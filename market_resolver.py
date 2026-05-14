from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any
from urllib.parse import urlparse

import requests


@dataclass(frozen=True)
class ResolvedMarket:
    slug: str
    market_id: str
    condition_id: str
    question: str
    up_token_id: str
    down_token_id: str
    start_time: datetime
    end_time: datetime
    raw: dict[str, Any]

    @property
    def seconds_to_close(self) -> float:
        return (self.end_time - datetime.now(timezone.utc)).total_seconds()


def extract_slug_from_url(url_or_slug: str) -> str:
    value = url_or_slug.strip()
    if not value:
        raise ValueError("Market URL or slug is empty")
    if not value.startswith(("http://", "https://")):
        return value.strip("/")
    parsed = urlparse(value)
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        raise ValueError(f"Could not extract slug from URL: {url_or_slug}")
    if "event" in parts:
        idx = parts.index("event")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return parts[-1]


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            if value.isdigit():
                return datetime.fromtimestamp(int(value), tz=timezone.utc)
    return None


def _infer_times_from_slug(slug: str) -> tuple[datetime, datetime]:
    match = re.search(r"-(\d{10})$", slug)
    if not match:
        now = datetime.now(timezone.utc)
        rounded = now.replace(second=0, microsecond=0)
        minute_bucket = rounded.minute - (rounded.minute % 5)
        start = rounded.replace(minute=minute_bucket)
        end = start + timedelta(minutes=5)
        if end <= now:
            start = end
            end = start + timedelta(minutes=5)
        return start, end
    ts = int(match.group(1))
    end = datetime.fromtimestamp(ts, tz=timezone.utc)
    return end - timedelta(minutes=5), end


def _token_ids_and_labels(market: dict[str, Any]) -> tuple[str, str]:
    token_ids = _maybe_json(market.get("clobTokenIds") or market.get("clob_token_ids") or [])
    outcomes = _maybe_json(market.get("outcomes") or market.get("shortOutcomes") or [])
    if isinstance(token_ids, str):
        token_ids = [item.strip() for item in token_ids.split(",") if item.strip()]
    if isinstance(outcomes, str):
        outcomes = [item.strip() for item in outcomes.split(",") if item.strip()]
    if len(token_ids) < 2:
        raise ValueError("Market does not expose two CLOB token IDs")

    labels = [str(item).strip().lower() for item in outcomes] if outcomes else []
    up_idx, down_idx = 0, 1
    for idx, label in enumerate(labels[: len(token_ids)]):
        if label in {"up", "yes", "higher", "above"}:
            up_idx = idx
        if label in {"down", "no", "lower", "below"}:
            down_idx = idx
    if up_idx == down_idx and len(token_ids) >= 2:
        up_idx, down_idx = 0, 1
    return str(token_ids[up_idx]), str(token_ids[down_idx])


class MarketResolver:
    def __init__(self, gamma_host: str = "https://gamma-api.polymarket.com", timeout: float = 10.0) -> None:
        self.gamma_host = gamma_host.rstrip("/")
        self.timeout = timeout

    def resolve_from_url(self, url_or_slug: str) -> ResolvedMarket:
        slug = extract_slug_from_url(url_or_slug)
        market = self._fetch_market_by_slug(slug)
        return self._build_market(slug, market)

    def resolve_current_btc_5m(self) -> ResolvedMarket:
        response = requests.get(
            f"{self.gamma_host}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": 100,
                "order": "endDate",
                "ascending": "true",
                "tag_slug": "bitcoin",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        markets = response.json()
        now = datetime.now(timezone.utc)
        candidates: list[ResolvedMarket] = []
        for market in markets:
            slug = str(market.get("slug", ""))
            question = str(market.get("question", "")).lower()
            if "btc" not in slug.lower() and "bitcoin" not in question:
                continue
            if "5m" not in slug.lower() and "5 minute" not in question and "5-minute" not in question:
                continue
            try:
                resolved = self._build_market(slug, market)
            except Exception:
                continue
            if resolved.end_time > now:
                candidates.append(resolved)
        if not candidates:
            raise LookupError("No active BTC 5-minute Polymarket market found")
        return min(candidates, key=lambda market: market.end_time)

    def _fetch_market_by_slug(self, slug: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.gamma_host}/markets",
            params={"slug": slug, "limit": 1},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list) and payload:
            return payload[0]

        # Some URLs are event slugs. Fetch events and use the first attached market.
        event_response = requests.get(
            f"{self.gamma_host}/events",
            params={"slug": slug, "limit": 1},
            timeout=self.timeout,
        )
        event_response.raise_for_status()
        events = event_response.json()
        if isinstance(events, list) and events:
            event = events[0]
            markets = event.get("markets") or []
            if markets:
                return markets[0]
        raise LookupError(f"Could not find market for slug {slug!r}")

    def _build_market(self, slug: str, market: dict[str, Any]) -> ResolvedMarket:
        up_token, down_token = _token_ids_and_labels(market)
        inferred_start, inferred_end = _infer_times_from_slug(slug)
        end_time = (
            _parse_time(market.get("endDate"))
            or _parse_time(market.get("endDateIso"))
            or _parse_time(market.get("end_time"))
            or inferred_end
        )
        start_time = (
            _parse_time(market.get("startDate"))
            or _parse_time(market.get("startDateIso"))
            or _parse_time(market.get("start_time"))
            or inferred_start
        )
        return ResolvedMarket(
            slug=str(market.get("slug") or slug),
            market_id=str(market.get("id") or market.get("marketId") or market.get("conditionId") or ""),
            condition_id=str(market.get("conditionId") or market.get("condition_id") or ""),
            question=str(market.get("question") or ""),
            up_token_id=up_token,
            down_token_id=down_token,
            start_time=start_time,
            end_time=end_time,
            raw=market,
        )
