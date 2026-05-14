from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
import websockets

from config import Settings
from logger import setup_logging


log = setup_logging()


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    token_id: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    last_update_ts: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> float | None:
        return max((level.price for level in self.bids), default=None)

    @property
    def best_ask(self) -> float | None:
        return min((level.price for level in self.asks), default=None)

    @property
    def mid(self) -> float | None:
        if self.best_bid is None and self.best_ask is None:
            return None
        if self.best_bid is None:
            return self.best_ask
        if self.best_ask is None:
            return self.best_bid
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        if self.best_bid is None or self.best_ask is None:
            return 1.0
        return max(0.0, self.best_ask - self.best_bid)

    def depth_to_price(self, side: str, limit_price: float) -> float:
        levels = self.asks if side.upper() == "BUY" else self.bids
        if side.upper() == "BUY":
            return sum(level.size for level in levels if level.price <= limit_price)
        return sum(level.size for level in levels if level.price >= limit_price)

    def top_depth(self, side: str, levels: int = 5) -> float:
        source = sorted(self.asks, key=lambda level: level.price) if side.upper() == "ASK" else sorted(self.bids, key=lambda level: level.price, reverse=True)
        return sum(level.size for level in source[:levels])


@dataclass
class MarketQuote:
    up: OrderBookSnapshot
    down: OrderBookSnapshot
    last_trade_up: float | None = None
    last_trade_down: float | None = None

    @property
    def price_up(self) -> float:
        return self.up.best_ask or self.up.mid or 0.0

    @property
    def price_down(self) -> float:
        return self.down.best_ask or self.down.mid or 0.0


def _levels(items: Any) -> list[OrderBookLevel]:
    levels: list[OrderBookLevel] = []
    if not items:
        return levels
    for item in items:
        try:
            if isinstance(item, dict):
                price = float(item.get("price"))
                size = float(item.get("size"))
            else:
                price = float(item[0])
                size = float(item[1])
            if 0 < price < 1.5 and size > 0:
                levels.append(OrderBookLevel(price=price, size=size))
        except (TypeError, ValueError, IndexError):
            continue
    return levels


def parse_orderbook(payload: dict[str, Any], token_id: str) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        token_id=str(payload.get("asset_id") or payload.get("token_id") or token_id),
        bids=_levels(payload.get("bids") or payload.get("buys") or []),
        asks=_levels(payload.get("asks") or payload.get("sells") or []),
    )


class PolymarketClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(base_url=settings.clob_host, timeout=10.0)
        self._sdk_client: Any | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def get_order_book(self, token_id: str) -> OrderBookSnapshot:
        response = await self._client.get("/book", params={"token_id": token_id})
        response.raise_for_status()
        return parse_orderbook(response.json(), token_id)

    async def get_market_quote(self, up_token_id: str, down_token_id: str) -> MarketQuote:
        up, down = await asyncio.gather(
            self.get_order_book(up_token_id),
            self.get_order_book(down_token_id),
        )
        last_up, last_down = await asyncio.gather(
            self.get_last_trade_price(up_token_id),
            self.get_last_trade_price(down_token_id),
            return_exceptions=True,
        )
        return MarketQuote(
            up=up,
            down=down,
            last_trade_up=None if isinstance(last_up, Exception) else last_up,
            last_trade_down=None if isinstance(last_down, Exception) else last_down,
        )

    async def get_last_trade_price(self, token_id: str) -> float | None:
        response = await self._client.get("/last-trade-price", params={"token_id": token_id})
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        value = payload.get("price") or payload.get("last_trade_price")
        return float(value) if value is not None else None

    def ensure_real_trading_ready(self) -> None:
        if not self.settings.real_trading:
            raise RuntimeError("Real trading is disabled. Set REAL_TRADING=true to enable.")
        if not self.settings.has_trading_credentials:
            raise RuntimeError("Real trading credentials are incomplete.")

    def _load_sdk(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        self.ensure_real_trading_ready()
        try:
            from py_clob_client_v2 import ApiCreds, ClobClient
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "py-clob-client-v2 is required for real trading. Install requirements.txt first."
            ) from exc

        creds = ApiCreds(
            api_key=self.settings.polymarket_api_key,
            api_secret=self.settings.polymarket_api_secret,
            api_passphrase=self.settings.polymarket_api_passphrase,
        )
        self._sdk_client = ClobClient(
            host=self.settings.clob_host,
            chain_id=self.settings.polygon_chain_id,
            key=self.settings.polymarket_private_key,
            creds=creds,
            signature_type=3,
            funder=self.settings.polymarket_funder,
        )
        return self._sdk_client

    async def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        tick_size: str = "0.01",
        neg_risk: bool = False,
    ) -> dict[str, Any]:
        sdk = self._load_sdk()

        def _post() -> dict[str, Any]:
            try:
                from py_clob_client_v2 import OrderArgs, PartialCreateOrderOptions
                from py_clob_client_v2.order_builder.constants import BUY, SELL
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("py-clob-client-v2 order classes unavailable") from exc
            order_side = BUY if side.upper() == "BUY" else SELL
            order = sdk.create_and_post_order(
                OrderArgs(token_id=token_id, price=round(price, 4), size=round(size, 4), side=order_side),
                options=PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk),
            )
            return order if isinstance(order, dict) else {"raw": order}

        return await asyncio.to_thread(_post)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        sdk = self._load_sdk()

        def _cancel() -> dict[str, Any]:
            for method_name in ("cancel_order", "cancelOrder"):
                method = getattr(sdk, method_name, None)
                if method:
                    result = method(order_id)
                    return result if isinstance(result, dict) else {"raw": result}
            raise RuntimeError("CLOB SDK does not expose a cancel_order method")

        return await asyncio.to_thread(_cancel)


class MarketWebSocketClient:
    def __init__(
        self,
        token_ids: list[str],
        on_book: Callable[[OrderBookSnapshot], None] | None = None,
        on_trade: Callable[[str, float], None] | None = None,
    ) -> None:
        self.token_ids = [str(token_id) for token_id in token_ids]
        self.on_book = on_book
        self.on_trade = on_trade
        self.url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self._stopped = asyncio.Event()

    def stop(self) -> None:
        self._stopped.set()

    async def run_forever(self) -> None:
        backoff = 1.0
        while not self._stopped.is_set():
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=10) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "assets_ids": self.token_ids,
                                "type": "market",
                                "custom_feature_enabled": True,
                            }
                        )
                    )
                    log.info("Polymarket market websocket subscribed for %s assets", len(self.token_ids))
                    backoff = 1.0
                    async for message in ws:
                        if self._stopped.is_set():
                            break
                        self._handle_message(message)
            except Exception as exc:
                log.warning("Polymarket websocket reconnecting after error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle_message(self, message: str | bytes) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        events = payload if isinstance(payload, list) else [payload]
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = event.get("event_type") or event.get("type")
            token_id = str(event.get("asset_id") or event.get("asset") or "")
            if event_type == "book" and token_id:
                snapshot = parse_orderbook(event, token_id)
                if self.on_book:
                    self.on_book(snapshot)
            elif event_type == "last_trade_price" and token_id and self.on_trade:
                try:
                    self.on_trade(token_id, float(event.get("price")))
                except (TypeError, ValueError):
                    continue
