from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx
import websockets

from config import Settings
from logger import setup_logging


log = setup_logging()


@dataclass
class PriceTick:
    source: str
    price: float
    ts: float


@dataclass
class FeedHealth:
    healthy: bool
    reason: str
    median_price: float
    max_deviation_bps: float
    sources: dict[str, float]


class PriceBuffer:
    def __init__(self, max_age_seconds: int = 600) -> None:
        self.max_age_seconds = max_age_seconds
        self.ticks: deque[PriceTick] = deque()

    def add(self, source: str, price: float, ts: float | None = None) -> None:
        now = ts or time.time()
        if price <= 0 or not math.isfinite(price):
            return
        self.ticks.append(PriceTick(source=source, price=price, ts=now))
        self.prune(now)

    def prune(self, now: float | None = None) -> None:
        ref = now or time.time()
        while self.ticks and ref - self.ticks[0].ts > self.max_age_seconds:
            self.ticks.popleft()

    def latest_by_source(self, max_age_seconds: float = 10.0) -> dict[str, PriceTick]:
        now = time.time()
        latest: dict[str, PriceTick] = {}
        for tick in reversed(self.ticks):
            if now - tick.ts > max_age_seconds:
                continue
            if tick.source not in latest:
                latest[tick.source] = tick
        return latest

    def latest_median(self, max_age_seconds: float = 10.0) -> float | None:
        latest = self.latest_by_source(max_age_seconds)
        if not latest:
            return None
        prices = sorted(tick.price for tick in latest.values())
        mid = len(prices) // 2
        if len(prices) % 2:
            return prices[mid]
        return (prices[mid - 1] + prices[mid]) / 2

    def price_at_or_before(self, seconds_ago: float) -> float | None:
        target = time.time() - seconds_ago
        candidates = [tick for tick in self.ticks if tick.ts <= target]
        if not candidates:
            return None
        return candidates[-1].price

    def prices_since(self, seconds: float) -> list[float]:
        cutoff = time.time() - seconds
        return [tick.price for tick in self.ticks if tick.ts >= cutoff]

    def return_over(self, seconds: float) -> float:
        now_price = self.latest_median() or (self.ticks[-1].price if self.ticks else 0.0)
        old = self.price_at_or_before(seconds)
        if not old or not now_price:
            return 0.0
        return (now_price / old) - 1.0

    def realized_vol(self, seconds: float) -> float:
        prices = self.prices_since(seconds)
        if len(prices) < 3:
            return 0.0
        returns = []
        for prev, cur in zip(prices, prices[1:]):
            if prev > 0:
                returns.append(math.log(cur / prev))
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        return math.sqrt(max(variance, 0.0))

    def ema(self, seconds: float) -> float:
        prices = self.prices_since(seconds)
        if not prices:
            return self.latest_median() or 0.0
        alpha = 2.0 / (len(prices) + 1.0)
        value = prices[0]
        for price in prices[1:]:
            value = alpha * price + (1.0 - alpha) * value
        return value

    def high_low(self, seconds: float) -> tuple[float, float]:
        prices = self.prices_since(seconds)
        if not prices:
            price = self.latest_median() or 0.0
            return price, price
        return max(prices), min(prices)

    def zscore_return(self, seconds: float) -> float:
        returns = []
        prices = self.prices_since(max(seconds * 5, 60))
        for prev, cur in zip(prices, prices[1:]):
            if prev > 0:
                returns.append((cur / prev) - 1.0)
        if len(returns) < 5:
            return 0.0
        current = self.return_over(seconds)
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / max(len(returns) - 1, 1)
        std = math.sqrt(max(variance, 1e-12))
        return max(min((current - mean) / std, 10.0), -10.0)


class BTCDataFeed:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.buffer = PriceBuffer()
        self._client = httpx.AsyncClient(timeout=8.0)
        self._stopped = asyncio.Event()

    async def close(self) -> None:
        self._stopped.set()
        await self._client.aclose()

    async def poll_once(self) -> FeedHealth:
        results = await asyncio.gather(
            self._fetch_binance_rest(),
            self._fetch_coinbase_rest(),
            self._fetch_chainlink_optional(),
            return_exceptions=True,
        )
        for source, result in zip(("binance", "coinbase", "chainlink"), results):
            if isinstance(result, Exception):
                log.debug("%s feed failed: %s", source, result)
                continue
            if result is not None:
                self.buffer.add(source, result)
        return self.health()

    def health(self) -> FeedHealth:
        latest = self.buffer.latest_by_source(max_age_seconds=15.0)
        sources = {source: tick.price for source, tick in latest.items()}
        if len(sources) < 2:
            return FeedHealth(False, "Need at least two fresh BTC price sources", 0.0, 0.0, sources)
        prices = sorted(sources.values())
        median = prices[len(prices) // 2] if len(prices) % 2 else (prices[len(prices) // 2 - 1] + prices[len(prices) // 2]) / 2
        deviations = [abs(price - median) / median * 10_000 for price in prices if median > 0]
        max_dev = max(deviations, default=0.0)
        if max_dev > self.settings.max_feed_deviation_bps:
            return FeedHealth(False, f"BTC feeds diverged by {max_dev:.1f} bps", median, max_dev, sources)
        return FeedHealth(True, "OK", median, max_dev, sources)

    async def _fetch_binance_rest(self) -> float:
        response = await self._client.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"})
        response.raise_for_status()
        return float(response.json()["price"])

    async def _fetch_coinbase_rest(self) -> float:
        response = await self._client.get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
        response.raise_for_status()
        return float(response.json()["data"]["amount"])

    async def _fetch_chainlink_optional(self) -> float | None:
        if not self.settings.chainlink_rpc_url or not self.settings.chainlink_btc_usd_feed_address:
            return None
        try:
            from web3 import Web3
        except Exception:
            return None

        abi = [
            {
                "inputs": [],
                "name": "latestRoundData",
                "outputs": [
                    {"internalType": "uint80", "name": "roundId", "type": "uint80"},
                    {"internalType": "int256", "name": "answer", "type": "int256"},
                    {"internalType": "uint256", "name": "startedAt", "type": "uint256"},
                    {"internalType": "uint256", "name": "updatedAt", "type": "uint256"},
                    {"internalType": "uint80", "name": "answeredInRound", "type": "uint80"},
                ],
                "stateMutability": "view",
                "type": "function",
            },
            {
                "inputs": [],
                "name": "decimals",
                "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
                "stateMutability": "view",
                "type": "function",
            },
        ]

        def _read() -> float | None:
            web3 = Web3(Web3.HTTPProvider(self.settings.chainlink_rpc_url, request_kwargs={"timeout": 5}))
            contract = web3.eth.contract(address=Web3.to_checksum_address(self.settings.chainlink_btc_usd_feed_address), abi=abi)
            decimals = contract.functions.decimals().call()
            data = contract.functions.latestRoundData().call()
            updated_at = int(data[3])
            if time.time() - updated_at > 60:
                return None
            return float(data[1]) / (10**decimals)

        return await asyncio.to_thread(_read)

    async def run_websockets(self) -> None:
        await asyncio.gather(self._run_binance_ws(), self._run_coinbase_ws())

    async def _run_binance_ws(self) -> None:
        url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
        backoff = 1.0
        while not self._stopped.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    backoff = 1.0
                    async for raw in ws:
                        payload = json.loads(raw)
                        self.buffer.add("binance", float(payload["p"]), ts=float(payload["T"]) / 1000.0)
            except Exception as exc:
                log.warning("Binance websocket reconnecting after error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _run_coinbase_ws(self) -> None:
        url = "wss://ws-feed.exchange.coinbase.com"
        subscribe = {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["ticker"],
        }
        backoff = 1.0
        while not self._stopped.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    await ws.send(json.dumps(subscribe))
                    backoff = 1.0
                    async for raw in ws:
                        payload = json.loads(raw)
                        if payload.get("type") == "ticker" and payload.get("price"):
                            self.buffer.add("coinbase", float(payload["price"]))
            except Exception as exc:
                log.warning("Coinbase websocket reconnecting after error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
