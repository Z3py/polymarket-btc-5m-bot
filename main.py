from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from btc_data_feed import BTCDataFeed
from config import load_settings
from database import BotDatabase
from execution_engine import ExecutionEngine, ExecutionResult
from feature_engineering import FeatureVector, build_features
from logger import setup_logging
from market_resolver import MarketResolver, ResolvedMarket
from models import EnsemblePredictor, Prediction
from polymarket_client import MarketQuote, MarketWebSocketClient, OrderBookSnapshot, PolymarketClient
from risk_manager import RiskDecision, RiskManager


log = setup_logging()


@dataclass
class OpenPaperPosition:
    prediction_id: int
    side: str
    entry_price: float
    stake: float
    start_price: float
    end_time: datetime
    features: FeatureVector


class BotRuntime:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.db = BotDatabase(self.settings.database_url)
        self.resolver = MarketResolver(self.settings.gamma_host)
        self.client = PolymarketClient(self.settings)
        self.feed = BTCDataFeed(self.settings)
        self.predictor = EnsemblePredictor(fees=self.settings.fee_rate, slippage=self.settings.estimated_slippage)
        self.risk = RiskManager(self.settings)
        self.execution = ExecutionEngine(self.settings, self.client, self.risk)
        self.market: ResolvedMarket | None = None
        self.ws_task: asyncio.Task | None = None
        self.ws_books: dict[str, OrderBookSnapshot] = {}
        self.open_positions: list[OpenPaperPosition] = []

    async def close(self) -> None:
        if self.ws_task:
            self.ws_task.cancel()
        await self.feed.close()
        await self.client.close()
        self.db.close()

    async def run(self, hours: float = 24.0, settle_open_positions: bool = False) -> None:
        if self.settings.real_trading:
            log.warning("REAL_TRADING is enabled. Orders will be live limit orders.")
        else:
            log.info("PAPER_TRADING mode active. No live orders will be sent.")

        ws_price_task = asyncio.create_task(self.feed.run_websockets())
        stop_at = datetime.now(timezone.utc) + timedelta(hours=hours)
        try:
            while datetime.now(timezone.utc) < stop_at:
                try:
                    await self._tick()
                except Exception as exc:
                    log.exception("Tick failed; bot will retry on next poll: %s", exc)
                await asyncio.sleep(self.settings.poll_seconds)
            if settle_open_positions and self.open_positions:
                await self._wait_for_open_paper_positions()
        finally:
            ws_price_task.cancel()
            await self.close()

    async def _tick(self) -> None:
        market = await self._current_market()
        feed_health = await self.feed.poll_once()
        quote = await self._market_quote(market)
        features = build_features(self.feed.buffer, quote, market)
        prediction = self.predictor.predict(features)
        metrics = self.db.compute_metrics()
        if metrics.safe_mode_required:
            self.risk.set_safe_mode(True, "rolling winrate below 60% over 50 settled trades")
        decision = self.risk.evaluate(prediction, features, quote, feed_health)
        execution = await self.execution.execute(market, decision) if decision.approved else None
        prediction_id = self._persist(market, prediction, features, decision, execution)
        if execution and execution.mode in {"PAPER", "SAFE_MODE"}:
            self.open_positions.append(
                OpenPaperPosition(
                    prediction_id=prediction_id,
                    side=execution.side,
                    entry_price=execution.entry_price,
                    stake=execution.filled_notional,
                    start_price=features.btc_start,
                    end_time=market.end_time,
                    features=features,
                )
            )
        await self._settle_paper_positions()
        await self._settle_shadow_predictions()
        print_signal(market, features, prediction, decision, self.settings.mode_label, self.risk.state.safe_mode)

    async def _current_market(self) -> ResolvedMarket:
        now = datetime.now(timezone.utc)
        if self.market and self.market.end_time > now:
            return self.market
        try:
            market = self.resolver.resolve_from_url(self.settings.market_url)
            if market.end_time <= now:
                log.info("Configured market is expired; searching for current BTC 5m market.")
                market = self.resolver.resolve_current_btc_5m()
        except Exception as exc:
            log.warning("Configured market resolution failed, searching active BTC 5m market: %s", exc)
            market = self.resolver.resolve_current_btc_5m()
        self.market = market
        self._start_market_websocket(market)
        return market

    def _start_market_websocket(self, market: ResolvedMarket) -> None:
        if self.ws_task:
            self.ws_task.cancel()
        ws = MarketWebSocketClient(
            [market.up_token_id, market.down_token_id],
            on_book=lambda book: self.ws_books.__setitem__(book.token_id, book),
        )
        self.ws_task = asyncio.create_task(ws.run_forever())

    async def _market_quote(self, market: ResolvedMarket) -> MarketQuote:
        if market.up_token_id in self.ws_books and market.down_token_id in self.ws_books:
            return MarketQuote(up=self.ws_books[market.up_token_id], down=self.ws_books[market.down_token_id])
        return await self.client.get_market_quote(market.up_token_id, market.down_token_id)

    def _persist(
        self,
        market: ResolvedMarket,
        prediction: Prediction,
        features: FeatureVector,
        decision: RiskDecision,
        execution: ExecutionResult | None,
    ) -> int:
        side = decision.side if decision.approved else "SKIP"
        edge = prediction.edge_up if side == "UP" else prediction.edge_down if side == "DOWN" else 0.0
        ev = (
            prediction.expected_value_up
            if side == "UP"
            else prediction.expected_value_down
            if side == "DOWN"
            else max(prediction.expected_value_up, prediction.expected_value_down)
        )
        return self.db.save_prediction(
            market_id=market.market_id or market.condition_id,
            slug=market.slug,
            start_time=market.start_time,
            end_time=market.end_time,
            p_up=prediction.p_up,
            p_down=prediction.p_down,
            price_up=features.price_up,
            price_down=features.price_down,
            chosen_side=side,
            entry_price=execution.entry_price if execution else None,
            edge=edge,
            expected_value=ev,
            confidence_score=prediction.confidence_score,
            features=features.to_dict() | {"model_components": prediction.components},
            mode="SAFE_MODE" if self.risk.state.safe_mode else self.settings.mode_label,
            reason=decision.reason,
            order_id=execution.order_id if execution else None,
            position_size=decision.position_size if decision.approved else 0.0,
            model_side=prediction.recommended_side,
        )

    async def _settle_paper_positions(self) -> None:
        if not self.open_positions:
            return
        now = datetime.now(timezone.utc)
        btc_now = self.feed.buffer.latest_median(max_age_seconds=30.0)
        if btc_now is None:
            return
        remaining: list[OpenPaperPosition] = []
        for position in self.open_positions:
            if now < position.end_time:
                remaining.append(position)
                continue
            market_up = btc_now >= position.start_price
            won = (position.side == "UP" and market_up) or (position.side == "DOWN" and not market_up)
            pnl = position.stake * ((1 - position.entry_price) / position.entry_price) if won else -position.stake
            self.db.update_result(position.prediction_id, "WIN" if won else "LOSS", pnl)
            self.risk.register_result(pnl)
            self.predictor.update(position.features, 1 if market_up else 0)
        self.open_positions = remaining

    async def _settle_shadow_predictions(self) -> None:
        btc_now = self.feed.buffer.latest_median(max_age_seconds=30.0)
        if btc_now is None:
            return
        now = datetime.now(timezone.utc)
        rows = self.db.unsettled_shadow_predictions(now.isoformat())
        for row in rows:
            end_time = _parse_dt(row["end_time"])
            if end_time is None or now - end_time > timedelta(minutes=10):
                continue
            features = _feature_vector_from_json(row["features"])
            if features is None or features.btc_start <= 0:
                continue
            market_up = btc_now >= features.btc_start
            outcome = "UP" if market_up else "DOWN"
            model_side = str(row["model_side"] or "").upper()
            shadow_side = model_side if model_side in {"UP", "DOWN"} else ("UP" if float(row["p_up"] or 0.0) >= float(row["p_down"] or 0.0) else "DOWN")
            correct = int(shadow_side == outcome)
            self.db.update_shadow_result(
                int(row["id"]),
                outcome=outcome,
                shadow_side=shadow_side,
                shadow_result="WIN" if correct else "LOSS",
                shadow_correct=correct,
            )
            self.predictor.update(features, 1 if market_up else 0)

    async def _wait_for_open_paper_positions(self, max_wait_seconds: float = 360.0) -> None:
        deadline = datetime.now(timezone.utc) + timedelta(seconds=max_wait_seconds)
        while self.open_positions and datetime.now(timezone.utc) < deadline:
            await self.feed.poll_once()
            await self._settle_paper_positions()
            if self.open_positions:
                await asyncio.sleep(min(self.settings.poll_seconds, 5.0))


def _parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _feature_vector_from_json(value: object) -> FeatureVector | None:
    try:
        payload = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return None
    fields = set(FeatureVector.__dataclass_fields__.keys())
    cleaned = {name: payload.get(name, 0.0) for name in fields}
    try:
        return FeatureVector(**cleaned)
    except TypeError:
        return None


def print_signal(
    market: ResolvedMarket,
    features: FeatureVector,
    prediction: Prediction,
    decision: RiskDecision,
    mode: str,
    safe_mode: bool,
) -> None:
    mode_label = "SAFE_MODE" if safe_mode else mode
    signal = decision.side if decision.approved else "SKIP"
    print(
        "\n[BTC 5M SIGNAL]\n"
        f"Market: {market.slug}\n"
        f"Time left: {features.seconds_to_expiry:.1f}s\n"
        f"BTC start: {features.btc_start:.2f}\n"
        f"BTC now: {features.btc_now:.2f}\n"
        f"p_up: {prediction.p_up:.4f}\n"
        f"p_down: {prediction.p_down:.4f}\n"
        f"price_up: {features.price_up:.4f}\n"
        f"price_down: {features.price_down:.4f}\n"
        f"edge_up: {prediction.edge_up:.4f}\n"
        f"edge_down: {prediction.edge_down:.4f}\n"
        f"EV_up: {prediction.expected_value_up:.4f}\n"
        f"EV_down: {prediction.expected_value_down:.4f}\n"
        f"confidence: {prediction.confidence_score:.1f}\n"
        f"signal: {signal}\n"
        f"position_size: {decision.position_size:.2f}\n"
        f"reason: {decision.reason}\n"
        f"mode: {mode_label}\n"
    )


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket BTC Up/Down 5-minute probability bot")
    parser.add_argument("--hours", type=float, default=24.0, help="How long to run the live loop")
    args = parser.parse_args()
    runtime = BotRuntime()
    await runtime.run(hours=args.hours)


if __name__ == "__main__":
    asyncio.run(async_main())
