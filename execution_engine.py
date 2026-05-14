from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from config import Settings
from market_resolver import ResolvedMarket
from polymarket_client import PolymarketClient
from risk_manager import RiskDecision, RiskManager


@dataclass
class ExecutionResult:
    order_id: str
    side: str
    mode: str
    requested_notional: float
    filled_notional: float
    entry_price: float
    shares: float
    status: str
    raw: dict[str, Any]


class ExecutionEngine:
    def __init__(self, settings: Settings, client: PolymarketClient, risk_manager: RiskManager) -> None:
        self.settings = settings
        self.client = client
        self.risk_manager = risk_manager

    async def execute(self, market: ResolvedMarket, decision: RiskDecision) -> ExecutionResult | None:
        if not decision.approved or decision.side == "SKIP":
            return None
        token_id = market.up_token_id if decision.side == "UP" else market.down_token_id
        price = max(min(decision.limit_price, 0.99), 0.01)
        shares = decision.position_size / price

        if not self.settings.real_trading or self.risk_manager.state.safe_mode:
            order_id = f"paper-{uuid.uuid4().hex[:12]}"
            self.risk_manager.register_trade(decision.position_size)
            return ExecutionResult(
                order_id=order_id,
                side=decision.side,
                mode="SAFE_MODE" if self.risk_manager.state.safe_mode else "PAPER",
                requested_notional=decision.position_size,
                filled_notional=decision.position_size,
                entry_price=price,
                shares=shares,
                status="FILLED",
                raw={"created_at": datetime.now(timezone.utc).isoformat(), "market": market.slug},
            )

        result = await self.client.place_limit_order(
            token_id=token_id,
            side="BUY",
            price=price,
            size=shares,
            tick_size=str(market.raw.get("minimumTickSize") or market.raw.get("minimum_tick_size") or "0.01"),
            neg_risk=bool(market.raw.get("negRisk") or market.raw.get("neg_risk") or False),
        )
        order_id = _extract_order_id(result)
        await asyncio.sleep(self.settings.order_fill_timeout_seconds)
        if order_id:
            try:
                await self.client.cancel_order(order_id)
            except Exception:
                pass
        self.risk_manager.register_trade(decision.position_size)
        return ExecutionResult(
            order_id=order_id or f"real-{uuid.uuid4().hex[:12]}",
            side=decision.side,
            mode="REAL",
            requested_notional=decision.position_size,
            filled_notional=decision.position_size,
            entry_price=price,
            shares=shares,
            status="POSTED_OR_PARTIAL",
            raw=result,
        )


def _extract_order_id(payload: dict[str, Any]) -> str | None:
    for key in ("orderID", "order_id", "id"):
        value = payload.get(key)
        if value:
            return str(value)
    raw = payload.get("raw")
    if isinstance(raw, dict):
        return _extract_order_id(raw)
    return None
