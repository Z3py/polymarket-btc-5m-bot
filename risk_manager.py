from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from btc_data_feed import FeedHealth
from config import Settings
from feature_engineering import FeatureVector
from models import Prediction
from polymarket_client import MarketQuote, OrderBookSnapshot


@dataclass
class RiskState:
    starting_balance: float
    current_balance: float
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    trade_timestamps: list[datetime] = field(default_factory=list)
    pause_until: datetime | None = None
    safe_mode: bool = False

    @property
    def daily_loss_pct(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return max(0.0, -self.daily_pnl / self.starting_balance)


@dataclass
class RiskDecision:
    approved: bool
    side: str
    position_size: float
    limit_price: float
    reason: str
    conviction: str = "NONE"
    safe_mode: bool = False


class RiskManager:
    def __init__(self, settings: Settings, starting_balance: float | None = None) -> None:
        balance = starting_balance if starting_balance is not None else settings.paper_starting_balance
        self.settings = settings
        self.state = RiskState(starting_balance=balance, current_balance=balance)

    def evaluate(
        self,
        prediction: Prediction,
        features: FeatureVector,
        quote: MarketQuote,
        feed_health: FeedHealth,
        requested_all_in: bool = False,
    ) -> RiskDecision:
        now = datetime.now(timezone.utc)
        self._clean_trade_timestamps(now)
        all_in_note = "ALL-IN REQUEST BLOCKED BY RISK MANAGER. Position capped at 10%. " if requested_all_in else ""
        safe_mode_note = "SAFE_MODE active: paper-only. " if self.state.safe_mode else ""

        if not feed_health.healthy:
            return self._skip(f"Data feed unhealthy: {feed_health.reason}")
        if self.state.pause_until and now < self.state.pause_until:
            return self._skip(f"Paused after losses until {self.state.pause_until.isoformat()}")
        if self.state.daily_loss_pct >= self.settings.max_daily_loss_pct:
            return self._skip("Max daily loss reached; trading stopped")
        if len(self.state.trade_timestamps) >= self.settings.max_trades_per_hour:
            return self._skip("Max trades per hour reached")
        if features.seconds_to_expiry < self.settings.entry_min_seconds_to_close:
            return self._skip("Too close to market close")
        if features.seconds_to_expiry > self.settings.entry_max_seconds_to_close:
            return self._skip("Too early for configured entry window")
        if features.time_window_score <= 0:
            return self._skip("Invalid time-to-expiry window")
        if not _prices_normal(features.price_up, features.price_down, self.settings.price_sum_max_deviation):
            return self._skip("UP/DOWN prices are abnormal")

        side = self._choose_side(prediction)
        if side == "SKIP":
            return self._skip("No positive EV side")

        p_win = prediction.p_up if side == "UP" else prediction.p_down
        edge = prediction.edge_up if side == "UP" else prediction.edge_down
        ev = prediction.expected_value_up if side == "UP" else prediction.expected_value_down
        book = quote.up if side == "UP" else quote.down
        price = features.price_up if side == "UP" else features.price_down
        spread = book.spread

        if ev <= 0:
            return self._skip("EV <= 0 after fees/slippage")
        if spread > self.settings.max_spread:
            return self._skip(f"Spread too wide: {spread:.4f}")
        if self.settings.estimated_slippage > self.settings.max_slippage:
            return self._skip("Estimated slippage exceeds cap")
        if edge < self.settings.edge_high:
            return self._skip(f"Edge below required threshold: {edge:.4f} < {self.settings.edge_high:.4f}")
        if prediction.confidence_score < self.settings.min_confidence_high:
            return self._skip("Confidence below high threshold")

        conviction = "LOW"
        pct = 0.0
        if p_win >= 0.80:
            conviction = "HIGH"
            pct = min(self.settings.max_position_pct, 0.10)
        elif p_win >= 0.70:
            conviction = "MEDIUM"
            pct = min(0.05, max(0.03, self.settings.max_position_pct / 2))
        else:
            return self._skip("LOW CONVICTION: estimated winrate below 70%")

        if self.state.daily_loss_pct >= 0.10:
            pct *= 0.5
        max_notional = self.state.current_balance * min(pct, 0.10)
        if max_notional <= 0:
            return self._skip("No available balance")

        required_liquidity = self.settings.min_liquidity_multiple * (max_notional / max(price, 0.01))
        available_liquidity = _available_liquidity(book, price, side)
        if available_liquidity < required_liquidity:
            return self._skip(
                f"Insufficient liquidity: available {available_liquidity:.2f}, required {required_liquidity:.2f}"
            )

        reason = all_in_note + safe_mode_note + (
            f"{conviction} conviction: p_win={p_win:.3f}, edge={edge:.3f}, "
            f"EV={ev:.3f}, confidence={prediction.confidence_score:.1f}"
        )
        return RiskDecision(
            approved=True,
            side=side,
            position_size=max_notional,
            limit_price=min(price + self.settings.max_slippage, 0.99),
            reason=reason,
            conviction=conviction,
            safe_mode=self.state.safe_mode,
        )

    def register_trade(self, notional: float) -> None:
        self.state.trade_timestamps.append(datetime.now(timezone.utc))

    def register_result(self, pnl: float) -> None:
        self.state.daily_pnl += pnl
        self.state.current_balance += pnl
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        if self.state.consecutive_losses >= self.settings.max_consecutive_losses:
            self.state.pause_until = datetime.now(timezone.utc) + timedelta(
                minutes=self.settings.consecutive_loss_pause_minutes
            )
        if self.state.daily_loss_pct >= self.settings.max_daily_loss_pct:
            self.state.safe_mode = True

    def set_safe_mode(self, enabled: bool, reason: str = "") -> None:
        self.state.safe_mode = enabled
        if enabled:
            self.state.pause_until = None

    def _choose_side(self, prediction: Prediction) -> str:
        if prediction.expected_value_up > 0 and prediction.edge_up > prediction.edge_down:
            return "UP"
        if prediction.expected_value_down > 0 and prediction.edge_down > prediction.edge_up:
            return "DOWN"
        return "SKIP"

    def _clean_trade_timestamps(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=1)
        self.state.trade_timestamps = [ts for ts in self.state.trade_timestamps if ts >= cutoff]

    @staticmethod
    def _skip(reason: str, safe_mode: bool = False) -> RiskDecision:
        return RiskDecision(False, "SKIP", 0.0, 0.0, reason, safe_mode=safe_mode)


def _prices_normal(price_up: float, price_down: float, max_deviation: float) -> bool:
    if not (0.01 <= price_up <= 0.99 and 0.01 <= price_down <= 0.99):
        return False
    return abs((price_up + price_down) - 1.0) <= max_deviation


def _available_liquidity(book: OrderBookSnapshot, target_price: float, side: str) -> float:
    if side in {"UP", "DOWN"}:
        return book.depth_to_price("BUY", target_price)
    return 0.0
