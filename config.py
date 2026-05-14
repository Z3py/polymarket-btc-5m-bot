from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent


def _bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a float, got {raw!r}") from None


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


@dataclass(frozen=True)
class Settings:
    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""
    polymarket_funder: str = ""
    real_trading: bool = False

    max_position_pct: float = 0.10
    max_daily_loss_pct: float = 0.20
    edge_high: float = 0.16
    edge_medium: float = 0.08
    min_confidence_high: float = 85.0
    min_confidence_medium: float = 75.0
    paper_starting_balance: float = 1000.0

    market_url: str = "https://polymarket.com/id/event/btc-updown-5m-1778690700"
    database_url: str = "sqlite:///bot.sqlite3"

    chainlink_rpc_url: str = ""
    chainlink_btc_usd_feed_address: str = ""

    max_spread: float = 0.04
    max_slippage: float = 0.02
    min_liquidity_multiple: float = 3.0
    max_trades_per_hour: int = 12
    max_consecutive_losses: int = 3
    consecutive_loss_pause_minutes: int = 30
    entry_min_seconds_to_close: int = 20
    entry_max_seconds_to_close: int = 150
    entry_ideal_min_seconds_to_close: int = 45
    max_feed_deviation_bps: float = 20.0
    poll_seconds: float = 2.0

    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    polygon_chain_id: int = 137
    fee_rate: float = 0.0
    estimated_slippage: float = 0.01
    order_fill_timeout_seconds: float = 3.0
    price_sum_max_deviation: float = 0.10

    @property
    def mode_label(self) -> str:
        return "REAL" if self.real_trading else "PAPER"

    @property
    def has_trading_credentials(self) -> bool:
        return all(
            [
                self.polymarket_private_key,
                self.polymarket_api_key,
                self.polymarket_api_secret,
                self.polymarket_api_passphrase,
                self.polymarket_funder,
            ]
        )

    def validate(self) -> None:
        if self.max_position_pct > 0.10:
            raise ValueError("MAX_POSITION_PCT may not exceed 0.10")
        if self.real_trading and not self.has_trading_credentials:
            raise ValueError(
                "REAL_TRADING=true requires POLYMARKET_PRIVATE_KEY, "
                "POLYMARKET_API_KEY, POLYMARKET_API_SECRET, "
                "POLYMARKET_API_PASSPHRASE, and POLYMARKET_FUNDER."
            )


def load_settings(env_file: str | Path | None = None) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    settings = Settings(
        polymarket_private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
        polymarket_api_key=os.getenv("POLYMARKET_API_KEY", ""),
        polymarket_api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
        polymarket_api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
        polymarket_funder=os.getenv("POLYMARKET_FUNDER", ""),
        real_trading=_bool(os.getenv("REAL_TRADING"), False),
        max_position_pct=min(_float("MAX_POSITION_PCT", 0.10), 0.10),
        max_daily_loss_pct=_float("MAX_DAILY_LOSS_PCT", 0.20),
        edge_high=_float("EDGE_HIGH", 0.16),
        edge_medium=_float("EDGE_MEDIUM", 0.08),
        min_confidence_high=_float("MIN_CONFIDENCE_HIGH", 85.0),
        min_confidence_medium=_float("MIN_CONFIDENCE_MEDIUM", 75.0),
        paper_starting_balance=_float("PAPER_STARTING_BALANCE", 1000.0),
        market_url=os.getenv("MARKET_URL", Settings.market_url),
        database_url=os.getenv("DATABASE_URL", Settings.database_url),
        chainlink_rpc_url=os.getenv("CHAINLINK_RPC_URL", ""),
        chainlink_btc_usd_feed_address=os.getenv("CHAINLINK_BTC_USD_FEED_ADDRESS", ""),
        max_spread=_float("MAX_SPREAD", 0.04),
        max_slippage=_float("MAX_SLIPPAGE", 0.02),
        min_liquidity_multiple=_float("MIN_LIQUIDITY_MULTIPLE", 3.0),
        max_trades_per_hour=_int("MAX_TRADES_PER_HOUR", 12),
        max_consecutive_losses=_int("MAX_CONSECUTIVE_LOSSES", 3),
        consecutive_loss_pause_minutes=_int("CONSECUTIVE_LOSS_PAUSE_MINUTES", 30),
        entry_min_seconds_to_close=_int("ENTRY_MIN_SECONDS_TO_CLOSE", 20),
        entry_max_seconds_to_close=_int("ENTRY_MAX_SECONDS_TO_CLOSE", 150),
        entry_ideal_min_seconds_to_close=_int("ENTRY_IDEAL_MIN_SECONDS_TO_CLOSE", 45),
        max_feed_deviation_bps=_float("MAX_FEED_DEVIATION_BPS", 20.0),
        poll_seconds=_float("POLL_SECONDS", 2.0),
    )
    settings.validate()
    return settings
