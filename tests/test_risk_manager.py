from datetime import datetime, timedelta, timezone

from btc_data_feed import FeedHealth
from config import Settings
from feature_engineering import FeatureVector
from models import Prediction
from polymarket_client import MarketQuote, OrderBookLevel, OrderBookSnapshot
from risk_manager import RiskManager


def _features(price_up: float = 0.62, price_down: float = 0.38) -> FeatureVector:
    return FeatureVector(
        return_1m=0.001,
        return_3m=0.002,
        return_5m=0.003,
        momentum_short=0.001,
        vol_1m=0.0005,
        vol_3m=0.0007,
        vol_5m=0.001,
        bid_depth_up=1000,
        ask_depth_up=1000,
        bid_depth_down=1000,
        ask_depth_down=1000,
        spread_up=0.01,
        spread_down=0.01,
        depth_imbalance=0.2,
        price_up=price_up,
        price_down=price_down,
        normalized_prob_up=price_up / (price_up + price_down),
        ema_10s=100,
        ema_30s=99,
        ema_60s=98,
        ema_slope_10_60=0.01,
        breakout_up_1m=1,
        breakout_down_1m=0,
        zscore_return_1m=0.5,
        seconds_to_expiry=90,
        time_window_score=1,
        btc_now=100,
        btc_start=99,
    )


def _quote() -> MarketQuote:
    up = OrderBookSnapshot(
        token_id="up",
        bids=[OrderBookLevel(0.61, 1000)],
        asks=[OrderBookLevel(0.62, 1000)],
    )
    down = OrderBookSnapshot(
        token_id="down",
        bids=[OrderBookLevel(0.37, 1000)],
        asks=[OrderBookLevel(0.38, 1000)],
    )
    return MarketQuote(up=up, down=down)


def _prediction() -> Prediction:
    return Prediction(
        p_up=0.82,
        p_down=0.18,
        confidence_score=88,
        expected_value_up=0.19,
        expected_value_down=-0.21,
        edge_up=0.20,
        edge_down=-0.20,
        recommended_side="UP",
        components={},
    )


def test_high_conviction_is_capped_at_ten_percent() -> None:
    settings = Settings(max_position_pct=0.50)
    manager = RiskManager(settings, starting_balance=1000)
    decision = manager.evaluate(
        _prediction(),
        _features(),
        _quote(),
        FeedHealth(True, "OK", 100, 1, {"binance": 100, "coinbase": 100.01}),
    )
    assert decision.approved
    assert decision.side == "UP"
    assert decision.position_size <= 100


def test_all_in_request_is_blocked() -> None:
    manager = RiskManager(Settings(), starting_balance=1000)
    decision = manager.evaluate(
        _prediction(),
        _features(),
        _quote(),
        FeedHealth(True, "OK", 100, 1, {"binance": 100, "coinbase": 100.01}),
        requested_all_in=True,
    )
    assert decision.approved
    assert decision.position_size <= 100
    assert "ALL-IN REQUEST BLOCKED" in decision.reason


def test_spread_too_wide_skips() -> None:
    manager = RiskManager(Settings(), starting_balance=1000)
    quote = _quote()
    quote.up.asks = [OrderBookLevel(0.70, 1000)]
    decision = manager.evaluate(
        _prediction(),
        _features(price_up=0.70, price_down=0.30),
        quote,
        FeedHealth(True, "OK", 100, 1, {"binance": 100, "coinbase": 100.01}),
    )
    assert not decision.approved
    assert "Spread too wide" in decision.reason


def test_edge_below_sixteen_percent_skips() -> None:
    manager = RiskManager(Settings(), starting_balance=1000)
    prediction = _prediction()
    prediction.p_up = 0.75
    prediction.p_down = 0.25
    prediction.edge_up = 0.13
    prediction.expected_value_up = 0.10
    decision = manager.evaluate(
        prediction,
        _features(price_up=0.62, price_down=0.38),
        _quote(),
        FeedHealth(True, "OK", 100, 1, {"binance": 100, "coinbase": 100.01}),
    )
    assert not decision.approved
    assert "Edge below required threshold" in decision.reason


def test_consecutive_losses_triggers_pause() -> None:
    manager = RiskManager(Settings(), starting_balance=1000)
    for _ in range(3):
        manager.register_result(-10)
    assert manager.state.pause_until is not None
    assert manager.state.pause_until > datetime.now(timezone.utc) - timedelta(seconds=1)
