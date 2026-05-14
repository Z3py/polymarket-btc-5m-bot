from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

from btc_data_feed import PriceBuffer
from market_resolver import ResolvedMarket
from polymarket_client import MarketQuote


@dataclass
class FeatureVector:
    return_1m: float
    return_3m: float
    return_5m: float
    momentum_short: float
    vol_1m: float
    vol_3m: float
    vol_5m: float
    bid_depth_up: float
    ask_depth_up: float
    bid_depth_down: float
    ask_depth_down: float
    spread_up: float
    spread_down: float
    depth_imbalance: float
    price_up: float
    price_down: float
    normalized_prob_up: float
    ema_10s: float
    ema_30s: float
    ema_60s: float
    ema_slope_10_60: float
    breakout_up_1m: float
    breakout_down_1m: float
    zscore_return_1m: float
    seconds_to_expiry: float
    time_window_score: float
    btc_now: float
    btc_start: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_features(price_buffer: PriceBuffer, quote: MarketQuote, market: ResolvedMarket) -> FeatureVector:
    btc_now = price_buffer.latest_median(max_age_seconds=20.0) or 0.0
    btc_start = _market_start_price(price_buffer, market) or btc_now
    ema_10 = price_buffer.ema(10)
    ema_30 = price_buffer.ema(30)
    ema_60 = price_buffer.ema(60)
    recent_prices = price_buffer.prices_since(60)
    prior_prices = recent_prices[:-1] if len(recent_prices) > 1 else recent_prices
    high_1m = max(prior_prices) if prior_prices else btc_now
    low_1m = min(prior_prices) if prior_prices else btc_now
    price_sum = max(quote.price_up + quote.price_down, 1e-9)
    seconds_to_expiry = max(0.0, (market.end_time - datetime.now(timezone.utc)).total_seconds())
    bid_depth_up = quote.up.top_depth("BID")
    ask_depth_up = quote.up.top_depth("ASK")
    bid_depth_down = quote.down.top_depth("BID")
    ask_depth_down = quote.down.top_depth("ASK")
    total_depth = bid_depth_up + ask_depth_up + bid_depth_down + ask_depth_down
    depth_imbalance = (
        (bid_depth_up + ask_depth_down - ask_depth_up - bid_depth_down) / total_depth
        if total_depth > 0
        else 0.0
    )
    return FeatureVector(
        return_1m=price_buffer.return_over(60),
        return_3m=price_buffer.return_over(180),
        return_5m=price_buffer.return_over(300),
        momentum_short=(ema_10 / ema_60 - 1.0) if ema_60 else 0.0,
        vol_1m=price_buffer.realized_vol(60),
        vol_3m=price_buffer.realized_vol(180),
        vol_5m=price_buffer.realized_vol(300),
        bid_depth_up=bid_depth_up,
        ask_depth_up=ask_depth_up,
        bid_depth_down=bid_depth_down,
        ask_depth_down=ask_depth_down,
        spread_up=quote.up.spread,
        spread_down=quote.down.spread,
        depth_imbalance=depth_imbalance,
        price_up=quote.price_up,
        price_down=quote.price_down,
        normalized_prob_up=quote.price_up / price_sum,
        ema_10s=ema_10,
        ema_30s=ema_30,
        ema_60s=ema_60,
        ema_slope_10_60=(ema_10 - ema_60) / ema_60 if ema_60 else 0.0,
        breakout_up_1m=1.0 if btc_now > high_1m and btc_now > 0 else 0.0,
        breakout_down_1m=1.0 if btc_now < low_1m and btc_now > 0 else 0.0,
        zscore_return_1m=price_buffer.zscore_return(60),
        seconds_to_expiry=seconds_to_expiry,
        time_window_score=_time_window_score(seconds_to_expiry),
        btc_now=btc_now,
        btc_start=btc_start,
    )


def _market_start_price(price_buffer: PriceBuffer, market: ResolvedMarket) -> float | None:
    target_age = (datetime.now(timezone.utc) - market.start_time).total_seconds()
    if target_age <= 0:
        return price_buffer.latest_median()
    return price_buffer.price_at_or_before(target_age)


def _time_window_score(seconds_to_expiry: float) -> float:
    if seconds_to_expiry < 20:
        return 0.0
    if 45 <= seconds_to_expiry <= 150:
        return 1.0
    if seconds_to_expiry < 45:
        return (seconds_to_expiry - 20) / 25
    if seconds_to_expiry <= 240:
        return max(0.0, 1.0 - (seconds_to_expiry - 150) / 90)
    return 0.0
