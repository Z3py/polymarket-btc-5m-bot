from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from feature_engineering import FeatureVector


FEATURE_NAMES = [
    "return_1m",
    "return_3m",
    "return_5m",
    "momentum_short",
    "vol_1m",
    "vol_3m",
    "vol_5m",
    "depth_imbalance",
    "normalized_prob_up",
    "ema_slope_10_60",
    "breakout_up_1m",
    "breakout_down_1m",
    "zscore_return_1m",
    "time_window_score",
]


@dataclass
class Prediction:
    p_up: float
    p_down: float
    confidence_score: float
    expected_value_up: float
    expected_value_down: float
    edge_up: float
    edge_down: float
    recommended_side: str
    components: dict[str, float]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def logit(prob: float) -> float:
    clipped = min(max(prob, 1e-6), 1 - 1e-6)
    return math.log(clipped / (1 - clipped))


def clamp_prob(value: float) -> float:
    return min(max(value, 0.01), 0.99)


def calculate_edge_ev(
    p_up: float,
    price_up: float,
    price_down: float,
    fees: float = 0.0,
    slippage: float = 0.0,
) -> tuple[float, float, float, float]:
    p_up = clamp_prob(p_up)
    p_down = 1.0 - p_up
    ev_up = p_up * (1 - price_up) - (1 - p_up) * price_up - fees - slippage
    ev_down = p_down * (1 - price_down) - (1 - p_down) * price_down - fees - slippage
    edge_up = p_up - price_up
    edge_down = p_down - price_down
    return ev_up, ev_down, edge_up, edge_down


class OnlineLogisticRegression:
    def __init__(self, learning_rate: float = 0.05, l2: float = 0.001) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.bias = 0.0
        self.weights = {name: 0.0 for name in FEATURE_NAMES}

    def predict(self, features: FeatureVector) -> float:
        values = _scaled_feature_values(features)
        score = self.bias + sum(self.weights[name] * values[name] for name in FEATURE_NAMES)
        return clamp_prob(sigmoid(score))

    def update(self, features: FeatureVector, y_up: int) -> None:
        values = _scaled_feature_values(features)
        prediction = self.predict(features)
        error = y_up - prediction
        self.bias += self.learning_rate * error
        for name, value in values.items():
            self.weights[name] += self.learning_rate * (error * value - self.l2 * self.weights[name])

    def to_json(self) -> str:
        return json.dumps({"bias": self.bias, "weights": self.weights})

    def load_json(self, payload: str) -> None:
        data = json.loads(payload)
        self.bias = float(data.get("bias", 0.0))
        loaded = data.get("weights", {})
        self.weights.update({name: float(loaded.get(name, self.weights[name])) for name in FEATURE_NAMES})


class BayesianAdjuster:
    def __init__(self) -> None:
        self.alpha_up = 1.0
        self.alpha_down = 1.0

    def adjust(self, p_up: float, market_prob_up: float) -> float:
        prior = self.alpha_up / (self.alpha_up + self.alpha_down)
        blended_logit = 0.55 * logit(p_up) + 0.25 * logit(prior) + 0.20 * logit(market_prob_up)
        return clamp_prob(sigmoid(blended_logit))

    def update(self, y_up: int) -> None:
        self.alpha_up += 1 if y_up else 0
        self.alpha_down += 0 if y_up else 1


class KalmanTrend:
    def __init__(self) -> None:
        self.x = 0.0
        self.p = 1.0
        self.q = 0.0001
        self.r = 0.01

    def update(self, observed_return: float) -> float:
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (observed_return - self.x)
        self.p = (1 - k) * self.p
        return self.x

    def probability(self) -> float:
        return clamp_prob(sigmoid(self.x * 250.0))


class RuleMicrostructureSignal:
    def predict(self, features: FeatureVector) -> float:
        score = 0.0
        score += 2.0 * features.depth_imbalance
        score += 180.0 * features.ema_slope_10_60
        score += 90.0 * features.return_1m
        score += 0.35 * features.breakout_up_1m
        score -= 0.35 * features.breakout_down_1m
        score -= 0.08 * features.zscore_return_1m
        score += 0.25 * (features.normalized_prob_up - 0.5)
        score *= max(features.time_window_score, 0.25)
        return clamp_prob(sigmoid(score))


class OptionalGradientBooster:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.samples: list[list[float]] = []
        self.labels: list[int] = []
        self.enabled = False
        try:
            from xgboost import XGBClassifier

            self.model = XGBClassifier(
                n_estimators=25,
                max_depth=2,
                learning_rate=0.08,
                subsample=0.8,
                eval_metric="logloss",
            )
            self.enabled = True
        except Exception:
            self.enabled = False

    def predict(self, features: FeatureVector) -> float | None:
        if not self.enabled or self.model is None or len(set(self.labels)) < 2 or len(self.labels) < 30:
            return None
        try:
            return clamp_prob(float(self.model.predict_proba([_feature_list(features)])[0][1]))
        except Exception:
            return None

    def update(self, features: FeatureVector, y_up: int) -> None:
        if not self.enabled or self.model is None:
            return
        self.samples.append(_feature_list(features))
        self.labels.append(int(y_up))
        if len(self.labels) >= 30 and len(set(self.labels)) >= 2:
            try:
                self.model.fit(self.samples[-500:], self.labels[-500:])
            except Exception:
                return


class EnsemblePredictor:
    def __init__(self, fees: float = 0.0, slippage: float = 0.01) -> None:
        self.logistic = OnlineLogisticRegression()
        self.bayes = BayesianAdjuster()
        self.kalman = KalmanTrend()
        self.rules = RuleMicrostructureSignal()
        self.booster = OptionalGradientBooster()
        self.fees = fees
        self.slippage = slippage

    def predict(self, features: FeatureVector) -> Prediction:
        p_log = self.logistic.predict(features)
        trend = self.kalman.update(features.return_1m)
        p_kalman = self.kalman.probability()
        p_rules = self.rules.predict(features)
        p_bayes = self.bayes.adjust(p_log, features.normalized_prob_up)
        components = {
            "logistic": p_log,
            "bayesian": p_bayes,
            "kalman": p_kalman,
            "rules": p_rules,
            "kalman_trend": trend,
        }
        weights = {"logistic": 0.30, "bayesian": 0.25, "kalman": 0.20, "rules": 0.25}
        p_boost = self.booster.predict(features)
        if p_boost is not None:
            components["xgboost"] = p_boost
            weights = {"logistic": 0.24, "bayesian": 0.20, "kalman": 0.16, "rules": 0.20, "xgboost": 0.20}
        p_up = clamp_prob(sum(components[name] * weight for name, weight in weights.items()))
        p_down = 1.0 - p_up
        ev_up, ev_down, edge_up, edge_down = calculate_edge_ev(
            p_up=p_up,
            price_up=features.price_up,
            price_down=features.price_down,
            fees=self.fees,
            slippage=self.slippage,
        )
        confidence = self._confidence_score(p_up, components, features)
        side = "UP" if ev_up > ev_down else "DOWN"
        if max(ev_up, ev_down) <= 0:
            side = "SKIP"
        return Prediction(
            p_up=p_up,
            p_down=p_down,
            confidence_score=confidence,
            expected_value_up=ev_up,
            expected_value_down=ev_down,
            edge_up=edge_up,
            edge_down=edge_down,
            recommended_side=side,
            components=components,
        )

    def update(self, features: FeatureVector, y_up: int) -> None:
        self.logistic.update(features, y_up)
        self.bayes.update(y_up)
        self.booster.update(features, y_up)

    def _confidence_score(self, p_up: float, components: dict[str, float], features: FeatureVector) -> float:
        directional = abs(p_up - 0.5) * 200
        agreement = 100 - min(100, _std([value for key, value in components.items() if key != "kalman_trend"]) * 250)
        micro = min(100, abs(features.depth_imbalance) * 80 + abs(features.ema_slope_10_60) * 8000)
        time_score = features.time_window_score * 100
        score = 0.45 * directional + 0.25 * agreement + 0.15 * micro + 0.15 * time_score
        return max(0.0, min(100.0, score))


def _scaled_feature_values(features: FeatureVector) -> dict[str, float]:
    raw = features.to_dict()
    scaled = {}
    for name in FEATURE_NAMES:
        value = float(raw[name])
        if name.startswith("return") or name.startswith("vol") or name in {"momentum_short", "ema_slope_10_60"}:
            value *= 100.0
        if name == "zscore_return_1m":
            value /= 5.0
        if name == "normalized_prob_up":
            value -= 0.5
        scaled[name] = max(min(value, 5.0), -5.0)
    return scaled


def _feature_list(features: FeatureVector) -> list[float]:
    values = _scaled_feature_values(features)
    return [values[name] for name in FEATURE_NAMES]


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
