from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Metrics:
    rolling_winrate_20: float | None
    rolling_winrate_50: float | None
    brier_score: float | None
    calibration_error: float | None
    profit_factor: float | None
    max_drawdown: float
    average_ev_vs_actual_pnl: float | None
    trade_count: int
    safe_mode_required: bool


@dataclass
class ShadowMetrics:
    predictions: int
    correct: int
    winrate: float | None


class BotDatabase:
    def __init__(self, database_url: str = "sqlite:///bot.sqlite3") -> None:
        self.database_url = database_url
        self.is_postgres = database_url.startswith(("postgresql://", "postgres://"))
        if database_url.startswith("sqlite:///"):
            path = database_url.replace("sqlite:///", "", 1)
            self.path = Path(path)
            self.conn = sqlite3.connect(self.path)
            self.conn.row_factory = sqlite3.Row
        elif self.is_postgres:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except Exception as exc:
                raise RuntimeError("PostgreSQL DATABASE_URL requires psycopg[binary]") from exc
            self.path = Path("")
            self.conn = psycopg.connect(database_url, row_factory=dict_row)
        else:
            path = "bot.sqlite3"
            self.path = Path(path)
            self.conn = sqlite3.connect(self.path)
            self.conn.row_factory = sqlite3.Row
        self.migrate()

    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> None:
        if self.is_postgres:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    market_id TEXT,
                    slug TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    p_up DOUBLE PRECISION,
                    p_down DOUBLE PRECISION,
                    price_up DOUBLE PRECISION,
                    price_down DOUBLE PRECISION,
                    chosen_side TEXT,
                    entry_price DOUBLE PRECISION,
                    result TEXT,
                    pnl DOUBLE PRECISION,
                    edge DOUBLE PRECISION,
                    expected_value DOUBLE PRECISION,
                    confidence_score DOUBLE PRECISION,
                    position_size DOUBLE PRECISION,
                    model_side TEXT,
                    outcome TEXT,
                    shadow_side TEXT,
                    shadow_result TEXT,
                    shadow_correct INTEGER,
                    features TEXT,
                    mode TEXT,
                    reason TEXT,
                    order_id TEXT
                )
                """
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_market ON predictions(market_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
        else:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    market_id TEXT,
                    slug TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    p_up REAL,
                    p_down REAL,
                    price_up REAL,
                    price_down REAL,
                    chosen_side TEXT,
                    entry_price REAL,
                    result TEXT,
                    pnl REAL,
                    edge REAL,
                    expected_value REAL,
                    confidence_score REAL,
                    position_size REAL,
                    model_side TEXT,
                    outcome TEXT,
                    shadow_side TEXT,
                    shadow_result TEXT,
                    shadow_correct INTEGER,
                    features TEXT,
                    mode TEXT,
                    reason TEXT,
                    order_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_predictions_market ON predictions(market_id);
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
                """
            )
        self._ensure_column("predictions", "position_size", "DOUBLE PRECISION" if self.is_postgres else "REAL")
        self._ensure_column("predictions", "model_side", "TEXT")
        self._ensure_column("predictions", "outcome", "TEXT")
        self._ensure_column("predictions", "shadow_side", "TEXT")
        self._ensure_column("predictions", "shadow_result", "TEXT")
        self._ensure_column("predictions", "shadow_correct", "INTEGER")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        if self.is_postgres:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {column_type}")
            return
        columns = [row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")]
        if column not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def save_prediction(
        self,
        market_id: str,
        slug: str,
        start_time: datetime,
        end_time: datetime,
        p_up: float,
        p_down: float,
        price_up: float,
        price_down: float,
        chosen_side: str,
        entry_price: float | None,
        edge: float,
        expected_value: float,
        confidence_score: float,
        features: dict[str, Any],
        mode: str,
        reason: str,
        order_id: str | None = None,
        position_size: float | None = None,
        model_side: str | None = None,
    ) -> int:
        values = (
            datetime.now(timezone.utc).isoformat(),
            market_id,
            slug,
            start_time.isoformat(),
            end_time.isoformat(),
            p_up,
            p_down,
            price_up,
            price_down,
            chosen_side,
            entry_price,
            edge,
            expected_value,
            confidence_score,
            position_size,
            model_side,
            json.dumps(features, separators=(",", ":")),
            mode,
            reason,
            order_id,
        )
        if self.is_postgres:
            cursor = self.conn.execute(
                """
                INSERT INTO predictions (
                    timestamp, market_id, slug, start_time, end_time, p_up, p_down,
                    price_up, price_down, chosen_side, entry_price, result, pnl,
                    edge, expected_value, confidence_score, position_size, model_side,
                    outcome, shadow_side, shadow_result, shadow_correct, features, mode, reason, order_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL, %s, %s, %s, %s, %s, NULL, NULL, NULL, NULL, %s, %s, %s, %s)
                RETURNING id
                """,
                values,
            )
            prediction_id = int(cursor.fetchone()["id"])
        else:
            cursor = self.conn.execute(
                """
                INSERT INTO predictions (
                    timestamp, market_id, slug, start_time, end_time, p_up, p_down,
                    price_up, price_down, chosen_side, entry_price, result, pnl,
                    edge, expected_value, confidence_score, position_size, model_side,
                    outcome, shadow_side, shadow_result, shadow_correct, features, mode, reason, order_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, ?, ?)
                """,
                values,
            )
            prediction_id = int(cursor.lastrowid)
        self.conn.commit()
        return prediction_id

    def update_result(self, prediction_id: int, result: str, pnl: float) -> None:
        placeholder = "%s" if self.is_postgres else "?"
        self.conn.execute(
            f"UPDATE predictions SET result = {placeholder}, pnl = {placeholder} WHERE id = {placeholder}",
            (result, pnl, prediction_id),
        )
        self.conn.commit()

    def update_shadow_result(
        self,
        prediction_id: int,
        outcome: str,
        shadow_side: str,
        shadow_result: str,
        shadow_correct: int,
    ) -> None:
        placeholder = "%s" if self.is_postgres else "?"
        self.conn.execute(
            f"""
            UPDATE predictions
            SET outcome = {placeholder}, shadow_side = {placeholder},
                shadow_result = {placeholder}, shadow_correct = {placeholder}
            WHERE id = {placeholder}
            """,
            (outcome, shadow_side, shadow_result, shadow_correct, prediction_id),
        )
        self.conn.commit()

    def unsettled_shadow_predictions(self, now_iso: str, limit: int = 500) -> list[Any]:
        if self.is_postgres:
            return list(
                self.conn.execute(
                    """
                    SELECT id, p_up, p_down, end_time, features, model_side
                    FROM predictions
                    WHERE outcome IS NULL AND end_time <= %s
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (now_iso, limit),
                )
            )
        return list(
            self.conn.execute(
                """
                SELECT id, p_up, p_down, end_time, features, model_side
                FROM predictions
                WHERE outcome IS NULL AND end_time <= ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (now_iso, limit),
            )
        )

    def last_predictions(self, limit: int = 20) -> list[sqlite3.Row]:
        if self.is_postgres:
            return list(self.conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT %s", (limit,)))
        return list(self.conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)))

    def compute_metrics(self) -> Metrics:
        rows = list(
            self.conn.execute(
                """
                SELECT p_up, chosen_side, result, pnl, expected_value
                FROM predictions
                WHERE chosen_side IN ('UP', 'DOWN') AND result IN ('WIN', 'LOSS')
                ORDER BY id ASC
                """
            )
        )
        trade_count = len(rows)
        wins = [1 if row["result"] == "WIN" else 0 for row in rows]
        pnl = [float(row["pnl"] or 0.0) for row in rows]
        brier_values = []
        calibration_errors = []
        ev_diff = []
        for row in rows:
            y = 1.0 if row["result"] == "WIN" and row["chosen_side"] == "UP" else 0.0
            if row["chosen_side"] == "DOWN":
                y = 1.0 if row["result"] == "LOSS" else 0.0
            p_up = float(row["p_up"])
            brier_values.append((p_up - y) ** 2)
            calibration_errors.append(abs(p_up - y))
            if row["expected_value"] is not None and row["pnl"] is not None:
                ev_diff.append(float(row["pnl"]) - float(row["expected_value"]))
        gross_profit = sum(value for value in pnl if value > 0)
        gross_loss = abs(sum(value for value in pnl if value < 0))
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for value in pnl:
            equity += value
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        rolling_50 = _mean(wins[-50:]) if trade_count >= 50 else None
        return Metrics(
            rolling_winrate_20=_mean(wins[-20:]) if trade_count >= 20 else None,
            rolling_winrate_50=rolling_50,
            brier_score=_mean(brier_values) if brier_values else None,
            calibration_error=_mean(calibration_errors) if calibration_errors else None,
            profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else (math.inf if gross_profit > 0 else None),
            max_drawdown=max_dd,
            average_ev_vs_actual_pnl=_mean(ev_diff) if ev_diff else None,
            trade_count=trade_count,
            safe_mode_required=(rolling_50 is not None and rolling_50 < 0.60),
        )

    def compute_shadow_metrics(self) -> ShadowMetrics:
        rows = list(
            self.conn.execute(
                """
                SELECT shadow_correct
                FROM predictions
                WHERE shadow_result IN ('WIN', 'LOSS')
                ORDER BY id ASC
                """
            )
        )
        total = len(rows)
        correct = sum(1 for row in rows if int(row["shadow_correct"] or 0) == 1)
        return ShadowMetrics(
            predictions=total,
            correct=correct,
            winrate=(correct / total if total else None),
        )


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
