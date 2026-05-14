from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import load_settings
from database import BotDatabase
from models import calculate_edge_ev


@dataclass
class BacktestSummary:
    trades: int
    wins: int
    pnl: float
    winrate: float


@dataclass
class EdgeWinrateReport:
    hours: float
    predictions: int
    skips: int
    trades: int
    wins: int
    losses: int
    winrate: float | None
    average_edge: float | None
    average_ev: float | None
    total_pnl: float
    profit_factor: float | None
    max_drawdown: float
    brier_score: float | None
    safe_mode_required: bool
    shadow_predictions: int = 0
    shadow_correct: int = 0
    shadow_winrate: float | None = None
    unique_markets: int = 0
    entry_window_scans: int = 0
    actionable_skips: int = 0


def simple_prediction_log_backtest(path: str | Path, starting_balance: float = 1000.0) -> BacktestSummary:
    """Replay a CSV prediction log with columns p_up, price_up, price_down, result_price_move."""
    balance = starting_balance
    wins = 0
    trades = 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            p_up = float(row["p_up"])
            price_up = float(row["price_up"])
            price_down = float(row["price_down"])
            ev_up, ev_down, edge_up, edge_down = calculate_edge_ev(p_up, price_up, price_down)
            side = "UP" if ev_up > ev_down and edge_up >= 0.16 else "DOWN" if ev_down > 0 and edge_down >= 0.16 else "SKIP"
            if side == "SKIP":
                continue
            stake = min(balance * 0.03, balance * 0.10)
            market_up = int(row["result_price_move"]) > 0
            won = (side == "UP" and market_up) or (side == "DOWN" and not market_up)
            entry = price_up if side == "UP" else price_down
            pnl = stake * ((1 - entry) / entry) if won else -stake
            balance += pnl
            wins += int(won)
            trades += 1
    return BacktestSummary(trades=trades, wins=wins, pnl=balance - starting_balance, winrate=(wins / trades if trades else 0.0))


async def run_paper_forward_test(hours: float = 24.0) -> None:
    """Run the live bot in forced paper mode for edge/winrate validation."""
    os.environ["REAL_TRADING"] = "false"
    from main import BotRuntime

    runtime = BotRuntime()
    await runtime.run(hours=hours, settle_open_positions=True)


def build_edge_winrate_report(database_url: str, hours: float = 24.0) -> EdgeWinrateReport:
    db = BotDatabase(database_url)
    try:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        placeholder = "%s" if db.is_postgres else "?"
        rows = list(
            db.conn.execute(
                f"""
                SELECT timestamp, slug, p_up, chosen_side, result, pnl, edge, expected_value, reason, features
                FROM predictions
                WHERE timestamp >= {placeholder}
                ORDER BY id ASC
                """,
                (since,),
            )
        )
        shadow = db.compute_shadow_metrics()
    finally:
        db.close()

    predictions = len(rows)
    skips = sum(1 for row in rows if _row(row, "chosen_side") == "SKIP")
    early_or_late_skips = sum(
        1
        for row in rows
        if _row(row, "chosen_side") == "SKIP"
        and (
            "Too early" in str(_row(row, "reason") or "")
            or "Too close" in str(_row(row, "reason") or "")
        )
    )
    entry_window_scans = sum(1 for row in rows if 20 <= _seconds_to_expiry(row) <= 150)
    trades = [row for row in rows if _row(row, "chosen_side") in {"UP", "DOWN"} and _row(row, "result") in {"WIN", "LOSS"}]
    wins = sum(1 for row in trades if _row(row, "result") == "WIN")
    losses = sum(1 for row in trades if _row(row, "result") == "LOSS")
    pnl = [float(_row(row, "pnl") or 0.0) for row in trades]
    gross_profit = sum(value for value in pnl if value > 0)
    gross_loss = abs(sum(value for value in pnl if value < 0))
    brier = []
    for row in trades:
        chosen_side = _row(row, "chosen_side")
        result = _row(row, "result")
        actual_up = 1.0 if (chosen_side == "UP" and result == "WIN") or (chosen_side == "DOWN" and result == "LOSS") else 0.0
        brier.append((float(_row(row, "p_up") or 0.5) - actual_up) ** 2)

    return EdgeWinrateReport(
        hours=hours,
        predictions=predictions,
        skips=skips,
        trades=len(trades),
        wins=wins,
        losses=losses,
        winrate=(wins / len(trades) if trades else None),
        average_edge=_mean_float([_row(row, "edge") for row in trades]),
        average_ev=_mean_float([_row(row, "expected_value") for row in trades]),
        total_pnl=sum(pnl),
        profit_factor=(gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else None)),
        max_drawdown=_max_drawdown(pnl),
        brier_score=_mean_float(brier),
        safe_mode_required=(len(trades) >= 50 and (wins / len(trades)) < 0.60),
        shadow_predictions=shadow.predictions,
        shadow_correct=shadow.correct,
        shadow_winrate=shadow.winrate,
        unique_markets=len({str(_row(row, "slug") or "") for row in rows if _row(row, "slug")}),
        entry_window_scans=entry_window_scans,
        actionable_skips=max(0, skips - early_or_late_skips),
    )


def print_report(report: EdgeWinrateReport) -> None:
    print("\n[24H EDGE / WINRATE REPORT]")
    print(f"Window hours: {report.hours:g}")
    print(f"Predictions logged: {report.predictions}")
    print(f"Unique markets: {report.unique_markets}")
    print(f"Entry-window scans: {report.entry_window_scans}")
    print(f"Skips: {report.skips}")
    print(f"Actionable skips: {report.actionable_skips}")
    print(f"Settled trades: {report.trades}")
    print(f"Wins: {report.wins}")
    print(f"Losses: {report.losses}")
    print(f"Winrate: {_fmt_pct(report.winrate)}")
    print(f"Average edge: {_fmt_num(report.average_edge)}")
    print(f"Average EV: {_fmt_num(report.average_ev)}")
    print(f"Total paper PnL: {_fmt_num(report.total_pnl)}")
    print(f"Profit factor: {_fmt_num(report.profit_factor)}")
    print(f"Max drawdown: {_fmt_num(report.max_drawdown)}")
    print(f"Brier score: {_fmt_num(report.brier_score)}")
    print(f"SAFE_MODE required: {'YES' if report.safe_mode_required else 'NO'}")
    print(f"Shadow labeled predictions: {report.shadow_predictions}")
    print(f"Shadow directional WR: {_fmt_pct(report.shadow_winrate)} ({report.shadow_correct}/{report.shadow_predictions})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper forward-test and reporting tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    forward = subparsers.add_parser("forward", help="Run forced-paper live forward-test")
    forward.add_argument("--hours", type=float, default=24.0)

    report = subparsers.add_parser("report", help="Print edge/winrate report from the database")
    report.add_argument("--hours", type=float, default=24.0)
    report.add_argument("--database-url", default=None)

    args = parser.parse_args()
    if args.command == "forward":
        asyncio.run(run_paper_forward_test(hours=args.hours))
    elif args.command == "report":
        settings = load_settings()
        print_report(build_edge_winrate_report(args.database_url or settings.database_url, hours=args.hours))


def _row(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return row[key]


def _seconds_to_expiry(row: Any) -> float:
    raw = _row(row, "features")
    if not raw:
        return 0.0
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return 0.0
    try:
        return float(parsed.get("seconds_to_expiry") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _mean_float(values: list[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _max_drawdown(pnl: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == float("inf"):
        return "inf"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
