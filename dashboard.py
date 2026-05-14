from __future__ import annotations

import html
import json
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from config import load_settings
from database import BotDatabase


EDGE_TARGET = 0.16
HOT_EDGE_TARGET = 0.16
WR_TARGET = 0.80
CONFIDENCE_TARGET = 85.0
PROFIT_TARGET_MIN = 8.0
PROFIT_TARGET_MAX = 60.0
PRESETS = [100, 200, 500, 1000]


def safe(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def pct(value: float | None, decimals: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{decimals}f}%"


def num(value: float | None, decimals: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{decimals}f}"


def parse_features(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def month_label(timestamp: Any) -> str:
    if not timestamp:
        return "Live"
    months = {
        1: "Januari",
        2: "Februari",
        3: "Maret",
        4: "April",
        5: "Mei",
        6: "Juni",
        7: "Juli",
        8: "Agustus",
        9: "September",
        10: "Oktober",
        11: "November",
        12: "Desember",
    }
    try:
        dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        return f"{dt.day} {months.get(dt.month, dt.month)} {dt.year}"
    except ValueError:
        return "Live"


def metric_card(label: str, value: str, sub: str, color: str) -> str:
    return f"""
    <div class="v7-card">
      <div class="v7-card-value" style="color:{color}">{safe(value)}</div>
      <div class="v7-card-label">{safe(label)}</div>
      <div class="v7-card-sub">{safe(sub)}</div>
    </div>
    """


def verdict_card(title: str, actual: str, target: str, ok: bool, note: str) -> str:
    color = "#10b981" if ok else "#f59e0b"
    icon = "✅" if ok else "⚠️"
    return f"""
    <div class="v7-verdict" style="border-left-color:{color}">
      <div class="v7-verdict-head">
        <span>{safe(title)}</span>
        <b style="color:{color}">{icon} {safe(actual)} / {safe(target)}</b>
      </div>
      <div class="v7-verdict-note">{safe(note)}</div>
    </div>
    """


def signal_color(side: str) -> str:
    if side == "UP":
        return "#10b981"
    if side == "DOWN":
        return "#ef4444"
    return "#475569"


def pwin_for_row(row: pd.Series) -> float:
    side = str(row.get("chosen_side", "SKIP"))
    if side == "UP":
        return float(row.get("p_up") or 0.0)
    if side == "DOWN":
        return float(row.get("p_down") or 0.0)
    return 0.0


def profit_pct_for_row(row: pd.Series) -> float:
    entry = float(row.get("entry_price") or 0.0)
    if entry <= 0:
        side = str(row.get("chosen_side", "SKIP"))
        entry = float(row.get("price_up") or 0.0) if side == "UP" else float(row.get("price_down") or 0.0)
    if entry <= 0:
        return 0.0
    return ((1.0 - entry) / entry) * 100.0


def reason_counts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "chosen_side" not in frame or "reason" not in frame:
        return pd.DataFrame(columns=["Reason", "Count"])
    skipped = frame[frame["chosen_side"].eq("SKIP")]
    if skipped.empty:
        return pd.DataFrame(columns=["Reason", "Count"])
    counts = skipped["reason"].fillna("No reason").value_counts()
    return pd.DataFrame({"Reason": counts.index, "Count": counts.values})


st.set_page_config(page_title="Bot v7 Backtesting Scorecard", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background:#0f172a; color:#e2e8f0; }
      div[data-testid="stHeader"] { background:transparent; }
      .block-container { max-width:980px; padding:14px 0 32px; }
      h1, h2, h3, p { margin:0; }
      .v7-title { color:#38bdf8; font-size:23px; font-weight:500; margin-bottom:4px; }
      .v7-subtitle { color:#64748b; font-size:16px; margin-bottom:16px; }
      .bankroll-row { display:flex; align-items:center; gap:8px; margin:4px 0 14px; color:#64748b; }
      .bankroll-pill {
        background:#1e293b; color:#94a3b8; border-radius:7px; padding:9px 20px;
        min-width:74px; text-align:center; font-weight:500;
      }
      .bankroll-pill.active { background:#38bdf8; color:#020617; font-weight:800; }
      .v7-hero {
        background:linear-gradient(135deg,#0c1e34,#172338);
        border:1px solid #1e4972; border-radius:14px; padding:24px 22px 22px; margin-bottom:16px;
      }
      .v7-section-label { color:#64748b; font-size:16px; font-weight:800; margin-bottom:14px; }
      .v7-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:11px; margin-bottom:11px; }
      .v7-card { background:#0f172a; border-radius:9px; padding:22px 16px 15px; min-height:112px; }
      .v7-card-value { font-size:36px; font-weight:850; line-height:1; margin-bottom:10px; }
      .v7-card-label { color:#a8b3c7; font-size:16px; }
      .v7-card-sub { color:#475569; font-size:13px; margin-top:5px; }
      .v7-tabs {
        display:flex; gap:7px; flex-wrap:wrap; margin:8px 0 16px;
      }
      .v7-tab {
        background:#1e293b; color:#a8b3c7; border-radius:7px; padding:11px 21px; font-size:16px;
      }
      .v7-tab.active { background:#38bdf8; color:#020617; font-weight:800; }
      .v7-panel { background:#1e293b; border-radius:10px; padding:24px 22px; margin-bottom:14px; }
      .v7-panel-title { color:#38bdf8; font-size:18px; font-weight:850; margin-bottom:18px; }
      .v7-verdict-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
      .v7-verdict { background:#0f172a; border-radius:9px; border-left:4px solid #f59e0b; padding:18px 16px; min-height:112px; }
      .v7-verdict-head { display:flex; justify-content:space-between; gap:10px; color:#e5e7eb; font-size:17px; font-weight:850; }
      .v7-verdict-note { color:#64748b; font-size:15px; margin-top:12px; line-height:1.55; }
      .signal-card { background:#1e293b; border-radius:10px; padding:14px; margin-bottom:10px; border-left:4px solid #38bdf8; }
      .hot-card { background:#092414; border:1px solid #10b98155; border-radius:10px; padding:15px; margin-bottom:10px; }
      .badge { display:inline-block; border-radius:4px; padding:3px 9px; font-size:12px; font-weight:800; color:#fff; margin-right:5px; }
      .muted { color:#64748b; }
      .tiny { color:#475569; font-size:11px; margin-top:4px; }
      div[data-testid="stButton"] > button {
        background:#1e293b; color:#94a3b8; border:0; border-radius:7px; padding:9px 20px; min-width:74px;
      }
      div[data-testid="stButton"] > button:hover { background:#24344d; color:#dbeafe; border:0; }
      .stDataFrame { border-radius:8px; overflow:hidden; }
      @media (max-width:900px) {
        .block-container { padding-left:10px; padding-right:10px; }
        .v7-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .v7-verdict-grid { grid-template-columns:1fr; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = load_settings()
db = BotDatabase(settings.database_url)
metrics = db.compute_metrics()
shadow_metrics = db.compute_shadow_metrics()
rows = db.last_predictions(500)
db.close()

frame = pd.DataFrame([dict(row) for row in rows])

if frame.empty:
    settled = pd.DataFrame()
    signals = pd.DataFrame()
    skipped = pd.DataFrame()
    shadow = pd.DataFrame()
    hot = pd.DataFrame()
    latest = None
else:
    for col in ["edge", "expected_value", "confidence_score", "pnl", "position_size", "p_up", "p_down", "price_up", "price_down", "entry_price"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    settled = frame[frame["result"].isin(["WIN", "LOSS"])].copy()
    signals = frame[frame["chosen_side"].isin(["UP", "DOWN"])].copy()
    skipped = frame[frame["chosen_side"].eq("SKIP")].copy()
    shadow = frame[frame["shadow_result"].isin(["WIN", "LOSS"])].copy() if "shadow_result" in frame else pd.DataFrame()
    hot = signals[signals["edge"] >= HOT_EDGE_TARGET].copy()
    latest = frame.iloc[0]

scan_count = len(frame)
settled_wins = int(settled["result"].eq("WIN").sum()) if not settled.empty else 0
settled_losses = int(settled["result"].eq("LOSS").sum()) if not settled.empty else 0
settled_wr = settled_wins / len(settled) if len(settled) else None
avg_edge = float(signals["edge"].mean()) if not signals.empty else 0.0
avg_wr_est = float(signals.apply(pwin_for_row, axis=1).mean()) if not signals.empty else 0.0
total_bet = float(signals["position_size"].sum()) if "position_size" in signals else 0.0
total_ev = float((signals["position_size"] * signals["expected_value"]).sum()) if not signals.empty and "position_size" in signals else 0.0
total_pnl = float(frame["pnl"].sum()) if not frame.empty else 0.0
on_target = signals[signals.apply(lambda row: PROFIT_TARGET_MIN <= profit_pct_for_row(row) <= PROFIT_TARGET_MAX, axis=1)] if not signals.empty else signals
shadow_wr = shadow_metrics.winrate
date_label = month_label(latest.get("timestamp") if latest is not None else None)
safe_mode = metrics.safe_mode_required

st.markdown(
    f"""
    <div class="v7-title">📡 Bot v7 — Backtesting Data Live {safe(date_label)}</div>
    <div class="v7-subtitle">{scan_count} market real dari Polymarket · BTC 5M live · Paper forward-test</div>
    """,
    unsafe_allow_html=True,
)

selected_bankroll = 200
bankroll_html = "<div class='bankroll-row'><span>Bankroll:</span>" + "".join(
    f"<span class='bankroll-pill {'active' if value == selected_bankroll else ''}'>${value}</span>"
    for value in PRESETS
) + "</div>"
st.markdown(bankroll_html, unsafe_allow_html=True)

edge_pass = avg_edge >= EDGE_TARGET
wr_pass = avg_wr_est >= WR_TARGET
hot_pass = len(hot) >= 1

st.markdown(
    f"""
    <div class="v7-hero">
      <div class="v7-section-label">📊 HASIL BACKTESTING — {safe(date_label)} · {scan_count} MARKET DISCAN</div>
      <div class="v7-grid">
        {metric_card("Signal Lolos", str(len(signals)), f"dari {scan_count} market", "#10b981")}
        {metric_card("Avg Edge v7", pct(avg_edge), "✅ Target tercapai" if edge_pass else "⚠️ Mendekati target", "#10b981" if edge_pass else "#f59e0b")}
        {metric_card("Avg WR Est", pct(avg_wr_est, 0), "✅ Di atas target" if wr_pass else "⚠️ Di bawah target", "#10b981" if wr_pass else "#f59e0b")}
        {metric_card(f"Signal HOT ≥{int(HOT_EDGE_TARGET * 100)}%", str(len(hot)), "edge di atas threshold", "#f59e0b")}
      </div>
      <div class="v7-grid">
        {metric_card("Shadow WR", pct(shadow_wr), f"{shadow_metrics.correct}/{shadow_metrics.predictions} prediksi berlabel", "#10b981" if (shadow_wr or 0) >= 0.60 else "#a78bfa")}
        {metric_card("Total Bet/scan", money(total_bet), f"dari ${selected_bankroll} bankroll", "#38bdf8")}
        {metric_card("Total EV/scan", money(total_ev), "expected value per scan", "#10b981" if total_ev >= 0 else "#ef4444")}
        {metric_card("Diblok filter", str(len(skipped)), "Season/Resolved/low vol", "#ef4444")}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

score_tab, signals_tab, hot_tab, strategy_tab, skip_tab = st.tabs(
    [
        "🏆 Scorecard",
        f"🎯 Signals ({len(signals)})",
        f"🔥 Hot ≥{int(HOT_EDGE_TARGET * 100)}% ({len(hot)})",
        "📊 Per Strategi",
        f"🚫 Skip ({len(skipped)})",
    ]
)

with score_tab:
    st.markdown(
        f"""
        <div class="v7-panel">
          <div class="v7-panel-title">🏆 Verdict v7 vs Target — Data {safe(date_label)}</div>
          <div class="v7-verdict-grid">
            {verdict_card("Avg Edge", pct(avg_edge), f"≥ {int(EDGE_TARGET * 100)}%", edge_pass, "Risk manager hanya mengizinkan entry jika edge side terpilih minimal 16%.")}
            {verdict_card("WR Estimasi", pct(avg_wr_est, 0), f"≥ {int(WR_TARGET * 100)}%", wr_pass, "Estimasi model dipakai sebagai gate, bukan janji winrate. Validasi tetap dari settled paper trades.")}
            {verdict_card("Filter Akurasi", f"{len(skipped)}/{scan_count}", "Blok sinyal lemah", len(skipped) >= len(signals), "SKIP dipakai ketika EV, edge, spread, liquidity, feed, atau confidence tidak valid.")}
            {verdict_card("Shadow Learning", pct(shadow_wr), "monitor", shadow_metrics.predictions >= 20, "Semua prediksi diberi label setelah market selesai agar model online bisa belajar meski risk manager memilih SKIP.")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if latest is None:
        st.info("Belum ada prediksi tersimpan. Jalankan `python backtester.py forward --hours 24` dulu.")
    else:
        latest_features = parse_features(latest.get("features"))
        side = str(latest.get("chosen_side", "SKIP"))
        st.markdown('<div class="v7-panel-title">📡 Current BTC 5M Signal</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f"<span class='badge' style='background:{signal_color(side)}'>{safe(side)}</span>", unsafe_allow_html=True)
        c2.metric("p_up", num(float(latest.get("p_up") or 0), 4))
        c3.metric("p_down", num(float(latest.get("p_down") or 0), 4))
        c4.metric("edge", num(float(latest.get("edge") or 0), 4))
        c5.metric("confidence", num(float(latest.get("confidence_score") or 0), 1))
        st.caption(str(latest.get("reason") or ""))

        st.markdown('<div class="v7-panel-title">📋 Tabel Analisis Semua Market / Prediksi</div>', unsafe_allow_html=True)
        display_cols = [
            "timestamp",
            "slug",
            "p_up",
            "p_down",
            "price_up",
            "price_down",
            "chosen_side",
            "entry_price",
            "position_size",
            "edge",
            "expected_value",
            "confidence_score",
            "result",
            "outcome",
            "shadow_side",
            "shadow_result",
            "pnl",
            "reason",
        ]
        st.dataframe(frame[[col for col in display_cols if col in frame.columns]].head(80), use_container_width=True, hide_index=True)

with signals_tab:
    if signals.empty:
        st.info("Belum ada signal UP/DOWN yang lolos risk manager.")
    else:
        ordered = signals.copy()
        ordered["p_win"] = ordered.apply(pwin_for_row, axis=1)
        ordered["profit_pct"] = ordered.apply(profit_pct_for_row, axis=1)
        for _, row in ordered.sort_values(["edge", "confidence_score"], ascending=False).head(100).iterrows():
            side = str(row.get("chosen_side", "SKIP"))
            color = signal_color(side)
            edge = float(row.get("edge") or 0.0)
            st.markdown(
                f"""
                <div class="signal-card" style="border-left-color:{color}">
                  <div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;">
                    <div>
                      <span class="badge" style="background:{color}">{safe(side)} @{num(float(row.get('entry_price') or 0), 4)}</span>
                      <span class="badge" style="background:#1e3a5f;color:#38bdf8">Q {num(float(row.get('confidence_score') or 0), 1)}</span>
                      <span class="badge" style="background:#1a1f2e;color:#a78bfa">WR~{pct(float(row.get('p_win') or 0), 0)}</span>
                    </div>
                    <b style="color:{'#10b981' if edge >= EDGE_TARGET else '#f59e0b'}">Edge {pct(edge)}</b>
                  </div>
                  <div style="font-size:15px;color:#e2e8f0;font-weight:800;margin-top:8px">{safe(row.get('slug'))}</div>
                  <div class="muted" style="font-size:12px;margin-top:5px">
                    Bet {money(float(row.get('position_size') or 0))} · EV/unit {num(float(row.get('expected_value') or 0), 4)}
                    · Profit WIN {num(float(row.get('profit_pct') or 0), 1)}% · PnL {money(float(row.get('pnl') or 0))}
                  </div>
                  <div class="tiny">{safe(row.get('reason'))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with hot_tab:
    if hot.empty:
        st.warning(f"Belum ada signal HOT dengan edge ≥ {EDGE_TARGET:.0%}.")
    else:
        for _, row in hot.sort_values(["edge", "confidence_score"], ascending=False).head(50).iterrows():
            side = str(row.get("chosen_side", "SKIP"))
            st.markdown(
                f"""
                <div class="hot-card">
                  <div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;">
                    <div>
                      <span class="badge" style="background:{signal_color(side)}">BET {safe(side)}</span>
                      <span class="badge" style="background:#0f172a;color:#38bdf8">ENTRY {num(float(row.get('entry_price') or 0), 4)}</span>
                    </div>
                    <div style="color:#10b981;font-weight:900;font-size:18px">EDGE {pct(float(row.get('edge') or 0))}</div>
                  </div>
                  <div style="font-size:15px;color:#e2e8f0;font-weight:800;margin-top:8px">{safe(row.get('slug'))}</div>
                  <div class="v7-grid" style="margin-top:10px">
                    {metric_card("Bet", money(float(row.get('position_size') or 0)), "risk capped", "#38bdf8")}
                    {metric_card("Confidence", num(float(row.get('confidence_score') or 0), 1), "high gate", "#10b981")}
                    {metric_card("EV/unit", num(float(row.get('expected_value') or 0), 4), "after costs", "#a78bfa")}
                    {metric_card("PnL", money(float(row.get('pnl') or 0)), str(row.get('result') or "OPEN"), "#10b981" if float(row.get('pnl') or 0) >= 0 else "#ef4444")}
                  </div>
                  <div class="tiny">{safe(row.get('reason'))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with strategy_tab:
    st.markdown('<div class="v7-panel-title">📊 Ringkasan Per Strategi / Model</div>', unsafe_allow_html=True)
    if signals.empty:
        st.info("Belum ada trade untuk diringkas.")
    else:
        by_side = signals.groupby("chosen_side").agg(
            n=("chosen_side", "count"),
            avg_edge=("edge", "mean"),
            avg_conf=("confidence_score", "mean"),
            total_bet=("position_size", "sum"),
            total_ev=("expected_value", "mean"),
            pnl=("pnl", "sum"),
        ).reset_index()
        st.dataframe(by_side, use_container_width=True, hide_index=True)
    latest_features = parse_features(latest.get("features") if latest is not None else None)
    model_components = latest_features.get("model_components", {})
    if model_components:
        st.markdown("Model components terbaru")
        st.json(model_components)

    st.markdown('<div class="v7-panel-title">📐 Apakah Target 16% & 80% WR Tercapai?</div>', unsafe_allow_html=True)
    target_rows = pd.DataFrame(
        [
            {"Metrik": "Avg Edge", "Target": ">=16%", "Hasil": pct(avg_edge), "Status": "PASS" if edge_pass else "WATCH"},
            {"Metrik": "Avg WR Est", "Target": ">=80%", "Hasil": pct(avg_wr_est, 0), "Status": "PASS" if wr_pass else "WATCH"},
            {"Metrik": "Signal HOT", "Target": ">=1", "Hasil": str(len(hot)), "Status": "PASS" if hot_pass else "WATCH"},
            {"Metrik": "Settled WR", "Target": "monitor", "Hasil": pct(settled_wr), "Status": "INFO"},
            {"Metrik": "Shadow WR", "Target": "monitor", "Hasil": pct(shadow_wr), "Status": "INFO"},
            {"Metrik": "SAFE_MODE", "Target": "OFF", "Hasil": "ON" if safe_mode else "OFF", "Status": "WATCH" if safe_mode else "PASS"},
        ]
    )
    st.dataframe(target_rows, use_container_width=True, hide_index=True)

with skip_tab:
    st.markdown('<div class="v7-panel-title">🚫 Market / Prediksi Diblok Filter</div>', unsafe_allow_html=True)
    rc = reason_counts(frame)
    if rc.empty:
        st.info("Belum ada SKIP.")
    else:
        st.dataframe(rc, use_container_width=True, hide_index=True)

    for _, row in skipped.head(100).iterrows():
        st.markdown(
            f"""
            <div class="signal-card" style="opacity:.68;border-left-color:#334155">
              <div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;">
                <b style="color:#94a3b8">{safe(row.get('slug'))}</b>
                <span class="badge" style="background:#334155">SKIP</span>
              </div>
              <div style="color:#ef4444;font-size:12px;margin-top:5px">{safe(row.get('reason'))}</div>
              <div class="tiny">edge {pct(float(row.get('edge') or 0))} · EV {num(float(row.get('expected_value') or 0), 4)} · confidence {num(float(row.get('confidence_score') or 0), 1)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
