# ============================================================
# QuantBengal Pro SMC Terminal  |  app.py
# Senior Quant Dev & Market Microstructure Analyst build
# Strategy: Smart Money Concepts (SMC) Liquidity Sweep Detection
# Stack: Python 3.11+ | Streamlit | Plotly | Pandas | yfinance
# Architecture: Single-file. No local imports.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# PAGE CONFIG  — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantBengal Pro | SMC Terminal",
    page_icon="🐯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [data-testid="stAppViewContainer"] {
      background-color: #0a0e14;
      color: #cdd6f4;
      font-family: 'JetBrains Mono', 'Courier New', monospace;
  }
  [data-testid="stSidebar"] {
      background-color: #0d1117;
      border-right: 1px solid #1e2a3a;
  }
  .terminal-header {
      background: linear-gradient(90deg, #0d1117 0%, #0f2027 60%, #0d1117 100%);
      border-bottom: 1px solid #1e6fa5;
      padding: 14px 24px 10px 24px;
      display: flex; align-items: baseline; gap: 12px;
  }
  .terminal-header h1 {
      font-size: 1.45rem; font-weight: 700;
      letter-spacing: 0.08em; color: #89dceb; margin: 0;
  }
  .terminal-header span {
      font-size: 0.72rem; color: #6c7086;
      letter-spacing: 0.15em; text-transform: uppercase;
  }
  .kpi-grid { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
  .kpi-tile {
      background: #0d1117; border: 1px solid #1e2a3a;
      border-radius: 6px; padding: 10px 18px; min-width: 140px; flex: 1;
  }
  .kpi-tile .label {
      font-size: 0.65rem; color: #6c7086;
      letter-spacing: 0.12em; text-transform: uppercase;
  }
  .kpi-tile .value { font-size: 1.35rem; font-weight: 700; color: #cdd6f4; margin-top: 2px; }
  .kpi-tile .value.up   { color: #a6e3a1; }
  .kpi-tile .value.down { color: #f38ba8; }
  .kpi-tile .value.neutral { color: #89dceb; }
  .badge { display: inline-block; border-radius: 4px; padding: 2px 8px;
           font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; }
  .badge.buy  { background: #1a3a2a; color: #a6e3a1; border: 1px solid #a6e3a1; }
  .badge.sell { background: #3a1a2a; color: #f38ba8; border: 1px solid #f38ba8; }
  .badge.none { background: #1a1e2a; color: #6c7086; border: 1px solid #313244; }
  .signal-table { font-size: 0.78rem; }
  [data-testid="stDataFrame"] { border: 1px solid #1e2a3a; border-radius: 6px; }
  [data-testid="stTab"] { color: #6c7086 !important; font-size: 0.82rem; }
  [aria-selected="true"] { color: #89dceb !important; border-bottom: 2px solid #89dceb !important; }
  hr { border-color: #1e2a3a; margin: 8px 0; }
  .bt-stat-grid { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
  .bt-stat {
      background: #0d1117; border: 1px solid #1e2a3a;
      border-radius: 6px; padding: 8px 14px; min-width: 120px; flex: 1;
  }
  .bt-stat .s-label { font-size: 0.62rem; color: #6c7086; letter-spacing: 0.1em; text-transform: uppercase; }
  .bt-stat .s-value { font-size: 1.1rem; font-weight: 700; color: #cdd6f4; margin-top: 2px; }
  .bt-stat .s-value.pos { color: #a6e3a1; }
  .bt-stat .s-value.neg { color: #f38ba8; }
  .bt-stat .s-value.neu { color: #89dceb; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ROLLING_WINDOW: int   = 20
VOLUME_MULT: float    = 1.2
RR_RATIO: float       = 2.5
LOOKBACK_DAYS: int    = 30
INTERVAL: str         = "15m"
BT_TARGET_PTS: float  = 100.0
BT_SL_PTS: float      = 40.0
BT_MAX_BARS: int      = 96

ASSET_MAP: dict = {
    "BankNifty (^NSEBANK)": "^NSEBANK",
    "Nifty 50 (^NSEI)":     "^NSEI",
    "Nifty IT (^CNXIT)":    "^CNXIT",
    "S&P 500 (^GSPC)":      "^GSPC",
    "NASDAQ 100 (^NDX)":    "^NDX",
    "Gold (GC=F)":          "GC=F",
    "Crude Oil (CL=F)":     "CL=F",
    "Bitcoin (BTC-USD)":    "BTC-USD",
}


# ─────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────
def fetch_ohlcv(ticker: str, days: int = LOOKBACK_DAYS, interval: str = INTERVAL) -> pd.DataFrame:
    """
    Download OHLCV via yfinance with three safeguards:
      1. MultiIndex column flattening (yfinance >= 0.2.x)
      2. Minimum 50-row check before rolling calculations
      3. UTC-naive timezone normalisation for Plotly compatibility
    Bug fix: removed threads=False (removed in yfinance 0.2.38+).
    """
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)

    raw = yf.download(
        tickers=ticker,
        start=start_dt,
        end=end_dt,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        return pd.DataFrame()

    # Safeguard 1: flatten MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.strip().capitalize() for c in raw.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(raw.columns)):
        return pd.DataFrame()

    df = raw[list(required)].copy()
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    # Safeguard 3: timezone normalisation
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    # Safeguard 2: minimum rows
    if len(df) < 50:
        return pd.DataFrame()

    return df


# ─────────────────────────────────────────────
# SMC SIGNAL ENGINE
# ─────────────────────────────────────────────
def compute_smc_signals(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    vol_mult: float = VOLUME_MULT,
    rr_ratio: float = RR_RATIO,
) -> pd.DataFrame:
    """
    Vectorised SMC Liquidity Sweep detection.
    shift(1) on all rolling calculations prevents lookahead bias.
    """
    out = df.copy()

    out["support"]      = out["Low"].rolling(window).min().shift(1)
    out["resistance"]   = out["High"].rolling(window).max().shift(1)
    out["vol_sma"]      = out["Volume"].rolling(window).mean().shift(1)
    out["vol_threshold"]= out["vol_sma"] * vol_mult

    high_vol  = out["Volume"] > out["vol_threshold"]
    bull_mask = (out["Low"] < out["support"]) & (out["Close"] > out["support"]) & high_vol
    bear_mask = (out["High"] > out["resistance"]) & (out["Close"] < out["resistance"]) & high_vol

    out["signal"] = np.where(bull_mask, "BUY", np.where(bear_mask, "SELL", ""))

    buy_sl  = out["support"]
    buy_tp  = out["Close"] + (out["Close"] - buy_sl) * rr_ratio
    sell_sl = out["resistance"]
    sell_tp = out["Close"] - (sell_sl - out["Close"]) * rr_ratio

    out["sl"] = np.where(bull_mask, buy_sl,  np.where(bear_mask, sell_sl, np.nan))
    out["tp"] = np.where(bull_mask, buy_tp,  np.where(bear_mask, sell_tp, np.nan))

    out["candle_range"] = out["High"] - out["Low"]
    out["rel_volume"]   = (out["Volume"] / out["vol_sma"]).round(2)
    out["prev_close"]   = out["Close"].shift(1)

    return out


# ─────────────────────────────────────────────
# CHARTING ENGINE
# ─────────────────────────────────────────────
def build_chart(df: pd.DataFrame, ticker: str, n_candles: int = 100) -> go.Figure:
    """
    Dual-panel chart: candlestick + SMC overlays (panel 1), volume (panel 2).
    Slices to last n_candles before building — full history used for signal detection.
    Marker offset is price-range-relative to self-calibrate across assets.
    """
    view = df.tail(n_candles).copy()
    mean_range: float   = float(view["candle_range"].mean())
    marker_offset: float = mean_range * 0.35

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.018,
    )

    # Layer 1 — Candlestick
    fig.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"],
        low=view["Low"], close=view["Close"], name="Price",
        increasing_line_color="#a6e3a1", decreasing_line_color="#f38ba8",
        increasing_fillcolor="#a6e3a1", decreasing_fillcolor="#f38ba8",
        whiskerwidth=0.3, line=dict(width=1), hoverinfo="x+y",
    ), row=1, col=1)

    # Layer 2 — Support
    fig.add_trace(go.Scatter(
        x=view.index, y=view["support"], mode="lines",
        name="Support (20p Low)",
        line=dict(color="#a6e3a1", width=1.5, dash="dash"),
        opacity=0.85, hovertemplate="Support: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Layer 3 — Resistance
    fig.add_trace(go.Scatter(
        x=view.index, y=view["resistance"], mode="lines",
        name="Resistance (20p High)",
        line=dict(color="#f38ba8", width=1.5, dash="dash"),
        opacity=0.85, hovertemplate="Resistance: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Layer 4 — BUY markers
    buys = view[view["signal"] == "BUY"]
    fig.add_trace(go.Scatter(
        x=buys.index,
        y=(buys["Low"] - marker_offset) if not buys.empty else pd.Series(dtype=float),
        mode="markers+text", name="Bullish Sweep ▲",
        marker=dict(symbol="triangle-up", size=15, color="#a6e3a1",
                    line=dict(color="#0d1117", width=1.5)),
        text=["BUY"] * len(buys), textposition="bottom center",
        textfont=dict(size=8, color="#a6e3a1", family="JetBrains Mono, monospace"),
        hovertemplate=(
            "<b>BULLISH SWEEP</b><br>Time : %{x}<br>"
            "Close: %{customdata[0]:.2f}<br>SL: %{customdata[1]:.2f}<br>"
            "TP: %{customdata[2]:.2f}<br>RelVol: %{customdata[3]:.2f}x<extra></extra>"
        ),
        customdata=buys[["Close","sl","tp","rel_volume"]].values if not buys.empty else np.empty((0,4)),
    ), row=1, col=1)

    # Layer 5 — SELL markers
    sells = view[view["signal"] == "SELL"]
    fig.add_trace(go.Scatter(
        x=sells.index,
        y=(sells["High"] + marker_offset) if not sells.empty else pd.Series(dtype=float),
        mode="markers+text", name="Bearish Sweep ▼",
        marker=dict(symbol="triangle-down", size=15, color="#f38ba8",
                    line=dict(color="#0d1117", width=1.5)),
        text=["SELL"] * len(sells), textposition="top center",
        textfont=dict(size=8, color="#f38ba8", family="JetBrains Mono, monospace"),
        hovertemplate=(
            "<b>BEARISH SWEEP</b><br>Time : %{x}<br>"
            "Close: %{customdata[0]:.2f}<br>SL: %{customdata[1]:.2f}<br>"
            "TP: %{customdata[2]:.2f}<br>RelVol: %{customdata[3]:.2f}x<extra></extra>"
        ),
        customdata=sells[["Close","sl","tp","rel_volume"]].values if not sells.empty else np.empty((0,4)),
    ), row=1, col=1)

    # Layer 6 — SL / TP extension lines (legend-only by default)
    right_edge = view.index[-1]
    sl_x, sl_y, tp_x, tp_y = [], [], [], []
    for ts, bar in view[view["signal"] != ""].iterrows():
        if pd.notna(bar["sl"]):
            sl_x.extend([ts, right_edge, None]); sl_y.extend([bar["sl"], bar["sl"], None])
        if pd.notna(bar["tp"]):
            tp_x.extend([ts, right_edge, None]); tp_y.extend([bar["tp"], bar["tp"], None])
    if sl_x:
        fig.add_trace(go.Scatter(x=sl_x, y=sl_y, mode="lines", name="Stop Loss levels",
            line=dict(color="#f38ba8", width=0.9, dash="longdash"),
            opacity=0.55, visible="legendonly", hoverinfo="skip"), row=1, col=1)
    if tp_x:
        fig.add_trace(go.Scatter(x=tp_x, y=tp_y, mode="lines", name="Take Profit levels",
            line=dict(color="#a6e3a1", width=0.9, dash="longdash"),
            opacity=0.55, visible="legendonly", hoverinfo="skip"), row=1, col=1)

    # Layer 7 — Volume bars
    vol_colors = np.where(view["Close"] >= view["Open"], "#a6e3a1", "#f38ba8").tolist()
    fig.add_trace(go.Bar(
        x=view.index, y=view["Volume"], name="Volume",
        marker=dict(color=vol_colors, line=dict(width=0)),
        opacity=0.50, showlegend=False, hovertemplate="Vol: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    # Layer 8 — Volume SMA
    fig.add_trace(go.Scatter(
        x=view.index, y=view["vol_sma"], mode="lines",
        name=f"Vol SMA ({ROLLING_WINDOW})",
        line=dict(color="#89dceb", width=1.4),
        hovertemplate="Vol SMA: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    t_start    = view.index[0].strftime("%d %b")
    t_end      = view.index[-1].strftime("%d %b '%y")
    n_buy_vis  = int((view["signal"] == "BUY").sum())
    n_sell_vis = int((view["signal"] == "SELL").sum())

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14",
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(
            text=(f"<b style='color:#89dceb'>{ticker}</b>"
                  f"<span style='color:#6c7086'>  ·  SMC Liquidity Sweep  ·  {INTERVAL}  ·  "
                  f"last {n_candles} bars  ({t_start} to {t_end})  ·  "
                  f"<span style='color:#a6e3a1'>{n_buy_vis} BUY</span>  "
                  f"<span style='color:#f38ba8'>{n_sell_vis} SELL</span></span>"),
            font=dict(family="JetBrains Mono, monospace", size=12), x=0.005, xanchor="left",
        ),
        legend=dict(orientation="h", x=0, y=1.055,
                    font=dict(size=10, color="#6c7086", family="JetBrains Mono, monospace"),
                    bgcolor="rgba(0,0,0,0)", itemclick="toggle", itemdoubleclick="toggleothers"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0d1117", bordercolor="#1e2a3a", font_size=11,
                        font_family="JetBrains Mono, monospace"),
        xaxis=dict(rangeslider=dict(visible=False)),
    )
    fig.update_yaxes(row=1, col=1, gridcolor="#1a2235", gridwidth=0.5,
                     zerolinecolor="#1e2a3a", side="right", showgrid=True,
                     tickfont=dict(size=10, color="#6c7086"), tickformat=",.2f")
    fig.update_yaxes(row=2, col=1, gridcolor="#1a2235", gridwidth=0.5,
                     zerolinecolor="#1e2a3a", side="right", showgrid=True,
                     tickfont=dict(size=9, color="#6c7086"), tickformat=".2s")
    fig.update_xaxes(gridcolor="#1a2235", gridwidth=0.5, showgrid=False, zeroline=False,
                     tickfont=dict(size=9, color="#6c7086"),
                     rangeslider=dict(visible=False))
    return fig


# ─────────────────────────────────────────────
# KPI HELPER
# ─────────────────────────────────────────────
def build_kpi_html(df: pd.DataFrame) -> str:
    last   = df.iloc[-1]
    close  = float(last["Close"])
    open_  = float(df.iloc[-2]["Close"]) if len(df) > 1 else close
    chg    = close - open_
    pct    = (chg / open_) * 100 if open_ != 0 else 0.0
    dir_cls = "up" if chg >= 0 else "down"
    arrow   = "▲" if chg >= 0 else "▼"

    n_buy  = int((df["signal"] == "BUY").sum())
    n_sell = int((df["signal"] == "SELL").sum())
    sig_mask = df["signal"] != ""
    last_sig = df.loc[sig_mask, "signal"].iloc[-1] if sig_mask.any() else "—"
    sig_cls  = "up" if last_sig == "BUY" else ("down" if last_sig == "SELL" else "neutral")

    sup_val = f"{float(last['support']):.2f}" if pd.notna(last.get("support")) else "—"
    res_val = f"{float(last['resistance']):.2f}" if pd.notna(last.get("resistance")) else "—"
    rv_val  = f"{float(last['rel_volume']):.2f}x" if pd.notna(last.get("rel_volume")) else "—"

    tiles = [
        ("LTP",          f"{close:,.2f}",                    dir_cls),
        (f"Change {arrow}", f"{chg:+.2f} ({pct:+.2f}%)",   dir_cls),
        ("Last Signal",  last_sig,                            sig_cls),
        ("Buy Sweeps",   str(n_buy),                         "up"),
        ("Sell Sweeps",  str(n_sell),                        "down"),
        ("Support",      sup_val,                             "neutral"),
        ("Resistance",   res_val,                             "neutral"),
        ("Rel Volume",   rv_val,                              "neutral"),
    ]
    html = '<div class="kpi-grid">'
    for label, value, cls in tiles:
        html += (f'<div class="kpi-tile">'
                 f'<div class="label">{label}</div>'
                 f'<div class="value {cls}">{value}</div>'
                 f'</div>')
    html += '</div>'
    return html


# ─────────────────────────────────────────────
# SIGNAL LOG TABLE
# ─────────────────────────────────────────────
def build_signal_table(df: pd.DataFrame) -> pd.DataFrame:
    signals = df[df["signal"] != ""].copy()
    if signals.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Datetime":   signals.index.strftime("%Y-%m-%d %H:%M"),
        "Signal":     signals["signal"].values,
        "Close":      signals["Close"].round(2).values,
        "SL":         signals["sl"].round(2).values,
        "TP":         signals["tp"].round(2).values,
        "Support":    signals["support"].round(2).values,
        "Resistance": signals["resistance"].round(2).values,
        "Rel Vol":    signals["rel_volume"].values,
    })
    return out.sort_values("Datetime", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def run_backtest(
    df: pd.DataFrame,
    target_pts: float = BT_TARGET_PTS,
    sl_pts: float     = BT_SL_PTS,
    max_bars: int     = BT_MAX_BARS,
) -> pd.DataFrame:
    """
    Walk-forward fixed-point backtest.
    Entry at signal candle Close. Exit scan: bar-by-bar forward up to max_bars.
    Same-candle TP+SL breach => Loss (conservative).
    Inner scan uses pre-extracted NumPy arrays for speed.
    """
    signal_rows = df[df["signal"] != ""]
    if signal_rows.empty:
        return pd.DataFrame()

    all_highs  = df["High"].to_numpy(dtype=np.float64)
    all_lows   = df["Low"].to_numpy(dtype=np.float64)
    all_closes = df["Close"].to_numpy(dtype=np.float64)
    all_times  = df.index.to_numpy()
    n_bars     = len(df)
    time_to_pos: dict = {t: i for i, t in enumerate(df.index)}

    records: list = []

    for entry_time, sig_row in signal_rows.iterrows():
        sig_type  = sig_row["signal"]
        entry_px  = float(sig_row["Close"])

        if sig_type == "BUY":
            tp_price = entry_px + target_pts
            sl_price = entry_px - sl_pts
        else:
            tp_price = entry_px - target_pts
            sl_price = entry_px + sl_pts

        entry_pos   = time_to_pos[entry_time]
        scan_start  = entry_pos + 1
        scan_end    = min(scan_start + max_bars, n_bars)

        result    = "Open"
        exit_px   = float(all_closes[-1])
        exit_time = all_times[-1]

        for j in range(scan_start, scan_end):
            bh = all_highs[j]
            bl = all_lows[j]
            hit_tp = (bh >= tp_price) if sig_type == "BUY" else (bl <= tp_price)
            hit_sl = (bl <= sl_price) if sig_type == "BUY" else (bh >= sl_price)

            if hit_tp and hit_sl:
                result = "Loss"; exit_px = sl_price; exit_time = all_times[j]; break
            elif hit_tp:
                result = "Win";  exit_px = tp_price; exit_time = all_times[j]; break
            elif hit_sl:
                result = "Loss"; exit_px = sl_price; exit_time = all_times[j]; break

        pnl_pts = (exit_px - entry_px) if sig_type == "BUY" else (entry_px - exit_px)
        records.append({
            "entry_time": entry_time, "exit_time": exit_time,
            "type": sig_type, "entry_px": round(entry_px, 2),
            "exit_px": round(exit_px, 2), "pnl_pts": round(pnl_pts, 2), "result": result,
        })

    if not records:
        return pd.DataFrame()

    trades = pd.DataFrame(records)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"])

    closed_mask = trades["result"].isin(["Win", "Loss"])
    trades["cum_pnl"] = (
        trades.loc[closed_mask, "pnl_pts"]
        .cumsum()
        .reindex(trades.index)
        .ffill()
        .fillna(0)
    )
    return trades


# ─────────────────────────────────────────────
# BACKTEST SUMMARY STATS
# ─────────────────────────────────────────────
def compute_bt_stats(trades: pd.DataFrame) -> dict:
    """Stats over closed trades only. Open trades excluded from rates."""
    closed = trades[trades["result"].isin(["Win", "Loss"])].copy()
    open_  = trades[trades["result"] == "Open"]

    n_total  = len(trades)
    n_closed = len(closed)
    n_open   = len(open_)

    if n_closed == 0:
        return {
            "n_total": n_total, "n_closed": 0, "n_open": n_open,
            "n_wins": 0, "n_losses": 0, "win_rate": 0.0,
            "total_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "expectancy": 0.0, "profit_factor": 0.0, "max_consec_loss": 0,
        }

    wins   = closed[closed["result"] == "Win"]
    losses = closed[closed["result"] == "Loss"]
    n_wins = len(wins); n_losses = len(losses)
    win_rate = n_wins / n_closed

    total_pnl    = float(closed["pnl_pts"].sum())
    avg_win      = float(wins["pnl_pts"].mean())   if n_wins   else 0.0
    avg_loss     = float(losses["pnl_pts"].mean()) if n_losses else 0.0
    expectancy   = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    gross_profit = float(wins["pnl_pts"].sum())            if n_wins   else 0.0
    gross_loss   = abs(float(losses["pnl_pts"].sum()))     if n_losses else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    consec = max_consec = 0
    for r in closed["result"].tolist():
        if r == "Loss":
            consec += 1; max_consec = max(max_consec, consec)
        else:
            consec = 0

    return {
        "n_total": n_total, "n_closed": n_closed, "n_open": n_open,
        "n_wins": n_wins, "n_losses": n_losses, "win_rate": win_rate,
        "total_pnl": total_pnl, "avg_win": avg_win, "avg_loss": avg_loss,
        "expectancy": expectancy, "profit_factor": profit_factor,
        "max_consec_loss": max_consec,
    }


# ─────────────────────────────────────────────
# EQUITY CURVE  (time-series)
# ─────────────────────────────────────────────
def build_equity_curve_timeseries(trades: pd.DataFrame) -> go.Figure:
    """
    Point Equity Curve with x-axis = calendar entry_time.
    Matches spec: 'cumulative points gained over 30 days'.
    """
    closed = trades[trades["result"].isin(["Win","Loss"])].copy()
    open_  = trades[trades["result"] == "Open"].copy()

    if closed.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e14",
            plot_bgcolor="#0a0e14", margin=dict(l=0,r=0,t=36,b=0),
            title=dict(text="Point Equity Curve — no closed trades yet",
                       font=dict(family="JetBrains Mono, monospace", size=12, color="#6c7086"),
                       x=0.005, xanchor="left"),
        )
        fig.add_annotation(text="No closed trades to plot.", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color="#6c7086", family="JetBrains Mono, monospace"),
                           xanchor="center", yanchor="middle")
        return fig

    closed = closed.sort_values("entry_time").reset_index(drop=True)
    cum_pnl      = closed["pnl_pts"].cumsum()
    last_pnl     = float(cum_pnl.iloc[-1])
    pnl_color    = "#a6e3a1" if last_pnl >= 0 else "#f38ba8"
    point_colors = ["#a6e3a1" if v >= 0 else "#f38ba8" for v in cum_pnl]

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#313244", width=1, dash="dot"))

    fig.add_trace(go.Scatter(
        x=closed["entry_time"], y=cum_pnl, mode="lines+markers",
        name="Cumulative P&L",
        line=dict(color="#89dceb", width=2.5, shape="hv"),
        fill="tozeroy", fillcolor="rgba(137,220,235,0.07)",
        marker=dict(color=point_colors, size=8,
                    symbol=["triangle-up" if r == "Win" else "triangle-down" for r in closed["result"]],
                    line=dict(color="#0a0e14", width=1.2)),
        hovertemplate=(
            "<b>%{customdata[0]}</b>  %{x|%d %b %H:%M}<br>"
            "P&L this trade : <b>%{customdata[1]:+.1f} pts</b><br>"
            "Cumulative     : <b>%{y:+.1f} pts</b><extra></extra>"
        ),
        customdata=list(zip(closed["type"], closed["pnl_pts"])),
    ))

    bar_colors = ["#a6e3a1" if p > 0 else "#f38ba8" for p in closed["pnl_pts"]]
    fig.add_trace(go.Bar(
        x=closed["entry_time"], y=closed["pnl_pts"], name="Per-trade P&L",
        marker=dict(color=bar_colors, opacity=0.35, line=dict(width=0)),
        yaxis="y2", hovertemplate="%{x|%d %b %H:%M}  <b>%{y:+.1f} pts</b><extra></extra>",
    ))

    if not open_.empty:
        fig.add_trace(go.Scatter(
            x=open_["entry_time"], y=[last_pnl] * len(open_),
            mode="markers+text", name="Open (unresolved)",
            marker=dict(symbol="circle-open", size=10, color="#89dceb",
                        line=dict(color="#89dceb", width=2)),
            text=["OPEN"] * len(open_), textposition="top center",
            textfont=dict(size=8, color="#89dceb", family="JetBrains Mono, monospace"),
            hovertemplate="OPEN trade  %{x|%d %b %H:%M}<br>Not yet resolved<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14",
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(
            text=(f"Point Equity Curve  ·  30-day SMC Backtest  ·  "
                  f"{len(closed)} closed trades  ·  "
                  f"<span style='color:{pnl_color}'>{last_pnl:+.1f} pts net</span>"),
            font=dict(family="JetBrains Mono, monospace", size=12, color="#89dceb"),
            x=0.005, xanchor="left",
        ),
        legend=dict(orientation="h", x=0, y=1.055,
                    font=dict(size=10, color="#6c7086"), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0d1117", bordercolor="#1e2a3a",
                        font_size=11, font_family="JetBrains Mono, monospace"),
        xaxis=dict(gridcolor="#1a2235", tickfont=dict(size=9, color="#6c7086"),
                   tickformat="%d %b\n%H:%M", rangeslider=dict(visible=False)),
        yaxis=dict(title="Cumulative P&L (pts)", gridcolor="#1a2235", side="left",
                   tickfont=dict(size=9, color="#6c7086"), tickformat="+,.0f",
                   zeroline=True, zerolinecolor="#313244", zerolinewidth=1),
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    tickfont=dict(size=9, color="#6c7086"), tickformat="+,.0f",
                    title="Per-trade P&L (pts)"),
        bargap=0.2,
    )
    return fig


# ─────────────────────────────────────────────
# DONUT CHART
# ─────────────────────────────────────────────
def build_result_donut(stats: dict) -> go.Figure:
    """Win/Loss/Open distribution donut."""
    values = [stats["n_wins"], stats["n_losses"], stats["n_open"]]
    if sum(values) == 0:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0e14",
                          plot_bgcolor="#0a0e14", margin=dict(l=0,r=0,t=28,b=0))
        fig.add_annotation(text="No closed trades", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color="#6c7086", family="JetBrains Mono, monospace"),
                           xanchor="center", yanchor="middle")
        return fig

    fig = go.Figure(go.Pie(
        labels=["Wins", "Losses", "Open"], values=values, hole=0.62,
        marker=dict(colors=["#a6e3a1","#f38ba8","#89dceb"],
                    line=dict(color="#0a0e14", width=3)),
        textinfo="label+percent",
        textfont=dict(size=11, family="JetBrains Mono, monospace"),
        hovertemplate="%{label}: %{value} trades (%{percent})<extra></extra>",
        sort=False,
    ))
    wr = stats["win_rate"] * 100
    wr_color = "#a6e3a1" if wr >= 50 else "#f38ba8"
    fig.add_annotation(
        text=f"<b style='color:{wr_color}'>{wr:.1f}%</b><br>"
             f"<span style='color:#6c7086;font-size:10px'>Win Rate</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, family="JetBrains Mono, monospace"),
        xanchor="center", yanchor="middle",
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14",
        margin=dict(l=0,r=0,t=28,b=0),
        title=dict(text="Outcome Distribution",
                   font=dict(family="JetBrains Mono, monospace", size=12, color="#89dceb"),
                   x=0.5, xanchor="center"),
        legend=dict(orientation="v", x=1.02, y=0.5,
                    font=dict(size=10, color="#6c7086"), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ─────────────────────────────────────────────
# LIVE RADAR UI HELPERS
# ─────────────────────────────────────────────
def render_radar_banner(last_row: pd.Series) -> None:
    """Full-width signal banner with pulse animation when active."""
    sig = str(last_row.get("signal", ""))

    if sig == "BUY":
        bg = "linear-gradient(90deg, #0d2b1a 0%, #1a4a2a 50%, #0d2b1a 100%)"
        border_c = "#a6e3a1"; icon = "▲"; label = "ACTIVE BUY SIGNAL"
        sub = "Bullish Liquidity Sweep Confirmed — Institutional Demand Detected"
        txt_col = "#a6e3a1"; pulse = "#a6e3a1"
    elif sig == "SELL":
        bg = "linear-gradient(90deg, #2b0d0d 0%, #4a1a1a 50%, #2b0d0d 100%)"
        border_c = "#f38ba8"; icon = "▼"; label = "ACTIVE SELL SIGNAL"
        sub = "Bearish Liquidity Sweep Confirmed — Institutional Supply Detected"
        txt_col = "#f38ba8"; pulse = "#f38ba8"
    else:
        bg = "#0d1117"; border_c = "#1e2a3a"; icon = "◉"
        label = "MONITORING — NO ACTIVE SIGNAL"
        sub = "Last candle closed neutral. Watching for next SMC sweep."
        txt_col = "#6c7086"; pulse = "#6c7086"

    ts_str = ""
    if hasattr(last_row.name, "strftime"):
        ts_str = last_row.name.strftime("Last bar: %d %b %Y  %H:%M UTC")

    anim = "animation: radar-pulse 1.8s infinite;" if sig in ("BUY", "SELL") else ""

    st.markdown(f"""
    <style>
      @keyframes radar-pulse {{
        0%   {{ box-shadow: 0 0 0 0   {pulse}44; }}
        70%  {{ box-shadow: 0 0 0 10px {pulse}00; }}
        100% {{ box-shadow: 0 0 0 0   {pulse}00; }}
      }}
      .radar-banner {{
        background: {bg}; border: 1px solid {border_c};
        border-left: 4px solid {border_c}; border-radius: 6px;
        padding: 14px 22px; margin-bottom: 12px; {anim}
        display: flex; align-items: center; justify-content: space-between;
      }}
      .radar-banner .rb-left  {{ display:flex; align-items:center; gap:14px; }}
      .radar-banner .rb-icon  {{ font-size:2rem; color:{txt_col}; line-height:1; }}
      .radar-banner .rb-label {{ font-size:1.05rem; font-weight:700; color:{txt_col};
                                 letter-spacing:0.1em; font-family:'JetBrains Mono',monospace; }}
      .radar-banner .rb-sub   {{ font-size:0.68rem; color:#6c7086; margin-top:3px; letter-spacing:0.06em; }}
      .radar-banner .rb-ts    {{ font-size:0.62rem; color:#45475a; font-family:'JetBrains Mono',monospace; }}
    </style>
    <div class="radar-banner">
      <div class="rb-left">
        <div class="rb-icon">{icon}</div>
        <div>
          <div class="rb-label">{label}</div>
          <div class="rb-sub">{sub}</div>
        </div>
      </div>
      <div class="rb-ts">{ts_str}</div>
    </div>
    """, unsafe_allow_html=True)


def render_radar_metrics(last_row: pd.Series, bt_target: float, bt_sl: float) -> None:
    """Four st.metric boxes: Live Price, Target, Stop Loss, Risk:Reward."""
    sig        = str(last_row.get("signal", ""))
    close      = float(last_row["Close"])
    prev_close = float(last_row.get("prev_close", close))
    delta_pts  = close - prev_close
    delta_pct  = (delta_pts / prev_close * 100) if prev_close != 0 else 0.0

    if sig == "BUY":
        tp_display = close + bt_target; sl_display = close - bt_sl
        rr_display = bt_target / bt_sl if bt_sl > 0 else 0.0
        tp_label = f"Target (+{bt_target:.0f} pts)"; sl_label = f"Stop Loss (-{bt_sl:.0f} pts)"
    elif sig == "SELL":
        tp_display = close - bt_target; sl_display = close + bt_sl
        rr_display = bt_target / bt_sl if bt_sl > 0 else 0.0
        tp_label = f"Target (-{bt_target:.0f} pts)"; sl_label = f"Stop Loss (+{bt_sl:.0f} pts)"
    else:
        tp_raw = last_row.get("tp"); sl_raw = last_row.get("sl")
        tp_display = float(tp_raw) if tp_raw is not None and pd.notna(tp_raw) else float("nan")
        sl_display = float(sl_raw) if sl_raw is not None and pd.notna(sl_raw) else float("nan")
        rr_display = bt_target / bt_sl if bt_sl > 0 else 0.0
        tp_label = "Last TP Level"; sl_label = "Last SL Level"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live Price", f"{close:,.2f}", f"{delta_pts:+.2f} ({delta_pct:+.2f}%)")
    c2.metric(
        tp_label,
        f"{tp_display:,.2f}" if not np.isnan(tp_display) else "—",
        delta=f"+{bt_target:.0f} pts" if sig == "BUY" else (f"-{bt_target:.0f} pts" if sig == "SELL" else None),
        delta_color="normal" if sig == "BUY" else ("inverse" if sig == "SELL" else "off"),
    )
    c3.metric(
        sl_label,
        f"{sl_display:,.2f}" if not np.isnan(sl_display) else "—",
        delta=f"-{bt_sl:.0f} pts" if sig in ("BUY","SELL") else None,
        delta_color="inverse",
    )
    c4.metric("Risk : Reward", f"1 : {rr_display:.2f}", f"{bt_target:.0f} / {bt_sl:.0f} pts",
              delta_color="off")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐯 QuantBengal Pro")
    st.markdown("**SMC Terminal**  `v1.0`")
    st.markdown("---")

    selected_label = st.selectbox("Asset", list(ASSET_MAP.keys()), index=0)
    ticker = ASSET_MAP[selected_label]

    st.markdown("---")
    st.markdown("**Strategy Parameters**")
    # Bug fix: removed format="%.1f×" and "1 : %.1f" — printf-style
    # injection of non-ASCII chars causes ValueError on Streamlit >= 1.32
    roll_window    = st.slider("Rolling Window (bars)", 10, 50, ROLLING_WINDOW, step=5)
    vol_multiplier = st.slider("Volume Multiplier", 1.0, 2.5, VOLUME_MULT, step=0.1)
    rr             = st.slider("Risk Reward Ratio", 1.5, 5.0, RR_RATIO, step=0.5)

    st.markdown("---")
    st.markdown("**Backtest Parameters**")
    bt_target = st.number_input("Target (pts)", min_value=10.0, max_value=500.0,
                                value=BT_TARGET_PTS, step=10.0)
    bt_sl     = st.number_input("Stop Loss (pts)", min_value=5.0, max_value=200.0,
                                value=BT_SL_PTS, step=5.0)
    bt_rr_display = bt_target / bt_sl if bt_sl > 0 else 0.0
    st.markdown(
        f"<div style='font-size:0.68rem; color:#89dceb; margin-top:-6px'>"
        f"Implied RR: 1 : {bt_rr_display:.2f}</div>",
        unsafe_allow_html=True,
    )
    bt_max_bars = st.slider("Max Hold (bars)", 12, 200, BT_MAX_BARS, step=4)

    st.markdown("---")
    st.button("Refresh Data", use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.65rem; color:#6c7086; line-height:1.7'>"
        f"<b style='color:#89dceb'>BUY Trigger</b><br>"
        f"Low &lt; Support({roll_window}) AND<br>"
        f"Close &gt; Support({roll_window}) AND<br>"
        f"Vol &gt; {vol_multiplier:.1f}x Vol SMA<br><br>"
        f"<b style='color:#89dceb'>SELL Trigger</b><br>"
        f"High &gt; Resistance({roll_window}) AND<br>"
        f"Close &lt; Resistance({roll_window}) AND<br>"
        f"Vol &gt; {vol_multiplier:.1f}x Vol SMA<br><br>"
        f"<b style='color:#89dceb'>Risk Management</b><br>"
        f"Ratio 1 : {rr:.1f}<br>"
        f"SL at liquidity level<br>"
        f"TP = Close +/- (risk x RR)"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="terminal-header">
  <h1>🐯 QuantBengal Pro SMC Terminal</h1>
  <span>Smart Money Concepts · Liquidity Sweep Detection · {INTERVAL} bars · 30-day window</span>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA FETCH + PROCESSING  (no @st.cache_data)
# ─────────────────────────────────────────────
with st.spinner(f"Fetching {ticker} · {INTERVAL} · 30d ..."):
    raw_df = fetch_ohlcv(ticker, days=LOOKBACK_DAYS, interval=INTERVAL)

if raw_df.empty:
    st.error(
        f"**Data pipeline returned no usable rows for `{ticker}`.**  "
        "Possible causes: market closed, ticker unsupported for intraday, "
        "or yfinance rate limit. Try BTC-USD (trades 24/7) or click Refresh."
    )
    st.stop()

processed_df = compute_smc_signals(raw_df, window=roll_window,
                                   vol_mult=vol_multiplier, rr_ratio=rr)


# ─────────────────────────────────────────────
# KPI BAR  (always visible above tabs)
# ─────────────────────────────────────────────
st.markdown(build_kpi_html(processed_df), unsafe_allow_html=True)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BACKTEST  (computed once, used in Tab 2)
# ─────────────────────────────────────────────
trades_df = run_backtest(processed_df, target_pts=bt_target,
                         sl_pts=bt_sl, max_bars=bt_max_bars)
bt_stats  = compute_bt_stats(trades_df) if not trades_df.empty else {
    "n_total": 0, "n_closed": 0, "n_open": 0,
    "n_wins": 0, "n_losses": 0, "win_rate": 0.0,
    "total_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
    "expectancy": 0.0, "profit_factor": 0.0, "max_consec_loss": 0,
}


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab_radar, tab_backtest, tab_signals, tab_data = st.tabs([
    "📡  Institutional Live Radar",
    "🔬  SMC Performance Backtest",
    "⚡  Signal Log",
    "🗂  Raw Data",
])


# ══════════════════════════════════════════════
# TAB 1 — INSTITUTIONAL LIVE RADAR
# ══════════════════════════════════════════════
with tab_radar:
    last_bar = processed_df.iloc[-1]

    render_radar_banner(last_bar)
    render_radar_metrics(last_bar, bt_target=bt_target, bt_sl=bt_sl)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    n_candles_radar = st.slider(
        "Visible candles", min_value=50, max_value=300, value=100, step=10,
        key="radar_candles",
        help="Number of 15-min bars rendered. Signal engine uses full 30-day history.",
    )
    st.plotly_chart(
        build_chart(processed_df, ticker, n_candles=n_candles_radar),
        use_container_width=True,
        config={"scrollZoom": True, "displayModeBar": True},
    )
    st.markdown(
        "<div style='font-size:0.67rem; color:#45475a; margin-top:-8px; line-height:1.9'>"
        "<span style='color:#a6e3a1'>&#9650; green triangle</span> = Bullish Sweep (BUY) &nbsp;&middot;&nbsp;"
        "<span style='color:#f38ba8'>&#9660; red triangle</span> = Bearish Sweep (SELL) &nbsp;&middot;&nbsp;"
        "<span style='color:#a6e3a1'>-- dashed green</span> = Support (20p Low) &nbsp;&middot;&nbsp;"
        "<span style='color:#f38ba8'>-- dashed red</span> = Resistance (20p High) &nbsp;&middot;&nbsp;"
        "SL/TP lines hidden by default — toggle via legend. "
        "<em>Simulation only. Not financial advice.</em>"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# TAB 2 — SMC PERFORMANCE BACKTEST
# ══════════════════════════════════════════════
with tab_backtest:
    s  = bt_stats
    pf = s["profit_factor"]

    # Bug fix: check n_closed explicitly; empty dict guard was insufficient
    if trades_df.empty or s["n_closed"] == 0:
        st.info(
            "No closed trades to display. Either no signals were detected or all trades "
            "are still open within the scan window. "
            "Try reducing the Volume Multiplier or Rolling Window in the sidebar."
        )
        if not trades_df.empty and s["n_open"] > 0:
            st.markdown(f"**{s['n_open']} open trade(s)** detected — none resolved within "
                        f"{bt_max_bars} bars ({bt_max_bars * 15 // 60}h). "
                        "Increase Max Hold in the sidebar to resolve them.")
    else:
        # ── Four headline KPIs ────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.72rem; color:#89dceb; letter-spacing:0.1em; "
            "text-transform:uppercase; margin-bottom:6px'>Performance Summary</div>",
            unsafe_allow_html=True,
        )
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("Total Trades", str(s["n_total"]),
                   f"{s['n_closed']} closed - {s['n_open']} open", delta_color="off")
        win_rate_pct = s["win_rate"] * 100
        kc2.metric("Win Rate", f"{win_rate_pct:.1f}%",
                   f"{s['n_wins']}W / {s['n_losses']}L",
                   delta_color="normal" if win_rate_pct >= 50 else "inverse")
        kc3.metric("Net Points Accrued", f"{s['total_pnl']:+.1f} pts",
                   f"Expectancy {s['expectancy']:+.2f} pts/trade",
                   delta_color="normal" if s["total_pnl"] >= 0 else "inverse")
        kc4.metric("Profit Factor",
                   f"{pf:.2f}x" if pf != float("inf") else "inf",
                   "Gross Wins / Gross Losses",
                   delta_color="normal" if pf >= 1.0 else "inverse")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # ── Secondary stat cards ──────────────────────────────────────────
        sec_cards = [
            ("Avg Win",          f"{s['avg_win']:+.1f} pts",         "pos"),
            ("Avg Loss",         f"{s['avg_loss']:+.1f} pts",         "neg"),
            ("Expectancy",       f"{s['expectancy']:+.2f} pts/trade",
             "pos" if s["expectancy"] >= 0 else "neg"),
            ("Max Consec. Loss", str(s["max_consec_loss"]),            "neg"),
            ("Target / SL",      f"{bt_target:.0f} / {bt_sl:.0f} pts","neu"),
            ("Implied RR",       f"1 : {bt_rr_display:.2f}",         "neu"),
        ]
        sec_html = '<div class="bt-stat-grid">'
        for lbl, val, cls in sec_cards:
            sec_html += (f'<div class="bt-stat"><div class="s-label">{lbl}</div>'
                         f'<div class="s-value {cls}">{val}</div></div>')
        sec_html += '</div>'
        st.markdown(sec_html, unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── Methodology note ──────────────────────────────────────────────
        st.markdown(
            f"<div style='font-size:0.67rem; color:#6c7086; background:#0d1117; "
            f"border:1px solid #1e2a3a; border-radius:6px; padding:8px 16px; "
            f"line-height:1.9; margin-bottom:12px'>"
            f"<b style='color:#89dceb'>Simulation</b> &nbsp;&middot;&nbsp;"
            f"Entry at signal candle Close &nbsp;&middot;&nbsp;"
            f"Target <b style='color:#a6e3a1'>+{bt_target:.0f} pts</b> limit fill &nbsp;&middot;&nbsp;"
            f"Stop <b style='color:#f38ba8'>-{bt_sl:.0f} pts</b> stop fill &nbsp;&middot;&nbsp;"
            f"Same-candle TP+SL = Loss (conservative) &nbsp;&middot;&nbsp;"
            f"Max hold {bt_max_bars} bars ({bt_max_bars * 15 // 60}h) &nbsp;&middot;&nbsp;"
            f"No slippage or commission modelled."
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Point Equity Curve ────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.72rem; color:#89dceb; letter-spacing:0.1em; "
            "text-transform:uppercase; margin-bottom:4px'>Point Equity Curve — 30 Days</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_equity_curve_timeseries(trades_df),
                        use_container_width=True, config={"displayModeBar": False})

        # ── Detailed Trade Log ────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.72rem; color:#89dceb; letter-spacing:0.1em; "
            "text-transform:uppercase; margin:10px 0 4px'>Detailed Trade Log</div>",
            unsafe_allow_html=True,
        )

        display_trades = trades_df.copy()
        display_trades["entry_time"] = display_trades["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
        display_trades["exit_time"]  = display_trades["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
        display_trades.index = range(1, len(display_trades) + 1)
        display_trades.index.name = "#"
        display_trades.columns = [
            "Entry Time", "Exit Time", "Type",
            "Entry Px", "Exit Px", "P&L (pts)", "Result", "Cum P&L",
        ]

        def _style_trades(row):
            if row["Result"] == "Win":   return ["background-color:#1a3a2a"] * len(row)
            if row["Result"] == "Loss":  return ["background-color:#3a1a1a"] * len(row)
            return ["background-color:#1a1e2a"] * len(row)

        def _colour_pnl(val):
            try:
                return ("color:#a6e3a1;font-weight:700" if float(val) >= 0
                        else "color:#f38ba8;font-weight:700")
            except (ValueError, TypeError):
                return ""

        def _colour_result(val):
            if val == "Win":  return "color:#a6e3a1;font-weight:700"
            if val == "Loss": return "color:#f38ba8;font-weight:700"
            return "color:#89dceb"

        def _colour_type(val):
            if val == "BUY":  return "color:#a6e3a1;font-weight:700"
            if val == "SELL": return "color:#f38ba8;font-weight:700"
            return ""

        st.dataframe(
            display_trades.style
            .apply(_style_trades, axis=1)
            .map(_colour_pnl,    subset=["P&L (pts)", "Cum P&L"])
            .map(_colour_result, subset=["Result"])
            .map(_colour_type,   subset=["Type"])
            .format({"Entry Px": "{:.2f}", "Exit Px": "{:.2f}",
                     "P&L (pts)": "{:+.2f}", "Cum P&L": "{:+.2f}"}),
            use_container_width=True, height=420,
        )
        st.markdown(
            f"<div style='font-size:0.62rem; color:#45475a; margin-top:4px'>"
            f"{len(trades_df)} total &nbsp;&middot;&nbsp; "
            f"{s['n_closed']} closed ({s['n_wins']}W / {s['n_losses']}L) "
            f"&nbsp;&middot;&nbsp; {s['n_open']} open &nbsp;&middot;&nbsp; "
            f"Period: {processed_df.index[0].strftime('%d %b')} to "
            f"{processed_df.index[-1].strftime('%d %b %Y')}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
# TAB 3 — SIGNAL LOG
# ══════════════════════════════════════════════
with tab_signals:
    sig_df = build_signal_table(processed_df)

    if sig_df.empty:
        st.info("No SMC liquidity sweep signals in the current window. "
                "Try reducing the Volume Multiplier or Rolling Window in the sidebar.")
    else:
        total_buy  = int((sig_df["Signal"] == "BUY").sum())
        total_sell = int((sig_df["Signal"] == "SELL").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Signals",  len(sig_df))
        c2.metric("Bullish Sweeps", total_buy,  delta=f"+{total_buy}")
        c3.metric("Bearish Sweeps", total_sell, delta=f"-{total_sell}", delta_color="inverse")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        def _colour_sig(val: str) -> str:
            if val == "BUY":  return "background-color:#1a3a2a; color:#a6e3a1; font-weight:700"
            if val == "SELL": return "background-color:#3a1a2a; color:#f38ba8; font-weight:700"
            return ""

        st.dataframe(sig_df.style.map(_colour_sig, subset=["Signal"]),
                     use_container_width=True, height=460)


# ══════════════════════════════════════════════
# TAB 4 — RAW DATA
# ══════════════════════════════════════════════
with tab_data:
    display_cols = ["Open", "High", "Low", "Close", "Volume",
                    "support", "resistance", "vol_sma",
                    "signal", "sl", "tp", "rel_volume"]
    display_df = processed_df[display_cols].copy()

    # Stamp backtest result onto signal bars
    display_df["bt_result"] = ""
    if not trades_df.empty:
        bt_map = dict(zip(
            pd.to_datetime(trades_df["entry_time"]).dt.tz_localize(None),
            trades_df["result"],
        ))
        display_df["bt_result"] = display_df.index.map(lambda ts: bt_map.get(ts, ""))

    display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M")
    display_df = display_df.sort_index(ascending=False)

    fmt = {
        "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}", "Close": "{:.2f}",
        "support": "{:.2f}", "resistance": "{:.2f}",
        "vol_sma": "{:,.0f}", "Volume": "{:,.0f}",
        "sl": "{:.2f}", "tp": "{:.2f}", "rel_volume": "{:.2f}",
    }

    def _csig(val: str) -> str:
        if val == "BUY":  return "background-color:#1a3a2a; color:#a6e3a1; font-weight:700"
        if val == "SELL": return "background-color:#3a1a2a; color:#f38ba8; font-weight:700"
        return ""

    def _cbtr(val: str) -> str:
        if val == "Win":  return "color:#a6e3a1; font-weight:700"
        if val == "Loss": return "color:#f38ba8; font-weight:700"
        if val == "Open": return "color:#89dceb; font-weight:700"
        return "color:#45475a"

    def _crvol(val) -> str:
        try:
            v = float(val)
            if v >= 2.0:                return "color:#f9e2af; font-weight:700"
            if v >= float(VOLUME_MULT): return "color:#cdd6f4"
        except (ValueError, TypeError):
            pass
        return "color:#45475a"

    st.dataframe(
        display_df.style
        .map(_csig,  subset=["signal"])
        .map(_cbtr,  subset=["bt_result"])
        .map(_crvol, subset=["rel_volume"])
        .format(fmt),
        use_container_width=True, height=500,
    )

    n_sigs = int((processed_df["signal"] != "").sum())
    n_bt   = len(trades_df) if not trades_df.empty else 0
    st.markdown(
        f"<div style='font-size:0.62rem; color:#45475a; margin-top:4px; line-height:1.9'>"
        f"<b style='color:#6c7086'>Rows:</b> {len(processed_df):,} &nbsp;&middot;&nbsp; "
        f"<b style='color:#6c7086'>Period:</b> "
        f"{processed_df.index[0].strftime('%Y-%m-%d %H:%M')} to "
        f"{processed_df.index[-1].strftime('%Y-%m-%d %H:%M')} &nbsp;&middot;&nbsp; "
        f"<b style='color:#6c7086'>Interval:</b> {INTERVAL} &nbsp;&middot;&nbsp; "
        f"<b style='color:#6c7086'>Signals:</b> {n_sigs} &nbsp;&middot;&nbsp; "
        f"<b style='color:#6c7086'>BT trades:</b> {n_bt}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='font-size:0.62rem; color:#45475a; text-align:center; padding-bottom:8px'>"
    f"QuantBengal Pro SMC Terminal &nbsp;&middot;&nbsp; "
    f"Strategy: SMC Liquidity Sweep &nbsp;&middot;&nbsp; "
    f"Backtest: {bt_target:.0f} pt target / {bt_sl:.0f} pt SL "
    f"(1:{bt_rr_display:.2f} RR) &nbsp;&middot;&nbsp; "
    f"All signals are probabilistic historical simulation. "
    f"Not financial advice."
    f"</div>",
    unsafe_allow_html=True,
)
