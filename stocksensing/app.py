
import os
import sys
import glob
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Sense",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
BLUE   = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
PURPLE = "#bc8cff"
DARK   = "#0d1117"
CARD   = "#161b22"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
    .signal-up {
        background: #1a3a2a;
        border: 1px solid #3fb950;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .signal-down {
        background: #3a1a1a;
        border: 1px solid #f85149;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    div[data-testid="stMetricValue"] { color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "engineered_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "stock_model_best.keras")
SCALER_PATH= os.path.join(BASE_DIR, "models", "scaler.pkl")
META_GLOB  = os.path.join(BASE_DIR, "models", "metadata_*.json")
LOGS_GLOB  = os.path.join(BASE_DIR, "logs", "training_log_*.csv")

FEATURES = [
    "rsi", "macd", "macd_signal", "macd_diff",
    "stoch", "momentum", "ema_sma_ratio",
    "bb_width", "bb_position", "atr_pct", "volatility_10d",
    "close_position", "price_range",
    "macd_cross", "rsi_overbought", "rsi_oversold",
    "volume_zscore", "volume_change", "v_lag1", "v_lag2",
    "returns", "r_lag1", "r_lag2", "r_lag3",
    "trend_3d", "trend_5d",
]


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["formatted_date"])
    df.sort_values("formatted_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_resource
def load_model_and_scaler():
    model  = load_model(MODEL_PATH)  if os.path.exists(MODEL_PATH)  else None
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return model, scaler

def load_metadata():
    files = glob.glob(META_GLOB)
    if not files:
        return {}
    return json.load(open(max(files, key=os.path.getmtime)))

def load_training_log():
    files = glob.glob(LOGS_GLOB)
    if not files:
        return None
    return pd.read_csv(max(files, key=os.path.getmtime))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Sense")
    st.markdown("---")

    df_full = load_data()

    date_min = df_full["formatted_date"].min().date()
    date_max = df_full["formatted_date"].max().date()

    st.markdown("### 📅 Date Range")
    start_date = st.date_input("From", value=date_min, min_value=date_min, max_value=date_max)
    end_date   = st.date_input("To",   value=date_max, min_value=date_min, max_value=date_max)

    st.markdown("### 🎛️ Indicators")
    show_bb    = st.checkbox("Bollinger Bands",  value=True)
    show_ema   = st.checkbox("EMA 10",           value=True)
    show_sma   = st.checkbox("SMA 10",           value=True)
    show_vol   = st.checkbox("Volume",           value=True)

    st.markdown("### 🤖 Model")
    threshold  = st.slider("Prediction Threshold", 0.30, 0.70, 0.50, 0.01)

    st.markdown("---")
    st.caption(f"Data: {date_min} → {date_max}")
    st.caption(f"Total rows: {len(df_full):,}")


# ── Filter data ───────────────────────────────────────────────────────────────
df = df_full[
    (df_full["formatted_date"].dt.date >= start_date) &
    (df_full["formatted_date"].dt.date <= end_date)
].copy()


# ── Run predictions ───────────────────────────────────────────────────────────
model, scaler = load_model_and_scaler()
meta = load_metadata()

predictions = None
probabilities = None

if model and scaler:
    feat_cols = [f for f in FEATURES if f in df.columns]
    X = df[feat_cols].values
    X_scaled = scaler.transform(X)
    probabilities = model.predict(X_scaled, verbose=0).flatten()
    predictions   = (probabilities > threshold).astype(int)
    df["prediction"]   = predictions
    df["probability"]  = probabilities


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Stock Sense Dashboard")
st.markdown(f"**S&P 500** · {start_date} → {end_date} · {len(df):,} trading days")
st.markdown("---")


# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

period_return = ((df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100) if len(df) > 1 else 0
volatility    = df["returns"].std() * np.sqrt(252) * 100 if "returns" in df.columns else 0
last_rsi      = df["rsi"].iloc[-1] if "rsi" in df.columns else 0
last_close    = df["close"].iloc[-1] if "close" in df.columns else 0
prev_close    = df["close"].iloc[-2] if len(df) > 1 else last_close
day_change    = (last_close - prev_close) / prev_close * 100 if prev_close else 0

k1.metric("Last Close",     f"{last_close:,.2f}",  f"{day_change:+.2f}%")
k2.metric("Period Return",  f"{period_return:+.2f}%")
k3.metric("Annualized Vol", f"{volatility:.1f}%")
k4.metric("Last RSI",       f"{last_rsi:.1f}",
          "Overbought" if last_rsi > 70 else ("Oversold" if last_rsi < 30 else "Neutral"))

if meta:
    k5.metric("Model Accuracy", f"{meta.get('metrics', {}).get('accuracy', 0)*100:.1f}%")
    k6.metric("ROC-AUC",        f"{meta.get('metrics', {}).get('roc_auc', 0):.4f}")

st.markdown("---")


# ── Prediction Signal ─────────────────────────────────────────────────────────
if probabilities is not None:
    last_prob = probabilities[-1]
    last_pred = predictions[-1]
    conf      = last_prob if last_pred == 1 else (1 - last_prob)

    col_sig, col_conf, col_streak = st.columns([1, 1, 2])

    with col_sig:
        if last_pred == 1:
            st.markdown(f"""
            <div class="signal-up">
                <div style="font-size:2.5rem">🟢</div>
                <div style="font-size:1.4rem; font-weight:bold; color:#3fb950">UP</div>
                <div style="color:#8b949e; font-size:0.85rem">Next Day Signal</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="signal-down">
                <div style="font-size:2.5rem">🔴</div>
                <div style="font-size:1.4rem; font-weight:bold; color:#f85149">DOWN</div>
                <div style="color:#8b949e; font-size:0.85rem">Next Day Signal</div>
            </div>""", unsafe_allow_html=True)

    with col_conf:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{GREEN if last_pred==1 else RED}">{conf*100:.1f}%</div>
            <div class="metric-label">Model Confidence</div>
        </div>""", unsafe_allow_html=True)

    with col_streak:
        if predictions is not None and len(predictions) >= 5:
            recent = predictions[-5:]
            labels = ["🟢 UP" if p == 1 else "🔴 DN" for p in recent]
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1rem; letter-spacing:6px">{"  ".join(labels)}</div>
                <div class="metric-label">Last 5 Predictions</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")


# ── Price Chart with Indicators ───────────────────────────────────────────────
st.markdown("### 📊 Price Chart & Technical Indicators")

rows   = 4 if show_vol else 3
heights= [0.5, 0.18, 0.17, 0.15] if show_vol else [0.55, 0.22, 0.23]
row_titles = ["Price", "RSI", "MACD"] + (["Volume"] if show_vol else [])

fig = make_subplots(
    rows=rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=heights,
    subplot_titles=row_titles,
)

# ── Price + overlays ──────────────────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=df["formatted_date"], y=df["close"],
    name="Close", line=dict(color=BLUE, width=2),
), row=1, col=1)

if show_bb and "bb_high" in df.columns:
    fig.add_trace(go.Scatter(
        x=df["formatted_date"], y=df["bb_high"],
        name="BB High", line=dict(color=YELLOW, width=1, dash="dot"),
        opacity=0.7,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["formatted_date"], y=df["bb_low"],
        name="BB Low", line=dict(color=YELLOW, width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(210,153,34,0.05)", opacity=0.7,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["formatted_date"], y=df["bb_mid"],
        name="BB Mid", line=dict(color=YELLOW, width=1, dash="dash"),
        opacity=0.5,
    ), row=1, col=1)

if show_ema and "ema_10" in df.columns:
    fig.add_trace(go.Scatter(
        x=df["formatted_date"], y=df["ema_10"],
        name="EMA 10", line=dict(color=GREEN, width=1.5),
    ), row=1, col=1)

if show_sma and "sma_10" in df.columns:
    fig.add_trace(go.Scatter(
        x=df["formatted_date"], y=df["sma_10"],
        name="SMA 10", line=dict(color=PURPLE, width=1.5),
    ), row=1, col=1)

# Prediction markers
if predictions is not None:
    up_mask   = df["prediction"] == 1
    down_mask = df["prediction"] == 0
    fig.add_trace(go.Scatter(
        x=df.loc[up_mask, "formatted_date"],
        y=df.loc[up_mask, "close"],
        mode="markers", name="Pred UP",
        marker=dict(color=GREEN, size=4, symbol="triangle-up"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.loc[down_mask, "formatted_date"],
        y=df.loc[down_mask, "close"],
        mode="markers", name="Pred DOWN",
        marker=dict(color=RED, size=4, symbol="triangle-down"),
    ), row=1, col=1)

# ── RSI ───────────────────────────────────────────────────────────────────────
if "rsi" in df.columns:
    fig.add_trace(go.Scatter(
        x=df["formatted_date"], y=df["rsi"],
        name="RSI", line=dict(color=PURPLE, width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=RED,   opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=GREEN, opacity=0.5, row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=RED,   opacity=0.05, row=2, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=GREEN, opacity=0.05, row=2, col=1)

# ── MACD ──────────────────────────────────────────────────────────────────────
if "macd" in df.columns:
    colors_macd = [GREEN if v >= 0 else RED for v in df["macd_diff"]]
    fig.add_trace(go.Bar(
        x=df["formatted_date"], y=df["macd_diff"],
        name="MACD Diff", marker_color=colors_macd, opacity=0.7,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df["formatted_date"], y=df["macd"],
        name="MACD", line=dict(color=BLUE, width=1.5),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df["formatted_date"], y=df["macd_signal"],
        name="Signal", line=dict(color=YELLOW, width=1.5),
    ), row=3, col=1)

# ── Volume ────────────────────────────────────────────────────────────────────
if show_vol and "volume" in df.columns:
    vol_colors = [GREEN if r >= 0 else RED
                  for r in df["returns"].fillna(0)]
    fig.add_trace(go.Bar(
        x=df["formatted_date"], y=df["volume"],
        name="Volume", marker_color=vol_colors, opacity=0.6,
    ), row=4, col=1)

fig.update_layout(
    height=750,
    paper_bgcolor=DARK,
    plot_bgcolor=CARD,
    font=dict(color=TEXT, family="monospace"),
    legend=dict(bgcolor=CARD, bordercolor="#21262d", borderwidth=1,
                orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
)
fig.update_xaxes(gridcolor="#21262d", showgrid=True)
fig.update_yaxes(gridcolor="#21262d", showgrid=True)

st.plotly_chart(fig, use_container_width=True)


# ── Stochastic ────────────────────────────────────────────────────────────────
if "stoch" in df.columns:
    st.markdown("### 🎯 Stochastic Oscillator")
    fig_stoch = go.Figure()
    fig_stoch.add_trace(go.Scatter(
        x=df["formatted_date"], y=df["stoch"],
        name="Stochastic", line=dict(color=PURPLE, width=2),
    ))
    fig_stoch.add_hline(y=80, line_dash="dash", line_color=RED,   opacity=0.6)
    fig_stoch.add_hline(y=20, line_dash="dash", line_color=GREEN, opacity=0.6)
    fig_stoch.add_hrect(y0=80, y1=100, fillcolor=RED,   opacity=0.05)
    fig_stoch.add_hrect(y0=0,  y1=20,  fillcolor=GREEN, opacity=0.05)
    fig_stoch.update_layout(
        height=200, paper_bgcolor=DARK, plot_bgcolor=CARD,
        font=dict(color=TEXT), margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False, hovermode="x unified",
    )
    fig_stoch.update_xaxes(gridcolor="#21262d")
    fig_stoch.update_yaxes(gridcolor="#21262d", range=[0, 100])
    st.plotly_chart(fig_stoch, use_container_width=True)


# ── Prediction Probability Timeline ───────────────────────────────────────────
if probabilities is not None:
    st.markdown("### 🤖 Model Prediction Confidence Over Time")
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(
        x=df["formatted_date"], y=df["probability"],
        name="UP Probability",
        line=dict(color=BLUE, width=1.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
    ))
    fig_prob.add_hline(y=threshold, line_dash="dash",
                       line_color=YELLOW, opacity=0.8,
                       annotation_text=f"Threshold {threshold:.2f}")
    fig_prob.update_layout(
        height=220, paper_bgcolor=DARK, plot_bgcolor=CARD,
        font=dict(color=TEXT), margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified", showlegend=False,
        yaxis=dict(range=[0, 1], gridcolor="#21262d"),
        xaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig_prob, use_container_width=True)


# ── Feature Distributions ─────────────────────────────────────────────────────
st.markdown("### 📐 Technical Indicator Distributions")

ind_cols = st.columns(3)

indicator_pairs = [
    ("RSI",       "rsi",      [0, 100],  [30, 70]),
    ("Momentum",  "momentum", None,      [0]),
    ("BB Width",  "bb_width", None,      None),
]

for idx, (label, col, xrange, vlines) in enumerate(indicator_pairs):
    if col not in df.columns:
        continue
    with ind_cols[idx % 3]:
        fig_dist = px.histogram(
            df, x=col, nbins=50,
            title=label,
            color_discrete_sequence=[BLUE],
        )
        if xrange:
            fig_dist.update_xaxes(range=xrange)
        if vlines:
            for v in vlines:
                fig_dist.add_vline(x=v, line_dash="dash", line_color=YELLOW, opacity=0.7)
        fig_dist.update_layout(
            height=220, paper_bgcolor=DARK, plot_bgcolor=CARD,
            font=dict(color=TEXT, size=11),
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# ── Training History ──────────────────────────────────────────────────────────
train_log = load_training_log()
if train_log is not None:
    st.markdown("### 🏋️ Model Training History")
    col_loss, col_acc = st.columns(2)

    with col_loss:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=train_log["loss"], name="Train Loss",
            line=dict(color=BLUE, width=2),
        ))
        fig_loss.add_trace(go.Scatter(
            y=train_log["val_loss"], name="Val Loss",
            line=dict(color=RED, width=2, dash="dash"),
        ))
        fig_loss.update_layout(
            title="Loss", height=280,
            paper_bgcolor=DARK, plot_bgcolor=CARD,
            font=dict(color=TEXT), margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(bgcolor=CARD), hovermode="x unified",
            xaxis=dict(gridcolor="#21262d", title="Epoch"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_acc:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            y=train_log["accuracy"], name="Train Acc",
            line=dict(color=GREEN, width=2),
        ))
        fig_acc.add_trace(go.Scatter(
            y=train_log["val_accuracy"], name="Val Acc",
            line=dict(color=RED, width=2, dash="dash"),
        ))
        fig_acc.update_layout(
            title="Accuracy", height=280,
            paper_bgcolor=DARK, plot_bgcolor=CARD,
            font=dict(color=TEXT), margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(bgcolor=CARD), hovermode="x unified",
            xaxis=dict(gridcolor="#21262d", title="Epoch"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_acc, use_container_width=True)


# ── Model Metadata ────────────────────────────────────────────────────────────
if meta:
    st.markdown("### 🗂️ Model Metadata")
    m = meta.get("metrics", {})
    cfg = meta.get("config", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",    f"{m.get('accuracy',0)*100:.2f}%")
    c2.metric("ROC-AUC",     f"{m.get('roc_auc',0):.4f}")
    c3.metric("Epochs",      meta.get("epochs_trained", "—"))
    c4.metric("Batch Size",  cfg.get("batch_size", "—"))
    c5.metric("LR",          cfg.get("learning_rate", "—"))

    with st.expander("Full Metadata JSON"):
        st.json(meta)


# ── Raw Data ──────────────────────────────────────────────────────────────────
st.markdown("### 🗃️ Data Preview")
display_cols = ["formatted_date", "close", "rsi", "macd", "macd_signal",
                "bb_high", "bb_low", "stoch", "atr_pct", "returns"]
if predictions is not None:
    display_cols += ["probability", "prediction"]

available = [c for c in display_cols if c in df.columns]
st.dataframe(
    df[available].tail(50).style.format({
        "close"      : "{:.2f}",
        "rsi"        : "{:.2f}",
        "macd"       : "{:.4f}",
        "macd_signal": "{:.4f}",
        "bb_high"    : "{:.2f}",
        "bb_low"     : "{:.2f}",
        "stoch"      : "{:.2f}",
        "atr_pct"    : "{:.5f}",
        "returns"    : "{:.4f}",
        "probability": "{:.4f}",
    }),
    use_container_width=True,
    height=400,
)

st.markdown("---")
st.caption("Stock Sense · ANN-powered S&P 500 trend prediction · Built with Streamlit & Plotly")