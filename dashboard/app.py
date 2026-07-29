"""
Supply Chain Intelligence Platform — Decision Dashboard
=========================================================

This is a CLIENT of your FastAPI server (api/main.py). It does NOT load
models or touch MySQL directly — everything here is built from what
/categories and /forecast return. If you see connection errors, start
the API first:

    python -m uvicorn api.main:app --reload --port 8000

Then run:

    streamlit run dashboard/app.py

HONEST NOTE ON SCOPE:
- Forecasts, intervals, model routing, confidence -> come straight from your API.
- EOQ / Safety Stock / Reorder Point / Stockout-cost simulation -> computed
  HERE in the dashboard from your forecast + interval output, using
  standard inventory formulas. Your API does not compute these.
- Model leaderboard numbers (GBM 18.56% MAPE, Baseline 19.17%, LSTM 20.44%)
  are static, taken from your own evaluation report — not re-fetched live.
- ₹ Business impact estimates use standard inventory accounting assumptions
  (stockout cost = 2× unit cost, clearly labeled) — not from the API.
- ABC classification based on 7-day demand share from portfolio forecasts.
"""

from __future__ import annotations

import math
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import os
API_BASE_URL = os.environ.get("API_BASE_URL", "https://supply-chain-intelligence-fyff.onrender.com")

st.set_page_config(
    page_title="Supply Chain Intelligence Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DESIGN TOKENS — fixed palette, used everywhere
# ─────────────────────────────────────────────
# GBM      → CYAN    #00E5FF
# LSTM     → ORANGE  #FFA94D
# ALERTS   → RED     #FF5252
# SUCCESS  → GREEN   #00E676
# NEUTRAL  → PURPLE  #B388FF  (auto/misc)
# BG_CARD  → #131C2C
# BG_APP   → #0A0E1A
# BORDER   → #1E2A3E
# TEXT_DIM → #FFFFFF
# TEXT     → #EDF2F7

PALETTE = {
    "gbm":     "#00E5FF",
    "lstm":    "#FFA94D",
    "alert":   "#FF5252",
    "success": "#00E676",
    "auto":    "#B388FF",
    "cost_ordering": "#FFA94D",
    "cost_holding":  "#00E5FF",
    "cost_total":    "#B388FF",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }
.stApp { background: #0A0E1A !important; font-family: 'Inter', sans-serif !important; }
html, body, [class*="css"] { font-size: 16px !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { background: #0F1525 !important; border-right: 1px solid #1E2A3E !important; }
[data-testid="stSidebar"] * { color: #EDF2F7 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #131C2C !important; border: 1px solid #1E2A3E !important;
    border-radius: 10px !important; margin-bottom: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background: #131C2C !important; color: #FFA94D !important; border-radius: 10px !important; padding: 10px 14px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p { font-size: 12px !important; font-weight: 600 !important; }

/* Sidebar inputs — higher contrast so they don't look disabled */
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] .stNumberInput input {
    background: #1E2A3E !important;
    color: #EDF2F7 !important;
    border: 1px solid #2D3A5E !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] { color: #00E5FF !important; }

.main .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1400px !important; }

.main-heading {
    font-size: 44px !important; font-weight: 800 !important;
    background: linear-gradient(135deg, #00E5FF, #FFA94D);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 0.25rem !important; letter-spacing: -0.02em !important;
}
.sub-heading { font-size: 16px !important; font-weight: 500 !important; color: #FFFFFF !important; margin-bottom: 0.25rem !important; }
.tech-line { font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; color: #00E5FF !important; }

h2 { font-size: 26px !important; font-weight: 700 !important; color: #EDF2F7 !important; margin-top: 1rem !important; }
h3 { font-size: 19px !important; font-weight: 600 !important; color: #EDF2F7 !important; }

[data-testid="stTabs"] button { font-size: 13px !important; font-weight: 600 !important; color: #FFFFFF !important; padding: 8px 12px !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #00E5FF !important; border-bottom: 2px solid #00E5FF !important; }

[data-baseweb="select"] > div:first-child { background: #131C2C !important; border: 1px solid #2D3A5E !important; border-radius: 10px !important; }
[data-baseweb="select"] span, [data-baseweb="select"] div, [data-baseweb="select"] p { color: #EDF2F7 !important; }
[role="option"], [data-baseweb="option"] { background: #131C2C !important; color: #EDF2F7 !important; }
[role="option"]:hover { background: #1E2A3E !important; color: #00E5FF !important; }

.stButton > button {
    background: linear-gradient(135deg, #00E5FF, #0099CC) !important; color: #0A0E1A !important;
    font-weight: 700 !important; border: none !important; border-radius: 12px !important; padding: 12px 32px !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 30px rgba(0,229,255,0.3) !important; }

[data-testid="metric-container"] {
    background: #131C2C !important; border: 1px solid #1E2A3E !important; border-radius: 16px !important; padding: 18px !important;
}
[data-testid="metric-container"] label { color: #00E5FF !important; font-size: 12px !important; text-transform: uppercase !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00E5FF !important; font-family: 'JetBrains Mono', monospace !important; font-size: 30px !important; font-weight: 700 !important;
}

/* Toast notification styles */
.toast-success {
    position: fixed; top: 80px; right: 24px; z-index: 9999;
    background: #0D2E1A; border: 1px solid #00E676; border-radius: 12px;
    padding: 14px 20px; color: #00E676; font-size: 13px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    animation: slideIn 0.3s ease;
}
.toast-error {
    position: fixed; top: 80px; right: 24px; z-index: 9999;
    background: #2E0D0D; border: 1px solid #FF5252; border-radius: 12px;
    padding: 14px 20px; color: #FF5252; font-size: 13px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    animation: slideIn 0.3s ease;
}
@keyframes slideIn { from { transform: translateX(120%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

/* Calibrated/estimated badge */
.badge-estimated {
    background: rgba(255,165,0,0.15); color: #FFA94D;
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    padding: 3px 8px; border-radius: 6px; font-weight: 700;
    border: 1px solid rgba(255,165,0,0.3); margin-left: 6px;
}
.badge-calibrated {
    background: rgba(0,230,118,0.12); color: #00E676;
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    padding: 3px 8px; border-radius: 6px; font-weight: 700;
    border: 1px solid rgba(0,230,118,0.3); margin-left: 6px;
}

/* ABC class badges */
.abc-a { background: rgba(255,82,82,0.15); color: #FF5252; border: 1px solid rgba(255,82,82,0.35); border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.abc-b { background: rgba(255,169,77,0.15); color: #FFA94D; border: 1px solid rgba(255,169,77,0.35); border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.abc-c { background: rgba(0,229,255,0.12); color: #00E5FF; border: 1px solid rgba(0,229,255,0.3); border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

hr { border-color: #1E2A3E !important; }
label { color: #FFFFFF !important; font-size: 13px !important; font-weight: 600 !important; text-transform: uppercase !important; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0F1525; }
::-webkit-scrollbar-thumb { background: #2D3A5E; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CATEGORY STATS (from category_stats.json — ground truth for sanity checks)
# ─────────────────────────────────────────────
CATEGORY_STATS = {
    "bed_bath_table":      {"avg_daily_demand": 18.35},
    "health_beauty":       {"avg_daily_demand": 15.72},
    "sports_leisure":      {"avg_daily_demand": 13.98},
    "furniture_decor":     {"avg_daily_demand": 13.51},
    "computers_accessories": {"avg_daily_demand": 12.89},
    "housewares":          {"avg_daily_demand": 11.60},
    "watches_gifts":       {"avg_daily_demand": 10.26},
    "telephony":           {"avg_daily_demand": 7.59},
    "garden_tools":        {"avg_daily_demand": 7.33},
    "auto":                {"avg_daily_demand": 7.15},
}

# ─────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_categories() -> dict:
    r = requests.get(f"{API_BASE_URL}/categories", timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_forecast(category: str, model: str = "auto") -> dict:
    r = requests.post(f"{API_BASE_URL}/forecast", json={"category": category, "model": model}, timeout=30)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"[{r.status_code}] {detail}")
    return r.json()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_all_forecasts(categories: tuple, model: str = "auto") -> dict:
    out = {}
    for c in categories:
        try:
            out[c] = fetch_forecast(c, model)
        except Exception as e:
            out[c] = {"error": str(e)}
    return out


def fetch_explain(category: str) -> dict:
    r = requests.post(f"{API_BASE_URL}/explain", json={"category": category, "model": "auto"}, timeout=30)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"[{r.status_code}] {detail}")
    return r.json()


# ─────────────────────────────────────────────
# STATIC REFERENCE DATA
# ─────────────────────────────────────────────
LEADERBOARD = [
    ("GBM (LightGBM)", 18.56, "Production — 9/10 categories"),
    ("LSTM",           20.44, "Production — telephony only"),
]
# Baseline deliberately excluded from LEADERBOARD display —
# API returns 501, so listing it would be misleading.

UNIT_COST_DEFAULT     = 25.0
HOLDING_RATE_DEFAULT  = 0.20
ORDERING_COST_DEFAULT = 500.0

# ─────────────────────────────────────────────
# INVENTORY MATH
# ─────────────────────────────────────────────
def z_score_for_service_level(sl: float) -> float:
    table = {0.50: 0.00, 0.80: 0.84, 0.85: 1.04, 0.90: 1.28,
             0.95: 1.65, 0.975: 1.96, 0.99: 2.33, 0.995: 2.58, 0.999: 3.09}
    return table[min(table.keys(), key=lambda k: abs(k - sl))]


def compute_inventory_plan(forecast: list[float], interval_95: dict | None,
                            lead_time_days: int, service_level: float,
                            unit_cost: float, holding_rate: float,
                            ordering_cost: float) -> dict:
    avg_daily_demand = float(np.mean(forecast)) if forecast else 0.0
    interval_based_std = False

    if interval_95 and interval_95.get("upper") and interval_95.get("lower"):
        widths = [u - l for u, l in zip(interval_95["upper"], interval_95["lower"])]
        if widths and max(widths) > 0:
            avg_std = float(np.mean(widths)) / (2 * 1.96)
            interval_based_std = True
        else:
            avg_std = avg_daily_demand * 0.25
    else:
        avg_std = avg_daily_demand * 0.25

    z = z_score_for_service_level(service_level)
    lead_time_demand = avg_daily_demand * lead_time_days
    safety_stock = z * avg_std * math.sqrt(max(lead_time_days, 1))
    reorder_point = lead_time_demand + safety_stock

    annual_demand = avg_daily_demand * 365
    holding_cost_per_unit = unit_cost * holding_rate
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit) if holding_cost_per_unit > 0 else 0.0
    orders_per_year = annual_demand / eoq if eoq > 0 else 0.0
    annual_ordering_cost = orders_per_year * ordering_cost
    annual_holding_cost = (eoq / 2) * holding_cost_per_unit if eoq > 0 else 0.0
    total_annual_cost = annual_ordering_cost + annual_holding_cost

    # Stockout risk: P(stockout during LT) = 1 - service_level
    stockout_prob = 1.0 - service_level
    stockout_cost_per_unit = 2.0 * unit_cost   # standard: lost sale + goodwill = 2× unit cost
    expected_stockout_cost = stockout_prob * avg_daily_demand * lead_time_days * stockout_cost_per_unit

    return {
        "avg_daily_demand":      avg_daily_demand,
        "avg_std":               avg_std,
        "lead_time_demand":      lead_time_demand,
        "safety_stock":          safety_stock,
        "reorder_point":         reorder_point,
        "eoq":                   eoq,
        "orders_per_year":       orders_per_year,
        "annual_ordering_cost":  annual_ordering_cost,
        "annual_holding_cost":   annual_holding_cost,
        "total_annual_cost":     total_annual_cost,
        "annual_demand":         annual_demand,
        "interval_based_std":    interval_based_std,
        "stockout_prob":         stockout_prob,
        "stockout_cost_per_unit": stockout_cost_per_unit,
        "expected_stockout_cost": expected_stockout_cost,
    }


def compute_business_impact(plan: dict, unit_cost: float,
                             gbm_mape: float = 18.56,
                             baseline_mape: float = 19.17) -> dict:
    """
    Estimate ₹ annual savings vs naive (baseline) ordering.

    Logic:
    - MAPE improvement = baseline_mape - gbm_mape = 0.61%
    - Each 1% MAPE maps to ~1% error in demand estimate
    - Higher demand error → more safety stock needed (or more stockouts)
    - We model it as: reduction in excess holding cost + reduction in stockouts
    - Stockout cost = 2× unit cost (standard assumption)
    - Holding cost = unit_cost × holding_rate (from sidebar)

    All assumptions are labeled in the UI.
    """
    mape_improvement_pct = baseline_mape - gbm_mape  # 0.61%
    annual_demand = plan["annual_demand"]

    # Reduction in demand error (units/year)
    units_better_estimated = annual_demand * (mape_improvement_pct / 100.0)

    # Fraction that would have been excess stock (held at cost)
    excess_holding_savings = units_better_estimated * unit_cost * 0.20   # holding rate

    # Fraction that would have been stockouts at 2× unit cost
    stockout_savings = units_better_estimated * unit_cost * 2.0 * 0.5   # ~half goes to stockouts

    total_savings = excess_holding_savings + stockout_savings

    # Cost per 1% MAPE improvement
    cost_per_mape_pct = total_savings / mape_improvement_pct if mape_improvement_pct > 0 else 0

    return {
        "mape_improvement_pct":   mape_improvement_pct,
        "total_annual_savings":   total_savings,
        "excess_holding_savings": excess_holding_savings,
        "stockout_savings":       stockout_savings,
        "cost_per_mape_pct":      cost_per_mape_pct,
    }


def abc_classify(demand_share_pct: float) -> str:
    """Classify based on cumulative revenue/demand share (Pareto principle)."""
    if demand_share_pct >= 70:
        return "A"
    elif demand_share_pct >= 90:
        return "B"
    else:
        return "C"


def abc_classify_from_rank(rank: int, total: int) -> str:
    """Top 20% → A, next 30% → B, bottom 50% → C."""
    pct = rank / total
    if pct <= 0.20:
        return "A"
    elif pct <= 0.50:
        return "B"
    else:
        return "C"


def risk_color(conf: str) -> str:
    return {"high": "#00E676", "low": "#FFB74D"}.get(conf, None)


def source_tag(source: str) -> str:
    color = "#00E5FF" if source == "API" else "#FFA94D"
    return (f"<span style='background:{color}22;color:{color};font-size:9px;"
            f"font-family:JetBrains Mono,monospace;padding:2px 6px;border-radius:6px;"
            f"margin-left:6px;font-weight:700;'>{source}</span>")


def show_toast(message: str, kind: str = "success"):
    icon = "✓" if kind == "success" else "✗"
    cls = f"toast-{kind}"
    st.markdown(f"<div class='{cls}'>{icon} {message}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONNECT
# ─────────────────────────────────────────────
try:
    cat_info = fetch_categories()
    known_categories = tuple(cat_info["known_categories"])
    API_UP = True
except requests.exceptions.ConnectionError:
    API_UP = False
    known_categories = tuple()
except requests.exceptions.RequestException as e:
    API_UP = False
    known_categories = tuple()
    st.error(f"Unexpected error reaching the API: {e}")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:25px 0 15px;'>
        <div style='font-size:28px;background:linear-gradient(135deg, #00E5FF, #FFA94D);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    font-weight:800;letter-spacing:-0.5px;'>📦 SupplyChain IQ</div>
        <div style='color:#FFFFFF;font-size:11px;letter-spacing:1px;margin-top:6px;font-weight:600;'>
            DEMAND FORECASTING & DECISION INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    status_color = "#00E676" if API_UP else "#FF5252"
    status_text  = "API CONNECTED" if API_UP else "API OFFLINE"
    st.markdown(f"""
    <div style='background:#131C2C;border:1px solid {status_color};border-radius:10px;
                padding:10px 14px;margin-bottom:14px;text-align:center;'>
        <span style='color:{status_color};font-family:JetBrains Mono,monospace;font-size:12px;font-weight:700;'>
            ● {status_text}
        </span>
    </div>""", unsafe_allow_html=True)

    with st.expander("🏆 MODEL LEADERBOARD", expanded=True):
        st.caption("Baseline excluded — API returns 501 (not implemented).")
        for name, mape, note in LEADERBOARD:
            color = PALETTE["gbm"] if "GBM" in name else PALETTE["lstm"]
            st.markdown(f"""
            <div style='margin-bottom:10px;'>
                <div style='display:flex;justify-content:space-between;'>
                    <span style='color:#EDF2F7;font-size:12px;'>{name}</span>
                    <span style='color:{color};font-family:JetBrains Mono,monospace;font-size:12px;font-weight:700;'>{mape}%</span>
                </div>
                <div style='color:#FFFFFF;font-size:10px;'>{note}</div>
            </div>""", unsafe_allow_html=True)

    with st.expander("⚙️ INVENTORY ASSUMPTIONS", expanded=True):
        st.caption("Used in Inventory & What-If tabs — not from the API.")
        sb_unit_cost      = st.number_input("Unit cost (₹/unit)", value=UNIT_COST_DEFAULT, min_value=1.0, step=1.0)
        sb_holding_rate   = st.slider("Annual holding rate (% of unit cost)", 5, 50, int(HOLDING_RATE_DEFAULT * 100)) / 100
        sb_ordering_cost  = st.number_input("Cost per order (₹)", value=ORDERING_COST_DEFAULT, min_value=10.0, step=50.0)
        sb_lead_time      = st.slider("Supplier lead time (days)", 1, 30, 7)
        sb_service_level  = st.select_slider("Target service level", options=[0.80, 0.85, 0.90, 0.95, 0.975, 0.99], value=0.95)

    st.divider()
    st.markdown("""
    <div style='color:#FFFFFF;font-size:10px;text-align:center;padding:10px;'>
        ⚡ LightGBM + LSTM · Auto Model Routing<br>
        🎯 Conformal Intervals · SHAP<br>
        🐍 FastAPI · MySQL · Olist Dataset
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:24px;'>
    <div class='main-heading'>Supply Chain Intelligence Platform</div>
    <div class='sub-heading'>Demand Forecasting → Inventory Decisions, with Uncertainty Quantification</div>
    <div class='tech-line'>LightGBM + LSTM · Auto Model Selection · Conformal Prediction Intervals · FastAPI</div>
</div>
""", unsafe_allow_html=True)

if not API_UP:
    st.error(
        f"Can't reach the forecast API at `{API_BASE_URL}`. Start it first:\n\n"
        "`python -m uvicorn api.main:app --reload --port 8000`\n\n"
        "Everything below depends on it."
    )
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Forecast Explorer",
    "📦 Inventory Engine",
    "🔄 What-If Simulator",
    "🌐 Portfolio Overview",
    "🧠 Model Trust Center",
    "📋 Executive Summary",
    "ℹ️ System Overview",
])

# ══════════════════════════════════════════════════════
# TAB 1 — FORECAST EXPLORER
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown("### Category Forecast Explorer")
    st.markdown("<p style='color:#00E5FF;font-size:14px;'>Pick a category, see the 7-day forecast with calibrated uncertainty bands.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        category = st.selectbox("Category", known_categories, index=0, key="cat1")
    with c2:
        # Baseline is disabled — shown with label so user knows why
        model_options = ["auto", "gbm", "lstm", "── baseline (unavailable) ──"]
        model_choice_raw = st.selectbox("Model", model_options, index=0, key="model1")
        if "baseline" in model_choice_raw:
            st.caption("⚠️ Baseline not implemented in API (returns 501). Select auto/gbm/lstm.")
            model_choice = None
        else:
            model_choice = model_choice_raw

    if st.button("Get Forecast", type="primary", key="btn1", disabled=(model_choice is None)):
        with st.spinner("Calling forecast API..."):
            try:
                result = fetch_forecast(category, model_choice)
                show_toast(f"Forecast loaded for {category}", "success")
            except RuntimeError as e:
                show_toast(f"Forecast failed: {e}", "error")
                st.error(f"Forecast request failed: {e}")
                st.stop()
            except requests.exceptions.RequestException as e:
                show_toast("Could not reach the API", "error")
                st.error(f"Could not reach the API: {e}")
                st.stop()
        st.session_state["last_forecast"] = result
        st.session_state["last_category"] = category

    if "last_forecast" not in st.session_state:
        st.info("Pick a category and model, then click **Get Forecast**.")
    else:
        result   = st.session_state["last_forecast"]
        category = st.session_state["last_category"]
        conf     = result.get("confidence", "unknown")
        badge_color = risk_color(conf)

        # ── Metric cards ─────────────────────────────
        avg_d    = float(np.mean(result["forecast"]))
        total_7d = float(np.sum(result["forecast"]))
        expected_daily = CATEGORY_STATS.get(category, {}).get("avg_daily_demand", None)

        m1, m2, m3, m4 = st.columns(4)

        # Model used card
        model_color = PALETTE["gbm"] if result["model_used"].lower() == "gbm" else PALETTE["lstm"]
        m1.markdown(f"""
        <div style='background:#131C2C;border:1px solid {model_color};border-radius:16px;padding:18px;text-align:center;'>
            <div style='color:#FFFFFF;font-size:11px;text-transform:uppercase;margin-bottom:8px;letter-spacing:1px;'>Model Used</div>
            <div style='color:{model_color};font-size:24px;font-weight:700;font-family:JetBrains Mono,monospace;'>{result['model_used'].upper()}</div>
        </div>""", unsafe_allow_html=True)

        # Confidence card — only show if known
        if badge_color:
            conf_label = conf.upper()
            conf_icon  = "●" if conf == "high" else "◐"
        else:
            conf_label = "N/A"
            conf_icon  = "○"
            badge_color = "#2D3A5E"

        m2.markdown(f"""
        <div style='background:#131C2C;border:1px solid {badge_color};border-radius:16px;padding:18px;text-align:center;'>
            <div style='color:#FFFFFF;font-size:11px;text-transform:uppercase;margin-bottom:8px;letter-spacing:1px;'>Confidence</div>
            <div style='color:{badge_color};font-size:24px;font-weight:700;font-family:JetBrains Mono,monospace;'>{conf_icon} {conf_label}</div>
        </div>""", unsafe_allow_html=True)

        # Avg daily demand with sanity check indicator
        demand_ok = True
        if expected_daily and avg_d > 0:
            ratio = avg_d / expected_daily
            demand_ok = 0.3 < ratio < 3.0   # within 3× of historical avg

        demand_color = PALETTE["alert"] if not demand_ok else "#FFA94D"
        demand_flag  = " ⚠️" if not demand_ok else ""
        m3.markdown(f"""
        <div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:16px;padding:18px;text-align:center;'>
            <div style='color:#FFFFFF;font-size:11px;text-transform:uppercase;margin-bottom:8px;letter-spacing:1px;'>Avg Daily Demand{demand_flag}</div>
            <div style='color:{demand_color};font-size:24px;font-weight:700;font-family:JetBrains Mono,monospace;'>{avg_d:.2f}</div>
            {"<div style='color:#FFFFFF;font-size:10px;margin-top:4px;'>Historical avg: " + f"{expected_daily:.2f}" + "</div>" if expected_daily else ""}
        </div>""", unsafe_allow_html=True)

        m4.markdown(f"""
        <div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:16px;padding:18px;text-align:center;'>
            <div style='color:#FFFFFF;font-size:11px;text-transform:uppercase;margin-bottom:8px;letter-spacing:1px;'>7-Day Total</div>
            <div style='color:#FFA94D;font-size:24px;font-weight:700;font-family:JetBrains Mono,monospace;'>{total_7d:.1f}</div>
        </div>""", unsafe_allow_html=True)

        # Scaling bug warning
        if not demand_ok and expected_daily:
            st.warning(
                f"⚠️ **Scaling mismatch detected for '{category}'**: "
                f"Forecast avg = {avg_d:.3f} units/day but historical avg = {expected_daily:.2f} units/day "
                f"(ratio: {avg_d/expected_daily:.3f}×). "
                f"This likely means the API is returning log-space predictions without `expm1()`. "
                f"Inventory numbers below will be unreliable until this is fixed in the API."
            )

        st.caption(result.get("selection_reason", ""))
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Forecast chart ────────────────────────────
        days = [f"Day {i+1}" for i in range(len(result["forecast"]))]
        fig  = go.Figure()

        has_interval = False
        if result.get("interval_95"):
            iv = result["interval_95"]
            upper = iv.get("upper", [])
            lower = iv.get("lower", [])
            if upper and lower and any(u > 0 for u in upper):
                has_interval = True
                fig.add_trace(go.Scatter(
                    x=days + days[::-1], y=upper + lower[::-1],
                    fill="toself", fillcolor=f"rgba(0,229,255,0.10)",
                    line=dict(color="rgba(255,255,255,0)"), name="95% interval", hoverinfo="skip"))

        if result.get("interval_80"):
            iv = result["interval_80"]
            upper80 = iv.get("upper", [])
            lower80 = iv.get("lower", [])
            if upper80 and lower80 and any(u > 0 for u in upper80):
                fig.add_trace(go.Scatter(
                    x=days + days[::-1], y=upper80 + lower80[::-1],
                    fill="toself", fillcolor=f"rgba(0,229,255,0.22)",
                    line=dict(color="rgba(255,255,255,0)"), name="80% interval", hoverinfo="skip"))

        line_color = PALETTE["gbm"] if result["model_used"].lower() == "gbm" else PALETTE["lstm"]
        fig.add_trace(go.Scatter(
            x=days, y=result["forecast"], mode="lines+markers",
            line=dict(color=line_color, width=3), marker=dict(size=9, color=line_color), name="Forecast"))

        fig.update_layout(
            title=f"7-Day Demand Forecast — {category} ({result['model_used'].upper()})",
            paper_bgcolor="#131C2C", plot_bgcolor="#131C2C", font={"color": "#EDF2F7"},
            xaxis={"gridcolor": "#1E2A3E"}, yaxis={"gridcolor": "#1E2A3E", "title": "Units Sold"},
            hovermode="x unified", height=440, margin=dict(t=60, b=40),
            legend={"font": {"color": "#FFFFFF"}, "bgcolor": "rgba(0,0,0,0)"})
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        if not has_interval:
            st.caption("ℹ️ No calibrated interval returned for this model/category (known gap on LSTM routes). "
                       "Inventory tab will use a 25%-of-demand heuristic fallback.")

        with st.expander("Raw API response"):
            st.json(result)

        # ── Export button ─────────────────────────────
        forecast_df = pd.DataFrame({"Day": days, "Forecast_Units": result["forecast"]})
        if result.get("interval_95") and has_interval:
            forecast_df["Lower_95"] = result["interval_95"]["lower"]
            forecast_df["Upper_95"] = result["interval_95"]["upper"]
        st.download_button(
            "📥 Export Forecast (CSV)",
            data=forecast_df.to_csv(index=False),
            file_name=f"forecast_{category}_{result['model_used']}.csv",
            mime="text/csv")

# ══════════════════════════════════════════════════════
# TAB 2 — INVENTORY DECISION ENGINE
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("### Inventory Decision Engine")
    st.markdown(
        "<p style='color:#00E5FF;font-size:14px;'>Turns the forecast into the three numbers a "
        "supply planner actually needs: safety stock, reorder point, and EOQ. "
        "Business impact estimates assume stockout cost = 2× unit cost (labeled throughout).</p>",
        unsafe_allow_html=True)

    if "last_forecast" not in st.session_state:
        st.info("Run a forecast in the **Forecast Explorer** tab first.")
    else:
        result   = st.session_state["last_forecast"]
        category = st.session_state["last_category"]
        plan = compute_inventory_plan(
            result["forecast"], result.get("interval_95"),
            sb_lead_time, sb_service_level, sb_unit_cost, sb_holding_rate, sb_ordering_cost)
        impact = compute_business_impact(plan, sb_unit_cost)

        # ── Fallback warning banner ───────────────────
        if not plan["interval_based_std"]:
            st.markdown("""
            <div style='background:#1E1500;border:1px solid #FFA94D;border-radius:10px;
                        padding:12px 16px;margin-bottom:16px;'>
                <span style='color:#FFA94D;font-family:JetBrains Mono,monospace;font-size:11px;font-weight:700;'>
                    ⚠ ESTIMATED — NOT CALIBRATED
                </span><br>
                <span style='color:#EDF2F7;font-size:13px;'>
                    No 95% interval returned (LSTM route — known gap). Safety stock uses a 25%-of-demand 
                    heuristic. Numbers here are directionally useful but less precise than GBM-routed categories.
                </span>
            </div>""", unsafe_allow_html=True)

        calibration_badge = (
            '<span class="badge-calibrated">CALIBRATED</span>' if plan["interval_based_std"]
            else '<span class="badge-estimated">ESTIMATED</span>'
        )

        st.markdown(
            f"#### Plan for: **{category}** — {sb_lead_time}d lead time, "
            f"{int(sb_service_level*100)}% service level &nbsp;{calibration_badge}",
            unsafe_allow_html=True)
        st.markdown(
            f"{source_tag('API')} forecast + interval &nbsp;&nbsp; {source_tag('DASHBOARD')} EOQ / safety stock / reorder point",
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Inventory decision cards with color-coded thresholds ──
        c1, c2, c3, c4 = st.columns(4)

        rop   = plan["reorder_point"]
        ss    = plan["safety_stock"]
        eoq   = plan["eoq"]
        opy   = plan["orders_per_year"]

        # Color-code: if safety stock < 5 units flag it red (very low buffer)
        ss_color   = PALETTE["alert"] if ss < 5 else PALETTE["success"]
        rop_color  = "#FFA94D"

        c1.markdown(f"""
        <div style='background:#131C2C;border:2px solid {rop_color};border-radius:16px;padding:20px;text-align:center;'>
            <div style='font-size:20px;margin-bottom:6px;'>🔁</div>
            <div style='color:#FFFFFF;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Reorder Point</div>
            <div style='color:{rop_color};font-size:28px;font-weight:700;font-family:JetBrains Mono,monospace;'>{rop:.0f}</div>
            <div style='color:#FFFFFF;font-size:11px;margin-top:4px;'>units</div>
        </div>""", unsafe_allow_html=True)

        c2.markdown(f"""
        <div style='background:#131C2C;border:2px solid {ss_color};border-radius:16px;padding:20px;text-align:center;'>
            <div style='font-size:20px;margin-bottom:6px;'>🛡️</div>
            <div style='color:#FFFFFF;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Safety Stock</div>
            <div style='color:{ss_color};font-size:28px;font-weight:700;font-family:JetBrains Mono,monospace;'>{ss:.0f}</div>
            <div style='color:#FFFFFF;font-size:11px;margin-top:4px;'>{"⚠ low buffer" if ss < 5 else "units"}</div>
        </div>""", unsafe_allow_html=True)

        c3.markdown(f"""
        <div style='background:#131C2C;border:2px solid {PALETTE["gbm"]};border-radius:16px;padding:20px;text-align:center;'>
            <div style='font-size:20px;margin-bottom:6px;'>📦</div>
            <div style='color:#FFFFFF;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>EOQ</div>
            <div style='color:{PALETTE["gbm"]};font-size:28px;font-weight:700;font-family:JetBrains Mono,monospace;'>{eoq:.0f}</div>
            <div style='color:#FFFFFF;font-size:11px;margin-top:4px;'>units/order</div>
        </div>""", unsafe_allow_html=True)

        c4.markdown(f"""
        <div style='background:#131C2C;border:2px solid #B388FF;border-radius:16px;padding:20px;text-align:center;'>
            <div style='font-size:20px;margin-bottom:6px;'>🔄</div>
            <div style='color:#FFFFFF;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Orders / Year</div>
            <div style='color:#B388FF;font-size:28px;font-weight:700;font-family:JetBrains Mono,monospace;'>{opy:.1f}</div>
            <div style='color:#FFFFFF;font-size:11px;margin-top:4px;'>orders</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Stockout risk card ───────────────────────
        stockout_pct = plan["stockout_prob"] * 100
        revenue_at_risk = plan["expected_stockout_cost"]
        st.markdown(f"""
        <div style='background:#1A0E0E;border:1px solid {PALETTE["alert"]};border-radius:12px;padding:16px 20px;margin-bottom:16px;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <div style='color:{PALETTE["alert"]};font-family:JetBrains Mono,monospace;font-size:11px;font-weight:700;'>⚠ STOCKOUT RISK AT {int(sb_service_level*100)}% SERVICE LEVEL</div>
                    <div style='color:#EDF2F7;font-size:13px;margin-top:6px;'>
                        {stockout_pct:.1f}% chance of stockout during the {sb_lead_time}-day lead time window.
                        At 2× unit cost assumption, expected cost of a stockout event = 
                        <span style='color:{PALETTE["alert"]};font-weight:700;font-family:JetBrains Mono,monospace;'>
                            ₹{revenue_at_risk:,.0f}
                        </span>
                    </div>
                    <div style='color:#FFFFFF;font-size:10px;margin-top:6px;'>Stockout cost assumed = 2× unit cost (lost sale + goodwill). Adjust unit cost in sidebar to recalculate.</div>
                </div>
                <div style='text-align:center;padding-left:20px;min-width:80px;'>
                    <div style='color:{PALETTE["alert"]};font-size:32px;font-weight:800;font-family:JetBrains Mono,monospace;'>{stockout_pct:.0f}%</div>
                    <div style='color:#FFFFFF;font-size:10px;'>risk</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Business impact card ─────────────────────
        st.markdown(f"""
        <div style='background:#0A1E0A;border:1px solid {PALETTE["success"]};border-radius:12px;padding:16px 20px;margin-bottom:16px;'>
            <div style='color:{PALETTE["success"]};font-family:JetBrains Mono,monospace;font-size:11px;font-weight:700;'>
                💰 ESTIMATED ANNUAL SAVINGS vs NAIVE BASELINE
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:12px;'>
                <div style='text-align:center;'>
                    <div style='color:#FFFFFF;font-size:10px;text-transform:uppercase;letter-spacing:1px;'>MAPE Improvement</div>
                    <div style='color:{PALETTE["success"]};font-size:22px;font-weight:700;font-family:JetBrains Mono,monospace;'>{impact["mape_improvement_pct"]:.2f}%</div>
                    <div style='color:#FFFFFF;font-size:10px;'>GBM vs baseline</div>
                </div>
                <div style='text-align:center;'>
                    <div style='color:#FFFFFF;font-size:10px;text-transform:uppercase;letter-spacing:1px;'>Holding Cost Saved</div>
                    <div style='color:{PALETTE["success"]};font-size:22px;font-weight:700;font-family:JetBrains Mono,monospace;'>₹{impact["excess_holding_savings"]:,.0f}</div>
                    <div style='color:#FFFFFF;font-size:10px;'>less excess stock</div>
                </div>
                <div style='text-align:center;'>
                    <div style='color:#FFFFFF;font-size:10px;text-transform:uppercase;letter-spacing:1px;'>Total Annual Savings</div>
                    <div style='color:{PALETTE["success"]};font-size:26px;font-weight:800;font-family:JetBrains Mono,monospace;'>₹{impact["total_annual_savings"]:,.0f}</div>
                    <div style='color:#FFFFFF;font-size:10px;'>vs naive ordering</div>
                </div>
            </div>
            <div style='color:#FFFFFF;font-size:10px;margin-top:10px;border-top:1px solid #1E2A3E;padding-top:8px;'>
                Assumptions: stockout cost = 2× unit cost · holding rate = {sb_holding_rate*100:.0f}% · 
                GBM MAPE 18.56% vs baseline 19.17% · per-category estimate scaled from annual demand.
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Cost breakdown + Demand inputs ───────────
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**📊 Annual Cost Breakdown at EOQ**")
            fig_cost = go.Figure(go.Bar(
                x=["Ordering", "Holding", "Total"],
                y=[plan["annual_ordering_cost"], plan["annual_holding_cost"], plan["total_annual_cost"]],
                marker_color=[PALETTE["cost_ordering"], PALETTE["cost_holding"], PALETTE["cost_total"]],
                text=[f"₹{v:,.0f}" for v in [plan["annual_ordering_cost"], plan["annual_holding_cost"], plan["total_annual_cost"]]],
                textposition="outside"))
            fig_cost.update_layout(paper_bgcolor="#131C2C", plot_bgcolor="#131C2C", font={"color": "#EDF2F7"},
                height=300, margin=dict(l=20, r=20, t=20, b=20),
                yaxis={"gridcolor": "#1E2A3E", "title": "₹/year"}, xaxis={"color": "#FFFFFF"},
                showlegend=False)
            st.plotly_chart(fig_cost, use_container_width=True, config={"displaylogo": False})

        with cc2:
            st.markdown("**📐 Demand Inputs Used**")
            for label, val in [
                ("Avg daily demand",   f"{plan['avg_daily_demand']:.2f} units"),
                ("Demand std (proxy)", f"{plan['avg_std']:.2f} units"),
                ("Lead-time demand",   f"{plan['lead_time_demand']:.2f} units"),
                ("Annualised demand",  f"{plan['annual_demand']:.0f} units"),
                ("Unit cost",          f"₹{sb_unit_cost:.2f}"),
                ("Holding rate",       f"{sb_holding_rate*100:.0f}% / year"),
            ]:
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1E2A3E;'>
                    <span style='color:#FFFFFF;font-size:13px;'>{label}</span>
                    <span style='color:#00E5FF;font-family:JetBrains Mono,monospace;font-size:13px;'>{val}</span>
                </div>""", unsafe_allow_html=True)

        # ── Action card ───────────────────────────────
        st.markdown(f"""
        <div style='background:#0D2E1A;border:1px solid {PALETTE["success"]};border-radius:12px;padding:18px 22px;margin-top:16px;'>
            <div style='color:{PALETTE["success"]};font-family:JetBrains Mono,monospace;font-size:11px;font-weight:700;margin-bottom:8px;'>
                📋 RECOMMENDED ACTION
            </div>
            <div style='color:#EDF2F7;font-size:14px;line-height:1.7;'>
                When stock for <b>{category}</b> hits <b style='color:#FFA94D;'>{rop:.0f} units</b>, 
                place an order of <b style='color:{PALETTE["gbm"]};'>{eoq:.0f} units</b>. 
                This covers the {sb_lead_time}-day lead time at a {int(sb_service_level*100)}% service level,
                with a safety buffer of <b style='color:{ss_color};'>{ss:.0f} units</b> 
                against demand variability.
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 3 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown("### What-If Simulator")
    st.markdown(
        "<p style='color:#00E5FF;font-size:14px;'>Adjust lead time, service level, and demand scenarios. "
        "No API calls on slider moves — all computed locally from your last forecast.</p>",
        unsafe_allow_html=True)

    if "last_forecast" not in st.session_state:
        st.info("Run a forecast in the **Forecast Explorer** tab first.")
    else:
        result   = st.session_state["last_forecast"]
        category = st.session_state["last_category"]

        w1, w2 = st.columns(2)
        with w1:
            sim_lead_time = st.slider("Simulated lead time (days)", 1, 30, sb_lead_time, key="sim_lt")
        with w2:
            sim_service_level = st.select_slider(
                "Simulated service level", options=[0.80, 0.85, 0.90, 0.95, 0.975, 0.99],
                value=sb_service_level, key="sim_sl")

        # ── Festive / demand spike section ───────────
        st.markdown("---")
        st.markdown("#### 🎉 Festive Season / Demand Spike Stress Test")
        st.caption(
            "Simulates a demand surge (e.g. Black Friday, Diwali equivalent on Olist). "
            "Multiplies forecast by uplift factor to show how safety stock requirements change. "
            "No retraining — this is a planning overlay, not a new model prediction."
        )

        spike_col1, spike_col2 = st.columns([2, 1])
        with spike_col1:
            demand_uplift_pct = st.slider("Expected demand uplift (%)", 0, 200, 0, step=10,
                                           key="demand_uplift",
                                           help="0% = baseline forecast. 100% = demand doubles.")
        with spike_col2:
            spike_duration_days = st.slider("Spike duration (days)", 1, 14, 7, key="spike_dur")

        uplift_factor = 1.0 + (demand_uplift_pct / 100.0)
        spiked_forecast = [v * uplift_factor for v in result["forecast"]]

        # Scale intervals if available
        spiked_interval = None
        if result.get("interval_95") and result["interval_95"].get("upper"):
            spiked_interval = {
                "upper": [v * uplift_factor for v in result["interval_95"]["upper"]],
                "lower": [v * uplift_factor for v in result["interval_95"]["lower"]],
            }

        base_plan = compute_inventory_plan(result["forecast"], result.get("interval_95"),
                                            sim_lead_time, sim_service_level,
                                            sb_unit_cost, sb_holding_rate, sb_ordering_cost)
        sim_plan  = compute_inventory_plan(spiked_forecast, spiked_interval,
                                            sim_lead_time, sim_service_level,
                                            sb_unit_cost, sb_holding_rate, sb_ordering_cost)

        rop_delta  = sim_plan["reorder_point"]   - base_plan["reorder_point"]
        ss_delta   = sim_plan["safety_stock"]    - base_plan["safety_stock"]
        cost_delta = sim_plan["total_annual_cost"] - base_plan["total_annual_cost"]

        r1, r2, r3 = st.columns(3)
        r1.metric("Reorder Point",     f"{sim_plan['reorder_point']:.1f}",    delta=f"{rop_delta:+.1f} vs base")
        r2.metric("Safety Stock",      f"{sim_plan['safety_stock']:.1f}",     delta=f"{ss_delta:+.1f} vs base")
        r3.metric("Total Annual Cost", f"₹{sim_plan['total_annual_cost']:,.0f}", delta=f"₹{cost_delta:+,.0f} vs base")

        if demand_uplift_pct > 0:
            ss_gap = sim_plan["safety_stock"] - base_plan["safety_stock"]
            st.markdown(f"""
            <div style='background:#1A0E00;border:1px solid #FFA94D;border-radius:10px;padding:14px 18px;margin-top:8px;'>
                <span style='color:#FFA94D;font-family:JetBrains Mono,monospace;font-size:11px;font-weight:700;'>
                    🎉 SPIKE IMPACT — {demand_uplift_pct}% demand uplift for {spike_duration_days} days
                </span><br>
                <span style='color:#EDF2F7;font-size:13px;'>
                    Current safety stock ({base_plan["safety_stock"]:.0f} units) would need to increase by 
                    <b style='color:#FFA94D;'>{ss_gap:.0f} units</b> to maintain the same {int(sim_service_level*100)}% service level.
                    Pre-positioning <b>{sim_plan["eoq"]:.0f} units</b> before the spike avoids a stockout.
                </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tradeoff charts ───────────────────────────
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("#### 📈 Service Level vs Safety Stock")
            sl_range  = [0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
            ss_values = [compute_inventory_plan(spiked_forecast, spiked_interval,
                                                 sim_lead_time, sl,
                                                 sb_unit_cost, sb_holding_rate, sb_ordering_cost)["safety_stock"]
                         for sl in sl_range]
            fig_tradeoff = go.Figure(go.Scatter(
                x=[f"{int(s*100)}%" for s in sl_range], y=ss_values,
                mode="lines+markers",
                line=dict(color=PALETTE["gbm"], width=3),
                marker=dict(size=10, color=PALETTE["lstm"])))
            fig_tradeoff.update_layout(
                paper_bgcolor="#131C2C", plot_bgcolor="#131C2C", font={"color": "#EDF2F7"},
                height=300, margin=dict(l=20, r=20, t=20, b=20),
                xaxis={"title": "Target Service Level", "color": "#FFFFFF"},
                yaxis={"gridcolor": "#1E2A3E", "title": "Safety Stock (units)"})
            st.plotly_chart(fig_tradeoff, use_container_width=True, config={"displaylogo": False})
            st.caption("Higher service levels require more safety stock — use this to justify your chosen level.")

        with ch2:
            st.markdown("#### 🚚 Lead Time vs Reorder Point")
            lt_range   = list(range(1, 22, 2))
            rop_values = [compute_inventory_plan(spiked_forecast, spiked_interval,
                                                  lt, sim_service_level,
                                                  sb_unit_cost, sb_holding_rate, sb_ordering_cost)["reorder_point"]
                          for lt in lt_range]
            bar_colors = [PALETTE["alert"] if abs(lt - sim_lead_time) <= 1 else "#2D3A5E" for lt in lt_range]
            fig_lt = go.Figure(go.Bar(
                x=[f"{lt}d" for lt in lt_range], y=rop_values,
                marker_color=bar_colors,
                text=[f"{v:.0f}" for v in rop_values], textposition="outside"))
            fig_lt.update_layout(
                paper_bgcolor="#131C2C", plot_bgcolor="#131C2C", font={"color": "#EDF2F7"},
                height=300, margin=dict(l=20, r=20, t=20, b=20),
                xaxis={"title": "Supplier Lead Time", "color": "#FFFFFF"},
                yaxis={"gridcolor": "#1E2A3E", "title": "Reorder Point (units)"},
                showlegend=False)
            st.plotly_chart(fig_lt, use_container_width=True, config={"displaylogo": False})
            st.caption(f"Red bar = current sim lead time ({sim_lead_time}d). Every extra day raises reorder point.")

# ══════════════════════════════════════════════════════
# TAB 4 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown("### Portfolio Overview — All Categories")
    st.markdown(
        "<p style='color:#00E5FF;font-size:14px;'>Forecast for every category in one view. "
        "ABC classification: A = top 20% demand (tightest control), B = next 30%, C = bottom 50%.</p>",
        unsafe_allow_html=True)

    if st.button("🔄 Refresh Portfolio", key="btn_portfolio"):
        st.cache_data.clear()

    with st.spinner(f"Fetching forecasts for {len(known_categories)} categories..."):
        all_results = fetch_all_forecasts(known_categories, "auto")

    rows = []
    for cat, res in all_results.items():
        if "error" in res:
            rows.append({"Category": cat, "Model": "ERROR", "Avg Daily (Forecast)": None,
                         "7-Day Total": None, "Confidence": "—", "ABC": "—"})
            continue
        rows.append({
            "Category":             cat,
            "Model":                res["model_used"].upper(),
            "Avg Daily (Forecast)": round(float(np.mean(res["forecast"])), 3),
            "7-Day Total":          round(float(np.sum(res["forecast"])), 1),
            "Historical Avg/Day":   CATEGORY_STATS.get(cat, {}).get("avg_daily_demand", None),
            "Confidence":           res.get("confidence", "—"),
        })

    port_df = pd.DataFrame(rows).sort_values("7-Day Total", ascending=False, na_position="last").reset_index(drop=True)

    # ABC classification by rank
    valid_rows = port_df.dropna(subset=["7-Day Total"])
    total_cats = len(valid_rows)
    abc_map = {}
    for i, (_, row) in enumerate(valid_rows.iterrows()):
        abc_map[row["Category"]] = abc_classify_from_rank(i + 1, total_cats)
    port_df["ABC"] = port_df["Category"].map(abc_map).fillna("—")

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Categories Tracked", len(port_df))
    c2.metric("GBM-Routed",  int((port_df["Model"] == "GBM").sum()))
    c3.metric("LSTM-Routed", int((port_df["Model"] == "LSTM").sum()))
    total_7d_all = port_df["7-Day Total"].sum()
    c4.metric("Total 7-Day Demand", f"{total_7d_all:.0f} units")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ABC explanation ──────────────────────────
    st.markdown(f"""
    <div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:10px;padding:14px 18px;margin-bottom:16px;'>
        <span style='color:#00E5FF;font-family:JetBrains Mono,monospace;font-size:11px;font-weight:700;'>ABC / PARETO CLASSIFICATION</span><br>
        <span style='color:#EDF2F7;font-size:13px;'>
            <span class='abc-a'>A</span> Top 20% demand share — tightest safety stock policy, most frequent review.<br>
            <span class='abc-b'>B</span> Next 30% — moderate control, review weekly.<br>
            <span class='abc-c'>C</span> Bottom 50% — loose control, review monthly. Lower holding cost priority.
        </span>
    </div>""", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────
    cc1, cc2 = st.columns([1.6, 1])
    with cc1:
        st.markdown("**Demand by Category (colored by model)**")
        fig_bar = px.bar(
            port_df.dropna(subset=["7-Day Total"]),
            x="Category", y="7-Day Total", color="Model",
            color_discrete_map={"GBM": PALETTE["gbm"], "LSTM": PALETTE["lstm"], "ERROR": PALETTE["alert"]})
        fig_bar.update_layout(
            paper_bgcolor="#131C2C", plot_bgcolor="#131C2C", font={"color": "#EDF2F7"},
            height=360, margin=dict(l=20, r=20, t=20, b=80),
            xaxis={"color": "#FFFFFF", "tickangle": -35},
            yaxis={"gridcolor": "#1E2A3E", "title": "Units (7-day)"},
            legend={"font": {"color": "#FFFFFF"}, "bgcolor": "rgba(0,0,0,0)", "title": None})
        st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False})

    with cc2:
        st.markdown("**Model Routing Split**")
        route_counts = port_df[port_df["Model"] != "ERROR"]["Model"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=route_counts.index,
            values=route_counts.values,
            hole=0.60,
            marker_colors=[PALETTE["gbm"], PALETTE["lstm"]],
            textfont={"color": "#EDF2F7", "size": 13},
            showlegend=True))
        # Annotation in donut hole
        fig_pie.add_annotation(
            text=f"{len(port_df)}<br><span style='font-size:10px'>cats</span>",
            x=0.5, y=0.5, showarrow=False,
            font={"color": "#EDF2F7", "size": 18, "family": "JetBrains Mono"})
        fig_pie.update_layout(
            paper_bgcolor="#131C2C", font={"color": "#EDF2F7"}, height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            legend={"font": {"color": "#FFFFFF"}, "bgcolor": "rgba(0,0,0,0)"})
        st.plotly_chart(fig_pie, use_container_width=True, config={"displaylogo": False})

    # ── Full table with ABC and scaling check ─────
    st.markdown("**Full Portfolio Table**")

    for _, row in port_df.iterrows():
        abc = row.get("ABC", "—")
        hist = row.get("Historical Avg/Day")
        fcast = row.get("Avg Daily (Forecast)")
        scaling_flag = ""
        if hist and fcast and fcast > 0:
            ratio = fcast / hist
            if not (0.3 < ratio < 3.0):
                scaling_flag = " ⚠️"

        abc_html = f'<span class="abc-{abc.lower()}">{abc}</span>' if abc in ("A","B","C") else abc
        model_color = PALETTE["gbm"] if row["Model"] == "GBM" else (PALETTE["lstm"] if row["Model"] == "LSTM" else PALETTE["alert"])
        conf_val = row.get("Confidence", "—")
        conf_color = risk_color(str(conf_val).lower()) or "#FFFFFF"

        st.markdown(f"""
        <div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:10px;
                    padding:12px 16px;margin-bottom:6px;display:flex;align-items:center;gap:16px;'>
            <div style='min-width:160px;color:#EDF2F7;font-size:13px;font-weight:600;'>{row['Category']}{scaling_flag}</div>
            <div style='min-width:60px;'>{abc_html}</div>
            <div style='min-width:60px;color:{model_color};font-family:JetBrains Mono,monospace;font-size:12px;font-weight:700;'>{row['Model']}</div>
            <div style='min-width:100px;color:#FFA94D;font-family:JetBrains Mono,monospace;font-size:12px;'>
                {f"{row['7-Day Total']:.1f} units" if row['7-Day Total'] else "—"}</div>
            <div style='color:{conf_color};font-family:JetBrains Mono,monospace;font-size:11px;'>{conf_val.upper() if isinstance(conf_val, str) else conf_val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "📥 Download Portfolio Snapshot (CSV)",
        data=port_df.to_csv(index=False),
        file_name="supply_chain_portfolio_snapshot.csv",
        mime="text/csv")

# ══════════════════════════════════════════════════════
# TAB 5 — MODEL TRUST CENTER
# ══════════════════════════════════════════════════════
with tab5:
    st.markdown("### Model Trust Center")
    st.markdown(
        "<p style='color:#00E5FF;font-size:14px;'>Why trust this forecast? "
        "Evaluation metrics, live model comparison, SHAP explainability — all in one place.</p>",
        unsafe_allow_html=True)

    st.markdown("#### 🏆 Model Leaderboard (held-out evaluation)")
    st.caption("Baseline (Seasonal Naive/Prophet) is excluded — API returns 501 for it, "
               "so including it would compare against a stub, not a real model.")

    hdr = st.columns([2, 1, 3])
    for col, h in zip(hdr, ["MODEL", "AVG MAPE", "NOTES"]):
        col.markdown(f"<div style='color:#00E5FF;font-family:JetBrains Mono,monospace;font-size:11px;"
                     f"font-weight:600;border-bottom:2px solid #2D3A5E;padding:8px 0;'>{h}</div>", unsafe_allow_html=True)

    for name, mape, note in LEADERBOARD:
        is_best    = mape == min(m for _, m, _ in LEADERBOARD)
        bg         = "#0D1E35" if is_best else "#0F1525"
        name_color = PALETTE["gbm"] if "GBM" in name else PALETTE["lstm"]
        cols = st.columns([2, 1, 3])
        cols[0].markdown(f"<div style='background:{bg};padding:10px 8px;border-radius:4px;color:{name_color};font-size:13px;font-weight:600;'>{'✅ ' if is_best else ''}{name}</div>", unsafe_allow_html=True)
        cols[1].markdown(f"<div style='background:{bg};padding:10px 8px;border-radius:4px;color:#FFA94D;font-family:JetBrains Mono,monospace;font-weight:700;'>{mape}%</div>", unsafe_allow_html=True)
        cols[2].markdown(f"<div style='background:{bg};padding:10px 8px;border-radius:4px;color:#FFFFFF;font-size:13px;'>{note}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Live routing comparison (no baseline) ────
    st.markdown("#### 🔁 Live Model Comparison (GBM vs LSTM vs Auto)")
    st.caption("Baseline excluded — API not implemented.")
    check_cat = st.selectbox("Category to compare", known_categories, key="trust_cat")
    if st.button("Compare GBM vs LSTM vs Auto", key="btn_compare"):
        with st.spinner("Calling /forecast for gbm, lstm, auto..."):
            comparisons = {}
            for m in ["gbm", "lstm", "auto"]:
                try:
                    comparisons[m] = fetch_forecast(check_cat, m)
                except Exception as e:
                    comparisons[m] = {"error": str(e)}

        if comparisons:
            fig_cmp = go.Figure()
            cmp_colors = {"gbm": PALETTE["gbm"], "lstm": PALETTE["lstm"], "auto": PALETTE["auto"]}
            for m, res in comparisons.items():
                if "error" in res:
                    st.warning(f"`{m}` failed: {res['error']}")
                    continue
                fig_cmp.add_trace(go.Scatter(
                    x=[f"Day {i+1}" for i in range(len(res["forecast"]))],
                    y=res["forecast"], mode="lines+markers",
                    name=f"{m.upper()} (→{res['model_used'].upper()})",
                    line=dict(color=cmp_colors.get(m, "#FFFFFF"), width=2.5),
                    marker=dict(size=8)))
            fig_cmp.update_layout(
                paper_bgcolor="#131C2C", plot_bgcolor="#131C2C", font={"color": "#EDF2F7"},
                height=380, margin=dict(l=20, r=20, t=20, b=20),
                xaxis={"color": "#FFFFFF"}, yaxis={"gridcolor": "#1E2A3E", "title": "Forecast (units)"},
                legend={"font": {"color": "#FFFFFF"}, "bgcolor": "rgba(0,0,0,0)"})
            st.plotly_chart(fig_cmp, use_container_width=True, config={"displaylogo": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP — directional only ───────────────────
    st.markdown(f"#### 🔬 Feature Importance — Direction Only (SHAP) {source_tag('API')}", unsafe_allow_html=True)
    st.caption(
        "SHAP values from your trained GBM models via /explain. "
        "Only GBM-routed categories have SHAP support. "
        "**Bars show direction only** — red = pushes demand up, green = pushes demand down. "
        "Exact magnitudes are in scaled model-output space and are not shown to avoid misinterpretation."
    )
    explain_cat = st.selectbox("Category to explain", known_categories, key="explain_cat")
    if st.button("Explain Day-1 Forecast", key="btn_explain"):
        try:
            exp = fetch_explain(explain_cat)
        except RuntimeError as e:
            st.warning(f"Not available for this category: {e}")
            exp = None
        except requests.exceptions.RequestException as e:
            show_toast("Could not reach the API for SHAP", "error")
            exp = None

        if exp:
            st.caption(exp.get("note", ""))
            feats  = exp["top_features"]
            names  = [f["feature"] for f in feats][::-1]
            vals   = [f["shap_value"] for f in feats][::-1]

            # Direction only: normalize to [-1, 1] range for display, drop raw magnitude
            max_abs = max(abs(v) for v in vals) if vals else 1.0
            norm_vals   = [v / max_abs for v in vals]
            bar_colors  = [PALETTE["alert"] if v > 0 else PALETTE["success"] for v in norm_vals]
            bar_labels  = ["▲ demand up" if v > 0 else "▼ demand down" for v in norm_vals]

            fig_shap = go.Figure(go.Bar(
                x=norm_vals, y=names, orientation="h",
                marker_color=bar_colors,
                text=bar_labels,
                textposition="outside",
                hovertemplate="%{y}: %{text}<extra></extra>"))
            fig_shap.update_layout(
                paper_bgcolor="#131C2C", plot_bgcolor="#131C2C", font={"color": "#EDF2F7"},
                height=340, margin=dict(l=20, r=20, t=20, b=20),
                xaxis={"gridcolor": "#1E2A3E", "title": "Direction (normalized)", "range": [-1.4, 1.4]},
                yaxis={"color": "#FFFFFF"},
                showlegend=False)
            st.plotly_chart(fig_shap, use_container_width=True, config={"displaylogo": False})
            st.caption("🔴 feature pushes Day-1 demand **up** &nbsp;&nbsp; 🟢 pushes Day-1 demand **down**. "
                       "Magnitude not shown — direction is reliable, raw SHAP scale is not interpretable in original units.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚠️ Known Gaps")
    for gap in [
        "LSTM-routed forecasts (telephony) sometimes return null confidence intervals — documented gap, not yet fixed.",
        "LSTM tied with seasonal-naive baseline on MAPE (~20.4%) — in production for telephony by routing default, not because it clearly beats baseline.",
        "Telephony avg_daily_demand scaling bug: API likely returning log-space values without expm1(). Historical avg = 7.59 units/day; if forecast shows < 1 unit/day, this is the cause.",
        "Safety stock falls back to 25%-of-demand heuristic when 95% interval is missing.",
        "Baseline model returns HTTP 501 from the API — excluded from all comparisons in this dashboard.",
    ]:
        st.markdown(f"<p style='color:#FFFFFF;font-size:13px;margin:6px 0;'>• {gap}</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 6 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════
with tab6:
    st.markdown("### Executive Summary")
    st.markdown(
        "<p style='color:#00E5FF;font-size:14px;'>3-minute business briefing. "
        "No technical jargon — just the numbers that matter to a supply chain manager.</p>",
        unsafe_allow_html=True)

    # Pull portfolio data (cached)
    with st.spinner("Loading portfolio data..."):
        exec_results = fetch_all_forecasts(known_categories, "auto")

    total_categories = len(known_categories)
    total_7d_demand  = sum(
        float(np.sum(r["forecast"])) for r in exec_results.values() if "error" not in r
    )
    avg_daily_total  = total_7d_demand / 7.0

    # ₹ estimates — using sidebar unit cost, standard assumptions
    total_inventory_value = avg_daily_total * sb_lead_time * sb_unit_cost   # stock needed for lead time
    annual_demand_units   = avg_daily_total * 365
    # Use 18.56% vs 19.17% MAPE improvement logic, scaled to total demand
    mape_improvement  = 19.17 - 18.56
    units_better      = annual_demand_units * (mape_improvement / 100.0)
    est_annual_savings = units_better * sb_unit_cost * (0.20 + 2.0 * 0.5)  # holding + stockout savings
    stockout_risk_reduced = (19.17 - 18.56)  # percentage points

    gbm_routed = sum(1 for r in exec_results.values() if "error" not in r and r.get("model_used","").upper() == "GBM")
    lstm_routed = total_categories - gbm_routed

    # ── Top KPI cards ────────────────────────────
    e1, e2, e3, e4 = st.columns(4)

    e1.markdown(f"""
    <div style='background:#131C2C;border:2px solid {PALETTE["gbm"]};border-radius:16px;padding:22px;text-align:center;'>
        <div style='font-size:24px;margin-bottom:8px;'>📦</div>
        <div style='color:#FFFFFF;font-size:11px;text-transform:uppercase;letter-spacing:1px;'>Categories Automated</div>
        <div style='color:{PALETTE["gbm"]};font-size:36px;font-weight:800;font-family:JetBrains Mono,monospace;margin:6px 0;'>{total_categories}</div>
        <div style='color:#FFFFFF;font-size:11px;'>{gbm_routed} GBM · {lstm_routed} LSTM</div>
    </div>""", unsafe_allow_html=True)

    e2.markdown(f"""
    <div style='background:#131C2C;border:2px solid {PALETTE["success"]};border-radius:16px;padding:22px;text-align:center;'>
        <div style='font-size:24px;margin-bottom:8px;'>💰</div>
        <div style='color:#FFFFFF;font-size:11px;text-transform:uppercase;letter-spacing:1px;'>Est. Annual Savings</div>
        <div style='color:{PALETTE["success"]};font-size:30px;font-weight:800;font-family:JetBrains Mono,monospace;margin:6px 0;'>₹{est_annual_savings:,.0f}</div>
        <div style='color:#FFFFFF;font-size:11px;'>vs naive baseline ordering</div>
    </div>""", unsafe_allow_html=True)

    e3.markdown(f"""
    <div style='background:#131C2C;border:2px solid {PALETTE["lstm"]};border-radius:16px;padding:22px;text-align:center;'>
        <div style='font-size:24px;margin-bottom:8px;'>📊</div>
        <div style='color:#FFFFFF;font-size:11px;text-transform:uppercase;letter-spacing:1px;'>Inventory Value Covered</div>
        <div style='color:{PALETTE["lstm"]};font-size:30px;font-weight:800;font-family:JetBrains Mono,monospace;margin:6px 0;'>₹{total_inventory_value:,.0f}</div>
        <div style='color:#FFFFFF;font-size:11px;'>{sb_lead_time}d lead time stock at ₹{sb_unit_cost:.0f}/unit</div>
    </div>""", unsafe_allow_html=True)

    e4.markdown(f"""
    <div style='background:#131C2C;border:2px solid {PALETTE["auto"]};border-radius:16px;padding:22px;text-align:center;'>
        <div style='font-size:24px;margin-bottom:8px;'>🎯</div>
        <div style='color:#FFFFFF;font-size:11px;text-transform:uppercase;letter-spacing:1px;'>Forecast Accuracy</div>
        <div style='color:{PALETTE["auto"]};font-size:36px;font-weight:800;font-family:JetBrains Mono,monospace;margin:6px 0;'>18.56%</div>
        <div style='color:#FFFFFF;font-size:11px;'>avg MAPE (GBM, held-out)</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Business narrative cards ──────────────────
    n1, n2 = st.columns(2)
    with n1:
        st.markdown(f"""
        <div style='background:#0A1E0A;border:1px solid {PALETTE["success"]};border-radius:14px;padding:20px;margin-bottom:16px;'>
            <div style='color:{PALETTE["success"]};font-size:13px;font-weight:700;margin-bottom:10px;'>✅ WHAT THIS SYSTEM DOES</div>
            <div style='color:#EDF2F7;font-size:13px;line-height:1.8;'>
                • Automatically forecasts demand for <b>{total_categories} product categories</b> 7 days ahead<br>
                • Tells planners exactly <b>when to reorder</b> and <b>how much</b>, removing guesswork<br>
                • Quantifies forecast uncertainty with calibrated 80%/95% intervals<br>
                • Identifies which categories need the tightest inventory control (ABC analysis)<br>
                • Translates model accuracy into ₹ business impact a manager can act on
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:14px;padding:20px;'>
            <div style='color:#FFA94D;font-size:13px;font-weight:700;margin-bottom:10px;'>📐 HOW SAVINGS ARE CALCULATED</div>
            <div style='color:#EDF2F7;font-size:13px;line-height:1.8;'>
                GBM MAPE 18.56% vs baseline 19.17% = <b>{mape_improvement:.2f}% improvement</b>.<br>
                Each 1% MAPE improvement means fewer wrongly-ordered units per year.<br>
                Wrongly-ordered units cause: excess holding cost (₹{sb_unit_cost:.0f} × {sb_holding_rate*100:.0f}%/yr) 
                or stockouts (₹{sb_unit_cost*2:.0f}/unit lost sale + goodwill).<br>
                Combined across all categories at {avg_daily_total:.1f} units/day average = 
                <b style='color:{PALETTE["success"]};'>₹{est_annual_savings:,.0f}/year</b>.
            </div>
            <div style='color:#FFFFFF;font-size:10px;margin-top:8px;'>
                Assumption: stockout cost = 2× unit cost. Adjust unit cost in sidebar to recalculate.
            </div>
        </div>""", unsafe_allow_html=True)

    with n2:
        st.markdown(f"""
        <div style='background:#1A0E00;border:1px solid {PALETTE["lstm"]};border-radius:14px;padding:20px;margin-bottom:16px;'>
            <div style='color:{PALETTE["lstm"]};font-size:13px;font-weight:700;margin-bottom:10px;'>⚠️ KNOWN LIMITATIONS</div>
            <div style='color:#EDF2F7;font-size:13px;line-height:1.8;'>
                • LSTM and baseline perform similarly on MAPE (~20%) — <b>directional accuracy</b> is the LSTM's strength, not raw MAPE<br>
                • Category-level only — real ordering decisions happen at SKU level<br>
                • Lead time assumed fixed — variable supplier delay not yet modelled<br>
                • Telephony forecast may have a scaling bug — verify before using those numbers in production
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:14px;padding:20px;'>
            <div style='color:{PALETTE["gbm"]};font-size:13px;font-weight:700;margin-bottom:10px;'>🔑 KEY INTERVIEW NUMBERS</div>
            <div style='color:#EDF2F7;font-size:13px;line-height:1.8;'>
                <span style='color:#FFFFFF;'>Best model MAPE:</span> <b>18.56%</b> (GBM)<br>
                <span style='color:#FFFFFF;'>vs baseline:</span> <b>19.17%</b> → {mape_improvement:.2f}% improvement<br>
                <span style='color:#FFFFFF;'>Interval coverage:</span> <b>94.9%</b> at 95% target<br>
                <span style='color:#FFFFFF;'>Categories automated:</span> <b>{total_categories}</b><br>
                <span style='color:#FFFFFF;'>Estimated annual savings:</span> <b style='color:{PALETTE["success"]};'>₹{est_annual_savings:,.0f}</b><br>
                <span style='color:#FFFFFF;'>LSTM directional lift:</span> <b>+14–16pp</b> over random guess
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#FFFFFF;font-size:11px;font-family:JetBrains Mono,monospace;'>All ₹ figures recalculate dynamically based on sidebar unit cost (₹{sb_unit_cost:.0f}) and holding rate ({sb_holding_rate*100:.0f}%). Change them in the sidebar and this page updates.</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 7 — SYSTEM OVERVIEW
# ══════════════════════════════════════════════════════
with tab7:
    st.markdown("### System Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:16px;padding:24px;margin-bottom:20px;'>
            <div style='font-family:JetBrains Mono,monospace;color:#00E5FF;font-size:11px;font-weight:600;margin-bottom:14px;'>🎯 PROJECT OVERVIEW</div>
            <p style='color:#FFFFFF;font-size:14px;line-height:1.7;'>
                Demand forecasting platform on the Olist Brazilian e-commerce dataset 
                (~6,200 category-day rows, 10 product categories, MySQL-backed). 
                LightGBM and LSTM evaluated against seasonal-naive/Prophet baselines 
                using leakage-free walk-forward validation. GBM won 9/10 categories 
                (18.56% avg MAPE) and is auto-routed; LSTM serves telephony. 
                Conformal calibration delivers 80%/95% interval coverage of ~80.0%/94.9%. 
                Dashboard turns forecast + interval into reorder point, safety stock, 
                and EOQ a planner can act on — with ₹ business impact translation.
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:16px;padding:24px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:JetBrains Mono,monospace;color:#00E5FF;font-size:11px;font-weight:600;margin-bottom:14px;'>⚙️ CAPABILITIES</div>", unsafe_allow_html=True)
        for cap in [
            "📈 7-day demand forecast with calibrated 80%/95% intervals",
            "🤖 Automatic GBM/LSTM model routing per category",
            "📦 Reorder point, safety stock, EOQ from forecast output",
            "💰 ₹ business impact — savings vs naive baseline ordering",
            "⚠️ Stockout risk quantification with revenue-at-risk estimate",
            "🎉 Festive/demand spike stress test simulator",
            "🔖 ABC/Pareto classification of categories by demand share",
            "🔄 What-if simulation across lead time and service level",
            "🌐 Portfolio-wide view across all tracked categories",
            "🧠 SHAP feature direction (GBM only) — direction reliable, magnitude not",
        ]:
            st.markdown(f"<p style='color:#FFFFFF;font-size:13px;margin:7px 0;'>• {cap}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Evaluation metrics as visual cards, not a plain table
        st.markdown("<div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:16px;padding:24px;margin-bottom:20px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:JetBrains Mono,monospace;color:#00E5FF;font-size:11px;font-weight:600;margin-bottom:14px;'>📊 EVALUATION METRICS</div>", unsafe_allow_html=True)

        metric_items = [
            ("Best model MAPE (GBM)",       "18.56%",  PALETTE["gbm"],     "held-out walk-forward CV"),
            ("Baseline MAPE",               "19.17%",  "#FFFFFF",           "seasonal naive / Prophet"),
            ("LSTM MAPE",                   "20.44%",  PALETTE["lstm"],     "telephony only"),
            ("80% interval coverage",       "80.0%",   PALETTE["success"],  "calibrated, target 80%"),
            ("95% interval coverage",       "94.9%",   PALETTE["success"],  "calibrated, target 95%"),
            ("GBM Day-1 MAPE",              "6.9%",    PALETTE["gbm"],      "best at short horizon"),
            ("GBM Day-7 MAPE",              "30.1%",   PALETTE["alert"],    "degrades at longer horizon"),
            ("Categories tracked",          str(len(known_categories)) if known_categories else "10", PALETTE["auto"], "auto-routed"),
        ]

        for label, value, color, note in metric_items:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:10px 0;border-bottom:1px solid #1E2A3E;'>
                <div>
                    <span style='color:#FFFFFF;font-size:13px;'>{label}</span>
                    <div style='color:#FFFFFF;font-size:10px;'>{note}</div>
                </div>
                <span style='color:{color};font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;'>{value}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:16px;padding:24px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:JetBrains Mono,monospace;color:#00E5FF;font-size:11px;font-weight:600;margin-bottom:14px;'>🛠️ TECH STACK</div>", unsafe_allow_html=True)
        for tech, detail in [
            ("ML Models",       "LightGBM (7 horizon-specific boosters), LSTM (anchor-delta head)"),
            ("Uncertainty",     "Conformal prediction intervals (80% + 95%)"),
            ("Explainability",  "SHAP — directional feature importance per category"),
            ("Serving",         "FastAPI — /forecast, /explain, /categories"),
            ("Dashboard",       "Streamlit + Plotly — decision layer on top of API"),
            ("Data",            "MySQL — Olist Brazilian e-commerce dataset"),
            ("Tracking",        "MLflow — experiment tracking + model registry"),
        ]:
            st.markdown(f"""
            <div style='padding:8px 0;border-bottom:1px solid #1E2A3E;'>
                <span style='color:#EDF2F7;font-size:13px;font-weight:600;'>{tech}</span><br>
                <span style='color:#FFFFFF;font-size:12px;'>{detail}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style='text-align:center;color:#FFFFFF;font-size:11px;font-family:JetBrains Mono,monospace;padding:20px;'>
        SUPPLY CHAIN INTELLIGENCE PLATFORM &nbsp;·&nbsp; LightGBM + LSTM + FastAPI + Streamlit &nbsp;·&nbsp;
        Unit cost ₹{sb_unit_cost:.0f} · Holding {sb_holding_rate*100:.0f}%/yr · Lead time {sb_lead_time}d
    </div>""", unsafe_allow_html=True)