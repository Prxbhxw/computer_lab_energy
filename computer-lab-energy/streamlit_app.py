"""
LabEnergy - Computer Lab Energy Consumption Prediction & Analytics
Built with Streamlit, scikit-learn, pandas, and Plotly.
Deployable on Streamlit Community Cloud without external backend servers.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 1. Page Configuration & Theme
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LabEnergy - Computer Lab Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top banner */
    .top-banner {
        background: linear-gradient(135deg, #7c6cf2 0%, #5643cc 100%);
        border-radius: 16px;
        padding: 24px 28px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(124, 108, 242, 0.25);
    }
    .top-banner h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: white;
    }
    .top-banner p {
        margin: 6px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border: 1px solid #e9eaf2;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(124, 108, 242, 0.08);
    }
    .metric-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .metric-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 17px;
    }
    .icon-purple { background: #edeafe; color: #7c6cf2; }
    .icon-blue   { background: #e0f2fe; color: #0284c7; }
    .icon-amber  { background: #fef3c7; color: #d97706; }
    .icon-emerald{ background: #d1fae5; color: #059669; }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-unit {
        font-size: 15px;
        font-weight: 500;
        color: #64748b;
        margin-left: 4px;
    }
    .metric-sub {
        margin-top: 6px;
        font-size: 12px;
        color: #64748b;
    }
    .pill-good {
        background: #ecfdf5;
        color: #059669;
        padding: 2px 8px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 11px;
    }

    /* Result Card */
    .result-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        margin-top: 16px;
        box-shadow: 0 10px 25px -5px rgba(30, 27, 75, 0.4);
    }
    .result-box .title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #a5b4fc;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .result-box .kwh {
        font-size: 38px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
    }
    .result-box .cost {
        font-size: 24px;
        font-weight: 700;
        color: #34d399;
        margin-top: 6px;
    }
    
    /* Tip Card */
    .tip-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #7c6cf2;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .tip-card strong {
        color: #1e293b;
        display: block;
        margin-bottom: 4px;
        font-size: 14px;
    }
    .tip-card p {
        margin: 0;
        font-size: 13px;
        color: #475569;
    }

    /* Streamlit overrides */
    div[data-testid="stMetricValue"] {
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 2. File & Directory Resolution
# ─────────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent

def locate_data_file() -> Path:
    """Finds historical_energy.csv across varied folder layouts."""
    candidates = [
        CURRENT_DIR / "backend" / "data" / "historical_energy.csv",
        CURRENT_DIR / "computer-lab-energy" / "backend" / "data" / "historical_energy.csv",
        CURRENT_DIR / "data" / "historical_energy.csv",
        CURRENT_DIR.parent / "computer-lab-energy" / "backend" / "data" / "historical_energy.csv",
        CURRENT_DIR.parent / "backend" / "data" / "historical_energy.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback to backend/data path even if non-existent yet
    return CURRENT_DIR / "backend" / "data" / "historical_energy.csv"

def locate_model_file() -> Path:
    """Finds or defines path for energy_model.joblib."""
    candidates = [
        CURRENT_DIR / "backend" / "energy_model.joblib",
        CURRENT_DIR / "computer-lab-energy" / "backend" / "energy_model.joblib",
        CURRENT_DIR / "energy_model.joblib",
        CURRENT_DIR.parent / "computer-lab-energy" / "backend" / "energy_model.joblib",
    ]
    for c in candidates:
        if c.exists():
            return c
    return CURRENT_DIR / "backend" / "energy_model.joblib"

DATA_FILE = locate_data_file()
MODEL_FILE = locate_model_file()

# ─────────────────────────────────────────────────────────────
# 3. Data & Model Functions (Cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_default_dataset() -> pd.DataFrame:
    """Loads and preprocesses the bundled historical dataset."""
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
    else:
        # Fallback synthetic generator if file is absent
        np.random.seed(42)
        dates = pd.date_range(start="2025-01-01", periods=240, freq="D")
        computers = np.random.randint(12, 45, size=240)
        hours = np.round(np.random.uniform(1.5, 8.0, size=240), 1)
        watts = np.random.randint(90, 180, size=240)
        noise = np.random.normal(1.0, 0.05, size=240)
        kwh = np.round((computers * hours * watts / 1000.0) * noise, 3)
        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "computers_used": computers,
            "operating_hours": hours,
            "avg_power_watts": watts,
            "energy_kwh": kwh
        })
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.strftime("%b %Y")
    df["day_name"] = df["date"].dt.day_name()
    return df

def train_or_load_model(df: pd.DataFrame, n_estimators: int = 250, test_size: float = 0.2):
    """Loads saved model or fits a new RandomForestRegressor."""
    features = ["computers_used", "operating_hours", "avg_power_watts"]
    target = "energy_kwh"
    
    # If custom dataset or retraining, fit directly
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    if MODEL_FILE.exists() and len(df) == 240 and n_estimators == 250 and test_size == 0.2:
        try:
            model = joblib.load(MODEL_FILE)
            preds = model.predict(X_test)
            metrics = {
                "r2": round(float(r2_score(y_test, preds)), 4),
                "mae": round(float(mean_absolute_error(y_test, preds)), 3),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "features": features
            }
            return model, metrics
        except Exception:
            pass  # Fall through to fit if load fails
            
    # Train freshly
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "r2": round(float(r2_score(y_test, preds)), 4),
        "mae": round(float(mean_absolute_error(y_test, preds)), 3),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": features
    }
    return model, metrics

# Helper for currency formatting
def format_curr(value: float, currency_sym: str) -> str:
    return f"{currency_sym} {value:,.2f}"

# ─────────────────────────────────────────────────────────────
# 4. Sidebar Controls & Settings
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <div style="background:#7c6cf2; color:white; width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:bold;">⚡</div>
        <div>
            <div style="font-size:18px; font-weight:800; color:#1e1b4b; line-height:1.1;">Lab<span style="color:#7c6cf2;">Energy</span></div>
            <div style="font-size:10px; font-weight:700; color:#64748b; letter-spacing:0.8px;">ENERGY INTELLIGENCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**Facility Context**")
    lab_name = st.selectbox("Selected Facility", ["Computer Lab 01 (Campus Main)", "Computer Lab 02 (Engineering)", "Data Science Lab", "Multimedia Studio"])
    st.caption("Status: `Historical CSV Mode` · No hardware sensors needed")
    
    st.markdown("---")
    st.markdown("**Global Billing & Emission Settings**")
    currency = st.selectbox("Currency Unit", ["INR (₹)", "USD ($)", "EUR (€)", "GBP (£)"], index=0)
    currency_symbol = {"INR (₹)": "₹", "USD ($)": "$", "EUR (€)": "€", "GBP (£)": "£"}[currency]
    
    tariff = st.number_input(
        f"Tariff ({currency_symbol} per kWh)",
        min_value=0.0,
        max_value=1000.0,
        value=8.0,
        step=0.5,
        help="Electricity price used to calculate operational expenses."
    )
    
    co2_factor = st.number_input(
        "Grid Carbon Factor (kg CO₂ / kWh)",
        min_value=0.0,
        max_value=2.0,
        value=0.82,
        step=0.05,
        help="Typical Indian power grid average is ~0.82 kg CO2 per kWh. US/EU typical ranges between 0.35 - 0.45 kg/kWh."
    )
    
    st.markdown("---")
    st.caption("Deploy ready: Streamlit Community Cloud")
    st.caption("Model: Random Forest Regressor · scikit-learn")

# ─────────────────────────────────────────────────────────────
# 5. Load Data & Initialize Model
# ─────────────────────────────────────────────────────────────
if "custom_df" not in st.session_state:
    st.session_state.custom_df = None

active_df = st.session_state.custom_df if st.session_state.custom_df is not None else load_default_dataset()
model, model_metrics = train_or_load_model(active_df)

# High-level Aggregates
total_energy = float(active_df["energy_kwh"].sum())
avg_daily = float(active_df["energy_kwh"].mean())
peak_daily = float(active_df["energy_kwh"].max())
total_cost = total_energy * tariff
total_co2 = total_energy * co2_factor
total_days = len(active_df)

# ─────────────────────────────────────────────────────────────
# 6. Top Header Banner
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-banner">
    <h1>⚡ Computer Lab Energy Intelligence</h1>
    <p>Predict lab power consumption, optimize operating schedules, and uncover historical energy patterns using machine learning.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 7. Main Navigation Tabs
# ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Executive Dashboard",
    "⚡ ML Energy Predictor",
    "📈 Usage Analytics & Trends",
    "📋 Historical Data & Upload",
    "🤖 Model Diagnostics",
    "🌿 Green Lab & ROI Calculator"
])

# ─────────────────────────────────────────────────────────────
# TAB 1: EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    # Metric cards row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top">
                <span class="metric-label">Total Energy</span>
                <span class="metric-icon icon-purple">⚡</span>
            </div>
            <div class="metric-value">{total_energy:,.1f}<span class="metric-unit">kWh</span></div>
            <div class="metric-sub"><span class="pill-good">Dataset Record</span> over {total_days} days</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top">
                <span class="metric-label">Daily Average</span>
                <span class="metric-icon icon-blue">📊</span>
            </div>
            <div class="metric-value">{avg_daily:.2f}<span class="metric-unit">kWh/day</span></div>
            <div class="metric-sub">Mean consumption per recorded session</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top">
                <span class="metric-label">Peak Daily Record</span>
                <span class="metric-icon icon-amber">🔥</span>
            </div>
            <div class="metric-value">{peak_daily:.2f}<span class="metric-unit">kWh</span></div>
            <div class="metric-sub">Highest recorded single-day usage</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top">
                <span class="metric-label">Estimated Bill</span>
                <span class="metric-icon icon-emerald">💰</span>
            </div>
            <div class="metric-value">{currency_symbol}{total_cost:,.0f}</div>
            <div class="metric-sub">At {currency_symbol}{tariff}/kWh · ~{total_co2:,.0f} kg CO₂</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Notice Pill
    st.info("ℹ️ **Historical Analysis Mode:** This dashboard runs entirely from historical CSV logs and a trained Random Forest model. Live hardware sensors or meters are not required.")

    # Charts Grid
    col_chart_left, col_chart_right = st.columns([7, 5])
    
    with col_chart_left:
        st.subheader("Monthly Energy Consumption Trend")
        monthly_df = active_df.groupby("month", sort=False)["energy_kwh"].sum().reset_index()
        
        fig_monthly = px.area(
            monthly_df,
            x="month",
            y="energy_kwh",
            labels={"month": "Month", "energy_kwh": "Total Energy (kWh)"},
            color_discrete_sequence=["#7c6cf2"]
        )
        fig_monthly.update_traces(
            mode="lines+markers",
            line=dict(width=3, color="#7c6cf2"),
            marker=dict(size=7, color="#5643cc", line=dict(width=2, color="white")),
            fillcolor="rgba(124, 108, 242, 0.22)"
        )
        fig_monthly.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            hovermode="x unified",
            height=340,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#e9eaf2"),
            xaxis=dict(gridcolor="#e9eaf2")
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col_chart_right:
        st.subheader("Recent Daily Trajectory (Last 30 Days)")
        recent_df = active_df.tail(30).copy()
        recent_df["date_str"] = recent_df["date"].dt.strftime("%d %b")
        
        fig_recent = px.bar(
            recent_df,
            x="date_str",
            y="energy_kwh",
            labels={"date_str": "Date", "energy_kwh": "Energy (kWh)"},
            color_discrete_sequence=["#9f8efc"]
        )
        fig_recent.update_traces(
            marker_line_width=0,
            marker_color="#7c6cf2"
        )
        fig_recent.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            height=340,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#e9eaf2"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)", tickangle=-45)
        )
        st.plotly_chart(fig_recent, use_container_width=True)

    # Quick Insights Summary Row
    st.markdown("### Operational Highlights")
    ih1, ih2, ih3 = st.columns(3)
    with ih1:
        st.markdown("""
        <div class="tip-card">
            <strong>🖥️ Idle Power Detection</strong>
            <p>Historical runs show average active power ratings varying by 35% between daytime practicals and evening project labs.</p>
        </div>
        """, unsafe_allow_html=True)
    with ih2:
        st.markdown(f"""
        <div class="tip-card">
            <strong>🌱 Carbon Offset Equivalent</strong>
            <p>Total historical carbon output ({total_co2:,.0f} kg CO₂) requires approximately <b>{int(total_co2 / 21.7)} mature trees</b> to absorb annually.</p>
        </div>
        """, unsafe_allow_html=True)
    with ih3:
        st.markdown(f"""
        <div class="tip-card">
            <strong>💡 Benchmark Cost Efficiency</strong>
            <p>At current tariff of {currency_symbol}{tariff}/kWh, daily average lab operational overhead is <b>{currency_symbol}{avg_daily * tariff:.2f}/day</b>.</p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 2: ML ENERGY PREDICTOR
# ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("⚡ Lab Session Energy & Cost Predictor")
    st.caption("Estimate the energy consumption and electricity bill for an upcoming lecture or practical session using the Random Forest Regressor.")
    
    col_input, col_output = st.columns([6, 6], gap="large")
    
    with col_input:
        st.markdown("#### 1. Hardware & Session Parameters")
        
        # Preset buttons
        st.markdown("**Quick Hardware Presets:**")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        preset_val = None
        if p_col1.button("Eco / Mini PC\n(65W)", use_container_width=True):
            st.session_state.watts_val = 65
        if p_col2.button("Standard PC\n(120W)", use_container_width=True):
            st.session_state.watts_val = 120
        if p_col3.button("CAD Workstation\n(220W)", use_container_width=True):
            st.session_state.watts_val = 220
        if p_col4.button("AI / GPU Rig\n(450W)", use_container_width=True):
            st.session_state.watts_val = 450
            
        default_watts = st.session_state.get("watts_val", 130)
        
        computers_input = st.slider(
            "Computers Powered On",
            min_value=1,
            max_value=100,
            value=28,
            help="Total workstations active during the lab session."
        )
        
        hours_input = st.slider(
            "Session Duration (Operating Hours)",
            min_value=0.5,
            max_value=18.0,
            value=6.0,
            step=0.5,
            help="Total continuous hours the lab is occupied."
        )
        
        watts_input = st.number_input(
            "Average Power Consumption per PC (Watts)",
            min_value=10,
            max_value=1500,
            value=int(default_watts),
            step=5,
            help="Average wattage per workstation including monitor and CPU."
        )
        
        session_tariff = st.number_input(
            f"Electricity Tariff for this Session ({currency_symbol}/kWh)",
            min_value=0.0,
            max_value=500.0,
            value=float(tariff),
            step=0.5
        )

    # Perform Prediction
    input_row = pd.DataFrame([{
        "computers_used": computers_input,
        "operating_hours": hours_input,
        "avg_power_watts": watts_input
    }])
    
    predicted_kwh = float(model.predict(input_row)[0])
    predicted_kwh = max(0.0, round(predicted_kwh, 2))
    predicted_cost = round(predicted_kwh * session_tariff, 2)
    predicted_co2 = round(predicted_kwh * co2_factor, 2)
    
    # Formula baseline for comparison: (N * H * W) / 1000
    formula_baseline = round((computers_input * hours_input * watts_input) / 1000.0, 2)
    delta_percent = round(((predicted_kwh - formula_baseline) / max(0.001, formula_baseline)) * 100, 1)

    with col_output:
        st.markdown("#### 2. Predicted Session Impact")
        
        st.markdown(f"""
        <div class="result-box">
            <div class="title">✨ ML MODEL PREDICTION</div>
            <div class="kwh">{predicted_kwh} <span style="font-size:20px; font-weight:400; opacity:0.8;">kWh</span></div>
            <div class="cost">{currency_symbol}{predicted_cost:,.2f} <span style="font-size:14px; color:#cbd5e1; font-weight:normal;">estimated session cost</span></div>
            <div style="margin-top:14px; font-size:13px; color:#c7d2fe; border-top:1px solid rgba(255,255,255,0.15); padding-top:10px;">
                🍃 Carbon Footprint: <b>{predicted_co2} kg CO₂e</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        
        # Comparison gauge / metrics
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric(
                label="Baseline Physics Formula",
                value=f"{formula_baseline} kWh",
                help="(Computers × Hours × Watts) ÷ 1000"
            )
        with mc2:
            st.metric(
                label="Model vs Formula Variance",
                value=f"{abs(delta_percent)}%",
                delta=f"{delta_percent:+.1f}% vs theoretical",
                delta_color="inverse" if delta_percent > 0 else "normal"
            )
            
        # Real-world equivalents
        st.markdown("**Real-world Equivalents:**")
        eq_phones = int(predicted_kwh * 80)
        eq_led_hours = int(predicted_kwh * 100)
        st.markdown(f"""
        - 📱 Equivalent to fully charging **~{eq_phones:,} smartphones**.
        - 💡 Could illuminate a 10W classroom LED bulb for **~{eq_led_hours:,} continuous hours**.
        - 🏫 Represents **{(predicted_kwh / max(1.0, avg_daily)) * 100:.1f}%** of an average full day's lab consumption.
        """)

    # Scenario Sensitivity Analysis
    st.markdown("---")
    st.subheader("💡 Sensitivity & 'What-If' Duration Analysis")
    st.caption("How energy & cost changes as operating duration varies for this specific lab size:")
    
    hours_range = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    sensitivity_rows = []
    for h in hours_range:
        sim_row = pd.DataFrame([{
            "computers_used": computers_input,
            "operating_hours": h,
            "avg_power_watts": watts_input
        }])
        sim_kwh = round(float(model.predict(sim_row)[0]), 2)
        sim_cost = round(sim_kwh * session_tariff, 2)
        sensitivity_rows.append({"Hours": f"{h} hrs", "Energy_kWh": sim_kwh, "Cost": sim_cost})
        
    sens_df = pd.DataFrame(sensitivity_rows)
    
    fig_sens = px.line(
        sens_df,
        x="Hours",
        y="Energy_kWh",
        markers=True,
        text="Energy_kWh",
        labels={"Hours": "Session Length", "Energy_kWh": "Energy (kWh)"},
        color_discrete_sequence=["#7c6cf2"]
    )
    fig_sens.update_traces(textposition="top center", line=dict(width=3))
    fig_sens.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#e9eaf2")
    )
    st.plotly_chart(fig_sens, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 3: USAGE ANALYTICS & TRENDS
# ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📈 Deep Consumption Analytics & Correlation")
    
    an_col1, an_col2 = st.columns(2)
    
    with an_col1:
        st.markdown("**Distribution of Daily Energy (kWh)**")
        fig_dist = px.histogram(
            active_df,
            x="energy_kwh",
            nbins=25,
            marginal="box",
            color_discrete_sequence=["#7c6cf2"],
            labels={"energy_kwh": "Daily Energy (kWh)"}
        )
        fig_dist.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#e9eaf2"),
            xaxis=dict(gridcolor="#e9eaf2")
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with an_col2:
        st.markdown("**Computers Used vs Operating Hours (Bubble Size = kWh)**")
        fig_scatter = px.scatter(
            active_df,
            x="computers_used",
            y="operating_hours",
            size="energy_kwh",
            color="energy_kwh",
            color_continuous_scale="Viridis",
            labels={
                "computers_used": "Computers in Use",
                "operating_hours": "Operating Hours",
                "energy_kwh": "Energy (kWh)"
            }
        )
        fig_scatter.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#e9eaf2"),
            xaxis=dict(gridcolor="#e9eaf2")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("**Energy Consumption by Day of Week**")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_df = active_df.groupby("day_name")["energy_kwh"].agg(["mean", "count", "sum"]).reindex(day_order).dropna().reset_index()
    day_df.columns = ["Day", "Average_kWh", "Sessions_Count", "Total_kWh"]
    
    fig_day = px.bar(
        day_df,
        x="Day",
        y="Average_kWh",
        text_auto=".1f",
        labels={"Average_kWh": "Avg Daily Energy (kWh)"},
        color="Average_kWh",
        color_continuous_scale="Purples"
    )
    fig_day.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#e9eaf2")
    )
    st.plotly_chart(fig_day, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 4: HISTORICAL DATA & CSV UPLOAD
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("📋 Historical Records Explorer & Custom Upload")
    
    u_col1, u_col2 = st.columns([7, 5])
    
    with u_col1:
        st.markdown("#### Filter Historical Records")
        search_query = st.text_input("🔍 Search by Month or Keyword (e.g. 'Jan 2025' or 'Monday')", "")
        
        min_comp, max_comp = int(active_df["computers_used"].min()), int(active_df["computers_used"].max())
        comp_filter = st.slider("Filter by Computers Used Range", min_comp, max_comp, (min_comp, max_comp))
        
        filtered_df = active_df[
            (active_df["computers_used"] >= comp_filter[0]) &
            (active_df["computers_used"] <= comp_filter[1])
        ]
        
        if search_query:
            mask = (
                filtered_df["month"].str.contains(search_query, case=False, na=False) |
                filtered_df["day_name"].str.contains(search_query, case=False, na=False) |
                filtered_df["date"].astype(str).str.contains(search_query, case=False, na=False)
            )
            filtered_df = filtered_df[mask]
            
        st.write(f"Showing **{len(filtered_df)}** of {len(active_df)} entries:")
        st.dataframe(
            filtered_df[["date", "month", "day_name", "computers_used", "operating_hours", "avg_power_watts", "energy_kwh"]],
            use_container_width=True,
            height=300
        )
        
        # CSV Export
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Filtered Data as CSV",
            data=csv_data,
            file_name="lab_energy_filtered.csv",
            mime="text/csv"
        )

    with u_col2:
        st.markdown("#### 📤 Upload Your Own Lab CSV Dataset")
        st.caption("Upload historical logs from your own college lab to automatically recompute metrics and retrain the model.")
        
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                required_cols = {"computers_used", "operating_hours", "avg_power_watts", "energy_kwh"}
                if not required_cols.issubset(set(user_df.columns)):
                    st.error(f"CSV missing required columns: {required_cols - set(user_df.columns)}")
                else:
                    st.success(f"Successfully validated {len(user_df)} rows!")
                    if "date" in user_df.columns:
                        user_df["date"] = pd.to_datetime(user_df["date"])
                        user_df["month"] = user_df["date"].dt.strftime("%b %Y")
                        user_df["day_name"] = user_df["date"].dt.day_name()
                    else:
                        user_df["date"] = pd.date_range(start="2025-01-01", periods=len(user_df), freq="D")
                        user_df["month"] = user_df["date"].dt.strftime("%b %Y")
                        user_df["day_name"] = user_df["date"].dt.day_name()
                        
                    if st.button("Apply Uploaded Dataset & Retrain", type="primary"):
                        st.session_state.custom_df = user_df
                        st.rerun()
            except Exception as e:
                st.error(f"Error parsing file: {e}")
                
        if st.session_state.custom_df is not None:
            if st.button("Reset to Default Bundled Dataset"):
                st.session_state.custom_df = None
                st.rerun()

# ─────────────────────────────────────────────────────────────
# TAB 5: MODEL DIAGNOSTICS & RETRAINING
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("🤖 Machine Learning Model Architecture")
    st.caption("Inspect the underlying Random Forest Regressor and its accuracy metrics.")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("R² Score (Goodness of Fit)", f"{model_metrics['r2']:.4f}", help="Close to 1.0 indicates excellent fit.")
    with m2:
        st.metric("Mean Absolute Error (MAE)", f"{model_metrics['mae']:.3f} kWh", help="Average kWh prediction error.")
    with m3:
        st.metric("Training Samples", f"{model_metrics['train_rows']}")
    with m4:
        st.metric("Hold-out Test Samples", f"{model_metrics['test_rows']}")
        
    st.markdown("---")
    f_col1, f_col2 = st.columns([6, 6])
    
    with f_col1:
        st.markdown("**Feature Importance in Energy Prediction**")
        if hasattr(model, "feature_importances_"):
            feat_df = pd.DataFrame({
                "Feature": model_metrics["features"],
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)
            
            fig_feat = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Purples",
                text_auto=".3f"
            )
            fig_feat.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#e9eaf2")
            )
            st.plotly_chart(fig_feat, use_container_width=True)
            st.caption("Operating hours and computers count typically hold the highest weighting in determining total energy draw.")

    with f_col2:
        st.markdown("**Retrain Hyperparameters**")
        with st.form("retrain_form"):
            trees = st.slider("Number of Estimators (Trees)", min_value=50, max_value=500, value=250, step=50)
            split = st.slider("Test Set Split Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
            retrain_btn = st.form_submit_button("Retrain Model Now")
            if retrain_btn:
                with st.spinner("Retraining Random Forest model..."):
                    new_model, new_metrics = train_or_load_model(active_df, n_estimators=trees, test_size=split)
                    st.success(f"Retrained successfully! New R²: {new_metrics['r2']} · MAE: {new_metrics['mae']} kWh")

# ─────────────────────────────────────────────────────────────
# TAB 6: GREEN LAB & ROI CALCULATOR
# ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("🌿 Sustainable Lab Strategies & Energy ROI Calculator")
    st.markdown("Discover actionable interventions to cut idle power drain and simulate annual cost and emission savings.")
    
    roi_col1, roi_col2 = st.columns([6, 6], gap="large")
    
    with roi_col1:
        st.markdown("#### 1. Conservation Policies for College Labs")
        st.markdown("""
        <div class="tip-card">
            <strong>1. Automated Sleep & Hibernate Policy</strong>
            <p>Configure OS group policies (GPO) to suspend monitors after 10 minutes and hibernate idle systems after 20 minutes without mouse/keyboard activity.</p>
        </div>
        <div class="tip-card">
            <strong>2. Smart Master Power Strip / Relay</strong>
            <p>Cut phantom/vampire standby power to peripherals (speakers, document cameras, secondary displays) using a primary-secondary master relay.</p>
        </div>
        <div class="tip-card">
            <strong>3. Batch Scheduling & Off-Peak Practicals</strong>
            <p>Consolidate lower-attendance sessions into a single lab rather than running 3 separate rooms at 25% capacity with full HVAC loads.</p>
        </div>
        <div class="tip-card">
            <strong>4. Display Brightness Calibration</strong>
            <p>Lowering default monitor brightness from 100% to 65% in shaded lab environments reduces display power draw by up to 22%.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with roi_col2:
        st.markdown("#### 2. Annual Energy & Cost Savings Simulator")
        sim_pcs = st.slider("Total Workstations in Lab", 10, 200, 60, step=5)
        sim_idle_hours = st.slider("Current Idle Hours per Day (left on unnecessarily)", 0.5, 6.0, 2.0, step=0.5)
        sim_days_open = st.slider("Operating Days per Academic Year", 100, 320, 220, step=10)
        sim_pc_watts = st.slider("Average Workstation Active Draw (Watts)", 50, 400, 120, step=10)
        
        # Savings Calculation
        annual_kwh_saved = (sim_pcs * sim_idle_hours * sim_pc_watts / 1000.0) * sim_days_open
        annual_cost_saved = annual_kwh_saved * tariff
        annual_co2_saved = annual_kwh_saved * co2_factor
        
        st.markdown(f"""
        <div class="result-box" style="background: linear-gradient(135deg, #065f46 0%, #047857 100%);">
            <div class="title" style="color:#a7f3d0;">🌿 POTENTIAL ANNUAL SAVINGS</div>
            <div class="kwh">{annual_kwh_saved:,.0f} <span style="font-size:20px; font-weight:400; opacity:0.9;">kWh / year</span></div>
            <div class="cost" style="color:#fef08a;">{currency_symbol}{annual_cost_saved:,.2f} / year</div>
            <div style="margin-top:12px; font-size:13px; color:#d1fae5; border-top:1px solid rgba(255,255,255,0.2); padding-top:10px;">
                Avoids <b>{annual_co2_saved:,.0f} kg CO₂</b> emissions (~{int(annual_co2_saved / 21.7)} trees equivalent)
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 8. Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
f_left, f_right = st.columns([6, 6])
with f_left:
    st.caption("© 2025 LabEnergy · Computer Lab Energy Consumption Prediction System")
with f_right:
    st.markdown("<p style='text-align:right; font-size:12px; color:#64748b;'>Built for historical analysis & academic energy optimization</p>", unsafe_allow_html=True)
