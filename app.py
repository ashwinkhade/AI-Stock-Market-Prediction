"""
app.py
------
Streamlit dashboard for the AI Stock Market Prediction System.

Run with:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_collection import fetch_stock_data
from preprocessing   import preprocess, engineer_features, clean_data
from train_model     import train_all_models

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0D1117; }
    .stApp { background-color: #0D1117; }
    [data-testid="stSidebar"] { background-color: #161B22; }
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #58A6FF; }
    .metric-label { font-size: 0.85rem; color: #8B949E; margin-top: 4px; }
    h1, h2, h3 { color: #E6EDF3 !important; }
    .stSelectbox label, .stSlider label, .stTextInput label { color: #8B949E !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:10px 0 20px 0;">
    <h1 style="font-size:2.4rem; background: linear-gradient(90deg,#58A6FF,#3FB950);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        📈 AI Stock Market Prediction System
    </h1>
    <p style="color:#8B949E; font-size:1rem;">
        Machine Learning powered stock price forecasting with Linear Regression,
        Random Forest &amp; Decision Tree
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    ticker = st.text_input("Stock Ticker Symbol", value="AAPL",
                           placeholder="e.g. AAPL, TSLA, GOOGL").upper().strip()
    period = st.selectbox("Historical Period",
                          ["1y", "2y", "3y", "5y", "10y"], index=3)
    test_ratio = st.slider("Test Set Size (%)", 10, 40, 20) / 100

    run_btn = st.button("🚀 Run Prediction", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 📌 Popular Tickers")
    cols = st.columns(2)
    quick = ["AAPL","MSFT","GOOGL","TSLA","AMZN","NVDA","META","NFLX"]
    for i, t in enumerate(quick):
        if cols[i % 2].button(t, use_container_width=True):
            ticker = t
            run_btn = True

    st.markdown("---")
    st.markdown("""
    <div style='color:#8B949E; font-size:0.8rem;'>
    <b>Models</b><br>
    • Linear Regression<br>
    • Random Forest Regressor<br>
    • Decision Tree Regressor<br><br>
    <b>Metrics</b><br>
    MAE · MSE · RMSE · R²
    </div>
    """, unsafe_allow_html=True)


# ── Helper: run full pipeline ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(ticker, period, test_ratio):
    df      = fetch_stock_data(ticker, period)
    df_feat = engineer_features(df)
    df_feat = clean_data(df_feat)
    split   = int(len(df_feat) * (1 - test_ratio))
    train_df, test_df = df_feat.iloc[:split], df_feat.iloc[split:]

    FEAT = ["Open","High","Low","Volume","MA_10","MA_20","MA_50",
            "EMA_12","EMA_26","MACD","RSI","Volatility","Daily_Return",
            "Lag_1","Lag_2","Lag_3","Lag_4","Lag_5"]

    X_train = train_df[FEAT].values; y_train = train_df["Close"].values
    X_test  = test_df[FEAT].values;  y_test  = test_df["Close"].values

    results = train_all_models(X_train, X_test, y_train, y_test)
    return df_feat, train_df, test_df, y_test, results


# ── Main content ───────────────────────────────────────────────────────────────
if run_btn or "results" in st.session_state:
    if run_btn:
        st.session_state.pop("results", None)

    with st.spinner(f"Fetching data and training models for **{ticker}** …"):
        try:
            df_feat, train_df, test_df, y_test, results = run_pipeline(
                ticker, period, test_ratio)
            st.session_state["results"] = (df_feat, train_df, test_df, y_test, results)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    df_feat, train_df, test_df, y_test, results = st.session_state["results"]

    # ── KPI strip ─────────────────────────────────────────────────────────────
    best_r2 = max(results[n]["metrics"]["R2"] for n in results)
    best_model = max(results, key=lambda n: results[n]["metrics"]["R2"])
    current_price = df_feat["Close"].iloc[-1]
    price_change  = df_feat["Close"].pct_change().iloc[-1] * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Current Price",  f"${current_price:.2f}", "#58A6FF"),
        ("1-Day Change",   f"{price_change:+.2f}%",
         "#3FB950" if price_change >= 0 else "#F78166"),
        ("Best Model",     best_model,               "#BC8CFF"),
        ("Best R² Score",  f"{best_r2:.4f}",          "#E3B341"),
        ("Data Points",    f"{len(df_feat):,}",        "#58A6FF"),
    ]
    for col, (label, val, color) in zip([k1,k2,k3,k4,k5], kpis):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color};">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Stock Data", "📈 Predictions", "📉 Model Comparison",
         "🔍 EDA", "📋 Raw Data"])

    # ── TAB 1: Historical data ────────────────────────────────────────────────
    with tab1:
        st.subheader(f"📊 {ticker} Historical Stock Data")

        fig = make_subplots(rows=3, cols=1,
                            shared_xaxes=True,
                            row_heights=[0.55, 0.25, 0.20],
                            vertical_spacing=0.04)

        fig.add_trace(go.Candlestick(
            x=df_feat.index,
            open=df_feat["Open"], high=df_feat["High"],
            low=df_feat["Low"],   close=df_feat["Close"],
            name="OHLC",
            increasing_line_color="#3FB950",
            decreasing_line_color="#F78166"), row=1, col=1)

        for ma, color in [("MA_20","#E3B341"),("MA_50","#BC8CFF")]:
            fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat[ma],
                                     name=ma, line=dict(color=color, width=1.2, dash="dash")),
                          row=1, col=1)

        vol_colors = ["#3FB950" if c >= o else "#F78166"
                      for c, o in zip(df_feat["Close"], df_feat["Open"])]
        fig.add_trace(go.Bar(x=df_feat.index, y=df_feat["Volume"],
                             name="Volume", marker_color=vol_colors, opacity=0.7),
                      row=2, col=1)

        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["RSI"],
                                  name="RSI", line=dict(color="#58A6FF", width=1.2)),
                      row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#F78166", opacity=0.6, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#3FB950", opacity=0.6, row=3, col=1)

        fig.update_layout(
            height=700, template="plotly_dark",
            paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: Predictions ────────────────────────────────────────────────────
    with tab2:
        st.subheader("📈 Actual vs Predicted Closing Prices")

        model_choice = st.selectbox("Select Model", list(results.keys()))
        res = results[model_choice]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=test_df.index, y=y_test,
                                   name="Actual", line=dict(color="#8B949E", width=2)))
        fig2.add_trace(go.Scatter(x=test_df.index, y=res["y_pred"],
                                   name="Predicted",
                                   line=dict(color="#58A6FF", width=2.5)))
        fig2.add_trace(go.Scatter(
            x=list(test_df.index) + list(test_df.index[::-1]),
            y=list(res["y_pred"]) + list(y_test[::-1]),
            fill="toself", fillcolor="rgba(88,166,255,0.08)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False, name="Error region"))
        fig2.update_layout(
            height=450, template="plotly_dark",
            paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
            title=f"{model_choice}  —  Test Period Prediction",
            xaxis_title="Date", yaxis_title="Price (USD)",
            margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        m = res["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, val) in zip([c1,c2,c3,c4], [
            ("MAE",  f"{m['MAE']:.4f}"),
            ("MSE",  f"{m['MSE']:.4f}"),
            ("RMSE", f"{m['RMSE']:.4f}"),
            ("R²",   f"{m['R2']:.4f}"),
        ]):
            col.metric(label, val)

    # ── TAB 3: Model comparison ───────────────────────────────────────────────
    with tab3:
        st.subheader("📉 Model Performance Comparison")

        names  = list(results.keys())
        colors = ["#58A6FF", "#3FB950", "#F78166"]

        c1, c2 = st.columns(2)

        with c1:
            fig_mae = go.Figure(go.Bar(
                x=names,
                y=[results[n]["metrics"]["MAE"]  for n in names],
                marker_color=colors, text=[f"{results[n]['metrics']['MAE']:.3f}" for n in names],
                textposition="outside"))
            fig_mae.update_layout(title="Mean Absolute Error (↓ lower is better)",
                                  template="plotly_dark",
                                  paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                                  height=350, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig_mae, use_container_width=True)

        with c2:
            fig_rmse = go.Figure(go.Bar(
                x=names,
                y=[results[n]["metrics"]["RMSE"] for n in names],
                marker_color=colors, text=[f"{results[n]['metrics']['RMSE']:.3f}" for n in names],
                textposition="outside"))
            fig_rmse.update_layout(title="Root Mean Squared Error (↓ lower is better)",
                                   template="plotly_dark",
                                   paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                                   height=350, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig_rmse, use_container_width=True)

        fig_r2 = go.Figure(go.Bar(
            x=names,
            y=[results[n]["metrics"]["R2"] for n in names],
            marker_color=colors, text=[f"{results[n]['metrics']['R2']:.4f}" for n in names],
            textposition="outside"))
        fig_r2.update_layout(title="R² Score (↑ higher is better)",
                              template="plotly_dark",
                              paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                              yaxis=dict(range=[0, 1.1]),
                              height=380, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig_r2, use_container_width=True)

        st.subheader("📋 Metrics Table")
        rows = []
        for name in names:
            m = results[name]["metrics"]
            rows.append({"Model": name,
                         "MAE":  round(m["MAE"],  4),
                         "MSE":  round(m["MSE"],  4),
                         "RMSE": round(m["RMSE"], 4),
                         "R²":   round(m["R2"],   4)})
        st.dataframe(pd.DataFrame(rows).set_index("Model"),
                     use_container_width=True)

    # ── TAB 4: EDA ────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("🔍 Exploratory Data Analysis")
        c1, c2 = st.columns(2)

        with c1:
            fig_ret = px.histogram(df_feat, x="Daily_Return",
                                   nbins=80, title="Distribution of Daily Returns",
                                   color_discrete_sequence=["#58A6FF"])
            fig_ret.update_layout(template="plotly_dark",
                                  paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                                  height=350, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig_ret, use_container_width=True)

        with c2:
            corr_cols = ["Close","MA_20","MA_50","RSI","MACD","Volatility","Volume"]
            corr = df_feat[corr_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  title="Feature Correlation Heatmap", aspect="auto")
            fig_corr.update_layout(template="plotly_dark",
                                   paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                                   height=350, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig_corr, use_container_width=True)

        fig_vol = px.line(df_feat, x=df_feat.index, y="Volatility",
                          title="10-Day Rolling Volatility",
                          color_discrete_sequence=["#F78166"])
        fig_vol.update_layout(template="plotly_dark",
                               paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                               height=300, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig_vol, use_container_width=True)

    # ── TAB 5: Raw data ───────────────────────────────────────────────────────
    with tab5:
        st.subheader("📋 Historical Stock Data")
        display_cols = ["Open","High","Low","Close","Volume","MA_20","MA_50","RSI","MACD"]
        st.dataframe(df_feat[display_cols].sort_index(ascending=False).head(200),
                     use_container_width=True)
        csv = df_feat.to_csv().encode("utf-8")
        st.download_button("⬇️ Download full dataset (CSV)",
                           data=csv,
                           file_name=f"{ticker}_stock_data.csv",
                           mime="text/csv")

else:
    # ── Landing prompt ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;">
        <div style="font-size:5rem;">📈</div>
        <h2 style="color:#58A6FF; margin-bottom:12px;">Ready to Predict</h2>
        <p style="color:#8B949E; font-size:1.1rem; max-width:500px; margin:auto;">
            Enter a stock ticker symbol in the sidebar and click
            <strong style="color:#3FB950;">🚀 Run Prediction</strong>
            to start the full ML pipeline.
        </p>
        <br>
        <p style="color:#8B949E; font-size:0.9rem;">
            Supports any Yahoo Finance ticker · AAPL · TSLA · GOOGL · MSFT · AMZN · NVDA …
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#30363D; margin-top:40px;">
<div style="text-align:center; color:#8B949E; font-size:0.8rem; padding:10px 0;">
    AI Stock Market Prediction System · Built with Python, Scikit-learn &amp; Streamlit ·
    <a href="https://github.com" style="color:#58A6FF;">GitHub</a>
</div>
""", unsafe_allow_html=True)
