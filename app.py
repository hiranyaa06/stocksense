# app.py — StockSense Professional Dashboard
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from model import train_model, predict_7_days

# page_confg
st.set_page_config(
    page_title="StockSense",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS (professional look!) ───────────────────
st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00d4aa;
    }
    .title {
        font-size: 42px;
        font-weight: 700;
        color: #00d4aa;
    }
    .subtitle {
        color: #888;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────
st.markdown('<p class="title">📈 StockSense</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered stock price forecasting</p>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    popular_tickers = [
        "AAPL",
        "TSLA",
        "GOOGL",
        "MSFT",
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS"
    ]

    selected_ticker = st.selectbox(
        "Quick Select",
        popular_tickers,
        index=0
    )

    ticker = st.text_input(
        "Or Enter Custom Ticker",
        value=selected_ticker
    ).upper()

    period = st.selectbox(
        "Historical Data",
        ["1y", "2y", "5y"],
        index=1
    )

    st.divider()

    predict_btn = st.button(
        "🔮 Predict Now",
        use_container_width=True
    )

    st.divider()

    st.markdown("### Popular Stocks")
    st.markdown(
        """
        🇺🇸 AAPL • TSLA • GOOGL • MSFT

        🇮🇳 RELIANCE.NS • TCS.NS • INFY.NS
        """
    )

# ── Main Content ──────────────────────────────────────
if predict_btn:
    with st.spinner(f"Fetching data & training model for {ticker}..."):
        try:
            # Train model
            model, df, feature_cols, mae, rmse = train_model(ticker)

            # Predict 7 days
            pred_7 = predict_7_days(model, df, feature_cols)

            # ── Metric Cards ──────────────────────────
            st.subheader(f"📊 {ticker} Overview")
            col1, col2, col3, col4 = st.columns(4)

            latest_price = float(df['Close'].values[-1])
            prev_price   = float(df['Close'].values[-2])
            change       = latest_price - prev_price
            change_pct   = (change / prev_price) * 100

            col1.metric("Latest Price",  f"${latest_price:.2f}")
            col2.metric("Daily Change",  f"${change:.2f}",  f"{change_pct:.2f}%")
            col3.metric("MAE",           f"${mae:.2f}")
            col4.metric("RMSE",          f"${rmse:.2f}")

            st.divider()

            # ── Historical Price Chart ─────────────────
            st.subheader("📉 Historical Price")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df.index[-90:],
                y=df['Close'].values[-90:],
                mode='lines',
                name='Close Price',
                line=dict(color='#00d4aa', width=2)
            ))
            fig1.update_layout(
                template='plotly_dark',
                height=350,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig1, use_container_width=True)

            st.divider()

            # ── 7 Day Prediction Chart ─────────────────
            st.subheader("🔮 7-Day Price Forecast")
            fig2 = go.Figure()

            # Last 30 days actual
            fig2.add_trace(go.Scatter(
                x=df.index[-30:],
                y=df['Close'].values[-30:],
                mode='lines',
                name='Actual',
                line=dict(color='#00d4aa', width=2)
            ))

            # 7 day predictions
            fig2.add_trace(go.Scatter(
                x=pred_7.index,
                y=pred_7['price'].values,
                mode='lines+markers',
                name='Predicted',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                marker=dict(size=8)
            ))

            fig2.update_layout(
                template='plotly_dark',
                height=350,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig2, use_container_width=True)

            # ── Prediction Table ───────────────────────
            st.subheader("📅 7-Day Predictions Table")
            st.dataframe(
                pred_7.style.format("${:.2f}"),
                use_container_width=True
            )

        except Exception as e:
            import traceback
            traceback.print_exc()

else:
    # Welcome screen
    st.info("Enter a stock ticker in the sidebar and click **Predict Now** to get started!")
    st.markdown("""
    ### What StockSense does:
    - 📥 Fetches real-time stock data
    - 🤖 Trains a Random Forest model
    - 🔮 Predicts next 7 days of prices
    - 📊 Shows beautiful interactive charts
    """)