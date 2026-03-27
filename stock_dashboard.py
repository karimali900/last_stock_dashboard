import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from sqlalchemy import create_engine, text
from datetime import datetime

st.set_page_config(page_title="Karim Stock Market Hub", page_icon="📈", layout="wide")
st.title("📊 Karim Stock Market Hub")
st.caption("Live prices • Technical analysis • Sentiment • News • Forecasts")

# ====================== DATABASE ======================
@st.cache_resource
def get_db_engine():
    return create_engine("postgresql://postgres:karim@localhost:5432/stocks_2026")

db = get_db_engine()

# ====================== HELPERS (now included) ======================
def get_live_price(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        p = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        ch = info.get('regularMarketChangePercent') or 0.0
        return round(p, 4) if p else None, round(ch, 2)
    except:
        return None, 0.0

def compute_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(df):
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def compute_bollinger_bands(df, window=20):
    mb = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    return mb + 2*std, mb, mb - 2*std

# ====================== SIDEBAR ======================
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Refresh All Data Now"):
    st.rerun()

tickers = ["AAPL", "TSLA", "COMI.CA", "HRHO.CA", "2222.SR", "VOD.L", "BP.L"]

# ====================== LIVE PRICES ======================
st.subheader("📈 Live Prices")
live_data = []
for t in tickers:
    p, ch = get_live_price(t)
    arrow = "▲" if ch > 0 else "▼" if ch < 0 else "─"
    live_data.append({"Ticker": t, "Price": p or "N/A", "Change %": ch, "Arrow": arrow})

live_df = pd.DataFrame(live_data)
st.dataframe(
    live_df.style.format({"Price": "${:.2f}", "Change %": "{:+.2f}%"}),
    use_container_width=True,
    hide_index=True
)

# ====================== LATEST DATA FROM DATABASE ======================
st.subheader("📋 Latest Analysis from Database")
latest_query = text("""
    SELECT DISTINCT ON ("Ticker") *
    FROM stock_history 
    ORDER BY "Ticker", "Datetime" DESC
""")
latest_df = pd.read_sql(latest_query, db)

# ====================== INTERACTIVE CHARTS ======================
st.subheader("📉 Interactive Charts")
col1, col2 = st.columns([1, 1])

with col1:
    ticker_choice = st.selectbox("Select stock", tickers, index=0)
    
    df_t = pd.read_sql(text(f"""
        SELECT * FROM stock_history 
        WHERE "Ticker" = '{ticker_choice}' 
        ORDER BY "Datetime" DESC LIMIT 90
    """), db).sort_values("Datetime")

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df_t['Datetime'], y=df_t['Close'], name="Close Price", line=dict(color="blue")))
    ub, mb, lb = compute_bollinger_bands(df_t)
    fig_price.add_trace(go.Scatter(x=df_t['Datetime'], y=ub, name="Upper BB", line=dict(dash="dot", color="red")))
    fig_price.add_trace(go.Scatter(x=df_t['Datetime'], y=lb, name="Lower BB", line=dict(dash="dot", color="green")))
    fig_price.update_layout(title=f"{ticker_choice} Price + Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    rsi = compute_rsi(df_t)
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df_t['Datetime'], y=rsi, name="RSI", line=dict(color="purple")))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(title=f"{ticker_choice} RSI (14)", xaxis_title="Date", yaxis_title="RSI")
    st.plotly_chart(fig_rsi, use_container_width=True)

# ====================== NEWS ======================
st.subheader("📰 Latest Stock News")
for t in tickers:
    try:
        stock = yf.Ticker(t)
        news_list = stock.news[:3]
        st.write(f"**{t}**")
        for item in news_list:
            st.markdown(f"• [{item.get('title')}]({item.get('link')})")
    except:
        pass

st.success(f"✅ Dashboard updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
