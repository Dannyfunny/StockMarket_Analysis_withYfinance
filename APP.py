import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import date, timedelta

# ---------------------------- Data Fetch ---------------------------- #
def get_data(ticker, start, end, interval):
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval)
        return df.dropna()
    except:
        return pd.DataFrame()

# ---------------------------- Charting Functions ---------------------------- #
def plot_candlestick(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    candlestick_data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])

    if candlestick_data.empty:
        st.warning("⚠ No valid candlestick data to display.")
        return

    st.subheader(f"📈 {ticker} - Candlestick Chart")
    fig = go.Figure(data=[
        go.Candlestick(x=candlestick_data.index,
                       open=candlestick_data['Open'],
                       high=candlestick_data['High'],
                       low=candlestick_data['Low'],
                       close=candlestick_data['Close'])
    ])
    fig.update_layout(xaxis_rangeslider_visible=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Analysis Functions ---------------------------- #
def intraday_analysis(ticker):
    st.subheader("📊 Intraday Analysis (5-minute intervals)")
    st.warning("⚠️ Intraday data is only available for the latest trading day.")
    data = yf.download(ticker, period="1d", interval="5m")

    if data.empty or "Close" not in data.columns:
        st.error("No intraday data available. Market may be closed or data restricted.")
        return

    st.line_chart(data['Close'])
    plot_candlestick(data, ticker)

    data['EMA_9'] = data['Close'].ewm(span=9).mean()
    st.write("📈 9-period EMA Overlay")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close')
    ax.plot(data.index, data['EMA_9'], label='9-EMA', linestyle='--')
    ax.legend()
    st.pyplot(fig)

def short_term_analysis(ticker, start, end):
    st.subheader("📈 Short-Term Analysis (Daily)")

    with st.spinner("📡 Fetching data..."):
        try:
            data = yf.download(ticker, start=start, end=end)
        except Exception as e:
            st.error(f"⚠ Error fetching data: {e}")
            st.stop()

    if data.empty:
        st.error("⚠ No historical data found for this ticker.")
        st.stop()

    st.write("📊 Sample Data", data.tail())
    plot_candlestick(data, ticker)

    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

    st.write("📊 10-day and 20-day SMA Crossover")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, data['SMA_10'], label='10-SMA')
    ax.plot(data.index, data['SMA_20'], label='20-SMA')
    ax.legend()
    st.pyplot(fig)

    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    st.write("📉 RSI (14-period)")
    fig2, ax2 = plt.subplots()
    ax2.plot(data.index, rsi, label='RSI', color='orange')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='green', linestyle='--')
    ax2.set_title("RSI Indicator")
    st.pyplot(fig2)

def long_term_analysis(ticker, start, end):
    st.subheader("📉 Long-Term Analysis (6 months to 5 years)")

    with st.spinner("📡 Fetching data..."):
        try:
            data = yf.download(ticker, start=start, end=end)
        except Exception as e:
            st.error(f"⚠ Error fetching data: {e}")
            st.stop()

    if data.empty:
        st.error("⚠ No historical data found for this ticker.")
        st.stop()

    st.write("📊 Sample Data", data.tail())
    plot_candlestick(data, ticker)

    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_100'] = data['Close'].rolling(window=100).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    st.write("📊 SMA Trend (50/100/200)")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close')
    ax.plot(data.index, data['SMA_50'], label='50-SMA')
    ax.plot(data.index, data['SMA_100'], label='100-SMA')
    ax.plot(data.index, data['SMA_200'], label='200-SMA')
    ax.legend()
    st.pyplot(fig)

    try:
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        years = (data.index[-1] - data.index[0]).days / 365.0
        if years > 0:
            cagr = ((end_price / start_price) ** (1 / years)) - 1
            st.write(f"📈 CAGR: **{cagr:.2%}** over {years:.2f} years")
        else:
            st.warning("⚠️ Insufficient duration for CAGR calculation.")
    except Exception as e:
        st.warning(f"⚠️ Unable to calculate CAGR: {e}")

    index_data = get_data('^NSEI', start, end, '1d')
    combined = pd.concat([data['Close'], index_data['Close']], axis=1)
    combined.columns = ['Stock', 'Index']
    combined.dropna(inplace=True)
    if not combined.empty:
        cov_matrix = combined.pct_change().cov()
        beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1]
        st.write(f"📎 Beta vs NIFTY: **{beta:.2f}**")
    else:
        st.warning("⚠️ Could not calculate Beta — missing Index data.")

# ---------------------------- Streamlit UI ---------------------------- #
st.set_page_config(page_title="Stock Analysis App", layout="wide")
st.title("📊 Stock Market Analysis App")
st.write("Welcome Hustler, I will help you with basic Stock Prediction Analysis.")

ticker = st.text_input("Enter Stock Ticker (e.g., INFY.NS)", "INFY.NS")
analysis_type = st.selectbox("Select Analysis Type", ["Intraday", "Short Term", "Long Term"])

today = date.today()
min_date = today - timedelta(days=365 * 5)

if analysis_type == "Short Term":
    st.info("📅 Recommended: 30 to 90 days for Short-Term analysis.")
    start_date = st.date_input("Select Start Date", value=today - timedelta(days=60), min_value=min_date, max_value=today)
    end_date = st.date_input("Select End Date", value=today, min_value=min_date, max_value=today)

elif analysis_type == "Long Term":
    st.info("📅 Recommended: 6 months to 5 years for Long-Term analysis.")
    start_date = st.date_input("Select Start Date", value=today - timedelta(days=730), min_value=min_date, max_value=today)
    end_date = st.date_input("Select End Date", value=today, min_value=min_date, max_value=today)

if st.button("Run Analysis"):
    if analysis_type != "Intraday" and start_date >= end_date:
        st.warning("❗ Start date must be earlier than end date.")
    else:
        if analysis_type == "Intraday":
            intraday_analysis(ticker)
        elif analysis_type == "Short Term":
            short_term_analysis(ticker, start_date, end_date)
        elif analysis_type == "Long Term":
            long_term_analysis(ticker, start_date, end_date)

        st.success("✅ Analysis Complete!")
        satisfied = st.radio("Are you satisfied with the results?", ["Yes", "No"])
        if satisfied == "Yes":
            st.balloons()
            st.write("🚀 Hope you make the money you desire, Hustler!")
        else:
            st.write("💡 Try other stocks or date ranges!")
