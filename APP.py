# stock_analysis_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import date

# ---------------------------- Helper Functions ---------------------------- #

def get_data(ticker, start, end, interval):
    data = yf.download(ticker, start=start, end=end, interval=interval)
    return data

def plot_candlestick(data, title="Candlestick Chart"):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

def intraday_analysis(ticker, start, end):
    st.subheader("📊 Intraday Analysis (5-minute)")
    data = get_data(ticker, start, end, '5m')
    if data.empty:
        st.error("No intraday data available for selected date.")
        return
    
    st.write("Line Chart of Closing Price:")
    st.line_chart(data['Close'])

    st.write("Candlestick Chart:")
    plot_candlestick(data)

    data['EMA_9'] = data['Close'].ewm(span=9).mean()
    st.write("9-Period EMA Overlay:")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close')
    ax.plot(data.index, data['EMA_9'], label='9-EMA', linestyle='--')
    ax.legend()
    st.pyplot(fig)

def short_term_analysis(ticker, start, end):
    st.subheader("📈 Short-Term Analysis (Daily)")
    data = get_data(ticker, start, end, '1d')
    if data.empty:
        st.error("No data available.")
        return

    st.write("Line Chart of Closing Price:")
    st.line_chart(data['Close'])

    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    
    st.write("10-day and 20-day SMA Crossover:")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close')
    ax.plot(data.index, data['SMA_10'], label='10-SMA')
    ax.plot(data.index, data['SMA_20'], label='20-SMA')
    ax.legend()
    st.pyplot(fig)

    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    st.write("RSI (14-period):")
    fig2, ax2 = plt.subplots()
    ax2.plot(data.index, rsi, label='RSI', color='orange')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='green', linestyle='--')
    st.pyplot(fig2)

def long_term_analysis(ticker, start, end):
    st.subheader("📉 Long-Term Analysis (Daily)")
    data = get_data(ticker, start, end, '1d')
    if data.empty:
        st.error("No data available.")
        return

    st.write("Line Chart of Closing Price:")
    st.line_chart(data['Close'])

    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['SMA_100'] = data['Close'].rolling(100).mean()
    data['SMA_200'] = data['Close'].rolling(200).mean()

    st.write("SMA 50 / 100 / 200:")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close')
    ax.plot(data.index, data['SMA_50'], label='50-SMA')
    ax.plot(data.index, data['SMA_100'], label='100-SMA')
    ax.plot(data.index, data['SMA_200'], label='200-SMA')
    ax.legend()
    st.pyplot(fig)

    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    years = (data.index[-1] - data.index[0]).days / 365.0
    cagr = ((end_price / start_price) ** (1 / years)) - 1
    st.write(f"📈 CAGR: **{cagr:.2%}**")

    index_data = get_data('^NSEI', start, end, '1d')
    combined = pd.concat([data['Close'], index_data['Close']], axis=1)
    combined.columns = ['Stock', 'Index']
    combined.dropna(inplace=True)
    
    beta = combined.pct_change().cov().iloc[0, 1] / combined.pct_change().cov().iloc[1, 1]
    st.write(f"🧮 Beta vs NIFTY: **{beta:.2f}**")

# ---------------------------- Streamlit UI ---------------------------- #

st.title("📊 Stock Market Analysis Tool")
st.write("Welcome Hustler, I will help you with basic Stock Prediction Analysis.")

ticker = st.text_input("Enter Stock Ticker (e.g., INFY.NS for Infosys):", value="INFY.NS")
analysis_type = st.selectbox("Select Analysis Type:", ["Intraday", "Short Term", "Long Term"])

today = date.today()
default_start = today.replace(year=today.year - 1)

start_date = st.date_input("Select Start Date:", value=default_start)
end_date = st.date_input("Select End Date:", value=today)

if st.button("Run Analysis"):
    if start_date >= end_date:
        st.warning("Start date must be earlier than end date.")
    else:
        if analysis_type == "Intraday":
            intraday_analysis(ticker, start_date, end_date)
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
            st.write("💡 Try different stocks, dates or analysis types!")
