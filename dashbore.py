# app.py
import streamlit as st
import pandas as pd
import numpy as np
from kiteconnect import KiteConnect
import talib

# ====== Kite API setup ======
API_KEY = "gqmhg28m3qd411ri"
API_SECRET = "p5vuu9taqvkaduesamtjyig6d0qvztet"
ACCESS_TOKEN = "n3naMoUEkzhHvZyTPmlL0oAN1F7KYYKF"

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ====== Streamlit UI ======
st.title("Nifty 5-Min Intraday Score Meter")
st.write("Calculating bullish/bearish score using 100 indicators...")

# ====== Fetch historical 5-min Nifty data ======
def get_nifty_data():
    instrument_token = 256265  # NIFTY 50 index token
    data = kite.historical_data(
        instrument_token,
        "2025-11-03 09:15:00",
        "2025-11-03 15:30:00",
        interval="5minute"
    )
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df

df = get_nifty_data()

# ====== Indicator calculations ======
# Simple EMA/SMA examples, extend to all 100
df['EMA9'] = talib.EMA(df['close'], timeperiod=9)
df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
df['EMA50'] = talib.EMA(df['close'], timeperiod=50)
df['EMA100'] = talib.EMA(df['close'], timeperiod=100)
df['EMA200'] = talib.EMA(df['close'], timeperiod=200)

df['SMA20'] = talib.SMA(df['close'], timeperiod=20)
df['SMA50'] = talib.SMA(df['close'], timeperiod=50)
df['SMA100'] = talib.SMA(df['close'], timeperiod=100)
df['SMA200'] = talib.SMA(df['close'], timeperiod=200)

df['RSI'] = talib.RSI(df['close'], timeperiod=14)
macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['MACD'] = macd
df['MACD_signal'] = macd_signal

# ====== Scoring ======
def calculate_score(row):
    score = 0

    # Example: EMA9
    if row['close'] > row['EMA9']:
        score += 1
    else:
        score -= 1

    # EMA20
    if row['close'] > row['EMA20']:
        score += 1
    else:
        score -= 1

    # EMA50
    if row['close'] > row['EMA50']:
        score += 1
    else:
        score -= 1

    # EMA100
    if row['close'] > row['EMA100']:
        score += 1
    else:
        score -= 1

    # EMA200
    if row['close'] > row['EMA200']:
        score += 1
    else:
        score -= 1

    # SMA20
    if row['close'] > row['SMA20']:
        score += 1
    else:
        score -= 1

    # SMA50
    if row['close'] > row['SMA50']:
        score += 1
    else:
        score -= 1

    # SMA100
    if row['close'] > row['SMA100']:
        score += 1
    else:
        score -= 1

    # SMA200
    if row['close'] > row['SMA200']:
        score += 1
    else:
        score -= 1

    # MACD
    if row['MACD'] > row['MACD_signal']:
        score += 1
    else:
        score -= 1

    # RSI
    if row['RSI'] > 55:
        score += 1
    elif row['RSI'] < 45:
        score -= 1

    # TODO: add remaining 89 indicators similarly

    return score

df['Score'] = df.apply(calculate_score, axis=1)
df['Bullish %'] = (df['Score'] / 100) * 100
df['Bearish %'] = 100 - df['Bullish %']

# ====== Display ======
st.subheader("Latest Candle Score")
latest = df.iloc[-1]
st.write(f"Time: {latest.name}")
st.write(f"Bullish %: {latest['Bullish %']:.2f}%")
st.write(f"Bearish %: {latest['Bearish %']:.2f}%")

st.subheader("Score Table (last 10 candles)")
st.dataframe(df[['close', 'Score', 'Bullish %', 'Bearish %']].tail(10))
