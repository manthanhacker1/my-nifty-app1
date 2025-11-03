import streamlit as st
import pandas as pd
import pandas_ta as ta
from kiteconnect import KiteConnect
import numpy as np
import sys

# --- Configuration ---
st.set_page_config(layout="wide")
st.title("100-Indicator Technical Analysis Dashboard")

# --- 🚨 SECURITY WARNING ---
st.warning("""
**SECURITY WARNING:** You have hardcoded your API credentials.
This is **extremely dangerous**. Anyone with this code can access your account.
Please use Streamlit Secrets or Environment Variables to store your keys.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Broker Credentials")
API_KEY = st.sidebar.text_input("API Key", value="gqmhg28m3qd411ri")
API_SECRET = st.sidebar.text_input("API Secret", value="p5vuu9taqvkaduesamtjyig6d0qvztet", type="password")
ACCESS_TOKEN = st.sidebar.text_input("Access Token", value="n3naMoUEkzhHvZyTPmlL0oAN1F7KYYKF", type="password")

st.sidebar.header("Trade Settings")
# NIFTY 50 Index Token = 256265
INSTRUMENT_TOKEN = st.sidebar.text_input("Instrument Token", value="256265")

# --- Weighting System (Your "give each % weightge") ---
st.sidebar.header("Indicator Weights")
st.sidebar.info("Set the importance (weight) for each signal. 0 = off.")

# Define default weights. You can add all 100 here.
# A weight of 1 is the default.
weights = {
    'EMA_9': 1,
    'EMA_20': 1,
    'SMA_50': 1,
    'MACD': 2,
    'RSI': 2,
    'BBANDS': 1,
    'SUPERTREND': 3,
}

# Create sliders in the sidebar to override default weights
weights['EMA_9'] = st.sidebar.slider("EMA 9 Weight", 0, 10, weights['EMA_9'])
weights['EMA_20'] = st.sidebar.slider("EMA 20 Weight", 0, 10, weights['EMA_20'])
weights['SMA_50'] = st.sidebar.slider("SMA 50 Weight", 0, 10, weights['SMA_50'])
weights['MACD'] = st.sidebar.slider("MACD Weight", 0, 10, weights['MACD'])
weights['RSI'] = st.sidebar.slider("RSI Weight", 0, 10, weights['RSI'])
weights['BBANDS'] = st.sidebar.slider("Bollinger Bands Weight", 0, 10, weights['BBANDS'])
weights['SUPERTREND'] = st.sidebar.slider("SuperTrend Weight", 0, 10, weights['SUPERTREND'])


# --- Main Application ---

def fetch_data(api_key, api_secret, access_token, instrument_token):
    """Connects to Kite and fetches 5-minute historical data."""
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Fetch 100 days of 5-minute data
        from datetime import datetime, timedelta
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=100)
        
        data = kite.historical_data(instrument_token, from_date, to_date, "5minute")
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        st.error(f"API Error: {e}")
        st.stop()

def calculate_indicators(df):
    """Calculates all technical indicators using pandas-ta."""
    st.write("Calculating Indicators...")
    
    # 1. EMAs (Close > EMA)
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=20, append=True)
    
    # 2. SMA (Close > SMA)
    df.ta.sma(length=50, append=True)

    # 3. MACD (MACD > Signal)
    # This adds: MACD_12_26_9, MACDh_12_26_9 (Histogram), MACDs_12_26_9 (Signal)
    df.ta.macd(append=True)

    # 4. RSI (RSI > 55 / < 45)
    df.ta.rsi(length=14, append=True)

    # 5. Bollinger Bands (Close > Midband)
    # Adds: BBL_20_2.0, BBM_20_2.0 (Midband), BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
    df.ta.bbands(length=20, append=True)

    # 6. SuperTrend (Green)
    # Adds: SUPERT_7_3.0, SUPERTd_7_3.0 (Direction), SUPERTl_7_3.0, SUPERTs_7_3.0
    df.ta.supertrend(length=7, multiplier=3.0, append=True)

    # Add more of your 100 indicators here...
    # Example: df.ta.adx(append=True)
    # Example: df.ta.vwap(append=True)
    
    df.dropna(inplace=True)
    return df

def generate_signals(df):
    """Generates Bullish (1), Bearish (-1), or Neutral (0) signals."""
    st.write("Generating Signals...")
    signals = pd.DataFrame(index=df.index)
    
    # Signal: 1 = Bullish, -1 = Bearish, 0 = Neutral
    
    # 1. EMA 9 (Indicator 1)
    signals['EMA_9'] = np.where(df['close'] > df['EMA_9'], 1, -1)
    
    # 2. EMA 20 (Indicator 2)
    signals['EMA_20'] = np.where(df['close'] > df['EMA_20'], 1, -1)

    # 3. SMA 50 (Indicator 7)
    signals['SMA_50'] = np.where(df['close'] > df['SMA_50_20'], 1, -1) # Note: pandas-ta uses SMA_length_lookback

    # 4. MACD (Indicator 10)
    # Your condition: MACD > Signal. This is true when Histogram > 0
    signals['MACD'] = np.where(df['MACDh_12_26_9'] > 0, 1, -1)

    # 5. RSI (Indicator 11)
    # Your condition: RSI > 55 (Bullish), RSI < 45 (Bearish)
    signals['RSI'] = np.where(df['RSI_14'] > 55, 1, np.where(df['RSI_14'] < 45, -1, 0))

    # 6. Bollinger Bands (Indicator 14)
    # Your condition: Close > Midband
    signals['BBANDS'] = np.where(df['close'] > df['BBM_20_2.0'], 1, -1)

    # 7. SuperTrend (Indicator 16)
    # Your condition: Green (1) vs Red (-1)
    # `SUPERTd_7_3.0` column holds the direction (1 for up, -1 for down)
    signals['SUPERTREND'] = df['SUPERTd_7_3.0']

    return signals

def calculate_weighted_score(signals, weights):
    """Applies weights and calculates a final 0-100 score."""
    st.write("Calculating Weighted Score...")
    
    # Get the most recent signal for each indicator
    latest_signals = signals.iloc[-1]
    
    total_score = 0
    max_score = 0
    
    score_breakdown = {}
    
    for indicator, weight in weights.items():
        if indicator in latest_signals:
            signal = latest_signals[indicator]
            
            # Add to scores
            total_score += signal * weight
            max_score += weight # The max possible score is the sum of all weights
            
            # Store for breakdown table
            score_breakdown[indicator] = {
                'Signal': 'Bullish' if signal == 1 else ('Bearish' if signal == -1 else 'Neutral'),
                'Weight': weight,
                'Score': signal * weight
            }

    # Normalize the score
    # Raw score is from -max_score to +max_score
    # We convert this to a 0-100% scale
    if max_score == 0:
        return 0, {} # Avoid division by zero
        
    normalized_score = (total_score / max_score) # Scale: -1 to +1
    final_percentage = (normalized_score + 1) / 2 * 100 # Scale: 0 to 100
    
    return final_percentage, score_breakdown


# --- Run Button ---
if st.sidebar.button("Analyze Instrument"):
    
    # 1. Fetch Data
    with st.spinner("Fetching 5-min data..."):
        df = fetch_data(API_KEY, API_SECRET, ACCESS_TOKEN, INSTRUMENT_TOKEN)

    # 2. Calculate Indicators
    with st.spinner("Calculating indicators..."):
        df_with_ta = calculate_indicators(df.copy())

    # 3. Generate Signals
    with st.spinner("Generating signals..."):
        signals_df = generate_signals(df_with_ta)

    # 4. Calculate Final Score
    with st.spinner("Calculating final score..."):
        final_score, breakdown = calculate_weighted_score(signals_df, weights)

    # 5. Display Results
    st.header("Analysis for Instrument: " + INSTRUMENT_TOKEN)
    
    # Get latest candle time
    latest_candle_time = signals_df.index[-1]
    st.subheader(f"Latest Candle: {latest_candle_time} (UTC)")

    # Big Score Metric
    if final_score > 60:
        st.success(f"**Overall Bullish Score: {final_score:.2f}%**")
    elif final_score < 40:
        st.error(f"**Overall Bearish Score: {final_score:.2f}%**")
    else:
        st.warning(f"**Overall Neutral Score: {final_score:.2f}%**")

    # --- Score Breakdown ---
    st.subheader("Score Breakdown")
    breakdown_df = pd.DataFrame.from_dict(breakdown, orient='index')
    st.dataframe(breakdown_df, use_container_width=True)

    # --- Data Tables ---
    st.subheader("Latest Signals (Most Recent 10 Candles)")
    st.dataframe(signals_df.tail(10), use_container_width=True)
    
    st.subheader("Raw Data + Indicators (Most Recent 10 Candles)")
    st.dataframe(df_with_ta.tail(10), use_container_width=True)

else:
    st.info("Click 'Analyze Instrument' in the sidebar to begin.")
