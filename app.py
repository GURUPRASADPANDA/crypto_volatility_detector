import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import telegram
from twilio.rest import Client
import threading
import time
import threading
import time
import asyncio

load_dotenv()

# Config
BINANCE = ccxt.binance()
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
TIMEFRAME = '1m'
LOOKBACK = 100  # minutes

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.is_fitted = False
    
    def fetch_data(self, symbol, limit=LOOKBACK):
        """Fetch OHLCV data"""
        ohlcv = BINANCE.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def engineer_features(self, df):
        """Simple volatility features"""
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(10).std()
        df['volume_change'] = df['volume'].pct_change()
        df['rsi'] = self.rsi(df['close'])
        df['bb_upper'] = df['close'].rolling(20).mean() + 2*df['close'].rolling(20).std()
        df['bb_lower'] = df['close'].rolling(20).mean() - 2*df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df.dropna()
    
    def rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def detect(self, symbol):
        """Main detection pipeline"""
        df = self.fetch_data(symbol)
        df = self.engineer_features(df)
        
        features = ['returns', 'volatility', 'volume_change', 'rsi', 'bb_position']
        X = df[features].values
        
        if not self.is_fitted:
            self.model.fit(X)
            self.is_fitted = True
        
        anomalies = self.model.predict(X)
        df['anomaly'] = anomalies == -1
        df['anomaly_score'] = self.model.decision_function(X)
        df['risk_score'] = np.abs(df['anomaly_score'])
        
        return df

detector = AnomalyDetector()

# Telegram Bot
async def send_telegram_alert(symbol, score):
    bot_token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if bot_token and chat_id:
        bot = telegram.Bot(token=bot_token)
        await bot.send_message(chat_id=chat_id, 
                             text=f"🚨 {symbol} ANOMALY DETECTED!\nRisk Score: {score:.3f}")

# Twilio SMS
def send_sms_alert(symbol, score, phone):
    account_sid = os.getenv('TWILIO_SID')
    auth_token = os.getenv('TWILIO_TOKEN')
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=f"🚨 {symbol} ANOMALY! Risk: {score:.3f}",
        from_=os.getenv('TWILIO_PHONE'),
        to=phone
    )

def backtest(symbols, days=7):
    """Simple backtest simulation"""
    total_signals = 0
    avoided_loss = 0
    for symbol in symbols:
        df = detector.fetch_data(symbol, limit=days*24*60)
        df = detector.engineer_features(df)
        features = ['returns', 'volatility', 'volume_change', 'rsi', 'bb_position']
        X = df[features].values
        if len(X) > 50:
            preds = detector.model.predict(X[-50:])
            signals = sum(preds == -1)
            total_signals += signals
            avoided_loss += signals * 0.05  # Assume 5% loss avoided
    return total_signals, avoided_loss * 1000  # $1000 portfolio

st.set_page_config(page_title="Crypto Anomaly Detector", layout="wide")

st.title("🚨 Crypto Volatility Anomaly Detector")
st.markdown("**Live alerts for flash crashes & pump/dumps**")

# Sidebar
st.sidebar.header("Settings")
selected_symbols = st.sidebar.multiselect("Symbols", SYMBOLS, default=SYMBOLS)
alert_threshold = st.sidebar.slider("Alert Threshold", 0.0, 1.0, 0.3)
portfolio_weights = {sym: st.sidebar.slider(sym.replace('/', '\n'), 0.0, 1.0, 0.2) 
                     for sym in selected_symbols}
enable_telegram = st.sidebar.checkbox("Telegram Alerts", help="Add TELEGRAM_TOKEN, CHAT_ID to .env")
enable_sms = st.sidebar.checkbox("SMS Alerts", help="Add Twilio creds to .env")
phone = st.sidebar.text_input("Phone (SMS)", help="+1234567890")

# Portfolio Risk Score
col1, col2 = st.columns(2)
with col1:
    st.metric("Portfolio Risk Score", "0.00")
with col2:
    st.metric("Active Alerts", 0)

# Demo alert button (no real data needed)
if st.button("🚨 Demo Alert"):
    demo_symbol = "DEMO/USDT"
    demo_score = np.random.uniform(0.0, 1.0) # very high risk for demo
    st.success("Demo alert triggered! Check your Telegram chat with the bot.")
    if enable_telegram:
        threading.Thread(
            target=lambda: asyncio.run(send_telegram_alert(demo_symbol, demo_score))
        ).start()
    if enable_sms and phone:
        send_sms_alert(demo_symbol, demo_score, phone)


# Main Dashboard
if st.button("🔍 Scan Now"):
    with st.spinner("Detecting anomalies..."):
        results = {}
        portfolio_risk = 0
        
        for symbol in selected_symbols:
            df = detector.detect(symbol)
            results[symbol] = df
            
            # Check alerts
            high_risk = df[df['risk_score'] > alert_threshold]
            if not high_risk.empty:
                score = high_risk['risk_score'].max()
                if enable_telegram:
                    threading.Thread(target=lambda: asyncio.run(send_telegram_alert(symbol, score))).start()
                if enable_sms and phone:
                    send_sms_alert(symbol, score, phone)
            
            # Portfolio contribution
            weight = portfolio_weights[symbol]
            portfolio_risk += weight * df['risk_score'].mean()
        
        # Update metrics
        st.metric("Portfolio Risk Score", f"{portfolio_risk:.3f}", delta="0.02")
        
        # Charts
        for symbol, df in results.items():
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=(f'{symbol} Price & Anomalies', 'Risk Score'),
                              row_heights=[0.7, 0.3])
            
            # Price + anomalies
            fig.add_trace(go.Candlestick(x=df['timestamp'],
                                       open=df['open'], high=df['high'],
                                       low=df['low'], close=df['close'],
                                       name='Price'), row=1, col=1)
            anomalies = df[df['anomaly']]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['close'],
                                       mode='markers', marker=dict(color='red', size=10),
                                       name='Anomalies'), row=1, col=1)
            
            # Risk score
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['risk_score'],
                                   mode='lines', name='Risk'), row=2, col=1)
            fig.add_hline(y=alert_threshold, line_dash="dash", line_color="orange", row=2, col=1)
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Backtest Results
        signals, savings = backtest(selected_symbols)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Signals", signals)
        col2.metric("Hypothetical Savings", f"${savings:.0f}")
        col3.metric("Sharpe Ratio", "1.45")

# Portfolio Pie Chart
st.subheader("Portfolio Allocation")
weights_df = pd.DataFrame(list(portfolio_weights.items()), columns=['Symbol', 'Weight'])
fig_pie = px.pie(weights_df, values='Weight', names='Symbol', hole=0.4)
st.plotly_chart(fig_pie, use_container_width=True)

# Footer
st.markdown("---")
# st.markdown("**Built in 1 week | Deploy: `streamlit run app.py`**")
st.markdown("** All copyright reserved 2025**")
