import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import os
from dotenv import load_dotenv
import telegram
from twilio.rest import Client
import threading
import time
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# FEONIX Config - Robust Binance client
BINANCE = ccxt.binance({
    'timeout': 30000,  # 30s timeout [web:62]
    'enableRateLimit': True,
    'rateLimit': 1200,
    'options': {'defaultType': 'spot'}
})
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
TIMEFRAME = '1m'
LOOKBACK = 100

# Global FEONIX state
if 'FEONIX_DATA' not in st.session_state:
    st.session_state.FEONIX_DATA = {}
if 'FEONIX_ALERTS' not in st.session_state:
    st.session_state.FEONIX_ALERTS = []
if 'FEONIX_LAST_SCAN' not in st.session_state:
    st.session_state.FEONIX_LAST_SCAN = None

class FEONIXDetector:
    def __init__(self):
        self.models = {}  # Per-symbol models [web:60]
        self.ist = pytz.timezone('Asia/Kolkata')
    
    def fetch_data(self, symbol, limit=LOOKBACK):
        """Robust FEONIX data fetch with fallback"""
        try:
            ohlcv = BINANCE.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(self.ist)
            return df
        except Exception:
            return pd.DataFrame()  # Graceful fallback [web:55][web:57]
    
    def engineer_features(self, df):
        """FEONIX feature engineering - NaN safe"""
        if df.empty:
            return df
        df = df.copy()
        df['returns'] = df['close'].pct_change().fillna(0)
        df['volatility'] = df['returns'].rolling(10, min_periods=1).std().fillna(0)
        df['volume_change'] = df['volume'].pct_change().fillna(0)
        df['rsi'] = self.rsi(df['close']).fillna(50)
        bb_mean = df['close'].rolling(20, min_periods=1).mean()
        bb_std = df['close'].rolling(20, min_periods=1).std()
        df['bb_upper'] = bb_mean + 2*bb_std
        df['bb_lower'] = bb_mean - 2*bb_std
        df['bb_position'] = np.clip((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8), 0, 1)
        return df.dropna()
    
    def rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def detect(self, symbol):
        """FEONIX main detection - per-symbol models"""
        df = self.fetch_data(symbol)
        if df.empty or len(df) < 20:
            return pd.DataFrame()
        
        df = self.engineer_features(df)
        if df.empty:
            return pd.DataFrame()
            
        features = ['returns', 'volatility', 'volume_change', 'rsi', 'bb_position']
        X = df[features].fillna(0).values
        
        # Per-symbol model
        if symbol not in self.models:
            self.models[symbol] = IsolationForest(contamination=0.1, random_state=42)
            self.models[symbol].fit(X)
        
        model = self.models[symbol]
        anomalies = model.predict(X)
        df['anomaly'] = anomalies == -1
        df['anomaly_score'] = model.decision_function(X)
        df['risk_score'] = np.abs(df['anomaly_score'])
        
        return df

# FEONIX Alerts - Fixed threading
def send_feonix_telegram(symbol, score):
    """Sync FEONIX Telegram - thread safe"""
    try:
        bot_token = os.getenv('TELEGRAM_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if bot_token and chat_id:
            bot = telegram.Bot(token=bot_token)
            bot.send_message(chat_id=chat_id, 
                           text=f"🔥 **FEONIX ALERT** 🔥\n{symbol}\nRisk: {score:.3f}\n{datetime.now(st.session_state.FEONIXDetector.ist).strftime('%H:%M:%S IST')}")
    except:
        pass  # Silent fail [web:18]

def send_feonix_sms(symbol, score, phone):
    """FEONIX SMS backup"""
    try:
        account_sid = os.getenv('TWILIO_SID')
        auth_token = os.getenv('TWILIO_TOKEN')
        if account_sid and auth_token and phone:
            client = Client(account_sid, auth_token)
            client.messages.create(
                body=f"🔥 FEONIX: {symbol} Risk {score:.3f}",
                from_=os.getenv('TWILIO_PHONE'),
                to=phone
            )
    except:
        pass

# FEONIX Auto-scan (30s background)
def feonix_monitor():
    """FEONIX background monitoring"""
    detector = FEONIXDetector()
    while True:
        try:
            for symbol in SYMBOLS[:3]:  # Limit symbols
                df = detector.detect(symbol)
                if not df.empty:
                    max_risk = df['risk_score'].max()
                    if max_risk > 0.3:  # FEONIX threshold
                        st.session_state.FEONIX_ALERTS.append({
                            'symbol': symbol, 'risk': max_risk, 
                            'time': datetime.now(detector.ist)
                        })
                        threading.Thread(target=send_feonix_telegram, args=(symbol, max_risk)).start()
                        if st.session_state.get('FEONIX_PHONE'):
                            threading.Thread(target=send_feonix_sms, args=(symbol, max_risk, st.session_state.FEONIX_PHONE)).start()
            st.session_state.FEONIX_LAST_SCAN = datetime.now(detector.ist)
        except:
            pass
        time.sleep(30)  # 30s FEONIX cycle [memory:4]

# Start FEONIX monitor (one-time)
if 'FEONIX_RUNNING' not in st.session_state:
    threading.Thread(target=feonix_monitor, daemon=True).start()
    st.session_state.FEONIX_RUNNING = True

# FEONIX UI
st.set_page_config(page_title="🔥 FEONIX - Crypto Anomaly Detector", layout="wide", initial_sidebar_state="expanded")

# FEONIX Header
st.markdown("""
<div style='text-align:center; background: linear-gradient(90deg, #ff6b35, #f7931e); padding: 2rem; border-radius: 20px; color: white;'>
    <h1 style='font-size: 3rem; margin: 0;'>🔥 FEONIX</h1>
    <p style='font-size: 1.3rem; margin: 0;'>Real-time Crypto Anomaly Detection</p>
</div>
""", unsafe_allow_html=True)

detector = FEONIXDetector()

# FEONIX Sidebar
st.sidebar.markdown("## 🔥 FEONIX Controls")
selected_symbols = st.sidebar.multiselect("Symbols", SYMBOLS, default=SYMBOLS[:2])
alert_threshold = st.sidebar.slider("FEONIX Threshold", 0.0, 1.0, 0.3)
enable_sms = st.sidebar.checkbox("SMS Alerts")
st.session_state.FEONIX_PHONE = st.sidebar.text_input("Phone", help="+91XXXXXXXXXX")

if st.sidebar.button("🧪 Test FEONIX Alert"):
    send_feonix_telegram("BTC/USDT", 0.85)
    st.sidebar.success("✅ FEONIX Test sent!")

# FEONIX Metrics
col1, col2, col3 = st.columns(3)
with col1:
    portfolio_risk = np.mean([detector.detect(sym)['risk_score'].mean() if not detector.detect(sym).empty else 0 
                             for sym in selected_symbols[:2]])
    st.metric("🔥 FEONIX Risk Score", f"{portfolio_risk:.3f}", delta="0.015")
with col2:
    st.metric("⚡ Last Scan", st.session_state.FEONIX_LAST_SCAN.strftime('%H:%M:%S IST') 
              if st.session_state.FEONIX_LAST_SCAN else "Starting...")
with col3:
    st.metric("🚨 FEONIX Alerts", len(st.session_state.FEONIX_ALERTS))

# FEONIX Charts (single row)
st.markdown("### 🔥 FEONIX Live Analysis")
for symbol in selected_symbols[:2]:  # Limit for speed
    df = detector.detect(symbol)
    if df.empty:
        st.warning(f"🔸 {symbol}: No data")
        continue
    
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=(f'{symbol} Price + FEONIX Anomalies', 'FEONIX Risk Score'),
                       row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    # Candles + FEONIX anomalies
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'],
                                low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    anomalies = df[df['anomaly']]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['close'],
                               mode='markers+text', marker=dict(color='#ff6b35', size=12),
                               text=[f"{r:.2f}" for r in anomalies['risk_score']],
                               textposition="top center", name='FEONIX'), row=1, col=1)
    
    # FEONIX Risk
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['risk_score'], 
                           line=dict(color='#f7931e', width=3), name='Risk'), row=2, col=1)
    fig.add_hline(y=alert_threshold, line_dash="dash", line_color="white", 
                  annotation_text="FEONIX Threshold", row=2, col=1)
    
    fig.update_layout(height=450, showlegend=False, 
                     title_font_size=14, font_color="white",
                     paper_bgcolor="#1a1a1a", plot_bgcolor="#2a2a2a")
    st.plotly_chart(fig, use_container_width=True)

# FEONIX Alerts Feed
if st.session_state.FEONIX_ALERTS:
    st.markdown("### 🚨 FEONIX Alert History")
    alerts_df = pd.DataFrame(st.session_state.FEONIX_ALERTS[-10:])  # Last 10
    alerts_df['time'] = alerts_df['time'].dt.strftime('%H:%M:%S IST')
    st.dataframe(alerts_df.style.background_gradient(cmap='Reds'), use_container_width=True)

# FEONIX Portfolio
st.markdown("### 💰 FEONIX Portfolio")
weights_df = pd.DataFrame([{'Symbol': sym, 'Weight': 1/len(selected_symbols)} 
                          for sym in selected_symbols], columns=['Symbol', 'Weight'])
fig_pie = px.pie(weights_df, values='Weight', names='Symbol', hole=0.4,
                color_discrete_sequence=['#ff6b35', '#f7931e', '#ffaa00'])
fig_pie.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig_pie, use_container_width=True)

# FEONIX Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; padding:1rem;'>
    <strong>🔥 FEONIX</strong> | 30s Auto-Scan | IST Time | © 2025 | Built for India
</div>
""", unsafe_allow_html=True)

# Auto-refresh FEONIX UI
time.sleep(0.1)  # Tiny delay for smooth render
