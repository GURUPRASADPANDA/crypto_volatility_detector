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
import threading
import time
import asyncio
from streamlit_autorefresh import st_autorefresh

load_dotenv()

# Set page config
st.set_page_config(
    page_title="Crypto Anomaly Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0f0f0f; }
    .main .block-container { padding-top: 2rem; }
    div.element-container .stSpinner { display: none !important; }
    .stMetric > label { color: #00ff88 !important; font-size: 14px; font-weight: 600; }
    .stMetric > div > div { color: #ffffff !important; font-size: 28px; font-weight: 700; }
    .stPlotlyChart {
        border-radius: 12px; border: 1px solid #333; background: #1a1a1a;
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.1);
    }
    .stButton > button {
        background: linear-gradient(90deg, #00ff88, #00cc6a); color: #000;
        border: none; border-radius: 8px; font-weight: 600; height: 42px;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #00cc6a, #00ff88);
        box-shadow: 0 6px 20px rgba(0, 255, 136, 0.4);
    }
    .stSlider > div > div > div > div { background: linear-gradient(90deg, #00ff88, #ff4444); }
    .stRadio > div > label { color: #00ff88 !important; }
    h1 { color: #00ff88 !important; font-size: 2.8rem !important; font-weight: 700 !important;
         text-shadow: 0 0 20px rgba(0, 255, 136, 0.3); }
    .stMarkdown { color: #ffffff !important; }
    .stError {
        background-color: #ff4444 !important; color: white !important;
        border-radius: 8px; padding: 12px; border-left: 5px solid #ff6666;
    }
    .alert-banner {
        background: linear-gradient(90deg, #ff4444, #ff6666);
        color: white; padding: 15px; border-radius: 10px; margin: 10px 0;
        font-weight: 600; text-align: center; box-shadow: 0 4px 15px rgba(255,68,68,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Config
BINANCE = ccxt.binance({
    'sandbox': False, 'rateLimit': 1200, 'enableRateLimit': True, 'options': {'defaultType': 'spot'}
})

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT','XRP/USD']
TIMEFRAME = '1m'
LOOKBACK = 150

# Global state
if 'latest_results' not in st.session_state:
    st.session_state.latest_results = {}
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {}
if 'active_alerts' not in st.session_state:
    st.session_state.active_alerts = {}

# FIXED: Static functions for caching
@st.cache_data(ttl=10)
def fetch_realtime_data(symbol):
    """Fetch real-time data - FIXED"""
    try:
        since = int((datetime.now(IST) - timedelta(minutes=LOOKBACK)).timestamp() * 1000)
        ohlcv = BINANCE.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=LOOKBACK)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(IST)
        return df.sort_values('timestamp').tail(LOOKBACK).reset_index(drop=True)
    except:
        return generate_demo_data(symbol)

@st.cache_data(ttl=10)
def generate_demo_data(symbol):
    """Generate demo data - FIXED"""
    np.random.seed(int(time.time()) % 100)
    timestamps = pd.date_range(end=datetime.now(IST), periods=LOOKBACK, freq='1T')
    base_price = 65000 if 'BTC' in symbol else 3200 if 'ETH' in symbol else 320
    
    prices = [base_price]
    for i in range(1, LOOKBACK):
        change = np.random.normal(0, 0.0015)
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'timestamp': timestamps, 'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.0008))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.0008))) for p in prices],
        'close': prices,
        'volume': np.random.randint(500, 80000, LOOKBACK) * np.random.uniform(0.5, 2, LOOKBACK)
    })
    
    current_min = datetime.now(IST).minute % LOOKBACK
    for pos in [(current_min - 3) % LOOKBACK, (current_min - 15) % LOOKBACK]:
        if 0 <= pos < len(df):
            df.loc[pos, 'close'] *= 0.93 if pos % 2 == 0 else 1.08
            df.loc[pos, 'low' if pos % 2 == 0 else 'high'] *= 0.90 if pos % 2 == 0 else 1.13
            df.loc[pos, 'volume'] *= 7
    
    return df

def engineer_features(df):
    """Feature engineering"""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(8).std()
    df['volume_change'] = df['volume'].pct_change()
    df['rsi'] = rsi(df['close'])
    df['bb_position'] = bb_position(df['close'])
    df['momentum'] = df['close'].pct_change(3)
    df['volume_sma'] = df['volume'] / df['volume'].rolling(20).mean()
    return df.dropna()

def rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bb_position(prices):
    bb_upper = prices.rolling(20).mean() + 2 * prices.rolling(20).std()
    bb_lower = prices.rolling(20).mean() - 2 * prices.rolling(20).std()
    return (prices - bb_lower) / (bb_upper - bb_lower)

model = IsolationForest(contamination=0.08, random_state=42)
is_fitted = False

def detect(symbol, use_demo=False):
    """Main detection - FIXED"""
    global model, is_fitted
    
    if use_demo:
        df = generate_demo_data(symbol)
    else:
        df = fetch_realtime_data(symbol)
    
    df_eng = engineer_features(df)
    features = ['returns', 'volatility', 'volume_change', 'rsi', 'bb_position', 'momentum', 'volume_sma']
    X = df_eng[features].fillna(0).values
    
    if len(X) > 15 and not is_fitted:
        model.fit(X)
        is_fitted = True
    
    if len(X) > 0:
        preds = model.predict(X)
        scores = model.decision_function(X)
        df['anomaly'] = pd.Series(preds == -1, index=df_eng.index).reindex(df.index, fill_value=False)
        df['risk_score'] = np.abs(pd.Series(scores, index=df_eng.index)).reindex(df.index, fill_value=0).fillna(0)
    
    return df

async def send_telegram_alert(symbol, score):
    bot_token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if bot_token and chat_id:
        try:
            bot = telegram.Bot(token=bot_token)
            await bot.send_message(
                chat_id=chat_id,
                text=f"LIVE ALERT: {symbol}\nRisk: {score:.3f}\n{datetime.now(IST).strftime('%H:%M:%S IST')}"
            )
        except:
            pass

# REAL-TIME MONITORING (5s intervals)
def check_realtime_anomalies():
    while True:
        try:
            for symbol in SYMBOLS:
                df = detect(symbol, use_demo=False)
                high_risk = df[df['risk_score'] > 0.4]
                
                if not high_risk.empty:
                    score = high_risk['risk_score'].max()
                    now = time.time()
                    
                    if (symbol not in st.session_state.last_alert_time or 
                        now - st.session_state.last_alert_time.get(symbol, 0) > 60):
                        
                        st.session_state.last_alert_time[symbol] = now
                        threading.Thread(target=lambda: asyncio.run(send_telegram_alert(symbol, score))).start()
                        st.session_state.active_alerts[symbol] = {'score': score, 'time': datetime.now(IST)}
            
            time.sleep(5)
        except:
            time.sleep(5)

# Start monitoring
if 'monitoring_thread' not in st.session_state:
    st.session_state.monitoring_thread = threading.Thread(target=check_realtime_anomalies, daemon=True)
    st.session_state.monitoring_thread.start()

# UI
st.markdown("# Crypto Anomaly Detection Dashboard")
st.markdown("*Real-time monitoring | Indian Standard Time*")

st_autorefresh(interval=30 * 1000)

# Top Status
col1, col2 = st.columns([3, 1])
col1.metric("Status", f"Live - {datetime.now(IST).strftime('%H:%M:%S IST')}")
if col2.button("Force Refresh", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("## Configuration")
    mode = st.radio("Display Mode:", ["Live Market Data", "Demo Mode"])
    use_demo = mode == "Demo Mode"
    
    selected_symbols = st.multiselect("Symbols:", SYMBOLS, default=SYMBOLS)
    threshold = st.slider("Alert Threshold:", 0.0, 0.8, 0.4)
    
    st.markdown("---")
    st.info("Real-time monitoring: Active (5s intervals)")
    if st.button("Test Alerts"):
        for symbol in SYMBOLS[:2]:
            threading.Thread(target=lambda s=symbol: asyncio.run(send_telegram_alert(s, 0.75))).start()

# Live data
portfolio_risk = 0
results = {}
live_alerts_count = len(st.session_state.active_alerts)

for symbol in selected_symbols:
    df = detect(symbol, use_demo=use_demo)
    results[symbol] = df
    
    high_risk = df[df['risk_score'] > threshold]
    if not high_risk.empty:
        max_score = high_risk['risk_score'].max()
        portfolio_risk += max_score * 0.3
        
        st.markdown(f"""
        <div class="alert-banner">
            LIVE ALERT: {symbol} | Risk: {max_score:.3f} | {datetime.now(IST).strftime('%H:%M:%S IST')}
        </div>
        """, unsafe_allow_html=True)
    else:
        portfolio_risk += df['risk_score'].tail(10).mean() * 0.1

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Risk", f"{portfolio_risk:.3f}")
col2.metric("Live Anomalies", live_alerts_count)
col3.metric("Check Rate", "5 seconds")
col4.metric("Refresh", "30 seconds")

# Charts
st.markdown("## Real-time Analysis")
for symbol, df in results.items():
    st.markdown(f"### {symbol}")
    
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                       subplot_titles=(f"{symbol} Price", "Risk Score"),
                       vertical_spacing=0.08)
    
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'],
                                low=df['low'], close=df['close'],
                                increasing_line_color="#00ff88",
                                decreasing_line_color="#ff4444"), row=1, col=1)
    
    anomalies = df[df['anomaly']]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['high']*1.01,
                                mode='markers', marker=dict(color='#ff4444', size=14,
                                symbol='diamond', line=dict(width=2, color='white')),
                                hovertemplate='<b>ALERT</b><br>Risk: %{customdata:.3f}<extra></extra>',
                                customdata=anomalies['risk_score']), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['risk_score'], mode='lines',
                            line=dict(color='#ffaa00', width=3)), row=2, col=1)
    
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ff4444", row=2, col=1)
    
    fig.update_layout(height=520, showlegend=False, plot_bgcolor='#1a1a1a',
                     paper_bgcolor='#1a1a1a', font_color="#ffffff",
                     xaxis_rangeslider_visible=False)
    fig.update_xaxes(showgrid=True, gridcolor='#333', color="#00ff88")
    fig.update_yaxes(showgrid=True, gridcolor='#333', color="#00ff88")
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("")

st.markdown("---")
st.markdown("*5s anomaly detection | 30s visual refresh | © 2025*")
