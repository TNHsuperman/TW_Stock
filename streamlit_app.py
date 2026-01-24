import sys
import warnings
import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import urllib3
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 系統初始化與安全設定 ---
if 'warnings' not in sys.modules:
    sys.modules['warnings'] = warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="台股智慧選股", layout="centered")

# --- 2. 資料抓取與緩存 ---
@st.cache_data(ttl=86400)
def get_stock_info_map():
    stock_info_map = {}
    stocks_list = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    urls = [
        ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', ".TW"),
        ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', ".TWO")
    ]
    for url, suffix in urls:
        try:
            response = requests.get(url, headers=headers, timeout=15, verify=False)
            df_list = pd.read_html(StringIO(response.text), flavor='lxml')
            df = df_list[0]
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            for index, row in df.iterrows():
                val = row['有價證券代號及名稱']
                industry = row['產業別']
                if val and '　' in str(val):
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        ticker = f"{code}{suffix}"
                        stocks_list.append(ticker)
                        stock_info_map[ticker] = {"name": name, "industry": industry if pd.notna(industry) else "其他"}
        except: continue
    return stocks_list, stock_info_map

# --- 3. UI 策略與指標設定 ---
st.sidebar.header("⚙️ 策略參數")
strategy_option = st.sidebar.radio("選擇選股策略：", ("均線多頭回測", "均線糾結偵測"))
indicator_choice = st.sidebar.selectbox("查看確認指標：", ["都不顯示", "RSI (強弱指標)", "MACD (趨勢指標)"])

min_volume = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)
use_filter = st.sidebar.checkbox("僅顯示轉強標的 (RSI > 50 或 MACD 柱狀體 > 0)")

# --- 4. 核心掃描邏輯 ---
if st.button(f"🔍 開始全市場掃描", use_container_width=True):
    all_stocks, info_map = get_stock_info_map()
    results = []
    progress_bar = st.progress(0)
    
    batch_size = 100
    for i in range(0, len(all_stocks), batch_size):
        batch = all_stocks[i:i + batch_size]
        try:
            data = yf.download(batch, period="150d", group_by='ticker', progress=False)
            for ticker in batch:
                try:
                    df = data[ticker] if len(batch) > 1 else data
                    df = df.dropna(subset=['Close'])
                    if len(df) < 60: continue
                    
                    avg_vol = df['Volume'].tail(5).mean() / 1000
                    if avg_vol < min_volume: continue

                    close = float(df['Close'].iloc[-1])
                    m30, m45, m60 = df['Close'].rolling(30).mean().iloc[-1], df['Close'].rolling(45).mean().iloc[-1], df['Close'].rolling(60).mean().iloc[-1]
                    
                    delta = df['Close'].diff()
                    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rsi = (100 - (100 / (1 + gain/loss))).iloc[-1]

                    exp1, exp2 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
                    macd_v, signal_v = exp1 - exp2, (exp1 - exp2).ewm(span=9, adjust=False).mean()
                    hist_v = (macd_v - signal_v).iloc[-1]
                    
                    keep = False
                    if strategy_option == "均線多頭回測":
                        if m30 > m45 > m60 and close > m30 and (close - m30) / m30 <= 0.02: keep = True
                    elif strategy_option == "均線糾結偵測":
                        ma_spread = (max(m30, m45, m60) - min(m30, m45, m60)) / min(m30, m45, m60)
                        if ma_spread <= 0.015 and abs(close - m30) / m30 <= 0.02: keep = True
                    
                    if keep and use_filter:
                        if indicator_choice == "RSI (強弱指標)" and rsi < 50: keep = False
                        if indicator_choice == "MACD (趨勢指標)" and hist_v < 0: keep = False

                    if keep:
                        stock_data = info_map.get(ticker, {"name": "未知", "industry": "其他"})
                        results.append({"ID": ticker, "代碼": ticker.split('.')[0], "名稱": stock_data["name"], "類股": stock_data["industry"], "收盤": round(close, 2)})
                except: continue
        except: continue
        progress_bar.progress(min((i + batch_size) / len(all_stocks), 1.0))
    st.session_state['scan_results'] = pd.DataFrame(results)
    st.session_state['selected_index'] = 0

# --- 5. 顯示與同步邏輯 ---
if 'scan_results' in st.session_state and not st.session_state['scan_results'].empty:
    df_raw = st.session_state['scan_results']
    selected_industry = st.selectbox("🎯 篩選類股：", ["全部"] + sorted(df_raw["類股"].unique().tolist()))
    df_filtered = df_raw if selected_industry == "全部" else df_raw[df_raw["
