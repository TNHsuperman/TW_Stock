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

st.set_page_config(page_title="台股趨勢選股", layout="centered")

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
indicator_choice = st.sidebar.selectbox("查看確認指標：", ["RSI (強弱指標)", "MACD (趨勢指標)"])

# 成交量濾網
min_volume = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)

# 趨勢過濾開關
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
        except: continue
        
        for ticker in batch:
            try:
                df = data[ticker] if len(batch) > 1 else data
                df = df.dropna(subset=['Close'])
                if len(df) < 60: continue
                
                # 成交量過濾
                avg_vol = df['Volume'].tail(5).mean() / 1000
                if avg_vol < min_volume: continue

                close = float(df['Close'].iloc[-1])
                # 計算 MA
                m30 = df['Close'].rolling(30).mean().iloc[-1]
