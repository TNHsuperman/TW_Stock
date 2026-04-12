import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
import random
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================
# 1. 基礎設定與環境初始化
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股 v9.9.3", layout="wide")

@st.cache_resource
def get_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50))
    return s

session = get_session()

# 初始化 Session State
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# ============================================================
# 2. 核心抓取函數 (優化計算速度)
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_list():
    stocks = []
    try:
        for mode in ["2", "4"]:
            url = f'https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}'
            r = session.get(url, timeout=10)
            df = pd.read_html(StringIO(r.text))[0]
            for val, ind in zip(df.iloc[1:, 0], df.iloc[1:, 4]):
                if '　' in str(val):
                    c, n = val.split('　')
                    if len(c) == 4 and c.isdigit():
                        stocks.append({"ticker": f"{c}.{'TW' if mode=='2' else 'TWO'}", "name": n, "industry": ind, "code": c})
    except: pass
    return stocks

def run_check(s, bias_limit, vol_limit):
    try:
        # 提速：直接調用快取 API
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}?range=1y&interval=1d"
        r = session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        res = r.json()['chart']['result'][0]
        c = pd.Series(res['indicators']['quote'][0]['close']).ffill().values
        v = pd.Series(res['indicators']['quote'][0]['volume']).ffill().values
        
        if len(c) < 65: return None
        
        # 轉用 Numpy 運算提高效能
        ma30, ma45, ma60 = np.mean(c[-30:]), np.mean(c[-45:]), np.mean(c[-60:])
        curr_p = c[-1]
        bias = ((curr_p - ma30) / ma30) * 100
        avg_v5 = np.mean(v[-5:]) / 1000

        if (ma30 > ma45 > ma60) and (0 <= bias <= bias_limit) and (avg_v5 >= vol_limit):
            v_change = ((v[-1] - v[-2]) / v[-2]) * 100 if v[-2] > 0 else 0
            return {**s, "收盤": round(curr_p, 2), "乖離30MA(%)": round(bias, 2), 
                    "成交量(張)": int(v[-1]/1000), "量變動(%)": round(v_change, 2)}
    except: return None

def fetch_extra(ticker):
    res = {"pe": np.nan, "mom": np.nan, "yoy": np.nan}
    try:
        yt = yf.Ticker(ticker)
        res["pe"] = yt.info.get('trailingPE', np.nan)
        code = ticker.split('.')[0]
        resp = session.get(f"https://tw.stock.yahoo.com/quote/{code}/revenue", timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        spans = [s.text.replace('%','') for s in soup.select('span') if '%' in s.text]
        if len(spans) >= 2:
            res["mom"], res["yoy"] = float(spans[0]), float(spans[1])
    except: pass
    return res

# ============================================================
# 3. UI 邏輯 (修復沒反應的問題)
# ============================================================

st.sidebar.header("🎯 策略設定")
bias_input = st.sidebar.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0)
vol_input = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

# 使用狀態變數控制按鈕
if st.sidebar.button("🚀 開始掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning = True
    st.rerun()

if st.session_state.is_scanning:
    # 使用 st.status 確保前端能看到後台在動
    with st.status("🔍 正在執行全市場掃描...", expanded=True) as status:
        stocks = get_stock_list()
        hits = []
        
        # 線程降至 25 以確保 Streamlit 穩定
        with ThreadPoolExecutor(max_workers=25) as ex:
            future_to_stock = {ex.submit(run_check, s, bias_input, vol_input): s for s in stocks}
            for i, future in enumerate(as_completed(future_to_stock), 1):
                res = future.result()
                if res: hits.append(res)
                if i % 100 == 0:
                    status.update(label=f"🔍 已掃描 {i} / {len(stocks)} 檔標的...")
        
        if hits:
            status.update(label=f"📊 命中 {len(hits)} 檔，抓取深度財務數據...")
            final = []
            with ThreadPoolExecutor(max_workers=10) as ex:
                future_to_hit = {ex.submit(fetch_extra, h['ticker']): h for h in hits}
                for future in as_completed(future_to_hit):
                    extra = future.result()
                    final.append({**future_to_hit[future], **extra})
            st.session_state.scan_results = pd.DataFrame(final)
        else:
            st.session_state.scan_results = pd.DataFrame()
            st.warning("查無標的。")
        
        status.update(label="✅ 掃描完成！", state="complete", expanded=False)
        st.session_state.is_scanning = False
        st.rerun()

# ============================================================
# 4. 結果顯示 (K線圖部分略)
# ============================================================

if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    st.success(f"找到 {len(df)} 支符合條件股票")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # 底部切換邏輯
    st.divider()
    idx = st.session_state.current_idx % len(df)
    c1, c2, c3 = st.columns([1,2,1])
    if c1.button("⬅️ 上一支"): st.session_state.current_idx -= 1; st.rerun()
    c2.markdown(f"<center><h3>{df.iloc[idx]['code']} {df.iloc[idx]['name']}</h3></center>", unsafe_allow_html=True)
    if c3.button("下一支 ➡️"): st.session_state.current_idx += 1; st.rerun()
    
    # 這裡可以放 draw_k_line 繪圖邏輯...
else:
    if not st.session_state.is_scanning:
        st.info("請點擊左側「開始掃描」按鈕。")
