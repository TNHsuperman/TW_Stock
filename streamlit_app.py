import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import urllib3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import time
import random

# ============================================================
# 1. 基礎設定
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板 v5.0", layout="wide")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
]

def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Referer': 'https://tw.stock.yahoo.com/'
    }

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False

# ============================================================
# 2. 核心抓取功能
# ============================================================

def fetch_pe_ratio(ticker: str) -> str:
    """ 雙方案取本益比 """
    code = ticker.split('.')[0]
    # 方案 A: Google 快速提取 (極度穩定)
    try:
        g_url = f"https://www.google.com/search?q={code}+本益比"
        g_resp = requests.get(g_url, headers=get_headers(), timeout=5)
        match = re.search(r'本益比[:：]?\s*(\d+\.\d+)', g_resp.text)
        if match: return match.group(1)
    except: pass
    
    # 方案 B: Yahoo 頁面匹配
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}"
        resp = requests.get(url, headers=get_headers(), timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        label = soup.find('span', string=re.compile("本益比"))
        if label:
            val = label.find_next_sibling('span')
            if val: return val.get_text(strip=True)
    except: pass
    return "N/A"

def fetch_revenue_info(ticker: str) -> tuple[str, str]:
    code = ticker.split('.')[0]
    url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            spans = [s.text.strip() for s in row.find_all('span') if '%' in s.text]
            if len(spans) >= 2: return spans[0], spans[1]
    except: pass
    return "N/A", "N/A"

def fetch_dynamic_themes(ticker: str, name: str) -> str:
    """ 動態題材分析 (從新聞摘要) """
    code = ticker.split('.')[0]
    query = f"{name} {code} 題材"
    url = f"https://www.google.com/search?q={query}"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=5)
        text = resp.text
        # 常見產業關鍵字
        keywords = ["AI", "散熱", "半導體", "伺服器", "重電", "矽光子", "車用", "生技", "機器人"]
        found = [kw for kw in keywords if kw in text]
        # 額外抓取
        dynamic = re.findall(r'[\u4e00-\u9fa5]{2,4}(?:概念|題材)', text)
        found += [d.replace('概念','').replace('題材','') for d in dynamic]
        return "、".join(list(dict.fromkeys(found))[:3]) if found else "觀察中"
    except: return ""

@st.cache_data(ttl=86400)
def get_stock_list():
    data = []
    urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
            ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
    for url, market in urls:
        try:
            r = requests.get(url, headers=get_headers(), timeout=15, verify=False)
            df = pd.read_html(StringIO(r.text))[0]
            df.columns = df.iloc[0]
            for _, row in df.iloc[1:].iterrows():
                val = row['有價證券代號及名稱']
                if val and '　' in str(val):
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        data.append({"ticker": f"{code}.{market}", "name": name, "industry": row['產業別'], "market": market, "code": code})
        except: continue
    return data

def fetch_yf_history(ticker: str) -> pd.DataFrame:
    p2, p1 = int(time.time()), int((datetime.now() - timedelta(days=150)).timestamp())
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}", 
                         params={"period1":p1, "period2":p2, "interval":"1d"}, headers=get_headers(), timeout=10)
        json_data = r.json()['chart']['result'][0]
        df = pd.DataFrame({
            'Date': pd.to_datetime(json_data['timestamp'], unit='s').normalize(),
            'Close': json_data['indicators']['quote'][0]['close'],
            'Volume': json_data['indicators']['quote'][0]['volume']
        }).dropna().set_index('Date')
        return df
    except: return pd.DataFrame()

# ============================================================
# 3. UI 與 主邏輯
# ============================================================

st.sidebar.header("⚙️ 參數設定")
max_bias = st.sidebar.number_input("乖離率上限 (%)", 0.1, 10.0, 2.5)
min_vol = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if st.button("🚀 開始智慧掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning = True
    st.rerun()

if st.session_state.is_scanning:
    status = st.empty()
    bar = st.progress(0)
    
    status.text("📡 下載股票清單中...")
    stocks = get_stock_list()
    
    initial_results = [] # 確保初始化
    status.text(f"🔍 技術面篩選中 (共 {len(stocks)} 支)...")
    
    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(fetch_yf_history, s['ticker']): s for s in stocks}
        for i, f in enumerate(as_completed(futures), 1):
            if i % 100 == 0: bar.progress(i / len(stocks))
            s = futures[f]
            df = f.result()
            if len(df) > 60:
                if (df['Volume'].tail(5).mean() / 1000) >= min_vol:
                    c = df['Close'].iloc[-1]
                    ma30 = df['Close'].rolling(30).mean().iloc[-1]
                    bias = ((c - ma30) / ma30) * 100
                    if 0 < bias <= max_bias:
                        initial_results.append({**s, "收盤": round(c, 2), "乖離(%)": round(bias, 2)})

    if initial_results:
        status.text(f"📊 初步篩選出 {len(initial_results)} 支，深度分析基本面與題材...")
        final_data = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            enrich_futures = []
            for r in initial_results:
                def task(row):
                    pe = fetch_pe_ratio(row['ticker'])
                    mom, yoy = fetch_revenue_info(row['ticker'])
                    theme = fetch_dynamic_themes(row['ticker'], row['name'])
                    return {**row, "本益比": pe, "營收月增": mom, "營收年增": yoy, "熱門題材": theme}
                enrich_futures.append(ex.submit(task, r))
            
            for j, f in enumerate(as_completed(enrich_futures), 1):
                status.text(f"深度分析進度: {j} / {len(initial_results)}")
                final_data.append(f.result())
        
        st.session_state.scan_results = pd.DataFrame(final_data)
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("查無標的")

    st.session_state.is_scanning = False
    st.rerun()

# 顯示結果
if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    st.subheader(f"✅ 找到 {len(df)} 支標的")
    cols = ["code", "name", "market", "industry", "收盤", "乖離(%)", "本益比", "營收月增", "營收年增", "熱門題材"]
    st.dataframe(df[cols].rename(columns={"code":"代碼","name":"名稱"}), use_container_width=True, hide_index=True)
else:
    if not st.session_state.is_scanning:
        st.info("💡 請點擊按鈕開始掃描。")
