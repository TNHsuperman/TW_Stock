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
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
]

def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://tw.stock.yahoo.com/'
    }

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False

# ============================================================
# 2. 核心功能：本益比 (yfinance + 網頁爬取雙方案)
# ============================================================

def fetch_pe_ratio(ticker: str) -> str:
    """ 
    雙方案取本益比：
    1. 優先嘗試解析 Yahoo 奇摩股市頁面文字
    2. 備援嘗試 Google 搜尋快速結果
    """
    code = ticker.split('.')[0]
    # 方案 A: Yahoo 奇摩
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}"
        resp = requests.get(url, headers=get_headers(), timeout=7)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 尋找包含「本益比」字樣的元件
        for el in soup.find_all(['span', 'div', 'li']):
            if '本益比' in el.text and len(el.text) < 20:
                # 通常數值在下一個兄弟節點或子節點
                val = el.find_next_sibling() or el.parent.find_all(recursive=False)[-1]
                txt = val.get_text(strip=True)
                if re.match(r'^\d+\.?\d*$', txt):
                    return txt
    except: pass

    # 方案 B: Google Search 靜態抓取 (快速且難擋)
    try:
        g_url = f"https://www.google.com/search?q={code}+本益比"
        g_resp = requests.get(g_url, headers=get_headers(), timeout=5)
        # 搜尋結果中常見格式: "本益比：15.2"
        match = re.search(r'本益比[:：]?\s*(\d+\.\d+)', g_resp.text)
        if match: return match.group(1)
    except: pass
    
    return "N/A"

def fetch_revenue_info(ticker: str) -> tuple[str, str]:
    """ 修正營收：抓取百分比 """
    code = ticker.split('.')[0]
    url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=7)
        soup = BeautifulSoup(resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            spans = [s.text.strip() for s in row.find_all('span') if '%' in s.text]
            if len(spans) >= 2: return spans[0], spans[1]
    except: pass
    return "N/A", "N/A"

def fetch_dynamic_themes(ticker: str, name: str) -> str:
    """ 
    動態題材分析：
    抓取 Google News 標題並進行詞頻過濾
    """
    code = ticker.split('.')[0]
    query = f"{name} {code} 題材 概念股"
    url = f"https://www.google.com/search?q={query}&tbm=nws" # 強制搜尋新聞
    try:
        resp = requests.get(url, headers=get_headers(), timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        text = soup.get_text()
        
        # 產業關鍵字庫 (輔助過濾)
        keywords = ["AI", "散熱", "半導體", "伺服器", "綠能", "重電", "軍工", "低軌衛星", 
                    "矽光子", "車用", "生技", "機器人", "摺疊機", "面板", "封裝"]
        
        found = []
        for kw in keywords:
            if kw in text: found.append(kw)
            
        # 額外動態提取 2-4 字的名詞
        dynamic = re.findall(r'[\u4e00-\u9fa5]{2,4}(?:題材|概念|供應鏈)', text)
        for d in dynamic:
            found.append(d.replace('題材','').replace('概念','').replace('供應鏈',''))

        if found:
            return "、".join(list(dict.fromkeys(found))[:3])
    except: pass
    return "熱門題材觀察中"

# ============================================================
# 3. 掃描與並行處理邏輯
# ============================================================

def enrich_data_task(row: dict):
    """ 同時補充本益比、營收與題材 """
    t, n = row['_ticker'], row['名稱']
    time.sleep(random.uniform(0.5, 1.2)) # 避免被封鎖
    
    pe = fetch_pe_ratio(t)
    mom, yoy = fetch_revenue_info(t)
    theme = fetch_dynamic_themes(t, n)
    
    return {
        "_ticker": t,
        "本益比": pe,
        "營收月增": mom,
        "營收年增": yoy,
        "熱門題材": theme
    }

# ============================================================
# 4. Streamlit UI 與 主邏輯
# ============================================================

st.sidebar.header("🚀 策略篩選參數")
max_bias = st.sidebar.number_input("乖離率上限 (%)", 0.1, 10.0, 2.5)
min_vol = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if st.button("🔍 開始全市場智慧掃描", use_container_width=True):
    st.session_state.is_scanning = True
    st.rerun()

if st.session_state.is_scanning:
    # --- 步驟 1: 抓取股票清單 (此處假設已有 get_stock_list 函數) ---
    # ... (原有抓取清單邏輯)
    
    # --- 步驟 2: 技術面過濾 (ThreadPoolExecutor 15) ---
    # ... (原有掃描邏輯，將符合條件的存入 initial_results)
    
    # 範例測試數據 (initial_results)
    if initial_results:
        status = st.empty()
        final_results = []
        status.text(f"📊 初步篩選出 {len(initial_results)} 支，正在分析基本面...")
        
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(enrich_data_task, r) for r in initial_results]
            for i, f in enumerate(as_completed(futures), 1):
                status.text(f"分析進度: {i} / {len(initial_results)}")
                final_results.append(f.result())
        
        # 合併數據並存入 session_state
        res_df = pd.DataFrame(initial_results).merge(pd.DataFrame(final_results), on='_ticker')
        st.session_state.scan_results = res_df
        
    st.session_state.is_scanning = False
    st.rerun()

# --- 顯示結果 ---
if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    # 確保顯示順序與過濾
    display_cols = ["代碼", "名稱", "市場", "類股", "收盤", "乖離(%)", "本益比", "營收月增", "營收年增", "熱門題材"]
    st.success(f"✅ 掃描完成！符合條件標的：{len(df)} 支")
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
