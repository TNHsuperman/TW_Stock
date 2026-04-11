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

# ============================================================
# 1. 基礎設定
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股監控 v3.0", layout="wide")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer': 'https://tw.stock.yahoo.com/'
}

# 預定義熱門題材關鍵字 (用於 Google 爬蟲分析)
THEME_KEYWORDS = [
    "AI", "伺服器", "散熱", "半導體", "台積電供應鏈", "低軌衛星", "重電", "儲能", "綠能", 
    "電動車", "CPO", "矽光子", "機器人", "軍工", "摺疊機", "Wi-Fi 7", "生技", "水資源"
]

# ============================================================
# 2. 核心抓取邏輯修正
# ============================================================

def fetch_yf_pe(ticker: str) -> str:
    """ 修正本益比：多重路徑解析 """
    code = ticker.split('.')[0]
    url = f"https://tw.stock.yahoo.com/quote/{code}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=7)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 1. 找包含「本益比」文字的 li
        items = soup.find_all('li', class_='Fxg(1)')
        for item in items:
            label = item.find('span', class_='C($c-primary-text)')
            if label and '本益比' in label.text:
                val = item.find_all('span')[-1] # 最後一個 span 通常是數值
                return val.text.strip()
        
        # 2. 備用：直接找特定 class
        pe_val = soup.select_one('span.Fw\(b\).Fz\(18px\)') # 有時會在這個位置
        if pe_val and '.' in pe_val.text:
            return pe_val.text.strip()
            
    except: pass
    return "N/A"

def fetch_revenue_growth(ticker: str) -> tuple[str, str]:
    """ 修正營收：抓取百分比而非數值 """
    code = ticker.split('.')[0]
    url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=7)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 抓取第一列數據
        row = soup.select_one('li.List\(n\)')
        if row:
            cells = row.find_all('span')
            # 根據 Yahoo 結構：[月份, 營收, 月增%, 年增%, ...]
            # 我們尋找帶有 % 符號的字串
            percents = [c.text.strip() for c in cells if '%' in c.text]
            if len(percents) >= 2:
                return percents[0], percents[1] # 月增, 年增
    except: pass
    return "N/A", "N/A"

def fetch_google_themes(ticker: str, name: str) -> str:
    """ 使用 Google 搜尋爬蟲判斷熱門題材 """
    code = ticker.split('.')[0]
    query = f"{name} {code} 題材 概念股"
    url = f"https://www.google.com/search?q={query}"
    
    try:
        # Google 需要特定的 User-Agent 避免被擋
        google_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        resp = requests.get(url, headers=google_headers, timeout=5)
        text = resp.text
        
        found_themes = []
        for kw in THEME_KEYWORDS:
            if kw in text:
                found_themes.append(kw)
        
        # 移除重複並回傳前三個最相關的
        if found_themes:
            return "、".join(list(dict.fromkeys(found_themes))[:3])
    except: pass
    return ""

# ============================================================
# 3. 資料處理與並行優化
# ============================================================

def enrich_one(row_data: dict) -> dict:
    ticker = row_data['_ticker']
    name = row_data['名稱']
    
    # 這裡加入一點點隨機延遲，防止被 Yahoo/Google 同時封鎖
    time.sleep(np.random.uniform(0.5, 1.5))
    
    pe = fetch_yf_pe(ticker)
    mom, yoy = fetch_revenue_growth(ticker)
    theme = fetch_google_themes(ticker, name)
    
    return {
        "_ticker": ticker,
        "本益比": pe,
        "營收月增": mom,
        "營收年增": yoy,
        "熱門題材": theme
    }

def enrich_results(res_df: pd.DataFrame, status_text) -> pd.DataFrame:
    total = len(res_df)
    rows_input = res_df.to_dict('records')
    enrich_map = {}
    
    # 降低並行數量 (max_workers=3)，因為 Google 爬蟲查太快會跳驗證碼
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(enrich_one, row): row['_ticker'] for row in rows_input}
        for i, future in enumerate(as_completed(futures), 1):
            status_text.text(f"🔍 深度分析中 ({i}/{total}): 正在查詢 {futures[future]}...")
            try:
                res = future.result()
                enrich_map[res['_ticker']] = res
            except: continue
    
    # 將結果填回原 DataFrame
    res_df['本益比'] = res_df['_ticker'].map(lambda t: enrich_map.get(t, {}).get('本益比', 'N/A'))
    res_df['營收月增'] = res_df['_ticker'].map(lambda t: enrich_map.get(t, {}).get('營收月增', 'N/A'))
    res_df['營收年增'] = res_df['_ticker'].map(lambda t: enrich_map.get(t, {}).get('營收年增', 'N/A'))
    res_df['熱門題材'] = res_df['_ticker'].map(lambda t: enrich_map.get(t, {}).get('熱門題材', ''))
    return res_df

# ============================================================
# 4. 歷史資料下載與掃描邏輯 (簡化版)
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_info_map():
    # ... (保持原本抓取 TWSE/TPEX 清單的邏輯)
    # 這裡省略，請套用你原本代碼中的 get_stock_info_map
    pass

def fetch_yf_history(ticker: str, days: int = 130) -> pd.DataFrame:
    # ... (保持原本抓取歷史資料的邏輯)
    pass

def run_scan(all_stocks, info_map, strategy, min_vol, progress_bar, status_text):
    # ... (原本的掃描邏輯)
    # 確保在完成掃描後呼叫 enrich_results
    pass

# ============================================================
# 5. 主程式入口
# ============================================================

# (其餘 Streamlit UI 邏輯與你的代碼保持一致)
# 只要確保在掃描結束後執行：
# if not res_df.empty:
#     res_df = enrich_results(res_df, status_text)
