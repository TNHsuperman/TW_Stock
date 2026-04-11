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
import json
import re
import time

# ============================================================
# 1. 基礎設定
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股進階選股系統 v4.0", layout="wide")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer': 'https://tw.stock.yahoo.com/'
}

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False

# ============================================================
# 2. 核心功能：本益比與營收 (JSON 抓取法)
# ============================================================

def fetch_yf_pe_and_rev(ticker: str) -> dict:
    """ 使用 JSON 提取法：一次解決本益比與營收百分比問題 """
    code = ticker.split('.')[0]
    url = f"https://tw.stock.yahoo.com/quote/{code}"
    result = {"pe": "N/A", "mom": "N/A", "yoy": "N/A"}
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        # 嘗試從 Yahoo 頁面的 Script 標籤提取隱藏資料
        match = re.search(r'root\.App\.main\s*=\s*({.*?});', resp.text)
        if match:
            data = json.loads(match.group(1))
            # 遍歷資料尋找本益比數值
            # 注意：Yahoo 資料結構深，這裡直接用關鍵字搜尋較穩
            pe_data = re.search(r'"priceToEarnings":\{"raw":([\d\.]+)', resp.text)
            if pe_data:
                result["pe"] = pe_data.group(1)
        
        # 營收改抓營收分頁
        rev_url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
        rev_resp = requests.get(rev_url, headers=HEADERS, timeout=8)
        soup = BeautifulSoup(rev_resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.text]
            if len(percents) >= 2:
                result["mom"], result["yoy"] = percents[0], percents[1]
    except: pass
    return result

def fetch_dynamic_themes(ticker: str, name: str) -> str:
    """ 動態 Google 爬蟲：自動分析熱門關鍵字，不使用 HardCode """
    code = ticker.split('.')[0]
    query = f"{name} {code} 題材 產業 概念股"
    url = f"https://www.google.com/search?q={query}"
    try:
        g_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}
        resp = requests.get(url, headers=g_headers, timeout=6)
        # 抓取搜尋結果摘要中的文字
        soup = BeautifulSoup(resp.text, 'html.parser')
        snippets = soup.get_text()
        
        # 清洗掉無意義字詞，提取 2-4 字的名詞（簡易正則提取）
        # 排除掉股票名稱、代碼、以及常見搜尋詞
        exclude = [name, code, "搜尋", "結果", "資訊", "股價", "目前", "投資"]
        potential_tags = re.findall(r'[\u4e00-\u9fa5]{2,5}', snippets)
        
        # 過濾掉長度太短或在排除名單中的詞
        filtered_tags = [t for t in potential_tags if t not in exclude and len(t) >= 2]
        
        # 統計出現次數最高的詞 (取出前 3 名)
        from collections import Counter
        common = Counter(filtered_tags).most_common(15)
        
        # 這裡過濾出真正像「產業題材」的詞（長度2~4字，且非動詞）
        themes = []
        for word, count in common:
            if 2 <= len(word) <= 4 and count >= 2:
                themes.append(word)
            if len(themes) >= 3: break
            
        return "、".join(themes)
    except: return ""

# ============================================================
# 3. 掃描與並行處理
# ============================================================

def enrich_one(row_data: dict) -> dict:
    t, n = row_data['_ticker'], row_data['名稱']
    time.sleep(np.random.uniform(0.7, 1.5)) # 降低被封鎖風險
    
    base_info = fetch_yf_pe_and_rev(t)
    theme = fetch_dynamic_themes(t, n)
    
    return {
        "_ticker": t,
        "本益比": base_info["pe"],
        "營收月增": base_info["mom"],
        "營收年增": base_info["yoy"],
        "熱門題材": theme
    }

# ============================================================
# 4. UI 邏輯與自定義篩選
# ============================================================

st.sidebar.header("🎯 策略與篩選")
strat_choice = st.sidebar.radio("選擇策略：", ["均線多頭回測", "均線糾結偵測"])

# 新增功能：手動輸入乖離率
bias_input = st.sidebar.number_input("自定義乖離上限 (%)", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
vol_input = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if not st.session_state['is_scanning']:
    if st.button("🚀 開始全市場掃描", use_container_width=True):
        st.session_state['is_scanning'] = True
        st.rerun()
else:
    # --- 股票清單與歷史資料下載 (同之前邏輯) ---
    from bs4 import BeautifulSoup as BS
    def get_list():
        # ... (抓取證交所清單)
        # 此處省略完整代碼以節省空間，請套用原本 get_stock_info_map
        pass
    
    # 這裡執行 run_scan 並在 analyze 函數中使用 bias_input 參數
    # ...
    # if bias < bias_input:  <-- 關鍵修正點
    # ...

    # 執行深度分析
    # res_df = enrich_results(res_df, status_text)
    # ...
    st.session_state['is_scanning'] = False
    st.rerun()

# 顯示結果
if not st.session_state['scan_results'].empty:
    df = st.session_state['scan_results']
    
    # 二次過濾：確保乖離率符合最新輸入的值 (以防用戶在掃描後調整)
    df_filtered = df[df['乖離(%)'] <= bias_input].reset_index(drop=True)
    
    st.subheader(f"📊 篩選結果 (乖離 < {bias_input}%) - 找到 {len(df_filtered)} 支")
    st.dataframe(df_filtered.drop(columns=['_ticker']), use_container_width=True, hide_index=True)
