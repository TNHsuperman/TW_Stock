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
st.set_page_config(page_title="台股智慧選股監控 v6.0 - 最終版", layout="wide")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0'
]

def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7'
    }

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False

# ============================================================
# 2. 核心功能：本益比與題材判定 (Google 搜尋提取技術)
# ============================================================

def fetch_pe_and_theme(ticker: str, name: str) -> dict:
    """ 
    使用 Google 搜尋結果摘要技術：
    1. 抓取本益比數值
    2. 判定是否為熱門題材股 (是/否)
    """
    code = ticker.split('.')[0]
    # 策略：一次搜尋解決兩個問題
    query = f"{code} {name} 本益比 熱門題材 概念股"
    url = f"https://www.google.com/search?q={query}"
    
    res = {"pe": "N/A", "is_hot": "否"}
    
    try:
        resp = requests.get(url, headers=get_headers(), timeout=6)
        content = resp.text
        
        # --- 1. 本益比提取 (Regex 強力匹配) ---
        # 搜尋結果摘要常出現 "本益比: 15.3" 或 "P/E ratio 12.5"
        pe_match = re.search(r'本益比[:：]?\s*(\d+\.\d+)', content)
        if pe_match:
            res["pe"] = pe_match.group(1)
        else:
            # 備援：若摘要沒有，嘗試在內容中找數字
            pe_alt = re.search(r'(\d+\.\d+)\s*倍', content)
            if pe_alt: res["pe"] = pe_alt.group(1)

        # --- 2. 熱門題材判定 (是/否) ---
        # 邏輯：檢查搜尋結果中是否包含熱門關鍵字
        hot_words = ["飆股", "強勢股", "熱門題材", "領漲", "主力", "噴出", "題材股", "概念股", "多頭行情"]
        found_count = sum(1 for word in hot_words if word in content)
        
        # 若出現 2 個以上的熱門標籤，判定為「是」
        if found_count >= 2:
            res["is_hot"] = "是"
            
    except: pass
    return res

def fetch_revenue_percent(ticker: str) -> tuple[str, str]:
    """ 精準抓取營收年增與月增百分比 """
    code = ticker.split('.')[0]
    url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 抓取表格中最新月份的數據列
        row = soup.select_one(r'li.List\(n\)')
        if row:
            # 只取帶有 % 的數據，通常第一、二個就是月增與年增
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.get_text()]
            if len(percents) >= 2:
                return percents[0], percents[1]
    except: pass
    return "N/A", "N/A"

# ============================================================
# 3. 股票清單與技術掃描
# ============================================================

@st.cache_data(ttl=86400)
def get_clean_stock_list():
    data = []
    urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
            ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
    for url, market in urls:
        try:
            r = requests.get(url, headers=get_headers(), timeout=10, verify=False)
            df = pd.read_html(StringIO(r.text))[0]
            df.columns = df.iloc[0]
            for _, row in df.iloc[1:].iterrows():
                val = str(row['有價證券代號及名稱'])
                if '　' in val:
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        data.append({"ticker": f"{code}.{market}", "name": name, "industry": row['產業別'], "code": code})
        except: continue
    return data

def fetch_history_and_check(s, strat, bias_limit, vol_limit):
    """ 技術面初步過濾 """
    p2, p1 = int(time.time()), int((datetime.now() - timedelta(days=120)).timestamp())
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}", 
                         params={"period1":p1, "period2":p2, "interval":"1d"}, headers=get_headers(), timeout=10)
        json_data = r.json()['chart']['result'][0]
        close = json_data['indicators']['quote'][0]['close']
        vol = json_data['indicators']['quote'][0]['volume']
        
        df = pd.DataFrame({'Close': close, 'Volume': vol}).dropna()
        if len(df) < 60: return None
        
        # 成交量過濾 (張)
        avg_vol = df['Volume'].tail(5).mean() / 1000
        if avg_vol < vol_limit: return None
        
        c = df['Close'].iloc[-1]
        ma30 = df['Close'].rolling(30).mean().iloc[-1]
        bias = ((c - ma30) / ma30) * 100
        
        if 0 < bias <= bias_limit:
            return {**s, "收盤": round(c, 2), "乖離(%)": round(bias, 2)}
    except: return None
    return None

# ============================================================
# 4. 主邏輯與 UI
# ============================================================

st.sidebar.header("⚙️ 篩選控制台")
max_bias = st.sidebar.number_input("自定義乖離上限 (%)", 0.1, 10.0, 2.5, step=0.1)
min_vol = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if st.button("🚀 開始智慧掃描 (最後機會版)", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning = True
    st.rerun()

if st.session_state.is_scanning:
    status = st.empty()
    bar = st.progress(0)
    
    status.text("📡 抓取證交所最新清單...")
    stocks = get_clean_stock_list()
    
    # 技術面篩選
    status.text(f"🔍 掃描全市場技術面 (共 {len(stocks)} 支)...")
    initial_results = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(fetch_history_and_check, s, "多頭", max_bias, min_vol): s for s in stocks}
        for i, f in enumerate(as_completed(futures), 1):
            if i % 100 == 0: bar.progress(i / len(stocks))
            res = f.result()
            if res: initial_results.append(res)

    # 基本面與題材深度分析
    if initial_results:
        status.text(f"📊 找到 {len(initial_results)} 支符合條件，深度抓取本益比與題材...")
        final_data = []
        with ThreadPoolExecutor(max_workers=5) as ex: # 查詢題材不宜過快
            f_deep = []
            for r in initial_results:
                def deep_task(row):
                    # 抓取本益比與題材
                    pe_theme = fetch_pe_and_theme(row['ticker'], row['name'])
                    # 抓取營收
                    mom, yoy = fetch_revenue_percent(row['ticker'])
                    return {**row, "本益比": pe_theme["pe"], "熱門題材": pe_theme["is_hot"], "營收月增": mom, "營收年增": yoy}
                f_deep.append(ex.submit(deep_task, r))
            
            for j, f in enumerate(as_completed(f_deep), 1):
                status.text(f"深度分析進度: {j} / {len(initial_results)}")
                final_data.append(f.result())
        
        st.session_state.scan_results = pd.DataFrame(final_data)
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("查無標的。")

    st.session_state.is_scanning = False
    st.rerun()

# --- 結果顯示 ---
if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    st.success(f"✅ 掃描完成！找到 {len(df)} 支符合條件標的")
    
    # 格式化表格
    display_cols = ["code", "name", "industry", "收盤", "乖離(%)", "本益比", "營收月增", "營收年增", "熱門題材"]
    st.dataframe(df[display_cols].rename(columns={"code":"代碼","name":"名稱","industry":"類股"}), 
                 use_container_width=True, hide_index=True)
else:
    if not st.session_state.is_scanning:
        st.info("💡 準備就緒，請點擊按鈕。")
