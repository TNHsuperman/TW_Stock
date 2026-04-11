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
st.set_page_config(page_title="台股均線策略選股儀表板", layout="wide")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
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
# 2. 核心功能：資料抓取 (本益比、題材、營收)
# ============================================================

def fetch_deep_info(ticker: str, name: str) -> dict:
    """ 
    使用 Google 搜尋與 Yahoo 營收頁面進行深度資料抓取
    """
    code = ticker.split('.')[0]
    res = {"pe": "N/A", "is_hot": "否", "mom": "N/A", "yoy": "N/A"}
    
    # --- A. 本益比與熱門題材 (Google 搜尋提取) ---
    try:
        g_url = f"https://www.google.com/search?q={code}+{name}+本益比+熱門題材"
        g_resp = requests.get(g_url, headers=get_headers(), timeout=7)
        content = g_resp.text
        
        # 本益比提取
        pe_match = re.search(r'本益比[:：]?\s*(\d+\.\d+)', content)
        if pe_match:
            res["pe"] = pe_match.group(1)
            
        # 熱門題材判定 (是/否)
        hot_keywords = ["飆股", "領漲", "強勢", "題材", "噴出", "熱門", "多頭"]
        count = sum(1 for kw in hot_keywords if kw in content)
        if count >= 2:
            res["is_hot"] = "是"
    except: pass

    # --- B. 營收百分比 (Yahoo 營收頁面) ---
    try:
        rev_url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
        rev_resp = requests.get(rev_url, headers=get_headers(), timeout=5)
        soup = BeautifulSoup(rev_resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.get_text()]
            if len(percents) >= 2:
                res["mom"], res["yoy"] = percents[0], percents[1]
    except: pass
    
    return res

# ============================================================
# 3. 技術分析與掃描
# ============================================================

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
                val = str(row['有價證券代號及名稱'])
                if '　' in val:
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        data.append({"ticker": f"{code}.{market}", "name": name, "industry": row['產業別'], "code": code})
        except: continue
    return data

def run_strategy_check(s, bias_limit, vol_limit):
    """ 
    嚴格執行策略：
    1. 均線多頭: 30MA > 45MA > 60MA
    2. 乖離限制: 收盤價與 15MA 乖離 <= bias_limit
    """
    p2, p1 = int(time.time()), int((datetime.now() - timedelta(days=150)).timestamp())
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}", 
                         params={"period1":p1, "period2":p2, "interval":"1d"}, headers=get_headers(), timeout=10)
        json_res = r.json()['chart']['result'][0]
        close = pd.Series(json_res['indicators']['quote'][0]['close'])
        volume = pd.Series(json_res['indicators']['quote'][0]['volume'])
        
        if len(close) < 60: return None
        
        # 成交量過濾
        if (volume.tail(5).mean() / 1000) < vol_limit: return None
        
        ma15 = close.rolling(15).mean().iloc[-1]
        ma30 = close.rolling(30).mean().iloc[-1]
        ma45 = close.rolling(45).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        curr_price = close.iloc[-1]
        
        # 乖離率計算 (收盤價 vs 15MA)
        bias_15 = ((curr_price - ma15) / ma15) * 100
        
        # 條件判斷
        is_bull_trend = ma30 > ma45 > ma60
        is_within_bias = 0 < bias_15 <= bias_limit
        
        if is_bull_trend and is_within_bias:
            return {**s, "收盤": round(curr_price, 2), "乖離15MA(%)": round(bias_15, 2)}
    except: return None
    return None

# ============================================================
# 4. 主 UI 介面
# ============================================================

st.sidebar.header("⚙️ 策略參數")
user_bias = st.sidebar.number_input("15MA 乖離上限 (%)", 0.1, 10.0, 2.5, step=0.1)
user_vol = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if st.button("🚀 開始多頭均線選股 (30/45/60 MA)", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning = True
    st.rerun()

if st.session_state.is_scanning:
    status = st.empty()
    bar = st.progress(0)
    
    status.text("📡 獲取股票清單...")
    all_stocks = get_stock_list()
    
    # 第一階段：技術面掃描
    initial_hits = []
    status.text(f"🔍 掃描均線趨勢與乖離 (共 {len(all_stocks)} 支)...")
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(run_strategy_check, s, user_bias, user_vol): s for s in all_stocks}
        for i, f in enumerate(as_completed(futures), 1):
            if i % 100 == 0: bar.progress(i / len(all_stocks))
            res = f.result()
            if res: initial_hits.append(res)
            
    # 第二階段：深度資訊補完
    if initial_hits:
        status.text(f"📊 找到 {len(initial_hits)} 支標的，分析基本面與熱門題材...")
        final_list = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            f_deep = {ex.submit(fetch_deep_info, r['ticker'], r['name']): r for r in initial_hits}
            for j, f in enumerate(as_completed(f_deep), 1):
                status.text(f"深度分析進度: {j} / {len(initial_hits)}")
                deep_info = f.result()
                final_list.append({**f_deep[f], "本益比": deep_info["pe"], "熱門題材": deep_info["is_hot"], "營收月增": deep_info["mom"], "營收年增": deep_info["yoy"]})
        
        st.session_state.scan_results = pd.DataFrame(final_list)
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("無符合多頭回測條件之標的。")

    st.session_state.is_scanning = False
    st.rerun()

# --- 結果呈現 ---
if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    st.subheader(f"✅ 掃描完成！符合條件：{len(df)} 支")
    
    # 欄位排序
    show_cols = ["code", "name", "收盤", "乖離15MA(%)", "本益比", "營收月增", "營收年增", "熱門題材", "industry"]
    st.dataframe(df[show_cols].rename(columns={"code":"代碼","name":"名稱","industry":"類股"}), 
                 use_container_width=True, hide_index=True)
else:
    if not st.session_state.is_scanning:
        st.info("💡 點擊上方按鈕開始掃描 (均線多頭且 15MA 乖離過濾)。")
