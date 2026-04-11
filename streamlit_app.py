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
import yfinance as yf

# ============================================================
# 1. 基礎設定
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股 30/45/60MA 量價選股儀表板", layout="wide")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
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
# 2. 核心功能：深度資訊抓取 (PE/營收)
# ============================================================

def fetch_deep_info(ticker: str) -> dict:
    code = ticker.split('.')[0]
    res = {"pe": "N/A", "mom": "N/A", "yoy": "N/A"}
    
    # --- A. 本益比 (yfinance + Google 備援) ---
    try:
        yt = yf.Ticker(ticker)
        pe = yt.info.get('trailingPE')
        if pe: res["pe"] = f"{pe:.2f}"
    except: pass

    if res["pe"] == "N/A":
        try:
            query = f"{code} 本益比"
            g_url = f"https://www.google.com/search?q={query}"
            g_resp = requests.get(g_url, headers=get_headers(), timeout=5)
            pe_match = re.search(r'本益比[:：]?\s*(\d+\.\d+)', g_resp.text)
            if pe_match: res["pe"] = pe_match.group(1)
        except: pass

    # --- B. 營收百分比 ---
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
# 3. 技術分析掃描 (30/45/60 MA 策略 + 成交量變動)
# ============================================================

def run_strategy_check(s, bias_limit, vol_limit):
    """
    1. 均線多頭: 30MA > 45MA > 60MA
    2. 乖離限制: 收盤價與 30MA 乖離 <= bias_limit
    3. 成交量變動: 計算今日與昨日之百分比
    """
    p2, p1 = int(time.time()), int((datetime.now() - timedelta(days=150)).timestamp())
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}", 
                         params={"period1":p1, "period2":p2, "interval":"1d"}, headers=get_headers(), timeout=10)
        data = r.json()['chart']['result'][0]
        c_series = pd.Series(data['indicators']['quote'][0]['close'])
        v_series = pd.Series(data['indicators']['quote'][0]['volume'])
        
        if len(c_series) < 65: return None
        
        # 成交量計算
        vol_today = v_series.iloc[-1]
        vol_yesterday = v_series.iloc[-2]
        
        if vol_yesterday > 0:
            vol_change_pct = ((vol_today - vol_yesterday) / vol_yesterday) * 100
        else:
            vol_change_pct = 0.0
            
        curr_vol_int = int(vol_today / 1000) # 今日成交量 (張)
        avg_vol_5d = v_series.tail(5).mean() / 1000 # 5日均量
        
        # 門檻過濾 (以5日均量為準)
        if avg_vol_5d < vol_limit: return None
        
        ma30 = c_series.rolling(30).mean().iloc[-1]
        ma45 = c_series.rolling(45).mean().iloc[-1]
        ma60 = c_series.rolling(60).mean().iloc[-1]
        curr_price = c_series.iloc[-1]
        
        # 乖離率計算
        bias_30 = ((curr_price - ma30) / ma30) * 100
        
        if (ma30 > ma45 > ma60) and (0 < bias_30 <= bias_limit):
            return {
                **s, 
                "收盤": round(curr_price, 2), 
                "乖離30MA(%)": round(bias_30, 2),
                "成交量(張)": curr_vol_int,
                "量變動(%)": f"{vol_change_pct:+.1f}%"
            }
    except: return None
    return None

# ============================================================
# 4. Streamlit UI 與 主邏輯
# ============================================================

st.sidebar.header("🎯 策略設定")
user_bias = st.sidebar.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0, step=0.1)
user_vol = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if st.button("🚀 開始全市場智慧掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning = True
    st.rerun()

if st.session_state.is_scanning:
    status = st.empty()
    bar = st.progress(0)
    
    # 載入清單
    stocks_list = []
    try:
        urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
                ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
        for url, mkt in urls:
            r = requests.get(url, headers=get_headers(), timeout=10, verify=False)
            df = pd.read_html(StringIO(r.text))[0]
            df.columns = df.iloc[0]
            for _, row in df.iloc[1:].iterrows():
                val = str(row['有價證券代號及名稱'])
                if '　' in val:
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        stocks_list.append({"ticker": f"{code}.{mkt}", "name": name, "industry": row['產業別'], "code": code})
    except: pass

    # 第一階段：技術過濾
    initial_hits = []
    status.text(f"🔍 正在篩選符合 30/45/60MA 趨勢之標的...")
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(run_strategy_check, s, user_bias, user_vol): s for s in stocks_list}
        for i, f in enumerate(as_completed(futures), 1):
            if i % 100 == 0: bar.progress(i / len(stocks_list))
            res = f.result()
            if res: initial_hits.append(res)
            
    # 第二階段：基本面分析
    if initial_hits:
        status.text(f"📊 找到 {len(initial_hits)} 支標的，補充本益比與營收數據...")
        final_list = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            f_deep = {ex.submit(fetch_deep_info, r['ticker']): r for r in initial_hits}
            for j, f in enumerate(as_completed(f_deep), 1):
                status.text(f"數據分析進度: {j} / {len(initial_hits)}")
                deep_res = f.result()
                final_list.append({
                    **f_deep[f], 
                    "本益比": deep_res["pe"], 
                    "營收月增": deep_res["mom"], 
                    "營收年增": deep_res["yoy"]
                })
        
        st.session_state.scan_results = pd.DataFrame(final_list)
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("查無符合條件標的。")

    st.session_state.is_scanning = False
    st.rerun()

# --- 結果顯示 ---
if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    st.success(f"✅ 掃描完成！找到 {len(df)} 支標的")
    
    # 顯示欄位
    show_cols = ["code", "name", "收盤", "乖離30MA(%)", "成交量(張)", "量變動(%)", "本益比", "營收月增", "營收年增", "industry"]
    st.dataframe(df[show_cols].rename(columns={"code":"代碼","name":"名稱","industry":"類股"}), 
                 use_container_width=True, hide_index=True)
else:
    if not st.session_state.is_scanning:
        st.info("💡 點擊按鈕執行選股。")
