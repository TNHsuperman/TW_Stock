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
import time
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# 1. 基礎設定與環境初始化
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板 v12.0 - 官方 API 版", layout="wide")

def get_headers():
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'X-Requested-With': 'XMLHttpRequest'
    }

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False
if 'current_idx' not in st.session_state:
    st.session_state['current_idx'] = 0

def get_tw_now():
    return datetime.now(timezone(timedelta(hours=8)))

# ============================================================
# 2. 核心數據抓取 (證交所官方 API 整合)
# ============================================================

@st.cache_data(ttl=3600)
def fetch_official_pe_data():
    """
    從證交所與櫃買中心抓取當日所有股票的本益比清單
    這是最穩定且不會漏掉數據的方法
    """
    pe_map = {}
    today_str = get_tw_now().strftime("%Y%m%d")
    
    # --- A. 抓取上市股票本益比 (TWSE) ---
    try:
        twse_url = f"https://www.twse.com.tw/exchangeReport/BWIBBU_d?response=json&date={today_str}&selectType=ALL"
        r = requests.get(twse_url, headers=get_headers(), timeout=15)
        data = r.json()
        if data.get('data'):
            for row in data['data']:
                # row[0]=代碼, row[4]=本益比
                code = row[0]
                try: pe_map[code] = float(row[4])
                except: pe_map[code] = np.nan
    except: pass

    # --- B. 抓取上櫃股票本益比 (TPEx) ---
    try:
        # 櫃買中心 API 格式略有不同
        tpex_url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php?l=zh-tw&d={get_tw_now().year-1911}/{get_tw_now().strftime('%m/%d')}"
        r = requests.get(tpex_url, headers=get_headers(), timeout=15)
        data = r.json()
        if data.get('aaData'):
            for row in data['aaData']:
                # row[0]=代碼, row[15]=本益比
                code = row[0]
                try: pe_map[code] = float(row[15])
                except: pe_map[code] = np.nan
    except: pass
    
    return pe_map

@st.cache_data(ttl=86400)
def get_stock_market_list():
    stocks = []
    try:
        urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
                ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
        for url, mkt in urls:
            r = requests.get(url, timeout=15, verify=False)
            df_isin = pd.read_html(StringIO(r.text))[0]
            df_isin.columns = df_isin.iloc[0]
            for _, row in df_isin.iloc[1:].iterrows():
                val = str(row['有價證券代號及名稱'])
                if '　' in val:
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        stocks.append({"ticker": f"{code}.{mkt}", "name": name, "industry": row['產業別'], "code": code})
    except: pass
    return stocks

def fetch_revenue_info(code: str) -> dict:
    """僅抓取營收，本益比改由官方 API 全域匹配"""
    res = {"mom": np.nan, "yoy": np.nan}
    try:
        rev_url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
        rev_resp = requests.get(rev_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(rev_resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.get_text()]
            if len(percents) >= 2:
                res["mom"] = float(percents[0].replace('%', '').replace(',', ''))
                res["yoy"] = float(percents[1].replace('%', '').replace(',', ''))
    except: pass
    return res

# ============================================================
# 3. 技術分析與掃描邏輯
# ============================================================

def run_strategy_check(s, bias_limit, vol_limit):
    now_ts = int(get_tw_now().timestamp())
    start_ts = int((get_tw_now() - timedelta(days=200)).timestamp()) 
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}", 
                         params={"period1":start_ts, "period2":now_ts, "interval":"1d"}, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        data = r.json()['chart']['result'][0]
        c_series = pd.Series(data['indicators']['quote'][0]['close']).ffill().dropna()
        v_series = pd.Series(data['indicators']['quote'][0]['volume']).ffill().dropna()
        if len(c_series) < 65: return None
        
        curr_price = c_series.iloc[-1]
        ma30 = c_series.rolling(30).mean().iloc[-1]
        ma45 = c_series.rolling(45).mean().iloc[-1]
        ma60 = c_series.rolling(60).mean().iloc[-1]
        bias_30 = ((curr_price - ma30) / ma30) * 100
        
        avg_vol_5d = v_series.tail(5).mean() / 1000
        if avg_vol_5d < vol_limit: return None

        if (ma30 > ma45 > ma60) and (0 <= bias_30 <= bias_limit):
            vol_change = ((v_series.iloc[-1] - v_series.iloc[-2]) / v_series.iloc[-2]) * 100 if v_series.iloc[-2] > 0 else 0
            return {**s, "收盤": round(curr_price, 2), "乖離30MA(%)": round(bias_30, 2), "成交量(張)": int(v_series.iloc[-1]/1000), "量變動(%)": round(vol_change, 2)}
    except: return None
    return None

# ============================================================
# 4. Streamlit UI
# ============================================================

st.sidebar.header("🎯 策略設定")
user_bias = st.sidebar.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0, step=0.1)
user_vol = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if st.button("🚀 開始官方資料整合掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning = True
    st.rerun()

if st.session_state.is_scanning:
    with st.status("📡 正在串接官方 API...", expanded=True) as status:
        # 第一步：獲取全市場本益比清單 (一次拿完所有股票 PE)
        status.write("正在獲取證交所/櫃買中心全市場本益比清單...")
        pe_master_map = fetch_official_pe_data()
        
        # 第二步：掃描符合技術面條件的股票
        status.write("正在篩選技術面符合標的...")
        stocks_list = get_stock_market_list()
        initial_hits = []
        with ThreadPoolExecutor(max_workers=30) as ex:
            futures = {ex.submit(run_strategy_check, s, user_bias, user_vol): s for s in stocks_list}
            for f in as_completed(futures):
                res = f.result()
                if res: initial_hits.append(res)
        
        # 第三步：補完營收與 PE
        if initial_hits:
            status.write(f"正在匹配數據 (共 {len(initial_hits)} 支)...")
            final_list = []
            with ThreadPoolExecutor(max_workers=10) as ex:
                f_rev = {ex.submit(fetch_revenue_info, r['code']): r for r in initial_hits}
                for f in as_completed(f_rev):
                    orig_data = f_rev[f]
                    rev_res = f.result()
                    # 從官方 map 中直接讀取 PE，抓不到則為 0
                    pe_val = pe_master_map.get(orig_data['code'], 0.0)
                    final_list.append({**orig_data, "本益比": pe_val, "營收月增": rev_res["mom"], "營收年增": rev_res["yoy"]})
            st.session_state.scan_results = pd.DataFrame(final_list)
        else:
            st.session_state.scan_results = pd.DataFrame()
            st.warning("查無符合條件標的。")
            
    st.session_state.is_scanning = False
    st.rerun()

# ============================================================
# 5. 結果顯示
# ============================================================

if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results.copy()
    st.dataframe(
        df[["code", "name", "收盤", "乖離30MA(%)", "成交量(張)", "量變動(%)", "本益比", "營收月增", "營收年增", "industry"]],
        use_container_width=True, hide_index=True,
        column_config={
            "本益比": st.column_config.NumberColumn("PE (證交所)", format="%.2f"),
            "乖離30MA(%)": st.column_config.NumberColumn("乖離%", format="%.2f%%"),
            "成交量(張)": st.column_config.NumberColumn("量(張)", format="%d"),
        }
    )
    st.caption("💡 數據來源說明：本益比由證交所/櫃買中心 API 即時提供；技術面與營收由 Yahoo 數據整合。")
