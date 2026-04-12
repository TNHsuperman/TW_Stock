import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import random
import time
import yfinance as yf

# ============================================================
# 1. 基礎設定與環境初始化
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板 v7.1", layout="wide")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
]

def get_headers():
    return {'User-Agent': random.choice(USER_AGENTS)}

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False

# ============================================================
# 2. 數據抓取與快取邏輯
# ============================================================

@st.cache_data(ttl=86400)
def get_all_stock_list():
    """獲取全市場股票清單 (上市與上櫃)"""
    stocks = []
    try:
        urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
                ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
        for url, mkt in urls:
            r = requests.get(url, headers=get_headers(), timeout=10, verify=False)
            df_isin = pd.read_html(StringIO(r.text))[0]
            df_isin.columns = df_isin.iloc[0]
            for _, row in df_isin.iloc[1:].iterrows():
                val = str(row['有價證券代號及名稱'])
                if '　' in val:
                    code, name = val.split('　')
                    # 只抓取 4 位數的普通股
                    if len(code) == 4 and code.isdigit():
                        stocks.append({"ticker": f"{code}.{mkt}", "name": name, "industry": row['產業別'], "code": code})
    except Exception as e:
        st.error(f"獲取股票清單失敗: {e}")
    return stocks

def clean_percent(text):
    if not text or text == "N/A": return np.nan
    try:
        return float(text.replace('%', '').replace(',', ''))
    except:
        return np.nan

def fetch_deep_info(ticker: str) -> dict:
    code = ticker.split('.')[0]
    res = {"pe": np.nan, "mom": np.nan, "yoy": np.nan}
    try:
        yt = yf.Ticker(ticker)
        pe = yt.info.get('trailingPE')
        if pe: res["pe"] = float(pe)
    except: pass

    try:
        rev_url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
        rev_resp = requests.get(rev_url, headers=get_headers(), timeout=5)
        soup = BeautifulSoup(rev_resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.get_text()]
            if len(percents) >= 2:
                res["mom"] = clean_percent(percents[0])
                res["yoy"] = clean_percent(percents[1])
    except: pass
    return res

# ============================================================
# 3. 技術分析策略邏輯
# ============================================================

def run_strategy_check(s, bias_limit, vol_limit):
    # 設定回測區間 (200天確保均線計算足夠)
    p2, p1 = int(time.time()), int((datetime.now() - timedelta(days=200)).timestamp())
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}", 
                         params={"period1":p1, "period2":p2, "interval":"1d"}, headers=get_headers(), timeout=10)
        data = r.json()['chart']['result'][0]
        c_series = pd.Series(data['indicators']['quote'][0]['close']).ffill().dropna()
        v_series = pd.Series(data['indicators']['quote'][0]['volume']).ffill().dropna()
        
        if len(c_series) < 65: return None
        
        vol_today = v_series.iloc[-1]
        vol_yesterday = v_series.iloc[-2]
        vol_change = ((vol_today - vol_yesterday) / vol_yesterday) * 100 if vol_yesterday > 0 else 0
        curr_vol_int = int(vol_today / 1000) # 轉為張
        avg_vol_5d = v_series.tail(5).mean() / 1000
        
        # 篩選：成交量門檻
        if avg_vol_5d < vol_limit: return None
        
        # 計算均線
        ma30 = c_series.rolling(30).mean().iloc[-1]
        ma45 = c_series.rolling(45).mean().iloc[-1]
        ma60 = c_series.rolling(60).mean().iloc[-1]
        curr_price = c_series.iloc[-1]
        bias_30 = ((curr_price - ma30) / ma30) * 100
        
        # 條件：多頭排列且乖離在設定範圍內 (0~N%)
        if (ma30 > ma45 > ma60) and (0 <= bias_30 <= bias_limit):
            return {
                **s, 
                "收盤": round(curr_price, 2), 
                "乖離30MA(%)": round(bias_30, 2),
                "成交量(張)": curr_vol_int,
                "量變動(%)": round(vol_change, 2)
            }
    except: return None
    return None

# ============================================================
# 4. Streamlit UI 介面
# ============================================================

st.sidebar.header("🎯 策略設定")
user_bias = st.sidebar.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0, step=0.1)
user_vol = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

col1, col2 = st.columns([4, 1])
with col1:
    btn_scan = st.button("🚀 開始全市場智慧掃描", use_container_width=True, disabled=st.session_state.is_scanning)

if btn_scan:
    st.session_state.is_scanning = True
    st.rerun()

if st.session_state.is_scanning:
    status = st.empty()
    bar = st.progress(0)
    
    # 獲取清單 (使用快取)
    stocks_list = get_all_stock_list()
    
    initial_hits = []
    status.text(f"🔍 正在掃描全市場 {len(stocks_list)} 支標的...")
    
    # 第一階段：技術面篩選
    with ThreadPoolExecutor(max_workers=30) as ex:
        futures = {ex.submit(run_strategy_check, s, user_bias, user_vol): s for s in stocks_list}
        for i, f in enumerate(as_completed(futures), 1):
            if i % 100 == 0: bar.progress(i / len(stocks_list))
            res = f.result()
            if res: initial_hits.append(res)
            
    # 第二階段：基本面抓取
    if initial_hits:
        status.text(f"📊 正在抓取篩選出 {len(initial_hits)} 支標的的財報數據...")
        final_list = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            f_deep = {ex.submit(fetch_deep_info, r['ticker']): r for r in initial_hits}
            for j, f in enumerate(as_completed(f_deep), 1):
                status.text(f"財報進度: {j} / {len(initial_hits)}")
                deep_res = f.result()
                final_list.append({**f_deep[f], "本益比": deep_res["pe"], "營收月增": deep_res["mom"], "營收年增": deep_res["yoy"]})
        st.session_state.scan_results = pd.DataFrame(final_list)
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("查無符合條件之標的。")

    st.session_state.is_scanning = False
    st.rerun()

# ============================================================
# 5. 結果顯示與下載按鈕
# ============================================================

if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results.copy()
    
    # UI 頂部控制列
    st.divider()
    res_col1, res_col2 = st.columns([3, 1])
    with res_col1:
        st.success(f"✅ 掃描完成！找到 {len(df)} 支技術面轉強標的")
    with res_col2:
        # 下載 CSV 功能
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下載選股結果 (CSV)",
            data=csv,
            file_name=f'TW_Stock_Scan_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            use_container_width=True
        )

    # 欄位重新命名與格式化顯示
    show_cols = ["code", "name", "收盤", "乖離30MA(%)", "成交量(張)", "量變動(%)", "本益比", "營收月增", "營收年增", "industry"]
    df_display = df[show_cols].rename(columns={"code":"代碼","name":"名稱","industry":"類股"})

    # 台灣股市配色習慣
    def color_tw_style(val):
        if pd.isna(val): return ''
        color = '#ef5350' if val > 0 else '#26a69a' if val < 0 else 'white'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df_display.style.map(color_tw_style, subset=['量變動(%)', '營收月增', '營收年增']),
        use_container_width=True,
        hide_index=True,
        column_config={
            "代碼": st.column_config.TextColumn("代碼"),
            "名稱": st.column_config.TextColumn("名稱"),
            "收盤": st.column_config.NumberColumn("價格", format="%.2f"),
            "乖離30MA(%)": st.column_config.ProgressColumn(
                "30MA 乖離",
                help=f"進度條越短代表越貼近 30MA 支撐。上限為 {user_bias}%",
                format="%.2f%%",
                min_value=0,
                max_value=user_bias,
            ),
            "量變動(%)": st.column_config.NumberColumn("量變動", format="%.1f%%"),
            "營收月增": st.column_config.NumberColumn("營收月增", format="%.1f%%"),
            "營收年增": st.column_config.NumberColumn("營收年增", format="%.1f%%"),
            "本益比": st.column_config.NumberColumn("PE", format="%.1f"),
            "成交量(張)": st.column_config.NumberColumn("成交量", format="%d 📦"),
            "類股": st.column_config.TextColumn("產業別")
        }
    )
    
    st.caption(f"💡 數據更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
else:
    if not st.session_state.is_scanning:
        st.info("💡 請調整左側參數（乖離率與成交量）後，點擊「開始智慧掃描」按鈕。")
