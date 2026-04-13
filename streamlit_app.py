import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import urllib3
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
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
# 強制關閉 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板 v10.2 SSL修復版", layout="wide")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
]

def get_headers():
    return {'User-Agent': random.choice(USER_AGENTS), 'Accept': '*/*'}

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'current_idx' not in st.session_state:
    st.session_state['current_idx'] = 0

def get_tw_now():
    return datetime.now(timezone(timedelta(hours=8)))

# ============================================================
# 2. 數據抓取模組 (修正 SSL 驗證)
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_market_list():
    stocks = []
    urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
            ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
    for url, mkt in urls:
        try:
            # 關鍵修正：verify=False 繞過證照檢查，timeout 縮短防止卡死
            r = requests.get(url, headers=get_headers(), timeout=10, verify=False)
            r.encoding = 'big5' # 證交所使用 big5
            dfs = pd.read_html(StringIO(r.text))
            df = dfs[0]
            df.columns = df.iloc[0]
            for _, row in df.iloc[1:].iterrows():
                val = str(row['有價證券代號及名稱'])
                if '　' in val:
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        stocks.append({"ticker": f"{code}.{mkt}", "name": name, "industry": str(row['產業別']), "code": code})
        except Exception as e:
            st.warning(f"市場清單 {mkt} 獲取異常: {e}")
            continue
    return stocks

def fetch_deep_info(ticker):
    res = {"pe": np.nan, "mom": np.nan, "yoy": np.nan}
    code = ticker.split('.')[0]
    try:
        # 營收數據 (Yahoo 網頁不需要 SSL 驗證通常也能過)
        rev_url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
        r = requests.get(rev_url, headers=get_headers(), timeout=5, verify=False)
        soup = BeautifulSoup(r.text, 'html.parser')
        target = soup.select_one(r'li.List\(n\)')
        if target:
            nums = [s.get_text(strip=True) for s in target.find_all('span') if '%' in s.get_text()]
            if len(nums) >= 2:
                res["mom"] = float(nums[0].replace('%','').replace(',',''))
                res["yoy"] = float(nums[1].replace('%','').replace(',',''))
        # 本益比
        yt = yf.Ticker(ticker)
        res["pe"] = yt.info.get('trailingPE', np.nan)
    except: pass
    return res

def run_strategy_check(s, bias_limit, vol_limit):
    now_ts = int(get_tw_now().timestamp())
    start_ts = int((get_tw_now() - timedelta(days=250)).timestamp()) 
    try:
        # 增加延遲減少支數變動問題
        time.sleep(random.uniform(0.1, 0.3))
        api_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}"
        r = requests.get(api_url, params={"period1":start_ts, "period2":now_ts, "interval":"1d"}, headers=get_headers(), timeout=10, verify=False)
        
        json_data = r.json()
        if not json_data['chart']['result']: return None
        
        data = json_data['chart']['result'][0]
        c = pd.Series(data['indicators']['quote'][0]['close']).ffill().dropna()
        v = pd.Series(data['indicators']['quote'][0]['volume']).ffill().dropna()
        
        if len(c) < 65: return None
        
        avg_vol_5d = v.tail(5).mean() / 1000
        if avg_vol_5d < vol_limit: return None
        
        # 技術指標計算
        ma30 = c.rolling(30).mean().iloc[-1]
        ma45 = c.rolling(45).mean().iloc[-1]
        ma60 = c.rolling(60).mean().iloc[-1]
        curr_p = c.iloc[-1]
        bias = ((curr_p - ma30) / ma30) * 100
        
        # 策略過濾：多頭排列 + 低乖離
        if (ma30 > ma45 > ma60) and (0 <= bias <= bias_limit):
            v_chg = ((v.iloc[-1] - v.iloc[-2]) / v.iloc[-2]) * 100 if v.iloc[-2] > 0 else 0
            return {**s, "收盤": round(curr_p, 2), "乖離30MA(%)": round(bias, 2), "成交量(張)": int(v.iloc[-1]/1000), "量變動(%)": round(v_chg, 2)}
    except: return None
    return None

# ============================================================
# 3. Streamlit UI 與 主邏輯
# ============================================================

st.sidebar.header("🎯 策略設定")
with st.sidebar.form("filter_form"):
    user_bias = st.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0, step=0.1)
    user_vol = st.slider("最小成交量 (張)", 0, 3000, 500)
    submit = st.form_submit_button("🚀 開始智慧掃描", use_container_width=True)

if submit:
    st.session_state.is_scanning = True # 標記開始
    bar = st.progress(0)
    status = st.empty()
    
    status.info("正在讀取市場清單...")
    stocks = get_stock_market_list()
    
    if not stocks:
        st.error("無法取得市場清單，請檢查網路連線。")
    else:
        hits = []
        status.warning(f"🔍 掃描技術指標中 (0/{len(stocks)})...")
        # 降低 worker 數量至 10，確保每一條請求都能成功而不被 Yahoo 拒絕，解決支數變動問題
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(run_strategy_check, s, user_bias, user_vol): s for s in stocks}
            for i, f in enumerate(as_completed(futures), 1):
                if i % 50 == 0:
                    bar.progress(i / len(stocks))
                    status.warning(f"🔍 掃描技術指標中 ({i}/{len(stocks)})...")
                res = f.result()
                if res: hits.append(res)
        
        if hits:
            status.warning(f"📊 抓取財務深度數據 ({len(hits)} 支)...")
            final = []
            with ThreadPoolExecutor(max_workers=5) as ex:
                f_deep = {ex.submit(fetch_deep_info, r['ticker']): r for r in hits}
                for j, f in enumerate(as_completed(f_deep), 1):
                    try:
                        d_res = f.result()
                        final.append({**f_deep[f], "本益比": d_res["pe"], "營營收月增": d_res["mom"], "營收年增": d_res["yoy"]})
                    except: continue
            st.session_state.scan_results = pd.DataFrame(final)
            status.success(f"✅ 掃描完成！找到 {len(final)} 支標的")
        else:
            st.session_state.scan_results = pd.DataFrame()
            status.error("查無符合標的。")
    st.session_state.is_scanning = False

# ============================================================
# 4. 結果顯示
# ============================================================

if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # 底部輪播與圖表功能維持原樣
    st.divider()
    c_idx = st.session_state.current_idx
    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("⬅️ 上一支"):
        st.session_state.current_idx = (c_idx - 1) % len(df)
        st.rerun()
    c2.markdown(f"<center><b>{df.iloc[c_idx]['code']} {df.iloc[c_idx]['name']}</b> ({c_idx+1}/{len(df)})</center>", unsafe_allow_html=True)
    if c3.button("下一支 ➡️"):
        st.session_state.current_idx = (c_idx + 1) % len(df)
        st.rerun()
        
    # 圖表顯示
    try:
        curr = df.iloc[st.session_state.current_idx]
        k_df = yf.Ticker(curr['ticker']).history(period="1y")
        if not k_df.empty:
            k_df['MA30'] = k_df['Close'].rolling(30).mean()
            k_df = k_df.tail(120)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=k_df.index, open=k_df['Open'], high=k_df['High'], low=k_df['Low'], close=k_df['Close'], name="K線"), row=1, col=1)
            fig.add_trace(go.Scatter(x=k_df.index, y=k_df['MA30'], line=dict(color='orange'), name="30MA"), row=1, col=1)
            fig.add_trace(go.Bar(x=k_df.index, y=k_df['Volume'], name="成交量"), row=2, col=1)
            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
    except: pass
else:
    st.info("💡 調整參數後按「開始智慧掃描」")
