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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================
# 1. 基礎設定與環境初始化
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板 v9.9 (急速版)", layout="wide")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
]

def get_robust_session():
    session = requests.Session()
    # 稍微降低重試次數以求速度，但保留關鍵重試
    retry = Retry(
        total=2, 
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    # 增加 Pool Size 以應付 50 個執行緒
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

session = get_robust_session()

def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': '*/*',
        'Connection': 'keep-alive'
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
# 2. 數據清洗與深度資訊抓取
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_market_list():
    stocks = []
    try:
        urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
                ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
        for url, mkt in urls:
            r = session.get(url, headers=get_headers(), timeout=10)
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

def fetch_deep_info(ticker: str) -> dict:
    code = ticker.split('.')[0]
    res = {"pe": np.nan, "mom": np.nan, "yoy": np.nan}
    try:
        # 深度資訊抓取維持適中速度，避免被封鎖
        yt = yf.Ticker(ticker)
        res["pe"] = yt.info.get('trailingPE', np.nan)
    except: pass
    try:
        rev_url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
        rev_resp = session.get(rev_url, headers=get_headers(), timeout=5)
        soup = BeautifulSoup(rev_resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            percents = [s.get_text(strip=True).replace('%','') for s in row.find_all('span') if '%' in s.get_text()]
            if len(percents) >= 2:
                res["mom"] = float(percents[0])
                res["yoy"] = float(percents[1])
    except: pass
    return res

# ============================================================
# 3. 核心技術過濾 (提速關鍵：精簡 Request)
# ============================================================

def run_strategy_check(s, bias_limit, vol_limit):
    try:
        # 使用 range=1y 取代確切時間戳，減少後端運算時間
        api_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}?range=1y&interval=1d"
        r = session.get(api_url, headers=get_headers(), timeout=7)
        
        data = r.json()['chart']['result'][0]
        quote = data['indicators']['quote'][0]
        c_raw = quote['close']
        v_raw = quote['volume']
        
        # 快速過濾無效數據
        if not c_raw or len(c_raw) < 65: return None
        
        # 使用 numpy 加速計算
        c = pd.Series(c_raw).ffill().values
        v = pd.Series(v_raw).ffill().values
        
        vol_today = v[-1]
        avg_vol_5d = np.mean(v[-5:]) / 1000
        
        if avg_vol_5d < vol_limit: return None
        
        # 計算均線
        ma30 = np.mean(c[-30:])
        ma45 = np.mean(c[-45:])
        ma60 = np.mean(c[-60:])
        curr_price = c[-1]
        bias_30 = ((curr_price - ma30) / ma30) * 100
        
        if (ma30 > ma45 > ma60) and (0 <= bias_30 <= bias_limit):
            vol_yesterday = v[-2]
            vol_change = ((vol_today - vol_yesterday) / vol_yesterday) * 100 if vol_yesterday > 0 else 0
            return {**s, "收盤": round(curr_price, 2), "乖離30MA(%)": round(bias_30, 2), 
                    "成交量(張)": int(vol_today / 1000), "量變動(%)": round(vol_change, 2)}
    except:
        return None
    return None

# (其餘 K 線繪圖與新聞抓取函數保持不變)
def draw_k_line(ticker, name):
    try:
        yt = yf.Ticker(ticker)
        df = yt.history(period="1y") 
        if df.empty or len(df) < 60: return None
        df['MA30'] = df['Close'].rolling(30).mean()
        df['MA45'] = df['Close'].rolling(45).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df = df.tail(180).copy()
        colors = ['#ef5350' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#26a69a' for i in range(len(df))]
        df.index = df.index.strftime('%Y-%m-%d')
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                                     name='K線', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA30'], line=dict(color='orange', width=1.5), name='30MA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA45'], line=dict(color='blue', width=1.5), name='45MA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='purple', width=1.5), name='60MA'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors), row=2, col=1)
        fig.update_layout(title=f"{name} ({ticker}) 180日K線與均線圖", xaxis_rangeslider_visible=False, height=600, template='plotly_dark')
        fig.update_xaxes(type='category') 
        return fig
    except: return None

def get_tw_stock_news(code):
    try:
        news_url = f"https://tw.stock.yahoo.com/quote/{code}/news"
        resp = session.get(news_url, headers=get_headers(), timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        news_links = soup.find_all('a', href=True)
        results = []
        seen_titles = set()
        for link in news_links:
            href = link.get('href')
            if '/news/' in href:
                title = link.get_text(strip=True)
                if len(title) < 8 or title in seen_titles: continue
                full_link = href if href.startswith('http') else "https://tw.stock.yahoo.com" + href
                results.append({"title": title, "link": full_link, "sentiment": "💡 資訊", "color": "#888888", "publisher": "Yahoo股市"})
                seen_titles.add(title)
                if len(results) >= 5: break
        return results
    except: return None

# ============================================================
# 4. Streamlit UI 
# ============================================================

st.sidebar.header("🎯 策略設定")
user_bias = st.sidebar.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0, step=0.1)
user_vol = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if st.button("🚀 急速全市場掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning = True
    st.session_state.current_idx = 0 
    st.rerun()

if st.session_state.is_scanning:
    status = st.empty()
    bar = st.progress(0)
    stocks_list = get_stock_market_list()
    initial_hits = []
    
    # 提速點：提高 max_workers 至 50
    with ThreadPoolExecutor(max_workers=50) as ex:
        futures = {ex.submit(run_strategy_check, s, user_bias, user_vol): s for s in stocks_list}
        for i, f in enumerate(as_completed(futures), 1):
            if i % 100 == 0: 
                bar.progress(i / len(stocks_list))
                status.text(f"⚡ 急速掃描中: {i} / {len(stocks_list)}")
            res = f.result()
            if res: initial_hits.append(res)
            
    if initial_hits:
        status.text(f"📊 命中 {len(initial_hits)} 檔，同步財報數據...")
        final_list = []
        with ThreadPoolExecutor(max_workers=20) as ex:
            f_deep = {ex.submit(fetch_deep_info, r['ticker']): r for r in initial_hits}
            for f in as_completed(f_deep):
                deep_res = f.result()
                final_list.append({**f_deep[f], "本益比": deep_res["pe"], "營收月增": deep_res["mom"], "營收年增": deep_res["yoy"]})
        st.session_state.scan_results = pd.DataFrame(final_list)
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("查無標的。")
    st.session_state.is_scanning = False
    st.rerun()

# (表格顯示與切換邏輯與之前相同，略作精簡)
if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    st.success(f"✅ 掃描完成！找到 {len(df)} 支標的")
    st.dataframe(df.style.highlight_max(axis=0, subset=['營收年增'], color='#3d0000'), use_container_width=True)
    
    st.divider()
    c_idx = st.session_state.current_idx % len(df)
    col1, col2, col3 = st.columns([1,2,1])
    if col1.button("⬅️ 上一支"): st.session_state.current_idx -= 1; st.rerun()
    col2.markdown(f"<center><b>{df.iloc[c_idx]['code']} {df.iloc[c_idx]['name']}</b></center>", unsafe_allow_html=True)
    if col3.button("下一支 ➡️"): st.session_state.current_idx += 1; st.rerun()
    
    k_fig = draw_k_line(df.iloc[c_idx]['ticker'], df.iloc[c_idx]['name'])
    if k_fig: st.plotly_chart(k_fig, use_container_width=True)
