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
import re

# ============================================================
# 1. 基礎設定與環境初始化
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板 v10.0", layout="wide")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
]

def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://tw.stock.yahoo.com/',
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
# 2. 數據清洗與深度資訊抓取 (多源 PE 強力抓取版)
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_market_list():
    stocks = []
    try:
        urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
                ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
        for url, mkt in urls:
            r = requests.get(url, headers=get_headers(), timeout=15, verify=False)
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

def clean_percent(text):
    if not text or text == "N/A": return np.nan
    try: return float(text.replace('%', '').replace(',', ''))
    except: return np.nan

def fetch_deep_info(ticker: str) -> dict:
    """四重抓取機制，確保 PE 不遺漏"""
    code = ticker.split('.')[0]
    res = {"pe": np.nan, "mom": np.nan, "yoy": np.nan}
    
    # --- 步驟 1: 抓取營收 (Yahoo) ---
    try:
        rev_url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
        rev_resp = requests.get(rev_url, headers=get_headers(), timeout=10)
        soup = BeautifulSoup(rev_resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.get_text()]
            if len(percents) >= 2:
                res["mom"] = clean_percent(percents[0])
                res["yoy"] = clean_percent(percents[1])
    except: pass

    # --- 步驟 2: 本益比多源抓取計畫 ---
    
    # 來源 A: Yahoo 股市 (直接解析詳情頁)
    try:
        pe_url = f"https://tw.stock.yahoo.com/quote/{code}"
        pe_resp = requests.get(pe_url, headers=get_headers(), timeout=10)
        pe_soup = BeautifulSoup(pe_resp.text, 'html.parser')
        # 遍歷所有 span 尋找關鍵字
        for s in pe_soup.find_all('span'):
            if "本益比" in s.get_text():
                val = s.find_next_sibling()
                if val and val.get_text() != '-':
                    res["pe"] = float(val.get_text().replace(',', ''))
                    break
    except: pass

    # 來源 B: PChome 股市 (備援，若 Yahoo 抓不到)
    if np.isnan(res["pe"]):
        try:
            pc_url = f"https://stock.pchome.com.tw/stock/sid{code}.html"
            pc_resp = requests.get(pc_url, headers=get_headers(), timeout=10)
            pc_soup = BeautifulSoup(pc_resp.text, 'html.parser')
            # 尋找包含本益比的表格格位
            pe_cell = pc_soup.find(text=re.compile("本益比"))
            if pe_cell:
                val_text = pe_cell.find_parent().find_next_sibling().get_text(strip=True)
                if val_text and val_text != '-':
                    res["pe"] = float(val_text)
        except: pass

    # 來源 C: yfinance
    if np.isnan(res["pe"]):
        try:
            yt = yf.Ticker(ticker)
            pe_val = yt.info.get('trailingPE') or yt.info.get('forwardPE')
            if pe_val: res["pe"] = float(pe_val)
        except: pass

    return res

# ============================================================
# 3. 技術分析、繪圖與新聞抓取
# ============================================================

def run_strategy_check(s, bias_limit, vol_limit):
    now_ts = int(get_tw_now().timestamp())
    start_ts = int((get_tw_now() - timedelta(days=250)).timestamp()) 
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}", 
                         params={"period1":start_ts, "period2":now_ts, "interval":"1d"}, headers=get_headers(), timeout=10)
        data = r.json()['chart']['result'][0]
        c_series = pd.Series(data['indicators']['quote'][0]['close']).ffill().dropna()
        v_series = pd.Series(data['indicators']['quote'][0]['volume']).ffill().dropna()
        if len(c_series) < 65: return None
        
        vol_today = v_series.iloc[-1]
        vol_yesterday = v_series.iloc[-2]
        vol_change = ((vol_today - vol_yesterday) / vol_yesterday) * 100 if vol_yesterday > 0 else 0
        curr_vol_int = int(vol_today / 1000)
        avg_vol_5d = v_series.tail(5).mean() / 1000
        if avg_vol_5d < vol_limit: return None
        
        ma30 = c_series.rolling(30).mean().iloc[-1]
        ma45 = c_series.rolling(45).mean().iloc[-1]
        ma60 = c_series.rolling(60).mean().iloc[-1]
        curr_price = c_series.iloc[-1]
        bias_30 = ((curr_price - ma30) / ma30) * 100
        
        if (ma30 > ma45 > ma60) and (0 <= bias_30 <= bias_limit):
            return {**s, "收盤": round(curr_price, 2), "乖離30MA(%)": round(bias_30, 2), "成交量(張)": curr_vol_int, "量變動(%)": round(vol_change, 2)}
    except: return None
    return None

def draw_k_line(ticker, name):
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

def get_tw_stock_news(code):
    try:
        news_url = f"https://tw.stock.yahoo.com/quote/{code}/news"
        resp = requests.get(news_url, headers=get_headers(), timeout=10)
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
                if len(results) >= 8: break
        return results
    except: return None

# ============================================================
# 4. Streamlit UI
# ============================================================

st.sidebar.header("🎯 策略設定")
user_bias = st.sidebar.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0, step=0.1)
user_vol = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if st.button("🚀 開始全市場智慧掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning = True
    st.session_state.current_idx = 0 
    st.rerun()

if st.session_state.is_scanning:
    status = st.empty()
    bar = st.progress(0)
    stocks_list = get_stock_market_list()
    initial_hits = []
    status.text(f"🔍 第一階段：掃描全市場技術面...")
    with ThreadPoolExecutor(max_workers=30) as ex:
        futures = {ex.submit(run_strategy_check, s, user_bias, user_vol): s for s in stocks_list}
        for i, f in enumerate(as_completed(futures), 1):
            if i % 100 == 0: bar.progress(i / len(stocks_list))
            res = f.result()
            if res: initial_hits.append(res)
    if initial_hits:
        status.text(f"📊 第二階段：想盡辦法抓取 PE/營收 (進度：0/{len(initial_hits)})")
        final_list = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            f_deep = {ex.submit(fetch_deep_info, r['ticker']): r for r in initial_hits}
            for j, f in enumerate(as_completed(f_deep), 1):
                status.text(f"進度: {j} / {len(initial_hits)}")
                deep_res = f.result()
                final_list.append({**f_deep[f], "本益比": deep_res["pe"], "營收月增": deep_res["mom"], "營收年增": deep_res["yoy"]})
        st.session_state.scan_results = pd.DataFrame(final_list)
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("查無條件標的。")
    st.session_state.is_scanning = False
    st.rerun()

# ============================================================
# 5. 結果呈現
# ============================================================

if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results.copy()
    
    # 針對 PE 為空的處理：若仍為空，標註為 0 方便排序，或顯示為 N/A
    df['本益比'] = df['本益比'].fillna(0.0)

    st.success(f"✅ 掃描完成！找到 {len(df)} 支標的")
    
    def color_tw_style(val):
        if pd.isna(val) or val == 0: return ''
        color = '#ef5350' if val > 0 else '#26a69a' if val < 0 else 'white'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df.style.map(color_tw_style, subset=[c for c in ['量變動(%)', '營收月增', '營收年增'] if c in df.columns]),
        use_container_width=True, hide_index=True,
        column_config={
            "code": "代碼", "name": "名稱", "收盤": "價格",
            "乖離30MA(%)": st.column_config.NumberColumn("乖離30MA", format="%.2f%%"),
            "本益比": st.column_config.NumberColumn("PE (0=虧損/缺資料)", format="%.2f"),
            "成交量(張)": st.column_config.NumberColumn("成交量", format="%d"),
            "industry": "產業別"
        }
    )
    
    # K線切換與新聞功能保持不變...
    st.divider()
    total_found = len(df)
    c_idx = st.session_state.current_idx
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col1:
        if st.button("⬅️ 上一支", use_container_width=True):
            st.session_state.current_idx = (c_idx - 1) % total_found
            st.rerun()
    with btn_col2:
        st.markdown(f"<center><b>{df.iloc[c_idx]['code']} {df.iloc[c_idx]['name']}</b></center>", unsafe_allow_html=True)
    with btn_col3:
        if st.button("下一支 ➡️", use_container_width=True):
            st.session_state.current_idx = (c_idx + 1) % total_found
            st.rerun()
    
    current_stock = df.iloc[c_idx]
    k_fig = draw_k_line(current_stock['ticker'], current_stock['name'])
    if k_fig: st.plotly_chart(k_fig, use_container_width=True)
    
    st.subheader(f"📰 {current_stock['name']} 相關新聞")
    news_list = get_tw_stock_news(current_stock['code'])
    if news_list:
        for n in news_list:
            st.markdown(f"• [{n['title']}]({n['link']})")
    else: st.info("無即時新聞。")
else:
    if not st.session_state.is_scanning: st.info("💡 調整參數後執行掃描。")
