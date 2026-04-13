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
st.set_page_config(page_title="台股智慧選股儀表板", layout="wide", page_icon="📈")

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

# ── Session State 初始化 ──────────────────────────────────────
if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False
if 'current_idx' not in st.session_state:
    st.session_state['current_idx'] = 0
if 'last_selected_row' not in st.session_state:
    st.session_state['last_selected_row'] = None
if 'table_key' not in st.session_state:
    st.session_state['table_key'] = 0

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

@st.cache_data(ttl=3600)
def download_batch_history(tickers: tuple) -> dict:
    if not tickers: return {}
    ticker_str = " ".join(tickers)
    try:
        raw = yf.download(ticker_str, period="4mo", interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=True)
    except: return {}

    result = {}
    if len(tickers) == 1:
        tk = tickers[0]
        try:
            df = raw[["Close", "Volume"]].dropna()
            df.columns = ["close", "volume"]
            df["volume"] = (df["volume"] / 1000).astype(int)
            result[tk] = df.reset_index(drop=True)
        except: pass
    else:
        for tk in tickers:
            try:
                df = raw[tk][["Close", "Volume"]].dropna()
                df.columns = ["close", "volume"]
                df["volume"] = (df["volume"] / 1000).astype(int)
                result[tk] = df.reset_index(drop=True)
            except: pass
    return result

def calc_ma_signals(history_map: dict, stock_map: list, bias_limit: float, vol_limit: int) -> list:
    hits = []
    for s in stock_map:
        tk = s["ticker"]
        df = history_map.get(tk)
        if df is None or len(df) < 65: continue
        closes, volumes = df["close"], df["volume"]
        avg_vol_5d = volumes.tail(5).mean()
        if avg_vol_5d < vol_limit: continue

        ma30, ma45, ma60 = closes.rolling(30).mean().iloc[-1], closes.rolling(45).mean().iloc[-1], closes.rolling(60).mean().iloc[-1]
        curr_price = float(closes.iloc[-1])
        vol_today, vol_yesterday = int(volumes.iloc[-1]), float(volumes.iloc[-2])
        bias_30 = ((curr_price - ma30) / ma30) * 100
        vol_change = ((vol_today - vol_yesterday) / vol_yesterday * 100) if vol_yesterday > 0 else 0

        if (ma30 > ma45 > ma60) and (0 <= bias_30 <= bias_limit):
            hits.append({**s, "收盤": round(curr_price, 2), "乖離30MA(%)": round(bias_30, 2), "成交量(張)": vol_today, "量變動(%)": round(vol_change, 2)})
    return hits

def clean_percent(text):
    if not text or text == "N/A": return np.nan
    try: return float(text.replace('%', '').replace(',', ''))
    except: return np.nan

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
        rev_resp = requests.get(rev_url, headers=get_headers(), timeout=10)
        soup = BeautifulSoup(rev_resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.get_text()]
            if len(percents) >= 2:
                res["mom"], res["yoy"] = clean_percent(percents[0]), clean_percent(percents[1])
    except: pass
    return res

@st.cache_data(ttl=3600)
def get_kline_data(code: str, market: str) -> pd.DataFrame:
    rows = []
    now = get_tw_now()
    try:
        if market == "TW":
            for delta in range(6):
                d = now - timedelta(days=30 * delta)
                url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={d.year}{d.month:02d}01&stockNo={code}"
                r = requests.get(url, headers=get_headers(), timeout=10)
                data = r.json().get("data", [])
                for row in data:
                    yy, mm, dd = row[0].split("/")
                    rows.append({"date": f"{int(yy)+1911}-{mm}-{dd}", "open": float(str(row[3]).replace(",","")), "high": float(str(row[4]).replace(",","")), "low": float(str(row[5]).replace(",","")), "close": float(str(row[6]).replace(",","")), "volume": int(str(row[1]).replace(",","")) // 1000})
        else:
            for delta in range(6):
                d = now - timedelta(days=30 * delta)
                roc_ym = f"{d.year - 1911}/{d.month:02d}"
                url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d={roc_ym}&stkno={code}"
                r = requests.get(url, headers=get_headers(), timeout=10)
                data = r.json().get("aaData", [])
                for row in data:
                    yy, mm, dd = row[0].split("/")
                    rows.append({"date": f"{int(yy)+1911}-{mm}-{dd}", "open": float(str(row[3]).replace(",","")), "high": float(str(row[4]).replace(",","")), "low": float(str(row[5]).replace(",","")), "close": float(str(row[6]).replace(",","")), "volume": int(str(row[1]).replace(",","")) // 1000})
    except: pass
    return pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()

# ============================================================
# 3. 繪圖函數修正 (核心修復處)
# ============================================================

def draw_k_line(ticker, name):
    code = ticker.split(".")[0]
    market = "TW" if ticker.endswith(".TW") else "TWO"
    df = get_kline_data(code, market)
    
    # 備援機制
    if df.empty or len(df) < 10:
        try:
            yt = yf.Ticker(ticker)
            raw = yt.history(period="6mo")
            if raw.empty: return None
            df = raw.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}).reset_index()
            df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
            df["volume"] = df["volume"] // 1000
        except: return None

    df = df.tail(180).copy()
    if df.empty: return None

    df['MA30'] = df['close'].rolling(30).mean()
    df['MA45'] = df['close'].rolling(45).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    colors = ['#ef5350' if df['close'].iloc[i] >= df['open'].iloc[i] else '#26a69a' for i in range(len(df))]

    # 建立畫布並增加安全檢查
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K線', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA30'], line=dict(color='orange', width=1.5), name='30MA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA45'], line=dict(color='blue', width=1.5), name='45MA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], line=dict(color='purple', width=1.5), name='60MA'), row=1, col=1)
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='成交量', marker_color=colors), row=2, col=1)

        fig.update_layout(title=f"{name} ({ticker})", xaxis_rangeslider_visible=False, height=620, template='plotly_dark', paper_bgcolor='#050d1a', plot_bgcolor='#070f1f', font=dict(color="#7a9aaa"), margin=dict(l=12, r=12, t=48, b=12))
        fig.update_xaxes(type='category', gridcolor='#ffffff10', showgrid=True, zeroline=False)
        fig.update_yaxes(gridcolor='#ffffff10', showgrid=True, zeroline=False)
        return fig
    except: return None

def get_tw_stock_news(code):
    try:
        news_url = f"https://tw.stock.yahoo.com/quote/{code}/news"
        resp = requests.get(news_url, headers=get_headers(), timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        news_links = soup.find_all('a', href=True)
        results, seen_titles = [], set()
        for link in news_links:
            href, title = link.get('href'), link.get_text(strip=True)
            if '/news/' in href and len(title) > 8 and title not in seen_titles:
                full_link = href if href.startswith('http') else "https://tw.stock.yahoo.com" + href
                results.append({"title": title, "link": full_link, "sentiment": "💡 資訊", "color": "#888888", "publisher": "Yahoo股市"})
                seen_titles.add(title)
                if len(results) >= 8: break
        return results
    except: return None

# ============================================================
# 4. Streamlit UI (CSS 保持不變)
# ============================================================
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Noto+Sans+TC:wght@300;400;500;700&display=swap');
html, body, [data-testid="stAppViewContainer"] { background-color: #050d1a !important; color: #c8d8e8 !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #070f1f 0%, #0a1628 100%) !important; }
[data-testid="stButton"] > button { background: linear-gradient(135deg, #003d2e 0%, #00251a 100%) !important; color: #00ffc8 !important; font-family: 'Share Tech Mono' !important; }
p, span, label { font-family: 'Noto Sans TC', sans-serif !important; }
</style>""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("🎯 策略設定")
user_bias = st.sidebar.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0, step=0.1)
user_vol  = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if st.button("🚀 開始智慧掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning, st.session_state.current_idx, st.session_state.last_selected_row = True, 0, None
    st.rerun()

if st.session_state.is_scanning:
    status, bar = st.empty(), st.progress(0)
    stock_map = get_stock_market_list()
    all_tickers = [s["ticker"] for s in stock_map]
    history_map, BATCH = {}, 200
    batches = [all_tickers[i:i+BATCH] for i in range(0, len(all_tickers), BATCH)]
    for bi, batch in enumerate(batches):
        status.text(f"📥 批次下載 {bi+1}/{len(batches)}...")
        bar.progress(0.03 + 0.72 * (bi / len(batches)))
        history_map.update(download_batch_history(tuple(batch)))
    
    hits = calc_ma_signals(history_map, stock_map, user_bias, user_vol)
    if hits:
        final_list = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            f_deep = {ex.submit(fetch_deep_info, r["ticker"]): r for r in hits}
            for j, f in enumerate(as_completed(f_deep), 1):
                res, base = f.result(), f_deep[f]
                final_list.append({**base, "本益比": res["pe"], "營收月增": res["mom"], "營收年增": res["yoy"]})
        st.session_state.scan_results = pd.DataFrame(final_list)
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("查無標的。")
    st.session_state.is_scanning = False
    st.rerun()

# ============================================================
# 5. 結果顯示與 K 線 (修正索引安全)
# ============================================================
if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results.copy()
    if st.session_state.current_idx >= len(df): st.session_state.current_idx = 0
    
    # 顯示表格
    df_display = df[["code", "name", "收盤", "乖離30MA(%)", "成交量(張)", "量變動(%)", "industry"]].rename(columns={"code":"代碼","name":"名稱","industry":"類股"})
    event = st.dataframe(df_display, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key=f"stock_table_{st.session_state.table_key}")

    if event and "selection" in event and event["selection"]["rows"]:
        clicked_row = event["selection"]["rows"][0]
        if clicked_row != st.session_state.last_selected_row:
            st.session_state.current_idx, st.session_state.last_selected_row = clicked_row, clicked_row

    # 切換按鈕
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ 上一支", use_container_width=True):
            st.session_state.current_idx = (st.session_state.current_idx - 1) % len(df)
            st.session_state.table_key += 1
            st.rerun()
    with col2:
        curr = df.iloc[st.session_state.current_idx]
        st.markdown(f"<div style='text-align:center; color:#00ffc8;'><b>{curr['code']} {curr['name']}</b></div>", unsafe_allow_html=True)
    with col3:
        if st.button("下一支 ➡️", use_container_width=True):
            st.session_state.current_idx = (st.session_state.current_idx + 1) % len(df)
            st.session_state.table_key += 1
            st.rerun()

    # 繪製 K 線 (加強錯誤捕獲)
    k_fig = draw_k_line(curr['ticker'], curr['name'])
    if k_fig:
        st.plotly_chart(k_fig, use_container_width=True)
    else:
        st.error("⚠️ 該標的 K 線資料讀取失敗。")
else:
    if not st.session_state.is_scanning: st.info("調整參數後點擊掃描按鈕。")
