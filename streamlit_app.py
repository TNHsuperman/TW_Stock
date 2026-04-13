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
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
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
# 記錄上一次點擊的列，用來偵測「是否真的換了一列」
if 'last_selected_row' not in st.session_state:
    st.session_state['last_selected_row'] = None
# 每次按鈕切換時改變 key，強制 Streamlit 重建表格 widget → selection 自動清空
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

# ============================================================
# 2-A. yfinance 批次下載（核心加速）
# ============================================================

@st.cache_data(ttl=3600)
def download_batch_history(tickers: tuple) -> dict:
    """
    用 yfinance.download() 一次下載多支股票的歷史資料。
    比逐支請求快 10 倍以上，且不容易被限流。
    回傳 dict: { "2330.TW": DataFrame(close, volume), ... }
    """
    if not tickers:
        return {}
    ticker_str = " ".join(tickers)
    try:
        raw = yf.download(
            ticker_str,
            period="4mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return {}

    result = {}
    # 單支時 columns 是 [Open,High,Low,Close,Volume]；多支時是 MultiIndex
    if len(tickers) == 1:
        tk = tickers[0]
        try:
            df = raw[["Close", "Volume"]].dropna()
            df.columns = ["close", "volume"]
            df["volume"] = (df["volume"] / 1000).astype(int)
            result[tk] = df.reset_index(drop=True)
        except Exception:
            pass
    else:
        for tk in tickers:
            try:
                df = raw[tk][["Close", "Volume"]].dropna()
                df.columns = ["close", "volume"]
                df["volume"] = (df["volume"] / 1000).astype(int)
                result[tk] = df.reset_index(drop=True)
            except Exception:
                pass
    return result


def calc_ma_signals(history_map: dict, stock_map: list,
                    bias_limit: float, vol_limit: int) -> list:
    """
    純 CPU 計算：對已下載好的歷史資料做均線條件判斷，不發任何網路請求。
    """
    hits = []
    for s in stock_map:
        tk = s["ticker"]
        df = history_map.get(tk)
        if df is None or len(df) < 65:
            continue
        closes  = df["close"]
        volumes = df["volume"]

        avg_vol_5d = volumes.tail(5).mean()
        if avg_vol_5d < vol_limit:
            continue

        ma30 = closes.rolling(30).mean().iloc[-1]
        ma45 = closes.rolling(45).mean().iloc[-1]
        ma60 = closes.rolling(60).mean().iloc[-1]
        curr_price    = float(closes.iloc[-1])
        vol_today     = int(volumes.iloc[-1])
        vol_yesterday = float(volumes.iloc[-2])
        bias_30       = ((curr_price - ma30) / ma30) * 100
        vol_change    = ((vol_today - vol_yesterday) / vol_yesterday * 100) if vol_yesterday > 0 else 0

        if (ma30 > ma45 > ma60) and (0 <= bias_30 <= bias_limit):
            hits.append({**s,
                "收盤":        round(curr_price, 2),
                "乖離30MA(%)": round(bias_30, 2),
                "成交量(張)":  vol_today,
                "量變動(%)":   round(vol_change, 2),
            })
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
                res["mom"] = clean_percent(percents[0])
                res["yoy"] = clean_percent(percents[1])
    except: pass
    return res


@st.cache_data(ttl=3600)
def get_kline_data(code: str, market: str) -> pd.DataFrame:
    """抓 K 線用歷史資料（含 OHLCV），最多 6 個月，來自官方 API。"""
    rows = []
    now = get_tw_now()
    months = 6
    if market == "TW":
        for delta in range(months):
            d = now - timedelta(days=30 * delta)
            yyyymm = f"{d.year}{d.month:02d}01"
            url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={yyyymm}&stockNo={code}"
            try:
                r = requests.get(url, headers=get_headers(), timeout=10)
                for row in r.json().get("data", []):
                    try:
                        # [日期, 成交股數, 成交金額, 開盤, 最高, 最低, 收盤, 漲跌, 筆數]
                        date   = str(row[0])
                        # 民國年轉西元
                        yy, mm, dd = date.split("/")
                        date_str = f"{int(yy)+1911}-{mm}-{dd}"
                        rows.append({
                            "date":   date_str,
                            "open":   float(str(row[3]).replace(",","")),
                            "high":   float(str(row[4]).replace(",","")),
                            "low":    float(str(row[5]).replace(",","")),
                            "close":  float(str(row[6]).replace(",","")),
                            "volume": int(str(row[1]).replace(",","")) // 1000,
                        })
                    except: pass
            except: pass
    else:
        for delta in range(months):
            d = now - timedelta(days=30 * delta)
            roc_ym = f"{d.year - 1911}/{d.month:02d}"
            url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d={roc_ym}&stkno={code}"
            try:
                r = requests.get(url, headers=get_headers(), timeout=10)
                for row in r.json().get("aaData", []):
                    try:
                        # [日期, 成交股數, 成交金額, 開盤, 最高, 最低, 收盤, ...]
                        date_raw = str(row[0])
                        yy, mm, dd = date_raw.split("/")
                        date_str = f"{int(yy)+1911}-{mm}-{dd}"
                        rows.append({
                            "date":   date_str,
                            "open":   float(str(row[3]).replace(",","")),
                            "high":   float(str(row[4]).replace(",","")),
                            "low":    float(str(row[5]).replace(",","")),
                            "close":  float(str(row[6]).replace(",","")),
                            "volume": int(str(row[1]).replace(",","")) // 1000,
                        })
                    except: pass
            except: pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


def draw_k_line(ticker, name):
    code   = ticker.split(".")[0]
    market = "TW" if ticker.endswith(".TW") else "TWO"
    df = get_kline_data(code, market)
    # fallback to yfinance if official API fails
    if df.empty or len(df) < 30:
        yt = yf.Ticker(ticker)
        raw = yt.history(period="6mo")
        if raw.empty: return None
        df = raw.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}).reset_index()
        df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
        df["volume"] = df["volume"] // 1000
    df = df.tail(180).copy()
    if len(df) < 10: return None
    df['MA30'] = df['close'].rolling(30).mean()
    df['MA45'] = df['close'].rolling(45).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    colors = ['#ef5350' if df['close'].iloc[i] >= df['open'].iloc[i] else '#26a69a' for i in range(len(df))]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='K線',
                                 increasing_line_color='#ef5350', decreasing_line_color='#26a69a'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA30'], line=dict(color='orange', width=1.5), name='30MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA45'], line=dict(color='blue',   width=1.5), name='45MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], line=dict(color='purple', width=1.5), name='60MA'), row=1, col=1)
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='成交量', marker_color=colors), row=2, col=1)
    fig.update_layout(
        title=dict(
            text=f"<b>{name}</b>  <span style='color:#4a7a8a;font-size:14px;'>({ticker})</span>",
            font=dict(family="Share Tech Mono, monospace", size=16, color="#00ffc8"),
        ),
        xaxis_rangeslider_visible=False,
        height=620,
        template='plotly_dark',
        paper_bgcolor='#050d1a',
        plot_bgcolor='#070f1f',
        font=dict(family="Share Tech Mono, monospace", color="#7a9aaa"),
        legend=dict(
            bgcolor='#0a1628', bordercolor='#0ff2', borderwidth=1,
            font=dict(size=11, color="#a0c4d8"),
        ),
        margin=dict(l=12, r=12, t=48, b=12),
    )
    fig.update_xaxes(type='category', gridcolor='#0ff1', showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor='#0ff1', showgrid=True, zeroline=False)
    return fig

def get_tw_stock_news(code):
    try:
        news_url = f"https://tw.stock.yahoo.com/quote/{code}/news"
        resp = requests.get(news_url, headers=get_headers(), timeout=10)
        if resp.status_code != 200: return None
        soup = BeautifulSoup(resp.text, 'html.parser')
        news_links = soup.find_all('a', href=True)
        pos_words = ["成長", "新高", "利多", "噴發", "買進", "展望佳", "獲利", "創高", "轉盈", "法說", "漲", "配息", "訂單", "營收亮眼"]
        neg_words = ["衰退", "減少", "利空", "調降", "跌", "虧損", "賣出", "縮減", "保守", "淡季", "壓力", "下修"]
        results, seen_titles = [], set()
        for link in news_links:
            href = link.get('href')
            if '/news/' in href and 'tw.stock.yahoo.com' in href or href.startswith('/news/'):
                title = link.get_text(strip=True)
                if len(title) < 8 or title in seen_titles: continue
                full_link = href if href.startswith('http') else "https://tw.stock.yahoo.com" + href
                sentiment, color = "💡 資訊", "#888888"
                if any(w in title for w in pos_words):
                    sentiment, color = "📈 利多", "#ef5350"
                elif any(w in title for w in neg_words):
                    sentiment, color = "📉 利空", "#26a69a"
                results.append({"title": title, "link": full_link, "sentiment": sentiment,
                                 "color": color, "publisher": "Yahoo股市"})
                seen_titles.add(title)
                if len(results) >= 8: break
        return results
    except: return None

# ============================================================
# 4. Streamlit UI 介面
# ============================================================

# ── 全域 CSS 主題注入 ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Noto+Sans+TC:wght@300;400;500;700&display=swap');

/* ── 全域底色 ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #050d1a !important;
    color: #c8d8e8 !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070f1f 0%, #0a1628 100%) !important;
    border-right: 1px solid #0ff3 !important;
}
[data-testid="stSidebar"] * { color: #a0c4d8 !important; }

/* ── 掃描按鈕 ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #003d2e 0%, #00251a 100%) !important;
    color: #00ffc8 !important;
    border: 1px solid #00ffc855 !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 15px !important;
    letter-spacing: 2px !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #005a44 0%, #003d2e 100%) !important;
    border-color: #00ffc8 !important;
    box-shadow: 0 0 18px #00ffc840, 0 0 40px #00ffc820 !important;
    color: #ffffff !important;
}
[data-testid="stButton"] > button:disabled {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
}

/* ── 下載按鈕 ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #001f3d 0%, #000d1a 100%) !important;
    color: #4db8ff !important;
    border: 1px solid #4db8ff55 !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1px !important;
    transition: all 0.25s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #4db8ff !important;
    box-shadow: 0 0 15px #4db8ff40 !important;
    color: #ffffff !important;
}

/* ── 表格 ── */
[data-testid="stDataFrame"] {
    border: 1px solid #0ff2 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

/* ── 進度條 ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00ffc8, #4db8ff) !important;
    box-shadow: 0 0 10px #00ffc860 !important;
}
[data-testid="stProgress"] > div {
    background: #0a1628 !important;
    border: 1px solid #0ff2 !important;
    border-radius: 4px !important;
}

/* ── 成功 / 警告訊息框 ── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    border-left: 3px solid #00ffc8 !important;
    background: #00ffc808 !important;
}

/* ── slider & number_input ── */
[data-testid="stSlider"] div[role="slider"] {
    background: #00ffc8 !important;
    box-shadow: 0 0 8px #00ffc880 !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #070f1f !important;
    border: 1px solid #0ff3 !important;
    color: #c8d8e8 !important;
    border-radius: 4px !important;
}

/* ── divider ── */
hr {
    border-color: #0ff2 !important;
    margin: 16px 0 !important;
}

/* ── caption ── */
[data-testid="stCaptionContainer"] {
    color: #4a6070 !important;
    font-size: 11px !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── 頁面內所有純文字 ── */
p, span, label, .stMarkdown { font-family: 'Noto Sans TC', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ── 頁頭 Banner ───────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(90deg, #050d1a 0%, #071828 40%, #050d1a 100%);
    border-bottom: 1px solid #00ffc830;
    padding: 22px 32px 16px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
">
  <!-- 背景裝飾線 -->
  <div style="position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,#00ffc8,#4db8ff,transparent);"></div>

  <div style="display:flex; align-items:center; gap:16px;">
    <div style="font-size:32px; filter:drop-shadow(0 0 8px #00ffc8);">📈</div>
    <div>
      <div style="
        font-family:'Share Tech Mono',monospace;
        font-size:22px;
        font-weight:700;
        color:#00ffc8;
        letter-spacing:4px;
        text-shadow: 0 0 20px #00ffc860;
        line-height:1.1;
      ">台股智慧選股系統</div>
      <div style="
        font-family:'Share Tech Mono',monospace;
        font-size:11px;
        color:#4a7a8a;
        letter-spacing:3px;
        margin-top:3px;
      ">TAIWAN STOCK SMART SCANNER · MA STRATEGY ENGINE</div>
    </div>
    <div style="margin-left:auto; text-align:right;">
      <div style="
        font-family:'Share Tech Mono',monospace;
        font-size:11px;
        color:#4a7a8a;
        letter-spacing:1px;
      ">STRATEGY</div>
      <div style="
        font-family:'Share Tech Mono',monospace;
        font-size:12px;
        color:#00ffc8;
        letter-spacing:1px;
      ">MA30 &gt; MA45 &gt; MA60 · BULL ALIGNMENT</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar 美化 ──────────────────────────────────────────────
st.sidebar.markdown("""
<div style="
    font-family:'Share Tech Mono',monospace;
    font-size:13px;
    color:#00ffc8;
    letter-spacing:3px;
    padding:8px 0 12px;
    border-bottom:1px solid #00ffc830;
    margin-bottom:12px;
">⚙ STRATEGY CONFIG</div>
""", unsafe_allow_html=True)
st.sidebar.header("🎯 策略設定")
user_bias = st.sidebar.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0, step=0.1)
user_vol  = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if st.button("🚀 開始全市場智慧掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning   = True
    st.session_state.current_idx   = 0
    st.session_state.last_selected_row = None
    st.rerun()

if st.session_state.is_scanning:
    status = st.empty()
    bar    = st.progress(0)
    BATCH  = 200   # 每次 yfinance.download() 的股票數，太大會超時

    # ── Step 1：取得股票清單 ──────────────────────────────────
    status.text("📋 Step 1/3：載入股票清單...")
    bar.progress(0.03)
    stock_map = get_stock_market_list()
    all_tickers = [s["ticker"] for s in stock_map]
    total_tickers = len(all_tickers)

    # ── Step 2：分批批次下載歷史資料 ─────────────────────────
    history_map = {}
    batches = [all_tickers[i:i+BATCH] for i in range(0, total_tickers, BATCH)]
    for bi, batch in enumerate(batches):
        status.text(f"📥 Step 2/3：批次下載歷史資料 {bi+1}/{len(batches)}（每批 {BATCH} 支）...")
        bar.progress(0.03 + 0.72 * (bi / len(batches)))
        batch_data = download_batch_history(tuple(batch))
        history_map.update(batch_data)

    bar.progress(0.75)
    status.text(f"✅ 已下載 {len(history_map)} 支股票資料，計算均線中...")

    # ── Step 3：純 CPU 均線篩選（毫秒級）────────────────────
    initial_hits = calc_ma_signals(history_map, stock_map, user_bias, user_vol)
    bar.progress(0.80)

    if initial_hits:
        # ── Step 4：抓財報數據（只對符合條件的股票）──────────
        status.text(f"📈 找到 {len(initial_hits)} 支！抓取財報數據中...")
        final_list = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            f_deep = {ex.submit(fetch_deep_info, r["ticker"]): r for r in initial_hits}
            for j, f in enumerate(as_completed(f_deep), 1):
                bar.progress(0.80 + 0.19 * j / len(initial_hits))
                deep_res = f.result()
                base     = f_deep[f]
                final_list.append({
                    "ticker":      base["ticker"],
                    "code":        base["code"],
                    "name":        base["name"],
                    "industry":    base["industry"],
                    "收盤":        base["收盤"],
                    "乖離30MA(%)": base["乖離30MA(%)"],
                    "成交量(張)":  base["成交量(張)"],
                    "量變動(%)":   base["量變動(%)"],
                    "本益比":      deep_res["pe"],
                    "營收月增":    deep_res["mom"],
                    "營收年增":    deep_res["yoy"],
                })
        bar.progress(1.0)
        st.session_state.scan_results = pd.DataFrame(final_list)
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("查無條件標的。")

    st.session_state.is_scanning = False
    st.rerun()

# ============================================================
# 5. 結果顯示
# ============================================================

if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results.copy()

    col_msg, col_dl = st.columns([3, 1])
    with col_msg:
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #002a1a, #001a10);
            border: 1px solid #00ffc840;
            border-left: 3px solid #00ffc8;
            border-radius: 4px;
            padding: 12px 20px;
            display: flex; align-items: center; gap: 12px;
        ">
            <span style="font-size:20px;">✅</span>
            <span style="font-family:'Share Tech Mono',monospace; color:#00ffc8; letter-spacing:2px; font-size:14px;">
                掃描完成 &nbsp;·&nbsp; 找到 <b style="font-size:20px; color:#fff;">{len(df)}</b> 支符合條件標的
            </span>
            <span style="margin-left:auto; font-family:'Share Tech Mono',monospace;
                color:#4a7a8a; font-size:11px; letter-spacing:1px;">
                {get_tw_now().strftime("%Y-%m-%d %H:%M")} TWN
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col_dl:
        csv     = df.to_csv(index=False).encode('utf-8-sig')
        tw_date = get_tw_now().strftime("%Y%m%d")
        st.download_button(label="⬇ 匯出 CSV", data=csv,
                           file_name=f'tw_stock_scan_{tw_date}.csv',
                           mime='text/csv', use_container_width=True)

    show_cols      = ["code", "name", "收盤", "乖離30MA(%)", "成交量(張)", "量變動(%)", "本益比", "營收月增", "營收年增", "industry"]
    available_cols = [c for c in show_cols if c in df.columns]
    df_display     = df[available_cols].rename(columns={"code": "代碼", "name": "名稱", "industry": "類股"})

    def color_tw_style(val):
        if pd.isna(val): return ''
        color = '#ef5350' if val > 0 else '#26a69a' if val < 0 else 'white'
        return f'color: {color}; font-weight: bold'

    # ── 確保索引不溢出 ──────────────────────────────────────────
    total_found = len(df)
    if st.session_state.current_idx >= total_found:
        st.session_state.current_idx = 0

    # ── 顯示表格，捕捉點擊事件 ─────────────────────────────────
    event = st.dataframe(
        df_display.style.map(
            color_tw_style,
            subset=[c for c in ['量變動(%)', '營收月增', '營收年增'] if c in df_display.columns]
        ),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",          # 點擊任一列時觸發 rerun
        selection_mode="single-row",
        key=f"stock_table_{st.session_state.table_key}",  # ★ key 改變時表格重建，selection 清空
        column_config={
            "代碼":        st.column_config.TextColumn("代碼"),
            "名稱":        st.column_config.TextColumn("名稱"),
            "收盤":        st.column_config.NumberColumn("價格",    format="%.2f"),
            "乖離30MA(%)": st.column_config.ProgressColumn("30MA 乖離", help=f"上限設定為 {user_bias}%",
                                                            format="%.2f%%", min_value=0, max_value=user_bias),
            "量變動(%)":   st.column_config.NumberColumn("量變動",  format="%.1f%%"),
            "營收月增":    st.column_config.NumberColumn("營收月增", format="%.1f%%"),
            "營收年增":    st.column_config.NumberColumn("營收年增", format="%.1f%%"),
            "本益比":      st.column_config.NumberColumn("PE",       format="%.1f"),
            "成交量(張)":  st.column_config.NumberColumn("成交量",  format="%d 📦"),
            "類股":        st.column_config.TextColumn("產業別"),
        }
    )

    # ── ★ 核心修正：點擊列 → 同步更新 current_idx ──────────────
    # 關鍵：last_selected_row 永遠與 current_idx 保持一致。
    # 表格 selection 的列號若等於 last_selected_row，代表是「舊的殘留選取」→ 忽略。
    # 只有使用者真正點了不同列，clicked_row 才會與 last_selected_row 不同 → 更新。
    if event and "selection" in event and event["selection"]["rows"]:
        clicked_row = event["selection"]["rows"][0]
        if clicked_row != st.session_state.last_selected_row:
            st.session_state.current_idx       = clicked_row
            st.session_state.last_selected_row = clicked_row

    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace; font-size:11px;
        color:#2a4a5a; padding:6px 4px; letter-spacing:0.5px; line-height:2;">
        ▸ 進度條滿格 = 乖離率接近上限 ({user_bias}%)，越短代表越貼近 30MA
        &nbsp;&nbsp;▸ 紅字 = 正增長 / 綠字 = 負增長
        &nbsp;&nbsp;▸ 點擊任一列查看 K 線圖
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # 6. 上一支 / 下一支 按鈕 + K 線圖
    # ============================================================
    st.divider()

    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])

    with btn_col1:
        if st.button("⬅️ 上一支", use_container_width=True):
            new_idx = (st.session_state.current_idx - 1) % total_found
            st.session_state.current_idx       = new_idx
            st.session_state.last_selected_row = None          # 表格重建後無選取
            st.session_state.table_key        += 1             # ★ 強制重建表格，清除打勾
            st.rerun()

    with btn_col2:
        c_idx = st.session_state.current_idx
        st.markdown(f"""
        <div style="
            text-align:center;
            font-family:'Share Tech Mono',monospace;
            padding: 8px 0;
        ">
            <span style="color:#4a7a8a; font-size:11px; letter-spacing:2px;">
                [ {c_idx+1} / {total_found} ]
            </span><br>
            <span style="color:#00ffc8; font-size:16px; font-weight:700;
                text-shadow: 0 0 12px #00ffc860; letter-spacing:2px;">
                {df.iloc[c_idx]['code']}
            </span>
            <span style="color:#c8d8e8; font-size:15px; margin-left:8px;">
                {df.iloc[c_idx]['name']}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with btn_col3:
        if st.button("下一支 ➡️", use_container_width=True):
            new_idx = (st.session_state.current_idx + 1) % total_found
            st.session_state.current_idx       = new_idx
            st.session_state.last_selected_row = None          # 表格重建後無選取
            st.session_state.table_key        += 1             # ★ 強制重建表格，清除打勾
            st.rerun()

    # ── K 線圖 ────────────────────────────────────────────────
    current_stock = df.iloc[st.session_state.current_idx]
    k_fig = draw_k_line(current_stock['ticker'], current_stock['name'])
    if k_fig:
        st.plotly_chart(k_fig, use_container_width=True)
    else:
        st.warning("⚠️ 無法載入 K 線資料，請稍後再試。")

    # ── 新聞 ──────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        font-family:'Share Tech Mono',monospace;
        font-size:13px; color:#4db8ff; letter-spacing:3px;
        padding: 16px 0 8px;
        border-top: 1px solid #4db8ff20;
        margin-top: 8px;
    ">
        📡 &nbsp; LIVE NEWS &nbsp;·&nbsp;
        <span style="color:#c8d8e8;">{current_stock['name']} ({current_stock['code']})</span>
    </div>
    """, unsafe_allow_html=True)
    news_list = get_tw_stock_news(current_stock['code'])
    if news_list:
        for n in news_list:
            st.markdown(f"""
            <div style="padding:15px; border-bottom:1px solid #444; background-color:rgba(255,255,255,0.02); margin-bottom:8px; border-radius:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="color:{n['color']}; font-weight:bold; border:1px solid {n['color']}; padding:3px 10px; border-radius:15px; font-size:12px;">{n['sentiment']}</span>
                    <span style="color:#aaa; font-size:12px;">{n['publisher']}</span>
                </div>
                <a href="{n['link']}" target="_blank" style="text-decoration:none; color:#ffffff; font-size:17px; font-weight:500; line-height:1.4;">{n['title']}</a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ 無法獲取即時新聞。")

else:
    if not st.session_state.is_scanning:
        st.markdown("""
        <div style="
            text-align:center;
            padding: 80px 20px;
            font-family:'Share Tech Mono',monospace;
        ">
            <div style="font-size:56px; margin-bottom:20px;
                filter:drop-shadow(0 0 20px #00ffc860);">📈</div>
            <div style="font-size:20px; color:#00ffc8; letter-spacing:4px;
                text-shadow:0 0 20px #00ffc840; margin-bottom:12px;">
                READY TO SCAN
            </div>
            <div style="font-size:13px; color:#2a5060; letter-spacing:2px; margin-bottom:6px;">
                均線多頭排列策略 · MA30 &gt; MA45 &gt; MA60
            </div>
            <div style="font-size:12px; color:#1a3040; letter-spacing:1px;">
                調整左側策略參數，點擊掃描按鈕開始選股
            </div>
        </div>
        """, unsafe_allow_html=True)
