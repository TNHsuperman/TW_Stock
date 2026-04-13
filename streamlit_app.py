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
st.set_page_config(page_title="台股智慧選股儀表板", layout="wide", page_icon="💰")

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
        title=f"{name} ({ticker})",
        xaxis_rangeslider_visible=False,
        height=620,
        template='plotly_dark',
        paper_bgcolor='#050d1a',
        plot_bgcolor='#070f1f',
        font=dict(color="#7a9aaa"),
        legend=dict(
            bgcolor='#0a1628', bordercolor='#1a3a4a', borderwidth=1,
            font=dict(size=11, color="#a0c4d8"),
        ),
        margin=dict(l=12, r=12, t=48, b=12),
    )
    fig.update_xaxes(type='category', gridcolor='rgba(255,255,255,0.06)', showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.06)', showgrid=True, zeroline=False)
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
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Noto+Sans+TC:wght@300;400;500;700&display=swap');

/* ── 隱藏 sidebar collapse 按鈕 ── */
[data-testid='stSidebarCollapseButton'],
[data-testid='collapsedControl'] { display: none !important; }

/* ── 全域底色 ── */
html, body, [data-testid='stAppViewContainer'], [data-testid='stMain'] {
    background-color: #02080f !important;
    color: #c8d8e8 !important;
}

/* ── 主內容區背景加細網格紋理 ── */
[data-testid='stMain'] {
    background-image:
        linear-gradient(rgba(0,180,120,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,180,120,0.03) 1px, transparent 1px) !important;
    background-size: 40px 40px !important;
}

/* ── Sidebar ── */
[data-testid='stSidebar'] {
    background: linear-gradient(180deg, #040c18 0%, #061220 100%) !important;
    border-right: 1px solid rgba(0,255,180,0.12) !important;
}
[data-testid='stSidebar'] * { color: #90b8c8 !important; }
[data-testid='stSidebar'] label { font-size: 12px !important; letter-spacing: 1px !important; }

/* ── 主掃描按鈕 ── */
[data-testid='stButton'] > button {
    background: linear-gradient(135deg, #003828 0%, #001a12 100%) !important;
    color: #00ffc0 !important;
    border: 1px solid rgba(0,255,180,0.3) !important;
    border-radius: 2px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 3px !important;
    padding: 14px 0 !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    position: relative !important;
}
[data-testid='stButton'] > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,255,180,0.08), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
[data-testid='stButton'] > button:hover {
    border-color: #00ffc0 !important;
    box-shadow: 0 0 24px rgba(0,255,180,0.25), 0 0 60px rgba(0,255,180,0.1), inset 0 0 20px rgba(0,255,180,0.05) !important;
    color: #ffffff !important;
    letter-spacing: 4px !important;
}
[data-testid='stButton'] > button:disabled { opacity: 0.25 !important; }

/* ── 下載按鈕 ── */
[data-testid='stDownloadButton'] > button {
    background: linear-gradient(135deg, #001828 0%, #000c18 100%) !important;
    color: #38a8e8 !important;
    border: 1px solid rgba(56,168,232,0.25) !important;
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    transition: all 0.25s ease !important;
}
[data-testid='stDownloadButton'] > button:hover {
    border-color: #38a8e8 !important;
    box-shadow: 0 0 16px rgba(56,168,232,0.2) !important;
    color: #ffffff !important;
}

/* ── 表格 ── */
[data-testid='stDataFrame'] {
    border: 1px solid rgba(0,200,140,0.15) !important;
    border-radius: 2px !important;
    box-shadow: 0 0 30px rgba(0,200,140,0.05) !important;
}

/* ── 進度條 ── */
[data-testid='stProgress'] > div > div {
    background: linear-gradient(90deg, #00b478, #00ffc0, #38a8e8) !important;
    box-shadow: 0 0 12px rgba(0,255,180,0.5) !important;
    border-radius: 1px !important;
}
[data-testid='stProgress'] > div {
    background: #060e1a !important;
    border: 1px solid rgba(0,200,140,0.2) !important;
    border-radius: 1px !important;
    height: 6px !important;
}

/* ── Alert ── */
[data-testid='stAlert'] {
    border-radius: 2px !important;
    border-left: 2px solid #00ffc0 !important;
    background: rgba(0,255,180,0.04) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Slider ── */
[data-testid='stSlider'] div[role='slider'] {
    background: #00ffc0 !important;
    box-shadow: 0 0 10px rgba(0,255,180,0.6) !important;
    width: 14px !important; height: 14px !important;
}
[data-testid='stSlider'] > div > div > div {
    background: linear-gradient(90deg, #00ffc0, #38a8e8) !important;
}

/* ── Number input ── */
[data-testid='stNumberInput'] input {
    background: #040c18 !important;
    border: 1px solid rgba(0,200,140,0.2) !important;
    color: #00ffc0 !important;
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}
[data-testid='stNumberInput'] input:focus {
    border-color: rgba(0,255,180,0.5) !important;
    box-shadow: 0 0 10px rgba(0,255,180,0.15) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,200,140,0.12) !important;
    margin: 20px 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #02080f; }
::-webkit-scrollbar-thumb { background: rgba(0,200,140,0.3); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,255,180,0.5); }

/* ── 字型 ── */
p, li, .stMarkdown { font-family: 'Noto Sans TC', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ── 頁頭 Banner ───────────────────────────────────────────────
_now_str = get_tw_now().strftime("%Y-%m-%d %H:%M")
st.markdown(f"""
<div style="background:linear-gradient(135deg,#02080f 0%,#04121e 50%,#02080f 100%);
  border-bottom:1px solid rgba(0,200,140,0.2);padding:24px 36px 20px;
  margin-bottom:12px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,#00ffc0 30%,#38a8e8 70%,transparent);opacity:0.6;"></div>
  <div style="position:absolute;right:-60px;top:-60px;width:200px;height:200px;
    border:1px solid rgba(0,200,140,0.06);border-radius:50%;"></div>
  <div style="position:absolute;right:-30px;top:-30px;width:140px;height:140px;
    border:1px solid rgba(0,200,140,0.08);border-radius:50%;"></div>
  <div style="display:flex;align-items:center;gap:20px;position:relative;">
    <div style="width:52px;height:52px;border:1px solid rgba(0,255,180,0.3);
      border-radius:4px;display:flex;align-items:center;justify-content:center;
      font-size:26px;background:linear-gradient(135deg,rgba(0,100,60,0.4),rgba(0,50,30,0.6));
      box-shadow:0 0 20px rgba(0,255,180,0.15),inset 0 0 15px rgba(0,255,180,0.05);
      flex-shrink:0;">📈</div>
    <div style="flex:1;">
      <div style="font-family:Orbitron,monospace;font-size:20px;font-weight:700;
        color:#00ffc0;letter-spacing:5px;text-shadow:0 0 30px rgba(0,255,180,0.5);
        line-height:1.2;margin-bottom:5px;">台股智慧選股系統</div>
      <div style="display:flex;align-items:center;gap:16px;">
        <span style="font-family:Share Tech Mono,monospace;font-size:10px;
          color:#3a9a7a;letter-spacing:2px;">TAIWAN STOCK SCANNER</span>
        <span style="width:1px;height:10px;background:rgba(0,200,140,0.3);"></span>
        <span style="font-family:Share Tech Mono,monospace;font-size:10px;
          color:#3a9a7a;letter-spacing:2px;">MA STRATEGY ENGINE v9.9</span>
      </div>
    </div>
    <div style="text-align:right;flex-shrink:0;">
      <div style="font-family:Share Tech Mono,monospace;font-size:9px;
        color:#3a8060;letter-spacing:2px;margin-bottom:6px;">ACTIVE STRATEGY</div>
      <div style="font-family:Orbitron,monospace;font-size:11px;color:#00ffc0;
        letter-spacing:2px;border:1px solid rgba(0,255,180,0.2);padding:4px 12px;
        border-radius:2px;background:rgba(0,100,60,0.15);">MA30 › MA45 › MA60</div>
      <div style="font-family:Share Tech Mono,monospace;font-size:9px;
        color:#3a8060;letter-spacing:1px;margin-top:5px;">{_now_str} &nbsp;TWN +08:00</div>
    </div>
  </div>
  <div style="margin-top:16px;padding-top:12px;border-top:1px solid rgba(0,200,140,0.08);
    display:flex;gap:32px;">
    <div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#3a9070;letter-spacing:1px;">
      <span style="color:#00c890;">◆</span> &nbsp;上市 TWSE &nbsp;+&nbsp; 上櫃 TPEX 全市場掃描</div>
    <div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#3a9070;letter-spacing:1px;">
      <span style="color:#00c890;">◆</span> &nbsp;均線多頭排列 · 乖離率過濾 · 成交量門檻</div>
    <div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#3a9070;letter-spacing:1px;">
      <span style="color:#00c890;">◆</span> &nbsp;PE 本益比 · 月增率 · 年增率 財務篩選</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar 美化 ──────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding: 8px 0 20px;">
  <!-- 頂部 logo 區 -->
  <div style="
    font-family:Orbitron,monospace;
    font-size:11px; font-weight:700;
    color:#00c890; letter-spacing:4px;
    padding-bottom:14px;
    border-bottom:1px solid rgba(0,200,140,0.15);
    margin-bottom:18px;
  ">⚙ &nbsp;STRATEGY CONFIG</div>

  <!-- 策略說明 badge -->
  <div style="
    background:rgba(0,80,50,0.2);
    border:1px solid rgba(0,200,140,0.15);
    border-radius:2px;
    padding:10px 12px;
    margin-bottom:18px;
  '>
    <div style="font-family:Share Tech Mono,monospace; font-size:11px;
      color:#3a9070; letter-spacing:2px; margin-bottom:6px;">SIGNAL CONDITION</div>
    <div style="font-family:Share Tech Mono,monospace; font-size:11px; color:#00c890;">
      MA30 &gt; MA45 &gt; MA60
    </div>
    <div style="font-family:Share Tech Mono,monospace; font-size:11px;
      color:#3a9070; letter-spacing:1px; margin-top:4px;">
      BULLISH ALIGNMENT · BIAS FILTER
    </div>
  </div>

  <div style="font-family:Noto Sans TC,sans-serif;
    font-size:13px; font-weight:500; color:#70a0b0;
    letter-spacing:2px; margin-bottom:4px;">
    📐 &nbsp;參數設定
  </div>
</div>
""", unsafe_allow_html=True)
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

    # ── 統計卡片列 ──────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        display:grid; grid-template-columns: 1fr 1fr 1fr 1fr;
        gap:10px; margin-bottom:14px;
    ">
      <div style="
        background:linear-gradient(135deg,rgba(0,80,50,0.3),rgba(0,40,25,0.5));
        border:1px solid rgba(0,200,140,0.2); border-radius:2px;
        padding:14px 18px;
        border-top: 2px solid #00c890;
      ">
        <div style="font-family:Share Tech Mono,monospace;font-size:11px;
          color:#3a9070;letter-spacing:2px;margin-bottom:6px;">SIGNALS FOUND</div>
        <div style="font-family:Orbitron,monospace;font-size:26px;
          font-weight:700;color:#00ffc0;text-shadow:0 0 15px rgba(0,255,180,0.4);">{len(df)}</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#3a9070;margin-top:3px;">支符合條件標的</div>
      </div>
      <div style="
        background:linear-gradient(135deg,rgba(0,40,80,0.3),rgba(0,20,50,0.5));
        border:1px solid rgba(56,168,232,0.15); border-radius:2px;
        padding:14px 18px;
        border-top: 2px solid #38a8e8;
      ">
        <div style="font-family:Share Tech Mono,monospace;font-size:11px;
          color:#3a6888;letter-spacing:2px;margin-bottom:6px;">STRATEGY</div>
        <div style="font-family:Orbitron,monospace;font-size:13px;
          font-weight:600;color:#38a8e8;margin-top:4px;">MA BULL</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#3a6888;margin-top:3px;">30 › 45 › 60 均線排列</div>
      </div>
      <div style="
        background:linear-gradient(135deg,rgba(60,30,0,0.3),rgba(40,20,0,0.5));
        border:1px solid rgba(255,180,0,0.12); border-radius:2px;
        padding:14px 18px;
        border-top: 2px solid #c8900a;
      ">
        <div style="font-family:Share Tech Mono,monospace;font-size:11px;
          color:#886820;letter-spacing:2px;margin-bottom:6px;">BIAS LIMIT</div>
        <div style="font-family:Orbitron,monospace;font-size:26px;
          font-weight:700;color:#e0a020;">{user_bias}<span style="font-size:14px;">%</span></div>
        <div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#886820;margin-top:3px;">30MA 乖離上限</div>
      </div>
      <div style="
        background:linear-gradient(135deg,rgba(40,0,60,0.3),rgba(20,0,40,0.5));
        border:1px solid rgba(160,100,255,0.12); border-radius:2px;
        padding:14px 18px;
        border-top: 2px solid #7840c0;
      ">
        <div style="font-family:Share Tech Mono,monospace;font-size:11px;
          color:#604888;letter-spacing:2px;margin-bottom:6px;">SCAN TIME</div>
        <div style="font-family:Orbitron,monospace;font-size:13px;
          font-weight:600;color:#a070e0;margin-top:4px;">{get_tw_now().strftime("%H:%M")}</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#604888;margin-top:3px;">{get_tw_now().strftime("%Y-%m-%d")} TWN</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_spacer, col_dl = st.columns([3, 1])
    with col_dl:
        csv     = df.to_csv(index=False).encode('utf-8-sig')
        tw_date = get_tw_now().strftime("%Y%m%d")
        st.download_button(label="⬇ &nbsp;EXPORT CSV", data=csv,
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
    <div style="font-family:Share Tech Mono,monospace; font-size:11px;
        color:#4a7a8a; padding:6px 4px; letter-spacing:0.5px; line-height:2; font-size:14px;">
        ▸ 進度條滿格 = 乖離率接近上限 ({user_bias}%)，越短代表越貼近 30MA
        &nbsp;&nbsp;▸ 紅字 = 正增長 / 綠字 = 負增長
        &nbsp;&nbsp;▸ 點擊任一列查看 K 線圖
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # 6. 上一支 / 下一支 按鈕 + K 線圖
    # ============================================================
    # ── K 線圖標題列 ─────────────────────────────────────────────
    st.markdown("""
    <div style="
        display:flex; align-items:center; gap:12px;
        padding: 14px 0 10px;
        border-top: 1px solid rgba(0,200,140,0.12);
        margin-top:8px;
    ">
        <div style="width:3px; height:20px; background:linear-gradient(180deg,#00ffc0,#38a8e8);
            border-radius:2px; flex-shrink:0;"></div>
        <div style="font-family:Orbitron,monospace; font-size:11px; font-weight:600;
            color:#00c890; letter-spacing:3px;">K-LINE CHART</div>
        <div style="flex:1; height:1px; background:rgba(0,200,140,0.08);"></div>
        <div style="font-family:Share Tech Mono,monospace; font-size:9px;
            color:#3a9070; letter-spacing:2px;">MA30 · MA45 · MA60</div>
    </div>
    """, unsafe_allow_html=True)

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
            font-family:Share Tech Mono,monospace;
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
        font-family:Share Tech Mono,monospace;
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
            badge_bg = "rgba(0,80,40,0.3)" if "利多" in n["sentiment"] else "rgba(80,0,20,0.3)" if "利空" in n["sentiment"] else "rgba(20,40,60,0.3)"
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(4,14,26,0.8), rgba(2,8,15,0.9));
                border: 1px solid rgba(40,80,100,0.2);
                border-left: 2px solid {n['color']};
                border-radius:2px;
                padding:14px 18px;
                margin-bottom:8px;
                transition: all 0.2s;
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span style="
                        background:{badge_bg};
                        color:{n['color']};
                        border:1px solid {n['color']}40;
                        padding:3px 12px; border-radius:1px;
                        font-family:Share Tech Mono,monospace;
                        font-size:11px; letter-spacing:1px;
                    ">{n['sentiment']}</span>
                    <span style="font-family:Share Tech Mono,monospace;
                        color:#4a7080; font-size:11px; letter-spacing:1px;">{n['publisher']}</span>
                </div>
                <a href="{n['link']}" target="_blank" style="
                    text-decoration:none; color:#c0d8e8;
                    font-family:Noto Sans TC,sans-serif;
                    font-size:15px; font-weight:400; line-height:1.6;
                    letter-spacing:0.5px;
                ">{n['title']}</a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ 無法獲取即時新聞。")

else:
    if not st.session_state.is_scanning:
        st.markdown("""
        <div style="text-align:center;padding:70px 20px 80px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
            width:320px;height:320px;border:1px solid rgba(0,200,140,0.04);border-radius:50%;"></div>
          <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
            width:220px;height:220px;border:1px solid rgba(0,200,140,0.06);border-radius:50%;"></div>
          <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
            width:130px;height:130px;border:1px solid rgba(0,200,140,0.08);border-radius:50%;"></div>
          <div style="position:relative;">
            <div style="font-size:48px;margin-bottom:24px;
              filter:drop-shadow(0 0 24px rgba(0,255,180,0.5));">📈</div>
            <div style="font-family:Orbitron,monospace;font-size:22px;font-weight:700;
              color:#00ffc0;letter-spacing:6px;text-shadow:0 0 30px rgba(0,255,180,0.4);
              margin-bottom:10px;">READY TO SCAN</div>
            <div style="width:120px;height:1px;margin:0 auto 20px;
              background:linear-gradient(90deg,transparent,rgba(0,255,192,0.25),transparent);"></div>
            <div style="font-family:Share Tech Mono,monospace;font-size:12px;
              color:#3a9070;letter-spacing:3px;margin-bottom:8px;">均線多頭排列策略</div>
            <div style="font-family:Orbitron,monospace;font-size:13px;font-weight:600;
              color:#009060;letter-spacing:2px;margin-bottom:24px;">MA30 › MA45 › MA60</div>
            <div style="display:inline-block;background:rgba(0,50,30,0.3);
              border:1px solid rgba(0,200,140,0.1);border-radius:2px;padding:10px 24px;
              font-family:Share Tech Mono,monospace;font-size:11px;
              color:#3a8060;letter-spacing:1px;">← 調整左側參數後點擊掃描按鈕開始</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
