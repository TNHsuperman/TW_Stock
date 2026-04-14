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
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# 1. 基礎設定與環境初始化
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(
    page_title="台股智慧選股",
    layout="wide",
    page_icon="https://cdn-icons-png.flaticon.com/512/2953/2953423.png",
    initial_sidebar_state="collapsed",   # 手機預設收起 sidebar
)

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
for key, default in [
    ('scan_results', pd.DataFrame()),
    ('is_scanning', False),
    ('current_idx', 0),
    ('last_selected_row', None),
    ('table_key', 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def get_tw_now():
    return datetime.now(timezone(timedelta(hours=8)))

# ============================================================
# 2. 數據抓取
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_market_list():
    stocks = []
    try:
        session = requests.Session()
        session.verify = False
        adapter = requests.adapters.HTTPAdapter(max_retries=2)
        session.mount('https://', adapter)

        urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
                ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
        for url, mkt in urls:
            r = session.get(url, headers=get_headers(), timeout=8)
            df_isin = pd.read_html(StringIO(r.text))[0]
            df_isin.columns = df_isin.iloc[0]
            for _, row in df_isin.iloc[1:].iterrows():
                val = str(row['有價證券代號及名稱'])
                if '　' in val:
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        stocks.append({"ticker": f"{code}.{mkt}", "name": name, "industry": row['產業別'], "code": code})
    except:
        pass
    return stocks


@st.cache_data(ttl=3600)
def download_batch_history(tickers: tuple) -> dict:
    if not tickers:
        return {}
    ticker_str = " ".join(tickers)
    try:
        raw = yf.download(ticker_str, period="4mo", interval="1d",
                          group_by="ticker", auto_adjust=True, progress=False, threads=True)
    except Exception:
        return {}
    result = {}
    if len(tickers) == 1:
        tk = tickers[0]
        try:
            df = raw[["Close", "Volume"]].dropna()
            df.columns = ["close", "volume"]
            df["volume"] = (df["volume"] / 1000).astype(int)
            result[tk] = df.reset_index(drop=True)
        except:
            pass
    else:
        for tk in tickers:
            try:
                df = raw[tk][["Close", "Volume"]].dropna()
                df.columns = ["close", "volume"]
                df["volume"] = (df["volume"] / 1000).astype(int)
                result[tk] = df.reset_index(drop=True)
            except:
                pass
    return result


def calc_ma_signals(history_map, stock_map, bias_limit, vol_limit):
    hits = []
    for s in stock_map:
        tk = s["ticker"]
        df = history_map.get(tk)
        if df is None or len(df) < 65:
            continue
        closes = df["close"]
        volumes = df["volume"]
        if volumes.tail(5).mean() < vol_limit:
            continue
        ma30 = closes.rolling(30).mean().iloc[-1]
        ma45 = closes.rolling(45).mean().iloc[-1]
        ma60 = closes.rolling(60).mean().iloc[-1]
        curr_price = float(closes.iloc[-1])
        vol_today = int(volumes.iloc[-1])
        vol_yesterday = float(volumes.iloc[-2])
        bias_30 = ((curr_price - ma30) / ma30) * 100
        vol_change = ((vol_today - vol_yesterday) / vol_yesterday * 100) if vol_yesterday > 0 else 0
        if (ma30 > ma45 > ma60) and (0 <= bias_30 <= bias_limit):
            hits.append({**s,
                "收盤": round(curr_price, 2),
                "乖離30MA(%)": round(bias_30, 2),
                "成交量(張)": vol_today,
                "量變動(%)": round(vol_change, 2),
            })
    return hits


def clean_percent(text):
    if not text or text == "N/A":
        return np.nan
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
        if pe:
            res["pe"] = float(pe)
    except:
        pass
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
    except:
        pass
    return res


@st.cache_data(ttl=3600)
def get_kline_data(code: str, market: str) -> pd.DataFrame:
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
                        yy, mm, dd = str(row[0]).split("/")
                        rows.append({
                            "date": f"{int(yy)+1911}-{mm}-{dd}",
                            "open": float(str(row[3]).replace(",", "")),
                            "high": float(str(row[4]).replace(",", "")),
                            "low": float(str(row[5]).replace(",", "")),
                            "close": float(str(row[6]).replace(",", "")),
                            "volume": int(str(row[1]).replace(",", "")) // 1000,
                        })
                    except:
                        pass
            except:
                pass
    else:
        for delta in range(months):
            d = now - timedelta(days=30 * delta)
            roc_ym = f"{d.year - 1911}/{d.month:02d}"
            url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d={roc_ym}&stkno={code}"
            try:
                r = requests.get(url, headers=get_headers(), timeout=10)
                for row in r.json().get("aaData", []):
                    try:
                        yy, mm, dd = str(row[0]).split("/")
                        rows.append({
                            "date": f"{int(yy)+1911}-{mm}-{dd}",
                            "open": float(str(row[3]).replace(",", "")),
                            "high": float(str(row[4]).replace(",", "")),
                            "low": float(str(row[5]).replace(",", "")),
                            "close": float(str(row[6]).replace(",", "")),
                            "volume": int(str(row[1]).replace(",", "")) // 1000,
                        })
                    except:
                        pass
            except:
                pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


def draw_k_line(ticker, name):
    code = ticker.split(".")[0]
    market = "TW" if ticker.endswith(".TW") else "TWO"
    df = get_kline_data(code, market)
    if df.empty or len(df) < 30:
        yt = yf.Ticker(ticker)
        raw = yt.history(period="6mo")
        if raw.empty:
            return None
        df = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}).reset_index()
        df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
        df["volume"] = df["volume"] // 1000
    df = df.tail(70).copy()   # 70天
    if len(df) < 10:
        return None

    df['MA30'] = df['close'].rolling(30).mean()
    df['MA45'] = df['close'].rolling(45).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    colors = ['#ef5350' if df['close'].iloc[i] >= df['open'].iloc[i] else '#26a69a' for i in range(len(df))]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name=f'{name} ({code})',          # ← 顯示股票名稱
        increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
        hoverinfo='none', showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'], mode='none', name='', showlegend=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "開盤：%{customdata[0]:.2f}<br>"
            "收盤：%{customdata[1]:.2f}<br>"
            "最高：%{customdata[2]:.2f}<br>"
            "最低：%{customdata[3]:.2f}<br>"
            "─────────────<br>"
            "30MA：%{customdata[4]:.2f}<br>"
            "45MA：%{customdata[5]:.2f}<br>"
            "60MA：%{customdata[6]:.2f}<br>"
            "─────────────<br>"
            "成交量：%{customdata[7]:,} 張"
            "<extra></extra>"
        ),
        customdata=df[['open', 'close', 'high', 'low', 'MA30', 'MA45', 'MA60', 'volume']].values,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['date'], y=df['MA30'], line=dict(color='orange', width=1.5), name='30MA', hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA45'], line=dict(color='#4488ff', width=1.5), name='45MA', hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], line=dict(color='#cc66ff', width=1.5), name='60MA', hoverinfo='skip'), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'], name='成交量',
        marker_color=colors, showlegend=False,
        hovertemplate="<b>%{x}</b><br>成交量：%{y:,} 張<extra></extra>",
    ), row=2, col=1)

    spike_cfg = dict(
        type='category', showgrid=True, gridcolor='rgba(255,255,255,0.06)',
        zeroline=False, showspikes=True, spikemode='across',
        spikesnap='cursor', spikecolor='rgba(0,255,192,0.5)',
        spikethickness=1, spikedash='dot', showline=False,
    )

    fig.update_layout(
        title=dict(
            text=f'{name}．{ticker}',
            font=dict(size=13, color='#00ffc0', family='Noto Sans TC, sans-serif'),
            x=0, xanchor='left', pad=dict(l=8, t=4),
        ),
        xaxis_rangeslider_visible=False,
        height=500,
        template='plotly_dark',
        paper_bgcolor='#050d1a',
        plot_bgcolor='#070f1f',
        font=dict(color="#7a9aaa", size=11),
        legend=dict(
            bgcolor='rgba(5,13,26,0.75)',
            bordercolor='rgba(0,200,140,0.15)', borderwidth=1,
            font=dict(size=10, color="#a0c4d8"),
            orientation='h',
            yanchor='bottom', y=0.29,
            xanchor='left', x=0.01,
        ),
        margin=dict(l=8, r=8, t=36, b=8),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#0d1f35', bordercolor='rgba(0,200,180,0.4)',
            font=dict(size=11, color='#c8e8ff', family='Share Tech Mono, monospace'),
            namelength=0,
        ),
        hoverdistance=100,
        spikedistance=-1,
        dragmode=False,
        xaxis=dict(
            **spike_cfg,
            fixedrange=True,
            showticklabels=False,
        ),
        xaxis2=dict(
            **spike_cfg,
            matches='x',
            fixedrange=True,
            showticklabels=False,          # 不顯示日期
        ),
    )

    fig.update_yaxes(fixedrange=True, gridcolor='rgba(255,255,255,0.06)', showgrid=True,
                     zeroline=False, showspikes=False, tickfont=dict(size=10))

    return fig


def get_tw_stock_news(code):
    try:
        news_url = f"https://tw.stock.yahoo.com/quote/{code}/news"
        resp = requests.get(news_url, headers=get_headers(), timeout=10)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, 'html.parser')
        news_links = soup.find_all('a', href=True)
        pos_words = ["成長", "新高", "利多", "噴發", "買進", "展望佳", "獲利", "創高", "轉盈", "法說", "漲", "配息", "訂單", "營收亮眼"]
        neg_words = ["衰退", "減少", "利空", "調降", "跌", "虧損", "賣出", "縮減", "保守", "淡季", "壓力", "下修"]
        results, seen_titles = [], set()
        for link in news_links:
            href = link.get('href')
            if '/news/' in href and 'tw.stock.yahoo.com' in href or href.startswith('/news/'):
                title = link.get_text(strip=True)
                if len(title) < 8 or title in seen_titles:
                    continue
                full_link = href if href.startswith('http') else "https://tw.stock.yahoo.com" + href
                sentiment, color = "💡 資訊", "#888888"
                if any(w in title for w in pos_words):
                    sentiment, color = "📈 利多", "#ef5350"
                elif any(w in title for w in neg_words):
                    sentiment, color = "📉 利空", "#26a69a"
                results.append({"title": title, "link": full_link,
                                 "sentiment": sentiment, "color": color, "publisher": "Yahoo股市"})
                seen_titles.add(title)
                if len(results) >= 8:
                    break
        return results
    except:
        return None

# ============================================================
# 3. 全域 CSS（PC + 手機 RWD）
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Noto+Sans+TC:wght@300;400;500;700&display=swap');

/* ── 隱藏 sidebar collapse 按鈕 ── */
[data-testid='stSidebarCollapseButton'],
[data-testid='collapsedControl'] { display: none !important; }

/* ── 隱藏頂部工具列（Share / Star / Fork 按鈕列）── */
[data-testid='stHeader'],
header[data-testid='stHeader'],
[data-testid='stToolbar'] { display: none !important; }

/* ── 補回頂部空間 ── */
.block-container { padding-top: 0.5rem !important; }

/* ── 全域底色 ── */
html, body, [data-testid='stAppViewContainer'], [data-testid='stMain'] {
    background-color: #02080f !important;
    color: #c8d8e8 !important;
}
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
}
[data-testid='stButton'] > button:hover {
    border-color: #00ffc0 !important;
    box-shadow: 0 0 24px rgba(0,255,180,0.25) !important;
    color: #ffffff !important;
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
}

/* ── 進度條 ── */
[data-testid='stProgress'] > div > div {
    background: linear-gradient(90deg, #00b478, #00ffc0, #38a8e8) !important;
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
    width: 14px !important; height: 14px !important;
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

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid rgba(0,200,140,0.12) !important; margin: 16px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #02080f; }
::-webkit-scrollbar-thumb { background: rgba(0,200,140,0.3); border-radius: 2px; }

/* ── 字型 ── */
p, li, .stMarkdown { font-family: 'Noto Sans TC', sans-serif !important; }

/* ══════════════════════════════════════════
   手機 RWD (max-width: 768px)
   ══════════════════════════════════════════ */
@media (max-width: 768px) {

    /* padding 縮小 */
    [data-testid='stMain'] > div { padding: 0 8px !important; }
    .block-container { padding: 0.5rem 0.5rem 2rem !important; max-width: 100% !important; }

    /* Banner 字體縮小 */
    .banner-title { font-size: 15px !important; letter-spacing: 3px !important; }
    .banner-sub   { display: none !important; }          /* 隱藏副標題列 */
    .banner-stats { display: none !important; }          /* 隱藏下方三欄說明 */
    .banner-wrap  { padding: 14px 16px 12px !important; }

    /* 統計卡片：2欄 */
    .stat-grid { grid-template-columns: 1fr 1fr !important; gap: 8px !important; }

    /* K 線圖高度手機版略降 */
    [data-testid='stPlotlyChart'] { min-height: 380px !important; }

    /* 按鈕字體 */
    [data-testid='stButton'] > button {
        font-size: 11px !important;
        letter-spacing: 1px !important;
        padding: 10px 0 !important;
    }

    /* 表格字體縮小 */
    [data-testid='stDataFrame'] * { font-size: 12px !important; }

    /* 新聞卡片 padding 縮小 */
    .news-card { padding: 10px 12px !important; }
    .news-title { font-size: 13px !important; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 4. Banner 頁頭
# ============================================================
_now_str = get_tw_now().strftime("%Y-%m-%d %H:%M")
st.markdown(f"""
<div class="banner-wrap" style="
    background:linear-gradient(135deg,#02080f 0%,#04121e 50%,#02080f 100%);
    border-bottom:1px solid rgba(0,200,140,0.2);
    padding:20px 28px 16px;
    margin-bottom:12px;
    position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,#00ffc0 30%,#38a8e8 70%,transparent);opacity:0.6;"></div>

  <!-- 主標題行 -->
  <div style="display:flex;align-items:center;gap:14px;position:relative;flex-wrap:wrap;">
    <div style="width:44px;height:44px;border:1px solid rgba(0,255,180,0.3);
      border-radius:4px;display:flex;align-items:center;justify-content:center;
      background:linear-gradient(135deg,rgba(0,100,60,0.4),rgba(0,50,30,0.6));
      flex-shrink:0;overflow:hidden;padding:6px;">
      <img src="https://cdn-icons-png.flaticon.com/512/5501/5501377.png" style="width:100%;height:100%;object-fit:contain;">
    </div>
    <div style="flex:1;min-width:180px;">
      <div class="banner-title" style="font-family:Orbitron,monospace;font-size:18px;font-weight:700;
        color:#00ffc0;letter-spacing:4px;text-shadow:0 0 30px rgba(0,255,180,0.5);line-height:1.3;margin-bottom:4px;">
        台股智慧選股系統</div>
      <div class="banner-sub" style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <span style="font-family:Share Tech Mono,monospace;font-size:10px;color:#3a9a7a;letter-spacing:2px;">TAIWAN STOCK SCANNER</span>
        <span style="font-family:Share Tech Mono,monospace;font-size:10px;color:#3a9a7a;letter-spacing:2px;">MA STRATEGY ENGINE v9.9</span>
      </div>
    </div>
    <div style="text-align:right;flex-shrink:0;">
      <div style="font-family:Orbitron,monospace;font-size:10px;color:#00ffc0;
        letter-spacing:2px;border:1px solid rgba(0,255,180,0.2);padding:3px 10px;
        border-radius:2px;background:rgba(0,100,60,0.15);margin-bottom:4px;">MA30 › MA45 › MA60</div>
      <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#3a8060;letter-spacing:1px;">
        {_now_str} TWN +08:00</div>
    </div>
  </div>

  <!-- 下方說明列（手機隱藏）-->
  <div class="banner-stats" style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(0,200,140,0.08);
    display:flex;gap:24px;flex-wrap:wrap;">
    <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a9070;letter-spacing:1px;">
      <span style="color:#00c890;">◆</span> &nbsp;上市 TWSE + 上櫃 TPEX 全市場掃描</div>
    <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a9070;letter-spacing:1px;">
      <span style="color:#00c890;">◆</span> &nbsp;均線多頭排列 · 乖離率過濾 · 成交量門檻</div>
    <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a9070;letter-spacing:1px;">
      <span style="color:#00c890;">◆</span> &nbsp;PE 本益比 · 月增率 · 年增率 財務篩選</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 5. Sidebar 參數設定
# ============================================================
st.sidebar.markdown("""
<div style="padding:8px 0 20px;">
  <div style="font-family:Orbitron,monospace;font-size:11px;font-weight:700;
    color:#00c890;letter-spacing:4px;padding-bottom:14px;
    border-bottom:1px solid rgba(0,200,140,0.15);margin-bottom:18px;">
    ⚙ &nbsp;STRATEGY CONFIG</div>
  <div style="background:rgba(0,80,50,0.2);border:1px solid rgba(0,200,140,0.15);
    border-radius:2px;padding:10px 12px;margin-bottom:18px;">
    <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a9070;letter-spacing:2px;margin-bottom:6px;">SIGNAL CONDITION</div>
    <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#00c890;">MA30 &gt; MA45 &gt; MA60</div>
    <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a9070;letter-spacing:1px;margin-top:4px;">BULLISH ALIGNMENT · BIAS FILTER</div>
  </div>
  <div style="font-family:Noto Sans TC,sans-serif;font-size:13px;font-weight:500;
    color:#70a0b0;letter-spacing:2px;margin-bottom:4px;">📐 &nbsp;參數設定</div>
</div>
""", unsafe_allow_html=True)

user_bias = st.sidebar.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0, step=0.1)
user_vol  = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

# ── 手機版：在主畫面也放一排參數快速設定 ──
with st.expander("⚙ 篩選參數", expanded=False):
    mc1, mc2 = st.columns(2)
    with mc1:
        user_bias = st.number_input("30MA 乖離上限 (%)", 0.1, 15.0, user_bias, step=0.1, key="mb")
    with mc2:
        user_vol = st.slider("最小成交量 (張)", 0, 3000, user_vol, key="mv")

# ── 掃描按鈕 ──
if st.button("🚀 開始全市場智慧掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning = True
    st.session_state.current_idx = 0
    st.session_state.last_selected_row = None
    st.rerun()

# ============================================================
# 6. 掃描流程
# ============================================================
if st.session_state.is_scanning:
    status = st.empty()
    bar = st.progress(0)
    BATCH = 200

    status.text("📋 Step 1/3：載入股票清單...")
    bar.progress(0.03)
    stock_map = get_stock_market_list()
    all_tickers = [s["ticker"] for s in stock_map]
    total_tickers = len(all_tickers)

    history_map = {}
    batches = [all_tickers[i:i+BATCH] for i in range(0, total_tickers, BATCH)]
    for bi, batch in enumerate(batches):
        status.text(f"📥 Step 2/3：批次下載 {bi+1}/{len(batches)}...")
        bar.progress(0.03 + 0.72 * (bi / len(batches)))
        history_map.update(download_batch_history(tuple(batch)))

    bar.progress(0.75)
    status.text("✅ 計算均線中...")
    initial_hits = calc_ma_signals(history_map, stock_map, user_bias, user_vol)
    bar.progress(0.80)

    if initial_hits:
        status.text(f"📈 找到 {len(initial_hits)} 支！抓取財報數據中...")
        final_list = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            f_deep = {ex.submit(fetch_deep_info, r["ticker"]): r for r in initial_hits}
            for j, f in enumerate(as_completed(f_deep), 1):
                bar.progress(0.80 + 0.19 * j / len(initial_hits))
                deep_res = f.result()
                base = f_deep[f]
                final_list.append({
                    "ticker": base["ticker"], "code": base["code"],
                    "name": base["name"], "industry": base["industry"],
                    "收盤": base["收盤"], "乖離30MA(%)": base["乖離30MA(%)"],
                    "成交量(張)": base["成交量(張)"], "量變動(%)": base["量變動(%)"],
                    "本益比": deep_res["pe"], "營收月增": deep_res["mom"], "營收年增": deep_res["yoy"],
                })
        bar.progress(1.0)
        st.session_state.scan_results = pd.DataFrame(final_list)
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("查無條件標的。")

    st.session_state.is_scanning = False
    st.rerun()

# ============================================================
# 7. 結果顯示
# ============================================================
if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results.copy()
    total_found = len(df)
    if st.session_state.current_idx >= total_found:
        st.session_state.current_idx = 0

    # ── 統計卡片（手機 2 欄 / PC 4 欄）──
    st.markdown(f"""
    <div class="stat-grid" style="
        display:grid; grid-template-columns:1fr 1fr 1fr 1fr;
        gap:10px; margin-bottom:14px;">
      <div style="background:linear-gradient(135deg,rgba(0,80,50,0.3),rgba(0,40,25,0.5));
        border:1px solid rgba(0,200,140,0.2);border-radius:2px;padding:12px 14px;border-top:2px solid #00c890;">
        <div style="font-family:Share Tech Mono,monospace;font-size:10px;color:#3a9070;letter-spacing:2px;margin-bottom:4px;">SIGNALS FOUND</div>
        <div style="font-family:Orbitron,monospace;font-size:24px;font-weight:700;color:#00ffc0;">{total_found}</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a9070;">支符合條件標的</div>
      </div>
      <div style="background:linear-gradient(135deg,rgba(0,40,80,0.3),rgba(0,20,50,0.5));
        border:1px solid rgba(56,168,232,0.15);border-radius:2px;padding:12px 14px;border-top:2px solid #38a8e8;">
        <div style="font-family:Share Tech Mono,monospace;font-size:10px;color:#3a6888;letter-spacing:2px;margin-bottom:4px;">STRATEGY</div>
        <div style="font-family:Orbitron,monospace;font-size:12px;font-weight:600;color:#38a8e8;margin-top:4px;">MA BULL</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a6888;">30 › 45 › 60</div>
      </div>
      <div style="background:linear-gradient(135deg,rgba(60,30,0,0.3),rgba(40,20,0,0.5));
        border:1px solid rgba(255,180,0,0.12);border-radius:2px;padding:12px 14px;border-top:2px solid #c8900a;">
        <div style="font-family:Share Tech Mono,monospace;font-size:10px;color:#886820;letter-spacing:2px;margin-bottom:4px;">BIAS LIMIT</div>
        <div style="font-family:Orbitron,monospace;font-size:24px;font-weight:700;color:#e0a020;">{user_bias}<span style="font-size:12px;">%</span></div>
        <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#886820;">30MA 乖離上限</div>
      </div>
      <div style="background:linear-gradient(135deg,rgba(40,0,60,0.3),rgba(20,0,40,0.5));
        border:1px solid rgba(160,100,255,0.12);border-radius:2px;padding:12px 14px;border-top:2px solid #7840c0;">
        <div style="font-family:Share Tech Mono,monospace;font-size:10px;color:#604888;letter-spacing:2px;margin-bottom:4px;">SCAN TIME</div>
        <div style="font-family:Orbitron,monospace;font-size:12px;font-weight:600;color:#a070e0;margin-top:4px;">{get_tw_now().strftime("%H:%M")}</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#604888;">{get_tw_now().strftime("%Y-%m-%d")}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 下載按鈕 ──
    col_spacer, col_dl = st.columns([3, 1])
    with col_dl:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="⬇ EXPORT CSV", data=csv,
                           file_name=f'tw_stock_scan_{get_tw_now().strftime("%Y%m%d")}.csv',
                           mime='text/csv', use_container_width=True)

    # ── 結果表格 ──
    show_cols = ["code", "name", "收盤", "乖離30MA(%)", "成交量(張)", "量變動(%)", "本益比", "營收月增", "營收年增", "industry"]
    available_cols = [c for c in show_cols if c in df.columns]
    df_display = df[available_cols].rename(columns={"code": "代碼", "name": "名稱", "industry": "類股"})

    def color_tw_style(val):
        if pd.isna(val): return ''
        color = '#ef5350' if val > 0 else '#26a69a' if val < 0 else 'white'
        return f'color: {color}; font-weight: bold'

    event = st.dataframe(
        df_display.style.map(
            color_tw_style,
            subset=[c for c in ['量變動(%)', '營收月增', '營收年增'] if c in df_display.columns]
        ),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"stock_table_{st.session_state.table_key}",
        column_config={
            "代碼":        st.column_config.TextColumn("代碼",    width=70),
            "名稱":        st.column_config.TextColumn("名稱",    width=100),
            "收盤":        st.column_config.NumberColumn("價格",  width=75,  format="%.2f"),
            "乖離30MA(%)": st.column_config.ProgressColumn("30MA乖離", width=120,
                                                            help=f"上限 {user_bias}%",
                                                            format="%.2f%%", min_value=0, max_value=user_bias),
            "量變動(%)":   st.column_config.NumberColumn("量變動", width=80, format="%.1f%%"),
            "營收月增":    st.column_config.NumberColumn("月增",   width=75, format="%.1f%%"),
            "營收年增":    st.column_config.NumberColumn("年增",   width=75, format="%.1f%%"),
            "本益比":      st.column_config.NumberColumn("PE",     width=65, format="%.1f"),
            "成交量(張)":  st.column_config.NumberColumn("成交量", width=95, format="%d 📦"),
            "類股":        st.column_config.TextColumn("產業別",  width=120),
        }
    )

    if event and "selection" in event and event["selection"]["rows"]:
        clicked_row = event["selection"]["rows"][0]
        if clicked_row != st.session_state.last_selected_row:
            st.session_state.current_idx = clicked_row
            st.session_state.last_selected_row = clicked_row

    st.markdown(f"""
    <div style="font-family:Share Tech Mono,monospace;font-size:12px;
        color:#4a7a8a;padding:4px;line-height:2;">
        ▸ 進度條滿格 = 乖離率接近上限 ({user_bias}%)
        &nbsp;▸ 紅字正增長 / 綠字負增長
        &nbsp;▸ 點擊列查看K線
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # 8. K 線圖區
    # ============================================================
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:14px 0 6px;
        border-top:1px solid rgba(0,200,140,0.12);margin-top:8px;">
      <div style="width:3px;height:20px;background:linear-gradient(180deg,#00ffc0,#38a8e8);border-radius:2px;flex-shrink:0;"></div>
      <div style="font-family:Orbitron,monospace;font-size:11px;font-weight:600;color:#00c890;letter-spacing:3px;">K-LINE CHART</div>
      <div style="flex:1;height:1px;background:rgba(0,200,140,0.08);"></div>
      <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#3a9070;letter-spacing:2px;">MA30 · MA45 · MA60</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 上一支 / 股票名稱 / 下一支（按鈕在兩側黃框位置）──
    c_idx = st.session_state.current_idx
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col1:
        if st.button("⬅ 上一支", use_container_width=True):
            st.session_state.current_idx = (st.session_state.current_idx - 1) % total_found
            st.session_state.last_selected_row = None
            st.session_state.table_key += 1
            st.rerun()
    with btn_col2:
        st.markdown(f"""
        <div style="text-align:center;font-family:Share Tech Mono,monospace;padding:8px 0;">
            <span style="color:#4a7a8a;font-size:11px;letter-spacing:2px;">[ {c_idx+1} / {total_found} ]</span><br>
            <span style="color:#00ffc8;font-size:16px;font-weight:700;letter-spacing:2px;">{df.iloc[c_idx]['code']}</span>
            <span style="color:#c8d8e8;font-size:14px;margin-left:6px;">{df.iloc[c_idx]['name']}</span>
        </div>
        """, unsafe_allow_html=True)
    with btn_col3:
        if st.button("下一支 ➡", use_container_width=True):
            st.session_state.current_idx = (st.session_state.current_idx + 1) % total_found
            st.session_state.last_selected_row = None
            st.session_state.table_key += 1
            st.rerun()

    current_stock = df.iloc[st.session_state.current_idx]
    k_fig = draw_k_line(current_stock['ticker'], current_stock['name'])
    if k_fig:
        st.plotly_chart(k_fig, use_container_width=True, config={
            "displayModeBar": False,   # 隱藏右上角工具列
            "scrollZoom": False,       # 停用滾輪縮放
            "doubleClick": False,      # 停用雙擊重置
            "showTips": False,
        })
    else:
        st.warning("⚠️ 無法載入 K 線資料，請稍後再試。")

    # ============================================================
    # 9. 新聞
    # ============================================================
    st.markdown(f"""
    <div style="font-family:Share Tech Mono,monospace;font-size:13px;color:#4db8ff;
        letter-spacing:3px;padding:16px 0 8px;border-top:1px solid #4db8ff20;margin-top:8px;">
        📡 &nbsp;LIVE NEWS &nbsp;·&nbsp;
        <span style="color:#c8d8e8;">{current_stock['name']} ({current_stock['code']})</span>
    </div>
    """, unsafe_allow_html=True)

    news_list = get_tw_stock_news(current_stock['code'])
    if news_list:
        for n in news_list:
            badge_bg = (
                "rgba(0,80,40,0.3)" if "利多" in n["sentiment"]
                else "rgba(80,0,20,0.3)" if "利空" in n["sentiment"]
                else "rgba(20,40,60,0.3)"
            )
            st.markdown(f"""
            <div class="news-card" style="
                background:linear-gradient(135deg,rgba(4,14,26,0.8),rgba(2,8,15,0.9));
                border:1px solid rgba(40,80,100,0.2);
                border-left:2px solid {n['color']};
                border-radius:2px;padding:12px 16px;margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:6px;">
                <span style="background:{badge_bg};color:{n['color']};border:1px solid {n['color']}40;
                  padding:2px 10px;border-radius:1px;font-family:Share Tech Mono,monospace;
                  font-size:11px;letter-spacing:1px;">{n['sentiment']}</span>
                <span style="font-family:Share Tech Mono,monospace;color:#4a7080;font-size:11px;">{n['publisher']}</span>
              </div>
              <a class="news-title" href="{n['link']}" target="_blank" style="
                text-decoration:none;color:#c0d8e8;
                font-family:Noto Sans TC,sans-serif;
                font-size:14px;font-weight:400;line-height:1.6;letter-spacing:0.5px;">
                {n['title']}</a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ 無法獲取即時新聞。")

else:
    if not st.session_state.is_scanning:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px 70px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
            width:280px;height:280px;border:1px solid rgba(0,200,140,0.04);border-radius:50%;"></div>
          <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
            width:190px;height:190px;border:1px solid rgba(0,200,140,0.06);border-radius:50%;"></div>
          <div style="position:relative;">
            <div style="font-size:42px;margin-bottom:20px;filter:drop-shadow(0 0 24px rgba(0,255,180,0.5));">📈</div>
            <div style="font-family:Orbitron,monospace;font-size:20px;font-weight:700;
              color:#00ffc0;letter-spacing:5px;text-shadow:0 0 30px rgba(0,255,180,0.4);margin-bottom:8px;">READY TO SCAN</div>
            <div style="width:100px;height:1px;margin:0 auto 16px;
              background:linear-gradient(90deg,transparent,rgba(0,255,192,0.25),transparent);"></div>
            <div style="font-family:Orbitron,monospace;font-size:12px;font-weight:600;
              color:#009060;letter-spacing:2px;margin-bottom:20px;">MA30 › MA45 › MA60</div>
            <div style="display:inline-block;background:rgba(0,50,30,0.3);
              border:1px solid rgba(0,200,140,0.1);border-radius:2px;padding:8px 20px;
              font-family:Share Tech Mono,monospace;font-size:11px;color:#3a8060;letter-spacing:1px;">
              展開上方篩選參數後點擊掃描按鈕</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
