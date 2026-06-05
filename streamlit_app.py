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
    initial_sidebar_state="collapsed",
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
# [修正2] 加入 user_bias / user_vol 的 session_state 預設值，避免雙重定義衝突
for key, default in [
    ('scan_results',      pd.DataFrame()),
    ('is_scanning',       False),
    ('current_idx',       0),
    ('last_selected_row', None),
    ('table_key',         0),
    ('user_bias',         3.0),
    ('user_vol',          500),
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

    # [修正3] 下載失敗時提前返回，避免後續 KeyError
    if raw is None or raw.empty:
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
        closes  = df["close"]
        volumes = df["volume"]
        if volumes.tail(5).mean() < vol_limit:
            continue
        ma30 = closes.rolling(30).mean().iloc[-1]
        ma45 = closes.rolling(45).mean().iloc[-1]
        ma60 = closes.rolling(60).mean().iloc[-1]
        curr_price   = float(closes.iloc[-1])
        vol_today    = int(volumes.iloc[-1])
        vol_yesterday = float(volumes.iloc[-2])
        bias_30   = ((curr_price - ma30) / ma30) * 100
        vol_change = ((vol_today - vol_yesterday) / vol_yesterday * 100) if vol_yesterday > 0 else 0
        if (ma30 > ma45 > ma60) and (0 <= bias_30 <= bias_limit):
            hits.append({**s,
                "收盤":       round(curr_price, 2),
                "乖離30MA(%)": round(bias_30, 2),
                "成交量(張)":  vol_today,
                "量變動(%)":   round(vol_change, 2),
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
    res  = {"pe": np.nan, "mom": np.nan, "yoy": np.nan}
    try:
        yt = yf.Ticker(ticker)
        pe = yt.info.get('trailingPE')
        if pe:
            res["pe"] = float(pe)
    except:
        pass
    try:
        rev_url  = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
        rev_resp = requests.get(rev_url, headers=get_headers(), timeout=10)
        soup     = BeautifulSoup(rev_resp.text, 'html.parser')

        # [修正4] 原 r'li.List(n)' CSS 選擇器含括號，BeautifulSoup 不支援
        # 改用 find_all + lambda 模糊匹配含 'List' 的 class
        list_items = soup.find_all('li', class_=lambda c: c and 'List' in ' '.join(c) if isinstance(c, list) else c and 'List' in c)
        row = list_items[0] if list_items else None

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
    now  = get_tw_now()
    months = 6
    if market == "TW":
        for delta in range(months):
            # [修正8] 使用精確的年月計算，避免 timedelta(days=30) 重複月份
            month_offset = now.month - delta
            year_offset  = now.year + (month_offset - 1) // 12
            month_val    = (month_offset - 1) % 12 + 1
            yyyymm = f"{year_offset}{month_val:02d}01"
            url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={yyyymm}&stockNo={code}"
            try:
                r = requests.get(url, headers=get_headers(), timeout=10)
                for row in r.json().get("data", []):
                    try:
                        yy, mm, dd = str(row[0]).split("/")
                        rows.append({
                            "date":   f"{int(yy)+1911}-{mm}-{dd}",
                            "open":   float(str(row[3]).replace(",", "")),
                            "high":   float(str(row[4]).replace(",", "")),
                            "low":    float(str(row[5]).replace(",", "")),
                            "close":  float(str(row[6]).replace(",", "")),
                            "volume": int(str(row[1]).replace(",", "")) // 1000,
                        })
                    except:
                        pass
            except:
                pass
    else:
        for delta in range(months):
            month_offset = now.month - delta
            year_offset  = now.year + (month_offset - 1) // 12
            month_val    = (month_offset - 1) % 12 + 1
            roc_ym = f"{year_offset - 1911}/{month_val:02d}"
            url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d={roc_ym}&stkno={code}"
            try:
                r = requests.get(url, headers=get_headers(), timeout=10)
                for row in r.json().get("aaData", []):
                    try:
                        yy, mm, dd = str(row[0]).split("/")
                        rows.append({
                            "date":   f"{int(yy)+1911}-{mm}-{dd}",
                            "open":   float(str(row[3]).replace(",", "")),
                            "high":   float(str(row[4]).replace(",", "")),
                            "low":    float(str(row[5]).replace(",", "")),
                            "close":  float(str(row[6]).replace(",", "")),
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
    code   = ticker.split(".")[0]
    market = "TW" if ticker.endswith(".TW") else "TWO"
    df     = get_kline_data(code, market)
    if df.empty or len(df) < 30:
        yt  = yf.Ticker(ticker)
        raw = yt.history(period="6mo")
        if raw.empty:
            return None
        df = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                  "Close": "close", "Volume": "volume"}).reset_index()
        df["date"]   = df["Date"].dt.strftime("%Y-%m-%d")
        df["volume"] = df["volume"] // 1000

    # [修正7] 先用全部資料計算 MA，再 tail(70) 截取顯示範圍
    # 避免 tail(70) 後 MA60 只有最後 11 天有值的問題
    df['MA30'] = df['close'].rolling(30).mean()
    df['MA45'] = df['close'].rolling(45).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    df = df.tail(70).copy()

    if len(df) < 10:
        return None

    colors = ['#22ab94' if df['close'].iloc[i] >= df['open'].iloc[i]
              else '#f23645' for i in range(len(df))]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name=f'{name} ({code})',
        increasing_line_color='#22ab94', decreasing_line_color='#f23645',
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

    fig.add_trace(go.Scatter(x=df['date'], y=df['MA30'],
                             line=dict(color='#2962ff', width=1.7), name='30MA', hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA45'],
                             line=dict(color='#f9a825', width=1.7), name='45MA', hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'],
                             line=dict(color='#ab47bc', width=1.7), name='60MA', hoverinfo='skip'), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'], name='成交量',
        marker_color=colors, showlegend=False,
        hovertemplate="<b>%{x}</b><br>成交量：%{y:,} 張<extra></extra>",
    ), row=2, col=1)

    spike_cfg = dict(
        type='category', showgrid=True, gridcolor='rgba(255,255,255,0.055)',
        zeroline=False, showspikes=True, spikemode='across',
        spikesnap='cursor', spikecolor='rgba(41,98,255,0.55)',
        spikethickness=1, spikedash='dot', showline=False,
    )

    fig.update_layout(
        title=dict(
            text=f'{name}．{ticker}',
            font=dict(size=14, color='#e6edf3', family='Noto Sans TC, sans-serif'),
            x=0, xanchor='left', pad=dict(l=8, t=4),
        ),
        xaxis_rangeslider_visible=False,
        height=500,
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color="#8b949e", size=11),
        legend=dict(
            bgcolor='rgba(13,17,23,0.88)',
            bordercolor='rgba(48,54,61,0.9)', borderwidth=1,
            font=dict(size=10, color="#c9d1d9"),
            orientation='h',
            yanchor='bottom', y=0.29,
            xanchor='left', x=0.01,
        ),
        margin=dict(l=8, r=8, t=36, b=8),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#161b22', bordercolor='rgba(41,98,255,0.45)',
            font=dict(size=11, color='#e6edf3', family='Share Tech Mono, monospace'),
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
            showticklabels=False,
        ),
    )

    fig.update_yaxes(fixedrange=True, gridcolor='rgba(255,255,255,0.055)', showgrid=True,
                     zeroline=False, showspikes=False, tickfont=dict(size=10))
    return fig

# [修正 新增] 新聞加上快取，避免每次切換股票都重新爬取
@st.cache_data(ttl=300)
def get_tw_stock_news(code):
    try:
        news_url = f"https://tw.stock.yahoo.com/quote/{code}/news"
        resp     = requests.get(news_url, headers=get_headers(), timeout=10)
        if resp.status_code != 200:
            return None
        soup       = BeautifulSoup(resp.text, 'html.parser')
        news_links = soup.find_all('a', href=True)
        pos_words  = ["成長", "新高", "利多", "噴發", "買進", "展望佳", "獲利", "創高", "轉盈", "法說", "漲", "配息", "訂單", "營收亮眼"]
        neg_words  = ["衰退", "減少", "利空", "調降", "跌", "虧損", "賣出", "縮減", "保守", "淡季", "壓力", "下修"]
        results, seen_titles = [], set()
        for link in news_links:
            href = link.get('href')
            if not href:
                continue
            # [修正5] 修正 href 條件判斷邏輯，加括號明確優先順序
            if '/news/' in href and ('tw.stock.yahoo.com' in href or href.startswith('/')):
                title = link.get_text(strip=True)
                if len(title) < 8 or title in seen_titles:
                    continue
                full_link = href if href.startswith('http') else "https://tw.stock.yahoo.com" + href
                sentiment, color = "💡 資訊", "#8b949e"
                if any(w in title for w in pos_words):
                    sentiment, color = "📈 利多", "#22ab94"
                elif any(w in title for w in neg_words):
                    sentiment, color = "📉 利空", "#f23645"
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500;600&family=Noto+Sans+TC:wght@300;400;500;700&display=swap');

:root {
    --tv-bg: #0d1117;
    --tv-panel: #161b22;
    --tv-panel-2: #111820;
    --tv-border: #30363d;
    --tv-text: #e6edf3;
    --tv-muted: #8b949e;
    --tv-blue: #2962ff;
    --tv-blue-soft: rgba(41, 98, 255, 0.16);
    --tv-green: #22ab94;
    --tv-red: #f23645;
    --tv-yellow: #f9a825;
}

[data-testid='stSidebarCollapseButton'],
[data-testid='collapsedControl'],
[data-testid='stHeader'],
header[data-testid='stHeader'],
[data-testid='stToolbar'] { display: none !important; }

html, body, [data-testid='stAppViewContainer'], [data-testid='stMain'] {
    background: var(--tv-bg) !important;
    color: var(--tv-text) !important;
    font-family: 'Inter','Noto Sans TC',sans-serif !important;
}
.block-container { padding-top: 0.75rem !important; max-width: 1500px !important; }

[data-testid='stSidebar'] {
    background: #0b1118 !important;
    border-right: 1px solid var(--tv-border) !important;
}
[data-testid='stSidebar'] * { color: var(--tv-muted) !important; }
[data-testid='stSidebar'] label { font-size: 12px !important; font-weight: 600 !important; letter-spacing: .3px !important; }

[data-testid='stButton'] > button,
[data-testid='stDownloadButton'] > button {
    background: #161b22 !important;
    color: var(--tv-text) !important;
    border: 1px solid var(--tv-border) !important;
    border-radius: 8px !important;
    font-family: 'Inter','Noto Sans TC',sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: .2px !important;
    padding: 11px 0 !important;
    transition: all .18s ease !important;
}
[data-testid='stButton'] > button:hover,
[data-testid='stDownloadButton'] > button:hover {
    border-color: var(--tv-blue) !important;
    background: #1b2430 !important;
    box-shadow: 0 0 0 3px rgba(41,98,255,.14) !important;
}
[data-testid='stButton'] > button:disabled { opacity: .35 !important; }

[data-testid='stDataFrame'] {
    background: var(--tv-panel) !important;
    border: 1px solid var(--tv-border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid='stDataFrame'] * {
    font-family: 'Roboto Mono','Noto Sans TC',monospace !important;
}

[data-testid='stProgress'] > div { background: #111820 !important; border-radius: 999px !important; height: 7px !important; }
[data-testid='stProgress'] > div > div { background: linear-gradient(90deg, var(--tv-blue), #4f8cff) !important; border-radius: 999px !important; }

[data-testid='stAlert'] {
    background: rgba(41,98,255,.08) !important;
    border: 1px solid rgba(41,98,255,.24) !important;
    border-left: 4px solid var(--tv-blue) !important;
    border-radius: 8px !important;
}

[data-testid='stSlider'] div[role='slider'] { background: var(--tv-blue) !important; }
[data-testid='stNumberInput'] input {
    background: #0d1117 !important;
    border: 1px solid var(--tv-border) !important;
    color: var(--tv-text) !important;
    border-radius: 8px !important;
    font-family: 'Roboto Mono',monospace !important;
}

.tv-banner, .tv-card, .tv-panel {
    background: linear-gradient(180deg, #161b22 0%, #111820 100%);
    border: 1px solid var(--tv-border);
    border-radius: 12px;
    box-shadow: 0 12px 30px rgba(0,0,0,.18);
}
.tv-banner { padding: 18px 22px; margin-bottom: 14px; }
.tv-title { font-size: 22px; font-weight: 800; color: var(--tv-text); letter-spacing: .2px; }
.tv-sub { color: var(--tv-muted); font-size: 12px; margin-top: 3px; }
.tv-pill { border: 1px solid var(--tv-border); background: rgba(41,98,255,.10); color: #8fb2ff; border-radius: 999px; padding: 5px 10px; font-size: 12px; font-weight: 700; }
.tv-card { padding: 14px 16px; }
.tv-label { color: var(--tv-muted); font-size: 11px; font-weight: 700; letter-spacing: .6px; text-transform: uppercase; }
.tv-value { color: var(--tv-text); font-family: 'Roboto Mono',monospace; font-size: 26px; font-weight: 700; margin-top: 5px; }
.tv-caption { color: var(--tv-muted); font-size: 12px; margin-top: 4px; }
.tv-section { color: var(--tv-text); font-size: 13px; font-weight: 800; letter-spacing: .5px; margin: 18px 0 9px; padding-left: 10px; border-left: 4px solid var(--tv-blue); }
.news-card { background: #161b22 !important; border: 1px solid var(--tv-border) !important; border-radius: 10px !important; }
.news-title:hover { color: #8fb2ff !important; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 999px; }

@media (max-width: 768px) {
    .block-container { padding: .6rem .65rem 2rem !important; }
    .banner-stats { display: none !important; }
    .stat-grid { grid-template-columns: 1fr 1fr !important; gap: 8px !important; }
    .tv-title { font-size: 17px !important; }
    [data-testid='stPlotlyChart'] { min-height: 390px !important; }
    [data-testid='stDataFrame'] * { font-size: 12px !important; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 4. Banner 頁頭
# ============================================================

_now_str = get_tw_now().strftime("%Y-%m-%d %H:%M")
st.markdown(f"""
<div class="tv-banner">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;">
    <div>
      <div class="tv-title">Taiwan Stock Screener Pro</div>
      <div class="tv-sub">Institutional MA Strategy Terminal · TWSE / TPEX · {_now_str} GMT+8</div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
      <span class="tv-pill">MA30 &gt; MA45 &gt; MA60</span>
      <span class="tv-pill">Bias Filter</span>
      <span class="tv-pill">Volume Gate</span>
    </div>
  </div>
  <div class="banner-stats" style="display:flex;gap:18px;flex-wrap:wrap;margin-top:14px;padding-top:13px;border-top:1px solid #30363d;">
    <div class="tv-caption">◆ 全市場智慧掃描</div>
    <div class="tv-caption">◆ 技術面多頭排列</div>
    <div class="tv-caption">◆ PE / MoM / YoY 財務濾網</div>
    <div class="tv-caption">◆ K 線與即時新聞整合</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 5. Sidebar 參數設定
# ============================================================

st.sidebar.markdown("""
<div style="padding:8px 0 18px;">
  <div style="font-family:Inter,Noto Sans TC,sans-serif;font-size:12px;font-weight:800;color:#e6edf3;letter-spacing:.5px;padding-bottom:12px;border-bottom:1px solid #30363d;margin-bottom:14px;">
    ⚙ STRATEGY CONFIG
  </div>
  <div class="tv-panel" style="padding:12px 13px;margin-bottom:16px;box-shadow:none;">
    <div class="tv-label">Signal Condition</div>
    <div style="font-family:Roboto Mono,monospace;font-size:13px;font-weight:700;color:#8fb2ff;margin-top:7px;">MA30 &gt; MA45 &gt; MA60</div>
    <div class="tv-caption">Bullish alignment · Bias filter</div>
  </div>
  <div class="tv-label">篩選參數</div>
</div>
""", unsafe_allow_html=True)

# [修正2] 統一用 session_state 管理參數，sidebar 與主畫面 expander 共用同一份值
sb_bias = st.sidebar.number_input(
    "30MA 乖離上限 (%)", 0.1, 15.0,
    value=st.session_state.user_bias, step=0.1, key="sb_bias"
)
sb_vol = st.sidebar.slider(
    "最小成交量 (張)", 0, 3000,
    value=st.session_state.user_vol, key="sb_vol"
)
st.session_state.user_bias = sb_bias
st.session_state.user_vol  = sb_vol

# ── 手機版：主畫面快速設定（與 sidebar 同步）──
with st.expander("⚙ 篩選參數", expanded=False):
    mc1, mc2 = st.columns(2)
    with mc1:
        mb_bias = st.number_input(
            "30MA 乖離上限 (%)", 0.1, 15.0,
            value=st.session_state.user_bias, step=0.1, key="mb_bias"
        )
        st.session_state.user_bias = mb_bias
    with mc2:
        mb_vol = st.slider(
            "最小成交量 (張)", 0, 3000,
            value=st.session_state.user_vol, key="mb_vol"
        )
        st.session_state.user_vol = mb_vol

# 統一讀取最終值
user_bias = st.session_state.user_bias
user_vol  = st.session_state.user_vol

# ── 掃描按鈕 ──
if st.button("🚀 開始全市場智慧掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning      = True
    st.session_state.current_idx      = 0
    st.session_state.last_selected_row = None
    st.rerun()

# ============================================================
# 6. 掃描流程
# ============================================================

if st.session_state.is_scanning:
    status = st.empty()
    bar    = st.progress(0)
    BATCH  = 200

    status.text("📋 Step 1/3：載入股票清單...")
    bar.progress(0.03)
    stock_map    = get_stock_market_list()
    all_tickers  = [s["ticker"] for s in stock_map]
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
        with ThreadPoolExecutor(max_workers=20) as ex:
            f_deep = {ex.submit(fetch_deep_info, r["ticker"]): r for r in initial_hits}
            for j, f in enumerate(as_completed(f_deep), 1):
                bar.progress(0.80 + 0.19 * j / len(initial_hits))
                deep_res = f.result()
                base     = f_deep[f]
                final_list.append({
                    "ticker":   base["ticker"],
                    "code":     base["code"],
                    "name":     base["name"],
                    "industry": base["industry"],
                    "收盤":       base["收盤"],
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
# 7. 結果顯示
# ============================================================

if not st.session_state.scan_results.empty:
    df          = st.session_state.scan_results.copy()
    total_found = len(df)
    if st.session_state.current_idx >= total_found:
        st.session_state.current_idx = 0

    avg_pe = df['本益比'].dropna().mean() if '本益比' in df.columns else np.nan
    avg_yoy = df['營收年增'].dropna().mean() if '營收年增' in df.columns else np.nan
    avg_pe_txt = f"{avg_pe:.1f}" if pd.notna(avg_pe) else "N/A"
    avg_yoy_txt = f"{avg_yoy:.1f}%" if pd.notna(avg_yoy) else "N/A"

    # ── 統計卡片 ──
    st.markdown(f"""
    <div class="stat-grid" style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px;">
      <div class="tv-card">
        <div class="tv-label">Total Signals</div>
        <div class="tv-value" style="color:#8fb2ff;">{total_found}</div>
        <div class="tv-caption">符合條件標的</div>
      </div>
      <div class="tv-card">
        <div class="tv-label">Strategy</div>
        <div class="tv-value" style="font-size:20px;color:#22ab94;">MA BULL</div>
        <div class="tv-caption">30MA &gt; 45MA &gt; 60MA</div>
      </div>
      <div class="tv-card">
        <div class="tv-label">Avg PE / Avg YoY</div>
        <div class="tv-value" style="font-size:20px;">{avg_pe_txt} <span style="color:#8b949e;font-size:14px;">/</span> {avg_yoy_txt}</div>
        <div class="tv-caption">財務估值與成長概況</div>
      </div>
      <div class="tv-card">
        <div class="tv-label">Bias / Volume</div>
        <div class="tv-value" style="font-size:20px;color:#f9a825;">{user_bias}% / {user_vol}</div>
        <div class="tv-caption">30MA 乖離上限 / 最小成交量</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 下載按鈕 ──
    col_spacer, col_dl = st.columns([3, 1])
    with col_dl:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="⬇ EXPORT CSV", data=csv,
            file_name=f'tw_stock_scan_{get_tw_now().strftime("%Y%m%d")}.csv',
            mime='text/csv', use_container_width=True
        )

    # ── 結果表格 ──
    show_cols      = ["code", "name", "收盤", "乖離30MA(%)", "成交量(張)", "量變動(%)", "本益比", "營收月增", "營收年增", "industry"]
    available_cols = [c for c in show_cols if c in df.columns]
    df_display     = df[available_cols].rename(columns={"code": "代碼", "name": "名稱", "industry": "類股"})

    def color_tw_style(val):
        if pd.isna(val): return ''
        color = '#22ab94' if val > 0 else '#f23645' if val < 0 else '#e6edf3'
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
            "成交量(張)":  st.column_config.NumberColumn("成交量", width=95, format="%d"),
            "類股":        st.column_config.TextColumn("產業別",  width=120),
        }
    )

    if event and "selection" in event and event["selection"]["rows"]:
        clicked_row = event["selection"]["rows"][0]
        if clicked_row != st.session_state.last_selected_row:
            st.session_state.current_idx      = clicked_row
            st.session_state.last_selected_row = clicked_row

    st.markdown(f"""
    <div class="tv-caption" style="padding:6px 2px 2px;">
        進度條滿格代表 30MA 乖離接近上限 {user_bias}%；綠字為正增長，紅字為負增長；點擊任一列可切換 K 線。
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # 8. K 線圖區
    # ============================================================

    st.markdown("""
    <div class="tv-section">K-LINE CHART · MA30 / MA45 / MA60</div>
    """, unsafe_allow_html=True)

    # ── 導航列 ──
    c_idx = st.session_state.current_idx
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("⬅ PREV", use_container_width=True, key="btn_prev"):
            st.session_state.current_idx      = (st.session_state.current_idx - 1) % total_found
            st.session_state.last_selected_row = None
            st.session_state.table_key        += 1
            st.rerun()
    with nav_col2:
        st.markdown(f"""
        <div class="tv-panel" style="padding:9px 12px;text-align:center;">
          <div class="tv-label">SIGNAL {c_idx+1} / {total_found}</div>
          <div style="display:flex;align-items:center;justify-content:center;gap:10px;margin-top:4px;">
            <span style="font-family:Roboto Mono,monospace;font-size:20px;font-weight:800;color:#8fb2ff;">{df.iloc[c_idx]['code']}</span>
            <span style="color:#30363d;">|</span>
            <span style="font-family:Noto Sans TC,sans-serif;font-size:15px;font-weight:700;color:#e6edf3;">{df.iloc[c_idx]['name']}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with nav_col3:
        # [修正1] 移除重複的 dead code，只保留一次 rerun
        if st.button("NEXT ➡", use_container_width=True, key="btn_next"):
            st.session_state.current_idx      = (st.session_state.current_idx + 1) % total_found
            st.session_state.last_selected_row = None
            st.session_state.table_key        += 1
            st.rerun()

    current_stock = df.iloc[st.session_state.current_idx]
    k_fig = draw_k_line(current_stock['ticker'], current_stock['name'])
    if k_fig:
        st.plotly_chart(k_fig, use_container_width=True, config={
            "displayModeBar": False,
            "scrollZoom":     False,
            "doubleClick":    False,
            "showTips":       False,
        })
    else:
        st.warning("⚠️ 無法載入 K 線資料，請稍後再試。")

    # ============================================================
    # 9. 新聞
    # ============================================================
    st.markdown(f"""
    <div class="tv-section">LIVE NEWS · {current_stock['name']} ({current_stock['code']})</div>
    """, unsafe_allow_html=True)

    news_list = get_tw_stock_news(current_stock['code'])
    if news_list:
        for n in news_list:
            badge_bg = (
                "rgba(34,171,148,0.14)"  if "利多" in n["sentiment"]
                else "rgba(242,54,69,0.14)" if "利空" in n["sentiment"]
                else "rgba(41,98,255,0.12)"
            )
            st.markdown(f"""
            <div class="news-card" style="
                background:linear-gradient(135deg,rgba(4,14,26,0.8),rgba(2,8,15,0.9));
                border:1px solid rgba(40,80,100,0.2);
                border-left:2px solid {n['color']};
                border-radius:10px;padding:13px 16px;margin-bottom:9px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:6px;">
                <span style="background:{badge_bg};color:{n['color']};border:1px solid {n['color']}40;
                  padding:2px 10px;border-radius:999px;font-family:Roboto Mono,monospace;
                  font-size:11px;letter-spacing:1px;">{n['sentiment']}</span>
                <span style="font-family:Roboto Mono,monospace;color:#8b949e;font-size:11px;">{n['publisher']}</span>
              </div>
              <a class="news-title" href="{n['link']}" target="_blank" style="
                text-decoration:none;color:#e6edf3;
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
        <div class="tv-panel" style="text-align:center;padding:56px 22px;margin-top:16px;">
          <div style="font-size:42px;margin-bottom:16px;">📈</div>
          <div class="tv-title">Ready to Scan</div>
          <div class="tv-sub" style="margin-top:8px;">MA30 &gt; MA45 &gt; MA60 · Bias Filter · Volume Gate</div>
          <div style="margin-top:18px;display:inline-block;" class="tv-pill">展開上方篩選參數後點擊掃描按鈕</div>
        </div>
        """, unsafe_allow_html=True)
