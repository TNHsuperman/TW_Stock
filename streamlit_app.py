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
    page_title="量化智慧選股終端",
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
for key, default in [
    ('scan_results',      pd.DataFrame()),
    ('is_scanning',       False),
    ('current_idx',       0),
    ('last_selected_row', None),
    ('table_key',          0),
    ('user_bias',         3.0),
    ('user_vol',          500),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def get_tw_now():
    return datetime.now(timezone(timedelta(hours=8)))

# ============================================================
# 2. 數據抓取與核心計算
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
                if ' ' in val:
                    code, name = val.split(' ')
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
                "收盤":        round(curr_price, 2),
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

    df['MA30'] = df['close'].rolling(30).mean()
    df['MA45'] = df['close'].rolling(45).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    df = df.tail(70).copy()

    if len(df) < 10:
        return None

    # 台股配色修正：上漲紅、下跌綠
    colors = ['#eb4d4b' if df['close'].iloc[i] >= df['open'].iloc[i] else '#2ecc71' for i in range(len(df))]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.72, 0.28])

    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name=f'{name} ({code})',
        increasing_line_color='#eb4d4b', increasing_fillcolor='#eb4d4b',
        decreasing_line_color='#2ecc71', decreasing_fillcolor='#2ecc71',
        hoverinfo='none', showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'], mode='none', name='', showlegend=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "開盤：%{customdata[0]:.2f}<br>"
            "最高：%{customdata[2]:.2f}<br>"
            "最低：%{customdata[3]:.2f}<br>"
            "收盤：%{customdata[1]:.2f}<br>"
            "─────────────<br>"
            "30MA：%{customdata[4]:.2f}<br>"
            "45MA：%{customdata[5]:.2f}<br>"
            "60MA：%{customdata[6]:.2f}<br>"
            "─────────────<br>"
            "成交：%{customdata[7]:,} 張"
            "<extra></extra>"
        ),
        customdata=df[['open', 'close', 'high', 'low', 'MA30', 'MA45', 'MA60', 'volume']].values,
    ), row=1, col=1)

    # 均線顏色改為低飽和、高辨識度金融配色
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA30'], line=dict(color='#f1c40f', width=1.2), name='30MA', hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA45'], line=dict(color='#3498db', width=1.2), name='45MA', hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], line=dict(color='#9b59b6', width=1.2), name='60MA', hoverinfo='skip'), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'], name='成交量',
        marker_color=colors, showlegend=False,
        hovertemplate="成交量：%{y:,} 張<extra></extra>",
    ), row=2, col=1)

    spike_cfg = dict(
        type='category', showgrid=True, gridcolor='#1e2736',
        zeroline=False, showspikes=True, spikemode='across',
        spikesnap='cursor', spikecolor='#7f8c8d',
        spikethickness=1, spikedash='solid', showline=True, linecolor='#2c3e50'
    )

    fig.update_layout(
        title=dict(
            text=f'{name} ({ticker}) 歷史K線終端',
            font=dict(size=14, color='#f1f2f6', family='Consolas, Noto Sans TC'),
            x=0, xanchor='left'
        ),
        xaxis_rangeslider_visible=False,
        height=480,
        template='plotly_dark',
        paper_bgcolor='#0b0e14',
        plot_bgcolor='#0b0e14',
        font=dict(color="#95a5a6", size=11),
        legend=dict(
            bgcolor='rgba(11,14,20,0.8)',
            bordercolor='#2c3e50', borderwidth=1,
            font=dict(size=11, color="#dcdde1"),
            orientation='h', yanchor='top', y=0.99, xanchor='left', x=0.01,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1c2431', bordercolor='#34495e',
            font=dict(size=11, color='#f1f2f6', family='Consolas'),
            namelength=0,
        ),
        dragmode=False,
        xaxis=dict(**spike_cfg, showticklabels=False),
        xaxis2=dict(**spike_cfg, matches='x', showticklabels=True, tickfont=dict(size=10, color='#7f8c8d')),
    )

    fig.update_yaxes(fixedrange=True, gridcolor='#1e2736', showgrid=True,
                     zeroline=False, showline=True, linecolor='#2c3e50', tickfont=dict(size=10, color='#7f8c8d'))
    return fig

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
            if '/news/' in href and ('tw.stock.yahoo.com' in href or href.startswith('/')):
                title = link.get_text(strip=True)
                if len(title) < 8 or title in seen_titles:
                    continue
                full_link = href if href.startswith('http') else "https://tw.stock.yahoo.com" + href
                sentiment, color = "📋 INFO", "#7f8c8d"
                if any(w in title for w in pos_words):
                    sentiment, color = "▲ BULL", "#eb4d4b"
                elif any(w in title for w in neg_words):
                    sentiment, color = "▼ BEAR", "#2ecc71"
                results.append({"title": title, "link": full_link,
                                 "sentiment": sentiment, "color": color, "publisher": "Yahoo Finance"})
                seen_titles.add(title)
                if len(results) >= 6:
                    break
        return results
    except:
        pass
    return None

# ============================================================
# 3. 全域 CSS（彭博終端 / 專業量化交易風格）
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Noto+Sans+TC:wght@400;500;700&display=swap');

/* 全局背景調色 - Bloomberg 深藍黑基調 */
html, body, [data-testid='stAppViewContainer'], [data-testid='stMain'] {
    background-color: #0b0e14 !important;
    color: #dcdde1 !important;
}

[data-testid='stSidebarCollapseButton'], [data-testid='collapsedControl'],
[data-testid='stHeader'], header[data-testid='stHeader'], [data-testid='stToolbar'] { 
    display: none !important; 
}

.block-container { padding-top: 0rem !important; padding-bottom: 1rem !important; }

/* 側邊欄與元件樣式優化 */
[data-testid='stSidebar'] {
    background-color: #0f141c !important;
    border-right: 1px solid #1e2736 !important;
}
[data-testid='stSidebar'] * { color: #a5b1c2 !important; }
[data-testid='stSidebar'] label { font-size: 11px !important; font-family: 'JetBrains Mono', sans-serif; text-transform: uppercase; }

/* 交易面板按鈕 - 深邃專業色澤 */
[data-testid='stButton'] > button {
    background: #1e2736 !important;
    color: #f1f2f6 !important;
    border: 1px solid #34495e !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', 'Noto Sans TC' !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    padding: 10px 0 !important;
    transition: all 0.2s ease !important;
}
[data-testid='stButton'] > button:hover {
    background: #2c3e50 !important;
    border-color: #4b6584 !important;
    box-shadow: none !important;
}

[data-testid='stDownloadButton'] > button {
    background: #0f141c !important;
    color: #4b6584 !important;
    border: 1px solid #1e2736 !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono' !important;
    font-size: 11px !important;
}
[data-testid='stDownloadButton'] > button:hover {
    border-color: #34495e !important;
    color: #f1f2f6 !important;
}

/* 資料表外框線淡化 */
[data-testid='stDataFrame'] {
    border: 1px solid #1e2736 !important;
}

/* 進度條改為專業單色系 */
[data-testid='stProgress'] > div > div {
    background: #4b6584 !important;
}
[data-testid='stProgress'] > div {
    background: #0f141c !important;
    border: 1px solid #1e2736 !important;
    height: 4px !important;
}

/* 數據輸入元件 */
[data-testid='stNumberInput'] input {
    background: #0f141c !important;
    border: 1px solid #1e2736 !important;
    color: #f1f2f6 !important;
    font-family: 'JetBrains Mono' !important;
}

/* 自訂字體與捲軸 */
p, li, .stMarkdown, span, label { font-family: 'Noto Sans TC', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0b0e14; }
::-webkit-scrollbar-thumb { background: #1e2736; border-radius: 2px; }

/* 響應式優化 */
@media (max-width: 768px) {
    .block-container { padding: 0.5rem !important; }
    [data-testid='stPlotlyChart'] { min-height: 360px !important; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 4. 終端頂部資訊列（Bloomberg Style）
# ============================================================

_now_str = get_tw_now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
<div style="background-color: #0f141c; border-bottom: 1px solid #1e2736; padding: 12px 20px; margin-bottom: 16px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="background: #2c3e50; color: #f1f2f6; font-family: 'JetBrains Mono'; font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 2px;">QUANT TRADING</span>
            <strong style="font-size: 16px; color: #f1f2f6; font-family: 'Noto Sans TC'; letter-spacing: 0.5px;">台股多頭均線篩選終端</strong>
            <span style="font-family: 'JetBrains Mono'; font-size: 11px; color: #57606f;">v10.0.1</span>
        </div>
        <div style="display: flex; align-items: center; gap: 20px; font-family: 'JetBrains Mono'; font-size: 12px;">
            <span style="color: #a5b1c2;"><span style="color: #eb4d4b;">●</span> TWSE / TPEX ACTIVE</span>
            <span style="color: #747d8c;">|</span>
            <span style="color: #f1f2f6;">{_now_str} (UTC+8)</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 5. 策略參數配置
# ============================================================

st.sidebar.markdown("""
<div style="font-family: 'JetBrains Mono'; font-size: 12px; font-weight: 700; color: #f1f2f6; letter-spacing: 1px; margin-bottom: 15px;">
    📊 STRATEGY ENGINE CONFIG
</div>
""", unsafe_allow_html=True)

sb_bias = st.sidebar.number_input("30MA 乖離上限 (%)", 0.1, 15.0, value=st.session_state.user_bias, step=0.1, key="sb_bias")
sb_vol = st.sidebar.slider("最小成交量門檻 (張)", 0, 3000, value=st.session_state.user_vol, key="sb_vol")
st.session_state.user_bias = sb_bias
st.session_state.user_vol  = sb_vol

with st.expander("⚙ FILTER CONFIGURATION (快速篩選面版)", expanded=False):
    mc1, mc2 = st.columns(2)
    with mc1:
        mb_bias = st.number_input("30MA 乖離上限 (%)", 0.1, 15.0, value=st.session_state.user_bias, step=0.1, key="mb_bias")
        st.session_state.user_bias = mb_bias
    with mc2:
        mb_vol = st.slider("最小成交量門檻 (張)", 0, 3000, value=st.session_state.user_vol, key="mb_vol")
        st.session_state.user_vol = mb_vol

user_bias = st.session_state.user_bias
user_vol  = st.session_state.user_vol

if st.button("RUN ENGINE / 執行全市場掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning       = True
    st.session_state.current_idx       = 0
    st.session_state.last_selected_row = None
    st.rerun()

# ============================================================
# 6. 掃描流程與進度控制
# ============================================================

if st.session_state.is_scanning:
    status = st.empty()
    bar    = st.progress(0)
    BATCH  = 200

    status.text("📋 初始化核心引擎：載入上市櫃有價證券名單...")
    bar.progress(0.05)
    stock_map    = get_stock_market_list()
    all_tickers  = [s["ticker"] for s in stock_map]
    total_tickers = len(all_tickers)

    history_map = {}
    batches = [all_tickers[i:i+BATCH] for i in range(0, total_tickers, BATCH)]
    for bi, batch in enumerate(batches):
        status.text(f"📥 正在下載 K 線數據源：批次處理 {bi+1}/{len(batches)}...")
        bar.progress(0.05 + 0.70 * (bi / len(batches)))
        history_map.update(download_batch_history(tuple(batch)))

    bar.progress(0.75)
    status.text("⚡ 正在進行矩陣運算與均線多頭排列過濾...")
    initial_hits = calc_ma_signals(history_map, stock_map, user_bias, user_vol)
    bar.progress(0.80)

    if initial_hits:
        status.text(f"🎯 初篩完成（命中 {len(initial_hits)} 檔）。正在並行查詢基本面本益比與營收動能...")
        final_list = []
        with ThreadPoolExecutor(max_workers=20) as ex:
            f_deep = {ex.submit(fetch_deep_info, r["ticker"]): r for r in initial_hits}
            for j, f in enumerate(as_completed(f_deep), 1):
                bar.progress(0.80 + 0.20 * j / len(initial_hits))
                deep_res = f.result()
                base     = f_deep[f]
                final_list.append({
                    "ticker":   base["ticker"],
                    "code":     base["code"],
                    "name":     base["name"],
                    "industry": base["industry"],
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
        st.warning("核心引擎未發現符合當前參數的標的。")

    st.session_state.is_scanning = False
    st.rerun()

# ============================================================
# 7. 量化數據看板展示
# ============================================================

if not st.session_state.scan_results.empty:
    df          = st.session_state.scan_results.copy()
    total_found = len(df)
    if st.session_state.current_idx >= total_found:
        st.session_state.current_idx = 0

    # ── 金融儀表板數據卡片（極簡現代風格） ──
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 16px;">
        <div style="background: #0f141c; border: 1px solid #1e2736; border-top: 3px solid #eb4d4b; padding: 12px 16px; border-radius: 2px;">
            <div style="font-family: 'JetBrains Mono'; font-size: 10px; color: #747d8c; letter-spacing: 0.5px;">SIGNALS MATCHED</div>
            <div style="font-family: 'JetBrains Mono'; font-size: 24px; font-weight: 700; color: #f1f2f6; margin: 2px 0;">{total_found}</div>
            <div style="font-size: 11px; color: #a5b1c2;">多頭排列型態確認標的</div>
        </div>
        <div style="background: #0f141c; border: 1px solid #1e2736; border-top: 3px solid #3498db; padding: 12px 16px; border-radius: 2px;">
            <div style="font-family: 'JetBrains Mono'; font-size: 10px; color: #747d8c; letter-spacing: 0.5px;">CORE STRATEGY</div>
            <div style="font-family: 'JetBrains Mono'; font-size: 14px; font-weight: 700; color: #3498db; margin: 9px 0 6px 0;">MA BULL ARCH</div>
            <div style="font-size: 11px; color: #a5b1c2;">階梯式：30MA &gt; 45MA &gt; 60MA</div>
        </div>
        <div style="background: #0f141c; border: 1px solid #1e2736; border-top: 3px solid #f1c40f; padding: 12px 16px; border-radius: 2px;">
            <div style="font-family: 'JetBrains Mono'; font-size: 10px; color: #747d8c; letter-spacing: 0.5px;">MAX BIAS THRESHOLD</div>
            <div style="font-family: 'JetBrains Mono'; font-size: 24px; font-weight: 700; color: #f1c40f; margin: 2px 0;">{user_bias}<span style="font-size:13px; font-weight:400;">%</span></div>
            <div style="font-size: 11px; color: #a5b1c2;">當前容忍扣高乖離上限</div>
        </div>
        <div style="background: #0f141c; border: 1px solid #1e2736; border-top: 3px solid #7f8c8d; padding: 12px 16px; border-radius: 2px;">
            <div style="font-family: 'JetBrains Mono'; font-size: 10px; color: #747d8c; letter-spacing: 0.5px;">MIN VOLUME FILTER</div>
            <div style="font-family: 'JetBrains Mono'; font-size: 24px; font-weight: 700; color: #dcdde1; margin: 2px 0;">{user_vol}</div>
            <div style="font-size: 11px; color: #a5b1c2;">五日均量低流動性過濾</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 輸出控制項 ──
    col_spacer, col_dl = st.columns([3, 1])
    with col_dl:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="💾 EXPORT DATA (CSV)", data=csv,
            file_name=f'tw_quant_signals_{get_tw_now().strftime("%Y%m%d")}.csv',
            mime='text/csv', use_container_width=True
        )

    # ── 專業級資料表格配置 ──
    show_cols      = ["code", "name", "收盤", "乖離30MA(%)", "成交量(張)", "量變動(%)", "本益比", "營收月增", "營收年增", "industry"]
    available_cols = [c for c in show_cols if c in df.columns]
    df_display     = df[available_cols].rename(columns={"code": "代碼", "name": "名稱", "industry": "市場板塊"})

    # 金融經典配色：上漲紅、下跌綠
    def color_tw_style(val):
        if pd.isna(val): return ''
        color = '#eb4d4b' if val > 0 else '#2ecc71' if val < 0 else '#dcdde1'
        return f'color: {color}; font-weight: 500; font-family: "JetBrains Mono"'

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
            "名稱":        st.column_config.TextColumn("名稱",    width=90),
            "收盤":        st.column_config.NumberColumn("收盤價",  width=80,  format="%.2f"),
            "乖離30MA(%)": st.column_config.ProgressColumn("30MA 乖離", width=120,
                                                           help=f"閾值上限 {user_bias}%",
                                                           format="%.2f%%", min_value=0, max_value=user_bias),
            "量變動(%)":   st.column_config.NumberColumn("日量變動", width=85, format="%.1f%%"),
            "營收月增":    st.column_config.NumberColumn("營收月增",  width=85, format="%.1f%%"),
            "營收年增":    st.column_config.NumberColumn("營收年增",  width=85, format="%.1f%%"),
            "本益比":      st.column_config.NumberColumn("PE (倍)",     width=75, format="%.1f"),
            "成交量(張)":  st.column_config.NumberColumn("當日成交量", width=100, format="%d"),
            "市場板塊":    st.column_config.TextColumn("產業別",  width=120),
        }
    )

    if event and "selection" in event and event["selection"]["rows"]:
        clicked_row = event["selection"]["rows"][0]
        if clicked_row != st.session_state.last_selected_row:
            st.session_state.current_idx      = clicked_row
            st.session_state.last_selected_row = clicked_row

    st.markdown("""
    <div style="font-family: 'JetBrains Mono', 'Noto Sans TC'; font-size: 11px; color: #747d8c; margin-top: -6px; margin-bottom: 20px;">
        * 註：橫條圖滿格代表乖離率已達上限設定值。點擊行項目可直接聯動下方即時K線圖與個股基本面新聞面資訊。
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # 8. K 線圖分析終端
    # ============================================================

    c_idx = st.session_state.current_idx
    
    st.markdown(f"""
    <div style="border-top: 1px solid #1e2736; padding-top: 16px; margin-top: 16px; display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 3px; height: 14px; background: #3498db; border-radius: 1px;"></div>
            <span style="font-family: 'JetBrains Mono'; font-size: 12px; font-weight: 700; color: #f1f2f6; letter-spacing: 0.5px;">TECHNICAL ANALYSIS TERMINAL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 歷史導航列 ──
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("◀ PREV ASSET", use_container_width=True, key="btn_prev"):
            st.session_state.current_idx      = (st.session_state.current_idx - 1) % total_found
            st.session_state.last_selected_row = None
            st.session_state.table_key        += 1
            st.rerun()
    with nav_col2:
        st.markdown(f"""
        <div style="background: #0f141c; border: 1px solid #1e2736; border-radius: 2px; padding: 6px 12px; text-align: center;">
            <div style="font-family: 'JetBrains Mono'; font-size: 10px; color: #747d8c; margin-bottom: 2px;">
                INDEX {c_idx+1} / {total_found}
            </div>
            <div style="font-family: 'JetBrains Mono', 'Noto Sans TC'; font-size: 15px; font-weight: 700; color: #f1f2f6;">
                {df.iloc[c_idx]['code']} &nbsp;·&nbsp; {df.iloc[c_idx]['name']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with nav_col3:
        if st.button("NEXT ASSET ▶", use_container_width=True, key="btn_next"):
            st.session_state.current_idx      = (st.session_state.current_idx + 1) % total_found
            st.session_state.last_selected_row = None
            st.session_state.table_key        += 1
            st.rerun()

    current_stock = df.iloc[st.session_state.current_idx]
    k_fig = draw_k_line(current_stock['ticker'], current_stock['name'])
    if k_fig:
        st.plotly_chart(k_fig, use_container_width=True, config={
            "displayModeBar": False, "scrollZoom": False, "doubleClick": False, "showTips": False,
        })
    else:
        st.warning("無法從路徑源取得該標的的歷史數據。")

    # ============================================================
    # 9. 新聞與市場情緒 (Bloomberg Layout)
    # ============================================================
    st.markdown(f"""
    <div style="font-family: 'JetBrains Mono'; font-size: 12px; font-weight: 700; color: #f1f2f6; letter-spacing: 0.5px; padding: 12px 0 8px 0; border-top: 1px solid #1e2736; margin-top: 12px;">
        📰 RELEVANT MARKET FEED / {current_stock['name']} ({current_stock['code']})
    </div>
    """, unsafe_allow_html=True)

    news_list = get_tw_stock_news(current_stock['code'])
    if news_list:
        for n in news_list:
            badge_bg = (
                "rgba(235,77,75,0.08)"  if "BULL" in n["sentiment"]
                else "rgba(46,204,113,0.08)" if "BEAR" in n["sentiment"]
                else "rgba(127,140,141,0.08)"
            )
            st.markdown(f"""
            <div style="background: #0f141c; border: 1px solid #1e2736; border-left: 3px solid {n['color']}; border-radius: 2px; padding: 10px 14px; margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="background: {badge_bg}; color: {n['color']}; border: 1px solid {n['color']}30; padding: 1px 6px; border-radius: 2px; font-family: 'JetBrains Mono'; font-size: 10px; font-weight: 700;">{n['sentiment']}</span>
                    <span style="font-family: 'JetBrains Mono'; color: #57606f; font-size: 11px;">{n['publisher']}</span>
                </div>
                <a href="{n['link']}" target="_blank" style="text-decoration: none; color: #dcdde1; font-family: 'Noto Sans TC'; font-size: 13px; font-weight: 400; line-height: 1.4;">
                    {n['title']}
                </a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("該標的當前暫無即時新聞動態。")

else:
    if not st.session_state.is_scanning:
        st.markdown("""
        <div style="text-align: center; padding: 80px 20px; background: #0f141c; border: 1px solid #1e2736; border-radius: 2px; margin-top: 20px;">
            <div style="font-size: 32px; margin-bottom: 12px;">📊</div>
            <div style="font-family: 'JetBrains Mono'; font-size: 16px; font-weight: 700; color: #f1f2f6; letter-spacing: 1px; margin-bottom: 4px;">QUANT TERMINAL STANDBY</div>
            <div style="font-family: 'JetBrains Mono'; font-size: 12px; color: #57606f; margin-bottom: 16px;">STRATEGY: MA30 &gt; MA45 &gt; MA60 BULLISH ALIGNMENT</div>
            <div style="display: inline-block; background: #1e2736; border: 1px solid #34495e; border-radius: 2px; padding: 6px 16px; font-family: 'Noto Sans TC'; font-size: 12px; color: #a5b1c2;">
                請確認或調整上方篩選參數，完成後點擊上方「執行全市場掃描」按鈕載入實時交易數據。
            </div>
        </div>
        """, unsafe_allow_html=True)
