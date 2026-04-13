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
st.set_page_config(page_title="台股智慧選股儀表板 v9.9", layout="wide")

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
# 2-A. 批次取得今日全市場行情（TWSE + TPEX）
# ============================================================

@st.cache_data(ttl=1800)  # 快取 30 分鐘，避免重複打 API
def get_today_quote_batch() -> dict:
    """
    一次拿回上市＋上櫃所有股票今日收盤價與成交量。
    回傳 dict: { "2330": {"close": 910.0, "volume": 45000, "market": "TW"}, ... }
    非交易日或抓取失敗則回傳空 dict。
    """
    result = {}
    today = get_tw_now().strftime("%Y%m%d")

    # ── 上市（TWSE）──
    try:
        url_twse = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={today}&type=ALL"
        r = requests.get(url_twse, headers=get_headers(), timeout=15)
        j = r.json()
        # 表格 9 = 個股行情（欄位：證券代號/名稱/成交股數/收盤 ...）
        for table in j.get("tables", []):
            fields = table.get("fields", [])
            if "收盤價" in fields and "成交股數" in fields:
                ci = fields.index("收盤價")
                vi = fields.index("成交股數")
                ni = fields.index("證券代號")
                for row in table.get("data", []):
                    code = row[ni].strip()
                    if not (len(code) == 4 and code.isdigit()):
                        continue
                    try:
                        close  = float(row[ci].replace(",", ""))
                        volume = int(row[vi].replace(",", "")) // 1000  # 股→張
                        result[code] = {"close": close, "volume": volume, "market": "TW"}
                    except: pass
    except: pass

    # ── 上櫃（TPEX）──
    try:
        d = get_tw_now()
        roc_date = f"{d.year - 1911}/{d.month:02d}/{d.day:02d}"
        url_tpex = f"https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php?l=zh-tw&d={roc_date}&se=AL"
        r = requests.get(url_tpex, headers=get_headers(), timeout=15)
        j = r.json()
        for row in j.get("aaData", []):
            # row: [代號, 名稱, 收盤, 漲跌, 開盤, 最高, 最低, 成交股數, ...]
            try:
                code   = str(row[0]).strip()
                if not (len(code) == 4 and code.isdigit()): continue
                close  = float(str(row[2]).replace(",", ""))
                volume = int(str(row[7]).replace(",", "")) // 1000  # 股→張
                result[code] = {"close": close, "volume": volume, "market": "TWO"}
            except: pass
    except: pass

    return result


@st.cache_data(ttl=86400)
def get_twse_history_batch(code: str) -> pd.DataFrame:
    """
    從 TWSE 抓單支上市股票近 3 個月逐日收盤，不限流、穩定。
    回傳 DataFrame columns: [date, close, volume(張)]
    """
    rows = []
    now = get_tw_now()
    # 抓最近 3 個月（每月一次請求，共 3 次）
    for delta_month in range(3):
        d = now - timedelta(days=30 * delta_month)
        yyyymm = f"{d.year}{d.month:02d}01"
        url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={yyyymm}&stockNo={code}"
        try:
            r = requests.get(url, headers=get_headers(), timeout=10)
            j = r.json()
            for row in j.get("data", []):
                try:
                    # row: [日期, 成交股數, 成交金額, 開盤, 最高, 最低, 收盤, 漲跌, 成交筆數]
                    close  = float(str(row[6]).replace(",", ""))
                    volume = int(str(row[1]).replace(",", "")) // 1000
                    rows.append({"close": close, "volume": volume})
                except: pass
        except: pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows[::-1])  # 最新在前 → 反轉為舊→新


@st.cache_data(ttl=86400)
def get_tpex_history_batch(code: str) -> pd.DataFrame:
    """
    從 TPEX 抓單支上櫃股票近 3 個月逐日收盤。
    """
    rows = []
    now = get_tw_now()
    for delta_month in range(3):
        d = now - timedelta(days=30 * delta_month)
        roc_ym = f"{d.year - 1911}/{d.month:02d}"
        url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d={roc_ym}&stkno={code}"
        try:
            r = requests.get(url, headers=get_headers(), timeout=10)
            j = r.json()
            for row in j.get("aaData", []):
                try:
                    close  = float(str(row[6]).replace(",", ""))
                    volume = int(str(row[1]).replace(",", "")) // 1000
                    rows.append({"close": close, "volume": volume})
                except: pass
        except: pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows[::-1])


def check_ma_condition(code: str, market: str, today_close: float,
                       today_vol: int, bias_limit: float) -> dict | None:
    """
    用官方 API 歷史資料判斷均線條件，完全不依賴 Yahoo Finance。
    """
    df_h = get_twse_history_batch(code) if market == "TW" else get_tpex_history_batch(code)
    if df_h.empty or len(df_h) < 62:
        return None

    closes  = df_h["close"]
    volumes = df_h["volume"]

    ma30 = closes.rolling(30).mean().iloc[-1]
    ma45 = closes.rolling(45).mean().iloc[-1]
    ma60 = closes.rolling(60).mean().iloc[-1]

    curr_price = today_close if today_close else closes.iloc[-1]
    bias_30    = ((curr_price - ma30) / ma30) * 100

    if not ((ma30 > ma45 > ma60) and (0 <= bias_30 <= bias_limit)):
        return None

    vol_yesterday = float(volumes.iloc[-2])
    vol_change    = ((today_vol - vol_yesterday) / vol_yesterday * 100) if vol_yesterday > 0 else 0

    ticker = f"{code}.{market}"
    return {
        "ticker": ticker,
        "收盤": round(curr_price, 2),
        "乖離30MA(%)": round(bias_30, 2),
        "成交量(張)": int(today_vol),
        "量變動(%)": round(vol_change, 2),
    }


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

# ============================================================
# 3. 技術分析、繪圖與中文新聞抓取
# ============================================================

def run_strategy_check(s, bias_limit, vol_limit):
    now_ts = int(get_tw_now().timestamp())
    start_ts = int((get_tw_now() - timedelta(days=250)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}"
    params = {"period1": start_ts, "period2": now_ts, "interval": "1d"}

    # ★ 最多重試 3 次，每次失敗後等待再重試，解決限流與 timeout 造成的結果不穩定
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=get_headers(), timeout=20)
            # 被限流時 Yahoo 回傳 429，等待後重試
            if r.status_code == 429:
                time.sleep(2 + attempt * 2)
                continue
            data = r.json()['chart']['result'][0]
            c_series = pd.Series(data['indicators']['quote'][0]['close']).ffill().dropna()
            v_series = pd.Series(data['indicators']['quote'][0]['volume']).ffill().dropna()
            if len(c_series) < 65: return None

            vol_today     = v_series.iloc[-1]
            vol_yesterday = v_series.iloc[-2]
            vol_change    = ((vol_today - vol_yesterday) / vol_yesterday) * 100 if vol_yesterday > 0 else 0
            curr_vol_int  = int(vol_today / 1000)
            avg_vol_5d    = v_series.tail(5).mean() / 1000
            if avg_vol_5d < vol_limit: return None

            ma30  = c_series.rolling(30).mean().iloc[-1]
            ma45  = c_series.rolling(45).mean().iloc[-1]
            ma60  = c_series.rolling(60).mean().iloc[-1]
            curr_price = c_series.iloc[-1]
            bias_30 = ((curr_price - ma30) / ma30) * 100

            if (ma30 > ma45 > ma60) and (0 <= bias_30 <= bias_limit):
                return {**s,
                        "收盤": round(curr_price, 2),
                        "乖離30MA(%)": round(bias_30, 2),
                        "成交量(張)": curr_vol_int,
                        "量變動(%)": round(vol_change, 2)}
            return None  # 條件不符，不需重試
        except requests.exceptions.Timeout:
            time.sleep(1 + attempt)   # timeout 時稍等再重試
            continue
        except Exception:
            return None               # 其他錯誤（格式異常等）直接放棄
    return None

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
    fig.update_layout(title=f"{name} ({ticker}) K線與均線圖",
                      xaxis_rangeslider_visible=False, height=600, template='plotly_dark')
    fig.update_xaxes(type='category')
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

    # ── Step 1：批次抓今日全市場行情（一次 API，約 1~2 秒）──
    status.text("📡 Step 1/3：取得全市場今日行情...")
    bar.progress(0.05)
    stock_map   = get_stock_market_list()           # [{ticker, name, industry, code}, ...]
    today_quote = get_today_quote_batch()           # {code: {close, volume, market}}

    if not today_quote:
        st.warning("⚠️ 今日行情尚未更新（可能為非交易日），將改用 Yahoo Finance 歷史末筆資料繼續掃描。")

    # ── Step 2：成交量初篩（純 CPU，毫秒級）──
    status.text("🔍 Step 2/3：成交量初篩...")
    bar.progress(0.10)
    candidates = []
    for s in stock_map:
        code = s["code"]
        q    = today_quote.get(code)
        if q:
            if q["volume"] >= user_vol:          # 通過成交量門檻
                candidates.append({**s, "_close": q["close"], "_volume": q["volume"]})
        else:
            # 批次 API 沒有資料（非交易日 or 新股）→ 仍列入候選，後續由 Yahoo 補
            candidates.append({**s, "_close": None, "_volume": None})

    status.text(f"🔍 Step 2/3：初篩通過 {len(candidates)} 支，開始均線計算...")

    # ── Step 3：對初篩股票抓歷史 K 線並做均線條件判斷 ──
    initial_hits = []
    total_c = len(candidates)

    def _check(s):
        code   = s["code"]
        market = "TW" if s["ticker"].endswith(".TW") else "TWO"
        close  = s["_close"]
        vol    = s["_volume"] if s["_volume"] else 0
        res    = check_ma_condition(code, market, close or 0, vol, user_bias)
        if res:
            return {**s, **res}
        return None

    with ThreadPoolExecutor(max_workers=50) as ex:  # 官方 API 無限流，可大幅提高並發
        futures = {ex.submit(_check, s): s for s in candidates}
        for i, f in enumerate(as_completed(futures), 1):
            bar.progress(0.10 + 0.75 * i / total_c)
            if i % 50 == 0:
                status.text(f"📊 Step 3/3：均線計算中... {i}/{total_c}")
            res = f.result()
            if res:
                initial_hits.append(res)

    bar.progress(0.85)

    if initial_hits:
        # ── Step 4：抓財報數據（只對符合條件的股票）──
        status.text(f"📈 找到 {len(initial_hits)} 支！抓取財報數據中...")
        final_list = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            f_deep = {ex.submit(fetch_deep_info, r["ticker"]): r for r in initial_hits}
            for j, f in enumerate(as_completed(f_deep), 1):
                bar.progress(0.85 + 0.14 * j / len(initial_hits))
                deep_res = f.result()
                base     = f_deep[f]
                final_list.append({
                    "ticker":    base["ticker"],
                    "code":      base["code"],
                    "name":      base["name"],
                    "industry":  base["industry"],
                    "收盤":      base["收盤"],
                    "乖離30MA(%)": base["乖離30MA(%)"],
                    "成交量(張)":  base["成交量(張)"],
                    "量變動(%)":   base["量變動(%)"],
                    "本益比":    deep_res["pe"],
                    "營收月增":  deep_res["mom"],
                    "營收年增":  deep_res["yoy"],
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
        st.success(f"✅ 掃描完成！找到 {len(df)} 支標的")
    with col_dl:
        csv     = df.to_csv(index=False).encode('utf-8-sig')
        tw_date = get_tw_now().strftime("%Y%m%d")
        st.download_button(label="📥 下載選股清單 (CSV)", data=csv,
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

    st.caption(f"💡 註1：進度條滿格代表乖離率接近你的上限值 ({user_bias}%)；條狀越短代表股價越貼近 30MA。")
    st.caption(f"💡 註2：營收增長與量變動如果為正數，會以紅色粗體顯示。")
    st.caption(f"💡 數據更新時間：{get_tw_now().strftime('%Y-%m-%d %H:%M:%S')} (台灣時間)")

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
        st.markdown(
            f"<div style='text-align:center; padding:6px 0;'>"
            f"第 <b>{c_idx + 1}</b> / {total_found} 支："
            f"<b>{df.iloc[c_idx]['code']} {df.iloc[c_idx]['name']}</b>"
            f"</div>",
            unsafe_allow_html=True
        )

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
    st.subheader(f"📰 {current_stock['name']} 即時中文新聞與情緒標籤")
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
        st.info("💡 調整左側參數後，點擊按鈕執行智慧選股。")
