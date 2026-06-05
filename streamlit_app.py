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
import re
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
    ('chart_mode',       'K線圖'),
    ('chart_period',     '日'),
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
            avg_vol20 = float(volumes.tail(20).mean())
            main_cost = calc_main_cost(df, 20)
            cost_gap = ((curr_price - main_cost) / main_cost * 100) if pd.notna(main_cost) and main_cost > 0 else np.nan
            high20 = float(closes.tail(20).max())
            high60 = float(closes.tail(60).max())
            prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else np.nan
            price_change = ((curr_price - prev_close) / prev_close * 100) if pd.notna(prev_close) and prev_close > 0 else np.nan
            hits.append({**s,
                "收盤":       round(curr_price, 2),
                "漲跌幅(%)":   round(price_change, 2) if pd.notna(price_change) else np.nan,
                "乖離30MA(%)": round(bias_30, 2),
                "成交量(張)":  vol_today,
                "量變動(%)":   round(vol_change, 2),
                "量比20日":    round(vol_today / avg_vol20, 2) if avg_vol20 > 0 else np.nan,
                "主力成本":    round(main_cost, 2) if pd.notna(main_cost) else np.nan,
                "主力成本乖離(%)": round(cost_gap, 2) if pd.notna(cost_gap) else np.nan,
                "RSI14":      round(calc_rsi(closes), 1),
                "MACD柱":     round(calc_macd_hist(closes), 3),
                "突破20日高":  curr_price >= high20,
                "接近60日高":  curr_price >= high60 * 0.97,
            })
    return hits

def clean_percent(text):
    if not text or text == "N/A":
        return np.nan
    try:
        return float(text.replace('%', '').replace(',', ''))
    except:
        return np.nan

def _to_float_or_nan(value):
    """將 Yahoo / yfinance 回傳的數值安全轉成 float。"""
    if value is None or value is False:
        return np.nan
    try:
        text = str(value).strip()
        if text in ["", "-", "--", "N/A", "None", "nan", "NaN"]:
            return np.nan
        text = text.replace(",", "").replace("倍", "")
        num = float(text)
        return num if np.isfinite(num) else np.nan
    except Exception:
        return np.nan

def _extract_pe_from_goodinfo_html(html: str) -> float:
    """
    從 Goodinfo 個股概況頁擷取 PER 欄位，也就是本益比。
    來源頁面格式：StockDetail.asp?STOCK_ID=台股代號
    """
    soup = BeautifulSoup(html, "html.parser")

    def norm_text(x):
        return re.sub(r"\s+", " ", str(x).replace("\xa0", " ")).strip()

    def valid_pe(x):
        pe = _to_float_or_nan(x)
        return pe if pd.notna(pe) and pe > 0 else np.nan

    # 方法 1：表格儲存格定位。通常 PER 會在表格欄名，下一格就是數值。
    cells = [norm_text(c.get_text(" ", strip=True)) for c in soup.find_all(["th", "td"])]
    cells = [c for c in cells if c]
    for i, c in enumerate(cells):
        if c.upper() == "PER":
            # 先找同一張交易資料表中 PER 後方最接近的數字
            for nxt in cells[i + 1:i + 6]:
                pe = valid_pe(nxt)
                if pd.notna(pe):
                    return pe

    # 方法 2：Goodinfo 有時會將「成交均價 PBR PER PEG」與下一列數值合併成純文字。
    lines = [norm_text(s) for s in soup.get_text("\n").splitlines()]
    lines = [x for x in lines if x]
    for i, line in enumerate(lines):
        if re.search(r"\bPBR\b.*\bPER\b.*\bPEG\b", line, re.I):
            # 下一列通常包含：成交張數 成交金額 成交筆數 成交均張 成交均價 PBR PER PEG
            for nxt in lines[i + 1:i + 5]:
                tokens = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", nxt)
                nums = []
                for token in tokens:
                    pe = _to_float_or_nan(token)
                    if pd.notna(pe):
                        nums.append(pe)
                # 最後三個通常為 PBR / PER / PEG，所以 PER 是倒數第二個。
                if len(nums) >= 3:
                    pe = nums[-2]
                    if pd.notna(pe) and pe > 0:
                        return pe

    # 方法 3：最後保險，搜尋 PER 鄰近文字。
    all_text = norm_text(soup.get_text(" "))
    m = re.search(r"\bPER\b\D{0,40}([0-9]+(?:\.[0-9]+)?)", all_text, re.I)
    if m:
        pe = valid_pe(m.group(1))
        if pd.notna(pe):
            return pe

    return np.nan


def _extract_pe_from_yahoo_html(html: str) -> float:
    """
    從 Yahoo 股市個股頁擷取本益比，僅作為 Goodinfo 失敗時的備援來源。
    """
    soup = BeautifulSoup(html, "html.parser")
    lines = [s.get_text(" ", strip=True) for s in soup.find_all(["li", "div", "span", "td"])]
    lines = [x for x in lines if x]

    for line in lines:
        if "本益比" in line or "PE" in line.upper():
            m = re.search(r"(?:本益比(?:\s*\(TTM\))?|PE(?:\s*Ratio)?)\D{0,30}([0-9]+(?:\.[0-9]+)?)", line, re.I)
            if m:
                pe = _to_float_or_nan(m.group(1))
                if pd.notna(pe) and pe > 0:
                    return pe

    for i, line in enumerate(lines):
        if "本益比" in line or "PE" in line.upper():
            for nxt in lines[i + 1:i + 8]:
                m = re.search(r"^-?[0-9]+(?:\.[0-9]+)?$", nxt.replace(",", ""))
                if m:
                    pe = _to_float_or_nan(nxt)
                    if pd.notna(pe) and pe > 0:
                        return pe
    return np.nan


def fetch_pe(ticker: str) -> float:
    """
    台股本益比取得順序：
    1) Goodinfo 個股概況頁 PER 欄位
    2) yfinance trailingPE / forwardPE
    3) yfinance 價格 ÷ trailingEps 手動換算
    4) Yahoo 股市個股頁文字解析
    """
    code = ticker.split('.')[0]

    # 1: Goodinfo PER 欄位。本益比以使用者指定網站為主。
    try:
        session = requests.Session()
        session.verify = False
        goodinfo_url = f"https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={code}"
        headers = get_headers()
        headers.update({
            "Referer": "https://goodinfo.tw/tw/index.asp",
            "Host": "goodinfo.tw",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        })
        resp = session.get(goodinfo_url, headers=headers, timeout=12)
        # Goodinfo 頁面常見 Big5/UTF-8 混用，優先採 apparent_encoding。
        if resp.encoding is None or resp.encoding.lower() in ["iso-8859-1", "ascii"]:
            resp.encoding = resp.apparent_encoding or "utf-8"
        if resp.status_code == 200 and resp.text:
            pe = _extract_pe_from_goodinfo_html(resp.text)
            if pd.notna(pe) and pe > 0:
                return pe
    except Exception:
        pass

    # 2 / 3: yfinance 備援
    try:
        yt = yf.Ticker(ticker)
        info = yt.info or {}
        for key in ["trailingPE", "forwardPE"]:
            pe = _to_float_or_nan(info.get(key))
            if pd.notna(pe) and pe > 0:
                return pe

        price = _to_float_or_nan(info.get("currentPrice") or info.get("regularMarketPrice"))
        eps = _to_float_or_nan(info.get("trailingEps"))
        if pd.notna(price) and pd.notna(eps) and eps > 0:
            return round(price / eps, 2)
    except Exception:
        pass

    # 4: Yahoo 股市頁面備援
    try:
        quote_url = f"https://tw.stock.yahoo.com/quote/{code}"
        quote_resp = requests.get(quote_url, headers=get_headers(), timeout=10)
        if quote_resp.status_code == 200:
            pe = _extract_pe_from_yahoo_html(quote_resp.text)
            if pd.notna(pe) and pe > 0:
                return pe
    except Exception:
        pass

    return np.nan


def fetch_deep_info(ticker: str) -> dict:
    code = ticker.split('.')[0]
    res  = {"pe": fetch_pe(ticker), "mom": np.nan, "yoy": np.nan}
    try:
        rev_url  = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
        rev_resp = requests.get(rev_url, headers=get_headers(), timeout=10)
        soup     = BeautifulSoup(rev_resp.text, 'html.parser')

        # 原 r'li.List(n)' CSS 選擇器含括號，BeautifulSoup 不支援；
        # 改用 find_all + lambda 模糊匹配含 'List' 的 class
        list_items = soup.find_all('li', class_=lambda c: c and 'List' in ' '.join(c) if isinstance(c, list) else c and 'List' in c)
        row = list_items[0] if list_items else None

        if row:
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.get_text()]
            if len(percents) >= 2:
                res["mom"] = clean_percent(percents[0])
                res["yoy"] = clean_percent(percents[1])
    except Exception:
        pass
    return res



def calc_rsi(close: pd.Series, period: int = 14) -> float:
    try:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else np.nan
    except Exception:
        return np.nan


def calc_macd_hist(close: pd.Series) -> float:
    try:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else np.nan
    except Exception:
        return np.nan


def calc_main_cost(df: pd.DataFrame, window: int = 20) -> float:
    """近 N 日量價加權平均成本，作為主力成本估算。"""
    try:
        tmp = df.tail(window).copy()
        vol_sum = tmp['volume'].sum()
        if vol_sum <= 0:
            return np.nan
        return float((tmp['close'] * tmp['volume']).sum() / vol_sum)
    except Exception:
        return np.nan


def calc_stock_score(row: dict) -> tuple[int, list[str]]:
    """0-100 分股票評分與飆股雷達條件。"""
    score = 0
    radar = []
    bias = row.get('乖離30MA(%)', np.nan)
    vol_ratio = row.get('量比20日', np.nan)
    vol_chg = row.get('量變動(%)', np.nan)
    yoy = row.get('營收年增', np.nan)
    mom = row.get('營收月增', np.nan)
    pe = row.get('本益比', np.nan)
    rsi = row.get('RSI14', np.nan)
    macd_hist = row.get('MACD柱', np.nan)
    cost_gap = row.get('主力成本乖離(%)', np.nan)
    breakout20 = row.get('突破20日高', False)
    near60 = row.get('接近60日高', False)

    score += 18
    if pd.notna(bias) and 0 <= bias <= 2.5:
        score += 7; radar.append('低乖離多頭')
    if breakout20:
        score += 5; radar.append('突破20日高')
    elif near60:
        score += 3; radar.append('接近60日高')

    if pd.notna(vol_ratio):
        if vol_ratio >= 2:
            score += 12; radar.append('量能放大2倍')
        elif vol_ratio >= 1.5:
            score += 9; radar.append('量能放大1.5倍')
        elif vol_ratio >= 1.1:
            score += 5
    if pd.notna(vol_chg) and vol_chg > 20:
        score += 4; radar.append('單日量增')
    if pd.notna(cost_gap) and 0 <= cost_gap <= 8:
        score += 4; radar.append('貼近主力成本')

    if pd.notna(yoy):
        if yoy >= 30:
            score += 14; radar.append('營收年增強')
        elif yoy >= 10:
            score += 9
        elif yoy > 0:
            score += 5
    if pd.notna(mom):
        if mom >= 10:
            score += 7; radar.append('營收月增強')
        elif mom > 0:
            score += 4
    if pd.notna(pe) and 0 < pe <= 25:
        score += 4; radar.append('PE合理')

    if pd.notna(rsi):
        if 55 <= rsi <= 70:
            score += 9; radar.append('RSI強勢未過熱')
        elif 45 <= rsi < 55:
            score += 5
        elif rsi > 78:
            score -= 5; radar.append('RSI過熱')
    if pd.notna(macd_hist) and macd_hist > 0:
        score += 8; radar.append('MACD偏多')

    if pd.notna(cost_gap) and cost_gap > 15:
        score -= 6; radar.append('遠離成本風險')
    if pd.notna(pe) and pe > 60:
        score -= 4; radar.append('PE偏高')
    return int(max(0, min(100, round(score)))), radar[:8]


def build_ai_report(stock: pd.Series) -> dict:
    score = stock.get('AI評分', np.nan)
    rsi = stock.get('RSI14', np.nan)
    macd_hist = stock.get('MACD柱', np.nan)
    cost_gap = stock.get('主力成本乖離(%)', np.nan)
    yoy = stock.get('營收年增', np.nan)
    mom = stock.get('營收月增', np.nan)
    vol_ratio = stock.get('量比20日', np.nan)
    bias = stock.get('乖離30MA(%)', np.nan)
    radar = stock.get('飆股雷達', '')

    strength = '強勢多頭' if pd.notna(score) and score >= 80 else '偏多觀察' if pd.notna(score) and score >= 65 else '中性偏多' if pd.notna(score) and score >= 50 else '訊號普通'
    trend = '均線多頭排列成立，趨勢結構偏多。'
    if pd.notna(bias) and bias > 5:
        trend += ' 但短線乖離偏大，追價風險提高。'
    elif pd.notna(bias) and bias <= 2.5:
        trend += ' 目前乖離相對收斂，位置較健康。'

    chip = '主力成本資料不足。'
    if pd.notna(cost_gap):
        if 0 <= cost_gap <= 8:
            chip = f'現價約高於近20日量價成本 {cost_gap:.1f}%，屬於貼近成本的偏多區。'
        elif cost_gap > 15:
            chip = f'現價高於近20日量價成本 {cost_gap:.1f}%，短線已明顯遠離成本。'
        elif cost_gap < 0:
            chip = f'現價低於近20日量價成本 {abs(cost_gap):.1f}%，需觀察是否重新站回成本線。'

    momentum = []
    if pd.notna(rsi): momentum.append(f'RSI14={rsi:.1f}')
    if pd.notna(macd_hist): momentum.append('MACD柱狀體為正' if macd_hist > 0 else 'MACD柱狀體為負')
    if pd.notna(vol_ratio): momentum.append(f'量比20日={vol_ratio:.2f}x')
    finance = []
    if pd.notna(yoy): finance.append(f'營收年增 {yoy:.1f}%')
    if pd.notna(mom): finance.append(f'營收月增 {mom:.1f}%')

    playbook = '高機率劇本：若守住主力成本線與短期均線，偏向沿均線震盪上攻；若跌破成本線且量縮，則轉為整理。'
    if pd.notna(score) and score >= 80:
        playbook = '高機率劇本：具備趨勢、量能與成長條件，偏向強勢股續航；回測主力成本線不破可視為觀察點。'
    elif pd.notna(score) and score < 55:
        playbook = '高機率劇本：訊號尚未完全共振，較適合等待量能或營收條件進一步確認。'
    return {'盤面強弱': strength, '趨勢結構': trend, '主力成本': chip, '動能訊號': '；'.join(momentum) if momentum else '動能資料不足', '財務訊號': '；'.join(finance) if finance else '財務資料不足', '飆股雷達': radar if radar else '尚未觸發高強度雷達條件', '劇本': playbook}


def fmt_num(value, pattern='{:.2f}', na='N/A'):
    return na if pd.isna(value) else pattern.format(value)


def render_hot_industries(df: pd.DataFrame):
    if df.empty or 'industry' not in df.columns or 'AI評分' not in df.columns:
        return
    hot = (df.groupby('industry')
             .agg(標的數=('code', 'count'), 平均分=('AI評分', 'mean'), 平均量變=('量變動(%)', 'mean'), 平均年增=('營收年增', 'mean'))
             .reset_index())
    hot = hot[hot['標的數'] >= 1].sort_values(['平均分', '標的數'], ascending=[False, False]).head(8)
    if hot.empty:
        return
    st.markdown('<div class="tv-section">HOT INDUSTRIES · 熱門族群</div>', unsafe_allow_html=True)
    cols = st.columns(min(4, len(hot)))
    for i, (_, r) in enumerate(hot.iterrows()):
        with cols[i % len(cols)]:
            yoy_txt = 'N/A' if pd.isna(r['平均年增']) else f"{r['平均年增']:.1f}%"
            vol_txt = 'N/A' if pd.isna(r['平均量變']) else f"{r['平均量變']:.1f}%"
            st.markdown(f"""
            <div class="tv-card" style="margin-bottom:10px;">
              <div class="tv-label">{r['industry']}</div>
              <div class="tv-value" style="font-size:22px;color:#8fb2ff;">{r['平均分']:.0f}</div>
              <div class="tv-caption">標的 {int(r['標的數'])} · 量變 {vol_txt} · YoY {yoy_txt}</div>
            </div>
            """, unsafe_allow_html=True)

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

def draw_k_line(ticker, name, chart_mode='K線圖', chart_period='日'):
    """畫出有實際切換功能的金融圖表。
    chart_mode: K線圖 / 走勢圖 / 技術指標
    chart_period: 日 / 週 / 月
    """
    code   = ticker.split(".")[0]
    market = "TW" if ticker.endswith(".TW") else "TWO"
    df     = get_kline_data(code, market)
    if df.empty or len(df) < 30:
        yt  = yf.Ticker(ticker)
        raw = yt.history(period="1y")
        if raw.empty:
            return None
        df = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                  "Close": "close", "Volume": "volume"}).reset_index()
        df["date"]   = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df["volume"] = df["volume"] // 1000

    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt")

    # 週/月是真正重取樣，不只是 UI 顯示。
    if chart_period in ["週", "月"]:
        rule = "W-FRI" if chart_period == "週" else "ME"
        df = (df.set_index("date_dt")
                .resample(rule)
                .agg({"open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"})
                .dropna()
                .reset_index())
        df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d")

    # 技術計算會因日/週/月切換而重新計算。
    df['MA30'] = df['close'].rolling(30).mean()
    df['MA45'] = df['close'].rolling(45).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    df['主力成本20'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['VOL_MA5'] = df['volume'].rolling(5).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI14'] = 100 - (100 / (1 + rs))
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['DIF'] - df['DEA']
    df['MACD_GOLD'] = (df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1))
    df['MACD_DEAD'] = (df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1))

    tail_n = 95 if chart_period == '日' else 78 if chart_period == '週' else 60
    df = df.tail(tail_n).copy()
    if len(df) < 10:
        return None

    up_color = '#26a69a'
    down_color = '#ef4444'
    colors = [up_color if df['close'].iloc[i] >= df['open'].iloc[i] else down_color for i in range(len(df))]
    max_vol = float(df['volume'].max()) if df['volume'].max() > 0 else 1.0

    fig = go.Figure()

    # 成交量與量均線整合在同一張圖底部。
    fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'], name='成交量', yaxis='y2',
        marker_color=colors, opacity=0.58,
        hovertemplate='<b>%{x}</b><br>成交量：%{y:,} 張<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['VOL_MA5'], yaxis='y2', mode='lines', name='成交量 MA5',
        line=dict(color='#facc15', width=1.05), hoverinfo='skip'
    ))

    if chart_mode == '走勢圖':
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['close'], mode='lines', name='收盤價',
            line=dict(color='#22c55e', width=2.2),
            hovertemplate='<b>%{x}</b><br>收盤：%{y:.2f}<extra></extra>'
        ))
    else:
        fig.add_trace(go.Candlestick(
            x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name=f'{name} ({code})',
            increasing_line_color=up_color, increasing_fillcolor='rgba(38,166,154,.88)',
            decreasing_line_color=down_color, decreasing_fillcolor='rgba(239,68,68,.88)',
            hoverinfo='none', showlegend=False,
        ))

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
            "主力成本20：%{customdata[7]:.2f}<br>"
            "RSI14：%{customdata[8]:.1f}<br>"
            "成交量：%{customdata[9]:,} 張"
            "<extra></extra>"
        ),
        customdata=df[['open', 'close', 'high', 'low', 'MA30', 'MA45', 'MA60', '主力成本20', 'RSI14', 'volume']].values,
    ))

    fig.add_trace(go.Scatter(x=df['date'], y=df['MA30'], line=dict(color='#3b82f6', width=1.5), name='MA30', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA45'], line=dict(color='#f59e0b', width=1.5), name='MA45', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], line=dict(color='#a855f7', width=1.5), name='MA60', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['主力成本20'], line=dict(color='#ef4444', width=1.35, dash='dash'), name='主力成本', hoverinfo='skip'))

    if chart_mode == '技術指標':
        gold = df[df['MACD_GOLD']]
        dead = df[df['MACD_DEAD']]
        overbought = df[df['RSI14'] >= 70]
        oversold = df[df['RSI14'] <= 30]
        fig.add_trace(go.Scatter(
            x=gold['date'], y=gold['low'] * 0.985, mode='markers', name='MACD 黃金交叉',
            marker=dict(symbol='triangle-up', size=11, color='#22c55e', line=dict(width=1, color='#d8fff0')),
            hovertemplate='<b>%{x}</b><br>MACD 黃金交叉<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=dead['date'], y=dead['high'] * 1.015, mode='markers', name='MACD 死亡交叉',
            marker=dict(symbol='triangle-down', size=11, color='#ef4444', line=dict(width=1, color='#ffd6d6')),
            hovertemplate='<b>%{x}</b><br>MACD 死亡交叉<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=overbought['date'], y=overbought['high'] * 1.028, mode='markers', name='RSI 過熱',
            marker=dict(symbol='circle-open', size=9, color='#f59e0b', line=dict(width=2)),
            hovertemplate='<b>%{x}</b><br>RSI 過熱：%{customdata:.1f}<extra></extra>', customdata=overbought['RSI14']
        ))
        fig.add_trace(go.Scatter(
            x=oversold['date'], y=oversold['low'] * 0.972, mode='markers', name='RSI 低檔',
            marker=dict(symbol='circle-open', size=9, color='#3b82f6', line=dict(width=2)),
            hovertemplate='<b>%{x}</b><br>RSI 低檔：%{customdata:.1f}<extra></extra>', customdata=oversold['RSI14']
        ))

    spike_cfg = dict(
        type='category', showgrid=True, gridcolor='rgba(148,163,184,0.09)',
        zeroline=False, showspikes=True, spikemode='across', spikesnap='cursor',
        spikecolor='rgba(59,130,246,0.65)', spikethickness=1, spikedash='dot',
        showline=False,
    )

    fig.update_layout(
        height=610,
        template='plotly_dark',
        paper_bgcolor='#0b121b',
        plot_bgcolor='#0b121b',
        font=dict(color='#9aa7b8', size=11, family='Inter, Noto Sans TC, sans-serif'),
        # 右側預留空間，避免價格刻度文字被截斷。
        margin=dict(l=10, r=74, t=18, b=12),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#111a26', bordercolor='rgba(59,130,246,0.45)',
            font=dict(size=12, color='#e6edf3', family='Roboto Mono, monospace'),
            namelength=0,
        ),
        legend=dict(
            orientation='h', yanchor='top', y=0.985, xanchor='left', x=0.012,
            bgcolor='rgba(7,13,20,.25)', bordercolor='rgba(33,48,64,.65)', borderwidth=0,
            font=dict(size=11, color='#cbd5e1'),
        ),
        dragmode=False,
        bargap=0.18,
        xaxis=dict(**spike_cfg, fixedrange=True, rangeslider=dict(visible=False), tickfont=dict(size=11), automargin=True),
        yaxis=dict(
            fixedrange=True, side='right', showgrid=True, gridcolor='rgba(148,163,184,0.09)',
            zeroline=False, tickfont=dict(size=11, color='#cbd5e1'), showspikes=False,
            automargin=True, ticks='outside', ticklabelposition='outside right', separatethousands=True,
        ),
        yaxis2=dict(
            overlaying='y', side='left', range=[0, max_vol / 0.22],
            showgrid=False, zeroline=False, showticklabels=False, fixedrange=True,
        ),
    )
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
# 3. 全域 CSS（TradingView 機構終端機風格）
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Roboto+Mono:wght@400;500;600;700&family=Noto+Sans+TC:wght@300;400;500;700;900&display=swap');

:root {
    --tv-bg: #070d14;
    --tv-bg-2: #0b121b;
    --tv-panel: #0f1722;
    --tv-panel-2: #111c29;
    --tv-card: #0c1520;
    --tv-border: #213040;
    --tv-border-2: #2d3e52;
    --tv-text: #d8dee9;
    --tv-muted: #8f9bad;
    --tv-faint: #5d6b7d;
    --tv-blue: #3b82f6;
    --tv-blue-soft: rgba(59,130,246,.16);
    --tv-green: #22c55e;
    --tv-green-2: #26a69a;
    --tv-red: #ef4444;
    --tv-red-2: #f23645;
    --tv-yellow: #facc15;
    --tv-orange: #f59e0b;
    --tv-purple: #a855f7;
}

[data-testid='stSidebarCollapseButton'],
[data-testid='collapsedControl'],
[data-testid='stHeader'],
header[data-testid='stHeader'],
[data-testid='stToolbar'] { display: none !important; }

html, body, [data-testid='stAppViewContainer'], [data-testid='stMain'] {
    background:
        radial-gradient(circle at 15% 0%, rgba(59,130,246,.10), transparent 30%),
        linear-gradient(180deg, #080e16 0%, #060b11 100%) !important;
    color: var(--tv-text) !important;
    font-family: 'Inter','Noto Sans TC',sans-serif !important;
}
.block-container {
    max-width: none !important;
    padding: 78px 24px 2.2rem 24px !important;
}

[data-testid='stSidebar'] { display:none !important; }

/* top bar / left rail */
.tv-topbar {
    position: fixed; top: 0; left: 0; right: 0; height: 58px; z-index: 9999;
    display: flex; align-items: center; justify-content: space-between; gap: 18px;
    padding: 0 18px 0 20px;
    background: rgba(7,13,20,.92);
    border-bottom: 1px solid var(--tv-border);
    backdrop-filter: blur(14px);
}
.tv-brand { display:flex;align-items:center;gap:10px;font-size:20px;font-weight:900;color:#f3f6fb;letter-spacing:.8px;white-space:nowrap; }
.tv-brand-icon { color: var(--tv-blue); font-size: 21px; }
.tv-tabs { display:flex;align-items:center;gap:30px;height:100%; }
.tv-tab { color:#b8c0cc;font-size:14px;font-weight:700;padding:19px 2px 16px;border-bottom:2px solid transparent; }
.tv-tab.active { color:#e8eef7;border-color:var(--tv-blue);box-shadow:0 8px 20px rgba(59,130,246,.16); }
.tv-top-meta { display:flex;align-items:center;gap:14px;color:#9aa7b8;font-family:'Roboto Mono',monospace;font-size:12px;white-space:nowrap; }
.tv-leftnav {
    position: fixed; left:0; top:58px; bottom:0; width:184px; z-index: 9998;
    background: rgba(8,14,22,.95); border-right:1px solid var(--tv-border);
    padding: 18px 14px; backdrop-filter: blur(14px);
}
.tv-navitem { display:flex;align-items:center;gap:12px;color:#9aa7b8;padding:13px 14px;border-radius:7px;margin-bottom:9px;font-weight:700;font-size:14px; }
.tv-navitem.active { color:#e8eef7;background:rgba(59,130,246,.08);border:1px solid var(--tv-border-2); }
.tv-navitem .ico { width:22px;text-align:center;font-size:19px;color:#9fb2c8; }
.tv-market-mini { position:absolute;left:14px;right:14px;bottom:30px;background:#0b131d;border:1px solid var(--tv-border);border-radius:8px;padding:14px 15px; }

[data-testid='stButton'] > button,
[data-testid='stDownloadButton'] > button {
    background: #111a26 !important;
    color: var(--tv-text) !important;
    border: 1px solid var(--tv-border) !important;
    border-radius: 6px !important;
    font-family: 'Inter','Noto Sans TC',sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 10px 0 !important;
    transition: all .16s ease !important;
}
[data-testid='stButton'] > button:hover,
[data-testid='stDownloadButton'] > button:hover {
    border-color: var(--tv-blue) !important;
    background: #172336 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.15) !important;
}
[data-testid='stButton'] > button:disabled { opacity: .35 !important; }

[data-testid='stDataFrame'] {
    background: var(--tv-panel) !important;
    border: 1px solid var(--tv-border) !important;
    border-radius: 9px !important;
    overflow: hidden !important;
}
[data-testid='stDataFrame'] * { font-family: 'Roboto Mono','Noto Sans TC',monospace !important; }

[data-testid='stPlotlyChart'] {
    background: var(--tv-panel) !important;
    border: 1px solid var(--tv-border) !important;
    border-radius: 9px !important;
    padding: 8px !important;
}

[data-testid='stProgress'] > div { background:#111a26 !important;border-radius:999px !important;height:7px !important; }
[data-testid='stProgress'] > div > div { background:linear-gradient(90deg,var(--tv-blue),#66a3ff) !important;border-radius:999px !important; }
[data-testid='stAlert'] { background:rgba(59,130,246,.08) !important;border:1px solid rgba(59,130,246,.25) !important;border-radius:8px !important; }
[data-testid='stSlider'] div[role='slider'] { background:var(--tv-blue) !important; }
[data-testid='stNumberInput'] input { background:#0b121b !important;border:1px solid var(--tv-border) !important;color:var(--tv-text) !important;border-radius:7px !important;font-family:'Roboto Mono',monospace !important; }

.tv-panel, .tv-card, .quote-panel, .side-card {
    background: linear-gradient(180deg, rgba(16,26,38,.98) 0%, rgba(10,18,28,.98) 100%);
    border: 1px solid var(--tv-border);
    border-radius: 9px;
    box-shadow: 0 18px 40px rgba(0,0,0,.20);
}
.tv-panel { padding: 14px 16px; }
.tv-card { padding: 13px 15px; }
.tv-label { color: var(--tv-muted); font-size: 11px; font-weight: 700; letter-spacing: .45px; text-transform: uppercase; }
.tv-value { color: var(--tv-text); font-family:'Roboto Mono',monospace; font-size:25px; font-weight:800; margin-top:5px; }
.tv-caption { color: var(--tv-muted); font-size: 12px; margin-top:4px; }
.tv-section { color:#dce6f2;font-size:13px;font-weight:900;letter-spacing:.4px;margin:15px 0 9px;padding-left:10px;border-left:3px solid var(--tv-blue); }
.tv-pill { border:1px solid var(--tv-border);background:#101a28;color:#bdc7d5;border-radius:5px;padding:6px 12px;font-size:13px;font-weight:700; }
.tv-pill.active { background:rgba(59,130,246,.18);border-color:#315b9b;color:#e9f1ff; }
.chart-control-label {font-size:11px;font-weight:800;color:var(--tv-muted);letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px;}
.chart-control-box {border:1px solid var(--tv-border);background:#0d1622;border-radius:8px;padding:8px 10px;}
.chart-control-box [data-testid='stRadio'] label {font-size:12px;color:#cbd5e1;font-weight:700;}
.chart-control-box [data-testid='stRadio'] div[role='radiogroup'] {gap:6px;}
.chart-control-box [data-testid='stRadio'] div[role='radiogroup'] label {border:1px solid var(--tv-border);background:#101a28;border-radius:5px;padding:4px 9px;margin-right:4px;}

.quote-panel { padding: 6px 0 12px; margin-bottom: 10px; border: 0; box-shadow:none; background:transparent; }
.quote-head { display:flex;align-items:center;gap:15px;flex-wrap:wrap; }
.quote-title { font-size:25px;font-weight:900;color:#e6edf6;letter-spacing:.5px; }
.quote-tag { background:#111a26;border:1px solid var(--tv-border);border-radius:6px;padding:8px 12px;color:#aeb9c7;font-weight:700;font-size:13px; }
.quote-price { font-family:'Roboto Mono',monospace;font-size:40px;font-weight:800;color:var(--tv-green);line-height:1.1;margin-top:14px; }
.quote-change { font-family:'Roboto Mono',monospace;font-size:18px;font-weight:700;margin-left:10px; }
.quote-metrics { display:grid;grid-template-columns:repeat(5,minmax(100px,1fr));gap:18px;margin-top:13px;max-width:780px; }
.metric-k { color:#8996a8;font-size:12px;font-weight:700; }
.metric-v { color:#e5edf7;font-family:'Roboto Mono',monospace;font-size:16px;font-weight:700;margin-top:4px; }
.chart-toolbar { display:flex;align-items:center;justify-content:space-between;gap:12px;margin:8px 0 8px; }
.chart-tabs { display:flex;gap:8px;align-items:center;flex-wrap:wrap; }
.side-card { padding:15px 16px;margin-bottom:14px; }
.side-title { display:flex;justify-content:space-between;align-items:center;color:#eef3fb;font-size:17px;font-weight:900;margin-bottom:12px; }
.bias-chip { background:rgba(34,197,94,.16);color:#8ef0aa;border:1px solid rgba(34,197,94,.25);border-radius:5px;padding:4px 10px;font-size:12px;font-weight:800; }
.report-row { display:flex;justify-content:space-between;gap:10px;border-bottom:1px solid rgba(45,62,82,.45);padding:7px 0;color:#c8d1dd;font-size:13px; }
.report-row span:last-child { font-family:'Roboto Mono',monospace;font-weight:700;color:#e8edf5;text-align:right; }
.radar-check { color:#b9c4d2;font-size:13px;line-height:1.9; }
.radar-check b { color:var(--tv-green);margin-right:8px; }
.news-card { background:#0f1722 !important;border:1px solid var(--tv-border) !important;border-radius:9px !important; }
.news-title:hover { color:#8fb7ff !important; }

/* Streamlit spacing cleanup */
div[data-testid='stVerticalBlock'] { gap: .55rem !important; }
hr { border-color: var(--tv-border) !important; }
::-webkit-scrollbar { width:8px;height:8px; }
::-webkit-scrollbar-track { background:#070d14; }
::-webkit-scrollbar-thumb { background:#2d3e52;border-radius:999px; }

@media (max-width: 1100px) {
    .block-container { padding-left: 18px !important; padding-top: 72px !important; }
    .tv-leftnav { display:none; }
    .tv-tabs { display:none; }
    .quote-metrics { grid-template-columns:repeat(2,1fr); }
}
@media (max-width: 768px) {
    .block-container { padding:68px 10px 2rem !important; }
    .tv-top-meta { display:none; }
    .tv-brand { font-size:16px; }
    .quote-price { font-size:32px; }
    [data-testid='stPlotlyChart'] { min-height:420px !important; }
    [data-testid='stDataFrame'] * { font-size:12px !important; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 4. 交易終端機 Top Bar（保留實際資訊，移除無作用導航）
# ============================================================

_now_str = get_tw_now().strftime("%Y-%m-%d %H:%M")
_signal_count = len(st.session_state.scan_results) if isinstance(st.session_state.scan_results, pd.DataFrame) else 0
st.markdown(f"""
<div class="tv-topbar">
  <div class="tv-brand"><span class="tv-brand-icon">◆</span> 台股智慧選股系統</div>
  <div class="tv-top-meta">
    <span>策略：MA30 &gt; MA45 &gt; MA60</span>
    <span>目前訊號：{_signal_count}</span>
    <span>更新時間：{_now_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 5. 篩選參數設定（實際可操作）
# ============================================================

# ── 主畫面篩選參數：這裡是實際會影響掃描結果的控制項 ──
with st.expander("▸ 篩選參數", expanded=False):
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
                row_data = {
                    "ticker":   base["ticker"],
                    "code":     base["code"],
                    "name":     base["name"],
                    "industry": base["industry"],
                    "收盤":       base["收盤"],
                    "漲跌幅(%)":   base.get("漲跌幅(%)", np.nan),
                    "乖離30MA(%)": base["乖離30MA(%)"],
                    "成交量(張)":  base["成交量(張)"],
                    "量變動(%)":   base["量變動(%)"],
                    "量比20日":    base.get("量比20日", np.nan),
                    "主力成本":    base.get("主力成本", np.nan),
                    "主力成本乖離(%)": base.get("主力成本乖離(%)", np.nan),
                    "RSI14":      base.get("RSI14", np.nan),
                    "MACD柱":     base.get("MACD柱", np.nan),
                    "突破20日高":  base.get("突破20日高", False),
                    "接近60日高":  base.get("接近60日高", False),
                    "本益比":      deep_res["pe"],
                    "營收月增":    deep_res["mom"],
                    "營收年增":    deep_res["yoy"],
                }
                score, radar = calc_stock_score(row_data)
                row_data["AI評分"] = score
                row_data["飆股雷達"] = "、".join(radar) if radar else "觀察"
                final_list.append(row_data)
        bar.progress(1.0)
        st.session_state.scan_results = pd.DataFrame(final_list).sort_values("AI評分", ascending=False).reset_index(drop=True)
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

    # ── 精簡統計列：移除 Avg AI Score / Rocket Radar 與熱門族群區塊 ──
    st.markdown(f"""
    <div class="stat-grid" style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:14px;">
      <div class="tv-card">
        <div class="tv-label">Total Signals</div>
        <div class="tv-value" style="color:#8fb2ff;">{total_found}</div>
        <div class="tv-caption">符合條件標的</div>
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
    show_cols      = ["code", "name", "AI評分", "收盤", "漲跌幅(%)", "乖離30MA(%)", "主力成本", "量比20日", "成交量(張)", "量變動(%)", "RSI14", "本益比", "營收月增", "營收年增", "industry"]
    available_cols = [c for c in show_cols if c in df.columns]
    df_display     = df[available_cols].rename(columns={"code": "代碼", "name": "名稱", "industry": "類股"})

    def color_tw_style(val):
        if pd.isna(val): return ''
        color = '#22ab94' if val > 0 else '#f23645' if val < 0 else '#e6edf3'
        return f'color: {color}; font-weight: bold'

    event = st.dataframe(
        df_display.style.map(
            color_tw_style,
            subset=[c for c in ['漲跌幅(%)', '量變動(%)', '營收月增', '營收年增'] if c in df_display.columns]
        ),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"stock_table_{st.session_state.table_key}",
        column_config={
            "代碼":        st.column_config.TextColumn("代碼",    width=70),
            "名稱":        st.column_config.TextColumn("名稱",    width=100),
            "AI評分":      st.column_config.ProgressColumn("AI評分", width=105, format="%d", min_value=0, max_value=100),
            "收盤":        st.column_config.NumberColumn("價格",  width=75,  format="%.2f"),
            "漲跌幅(%)":   st.column_config.NumberColumn("漲跌", width=75, format="%.1f%%"),
            "乖離30MA(%)": st.column_config.ProgressColumn("30MA乖離", width=120,
                                                            help=f"上限 {user_bias}%",
                                                            format="%.2f%%", min_value=0, max_value=user_bias),
            "主力成本":    st.column_config.NumberColumn("主力成本", width=90, format="%.2f"),
            "量比20日":    st.column_config.NumberColumn("量比", width=70, format="%.2fx"),
            "量變動(%)":   st.column_config.NumberColumn("量變動", width=80, format="%.1f%%"),
            "RSI14":       st.column_config.NumberColumn("RSI", width=65, format="%.1f"),
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
    # 8. 個股總覽 / K 線 + 成交量整合圖 / 右側 AI 面板
    # ============================================================

    current_stock = df.iloc[st.session_state.current_idx]
    report = build_ai_report(current_stock)
    c_idx = st.session_state.current_idx
    price = current_stock.get('收盤', np.nan)
    chg = current_stock.get('漲跌幅(%)', np.nan)
    chg_color = 'var(--tv-green)' if pd.notna(chg) and chg >= 0 else 'var(--tv-red)'
    chg_txt = 'N/A' if pd.isna(chg) else f"{chg:+.2f}%"
    prev_est = np.nan if pd.isna(price) or pd.isna(chg) or chg == -100 else price / (1 + chg / 100)
    chg_amt = np.nan if pd.isna(prev_est) else price - prev_est
    chg_amt_txt = 'N/A' if pd.isna(chg_amt) else f"{chg_amt:+.2f}"
    score_val = current_stock.get('AI評分', np.nan)
    score_txt = 'N/A' if pd.isna(score_val) else f"{int(score_val)} / 100"
    main_cost_txt = fmt_num(current_stock.get('主力成本', np.nan), '{:.2f}')
    cost_gap_txt = fmt_num(current_stock.get('主力成本乖離(%)', np.nan), '{:+.2f}%')
    rsi_txt = fmt_num(current_stock.get('RSI14', np.nan), '{:.1f}')
    macd_txt = fmt_num(current_stock.get('MACD柱', np.nan), '{:.3f}')
    vol_ratio_txt = fmt_num(current_stock.get('量比20日', np.nan), '{:.2f}x')
    radar_items = [x for x in str(current_stock.get('飆股雷達', '')).split('、') if x and x != '觀察']
    if not radar_items:
        radar_items = ['等待更強共振訊號']

    st.markdown(f"""
    <div class="quote-panel">
      <div class="quote-head">
        <div style="font-size:24px;color:var(--tv-yellow);">☆</div>
        <div class="quote-title">{current_stock['name']} · {current_stock['code']}.{ 'TW' if current_stock['ticker'].endswith('.TW') else 'TWO' }</div>
        <div class="quote-tag">{current_stock.get('industry', '產業別')}</div>
      </div>
      <div>
        <span class="quote-price">{fmt_num(price, '{:.2f}')}</span>
        <span class="quote-change" style="color:{chg_color};">{chg_amt_txt} ({chg_txt})</span>
      </div>
      <div class="quote-metrics">
        <div><div class="metric-k">AI 評分</div><div class="metric-v">{score_txt}</div></div>
        <div><div class="metric-k">主力成本</div><div class="metric-v">{main_cost_txt}</div></div>
        <div><div class="metric-k">成本乖離</div><div class="metric-v" style="color:{chg_color};">{cost_gap_txt}</div></div>
        <div><div class="metric-k">成交量</div><div class="metric-v">{fmt_num(current_stock.get('成交量(張)', np.nan), '{:,.0f}')} 張</div></div>
        <div><div class="metric-k">量比20日</div><div class="metric-v">{vol_ratio_txt}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    left_area, right_area = st.columns([4.7, 1.25], gap="medium")

    with left_area:
        nav_spacer, nav1, nav2 = st.columns([3.2, 0.9, 0.9])
        with nav1:
            if st.button("⬅ PREV", use_container_width=True, key="btn_prev"):
                st.session_state.current_idx = (st.session_state.current_idx - 1) % total_found
                st.session_state.last_selected_row = None
                st.session_state.table_key += 1
                st.rerun()
        with nav2:
            if st.button("NEXT ➡", use_container_width=True, key="btn_next"):
                st.session_state.current_idx = (st.session_state.current_idx + 1) % total_found
                st.session_state.last_selected_row = None
                st.session_state.table_key += 1
                st.rerun()

        # 固定顯示日 K 線圖：移除「圖表模式」與「週期」切換控制，只保留 K 線圖本體。
        k_fig = draw_k_line(
            current_stock['ticker'], current_stock['name'],
            chart_mode='K線圖', chart_period='日'
        )
        if k_fig:
            st.plotly_chart(k_fig, use_container_width=True, config={
                "displayModeBar": False,
                "scrollZoom": False,
                "doubleClick": False,
                "showTips": False,
            })
        else:
            st.warning("⚠️ 無法載入 K 線資料，請稍後再試。")

    with right_area:
        st.markdown(f"""
        <div class="side-card">
          <div class="side-title"><span>AI 分析報告</span><span class="bias-chip">{report['盤面強弱']}</span></div>
          <div class="report-row"><span>強度評分</span><span style="color:var(--tv-green);">{score_txt}</span></div>
          <div class="report-row"><span>主力成本</span><span>{main_cost_txt}</span></div>
          <div class="report-row"><span>成本乖離</span><span>{cost_gap_txt}</span></div>
          <div class="report-row"><span>RSI</span><span>{rsi_txt}</span></div>
          <div class="report-row"><span>MACD柱</span><span>{macd_txt}</span></div>
          <div style="color:#b8c4d3;font-size:13px;line-height:1.75;margin-top:12px;">{report['趨勢結構']}<br>{report['動能訊號']}</div>
        </div>
        <div class="side-card">
          <div class="side-title"><span>主力成本</span></div>
          <div style="font-family:Roboto Mono,monospace;font-size:27px;font-weight:800;color:var(--tv-green);">{main_cost_txt}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;">
            <div><div class="tv-caption">現價</div><div class="metric-v" style="color:var(--tv-green);">{fmt_num(price, '{:.2f}')}</div></div>
            <div><div class="tv-caption">浮動盈虧</div><div class="metric-v" style="color:{chg_color};">{cost_gap_txt}</div></div>
          </div>
          <div style="margin-top:14px;height:10px;border-radius:999px;background:linear-gradient(90deg,var(--tv-green) 0 24%, var(--tv-red) 24% 82%, #243446 82% 100%);"></div>
          <div style="display:flex;justify-content:space-between;color:#a7b3c2;font-size:12px;margin-top:7px;"><span>靠近成本</span><span>持有區</span><span>偏離風險</span></div>
        </div>
        <div class="side-card">
          <div class="side-title"><span>飆股雷達</span><span class="bias-chip">HOT</span></div>
          <div class="radar-check">{''.join([f'<div><b>✓</b>{item}</div>' for item in radar_items[:6]])}</div>
          <div style="display:flex;justify-content:space-between;align-items:end;margin-top:12px;">
            <span class="tv-caption">綜合評分</span>
            <span style="font-family:Roboto Mono,monospace;font-size:26px;font-weight:800;color:var(--tv-green);">{score_txt}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

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
