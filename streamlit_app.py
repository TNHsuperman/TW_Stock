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
import time
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import uuid
import os
import json
import threading

# ============================================================
# 1. 基礎設定與環境初始化
# ============================================================

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(
    page_title="台股智慧選股｜操作中心",
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
    ('show_guide',       True),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def get_tw_now():
    return datetime.now(timezone(timedelta(hours=8)))

# ============================================================
# 2. 數據抓取
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_market_list():
    """快速載入台股清單。

    舊版卡在 get_stock_market_list() 的主因通常是 isin.twse.com.tw + pd.read_html
    解析整頁 HTML 太慢或連線被拖住。這版改成：
    1) 先讀本機每日快取，幾乎秒開。
    2) 再抓 TWSE / TPEx OpenAPI JSON，避免 read_html 大表格解析。
    3) OpenAPI 失敗時才用舊 ISIN HTML 備援，且 timeout 較短。
    """
    cache_file = os.path.join(os.path.dirname(__file__) if '__file__' in globals() else '.', 'stock_market_cache_v2_industry_fixed.json')
    today = get_tw_now().strftime('%Y-%m-%d')


    INDUSTRY_CODE_MAP = {
        '01': '水泥工業', '02': '食品工業', '03': '塑膠工業', '04': '紡織纖維',
        '05': '電機機械', '06': '電器電纜', '07': '化學生技醫療', '08': '玻璃陶瓷',
        '09': '造紙工業', '10': '鋼鐵工業', '11': '橡膠工業', '12': '汽車工業',
        '14': '建材營造', '15': '航運業', '16': '觀光餐旅', '17': '金融保險業',
        '18': '貿易百貨業', '20': '其他業', '21': '化學工業', '22': '生技醫療業',
        '23': '油電燃氣業', '24': '半導體業', '25': '電腦及週邊設備業', '26': '光電業',
        '27': '通信網路業', '28': '電子零組件業', '29': '電子通路業', '30': '資訊服務業',
        '31': '其他電子業', '32': '文化創意業', '33': '農業科技業', '34': '電子商務',
        '35': '綠能環保', '36': '數位雲端', '37': '運動休閒', '38': '居家生活',
        '80': '管理股票', '91': '存託憑證', '92': 'ETF', '93': '受益證券', '94': '認購售權證',
    }

    def normalize_industry(value):
        """將 TWSE/TPEx OpenAPI 回傳的產業代碼轉成中文產業別。"""
        txt = str(value or '').strip()
        if txt in ['', '-', 'None', 'null', 'nan']:
            return '未分類'
        # 有些 OpenAPI 只給 05 / 36 這類產業代碼。
        code = txt.zfill(2) if txt.isdigit() and len(txt) <= 2 else txt
        if code in INDUSTRY_CODE_MAP:
            return INDUSTRY_CODE_MAP[code]
        # 有些欄位可能是「36 數位雲端」或「36,數位雲端」。
        m = re.match(r'^(\d{1,2})\s*[、,，\-:：]?\s*(.*)$', txt)
        if m:
            c = m.group(1).zfill(2)
            tail = m.group(2).strip()
            if tail and not tail.isdigit():
                return tail
            return INDUSTRY_CODE_MAP.get(c, txt)
        return txt

    def normalize_stock_name(value):
        """上櫃 OpenAPI 有時只提供公司全名，這裡轉成較適合表格的股票簡稱。"""
        name = str(value or '').strip().replace(' ', '')
        for suffix in ['股份有限公司', '有限公司']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name

    def load_local_cache(allow_stale=False):
        try:
            if not os.path.exists(cache_file):
                return []
            with open(cache_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if allow_stale or payload.get('date') == today:
                data = payload.get('data', [])
                if isinstance(data, list) and len(data) > 100:
                    for item in data:
                        if isinstance(item, dict):
                            item['industry'] = normalize_industry(item.get('industry', '未分類'))
                            item['name'] = normalize_stock_name(item.get('name', ''))
                    return data
        except Exception:
            pass
        return []

    def save_local_cache(data):
        try:
            if data and len(data) > 100:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({'date': today, 'data': data}, f, ensure_ascii=False)
        except Exception:
            pass

    cached = load_local_cache(allow_stale=False)
    if cached:
        return cached

    session = requests.Session()
    session.verify = False
    adapter = requests.adapters.HTTPAdapter(max_retries=0, pool_connections=20, pool_maxsize=20)
    session.mount('https://', adapter)
    session.mount('http://', adapter)


    INDUSTRY_CODE_MAP = {
        '01': '水泥工業', '02': '食品工業', '03': '塑膠工業', '04': '紡織纖維',
        '05': '電機機械', '06': '電器電纜', '07': '化學生技醫療', '08': '玻璃陶瓷',
        '09': '造紙工業', '10': '鋼鐵工業', '11': '橡膠工業', '12': '汽車工業',
        '14': '建材營造', '15': '航運業', '16': '觀光餐旅', '17': '金融保險業',
        '18': '貿易百貨業', '20': '其他業', '21': '化學工業', '22': '生技醫療業',
        '23': '油電燃氣業', '24': '半導體業', '25': '電腦及週邊設備業', '26': '光電業',
        '27': '通信網路業', '28': '電子零組件業', '29': '電子通路業', '30': '資訊服務業',
        '31': '其他電子業', '32': '文化創意業', '33': '農業科技業', '34': '電子商務',
        '35': '綠能環保', '36': '數位雲端', '37': '運動休閒', '38': '居家生活',
        '80': '管理股票', '91': '存託憑證', '92': 'ETF', '93': '受益證券', '94': '認購售權證',
    }

    def normalize_industry(value):
        """將 TWSE/TPEx OpenAPI 回傳的產業代碼轉成中文產業別。"""
        txt = str(value or '').strip()
        if txt in ['', '-', 'None', 'null', 'nan']:
            return '未分類'
        # 有些 OpenAPI 只給 05 / 36 這類產業代碼。
        code = txt.zfill(2) if txt.isdigit() and len(txt) <= 2 else txt
        if code in INDUSTRY_CODE_MAP:
            return INDUSTRY_CODE_MAP[code]
        # 有些欄位可能是「36 數位雲端」或「36,數位雲端」。
        m = re.match(r'^(\d{1,2})\s*[、,，\-:：]?\s*(.*)$', txt)
        if m:
            c = m.group(1).zfill(2)
            tail = m.group(2).strip()
            if tail and not tail.isdigit():
                return tail
            return INDUSTRY_CODE_MAP.get(c, txt)
        return txt

    def normalize_stock_name(value):
        """上櫃 OpenAPI 有時只提供公司全名，這裡轉成較適合表格的股票簡稱。"""
        name = str(value or '').strip().replace(' ', '')
        for suffix in ['股份有限公司', '有限公司']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name

    def pick(row, names, default=''):
        for n in names:
            if n in row and row.get(n) not in [None, '', 'null', 'None']:
                return str(row.get(n)).strip()
        return default

    def clean_code(code):
        m = re.search(r'\d{4}', str(code))
        return m.group(0) if m else ''

    def parse_openapi_rows(rows, market_suffix, market_type):
        out = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            code = clean_code(pick(row, ['Code', '證券代號', '公司代號', '有價證券代號', 'SecuritiesCompanyCode', '股票代號']))
            name = pick(row, ['Name', '證券名稱', '公司簡稱', '有價證券名稱', '股票名稱', '公司名稱', 'CompanyName'])
            industry = normalize_industry(pick(row, ['產業別', '產業類別', 'Industry', 'industry'], '未分類'))
            if len(code) == 4 and code.isdigit() and name:
                out.append({
                    'ticker': f'{code}.{market_suffix}',
                    'name': normalize_stock_name(name),
                    'industry': industry or '未分類',
                    'code': code,
                    '市場別': market_type,
                })
        return out

    stocks = []

    # 先用 OpenAPI JSON，速度遠快於 ISIN HTML。
    openapi_sources = [
        # 上市：公司基本資料，有產業別。
        ('https://openapi.twse.com.tw/v1/opendata/t187ap03_L', 'TW', '上市'),
        # 上櫃：公司基本資料，有產業別。
        ('https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O', 'TWO', '上櫃'),
        # 備援：上市每日行情，通常一定有代碼名稱，但可能沒有產業別。
        ('https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL', 'TW', '上市'),
        # 備援：上櫃每日行情。
        ('https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes', 'TWO', '上櫃'),
    ]

    seen = set()
    for url, suffix, market_type in openapi_sources:
        try:
            r = session.get(url, headers=get_headers(), timeout=(2.5, 5))
            if r.status_code != 200 or not r.text.strip():
                continue
            rows = r.json()
            parsed = parse_openapi_rows(rows, suffix, market_type)
            for s in parsed:
                key = s['ticker']
                # 若基本資料已經有產業別，不要被每日行情的未分類覆蓋。
                if key not in seen:
                    stocks.append(s)
                    seen.add(key)
            if len(stocks) >= 1200:
                # 已足夠涵蓋上市櫃，多半不用再跑慢速備援。
                pass
        except Exception:
            continue

    if len(stocks) > 500:
        stocks = sorted(stocks, key=lambda x: x['code'])
        save_local_cache(stocks)
        return stocks

    # OpenAPI 失敗才走舊 ISIN 備援；限制 timeout，避免卡太久。
    try:
        urls = [
            ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', 'TW', '上市'),
            ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', 'TWO', '上櫃'),
        ]
        for url, mkt, market_type in urls:
            r = session.get(url, headers=get_headers(), timeout=(2.5, 6))
            if r.status_code != 200 or not r.text.strip():
                continue
            df_isin = pd.read_html(StringIO(r.text))[0]
            df_isin.columns = df_isin.iloc[0]
            for _, row in df_isin.iloc[1:].iterrows():
                val = str(row.get('有價證券代號及名稱', ''))
                if '　' in val:
                    code, name = val.split('　', 1)
                    if len(code) == 4 and code.isdigit():
                        key = f'{code}.{mkt}'
                        if key in seen:
                            continue
                        stocks.append({
                            'ticker': key,
                            'name': normalize_stock_name(name),
                            'industry': normalize_industry(row.get('產業別', '未分類')), 
                            'code': code,
                            '市場別': market_type,
                        })
                        seen.add(key)
    except Exception:
        pass

    if len(stocks) > 100:
        stocks = sorted(stocks, key=lambda x: x['code'])
        save_local_cache(stocks)
        return stocks

    # 最後保底：回傳舊快取，即使不是今天，也比直接卡住或空白好。
    return load_local_cache(allow_stale=True)

# ============================================================
# [新功能] 公司簡介 與 季度 EPS 列表
# ============================================================
# 台灣公司財報依法為「季揭露」而非月揭露（月度只揭露營收，不含 EPS），
# 因此這裡呈現的是「近 N 季 EPS 列表」，並非每月更新，避免資料誤導使用者。

INDUSTRY_CODE_MAP_PROFILE = {
    '01': '水泥工業', '02': '食品工業', '03': '塑膠工業', '04': '紡織纖維',
    '05': '電機機械', '06': '電器電纜', '07': '化學生技醫療', '08': '玻璃陶瓷',
    '09': '造紙工業', '10': '鋼鐵工業', '11': '橡膠工業', '12': '汽車工業',
    '14': '建材營造', '15': '航運業', '16': '觀光餐旅', '17': '金融保險業',
    '18': '貿易百貨業', '20': '其他業', '21': '化學工業', '22': '生技醫療業',
    '23': '油電燃氣業', '24': '半導體業', '25': '電腦及週邊設備業', '26': '光電業',
    '27': '通信網路業', '28': '電子零組件業', '29': '電子通路業', '30': '資訊服務業',
    '31': '其他電子業', '32': '文化創意業', '33': '農業科技業', '34': '電子商務',
    '35': '綠能環保', '36': '數位雲端', '37': '運動休閒', '38': '居家生活',
}


def _roc_date_to_ad(text: str) -> str:
    """把 TWSE OpenAPI 日期字串轉成可讀日期，西元 8 碼與民國 7 碼皆容錯。"""
    s = re.sub(r"\D", "", str(text or ""))
    if len(s) == 8:  # 西元 YYYYMMDD
        try:
            return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
        except Exception:
            return text
    if len(s) == 7:  # 民國 YYYMMDD
        try:
            return f"{int(s[0:3]) + 1911}-{s[3:5]}-{s[5:7]}"
        except Exception:
            return text
    return text or "N/A"


@st.cache_data(ttl=86400, show_spinner=False)
def get_company_profile(code: str, market_suffix: str) -> dict:
    """公司基本資料：產業別、董事長、總經理、成立/上市日期、實收資本額、官網等。
    來源：TWSE OpenAPI t187ap03_L（上市）／ t187ap03_O（上櫃），公開資訊觀測站官方資料。
    抓不到時回傳空 dict，UI 端顯示「暫無資料」，不影響其他功能。
    """
    url = ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L" if market_suffix == "TW"
           else "https://openapi.twse.com.tw/v1/opendata/t187ap03_O")
    data = _fetch_json(url, timeout=10)
    if not isinstance(data, list):
        return {}

    rec = None
    for item in data:
        if isinstance(item, dict) and str(item.get("公司代號", "")).strip() == str(code):
            rec = item
            break
    if not rec:
        return {}

    industry_raw = str(rec.get("產業別", "")).strip()
    industry = INDUSTRY_CODE_MAP_PROFILE.get(industry_raw.zfill(2), industry_raw) if industry_raw else "未分類"

    cap_raw = _to_float_or_nan(rec.get("實收資本額"))
    cap_text = f"{cap_raw / 1e8:,.1f} 億元" if pd.notna(cap_raw) and cap_raw > 0 else "N/A"

    return {
        "公司名稱": rec.get("公司名稱", "N/A"),
        "產業別": industry or "未分類",
        "董事長": rec.get("董事長", "N/A") or "N/A",
        "總經理": rec.get("總經理", "N/A") or "N/A",
        "成立日期": _roc_date_to_ad(rec.get("成立日期", "")),
        "上市日期": _roc_date_to_ad(rec.get("上市日期", "")),
        "實收資本額": cap_text,
        "住址": rec.get("住址", "N/A") or "N/A",
        "網址": rec.get("網址", "") or "",
        "統一編號": rec.get("營利事業統一編號", "N/A") or "N/A",
    }


def _extract_yahoo_eps_quarterly_from_html(html: str) -> list:
    """從 Yahoo 台股「每股盈餘」頁面解析單季 EPS 列表。
    頁面格式如：2026 Q1 22.08 13.17% 58.28% 1,810.00（EPS／季增率／年增率／季均價）。
    只作為輔助參考資料，非官方財報數字，解析不到時回傳空 list。
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    pattern = r"(20\d{2})\s*Q([1-4])\s+(-?[\d,]+\.?\d*)\s+(-?[\d,]+\.?\d*)%\s+(-?[\d,]+\.?\d*)%\s+(-?[\d,]+\.?\d*)"
    rows = []
    seen = set()
    for m in re.finditer(pattern, text):
        year, q, eps, qoq, yoy, avg_price = m.groups()
        key = (year, q)
        if key in seen:
            continue
        seen.add(key)
        eps_val = _to_float_or_nan(eps)
        if pd.isna(eps_val):
            continue
        rows.append({
            "年季": f"{year} Q{q}",
            "EPS": eps_val,
            "季增率(%)": _to_float_or_nan(qoq),
            "年增率(%)": _to_float_or_nan(yoy),
            "季均價": _to_float_or_nan(avg_price),
        })
        if len(rows) >= 12:  # 最多保留近 12 季，避免頁面雜訊灌太多筆
            break
    return rows


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_quarterly_eps(code: str, market_suffix: str) -> list:
    """單季 EPS 列表（近期，依 Yahoo 股市頁面揭露筆數而定，通常近 8~20 季）。
    台灣公司財報依法為季揭露，這裡的清單以「季」為單位，非每月更新。
    """
    ticker = f"{code}.{market_suffix}"
    try:
        r = requests.get(f"https://tw.stock.yahoo.com/quote/{ticker}/eps",
                         headers=get_headers(), timeout=8, verify=False)
        if r.status_code == 200 and r.text:
            rows = _extract_yahoo_eps_quarterly_from_html(r.text)
            if rows:
                return rows
    except Exception:
        pass
    return []


def _trim_stale_trailing_days(df: pd.DataFrame) -> pd.DataFrame:
    """[修正] 移除資料尾端「非交易日佔位列」。

    yfinance 在週末/假日（或當日尚未收盤）時，有時會在最後補一筆
    「收盤價沿用前一日、成交量為 0」的列。若不處理，掃描邏輯拿
    「最後一筆 vs 倒數第二筆」算漲跌幅／量比時，兩筆會指向同一個
    真實交易日，導致算出 0.0% 漲跌、0.00x 量比（全市場齊漲跌幅掛零，
    一看就知道是資料問題而非市場真的沒漲跌）。

    做法：只裁尾端連續的 0 成交量列，保留最近一個「成交量 > 0」的
    真實交易日作為最後一筆，中間或更早的資料不受影響。
    """
    if df is None or df.empty or "volume" not in df.columns:
        return df
    nonzero_idx = df.index[df["volume"] > 0]
    if len(nonzero_idx) == 0:
        return df.iloc[0:0]
    last_valid = nonzero_idx[-1]
    if last_valid == df.index[-1]:
        return df
    return df.loc[:last_valid]


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
            df = _trim_stale_trailing_days(df)
            result[tk] = df.reset_index(drop=True)
        except:
            pass
    else:
        for tk in tickers:
            try:
                df = raw[tk][["Close", "Volume"]].dropna()
                df.columns = ["close", "volume"]
                df["volume"] = (df["volume"] / 1000).astype(int)
                df = _trim_stale_trailing_days(df)
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
    從 Goodinfo 個股概況頁的「PER」欄位擷取本益比。

    這版專門修正 Goodinfo HTML 有時會被壓成「純文字區塊」或 table 欄位錯位，
    導致嚴格 table 對位抓不到 PER 的問題。

    判斷原則：
    - 必須找到同一段裡的 PBR / PER / PEG 表頭。
    - 數值列最後三個數字依序視為 PBR / PER / PEG。
    - 回傳中間值，也就是 PER。
    """
    soup = BeautifulSoup(html, "html.parser")

    def norm_text(x):
        return re.sub(r"\s+", " ", str(x).replace("\xa0", " ")).strip()

    def is_label_per(x: str) -> bool:
        return norm_text(x).upper() == "PER"

    def valid_pe(x):
        pe = _to_float_or_nan(x)
        return pe if pd.notna(pe) and pe > 0 else np.nan

    def numbers_from_text(x: str):
        # Goodinfo 可能出現 7,098萬、1.08張/筆、+0.8(+1.33%)；只取獨立數值。
        vals = []
        for m in re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", str(x)):
            v = valid_pe(m)
            if pd.notna(v):
                vals.append(float(v))
        return vals

    # 方法 0：直接從 Goodinfo 純文字結構抓。這是最穩定的方式。
    # 例 A：同一列
    # 成交張數 成交金額 成交筆數 成交均張 成交均價 PBR PER PEG
    # 1,196 7,098萬 1,110 1.08張/筆 59.37 1.7 15.71 1.35
    # 例 B：每個儲存格都被拆成一行：PBR / PER / PEG / 1,196 / 7,098萬 / ... / 1.7 / 15.71 / 1.35
    text_lines = [norm_text(x) for x in soup.get_text("\n", strip=True).split("\n")]
    text_lines = [x for x in text_lines if x]

    # 0-1：PBR / PER / PEG 在同一行，下一行或後續數字列的最後三個數字通常為 PBR / PER / PEG。
    for i, line in enumerate(text_lines):
        compact = re.sub(r"\s+", " ", line.upper()).strip()
        if "PBR" in compact and "PER" in compact and "PEG" in compact:
            # 優先抓下一個有足夠數字的文字列。
            for nxt in text_lines[i + 1:i + 8]:
                nums = numbers_from_text(nxt)
                if len(nums) >= 3:
                    return round(float(nums[-2]), 2)

            # 有些情況表頭與數值被合在同一列。
            nums = numbers_from_text(line)
            if len(nums) >= 3:
                return round(float(nums[-2]), 2)

    # 0-2：PBR / PER / PEG 被拆成多個文字節點時，改用 token 序列掃描。
    upper_tokens = [x.upper() for x in text_lines]
    for i in range(len(upper_tokens) - 2):
        if upper_tokens[i] == "PBR" and upper_tokens[i + 1] == "PER" and upper_tokens[i + 2] == "PEG":
            nums = []
            # 往後多抓幾格，因為 Goodinfo 的下一列前面還有成交張數、成交金額、成交均價等欄位。
            for token in text_lines[i + 3:i + 25]:
                nums.extend(numbers_from_text(token))
            if len(nums) >= 3:
                # 最後三個數字為 PBR / PER / PEG，取中間 PER。
                return round(float(nums[-2]), 2)

    # 方法 1：逐列掃描表格，找到 PER 表頭後，抓下一列同欄位。
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [norm_text(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
            cells = [c for c in cells if c != ""]
            if cells:
                rows.append(cells)

        for r_idx, row in enumerate(rows):
            row_upper = [c.upper() for c in row]
            # 若同列同時有 PBR/PER/PEG，下一列最後三個數字通常就是 PBR/PER/PEG。
            if any(c == "PBR" for c in row_upper) and any(c == "PER" for c in row_upper) and any(c == "PEG" for c in row_upper):
                for next_row in rows[r_idx + 1:r_idx + 4]:
                    nums = []
                    for cell in next_row:
                        nums.extend(numbers_from_text(cell))
                    if len(nums) >= 3:
                        return round(float(nums[-2]), 2)

            per_indices = [i for i, c in enumerate(row) if is_label_per(c)]
            for per_idx in per_indices:
                for next_row in rows[r_idx + 1:r_idx + 4]:
                    if len(next_row) > per_idx:
                        pe = valid_pe(next_row[per_idx])
                        if pd.notna(pe):
                            return round(float(pe), 2)

    # 方法 2：pandas 解析 HTML table 後再找 PER。
    try:
        dfs = pd.read_html(StringIO(html))
        for df in dfs:
            df_str = df.astype(str).map(norm_text)
            values = df_str.values.tolist()

            for r_idx, row in enumerate(values):
                row_upper = [x.upper() for x in row]
                if "PBR" in row_upper and "PER" in row_upper and "PEG" in row_upper:
                    for next_row in values[r_idx + 1:r_idx + 4]:
                        nums = []
                        for cell in next_row:
                            nums.extend(numbers_from_text(cell))
                        if len(nums) >= 3:
                            return round(float(nums[-2]), 2)

                per_indices = [i for i, c in enumerate(row) if is_label_per(c)]
                for per_idx in per_indices:
                    for next_row in values[r_idx + 1:r_idx + 4]:
                        if len(next_row) > per_idx:
                            pe = valid_pe(next_row[per_idx])
                            if pd.notna(pe):
                                return round(float(pe), 2)
    except Exception:
        pass

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



# ── 官方 API 本益比高速擷取 ───────────────────────────────
# 改用 TWSE / TPEx 官方一次性批次資料，不再逐檔爬 Goodinfo。
# 優點：速度快、穩定、不容易被擋；掃描時只要抓上市 + 上櫃兩包資料。
PE_CACHE_FILE = os.path.join(
    os.path.dirname(__file__) if '__file__' in globals() else '.',
    "official_pe_cache.json"
)
PE_CACHE_LOCK = threading.Lock()
PE_CACHE_VERSION = "20260609_official_tpex_oldapi_yahoo_fallback_v2"


def _pe_cache_today_key() -> str:
    return get_tw_now().strftime("%Y-%m-%d")


def _load_pe_disk_cache() -> dict:
    try:
        if os.path.exists(PE_CACHE_FILE):
            with open(PE_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("version") == PE_CACHE_VERSION:
                return data
    except Exception:
        pass
    return {"version": PE_CACHE_VERSION, "date": "", "items": {}}


def _save_pe_disk_cache(items: dict, source: str = "official") -> None:
    try:
        payload = {
            "version": PE_CACHE_VERSION,
            "date": _pe_cache_today_key(),
            "source": source,
            "updated_at": get_tw_now().strftime("%Y-%m-%d %H:%M:%S"),
            "items": items,
        }
        tmp = PE_CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, PE_CACHE_FILE)
    except Exception:
        pass


def _clean_pe_value(value) -> float:
    """將官方資料的本益比欄位轉成 float；'-'、'N/A'、空值會回傳 NaN。"""
    if value is None:
        return np.nan
    text = str(value).strip()
    if text in ["", "-", "--", "N/A", "NA", "None", "nan", "NaN", "除權息"]:
        return np.nan
    text = text.replace(",", "").replace("倍", "").replace("％", "%")
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return np.nan
    try:
        pe = float(m.group(0))
        return pe if np.isfinite(pe) and pe > 0 else np.nan
    except Exception:
        return np.nan


def _find_record_code(record: dict) -> str:
    """兼容 TWSE / TPEx OpenAPI 中英文欄位名稱。"""
    if not isinstance(record, dict):
        return ""
    for k, v in record.items():
        key = str(k).lower()
        if key in ["code", "stockno", "stock_id", "stockid", "securitiescompanycode"]:
            s = re.sub(r"\D", "", str(v))
            if len(s) == 4:
                return s
        if any(word in str(k) for word in ["股票代號", "證券代號", "有價證券代號", "代號"]):
            s = re.sub(r"\D", "", str(v))
            if len(s) == 4:
                return s
    # 有些資料第一欄就是代號，但 key 名稱不固定。
    for v in record.values():
        s = re.sub(r"\D", "", str(v).strip())
        if len(s) == 4:
            return s
    return ""


def _find_record_pe(record: dict) -> float:
    """兼容 TWSE / TPEx OpenAPI 本益比欄位名稱。"""
    if not isinstance(record, dict):
        return np.nan

    preferred_keys = []
    for k in record.keys():
        key = str(k).strip()
        low = key.lower()
        if key == "本益比" or low in ["per", "peratio", "peratio", "pe", "peratio"]:
            preferred_keys.append(k)
        elif "本益比" in key or "pe" == low or "per" == low or "peratio" in low or "p/e" in low:
            preferred_keys.append(k)

    for k in preferred_keys:
        pe = _clean_pe_value(record.get(k))
        if pd.notna(pe):
            return round(float(pe), 2)
    return np.nan


def _merge_pe_records(items: dict, rows, source_name: str = "") -> int:
    """把 list[dict] 形式資料合併進 items。"""
    count = 0
    if not isinstance(rows, list):
        return count
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        code = _find_record_code(rec)
        pe = _find_record_pe(rec)
        if code and pd.notna(pe):
            items[code] = round(float(pe), 2)
            count += 1
    return count


def _fetch_json(url: str, timeout: int = 8):
    try:
        r = requests.get(url, headers=get_headers(), timeout=timeout, verify=False)
        if r.status_code != 200 or not r.content:
            return None
        # 官方 API 多為 UTF-8，但保留容錯。
        try:
            return r.json()
        except Exception:
            txt = r.content.decode(r.apparent_encoding or "utf-8", errors="ignore")
            return json.loads(txt)
    except Exception:
        return None


def _load_twse_pe(items: dict) -> int:
    """上市本益比：優先用 TWSE OpenAPI；失敗再用依日期查詢 API 回補。"""
    before = len(items)

    # 1) TWSE OpenAPI：通常直接回傳最新交易日全上市個股本益比。
    data = _fetch_json("https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_d", timeout=8)
    if isinstance(data, list):
        _merge_pe_records(items, data, "TWSE OpenAPI")

    if len(items) > before:
        return len(items) - before

    # 2) 備援：TWSE 舊式 JSON，需指定日期；往前找最近 14 天，避開假日與尚未收盤更新。
    now = get_tw_now()
    for d in range(0, 14):
        query_date = (now - timedelta(days=d)).strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/exchangeReport/BWIBBU_d?response=json&date={query_date}&selectType=ALL"
        payload = _fetch_json(url, timeout=8)
        if not isinstance(payload, dict):
            continue
        fields = payload.get("fields") or []
        data_rows = payload.get("data") or []
        if not fields or not data_rows:
            continue
        rows = []
        for row in data_rows:
            if isinstance(row, list):
                rows.append({str(fields[i]): row[i] for i in range(min(len(fields), len(row)))})
        _merge_pe_records(items, rows, "TWSE date API")
        if len(items) > before:
            return len(items) - before
    return 0


def _load_tpex_pe(items: dict) -> int:
    """
    上櫃本益比：先用 TPEx OpenAPI；若特定股票缺值，再用櫃買舊版 peratio_analysis 批次 JSON 回補。

    3324 雙鴻這類上櫃股票有時在 openapi/v1/tpex_mainboard_peratio_analysis
    欄位結構或資料更新時間不穩，容易合併不到本益比。舊版 pera_result.php
    是櫃買網站「個股本益比、殖利率、股價淨值比」頁面使用的批次資料，
    一次可抓整個上櫃市場，速度仍比逐檔爬 Goodinfo 快很多。
    """
    before = len(items)

    # 1) 新版 TPEx OpenAPI。
    candidate_urls = [
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis",
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis?response=json",
    ]
    for url in candidate_urls:
        data = _fetch_json(url, timeout=8)
        if isinstance(data, list):
            _merge_pe_records(items, data, "TPEx OpenAPI")
        elif isinstance(data, dict):
            for key in ["data", "Data", "aaData", "tables"]:
                if isinstance(data.get(key), list):
                    _merge_pe_records(items, data.get(key), "TPEx OpenAPI")

    # 2) 舊版櫃買批次 JSON 回補。這個來源對 3324 雙鴻較穩。
    now = get_tw_now()
    for d in range(0, 14):
        dt = now - timedelta(days=d)
        roc_date = f"{dt.year - 1911}/{dt.month:02d}/{dt.day:02d}"
        urls = [
            f"https://www.tpex.org.tw/web/stock/aftertrading/peratio_analysis/pera_result.php?l=zh-tw&o=json&d={roc_date}&c=&s=0,asc",
            f"https://www.tpex.org.tw/web/stock/aftertrading/peratio_analysis/pera_result.php?l=zh-tw&d={roc_date}&c=&s=0,asc",
        ]
        for url in urls:
            payload = _fetch_json(url, timeout=8)
            if not isinstance(payload, dict):
                continue
            rows = payload.get("aaData") or payload.get("data") or payload.get("Data") or []
            if not isinstance(rows, list) or not rows:
                continue

            parsed_rows = []
            for row in rows:
                # 常見格式：股票代號, 公司名稱, 本益比, 每股股利, 股利年度, 殖利率, 股價淨值比, 財報年/季
                if isinstance(row, list) and len(row) >= 3:
                    parsed_rows.append({"股票代號": row[0], "本益比": row[2]})
                elif isinstance(row, dict):
                    parsed_rows.append(row)
            _merge_pe_records(items, parsed_rows, "TPEx old pera_result")

            # 只要該日期有抓到一批資料，就不用再往前找。
            if len(parsed_rows) >= 100:
                return len(items) - before

    return len(items) - before


@st.cache_data(ttl=43200, show_spinner=False)
def load_official_pe_map(force_refresh: bool = False) -> dict:
    """
    一次載入上市 + 上櫃本益比對照表。
    掃描期間每支股票只查 dict，不再逐檔 HTTP request，因此速度會比 Goodinfo 快非常多。
    """
    if not force_refresh:
        with PE_CACHE_LOCK:
            cache = _load_pe_disk_cache()
        if cache.get("date") == _pe_cache_today_key() and isinstance(cache.get("items"), dict) and cache["items"]:
            return {str(k): _clean_pe_value(v) for k, v in cache["items"].items()}

    items = {}
    _load_twse_pe(items)
    _load_tpex_pe(items)

    # 若今日 API 因網路或交易所更新時間問題失敗，使用舊快取避免整欄 PE 變 None。
    if not items:
        with PE_CACHE_LOCK:
            old_cache = _load_pe_disk_cache()
        old_items = old_cache.get("items", {}) if isinstance(old_cache, dict) else {}
        if isinstance(old_items, dict) and old_items:
            return {str(k): _clean_pe_value(v) for k, v in old_items.items()}

    with PE_CACHE_LOCK:
        _save_pe_disk_cache(items, source="TWSE_TPEx_OfficialAPI")
    return {str(k): _clean_pe_value(v) for k, v in items.items()}


def _extract_twstock_yahoo_pe_from_html(html: str) -> float:
    """從 Yahoo 台股頁面解析本益比。只作為官方資料缺值時的單檔備援。"""
    if not html:
        return np.nan
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    patterns = [
        # 例：32.03 (84.33)本益比
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:\([^)]*\))?\s*本益比",
        # 例：本益比 32.03
        r"本益比\s*(?:\([^)]*\))?\s*([0-9]+(?:\.[0-9]+)?)",
        # 例：PE Ratio 32.03
        r"PE\s*(?:Ratio)?\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            pe = _clean_pe_value(m.group(1))
            if pd.notna(pe):
                return round(float(pe), 2)
    return np.nan


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_yahoo_tw_pe(code: str, market_suffix: str) -> float:
    """
    官方批次資料缺值時的快速單檔備援。
    Yahoo 台股頁面會顯示本益比，例如 3324.TWO 頁面可讀到「本益比」。
    只在官方 map 找不到時才呼叫，因此不會拖慢整體掃描。
    """
    ticker = f"{code}.{market_suffix}"
    urls = [
        f"https://tw.stock.yahoo.com/quote/{ticker}",
        f"https://tw.stock.yahoo.com/quote/{ticker}/profile",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=get_headers(), timeout=6, verify=False)
            if r.status_code != 200 or not r.text:
                continue
            pe = _extract_twstock_yahoo_pe_from_html(r.text)
            if pd.notna(pe):
                return round(float(pe), 2)
        except Exception:
            continue
    return np.nan


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_pe(ticker: str) -> float:
    """
    本益比取得順序：
    1. 官方批次資料：TWSE / TPEx，一次抓整包，速度最快。
    2. 舊快取：避免官方 API 當下失敗造成整欄空值。
    3. Yahoo 台股單檔備援：只針對官方缺值的少數股票補值，例如 3324 雙鴻。

    這樣可以保留批次速度，又能解決部分上櫃股 PE 缺值。
    """
    code = ticker.split('.')[0]
    market_suffix = "TW" if ticker.endswith(".TW") else "TWO"

    pe_map = load_official_pe_map(False)
    pe = _clean_pe_value(pe_map.get(code))
    if pd.notna(pe):
        return round(float(pe), 2)

    with PE_CACHE_LOCK:
        old_cache = _load_pe_disk_cache()
    old_items = old_cache.get("items", {}) if isinstance(old_cache, dict) else {}
    pe = _clean_pe_value(old_items.get(code))
    if pd.notna(pe):
        return round(float(pe), 2)

    # 官方缺值時才抓 Yahoo，避免每檔都逐頁爬造成變慢。
    pe = fetch_yahoo_tw_pe(code, market_suffix)
    if pd.notna(pe):
        return round(float(pe), 2)

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

@st.cache_data(ttl=3600, show_spinner=False)
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



def warm_kline_data_async(stocks):
    """背景預載前後股票 K 線資料，減少點 PREV / NEXT 時等待。"""
    if not stocks:
        return

    def _worker(items):
        for tk in items:
            try:
                code = str(tk).split(".")[0]
                market = "TW" if str(tk).endswith(".TW") else "TWO"
                get_kline_data(code, market)
            except Exception:
                pass

    try:
        threading.Thread(target=_worker, args=(list(stocks),), daemon=True).start()
    except Exception:
        pass

def draw_k_line(ticker, name, chart_mode='K線圖', chart_period='日'):
    """畫出有實際切換功能的金融圖表。
    chart_mode: K線圖 / 走勢圖 / 技術指標
    chart_period: 日 / 週 / 月
    """
    code   = ticker.split(".")[0]
    market = "TW" if ticker.endswith(".TW") else "TWO"
    df     = get_kline_data(code, market)
    if df.empty or len(df) < 30:
        # [修正] 原本沒有 try/except，Yahoo 連線異常時會直接讓整頁噴例外，
        # 改成失敗時安靜回傳 None，由呼叫端顯示「無法載入 K 線資料」提示。
        try:
            raw = yf.Ticker(ticker).history(period="1y")
        except Exception:
            raw = None
        if raw is None or raw.empty:
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
            "RSI14：%{customdata[7]:.1f}<br>"
            "成交量：%{customdata[8]:,} 張"
            "<extra></extra>"
        ),
        customdata=df[['open', 'close', 'high', 'low', 'MA30', 'MA45', 'MA60', 'RSI14', 'volume']].values,
    ))

    fig.add_trace(go.Scatter(x=df['date'], y=df['MA30'], line=dict(color='#3b82f6', width=1.5), name='MA30', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA45'], line=dict(color='#f59e0b', width=1.5), name='MA45', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], line=dict(color='#a855f7', width=1.5), name='MA60', hoverinfo='skip'))

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

    # 讓 X 軸日期不要每天都顯示：日線顯示每月第一個交易日，週線/月線則等距顯示。
    if chart_period == '日':
        tick_base = df.assign(_ym=df['date_dt'].dt.to_period('M'))
        tick_rows = tick_base.drop_duplicates('_ym', keep='first')
        tick_vals = tick_rows['date'].tolist()
        tick_text = []
        prev_year = None
        for d in tick_rows['date_dt']:
            if prev_year != d.year:
                tick_text.append(d.strftime('%Y-%m'))
                prev_year = d.year
            else:
                tick_text.append(d.strftime('%m月'))
    else:
        step = max(1, len(df) // 8)
        tick_rows = df.iloc[::step]
        tick_vals = tick_rows['date'].tolist()
        tick_text = [d.strftime('%Y-%m') for d in tick_rows['date_dt']]

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
        # 使用 closest + x/y spikelines，讓滑鼠移到圖上時有近似 TradingView 的十字準線。
        # Y 軸 spikeline 會顯示水平價位線，右側價格軸會對應目前游標附近價位。
        hovermode='closest',
        hoverdistance=80,
        spikedistance=-1,
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
        xaxis=dict(**spike_cfg, fixedrange=True, rangeslider=dict(visible=False), tickmode='array', tickvals=tick_vals, ticktext=tick_text, tickangle=0, tickfont=dict(size=11), automargin=True),
        yaxis=dict(
            fixedrange=True, side='right', showgrid=True, gridcolor='rgba(148,163,184,0.09)',
            zeroline=False, tickfont=dict(size=11, color='#cbd5e1'),
            # 水平準線：滑鼠移到 K 線圖上時，顯示對應價位的水平線。
            showspikes=True, spikemode='across+toaxis', spikesnap='cursor',
            spikecolor='rgba(148,163,184,0.82)', spikethickness=1, spikedash='dot',
            hoverformat='.2f',
            automargin=True, ticks='outside', ticklabelposition='outside right', separatethousands=True,
        ),
        yaxis2=dict(
            overlaying='y', side='left', range=[0, max_vol / 0.22],
            showgrid=False, zeroline=False, showticklabels=False, fixedrange=True,
        ),
    )
    return fig


def render_kline_chart_with_axis_price(fig, height=640):
    """用 HTML 方式渲染 Plotly，額外加入 TradingView 類似的游標價位標籤。

    Plotly 原生 spikeline 可以畫水平線，但不會在右側價格軸產生動態價位標籤；
    這裡用一層輕量 JS 讀取游標位置並換算 Y 軸價格，顯示在圖表右側。
    """
    div_id = f"kline_{uuid.uuid4().hex}"
    wrap_id = f"wrap_{uuid.uuid4().hex}"
    line_id = f"hline_{uuid.uuid4().hex}"
    label_id = f"ylabel_{uuid.uuid4().hex}"

    fig.update_layout(height=height, autosize=True)
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=div_id,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "showTips": False,
            "responsive": True,
        },
    )

    html = f"""
    <div id="{wrap_id}" style="position:relative;width:100%;height:{height}px;background:#0b121b;overflow:hidden;">
      {plot_html}
      <div id="{line_id}" style="display:none;position:absolute;height:0;border-top:1px dashed rgba(203,213,225,.88);pointer-events:none;z-index:9998;"></div>
      <div id="{label_id}" style="display:none;position:absolute;min-width:58px;padding:3px 8px;border-radius:4px;background:#2563eb;color:#ffffff;font-family:Roboto Mono,Consolas,monospace;font-size:12px;font-weight:700;text-align:center;line-height:18px;box-shadow:0 0 0 1px rgba(191,219,254,.45),0 6px 18px rgba(0,0,0,.35);pointer-events:none;z-index:9999;"></div>
    </div>
    <script>
    (function() {{
      const gd = document.getElementById("{div_id}");
      const wrap = document.getElementById("{wrap_id}");
      const hline = document.getElementById("{line_id}");
      const ylabel = document.getElementById("{label_id}");

      function formatPrice(v) {{
        if (!isFinite(v)) return "";
        const abs = Math.abs(v);
        const digits = abs >= 1000 ? 0 : abs >= 100 ? 1 : 2;
        return v.toLocaleString(undefined, {{ minimumFractionDigits: digits, maximumFractionDigits: digits }});
      }}

      function hideGuide() {{
        hline.style.display = "none";
        ylabel.style.display = "none";
      }}

      function updateGuide(ev) {{
        if (!gd || !gd._fullLayout || !gd._fullLayout.yaxis) return;
        const dragLayer = gd.querySelector('.nsewdrag');
        if (!dragLayer) return;

        const plotRect = dragLayer.getBoundingClientRect();
        const wrapRect = wrap.getBoundingClientRect();
        if (ev.clientX < plotRect.left || ev.clientX > plotRect.right || ev.clientY < plotRect.top || ev.clientY > plotRect.bottom) {{
          hideGuide();
          return;
        }}

        const yAxis = gd._fullLayout.yaxis;
        const py = ev.clientY - plotRect.top;
        const price = yAxis.p2l(py);
        const top = plotRect.top - wrapRect.top + py;
        const left = plotRect.left - wrapRect.left;
        const right = plotRect.right - wrapRect.left;

        hline.style.display = "block";
        hline.style.left = left + "px";
        hline.style.top = top + "px";
        hline.style.width = (plotRect.right - plotRect.left) + "px";

        ylabel.textContent = formatPrice(price);
        ylabel.style.display = "block";
        ylabel.style.top = (top - 12) + "px";
        ylabel.style.left = Math.min(right + 6, wrapRect.width - 72) + "px";
      }}

      function bind() {{
        if (!gd || !gd._fullLayout) {{ setTimeout(bind, 150); return; }}
        gd.addEventListener('mousemove', updateGuide);
        gd.addEventListener('mouseleave', hideGuide);
        window.addEventListener('resize', hideGuide);
      }}
      bind();
    }})();
    </script>
    """
    components.html(html, height=height + 8, scrolling=False)

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500;600&family=Noto+Sans+TC:wght@400;500;600;700;800&display=swap');
:root{
 --bg:#07111f;--panel:#0d1a2b;--panel2:#102138;--border:#233a55;
 --text:#edf4ff;--muted:#9db0c6;--blue:#4c8dff;--green:#35c48d;
 --red:#ff6474;--yellow:#ffc857;--shadow:0 14px 36px rgba(0,0,0,.22);
}
[data-testid='stHeader'],[data-testid='stToolbar'],[data-testid='stSidebar'],
[data-testid='collapsedControl'],[data-testid='stSidebarCollapseButton']{display:none!important;}
html,body,[data-testid='stAppViewContainer'],[data-testid='stMain']{
 background:radial-gradient(circle at 10% 0%,rgba(76,141,255,.13),transparent 30%),linear-gradient(180deg,#081321,#050c16)!important;
 color:var(--text)!important;font-family:'Inter','Noto Sans TC',sans-serif!important;
}
.block-container{max-width:1500px!important;padding:24px 28px 56px!important;}
.app-hero{display:flex;justify-content:space-between;align-items:center;gap:20px;padding:22px 24px;margin-bottom:16px;background:linear-gradient(135deg,rgba(16,33,56,.98),rgba(9,20,35,.98));border:1px solid var(--border);border-radius:18px;box-shadow:var(--shadow);}
.app-title{font-size:28px;font-weight:800;letter-spacing:.2px;color:#fff}.app-sub{font-size:13px;color:var(--muted);margin-top:7px;line-height:1.7}.app-meta{text-align:right;color:var(--muted);font-family:'Roboto Mono',monospace;font-size:12px;line-height:1.8;white-space:nowrap}
.workflow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:0 0 16px}.workflow-step{padding:12px 14px;border:1px solid var(--border);border-radius:12px;background:rgba(13,26,43,.85);color:var(--muted);font-size:12px;font-weight:700}.workflow-step b{display:inline-flex;width:23px;height:23px;border-radius:50%;align-items:center;justify-content:center;background:rgba(76,141,255,.18);color:#9fc0ff;margin-right:7px}.workflow-step.active{border-color:#4777b8;background:rgba(76,141,255,.10);color:#eaf2ff}
.section-head{display:flex;justify-content:space-between;align-items:end;margin:22px 0 10px}.section-title{font-size:18px;font-weight:800;color:#f4f8ff}.section-help{font-size:12px;color:var(--muted)}
.control-shell,.tv-panel,.tv-card,.side-card,[data-testid='stDataFrame']{background:linear-gradient(180deg,rgba(16,33,56,.96),rgba(9,20,35,.98))!important;border:1px solid var(--border)!important;border-radius:14px!important;box-shadow:var(--shadow)}
.control-shell{padding:8px 16px 14px;margin-bottom:12px}.control-note{padding:13px 15px;border-radius:11px;background:rgba(76,141,255,.08);border:1px solid rgba(76,141,255,.22);color:#bcd0e8;font-size:13px;line-height:1.75}
[data-testid='stExpander']{background:transparent!important;border:0!important}[data-testid='stExpander'] details{border:0!important}[data-testid='stExpander'] summary{font-weight:800!important;color:#f1f6ff!important;font-size:15px!important}
[data-testid='stButton']>button,[data-testid='stDownloadButton']>button{border-radius:10px!important;border:1px solid #34547b!important;background:#142944!important;color:#eff5ff!important;font-weight:800!important;min-height:44px!important;transition:.16s ease!important}
[data-testid='stButton']>button:hover,[data-testid='stDownloadButton']>button:hover{border-color:var(--blue)!important;background:#19365b!important;box-shadow:0 0 0 3px rgba(76,141,255,.13)!important}
[data-testid='stButton']>button[kind='primary']{background:linear-gradient(135deg,#3978e8,#4c8dff)!important;border-color:#679fff!important;color:#fff!important}
[data-testid='stNumberInput'] input{background:#091625!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:9px!important}.stSlider{padding-top:2px}
[data-testid='stProgress']>div{background:#13243a!important;border-radius:999px!important;height:9px!important}[data-testid='stProgress']>div>div{background:linear-gradient(90deg,#3978e8,#35c48d)!important;border-radius:999px!important}
[data-testid='stAlert']{border-radius:12px!important;background:rgba(76,141,255,.09)!important;border:1px solid rgba(76,141,255,.24)!important}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}.tv-card{padding:16px}.tv-label{color:var(--muted);font-size:11px;font-weight:800;letter-spacing:.06em}.tv-value{font-family:'Roboto Mono',monospace;font-size:25px;font-weight:800;margin-top:6px;color:var(--text)}.tv-caption{color:var(--muted);font-size:12px;margin-top:5px}
.tv-section{font-size:17px;font-weight:800;color:#f4f8ff;margin:22px 0 10px}.quote-panel{padding:18px 20px;margin:18px 0 12px;background:linear-gradient(135deg,rgba(16,33,56,.98),rgba(8,20,35,.98));border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow)}.quote-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.quote-title{font-size:24px;font-weight:800}.quote-tag,.bias-chip{border:1px solid var(--border);background:#13243a;border-radius:999px;padding:5px 10px;color:#b9c9dc;font-size:12px;font-weight:700}.quote-price{font-family:'Roboto Mono',monospace;font-size:38px;font-weight:800;color:var(--green);line-height:1.25}.quote-change{font-family:'Roboto Mono',monospace;font-size:17px;font-weight:700;margin-left:8px}.quote-metrics{display:grid;grid-template-columns:repeat(3,minmax(110px,1fr));gap:12px;margin-top:12px;max-width:700px}.metric-k{color:var(--muted);font-size:11px;font-weight:700}.metric-v{font-family:'Roboto Mono',monospace;font-size:15px;font-weight:700;margin-top:3px}
.side-card{padding:16px;margin-bottom:12px}.side-title{display:flex;justify-content:space-between;gap:8px;font-size:16px;font-weight:800;margin-bottom:10px}.report-row{display:flex;justify-content:space-between;border-bottom:1px solid rgba(35,58,85,.65);padding:8px 0;color:#c7d5e6;font-size:13px}.radar-check{font-size:13px;line-height:1.9;color:#c6d4e5}.radar-check b{color:var(--green);margin-right:7px}

[data-baseweb='tab-list']{gap:8px;background:rgba(13,26,43,.78);border:1px solid var(--border);border-radius:14px;padding:7px;margin:4px 0 18px;overflow-x:auto;}
[data-baseweb='tab']{height:46px;border-radius:10px;padding:0 18px;color:var(--muted)!important;font-weight:800!important;white-space:nowrap;}
[data-baseweb='tab'][aria-selected='true']{background:linear-gradient(135deg,#3978e8,#4c8dff)!important;color:#fff!important;box-shadow:0 8px 20px rgba(57,120,232,.25);}
[data-baseweb='tab-highlight']{display:none!important;}
[data-baseweb='tab-panel']{padding-top:2px;}
[data-testid='stDataFrame']{overflow:hidden!important}[data-testid='stDataFrame'] *{font-family:'Roboto Mono','Noto Sans TC',monospace!important}.news-card{background:linear-gradient(180deg,rgba(16,33,56,.9),rgba(9,20,35,.96))!important;border-radius:12px!important}.news-title:hover{color:#8eb6ff!important}
/* [新版面] 主從式雙欄工作台：左側候選清單常駐 + 右側個股工作台（K線/AI/新聞 分段切換） */
.workspace-left{background:linear-gradient(180deg,rgba(16,33,56,.96),rgba(9,20,35,.98));border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);padding:14px;max-height:900px;overflow-y:auto;}
.workspace-right{min-height:600px;}
.workspace-left [data-testid='stDataFrame']{border:0!important;box-shadow:none!important;}
[data-testid='stSegmentedControl']{margin-bottom:14px;}
[data-testid='stSegmentedControl'] label{border-radius:10px!important;font-weight:800!important;}
.candidate-row-hint{font-size:11px;color:var(--muted);margin:2px 0 8px;}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-track{background:#07111f}::-webkit-scrollbar-thumb{background:#29425f;border-radius:999px}
@media(max-width:1000px){.block-container{padding:16px 14px 40px!important}.workflow,.stat-grid{grid-template-columns:repeat(2,1fr)}.app-hero{align-items:flex-start}.app-meta{display:none}.workspace-left{max-height:420px;}}
@media(max-width:700px){.workflow,.stat-grid,.quote-metrics{grid-template-columns:1fr}.app-title{font-size:22px}.quote-price{font-size:31px}.section-help{display:none}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 4. 頁首與操作流程
# ============================================================
_now_str = get_tw_now().strftime("%Y-%m-%d %H:%M")
_signal_count = len(st.session_state.scan_results) if isinstance(st.session_state.scan_results, pd.DataFrame) else 0
_current_step = 3 if _signal_count else (2 if st.session_state.is_scanning else 1)
st.markdown(f"""
<div class="app-hero">
  <div>
    <div class="app-title">台股智慧選股</div>
    <div class="app-sub">依均線多頭排列、乖離率與成交量快速篩選，再整合 AI 評分、K 線、營收、本益比與個股新聞。</div>
  </div>
  <div class="app-meta">目前訊號 {_signal_count} 檔<br>資料時間 {_now_str}</div>
</div>
<div class="workflow">
  <div class="workflow-step {'active' if _current_step >= 1 else ''}"><b>1</b>設定篩選條件</div>
  <div class="workflow-step {'active' if _current_step >= 2 else ''}"><b>2</b>執行全市場掃描</div>
  <div class="workflow-step {'active' if _current_step >= 3 else ''}"><b>3</b>從清單選擇股票</div>
  <div class="workflow-step {'active' if _current_step >= 3 else ''}"><b>4</b>查看圖表與分析</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 5. 兩頁籤工作台（掃描 ／ 候選與分析工作台）
# ============================================================
# [新版面] 原本「候選清單／AI分析／個股新聞」三個頂層頁籤合併為一個
# 「主從式雙欄工作台」：左側候選清單常駐可見，右側個股工作台用分段
# 切換（K線圖／AI分析／個股新聞）取代原本三個各自獨立的頁籤與各自
# 獨立的 selectbox，選一次股票，三種檢視都同步。
user_bias = st.session_state.user_bias
user_vol = st.session_state.user_vol

tab_scan, tab_workspace = st.tabs([
    "🔍 選股掃描",
    "📊 候選與分析工作台",
])

# ------------------------------------------------------------
# TAB 1：選股掃描
# ------------------------------------------------------------
with tab_scan:
    st.markdown('<div class="section-head"><div><div class="section-title">設定掃描條件</div><div class="section-help">條件越嚴格，候選股票通常越少；第一次使用可保留預設值。</div></div></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="control-shell">', unsafe_allow_html=True)
        note_col, bias_col, vol_col = st.columns([1.6, 1, 1], gap="large")
        with note_col:
            st.markdown("""
            <div class="control-note">
              <b>選股策略</b><br>
              尋找 MA30 &gt; MA45 &gt; MA60 的多頭排列股票，限制股價不可距離 30 日均線過遠，並排除成交量不足的標的。
            </div>
            """, unsafe_allow_html=True)
        with bias_col:
            mb_bias = st.number_input(
                "30MA 乖離上限 (%)", 0.1, 15.0,
                value=float(st.session_state.user_bias), step=0.1, key="mb_bias",
                help="數值越小，越偏向尋找貼近 30 日均線的股票。"
            )
            st.session_state.user_bias = mb_bias
        with vol_col:
            mb_vol = st.slider(
                "最小成交量 (張)", 0, 3000,
                value=int(st.session_state.user_vol), key="mb_vol",
                help="排除流動性較低的股票。"
            )
            st.session_state.user_vol = mb_vol
        st.markdown('</div>', unsafe_allow_html=True)

    user_bias = st.session_state.user_bias
    user_vol = st.session_state.user_vol
    scan_left, scan_right = st.columns([3, 1])
    with scan_left:
        st.caption(f"目前條件：30MA 乖離 ≤ {user_bias:.1f}%｜成交量 ≥ {user_vol:,} 張｜MA30 > MA45 > MA60")
    with scan_right:
        scan_clicked = st.button(
            "開始全市場掃描", use_container_width=True,
            disabled=st.session_state.is_scanning, type="primary", key="scan_market"
        )

    if scan_clicked:
        st.session_state.is_scanning = True
        st.session_state.current_idx = 0
        st.session_state.last_selected_row = None
        st.rerun()

    if st.session_state.is_scanning:
        status = st.empty()
        bar = st.progress(0)
        BATCH = 200

        status.info("步驟 1/3：正在載入上市櫃股票清單…")
        bar.progress(0.03)
        stock_map = get_stock_market_list()
        all_tickers = [s["ticker"] for s in stock_map]
        total_tickers = len(all_tickers)

        history_map = {}
        batches = [all_tickers[i:i+BATCH] for i in range(0, total_tickers, BATCH)]
        for bi, batch in enumerate(batches):
            status.info(f"步驟 2/3：下載歷史行情批次 {bi+1}/{len(batches)}…")
            bar.progress(0.03 + 0.72 * (bi / max(1, len(batches))))
            history_map.update(download_batch_history(tuple(batch)))

        bar.progress(0.75)
        status.info("步驟 3/3：正在計算均線、量能與候選股票…")
        initial_hits = calc_ma_signals(history_map, stock_map, user_bias, user_vol)
        bar.progress(0.80)

        if initial_hits:
            status.info(f"已找到 {len(initial_hits)} 檔候選股，正在補齊本益比、營收與 AI 評分…")
            load_official_pe_map(False)
            final_list = []
            with ThreadPoolExecutor(max_workers=6) as ex:
                f_deep = {ex.submit(fetch_deep_info, r["ticker"]): r for r in initial_hits}
                for j, f in enumerate(as_completed(f_deep), 1):
                    bar.progress(0.80 + 0.19 * j / len(initial_hits))
                    deep_res = f.result()
                    base = f_deep[f]
                    row_data = {
                        "ticker": base["ticker"], "code": base["code"], "name": base["name"],
                        "industry": base["industry"],
                        "市場別": base.get("市場別", "上市" if str(base.get("ticker", "")).endswith(".TW") else "上櫃" if str(base.get("ticker", "")).endswith(".TWO") else "興櫃"),
                        "收盤": base["收盤"], "漲跌幅(%)": base.get("漲跌幅(%)", np.nan),
                        "乖離30MA(%)": base["乖離30MA(%)"], "成交量(張)": base["成交量(張)"],
                        "量變動(%)": base["量變動(%)"], "量比20日": base.get("量比20日", np.nan),
                        "主力成本": base.get("主力成本", np.nan),
                        "主力成本乖離(%)": base.get("主力成本乖離(%)", np.nan),
                        "RSI14": base.get("RSI14", np.nan), "MACD柱": base.get("MACD柱", np.nan),
                        "突破20日高": base.get("突破20日高", False), "接近60日高": base.get("接近60日高", False),
                        "本益比": deep_res["pe"], "營收月增": deep_res["mom"], "營收年增": deep_res["yoy"],
                    }
                    score, radar = calc_stock_score(row_data)
                    row_data["AI評分"] = score
                    row_data["飆股雷達"] = "、".join(radar) if radar else "觀察"
                    final_list.append(row_data)
            bar.progress(1.0)
            st.session_state.scan_results = pd.DataFrame(final_list).sort_values("AI評分", ascending=False).reset_index(drop=True)
            status.success(f"掃描完成，共找到 {len(st.session_state.scan_results)} 檔候選股票。")
        else:
            st.session_state.scan_results = pd.DataFrame()
            status.warning("查無符合目前條件的股票，請放寬乖離或成交量條件後再試。")

        st.session_state.is_scanning = False
        st.rerun()

    if not st.session_state.scan_results.empty:
        scan_df = st.session_state.scan_results
        avg_score = scan_df['AI評分'].mean() if 'AI評分' in scan_df.columns else np.nan
        strong_count = int((scan_df['AI評分'] >= 80).sum()) if 'AI評分' in scan_df.columns else 0
        st.markdown(f"""
        <div class="stat-grid" style="margin-top:18px;">
          <div class="tv-card"><div class="tv-label">最近掃描結果</div><div class="tv-value">{len(scan_df)}</div><div class="tv-caption">符合條件股票</div></div>
          <div class="tv-card"><div class="tv-label">平均 AI 評分</div><div class="tv-value">{'N/A' if pd.isna(avg_score) else f'{avg_score:.0f}'}</div><div class="tv-caption">滿分 100 分</div></div>
          <div class="tv-card"><div class="tv-label">強勢候選</div><div class="tv-value">{strong_count}</div><div class="tv-caption">AI 評分 80 分以上</div></div>
          <div class="tv-card"><div class="tv-label">下一步</div><div class="tv-value" style="font-size:18px">候選與分析工作台</div><div class="tv-caption">切換下一個頁籤選股並看圖</div></div>
        </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.is_scanning:
        st.markdown("""
        <div class="tv-panel" style="text-align:center;padding:42px 22px;margin-top:18px;">
          <div style="font-size:40px;margin-bottom:12px;">🔎</div>
          <div style="font-size:20px;font-weight:800;color:#f4f8ff;">尚未產生掃描結果</div>
          <div class="tv-caption" style="margin-top:9px;line-height:1.8;">設定條件後按下「開始全市場掃描」。<br>完成後請切換到「候選與分析工作台」頁籤。</div>
        </div>
        """, unsafe_allow_html=True)

# 共用結果與目前股票
has_results = isinstance(st.session_state.scan_results, pd.DataFrame) and not st.session_state.scan_results.empty
if has_results:
    df = st.session_state.scan_results.copy()
    total_found = len(df)
    if st.session_state.current_idx >= total_found:
        st.session_state.current_idx = 0
    current_stock = df.iloc[st.session_state.current_idx]
else:
    df = pd.DataFrame()
    total_found = 0
    current_stock = None

# ------------------------------------------------------------
# TAB 2：候選與分析工作台（主從式雙欄：左清單常駐 + 右側分段切換）
# ------------------------------------------------------------
with tab_workspace:
    if has_results:
        avg_score = df['AI評分'].mean() if 'AI評分' in df.columns else np.nan
        strong_count = int((df['AI評分'] >= 80).sum()) if 'AI評分' in df.columns else 0
        st.markdown(f"""
        <div class="stat-grid">
          <div class="tv-card"><div class="tv-label">符合條件</div><div class="tv-value">{total_found}</div><div class="tv-caption">本次候選股票</div></div>
          <div class="tv-card"><div class="tv-label">平均 AI 評分</div><div class="tv-value">{'N/A' if pd.isna(avg_score) else f'{avg_score:.0f}'}</div><div class="tv-caption">滿分 100 分</div></div>
          <div class="tv-card"><div class="tv-label">強勢候選</div><div class="tv-value">{strong_count}</div><div class="tv-caption">AI 評分 80 分以上</div></div>
          <div class="tv-card"><div class="tv-label">目前選擇</div><div class="tv-value" style="font-size:18px">{current_stock['code']} {current_stock['name']}</div><div class="tv-caption">K線／AI分析／新聞同步顯示</div></div>
        </div>
        """, unsafe_allow_html=True)

        left_col, right_col = st.columns([1.3, 2.4], gap="medium")

        # ══════════════ 左欄：候選清單（常駐，切換右側檢視時不消失）══════════════
        with left_col:
            with st.container(border=True):
                st.markdown('<div class="section-title" style="font-size:15px;margin-bottom:8px;">候選股票清單</div><div class="candidate-row-hint">點擊任一列即可切換右側個股工作台。</div>', unsafe_allow_html=True)

                quick_filter = st.selectbox(
                    "快速篩選", ["全部候選", "AI 評分 80 分以上", "營收年增為正", "量比 1.5 倍以上", "突破 20 日高"],
                    key="candidate_filter"
                )
                view_df = df.copy()
                if quick_filter == "AI 評分 80 分以上":
                    view_df = view_df[view_df['AI評分'] >= 80]
                elif quick_filter == "營收年增為正":
                    view_df = view_df[pd.to_numeric(view_df['營收年增'], errors='coerce') > 0]
                elif quick_filter == "量比 1.5 倍以上":
                    view_df = view_df[pd.to_numeric(view_df['量比20日'], errors='coerce') >= 1.5]
                elif quick_filter == "突破 20 日高":
                    view_df = view_df[view_df['突破20日高'] == True]

                csv = view_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("下載目前清單 CSV", csv, f'tw_stock_scan_{get_tw_now().strftime("%Y%m%d")}.csv', 'text/csv', use_container_width=True)

                show_cols = ["code", "name", "AI評分", "收盤", "漲跌幅(%)", "量比20日"]
                available_cols = [c for c in show_cols if c in view_df.columns]
                df_display = view_df[available_cols].rename(columns={"code": "代碼", "name": "名稱"})

                def color_tw_style(val):
                    if pd.isna(val): return ''
                    color = '#22ab94' if val > 0 else '#f23645' if val < 0 else '#e6edf3'
                    return f'color: {color}; font-weight: bold'

                event = st.dataframe(
                    df_display.style.map(color_tw_style, subset=[c for c in ['漲跌幅(%)'] if c in df_display.columns]),
                    use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row",
                    key=f"stock_table_{st.session_state.table_key}",
                    height=560,
                    column_config={
                        "代碼": st.column_config.TextColumn("代碼", width=62),
                        "名稱": st.column_config.TextColumn("名稱", width=80),
                        "AI評分": st.column_config.ProgressColumn("AI評分", width=90, format="%d", min_value=0, max_value=100),
                        "收盤": st.column_config.NumberColumn("價格", width=65, format="%.2f"),
                        "漲跌幅(%)": st.column_config.NumberColumn("漲跌", width=65, format="%.1f%%"),
                        "量比20日": st.column_config.NumberColumn("量比", width=60, format="%.2fx"),
                    }
                )
                if event and "selection" in event and event["selection"]["rows"]:
                    clicked_view_row = event["selection"]["rows"][0]
                    selected_code = str(view_df.iloc[clicked_view_row]['code'])
                    matches = df.index[df['code'].astype(str) == selected_code].tolist()
                    if matches:
                        st.session_state.current_idx = matches[0]
                        st.session_state.last_selected_row = clicked_view_row
                        current_stock = df.iloc[st.session_state.current_idx]

        # ══════════════ 右欄：個股工作台（報價 + 分段切換 K線／AI分析／新聞）══════════════
        with right_col:
            nav_space, nav1, nav2 = st.columns([3.4, 0.8, 0.8])
            with nav1:
                if st.button("← 上一檔", use_container_width=True, key="chart_prev"):
                    st.session_state.current_idx = (st.session_state.current_idx - 1) % total_found
                    st.rerun()
            with nav2:
                if st.button("下一檔 →", use_container_width=True, key="chart_next"):
                    st.session_state.current_idx = (st.session_state.current_idx + 1) % total_found
                    st.rerun()

            price = current_stock.get('收盤', np.nan)
            chg = current_stock.get('漲跌幅(%)', np.nan)
            chg_color = 'var(--green)' if pd.notna(chg) and chg >= 0 else 'var(--red)'
            chg_txt = 'N/A' if pd.isna(chg) else f"{chg:+.2f}%"
            prev_est = np.nan if pd.isna(price) or pd.isna(chg) or chg == -100 else price / (1 + chg / 100)
            chg_amt = np.nan if pd.isna(prev_est) else price - prev_est
            chg_amt_txt = 'N/A' if pd.isna(chg_amt) else f"{chg_amt:+.2f}"
            score_val = current_stock.get('AI評分', np.nan)
            score_txt = 'N/A' if pd.isna(score_val) else f"{int(score_val)} / 100"
            vol_ratio_txt = fmt_num(current_stock.get('量比20日', np.nan), '{:.2f}x')

            st.markdown(f"""
            <div class="quote-panel" style="margin-top:8px;">
              <div class="quote-head"><div class="quote-title">{current_stock['name']} · {current_stock['code']}</div><div class="quote-tag">{current_stock.get('市場別', '')}</div><div class="quote-tag">{current_stock.get('industry', '未分類')}</div></div>
              <div><span class="quote-price">{fmt_num(price, '{:.2f}')}</span><span class="quote-change" style="color:{chg_color};">{chg_amt_txt} ({chg_txt})</span></div>
              <div class="quote-metrics">
                <div><div class="metric-k">AI 評分</div><div class="metric-v">{score_txt}</div></div>
                <div><div class="metric-k">成交量</div><div class="metric-v">{fmt_num(current_stock.get('成交量(張)', np.nan), '{:,.0f}')} 張</div></div>
                <div><div class="metric-k">量比20日</div><div class="metric-v">{vol_ratio_txt}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # [新版面] 分段切換：取代原本 K線圖／AI分析／個股新聞 三個獨立頁籤，
            # 選一次股票、切換這裡即可，不會重新觸發選股、也不會弄丟左側清單。
            view_mode = st.segmented_control(
                "檢視模式",
                ["📈 K線圖", "🤖 AI 分析", "🏢 公司資訊", "📰 個股新聞"],
                default="📈 K線圖",
                key="detail_view_mode",
                label_visibility="collapsed",
            )
            view_mode = view_mode or "📈 K線圖"

            # ---------- K 線圖 ----------
            if view_mode == "📈 K線圖":
                k_fig = draw_k_line(current_stock['ticker'], current_stock['name'], chart_mode='K線圖', chart_period='日')
                if k_fig:
                    render_kline_chart_with_axis_price(k_fig, height=560)
                    try:
                        preload_tickers = [df.iloc[(st.session_state.current_idx + offset) % total_found]['ticker'] for offset in (-1, 1)]
                        warm_kline_data_async(preload_tickers)
                    except Exception:
                        pass
                else:
                    st.warning("無法載入 K 線資料，請稍後再試。")

            # ---------- AI 分析 ----------
            elif view_mode == "🤖 AI 分析":
                report = build_ai_report(current_stock)
                rsi_txt = fmt_num(current_stock.get('RSI14', np.nan), '{:.1f}')
                macd_txt = fmt_num(current_stock.get('MACD柱', np.nan), '{:.3f}')
                main_cost_txt = fmt_num(current_stock.get('主力成本', np.nan), '{:.2f}')
                cost_gap_txt = fmt_num(current_stock.get('主力成本乖離(%)', np.nan), '{:+.2f}%')
                radar_items = [x for x in str(current_stock.get('飆股雷達', '')).split('、') if x and x != '觀察'] or ['等待更強共振訊號']

                st.markdown(f"""
                <div class="side-card" style="margin-top:2px;">
                  <div class="side-title"><span>盤面強弱</span><span class="bias-chip">{report['盤面強弱']}</span></div>
                  <div class="report-row"><span>RSI14</span><span>{rsi_txt}</span></div>
                  <div class="report-row"><span>MACD 柱</span><span>{macd_txt}</span></div>
                  <div class="report-row"><span>主力成本</span><span>{main_cost_txt}</span></div>
                  <div class="report-row"><span>成本乖離</span><span>{cost_gap_txt}</span></div>
                </div>
                """, unsafe_allow_html=True)

                ai_left, ai_right = st.columns([1.6, 1], gap="medium")
                with ai_left:
                    st.markdown(f"""
                    <div class="side-card">
                      <div class="side-title">AI 綜合判讀</div>
                      <div style="color:#c7d5e6;font-size:13px;line-height:1.9;"><b>趨勢結構</b><br>{report['趨勢結構']}<br><br><b>動能訊號</b><br>{report['動能訊號']}<br><br><b>財務訊號</b><br>{report['財務訊號']}</div>
                    </div>
                    <div class="side-card"><div class="side-title">高機率劇本</div><div style="color:#c7d5e6;font-size:14px;line-height:1.9;">{report['劇本']}</div></div>
                    """, unsafe_allow_html=True)
                with ai_right:
                    st.markdown(f"""
                    <div class="side-card">
                      <div class="side-title"><span>飆股雷達</span><span class="bias-chip">HOT</span></div>
                      <div class="radar-check">{''.join([f'<div><b>✓</b>{item}</div>' for item in radar_items[:8]])}</div>
                    </div>
                    <div class="side-card"><div class="side-title">風險提醒</div><div style="color:#c7d5e6;font-size:13px;line-height:1.9;">{report['主力成本']}<br><br>AI 評分是條件整合結果，不代表保證獲利；仍需搭配停損、部位控管與市場環境判斷。</div></div>
                    """, unsafe_allow_html=True)

            # ---------- 公司資訊：公司簡介 + 季度 EPS 列表 ----------
            elif view_mode == "🏢 公司資訊":
                market_suffix = "TW" if current_stock['ticker'].endswith(".TW") else "TWO"
                profile = get_company_profile(current_stock['code'], market_suffix)

                st.markdown('<div class="section-head" style="margin-top:4px;"><div><div class="section-title" style="font-size:15px;">公司簡介</div><div class="section-help">資料來源：公開資訊觀測站（TWSE OpenAPI）。</div></div></div>', unsafe_allow_html=True)
                if profile:
                    website_html = f'<a href="{profile["網址"]}" target="_blank" style="color:#8eb6ff;">{profile["網址"]}</a>' if profile.get("網址") else "N/A"
                    st.markdown(f"""
                    <div class="side-card">
                      <div class="report-row"><span>公司全名</span><span>{profile['公司名稱']}</span></div>
                      <div class="report-row"><span>產業別</span><span>{profile['產業別']}</span></div>
                      <div class="report-row"><span>董事長</span><span>{profile['董事長']}</span></div>
                      <div class="report-row"><span>總經理</span><span>{profile['總經理']}</span></div>
                      <div class="report-row"><span>成立日期</span><span>{profile['成立日期']}</span></div>
                      <div class="report-row"><span>上市/上櫃日期</span><span>{profile['上市日期']}</span></div>
                      <div class="report-row"><span>實收資本額</span><span>{profile['實收資本額']}</span></div>
                      <div class="report-row"><span>統一編號</span><span>{profile['統一編號']}</span></div>
                      <div class="report-row" style="border-bottom:0;"><span>公司網址</span><span>{website_html}</span></div>
                      <div class="tv-caption" style="margin-top:10px;">{profile['住址']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("目前無法取得公司基本資料，可能是新股或官方資料尚未更新，請稍後再試。")

                st.markdown('<div class="section-head" style="margin-top:22px;"><div><div class="section-title" style="font-size:15px;">單季 EPS 列表</div><div class="section-help">台灣財報依法採季揭露，非每月更新；資料來源：Yahoo 股市，僅供參考。</div></div></div>', unsafe_allow_html=True)
                eps_rows = fetch_quarterly_eps(current_stock['code'], market_suffix)
                if eps_rows:
                    eps_df = pd.DataFrame(eps_rows)
                    ttm_eps = eps_df['EPS'].head(4).sum()
                    st.markdown(f"""
                    <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:10px;">
                      <div class="tv-card"><div class="tv-label">最新單季 EPS</div><div class="tv-value">{eps_df.iloc[0]['EPS']:.2f}</div><div class="tv-caption">{eps_df.iloc[0]['年季']}</div></div>
                      <div class="tv-card"><div class="tv-label">近四季 EPS 合計</div><div class="tv-value">{ttm_eps:.2f}</div><div class="tv-caption">近四季加總（TTM）</div></div>
                      <div class="tv-card"><div class="tv-label">季度年增率</div><div class="tv-value" style="font-size:20px;color:{'var(--green)' if pd.notna(eps_df.iloc[0]['年增率(%)']) and eps_df.iloc[0]['年增率(%)']>=0 else 'var(--red)'};">{fmt_num(eps_df.iloc[0]['年增率(%)'], '{:+.1f}%')}</div><div class="tv-caption">最新一季 YoY</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                    chart_df = eps_df.iloc[::-1]
                    fig = go.Figure(go.Bar(
                        x=chart_df['年季'], y=chart_df['EPS'],
                        marker_color=['#f23645' if v < 0 else '#35c48d' for v in chart_df['EPS']],
                        hovertemplate='%{x}<br>EPS %{y:.2f}<extra></extra>',
                    ))
                    fig.update_layout(
                        height=280, template='plotly_dark',
                        paper_bgcolor='#0b121b', plot_bgcolor='#0b121b',
                        margin=dict(l=20, r=20, t=10, b=20),
                        xaxis=dict(gridcolor='rgba(148,163,184,0.09)'),
                        yaxis=dict(title='EPS (元)', gridcolor='rgba(148,163,184,0.09)'),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(
                        eps_df, hide_index=True, use_container_width=True,
                        column_config={
                            "年季": st.column_config.TextColumn("年季", width=80),
                            "EPS": st.column_config.NumberColumn("EPS(元)", width=80, format="%.2f"),
                            "季增率(%)": st.column_config.NumberColumn("季增率", width=80, format="%.1f%%"),
                            "年增率(%)": st.column_config.NumberColumn("年增率", width=80, format="%.1f%%"),
                            "季均價": st.column_config.NumberColumn("季均價", width=80, format="%.2f"),
                        }
                    )
                else:
                    st.info("目前無法取得單季 EPS 資料，可能是新股、金融股（財報格式不同）或資料來源暫時無回應。")

            # ---------- 個股新聞 ----------
            else:
                st.markdown(f'<div class="section-head" style="margin-top:4px;"><div><div class="section-title" style="font-size:15px;">{current_stock["name"]} ({current_stock["code"]}) 最新新聞</div><div class="section-help">依標題關鍵字初步標記利多、利空或一般資訊。</div></div></div>', unsafe_allow_html=True)
                news_list = get_tw_stock_news(current_stock['code'])
                if news_list:
                    for n in news_list:
                        badge_bg = "rgba(34,171,148,0.14)" if "利多" in n["sentiment"] else "rgba(242,54,69,0.14)" if "利空" in n["sentiment"] else "rgba(41,98,255,0.12)"
                        st.markdown(f"""
                        <div class="news-card" style="border:1px solid rgba(40,80,100,0.2);border-left:2px solid {n['color']};padding:14px 16px;margin-bottom:10px;">
                          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:6px;">
                            <span style="background:{badge_bg};color:{n['color']};border:1px solid {n['color']}40;padding:2px 10px;border-radius:999px;font-size:11px;">{n['sentiment']}</span>
                            <span style="color:#8b949e;font-size:11px;">{n['publisher']}</span>
                          </div>
                          <a class="news-title" href="{n['link']}" target="_blank" style="text-decoration:none;color:#e6edf3;font-size:14px;line-height:1.65;">{n['title']}</a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("目前無法取得即時新聞，稍後重新整理即可再試。")
    else:
        st.info("目前沒有候選股票。請先到「選股掃描」頁籤執行掃描。")
