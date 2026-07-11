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
import multiprocessing as mp

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
    # [新功能] 大盤濾網 / 排除機制 / 回測
    ('use_market_filter', True),   # 空頭環境自動調降評分
    ('excl_disposal',     True),   # 排除處置股 / 全額交割股
    ('excl_attention',    False),  # 排除注意股（預設僅標示不排除）
    ('market_regime_info', None),  # 掃描當下的大盤環境快照
    ('backtest_result',   None),   # 回測明細 DataFrame
]:
    if key not in st.session_state:
        st.session_state[key] = default

def get_tw_now():
    return datetime.now(timezone(timedelta(hours=8)))

# ============================================================
# 1.5 [修正] 子行程隔離：避免 yfinance/curl_cffi 的 segmentation fault
#     直接把整個 Streamlit 主行程一起打死
# ============================================================
# 背景：curl_cffi（yfinance 底層 HTTP 函式庫）在高併發 / 連續請求時，
# 偶爾會觸發作業系統層級的 segmentation fault。這種錯誤是 C 底層崩潰，
# Python 的 try/except 完全攔截不到，會直接終止整個行程（也就是整個
# Streamlit App 一起爆掉，出現「Oh no」畫面）。
# 解法：把所有會呼叫 yfinance 的動作丟到獨立子行程執行；子行程就算崩潰，
# 也只會讓那一次資料抓取失敗（回傳 None），完全不影響主行程存活。

def _isolated_call(target, args=(), timeout=60):
    """在獨立子行程執行 target(*args)，回傳 (result, crashed)。
    - result 為 None 代表逾時 / 例外 / 無資料。
    - crashed=True 代表子行程是被系統訊號（例如 SIGSEGV）強制終止，
      而不是正常執行完畢，呼叫端可依此決定要不要降級重試。
    """
    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def _runner(_q):
        try:
            _q.put(target(*args))
        except Exception:
            try:
                _q.put(None)
            except Exception:
                pass

    p = ctx.Process(target=_runner, args=(q,), daemon=True)
    try:
        p.start()
    except Exception:
        return None, True

    result = None
    try:
        result = q.get(timeout=timeout)
    except Exception:
        result = None

    p.join(3)
    if p.is_alive():
        p.terminate()
        p.join(3)

    crashed = (p.exitcode is not None and p.exitcode != 0)
    return result, crashed

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

def _yf_download_worker(ticker_str: str, threads: bool):
    """實際執行 yf.download 的子行程工作函式，必須是模組層級函式才能被 fork 正確使用。"""
    try:
        return yf.download(ticker_str, period="4mo", interval="1d",
                           group_by="ticker", auto_adjust=True, progress=False, threads=threads)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def download_batch_history(tickers: tuple) -> dict:
    if not tickers:
        return {}
    ticker_str = " ".join(tickers)

    # [修正] 先以多執行緒（快）在子行程嘗試；若子行程崩潰或無資料，
    # 改用單執行緒（穩，但較慢）重試一次，犧牲一點速度換穩定性。
    raw, crashed = _isolated_call(_yf_download_worker, (ticker_str, True), timeout=60)
    if crashed or raw is None or (hasattr(raw, "empty") and raw.empty):
        raw, _ = _isolated_call(_yf_download_worker, (ticker_str, False), timeout=90)

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

# 非個股類型的產業別，掃描時一律排除（避免 ETF / 權證 / 存託憑證混入結果）。
NON_STOCK_INDUSTRIES = {'ETF', '存託憑證', '受益證券', '認購售權證', '管理股票'}


def calc_ma_signals(history_map, stock_map, bias_limit, vol_limit,
                    excluded_codes=None, attention_codes=None):
    """[新功能4] excluded_codes: 處置/全額交割等應排除的代碼集合。
    attention_codes: 注意股代碼集合，不排除但會標記警示。"""
    excluded_codes  = excluded_codes or set()
    attention_codes = attention_codes or set()
    hits = []
    for s in stock_map:
        tk = s["ticker"]
        # [新功能4] 排除機制：非個股類型 / 處置股 / 全額交割股
        if s.get('industry') in NON_STOCK_INDUSTRIES:
            continue
        if str(s.get('code', '')).startswith('00'):   # 00 開頭多為 ETF / ETN
            continue
        if s.get('code') in excluded_codes:
            continue
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
                # [新功能3] 保留均線數值，供出場守則計算防守價
                "MA30":       round(float(ma30), 2),
                "MA45":       round(float(ma45), 2),
                "MA60":       round(float(ma60), 2),
                # [新功能4] 注意股僅標示，不直接排除（可由設定改為排除）
                "注意股":     s.get('code') in attention_codes,
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

def _yf_ticker_history_worker(ticker: str):
    try:
        return yf.Ticker(ticker).history(period="1y")
    except Exception:
        return None


def draw_k_line(ticker, name, chart_mode='K線圖', chart_period='日'):
    """畫出有實際切換功能的金融圖表。
    chart_mode: K線圖 / 走勢圖 / 技術指標
    chart_period: 日 / 週 / 月
    """
    code   = ticker.split(".")[0]
    market = "TW" if ticker.endswith(".TW") else "TWO"
    df     = get_kline_data(code, market)
    if df.empty or len(df) < 30:
        # [修正] yf.Ticker(...).history() 一樣走子行程隔離，避免 curl_cffi 崩潰拖垮主行程
        raw, _crashed = _isolated_call(_yf_ticker_history_worker, (ticker,), timeout=25)
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
# 2.5 [新功能] 大盤濾網 / 排除清單 / 出場守則 / 策略回測
# ============================================================

def _yf_twii_worker():
    """子行程工作函式：抓取加權指數並直接算好摘要數值再回傳（回傳值需可被 pickle，
    plain dict 比整包 DataFrame 更輕量、更安全）。"""
    try:
        raw = yf.download("^TWII", period="9mo", interval="1d",
                          auto_adjust=True, progress=False, threads=False)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        closes = raw['Close'].dropna()
        if len(closes) < 60:
            return None
        close = float(closes.iloc[-1])
        prev  = float(closes.iloc[-2])
        ma20  = float(closes.rolling(20).mean().iloc[-1])
        ma60  = float(closes.rolling(60).mean().iloc[-1])
        return {'close': close, 'prev': prev, 'ma20': ma20, 'ma60': ma60}
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_market_regime() -> dict:
    """[新功能1] 大盤濾網：抓加權指數 (^TWII) 判斷多空環境。

    判斷規則（簡單且穩健）：
    - 多頭：收盤 > 60MA 且 20MA > 60MA
    - 空頭：收盤 < 60MA 且 20MA < 60MA
    - 其餘：震盪
    均線多頭排列策略在空頭市場勝率會明顯下降，掃描時據此提示與調降評分。

    [修正] 實際下載動作丟到獨立子行程執行，避免 yfinance/curl_cffi 偶發的
    segmentation fault 拖垮整個 Streamlit 主行程；子行程崩潰時直接視為
    「大盤資料暫時無法取得」，不影響掃描主流程。
    """
    info = {
        'regime': '未知', 'close': np.nan, 'chg_pct': np.nan,
        'ma20': np.nan, 'ma60': np.nan, 'position': 'N/A',
        'suggestion': '大盤資料暫時無法取得；策略照常執行，但請自行留意大盤風險。',
    }
    data, _crashed = _isolated_call(_yf_twii_worker, (), timeout=30)
    if not data:
        return info

    close, prev, ma20, ma60 = data['close'], data['prev'], data['ma20'], data['ma60']
    chg = (close / prev - 1) * 100 if prev > 0 else np.nan

    if close > ma60 and ma20 > ma60:
        regime, position = '多頭', '正常部位'
        suggestion = '大盤趨勢偏多，均線多頭訊號可信度較高，可依計畫正常配置部位。'
    elif close < ma60 and ma20 < ma60:
        regime, position = '空頭', '低水位 / 觀望'
        suggestion = '大盤處於空頭結構，多頭訊號勝率通常明顯下降；建議僅小量試單或觀望，嚴設停損。'
    else:
        regime, position = '震盪', '約五成部位'
        suggestion = '大盤位於震盪區間，訊號雜訊較多；建議降低部位並提高停損紀律。'

    info.update({'regime': regime, 'close': round(close, 2), 'chg_pct': round(chg, 2),
                 'ma20': round(ma20, 2), 'ma60': round(ma60, 2),
                 'position': position, 'suggestion': suggestion})
    return info


@st.cache_data(ttl=43200, show_spinner=False)
def get_excluded_stock_sets() -> dict:
    """[新功能4] 抓取官方警示名單：處置股、變更交易（全額交割）、注意股。

    來源為 TWSE / TPEx OpenAPI；任何來源抓取失敗都回傳空集合，
    不會影響掃描主流程（fail-open 設計）。
    """
    disposal, attention, altered = set(), set(), set()

    def collect(url, target: set):
        data = _fetch_json(url, timeout=6)
        rows = []
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            for key in ['data', 'Data', 'aaData', 'tables']:
                if isinstance(data.get(key), list):
                    rows = data[key]
                    break
        for rec in rows:
            if isinstance(rec, dict):
                code = _find_record_code(rec)
                if code:
                    target.add(code)

    # 處置有價證券（上市 / 上櫃）
    collect("https://openapi.twse.com.tw/v1/announcement/punish", disposal)
    collect("https://www.tpex.org.tw/openapi/v1/tpex_disposal_information", disposal)
    # 注意有價證券（上市 / 上櫃）
    collect("https://openapi.twse.com.tw/v1/announcement/notice", attention)
    collect("https://www.tpex.org.tw/openapi/v1/tpex_attention_information", attention)
    # 變更交易方法（全額交割）；API 名稱若調整，此處 fail-open 不影響掃描
    collect("https://openapi.twse.com.tw/v1/opendata/t187ap04_L", altered)

    return {'disposal': disposal, 'attention': attention, 'altered': altered}


def build_exit_plan(stock: pd.Series) -> dict:
    """[新功能3] 出場守則：回答「買了之後怎麼辦」。

    - 趨勢防守價：MA30，收盤跌破視為趨勢轉弱出場訊號。
    - 成本防守價：近 20 日量價加權主力成本。
    - 建議停損價：兩者取較高（較靠近現價）者，控制單筆虧損。
    - 另掃描 RSI 過熱 / MACD 柱轉負 / 跌破成本等即時警訊。
    """
    price     = stock.get('收盤', np.nan)
    ma30      = stock.get('MA30', np.nan)
    main_cost = stock.get('主力成本', np.nan)
    rsi       = stock.get('RSI14', np.nan)
    macd_hist = stock.get('MACD柱', np.nan)
    cost_gap  = stock.get('主力成本乖離(%)', np.nan)

    candidates = [v for v in [ma30, main_cost] if pd.notna(v) and v > 0]
    stop_price = max(candidates) if candidates else np.nan
    stop_gap   = ((stop_price / price) - 1) * 100 if pd.notna(stop_price) and pd.notna(price) and price > 0 else np.nan

    alerts = []
    if pd.notna(price) and pd.notna(ma30) and price < ma30:
        alerts.append('收盤已跌破 MA30，趨勢防守失守')
    if pd.notna(cost_gap) and cost_gap < 0:
        alerts.append('現價低於主力成本，籌碼優勢轉弱')
    if pd.notna(rsi) and rsi > 78:
        alerts.append(f'RSI {rsi:.0f} 過熱，避免追高並留意回檔')
    if pd.notna(macd_hist) and macd_hist < 0:
        alerts.append('MACD 柱轉負，短線動能轉弱')

    return {
        'ma30': ma30,
        'main_cost': main_cost,
        'stop_price': stop_price,
        'stop_gap': stop_gap,
        'alerts': alerts,
        'trail_rule': '移動停利：獲利超過 10% 後，改以 MA30 作為移動停損；收盤跌破即出場保住獲利。',
    }


def _yf_backtest_download_worker(ticker_str: str, threads: bool):
    try:
        return yf.download(ticker_str, period="1y", interval="1d",
                           group_by="ticker", auto_adjust=True,
                           progress=False, threads=threads)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def run_strategy_backtest(tickers: tuple, bias_limit: float, vol_limit: int) -> pd.DataFrame:
    """[新功能2] 策略回測：用與掃描完全相同的條件，檢驗近一年歷史訊號表現。

    條件：MA30 > MA45 > MA60、0 <= 30MA乖離 <= bias_limit、近5日均量 >= vol_limit。
    同一檔股票 5 個交易日內只取一次訊號（cooldown），避免連續訊號灌水。
    回傳每筆訊號的 5 / 10 / 20 日持有報酬明細。

    [修正] 下載動作在獨立子行程執行，避免 yfinance/curl_cffi 偶發崩潰拖垮主行程。
    """
    if not tickers:
        return pd.DataFrame()

    ticker_str = " ".join(tickers)
    raw, crashed = _isolated_call(_yf_backtest_download_worker, (ticker_str, True), timeout=90)
    if crashed or raw is None or (hasattr(raw, "empty") and raw.empty):
        raw, _ = _isolated_call(_yf_backtest_download_worker, (ticker_str, False), timeout=120)
    if raw is None or raw.empty:
        return pd.DataFrame()

    records = []
    for tk in tickers:
        try:
            if len(tickers) == 1:
                sub = raw[["Close", "Volume"]].dropna()
            else:
                sub = raw[tk][["Close", "Volume"]].dropna()
        except Exception:
            continue
        if len(sub) < 70:
            continue

        closes  = sub["Close"].astype(float)
        volumes = (sub["Volume"].astype(float) / 1000)  # 張
        ma30 = closes.rolling(30).mean()
        ma45 = closes.rolling(45).mean()
        ma60 = closes.rolling(60).mean()
        dates = sub.index

        last_signal_i = -99
        n = len(closes)
        for i in range(64, n - 1):
            if i - last_signal_i < 5:          # cooldown
                continue
            m30, m45, m60 = ma30.iloc[i], ma45.iloc[i], ma60.iloc[i]
            if pd.isna(m30) or pd.isna(m45) or pd.isna(m60):
                continue
            if not (m30 > m45 > m60):
                continue
            bias = (closes.iloc[i] - m30) / m30 * 100
            if not (0 <= bias <= bias_limit):
                continue
            if volumes.iloc[max(0, i - 4):i + 1].mean() < vol_limit:
                continue

            entry = float(closes.iloc[i])
            rec = {
                'ticker': tk,
                '訊號日': pd.Timestamp(dates[i]).strftime('%Y-%m-%d'),
                '進場價': round(entry, 2),
                '乖離30MA(%)': round(float(bias), 2),
            }
            for horizon in (5, 10, 20):
                if i + horizon < n:
                    rec[f'{horizon}日報酬(%)'] = round((float(closes.iloc[i + horizon]) / entry - 1) * 100, 2)
                else:
                    rec[f'{horizon}日報酬(%)'] = np.nan
            records.append(rec)
            last_signal_i = i

    return pd.DataFrame(records)


def render_backtest_section(result_df: pd.DataFrame, bias_limit: float, vol_limit: int):
    """[新功能2] 回測面板：顯示勝率 / 平均報酬 / 報酬分布。"""
    st.markdown('<div class="tv-section">STRATEGY BACKTEST · 策略歷史回測</div>', unsafe_allow_html=True)
    with st.expander("▸ 對目前掃描結果回測：近一年同條件訊號的 5 / 10 / 20 日表現", expanded=False):
        st.caption(
            "回測條件與掃描完全一致（MA30>MA45>MA60、乖離 0~上限、5日均量門檻），"
            "同一檔 5 個交易日內僅取一次訊號。結果僅用於驗證策略邏輯，過去績效不代表未來表現，亦非投資建議。"
        )
        if st.button("▶ 執行回測", key="btn_backtest", use_container_width=True):
            with st.spinner("回測中（下載一年歷史資料，約 10~60 秒）..."):
                bt = run_strategy_backtest(
                    tuple(result_df['ticker'].tolist()), float(bias_limit), int(vol_limit)
                )
            st.session_state.backtest_result = bt

        bt = st.session_state.backtest_result
        if bt is None:
            return
        if bt.empty:
            st.info("回測期間內沒有產生符合條件的歷史訊號。")
            return

        # 各持有期間統計摘要
        summary_rows = []
        for h in (5, 10, 20):
            col = f'{h}日報酬(%)'
            s = bt[col].dropna() if col in bt.columns else pd.Series(dtype=float)
            if s.empty:
                continue
            summary_rows.append({
                '持有期間': f'{h} 日',
                '訊號數': int(len(s)),
                '勝率(%)': round(float((s > 0).mean() * 100), 1),
                '平均報酬(%)': round(float(s.mean()), 2),
                '中位數(%)': round(float(s.median()), 2),
                '最差(%)': round(float(s.min()), 2),
                '最佳(%)': round(float(s.max()), 2),
            })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

        # 20 日報酬分布直方圖（若樣本不足退回 10 日 / 5 日）
        hist_col = next((f'{h}日報酬(%)' for h in (20, 10, 5)
                         if f'{h}日報酬(%)' in bt.columns and bt[f'{h}日報酬(%)'].notna().sum() >= 5), None)
        if hist_col:
            s = bt[hist_col].dropna()
            fig = go.Figure(go.Histogram(
                x=s, nbinsx=30, marker_color='#3b82f6', opacity=0.85,
                hovertemplate='報酬區間 %{x}<br>筆數 %{y}<extra></extra>',
            ))
            fig.add_vline(x=0, line_dash='dash', line_color='#f23645', line_width=1.2)
            fig.update_layout(
                title=dict(text=f'{hist_col} 分布（共 {len(s)} 筆訊號）', font=dict(size=13, color='#cbd5e1')),
                height=300, template='plotly_dark',
                paper_bgcolor='#0b121b', plot_bgcolor='#0b121b',
                margin=dict(l=20, r=20, t=40, b=20), bargap=0.05,
                xaxis=dict(title='報酬(%)', gridcolor='rgba(148,163,184,0.09)'),
                yaxis=dict(title='筆數', gridcolor='rgba(148,163,184,0.09)'),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            '⬇ 下載回測明細 CSV',
            bt.to_csv(index=False).encode('utf-8-sig'),
            file_name=f'backtest_detail_{get_tw_now().strftime("%Y%m%d")}.csv',
            mime='text/csv', use_container_width=True,
        )


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

    # [新功能1+4] 風控選項：大盤濾網 / 排除機制
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.session_state.use_market_filter = st.checkbox(
            "啟用大盤濾網", value=st.session_state.use_market_filter, key="cb_market",
            help="空頭環境自動調降個股評分（-12分）、震盪環境小幅調降（-5分），並提示建議部位水位。"
        )
    with fc2:
        st.session_state.excl_disposal = st.checkbox(
            "排除處置 / 全額交割股", value=st.session_state.excl_disposal, key="cb_disposal",
            help="依 TWSE / TPEx 官方公告名單，直接自掃描結果排除。"
        )
    with fc3:
        st.session_state.excl_attention = st.checkbox(
            "排除注意股", value=st.session_state.excl_attention, key="cb_attention",
            help="勾選則直接排除；未勾選時注意股仍會出現，但會加上 ⚠ 警示標記。"
        )

# 統一讀取最終值
user_bias = st.session_state.user_bias
user_vol  = st.session_state.user_vol
use_market_filter = st.session_state.use_market_filter
excl_disposal     = st.session_state.excl_disposal
excl_attention    = st.session_state.excl_attention

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

    # [新功能1] 大盤濾網：先判斷市場環境，作為後續評分調整與提示依據
    status.text("🌐 Step 1/4：判斷大盤環境（加權指數）...")
    bar.progress(0.01)
    regime_info = get_market_regime()
    st.session_state.market_regime_info = regime_info

    # [新功能4] 抓官方警示名單（處置 / 全額交割 / 注意股）
    excl_sets = get_excluded_stock_sets()
    excluded_codes = set()
    if excl_disposal:
        excluded_codes |= excl_sets['disposal'] | excl_sets['altered']
    if excl_attention:
        excluded_codes |= excl_sets['attention']
    attention_codes = excl_sets['attention']

    status.text("📋 Step 2/4：載入股票清單...")
    bar.progress(0.03)
    stock_map    = get_stock_market_list()
    all_tickers  = [s["ticker"] for s in stock_map]
    total_tickers = len(all_tickers)

    history_map = {}
    batches = [all_tickers[i:i+BATCH] for i in range(0, total_tickers, BATCH)]
    for bi, batch in enumerate(batches):
        status.text(f"📥 Step 3/4：批次下載 {bi+1}/{len(batches)}...")
        bar.progress(0.03 + 0.72 * (bi / len(batches)))
        history_map.update(download_batch_history(tuple(batch)))

    bar.progress(0.75)
    status.text("✅ Step 4/4：計算均線與排除警示股中...")
    initial_hits = calc_ma_signals(
        history_map, stock_map, user_bias, user_vol,
        excluded_codes=excluded_codes, attention_codes=attention_codes,
    )
    bar.progress(0.80)

    if initial_hits:
        status.text(f"📈 找到 {len(initial_hits)} 支！載入官方本益比快取與營收資料中...")
        load_official_pe_map(False)
        final_list = []
        with ThreadPoolExecutor(max_workers=6) as ex:
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
                    "市場別":     base.get("市場別", "上市" if str(base.get("ticker", "")).endswith(".TW") else "上櫃" if str(base.get("ticker", "")).endswith(".TWO") else "興櫃"),
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
                    # [新功能3] 出場守則使用的均線數值
                    "MA30":       base.get("MA30", np.nan),
                    "MA45":       base.get("MA45", np.nan),
                    "MA60":       base.get("MA60", np.nan),
                    # [新功能4] 注意股警示標記
                    "警示":       "⚠ 注意股" if base.get("注意股", False) else "",
                }
                score, radar = calc_stock_score(row_data)
                # [新功能1] 大盤濾網：空頭 / 震盪環境調降評分，反映系統性風險
                if use_market_filter:
                    regime = (regime_info or {}).get('regime', '未知')
                    if regime == '空頭':
                        score = max(0, score - 12)
                        radar.append('大盤空頭壓抑')
                    elif regime == '震盪':
                        score = max(0, score - 5)
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

    # ── [新功能1] 大盤環境卡片與空頭警示 ──
    regime_info = st.session_state.market_regime_info or {}
    regime = regime_info.get('regime', '未知')
    regime_color = {'多頭': 'var(--tv-green)', '空頭': 'var(--tv-red)',
                    '震盪': 'var(--tv-yellow)'}.get(regime, '#8f9bad')
    twii_txt = fmt_num(regime_info.get('close', np.nan), '{:,.0f}')
    twii_chg = regime_info.get('chg_pct', np.nan)
    twii_chg_txt = 'N/A' if pd.isna(twii_chg) else f"{twii_chg:+.2f}%"

    if regime == '空頭' and use_market_filter:
        st.warning(f"⚠️ 大盤濾網警示：加權指數目前為空頭結構（收盤 {twii_txt}，20MA 低於 60MA）。"
                   f"多頭訊號勝率通常明顯下降，所有評分已自動調降 12 分。{regime_info.get('suggestion','')}")

    # ── 精簡統計列（加入大盤環境）──
    st.markdown(f"""
    <div class="stat-grid" style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px;">
      <div class="tv-card">
        <div class="tv-label">Total Signals</div>
        <div class="tv-value" style="color:#8fb2ff;">{total_found}</div>
        <div class="tv-caption">符合條件標的</div>
      </div>
      <div class="tv-card">
        <div class="tv-label">Market Regime · 大盤環境</div>
        <div class="tv-value" style="font-size:20px;color:{regime_color};">{regime} <span style="font-size:13px;color:#8f9bad;">TAIEX {twii_txt} ({twii_chg_txt})</span></div>
        <div class="tv-caption">建議部位：{regime_info.get('position', 'N/A')}</div>
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
    show_cols      = ["code", "name", "市場別", "警示", "AI評分", "收盤", "漲跌幅(%)", "乖離30MA(%)", "量比20日", "成交量(張)", "量變動(%)", "本益比", "營收月增", "營收年增", "industry"]
    available_cols = [c for c in show_cols if c in df.columns]
    df_display     = df[available_cols].rename(columns={"code": "代碼", "name": "名稱", "industry": "類股", "市場別": "市場"})

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
            "市場":        st.column_config.TextColumn("市場",    width=70),
            "警示":        st.column_config.TextColumn("警示",    width=85, help="官方注意股名單標記"),
            "AI評分":      st.column_config.ProgressColumn("AI評分", width=105, format="%d", min_value=0, max_value=100),
            "收盤":        st.column_config.NumberColumn("價格",  width=75,  format="%.2f"),
            "漲跌幅(%)":   st.column_config.NumberColumn("漲跌", width=75, format="%.1f%%"),
            "乖離30MA(%)": st.column_config.ProgressColumn("30MA乖離", width=120,
                                                            help=f"上限 {user_bias}%",
                                                            format="%.2f%%", min_value=0, max_value=user_bias),
            "量比20日":    st.column_config.NumberColumn("量比", width=70, format="%.2fx"),
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

    # ── [新功能2] 策略回測面板 ──
    render_backtest_section(df, user_bias, user_vol)

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
            # 使用自訂 HTML 渲染，讓滑鼠水平準線右側可顯示即時對應股價。
            render_kline_chart_with_axis_price(k_fig, height=640)

            # 預載前後兩檔，點擊上一支 / 下一支時多數情況可直接從快取取圖，不再顯示 Running get_kline_data。
            try:
                preload_tickers = []
                for offset in (-1, 1):
                    preload_idx = (st.session_state.current_idx + offset) % total_found
                    preload_tickers.append(df.iloc[preload_idx]['ticker'])
                warm_kline_data_async(preload_tickers)
            except Exception:
                pass
        else:
            st.warning("⚠️ 無法載入 K 線資料，請稍後再試。")

    with right_area:
        st.markdown(f"""
        <div class="side-card">
          <div class="side-title"><span>AI 分析報告</span><span class="bias-chip">{report['盤面強弱']}</span></div>
          <div class="report-row"><span>強度評分</span><span style="color:var(--tv-green);">{score_txt}</span></div>
          <div class="report-row"><span>RSI</span><span>{rsi_txt}</span></div>
          <div class="report-row"><span>MACD柱</span><span>{macd_txt}</span></div>
          <div style="color:#b8c4d3;font-size:13px;line-height:1.75;margin-top:12px;">{report['趨勢結構']}<br>{report['動能訊號']}</div>
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

        # ── [新功能3] 出場守則卡片 ──
        exit_plan = build_exit_plan(current_stock)
        stop_txt = fmt_num(exit_plan['stop_price'], '{:.2f}')
        stop_gap_txt = '' if pd.isna(exit_plan['stop_gap']) else f"（距現價 {exit_plan['stop_gap']:+.1f}%）"
        if exit_plan['alerts']:
            alert_html = ''.join([
                f'<div style="color:#f8a5a5;font-size:13px;line-height:1.8;"><b style="color:var(--tv-red);margin-right:8px;">✕</b>{a}</div>'
                for a in exit_plan['alerts']
            ])
        else:
            alert_html = '<div style="color:#8ef0aa;font-size:13px;line-height:1.8;"><b style="color:var(--tv-green);margin-right:8px;">✓</b>目前未觸發任何出場警訊</div>'
        st.markdown(f"""
        <div class="side-card">
          <div class="side-title"><span>出場守則</span><span class="bias-chip" style="background:rgba(239,68,68,.14);color:#f8a5a5;border-color:rgba(239,68,68,.3);">RISK</span></div>
          <div class="report-row"><span>趨勢防守價 (MA30)</span><span>{fmt_num(exit_plan['ma30'], '{:.2f}')}</span></div>
          <div class="report-row"><span>成本防守價</span><span>{fmt_num(exit_plan['main_cost'], '{:.2f}')}</span></div>
          <div class="report-row"><span>建議停損價</span><span style="color:var(--tv-red);">{stop_txt}</span></div>
          <div class="tv-caption" style="margin-top:4px;">取 MA30 與主力成本較高者 {stop_gap_txt}，收盤跌破建議出場。</div>
          <div style="margin-top:10px;">{alert_html}</div>
          <div class="tv-caption" style="margin-top:10px;border-top:1px solid rgba(45,62,82,.45);padding-top:8px;">{exit_plan['trail_rule']}</div>
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
