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
PE_CACHE_FILE = os.path.join(os.path.dirname(__file__), "official_pe_cache.json")
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
    show_cols      = ["code", "name", "AI評分", "收盤", "漲跌幅(%)", "乖離30MA(%)", "量比20日", "成交量(張)", "量變動(%)", "RSI14", "本益比", "營收月增", "營收年增", "industry"]
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
