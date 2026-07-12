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
    # [新功能] 多策略切換 / 策略回測比較
    ('scan_strategy_used', '均線多頭排列'),
    ('backtest_results',   {}),   # {策略名稱: DataFrame} 讓不同策略的回測結果可以並排比較
    # [新功能] 自選股／追蹤清單
    ('watchlist_quotes',       pd.DataFrame()),
    ('watchlist_chart_target', None),  # (ticker, name)，選了哪一檔要在下方顯示 K 線
    ('industry_filter',        None),  # [新功能] 熱門族群卡片點擊後套用的產業別篩選
]:
    if key not in st.session_state:
        st.session_state[key] = default

def get_tw_now():
    return datetime.now(timezone(timedelta(hours=8)))

# ============================================================
# [新功能] 自選股／追蹤清單：JSON 檔案持久化
# ============================================================
# 沿用跟 get_stock_market_list() 相同的檔案路徑寫法（同一個部署環境下
# 存取邏輯保持一致）。存的是「輕量資訊」（代碼/名稱/加入時的價格與日期），
# 不存整包歷史資料，清單本身開啟速度才會快。
#
# 限制說明：這是「檔案系統持久化」，同一次部署運行期間會一直保留；
# 但 Streamlit Cloud 重新部署（例如推新 commit）時容器會重建，檔案不保證留著。
# 對單人使用的個人選股工具來說，這已經是不需要額外資料庫、最簡單可行的做法。

WATCHLIST_FILE = os.path.join(
    os.path.dirname(__file__) if '__file__' in globals() else '.',
    'watchlist_v1.json'
)


def load_watchlist() -> list:
    """讀取追蹤清單，檔案不存在或損毀時安全回傳空清單。"""
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception:
        pass
    return []


def save_watchlist(watchlist: list) -> bool:
    """寫回追蹤清單，失敗（例如唯讀檔案系統）時安靜略過，不影響其他功能。"""
    try:
        with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(watchlist, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def is_in_watchlist(code: str, watchlist: list) -> bool:
    return any(str(item.get('code')) == str(code) for item in watchlist)


def add_to_watchlist(stock: dict) -> list:
    """加入追蹤：記錄加入當下的日期／價格／AI評分，供之後計算累計報酬。"""
    watchlist = load_watchlist()
    code = str(stock.get('code'))
    if is_in_watchlist(code, watchlist):
        return watchlist
    watchlist.append({
        'code': code,
        'name': stock.get('name', ''),
        'ticker': stock.get('ticker', ''),
        'industry': stock.get('industry', '未分類'),
        '市場別': stock.get('市場別', ''),
        'added_date': get_tw_now().strftime('%Y-%m-%d'),
        'added_price': stock.get('收盤', None),
        'added_score': stock.get('AI評分', None),
        'note': '',
    })
    save_watchlist(watchlist)
    return watchlist


def remove_from_watchlist(code: str) -> list:
    watchlist = load_watchlist()
    watchlist = [item for item in watchlist if str(item.get('code')) != str(code)]
    save_watchlist(watchlist)
    return watchlist


def refresh_watchlist_quotes(watchlist: list) -> pd.DataFrame:
    """批次抓取追蹤清單目前報價，計算「加入以來累計報酬」與簡易健康度標示
    （跌破 MA30 / RSI 過熱），抓不到資料的股票仍會列出、只是報價欄顯示 N/A。
    """
    if not watchlist:
        return pd.DataFrame()

    tickers = tuple(item['ticker'] for item in watchlist if item.get('ticker'))
    history_map = download_batch_history(tickers) if tickers else {}
    hot_codes = get_hot_stock_tickers()  # [新功能] 熱門股判定，跟掃描流程共用同一份排行

    rows = []
    for item in watchlist:
        tk = item.get('ticker', '')
        df = history_map.get(tk)
        curr_price, chg_since_added, rsi_now, below_ma30, data_ok = np.nan, np.nan, np.nan, False, False
        if df is not None and not df.empty:
            closes = df['close']
            curr_price = float(closes.iloc[-1])
            data_ok = True
            added_price = item.get('added_price')
            if added_price and added_price > 0:
                chg_since_added = (curr_price / added_price - 1) * 100
            if len(closes) >= 30:
                ma30 = closes.rolling(30).mean().iloc[-1]
                below_ma30 = pd.notna(ma30) and curr_price < ma30
            rsi_now = calc_rsi(closes)
        rows.append({
            'code': item['code'], 'name': item['name'], 'ticker': tk,
            'industry': item.get('industry', '未分類'),
            'added_date': item.get('added_date', 'N/A'),
            'added_price': item.get('added_price'),
            '現價': round(curr_price, 2) if pd.notna(curr_price) else np.nan,
            '累計報酬(%)': round(chg_since_added, 2) if pd.notna(chg_since_added) else np.nan,
            'RSI14': round(rsi_now, 1) if pd.notna(rsi_now) else np.nan,
            '跌破MA30': below_ma30,
            '熱門股': item['code'] in hot_codes,
            '資料狀態': '正常' if data_ok else '暫無資料',
        })
    return pd.DataFrame(rows)


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

# ============================================================
# [新功能] 多策略切換：均線多頭排列 / 量價背離 / 型態突破
# ============================================================
# 設計原則：每種策略都拆成「單日條件判斷」函式（輸入 closes/volumes 與索引 i），
# 這組判斷函式同時被「即時掃描」（判斷最後一天）與「策略回測」（判斷歷史每一天）
# 共用——確保回測驗證的邏輯跟實際掃描時用的邏輯完全一致，回測結果才有意義。

def _check_ma_condition(closes: pd.Series, volumes: pd.Series, i: int, bias_limit: float):
    """策略1：均線多頭排列。MA30>MA45>MA60，且股價貼在 30 日均線之上、乖離不超過上限。"""
    if i < 59:
        return False, np.nan
    ma30 = closes.iloc[i-29:i+1].mean()
    ma45 = closes.iloc[i-44:i+1].mean()
    ma60 = closes.iloc[i-59:i+1].mean()
    price = closes.iloc[i]
    if pd.isna(ma30) or ma30 <= 0:
        return False, np.nan
    bias = (price - ma30) / ma30 * 100
    ok = (ma30 > ma45 > ma60) and (0 <= bias <= bias_limit)
    return ok, bias


def _check_divergence_condition(closes: pd.Series, volumes: pd.Series, i: int, vol_growth_threshold: float):
    """策略2：量價背離（止跌訊號）。近10日股價持平或小跌，但成交量較前10日
    明顯放大，且 RSI 從相對低檔回升，視為主力進場布局、量增價穩的背離訊號。"""
    if i < 24:
        return False, np.nan
    prev_price = closes.iloc[i-10]
    if pd.isna(prev_price) or prev_price <= 0:
        return False, np.nan
    price_chg_10d = (closes.iloc[i] / prev_price - 1) * 100
    recent_vol_avg = volumes.iloc[i-9:i+1].mean()
    prior_vol_avg = volumes.iloc[i-19:i-9].mean()
    if pd.isna(prior_vol_avg) or prior_vol_avg <= 0:
        return False, np.nan
    vol_chg = (recent_vol_avg / prior_vol_avg - 1) * 100
    rsi_now = calc_rsi(closes.iloc[:i+1])
    rsi_prev = calc_rsi(closes.iloc[:i-4]) if i >= 19 else np.nan
    if pd.isna(rsi_now) or pd.isna(rsi_prev):
        return False, np.nan
    ok = (-6 <= price_chg_10d <= 3) and (vol_chg >= vol_growth_threshold) and (rsi_now > rsi_prev) and (rsi_now < 70)
    return ok, vol_chg


def _check_breakout_condition(closes: pd.Series, volumes: pd.Series, i: int, vol_mult_threshold: float):
    """策略3：型態突破。經過一段整理（近20日波動幅度收斂 ≤12%）後，股價突破
    近20日高點，且成交量明顯放大確認（避免量能不足的假突破）。"""
    if i < 20:
        return False, np.nan
    price = closes.iloc[i]
    prior20 = closes.iloc[i-20:i]
    high20 = prior20.max()
    low20 = prior20.min()
    mean20 = prior20.mean()
    if pd.isna(mean20) or mean20 <= 0:
        return False, np.nan
    consolidation = (high20 - low20) / mean20 * 100
    avg_vol20 = volumes.iloc[i-20:i].mean()
    if pd.isna(avg_vol20) or avg_vol20 <= 0:
        return False, np.nan
    vol_ratio = volumes.iloc[i] / avg_vol20
    ok = (price > high20) and (vol_ratio >= vol_mult_threshold) and (consolidation <= 12)
    return ok, vol_ratio


def _build_common_signal_fields(s: dict, df: pd.DataFrame) -> dict:
    """三種策略共用的延伸欄位（漲跌、量比、主力成本、RSI、MACD…），
    確保切換策略不影響既有的 AI 評分／K線／AI分析／公司資訊等下游功能。"""
    closes = df["close"]
    volumes = df["volume"]
    curr_price = float(closes.iloc[-1])
    vol_today = int(volumes.iloc[-1])
    vol_yesterday = float(volumes.iloc[-2]) if len(volumes) >= 2 else np.nan
    avg_vol20 = float(volumes.tail(20).mean())
    ma30 = closes.rolling(30).mean().iloc[-1]
    main_cost = calc_main_cost(df, 20)
    cost_gap = ((curr_price - main_cost) / main_cost * 100) if pd.notna(main_cost) and main_cost > 0 else np.nan
    high20 = float(closes.tail(20).max())
    high60 = float(closes.tail(60).max())
    prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else np.nan
    price_change = ((curr_price - prev_close) / prev_close * 100) if pd.notna(prev_close) and prev_close > 0 else np.nan
    vol_change = ((vol_today - vol_yesterday) / vol_yesterday * 100) if pd.notna(vol_yesterday) and vol_yesterday > 0 else 0
    bias_30 = ((curr_price - ma30) / ma30 * 100) if pd.notna(ma30) and ma30 > 0 else np.nan
    return {**s,
        "收盤":       round(curr_price, 2),
        "漲跌幅(%)":   round(price_change, 2) if pd.notna(price_change) else np.nan,
        "乖離30MA(%)": round(bias_30, 2) if pd.notna(bias_30) else np.nan,
        "成交量(張)":  vol_today,
        "量變動(%)":   round(vol_change, 2),
        "量比20日":    round(vol_today / avg_vol20, 2) if avg_vol20 > 0 else np.nan,
        "主力成本":    round(main_cost, 2) if pd.notna(main_cost) else np.nan,
        "主力成本乖離(%)": round(cost_gap, 2) if pd.notna(cost_gap) else np.nan,
        "RSI14":      round(calc_rsi(closes), 1),
        "MACD柱":     round(calc_macd_hist(closes), 3),
        "突破20日高":  curr_price >= high20,
        "接近60日高":  curr_price >= high60 * 0.97,
    }


def calc_ma_signals(history_map, stock_map, bias_limit, vol_limit):
    """策略1：均線多頭排列。"""
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
        i = len(closes) - 1
        ok, bias = _check_ma_condition(closes, volumes, i, bias_limit)
        if ok:
            row = _build_common_signal_fields(s, df)
            row["策略"] = "均線多頭排列"
            row["訊號說明"] = f"MA30>MA45>MA60 多頭排列，乖離 {bias:.1f}%"
            hits.append(row)
    return hits


def calc_divergence_signals(history_map, stock_map, vol_growth_threshold, vol_limit):
    """策略2：量價背離（止跌訊號）。"""
    hits = []
    for s in stock_map:
        tk = s["ticker"]
        df = history_map.get(tk)
        if df is None or len(df) < 45:
            continue
        closes = df["close"]
        volumes = df["volume"]
        if volumes.tail(5).mean() < vol_limit:
            continue
        i = len(closes) - 1
        ok, vol_chg = _check_divergence_condition(closes, volumes, i, vol_growth_threshold)
        if ok:
            row = _build_common_signal_fields(s, df)
            price_chg_10d = (closes.iloc[i] / closes.iloc[i-10] - 1) * 100
            row["策略"] = "量價背離"
            row["訊號說明"] = f"近10日股價{price_chg_10d:+.1f}%、量能放大{vol_chg:+.0f}%，RSI回升"
            hits.append(row)
    return hits


def calc_breakout_signals(history_map, stock_map, vol_mult_threshold, vol_limit):
    """策略3：型態突破。"""
    hits = []
    for s in stock_map:
        tk = s["ticker"]
        df = history_map.get(tk)
        if df is None or len(df) < 45:
            continue
        closes = df["close"]
        volumes = df["volume"]
        if volumes.tail(5).mean() < vol_limit:
            continue
        i = len(closes) - 1
        ok, vol_ratio = _check_breakout_condition(closes, volumes, i, vol_mult_threshold)
        if ok:
            row = _build_common_signal_fields(s, df)
            high20_excl = float(closes.iloc[i-20:i].max())
            breakout_pct = (closes.iloc[i] / high20_excl - 1) * 100
            row["策略"] = "型態突破"
            row["訊號說明"] = f"突破20日高{breakout_pct:+.1f}%，爆量{vol_ratio:.1f}倍"
            hits.append(row)
    return hits


# 策略登記表：新增策略只要在這裡註冊，Tab1 的選單與掃描流程會自動支援。
STRATEGY_REGISTRY = {
    "均線多頭排列": {
        "func": calc_ma_signals,
        "desc": "尋找 MA30 &gt; MA45 &gt; MA60 的多頭排列股票，適合抓穩定趨勢股。",
        "param_label": "30MA 乖離上限 (%)", "param_help": "數值越小，越偏向尋找貼近 30 日均線的股票。",
        "param_min": 0.1, "param_max": 15.0, "param_default": 3.0, "param_step": 0.1,
    },
    "量價背離": {
        "func": calc_divergence_signals,
        "desc": "尋找股價持平或小跌、但成交量明顯放大、RSI 從低檔回升的止跌訊號，適合抓底部布局機會。",
        "param_label": "量增門檻 (%)", "param_help": "近10日均量相對前10日均量的放大幅度，數值越高訊號越嚴格。",
        "param_min": 5.0, "param_max": 100.0, "param_default": 25.0, "param_step": 5.0,
    },
    "型態突破": {
        "func": calc_breakout_signals,
        "desc": "尋找經過整理（近20日波動收斂）後爆量突破近20日高點的股票，適合抓噴出啟動點。",
        "param_label": "爆量倍數門檻 (x)", "param_help": "當日成交量相對近20日均量的倍數，數值越高代表要求突破時量能越強。",
        "param_min": 1.1, "param_max": 4.0, "param_default": 1.5, "param_step": 0.1,
    },
}



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


# ============================================================
# [新功能] 多空綜合判讀：市場常用技術指標的多／空判定
# ============================================================
# 這裡不是 AI 給的主觀敘述，是把市場上最常見的幾種技術指標各自獨立
# 算出「這個指標本身」目前偏多還是偏空，透明列出，讓使用者自己看得懂
# 每個判斷是怎麼來的，最後再統計「幾個指標偏多、幾個偏空」做綜合結論。
# 需要完整 OHLC K 線資料（不是只有收盤價），沿用 get_kline_data() 的資料格式。

def calc_bull_bear_indicators(df: pd.DataFrame) -> list:
    """計算 6 種常見技術指標的多空訊號：均線排列、KD、RSI、MACD、布林通道、乖離率。
    df 需含 open/high/low/close/volume 欄位且已依日期由舊到新排序（get_kline_data 格式）。
    回傳每個指標一筆 dict：{'指標', '數值', '訊號'(多/空/中性), '說明'}，資料不足時回傳空list。
    """
    if df is None or df.empty or len(df) < 60:
        return []
    close, high, low = df['close'], df['high'], df['low']
    price = float(close.iloc[-1])
    results = []

    # 1. 均線排列 (MA5 / MA20 / MA60)：短中長期均線的相對位置，判斷趨勢方向
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    if pd.notna(ma5) and pd.notna(ma20) and pd.notna(ma60):
        if ma5 > ma20 > ma60:
            verdict, note = '多', 'MA5 > MA20 > MA60，短中長期均線多頭排列'
        elif ma5 < ma20 < ma60:
            verdict, note = '空', 'MA5 < MA20 < MA60，短中長期均線空頭排列'
        else:
            verdict, note = '中性', '均線糾結交錯，尚未形成明確排列'
        results.append({'指標': '均線排列', '數值': f'{ma5:.2f} / {ma20:.2f} / {ma60:.2f}', '訊號': verdict, '說明': note})

    # 2. KD 隨機指標：常見的短線超買超賣與交叉訊號指標
    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    denom = (high9 - low9).replace(0, np.nan)
    rsv = ((close - low9) / denom * 100).fillna(50)
    k_line = rsv.ewm(alpha=1/3, adjust=False).mean()
    d_line = k_line.ewm(alpha=1/3, adjust=False).mean()
    if len(k_line) >= 2:
        k_now, d_now = float(k_line.iloc[-1]), float(d_line.iloc[-1])
        k_prev, d_prev = float(k_line.iloc[-2]), float(d_line.iloc[-2])
        cross_up = k_now > d_now and k_prev <= d_prev
        cross_down = k_now < d_now and k_prev >= d_prev
        if cross_up:
            verdict, note = '多', f'K({k_now:.0f}) 向上黃金交叉 D({d_now:.0f})'
        elif cross_down:
            verdict, note = '空', f'K({k_now:.0f}) 向下死亡交叉 D({d_now:.0f})'
        elif k_now > d_now:
            verdict, note = '多', f'K({k_now:.0f}) 位於 D({d_now:.0f}) 之上'
        else:
            verdict, note = '空', f'K({k_now:.0f}) 位於 D({d_now:.0f}) 之下'
        if k_now > 80:
            note += '，高檔區須留意過熱回檔'
        elif k_now < 20:
            note += '，低檔區留意反彈契機'
        results.append({'指標': 'KD隨機指標', '數值': f'K={k_now:.0f} D={d_now:.0f}', '訊號': verdict, '說明': note})

    # 3. RSI14：沿用既有 calc_rsi，補上多空判定門檻
    rsi = calc_rsi(close)
    if pd.notna(rsi):
        if rsi >= 55:
            verdict = '多'
        elif rsi <= 45:
            verdict = '空'
        else:
            verdict = '中性'
        extra = '，已達超買區間' if rsi > 70 else '，已達超賣區間' if rsi < 30 else ''
        results.append({'指標': 'RSI14', '數值': f'{rsi:.0f}', '訊號': verdict, '說明': f'RSI = {rsi:.0f}{extra}'})

    # 4. MACD：柱狀圖正負與翻轉，判斷動能方向
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    if len(hist) >= 2 and pd.notna(hist.iloc[-1]) and pd.notna(hist.iloc[-2]):
        hist_now, hist_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
        if hist_now > 0 and hist_prev <= 0:
            verdict, note = '多', 'MACD 柱狀圖翻紅（黃金交叉）'
        elif hist_now < 0 and hist_prev >= 0:
            verdict, note = '空', 'MACD 柱狀圖翻黑（死亡交叉）'
        elif hist_now > 0:
            verdict, note = '多', 'MACD 柱狀圖為正，動能偏多'
        else:
            verdict, note = '空', 'MACD 柱狀圖為負，動能偏空'
        results.append({'指標': 'MACD', '數值': f'{hist_now:+.3f}', '訊號': verdict, '說明': note})

    # 5. 布林通道：股價相對通道位置，判斷強弱與是否過熱／過冷
    ma20_boll = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = (ma20_boll + 2 * std20).iloc[-1]
    lower = (ma20_boll - 2 * std20).iloc[-1]
    mid = ma20_boll.iloc[-1]
    if pd.notna(upper) and pd.notna(lower) and pd.notna(mid):
        if price >= upper:
            verdict, note = '多', '股價觸及或突破布林上軌，短線強勢'
        elif price <= lower:
            verdict, note = '空', '股價跌破布林下軌，短線弱勢'
        elif price > mid:
            verdict, note = '多', '股價位於布林中軌之上'
        else:
            verdict, note = '空', '股價位於布林中軌之下'
        results.append({'指標': '布林通道', '數值': f'上{upper:.1f} 中{mid:.1f} 下{lower:.1f}', '訊號': verdict, '說明': note})

    # 6. 20日乖離率：股價偏離均線的幅度，判斷過熱／過冷程度
    if pd.notna(mid) and mid > 0:
        bias20 = (price - mid) / mid * 100
        if bias20 > 3:
            verdict = '多'
        elif bias20 < -3:
            verdict = '空'
        else:
            verdict = '中性'
        results.append({'指標': '20日乖離率', '數值': f'{bias20:+.1f}%', '訊號': verdict, '說明': f'股價與20日均線相差 {bias20:+.1f}%'})

    return results


def summarize_bull_bear(indicators: list) -> dict:
    """統計多空指標清單，給出綜合結論。多空各過半數以上（差距>=2個指標）才判定
    明確偏多／偏空，差距不大則視為多空拉鋸，避免用小樣本硬湊出過度自信的結論。
    """
    if not indicators:
        return {'verdict': '資料不足', 'bull': 0, 'bear': 0, 'neutral': 0, 'total': 0}
    bull = sum(1 for i in indicators if i['訊號'] == '多')
    bear = sum(1 for i in indicators if i['訊號'] == '空')
    neutral = sum(1 for i in indicators if i['訊號'] == '中性')
    total = len(indicators)
    if bull - bear >= 2:
        verdict = '偏多'
    elif bear - bull >= 2:
        verdict = '偏空'
    else:
        verdict = '多空拉鋸'
    return {'verdict': verdict, 'bull': bull, 'bear': bear, 'neutral': neutral, 'total': total}


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


# ============================================================
# [新功能] 策略回測：驗證目前選用策略的歷史勝率
# ============================================================
# 只針對「目前候選清單」的股票回測（而非全市場），控制下載與運算成本；
# 回測用的條件判斷函式跟即時掃描完全共用（_check_*_condition），
# 確保回測結果反映的就是你實際在用的那套邏輯，不是另一套假的驗證邏輯。

_STRATEGY_CHECKERS = {
    "均線多頭排列": _check_ma_condition,
    "量價背離": _check_divergence_condition,
    "型態突破": _check_breakout_condition,
}


@st.cache_data(ttl=3600, show_spinner=False)
def run_strategy_backtest(strategy_name: str, tickers: tuple, param_value: float, vol_limit: int) -> pd.DataFrame:
    """對候選清單股票重放近9個月歷史資料，找出歷史上符合『目前策略』條件的
    每一個訊號點，計算進場後 5 / 10 / 20 個交易日的持有報酬。
    同一檔股票 5 個交易日內只取一次訊號（cooldown），避免同一波段訊號重複灌水。
    """
    checker = _STRATEGY_CHECKERS.get(strategy_name)
    if checker is None or not tickers:
        return pd.DataFrame()

    ticker_str = " ".join(tickers)
    try:
        raw = yf.download(ticker_str, period="9mo", interval="1d", group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)
    except Exception:
        return pd.DataFrame()
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
        if len(sub) < 65:
            continue

        closes = sub["Close"].astype(float).reset_index(drop=True)
        volumes = (sub["Volume"].astype(float) / 1000).reset_index(drop=True)
        dates = sub.index
        n = len(closes)
        last_signal_i = -99
        for i in range(60, n - 1):
            if i - last_signal_i < 5:
                continue
            try:
                ok, _ = checker(closes, volumes, i, param_value)
            except Exception:
                continue
            if not ok:
                continue
            if volumes.iloc[max(0, i - 4):i + 1].mean() < vol_limit:
                continue
            entry = float(closes.iloc[i])
            rec = {
                'ticker': tk,
                '訊號日': pd.Timestamp(dates[i]).strftime('%Y-%m-%d'),
                '進場價': round(entry, 2),
            }
            for h in (5, 10, 20):
                rec[f'{h}日報酬(%)'] = round((float(closes.iloc[i + h]) / entry - 1) * 100, 2) if i + h < n else np.nan
            records.append(rec)
            last_signal_i = i

    return pd.DataFrame(records)


def summarize_backtest(bt: pd.DataFrame) -> list:
    """把回測明細整理成每個持有期間（5/10/20日）的勝率、平均報酬等摘要列。"""
    rows = []
    for h in (5, 10, 20):
        col = f'{h}日報酬(%)'
        s = bt[col].dropna() if col in bt.columns else pd.Series(dtype=float)
        if s.empty:
            continue
        rows.append({
            '持有期間': f'{h} 日', '訊號數': int(len(s)),
            '勝率(%)': round(float((s > 0).mean() * 100), 1),
            '平均報酬(%)': round(float(s.mean()), 2),
            '中位數(%)': round(float(s.median()), 2),
        })
    return rows


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
    """[設計] 統計『目前候選清單』裡各產業別的集中度與平均分數，用來二次確認
    訊號品質：某產業集中大量高分候選股，代表族群同步在動、訊號可信度較高；
    候選股分散在很多產業、各自只有 1-2 檔，代表比較像個股表現而非族群輪動。

    候選股總數 < 10 檔時直接不顯示：太少檔會讓每個產業只分到 1 檔，
    平均分數等於那一檔股票的分數，統計上沒有意義，顯示出來反而是雜訊。
    """
    if df.empty or 'industry' not in df.columns or 'AI評分' not in df.columns:
        return
    if len(df) < 10:
        return
    hot = (df.groupby('industry')
             .agg(標的數=('code', 'count'), 平均分=('AI評分', 'mean'), 平均量變=('量變動(%)', 'mean'), 平均年增=('營收年增', 'mean'))
             .reset_index())
    hot = hot[hot['標的數'] >= 1].sort_values(['平均分', '標的數'], ascending=[False, False]).head(8)
    if hot.empty:
        return

    head_col, clear_col = st.columns([4, 1])
    with head_col:
        st.markdown('<div class="tv-section" style="margin-bottom:2px;">HOT INDUSTRIES · 熱門族群</div><div class="candidate-row-hint" style="margin-bottom:10px;">卡片下方按「篩選」，候選清單只顯示該產業別；同一產業集中越多高分候選股，代表族群同步在動、訊號可信度越高。</div>', unsafe_allow_html=True)
    with clear_col:
        if st.session_state.get('industry_filter'):
            if st.button(f"✕ 清除「{st.session_state.industry_filter}」篩選", key="clear_industry_filter_top", use_container_width=True):
                st.session_state.industry_filter = None
                st.rerun()

    ncols = min(4, len(hot))
    cols = st.columns(ncols)
    for i, (_, r) in enumerate(hot.iterrows()):
        with cols[i % ncols]:
            yoy_txt = 'N/A' if pd.isna(r['平均年增']) else f"{r['平均年增']:.1f}%"
            vol_txt = 'N/A' if pd.isna(r['平均量變']) else f"{r['平均量變']:.1f}%"
            is_active = st.session_state.get('industry_filter') == r['industry']
            active_style = 'border-color:var(--blue)!important;box-shadow:0 0 0 2px rgba(76,141,255,.25);' if is_active else ''
            st.markdown(f"""
            <div class="tv-card" style="margin-bottom:6px;{active_style}">
              <div class="tv-label">{r['industry']}</div>
              <div class="tv-value" style="font-size:22px;color:#8fb2ff;">{r['平均分']:.0f}</div>
              <div class="tv-caption">標的 {int(r['標的數'])} · 量變 {vol_txt} · YoY {yoy_txt}</div>
            </div>
            """, unsafe_allow_html=True)
            # [新功能] 卡片下方的篩選按鈕：點擊後候選清單只顯示該產業別，
            # 再點一次（同一產業）視為取消篩選，回到全部候選。
            btn_label = "✓ 篩選中" if is_active else "篩選"
            if st.button(btn_label, key=f"hot_ind_{r['industry']}", use_container_width=True):
                st.session_state.industry_filter = None if is_active else r['industry']
                st.rerun()



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

# ============================================================
# [新功能] 熱門股判定：根據 Yahoo股市「社群爆紅榜．熱門搜尋」排行
# ============================================================
# 這個榜單是近7日「使用者實際搜尋量佔比」排序，反映的是當下市場話題／
# 時事熱度（例如法說會、財報、產業消息帶動的關注度），比單純用成交量
# 或漲跌幅判斷「熱門」更貼近「根據時事」的直覺意義。

@st.cache_data(ttl=3600, show_spinner=False)
def get_hot_stock_tickers() -> set:
    """抓取 Yahoo股市熱門搜尋排行，回傳目前『熱門股』的股票代碼集合（前30名）。
    抓不到資料時安全回傳空集合（fail-open），熱門股Tag單純不顯示，不影響其他功能。
    """
    try:
        r = requests.get("https://tw.stock.yahoo.com/community/rank/search",
                         headers=get_headers(), timeout=10, verify=False)
        if r.status_code != 200:
            return set()
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)

        # 頁面除了榜單本身，導覽列也會帶幾個範例代碼（如「個股比較」連結裡的
        # 2330.TW,2454.TW），為避免誤判，只擷取「資料時間：」到「我的自選股」
        # 之間、真正屬於榜單的區塊再解析，抓不到這兩個標記就直接放棄（寧可沒有
        # 標籤，也不要標錯）。
        start_idx = text.find("資料時間：")
        end_idx = text.find("我的自選股")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return set()
        table_text = text[start_idx:end_idx]

        codes = []
        for m in re.finditer(r'(\d{4,6})\.(TW|TWO)\b', table_text):
            code = m.group(1)
            if code not in codes:
                codes.append(code)
            if len(codes) >= 30:
                break
        return set(codes)
    except Exception:
        return set()


@st.cache_data(ttl=86400, show_spinner=False)
def get_etf_holdings(code: str, market_suffix: str) -> dict:
    """抓取 ETF 前十大持股與資產分佈。來源：Yahoo股市 ETF 持股分析頁面。
    Yahoo 只公開揭露『前十大持股』，不是完整成分股清單（完整清單通常要跟
    發行商或跟公開說明書才能拿到），資料通常每月更新一次，抓不到時回傳空 dict。
    """
    ticker = f"{code}.{market_suffix}"
    result = {'holdings': [], 'as_of': None, 'asset_alloc': []}
    try:
        r = requests.get(f"https://tw.stock.yahoo.com/quote/{ticker}/holding",
                         headers=get_headers(), timeout=10, verify=False)
        if r.status_code != 200:
            return result
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)

        # 前十大持股：從「前十大持股」標記之後才開始解析，避免跟前面的
        # 「資產分佈」區塊（同樣是「數字. 名稱 百分比%」格式）混在一起。
        start = text.find("前十大持股")
        if start != -1:
            end = text.find("我的自選股", start)
            segment = text[start:end] if end != -1 else text[start:start + 1500]
            date_m = re.search(r'資料時間：(\d{4}/\d{2}/\d{2})', segment)
            if date_m:
                result['as_of'] = date_m.group(1)
            for m in re.finditer(r'(\d{1,2})\s*\.\s*([\u4e00-\u9fffA-Za-z0-9\*]{1,16}?)\s+(\d{1,3}\.\d{1,2})%', segment):
                rank, name, pct = m.groups()
                result['holdings'].append({'排名': int(rank), '名稱': name, '占比(%)': float(pct)})
                if len(result['holdings']) >= 10:
                    break

        # 資產分佈（股票／債券／現金／其他），出現在「前十大持股」之前
        alloc_start = text.find("資產分佈")
        if alloc_start != -1:
            alloc_end = start if start != -1 else alloc_start + 400
            alloc_segment = text[alloc_start:alloc_end]
            for m in re.finditer(r'(股票|債券|現金|其他)\s+(\d{1,3}\.\d{1,2})%', alloc_segment):
                name, pct = m.groups()
                result['asset_alloc'].append({'類別': name, '占比(%)': float(pct)})
    except Exception:
        pass
    return result


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
/* [新功能] 進度條加強：加粗、發光、動態流動效果，並顯示 st.progress(text=...) 帶的百分比文字，掃描時一眼就能看到進度 */
[data-testid='stProgress']{margin:6px 0 4px;}
[data-testid='stProgress']>div{background:#0d1c30!important;border-radius:999px!important;height:26px!important;border:1px solid rgba(76,141,255,.28)!important;box-shadow:inset 0 0 10px rgba(0,0,0,.35)!important;overflow:visible!important;}
[data-testid='stProgress']>div>div{background:linear-gradient(90deg,#3978e8,#4c8dff,#35c48d)!important;background-size:200% 100%!important;animation:tv-progress-flow 2.2s linear infinite!important;border-radius:999px!important;box-shadow:0 0 16px rgba(76,141,255,.55),0 0 4px rgba(53,196,141,.5)!important;position:relative!important;min-width:26px!important;}
[data-testid='stProgress'] p{color:#eaf2ff!important;font-weight:800!important;font-size:14px!important;font-family:'Roboto Mono',monospace!important;margin-bottom:4px!important;}
@keyframes tv-progress-flow{0%{background-position:0% 0%}100%{background-position:200% 0%}}
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
/* [新功能] 手機版卡片清單：桌面顯示表格、手機顯示卡片，兩者都渲染由 CSS 依寬度切換，
   避免用 JS 偵測螢幕寬度，st.container(key=...) 產生的 st-key-* class 可直接用 CSS 選取。 */
.st-key-desktop_candidate_list{display:block;}
.st-key-mobile_candidate_list{display:none;}
.mobile-stock-card{background:linear-gradient(180deg,rgba(16,33,56,.92),rgba(9,20,35,.96));border:1px solid var(--border);border-radius:12px;padding:12px 14px;margin-bottom:8px;}
.mobile-stock-card .msc-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}
.mobile-stock-card .msc-name{font-size:15px;font-weight:800;color:#f4f8ff;}
.mobile-stock-card .msc-code{color:var(--muted);font-size:12px;margin-left:6px;}
.mobile-stock-card .msc-score{font-family:'Roboto Mono',monospace;font-weight:800;font-size:16px;color:#4c8dff;}
.mobile-stock-card .msc-metrics{display:flex;gap:14px;font-family:'Roboto Mono',monospace;font-size:12.5px;color:#c7d5e6;}
.mobile-stock-card .msc-metrics span{color:var(--muted);margin-right:3px;}
.strategy-badge{display:inline-block;background:rgba(76,141,255,.14);color:#9fc0ff;border:1px solid rgba(76,141,255,.3);border-radius:999px;padding:3px 11px;font-size:11px;font-weight:800;margin-left:8px;}
/* [新功能] 熱門股 Tag：橘紅色系，跟其他中性 quote-tag 拉開視覺區別，一眼看出是熱門股 */
.hot-tag{background:rgba(255,138,61,.16)!important;border-color:rgba(255,138,61,.45)!important;color:#ffb37a!important;font-weight:800!important;}
.hot-badge-inline{display:inline-block;margin-left:4px;font-size:12px;}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-track{background:#07111f}::-webkit-scrollbar-thumb{background:#29425f;border-radius:999px}
@media(max-width:1000px){.block-container{padding:16px 14px 40px!important}.workflow,.stat-grid{grid-template-columns:repeat(2,1fr)}.app-hero{align-items:flex-start}.app-meta{display:none}.workspace-left{max-height:420px;}}
@media(max-width:760px){
  .workflow,.stat-grid,.quote-metrics{grid-template-columns:1fr}.app-title{font-size:22px}.quote-price{font-size:31px}.section-help{display:none}
  .st-key-desktop_candidate_list{display:none;}
  .st-key-mobile_candidate_list{display:block;}
}
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

tab_scan, tab_workspace, tab_watchlist = st.tabs([
    "🔍 選股掃描",
    "📊 候選與分析工作台",
    "⭐ 自選股追蹤",
])

# ------------------------------------------------------------
# TAB 1：選股掃描
# ------------------------------------------------------------
with tab_scan:
    st.markdown('<div class="section-head"><div><div class="section-title">設定掃描條件</div><div class="section-help">條件越嚴格，候選股票通常越少；第一次使用可保留預設值。</div></div></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="control-shell">', unsafe_allow_html=True)

        # [新功能] 策略選單：切換策略時，下方參數與說明會跟著換
        strategy_name = st.selectbox(
            "選股策略", list(STRATEGY_REGISTRY.keys()),
            key="selected_strategy",
            help="切換策略後，下方的參數欄位與說明會自動對應調整。"
        )
        strategy_cfg = STRATEGY_REGISTRY[strategy_name]

        note_col, param_col, vol_col = st.columns([1.6, 1, 1], gap="large")
        with note_col:
            st.markdown(f"""
            <div class="control-note">
              <b>{strategy_name}</b><span class="strategy-badge">STRATEGY</span><br>
              {strategy_cfg['desc']}
            </div>
            """, unsafe_allow_html=True)
        with param_col:
            # [新功能] 每個策略有自己的關鍵參數（乖離上限 / 量增門檻 / 爆量倍數），
            # session_state key 依策略名稱區分，切換策略互不干擾、也記得上次調過的值。
            param_key = f"strategy_param__{strategy_name}"
            if param_key not in st.session_state:
                st.session_state[param_key] = strategy_cfg["param_default"]
            param_value = st.number_input(
                strategy_cfg["param_label"],
                strategy_cfg["param_min"], strategy_cfg["param_max"],
                value=float(st.session_state[param_key]), step=strategy_cfg["param_step"],
                key=f"input__{param_key}", help=strategy_cfg["param_help"],
            )
            st.session_state[param_key] = param_value
        with vol_col:
            mb_vol = st.slider(
                "最小成交量 (張)", 0, 3000,
                value=int(st.session_state.user_vol), key="mb_vol",
                help="排除流動性較低的股票。"
            )
            st.session_state.user_vol = mb_vol
        st.markdown('</div>', unsafe_allow_html=True)

    user_vol = st.session_state.user_vol
    param_value = st.session_state[f"strategy_param__{strategy_name}"]
    scan_left, scan_right = st.columns([3, 1])
    with scan_left:
        st.caption(f"目前策略：{strategy_name}｜{strategy_cfg['param_label']} = {param_value:g}｜成交量 ≥ {user_vol:,} 張")
    with scan_right:
        scan_clicked = st.button(
            "開始全市場掃描", use_container_width=True,
            disabled=st.session_state.is_scanning, type="primary", key="scan_market"
        )

    if scan_clicked:
        st.session_state.is_scanning = True
        st.session_state.current_idx = 0
        st.session_state.last_selected_row = None
        st.session_state.scan_strategy_used = strategy_name
        st.session_state.industry_filter = None  # [新功能] 新掃描結果的產業別分布會變，舊篩選清掉避免篩出空清單
        st.rerun()

    if st.session_state.is_scanning:
        status = st.empty()
        bar = st.progress(0)
        BATCH = 200
        active_strategy = st.session_state.get("scan_strategy_used", strategy_name)
        active_cfg = STRATEGY_REGISTRY[active_strategy]
        active_param = st.session_state[f"strategy_param__{active_strategy}"]

        def _update_progress(pct: float, msg: str):
            """[新功能] 進度條同時顯示百分比文字，狀態訊息同步更新，一目了然。"""
            pct = max(0.0, min(1.0, pct))
            status.info(msg)
            bar.progress(pct, text=f"{pct * 100:.0f}%")

        _update_progress(0.03, "步驟 1/3：正在載入上市櫃股票清單…")
        stock_map = get_stock_market_list()
        all_tickers = [s["ticker"] for s in stock_map]
        total_tickers = len(all_tickers)

        history_map = {}
        batches = [all_tickers[i:i+BATCH] for i in range(0, total_tickers, BATCH)]
        for bi, batch in enumerate(batches):
            _update_progress(
                0.03 + 0.72 * (bi / max(1, len(batches))),
                f"步驟 2/3：下載歷史行情批次 {bi+1}/{len(batches)}…"
            )
            history_map.update(download_batch_history(tuple(batch)))

        _update_progress(0.80, f"步驟 3/3：套用「{active_strategy}」策略計算候選股票…")
        initial_hits = active_cfg["func"](history_map, stock_map, active_param, user_vol)

        if initial_hits:
            _update_progress(0.80, f"已找到 {len(initial_hits)} 檔候選股，正在補齊本益比、營收與 AI 評分…")
            load_official_pe_map(False)
            hot_codes = get_hot_stock_tickers()  # [新功能] 熱門股判定：整批只抓一次排行榜，逐檔比對代碼
            final_list = []
            with ThreadPoolExecutor(max_workers=6) as ex:
                f_deep = {ex.submit(fetch_deep_info, r["ticker"]): r for r in initial_hits}
                for j, f in enumerate(as_completed(f_deep), 1):
                    _update_progress(
                        0.80 + 0.19 * j / len(initial_hits),
                        f"步驟 3/3：補齊個股資料中… ({j}/{len(initial_hits)})"
                    )
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
                        "策略": base.get("策略", active_strategy), "訊號說明": base.get("訊號說明", ""),
                        "熱門股": base["code"] in hot_codes,
                        "本益比": deep_res["pe"], "營收月增": deep_res["mom"], "營收年增": deep_res["yoy"],
                    }
                    score, radar = calc_stock_score(row_data)
                    row_data["AI評分"] = score
                    row_data["飆股雷達"] = "、".join(radar) if radar else "觀察"
                    final_list.append(row_data)
            _update_progress(1.0, "掃描完成，正在整理結果…")
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

        # ══════════════ 熱門族群：候選股集中度與平均分數，二次確認訊號品質 ══════════════
        render_hot_industries(df)

        # ══════════════ 策略回測比較：驗證目前策略的歷史勝率 ══════════════
        with st.expander(f"🧪 策略回測：驗證「{st.session_state.scan_strategy_used}」的歷史勝率（近9個月）", expanded=False):
            st.caption(
                "針對目前候選清單的股票，重放近9個月歷史資料找出符合『目前策略』條件的歷史訊號點，"
                "計算進場後 5／10／20 個交易日的持有報酬。切換策略、重新掃描後再跑一次回測，"
                "下方會列出每個策略跑過的結果，方便互相比較勝率。過去績效不代表未來表現，僅供參考，非投資建議。"
            )
            bt_strategy = st.session_state.scan_strategy_used
            bt_param = st.session_state.get(f"strategy_param__{bt_strategy}", STRATEGY_REGISTRY[bt_strategy]["param_default"])
            if st.button(f"▶ 執行「{bt_strategy}」回測", key="btn_run_backtest", use_container_width=True):
                with st.spinner(f"回測中（下載候選清單 {len(df)} 檔股票近9個月資料）..."):
                    bt_df = run_strategy_backtest(bt_strategy, tuple(df['ticker'].tolist()), float(bt_param), int(user_vol))
                st.session_state.backtest_results[bt_strategy] = bt_df

            if st.session_state.backtest_results:
                compare_rows = []
                for sname, bt_df in st.session_state.backtest_results.items():
                    if bt_df is None or bt_df.empty:
                        compare_rows.append({'策略': sname, '訊號數': 0, '20日勝率(%)': None, '20日平均報酬(%)': None})
                        continue
                    summary = summarize_backtest(bt_df)
                    row20 = next((r for r in summary if r['持有期間'] == '20 日'), None)
                    compare_rows.append({
                        '策略': sname,
                        '訊號數': row20['訊號數'] if row20 else len(bt_df),
                        '20日勝率(%)': row20['勝率(%)'] if row20 else None,
                        '20日平均報酬(%)': row20['平均報酬(%)'] if row20 else None,
                    })
                st.markdown('<div class="candidate-row-hint" style="margin-top:6px;">策略勝率比較（切換策略並各自執行過回測後會累積列在這裡）</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)

                current_bt = st.session_state.backtest_results.get(bt_strategy)
                if current_bt is not None and not current_bt.empty:
                    detail_rows = summarize_backtest(current_bt)
                    if detail_rows:
                        st.markdown(f'<div class="candidate-row-hint">「{bt_strategy}」各持有期間明細</div>', unsafe_allow_html=True)
                        st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)
                elif current_bt is not None:
                    st.info(f"「{bt_strategy}」在候選清單範圍內近9個月沒有找到符合條件的歷史訊號。")

        left_col, right_col = st.columns([1.5, 2.2], gap="medium")

        # ══════════════ 左欄：候選清單（常駐，切換右側檢視時不消失）══════════════
        with left_col:
            with st.container(border=True):
                st.markdown('<div class="section-title" style="font-size:15px;margin-bottom:8px;">候選股票清單</div><div class="candidate-row-hint">點擊任一列即可切換右側個股工作台。</div>', unsafe_allow_html=True)

                # [新功能] 熱門族群卡片點擊後的產業篩選提示（跟快速篩選是 AND 關係）
                if st.session_state.get('industry_filter'):
                    fc1, fc2 = st.columns([2.4, 1])
                    with fc1:
                        st.caption(f"🏷 產業篩選中：{st.session_state.industry_filter}")
                    with fc2:
                        if st.button("✕ 清除", key="clear_industry_filter_side", use_container_width=True):
                            st.session_state.industry_filter = None
                            st.rerun()

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
                if st.session_state.get('industry_filter'):
                    view_df = view_df[view_df['industry'] == st.session_state.industry_filter]

                csv = view_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("下載目前清單 CSV", csv, f'tw_stock_scan_{get_tw_now().strftime("%Y%m%d")}.csv', 'text/csv', use_container_width=True)

                # [新功能] 桌面版：完整表格。用 st.container(key=...) 讓 CSS 依螢幕寬度
                # 切換顯示／隱藏（st-key-desktop_candidate_list），手機版改顯示下方卡片清單。
                with st.container(key="desktop_candidate_list"):
                    show_cols = ["code", "name", "AI評分", "收盤", "漲跌幅(%)", "量比20日"]
                    available_cols = [c for c in show_cols if c in view_df.columns]
                    df_display = view_df[available_cols].rename(columns={"code": "代碼", "name": "名稱"})
                    if "熱門股" in view_df.columns:
                        # [新功能] 名稱後面加 🔥 標示熱門股，不額外佔一欄，維持窄欄表格的空間
                        df_display["名稱"] = [
                            f"🔥 {n}" if hot else n
                            for n, hot in zip(df_display["名稱"], view_df["熱門股"])
                        ]

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
                            # [修正] 名稱欄加寬（80→118px），台股名稱常見4~6個中文字，
                            # 加上熱門股🔥前綴後原本寬度會被截斷；其餘欄位微調挪出空間。
                            "代碼": st.column_config.TextColumn("代碼", width=56),
                            "名稱": st.column_config.TextColumn("名稱", width=118),
                            "AI評分": st.column_config.ProgressColumn("AI評分", width=80, format="%d", min_value=0, max_value=100),
                            "收盤": st.column_config.NumberColumn("價格", width=60, format="%.2f"),
                            "漲跌幅(%)": st.column_config.NumberColumn("漲跌", width=58, format="%.1f%%"),
                            "量比20日": st.column_config.NumberColumn("量比", width=54, format="%.2fx"),
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

                # [新功能] 手機版：卡片清單。CSS 在寬度 ≤760px 時才顯示這個容器，
                # 桌面版會被隱藏（不佔畫面），每張卡片用一個按鈕觸發選股，
                # 效果等同桌面表格的點列選股。
                with st.container(key="mobile_candidate_list"):
                    for _, row in view_df.iterrows():
                        chg = row.get('漲跌幅(%)', np.nan)
                        chg_color = '#22ab94' if pd.notna(chg) and chg > 0 else '#f23645' if pd.notna(chg) and chg < 0 else '#c7d5e6'
                        chg_txt = 'N/A' if pd.isna(chg) else f"{chg:+.1f}%"
                        score_val = row.get('AI評分', np.nan)
                        hot_badge = '<span class="hot-badge-inline">🔥</span>' if bool(row.get('熱門股', False)) else ''
                        st.markdown(f"""
                        <div class="mobile-stock-card">
                          <div class="msc-head">
                            <div><span class="msc-name">{row['name']}</span>{hot_badge}<span class="msc-code">{row['code']}</span></div>
                            <div class="msc-score">{'N/A' if pd.isna(score_val) else int(score_val)}</div>
                          </div>
                          <div class="msc-metrics">
                            <div><span>價</span>{fmt_num(row.get('收盤', np.nan), '{:.2f}')}</div>
                            <div><span>漲跌</span><span style="color:{chg_color};">{chg_txt}</span></div>
                            <div><span>量比</span>{fmt_num(row.get('量比20日', np.nan), '{:.2f}x')}</div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"查看 {row['name']} ({row['code']})", key=f"mobile_card_{row['code']}", use_container_width=True):
                            matches = df.index[df['code'].astype(str) == str(row['code'])].tolist()
                            if matches:
                                st.session_state.current_idx = matches[0]
                                st.rerun()

        # ══════════════ 右欄：個股工作台（報價 + 分段切換 K線／AI分析／新聞）══════════════
        with right_col:
            nav_star, nav_space, nav1, nav2 = st.columns([0.9, 2.5, 0.8, 0.8])
            with nav_star:
                # [新功能] 自選股／追蹤清單：星號切換加入/移除，重新讀檔即時反映目前狀態
                _wl_now = load_watchlist()
                _in_wl = is_in_watchlist(current_stock['code'], _wl_now)
                star_label = "★ 追蹤中" if _in_wl else "☆ 加入追蹤"
                if st.button(star_label, use_container_width=True, key="toggle_watchlist"):
                    if _in_wl:
                        remove_from_watchlist(current_stock['code'])
                    else:
                        add_to_watchlist(current_stock.to_dict())
                    st.session_state.watchlist_quotes = pd.DataFrame()  # 清空快取，下次進追蹤頁會提示重新整理
                    st.rerun()
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
            is_hot = bool(current_stock.get('熱門股', False))
            hot_tag_html = '<div class="quote-tag hot-tag">🔥 熱門股</div>' if is_hot else ''

            st.markdown(f"""
            <div class="quote-panel" style="margin-top:8px;">
              <div class="quote-head"><div class="quote-title">{current_stock['name']} · {current_stock['code']}</div><div class="quote-tag">{current_stock.get('市場別', '')}</div><div class="quote-tag">{current_stock.get('industry', '未分類')}</div>{hot_tag_html}</div>
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
                ["📈 K線圖", "🤖 AI 分析", "📐 多空指標", "🏢 公司資訊", "📰 個股新聞"],
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

            # ---------- 多空指標：市場常用技術指標的多空綜合判讀 ----------
            elif view_mode == "📐 多空指標":
                st.markdown(f'<div class="section-head" style="margin-top:4px;"><div><div class="section-title" style="font-size:15px;">{current_stock["name"]} 多空綜合判讀</div><div class="section-help">彙整市場上常用的技術指標各自獨立判斷，非 AI 主觀敘述，僅供參考，非投資建議。</div></div></div>', unsafe_allow_html=True)

                k_source_df = get_kline_data(current_stock['code'], "TW" if current_stock['ticker'].endswith(".TW") else "TWO")
                indicators = calc_bull_bear_indicators(k_source_df)

                if indicators:
                    summary = summarize_bull_bear(indicators)
                    verdict_color = {'偏多': 'var(--green)', '偏空': 'var(--red)', '多空拉鋸': 'var(--yellow)'}.get(summary['verdict'], '#8f9bad')
                    st.markdown(f"""
                    <div class="quote-panel" style="margin:4px 0 16px;">
                      <div class="quote-head"><div class="quote-title" style="color:{verdict_color};">綜合判讀：{summary['verdict']}</div></div>
                      <div class="quote-metrics" style="grid-template-columns:repeat(3,minmax(90px,1fr));">
                        <div><div class="metric-k">看多指標</div><div class="metric-v" style="color:var(--green);">{summary['bull']} / {summary['total']}</div></div>
                        <div><div class="metric-k">看空指標</div><div class="metric-v" style="color:var(--red);">{summary['bear']} / {summary['total']}</div></div>
                        <div><div class="metric-k">中性指標</div><div class="metric-v">{summary['neutral']} / {summary['total']}</div></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    for ind in indicators:
                        badge_color = {'多': 'var(--green)', '空': 'var(--red)', '中性': '#8f9bad'}.get(ind['訊號'], '#8f9bad')
                        badge_bg = {'多': 'rgba(34,171,148,0.14)', '空': 'rgba(242,54,69,0.14)', '中性': 'rgba(139,148,158,0.14)'}.get(ind['訊號'], 'rgba(139,148,158,0.14)')
                        st.markdown(f"""
                        <div class="side-card" style="margin-bottom:10px;">
                          <div class="side-title" style="margin-bottom:4px;">
                            <span>{ind['指標']}</span>
                            <span style="background:{badge_bg};color:{badge_color};border:1px solid {badge_color}40;border-radius:999px;padding:3px 12px;font-size:12px;font-weight:800;">{ind['訊號']}</span>
                          </div>
                          <div class="tv-caption" style="font-family:'Roboto Mono',monospace;margin-bottom:4px;">{ind['數值']}</div>
                          <div style="color:#c7d5e6;font-size:13px;">{ind['說明']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.caption("多空各項指標為市場常見的獨立技術分析工具，彼此可能出現不一致的訊號（例如趨勢指標偏多、短線震盪指標偏空），這是正常現象；綜合判讀僅是「多空指標數量對比」，不代表保證漲跌，仍請自行評估風險。")
                else:
                    st.info("目前無法取得足夠的歷史 K 線資料計算多空指標（需至少 60 個交易日），可能是新股或資料來源暫時無回應，請稍後再試。")

            # ---------- 公司資訊：公司簡介 + 季度 EPS 列表（ETF 改顯示成分股占比）----------
            elif view_mode == "🏢 公司資訊":
                market_suffix = "TW" if current_stock['ticker'].endswith(".TW") else "TWO"
                # [新功能] ETF 判定：產業別為「ETF」或代碼為 00 開頭（台股 ETF 代碼慣例）
                is_etf = (str(current_stock.get('industry', '')) == 'ETF') or bool(re.match(r'^00\d', str(current_stock['code'])))
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
                elif is_etf:
                    st.info(f"「{current_stock['name']}」是 ETF（基金），不是一般公司，沒有董事長／總經理等公司登記資料，屬正常情況。")
                else:
                    st.info("目前無法取得公司基本資料，可能是新股或官方資料尚未更新，請稍後再試。")

                if is_etf:
                    # [新功能] ETF 成分股與占比：取代一般個股才適用的 EPS 區塊
                    st.markdown('<div class="section-head" style="margin-top:22px;"><div><div class="section-title" style="font-size:15px;">ETF 成分股與占比</div><div class="section-help">資料來源：Yahoo 股市 ETF 持股分析頁面，僅公開揭露前十大持股，非完整成分股清單。</div></div></div>', unsafe_allow_html=True)
                    etf_data = get_etf_holdings(current_stock['code'], market_suffix)
                    holdings = etf_data.get('holdings', [])
                    if holdings:
                        as_of_txt = f"（資料時間：{etf_data['as_of']}）" if etf_data.get('as_of') else ""
                        st.caption(f"前十大持股占比{as_of_txt}，加總約為基金淨值的 {sum(h['占比(%)'] for h in holdings):.1f}%，其餘由更多分散持股組成。")

                        holdings_df = pd.DataFrame(holdings)
                        fig = go.Figure(go.Bar(
                            x=holdings_df['占比(%)'][::-1], y=holdings_df['名稱'][::-1], orientation='h',
                            marker_color='#4c8dff',
                            hovertemplate='%{y}<br>占比 %{x:.2f}%<extra></extra>',
                        ))
                        fig.update_layout(
                            height=320, template='plotly_dark',
                            paper_bgcolor='#0b121b', plot_bgcolor='#0b121b',
                            margin=dict(l=10, r=20, t=10, b=20),
                            xaxis=dict(title='占比 (%)', gridcolor='rgba(148,163,184,0.09)'),
                            yaxis=dict(gridcolor='rgba(148,163,184,0.09)'),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.dataframe(
                            holdings_df, hide_index=True, use_container_width=True,
                            column_config={
                                "排名": st.column_config.NumberColumn("排名", width=55),
                                "名稱": st.column_config.TextColumn("成分股名稱", width=140),
                                "占比(%)": st.column_config.ProgressColumn("占比", width=140, format="%.2f%%", min_value=0, max_value=max(h['占比(%)'] for h in holdings)),
                            }
                        )

                        if etf_data.get('asset_alloc'):
                            alloc_txt = "、".join(f"{a['類別']} {a['占比(%)']:.1f}%" for a in etf_data['asset_alloc'])
                            st.caption(f"資產配置：{alloc_txt}")
                    else:
                        st.info("目前無法取得這檔 ETF 的成分股資料，可能是新掛牌 ETF 或資料來源暫時無回應，請稍後再試。")
                else:
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

# ------------------------------------------------------------
# TAB 3：自選股／追蹤清單
# ------------------------------------------------------------
# [新功能] 讓還沒符合策略條件、但你想持續觀察的股票也能被記錄下來，
# 不用每次都全市場重新掃描才看得到它們。清單存在本機 JSON 檔案
# （watchlist_v1.json），跟 get_stock_market_list() 用同一套持久化寫法。
with tab_watchlist:
    st.markdown('<div class="section-head"><div><div class="section-title">自選股／追蹤清單</div><div class="section-help">在「候選與分析工作台」右上角點 ☆ 加入追蹤，這裡會持續記錄，不受重新掃描影響。</div></div></div>', unsafe_allow_html=True)

    watchlist = load_watchlist()
    if not watchlist:
        st.markdown("""
        <div class="tv-panel" style="text-align:center;padding:42px 22px;margin-top:8px;">
          <div style="font-size:40px;margin-bottom:12px;">⭐</div>
          <div style="font-size:20px;font-weight:800;color:#f4f8ff;">追蹤清單目前是空的</div>
          <div class="tv-caption" style="margin-top:9px;line-height:1.8;">掃描完成後，在「候選與分析工作台」右上角點「☆ 加入追蹤」，<br>就能把還沒達標、但值得持續觀察的股票留在這裡。</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        rc1, rc2 = st.columns([1, 3])
        with rc1:
            if st.button("🔄 更新報價", use_container_width=True, key="btn_refresh_watchlist", type="primary"):
                with st.spinner(f"正在更新 {len(watchlist)} 檔追蹤股票的報價..."):
                    st.session_state.watchlist_quotes = refresh_watchlist_quotes(watchlist)
        with rc2:
            st.caption(f"共追蹤 {len(watchlist)} 檔股票。首次查看或剛新增／移除後，請按「更新報價」重新整理數字。")

        quotes_df = st.session_state.watchlist_quotes
        # 追蹤清單本身有異動（新增/移除）時，快取的報價欄位數會對不上，提示需要重新整理
        stale = quotes_df is None or quotes_df.empty or len(quotes_df) != len(watchlist)

        if stale:
            st.info("報價尚未載入或追蹤清單剛更新過，請按上方「🔄 更新報價」。")
        else:
            for _, row in quotes_df.iterrows():
                ret = row.get('累計報酬(%)', np.nan)
                ret_color = 'var(--green)' if pd.notna(ret) and ret >= 0 else 'var(--red)' if pd.notna(ret) else '#8f9bad'
                ret_txt = 'N/A' if pd.isna(ret) else f"{ret:+.2f}%"
                warn_bits = []
                if row.get('跌破MA30'):
                    warn_bits.append('跌破MA30')
                rsi_v = row.get('RSI14', np.nan)
                if pd.notna(rsi_v) and rsi_v > 78:
                    warn_bits.append(f'RSI過熱{rsi_v:.0f}')
                warn_txt = f" · ⚠ {' / '.join(warn_bits)}" if warn_bits else ""

                with st.container(border=True):
                    head_col, price_col, ret_col, btn_col1, btn_col2 = st.columns([2, 1.2, 1.2, 1, 1])
                    with head_col:
                        hot_prefix = "🔥 " if bool(row.get('熱門股', False)) else ""
                        st.markdown(f"**{hot_prefix}{row['name']}** ({row['code']})　<span style='color:var(--muted);font-size:12px;'>{row['industry']}｜加入於 {row['added_date']}</span>", unsafe_allow_html=True)
                    with price_col:
                        added_p = row.get('added_price')
                        st.markdown(f"加入價 {fmt_num(added_p, '{:.2f}') if added_p else 'N/A'} → 現價 **{fmt_num(row.get('現價', np.nan), '{:.2f}')}**")
                    with ret_col:
                        st.markdown(f"<span style='color:{ret_color};font-weight:800;font-family:Roboto Mono,monospace;'>{ret_txt}</span>{warn_txt}", unsafe_allow_html=True)
                    with btn_col1:
                        if st.button("📈 K線", key=f"wl_chart_{row['code']}", use_container_width=True):
                            st.session_state.watchlist_chart_target = (row['ticker'], row['name'])
                    with btn_col2:
                        if st.button("🗑 移除", key=f"wl_remove_{row['code']}", use_container_width=True):
                            remove_from_watchlist(row['code'])
                            st.session_state.watchlist_quotes = pd.DataFrame()
                            if st.session_state.watchlist_chart_target and st.session_state.watchlist_chart_target[0] == row['ticker']:
                                st.session_state.watchlist_chart_target = None
                            st.rerun()

            if st.session_state.watchlist_chart_target:
                wl_tk, wl_name = st.session_state.watchlist_chart_target
                st.markdown(f'<div class="section-head" style="margin-top:20px;"><div><div class="section-title" style="font-size:16px;">{wl_name} K線圖</div></div></div>', unsafe_allow_html=True)
                wl_fig = draw_k_line(wl_tk, wl_name, chart_mode='K線圖', chart_period='日')
                if wl_fig:
                    render_kline_chart_with_axis_price(wl_fig, height=440)
                else:
                    st.warning("⚠️ 無法載入 K 線資料，請稍後再試。")
