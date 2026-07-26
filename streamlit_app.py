import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING
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
    # [新功能] 手動查詢個股：不需策略掃描，直接輸入代碼查詢
    ('manual_stocks',       []),   # 手動查詢過的股票清單（跟策略掃描結果同格式，最新查詢排最前面）
    ('manual_current_code', None), # 剛查到的代碼，用來把 current_idx 切過去
    # [新功能] 個股投資分析報告：量化資料 + 同業比較 + 波動率情境價 + AI敘事
    ('report_target_code', None),     # 目前報告頁籤鎖定要看的股票代碼
    ('report_cache',       {}),       # {代碼: 報告dict}，避免每次重跑都重新呼叫AI（成本較高）
    # [新功能] 風險控管與部位計算：ATR 停損倍數、單筆可承受風險金額
    ('risk_budget',        30000),   # 單筆交易可承受的最大虧損金額（元）
    ('atr_stop_mult',      2.0),     # 停損價 = 現價 − N × ATR20，N 即這個倍數
    ('show_position_cols', True),    # 是否在候選清單表格顯示停損／張數兩欄
    # [新功能] 持倉中心／Telegram 停損警示
    ('portfolio_quotes', pd.DataFrame()),
    ('portfolio_last_check', None),
    ('portfolio_alert_result', []),
    # report tab session state keys are initialized inline above
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
    """加入追蹤：記錄加入當下的日期／價格／財務評分，供之後計算累計報酬。"""
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


# ============================================================
# [新功能] 持倉管理／交易計畫／Telegram 停損警示
# ============================================================
# 持倉與 Telegram 設定採輕量 JSON 持久化，使用方式與追蹤清單一致。
# Streamlit Cloud 重新部署時容器檔案可能被重建；正式長期使用建議改接資料庫，
# Telegram Token 則優先放在 .streamlit/secrets.toml 或環境變數。

APP_DATA_DIR = os.path.dirname(__file__) if '__file__' in globals() else '.'
PORTFOLIO_FILE = os.path.join(APP_DATA_DIR, 'portfolio_v1.json')
TELEGRAM_CONFIG_FILE = os.path.join(APP_DATA_DIR, 'telegram_alert_config_v1.json')


def _safe_number(value, default=np.nan):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default


def is_etf_instrument(stock: dict) -> bool:
    """依股票資料判斷 ETF／ETN；這類商品的升降單位與一般股票不同。"""
    code = str(stock.get('code', '') or '').strip()
    industry = str(stock.get('industry', '') or '').strip().upper()
    name = str(stock.get('name', '') or '').strip().upper()
    return industry in {'ETF', 'ETN'} or code.startswith('00') or 'ETF' in name or 'ETN' in name


def get_tw_price_tick(price: float, is_etf: bool = False) -> float:
    """取得台股可申報價格的最小升降單位。"""
    p = _safe_number(price)
    if pd.isna(p) or p <= 0:
        return 0.01
    if is_etf:
        return 0.01 if p < 50 else 0.05
    if p < 10:
        return 0.01
    if p < 50:
        return 0.05
    if p < 100:
        return 0.10
    if p < 500:
        return 0.50
    if p < 1000:
        return 1.00
    return 5.00


def round_to_tw_tick(price: float, is_etf: bool = False, mode: str = 'nearest') -> float:
    """把計算價位修正成台股可實際下單的價格檔位。"""
    p = _safe_number(price)
    if pd.isna(p) or p <= 0:
        return np.nan
    tick = Decimal(str(get_tw_price_tick(p, is_etf)))
    value = Decimal(str(p)) / tick
    rounding = {
        'down': ROUND_FLOOR,
        'up': ROUND_CEILING,
        'nearest': ROUND_HALF_UP,
    }.get(mode, ROUND_HALF_UP)
    result = value.to_integral_value(rounding=rounding) * tick
    return float(result)


def load_positions() -> list:
    """讀取持倉紀錄；損毀或不存在時回傳空清單。"""
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception:
        pass
    return []


def save_positions(positions: list) -> bool:
    try:
        with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
            json.dump(positions, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def find_open_position(code: str, positions=None):
    positions = load_positions() if positions is None else positions
    return next((p for p in positions
                 if str(p.get('code')) == str(code) and p.get('status', 'open') == 'open'), None)


def upsert_position(position: dict) -> tuple[bool, str]:
    """同一股票只保留一筆未平倉部位；再次儲存時更新原紀錄。"""
    positions = load_positions()
    code = str(position.get('code', '')).strip()
    if not code:
        return False, '股票代碼不可空白'

    now_text = get_tw_now().isoformat(timespec='seconds')
    position = dict(position)
    position.setdefault('id', uuid.uuid4().hex)
    position.setdefault('created_at', now_text)
    position['updated_at'] = now_text
    position['status'] = 'open'
    position.setdefault('stop_alert_active', False)
    position.setdefault('last_stop_alert_at', '')

    replaced = False
    for i, old in enumerate(positions):
        if str(old.get('code')) == code and old.get('status', 'open') == 'open':
            position['id'] = old.get('id', position['id'])
            position['created_at'] = old.get('created_at', position['created_at'])
            position['stop_alert_active'] = old.get('stop_alert_active', False)
            position['last_stop_alert_at'] = old.get('last_stop_alert_at', '')
            positions[i] = position
            replaced = True
            break
    if not replaced:
        positions.append(position)

    ok = save_positions(positions)
    return ok, ('已更新既有持倉' if replaced else '已建立持倉') if ok else '持倉檔案寫入失敗'


def update_position(position_id: str, changes: dict) -> bool:
    positions = load_positions()
    changed = False
    for p in positions:
        if p.get('id') == position_id:
            p.update(changes)
            p['updated_at'] = get_tw_now().isoformat(timespec='seconds')
            changed = True
            break
    return save_positions(positions) if changed else False


def close_position(position_id: str, exit_price: float, exit_date: str, exit_note: str = '') -> bool:
    positions = load_positions()
    changed = False
    for p in positions:
        if p.get('id') == position_id and p.get('status', 'open') == 'open':
            p['status'] = 'closed'
            p['exit_price'] = round(float(exit_price), 2)
            p['exit_date'] = exit_date
            p['exit_note'] = exit_note
            p['updated_at'] = get_tw_now().isoformat(timespec='seconds')
            entry = _safe_number(p.get('entry_price'))
            shares = _safe_number(p.get('shares'), 0)
            if pd.notna(entry):
                p['realized_pnl'] = round((float(exit_price) - entry) * shares, 0)
                p['realized_return_pct'] = round((float(exit_price) / entry - 1) * 100, 2) if entry > 0 else np.nan
            changed = True
            break
    return save_positions(positions) if changed else False


def delete_position(position_id: str) -> bool:
    positions = load_positions()
    kept = [p for p in positions if p.get('id') != position_id]
    return save_positions(kept) if len(kept) != len(positions) else False


def _load_local_telegram_config() -> dict:
    try:
        if os.path.exists(TELEGRAM_CONFIG_FILE):
            with open(TELEGRAM_CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def get_telegram_config() -> dict:
    """Secrets／環境變數優先，本機 JSON 作為方便測試的備援。"""
    local = _load_local_telegram_config()
    token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
    chat_id = os.getenv('TELEGRAM_CHAT_ID', '').strip()
    try:
        token = token or str(st.secrets['TELEGRAM_BOT_TOKEN']).strip()
    except Exception:
        pass
    try:
        chat_id = chat_id or str(st.secrets['TELEGRAM_CHAT_ID']).strip()
    except Exception:
        pass
    return {
        'bot_token': token or str(local.get('bot_token', '')).strip(),
        'chat_id': chat_id or str(local.get('chat_id', '')).strip(),
        'auto_check': bool(local.get('auto_check', False)),
        'interval_minutes': int(local.get('interval_minutes', 5) or 5),
    }


def save_telegram_config(bot_token: str, chat_id: str, auto_check: bool, interval_minutes: int) -> bool:
    payload = {
        'bot_token': str(bot_token).strip(),
        'chat_id': str(chat_id).strip(),
        'auto_check': bool(auto_check),
        'interval_minutes': int(interval_minutes),
    }
    try:
        with open(TELEGRAM_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def send_telegram_message(message: str, config=None):
    config = get_telegram_config() if config is None else config
    token = str(config.get('bot_token', '')).strip()
    chat_id = str(config.get('chat_id', '')).strip()
    if not token or not chat_id:
        return False, '尚未設定 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID'
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{token}/sendMessage',
            data={'chat_id': chat_id, 'text': message, 'disable_web_page_preview': True},
            timeout=12,
        )
        payload = r.json() if r.text else {}
        if r.status_code == 200 and payload.get('ok'):
            return True, 'Telegram 訊息已送出'
        description = payload.get('description') if isinstance(payload, dict) else r.text
        return False, f'Telegram 回覆錯誤：{description or r.status_code}'
    except Exception as exc:
        return False, f'Telegram 連線失敗：{exc}'


def refresh_portfolio_positions(send_alerts: bool = False) -> tuple[pd.DataFrame, list]:
    """更新未平倉部位的最新價格、損益、R 倍數，必要時發出停損通知。"""
    positions = load_positions()
    open_positions = [p for p in positions if p.get('status', 'open') == 'open']
    if not open_positions:
        return pd.DataFrame(), []

    tickers = tuple(dict.fromkeys(p.get('ticker', '') for p in open_positions if p.get('ticker')))
    history_map = download_batch_history(tickers) if tickers else {}
    telegram_config = get_telegram_config()
    alert_results = []
    file_changed = False
    rows = []

    for p in open_positions:
        tk = p.get('ticker', '')
        history = history_map.get(tk)
        current = np.nan
        quote_date = ''
        if history is not None and not history.empty:
            current = _safe_number(history['close'].iloc[-1])
            try:
                quote_date = pd.Timestamp(history.index[-1]).strftime('%Y-%m-%d')
            except Exception:
                quote_date = get_tw_now().strftime('%Y-%m-%d')

        entry = _safe_number(p.get('entry_price'))
        stop = _safe_number(p.get('stop_price'))
        target1 = _safe_number(p.get('target1'))
        target2 = _safe_number(p.get('target2'))
        shares = _safe_number(p.get('shares'), 0)
        pnl = (current - entry) * shares if pd.notna(current) and pd.notna(entry) else np.nan
        ret = (current / entry - 1) * 100 if pd.notna(current) and pd.notna(entry) and entry > 0 else np.nan
        original_risk = (entry - stop) * shares if pd.notna(entry) and pd.notna(stop) and entry > stop else np.nan
        r_multiple = pnl / original_risk if pd.notna(pnl) and pd.notna(original_risk) and original_risk > 0 else np.nan
        stop_triggered = bool(pd.notna(current) and pd.notna(stop) and current <= stop)

        if stop_triggered:
            state = '🔴 跌破停損'
        elif pd.notna(target2) and pd.notna(current) and current >= target2:
            state = '🏆 達第二目標'
        elif pd.notna(target1) and pd.notna(current) and current >= target1:
            state = '🟢 達第一目標'
        else:
            state = '持有中'

        # 僅在「第一次跌破」時通知；單純更新報價不會提前消耗通知狀態。
        # 價格重新站回停損後，下一次再跌破才重新通知。
        was_active = bool(p.get('stop_alert_active', False))
        if stop_triggered and not was_active and send_alerts and bool(p.get('telegram_alert', True)):
            message = (
                f"【停損警示】{p.get('name', '')} ({p.get('code', '')})\n\n"
                f"最新價格：{current:.2f}\n"
                f"停損價格：{stop:.2f}\n"
                f"進場成本：{entry:.2f}\n"
                f"持有股數：{shares:,.0f} 股\n"
                f"未實現損益：{pnl:+,.0f} 元 ({ret:+.2f}%)\n"
                f"目前 R 倍數：{r_multiple:+.2f}R\n"
                f"策略：{p.get('strategy', '未填寫')}\n"
                f"檢查時間：{get_tw_now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "價格已觸及或跌破原定停損，請依交易計畫處理，不要任意放寬停損。"
            )
            ok, msg = send_telegram_message(message, telegram_config)
            alert_results.append({'code': p.get('code'), 'ok': ok, 'message': msg})
            if ok:
                p['stop_alert_active'] = True
                p['last_stop_alert_at'] = get_tw_now().isoformat(timespec='seconds')
                file_changed = True
        elif not stop_triggered and was_active:
            p['stop_alert_active'] = False
            file_changed = True

        p['last_price'] = round(current, 2) if pd.notna(current) else None
        p['last_quote_date'] = quote_date
        file_changed = True
        rows.append({
            '代碼': p.get('code', ''), '名稱': p.get('name', ''),
            '進場日': p.get('entry_date', ''), '成本': entry, '股數': shares,
            '現價': current, '損益(元)': pnl, '報酬率(%)': ret,
            'R倍數': r_multiple, '停損': stop, '目標1': target1, '目標2': target2,
            '風險至停損(元)': max((current - stop) * shares, 0) if pd.notna(current) and pd.notna(stop) else np.nan,
            '狀態': state, '策略': p.get('strategy', ''), '報價日': quote_date,
            '_id': p.get('id'),
        })

    if file_changed:
        save_positions(positions)
    return pd.DataFrame(rows), alert_results


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

def normalize_stock_name(value):
    """上櫃 OpenAPI 有時只提供公司全名，這裡轉成較適合表格的股票簡稱。
    （模組層級版本，供市場觀察等在 get_stock_market_list 外部的程式碼使用；
    get_stock_market_list 內另有一份同名區域函式，行為一致。）"""
    name = str(value or '').strip().replace(' ', '')
    for suffix in ['股份有限公司', '有限公司']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_market_list():
    """快速載入台股清單。

    舊版卡在 get_stock_market_list() 的主因通常是 isin.twse.com.tw + pd.read_html
    解析整頁 HTML 太慢或連線被拖住。這版改成：
    1) 先讀本機每日快取，幾乎秒開。
    2) 再抓 TWSE / TPEx OpenAPI JSON，避免 read_html 大表格解析。
    3) OpenAPI 失敗時才用舊 ISIN HTML 備援，且 timeout 較短。
    """
    cache_file = os.path.join(os.path.dirname(__file__) if '__file__' in globals() else '.', 'stock_market_cache_v3_industry_quality.json')
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

    def industry_quality_ok(data):
        """避免把只有代碼／名稱、產業全部未分類的行情備援寫入每日快取。"""
        if not isinstance(data, list) or len(data) <= 100:
            return False
        for market_name in ('上市', '上櫃'):
            subset = [x for x in data if isinstance(x, dict) and x.get('市場別') == market_name]
            if len(subset) < 50:
                return False
            valid = sum(
                1 for x in subset
                if normalize_industry(x.get('industry', '未分類'))
                not in ('', '未分類', 'None', 'nan')
            )
            # 公司基本資料正常時，絕大多數股票都有產業別；低於 45% 視為污染快取。
            if valid / max(len(subset), 1) < 0.45:
                return False
        return True

    def save_local_cache(data):
        try:
            if industry_quality_ok(data):
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({'date': today, 'data': data}, f, ensure_ascii=False)
        except Exception:
            pass

    cached = load_local_cache(allow_stale=False)
    if cached and industry_quality_ok(cached):
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
                # 若先讀到每日行情（產業未分類），後續公司基本資料可補回產業別。
                if key not in seen:
                    stocks.append(s)
                    seen.add(key)
                else:
                    current = next((x for x in stocks if x.get('ticker') == key), None)
                    if current is not None:
                        old_industry = normalize_industry(current.get('industry', '未分類'))
                        new_industry = normalize_industry(s.get('industry', '未分類'))
                        if old_industry == '未分類' and new_industry != '未分類':
                            current['industry'] = new_industry
                        if not current.get('name') and s.get('name'):
                            current['name'] = s['name']
            if len(stocks) >= 1200:
                # 已足夠涵蓋上市櫃，多半不用再跑慢速備援。
                pass
        except Exception:
            continue

    if len(stocks) > 500 and industry_quality_ok(stocks):
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
                        isin_industry = normalize_industry(row.get('產業別', '未分類'))
                        if key in seen:
                            current = next((x for x in stocks if x.get('ticker') == key), None)
                            if current is not None:
                                if normalize_industry(current.get('industry', '未分類')) == '未分類' and isin_industry != '未分類':
                                    current['industry'] = isin_industry
                                if not current.get('name'):
                                    current['name'] = normalize_stock_name(name)
                            continue
                        stocks.append({
                            'ticker': key,
                            'name': normalize_stock_name(name),
                            'industry': isin_industry,
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


# ============================================================
# [新功能] 個股三大法人買賣情況
# ============================================================
# 台灣證交所／櫃買中心官方 T86 開放資料只提供「最新一個交易日、全市場」的
# 三大法人買賣超，沒有現成的「單一個股、近期逐日」歷史 API 可直接查詢；
# 因此改用 Yahoo 股市個股「法人買賣」頁面一次性取得近期逐日資料，
# 跟既有 fetch_quarterly_eps() 走相同的「單頁解析＋快取＋抓不到就回空」寫法，
# 抓不到時 UI 端顯示「暫無資料」，不影響其他功能（fail-open）。

def _extract_institutional_daily_from_html(html: str) -> list:
    """從 Yahoo 台股「法人買賣」頁面解析近期逐日三大法人買賣超（張）。
    頁面格式如：2026/05/12 35,287 9,427 5,199 49,912 36.98% 10.00% 379,739
    （日期／外資／投信／自營商／合計／外資籌碼％／漲跌幅％／成交量）。
    僅作輔助參考，非官方逐筆對帳資料，解析不到時回傳空 list。
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    pattern = r"(20\d{2}/\d{2}/\d{2})\s+(-?[\d,]+)\s+(-?[\d,]+)\s+(-?[\d,]+)\s+(-?[\d,]+)\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+([\d,]+)"
    rows = []
    seen = set()
    for m in re.finditer(pattern, text):
        date_s, foreign, trust, dealer, total, foreign_pct, chg_pct, vol = m.groups()
        if date_s in seen:
            continue
        seen.add(date_s)
        f_val = _to_float_or_nan(foreign)
        t_val = _to_float_or_nan(trust)
        d_val = _to_float_or_nan(dealer)
        if pd.isna(f_val) and pd.isna(t_val) and pd.isna(d_val):
            continue
        rows.append({
            "日期": date_s.replace("/", "-"),
            "外資": f_val,
            "投信": t_val,
            "自營商": d_val,
            "合計": _to_float_or_nan(total),
            "外資籌碼(%)": _to_float_or_nan(foreign_pct),
            "漲跌幅(%)": _to_float_or_nan(chg_pct),
            "成交量": _to_float_or_nan(vol),
        })
        if len(rows) >= 60:  # 近 60 個交易日（約 3 個月），足夠看趨勢又不會抓過量雜訊
            break
    return rows


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_institutional_trading(code: str, market_suffix: str) -> list:
    """個股三大法人（外資／投信／自營商）近期逐日買賣超（張），最新一筆在最前面。
    來源：Yahoo 股市個股「法人買賣」頁面。交易日盤後才會更新，快取 1 小時。
    """
    ticker = f"{code}.{market_suffix}"
    try:
        r = requests.get(f"https://tw.stock.yahoo.com/quote/{ticker}/institutional-trading",
                         headers=get_headers(), timeout=8, verify=False)
        if r.status_code == 200 and r.text:
            rows = _extract_institutional_daily_from_html(r.text)
            if rows:
                return rows
    except Exception:
        pass
    return []


def _institutional_streak(rows: list, field: str) -> dict:
    """從逐日序列（rows[0] 為最新一日）計算目前連買／連賣天數與同方向累計張數。
    直接用序列本身往回累加推算，不額外解析頁面上的「連N買」文字，
    比較不受頁面版型調整影響。
    """
    vals = [r.get(field) for r in rows if pd.notna(r.get(field))]
    if not vals:
        return {"days": 0, "total": 0.0, "sign": 0}
    sign = 1 if vals[0] > 0 else (-1 if vals[0] < 0 else 0)
    if sign == 0:
        return {"days": 0, "total": 0.0, "sign": 0}
    days, total = 0, 0.0
    for v in vals:
        s = 1 if v > 0 else (-1 if v < 0 else 0)
        if s != sign:
            break
        days += 1
        total += v
    return {"days": days, "total": total, "sign": sign}


# ============================================================
# [新功能] 個股法人（外資／券商）目標價
# ============================================================
# 主來源：Anue鉅亨網「外資評等」公開表格（券商逐筆歷史）。
# 備援來源：Yahoo Finance 分析師共識（targetMean/High/Low + 評等），
#          Anue 抓不到或為空時自動啟用（fail-open）。
# 券商報告本身多為付費資訊，沒有官方免費完整 API。

def _clean_rating_txt(v, default="--"):
    """把 pandas NaN 安全轉成預設字串，避免 str(np.nan) 變成字面上的 'nan'。"""
    if pd.isna(v):
        return default
    s = str(v).strip()
    return s if s else default


def _fetch_analyst_from_anue(code: str) -> list:
    """Anue 鉅亨「外資評等」表格。成功回傳列，失敗回傳 []。"""
    url = f"https://www.cnyes.com/twstock/foreignrating.aspx?code={code}"
    try:
        r = requests.get(url, headers=get_headers(), timeout=10, verify=False)
        if r.status_code != 200 or not r.text:
            return []
        dfs = pd.read_html(StringIO(r.text))
    except Exception:
        return []

    target_df = None
    for df in dfs:
        cols = [str(c) for c in df.columns]
        if any("評等日期" in c for c in cols) and any("目標價" in c for c in cols):
            target_df = df
            break
    if target_df is None:
        return []

    target_df = target_df.rename(columns=lambda c: str(c).strip())
    rows = []
    for _, row in target_df.iterrows():
        date_raw = re.sub(r"\D", "", str(row.get("評等日期", "")))
        date_txt = (f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
                    if len(date_raw) == 8 else str(row.get("評等日期", "")).strip())
        target_price = _to_float_or_nan(row.get("目標價"))
        if pd.isna(target_price) or target_price <= 0:
            continue
        # Anue 表格的「現價」是頁面當下報價、每一列都相同，
        # 並非該評等日期的歷史收盤，若當成「當時股價」會誤導，故不採用。
        rows.append({
            "評等日期": date_txt,
            "券商": _clean_rating_txt(row.get("券商"), "N/A"),
            "新評等": _clean_rating_txt(row.get("新評等")),
            "升降": _clean_rating_txt(row.get("升/降")),
            "財測EPS": _clean_rating_txt(row.get("財測EPS(年度)")),
            "目標價": target_price,
            "現價": np.nan,
            "來源": "Anue",
        })
        if len(rows) >= 30:
            break
    return rows


def _fetch_analyst_from_yahoo(code: str, market_suffix: str = "TW") -> list:
    """Yahoo Finance 分析師共識目標價（備援）。
    只有共識均價／高低價與評等，沒有逐家券商歷史；成功時合成 1～3 筆列。
    """
    ticker = f"{code}.{market_suffix}"
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return []

    mean_t = _to_float_or_nan(info.get("targetMeanPrice"))
    high_t = _to_float_or_nan(info.get("targetHighPrice"))
    low_t = _to_float_or_nan(info.get("targetLowPrice"))
    cur = _to_float_or_nan(info.get("currentPrice") or info.get("regularMarketPrice"))
    rec = _clean_rating_txt(info.get("recommendationKey"), "--")
    rec_map = {
        "strong_buy": "強力買進", "buy": "買進", "hold": "中立",
        "underperform": "表現落後", "sell": "賣出", "none": "--",
    }
    rec_zh = rec_map.get(str(rec).lower(), rec if rec != "--" else "--")
    n_analysts = info.get("numberOfAnalystOpinions")
    n_txt = f"{int(n_analysts)} 家" if pd.notna(n_analysts) and n_analysts else "共識"

    if pd.isna(mean_t) or mean_t <= 0:
        return []

    today = get_tw_now().strftime("%Y-%m-%d")
    rows = [{
        "評等日期": today,
        "券商": f"Yahoo共識（{n_txt}）",
        "新評等": rec_zh,
        "升降": "--",
        "財測EPS": "--",
        "目標價": float(mean_t),
        "現價": cur,
        "來源": "Yahoo",
    }]
    if pd.notna(high_t) and high_t > 0 and abs(high_t - mean_t) / mean_t > 0.01:
        rows.append({
            "評等日期": today, "券商": "Yahoo共識高價", "新評等": rec_zh,
            "升降": "--", "財測EPS": "--", "目標價": float(high_t),
            "現價": cur, "來源": "Yahoo",
        })
    if pd.notna(low_t) and low_t > 0 and abs(low_t - mean_t) / mean_t > 0.01:
        rows.append({
            "評等日期": today, "券商": "Yahoo共識低價", "新評等": rec_zh,
            "升降": "--", "財測EPS": "--", "目標價": float(low_t),
            "現價": cur, "來源": "Yahoo",
        })
    return rows


@st.cache_data(ttl=21600, show_spinner=False)  # 評等不會逐分鐘變動，快取 6 小時
def fetch_analyst_target_price(code: str, market_suffix: str = "TW") -> list:
    """個股法人目標價：主來源 Anue，失敗或空白時自動改抓 Yahoo 共識。
    回傳列（最新在前）；每列含「來源」欄位（Anue / Yahoo）。
    """
    rows = _fetch_analyst_from_anue(str(code))
    if rows:
        return rows
    return _fetch_analyst_from_yahoo(str(code), market_suffix or "TW")


def summarize_target_price(rows: list, current_price: float) -> dict:
    """彙整法人目標價：近期（最近10筆評等）平均／最高／最低目標價，
    以及平均目標價相對目前股價的潛在漲跌幅。回傳 {} 代表資料不足。
    """
    if not rows:
        return {}
    prices = [r["目標價"] for r in rows if pd.notna(r["目標價"])]
    if not prices:
        return {}
    recent = prices[:10]
    avg_target = float(np.mean(recent))
    upside = np.nan
    if pd.notna(current_price) and current_price > 0:
        upside = (avg_target - current_price) / current_price * 100
    latest = rows[0]
    sources = sorted({str(r.get("來源", "Anue")) for r in rows})
    return {
        "latest_target": latest["目標價"],
        "latest_broker": latest["券商"],
        "latest_rating": latest["新評等"],
        "latest_date": latest["評等日期"],
        "avg_target": avg_target,
        "max_target": float(np.max(recent)),
        "min_target": float(np.min(recent)),
        "upside_pct": upside,
        "n_ratings": len(recent),
        "source": " / ".join(sources),
    }


# ============================================================
# [新功能] 個股股利政策／除權息
# ============================================================
# Yahoo 股市個股「股利」頁面是公開網頁、不需登入，會列出歷年現金股利／股票股利／
# 殖利率／除息日等資料，但跟法人買賣、EPS 頁面一樣是用 div 排版而非標準 <table>
# （pd.read_html 抓不到），所以延用本檔案既有的「get_text 攤平＋規則解析」手法。
# 頁面每一列固定 11 個欄位（發放期間／所屬期間／現金股利／股票股利／現金殖利率／
# 除息日昨收價／除息日／除權日／現金股利發放日／股票股利發放日／填息天數），
# 用「發放期間(西元年或-) + 所屬期間(西元年)」這組固定出現的兩個 token 當作
# 每一列的錨點，取「這個錨點到下一個錨點之間」當作這一列的欄位範圍（詳見
# _extract_dividend_rows_from_html 內的說明）。頁面上的殖利率欄位在不少個股
# 常常是空值、參考價值有限，所以只解析出來做內部驗證，不對外顯示。

def _extract_dividend_rows_from_html(html: str) -> list:
    """從 Yahoo 台股「股利」頁面解析歷年股利政策列表。
    僅作輔助參考資料，非官方股利公告，解析不到時回傳空 list。
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    marker = "歷年股利政策"
    idx = text.find(marker)
    if idx == -1:
        return []
    text = text[idx:]
    tokens = text.split(" ")

    # [修正] 先把所有列的起點錨點（發放期間 + 所屬期間）位置都找出來，
    # 用「這個錨點到下一個錨點之間」當作這一列的欄位範圍（動態寬度），
    # 而不是無條件固定往後抓 9 個 token。原本固定抓 9 個的寫法，
    # 一旦某一列實際欄位數跟預期不同（例如當年度股利尚未公告、
    # 頁面版型微調等情況），會把下一列的資料錯誤地吃進這一列，
    # 導致後面看到「同一年度出現兩筆、其中一筆全部空白」的錯位畫面。
    #
    # 錨點偵測還有一個容易忽略的陷阱：某一列自己的「填息天數＝-」剛好緊接在
    # 下一列開頭的真實年份前面時（例如 "... - 2025 2024 ..."），會被誤判成
    # 另一個「發放期間=- , 所屬期間=2025」的錨點，跟真正的錨點重疊巢狀。
    # 因此限制「發放期間＝-」只允許出現在整張表最前面那一列（尚未公告的
    # 預告股利，通常排在最新一列），中間所有列都必須是「YYYY YYYY」的
    # 完整年份配對；找到錨點後直接跳過這兩個 token 再繼續掃描，避免同一組
    # token 被重複解讀成不同錨點。
    this_year = get_tw_now().year
    anchor_positions = []
    i = 0
    found_first = False
    while i < len(tokens) - 1:
        tok, nxt = tokens[i], tokens[i + 1]
        is_year_pair = bool(re.fullmatch(r"\d{4}", tok)) and bool(re.fullmatch(r"\d{4}", nxt))
        is_predicted_pair = (not found_first) and tok == "-" and bool(re.fullmatch(r"\d{4}", nxt))
        if is_year_pair or is_predicted_pair:
            year_val = int(nxt)
            if 1960 <= year_val <= this_year + 1:  # 所屬期間要落在合理年份範圍，避免誤判
                anchor_positions.append(i)
                found_first = True
                i += 2
                continue
        i += 1

    rows = []
    for k, start in enumerate(anchor_positions):
        end = anchor_positions[k + 1] if k + 1 < len(anchor_positions) else min(start + 11, len(tokens))
        remainder = tokens[start + 2:end]
        if len(remainder) != 9:
            continue  # 欄位數對不上預期，寧可跳過這一列，也不要顯示錯位資料
        cash_div, stock_div, yield_pct, prev_close, ex_div_date, ex_right_date, cash_pay_date, stock_pay_date, fill_days = remainder
        ex_div_txt = ex_div_date if re.fullmatch(r"\d{4}/\d{2}/\d{2}", ex_div_date) else ("尚未公布" if ex_div_date == "尚未公布" else "-")
        rows.append({
            "所屬期間": tokens[start + 1],
            "現金股利": _to_float_or_nan(cash_div),
            "股票股利": _to_float_or_nan(stock_div),
            "除息日": ex_div_txt,
            "填息天數": _to_float_or_nan(fill_days),
        })
        if len(rows) >= 25:  # 最多近 25 個年度，避免頁面雜訊灌太多筆
            break

    # [修正] 保險起見再去重：同一個所屬年度理論上只會出現一次；如果因為解析誤差
    # 仍出現重複年度，保留欄位較完整（非空值數量較多）的那一筆，避免同時顯示
    # 兩筆同年度但互相矛盾的資料。dict 在 Python 3.7+ 會保留原本的插入順序，
    # 所以就算後面用更完整的資料覆蓋掉，列表原本「最新年度在前」的順序不會跑掉。
    def _filled_count(row: dict) -> int:
        n = 0
        for v in row.values():
            if v is None:
                continue
            if isinstance(v, float) and pd.isna(v):
                continue
            if v == "-":
                continue
            n += 1
        return n

    dedup = {}
    for r in rows:
        key = r["所屬期間"]
        if key not in dedup or _filled_count(r) > _filled_count(dedup[key]):
            dedup[key] = r
    return list(dedup.values())


def _extract_dividend_summary_from_html(html: str) -> dict:
    """解析頁面上「已連N年發放股利，合計X元。」這句摘要，
    直接用 Yahoo 自己算好的數字，比自己從逐年列表重算更準確、也更省事。
    抓不到就回傳 {}，UI 端改用逐年列表自行彙總當備援。
    """
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    m = re.search(r"已連\s*(\d+)\s*年發放股利，?合計\s*([\d.]+)\s*元", text)
    if not m:
        return {}
    years, total = m.groups()
    return {
        "連續配息年數": int(years),
        "歷年合計股利": _to_float_or_nan(total),
    }


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_dividend_history(code: str, market_suffix: str) -> dict:
    """個股歷年股利政策（現金股利／股票股利／除息日）與摘要統計。
    來源：Yahoo 股市個股「股利」頁面。股利政策一年頂多變動幾次，快取拉到 24 小時。
    回傳 {"rows": [...], "summary": {...}}，兩者皆可能是空值（fail-open）。
    """
    ticker = f"{code}.{market_suffix}"
    try:
        r = requests.get(f"https://tw.stock.yahoo.com/quote/{ticker}/dividend",
                         headers=get_headers(), timeout=8, verify=False)
        if r.status_code == 200 and r.text:
            rows = _extract_dividend_rows_from_html(r.text)
            summary = _extract_dividend_summary_from_html(r.text)
            if rows or summary:
                return {"rows": rows, "summary": summary}
    except Exception:
        pass
    return {"rows": [], "summary": {}}


# ============================================================
# [新功能] 個股／市場重要行事曆（法說會、除權息、股東會）
# ============================================================
# 個股：Yahoo 個股行事曆頁面
# 市場：Yahoo 台股行事曆（法說會 / 除權息 專頁）
# 抓不到時回傳空 list，UI 端顯示「暫無資料」，不影響其他功能（fail-open）。

def _extract_stock_calendar_from_html(html: str) -> list:
    """從 Yahoo 台股個股「行事曆」頁面解析近期重要事件。"""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    this_year = get_tw_now().year
    year_candidates = [this_year, this_year + 1, this_year - 1]
    pattern = r"(\d{2})/(\d{2})\s*週[一二三四五六日]\s*(法說會|除權息|股東會|董事會|停券|全額交割|暫停交易)"
    rows = []
    seen = set()
    for m in re.finditer(pattern, text):
        mm, dd, event = m.groups()
        best_date = None
        for y in year_candidates:
            try:
                d = datetime(y, int(mm), int(dd)).date()
                delta = (d - get_tw_now().date()).days
                if -60 <= delta <= 180:
                    if best_date is None or abs(delta) < abs((best_date - get_tw_now().date()).days):
                        best_date = d
            except ValueError:
                continue
        if best_date is None:
            continue
        key = (best_date.isoformat(), event)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "日期": best_date.strftime("%Y-%m-%d"),
            "事件": event,
            "距離天數": (best_date - get_tw_now().date()).days,
        })
        if len(rows) >= 30:
            break
    rows.sort(key=lambda x: x["日期"])
    return rows


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_stock_calendar(code: str, market_suffix: str) -> list:
    """個股重要行事曆（法說會、除權息、股東會等）。"""
    ticker = f"{code}.{market_suffix}"
    try:
        r = requests.get(f"https://tw.stock.yahoo.com/quote/{ticker}/calendar",
                         headers=get_headers(), timeout=8, verify=False)
        if r.status_code == 200 and r.text:
            rows = _extract_stock_calendar_from_html(r.text)
            if rows:
                return rows
    except Exception:
        pass
    return []


def get_next_earnings_related_event(calendar_rows: list) -> dict:
    """優先找未來法說會，其次股東會、除權息。"""
    if not calendar_rows:
        return {}
    for row in calendar_rows:
        if row.get("距離天數", -999) >= 0 and row.get("事件") == "法說會":
            return row
    for row in calendar_rows:
        if row.get("距離天數", -999) >= 0 and row.get("事件") == "股東會":
            return row
    for row in calendar_rows:
        if row.get("距離天數", -999) >= 0 and row.get("事件") == "除權息":
            return row
    return {}


def _extract_market_earnings_call_from_html(html: str) -> list:
    """從 Yahoo 台股「法說會」行事曆頁面解析近期法說會。"""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    pattern = r"([^\s]{2,12})\s+(\d{4}(?:\.TW|\.TWO)?)\s+(20\d{2}/\d{2}/\d{2})\s+(\d{1,2}:\d{2})"
    rows = []
    seen = set()
    for m in re.finditer(pattern, text):
        name, code_raw, date_s, time_s = m.groups()
        code = re.sub(r"\.(TW|TWO)$", "", code_raw)
        if not re.match(r"^\d{4}", code):
            continue
        key = (date_s, code)
        if key in seen:
            continue
        seen.add(key)
        try:
            d = datetime.strptime(date_s, "%Y/%m/%d").date()
            days = (d - get_tw_now().date()).days
        except Exception:
            continue
        if days < -1 or days > 60:
            continue
        rows.append({
            "日期": d.strftime("%Y-%m-%d"),
            "時間": time_s,
            "代碼": code,
            "名稱": name.strip(),
            "事件": "法說會",
            "距離天數": days,
        })
        if len(rows) >= 80:
            break
    rows.sort(key=lambda x: (x["日期"], x["時間"]))
    return rows


def _extract_market_dividend_from_html(html: str) -> list:
    """從 Yahoo 台股「除權息」行事曆頁面解析近期除權息。"""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    pattern = r"([^\s]{2,16})\s+(\d{4}[A-Z]?(?:\.TW|\.TWO)?)\s+(20\d{2}/\d{2}/\d{2})\s+(20\d{2}/\d{2}/\d{2}|—|-)\s+([\d.]+)"
    rows = []
    seen = set()
    for m in re.finditer(pattern, text):
        name, code_raw, ex_date, pay_date, cash = m.groups()
        code = re.sub(r"\.(TW|TWO)$", "", code_raw)
        if not re.match(r"^\d{4}", code):
            continue
        key = (ex_date, code)
        if key in seen:
            continue
        seen.add(key)
        try:
            d = datetime.strptime(ex_date, "%Y/%m/%d").date()
            days = (d - get_tw_now().date()).days
        except Exception:
            continue
        if days < -3 or days > 45:
            continue
        rows.append({
            "日期": d.strftime("%Y-%m-%d"),
            "代碼": code,
            "名稱": name.strip(),
            "事件": "除權息",
            "現金股利": _to_float_or_nan(cash),
            "距離天數": days,
        })
        if len(rows) >= 100:
            break
    rows.sort(key=lambda x: x["日期"])
    return rows


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_market_calendar_events() -> dict:
    """全市場近期重要事件：法說會 + 除權息。
    回傳 {"法說會": [...], "除權息": [...]}。
    """
    result = {"法說會": [], "除權息": []}
    try:
        r = requests.get("https://tw.stock.yahoo.com/calendar/earnings-call",
                         headers=get_headers(), timeout=10, verify=False)
        if r.status_code == 200 and r.text:
            result["法說會"] = _extract_market_earnings_call_from_html(r.text)
    except Exception:
        pass
    try:
        r = requests.get("https://tw.stock.yahoo.com/calendar/dividend",
                         headers=get_headers(), timeout=10, verify=False)
        if r.status_code == 200 and r.text:
            result["除權息"] = _extract_market_dividend_from_html(r.text)
    except Exception:
        pass
    return result


# ============================================================
# [新功能] 個股融資融券／資券變化
# ============================================================
# 跟三大法人一樣，官方 TWSE/TPEx 開放資料只提供「最新一天、全市場」的融資融券，
# 沒有「單一個股、近期逐日」的現成 API；改用 Yahoo 股市個股「資券變化」頁面，
# 走跟 institutional-trading 完全相同的「單頁解析＋規則正則＋抓不到就回空」寫法。

def _extract_margin_daily_from_html(html: str) -> list:
    """從 Yahoo 台股「資券變化」頁面解析近期逐日融資融券增減與餘額。
    頁面格式如：2026/04/17 12,805 144,081 4.58% 621 2,001 0.06% 1.39% 522
    （日期／融資增減／融資餘額／融資使用率％／融券增減／融券餘額／
    融券使用率％／券資比％／資券互抵，單位：張）。
    僅作輔助參考，非官方逐筆對帳資料，解析不到時回傳空 list。
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    pattern = r"(20\d{2}/\d{2}/\d{2})\s+(-?[\d,]+)\s+([\d,]+)\s+([\d.]+)%\s+(-?[\d,]+)\s+([\d,]+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d,]+)"
    rows = []
    seen = set()
    for m in re.finditer(pattern, text):
        date_s, m_chg, m_bal, m_use, s_chg, s_bal, s_use, ratio, offset = m.groups()
        if date_s in seen:
            continue
        seen.add(date_s)
        rows.append({
            "日期": date_s.replace("/", "-"),
            "融資增減": _to_float_or_nan(m_chg),
            "融資餘額": _to_float_or_nan(m_bal),
            "融資使用率(%)": _to_float_or_nan(m_use),
            "融券增減": _to_float_or_nan(s_chg),
            "融券餘額": _to_float_or_nan(s_bal),
            "融券使用率(%)": _to_float_or_nan(s_use),
            "券資比(%)": _to_float_or_nan(ratio),
            "資券互抵": _to_float_or_nan(offset),
        })
        if len(rows) >= 60:  # 近 60 個交易日（約 3 個月），足夠看趨勢又不會抓過量雜訊
            break
    return rows


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_margin_trading(code: str, market_suffix: str) -> list:
    """個股融資融券近期逐日增減與餘額，最新一筆在最前面。
    來源：Yahoo 股市個股「資券變化」頁面。交易日盤後才會更新，快取 1 小時。
    """
    ticker = f"{code}.{market_suffix}"
    try:
        r = requests.get(f"https://tw.stock.yahoo.com/quote/{ticker}/margin",
                         headers=get_headers(), timeout=8, verify=False)
        if r.status_code == 200 and r.text:
            rows = _extract_margin_daily_from_html(r.text)
            if rows:
                return rows
    except Exception:
        pass
    return []


# ============================================================
# [新功能] 個股財務體質評分
# ============================================================
# TWSE/TPEx 官方財報開放資料（資產負債表／損益表）依產業別分成好幾種格式
# （一般業／金融業／證券期貨業／金控業／保險業…欄位都不一樣），直接處理
# 官方原始財報格式很容易因為股票產業別不同而解析失敗；改用 Anue鉅亨網
# 「年度財務比率」頁面，該頁面已經把負債比率、流動比率、ROE、ROA 等
# 常用比率算好，且用的是標準 HTML <table>（不像法人買賣、股利頁面是
# div 排版），可以直接用 pd.read_html 解析，公開網頁不需登入。

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_financial_ratios(code: str) -> dict:
    """個股年度財務比率（負債比率、流動比率、ROE、ROA、獲利能力等）。
    來源：Anue鉅亨網「年度財務比率」頁面。回傳 {"year": "2025年", "metrics": {...}}，
    抓不到時回傳 {}（fail-open）。
    """
    url = f"https://www.cnyes.com/twstock/finratio2.aspx?code={code}"
    try:
        r = requests.get(url, headers=get_headers(), timeout=10, verify=False)
        if r.status_code != 200 or not r.text:
            return {}
        dfs = pd.read_html(StringIO(r.text))
    except Exception:
        return {}

    ratio_df = None
    for df in dfs:
        cols = [str(c) for c in df.columns]
        if any("項目" in c for c in cols) and any("名稱" in c for c in cols):
            ratio_df = df
            break
    if ratio_df is None:
        return {}

    ratio_df = ratio_df.rename(columns=lambda c: str(c).strip())
    # 年度欄位的排序在不同個股頁面並不一致（有由新到舊、也有由舊到新），
    # 不能直接取第一欄，要比大小挑出最新年度。
    year_cols = [c for c in ratio_df.columns if re.fullmatch(r"\d{4}年", c)]
    if not year_cols:
        return {}
    latest_col = max(year_cols, key=lambda c: int(c[:4]))

    metrics = {}
    for _, row in ratio_df.iterrows():
        name = str(row.get("名稱", "")).strip()
        if not name or name == "nan":
            continue
        val = _to_float_or_nan(row.get(latest_col))
        if pd.notna(val):
            metrics[name] = val
    if not metrics:
        return {}
    return {"year": latest_col, "metrics": metrics}


def _score_linear(value: float, lo: float, hi: float) -> float:
    """把數值線性映射到 0~100 分。lo 對應 0 分，hi 對應 100 分；
    lo > hi 時代表「數值越低越好」的指標（例如負債比率），會自動反向計算。
    結果一律限制在 0~100 之間，不會因為極端值爆出超出範圍的分數。
    """
    if pd.isna(value):
        return np.nan
    if lo == hi:
        return 50.0
    pct = (value - lo) / (hi - lo)
    return float(np.clip(pct * 100, 0, 100))


# 財務體質評分採用的指標：(對應財報項目名稱, 分類, 0分對應值, 100分對應值)。
# 門檻值是參考台股一般產業常見水準訂出的粗略區間，不同產業（例如重資產的
# 航運、金融）合理區間本來就不同，這裡的評分僅供快速篩選參考，不是嚴謹的
# 產業調整後估值模型。
FINANCIAL_HEALTH_METRICS = {
    "負債占資產比率": ("負債占資產比率", "財務結構", 80, 20),
    "流動比率":       ("流動比率", "償債能力", 80, 250),
    "速動比率":       ("速動比率", "償債能力", 40, 150),
    "營業利益率":     ("營業利益率", "獲利能力", 0, 20),
    "純益率":         ("純益率", "獲利能力", 0, 15),
    "總資產報酬率":   ("總資產報酬率", "獲利能力", 0, 10),
    "股東權益報酬率": ("股東權益報酬率", "獲利能力", 0, 20),
}


def calc_financial_health_score(metrics: dict) -> dict:
    """把財報比率轉成 0~100 的財務體質綜合評分，並列出各分類子分數。
    只用抓得到的指標計算，缺個一兩項不影響整體評分（fail-open）；
    完全抓不到指標時回傳 {}，由 UI 端顯示提示。
    """
    if not metrics:
        return {}

    items = []
    for label, (field, category, lo, hi) in FINANCIAL_HEALTH_METRICS.items():
        raw = metrics.get(field)
        if raw is None or pd.isna(raw):
            continue
        score = _score_linear(raw, lo, hi)
        items.append({"指標": label, "分類": category, "原始值": raw, "分數": score})

    if not items:
        return {}

    overall = float(np.mean([it["分數"] for it in items]))
    if overall >= 75:
        verdict, color = "體質優良", "var(--green)"
    elif overall >= 55:
        verdict, color = "體質穩健", "#8eb6ff"
    elif overall >= 35:
        verdict, color = "普通", "var(--yellow)"
    else:
        verdict, color = "體質偏弱", "var(--red)"

    by_category = {}
    for it in items:
        by_category.setdefault(it["分類"], []).append(it["分數"])
    category_scores = {cat: float(np.mean(scores)) for cat, scores in by_category.items()}

    return {
        "overall": overall, "verdict": verdict, "color": color,
        "items": items, "category_scores": category_scores,
        "n_metrics": len(items),
    }


def fetch_financial_health_score(code: str) -> float:
    """抓取個股財務比率並算出財務體質綜合評分（0~100）。
    取代原本清單／候選表格用的技術面 AI 評分：候選清單、快速篩選、
    統計卡片、加入追蹤紀錄現在都改用這個分數。抓不到資料或指標不足時
    回傳 NaN，呼叫端一律要能處理 NaN（fail-open，不讓抓取失敗中斷掃描）。
    """
    try:
        ratios = fetch_financial_ratios(code)
        metrics = ratios.get("metrics", {}) if ratios else {}
        health = calc_financial_health_score(metrics)
        return health.get("overall", np.nan) if health else np.nan
    except Exception:
        return np.nan


# ============================================================
# [新功能] 個股本益比河流圖
# ============================================================
# Yahoo 股市官方「本益比河流圖」頁面需要登入 VIP 帳號才能看到實際圖表資料，
# 沒有公開、免登入的資料來源可直接取用；因此改成自己還原：
# 用既有的季度 EPS（fetch_quarterly_eps）滾動加總近四季得到 TTM EPS，
# 乘上「這檔股票自己歷史本益比分布」的幾個分位數，得到隨 EPS 成長而變動的
# 理論價位帶，畫成河流狀的區域圖，再把實際股價疊在上面。
# 這是「相對自己過去本益比區間」的估算，不是官方資料、也不是目標價，僅供參考。

# ============================================================
# [市場觀察] 官方 OpenAPI：產業類股表現 ＋ 個股漲跌分布
# ============================================================
# 不再爬取玩股網 HTML。玩股網頁面採動態渲染／可能有反爬機制，requests 取得的
# HTML 常常沒有實際表格，因此會誤判成「沒有資料」。本區改用 TWSE／TPEx 官方
# OpenAPI 的最新收盤行情，再以公司基本資料的產業別於本地彙總。
#
# 另外加入磁碟快取：休市日或官方 API 暫時連線失敗時，顯示最近一次成功資料，
# 不會整個頁籤只剩警告訊息。

MARKET_PULSE_CACHE_FILE = os.path.join(
    os.path.dirname(__file__) if '__file__' in globals() else '.',
    'market_pulse_cache_v5.json',
)
MARKET_PROFILE_CACHE_FILE = os.path.join(
    os.path.dirname(__file__) if '__file__' in globals() else '.',
    'market_profile_cache_v3.json',
)

_MP_INDUSTRY_CODE_MAP = {
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


def _mp_to_float(v):
    """把行情欄位安全轉為 float，容許千分位、百分號、箭頭與 HTML 標籤。"""
    if v is None:
        return np.nan
    text = BeautifulSoup(str(v), "html.parser").get_text(" ", strip=True)
    text = (text.replace(",", "").replace("%", "").replace("＋", "+")
                .replace("－", "-").replace("−", "-").strip())
    if text in ("", "--", "---", "-", "X", "N/A", "null", "None", "nan"):
        return np.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    try:
        number = float(match.group(0))
        if any(mark in text for mark in ("▼", "▽", "跌")) and number > 0:
            number = -number
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan


def _mp_pick(row: dict, names, default=""):
    """從不同版本的官方欄位名稱中取第一個有效值。"""
    if not isinstance(row, dict):
        return default
    for name in names:
        if name in row:
            value = row.get(name)
            if value not in (None, "", "null", "None"):
                return value
    return default


def _mp_clean_code(value) -> str:
    """只取 4 碼股票代號；排除權證、債券等非一般股票商品。"""
    match = re.search(r"(?<!\d)(\d{4})(?!\d)", str(value or ""))
    return match.group(1) if match else ""


def _mp_normalize_industry(value) -> str:
    text = BeautifulSoup(str(value or ""), "html.parser").get_text(" ", strip=True)
    text = text.replace("　", " ").strip()
    if text in ("", "-", "--", "None", "null", "nan", "未分類"):
        return "未分類"
    code = text.zfill(2) if text.isdigit() and len(text) <= 2 else text
    if code in _MP_INDUSTRY_CODE_MAP:
        return _MP_INDUSTRY_CODE_MAP[code]
    match = re.match(r"^(\d{1,2})\s*[、,，\-:：]?\s*(.*)$", text)
    if match:
        industry_code = match.group(1).zfill(2)
        label = match.group(2).strip()
        if label and not label.isdigit():
            return label
        return _MP_INDUSTRY_CODE_MAP.get(industry_code, "未分類")
    # 官方資料有時使用「其他」而非「其他業」，保留原名稱即可。
    return text


def _mp_format_quote_date(value) -> str:
    """將民國 7 碼日期（1150724）或西元日期統一顯示成 YYYY-MM-DD。"""
    raw = str(value or "").strip()
    digits = re.sub(r"\D", "", raw)
    try:
        if len(digits) == 8 and digits.startswith("20"):
            return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
        if len(digits) == 7:
            return f"{int(digits[:3]) + 1911:04d}-{digits[3:5]}-{digits[5:7]}"
    except Exception:
        pass
    return raw


def _mp_apply_change_sign(change, sign_value):
    """部分 API 把漲跌符號與漲跌金額分欄，這裡合併成有正負號的數值。"""
    if pd.isna(change):
        return np.nan
    sign_text = BeautifulSoup(str(sign_value or ""), "html.parser").get_text(" ", strip=True)
    if any(mark in sign_text for mark in ("-", "－", "−", "▼", "▽", "跌")):
        return -abs(float(change))
    if any(mark in sign_text for mark in ("+", "＋", "▲", "△", "漲")):
        return abs(float(change))
    return float(change)


def _mp_api_headers(url: str) -> dict:
    """官方站台專用標頭；依資料端點帶入正確 Referer，並避免 CDN 舊快取。"""
    if "tpex.org.tw" in url:
        referer = "https://www.tpex.org.tw/zh-tw/mainboard/trading/info/indices-pricing.html"
    elif "www.twse.com.tw" in url or "wwwc.twse.com.tw" in url:
        referer = "https://www.twse.com.tw/zh/trading/historical/mi-index.html"
    else:
        referer = "https://openapi.twse.com.tw/"
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, text/html, */*",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.7",
        "Referer": referer,
        "Cache-Control": "no-cache, no-store, max-age=0",
        "Pragma": "no-cache",
    }


def _mp_fetch_json(url: str, timeout: int = 12):
    """官方 OpenAPI 下載；針對 403/429/5xx 做短暫重試。"""
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                headers=_mp_api_headers(url),
                timeout=(4, timeout),
                verify=False,
            )
            if response.status_code == 200 and response.content:
                try:
                    return response.json()
                except Exception:
                    encoding = response.apparent_encoding or "utf-8"
                    return json.loads(response.content.decode(encoding, errors="ignore").lstrip("\ufeff"))
            if response.status_code not in (403, 429, 500, 502, 503, 504):
                break
        except Exception:
            pass
        if attempt < 2:
            time.sleep(0.45 * (attempt + 1))
    return None


def _mp_payload_rows(payload) -> list:
    """容許 OpenAPI 直接回 list，或把 rows 包在 data／records 等節點。"""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("data", "aaData", "result", "records", "items", "rows"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    # 某些舊版回應會有 data9 這類欄位。
    for key, rows in payload.items():
        if str(key).lower().startswith("data") and isinstance(rows, list):
            return rows
    return []


def _mp_profile_is_valid(profile: dict) -> bool:
    return _mp_normalize_industry((profile or {}).get("industry")) != "未分類"


def _mp_load_profile_cache(market: str) -> dict:
    try:
        if not os.path.exists(MARKET_PROFILE_CACHE_FILE):
            return {}
        with open(MARKET_PROFILE_CACHE_FILE, "r", encoding="utf-8") as file:
            payload = json.load(file)
        rows = payload.get(market, {}).get("profiles", {}) if isinstance(payload, dict) else {}
        if not isinstance(rows, dict):
            return {}
        return {
            str(code): {
                "name": str(item.get("name", "") or "").strip(),
                "industry": _mp_normalize_industry(item.get("industry", "未分類")),
            }
            for code, item in rows.items()
            if _mp_clean_code(code) and isinstance(item, dict)
        }
    except Exception:
        return {}


def _mp_save_profile_cache(market: str, profiles: dict) -> None:
    valid_count = sum(1 for p in profiles.values() if _mp_profile_is_valid(p))
    if valid_count < 100:
        return
    try:
        payload = {}
        if os.path.exists(MARKET_PROFILE_CACHE_FILE):
            with open(MARKET_PROFILE_CACHE_FILE, "r", encoding="utf-8") as file:
                old = json.load(file)
                if isinstance(old, dict):
                    payload = old
        payload[market] = {
            "saved_at": get_tw_now().strftime("%Y-%m-%d %H:%M:%S"),
            "profiles": profiles,
        }
        temp_file = MARKET_PROFILE_CACHE_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temp_file, MARKET_PROFILE_CACHE_FILE)
    except Exception:
        pass


def _mp_fetch_official_profiles(market: str) -> dict:
    """直接取得公司產業別；TPEx 失敗時以公開發行公司基本資料交叉補值。"""
    if market == "TW":
        urls = [
            "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
            "https://openapi.twse.com.tw/v1/opendata/t187ap03_P",
        ]
    else:
        urls = [
            "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
            "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O?l=zh-tw",
            # 公開發行公司基本資料包含上櫃公司，可在 TPEx 公司資料端暫時被擋時補產業別。
            "https://openapi.twse.com.tw/v1/opendata/t187ap03_P",
            # 少數環境仍可讀到這個相容路徑，放最後嘗試，不依賴它。
            "https://openapi.twse.com.tw/v1/opendata/t187ap03_O",
        ]

    profiles = {}
    for url in urls:
        rows = _mp_payload_rows(_mp_fetch_json(url, timeout=14))
        if not rows:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = _mp_clean_code(_mp_pick(row, [
                "公司代號", "Code", "證券代號", "SecuritiesCompanyCode", "股票代號", "有價證券代號",
            ]))
            if not code:
                continue
            name = str(_mp_pick(row, [
                "公司簡稱", "公司名稱", "Name", "證券名稱", "CompanyName", "股票名稱",
            ], "") or "").strip()
            industry = _mp_normalize_industry(_mp_pick(row, [
                "產業別", "產業類別", "Industry", "industry", "IndustryCode", "產業代號",
            ], "未分類"))
            current = profiles.setdefault(code, {"name": "", "industry": "未分類"})
            if name and not current.get("name"):
                current["name"] = normalize_stock_name(name)
            if industry != "未分類":
                current["industry"] = industry
        # 已取得足夠產業資料就不再呼叫後續備援，減少等待。
        if sum(1 for p in profiles.values() if _mp_profile_is_valid(p)) >= 300:
            break
    return profiles


def _mp_stock_profiles(market: str) -> dict:
    """取得代碼、名稱、產業別；合併一般股票清單、獨立官方 API 與磁碟快取。"""
    market_label = "上市" if market == "TW" else "上櫃"
    profiles = {}

    # 先讀既有股票清單，名稱通常最完整；即使產業快取曾污染，也只把有效產業帶進來。
    try:
        stocks = get_stock_market_list()
    except Exception:
        stocks = []
    excluded_industries = {"ETF", "ETN", "受益證券", "認購售權證", "存託憑證"}
    for stock in stocks or []:
        if str(stock.get("市場別", "")).strip() != market_label:
            continue
        code = _mp_clean_code(stock.get("code", ""))
        if not code:
            continue
        name = str(stock.get("name", "") or "").strip()
        industry = _mp_normalize_industry(stock.get("industry", "未分類"))
        if code.startswith("00") or industry.upper() in excluded_industries:
            continue
        if "ETF" in name.upper() or "ETN" in name.upper() or "權證" in name:
            continue
        profiles[code] = {"name": name, "industry": industry}

    # 舊快取只補空值，不覆蓋本次抓到的新資料。
    cached_profiles = _mp_load_profile_cache(market)
    for code, item in cached_profiles.items():
        current = profiles.setdefault(code, {"name": "", "industry": "未分類"})
        if not current.get("name") and item.get("name"):
            current["name"] = item["name"]
        if not _mp_profile_is_valid(current) and _mp_profile_is_valid(item):
            current["industry"] = item["industry"]

    # 無論股票清單是否有資料，都直接嘗試官方公司基本資料，以修復「全部未分類」問題。
    official_profiles = _mp_fetch_official_profiles(market)
    for code, item in official_profiles.items():
        current = profiles.setdefault(code, {"name": "", "industry": "未分類"})
        if item.get("name"):
            current["name"] = item["name"]
        if _mp_profile_is_valid(item):
            current["industry"] = item["industry"]

    valid_profiles = {code: item for code, item in profiles.items() if not code.startswith("00")}
    _mp_save_profile_cache(market, valid_profiles)
    return valid_profiles


def _mp_normalize_quote(row: dict, market: str, profiles: dict):
    """把 TWSE／TPEx 不同欄名統一；即使產業表暫時失敗，仍保留漲跌家數。"""
    code = _mp_clean_code(_mp_pick(row, [
        "Code", "證券代號", "公司代號", "SecuritiesCompanyCode",
        "股票代號", "有價證券代號",
    ]))
    if not code or code.startswith("00"):
        return None

    profile = profiles.get(code, {})
    row_name = str(_mp_pick(row, [
        "Name", "證券名稱", "公司簡稱", "CompanyName", "股票名稱",
    ], "") or "").strip()
    name = str(profile.get("name", "") or row_name).strip()
    upper_name = name.upper()
    if "ETF" in upper_name or "ETN" in upper_name or "權證" in name:
        return None

    industry = _mp_normalize_industry(profile.get("industry", "未分類"))
    if industry == "未分類":
        industry = _mp_normalize_industry(_mp_pick(row, [
            "產業別", "產業類別", "Industry", "industry", "IndustryCode",
        ], "未分類"))
    if industry in ("ETF", "ETN", "受益證券", "認購售權證", "存託憑證"):
        return None

    close = _mp_to_float(_mp_pick(row, [
        "ClosingPrice", "Close", "收盤價", "收盤", "TodayClose",
    ]))
    open_price = _mp_to_float(_mp_pick(row, [
        "OpeningPrice", "Open", "開盤價", "開盤", "TodayOpen",
    ]))
    change = _mp_to_float(_mp_pick(row, [
        "Change", "漲跌價差", "漲跌", "ChangeAmount", "漲跌金額",
    ]))
    sign_value = _mp_pick(row, [
        "ChangeSign", "漲跌(+/-)", "漲跌符號", "Direction", "Trend",
    ])
    change = _mp_apply_change_sign(change, sign_value)

    change_pct = _mp_to_float(_mp_pick(row, [
        "ChangePercent", "ChangeRate", "漲跌幅", "漲跌幅(%)", "漲跌%", "ChangePercentString",
    ]))
    if pd.isna(change_pct) and pd.notna(close) and pd.notna(change):
        previous_close = close - change
        if previous_close > 0:
            change_pct = change / previous_close * 100

    trade_value = _mp_to_float(_mp_pick(row, [
        "TradeValue", "TransactionAmount", "成交金額", "成交值", "Amount", "成交金額(元)",
    ]))
    trade_volume = _mp_to_float(_mp_pick(row, [
        "TradeVolume", "TradingShares", "成交股數", "成交量", "Volume", "成交股數(股)",
    ]))
    quote_date = _mp_format_quote_date(_mp_pick(row, ["Date", "日期", "TradeDate", "資料日期"], ""))

    if pd.isna(change_pct):
        return None

    return {
        "code": code,
        "name": name,
        "industry": industry,
        "close": close,
        "open": open_price,
        "change": change,
        "change_pct": float(change_pct),
        "trade_value": trade_value,
        "trade_volume": trade_volume,
        "quote_date": quote_date,
        "market": market,
    }


@st.cache_data(ttl=600, show_spinner=False)
def fetch_official_market_quotes(market: str) -> pd.DataFrame:
    """取得上市／上櫃最新收盤行情，並統一欄位格式。"""
    profiles = _mp_stock_profiles(market)

    if market == "TW":
        urls = [
            "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
        ]
    else:
        urls = [
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes",
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes?l=zh-tw",
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
        ]

    for url in urls:
        payload = _mp_fetch_json(url)
        rows_payload = _mp_payload_rows(payload)
        if not rows_payload:
            continue

        rows = []
        for item in rows_payload:
            normalized = _mp_normalize_quote(item, market, profiles)
            if normalized:
                rows.append(normalized)
        if len(rows) >= 50:
            return pd.DataFrame(rows)

    return pd.DataFrame()



# 證交所 MI_INDEX 與櫃買中心 indexSummary 使用「官方類股價格指數」。
# 玩股網的上市／上櫃類股漲跌表也是以類股指數為基礎，不是把成分股漲跌幅做算術平均。
# 下列名稱統一層可把「水泥類指數」「水泥工業」等不同寫法合併到同一產業鍵。
_MP_CANONICAL_INDUSTRY_ALIASES = {
    "水泥": "水泥工業", "水泥類": "水泥工業", "水泥工業": "水泥工業",
    "食品": "食品工業", "食品類": "食品工業", "食品工業": "食品工業",
    "塑膠": "塑膠工業", "塑膠類": "塑膠工業", "塑膠工業": "塑膠工業",
    "紡織": "紡織纖維", "紡織類": "紡織纖維", "紡織纖維": "紡織纖維",
    "電機": "電機機械", "電機類": "電機機械", "電機機械": "電機機械",
    "電器電纜": "電器電纜", "電器電纜類": "電器電纜",
    "化學生技醫療": "化學生技醫療", "化學生技": "化學生技醫療",
    "玻璃陶瓷": "玻璃陶瓷", "玻璃陶瓷類": "玻璃陶瓷",
    "造紙": "造紙工業", "造紙類": "造紙工業", "造紙工業": "造紙工業",
    "鋼鐵": "鋼鐵工業", "鋼鐵類": "鋼鐵工業", "鋼鐵工業": "鋼鐵工業",
    "橡膠": "橡膠工業", "橡膠類": "橡膠工業", "橡膠工業": "橡膠工業",
    "汽車": "汽車工業", "汽車類": "汽車工業", "汽車工業": "汽車工業",
    "電子": "電子工業", "電子類": "電子工業", "電子工業": "電子工業",
    "建材營造": "建材營造", "建材營造類": "建材營造",
    "航運": "航運業", "航運類": "航運業", "航運業": "航運業",
    "觀光": "觀光餐旅", "觀光類": "觀光餐旅", "觀光餐旅": "觀光餐旅",
    "金融保險": "金融保險業", "金融保險類": "金融保險業", "金融保險業": "金融保險業",
    "貿易百貨": "貿易百貨業", "百貨貿易": "貿易百貨業", "貿易百貨類": "貿易百貨業", "貿易百貨業": "貿易百貨業",
    "油電燃氣": "油電燃氣業", "油電燃氣類": "油電燃氣業", "油電燃氣業": "油電燃氣業",
    "其他": "其他業", "其他類": "其他業", "其他業": "其他業",
    "化學": "化學工業", "化學類": "化學工業", "化學工業": "化學工業",
    "生技醫療": "生技醫療業", "生技醫療類": "生技醫療業", "生技醫療業": "生技醫療業",
    "半導體": "半導體業", "半導體類": "半導體業", "半導體業": "半導體業",
    "電腦及週邊設備": "電腦及週邊設備業", "電腦及週邊設備類": "電腦及週邊設備業", "電腦及週邊設備業": "電腦及週邊設備業",
    "光電": "光電業", "光電類": "光電業", "光電業": "光電業",
    "通信網路": "通信網路業", "通信網路類": "通信網路業", "通信網路業": "通信網路業",
    "電子零組件": "電子零組件業", "電子零組件類": "電子零組件業", "電子零組件業": "電子零組件業",
    "電子通路": "電子通路業", "電子通路類": "電子通路業", "電子通路業": "電子通路業",
    "資訊服務": "資訊服務業", "資訊服務類": "資訊服務業", "資訊服務業": "資訊服務業",
    "其他電子": "其他電子業", "其他電子類": "其他電子業", "其他電子業": "其他電子業",
    "文化創意": "文化創意業", "文化創意類": "文化創意業", "文化創意業": "文化創意業",
    "綠能環保": "綠能環保", "綠能環保類": "綠能環保",
    "數位雲端": "數位雲端", "數位雲端類": "數位雲端",
    "運動休閒": "運動休閒", "運動休閒類": "運動休閒",
    "居家生活": "居家生活", "居家生活類": "居家生活",
}

_MP_TWSE_INDUSTRY_KEYS = {
    "水泥工業", "食品工業", "塑膠工業", "紡織纖維", "電機機械", "電器電纜",
    "化學生技醫療", "玻璃陶瓷", "造紙工業", "鋼鐵工業", "橡膠工業", "汽車工業",
    "電子工業", "建材營造", "航運業", "觀光餐旅", "金融保險業", "貿易百貨業",
    "油電燃氣業", "其他業", "化學工業", "生技醫療業", "半導體業",
    "電腦及週邊設備業", "光電業", "通信網路業", "電子零組件業", "電子通路業",
    "資訊服務業", "其他電子業", "文化創意業", "綠能環保", "數位雲端", "運動休閒", "居家生活",
}

_MP_TPEX_INDUSTRY_KEYS = {
    "紡織纖維", "電機機械", "鋼鐵工業", "電子工業", "建材營造", "航運業",
    "觀光餐旅", "其他業", "化學工業", "生技醫療業", "半導體業",
    "電腦及週邊設備業", "光電業", "通信網路業", "電子零組件業", "電子通路業",
    "資訊服務業", "其他電子業", "文化創意業", "綠能環保", "數位雲端", "居家生活",
}

_MP_WANTGOO_LABELS = {
    "水泥工業": "水泥類", "食品工業": "食品類", "塑膠工業": "塑膠類",
    "紡織纖維": "紡織類", "電機機械": "電機類", "電器電纜": "電器電纜",
    "化學生技醫療": "化學生技", "玻璃陶瓷": "玻璃陶瓷類", "造紙工業": "造紙類",
    "鋼鐵工業": "鋼鐵類", "橡膠工業": "橡膠類", "汽車工業": "汽車類",
    "電子工業": "電子類", "建材營造": "建材營造", "航運業": "航運類",
    "觀光餐旅": "觀光餐旅", "金融保險業": "金融保險", "貿易百貨業": "百貨貿易",
    "油電燃氣業": "油電燃氣類", "其他業": "其他類", "化學工業": "化學工業",
    "生技醫療業": "生技醫療", "半導體業": "半導體", "電腦及週邊設備業": "電腦及週邊設備",
    "光電業": "光電業", "通信網路業": "通信網路業", "電子零組件業": "電子零組件",
    "電子通路業": "電子通路", "資訊服務業": "資訊服務類", "其他電子業": "其他電子",
    "文化創意業": "文化創意", "綠能環保": "綠能環保", "數位雲端": "數位雲端",
    "運動休閒": "運動休閒", "居家生活": "居家生活",
}


def _mp_canonical_industry(value) -> str:
    """把公司產業別與官方指數名稱轉為可合併的同一產業鍵。"""
    text = BeautifulSoup(str(value or ""), "html.parser").get_text(" ", strip=True)
    text = text.replace("臺灣", "").replace("上櫃", "").replace("價格", "").strip()
    # 報酬指數不是玩股網「類股漲跌表」所使用的價格指數，先保留字樣供上層排除。
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(類)?報酬指數$", "", text)
    text = re.sub(r"類指數$", "", text)
    text = re.sub(r"指數$", "", text)
    return _MP_CANONICAL_INDUSTRY_ALIASES.get(text, _MP_CANONICAL_INDUSTRY_ALIASES.get(_mp_normalize_industry(text), text))


def _mp_index_table_rows(payload) -> list:
    """把 TWSE／TPEx 的 list 或 tables/fields/data 結構展平成 dict rows。"""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []

    flattened = []
    tables = payload.get("tables")
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue
            title = str(table.get("title", "") or "")
            # TPEx 同頁也有報酬指數表；只採第一組價格指數，避免同產業重複。
            if "報酬指數" in title:
                continue
            fields = table.get("fields") or table.get("columns") or []
            data = table.get("data") or table.get("rows") or []
            if isinstance(fields, list) and isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        item = dict(row)
                    elif isinstance(row, (list, tuple)):
                        item = dict(zip(fields, row))
                    else:
                        continue
                    item["_table_title"] = title
                    flattened.append(item)
        if flattened:
            return flattened

    # 兼容 aaData/data/records 等常見包裝。
    rows = _mp_payload_rows(payload)
    return [row for row in rows if isinstance(row, dict)]


def _mp_parse_industry_indices(payload, market: str) -> pd.DataFrame:
    """從官方指數回應擷取產業價格指數的當日漲跌幅。"""
    allowed = _MP_TWSE_INDUSTRY_KEYS if market == "TW" else _MP_TPEX_INDUSTRY_KEYS
    parsed = []
    seen = set()

    for row in _mp_index_table_rows(payload):
        raw_name = _mp_pick(row, [
            "指數", "指數名稱", "名稱", "Index", "IndexName", "name",
        ], "")
        raw_name_text = BeautifulSoup(str(raw_name or ""), "html.parser").get_text(" ", strip=True)
        if not raw_name_text or "報酬" in raw_name_text:
            continue
        key = _mp_canonical_industry(raw_name_text)
        if key not in allowed or key in seen:
            continue

        pct = _mp_to_float(_mp_pick(row, [
            "漲跌百分比", "漲跌百分比(%)", "漲跌幅度 (%)", "漲跌幅度(%)",
            "漲跌幅", "漲跌%", "ChangePercent", "ChangeRate",
        ]))
        direction = _mp_pick(row, ["漲跌", "漲跌(+/-)", "漲跌符號", "Direction"], "")
        pct = _mp_apply_change_sign(pct, direction)
        if pd.isna(pct):
            continue

        close_index = _mp_to_float(_mp_pick(row, [
            "收盤指數", "收市指數", "指數值", "Close", "ClosingIndex",
        ]))
        parsed.append({
            "industry_key": key,
            "類股": _MP_WANTGOO_LABELS.get(key, key),
            "漲跌%": round(float(pct), 3),
            "指數值": close_index,
        })
        seen.add(key)

    return pd.DataFrame(parsed)


def _mp_fetch_json_no_store(url: str):
    """指數端點加時間戳避免中介快取；Streamlit 本身仍以 10 分鐘 TTL 控制更新頻率。"""
    separator = "&" if "?" in url else "?"
    return _mp_fetch_json(f"{url}{separator}_={int(time.time())}")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_official_industry_indices(market: str) -> pd.DataFrame:
    """取得官方類股價格指數漲跌幅；這才是玩股網類股行情可對照的口徑。"""
    if market == "TW":
        urls = [
            "https://openapi.twse.com.tw/v1/exchangeReport/MI_INDEX",
            "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date=&type=ALLBUT0999&response=json",
        ]
    else:
        urls = [
            "https://www.tpex.org.tw/www/zh-tw/afterTrading/indexSummary?date=&response=json",
        ]

    for url in urls:
        payload = _mp_fetch_json_no_store(url)
        result = _mp_parse_industry_indices(payload, market)
        minimum = 20 if market == "TW" else 12
        if len(result) >= minimum:
            return result

    # TPEx JSON 偶爾因站台切版或防護回傳非 JSON，官方 HTML 表格作為第二層備援。
    if market == "TWO":
        url = f"https://www.tpex.org.tw/www/zh-tw/afterTrading/indexSummary?date=&response=html&_={int(time.time())}"
        try:
            response = requests.get(
                url, headers=_mp_api_headers(url), timeout=(4, 15), verify=False,
            )
            if response.status_code == 200 and response.text:
                tables = pd.read_html(StringIO(response.text))
                for df in tables:
                    if df is None or df.empty:
                        continue
                    df.columns = [str(c).strip() for c in df.columns]
                    name_col = next((c for c in df.columns if str(c).strip() in ("指數", "指數名稱")), None)
                    pct_col = next((c for c in df.columns if "漲跌幅" in str(c)), None)
                    if not name_col or not pct_col:
                        continue
                    rows = []
                    for _, record in df.iterrows():
                        rows.append({
                            "指數": record.get(name_col),
                            "漲跌幅度 (%)": record.get(pct_col),
                            "收市指數": record.get(next((c for c in df.columns if "收市指數" in str(c)), ""), np.nan),
                        })
                    result = _mp_parse_industry_indices(rows, market)
                    if len(result) >= 12:
                        return result
        except Exception:
            pass

    return pd.DataFrame()

def _mp_json_records(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _mp_load_disk_cache(market: str) -> dict:
    try:
        if not os.path.exists(MARKET_PULSE_CACHE_FILE):
            return {}
        with open(MARKET_PULSE_CACHE_FILE, "r", encoding="utf-8") as file:
            payload = json.load(file)
        item = payload.get(market, {}) if isinstance(payload, dict) else {}
        if not isinstance(item, dict):
            return {}
        industry_rows = item.get("industry", [])
        # 即使舊快取沒有產業資料，也允許個股漲跌分布使用 breadth。
        if not industry_rows and not item.get("breadth"):
            return {}
        return {
            "industry": pd.DataFrame(industry_rows),
            "breadth": item.get("breadth", {}),
            "distribution": item.get("distribution", {}),
            "total": int(item.get("total", 0) or 0),
            "up": int(item.get("up", 0) or 0),
            "down": int(item.get("down", 0) or 0),
            "flat": int(item.get("flat", 0) or 0),
            "quote_date": _mp_format_quote_date(item.get("quote_date", "")),
            "saved_at": item.get("saved_at", ""),
            "source": item.get("source", "TWSE／TPEx 官方 OpenAPI"),
            "industry_coverage": float(item.get("industry_coverage", 0) or 0),
            "industry_method": item.get("industry_method", "official_index"),
            "index_source": item.get("index_source", ""),
            "is_cached": True,
        }
    except Exception:
        return {}


def _mp_save_disk_cache(market: str, pulse: dict) -> None:
    try:
        payload = {}
        if os.path.exists(MARKET_PULSE_CACHE_FILE):
            with open(MARKET_PULSE_CACHE_FILE, "r", encoding="utf-8") as file:
                old = json.load(file)
                if isinstance(old, dict):
                    payload = old
        payload[market] = {
            "industry": _mp_json_records(pulse.get("industry", pd.DataFrame())),
            "breadth": pulse.get("breadth", {}),
            "distribution": pulse.get("distribution", {}),
            "total": int(pulse.get("total", 0) or 0),
            "up": int(pulse.get("up", 0) or 0),
            "down": int(pulse.get("down", 0) or 0),
            "flat": int(pulse.get("flat", 0) or 0),
            "quote_date": pulse.get("quote_date", ""),
            "saved_at": get_tw_now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "TWSE／TPEx 官方 OpenAPI",
            "industry_coverage": float(pulse.get("industry_coverage", 0) or 0),
            "industry_method": pulse.get("industry_method", "official_index"),
            "index_source": pulse.get("index_source", ""),
        }
        temp_file = MARKET_PULSE_CACHE_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temp_file, MARKET_PULSE_CACHE_FILE)
    except Exception:
        pass


def _mp_build_change_distribution(pct: pd.Series) -> dict:
    """依圖示門檻把個股漲跌幅切成 9 個互斥區間；所有有效個股只會落入一格。"""
    values = pd.to_numeric(pct, errors="coerce").dropna()
    return {
        "漲停": int((values >= 9.5).sum()),
        "5%": int(((values >= 5.0) & (values < 9.5)).sum()),
        "2.5%": int(((values >= 2.5) & (values < 5.0)).sum()),
        "0.2%": int(((values > 0.2) & (values < 2.5)).sum()),
        "平盤": int(((values >= -0.2) & (values <= 0.2)).sum()),
        "-0.2%": int(((values > -2.5) & (values < -0.2)).sum()),
        "-2.5%": int(((values > -5.0) & (values <= -2.5)).sum()),
        "-5%": int(((values > -9.5) & (values <= -5.0)).sum()),
        "跌停": int((values <= -9.5).sum()),
    }


@st.cache_data(ttl=600, show_spinner=False)
def build_market_pulse(market: str) -> dict:
    """彙整官方行情。類股漲跌採官方價格指數；成交比重與漲跌家數採個股行情。"""
    quotes = fetch_official_market_quotes(market)
    if quotes.empty:
        return _mp_load_disk_cache(market)

    quote_dates = [
        _mp_format_quote_date(v) for v in quotes.get("quote_date", pd.Series(dtype=str)).tolist()
        if str(v).strip()
    ]
    quote_date = max(quote_dates) if quote_dates else ""

    quotes = quotes.copy()
    quotes["industry_key"] = quotes["industry"].map(_mp_canonical_industry)
    allowed = _MP_TWSE_INDUSTRY_KEYS if market == "TW" else _MP_TPEX_INDUSTRY_KEYS
    valid_industry_mask = quotes["industry_key"].isin(allowed)
    industry_base = quotes[valid_industry_mask].copy()
    industry_coverage = float(valid_industry_mask.mean()) if len(quotes) else 0.0

    # 個股資料只負責成交額、成交比重與家數；漲跌幅不再採等權平均。
    if industry_base.empty:
        industry_stats = pd.DataFrame()
    else:
        industry_stats = (
            industry_base.groupby("industry_key", as_index=False)
            .agg(
                **{
                    "等權平均漲跌%": ("change_pct", "mean"),
                    "成交額_元": ("trade_value", "sum"),
                    "股票數": ("code", "nunique"),
                    "上漲家數": ("change_pct", lambda s: int((s > 0).sum())),
                    "下跌家數": ("change_pct", lambda s: int((s < 0).sum())),
                }
            )
        )
        total_trade_value = float(industry_stats["成交額_元"].sum(skipna=True))
        industry_stats["成交額"] = industry_stats["成交額_元"] / 1e8
        industry_stats["成交比重%"] = (
            industry_stats["成交額_元"] / total_trade_value * 100
            if total_trade_value > 0 else np.nan
        )
        industry_stats["成交額"] = industry_stats["成交額"].round(2)
        industry_stats["成交比重%"] = industry_stats["成交比重%"].round(3)
        industry_stats["等權平均漲跌%"] = industry_stats["等權平均漲跌%"].round(3)
        industry_stats = industry_stats.drop(columns=["成交額_元"])

    official_indices = fetch_official_industry_indices(market)
    industry_method = "official_index"
    index_source = "證交所 MI_INDEX" if market == "TW" else "櫃買中心 indexSummary"

    if not official_indices.empty:
        industry = official_indices.copy()
        if not industry_stats.empty:
            industry = industry.merge(industry_stats, on="industry_key", how="left")
        industry = industry.sort_values("漲跌%", ascending=False)
    else:
        # 官方指數端點暫時失敗時，優先沿用同日已成功快取的官方類股指數。
        cached = _mp_load_disk_cache(market)
        cached_industry = cached.get("industry", pd.DataFrame()) if cached else pd.DataFrame()
        if (
            cached
            and cached.get("industry_method") == "official_index"
            and not cached_industry.empty
            and (not quote_date or cached.get("quote_date") == quote_date)
        ):
            industry = cached_industry.copy()
            industry_method = "official_index_cache"
            index_source = cached.get("index_source", index_source)
        elif not industry_stats.empty:
            # 最後保底才顯示等權平均，UI 會明確提示此口徑無法與玩股網直接比較。
            industry = industry_stats.copy()
            industry["類股"] = industry["industry_key"].map(lambda x: _MP_WANTGOO_LABELS.get(x, x))
            industry["漲跌%"] = industry["等權平均漲跌%"]
            industry = industry.sort_values("漲跌%", ascending=False)
            industry_method = "equal_weight_fallback"
            index_source = "成分股等權平均備援"
        else:
            industry = pd.DataFrame()
            industry_method = "unavailable"
            index_source = ""

    if not industry.empty:
        industry = industry.drop(columns=["industry_key", "等權平均漲跌%"], errors="ignore")
        industry["漲跌%"] = pd.to_numeric(industry["漲跌%"], errors="coerce").round(3)

    pct = pd.to_numeric(quotes["change_pct"], errors="coerce")
    open_price = pd.to_numeric(quotes["open"], errors="coerce")
    close_price = pd.to_numeric(quotes["close"], errors="coerce")
    up = int((pct > 0).sum())
    down = int((pct < 0).sum())
    flat = int((pct == 0).sum())
    breadth = {
        "上漲": up,
        "漲停": int((pct >= 9.5).sum()),
        "紅K": int((close_price > open_price).sum()),
        "下跌": down,
        "跌停": int((pct <= -9.5).sum()),
        "黑K": int((close_price < open_price).sum()),
        "平盤": flat,
    }
    distribution = _mp_build_change_distribution(pct)

    pulse = {
        "industry": industry,
        "breadth": breadth,
        "distribution": distribution,
        "total": int(len(quotes)),
        "up": up,
        "down": down,
        "flat": flat,
        "quote_date": quote_date,
        "saved_at": get_tw_now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "TWSE／TPEx 官方 OpenAPI",
        "industry_coverage": industry_coverage,
        "industry_method": industry_method,
        "index_source": index_source,
        "is_cached": False,
    }
    _mp_save_disk_cache(market, pulse)
    return pulse


@st.cache_data(ttl=86400, show_spinner=False)
def get_long_price_history(ticker: str, period: str = "3y") -> pd.DataFrame:
    """個股長期日收盤價，供本益比河流圖使用。用 yfinance，抓不到回傳空 DataFrame（fail-open）。"""
    try:
        raw = yf.Ticker(ticker).history(period=period)
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty or "Close" not in raw.columns:
        return pd.DataFrame()
    try:
        raw = raw.reset_index()
        date_col = "Date" if "Date" in raw.columns else raw.columns[0]
        dates = pd.to_datetime(raw[date_col])
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)
        close_col = raw["Close"]
        if isinstance(close_col, pd.DataFrame):  # 極少數情況欄位會是 MultiIndex，保險起見攤平
            close_col = close_col.iloc[:, 0]
        df = pd.DataFrame({"date": dates, "close": close_col}).dropna()
        # [修正] merge_asof 要求兩邊日期欄位的 dtype 完全一致（連解析度 ns/us 都要相同），
        # 這裡統一轉成 datetime64[ns]，避免跟 eps_df 那邊的日期 dtype 對不上而噴 MergeError。
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _quarter_report_effective_date(q_text: str):
    """把「2026 Q1」轉成該季財報大約公告完畢的日期（季底 + 概略公告遞延），
    當作這一季 TTM EPS 開始生效的時間點。Q4 是年報，公告遞延抓長一點（~90天）；
    其餘季報抓 ~45 天。屬概略估計，不是精確公告日，僅用來還原河流圖的時間軸。
    """
    try:
        year_s, q_s = q_text.split(" Q")
        year, q = int(year_s), q_s.strip()
        end_month = {"1": 3, "2": 6, "3": 9, "4": 12}[q]
        end_date = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
        lag_days = 90 if q == "4" else 45
        return end_date + pd.Timedelta(days=lag_days)
    except Exception:
        return pd.NaT


def build_pe_river_data(code: str, market_suffix: str, ticker: str) -> dict:
    """整合季度 EPS 與長期股價，還原本益比河流圖所需資料。
    資料不足，或合併過程出現任何預期外的錯誤（例如日期型別對不上），
    一律回傳 {}，由 UI 端顯示提示，不讓整頁掛掉（fail-open）。
    """
    eps_rows = fetch_quarterly_eps(code, market_suffix)
    if len(eps_rows) < 5:  # 至少要 5 季，才能滾動出 2 個以上的 TTM 資料點
        return {}

    try:
        eps_df = pd.DataFrame(eps_rows).iloc[::-1].reset_index(drop=True)  # 轉成舊到新排序
        eps_df["生效日"] = eps_df["年季"].apply(_quarter_report_effective_date)
        eps_df = eps_df.dropna(subset=["生效日"]).sort_values("生效日").reset_index(drop=True)
        # [修正] 跟 price_df 使用完全相同的 dtype（datetime64[ns]），避免 merge_asof 因為
        # 兩邊日期欄位型別（例如 ns / us 解析度）不一致而拋出 MergeError。
        eps_df["生效日"] = pd.to_datetime(eps_df["生效日"]).astype("datetime64[ns]")
        eps_df["TTM_EPS"] = eps_df["EPS"].rolling(4).sum()
        eps_df = eps_df.dropna(subset=["TTM_EPS"])
        if eps_df.empty:
            return {}

        price_df = get_long_price_history(ticker, period="3y")
        if price_df.empty or len(price_df) < 60:
            return {}

        merged = pd.merge_asof(
            price_df.sort_values("date"),
            eps_df[["生效日", "TTM_EPS"]].rename(columns={"生效日": "date"}).sort_values("date"),
            on="date", direction="backward",
        )
    except Exception:
        return {}

    merged = merged.dropna(subset=["TTM_EPS"])
    merged = merged[merged["TTM_EPS"] > 0]
    if len(merged) < 30:
        return {}

    # [修正] 對齊尾端收盤價。yfinance 的長期日線更新比交易所慢（尤其是台股
    # 收盤後到隔日開盤前那段時間），不補的話河流圖的「目前股價」會停在昨收，
    # 跟上方報價卡對不起來，連帶本益比與相對位置也一起算錯。
    # get_kline_data 是報價卡同一個來源（TWSE／TPEx），拿它的最後一根補齊：
    # 日期較新就補一列、同一天但價格不同就覆蓋。TTM_EPS 沿用最後一季，季報不會日日變動。
    try:
        _k_latest = get_kline_data(code, market_suffix)
        if _k_latest is not None and not _k_latest.empty:
            _k_date = pd.to_datetime(_k_latest["date"].iloc[-1]).normalize()
            _k_close = float(_k_latest["close"].iloc[-1])
            _m_date = pd.to_datetime(merged["date"].iloc[-1]).normalize()
            if pd.notna(_k_close) and _k_close > 0:
                if _k_date > _m_date:
                    _tail = merged.iloc[[-1]].copy()
                    _tail["date"] = _k_date
                    _tail["close"] = _k_close
                    merged = pd.concat([merged, _tail], ignore_index=True)
                elif _k_date == _m_date:
                    merged.iloc[-1, merged.columns.get_loc("close")] = _k_close
    except Exception:
        pass  # 補不到就沿用 yfinance 的序列，不讓整張圖掛掉

    merged["本益比"] = merged["close"] / merged["TTM_EPS"]
    latest_pe = float(merged["本益比"].iloc[-1])

    pe_series = merged["本益比"].replace([np.inf, -np.inf], np.nan).dropna()
    pe_series = pe_series[(pe_series > 0) & (pe_series < 200)]  # 濾掉明顯異常離群值
    if len(pe_series) < 30:
        return {}

    quantiles = {
        "低本益比(10%)": 0.10, "偏低(30%)": 0.30, "中位數(50%)": 0.50,
        "偏高(70%)": 0.70, "高本益比(90%)": 0.90,
    }
    bands = {label: float(pe_series.quantile(q)) for label, q in quantiles.items()}
    for label, pe_val in bands.items():
        merged[label] = merged["TTM_EPS"] * pe_val

    return {
        "df": merged.reset_index(drop=True),
        "bands": bands,
        "latest_pe": latest_pe,
        "pe_min": float(pe_series.min()),
        "pe_max": float(pe_series.max()),
    }


def _extract_ohlcv(frame) -> pd.DataFrame:
    """[新功能] 從 yfinance 回傳的單檔資料表萃取出掃描要用的欄位。

    以前只留 Close/Volume，導致算不出 ATR（真實波幅需要當日高低點）。
    yfinance 本來就把 High/Low 一起抓下來了，保留它們不會多花任何網路成本。

    fail-open：某些個股（新上市、長期停牌）可能缺 High/Low，這時退回只取
    Close/Volume，ATR 會改用收盤價振幅近似，而不是整檔資料被丟掉。
    """
    if frame is None:
        return None
    try:
        df = frame[["Close", "High", "Low", "Volume"]].dropna()
        df.columns = ["close", "high", "low", "volume"]
    except Exception:
        try:
            df = frame[["Close", "Volume"]].dropna()
            df.columns = ["close", "volume"]
        except Exception:
            return None
    if df.empty:
        return None
    try:
        df["volume"] = (df["volume"] / 1000).astype(int)
        df = _trim_stale_trailing_days(df)
        return df.reset_index(drop=True)
    except Exception:
        return None


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
        # [修正] 4mo 只有約 79~84 根 K 棒，剛好卡在策略4（需要 80 根、MA60+攻擊/拉回區間）
        # 的邊緣，遇到農曆年那種長假就會不足而整批掃不出東西。改成 6mo（約 122~127 根）
        # 留出餘裕；其餘策略最多只需要 65 根，不受影響。
        raw = yf.download(ticker_str, period="6mo", interval="1d",
                          group_by="ticker", auto_adjust=True, progress=False, threads=True)
    except Exception:
        return {}

    # [修正3] 下載失敗時提前返回，避免後續 KeyError
    if raw is None or raw.empty:
        return {}

    result = {}
    if len(tickers) == 1:
        tk = tickers[0]
        df = _extract_ohlcv(raw)
        if df is not None:
            result[tk] = df
    else:
        for tk in tickers:
            try:
                df = _extract_ohlcv(raw[tk])
            except Exception:
                continue
            if df is not None:
                result[tk] = df
    return result

# ============================================================
# [新功能] 多策略切換：均線多頭排列 / 量價背離 / 型態突破
# ============================================================
# 設計原則：每種策略都拆成「單日條件判斷」函式（輸入 closes/volumes 與索引 i），
# 這組判斷函式同時被「即時掃描」（判斷最後一天）與「策略回測」（判斷歷史每一天）
# 共用——確保回測驗證的邏輯跟實際掃描時用的邏輯完全一致，回測結果才有意義。

def _check_ma_condition(closes: pd.Series, volumes: pd.Series, i: int, bias_limit: float):
    """策略1：均線多頭排列。MA30>MA45>MA60，收盤價須 ≥ 30MA（明確限制，不追高超過乖離上限）。"""
    if i < 59:
        return False, np.nan
    ma30 = closes.iloc[i-29:i+1].mean()
    ma45 = closes.iloc[i-44:i+1].mean()
    ma60 = closes.iloc[i-59:i+1].mean()
    price = closes.iloc[i]
    if pd.isna(ma30) or ma30 <= 0:
        return False, np.nan
    bias = (price - ma30) / ma30 * 100
    # [新限制] 收盤價一定要 >= 30MA（price >= ma30），獨立寫成明確條件，
    # 不再只靠 bias >= 0 隱含達成；乖離上限則另外限制「不能貼太多、追高太多」。
    ok = (ma30 > ma45 > ma60) and (price >= ma30) and (bias <= bias_limit)
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


def _check_ma_attack_pullback_condition(closes: pd.Series, volumes: pd.Series, i: int, shrink_ratio_limit: float):
    """策略4：均線多頭排列 + 攻擊帶量、拉回量縮。在策略1（MA30>MA45>MA60多頭排列）
    的基礎上，往前找近期一段「攻擊上漲」（帶量走高、漲幅明顯）的高點，再確認目前
    正處於拉回整理，且拉回期間成交量較攻擊時明顯萎縮（量縮），同時股價尚未跌破
    45MA支撐——代表這是主力拉回洗浮額，而非趨勢轉弱。"""
    if i < 79:
        return False, np.nan
    ma30 = closes.iloc[i-29:i+1].mean()
    ma45 = closes.iloc[i-44:i+1].mean()
    ma60 = closes.iloc[i-59:i+1].mean()
    if pd.isna(ma30) or pd.isna(ma45) or pd.isna(ma60):
        return False, np.nan
    if not (ma30 > ma45 > ma60):
        return False, np.nan

    price = closes.iloc[i]
    if pd.isna(price) or price < ma45:
        return False, np.nan

    lookback = 15
    window_closes = closes.iloc[i-lookback:i+1]
    if window_closes.isna().any():
        return False, np.nan
    attack_high_pos = int(window_closes.values.argmax())
    attack_high_idx = i - lookback + attack_high_pos
    attack_high = float(window_closes.iloc[attack_high_pos])

    days_since_high = i - attack_high_idx
    if days_since_high < 2:  # 高點必須是「前段」，留出拉回天數
        return False, np.nan

    # 攻擊段：高點前 5 個交易日視為攻擊上漲區間，量能須明顯高於更早的基準量
    attack_start = max(0, attack_high_idx - 4)
    attack_vol_avg = volumes.iloc[attack_start:attack_high_idx+1].mean()
    base_start = max(0, attack_start - 20)
    if attack_start <= base_start:
        return False, np.nan
    base_vol_avg = volumes.iloc[base_start:attack_start].mean()
    if pd.isna(base_vol_avg) or base_vol_avg <= 0 or pd.isna(attack_vol_avg):
        return False, np.nan
    attack_vol_ratio = attack_vol_avg / base_vol_avg

    pre_attack_price = closes.iloc[attack_start]
    if pd.isna(pre_attack_price) or pre_attack_price <= 0:
        return False, np.nan
    attack_rise_pct = (attack_high / pre_attack_price - 1) * 100

    # 拉回段：高點之後到今天，量能須較攻擊段明顯萎縮
    pullback_vol_avg = volumes.iloc[attack_high_idx+1:i+1].mean()
    if pd.isna(pullback_vol_avg) or attack_vol_avg <= 0:
        return False, np.nan
    vol_shrink_ratio = pullback_vol_avg / attack_vol_avg

    pullback_pct = (price / attack_high - 1) * 100  # 負值＝拉回幅度

    ok = (
        attack_vol_ratio >= 1.3 and          # 攻擊段帶量：至少比基準量多30%
        attack_rise_pct >= 3.0 and           # 攻擊段漲幅至少3%，確認是真的上攻
        vol_shrink_ratio <= shrink_ratio_limit and  # 拉回量縮到門檻以下
        -15.0 <= pullback_pct <= -1.0        # 拉回幅度落在合理區間（太淺不算拉回，太深視為破壞）
    )
    return ok, vol_shrink_ratio


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
    # [新功能] 只存 ATR 原始值：停損價與建議張數跟使用者的風險設定有關，
    # 留到顯示時再算，調參數就不必重跑整輪掃描。
    atr20 = calc_atr(df, 20)
    return {**s,
        "ATR20":      round(atr20, 2) if pd.notna(atr20) else np.nan,
        "ATR比例(%)":  round(atr20 / curr_price * 100, 2) if pd.notna(atr20) and curr_price > 0 else np.nan,
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
            row["訊號說明"] = f"MA30>MA45>MA60 多頭排列，收盤價≥30MA，乖離 {bias:.1f}%"
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


def calc_ma_attack_pullback_signals(history_map, stock_map, shrink_ratio_limit, vol_limit):
    """策略4：均線多頭排列 + 攻擊帶量、拉回量縮。"""
    hits = []
    for s in stock_map:
        tk = s["ticker"]
        df = history_map.get(tk)
        # 條件函式要求 i >= 79，也就是至少 80 根 K 棒；原本寫 90 是比實際需求還嚴的
        # 保守值，配上當時 4mo 的下載視窗就變成「永遠不成立」，掃描必定 0 檔。
        if df is None or len(df) < 80:
            continue
        closes = df["close"]
        volumes = df["volume"]
        if volumes.tail(5).mean() < vol_limit:
            continue
        i = len(closes) - 1
        ok, vol_shrink_ratio = _check_ma_attack_pullback_condition(closes, volumes, i, shrink_ratio_limit)
        if ok:
            row = _build_common_signal_fields(s, df)
            row["策略"] = "均線多頭+攻擊量縮拉回"
            row["訊號說明"] = f"近期帶量攻擊上漲後拉回整理，拉回量縮至攻擊量的{vol_shrink_ratio*100:.0f}%，站穩45MA"
            hits.append(row)
    return hits


# 策略登記表：新增策略只要在這裡註冊，Tab1 的選單與掃描流程會自動支援。
STRATEGY_REGISTRY = {
    "均線多頭排列": {
        "func": calc_ma_signals,
        "desc": "尋找 MA30 &gt; MA45 &gt; MA60 的多頭排列股票，且收盤價須 ≥ 30MA，適合抓穩定趨勢股。",
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
    "均線多頭+攻擊量縮拉回": {
        "func": calc_ma_attack_pullback_signals,
        "desc": "在均線多頭排列（MA30&gt;MA45&gt;MA60）基礎上，尋找近期有帶量攻擊上漲、目前正拉回整理、且拉回量能明顯萎縮、股價仍站穩45MA的股票，適合抓主力洗盤後的續攻布局點。",
        "param_label": "拉回量縮比門檻 (倍)", "param_help": "拉回期間均量 ÷ 攻擊上漲期間均量的比值上限，數值越小代表要求拉回時量縮得越乾淨。",
        "param_min": 0.2, "param_max": 0.9, "param_default": 0.6, "param_step": 0.05,
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


# ============================================================
# [新功能] 產業別同儕比較
# ============================================================
# 完全用「已經全市場批次快取」的資料算，不逐股即時打 API：
#   - 股票清單／產業別：get_stock_market_list()（每日快取）
#   - 本益比：load_official_pe_map()（每日快取，跟本益比河流圖同一份資料）
# 樣本數太少（例如冷門產業只有 1、2 檔）時直接回傳空 dict，UI 端會顯示「資料不足」，
# 不勉強比較。

@st.cache_data(ttl=3600, show_spinner=False)
def build_industry_peer_stats(code: str, industry: str) -> dict:
    """算出這檔股票在「同產業」裡的本益比相對位置。

    回傳：{industry, peer_count, pe_sample_count, my_pe, median_pe, min_pe, max_pe,
           cheaper_than_pct, peer_pe_values}
    industry 為空、樣本數不足 3 檔、或抓不到股票清單／本益比資料時回傳空 dict。
    """
    industry = str(industry or "").strip()
    if not industry or industry in ("ETF", "未分類", "N/A", "-"):
        return {}

    stock_map = get_stock_market_list()
    if not stock_map:
        return {}
    peer_codes = [s.get("code") for s in stock_map if s.get("industry") == industry and s.get("code")]
    if len(peer_codes) < 3:
        return {}

    pe_map = load_official_pe_map(False)
    if not pe_map:
        return {}

    pe_pairs = [(c, _clean_pe_value(pe_map.get(c))) for c in peer_codes]
    pe_pairs = [(c, v) for c, v in pe_pairs if pd.notna(v)]
    if len(pe_pairs) < 3:
        return {}

    all_pe = [v for _, v in pe_pairs]
    my_pe = dict(pe_pairs).get(code, np.nan)

    percentile = np.nan
    if pd.notna(my_pe):
        # 本益比比「這檔股票」高（比較貴）的同業占比 → 這檔股票比多少同業便宜。
        pricier_count = sum(1 for v in all_pe if v > my_pe)
        percentile = round(100 * pricier_count / len(all_pe), 1)

    return {
        "industry": industry,
        "peer_count": len(peer_codes),
        "pe_sample_count": len(pe_pairs),
        "my_pe": my_pe,
        "median_pe": float(np.median(all_pe)),
        "min_pe": float(np.min(all_pe)),
        "max_pe": float(np.max(all_pe)),
        "cheaper_than_pct": percentile,
        "peer_pe_values": all_pe,
    }


# ============================================================
# [新功能] 供應鏈上下游（產業鏈地圖）
# ============================================================
# 台灣目前沒有免費、結構化的官方「供應鏈關係」API——證交所/櫃買中心合建的
# 「產業價值鏈資訊平台」(ic.tpex.org.tw) 資料最完整最權威，但該平台有防爬蟲
# 機制擋掉一般 requests 連線，且頁面本身是前端動態渲染，不是穩定可爬的資料源。
#
# 這裡採用「內建常見產業鏈對照表＋官方平台連結」雙軌做法：
#   1) INDUSTRY_CHAIN_DB：手動整理的熱門產業鏈上中下游對照表，只涵蓋幾條
#      最常被查詢的產業鏈（半導體、AI伺服器、PCB、面板、被動元件、記憶體、
#      紡織），每條鏈只列該階段幾檔具代表性、市值/知名度較高的股票，不是
#      窮舉清單。這份資料是整理自公開產業知識，「不是」官方即時資料，可能
#      有過時、疏漏，或分類見仁見智的地方，僅供快速參考起點。
#   2) 查不到、或想看完整/最新資料時，UI 會提供「產業價值鏈資訊平台」官方
#      連結，使用者自己輸入代碼查詢，資料權威性以官方平台為準。
#
# 如果之後想擴充涵蓋的產業鏈，直接在 INDUSTRY_CHAIN_DB 裡新增/修改即可，
# 不需要改動查詢邏輯。

INDUSTRY_CHAIN_DB = {
    "半導體（晶圓代工／IC設計／封測）": {
        "upstream": [("3105", "穩懋"), ("8069", "元太"), ("6488", "環球晶"), ("2455", "全新")],
        "midstream": [("2330", "台積電"), ("2303", "聯電"), ("5347", "世界先進"), ("3711", "日月光投控")],
        "downstream": [("2449", "京元電子"), ("6239", "力成"), ("8150", "南茂"), ("3374", "精材")],
    },
    "IC設計": {
        "upstream": [("3529", "力旺"), ("6274", "台燿")],
        "midstream": [("2454", "聯發科"), ("3034", "聯詠"), ("2379", "瑞昱"), ("3443", "創意"), ("3661", "世芯-KY")],
        "downstream": [("2449", "京元電子"), ("6239", "力成")],
    },
    "AI伺服器／雲端運算": {
        "upstream": [("3443", "創意"), ("3661", "世芯-KY"), ("2330", "台積電")],
        "midstream": [("2382", "廣達"), ("2317", "鴻海"), ("6669", "緯穎"), ("2356", "英業達"), ("3231", "緯創")],
        "downstream": [("3017", "奇鋐"), ("3324", "雙鴻"), ("2308", "台達電"), ("6412", "群電")],
    },
    "PCB／載板": {
        "upstream": [("6274", "台燿"), ("8046", "南電"), ("1305", "華夏")],
        "midstream": [("3037", "欣興")],
        "downstream": [],
    },
    "面板／顯示器": {
        "upstream": [("3034", "聯詠"), ("3661", "世芯-KY")],
        "midstream": [("2409", "友達"), ("3481", "群創")],
        "downstream": [("2371", "大同"), ("2492", "華新科")],
    },
    "被動元件": {
        "upstream": [("6274", "台燿")],
        "midstream": [("2327", "國巨"), ("2492", "華新科"), ("2456", "奇力新")],
        "downstream": [("2330", "台積電"), ("2317", "鴻海")],
    },
    "記憶體": {
        "upstream": [("2408", "南亞科"), ("2337", "旺宏"), ("2344", "華邦電")],
        "midstream": [("8299", "群聯"), ("2451", "創見")],
        "downstream": [("6230", "尼盛"), ("5289", "宜鼎")],
    },
    "紡織成衣": {
        "upstream": [("1402", "遠東新"), ("1303", "南亞")],
        "midstream": [("1476", "儒鴻"), ("1477", "聚陽"), ("1451", "年興")],
        "downstream": [],
    },
}

# 防呆過濾：如果之後手動編輯 INDUSTRY_CHAIN_DB 打錯代碼／漏填名稱，這裡統一擋掉，
# 避免顯示格式不完整的垃圾資料。
for _chain_name, _stages in INDUSTRY_CHAIN_DB.items():
    for _stage_name, _peers in _stages.items():
        _stages[_stage_name] = [(c, n) for c, n in _peers if n and re.fullmatch(r"\d{4,6}", c)]


def find_industry_chain_matches(code: str) -> list:
    """查這檔股票出現在 INDUSTRY_CHAIN_DB 裡的哪些產業鏈、哪個階段，
    並整理出「相對這檔股票」的上游／同階段／下游公司名單。

    一家公司可能同時出現在多條產業鏈裡（例如台積電同時是半導體中游、
    也是 AI 伺服器產業鏈的上游晶片供應方），所以回傳 list，可能有多筆。
    完全查表操作，沒有網路請求，速度可忽略不計。
    """
    matches = []
    for chain_name, stages in INDUSTRY_CHAIN_DB.items():
        stage_order = ["upstream", "midstream", "downstream"]
        stage_labels = {"upstream": "上游", "midstream": "中游", "downstream": "下游"}
        found_stage = None
        for stage in stage_order:
            if any(c == code for c, _ in stages.get(stage, [])):
                found_stage = stage
                break
        if not found_stage:
            continue

        idx = stage_order.index(found_stage)
        result = {"chain": chain_name, "stage": stage_labels[found_stage], "groups": []}
        for stage in stage_order:
            peers = [(c, n) for c, n in stages.get(stage, []) if c != code]
            if not peers:
                continue
            stage_idx = stage_order.index(stage)
            if stage_idx < idx:
                rel_label = f"⬅ 上游｜{stage_labels[stage]}"
            elif stage_idx > idx:
                rel_label = f"➡ 下游｜{stage_labels[stage]}"
            else:
                rel_label = f"⬌ 同階段｜{stage_labels[stage]}"
            result["groups"].append({"label": rel_label, "peers": peers})
        matches.append(result)
    return matches


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


def calc_atr(df: pd.DataFrame, period: int = 20) -> float:
    """[新功能] 平均真實區間 ATR（Wilder 平滑），用來衡量個股「日常會震多少」。

    真實區間 TR = max(當日高 − 當日低, |當日高 − 昨收|, |當日低 − 昨收|)，
    取 |高−昨收| 與 |低−昨收| 是為了把跳空缺口算進波動裡；只看高低差會低估。
    平滑用 Wilder 的 RMA（alpha = 1/period），跟看盤軟體的 ATR 定義一致，
    不是單純的算術平均。

    若資料缺 high/low（少數個股），退回用收盤價的日變動絕對值近似，
    這個近似值會略為低估波動（沒有把盤中振幅與跳空算進去）。
    """
    try:
        if df is None or df.empty or len(df) < period + 1:
            return np.nan
        close = df["close"].astype(float)
        if "high" in df.columns and "low" in df.columns:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
        else:
            tr = close.diff().abs()
        tr = tr.dropna()
        if len(tr) < period:
            return np.nan
        atr = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        return float(atr) if pd.notna(atr) and atr > 0 else np.nan
    except Exception:
        return np.nan


def calc_position_plan(price, atr, atr_mult=2.0, risk_budget=30000, lot_size=1000,
                       is_etf: bool = False) -> dict:
    """由 ATR 推出有效停損檔位，再由單筆可承受虧損反推部位大小。"""
    empty = {"停損價": np.nan, "停損幅度(%)": np.nan, "每張風險": np.nan,
             "建議張數": np.nan, "建議股數": np.nan, "投入金額": np.nan, "實際風險": np.nan}
    try:
        price = float(price)
        atr = float(atr)
        if not np.isfinite(price) or not np.isfinite(atr) or price <= 0 or atr <= 0:
            return empty

        # ATR 算出的理論價位先轉成交易所允許的升降單位，後續風險與張數
        # 全部以這個可實際下單的停損價重算，避免畫面出現 153.18、197.73 等無效檔位。
        stop = round_to_tw_tick(price - atr_mult * atr, is_etf=is_etf, mode='nearest')
        if pd.isna(stop) or stop <= 0 or stop >= price:
            return empty

        risk_per_share = price - stop
        risk_per_lot = risk_per_share * lot_size
        if risk_per_lot <= 0:
            return empty
        lots = int(max(0, np.floor(float(risk_budget) / risk_per_lot)))
        shares = int(max(0, np.floor(float(risk_budget) / risk_per_share)))
        unit_lots = lots if lots > 0 else shares / lot_size
        return {
            "停損價": stop,
            "停損幅度(%)": round(risk_per_share / price * 100, 2),
            "每張風險": round(risk_per_lot, 0),
            "建議張數": lots,
            "建議股數": shares,
            "投入金額": round(unit_lots * price * lot_size, 0),
            "實際風險": round(unit_lots * risk_per_lot, 0),
        }
    except Exception:
        return empty


def build_trade_plan(stock: dict, risk_budget: float, atr_mult: float) -> dict:
    """把 ATR 部位計算延伸為可直接建立持倉、且符合台股升降單位的交易計畫。"""
    price = _safe_number(stock.get('收盤'))
    atr = _safe_number(stock.get('ATR20'))
    is_etf = is_etf_instrument(stock)
    base = calc_position_plan(price, atr, atr_mult, risk_budget, is_etf=is_etf)
    stop = _safe_number(base.get('停損價'))
    if pd.isna(price) or pd.isna(stop) or price <= stop:
        return {
            'entry_low': np.nan, 'entry_high': np.nan, 'chase_limit': np.nan,
            'stop': stop, 'target1': np.nan, 'target2': np.nan,
            'rr1': np.nan, 'rr2': np.nan, 'price_tick': get_tw_price_tick(price, is_etf),
            'is_etf': is_etf, **base,
        }

    risk_per_share = price - stop
    entry_low = round_to_tw_tick(max(0.01, price - 0.30 * atr), is_etf)
    entry_high = round_to_tw_tick(price + 0.20 * atr, is_etf)
    chase_limit = round_to_tw_tick(price + 0.50 * atr, is_etf)
    target1 = round_to_tw_tick(price + 1.5 * risk_per_share, is_etf)
    target2 = round_to_tw_tick(price + 3.0 * risk_per_share, is_etf)

    # 極低價或極小 ATR 時，四捨五入後可能落在相同檔位；至少保留一個有效跳動單位。
    if pd.notna(target1) and target1 <= price:
        target1 = round_to_tw_tick(price + get_tw_price_tick(price, is_etf), is_etf, mode='up')
    if pd.notna(target2) and target2 <= target1:
        target2 = round_to_tw_tick(target1 + get_tw_price_tick(target1, is_etf), is_etf, mode='up')

    return {
        'entry_low': entry_low, 'entry_high': entry_high,
        'chase_limit': chase_limit, 'stop': stop,
        'target1': target1, 'target2': target2,
        'rr1': 1.5, 'rr2': 3.0,
        'price_tick': get_tw_price_tick(price, is_etf), 'is_etf': is_etf,
        **base,
    }


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
# [新功能] 手動查詢個股：不必等策略掃描，直接輸入代碼就能看完整工作台
# ============================================================
# 刻意重用跟策略掃描完全相同的技術指標計算管線
# （_build_common_signal_fields + fetch_deep_info + calc_stock_score 取飆股雷達，
# fetch_financial_health_score 取財務評分），確保手動查詢跟策略掃描出來的個股，
# 在 K線／AI分析／多空指標…等分頁呈現的資訊是同一套邏輯、不會兜不起來，
# 也不用另外維護一套簡化版計算。

def build_manual_stock_row(code: str) -> dict:
    """依股票代碼組出跟策略掃描結果同格式的一筆資料，供個股工作台直接使用。
    找不到代碼、或抓不到報價資料時回傳 None，由呼叫端顯示錯誤訊息（fail-open）。
    """
    code = str(code).strip().upper()
    if not code:
        return None

    try:
        stock_map = get_stock_market_list()
    except Exception:
        stock_map = []
    base = next((s for s in stock_map if str(s.get("code", "")).upper() == code), None)
    if base is None:
        return None

    market = "TW" if str(base["ticker"]).endswith(".TW") else "TWO"
    try:
        price_df = get_kline_data(base["code"], market)
    except Exception:
        price_df = pd.DataFrame()
    if price_df.empty or len(price_df) < 30:
        return None

    row_data = _build_common_signal_fields({
        "ticker": base["ticker"], "code": base["code"], "name": base["name"],
        "industry": base.get("industry", "未分類"),
        "市場別": "上市" if market == "TW" else "上櫃",
    }, price_df)

    try:
        deep_res = fetch_deep_info(base["ticker"])
    except Exception:
        deep_res = {"pe": np.nan, "mom": np.nan, "yoy": np.nan}
    row_data["本益比"] = deep_res.get("pe", np.nan)
    row_data["營收月增"] = deep_res.get("mom", np.nan)
    row_data["營收年增"] = deep_res.get("yoy", np.nan)

    try:
        hot_codes = get_hot_stock_tickers()
    except Exception:
        hot_codes = set()
    row_data["熱門股"] = base["code"] in hot_codes

    row_data["策略"] = "手動查詢"
    row_data["訊號說明"] = "使用者手動輸入代碼查詢，非策略掃描結果，僅顯示個股資訊與財務評分供參考。"

    _, radar = calc_stock_score(row_data)
    row_data["AI評分"] = fetch_financial_health_score(base["code"])
    row_data["飆股雷達"] = "、".join(radar) if radar else "觀察"
    return row_data


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
    "均線多頭+攻擊量縮拉回": _check_ma_attack_pullback_condition,
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
        # 策略4 的條件函式在 i < 79 一律回 False，資料太短等於整檔沒訊號，先擋掉
        min_bars = 80 if strategy_name == "均線多頭+攻擊量縮拉回" else 65
        if len(sub) < min_bars:
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


def _pct_color_style(val):
    """給 pandas Styler 用：正數綠、負數紅，跟全站漲跌配色（EPS長條圖等）一致。"""
    if pd.isna(val):
        return "color:#8f9bad;"
    color = "#35c48d" if val >= 0 else "#f23645"
    return f"color:{color};font-weight:700;"


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


@st.cache_data(ttl=3600, show_spinner=False)
def get_kline_data_adjusted(ticker: str) -> pd.DataFrame:
    """還原股價（除權息還原）K線資料，來源 yfinance auto_adjust=True。

    get_kline_data() 抓的是 TWSE/TPEx 官方原始成交價，不會因除權息往回調整；
    除息當天的價格缺口在原始價格序列上看起來像是「一根跳空長黑」，容易讓均線、
    型態辨識短暫失真——這也是策略掃描（download_batch_history）改用還原股價
    判斷訊號的原因。這裡提供同一套還原邏輯給 K 線圖切換使用，方便對照「掃描邏輯
    看到的樣子」跟「原始成交價看到的樣子」，兩者都能自己驗證。
    抓不到資料時回傳空 DataFrame，呼叫端會自動退回原始股價（fail-open）。
    """
    try:
        raw = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        # yfinance 新版有時回傳 MultiIndex 欄位（即使只查一檔），統一攤平。
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()
        df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                 "Low": "low", "Close": "close", "Volume": "volume"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["volume"] = (df["volume"] / 1000).astype(int)
        return df
    except Exception:
        return pd.DataFrame()


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

def draw_k_line(ticker, name, chart_mode='K線圖', chart_period='日', adjusted=False):
    """畫出有實際切換功能的金融圖表。
    chart_mode: K線圖 / 走勢圖 / 技術指標
    chart_period: 日 / 週 / 月
    adjusted: True 時改用除權息還原股價（跟策略掃描同一套邏輯），
              抓不到還原資料時自動退回原始成交價（fail-open）。
    """
    code   = ticker.split(".")[0]
    market = "TW" if ticker.endswith(".TW") else "TWO"
    used_adjusted = False
    lag_days = 0
    if adjusted:
        df = get_kline_data_adjusted(ticker)
        if not df.empty and len(df) >= 30:
            used_adjusted = True
            # [說明] yfinance 的還原股價資料來源跟 TWSE 官方原始股價（get_kline_data）
            # 不是同一家，Yahoo 的台股資料常常會晚官方收盤資料幾個小時到大半天才更新，
            # 尤其是台灣凌晨時段（Yahoo 後端多半還在處理前一個交易日的資料），這時候
            # 還原股價圖表看起來會「少一天」，但其實只是資料源還沒跟上，過幾個小時
            # 通常就會補齊，不是程式漏抓。這裡跟官方原始資料比對最新交易日，抓到落後
            # 幾天就在圖上提示，不讓使用者誤以為是程式漏抓資料。
            try:
                raw_ref = get_kline_data(code, market)
                if not raw_ref.empty and not df.empty:
                    raw_last = pd.to_datetime(raw_ref["date"]).max()
                    adj_last = pd.to_datetime(df["date"]).max()
                    if pd.notna(raw_last) and pd.notna(adj_last) and adj_last < raw_last:
                        lag_days = int((raw_last - adj_last).days)
            except Exception:
                lag_days = 0
        else:
            df = get_kline_data(code, market)
    else:
        df = get_kline_data(code, market)
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

    if used_adjusted:
        badge_text = '還原股價（除權息還原）'
        if lag_days > 0:
            badge_text += f'　⚠ 資料來源比官方收盤價慢約{lag_days}天，晚點再看通常會補上'
        fig.add_annotation(
            xref='paper', yref='paper', x=0.012, y=0.895,
            text=badge_text, showarrow=False, align='left',
            font=dict(size=11, color='#facc15'),
            bgcolor='rgba(7,13,20,.55)', bordercolor='rgba(250,204,21,.45)', borderwidth=1,
            borderpad=4,
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
# [新功能] 個股投資分析報告：純量化規則版本
# ============================================================
# 所有欄位（重點觀察、催化因素、估值旗標、同業比較、情境價格區間）
# 全部用規則與公開資料自行計算，不依賴任何外部AI服務，不需要API金鑰。


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_industry_peers_table(industry: str, exclude_code: str, max_peers: int = 4) -> pd.DataFrame:
    """同產業比較表：先用近20日均量篩出流動性較高的同業候選（避免對整個
    產業逐一呼叫財務評分，成本太高），再對候選計算財務體質評分，取分數
    最高的幾檔做比較。任何一步失敗都不中斷，最多回傳空表（fail-open）。"""
    if not industry or industry in ("未分類", ""):
        return pd.DataFrame()
    try:
        stock_map = get_stock_market_list()
    except Exception:
        return pd.DataFrame()
    peers = [s for s in stock_map if s.get("industry") == industry and str(s.get("code")) != str(exclude_code)]
    if not peers:
        return pd.DataFrame()

    peer_tickers = tuple(s["ticker"] for s in peers[:60])  # 產業股票數上限保護，避免超大產業拖慢下載
    history_map = download_batch_history(peer_tickers)

    liquidity = []
    for s in peers:
        df = history_map.get(s["ticker"])
        if df is None or df.empty:
            continue
        avg_vol = df["volume"].tail(20).mean()
        if pd.isna(avg_vol) or avg_vol <= 0:
            continue
        liquidity.append((s, df, avg_vol))
    if not liquidity:
        return pd.DataFrame()

    liquidity.sort(key=lambda x: x[2], reverse=True)
    candidates = liquidity[:10]  # 流動性前10檔再進一步算財務評分，控制成本

    def _score_one(item):
        s, df, _ = item
        try:
            score = fetch_financial_health_score(s["code"])
        except Exception:
            score = np.nan
        try:
            pe = fetch_pe(s["ticker"])
        except Exception:
            pe = np.nan
        return {
            "代碼": s["code"], "名稱": s["name"],
            "收盤": round(float(df["close"].iloc[-1]), 2),
            "本益比": pe, "財務評分": score,
        }

    with ThreadPoolExecutor(max_workers=5) as ex:
        rows = list(ex.map(_score_one, candidates))

    peer_df = pd.DataFrame(rows).dropna(subset=["財務評分"])
    if peer_df.empty:
        return pd.DataFrame()
    return peer_df.sort_values("財務評分", ascending=False).head(max_peers).reset_index(drop=True)


def calc_volatility_price_bands(closes: pd.Series, horizon_days: int = 20, lookback_days: int = 60) -> dict:
    """用近期歷史波動率反推情境價格區間：純統計上的機率分布估算（隨機漫步
    模型：日報酬標準差 × sqrt(horizon)），不是目標價、也不是基本面判斷。
    Bear/Bull 抓 ±1 個標準差，Stretched Bull 抓 +1.8 個標準差（機率更低
    的極端樂觀情境），資料不足時回傳空字典（fail-open）。"""
    if closes is None or len(closes) < lookback_days + 5:
        return {}
    recent = closes.tail(lookback_days).reset_index(drop=True)
    rets = np.log(recent / recent.shift(1)).dropna()
    if rets.empty or rets.std() == 0 or pd.isna(rets.std()):
        return {}
    daily_std = float(rets.std())
    horizon_std = daily_std * np.sqrt(horizon_days)
    current = float(closes.iloc[-1])
    return {
        "現價": round(current, 2),
        "悲觀 Bear (約-1σ)": round(current * np.exp(-1.0 * horizon_std), 2),
        "基準 Base": round(current, 2),
        "樂觀 Bull (約+1σ)": round(current * np.exp(1.0 * horizon_std), 2),
        "極樂觀 Stretched Bull (約+1.8σ)": round(current * np.exp(1.8 * horizon_std), 2),
        "年化波動率(%)": round(daily_std * np.sqrt(252) * 100, 1),
        "推算天數": horizon_days,
    }


def _valuation_flag(row_data: dict) -> str:
    """用財務評分＋乖離率粗略判斷估值旗標，純規則、不涉及AI。"""
    score = row_data.get("AI評分", np.nan)
    bias = row_data.get("乖離30MA(%)", np.nan)
    if pd.isna(score):
        return "資料不足"
    if score >= 75 and pd.notna(bias) and bias <= 8:
        return "偏低估／具吸引力"
    elif score >= 60:
        return "合理偏多"
    elif score >= 45:
        return "中性"
    else:
        return "偏弱／宜觀察"


# 催化因素關鍵字分類表：每個 tuple = (顯示標籤, 正面關鍵字list, 顏色)
_CATALYST_KEYWORD_RULES = [
    ("🏆 訂單/標案",  ["訂單", "標案", "得標", "得標金額", "接單", "供應", "合約", "採購"], "var(--green)"),
    ("📢 法說/財報",  ["法說", "法人說明會", "財報", "EPS", "獲利", "盈餘", "季報", "年報", "轉盈"], "var(--blue)"),
    ("🏗️ 政策/題材",  ["特別條例", "預算", "國防", "補貼", "政策", "條例", "立院", "立法院", "產業升級", "DDP"], "var(--yellow)"),
    ("📈 業績成長",   ["營收成長", "營收創高", "獲利成長", "年增", "季增", "創新高", "亮眼", "超預期"], "var(--green)"),
    ("🏭 擴產/新品",  ["擴產", "建廠", "新產品", "新機種", "量產", "導入", "出貨", "認證"], "#a78bfa"),
    ("💵 股利/除息",  ["配息", "除息", "配股", "殖利率", "股利"], "#f9a8d4"),
    ("⚠️ 法人調升",   ["調升", "目標價", "買進", "推薦", "看好", "上調", "增持"], "var(--blue)"),
    ("🔻 風險注意",   ["利空", "調降", "賣出", "下修", "虧損", "衰退", "減少", "壓力", "警示", "處置"], "var(--red)"),
]


def _rule_based_key_observations(row_data: dict, closes: pd.Series) -> list[str]:
    """把量化指標翻譯成繁體中文重點觀察條目，全部用規則產生，不呼叫AI。"""
    obs = []
    name = row_data.get("name", "")
    code = row_data.get("code", "")
    industry = row_data.get("industry", "")

    # 1. 基本定位
    obs.append(f"{name}（{code}）｜{industry}")

    # 2. 報價與52週位置
    price = row_data.get("收盤", np.nan)
    chg   = row_data.get("漲跌幅(%)", np.nan)
    if pd.notna(price) and pd.notna(chg):
        direction = "上漲" if chg >= 0 else "下跌"
        obs.append(f"現價 {price:.2f}，今日{direction} {abs(chg):.2f}%")

    if closes is not None and len(closes) >= 60:
        hi52 = float(closes.tail(252).max()) if len(closes) >= 252 else float(closes.max())
        lo52 = float(closes.tail(252).min()) if len(closes) >= 252 else float(closes.min())
        if hi52 > lo52 and pd.notna(price):
            pct_from_lo = (price - lo52) / (hi52 - lo52) * 100
            obs.append(f"52週區間 {lo52:.2f}–{hi52:.2f}，現價位於區間 {pct_from_lo:.0f}% 位置")

    # 3. EPS/本益比
    pe = row_data.get("本益比", np.nan)
    if pd.notna(pe):
        if pe > 100:
            pe_comment = "極高（題材評價為主）"
        elif pe > 40:
            pe_comment = "偏高"
        elif pe > 15:
            pe_comment = "合理"
        elif pe > 0:
            pe_comment = "偏低"
        else:
            pe_comment = "虧損中"
        obs.append(f"本益比 {pe:.1f}x，{pe_comment}")

    # 4. 營收年增/月增
    yoy = row_data.get("營收年增", np.nan)
    mom = row_data.get("營收月增", np.nan)
    if pd.notna(yoy):
        trend = "成長" if yoy > 0 else "衰退"
        obs.append(f"營收年增 {yoy:+.1f}%（{trend}）" + (f"、月增 {mom:+.1f}%" if pd.notna(mom) else ""))

    # 5. 財務評分
    score = row_data.get("AI評分", np.nan)
    if pd.notna(score):
        level = "優良" if score >= 75 else "中等" if score >= 50 else "偏弱"
        obs.append(f"財務體質評分 {score:.0f}/100（{level}）")

    # 6. RSI
    rsi = row_data.get("RSI14", np.nan)
    if pd.notna(rsi):
        if rsi >= 80:
            rsi_comment = "嚴重超買，短線需注意回落風險"
        elif rsi >= 70:
            rsi_comment = "超買區，動能強但留意壓力"
        elif rsi <= 30:
            rsi_comment = "超賣區，可能存在技術反彈機會"
        elif rsi <= 20:
            rsi_comment = "嚴重超賣"
        else:
            rsi_comment = "中性"
        obs.append(f"RSI14 = {rsi:.1f}，{rsi_comment}")

    # 7. 乖離率
    bias = row_data.get("乖離30MA(%)", np.nan)
    if pd.notna(bias):
        if bias > 15:
            bias_comment = "明顯偏高，短線存在壓力"
        elif bias > 5:
            bias_comment = "偏高"
        elif bias >= -3:
            bias_comment = "貼近均線，位置健康"
        else:
            bias_comment = "跌破均線下方"
        obs.append(f"30日乖離率 {bias:+.1f}%，{bias_comment}")

    # 8. 均線多頭/空頭
    if closes is not None and len(closes) >= 60:
        ma30 = float(closes.tail(30).mean())
        ma60 = float(closes.tail(60).mean())
        if pd.notna(price):
            if price > ma30 > ma60:
                obs.append("均線排列：多頭（價格 > MA30 > MA60）")
            elif price < ma30 < ma60:
                obs.append("均線排列：空頭（價格 < MA30 < MA60）")

    # 9. 飆股雷達
    radar = row_data.get("飆股雷達", "")
    if radar and radar != "觀察":
        obs.append(f"技術雷達：{radar}")

    return obs


def _rule_based_catalysts(news_list: list, row_data: dict) -> list[dict]:
    """用關鍵字分類新聞標題，產生催化因素條目列表。
    每筆格式：{label, titles, color}，titles 是符合該分類的標題列表。
    另外從量化指標本身（營收成長、評分等）補充結構性條目。"""
    # 從新聞標題找關鍵字符合的條目
    matched: dict[str, dict] = {}
    for n in (news_list or []):
        title = n.get("title", "")
        for label, keywords, color in _CATALYST_KEYWORD_RULES:
            if any(kw in title for kw in keywords):
                if label not in matched:
                    matched[label] = {"label": label, "titles": [], "color": color}
                if len(matched[label]["titles"]) < 3:  # 每類最多顯示3則標題
                    matched[label]["titles"].append(title)
                break  # 每則新聞只歸到第一個符合的分類

    # 從量化資料本身補充「結構性催化因素」（不依賴新聞）
    structural = []
    yoy = row_data.get("營收年增", np.nan)
    if pd.notna(yoy) and yoy >= 20:
        structural.append(f"營收年增 {yoy:+.1f}%，成長動能強勁")
    elif pd.notna(yoy) and yoy >= 5:
        structural.append(f"營收年增 {yoy:+.1f}%，穩健成長中")

    score = row_data.get("AI評分", np.nan)
    if pd.notna(score) and score >= 75:
        structural.append(f"財務體質評分 {score:.0f}/100，財務基本面扎實")

    if row_data.get("突破20日高"):
        structural.append("技術面突破近20日高點，短線動能轉強")

    if structural:
        matched["📊 量化訊號"] = {"label": "📊 量化訊號", "titles": structural, "color": "var(--blue)"}

    return list(matched.values())


def build_investment_report(row_data: dict, closes: pd.Series, industry_peers: pd.DataFrame, news_list: list) -> dict:
    """整合量化資料與規則分析，組出完整報告。所有欄位均由規則產生，不依賴外部AI。"""
    return {
        "code": row_data.get("code"), "name": row_data.get("name"), "industry": row_data.get("industry"),
        "收盤": row_data.get("收盤"), "漲跌幅(%)": row_data.get("漲跌幅(%)"),
        "本益比": row_data.get("本益比"), "財務評分": row_data.get("AI評分"),
        "營收年增": row_data.get("營收年增"), "營收月增": row_data.get("營收月增"),
        "RSI14": row_data.get("RSI14"), "MACD柱": row_data.get("MACD柱"),
        "乖離30MA(%)": row_data.get("乖離30MA(%)"), "主力成本乖離(%)": row_data.get("主力成本乖離(%)"),
        "飆股雷達": row_data.get("飆股雷達"), "估值判斷旗標": _valuation_flag(row_data),
        "重點觀察": _rule_based_key_observations(row_data, closes),
        "催化因素": _rule_based_catalysts(news_list, row_data),
        "情境價格區間": calc_volatility_price_bands(closes),
        "同業比較": industry_peers.to_dict("records") if industry_peers is not None and not industry_peers.empty else [],
        "新聞標題": [{"title": n["title"], "sentiment": n["sentiment"], "link": n.get("link", "")}
                    for n in (news_list or [])][:8],
    }





# ============================================================
# 3. 全域 CSS（TradingView 機構終端機風格）
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500;600&family=Noto+Sans+TC:wght@400;500;600;700;800&display=swap');

:root{
  --bg:#070d17;
  --surface:#0d1624;
  --surface-2:#111d2d;
  --surface-3:#162337;
  --border:rgba(148,163,184,.16);
  --border-strong:rgba(110,168,254,.38);
  --text:#e8eef8;
  --muted:#8796aa;
  --accent:#6ea8fe;
  --blue:#6ea8fe;
  --accent-strong:#3b82f6;
  --green:#36c99a;
  --red:#ff6b7a;
  --yellow:#f8c766;
  --shadow:0 12px 30px rgba(0,0,0,.18);
  --radius:16px;
}

[data-testid='stHeader'],[data-testid='stToolbar'],[data-testid='stSidebar'],
[data-testid='collapsedControl'],[data-testid='stSidebarCollapseButton']{display:none!important;}

html,body,[data-testid='stAppViewContainer'],[data-testid='stMain']{
  background:
    radial-gradient(circle at 18% -10%,rgba(59,130,246,.12),transparent 30%),
    radial-gradient(circle at 92% 0%,rgba(54,201,154,.06),transparent 24%),
    var(--bg)!important;
  color:var(--text)!important;
  font-family:'Inter','Noto Sans TC',sans-serif!important;
}

.block-container{max-width:min(2560px,100%)!important;padding:22px clamp(12px,2vw,56px) 72px!important;margin:0 auto!important;}

/* ── 頁首：單一主視覺，不再堆疊厚重卡片 ── */
.app-hero{
  display:flex;justify-content:space-between;align-items:center;gap:28px;
  padding:26px 28px;margin-bottom:12px;
  background:linear-gradient(115deg,rgba(17,29,45,.98),rgba(10,18,30,.98));
  border:1px solid var(--border);border-radius:22px;box-shadow:var(--shadow);
  position:relative;overflow:hidden;
}
.app-hero:after{content:'';position:absolute;right:-70px;top:-110px;width:250px;height:250px;border-radius:50%;background:rgba(110,168,254,.09);filter:blur(2px);pointer-events:none;}
.app-kicker{font-size:11px;font-weight:800;letter-spacing:.16em;color:var(--accent);margin-bottom:8px;}
.app-title{font-size:31px;font-weight:800;letter-spacing:-.02em;color:#f7faff;line-height:1.2;}
.app-sub{max-width:820px;font-size:13px;color:var(--muted);margin-top:9px;line-height:1.8;}
.app-meta{display:grid;grid-template-columns:repeat(2,minmax(112px,1fr));gap:10px;min-width:270px;z-index:1;}
.meta-item{padding:12px 14px;border:1px solid var(--border);border-radius:13px;background:rgba(255,255,255,.025);}
.meta-k{font-size:10px;font-weight:800;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;}
.meta-v{margin-top:5px;font-family:'Roboto Mono',monospace;font-size:14px;font-weight:700;color:#f4f8ff;white-space:nowrap;}

/* ── 精簡步驟條 ── */
.workflow{display:flex;align-items:center;gap:6px;margin:0 0 26px;padding:0 4px;overflow-x:auto;}
.workflow-step{display:flex;align-items:center;gap:7px;padding:8px 10px;color:#66778e;font-size:11px;font-weight:700;white-space:nowrap;}
.workflow-step:not(:last-child):after{content:'›';margin-left:7px;color:#35465c;font-size:15px;}
.workflow-step b{display:inline-flex;width:21px;height:21px;border-radius:50%;align-items:center;justify-content:center;background:#111c2c;border:1px solid var(--border);color:#6f8097;font-size:10px;}
.workflow-step.active{color:#cdd9e8;}.workflow-step.active b{background:rgba(110,168,254,.14);border-color:var(--border-strong);color:#9fc5ff;}

/* ── 頂層導覽 ──
   Streamlit 各版本的頁籤 DOM 不一致（.stTabs / [data-testid='stTabs'] /
   [data-baseweb='tab-list'] / role='tablist' 都出現過），只押一種寫法很容易整組落空，
   落空就會退回原生樣式：字擠在一起、底下還帶一條 primaryColor 的紅線。
   所以這裡把幾種寫法全部並列，並且直接選 button 而不是 button[role='tab']。 */
.stTabs [data-baseweb='tab-list'],
[data-testid='stTabs'] [data-baseweb='tab-list'],
[data-testid='stTabs'] [role='tablist'],
[data-baseweb='tab-list']{
  position:sticky;top:0;z-index:50;
  display:flex!important;flex-wrap:wrap!important;align-items:center!important;
  gap:8px!important;
  background:rgba(9,16,27,.92)!important;backdrop-filter:blur(14px);
  border:1px solid var(--border)!important;border-bottom:1px solid var(--border)!important;
  border-radius:15px!important;padding:7px!important;margin:0 0 28px!important;
}

/* 單顆頁籤：給足左右內距，別再讓四個標題黏成一行 */
.stTabs [data-baseweb='tab-list'] button,
[data-testid='stTabs'] [role='tablist'] button,
[data-baseweb='tab-list'] button,
[data-baseweb='tab']{
  flex:0 0 auto!important;
  height:auto!important;min-height:40px!important;
  padding:10px 20px!important;margin:0!important;
  border:0!important;border-bottom:0!important;border-radius:10px!important;
  background:transparent!important;color:#8796aa!important;
  font-size:13px!important;font-weight:700!important;white-space:nowrap!important;
  transition:background .15s ease,color .15s ease!important;
}

/* Streamlit 把標題包在 <p> 裡，<p> 自帶顏色與行距，必須明確覆蓋 */
.stTabs [data-baseweb='tab-list'] button p,
[data-testid='stTabs'] [role='tablist'] button p,
[data-baseweb='tab-list'] button p,
[data-baseweb='tab'] p{
  color:inherit!important;font-size:13px!important;font-weight:700!important;
  margin:0!important;line-height:1.2!important;letter-spacing:.01em!important;
}

.stTabs [data-baseweb='tab-list'] button:hover,
[data-testid='stTabs'] [role='tablist'] button:hover,
[data-baseweb='tab-list'] button:hover,
[data-baseweb='tab']:hover{background:rgba(255,255,255,.055)!important;color:#d9e4f2!important;}

.stTabs [data-baseweb='tab-list'] button[aria-selected='true'],
[data-testid='stTabs'] [role='tablist'] button[aria-selected='true'],
[data-baseweb='tab-list'] button[aria-selected='true'],
[data-baseweb='tab'][aria-selected='true']{
  background:var(--surface-3)!important;color:#ffffff!important;
  box-shadow:inset 0 0 0 1px rgba(110,168,254,.30)!important;
}

/* 原生的滑動底線／分隔線：紅線就是從這裡來的，一律關掉 */
.stTabs [data-baseweb='tab-highlight'],
[data-testid='stTabs'] [data-baseweb='tab-highlight'],
[data-baseweb='tab-highlight'],
.stTabs [data-baseweb='tab-border'],
[data-testid='stTabs'] [data-baseweb='tab-border'],
[data-baseweb='tab-border']{
  display:none!important;background:transparent!important;height:0!important;border:0!important;
}

[data-baseweb='tab-panel'],[data-testid='stTabs'] [role='tabpanel']{padding-top:0!important;}

/* ── 標題與留白 ── */
.section-head{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin:30px 0 14px;}
.section-title{font-size:18px;font-weight:800;letter-spacing:-.01em;color:#f3f7fd;}
.section-help{font-size:12px;color:var(--muted);line-height:1.65;}
.tv-section{font-size:15px;font-weight:800;color:#edf3fb;margin:18px 0 10px;}
.candidate-row-hint{font-size:11.5px;color:var(--muted);margin:3px 0 10px;line-height:1.6;}

/* ── 統一卡片語言：低陰影、細邊框、較大留白 ── */
.tv-panel,.tv-card,.side-card,.quote-panel,.news-card,[data-testid='stDataFrame']{
  background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important;box-shadow:none!important;
}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:0 0 20px;}
.tv-card{padding:17px 18px;min-height:104px;}
.tv-label{color:var(--muted);font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;}
.tv-value{font-family:'Roboto Mono',monospace;font-size:27px;font-weight:700;margin-top:9px;color:#f4f8ff;line-height:1.15;}
.tv-caption{color:var(--muted);font-size:11.5px;margin-top:7px;line-height:1.55;}
.side-card{padding:18px;margin-bottom:13px;}
.side-title{display:flex;justify-content:space-between;align-items:center;gap:10px;font-size:15px;font-weight:800;margin-bottom:10px;}
.report-row{display:flex;justify-content:space-between;gap:18px;border-bottom:1px solid rgba(148,163,184,.11);padding:10px 0;color:#cbd6e4;font-size:13px;}
.report-row:last-child{border-bottom:0;}
.radar-check{font-size:13px;line-height:1.95;color:#c8d4e3;}.radar-check b{color:var(--green);margin-right:7px;}

/* ── Streamlit 容器與表單 ── */
[data-testid='stVerticalBlockBorderWrapper']{background:var(--surface)!important;border-color:var(--border)!important;border-radius:var(--radius)!important;box-shadow:none!important;}
.st-key-scan_control_panel [data-testid='stVerticalBlockBorderWrapper']{padding:8px 4px 2px;}
.st-key-candidate_sidebar [data-testid='stVerticalBlockBorderWrapper']{position:sticky;top:70px;max-height:calc(100vh - 92px);overflow-y:auto;}
.control-note{padding:15px 16px;border-radius:13px;background:rgba(110,168,254,.07);border:1px solid rgba(110,168,254,.20);color:#bdcce0;font-size:13px;line-height:1.75;min-height:100%;}
.strategy-badge{display:inline-block;background:rgba(110,168,254,.11);color:#9fc5ff;border:1px solid rgba(110,168,254,.25);border-radius:999px;padding:3px 9px;font-size:9px;font-weight:800;letter-spacing:.08em;margin-left:7px;}

[data-baseweb='select']>div,[data-testid='stTextInput'] input,[data-testid='stNumberInput'] input{
  background:#0a1320!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:10px!important;min-height:42px!important;
}
[data-baseweb='select']>div:focus-within,[data-testid='stTextInput'] input:focus,[data-testid='stNumberInput'] input:focus{border-color:var(--border-strong)!important;box-shadow:0 0 0 3px rgba(59,130,246,.10)!important;}
label,[data-testid='stWidgetLabel'] p{color:#aab8ca!important;font-size:12px!important;font-weight:700!important;}
.stSlider{padding-top:4px;}

[data-testid='stButton']>button,[data-testid='stDownloadButton']>button{
  min-height:40px!important;border-radius:10px!important;border:1px solid var(--border)!important;
  background:#121f31!important;color:#e8eef8!important;font-weight:700!important;font-size:12px!important;transition:.16s ease!important;
}
[data-testid='stButton']>button:hover,[data-testid='stDownloadButton']>button:hover{border-color:rgba(110,168,254,.45)!important;background:#17283f!important;color:#fff!important;box-shadow:none!important;}
[data-testid='stButton']>button[kind='primary']{background:var(--accent-strong)!important;border-color:#5a96f5!important;color:#fff!important;}
[data-testid='stButton']>button[kind='primary']:hover{background:#4a8df5!important;}

[data-testid='stExpander']{background:transparent!important;border:0!important;margin:6px 0 14px;}
[data-testid='stExpander'] details{background:rgba(13,22,36,.62)!important;border:1px solid var(--border)!important;border-radius:14px!important;overflow:hidden;}
[data-testid='stExpander'] summary{font-weight:750!important;color:#dce6f3!important;font-size:13px!important;padding:4px 2px!important;}
[data-testid='stAlert']{border-radius:12px!important;background:rgba(110,168,254,.07)!important;border:1px solid rgba(110,168,254,.20)!important;}

/* ── 乾淨進度條 ──
   注意：不能用 [data-testid='stProgress']>div 來畫軌道。那個直接子層只是中介容器，
   把它塗成漸層會多長出一條假的滿版進度條，真正的條在下面，百分比文字還會被壓住。
   真正穩定的定位點是 role='progressbar'（軌道），它的子 div 才是填色條。 */
[data-testid='stProgress']{margin:16px 0 10px;}

/* 中介容器一律清乾淨，只留 progressbar 本身有樣式 */
[data-testid='stProgress'] > div:not([role='progressbar']){
  background:transparent!important;border:0!important;height:auto!important;
  border-radius:0!important;overflow:visible!important;box-shadow:none!important;
}

/* 百分比文字：獨立一行放在軌道上方，右對齊像儀表板讀數 */
[data-testid='stProgress'] [data-testid='stMarkdownContainer'] p,
[data-testid='stProgress'] p{
  color:#9fb2c9!important;font-weight:600!important;font-size:11.5px!important;
  font-family:'Roboto Mono',monospace!important;letter-spacing:.06em;
  margin:0 0 7px!important;line-height:1.4!important;text-align:right!important;
}

/* 軌道：細一點、內縮的膠囊 */
[data-testid='stProgress'] [role='progressbar']{
  background:#0f1a29!important;border:1px solid var(--border)!important;
  border-radius:999px!important;height:7px!important;overflow:hidden!important;
  box-shadow:inset 0 1px 2px rgba(0,0,0,.35)!important;
}

/* 填色條：藍→綠漸層，帶一點光暈與平滑推進 */
[data-testid='stProgress'] [role='progressbar'] > div{
  background:linear-gradient(90deg,var(--accent-strong),var(--green))!important;
  border-radius:999px!important;
  box-shadow:0 0 10px rgba(54,201,154,.28)!important;
  transition:width .28s cubic-bezier(.4,0,.2,1)!important;
}

/* ── 報價主卡 ── */
.quote-panel{padding:22px 24px;margin:12px 0 18px;position:relative;overflow:hidden;}
.quote-panel:before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--accent);}
.quote-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.quote-title{font-size:23px;font-weight:800;letter-spacing:-.015em;}
/* L2 狀態標籤：可掃視、不過度搶主價 */
.quote-tag,.bias-chip{
  border:1px solid var(--border);background:#121e2f;border-radius:999px;
  padding:5px 11px;color:#c5d2e3;font-size:12.5px;font-weight:700;
  line-height:1.25;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.quote-tag.earn-tag{max-width:none;font-size:13px;font-weight:800;letter-spacing:.01em;}
.quote-tag.earn-soon{
  background:rgba(242,169,0,.16)!important;border-color:rgba(242,169,0,.38)!important;color:#f5c84a!important;
}
.quote-tag.earn-later{
  background:rgba(76,141,255,.12)!important;border-color:rgba(76,141,255,.30)!important;color:#8eb6ff!important;
}
.quote-price{font-family:'Roboto Mono',monospace;font-size:42px;font-weight:700;color:var(--green);line-height:1.2;letter-spacing:-.04em;}
.quote-change{font-family:'Roboto Mono',monospace;font-size:15px;font-weight:700;margin-left:9px;}
.quote-metrics{display:grid;grid-template-columns:repeat(3,minmax(120px,1fr));gap:10px;margin-top:15px;max-width:720px;}
.quote-metrics>div{padding:10px 12px;background:rgba(255,255,255,.022);border:1px solid rgba(148,163,184,.10);border-radius:11px;}
.metric-k{color:var(--muted);font-size:11px;font-weight:700;letter-spacing:.02em;}
.metric-v{font-family:'Roboto Mono',monospace;font-size:14.5px;font-weight:700;margin-top:4px;}
.hot-tag{background:rgba(248,145,75,.12)!important;border-color:rgba(248,145,75,.32)!important;color:#ffb17a!important;}
.hot-badge-inline{display:inline-block;margin-left:4px;font-size:12px;}
.detail-nav-title{font-size:11px;font-weight:800;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;margin:4px 0 7px;}

/* ── 工作台 ── */
.workspace-left{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px;max-height:900px;overflow-y:auto;}
.workspace-right{min-height:600px;}
.workspace-left [data-testid='stDataFrame']{border:0!important;}
[data-testid='stDataFrame']{overflow:hidden!important;}
[data-testid='stDataFrame'] *{font-family:'Roboto Mono','Noto Sans TC',monospace!important;}
.news-card{padding:16px 18px!important;margin-bottom:11px!important;}.news-title:hover{color:#9fc5ff!important;}

/* ── 模組化資訊總覽（個股工作台新版面）── */
/* 垂直節奏：區塊之間固定節奏，避免「什麼都擠在一起」 */
.wb-section{margin:0 0 14px;}
.wb-section-gap{margin-top:18px;}
.wb-divider{height:1px;background:rgba(148,163,184,.12);margin:16px 0 14px;border:0;}
.wb-topbar{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;
  padding:0 0 10px;margin:0 0 10px;border-bottom:1px solid var(--border);}
.wb-topbar-title{font-size:clamp(17px,.95vw,22px);font-weight:800;color:#f3f7fd;letter-spacing:-.015em;}
.wb-topbar-sep{color:rgba(148,163,184,.35);margin:0 8px;font-weight:400;}
.wb-topbar-sub{color:var(--muted);font-weight:600;font-size:clamp(13px,.72vw,15px);}
.wb-topbar-note{font-size:clamp(11px,.58vw,13px);color:var(--muted);white-space:nowrap;}
/* 動作列：報價下方的一條工具列，不與報價搶視線 */
.wb-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 8px;}
.wb-toolbar-hint{font-size:11.5px;color:var(--muted);margin-left:auto;white-space:nowrap;}

/* 卡片等高網格 ──
   st.columns 每一欄的高度是各自獨立的，內容多寡不同就會長短不齊。
   這幾張卡片全是純 HTML／SVG（沒有 Streamlit widget），所以改成塞進
   同一個 CSS Grid：grid 的同一列預設就會 stretch 成等高，不必靠
   Streamlit 的 DOM 結構，換版本也不會壞。 */
.wb-grid6{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:clamp(8px,.7vw,16px);
  align-items:stretch;}
.wb-solo{display:grid;min-height:430px;}

/* 本益比河流圖：標題列 + 四格摘要 + 圖 + 彩色圖例 */
.wb-river-stat{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;
  background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:9px 11px;margin-bottom:7px;}
.wb-river-legend{display:flex;flex-wrap:wrap;gap:4px 10px;align-items:center;
  margin-top:6px;font-family:'Roboto Mono',monospace;color:#b3c2d4;font-size:clamp(11.5px,.62vw,14px);}
.wb-river-note{font-size:11.5px;color:var(--muted);line-height:1.65;margin-top:6px;}

/* 卡片外殼：改成 flex 直列，讓頁尾註解一律貼齊卡片底部 */
.wb-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:clamp(12px,.85vw,20px) clamp(13px,.9vw,22px);box-sizing:border-box;height:100%;
  display:flex;flex-direction:column;}
.wb-card>.wb-foot{margin-top:auto;}
.wb-card-head{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px;}
.wb-card-title{font-size:clamp(15px,.82vw,19px);font-weight:800;color:#eaf1fa;letter-spacing:-.01em;white-space:nowrap;}
.wb-card-note{font-size:clamp(11.5px,.62vw,14px);color:var(--muted);font-weight:700;white-space:nowrap;}
.wb-row{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:6.5px 0;
  border-bottom:1px solid rgba(148,163,184,.08);font-size:clamp(13px,.72vw,16px);color:#b3c2d4;}
.wb-row:last-child{border-bottom:0;}
.wb-row b{font-family:'Roboto Mono',monospace;font-weight:700;color:#eaf1fa;font-size:clamp(13.5px,.75vw,16.5px);
  white-space:nowrap;}
.wb-empty{color:var(--muted);font-size:clamp(13px,.7vw,16px);line-height:1.85;padding:14px 0;}
.wb-big{font-family:'Roboto Mono',monospace;font-size:clamp(29px,1.75vw,42px);font-weight:700;color:#f4f8ff;
  line-height:1.15;letter-spacing:-.03em;}
.wb-sub{font-size:clamp(12px,.66vw,15px);color:var(--muted);margin-top:4px;line-height:1.6;}
.wb-foot{font-size:clamp(11.5px,.62vw,14px);color:var(--muted);margin-top:9px;padding-top:8px;
  border-top:1px solid rgba(148,163,184,.09);}

/* 報價主卡（L1 現價最大、L2 標籤可掃、L3 指標等寬灰階） */
.wb-quote{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:14px 16px 15px;position:relative;overflow:hidden;}
.wb-quote:before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--accent);}
.wb-quote-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;}
.wb-quote-name{font-size:clamp(24px,1.45vw,36px);font-weight:800;letter-spacing:-.02em;color:#f4f8ff;line-height:1.15;}
.wb-quote-code{font-family:'Roboto Mono',monospace;font-size:clamp(18px,1.1vw,26px);font-weight:700;color:#8b9bb0;}
.wb-quote-body{display:flex;align-items:flex-start;gap:22px;flex-wrap:wrap;}
.wb-quote-left{flex:0 0 auto;min-width:0;}
.wb-quote-price{font-family:'Roboto Mono',monospace;font-size:clamp(40px,2.5vw,62px);font-weight:700;
  line-height:1.05;letter-spacing:-.04em;}
.wb-quote-chg{font-family:'Roboto Mono',monospace;font-size:clamp(15.5px,.95vw,22px);font-weight:700;margin-left:8px;}
.wb-mgrid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:7px 10px;flex:1 1 430px;
  align-content:start;padding-top:2px;}
.wb-mgrid>div{min-width:0;}
.wb-mk{font-size:clamp(11px,.6vw,13px);font-weight:700;color:#8b9bb0;white-space:nowrap;letter-spacing:.02em;}
.wb-mv{font-family:'Roboto Mono',monospace;font-size:clamp(14px,.78vw,17.5px);font-weight:700;color:#e9f0fa;
  margin-top:2px;white-space:nowrap;}

/* 迷你表格（三大法人逐日） */
.wb-tbl{width:100%;border-collapse:collapse;font-family:'Roboto Mono',monospace;font-size:clamp(12px,.66vw,15px);}
.wb-tbl th{color:var(--muted);font-weight:700;text-align:right;padding:4px 1px;font-size:clamp(11px,.6vw,13.5px);
  border-bottom:1px solid var(--border);}
.wb-tbl th:first-child,.wb-tbl td:first-child{text-align:left;}
.wb-tbl td{text-align:right;padding:4.5px 1px;color:#cdd9e8;
  border-bottom:1px solid rgba(148,163,184,.07);}
.wb-tbl tr:last-child td{border-bottom:0;}

/* 新聞列 */
.wb-news{display:block;text-decoration:none;padding:7px 0;
  border-bottom:1px solid rgba(148,163,184,.08);}
.wb-news:last-child{border-bottom:0;}
.wb-news-t{font-size:clamp(13px,.7vw,16px);line-height:1.55;color:#dbe6f3;font-weight:600;}
.wb-news:hover .wb-news-t{color:#9fc5ff;}
.wb-news-m{font-size:clamp(11px,.58vw,13px);color:var(--muted);margin-top:3px;font-family:'Roboto Mono',monospace;}

/* 點一下放大：純 CSS 覆蓋層（靠 tabindex + :focus，不需要 JS） */
.wb-zoom{outline:none;cursor:zoom-in;border-radius:10px;
  transition:background .15s ease;position:relative;}
.wb-zoom:hover{background:rgba(110,168,254,.07);}
.wb-zoom:hover:before{content:'⤢';position:absolute;top:2px;right:4px;
  font-size:11px;color:var(--accent);opacity:.75;}
.wb-zoom:focus{position:fixed;left:6vw;right:6vw;top:8vh;bottom:8vh;z-index:1000000;
  background:var(--surface);border:1px solid rgba(110,168,254,.55);border-radius:18px;
  box-shadow:0 26px 80px rgba(0,0,0,.7);cursor:zoom-out;
  display:flex;align-items:center;justify-content:center;padding:34px;}
.wb-zoom:focus:before{content:none;}
.wb-zoom:focus .wb-zoom-inner{width:100%;max-width:760px;}
.wb-zoom:focus svg{width:100%!important;max-width:100%!important;height:auto!important;
  max-height:70vh;}
.wb-zoom:focus:after{content:'點畫面任一處關閉';position:absolute;bottom:14px;left:0;right:0;
  text-align:center;font-size:12px;color:var(--muted);letter-spacing:.05em;}

/* 評等長條 */
.wb-bar{height:9px;border-radius:999px;background:rgba(148,163,184,.13);overflow:hidden;flex:1;}
.wb-bar>i{display:block;height:100%;border-radius:999px;}

/* 頁尾免責 */
.wb-disclaimer{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;
  margin:22px 0 4px;padding-top:13px;border-top:1px solid var(--border);
  font-size:11px;color:var(--muted);letter-spacing:.02em;}

/* 候選 chips 按鈕列：橫向捲動，一次瀏覽全部候選股 ──
   Streamlit 的 st.columns 預設是 flex:1 1 0 平均分寬，欄位一多每一欄就被壓到
   剩幾十 px。這裡把橫列改成 nowrap + overflow-x:auto，並給每一欄固定寬度，
   欄數再多也只是往右延伸、用捲軸瀏覽，不會互相擠壓。 */
.st-key-wb_chip_row [data-testid='stHorizontalBlock']{
  flex-wrap:nowrap!important;overflow-x:auto!important;overflow-y:hidden!important;
  gap:8px!important;padding:4px 2px 10px;
  scrollbar-width:thin;scrollbar-color:rgba(110,168,254,.55) rgba(148,163,184,.12);
  overscroll-behavior-x:contain;}
.st-key-wb_chip_row [data-testid='stHorizontalBlock']>div{
  flex:0 0 148px!important;width:148px!important;min-width:148px!important;max-width:148px!important;
  overflow:hidden!important;}
.st-key-wb_chip_row [data-testid='stHorizontalBlock']::-webkit-scrollbar{height:8px;}
.st-key-wb_chip_row [data-testid='stHorizontalBlock']::-webkit-scrollbar-track{
  background:rgba(148,163,184,.12);border-radius:999px;}
.st-key-wb_chip_row [data-testid='stHorizontalBlock']::-webkit-scrollbar-thumb{
  background:rgba(110,168,254,.55);border-radius:999px;}
.st-key-wb_chip_row [data-testid='stHorizontalBlock']::-webkit-scrollbar-thumb:hover{
  background:rgba(110,168,254,.8);}
.st-key-wb_chip_row [data-testid='stButton']>button{
  min-height:40px!important;height:40px!important;font-size:13px!important;font-weight:700!important;
  padding:4px 8px!important;line-height:1.25!important;
  white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important;}
/* 下方價位列：固定高度、不溢出、對比足夠 */
.wb-chip-sub{
  text-align:center;font-size:11.5px;line-height:1.3;margin:4px 0 2px;padding:0 2px;
  font-family:'Roboto Mono',monospace;font-weight:700;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.wb-chip-sub.up{color:#35c48d;}
.wb-chip-sub.down{color:#f23645;}
.wb-chip-sub.flat{color:#8b9bb0;}
.wb-chip-label{font-size:12px;font-weight:800;letter-spacing:.06em;color:var(--muted);
  margin:4px 0 6px;}

/* 窄螢幕：報價卡指標與六格模組一起收欄，維持等高不變 */
@media (max-width:1400px){.wb-grid6{grid-template-columns:repeat(3,minmax(0,1fr));}}
@media (max-width:1180px){.wb-mgrid{grid-template-columns:repeat(3,minmax(0,1fr));}}
@media (max-width:900px){.wb-grid6{grid-template-columns:repeat(2,minmax(0,1fr));}
  .wb-solo{min-height:0;}
  .wb-river-stat{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media (max-width:680px){.wb-mgrid{grid-template-columns:repeat(2,minmax(0,1fr));}
  .wb-quote-price{font-size:32px;}
  .wb-grid6{grid-template-columns:1fr;}}

/* 手機候選卡 */
.st-key-desktop_candidate_list{display:block;}.st-key-mobile_candidate_list{display:none;}
.mobile-stock-card{background:var(--surface-2);border:1px solid var(--border);border-radius:13px;padding:13px 14px;margin-bottom:9px;}
.mobile-stock-card .msc-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;}
.mobile-stock-card .msc-name{font-size:14px;font-weight:800;color:#f2f6fc;}.mobile-stock-card .msc-code{color:var(--muted);font-size:11px;margin-left:6px;}
.mobile-stock-card .msc-score{font-family:'Roboto Mono',monospace;font-weight:700;font-size:16px;color:var(--accent);}
.mobile-stock-card .msc-metrics{display:flex;gap:13px;font-family:'Roboto Mono',monospace;font-size:12px;color:#c8d4e3;}.mobile-stock-card .msc-metrics span{color:var(--muted);margin-right:3px;}

::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-track{background:#09111d}::-webkit-scrollbar-thumb{background:#26364b;border-radius:999px}::-webkit-scrollbar-thumb:hover{background:#334862}

/* 大螢幕：把固定 px 的全域元件一起放大，避免高解析度下整頁字太小 */
@media(min-width:1700px){
  .app-title{font-size:38px;}
  .app-sub{font-size:15px;max-width:1000px;}
  .app-kicker{font-size:12.5px;}
  .meta-k{font-size:11.5px;}
  .meta-v{font-size:17px;}
  .workflow-step{font-size:12.5px;padding:9px 12px;}
  .workflow-step b{width:24px;height:24px;font-size:11.5px;}
  .section-title{font-size:21px;}
  .section-help{font-size:13.5px;}
  .tv-label{font-size:11.5px;}
  .tv-value{font-size:32px;}
  .tv-caption{font-size:13px;}
  [data-testid='stTabs'] button p{font-size:16px!important;}
  [data-testid='stExpander'] summary p{font-size:15.5px!important;}
  /* 「完整分析明細」收合區裡的自訂元件 */
  .report-row{font-size:15px;padding:12px 0;}
  .tv-section{font-size:17.5px;}
  .metric-k{font-size:11.5px;}
  .control-note{font-size:15px;}
  .detail-nav-title{font-size:12.5px;}
  .candidate-row-hint{font-size:13px;}
  .radar-check{font-size:15px;}
  .strategy-badge{font-size:10.5px;padding:4px 11px;}
  .side-title{font-size:17px;}
  .quote-title{font-size:26px;}
  .quote-price{font-size:47px;}
  .quote-change{font-size:17px;}
}
@media(min-width:2300px){
  .app-title{font-size:44px;}
  .app-sub{font-size:16.5px;}
  .section-title{font-size:23px;}
  .tv-value{font-size:36px;}
  [data-testid='stTabs'] button p{font-size:18px!important;}
  [data-testid='stExpander'] summary p{font-size:17px!important;}
  .report-row{font-size:16.5px;}
  .tv-section{font-size:19px;}
  .control-note{font-size:16.5px;}
  .radar-check{font-size:16.5px;}
  .quote-title{font-size:29px;}
  .quote-price{font-size:52px;}
}

/* Streamlit 自己的元件（st.caption、st.markdown 內文、下拉選單、表格、
   欄位標籤…）全都是 rem 基準，直接調 html 的基準字級一次全部帶起來，
   不用逐個 data-testid 去猜。上面那些自訂 class 是 px，不受影響，
   所以兩邊要分開處理。 */
@media(min-width:1700px){ html{font-size:17.5px;} }
@media(min-width:2300px){ html{font-size:19px;} }

@media(max-width:1100px){
  .block-container{padding:18px 18px 52px!important;}
  .app-hero{align-items:flex-start;}.app-meta{min-width:230px;}
  .st-key-candidate_sidebar [data-testid='stVerticalBlockBorderWrapper']{position:static;max-height:none;}
}
@media(max-width:850px){
  .app-hero{display:block;padding:22px;}.app-meta{margin-top:18px;max-width:360px;min-width:0;}
  .section-help{display:none;}.quote-metrics{grid-template-columns:1fr 1fr;}
}
@media(max-width:760px){
  .block-container{padding:14px 12px 40px!important;}.app-title{font-size:25px;}.app-meta{grid-template-columns:1fr 1fr;}
  .workflow{padding:0;}.workflow-step{font-size:10px;padding:7px 5px;}.workflow-step:not(:last-child):after{margin-left:4px;}
  .stat-grid,.quote-metrics{grid-template-columns:1fr;}.quote-price{font-size:34px;}.quote-panel{padding:19px 17px;}
  .st-key-desktop_candidate_list{display:none;}.st-key-mobile_candidate_list{display:block;}
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
    <div class="app-kicker">TAIWAN EQUITY INTELLIGENCE</div>
    <div class="app-title">台股決策中心</div>
    <div class="app-sub">從全市場策略掃描，到個股技術面、基本面、法人籌碼與研究報告，集中在同一個清楚的分析工作台。</div>
  </div>
  <div class="app-meta">
    <div class="meta-item"><div class="meta-k">目前訊號</div><div class="meta-v">{_signal_count} 檔</div></div>
    <div class="meta-item"><div class="meta-k">資料時間</div><div class="meta-v">{_now_str}</div></div>
  </div>
</div>
<div class="workflow">
  <div class="workflow-step {'active' if _current_step >= 1 else ''}"><b>1</b>設定策略</div>
  <div class="workflow-step {'active' if _current_step >= 2 else ''}"><b>2</b>掃描市場</div>
  <div class="workflow-step {'active' if _current_step >= 3 else ''}"><b>3</b>挑選標的</div>
  <div class="workflow-step {'active' if _current_step >= 3 else ''}"><b>4</b>深入研究</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 5. 兩頁籤工作台（掃描 ／ 候選與分析工作台）
# ============================================================
# [新版面] 原本「候選清單／AI分析／個股新聞」三個頂層頁籤合併為一個
# 「主從式雙欄工作台」：左側候選清單常駐可見，右側個股工作台用分段
# 切換（K線圖／AI分析／個股新聞）取代原本三個各自獨立的頁籤與各自
# 獨立的 selectbox，選一次股票，三種檢視都同步。
# ------------------------------------------------------------
# [新版面] 模組化資訊總覽用的輕量 inline SVG 元件
# ------------------------------------------------------------
# 六格資訊模組每格實際只有約 170~200px 寬。Plotly 在這種寬度下
# 邊界、圖例與刻度字會互相重疊，而且一次掛六張圖表切換個股時
# 前端重繪明顯卡頓。這裡改用純字串組出來的 inline SVG：
# 沒有額外套件、沒有前端狀態，縮放時跟著卡片等比縮小。
# 全部回傳 HTML 字串，交給既有的 st.markdown(unsafe_allow_html=True) 輸出。

def wb_svg_gauge(pct, label, color="#36c99a"):
    """半圓儀表。pct 為 0~100（多方占比），label 是中央的判讀文字。"""
    pct = 0.0 if pd.isna(pct) else float(min(max(float(pct), 0.0), 100.0))
    r, cx, cy = 52.0, 70.0, 64.0
    arc = float(np.pi) * r                      # 半圓弧長
    filled = arc * pct / 100.0
    return (
        f'<svg viewBox="0 0 140 76" style="width:100%;max-width:158px;display:block;margin:2px auto 0;">'
        f'<path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}" fill="none" '
        f'stroke="rgba(148,163,184,.15)" stroke-width="10" stroke-linecap="round"/>'
        f'<path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}" fill="none" '
        f'stroke="{color}" stroke-width="10" stroke-linecap="round" '
        f'stroke-dasharray="{filled:.2f} {arc * 2:.2f}"/>'
        f'<text x="{cx}" y="{cy - 16}" text-anchor="middle" fill="{color}" '
        f'font-size="15.5" font-weight="800">{label}</text>'
        f'<text x="{cx}" y="{cy - 1}" text-anchor="middle" fill="#8796aa" font-size="13" '
        f'font-family="Roboto Mono,monospace">{pct:.0f}%</text>'
        f'</svg>'
    )


def wb_svg_radar(scores, color="#36c99a"):
    """多邊形雷達圖。scores 是 {軸名稱: 0~100} 的 dict，至少 3 軸才畫得出來。"""
    labels = [str(k) for k in scores.keys()]
    vals = [0.0 if pd.isna(v) else float(min(max(float(v), 0.0), 100.0)) for v in scores.values()]
    n = len(labels)
    if n < 3:
        return ''
    # viewBox 要留出軸標籤的寬度：最左／最右的標籤是 4 個中文字（約 40px），
    # 從圓心往外 69px，所以左右各留 120-69-40≈11px 的安全邊界，不會再被切掉。
    cx, cy, rmax = 129.0, 90.0, 56.0
    # 從正上方開始、順時針排列，跟一般財報雷達圖的閱讀習慣一致
    angles = [(-np.pi / 2) + (2 * np.pi * i / n) for i in range(n)]

    def _pt(ang, rad):
        return f"{cx + rad * float(np.cos(ang)):.1f},{cy + rad * float(np.sin(ang)):.1f}"

    grid = ''
    for ring in (0.34, 0.67, 1.0):
        pts = ' '.join(_pt(a, rmax * ring) for a in angles)
        grid += (f'<polygon points="{pts}" fill="none" stroke="rgba(148,163,184,.16)" '
                 f'stroke-width="1"/>')
    for a in angles:
        grid += (f'<line x1="{cx}" y1="{cy}" x2="{cx + rmax * float(np.cos(a)):.1f}" '
                 f'y2="{cy + rmax * float(np.sin(a)):.1f}" stroke="rgba(148,163,184,.14)" stroke-width="1"/>')

    data_pts = ' '.join(_pt(a, rmax * v / 100.0) for a, v in zip(angles, vals))
    text = ''
    for a, lb, v in zip(angles, labels, vals):
        lx = cx + (rmax + 15) * float(np.cos(a))
        ly = cy + (rmax + 15) * float(np.sin(a))
        ly = min(max(ly, 19.0), 158.0)      # 上下夾住，避免第二行的數字掉出畫布
        anchor = 'middle'
        if float(np.cos(a)) > 0.35:
            anchor = 'start'
        elif float(np.cos(a)) < -0.35:
            anchor = 'end'
        text += (f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" fill="#8796aa" '
                 f'font-size="11.5" font-weight="700">{lb}</text>'
                 f'<text x="{lx:.1f}" y="{ly + 12.5:.1f}" text-anchor="{anchor}" fill="#c8d4e3" '
                 f'font-size="11" font-family="Roboto Mono,monospace">{v:.0f}</text>')

    return (
        f'<svg viewBox="0 0 258 186" style="width:100%;display:block;margin:0 auto;">'
        f'{grid}'
        f'<polygon points="{data_pts}" fill="{color}33" stroke="{color}" stroke-width="1.7" '
        f'stroke-linejoin="round"/>'
        f'{text}</svg>'
    )


def wb_svg_sparkline(series, colors, height=52):
    """多條迷你走勢線。series 是 [[y...], [y...]]，共用同一組 min/max 做正規化。"""
    clean = []
    for s in series:
        vals = [float(v) for v in s if pd.notna(v)]
        clean.append(vals if len(vals) >= 2 else [])
    flat = [v for s in clean for v in s]
    if not flat:
        return ''
    lo, hi = min(flat), max(flat)
    span = (hi - lo) or 1.0
    w, pad = 200.0, 4.0
    paths = ''
    for vals, col in zip(clean, colors):
        if not vals:
            continue
        step = w / max(len(vals) - 1, 1)
        pts = ' '.join(
            f"{i * step:.1f},{pad + (height - 2 * pad) * (1 - (v - lo) / span):.1f}"
            for i, v in enumerate(vals)
        )
        paths += (f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8" '
                  f'stroke-linejoin="round" stroke-linecap="round"/>')
    return (f'<svg viewBox="0 0 {w:.0f} {height}" preserveAspectRatio="none" '
            f'style="width:100%;height:{height}px;display:block;">{paths}</svg>')


def wb_svg_donut(pct, center_text, color="#36c99a"):
    """環圈圖，用來表示占比。pct 為 0~100。"""
    pct = 0.0 if pd.isna(pct) else float(min(max(float(pct), 0.0), 100.0))
    r, cx, cy = 34.0, 46.0, 46.0
    circ = 2 * float(np.pi) * r
    return (
        f'<svg viewBox="0 0 92 92" style="width:100%;max-width:104px;display:block;margin:2px auto;">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(148,163,184,.15)" stroke-width="10"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="10" '
        f'stroke-linecap="round" stroke-dasharray="{circ * pct / 100:.2f} {circ:.2f}" '
        f'transform="rotate(-90 {cx} {cy})"/>'
        f'<text x="{cx}" y="{cy + 6}" text-anchor="middle" fill="#eaf1fa" font-size="18" '
        f'font-weight="700" font-family="Roboto Mono,monospace">{center_text}</text>'
        f'</svg>'
    )


def wb_financial_footnote(fin: dict, health: dict) -> str:
    """財務體質卡片的頁尾說明。

    原本寫成「2019年 年報｜採用 7 項指標」有三個問題：
      1.「2019年」＋「年報」的「年」重複。
      2. 卡片上只畫出 3 個分類，卻寫 7 項指標，看起來對不起來；
         實際上是 7 項指標先彙總成 3 個分類再畫雷達圖。
      3. 年度太舊時完全沒有提示，容易誤以為是最新財報。
    """
    raw = str(fin.get('year', '') or '')
    m = re.search(r'(\d{4})', raw)
    n_metrics = health.get('n_metrics', 0)
    n_cats = len(health.get('category_scores', {}))
    if m:
        fy = int(m.group(1))
        age = get_tw_now().year - fy
        if age >= 3:
            head = (f'<span style="color:var(--yellow);">財報年度 {fy}'
                    f'（已 {age} 年未更新）</span>')
        else:
            head = f'財報年度 {fy}'
    else:
        head = '財報年度不明'
    return (f'<div class="wb-foot">{head}<br>'
            f'{n_metrics} 項指標彙總為 {n_cats} 個分類</div>')


def wb_zoomable(inner_html, hint="點一下放大"):
    """把圖表包成「點一下放大」的容器。

    Streamlit 的 st.markdown 不會執行 JS，而這幾張卡片是純 HTML 塞在同一個
    CSS Grid 裡（為了等高），也不能中途插 st.button。所以改用純 CSS：
    給 div 一個 tabindex 讓它可以被點擊聚焦，再用 :focus 把它變成 position:fixed
    的全螢幕覆蓋層；點畫面其他地方會自動失焦收起來。不依賴任何腳本。
    """
    return (f'<div class="wb-zoom" tabindex="0" role="button" title="{hint}">'
            f'<div class="wb-zoom-inner">{inner_html}</div></div>')


def wb_bar_row(label, pct, color="#36c99a"):
    """評等分布用的水平長條列。"""
    pct = 0.0 if pd.isna(pct) else float(min(max(float(pct), 0.0), 100.0))
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:7px 0;">'
        f'<span style="font-size:12.5px;color:#b9c7d8;width:34px;flex:none;">{label}</span>'
        f'<span class="wb-bar"><i style="width:{pct:.0f}%;background:{color};"></i></span>'
        f'<span style="font-size:12.5px;color:#cdd9e8;font-family:Roboto Mono,monospace;'
        f'width:32px;text-align:right;flex:none;">{pct:.0f}%</span></div>'
    )



user_bias = st.session_state.user_bias
user_vol = st.session_state.user_vol

tab_scan, tab_workspace, tab_portfolio, tab_watchlist, tab_report, tab_market = st.tabs([
    "市場掃描",
    "個股工作台",
    "持倉中心",
    "追蹤清單",
    "研究報告",
    "市場觀察",
])

# ------------------------------------------------------------
# TAB 1：選股掃描
# ------------------------------------------------------------
with tab_scan:
    st.markdown('<div class="section-head"><div><div class="section-title">設定掃描條件</div><div class="section-help">條件越嚴格，候選股票通常越少；第一次使用可保留預設值。</div></div></div>', unsafe_allow_html=True)
    with st.container(border=True, key="scan_control_panel"):
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
            _update_progress(0.80, f"已找到 {len(initial_hits)} 檔候選股，正在補齊本益比、營收與財務體質評分…")
            load_official_pe_map(False)
            hot_codes = get_hot_stock_tickers()  # [新功能] 熱門股判定：整批只抓一次排行榜，逐檔比對代碼

            def _fetch_deep_and_health(ticker: str, code: str):
                """[新功能] 候選股清單分數改用財務體質綜合分：跟本益比／營收
                一起在同一個 worker 裡依序抓取，不額外開一個執行緒池，
                避免同時對兩個資料來源發出過多平行請求（fail-open：任一步
                失敗都不影響另一步的結果）。"""
                return fetch_deep_info(ticker), fetch_financial_health_score(code)

            final_list = []
            with ThreadPoolExecutor(max_workers=6) as ex:
                f_deep = {ex.submit(_fetch_deep_and_health, r["ticker"], r["code"]): r for r in initial_hits}
                for j, f in enumerate(as_completed(f_deep), 1):
                    _update_progress(
                        0.80 + 0.19 * j / len(initial_hits),
                        f"步驟 3/3：補齊個股資料中… ({j}/{len(initial_hits)})"
                    )
                    deep_res, health_score = f.result()
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
                        "ATR20": base.get("ATR20", np.nan), "ATR比例(%)": base.get("ATR比例(%)", np.nan),
                        "突破20日高": base.get("突破20日高", False), "接近60日高": base.get("接近60日高", False),
                        "策略": base.get("策略", active_strategy), "訊號說明": base.get("訊號說明", ""),
                        "熱門股": base["code"] in hot_codes,
                        "本益比": deep_res["pe"], "營收月增": deep_res["mom"], "營收年增": deep_res["yoy"],
                    }
                    _, radar = calc_stock_score(row_data)
                    row_data["AI評分"] = health_score
                    row_data["飆股雷達"] = "、".join(radar) if radar else "觀察"
                    final_list.append(row_data)
            _update_progress(1.0, "掃描完成，正在整理結果…")
            st.session_state.scan_results = pd.DataFrame(final_list).sort_values("AI評分", ascending=False, na_position="last").reset_index(drop=True)
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
        <div class="stat-grid" style="margin-top:20px;">
          <div class="tv-card"><div class="tv-label">符合條件</div><div class="tv-value">{len(scan_df)}</div><div class="tv-caption">本次掃描候選股票</div></div>
          <div class="tv-card"><div class="tv-label">平均財務評分</div><div class="tv-value">{'N/A' if pd.isna(avg_score) else f'{avg_score:.0f}'}</div><div class="tv-caption">候選標的平均分數</div></div>
          <div class="tv-card"><div class="tv-label">高分候選</div><div class="tv-value">{strong_count}</div><div class="tv-caption">財務評分 80 分以上</div></div>
        </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.is_scanning:
        st.markdown("""
        <div class="tv-panel" style="text-align:center;padding:42px 22px;margin-top:18px;">
          <div style="font-size:40px;margin-bottom:12px;">🔎</div>
          <div style="font-size:20px;font-weight:800;color:#f4f8ff;">尚未產生掃描結果</div>
          <div class="tv-caption" style="margin-top:9px;line-height:1.8;">設定條件後按下「開始全市場掃描」。<br>完成後請切換到「個股工作台」頁籤。</div>
        </div>
        """, unsafe_allow_html=True)

# 共用結果與目前股票：合併「策略掃描結果」跟「手動查詢股票」，
# 兩種來源共用同一份 current_idx / current_stock，個股工作台的程式碼完全不用區分來源。
has_results = isinstance(st.session_state.scan_results, pd.DataFrame) and not st.session_state.scan_results.empty
has_manual = bool(st.session_state.manual_stocks)
if has_results or has_manual:
    scan_df = st.session_state.scan_results.copy() if has_results else pd.DataFrame()
    manual_df = pd.DataFrame(st.session_state.manual_stocks) if has_manual else pd.DataFrame()
    if not manual_df.empty and not scan_df.empty:
        # 手動查詢的股票如果剛好也在掃描候選清單裡，優先用掃描結果那筆
        # （欄位較完整、含策略訊號），避免同一檔股票出現兩筆重複列。
        manual_df = manual_df[~manual_df["code"].isin(scan_df["code"])]
    df = pd.concat([scan_df, manual_df], ignore_index=True) if not manual_df.empty else scan_df
    total_found = len(df)
    if st.session_state.current_idx >= total_found:
        st.session_state.current_idx = 0
    if st.session_state.manual_current_code:  # 剛手動查到一檔股票，切過去給使用者看
        matches = df.index[df['code'].astype(str) == st.session_state.manual_current_code].tolist()
        if matches:
            st.session_state.current_idx = matches[0]
        st.session_state.manual_current_code = None
    current_stock = df.iloc[st.session_state.current_idx]
else:
    df = pd.DataFrame()
    total_found = 0
    current_stock = None

def render_manual_search_box(key_suffix: str, with_border: bool = True):
    """手動查詢個股輸入框：不需策略掃描，直接輸入代碼即可查看完整工作台。
    抽成函式方便在不同位置重複渲染（例如右欄工作台頂端、或完全沒有候選股票時的
    空狀態），用 key_suffix 讓每個位置的 widget key 不會互相衝突。
    """
    ctx = (st.expander("快速查詢其他股票", expanded=False)
           if key_suffix == "panel"
           else st.container(border=with_border, key=f"manual_search_{key_suffix}"))
    with ctx:
        st.markdown('<div class="candidate-row-hint">輸入上市／上櫃股票代碼，即可直接開啟完整個股分析，不必重新執行全市場掃描。</div>', unsafe_allow_html=True)
        msc1, msc2 = st.columns([3, 1])
        with msc1:
            manual_code_input = st.text_input(
                "股票代碼", key=f"manual_stock_code_input_{key_suffix}", placeholder="輸入代碼，例如：2330、2454",
                label_visibility="collapsed",
            )
        with msc2:
            manual_search_clicked = st.button("🔍 查詢", key=f"btn_manual_search_{key_suffix}", use_container_width=True)
        if manual_search_clicked:
            code_input = manual_code_input.strip()
            if not code_input:
                st.warning("請先輸入股票代碼。")
            else:
                with st.spinner(f"查詢 {code_input} 中…"):
                    manual_result = build_manual_stock_row(code_input)
                if manual_result is None:
                    st.error(f"找不到股票代碼「{code_input}」，請確認代碼是否正確（僅支援上市／上櫃股票）。")
                else:
                    # 避免同一檔股票在清單裡重複；最新查詢的排最前面，最多保留 20 檔。
                    st.session_state.manual_stocks = [
                        r for r in st.session_state.manual_stocks if r["code"] != manual_result["code"]
                    ]
                    st.session_state.manual_stocks.insert(0, manual_result)
                    st.session_state.manual_stocks = st.session_state.manual_stocks[:20]
                    st.session_state.manual_current_code = manual_result["code"]
                    st.rerun()


# ------------------------------------------------------------
# TAB 2：候選與分析工作台（主從式雙欄：左清單常駐 + 右側分段切換）
# ------------------------------------------------------------
with tab_workspace:
    if has_results:
        avg_score = scan_df['AI評分'].mean() if 'AI評分' in scan_df.columns else np.nan
        strong_count = int((scan_df['AI評分'] >= 80).sum()) if 'AI評分' in scan_df.columns else 0
        st.markdown(f"""
        <div class="stat-grid">
          <div class="tv-card"><div class="tv-label">候選股票</div><div class="tv-value">{len(scan_df)}</div><div class="tv-caption">目前策略掃描結果</div></div>
          <div class="tv-card"><div class="tv-label">平均財務評分</div><div class="tv-value">{'N/A' if pd.isna(avg_score) else f'{avg_score:.0f}'}</div><div class="tv-caption">候選標的整體品質</div></div>
          <div class="tv-card"><div class="tv-label">高分候選</div><div class="tv-value">{strong_count}</div><div class="tv-caption">財務評分 80 分以上</div></div>
        </div>
        """, unsafe_allow_html=True)

        # ══════════════ 熱門族群：改為收合區，避免一進工作台就被大量卡片占滿 ══════════════
        with st.expander("🔥 熱門族群與產業集中度", expanded=False):
            render_hot_industries(scan_df)

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
                with st.spinner(f"回測中（下載候選清單 {len(scan_df)} 檔股票近9個月資料）..."):
                    bt_df = run_strategy_backtest(bt_strategy, tuple(scan_df['ticker'].tolist()), float(bt_param), int(user_vol))
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

    if has_results or has_manual:
        # [新功能] 風險與部位設定：這兩個參數不影響選股條件，只影響「停損價」與
        # 「建議張數」怎麼算，所以放在掃描之外，改動後即時反映、不需要重新掃描。
        with st.expander("⚙️ 風險與部位設定（停損價與建議張數的計算基礎）", expanded=False):
            rc1, rc2, rc3 = st.columns([1, 1, 1.3], gap="large")
            with rc1:
                st.session_state.risk_budget = st.number_input(
                    "單筆可承受風險 (元)", min_value=1000, max_value=1000000,
                    value=int(st.session_state.get("risk_budget", 30000)), step=1000,
                    key="input_risk_budget",
                    help="這一筆交易若走到停損，你願意賠掉多少錢。建議抓總資金的 1~2%。",
                )
            with rc2:
                st.session_state.atr_stop_mult = st.number_input(
                    "ATR 停損倍數", min_value=0.5, max_value=5.0,
                    value=float(st.session_state.get("atr_stop_mult", 2.0)), step=0.5,
                    key="input_atr_mult",
                    help="停損價 = 現價 − 倍數 × ATR20。倍數越大停損越寬、越不容易被洗掉，但單張風險也越高。常用 2~3 倍。",
                )
            with rc3:
                st.session_state.show_position_cols = st.checkbox(
                    "在候選清單顯示「停損／張數」兩欄", 
                    value=bool(st.session_state.get("show_position_cols", True)),
                    key="input_show_pos_cols",
                    help="表格欄位較多時可以關掉，右側報價卡仍會顯示。",
                )
                st.caption(
                    f"目前設定：單筆最多賠 {int(st.session_state.risk_budget):,} 元，"
                    f"停損放在 {st.session_state.atr_stop_mult:g} 倍 ATR 之外。"
                )
            st.caption(
                "張數是由風險反推的：波動大的股票會自動建議買少一點，讓每一筆做錯時的損失都控制在同一個數字。"
                "ATR 用 20 日 Wilder 平滑，已把跳空缺口計入。"
            )

        # [新版面] 改為全寬單欄儀表板：候選清單移到頂部橫向 chips ＋ 下方收合表格，
        # 空出整個畫面寬度給六格資訊模組。right_col 保留成 container，
        # 下方既有程式碼的縮排與 with 區塊都不需要動。
        right_col = st.container()

        # ══════════════ 候選股票清單：完整表格移到收合區 ══════════════
        # 快速篩選、產業篩選、CSV 下載、桌面表格點列選股、手機卡片清單全部保留，
        # 只是預設收合，日常切換個股改用上方的候選 chips。
        with st.expander("📋 候選股票清單（快速篩選・完整表格・CSV 下載）", expanded=False):
            with st.container(border=True, key="candidate_sidebar"):
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
                    "快速篩選", ["全部候選", "財務評分 80 分以上", "營收年增為正", "量比 1.5 倍以上", "突破 20 日高"],
                    key="candidate_filter"
                )
                view_df = df.copy()
                if quick_filter == "財務評分 80 分以上":
                    view_df = view_df[view_df['AI評分'] >= 80]
                elif quick_filter == "營收年增為正":
                    view_df = view_df[pd.to_numeric(view_df['營收年增'], errors='coerce') > 0]
                elif quick_filter == "量比 1.5 倍以上":
                    view_df = view_df[pd.to_numeric(view_df['量比20日'], errors='coerce') >= 1.5]
                elif quick_filter == "突破 20 日高":
                    view_df = view_df[view_df['突破20日高'] == True]
                if st.session_state.get('industry_filter'):
                    view_df = view_df[view_df['industry'] == st.session_state.industry_filter]

                # [新功能] 依目前風險設定即時算出每一檔的停損價與建議張數。
                # 掃描結果只存 ATR20，所以調整設定不必重掃，這裡重算就好。
                _mult = float(st.session_state.get("atr_stop_mult", 2.0))
                _budget = float(st.session_state.get("risk_budget", 30000))
                _plans = [
                    calc_position_plan(
                        r.get("收盤", np.nan), r.get("ATR20", np.nan), _mult, _budget,
                        is_etf=is_etf_instrument(r.to_dict()),
                    )
                    for _, r in view_df.iterrows()
                ]
                view_df = view_df.copy()
                view_df["停損價"] = [pl["停損價"] for pl in _plans]
                view_df["停損幅度(%)"] = [pl["停損幅度(%)"] for pl in _plans]
                view_df["建議張數"] = [pl["建議張數"] for pl in _plans]

                csv = view_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("下載目前清單 CSV", csv, f'tw_stock_scan_{get_tw_now().strftime("%Y%m%d")}.csv', 'text/csv', use_container_width=True)

                # [新功能] 桌面版：完整表格。用 st.container(key=...) 讓 CSS 依螢幕寬度
                # 切換顯示／隱藏（st-key-desktop_candidate_list），手機版改顯示下方卡片清單。
                with st.container(key="desktop_candidate_list"):
                    show_cols = ["code", "name", "AI評分", "收盤", "漲跌幅(%)", "量比20日"]
                    if st.session_state.get("show_position_cols", True):
                        show_cols += ["停損價", "建議張數"]
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
                            "名稱": st.column_config.TextColumn("名稱", width=110),
                            "AI評分": st.column_config.ProgressColumn("財務評分", width=72, format="%d", min_value=0, max_value=100),
                            "收盤": st.column_config.NumberColumn("價格", width=58, format="%.2f"),
                            "漲跌幅(%)": st.column_config.NumberColumn("漲跌", width=56, format="%.1f%%"),
                            "量比20日": st.column_config.NumberColumn("量比", width=52, format="%.2fx"),
                            # [新功能] ATR 停損價與由風險反推的建議張數
                            "停損價": st.column_config.NumberColumn(
                                "停損", width=58, format="%.2f",
                                help="現價 − 倍數 × ATR20。跌破代表這次判斷已經錯了，不要凹。"),
                            "建議張數": st.column_config.NumberColumn(
                                "張數", width=52, format="%d",
                                help="以「單筆可承受風險 ÷ 每張到停損的虧損」反推。0 張代表這檔波動太大，用目前的風險預算買不下去。"),
                        }
                    )
                    if event and "selection" in event and event["selection"]["rows"]:
                        clicked_view_row = event["selection"]["rows"][0]
                        # [修正] st.dataframe 的選取狀態會在任何重跑（包含切換右側檢視分頁、
                        # 手動查詢股票）時持續回傳同一個值，不是只有「真的點了新的一列」才會有值。
                        # 如果每次都無條件套用，會把手動查詢或上一檔/下一檔剛設定好的 current_idx
                        # 蓋回這個表格上次點擊過的那一列。這裡改成只有「這次選取列跟上次記錄的不一樣」
                        # （代表使用者剛剛真的點了新的一列）才更新 current_idx。
                        if clicked_view_row != st.session_state.last_selected_row:
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
                          <div class="msc-metrics" style="margin-top:6px;">
                            <div><span>停損</span>{fmt_num(row.get('停損價', np.nan), '{:.2f}')}</div>
                            <div><span>張數</span>{fmt_num(row.get('建議張數', np.nan), '{:.0f}')}</div>
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
            # ══════════════ 模組化資訊總覽（全寬儀表板版面）══════════════
            # 版面順序：頂欄 → 報價卡＋動作區／候選 chips
            #           → K線走勢 ＋ 交易計畫 ＋ 持倉摘要 ＋ 風險設定（4 欄）
            #           → 多空／財務／三大法人／資券／目標價／新聞（6 格資訊模組）
            #           → 交易計畫表單、完整分析明細（皆為收合區）
            # 六格模組同時抓 6 個資料來源，全部都走既有的 @st.cache_data，
            # 同一檔股票在 TTL 內重複切換不會重打。
            _code = str(current_stock['code'])
            _mkt = "TW" if str(current_stock['ticker']).endswith(".TW") else "TWO"

            st.markdown(
                '<div class="wb-topbar">'
                '<div class="wb-topbar-title">個股工作台'
                '<span class="wb-topbar-sep">·</span>'
                '<span class="wb-topbar-sub">報價／圖表／計畫／分析</span></div>'
                '<div class="wb-topbar-note">TWSE／TPEx · Yahoo · Anue</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            render_manual_search_box(key_suffix="panel")

            # ---------- 共用序列：報價卡的 OHLC、多空指標、資券走勢都吃這一份 ----------
            _k_src = get_kline_data(_code, _mkt)

            def _k_at(col, idx=-1):
                """安全取用 K 線序列的單一數值；資料不足時回 NaN（fail-open）。"""
                try:
                    return float(_k_src[col].iloc[idx])
                except Exception:
                    return np.nan

            price = current_stock.get('收盤', np.nan)
            chg = current_stock.get('漲跌幅(%)', np.nan)
            _is_up = bool(pd.notna(chg) and chg >= 0)
            chg_color = 'var(--green)' if _is_up else 'var(--red)'
            chg_txt = 'N/A' if pd.isna(chg) else f"{chg:+.2f}%"
            _prev_close = _k_at('close', -2)
            if pd.isna(_prev_close) and pd.notna(price) and pd.notna(chg) and chg != -100:
                _prev_close = price / (1 + chg / 100)
            _chg_amt = np.nan if (pd.isna(_prev_close) or pd.isna(price)) else price - _prev_close
            _chg_amt_txt = 'N/A' if pd.isna(_chg_amt) else f"{_chg_amt:+.2f}"
            _arrow = '▲' if _is_up else '▼'

            _vol_lots = current_stock.get('成交量(張)', np.nan)
            _turnover = np.nan if (pd.isna(_vol_lots) or pd.isna(price)) else _vol_lots * 1000.0 * price
            _turnover_txt = 'N/A' if pd.isna(_turnover) else f"{_turnover / 1e8:.2f} 億"
            _close_5d = _k_at('close', -6)     # 5 個交易日前，用來算週漲跌
            _wk_amt = np.nan if (pd.isna(_close_5d) or pd.isna(price)) else price - _close_5d
            _wk_pct = (np.nan if (pd.isna(_close_5d) or _close_5d <= 0 or pd.isna(price))
                       else (price - _close_5d) / _close_5d * 100)
            _wk_color = 'var(--green)' if (pd.notna(_wk_pct) and _wk_pct >= 0) else 'var(--red)'

            # ATR 停損與建議張數：沿用側邊的風險設定，改參數不必重跑掃描
            _current_is_etf = is_etf_instrument(current_stock.to_dict())
            _risk_budget = float(st.session_state.get('risk_budget', 30000))
            _atr_mult = float(st.session_state.get('atr_stop_mult', 2.0))
            _plan = calc_position_plan(
                price, current_stock.get('ATR20', np.nan),
                _atr_mult, _risk_budget, is_etf=_current_is_etf,
            )
            _lots = _plan['建議張數']
            _shares_sug = _plan.get('建議股數', np.nan)
            if pd.isna(_lots):
                _lots_txt, _lots_color = 'N/A', '#e9f0fa'
            elif _lots > 0:
                _lots_txt, _lots_color = f"{int(_lots):,} 張", '#e9f0fa'
            elif pd.notna(_shares_sug) and _shares_sug > 0:
                _lots_txt, _lots_color = f"{int(_shares_sug):,} 股", 'var(--yellow)'
            else:
                _lots_txt, _lots_color = '0', 'var(--red)'

            _score_val = current_stock.get('AI評分', np.nan)
            _hot_tag = '<div class="quote-tag hot-tag">🔥 熱門股</div>' if bool(current_stock.get('熱門股', False)) else ''

            # [新功能] 距離下次法說會／財報相關事件天數
            _cal_suffix = "TW" if current_stock['ticker'].endswith(".TW") else "TWO"
            _cal_rows = fetch_stock_calendar(current_stock['code'], _cal_suffix)
            _next_earn = get_next_earnings_related_event(_cal_rows)
            _earn_tag = ""
            if _next_earn and pd.notna(_next_earn.get("距離天數")) and _next_earn["距離天數"] >= 0:
                _d = int(_next_earn["距離天數"])
                _evt = _next_earn.get("事件", "")
                if _d == 0:
                    _earn_tag = f'<div class="quote-tag earn-tag earn-soon" title="今日有{_evt}">📅 {_evt} 今天</div>'
                elif _d <= 14:
                    _earn_tag = f'<div class="quote-tag earn-tag earn-soon" title="{_d} 天後{_evt}">📅 {_evt} {_d}天</div>'
                else:
                    _earn_tag = f'<div class="quote-tag earn-tag earn-later" title="{_d} 天後{_evt}">📅 {_evt} {_d}天</div>'

            # ══════════════ 區塊 A：報價主卡（全寬，視線焦點）══════════════
            _metrics = [
                ("開盤", fmt_num(_k_at('open'), '{:.2f}'), ''),
                ("最高", fmt_num(_k_at('high'), '{:.2f}'), 'var(--green)'),
                ("最低", fmt_num(_k_at('low'), '{:.2f}'), 'var(--red)'),
                ("昨收", fmt_num(_prev_close, '{:.2f}'), ''),
                ("成交量", f"{fmt_num(_vol_lots, '{:,.0f}')} 張", ''),
                ("成交金額", _turnover_txt, ''),
                ("週漲跌", fmt_num(_wk_amt, '{:+.2f}'), _wk_color),
                ("週漲幅", fmt_num(_wk_pct, '{:+.2f}%'), _wk_color),
                ("量比20日", fmt_num(current_stock.get('量比20日', np.nan), '{:.2f}x'), ''),
                ("財務評分", 'N/A' if pd.isna(_score_val) else f"{int(_score_val)}", ''),
                ("ATR停損", fmt_num(_plan['停損價'], '{:.2f}'), 'var(--red)'),
                ("建議張數", _lots_txt, _lots_color),
            ]
            _mhtml = ''
            for _mk, _mv, _mc in _metrics:
                _mstyle = f' style="color:{_mc};"' if _mc else ''
                _mhtml += (f'<div><div class="wb-mk">{_mk}</div>'
                           f'<div class="wb-mv"{_mstyle}>{_mv}</div></div>')
            st.markdown(f"""
            <div class="wb-quote wb-section">
              <div class="wb-quote-head">
                <span class="wb-quote-name">{current_stock['name']}</span>
                <span class="wb-quote-code">{current_stock['code']}</span>
                <div class="quote-tag">{current_stock.get('市場別', '')}</div>
                <div class="quote-tag" title="{current_stock.get('industry', '未分類')}">{current_stock.get('industry', '未分類')}</div>{_hot_tag}{_earn_tag}
              </div>
              <div class="wb-quote-body">
                <div class="wb-quote-left">
                  <div class="wb-quote-price" style="color:{chg_color};">{fmt_num(price, '{:.2f}')}
                    <span class="wb-quote-chg">{_arrow} {_chg_amt_txt} ({chg_txt})</span></div>
                  <div class="wb-sub">更新時間 {get_tw_now().strftime('%m/%d %H:%M')}</div>
                </div>
                <div class="wb-mgrid">{_mhtml}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ══════════════ 區塊 B：動作工具列（固定四鍵，不與報價並排搶位）══════════════
            _tb1, _tb2, _tb3, _tb4 = st.columns([1, 1, 1, 1], gap="small")
            with _tb1:
                _wl_now = load_watchlist()
                _in_wl = is_in_watchlist(_code, _wl_now)
                if st.button("★ 已加自選" if _in_wl else "☆ 加入自選",
                             use_container_width=True, key="toggle_watchlist",
                             type="primary" if _in_wl else "secondary"):
                    if _in_wl:
                        remove_from_watchlist(_code)
                    else:
                        add_to_watchlist(current_stock.to_dict())
                    st.session_state.watchlist_quotes = pd.DataFrame()
                    st.rerun()
            with _tb2:
                if st.button("📋 交易計畫", use_container_width=True, key="wb_toggle_plan"):
                    st.session_state.wb_open_plan = not bool(st.session_state.get('wb_open_plan', False))
                    st.rerun()
            with _tb3:
                if st.button("← 上一檔", use_container_width=True, key="chart_prev"):
                    st.session_state.current_idx = (st.session_state.current_idx - 1) % total_found
                    st.rerun()
            with _tb4:
                if st.button("下一檔 →", use_container_width=True, key="chart_next"):
                    st.session_state.current_idx = (st.session_state.current_idx + 1) % total_found
                    st.rerun()

            # ══════════════ 區塊 C：候選股橫向 chips（獨立一列，可捲動切股）══════════════
            _max_chips = 150
            _n_chips = min(total_found, _max_chips)
            _more_txt = f'（僅列前 {_max_chips} 檔，其餘請用下方完整表格）' if total_found > _max_chips else ''
            st.markdown(
                f'<div class="wb-chip-label">候選股 {total_found} 檔 · 左右捲動切換{_more_txt}</div>',
                unsafe_allow_html=True)
            with st.container(key="wb_chip_row"):
                _chip_cols = st.columns(_n_chips) if _n_chips > 0 else []
                for _pos in range(_n_chips):
                    _crow = df.iloc[_pos]
                    _cchg = _crow.get('漲跌幅(%)', np.nan)
                    _cname = str(_crow.get('name', '') or '')
                    if len(_cname) > 5:
                        _cname = _cname[:5] + '…'
                    _clabel = f"{_cname} {_crow['code']}"
                    _cprice = fmt_num(_crow.get("收盤", np.nan), "{:.2f}")
                    _cpct = fmt_num(_cchg, "{:+.2f}%")
                    if pd.isna(_cchg):
                        _ccls = "flat"
                    elif _cchg >= 0:
                        _ccls = "up"
                    else:
                        _ccls = "down"
                    with _chip_cols[_pos]:
                        if st.button(_clabel, key=f"wb_chip_{_pos}_{_crow['code']}",
                                     use_container_width=True,
                                     type="primary" if _pos == st.session_state.current_idx else "secondary",
                                     help=f"{_crow.get('name', '')} ({_crow['code']})"):
                            st.session_state.current_idx = _pos
                            st.rerun()
                        st.markdown(
                            f'<div class="wb-chip-sub {_ccls}" title="{_crow.get("name", "")} {_cprice} {_cpct}">'
                            f'{_cprice}  {_cpct}</div>',
                            unsafe_allow_html=True)

            st.markdown('<div class="wb-divider"></div>', unsafe_allow_html=True)

            # ══════════════ 區塊 D：K線 ＋ 本益比河流（主圖區）══════════════
            st.markdown(
                '<div class="wb-chip-label" style="margin-top:2px;">圖表區 · K線與評價</div>',
                unsafe_allow_html=True)
            kl_col, river_col = st.columns([3.0, 2.2], gap="small")

            with kl_col:
                _kh1, _kh2 = st.columns([1.6, 2.4])
                with _kh1:
                    st.markdown('<div class="wb-card-title" style="padding-top:9px;">K線走勢</div>',
                                unsafe_allow_html=True)
                with _kh2:
                    show_adjusted = st.toggle(
                        "還原股價（除權息）", key="kline_show_adjusted",
                        help="開啟後改用還原除權息的股價繪製K線與均線（跟策略掃描使用的價格序列一致），"
                             "可以避免除息當天的價格缺口讓均線／型態短暫失真；關閉則顯示交易所公告的原始成交價。"
                             "抓不到還原資料時會自動退回原始股價。",
                    )
                k_fig = draw_k_line(current_stock['ticker'], current_stock['name'],
                                    chart_mode='K線圖', chart_period='日', adjusted=show_adjusted)
                if k_fig:
                    render_kline_chart_with_axis_price(k_fig, height=390)
                    try:
                        preload_tickers = [df.iloc[(st.session_state.current_idx + offset) % total_found]['ticker']
                                           for offset in (-1, 1)]
                        warm_kline_data_async(preload_tickers)
                    except Exception:
                        pass
                else:
                    st.warning("無法載入 K 線資料，請稍後再試。")

            # ---------- 交易計畫卡：先組好字串，下排跟其他模組一起排進 .wb-grid6 ----------
            # （與下方「交易計畫／建立持倉」表單同一組計算）
            _tp = build_trade_plan(current_stock.to_dict(), _risk_budget, _atr_mult)
            _tp_stop = _tp.get('stop', np.nan)
            _tp_t1 = _tp.get('target1', np.nan)
            _stop_pct = (np.nan if (pd.isna(_tp_stop) or pd.isna(price) or price <= 0)
                         else (_tp_stop - price) / price * 100)
            _t1_pct = (np.nan if (pd.isna(_tp_t1) or pd.isna(price) or price <= 0)
                       else (_tp_t1 - price) / price * 100)
            _strategy_txt = str(current_stock.get('策略', st.session_state.get('scan_strategy_used', '手動查詢')))
            _plan_card = (
                '<div class="wb-card">'
                '<div class="wb-card-head"><div class="wb-card-title">交易計畫</div></div>'
                f'<div class="wb-row"><span>策略方向</span><b>{_strategy_txt}</b></div>'
                f'<div class="wb-row"><span>理想進場區</span>'
                f'<b>{fmt_num(_tp.get("entry_low"), "{:.2f}")} ~ {fmt_num(_tp.get("entry_high"), "{:.2f}")}</b></div>'
                f'<div class="wb-row"><span>追價上限</span><b>{fmt_num(_tp.get("chase_limit"), "{:.2f}")}</b></div>'
                f'<div class="wb-row"><span>停損價</span><b style="color:var(--red);">'
                f'{fmt_num(_tp_stop, "{:.2f}")} ({fmt_num(_stop_pct, "{:.1f}%")})</b></div>'
                f'<div class="wb-row"><span>第一目標</span><b style="color:var(--green);">'
                f'{fmt_num(_tp_t1, "{:.2f}")} ({fmt_num(_t1_pct, "{:+.1f}%")})</b></div>'
                f'<div class="wb-row"><span>第二目標</span><b style="color:var(--green);">'
                f'{fmt_num(_tp.get("target2"), "{:.2f}")}</b></div>'
                f'<div class="wb-row"><span>風險報酬比</span>'
                f'<b>1 : {fmt_num(_tp.get("rr1"), "{:.1f}")}／1 : {fmt_num(_tp.get("rr2"), "{:.1f}")}</b></div>'
                '<div class="wb-foot">價位已依台股升降單位修正為可委託檔位</div>'
                '</div>'
            )


            with river_col:
                # ---------- 本益比河流圖 ----------
                # 用近四季 EPS 滾動加總（TTM）× 這檔股票「自己」歷史本益比的分位數，
                # 還原出隨獲利成長而移動的評價區間，再把實際股價疊上去。
                # 兩個資料來源（季度 EPS、3 年日線）都有 24 小時快取，切換個股時
                # 第一次會慢一點，之後同一天內都直接命中快取。
                _rh1, _rh2 = st.columns([1.9, 1.5])
                with _rh1:
                    st.markdown('<div class="wb-card-title" style="padding-top:9px;">本益比河流圖</div>',
                                unsafe_allow_html=True)
                with _rh2:
                    st.link_button(
                        "🔗 Goodinfo 官方圖",
                        f"https://goodinfo.tw/tw/ShowK_ChartFlow.asp?RPT_CAT=PER&STOCK_ID={_code}",
                        use_container_width=True,
                    )

                _river = build_pe_river_data(_code, _mkt, current_stock['ticker'])
                if _river:
                    _rdf = _river['df']
                    _bands = _river['bands']
                    _latest_pe = _river.get('latest_pe', np.nan)
                    _latest_close = float(_rdf['close'].iloc[-1])

                    # 目前本益比落在自身歷史的百分位：這是河流圖真正要回答的問題
                    _pe_ser = _rdf['本益比'].replace([np.inf, -np.inf], np.nan).dropna()
                    _pe_ser = _pe_ser[(_pe_ser > 0) & (_pe_ser < 200)]
                    _pe_pct = (float((_pe_ser < _latest_pe).mean() * 100)
                               if (len(_pe_ser) and pd.notna(_latest_pe)) else np.nan)
                    if pd.isna(_pe_pct):
                        _pos_txt, _pos_color = 'N/A', '#e9f0fa'
                    elif _pe_pct < 30:
                        _pos_txt, _pos_color = f'偏低 {_pe_pct:.0f}%', 'var(--green)'
                    elif _pe_pct <= 70:
                        _pos_txt, _pos_color = f'中間 {_pe_pct:.0f}%', '#8eb6ff'
                    else:
                        _pos_txt, _pos_color = f'偏高 {_pe_pct:.0f}%', 'var(--red)'

                    _rstats = [
                        ('目前股價', f'{_latest_close:.2f}', ''),
                        ('本益比TTM', f'{fmt_num(_latest_pe, "{:.1f}")}x', ''),
                        ('歷史區間', f'{_river["pe_min"]:.1f}~{_river["pe_max"]:.1f}', ''),
                        ('相對位置', _pos_txt, _pos_color),
                    ]
                    _rhtml = ''
                    for _rk, _rv, _rc in _rstats:
                        _rs = f' style="color:{_rc};"' if _rc else ''
                        _rhtml += (f'<div><div class="wb-mk">{_rk}</div>'
                                   f'<div class="wb-mv"{_rs}>{_rv}</div></div>')
                    st.markdown(f'<div class="wb-river-stat">{_rhtml}</div>', unsafe_allow_html=True)

                    # 由低到高疊出「河流」：相鄰兩條分位線之間填色，
                    # 綠（便宜）→ 藍 → 黃 → 紅（貴），比五條裸線好讀得多。
                    _band_labels = sorted(_bands, key=lambda k: _bands[k])
                    _fills = ['rgba(54,201,154,.20)', 'rgba(110,168,254,.16)',
                              'rgba(248,199,102,.15)', 'rgba(255,107,122,.17)']
                    _lines = ['#36c99a', '#6ea8fe', '#8eb6ff', '#f8c766', '#ff6b7a']
                    _fig = go.Figure()
                    for _bi, _lb in enumerate(_band_labels):
                        _fig.add_trace(go.Scatter(
                            x=_rdf['date'], y=_rdf[_lb], mode='lines',
                            name=f'{_lb} {_bands[_lb]:.1f}x',
                            line=dict(width=1.0, color=_lines[_bi % len(_lines)]),
                            fill=None if _bi == 0 else 'tonexty',
                            fillcolor=None if _bi == 0 else _fills[(_bi - 1) % len(_fills)],
                            hovertemplate=f'{_lb} {_bands[_lb]:.1f}x<br>%{{y:.2f}}<extra></extra>',
                        ))
                    _fig.add_trace(go.Scatter(
                        x=_rdf['date'], y=_rdf['close'], mode='lines', name='實際股價',
                        line=dict(width=2.2, color='#e6edf3'),
                        hovertemplate='%{x|%Y-%m-%d}<br>股價 %{y:.2f}<extra></extra>',
                    ))
                    _fig.update_layout(
                        height=296, template='plotly_dark', showlegend=False,
                        paper_bgcolor='#0d1624', plot_bgcolor='#0d1624',
                        margin=dict(l=6, r=6, t=6, b=6),
                        xaxis=dict(gridcolor='rgba(148,163,184,0.08)'),
                        yaxis=dict(gridcolor='rgba(148,163,184,0.08)', tickfont=dict(size=10)),
                        hovermode='x unified',
                    )
                    st.plotly_chart(_fig, use_container_width=True,
                                    key=f"pe_river_panel_{_code}")

                    # 圖太窄放不下 Plotly 圖例，改用一行彩色文字說明各區間
                    _legend = '　'.join(
                        f'<span style="color:{_lines[_i % len(_lines)]};">━</span>'
                        f'<span style="font-size:9.5px;">{_lb.split("(")[0]} {_bands[_lb]:.0f}x</span>'
                        for _i, _lb in enumerate(_band_labels)
                    )
                    # 只留彩色圖例（看圖要用），移除底下那段長說明。
                    # 完整的計算方式與免責說明仍在「完整分析明細 → 本益比河流圖」。
                    st.markdown(
                        f'<div class="wb-river-legend">{_legend}　'
                        f'<span style="color:#e6edf3;">━</span>'
                        f'<span>實際股價</span></div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="wb-card" style="min-height:372px;">'
                        '<div class="wb-empty">資料不足以計算本益比河流圖。<br><br>'
                        '需要至少 5 季 EPS（滾動出 2 個以上 TTM 資料點）與 3 年日線；'
                        '新股、EPS 揭露筆數太少、或本業虧損（TTM EPS ≤ 0）都會落到這裡。'
                        '<br><br>可改用右上角的 Goodinfo 官方河流圖。</div></div>',
                        unsafe_allow_html=True)

            # ══════════════ 區塊 E：資訊模組（交易計畫＋多空／財務／法人等）══════════════
            st.markdown(
                '<div class="wb-divider"></div>'
                '<div class="wb-chip-label">資訊模組 · 計畫與籌碼面摘要</div>',
                unsafe_allow_html=True)
            _mods = [_plan_card]

            # ---------- ① 多空指標 ----------
            _inds = calc_bull_bear_indicators(_k_src)
            if _inds:
                _sum = summarize_bull_bear(_inds)
                _bull_pct = (_sum['bull'] / _sum['total'] * 100) if _sum['total'] else 0.0
                _vcolor = {'偏多': '#36c99a', '偏空': '#ff6b7a', '多空拉鋸': '#f8c766'}.get(_sum['verdict'], '#8796aa')
                _rows = ''
                for _ind in _inds[:5]:
                    _c = {'多': 'var(--green)', '空': 'var(--red)', '中性': 'var(--muted)'}.get(_ind['訊號'], 'var(--muted)')
                    _rows += (f'<div class="wb-row"><span>{_ind["指標"]}</span>'
                              f'<b style="color:{_c};">{_ind["訊號"]}方</b></div>')
                _body = (wb_zoomable(wb_svg_gauge(_bull_pct, _sum['verdict'], _vcolor),
                                 '點一下放大多空儀表') + _rows
                         + f'<div class="wb-foot">看多 {_sum["bull"]}／看空 {_sum["bear"]}／'
                           f'中性 {_sum["neutral"]}（共 {_sum["total"]} 項）</div>')
            else:
                _body = '<div class="wb-empty">歷史資料不足 60 個交易日，無法計算多空指標。</div>'
            _mods.append('<div class="wb-card"><div class="wb-card-head">'
                         f'<div class="wb-card-title">多空指標</div></div>{_body}</div>')

            # ---------- ② 財務體質 ----------
            _fin = fetch_financial_ratios(_code)
            _health = calc_financial_health_score(_fin.get('metrics', {}))
            if _health:
                _hcolor = {'var(--green)': '#36c99a', 'var(--yellow)': '#f8c766',
                           'var(--red)': '#ff6b7a'}.get(_health['color'], '#8eb6ff')
                # 分類少於 3 種時畫不出多邊形（例如財報只解析到償債＋獲利兩類），
                # 這時退回條列顯示，不要讓卡片中間空一塊。
                _radar_html = wb_svg_radar(_health['category_scores'], _hcolor)
                if _radar_html:
                    _radar_html = wb_zoomable(_radar_html, '點一下放大財務體質雷達圖')
                if not _radar_html:
                    _radar_html = ''.join(
                        f'<div class="wb-row"><span>{_cat}</span><b>{_sc:.0f}</b></div>'
                        for _cat, _sc in _health['category_scores'].items()
                    )
                _body = (
                    f'<div style="text-align:center;margin-bottom:6px;">'
                    f'<span class="wb-big" style="color:{_hcolor};">{_health["overall"]:.0f}</span>'
                    f'<span style="font-size:12px;color:var(--muted);"> / 100</span>'
                    f'<div class="wb-sub">{_health["verdict"]}</div></div>'
                    f'{_radar_html}'
                    + wb_financial_footnote(_fin, _health)
                )
            else:
                _body = ('<div class="wb-empty">查無財務比率資料，可能是新股、金融／保險等'
                         '特殊財報格式，或來源暫時無回應。</div>')
            _mods.append('<div class="wb-card"><div class="wb-card-head">'
                         f'<div class="wb-card-title">財務體質</div></div>{_body}</div>')

            # ---------- ③ 三大法人 ----------
            _inst = fetch_institutional_trading(_code, _mkt)
            if _inst:
                _trs = ''
                for _r in _inst[:5]:
                    _tds = ''
                    for _f in ('外資', '投信', '自營商', '合計'):
                        _v = _r.get(_f, np.nan)
                        _c = '#36c99a' if (pd.notna(_v) and _v >= 0) else '#ff6b7a'
                        _tds += f'<td style="color:{_c};">{fmt_num(_v, "{:+,.0f}")}</td>'
                    _trs += f'<tr><td>{str(_r.get("日期", ""))[-5:]}</td>{_tds}</tr>'
                _cum = sum(_r.get('合計', 0) or 0 for _r in _inst[:5] if pd.notna(_r.get('合計')))
                _cum_c = 'var(--green)' if _cum >= 0 else 'var(--red)'
                _body = (
                    '<table class="wb-tbl"><thead><tr><th>日期</th><th>外資</th><th>投信</th>'
                    f'<th>自營</th><th>合計</th></tr></thead><tbody>{_trs}</tbody></table>'
                    '<div class="wb-row" style="margin-top:8px;border-top:1px solid var(--border);">'
                    f'<span>近5日累計</span><b style="color:{_cum_c};">{_cum:+,.0f} 張</b></div>'
                )
            else:
                _body = '<div class="wb-empty">查無三大法人資料，可能是新股或來源暫時無回應。</div>'
            _mods.append('<div class="wb-card"><div class="wb-card-head">'
                         '<div class="wb-card-title">三大法人</div>'
                         f'<div class="wb-card-note">近5日</div></div>{_body}</div>')

            # ---------- ④ 資券變化 ----------
            _mg = fetch_margin_trading(_code, _mkt)
            if _mg:
                _l = _mg[0]
                _mbal = _l.get('融資餘額', np.nan)
                _sbal = _l.get('融券餘額', np.nan)
                _ratio = ((_sbal / _mbal * 100) if (pd.notna(_mbal) and pd.notna(_sbal) and _mbal > 0)
                          else np.nan)
                _hist = list(reversed(_mg[:30]))
                _spark = wb_svg_sparkline(
                    [[r.get('融資餘額', np.nan) for r in _hist],
                     [r.get('融券餘額', np.nan) for r in _hist]],
                    ['#6ea8fe', '#f8c766'], height=48)
                _spark = wb_zoomable(_spark, '點一下放大資券走勢')
                _mc = 'var(--green)' if (pd.notna(_l.get('融資增減')) and _l['融資增減'] >= 0) else 'var(--red)'
                _sc = 'var(--green)' if (pd.notna(_l.get('融券增減')) and _l['融券增減'] >= 0) else 'var(--red)'
                _body = (
                    f'<div class="wb-row"><span>融資餘額</span><b>{fmt_num(_mbal, "{:,.0f}")} 張</b></div>'
                    f'<div class="wb-row"><span>融資增減</span>'
                    f'<b style="color:{_mc};">{fmt_num(_l.get("融資增減"), "{:+,.0f}")}</b></div>'
                    f'<div class="wb-row"><span>融券餘額</span><b>{fmt_num(_sbal, "{:,.0f}")} 張</b></div>'
                    f'<div class="wb-row"><span>融券增減</span>'
                    f'<b style="color:{_sc};">{fmt_num(_l.get("融券增減"), "{:+,.0f}")}</b></div>'
                    f'<div class="wb-row"><span>券資比</span><b>{fmt_num(_ratio, "{:.2f}%")}</b></div>'
                    f'<div style="margin-top:9px;">{_spark}</div>'
                    f'<div class="wb-foot"><span style="color:#6ea8fe;">━</span> 融資餘額　'
                    f'<span style="color:#f8c766;">━</span> 融券餘額（近 {len(_hist)} 日）</div>'
                )
            else:
                _body = '<div class="wb-empty">查無資券資料，可能是無信用交易資格或來源暫時無回應。</div>'
            _mods.append('<div class="wb-card"><div class="wb-card-head">'
                         '<div class="wb-card-title">資券變化</div>'
                         f'<div class="wb-card-note">近30日</div></div>{_body}</div>')

            # ---------- ⑤ 法人目標價 ----------
            _tgt = fetch_analyst_target_price(_code, _mkt)
            _tsum = summarize_target_price(_tgt, price) if _tgt else {}
            if _tsum:
                _up = _tsum.get('upside_pct', np.nan)
                _upc = 'var(--green)' if (pd.notna(_up) and _up >= 0) else 'var(--red)'
                # 依「新評等」文字歸類成買進／持有／中立三檔，算出近期評等分布
                _buy = _hold = _neu = 0
                for _r in _tgt[:12]:
                    _txt = str(_r.get('新評等', ''))
                    if any(w in _txt for w in ('買進', '買超', '增持', '優於', '強力', 'Buy', 'Outperform')):
                        _buy += 1
                    elif any(w in _txt for w in ('持有', '中立', '同步', 'Hold', 'Neutral')):
                        _hold += 1
                    else:
                        _neu += 1
                _tot = max(_buy + _hold + _neu, 1)
                _body = (
                    f'<div style="text-align:center;margin-bottom:4px;">'
                    f'<span class="wb-big">{fmt_num(_tsum["avg_target"], "{:.2f}")}</span>'
                    f'<div class="wb-sub">近 {_tsum["n_ratings"]} 筆平均目標價</div></div>'
                    f'<div style="text-align:center;font-family:Roboto Mono,monospace;'
                    f'font-size:17px;font-weight:700;color:{_upc};margin-bottom:8px;">'
                    f'{fmt_num(_up, "{:+.2f}%")}</div>'
                    f'{wb_bar_row("買進", _buy / _tot * 100, "#36c99a")}'
                    f'{wb_bar_row("持有", _hold / _tot * 100, "#6ea8fe")}'
                    f'{wb_bar_row("中立", _neu / _tot * 100, "#8796aa")}'
                    f'<div class="wb-row" style="margin-top:6px;"><span>最高／最低</span>'
                    f'<b>{fmt_num(_tsum["max_target"], "{:.1f}")} / {fmt_num(_tsum["min_target"], "{:.1f}")}</b></div>'
                    f'<div class="wb-foot">最新 {_tsum["latest_broker"]}｜{_tsum["latest_date"]}｜來源 {_tsum.get("source", "Anue")}</div>'
                )
            else:
                _body = '<div class="wb-empty">近期無外資／券商評等紀錄，或來源暫時無回應。</div>'
            _mods.append('<div class="wb-card"><div class="wb-card-head">'
                         f'<div class="wb-card-title">法人目標價</div></div>{_body}</div>')

            st.markdown(f'<div class="wb-grid6">{"".join(_mods)}</div>', unsafe_allow_html=True)

            st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)






            # [新功能] 交易計畫：沿用上方 ATR 停損與風險預算，可直接建立／更新持倉。
            with st.expander("🧭 交易計畫／建立持倉",
                             expanded=bool(st.session_state.get('wb_open_plan', False))):
                _trade_plan = build_trade_plan(
                    current_stock.to_dict(),
                    float(st.session_state.get('risk_budget', 30000)),
                    float(st.session_state.get('atr_stop_mult', 2.0)),
                )
                tp1, tp2, tp3, tp4 = st.columns(4)

                def _render_plan_price(label: str, value: str, badge: str = '', compact: bool = False):
                    badge_html = (f'<div style="display:inline-block;margin-top:7px;padding:2px 9px;border-radius:999px;'
                                  f'background:rgba(53,196,141,.18);color:#35c48d;font-size:13px;font-weight:800;">↑ {badge}</div>') if badge else ''
                    font_size = 'clamp(25px,2.35vw,38px)' if compact else 'clamp(28px,2.6vw,40px)'
                    st.markdown(
                        f'<div style="min-width:0;padding:4px 0 14px;overflow:visible;">'
                        f'<div style="font-size:14px;color:#91a7c4;margin-bottom:5px;">{label}</div>'
                        f'<div style="font-size:{font_size};line-height:1.12;color:#f4f8ff;white-space:nowrap;'
                        f'letter-spacing:-1.2px;font-variant-numeric:tabular-nums;overflow:visible;">{value}</div>'
                        f'{badge_html}</div>',
                        unsafe_allow_html=True,
                    )

                with tp1:
                    _render_plan_price(
                        "理想進場區",
                        f"{fmt_num(_trade_plan.get('entry_low'))}～{fmt_num(_trade_plan.get('entry_high'))}",
                        compact=True,
                    )
                with tp2:
                    _render_plan_price("追價上限", fmt_num(_trade_plan.get('chase_limit')))
                with tp3:
                    _render_plan_price("第一目標", fmt_num(_trade_plan.get('target1')), "1.5R")
                with tp4:
                    _render_plan_price("第二目標", fmt_num(_trade_plan.get('target2')), "3.0R")

                st.caption("以上價位已依台股股票／ETF 的最小升降單位，自動修正為可實際委託的價格檔位。")
                _default_shares = _trade_plan.get('建議股數', 1)
                if pd.isna(_default_shares) or _default_shares < 1:
                    _default_shares = 1
                _existing_position = find_open_position(current_stock['code'])
                _fallback_stop = _safe_number(_trade_plan.get('stop'), price * 0.95)
                _fallback_target1 = _safe_number(_trade_plan.get('target1'), price + (price - _fallback_stop) * 1.5)
                _fallback_target2 = _safe_number(_trade_plan.get('target2'), price + (price - _fallback_stop) * 3.0)

                # 舊版若已存入不符合升降單位的價格，載入表單時也一併校正。
                _entry_default = round_to_tw_tick(
                    _safe_number(_existing_position.get('entry_price'), price) if _existing_position else price,
                    _current_is_etf,
                )
                _stop_default = round_to_tw_tick(
                    _safe_number(_existing_position.get('stop_price'), _fallback_stop) if _existing_position else _fallback_stop,
                    _current_is_etf,
                )
                _target1_default = round_to_tw_tick(
                    _safe_number(_existing_position.get('target1'), _fallback_target1) if _existing_position else _fallback_target1,
                    _current_is_etf,
                )
                _target2_default = round_to_tw_tick(
                    _safe_number(_existing_position.get('target2'), _fallback_target2) if _existing_position else _fallback_target2,
                    _current_is_etf,
                )
                _entry_tick = get_tw_price_tick(_entry_default, _current_is_etf)
                _stop_tick = get_tw_price_tick(_stop_default, _current_is_etf)
                _target1_tick = get_tw_price_tick(_target1_default, _current_is_etf)
                _target2_tick = get_tw_price_tick(_target2_default, _current_is_etf)

                if _existing_position:
                    st.info("這檔股票已有未平倉紀錄；儲存後會更新原持倉，不會建立重複部位。")

                with st.form(key=f"trade_plan_form_{current_stock['code']}"):
                    f1, f2, f3 = st.columns(3)
                    with f1:
                        entry_price_input = st.number_input(
                            "實際／預計進場價", min_value=0.01,
                            value=float(_entry_default),
                            step=_entry_tick, format="%.2f",
                        )
                        shares_input = st.number_input(
                            "持有股數", min_value=1,
                            value=int(_existing_position.get('shares', _default_shares) if _existing_position else _default_shares),
                            step=1,
                        )
                    with f2:
                        stop_input = st.number_input(
                            "停損價", min_value=0.01,
                            value=float(_stop_default),
                            step=_stop_tick, format="%.2f",
                        )
                        target1_input = st.number_input(
                            "第一目標價", min_value=0.01,
                            value=float(_target1_default),
                            step=_target1_tick, format="%.2f",
                        )
                    with f3:
                        target2_input = st.number_input(
                            "第二目標價", min_value=0.01,
                            value=float(_target2_default),
                            step=_target2_tick, format="%.2f",
                        )
                        entry_date_input = st.date_input(
                            "進場日期",
                            value=pd.to_datetime(_existing_position.get('entry_date')).date() if _existing_position and _existing_position.get('entry_date') else get_tw_now().date(),
                        )
                    strategy_input = st.text_input(
                        "交易策略",
                        value=str(_existing_position.get('strategy', current_stock.get('策略', st.session_state.get('scan_strategy_used', '手動交易'))) if _existing_position else current_stock.get('策略', st.session_state.get('scan_strategy_used', '手動交易'))),
                    )
                    note_input = st.text_area(
                        "進場理由／失效條件",
                        value=str(_existing_position.get('note', current_stock.get('訊號說明', '')) if _existing_position else current_stock.get('訊號說明', '')),
                        placeholder="例如：守住 MA20 與前波突破點；跌破停損不凹單。",
                    )
                    telegram_alert_input = st.checkbox(
                        "跌破停損時發送 Telegram",
                        value=bool(_existing_position.get('telegram_alert', True) if _existing_position else True),
                    )
                    save_trade_clicked = st.form_submit_button(
                        "💼 建立／更新持倉", type="primary", use_container_width=True
                    )
                    if save_trade_clicked:
                        _entry_valid = round_to_tw_tick(entry_price_input, _current_is_etf)
                        _stop_valid = round_to_tw_tick(stop_input, _current_is_etf)
                        _target1_valid = round_to_tw_tick(target1_input, _current_is_etf)
                        _target2_valid = round_to_tw_tick(target2_input, _current_is_etf)
                        if _stop_valid >= _entry_valid:
                            st.error("停損價必須低於進場價。")
                        elif _target1_valid <= _entry_valid or _target2_valid <= _target1_valid:
                            st.error("第一目標必須高於進場價，第二目標必須高於第一目標。")
                        else:
                            position_payload = {
                                'code': str(current_stock['code']), 'name': current_stock['name'],
                                'ticker': current_stock['ticker'], 'industry': current_stock.get('industry', '未分類'),
                                '市場別': current_stock.get('市場別', ''),
                                'entry_date': entry_date_input.strftime('%Y-%m-%d'),
                                'entry_price': _entry_valid, 'shares': int(shares_input),
                                'stop_price': _stop_valid,
                                'target1': _target1_valid, 'target2': _target2_valid,
                                'strategy': strategy_input.strip(), 'note': note_input.strip(),
                                'telegram_alert': bool(telegram_alert_input),
                            }
                            ok, message = upsert_position(position_payload)
                            if ok:
                                st.session_state.portfolio_quotes = pd.DataFrame()
                                st.success(f"{message}：{current_stock['name']} ({current_stock['code']})")
                            else:
                                st.error(message)

            # ══════════════ 完整分析明細：原本的九選一詳細內容 ══════════════
            # 上方六格模組是「一眼看完」的摘要；這裡完整保留原本的
            # K線／多空／公司／財務／股利／法人／資券／目標價／新聞九個檢視，
            # 內容一行未改，只是收進來不佔畫面。
            #
            # 切換個股時：分析內容選項要保留（不要每次回到 K線圖）。
            # 原因：selectbox 若只放在 collapsed expander 內，部分 Streamlit 版本
            # 在 widget 未渲染的 run 會丢掉狀態；因此用獨立的 detail_view_pref 保存。
            view_options = [
                "📈 K線圖", "📐 多空指標", "🏢 公司資訊", "📅 重要行事曆",
                "🩺 財務體質", "💵 股利政策", "💰 三大法人", "📊 資券變化",
                "🎯 法人目標價", "📰 個股新聞",
            ]
            if "detail_view_pref" not in st.session_state:
                st.session_state.detail_view_pref = view_options[0]
            if st.session_state.detail_view_pref not in view_options:
                st.session_state.detail_view_pref = view_options[0]

            # 分析內容選單放在 expander 外，確保切換個股時 widget 仍會渲染、選項不會被重置
            st.markdown('<div class="detail-nav-title">分析內容</div>', unsafe_allow_html=True)
            if "detail_view_mode" not in st.session_state:
                st.session_state.detail_view_mode = st.session_state.detail_view_pref
            if st.session_state.detail_view_mode not in view_options:
                st.session_state.detail_view_mode = st.session_state.detail_view_pref
            view_mode = st.selectbox(
                "分析內容", view_options,
                key="detail_view_mode",
                label_visibility="collapsed",
            )
            st.session_state.detail_view_pref = view_mode

            with st.expander(
                f"🔍 完整分析明細 · 目前：{view_mode}",
                expanded=True,
            ):

                # ---------- K 線圖 ----------
                if view_mode == "📈 K線圖":
                    # [新版面] 「還原股價」開關已移到上方的「K線走勢」模組。
                    # 同一個 widget key 不能在畫面上出現兩次，這裡改成讀同一份
                    # session_state 值，兩邊永遠一致。
                    _adj_now = bool(st.session_state.get("kline_show_adjusted", False))
                    st.caption(
                        f"目前繪圖使用{'還原除權息股價' if _adj_now else '交易所原始成交價'}；"
                        "要切換請用上方「K線走勢」模組的「還原股價（除權息）」開關。"
                    )
                    if False:  # 保留原本的說明文字，方便日後查閱設計意圖
                        show_adjusted = st.toggle(
                            "還原股價（除權息）",
                            help="開啟後改用還原除權息的股價繪製K線與均線（跟策略掃描使用的價格序列一致），"
                                 "可以避免除息當天的價格缺口讓均線／型態短暫失真；關閉則顯示交易所公告的原始成交價。"
                                 "抓不到還原資料時會自動退回原始股價。",
                        )
                    k_fig = draw_k_line(current_stock['ticker'], current_stock['name'], chart_mode='K線圖', chart_period='日', adjusted=_adj_now)
                    if k_fig:
                        render_kline_chart_with_axis_price(k_fig, height=560)
                        try:
                            preload_tickers = [df.iloc[(st.session_state.current_idx + offset) % total_found]['ticker'] for offset in (-1, 1)]
                            warm_kline_data_async(preload_tickers)
                        except Exception:
                            pass
                    else:
                        st.warning("無法載入 K 線資料，請稍後再試。")

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

                # ---------- 重要行事曆（法說會／除權息／股東會）----------
                elif view_mode == "📅 重要行事曆":
                    market_suffix = "TW" if current_stock['ticker'].endswith(".TW") else "TWO"
                    st.markdown(f'<div class="section-head" style="margin-top:4px;"><div><div class="section-title" style="font-size:15px;">{current_stock["name"]} ({current_stock["code"]}) 重要行事曆</div><div class="section-help">資料來源：Yahoo 股市個股「行事曆」頁面；法說會通常與財報公布高度相關，僅供參考，實際日期以公司公告為準。</div></div></div>', unsafe_allow_html=True)

                    cal_rows = fetch_stock_calendar(current_stock['code'], market_suffix)
                    next_event = get_next_earnings_related_event(cal_rows)

                    if next_event:
                        days = next_event.get("距離天數", np.nan)
                        if pd.notna(days) and days >= 0:
                            if days == 0:
                                days_txt, days_color = "今天", "var(--yellow)"
                            elif days <= 7:
                                days_txt, days_color = f"{int(days)} 天", "var(--yellow)"
                            else:
                                days_txt, days_color = f"{int(days)} 天", "#8eb6ff"
                        else:
                            days_txt, days_color = "N/A", "#8f9bad"
                        st.markdown(f"""
                        <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:14px;">
                          <div class="tv-card"><div class="tv-label">下次重要事件</div><div class="tv-value" style="font-size:18px;">{next_event.get('事件', 'N/A')}</div><div class="tv-caption">{next_event.get('日期', '')}</div></div>
                          <div class="tv-card"><div class="tv-label">距離天數</div><div class="tv-value" style="color:{days_color};">{days_txt}</div><div class="tv-caption">以今日為基準</div></div>
                          <div class="tv-card"><div class="tv-label">說明</div><div class="tv-value" style="font-size:15px;">{"法說會 ≈ 財報時點" if next_event.get('事件') == '法說會' else next_event.get('事件', '')}</div><div class="tv-caption">優先顯示法說會</div></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("目前無法取得近期重要事件，可能是資料來源暫時無回應，或近期沒有已公告的法說會／除權息／股東會。")

                    if cal_rows:
                        future_rows = [r for r in cal_rows if r.get("距離天數", -999) >= -7]
                        if future_rows:
                            cal_df = pd.DataFrame(future_rows)
                            st.dataframe(
                                cal_df, hide_index=True, use_container_width=True,
                                column_config={
                                    "日期": st.column_config.TextColumn("日期", width=100),
                                    "事件": st.column_config.TextColumn("事件類型", width=100),
                                    "距離天數": st.column_config.NumberColumn("距離（天）", width=90, format="%+d"),
                                }
                            )
                            st.caption("正數＝未來天數，0＝今天，負數＝已過。")
                    else:
                        st.caption("若持續無法取得資料，可直接到 Yahoo 股市個股頁面查看最新行事曆。")

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

                            eps_styled = eps_df.style.map(_pct_color_style, subset=["季增率(%)", "年增率(%)"])
                            st.dataframe(
                                eps_styled, hide_index=True, use_container_width=True,
                                column_config={
                                    "年季": st.column_config.TextColumn("年季", width=80),
                                    "EPS": st.column_config.NumberColumn("EPS(元)", width=80, format="%.2f"),
                                    "季增率(%)": st.column_config.NumberColumn("季增率", width=80, format="%+.1f%%"),
                                    "年增率(%)": st.column_config.NumberColumn("年增率", width=80, format="%+.1f%%"),
                                    "季均價": st.column_config.NumberColumn("季均價", width=80, format="%.2f"),
                                }
                            )
                        else:
                            st.info("目前無法取得單季 EPS 資料，可能是新股、金融股（財報格式不同）或資料來源暫時無回應。")

                        # [新功能] 本益比河流圖：Yahoo 官方河流圖需登入 VIP 才能看，這裡用近四季
                        # EPS(TTM) × 自身歷史本益比分位數自行還原，屬估算版本，僅供參考。
                        # 旁邊並排一顆連結按鈕，直接開 Goodinfo 官方河流圖頁面讓使用者自行比對
                        # （純外部連結、開新分頁，不爬取／不截圖對方內容，避免反爬蟲與版權疑慮）。
                        river_head_col, river_link_col = st.columns([4, 1.3])
                        with river_head_col:
                            st.markdown('<div class="section-head" style="margin-top:22px;"><div><div class="section-title" style="font-size:15px;">本益比河流圖</div><div class="section-help">用近四季 EPS 滾動加總（TTM）× 自身歷史本益比分位數還原評價區間；估算版本，非官方資料，僅供參考、非投資建議。</div></div></div>', unsafe_allow_html=True)
                        with river_link_col:
                            st.link_button(
                                "🔗 Goodinfo 官方河流圖",
                                f"https://goodinfo.tw/tw/ShowK_ChartFlow.asp?RPT_CAT=PER&STOCK_ID={current_stock['code']}",
                                use_container_width=True,
                            )
                        pe_river = build_pe_river_data(current_stock['code'], market_suffix, current_stock['ticker'])
                        if pe_river:
                            river_df = pe_river['df']
                            bands = pe_river['bands']
                            latest_pe = pe_river.get('latest_pe', np.nan)
                            latest_close = float(river_df['close'].iloc[-1])

                            st.markdown(f"""
                            <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:10px;">
                              <div class="tv-card"><div class="tv-label">目前股價</div><div class="tv-value">{latest_close:.2f}</div><div class="tv-caption">最新收盤</div></div>
                              <div class="tv-card"><div class="tv-label">目前本益比（TTM）</div><div class="tv-value">{fmt_num(latest_pe, '{:.1f}')}</div><div class="tv-caption">股價 ÷ 近四季EPS</div></div>
                              <div class="tv-card"><div class="tv-label">近期本益比區間</div><div class="tv-value" style="font-size:18px;">{pe_river['pe_min']:.1f} ~ {pe_river['pe_max']:.1f}</div><div class="tv-caption">資料涵蓋範圍內</div></div>
                            </div>
                            """, unsafe_allow_html=True)

                            band_colors = ['#f23645', '#f2a900', '#35c48d', '#2962ff', '#a855f7']
                            band_labels = list(bands.keys())
                            fig = go.Figure()
                            for i, label in enumerate(band_labels):
                                fig.add_trace(go.Scatter(
                                    x=river_df['date'], y=river_df[label], mode='lines',
                                    name=f"{label} {bands[label]:.1f}x",
                                    line=dict(width=1.1, color=band_colors[i % len(band_colors)]),
                                    hovertemplate=f'{label} {bands[label]:.1f}x<br>' + '%{y:.2f}<extra></extra>',
                                ))
                            fig.add_trace(go.Scatter(
                                x=river_df['date'], y=river_df['close'], mode='lines', name='實際股價',
                                line=dict(width=2.4, color='#e6edf3'),
                                hovertemplate='%{x|%Y-%m-%d}<br>股價 %{y:.2f}<extra></extra>',
                            ))
                            fig.update_layout(
                                height=380, template='plotly_dark',
                                paper_bgcolor='#0b121b', plot_bgcolor='#0b121b',
                                margin=dict(l=20, r=20, t=10, b=20),
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10)),
                                xaxis=dict(gridcolor='rgba(148,163,184,0.09)'),
                                yaxis=dict(title='股價 (元)', gridcolor='rgba(148,163,184,0.09)'),
                                hovermode='x unified',
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("每條色線＝「近四季 EPS(TTM) × 該分位數的自身歷史本益比」推算出的理論價位，白線為實際股價。股價落在偏低區間，代表相對自己過去的本益比區間便宜；落在偏高區間代表相對較貴。這只是跟自己歷史比較的統計相對位置，不是目標價，也不是買賣建議。")
                        else:
                            st.info("目前資料不足以計算本益比河流圖，可能是新股、EPS 揭露筆數太少，或長期股價資料不足，請稍後再試。")

                        # [新功能] 產業別同儕比較：本益比放在同產業裡看相對位置，
                        # 全部用已經批次快取的資料算（股票清單＋官方本益比表），不逐股即時抓取。
                        st.markdown('<div class="section-head" style="margin-top:22px;"><div><div class="section-title" style="font-size:15px;">同產業比較</div><div class="section-help">用官方本益比批次資料（跟本益比河流圖同一份資料）比較同產業所有股票的本益比分布；只跟同業比較估值高低，不是買賣建議，成長股本益比偏高不一定代表偏貴。</div></div></div>', unsafe_allow_html=True)
                        peer_stats = build_industry_peer_stats(current_stock['code'], current_stock.get('industry', ''))
                        if peer_stats:
                            cheap_pct = peer_stats['cheaper_than_pct']
                            cheap_txt = f"比 {cheap_pct:.0f}% 的同業便宜" if pd.notna(cheap_pct) else "N/A"
                            cheap_color = 'var(--green)' if pd.notna(cheap_pct) and cheap_pct >= 50 else 'var(--red)'
                            st.markdown(f"""
                            <div class="stat-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:10px;">
                              <div class="tv-card"><div class="tv-label">本股本益比</div><div class="tv-value">{fmt_num(peer_stats['my_pe'], '{:.1f}')}</div><div class="tv-caption">{current_stock['name']}</div></div>
                              <div class="tv-card"><div class="tv-label">同業中位數</div><div class="tv-value">{fmt_num(peer_stats['median_pe'], '{:.1f}')}</div><div class="tv-caption">{peer_stats['industry']}</div></div>
                              <div class="tv-card"><div class="tv-label">同業本益比範圍</div><div class="tv-value" style="font-size:18px;">{peer_stats['min_pe']:.1f} ~ {peer_stats['max_pe']:.1f}</div><div class="tv-caption">有本益比樣本 {peer_stats['pe_sample_count']}/{peer_stats['peer_count']} 檔</div></div>
                              <div class="tv-card"><div class="tv-label">同業相對位置</div><div class="tv-value" style="font-size:18px;color:{cheap_color};">{cheap_txt}</div><div class="tv-caption">依本益比由低到高排序</div></div>
                            </div>
                            """, unsafe_allow_html=True)

                            fig = go.Figure()
                            fig.add_trace(go.Box(
                                x=peer_stats['peer_pe_values'], orientation='h', name='同業本益比分布',
                                boxpoints='all', jitter=0.55, pointpos=0,
                                marker=dict(color='rgba(140,160,190,0.55)', size=5),
                                line=dict(color='#4c8dff'), fillcolor='rgba(76,141,255,0.12)',
                                hovertemplate='本益比 %{x:.1f}<extra></extra>',
                            ))
                            if pd.notna(peer_stats['my_pe']):
                                fig.add_trace(go.Scatter(
                                    x=[peer_stats['my_pe']], y=['同業本益比分布'], mode='markers',
                                    name=current_stock['name'],
                                    marker=dict(color='#facc15', size=15, symbol='diamond', line=dict(color='#0b121b', width=1.5)),
                                    hovertemplate=f"{current_stock['name']} 本益比 " + '%{x:.1f}<extra></extra>',
                                ))
                            fig.update_layout(
                                height=190, template='plotly_dark',
                                paper_bgcolor='#0b121b', plot_bgcolor='#0b121b',
                                margin=dict(l=20, r=20, t=10, b=30),
                                showlegend=False,
                                xaxis=dict(title='本益比（倍）', gridcolor='rgba(148,163,184,0.09)'),
                                yaxis=dict(showticklabels=False),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"黃色菱形＝{current_stock['name']}目前本益比，箱型圖呈現「{peer_stats['industry']}」產業內有本益比資料的 {peer_stats['pe_sample_count']} 檔股票分布。本益比較低不代表股票比較好，也可能是市場認為成長性較差，請搭配其他指標一起看。")
                        else:
                            st.caption("這個產業分類的股票數量太少，或本益比資料不足（例如虧損股本益比為負，官方資料不會列出），暫時無法比較同業。")

                        # [新功能] 供應鏈上下游：內建常見熱門產業鏈對照表（非官方即時資料，
                        # 整理自公開產業知識，僅涵蓋幾條常見產業鏈的代表性股票），
                        # 搭配官方「產業價值鏈資訊平台」連結作為完整/權威資料的保底。
                        chain_head_col, chain_link_col = st.columns([4, 1.3])
                        with chain_head_col:
                            st.markdown('<div class="section-head" style="margin-top:22px;"><div><div class="section-title" style="font-size:15px;">供應鏈上下游</div><div class="section-help">內建常見熱門產業鏈對照表，整理自公開產業知識、非官方即時資料，只列代表性股票、不是完整清單，可能有過時或疏漏；一家公司也可能同時橫跨多條產業鏈。完整/最新資料請點右側官方平台連結，輸入代碼查詢。</div></div></div>', unsafe_allow_html=True)
                        with chain_link_col:
                            st.link_button(
                                "🔗 產業價值鏈資訊平台",
                                "https://ic.tpex.org.tw/",
                                use_container_width=True,
                            )
                        chain_matches = find_industry_chain_matches(current_stock['code'])
                        if chain_matches:
                            for m in chain_matches:
                                st.markdown(f"**{m['chain']}** － {current_stock['name']} 位於本表的**{m['stage']}**")
                                group_html = ""
                                for g in m['groups']:
                                    peer_txt = "、".join(f"{n}（{c}）" for c, n in g['peers'])
                                    group_html += f'<div class="report-row"><span>{g["label"]}</span><span style="text-align:right;max-width:70%;">{peer_txt}</span></div>'
                                st.markdown(f'<div class="side-card" style="margin-bottom:14px;">{group_html}</div>', unsafe_allow_html=True)
                        else:
                            st.caption(f"目前內建的產業鏈對照表還沒有收錄「{current_stock['name']}」，可以點上方連結到官方「產業價值鏈資訊平台」直接查詢完整上下游資料。")

                # ---------- 財務體質評分 ----------
                elif view_mode == "🩺 財務體質":
                    st.markdown(f'<div class="section-head" style="margin-top:4px;"><div><div class="section-title" style="font-size:15px;">{current_stock["name"]} ({current_stock["code"]}) 財務體質評分</div><div class="section-help">資料來源：Anue鉅亨網「年度財務比率」頁面，整合負債比率、流動比率、獲利能力等指標估算；不同產業合理區間本就不同，僅供快速篩選參考，非投資建議。</div></div></div>', unsafe_allow_html=True)

                    fin_data = fetch_financial_ratios(current_stock['code'])
                    fin_metrics = fin_data.get("metrics", {})
                    health = calc_financial_health_score(fin_metrics)

                    if health:
                        metric_cols = ''.join(
                            f'<div><div class="metric-k">{cat}</div><div class="metric-v">{score:.0f}</div></div>'
                            for cat, score in health['category_scores'].items()
                        )
                        st.markdown(f"""
                        <div class="quote-panel" style="margin:4px 0 16px;">
                          <div class="quote-head"><div class="quote-title" style="color:{health['color']};">綜合評分：{health['overall']:.0f} 分・{health['verdict']}</div></div>
                          <div class="quote-metrics" style="grid-template-columns:repeat({max(len(health['category_scores']), 1)},minmax(90px,1fr));">
                            {metric_cols}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                        # 「2019年 年報」的『年』會重複，年度太舊也沒有提示；
                        # 另外雷達圖畫的是分類數，跟指標數不同，這裡一併說明清楚。
                        _fy_m = re.search(r'(\d{4})', str(fin_data.get('year', '') or ''))
                        _fy_age = (get_tw_now().year - int(_fy_m.group(1))) if _fy_m else 0
                        _fy_txt = f'財報年度 {_fy_m.group(1)}' if _fy_m else '財報年度不明'
                        if _fy_m and _fy_age >= 3:
                            _fy_txt += f'（已 {_fy_age} 年未更新，僅供參考）'
                        st.caption(
                            f"{_fy_txt}｜{health['n_metrics']} 項指標彙總為 "
                            f"{len(health['category_scores'])} 個分類"
                        )

                        for it in health['items']:
                            badge_color = 'var(--green)' if it['分數'] >= 60 else ('var(--yellow)' if it['分數'] >= 35 else 'var(--red)')
                            st.markdown(f"""
                            <div class="side-card" style="margin-bottom:10px;">
                              <div class="side-title" style="margin-bottom:4px;">
                                <span>{it['指標']}（{it['分類']}）</span>
                                <span style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}40;border-radius:999px;padding:3px 12px;font-size:12px;font-weight:800;">{it['分數']:.0f} 分</span>
                              </div>
                              <div class="tv-caption" style="font-family:'Roboto Mono',monospace;">原始數值：{it['原始值']:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.caption("評分是用線性映射把每項指標換算成 0~100 分再平均，區間門檻是參考一般產業常見水準訂出的粗略估計，不同產業（例如重資產的航運、金融）合理區間本來就不同，僅適合同產業內快速比較參考，不是嚴謹的估值模型，也不是投資建議。")
                    else:
                        st.info("目前無法取得財務比率資料，可能是這檔股票剛上市、屬於金融／保險等特殊財報格式產業，或資料來源暫時無回應，請稍後再試。")

                # ---------- 股利政策／殖利率／除權息 ----------
                elif view_mode == "💵 股利政策":
                    market_suffix = "TW" if current_stock['ticker'].endswith(".TW") else "TWO"
                    st.markdown(f'<div class="section-head" style="margin-top:4px;"><div><div class="section-title" style="font-size:15px;">{current_stock["name"]} ({current_stock["code"]}) 股利政策</div><div class="section-help">資料來源：Yahoo 股市個股「股利」頁面，彙整歷年現金股利／股票股利／除息日，僅供參考，非投資建議。</div></div></div>', unsafe_allow_html=True)

                    div_data = fetch_dividend_history(current_stock['code'], market_suffix)
                    div_rows = div_data.get("rows", [])
                    div_summary = div_data.get("summary", {})

                    if div_rows or div_summary:
                        # 優先用 Yahoo 頁面上「已連N年配息…」這句官方算好的摘要；
                        # 算不到（例如頁面版型調整導致那句話解析失敗）時，改用逐年列表自行回補，
                        # 兩層 fallback 確保就算其中一種解析失效，畫面還是有數字可看。
                        streak_years = div_summary.get("連續配息年數")
                        if streak_years is None:
                            streak = 0
                            for r in div_rows:
                                has_div = (pd.notna(r.get("現金股利")) and r.get("現金股利") > 0) or (pd.notna(r.get("股票股利")) and r.get("股票股利") > 0)
                                if has_div:
                                    streak += 1
                                else:
                                    break
                            streak_years = streak if streak > 0 else None

                        total_div = div_summary.get("歷年合計股利")

                        latest_row = div_rows[0] if div_rows else {}
                        latest_cash_div = latest_row.get("現金股利", np.nan)
                        latest_year = latest_row.get("所屬期間", "")
                        latest_ex_date = latest_row.get("除息日", "N/A")

                        st.markdown(f"""
                        <div class="stat-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:10px;">
                          <div class="tv-card"><div class="tv-label">連續配息</div><div class="tv-value">{fmt_num(streak_years, '{:.0f}')}</div><div class="tv-caption">年</div></div>
                          <div class="tv-card"><div class="tv-label">歷年合計股利</div><div class="tv-value">{fmt_num(total_div, '{:.2f}')}</div><div class="tv-caption">元／股</div></div>
                          <div class="tv-card"><div class="tv-label">最新現金股利</div><div class="tv-value">{fmt_num(latest_cash_div, '{:.2f}')}</div><div class="tv-caption">{latest_year}年度</div></div>
                          <div class="tv-card"><div class="tv-label">最新除息日</div><div class="tv-value" style="font-size:18px;">{latest_ex_date}</div><div class="tv-caption">已公告或預告</div></div>
                        </div>
                        """, unsafe_allow_html=True)

                        if div_rows:
                            div_df = pd.DataFrame(div_rows)
                            chart_df = div_df.iloc[::-1]  # 轉成舊到新排序，繪圖用
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=chart_df['所屬期間'], y=chart_df['現金股利'], name='現金股利',
                                marker_color='#4c8dff',
                                hovertemplate='%{x}年度<br>現金股利 %{y:.2f}<extra></extra>',
                            ))
                            fig.update_layout(
                                height=320, template='plotly_dark',
                                paper_bgcolor='#0b121b', plot_bgcolor='#0b121b',
                                margin=dict(l=20, r=20, t=10, b=20),
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                                xaxis=dict(title='所屬年度', gridcolor='rgba(148,163,184,0.09)'),
                                yaxis=dict(title='現金股利 (元)', gridcolor='rgba(148,163,184,0.09)'),
                                hovermode='x unified',
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            st.dataframe(
                                div_df, hide_index=True, use_container_width=True,
                                column_config={
                                    "所屬期間": st.column_config.TextColumn("所屬年度", width=80),
                                    "現金股利": st.column_config.NumberColumn("現金股利(元)", width=90, format="%.2f"),
                                    "股票股利": st.column_config.NumberColumn("股票股利(元)", width=90, format="%.2f"),
                                    "除息日": st.column_config.TextColumn("除息日", width=100),
                                    "填息天數": st.column_config.NumberColumn("填息天數", width=80, format="%.0f"),
                                }
                            )
                            st.caption("填息天數是指除息後股價回到除息前一天收盤價所花的交易日數，空白代表尚未填息或資料不足；最上面一筆若除息日顯示「尚未公布」，代表是公司擬定中的預告股利，非正式決議，實際金額可能調整。")
                    else:
                        st.info("目前無法取得股利政策資料，可能是這檔股票從未配發股利（例如成長型無配息公司）、新股，或資料來源暫時無回應，請稍後再試。")

                # ---------- 三大法人買賣情況 ----------
                elif view_mode == "💰 三大法人":
                    market_suffix = "TW" if current_stock['ticker'].endswith(".TW") else "TWO"
                    st.markdown(f'<div class="section-head" style="margin-top:4px;"><div><div class="section-title" style="font-size:15px;">{current_stock["name"]} ({current_stock["code"]}) 三大法人買賣情況</div><div class="section-help">資料來源：Yahoo 股市個股「法人買賣」頁面，彙整外資、投信、自營商逐日買賣超，僅供參考，非官方逐筆對帳資料。</div></div></div>', unsafe_allow_html=True)

                    inst_rows = fetch_institutional_trading(current_stock['code'], market_suffix)
                    if inst_rows:
                        latest = inst_rows[0]
                        streak_f = _institutional_streak(inst_rows, "外資")
                        streak_t = _institutional_streak(inst_rows, "投信")
                        streak_d = _institutional_streak(inst_rows, "自營商")
                        streak_a = _institutional_streak(inst_rows, "合計")

                        def _streak_txt(s):
                            if s["days"] <= 1 or s["sign"] == 0:
                                return "非連續買賣"
                            word = "買" if s["sign"] > 0 else "賣"
                            return f"連{s['days']}{word}　累計 {s['total']:+,.0f} 張"

                        cards = [
                            ("外資（張）", latest.get("外資", np.nan), streak_f),
                            ("投信（張）", latest.get("投信", np.nan), streak_t),
                            ("自營商（張）", latest.get("自營商", np.nan), streak_d),
                            ("三大法人合計（張）", latest.get("合計", np.nan), streak_a),
                        ]
                        card_html = ""
                        for label, val, streak in cards:
                            color = "var(--green)" if pd.notna(val) and val >= 0 else "var(--red)"
                            val_txt = fmt_num(val, "{:+,.0f}")
                            card_html += f'<div class="tv-card"><div class="tv-label">{label}</div><div class="tv-value" style="color:{color};">{val_txt}</div><div class="tv-caption">{_streak_txt(streak)}</div></div>'

                        st.markdown(f"""
                        <div class="stat-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:6px;">
                          {card_html}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"資料時間：{latest['日期']}（近期逐日資料，正數為買超、負數為賣超；連買／連賣依此序列往回累加推算）")

                        chart_df = pd.DataFrame(inst_rows).iloc[::-1]  # 由舊到新排序，繪圖用
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=chart_df['日期'], y=chart_df['外資'], name='外資', marker_color='#2962ff'))
                        fig.add_trace(go.Bar(x=chart_df['日期'], y=chart_df['投信'], name='投信', marker_color='#f2a900'))
                        fig.add_trace(go.Bar(x=chart_df['日期'], y=chart_df['自營商'], name='自營商', marker_color='#a855f7'))
                        fig.update_layout(
                            barmode='relative', height=320, template='plotly_dark',
                            paper_bgcolor='#0b121b', plot_bgcolor='#0b121b',
                            margin=dict(l=20, r=20, t=10, b=20),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                            xaxis=dict(gridcolor='rgba(148,163,184,0.09)'),
                            yaxis=dict(title='買賣超（張）', gridcolor='rgba(148,163,184,0.09)'),
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        inst_df = pd.DataFrame(inst_rows)
                        st.dataframe(
                            inst_df, hide_index=True, use_container_width=True,
                            column_config={
                                "日期": st.column_config.TextColumn("日期", width=90),
                                "外資": st.column_config.NumberColumn("外資(張)", width=90, format="%,.0f"),
                                "投信": st.column_config.NumberColumn("投信(張)", width=90, format="%,.0f"),
                                "自營商": st.column_config.NumberColumn("自營商(張)", width=90, format="%,.0f"),
                                "合計": st.column_config.NumberColumn("合計(張)", width=90, format="%,.0f"),
                                "外資籌碼(%)": st.column_config.NumberColumn("外資籌碼", width=90, format="%.2f%%"),
                                "漲跌幅(%)": st.column_config.NumberColumn("漲跌幅", width=80, format="%.2f%%"),
                                "成交量": st.column_config.NumberColumn("成交量(張)", width=100, format="%,.0f"),
                            }
                        )
                    else:
                        st.info("目前無法取得三大法人買賣資料，可能是新股、資料來源暫時無回應，或該標的非集中市場／櫃買中心交易標的，請稍後再試。")

                # ---------- 資券變化 ----------
                elif view_mode == "📊 資券變化":
                    market_suffix = "TW" if current_stock['ticker'].endswith(".TW") else "TWO"
                    st.markdown(f'<div class="section-head" style="margin-top:4px;"><div><div class="section-title" style="font-size:15px;">{current_stock["name"]} ({current_stock["code"]}) 融資融券／資券變化</div><div class="section-help">資料來源：Yahoo 股市個股「資券變化」頁面，彙整融資融券逐日增減與餘額，僅供參考，非投資建議。</div></div></div>', unsafe_allow_html=True)

                    margin_rows = fetch_margin_trading(current_stock['code'], market_suffix)
                    if margin_rows:
                        latest = margin_rows[0]
                        streak_margin = _institutional_streak(margin_rows, "融資增減")
                        streak_short = _institutional_streak(margin_rows, "融券增減")

                        def _streak_txt(s, word_pos="增", word_neg="減"):
                            if s["days"] <= 1 or s["sign"] == 0:
                                return "非連續增減"
                            word = word_pos if s["sign"] > 0 else word_neg
                            return f"連{s['days']}{word}　累計 {s['total']:+,.0f} 張"

                        margin_color = "var(--green)" if pd.notna(latest.get("融資增減")) and latest["融資增減"] >= 0 else "var(--red)"
                        short_color = "var(--green)" if pd.notna(latest.get("融券增減")) and latest["融券增減"] >= 0 else "var(--red)"
                        st.markdown(f"""
                        <div class="stat-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:6px;">
                          <div class="tv-card"><div class="tv-label">融資餘額（張）</div><div class="tv-value">{fmt_num(latest.get('融資餘額'), '{:,.0f}')}</div><div class="tv-caption" style="color:{margin_color};">今日{fmt_num(latest.get('融資增減'), '{:+,.0f}')}張・{_streak_txt(streak_margin)}</div></div>
                          <div class="tv-card"><div class="tv-label">融券餘額（張）</div><div class="tv-value">{fmt_num(latest.get('融券餘額'), '{:,.0f}')}</div><div class="tv-caption" style="color:{short_color};">今日{fmt_num(latest.get('融券增減'), '{:+,.0f}')}張・{_streak_txt(streak_short)}</div></div>
                          <div class="tv-card"><div class="tv-label">券資比</div><div class="tv-value">{fmt_num(latest.get('券資比(%)'), '{:.2f}%')}</div><div class="tv-caption">融券餘額 ÷ 融資餘額</div></div>
                          <div class="tv-card"><div class="tv-label">資券互抵（張）</div><div class="tv-value">{fmt_num(latest.get('資券互抵'), '{:,.0f}')}</div><div class="tv-caption">當沖：同日融資買進＋融券賣出</div></div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"資料時間：{latest['日期']}（近期逐日資料）")

                        chart_df = pd.DataFrame(margin_rows).iloc[::-1]  # 轉成舊到新排序，繪圖用
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=chart_df['日期'], y=chart_df['融資餘額'], name='融資餘額', mode='lines',
                            line=dict(width=2.2, color='#2962ff'),
                            hovertemplate='%{x}<br>融資餘額 %{y:,.0f} 張<extra></extra>',
                        ))
                        fig.add_trace(go.Scatter(
                            x=chart_df['日期'], y=chart_df['融券餘額'], name='融券餘額', mode='lines', yaxis='y2',
                            line=dict(width=2.2, color='#f2a900'),
                            hovertemplate='%{x}<br>融券餘額 %{y:,.0f} 張<extra></extra>',
                        ))
                        fig.update_layout(
                            height=340, template='plotly_dark',
                            paper_bgcolor='#0b121b', plot_bgcolor='#0b121b',
                            margin=dict(l=20, r=40, t=10, b=20),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                            xaxis=dict(gridcolor='rgba(148,163,184,0.09)'),
                            yaxis=dict(title='融資餘額（張）', gridcolor='rgba(148,163,184,0.09)'),
                            yaxis2=dict(title='融券餘額（張）', overlaying='y', side='right', showgrid=False),
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        margin_df = pd.DataFrame(margin_rows)
                        st.dataframe(
                            margin_df, hide_index=True, use_container_width=True,
                            column_config={
                                "日期": st.column_config.TextColumn("日期", width=90),
                                "融資增減": st.column_config.NumberColumn("融資增減", width=85, format="%,.0f"),
                                "融資餘額": st.column_config.NumberColumn("融資餘額", width=85, format="%,.0f"),
                                "融資使用率(%)": st.column_config.NumberColumn("融資使用率", width=90, format="%.2f%%"),
                                "融券增減": st.column_config.NumberColumn("融券增減", width=85, format="%,.0f"),
                                "融券餘額": st.column_config.NumberColumn("融券餘額", width=85, format="%,.0f"),
                                "融券使用率(%)": st.column_config.NumberColumn("融券使用率", width=90, format="%.2f%%"),
                                "券資比(%)": st.column_config.NumberColumn("券資比", width=80, format="%.2f%%"),
                                "資券互抵": st.column_config.NumberColumn("資券互抵", width=80, format="%,.0f"),
                            }
                        )
                        st.caption("融資增減／融券增減是正數代表當日餘額比前一日增加，負數代表減少；券資比 = 融券餘額 ÷ 融資餘額，數值愈高代表放空的人相對愈多。資券互抵（資券當沖）是同一交易日內對同一檔股票融資買進又融券賣出、可互相沖銷的張數。")
                    else:
                        st.info("目前無法取得融資融券資料，可能是這檔股票不適用融資融券交易（例如興櫃股票或處置股），或資料來源暫時無回應，請稍後再試。")

                # ---------- 法人目標價 ----------
                elif view_mode == "🎯 法人目標價":
                    st.markdown(f'<div class="section-head" style="margin-top:4px;"><div><div class="section-title" style="font-size:15px;">{current_stock["name"]} ({current_stock["code"]}) 法人目標價</div><div class="section-help">主來源：Anue鉅亨「外資評等」；若無資料則自動改用 Yahoo Finance 分析師共識目標價。僅供參考，非投資建議。</div></div></div>', unsafe_allow_html=True)

                    market_suffix = "TW" if str(current_stock.get('ticker', '')).endswith(".TW") else "TWO"
                    target_rows = fetch_analyst_target_price(current_stock['code'], market_suffix)
                    if target_rows:
                        current_price = current_stock.get('收盤', np.nan)
                        summary = summarize_target_price(target_rows, current_price)

                        if summary:
                            upside = summary.get('upside_pct', np.nan)
                            upside_color = 'var(--green)' if pd.notna(upside) and upside >= 0 else 'var(--red)'
                            st.markdown(f"""
                            <div class="stat-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:10px;">
                              <div class="tv-card"><div class="tv-label">最新目標價</div><div class="tv-value">{fmt_num(summary['latest_target'], '{:.1f}')}</div><div class="tv-caption">{summary['latest_broker']}｜{summary['latest_date']}</div></div>
                              <div class="tv-card"><div class="tv-label">近{summary['n_ratings']}筆平均目標價</div><div class="tv-value">{fmt_num(summary['avg_target'], '{:.1f}')}</div><div class="tv-caption">{fmt_num(summary['min_target'], '{:.1f}')} ~ {fmt_num(summary['max_target'], '{:.1f}')}</div></div>
                              <div class="tv-card"><div class="tv-label">目前股價</div><div class="tv-value">{fmt_num(current_price, '{:.2f}')}</div><div class="tv-caption">最新收盤</div></div>
                              <div class="tv-card"><div class="tv-label">平均目標價潛在漲跌</div><div class="tv-value" style="color:{upside_color};">{fmt_num(upside, '{:+.1f}%')}</div><div class="tv-caption">相對目前股價</div></div>
                            </div>
                            """, unsafe_allow_html=True)

                        target_df = pd.DataFrame(target_rows)
                        chart_df = target_df.iloc[::-1]  # 轉成舊到新排序，繪圖用
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=chart_df['評等日期'], y=chart_df['目標價'], mode='lines+markers', name='法人目標價',
                            line=dict(width=2.2, color='#4c8dff'), marker=dict(size=6),
                            hovertemplate='%{x}<br>目標價 %{y:.1f}<extra></extra>',
                        ))
                        if pd.notna(current_price):
                            fig.add_hline(
                                y=current_price, line=dict(color='#e6edf3', width=1.4, dash='dot'),
                                annotation_text=f'目前股價 {current_price:.2f}', annotation_position='bottom right',
                            )
                        fig.update_layout(
                            height=320, template='plotly_dark',
                            paper_bgcolor='#0b121b', plot_bgcolor='#0b121b',
                            margin=dict(l=20, r=20, t=10, b=20),
                            xaxis=dict(gridcolor='rgba(148,163,184,0.09)'),
                            yaxis=dict(title='目標價 (元)', gridcolor='rgba(148,163,184,0.09)'),
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        def _rating_change_color(val):
                            s = str(val)
                            if "升" in s:
                                return "color:#35c48d;font-weight:700;"
                            if "降" in s:
                                return "color:#f23645;font-weight:700;"
                            return ""

                        target_styled = target_df.style.map(_rating_change_color, subset=["升降"])
                        st.dataframe(
                            target_styled, hide_index=True, use_container_width=True,
                            column_config={
                                "評等日期": st.column_config.TextColumn("評等日期", width=90),
                                "券商": st.column_config.TextColumn("券商/機構", width=100),
                                "新評等": st.column_config.TextColumn("投資評等", width=100),
                                "升降": st.column_config.TextColumn("升降", width=60),
                                "財測EPS": st.column_config.TextColumn("財測EPS(年度)", width=100),
                                "目標價": st.column_config.NumberColumn("目標價", width=80, format="%.1f"),
                                "現價": st.column_config.NumberColumn("參考現價", width=80, format="%.2f"),
                                "來源": st.column_config.TextColumn("來源", width=70),
                            }
                        )
                        st.caption("「近N筆平均目標價」取最近10筆評等平均（非近10天）。Anue 不提供評等當日歷史股價，故「參考現價」欄僅在 Yahoo 備援時顯示共識當下價格。目標價僅供參考，非投資建議。")
                    else:
                        st.info("目前無法取得法人目標價資料，可能是這檔股票近期沒有外資／券商發布評等報告，或資料來源暫時無回應，請稍後再試。")

                # ---------- 個股新聞 ----------
                elif view_mode == "📰 個股新聞":
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

            st.markdown(
                '<div class="wb-disclaimer">'
                '<span>本資料僅供參考・投資有風險・請審慎評估</span>'
                '<span>資料來源：TWSE／TPEx OpenAPI・Yahoo 股市・Anue鉅亨網</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    else:
        st.info("目前沒有候選股票。可以直接在下面輸入代碼查詢，或先到「市場掃描」頁籤執行全市場掃描。")
        render_manual_search_box(key_suffix="empty")

# ------------------------------------------------------------
# TAB 3：持倉管理中心／Telegram 停損警示
# ------------------------------------------------------------
with tab_portfolio:
    st.markdown('<div class="section-head"><div><div class="section-title">持倉管理中心</div><div class="section-help">管理進場成本、股數、停損與目標價，統一追蹤損益、R 倍數與 Telegram 停損警示。</div></div></div>', unsafe_allow_html=True)

    # Telegram 設定
    with st.expander("✈️ Telegram 停損警示設定", expanded=False):
        _tg_cfg = get_telegram_config()
        tg1, tg2 = st.columns(2)
        with tg1:
            tg_token = st.text_input(
                "Bot Token", value=_tg_cfg.get('bot_token', ''), type="password",
                help="建議正式部署時放在 .streamlit/secrets.toml 的 TELEGRAM_BOT_TOKEN。",
            )
        with tg2:
            tg_chat_id = st.text_input(
                "Chat ID", value=_tg_cfg.get('chat_id', ''),
                help="可填個人 Chat ID 或群組 Chat ID。",
            )
        tg3, tg4 = st.columns(2)
        with tg3:
            tg_auto = st.checkbox(
                "頁面開啟時自動定時檢查", value=bool(_tg_cfg.get('auto_check', False)),
                help="瀏覽器頁面必須保持開啟；頁面關閉後 Streamlit 不會在背景持續執行。",
            )
        with tg4:
            tg_interval = st.number_input(
                "檢查間隔（分鐘）", min_value=1, max_value=120,
                value=max(1, int(_tg_cfg.get('interval_minutes', 5))), step=1,
            )
        tgb1, tgb2 = st.columns(2)
        with tgb1:
            if st.button("💾 儲存 Telegram 設定", use_container_width=True, key="save_tg_config"):
                if save_telegram_config(tg_token, tg_chat_id, tg_auto, int(tg_interval)):
                    st.success("Telegram 設定已儲存。")
                else:
                    st.error("Telegram 設定檔寫入失敗。")
        with tgb2:
            if st.button("🧪 發送測試訊息", use_container_width=True, key="test_tg_message"):
                test_cfg = {'bot_token': tg_token, 'chat_id': tg_chat_id}
                ok, message = send_telegram_message(
                    f"【台股決策中心】Telegram 測試成功\n時間：{get_tw_now().strftime('%Y-%m-%d %H:%M:%S')}",
                    test_cfg,
                )
                st.success(message) if ok else st.error(message)
        st.caption("若直接在畫面儲存，Token 會寫入本機 JSON；公開或多人部署請改用 Streamlit Secrets，避免憑證外洩。")

    positions_all = load_positions()
    open_positions = [p for p in positions_all if p.get('status', 'open') == 'open']
    closed_positions = [p for p in positions_all if p.get('status') == 'closed']

    # 自動檢查：Streamlit 所有頁籤都會執行，因此只要頁面保持開啟即可定時刷新。
    _cfg_runtime = get_telegram_config()
    if _cfg_runtime.get('auto_check') and open_positions:
        _interval_seconds = max(60, int(_cfg_runtime.get('interval_minutes', 5)) * 60)
        _last_check = st.session_state.get('portfolio_last_check')
        _now_ts = time.time()
        if not _last_check or (_now_ts - float(_last_check)) >= _interval_seconds:
            _auto_df, _auto_alerts = refresh_portfolio_positions(send_alerts=True)
            st.session_state.portfolio_quotes = _auto_df
            st.session_state.portfolio_alert_result = _auto_alerts
            st.session_state.portfolio_last_check = _now_ts
        # 用瀏覽器重新整理觸發下一輪；頁面關閉後不會執行。
        components.html(
            f"<script>setTimeout(function(){{window.parent.location.reload();}}, {_interval_seconds * 1000});</script>",
            height=0,
        )

    ctl1, ctl2, ctl3 = st.columns([1.2, 1.2, 3])
    with ctl1:
        if st.button("🔄 更新持倉報價", use_container_width=True, type="primary", key="refresh_portfolio"):
            with st.spinner(f"正在更新 {len(open_positions)} 筆持倉..."):
                quote_df, alert_results = refresh_portfolio_positions(send_alerts=False)
            st.session_state.portfolio_quotes = quote_df
            st.session_state.portfolio_alert_result = alert_results
            st.session_state.portfolio_last_check = time.time()
    with ctl2:
        if st.button("🚨 檢查並推播停損", use_container_width=True, key="check_portfolio_alerts"):
            with st.spinner("正在檢查停損條件並傳送 Telegram..."):
                quote_df, alert_results = refresh_portfolio_positions(send_alerts=True)
            st.session_state.portfolio_quotes = quote_df
            st.session_state.portfolio_alert_result = alert_results
            st.session_state.portfolio_last_check = time.time()
            if not alert_results:
                st.success("本次沒有新的停損觸發。")
            else:
                for result in alert_results:
                    (st.success if result.get('ok') else st.error)(f"{result.get('code')}：{result.get('message')}")
    with ctl3:
        last_check = st.session_state.get('portfolio_last_check')
        check_txt = datetime.fromtimestamp(last_check, tz=timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S') if last_check else '尚未檢查'
        st.caption(f"未平倉 {len(open_positions)} 筆｜上次報價／警示檢查：{check_txt}。最新價格來自目前可取得的日線資料，不是券商逐筆即時成交價。")

    if not open_positions:
        st.markdown("""
        <div class="tv-panel" style="text-align:center;padding:42px 22px;margin-top:12px;">
          <div style="font-size:40px;margin-bottom:12px;">💼</div>
          <div style="font-size:20px;font-weight:800;color:#f4f8ff;">目前沒有未平倉部位</div>
          <div class="tv-caption" style="margin-top:9px;line-height:1.8;">到「個股工作台」展開「交易計畫／建立持倉」，<br>即可把股票加入持倉中心並設定 Telegram 停損警示。</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        quote_df = st.session_state.get('portfolio_quotes', pd.DataFrame())
        valid_quote_df = isinstance(quote_df, pd.DataFrame) and not quote_df.empty
        if valid_quote_df:
            total_value = float((quote_df['現價'] * quote_df['股數']).sum(skipna=True))
            total_pnl = float(quote_df['損益(元)'].sum(skipna=True))
            total_risk = float(quote_df['風險至停損(元)'].sum(skipna=True))
            stop_count = int(quote_df['狀態'].astype(str).str.contains('跌破停損').sum())
            s1, s2, s3, s4 = st.columns(4)
            with s1: st.metric("持倉市值", f"{total_value:,.0f} 元")
            with s2: st.metric("未實現損益", f"{total_pnl:+,.0f} 元")
            with s3: st.metric("目前至停損風險", f"{total_risk:,.0f} 元")
            with s4: st.metric("跌破停損", f"{stop_count} 筆")

            display_cols = ['代碼', '名稱', '進場日', '成本', '股數', '現價', '損益(元)', '報酬率(%)', 'R倍數', '停損', '目標1', '目標2', '狀態', '策略', '報價日']
            st.dataframe(
                quote_df[[c for c in display_cols if c in quote_df.columns]],
                hide_index=True, use_container_width=True,
                column_config={
                    '成本': st.column_config.NumberColumn(format='%.2f'),
                    '現價': st.column_config.NumberColumn(format='%.2f'),
                    '股數': st.column_config.NumberColumn(format='%,.0f'),
                    '損益(元)': st.column_config.NumberColumn(format='%+,.0f'),
                    '報酬率(%)': st.column_config.NumberColumn(format='%+.2f%%'),
                    'R倍數': st.column_config.NumberColumn(format='%+.2fR'),
                    '停損': st.column_config.NumberColumn(format='%.2f'),
                    '目標1': st.column_config.NumberColumn(format='%.2f'),
                    '目標2': st.column_config.NumberColumn(format='%.2f'),
                },
            )
        else:
            basic_rows = [{
                '代碼': p.get('code'), '名稱': p.get('name'), '進場日': p.get('entry_date'),
                '成本': p.get('entry_price'), '股數': p.get('shares'), '停損': p.get('stop_price'),
                '目標1': p.get('target1'), '目標2': p.get('target2'), '策略': p.get('strategy'),
            } for p in open_positions]
            st.dataframe(pd.DataFrame(basic_rows), hide_index=True, use_container_width=True)
            st.info("請按「更新持倉報價」載入現價、損益與 R 倍數。")

        st.markdown('<div class="section-title" style="margin-top:24px;font-size:16px;">✏️ 編輯／平倉持倉</div>', unsafe_allow_html=True)
        position_map = {f"{p.get('name')} ({p.get('code')})": p for p in open_positions}
        selected_label = st.selectbox("選擇持倉", list(position_map.keys()), key="manage_position_select")
        selected_position = position_map[selected_label]
        _selected_is_etf = is_etf_instrument(selected_position)
        _selected_price_base = _safe_number(selected_position.get('last_price'), _safe_number(selected_position.get('entry_price'), 0.01))
        _selected_tick = get_tw_price_tick(_selected_price_base, _selected_is_etf)
        _edit_entry_default = round_to_tw_tick(_safe_number(selected_position.get('entry_price'), 0.01), _selected_is_etf)
        _edit_stop_default = round_to_tw_tick(_safe_number(selected_position.get('stop_price'), 0.01), _selected_is_etf)
        _edit_target1_default = round_to_tw_tick(_safe_number(selected_position.get('target1'), 0.01), _selected_is_etf)
        _edit_target2_default = round_to_tw_tick(_safe_number(selected_position.get('target2'), 0.01), _selected_is_etf)

        with st.form("edit_position_form"):
            e1, e2, e3 = st.columns(3)
            with e1:
                edit_entry = st.number_input("進場成本", min_value=0.01, value=float(_edit_entry_default), step=_selected_tick, format='%.2f')
                edit_shares = st.number_input("持有股數", min_value=1, value=int(selected_position.get('shares', 1)), step=1)
            with e2:
                edit_stop = st.number_input("停損價", min_value=0.01, value=float(_edit_stop_default), step=_selected_tick, format='%.2f')
                edit_target1 = st.number_input("第一目標", min_value=0.01, value=float(_edit_target1_default), step=_selected_tick, format='%.2f')
            with e3:
                edit_target2 = st.number_input("第二目標", min_value=0.01, value=float(_edit_target2_default), step=_selected_tick, format='%.2f')
                edit_tg = st.checkbox("啟用 Telegram 停損警示", value=bool(selected_position.get('telegram_alert', True)))
            edit_strategy = st.text_input("策略", value=str(selected_position.get('strategy', '')))
            edit_note = st.text_area("交易筆記", value=str(selected_position.get('note', '')))
            if st.form_submit_button("儲存持倉修改", type="primary", use_container_width=True):
                _edit_entry_valid = round_to_tw_tick(edit_entry, _selected_is_etf)
                _edit_stop_valid = round_to_tw_tick(edit_stop, _selected_is_etf)
                _edit_target1_valid = round_to_tw_tick(edit_target1, _selected_is_etf)
                _edit_target2_valid = round_to_tw_tick(edit_target2, _selected_is_etf)
                if _edit_stop_valid >= _edit_entry_valid:
                    st.error("停損價必須低於進場成本。")
                elif _edit_target1_valid <= _edit_entry_valid or _edit_target2_valid <= _edit_target1_valid:
                    st.error("目標價設定不正確。")
                else:
                    ok = update_position(selected_position['id'], {
                        'entry_price': _edit_entry_valid, 'shares': int(edit_shares),
                        'stop_price': _edit_stop_valid,
                        'target1': _edit_target1_valid, 'target2': _edit_target2_valid,
                        'strategy': edit_strategy.strip(), 'note': edit_note.strip(),
                        'telegram_alert': bool(edit_tg), 'stop_alert_active': False,
                    })
                    if ok:
                        st.session_state.portfolio_quotes = pd.DataFrame()
                        st.success("持倉已更新。")
                        st.rerun()
                    else:
                        st.error("持倉更新失敗。")

        c1, c2 = st.columns(2)
        with c1:
            with st.form("close_position_form"):
                close_price_input = st.number_input("平倉價", min_value=0.01, value=float(_safe_number(selected_position.get('last_price'), _safe_number(selected_position.get('entry_price'), 0.01))), step=_selected_tick, format='%.2f')
                close_date_input = st.date_input("平倉日期", value=get_tw_now().date())
                close_note_input = st.text_input("平倉原因", placeholder="停損、達標、趨勢轉弱、分批停利…")
                if st.form_submit_button("✅ 確認平倉", use_container_width=True):
                    if close_position(selected_position['id'], round_to_tw_tick(close_price_input, _selected_is_etf), close_date_input.strftime('%Y-%m-%d'), close_note_input):
                        st.session_state.portfolio_quotes = pd.DataFrame()
                        st.success("已完成平倉並保留績效紀錄。")
                        st.rerun()
                    else:
                        st.error("平倉失敗。")
        with c2:
            st.warning("刪除會直接移除紀錄；正常出場請使用左側「確認平倉」，才能保留績效。")
            confirm_delete = st.checkbox("我確認要永久刪除這筆持倉", key="confirm_delete_position")
            if st.button("🗑 永久刪除", disabled=not confirm_delete, use_container_width=True, key="delete_position_btn"):
                if delete_position(selected_position['id']):
                    st.session_state.portfolio_quotes = pd.DataFrame()
                    st.success("持倉已刪除。")
                    st.rerun()

    if closed_positions:
        with st.expander(f"📚 已平倉紀錄（{len(closed_positions)} 筆）", expanded=False):
            closed_rows = [{
                '代碼': p.get('code'), '名稱': p.get('name'), '進場日': p.get('entry_date'),
                '平倉日': p.get('exit_date'), '進場價': p.get('entry_price'), '平倉價': p.get('exit_price'),
                '股數': p.get('shares'), '已實現損益': p.get('realized_pnl'),
                '報酬率(%)': p.get('realized_return_pct'), '策略': p.get('strategy'), '平倉原因': p.get('exit_note'),
            } for p in reversed(closed_positions)]
            st.dataframe(pd.DataFrame(closed_rows), hide_index=True, use_container_width=True)


# ------------------------------------------------------------
# TAB 4：自選股／追蹤清單
# ------------------------------------------------------------
# [新功能] 讓還沒符合策略條件、但你想持續觀察的股票也能被記錄下來，
# 不用每次都全市場重新掃描才看得到它們。清單存在本機 JSON 檔案
# （watchlist_v1.json），跟 get_stock_market_list() 用同一套持久化寫法。
with tab_watchlist:
    st.markdown('<div class="section-head"><div><div class="section-title">自選股／追蹤清單</div><div class="section-help">在「個股工作台」右上角點 ☆ 加入追蹤，這裡會持續記錄，不受重新掃描影響。</div></div></div>', unsafe_allow_html=True)

    watchlist = load_watchlist()
    if not watchlist:
        st.markdown("""
        <div class="tv-panel" style="text-align:center;padding:42px 22px;margin-top:8px;">
          <div style="font-size:40px;margin-bottom:12px;">⭐</div>
          <div style="font-size:20px;font-weight:800;color:#f4f8ff;">追蹤清單目前是空的</div>
          <div class="tv-caption" style="margin-top:9px;line-height:1.8;">掃描完成後，在「個股工作台」右上角點「☆ 加入追蹤」，<br>就能把還沒達標、但值得持續觀察的股票留在這裡。</div>
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
                wl_show_adjusted = st.toggle("還原股價（除權息）", key="wl_kline_show_adjusted")
                wl_fig = draw_k_line(wl_tk, wl_name, chart_mode='K線圖', chart_period='日', adjusted=wl_show_adjusted)
                if wl_fig:
                    render_kline_chart_with_axis_price(wl_fig, height=440)
                else:
                    st.warning("⚠️ 無法載入 K 線資料，請稍後再試。")

# ------------------------------------------------------------
# TAB 5：投資分析報告（純量化規則版，不需要 API 金鑰）
# ------------------------------------------------------------
with tab_report:
    st.markdown('<div class="section-head"><div><div class="section-title">個股投資分析報告</div><div class="section-help">輸入股票代碼，自動整合量化指標、新聞分類、同業比較與波動率情境價格區間，全部由規則產生，不需要 API 金鑰。</div></div></div>', unsafe_allow_html=True)

    # [版面] 代碼輸入、產生、重新產生三顆放同一列。原本「重新產生」是用一個
    # st.columns([5,1]) 的空白左欄把它擠到右邊、自己獨占一行，既浪費一整列高度，
    # 視覺上也像是漂在按鈕下方的孤兒元件。
    _has_report = bool(st.session_state.get("report_target_code"))
    rc1, rc2, rc3 = st.columns([2.6, 1, 1], gap="medium")
    with rc1:
        report_code_input = st.text_input("股票代碼", key="report_code_input", placeholder="例如 2330、8033", label_visibility="collapsed")
    with rc2:
        gen_clicked = st.button("📑 產生報告", use_container_width=True, type="primary", key="btn_gen_report")
    with rc3:
        regen_clicked = st.button("🔄 重新產生", use_container_width=True, key="btn_regenerate_report",
                                  disabled=not _has_report, help="略過快取，重新抓一次財務、同業與新聞資料。")

    if gen_clicked and report_code_input.strip():
        st.session_state.report_target_code = report_code_input.strip().upper()
        st.session_state.report_force_refresh = True
        st.rerun()

    if regen_clicked and _has_report:
        st.session_state.report_force_refresh = True
        st.rerun()

    target_code = st.session_state.get("report_target_code")

    if not target_code:
        st.markdown("""
        <div class="tv-panel" style="text-align:center;padding:42px 22px;margin-top:8px;">
          <div style="font-size:40px;margin-bottom:12px;">📑</div>
          <div style="font-size:20px;font-weight:800;color:#f4f8ff;">還沒有產生過報告</div>
          <div class="tv-caption" style="margin-top:9px;line-height:1.8;">在上方輸入股票代碼並按「產生報告」，<br>就能看到重點觀察、催化因素、同業比較與情境價格區間的完整分析。</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cache = st.session_state.setdefault("report_cache", {})
        force_refresh = st.session_state.pop("report_force_refresh", False)

        if target_code not in cache or force_refresh:
            with st.spinner(f"正在整理 {target_code} 的財務、同業與新聞資料…"):
                row_data = build_manual_stock_row(target_code)

            if row_data is None:
                st.error(f"找不到股票代碼「{target_code}」，或目前抓不到報價資料，請確認代碼是否正確、稍後再試。")
                cache.pop(target_code, None)
                target_code = None
            else:
                market = "TW" if str(row_data["ticker"]).endswith(".TW") else "TWO"
                price_df = get_kline_data(row_data["code"], market)
                closes = price_df["close"] if not price_df.empty else pd.Series(dtype=float)

                with st.spinner("正在尋找同業比較標的…"):
                    peers_df = fetch_industry_peers_table(row_data.get("industry", ""), row_data["code"])

                news_list = get_tw_stock_news(row_data["code"]) or []
                cache[target_code] = build_investment_report(row_data, closes, peers_df, news_list)

        if target_code and target_code in cache:
            rd = cache[target_code]

            # ---------- 報告頭卡：名稱、報價、估值旗標 ----------
            badge_color = {
                "偏低估／具吸引力": "var(--green)", "合理偏多": "var(--blue)",
                "中性": "var(--yellow)", "偏弱／宜觀察": "var(--red)", "資料不足": "#8f9bad",
            }.get(rd.get("估值判斷旗標"), "#8f9bad")
            chg = rd.get("漲跌幅(%)", np.nan)
            chg_color = "var(--green)" if pd.notna(chg) and chg >= 0 else "var(--red)"

            # [版面] 估值旗標原本自己佔一行（margin-top:14px），把卡片撐高又顯得零散。
            # 改成跟產業別同一行的 chip，卡片少一列、左右兩塊的視覺重量也比較平衡。
            st.markdown(f"""
            <div class="tv-panel" style="padding:24px 26px;margin:16px 0 4px;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:20px;">
                <div style="min-width:250px;">
                  <div style="font-size:23px;font-weight:800;line-height:1.35;">{rd.get('name','')} ({rd.get('code','')})</div>
                  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:11px;">
                    <span class="tv-caption" style="margin:0;">{rd.get('industry','')}</span>
                    <span style="background:{badge_color};color:#04121f;font-weight:800;padding:4px 12px;border-radius:20px;font-size:12.5px;white-space:nowrap;">{rd.get('估值判斷旗標','')}</span>
                  </div>
                </div>
                <div style="text-align:right;min-width:140px;">
                  <div style="font-size:28px;font-weight:800;font-family:'Roboto Mono',monospace;line-height:1.2;">{fmt_num(rd.get('收盤'))}</div>
                  <div style="color:{chg_color};font-weight:700;font-family:'Roboto Mono',monospace;margin-top:6px;">{fmt_num(chg, '{:+.2f}%')}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ---------- 量化指標卡 ----------
            # [版面] .tv-card 的 min-height 是 104px，只放「標籤＋數字」會留下一塊空白，
            # 四張並排時看起來特別空。補一行判讀說明，順便讓數字有解讀基準。
            m1, m2, m3, m4 = st.columns(4, gap="medium")
            for col, label, val, fmt, note in [
                (m1, "本益比", rd.get("本益比"), "{:.1f}", "須與同業對照才有意義"),
                (m2, "財務評分", rd.get("財務評分"), "{:.0f}", "0–100，越高體質越穩"),
                (m3, "營收年增(%)", rd.get("營收年增"), "{:+.1f}", "與去年同月比較"),
                (m4, "RSI14", rd.get("RSI14"), "{:.1f}", "＞70 過熱／＜30 超賣"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="tv-card" style="margin-top:12px;">'
                        f'<div class="tv-label">{label}</div>'
                        f'<div class="tv-value">{fmt_num(val, fmt)}</div>'
                        f'<div class="tv-caption">{note}</div></div>',
                        unsafe_allow_html=True)

            # ---------- 重點觀察 ----------
            obs_list = rd.get("重點觀察", [])
            if obs_list:
                st.markdown('<div class="section-title" style="margin-top:30px;font-size:16px;">📌 重點觀察</div>', unsafe_allow_html=True)
                obs_html = "".join(f"<div style='padding:5px 0;border-bottom:1px solid rgba(35,58,85,0.5);font-size:14px;'>{o}</div>" for o in obs_list)
                st.markdown(f'<div class="tv-panel" style="padding:12px 18px;">{obs_html}</div>', unsafe_allow_html=True)

            # ---------- 催化因素 ----------
            catalysts = rd.get("催化因素", [])
            if catalysts:
                st.markdown('<div class="section-title" style="margin-top:22px;font-size:16px;">🚀 催化因素（新聞關鍵字分類）</div>', unsafe_allow_html=True)
                for cat in catalysts:
                    label = cat.get("label", "")
                    color = cat.get("color", "var(--muted)")
                    titles = cat.get("titles", [])
                    titles_html = "".join(f"<div style='font-size:13px;color:var(--muted);padding:2px 0;'>· {t}</div>" for t in titles)
                    st.markdown(
                        f'<div style="border-left:3px solid {color};padding:10px 14px;background:rgba(13,26,43,0.7);border-radius:6px;margin-bottom:8px;">'
                        f'<b style="color:{color};">{label}</b>{titles_html}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.caption("目前沒有符合分類的新聞標題，可能是新聞資料尚未載入。")

            # ---------- 財務健康與同業比較 ----------
            st.markdown('<div class="section-title" style="margin-top:22px;font-size:16px;">🏢 財務健康與同業比較</div>', unsafe_allow_html=True)
            peers = rd.get("同業比較", [])
            if peers:
                peer_df_display = pd.DataFrame(peers)
                st.dataframe(
                    peer_df_display, use_container_width=True, hide_index=True,
                    column_config={
                        "代碼": st.column_config.TextColumn("代碼", width=70),
                        "名稱": st.column_config.TextColumn("名稱", width=100),
                        "收盤": st.column_config.NumberColumn("收盤", format="%.2f"),
                        "本益比": st.column_config.NumberColumn("本益比", format="%.1f"),
                        "財務評分": st.column_config.NumberColumn("財務評分", format="%.0f"),
                    },
                )
                st.caption("同業比較表：自動抓取同產業（依 industry 欄位分類）中流動性較高、財務評分最高的幾檔股票，僅供參考。")
            else:
                st.caption("目前找不到足夠的同產業比較標的（可能是產業分類過於冷門或資料不足）。")

            # ---------- 情境價格區間 ----------
            st.markdown('<div class="section-title" style="margin-top:22px;font-size:16px;">⚖️ 情境價格區間（波動率估算）</div>', unsafe_allow_html=True)
            bands = rd.get("情境價格區間", {})
            if bands:
                b1, b2, b3, b4 = st.columns(4)
                for col, label, key in [
                    (b1, "悲觀 Bear", "悲觀 Bear (約-1σ)"),
                    (b2, "基準 Base", "基準 Base"),
                    (b3, "樂觀 Bull", "樂觀 Bull (約+1σ)"),
                    (b4, "極樂觀 Stretched Bull", "極樂觀 Stretched Bull (約+1.8σ)"),
                ]:
                    with col:
                        st.markdown(f'<div class="tv-card"><div class="tv-label">{label}</div><div class="tv-value">{fmt_num(bands.get(key))}</div></div>', unsafe_allow_html=True)
                st.caption(f"用近60日歷史波動率（年化約 {fmt_num(bands.get('年化波動率(%)'), '{:.1f}')}%）反推未來約 {bands.get('推算天數','')} 個交易日的統計價格區間（隨機漫步模型）。屬機率分布估算，不是目標價、不是基本面判斷，僅供參考。")
            else:
                st.caption("歷史價格資料不足（需近65個交易日以上），無法計算情境價格區間。")

            # ---------- 參考新聞標題 ----------
            news_items = rd.get("新聞標題", [])
            if news_items:
                st.markdown('<div class="section-title" style="margin-top:22px;font-size:16px;">📰 參考新聞（Yahoo股市）</div>', unsafe_allow_html=True)
                for n in news_items:
                    link = n.get("link", "")
                    title = n.get("title", "")
                    sentiment = n.get("sentiment", "")
                    if link:
                        st.caption(f"{sentiment} [{title}]({link})")
                    else:
                        st.caption(f"{sentiment} {title}")

            st.markdown('<div class="tv-caption" style="margin-top:24px;">本報告由系統自動整理公開資料與統計方法產生，僅供研究參考，不構成投資建議，投資請自行判斷並留意風險。</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# TAB 6：市場觀察（產業類股表現 ＋ 個股漲跌分布）
# ------------------------------------------------------------
with tab_market:
    st.markdown(
        '<div class="section-head"><div>'
        '<div class="section-title">市場觀察</div>'
        '<div class="section-help">當日產業資金流向與個股漲跌結構，'
        '資料來源：證交所／櫃買中心官方類股指數與個股 OpenAPI；每 10 分鐘更新，連線失敗時自動顯示最近成功快取。</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    mp_market_label = st.radio(
        "資料範圍", ["上市", "上櫃"], horizontal=True, key="market_pulse_scope",
        label_visibility="collapsed",
    )
    mp_market = "TW" if mp_market_label == "上市" else "TWO"

    mp_sub = st.radio(
        "檢視", ["📊 產業類股表現", "📈 個股漲跌分布", "📅 市場行事曆"], horizontal=True,
        key="market_pulse_view", label_visibility="collapsed",
    )

    if mp_sub == "📅 市場行事曆":
        st.markdown(
            '<div class="section-head" style="margin-top:8px;"><div>'
            '<div class="section-title" style="font-size:16px;">市場重要行事曆</div>'
            '<div class="section-help">近期法說會與除權息一覽，資料來源：Yahoo 股市台股行事曆；'
            '法說會通常與財報公布高度相關，僅供參考，實際日期以公司公告為準。</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        with st.spinner("正在載入市場行事曆…"):
            cal_data = fetch_market_calendar_events()

        earn_rows = cal_data.get("法說會", [])
        div_rows = cal_data.get("除權息", [])

        cal_type = st.radio(
            "事件類型", ["法說會（財報相關）", "除權息"], horizontal=True,
            key="market_cal_type", label_visibility="collapsed",
        )

        if cal_type.startswith("法說會"):
            if earn_rows:
                near = [r for r in earn_rows if 0 <= r.get("距離天數", 999) <= 14]
                st.markdown(f"""
                <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:12px;">
                  <div class="tv-card"><div class="tv-label">近期法說會</div><div class="tv-value">{len(earn_rows)}</div><div class="tv-caption">已公告場次</div></div>
                  <div class="tv-card"><div class="tv-label">14 天內</div><div class="tv-value" style="color:var(--yellow);">{len(near)}</div><div class="tv-caption">需特別留意</div></div>
                  <div class="tv-card"><div class="tv-label">最近一場</div><div class="tv-value" style="font-size:16px;">{earn_rows[0].get('名稱','')} ({earn_rows[0].get('代碼','')})</div><div class="tv-caption">{earn_rows[0].get('日期','')} {earn_rows[0].get('時間','')}</div></div>
                </div>
                """, unsafe_allow_html=True)
                earn_df = pd.DataFrame(earn_rows)
                st.dataframe(
                    earn_df, hide_index=True, use_container_width=True,
                    column_config={
                        "日期": st.column_config.TextColumn("日期", width=95),
                        "時間": st.column_config.TextColumn("時間", width=70),
                        "代碼": st.column_config.TextColumn("代碼", width=70),
                        "名稱": st.column_config.TextColumn("名稱", width=100),
                        "事件": st.column_config.TextColumn("事件", width=70),
                        "距離天數": st.column_config.NumberColumn("距離(天)", width=80, format="%+d"),
                    },
                )
                st.caption("法說會通常會同步公布或討論最新季度財報與營運展望，是重要觀察時點。資料來源：Yahoo 股市。")
            else:
                st.info("目前無法取得法說會行事曆，可能是資料來源暫時無回應，請稍後再試。")
                st.link_button("開啟 Yahoo 法說會行事曆", "https://tw.stock.yahoo.com/calendar/earnings-call")
        else:
            if div_rows:
                near_div = [r for r in div_rows if 0 <= r.get("距離天數", 999) <= 7]
                st.markdown(f"""
                <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:12px;">
                  <div class="tv-card"><div class="tv-label">近期除權息</div><div class="tv-value">{len(div_rows)}</div><div class="tv-caption">已公告</div></div>
                  <div class="tv-card"><div class="tv-label">7 天內</div><div class="tv-value" style="color:var(--yellow);">{len(near_div)}</div><div class="tv-caption">即將除權息</div></div>
                  <div class="tv-card"><div class="tv-label">最近一檔</div><div class="tv-value" style="font-size:16px;">{div_rows[0].get('名稱','')} ({div_rows[0].get('代碼','')})</div><div class="tv-caption">{div_rows[0].get('日期','')}・現金 {fmt_num(div_rows[0].get('現金股利'), '{:.2f}')}</div></div>
                </div>
                """, unsafe_allow_html=True)
                div_df = pd.DataFrame(div_rows)
                st.dataframe(
                    div_df, hide_index=True, use_container_width=True,
                    column_config={
                        "日期": st.column_config.TextColumn("除權息日", width=95),
                        "代碼": st.column_config.TextColumn("代碼", width=70),
                        "名稱": st.column_config.TextColumn("名稱", width=100),
                        "事件": st.column_config.TextColumn("事件", width=70),
                        "現金股利": st.column_config.NumberColumn("現金股利", width=90, format="%.2f"),
                        "距離天數": st.column_config.NumberColumn("距離(天)", width=80, format="%+d"),
                    },
                )
                st.caption("除權息當日股價通常會調整，填息速度與殖利率可作為參考；實際發放與除權息日以公司公告為準。")
            else:
                st.info("目前無法取得除權息行事曆，可能是資料來源暫時無回應，請稍後再試。")
                st.link_button("開啟 Yahoo 除權息行事曆", "https://tw.stock.yahoo.com/calendar/dividend")

    else:
        with st.spinner("正在從證交所／櫃買中心取得市場資料…"):
            pulse = build_market_pulse(mp_market)

        if pulse and pulse.get("is_cached"):
            cached_at = pulse.get("saved_at", "")
            st.info(f"官方 API 暫時無法連線，現顯示最近一次成功快取{f'（{cached_at}）' if cached_at else ''}。")
        elif pulse:
            quote_date = pulse.get("quote_date", "")
            if quote_date:
                st.caption(f"官方最新資料日期：{quote_date}")

        if not pulse:
            st.warning(
                f"目前無法取得{mp_market_label}官方行情，且尚無可用快取。"
                "請稍後重新整理；首次部署若遇休市日，待下一個交易日成功抓取後便會建立快取。"
            )
        elif mp_sub == "📊 產業類股表現":
            ind = pulse.get("industry", pd.DataFrame())
            if ind.empty:
                coverage = float(pulse.get("industry_coverage", 0) or 0)
                st.warning(
                    "個股行情已取得，但公司產業對照資料暫時不可用。"
                    f"目前產業覆蓋率為 {coverage:.0%}；可先切換至「個股漲跌分布」，系統也會持續使用最近成功快取。"
                )
            else:
                _industry_method = pulse.get("industry_method", "official_index")
                if _industry_method == "equal_weight_fallback":
                    st.warning(
                        "官方類股指數端點目前暫時無法取得，以下漲跌幅為成分股等權平均備援；"
                        "此口徑不會與玩股網完全一致。官方指數恢復後會自動切回正確口徑。"
                    )
                elif _industry_method == "official_index_cache":
                    st.info("目前類股漲跌幅沿用同一交易日最近成功取得的官方指數快取。")

                pm_mode = st.radio(
                    "排序", ["漲幅", "跌幅", "成交比重"], horizontal=True,
                    key="market_pulse_ind_mode", label_visibility="collapsed",
                )
                _val_col = "漲跌%" if pm_mode != "成交比重" else "成交比重%"
                if pm_mode == "漲幅":
                    view = ind.sort_values("漲跌%", ascending=False).head(15)
                elif pm_mode == "跌幅":
                    view = ind.sort_values("漲跌%", ascending=True).head(15)
                else:
                    if "成交比重%" not in ind.columns:
                        st.info("此市場的資料未包含成交比重欄位。")
                        view = ind.head(0)
                    else:
                        view = ind.sort_values("成交比重%", ascending=False).head(15)

                if not view.empty:
                    if _val_col == "漲跌%":
                        colors = ["#e0505a" if v >= 0 else "#3fae7a" for v in view[_val_col]]
                        texts = [f"{v:+.2f}%" for v in view[_val_col]]
                    else:
                        colors = ["#6ea8fe"] * len(view)
                        texts = [f"{v:.2f}%" for v in view[_val_col]]

                    fig = go.Figure(go.Bar(
                        x=view["類股"], y=view[_val_col],
                        marker_color=colors, text=texts, textposition="outside",
                        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
                    ))
                    fig.update_layout(
                        height=420, template="plotly_dark",
                        paper_bgcolor="#0d1624", plot_bgcolor="#0d1624",
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis=dict(tickangle=-30, tickfont=dict(size=13)),
                        yaxis=dict(gridcolor="rgba(148,163,184,0.10)",
                                   title="漲跌幅 (%)" if _val_col == "漲跌%" else "成交比重 (%)"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"mp_ind_{mp_market}_{pm_mode}")

                if pulse.get("industry_method") in ("official_index", "official_index_cache"):
                    st.caption(
                        f"資料來源：{pulse.get('index_source', '證交所／櫃買中心官方類股指數')}。"
                        f"{mp_market_label}類股漲跌幅直接採官方價格指數，與玩股網類股行情屬相同計算口徑；"
                        "成交比重則依官方個股成交額彙總。"
                    )
                else:
                    st.caption(
                        "資料來源：證交所／櫃買中心官方個股行情。"
                        "目前為成分股等權平均備援，不能直接拿來與玩股網的類股價格指數比較。"
                    )

                with st.expander("完整產業列表", expanded=False):
                    _asc = pm_mode == "跌幅"
                    show = ind.sort_values(_val_col if _val_col in ind.columns else "漲跌%",
                                           ascending=_asc).copy()
                    show_cols = ["類股", "漲跌%"]
                    if "成交比重%" in show.columns:
                        show_cols.append("成交比重%")
                    if "成交額" in show.columns:
                        show_cols.append("成交額")

                    def _mp_color_tw(val):
                        if pd.isna(val): return ''
                        c = '#22ab94' if val > 0 else '#f23645' if val < 0 else '#e6edf3'
                        return f'color: {c}; font-weight: bold'

                    st.dataframe(
                        show[show_cols].style.map(
                            _mp_color_tw, subset=["漲跌%"]),
                        use_container_width=True, hide_index=True,
                        column_config={
                            "類股": st.column_config.TextColumn("類股"),
                            "漲跌%": st.column_config.NumberColumn("漲跌%", format="%.2f%%"),
                            "成交比重%": st.column_config.NumberColumn("成交比重", format="%.2f%%"),
                            "成交額": st.column_config.NumberColumn("成交額(億)", format="%.1f"),
                        },
                    )

        elif mp_sub == "📈 個股漲跌分布":
            breadth = pulse.get("breadth", {})
            distribution = pulse.get("distribution", {})
            if not breadth or not distribution:
                st.warning("無法取得漲跌家數分布，請稍後再試。")
            else:
                # 依使用者指定的圖二順序：漲停 → 5% → 2.5% → 0.2% → 平盤 → 負向區間 → 跌停。
                labels = ["漲停", "5%", "2.5%", "0.2%", "平盤", "-0.2%", "-2.5%", "-5%", "跌停"]
                values = [int(distribution.get(k, 0) or 0) for k in labels]
                ranges = [
                    "≥ +9.5%", "+5.0% ～ < +9.5%", "+2.5% ～ < +5.0%",
                    "> +0.2% ～ < +2.5%", "-0.2% ～ +0.2%",
                    "> -2.5% ～ < -0.2%", "> -5.0% ～ ≤ -2.5%",
                    "> -9.5% ～ ≤ -5.0%", "≤ -9.5%",
                ]
                bar_colors = [
                    "#b8323e", "#cf4653", "#df6973", "#efa0a7", "#94a3b8",
                    "#b4ddc9", "#82cba9", "#4eb282", "#2f8f68",
                ]

                fig = go.Figure(go.Bar(
                    x=labels, y=values, marker_color=bar_colors,
                    customdata=ranges,
                    text=[f"{v:,}" for v in values], textposition="outside",
                    hovertemplate="%{x}<br>區間：%{customdata}<br>%{y:,} 檔<extra></extra>",
                ))
                fig.update_layout(
                    height=440, template="plotly_dark",
                    paper_bgcolor="#0d1624", plot_bgcolor="#0d1624",
                    margin=dict(l=10, r=10, t=30, b=55),
                    xaxis=dict(
                        categoryorder="array", categoryarray=labels,
                        tickangle=-35, tickfont=dict(size=13),
                    ),
                    yaxis=dict(gridcolor="rgba(148,163,184,0.10)", title="家數", rangemode="tozero"),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"mp_dist_v2_{mp_market}")

                st.caption(
                    "分類門檻：漲停 ≥ 9.5%；5% 為 5～未滿 9.5%；2.5% 為 2.5～未滿 5%；"
                    "0.2% 為大於 0.2～未滿 2.5%；平盤為 -0.2～+0.2%；負向區間採對稱門檻。"
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f'<div class="tv-card"><div class="tv-label">上漲家數</div>'
                        f'<div class="tv-value" style="color:#e0505a;">{breadth.get("上漲", 0):,}</div>'
                        f'<div class="tv-caption">含漲停 {breadth.get("漲停", 0):,} 檔</div></div>',
                        unsafe_allow_html=True)
                with c2:
                    st.markdown(
                        f'<div class="tv-card"><div class="tv-label">紅K / 黑K</div>'
                        f'<div class="tv-value" style="color:#e6edf3;">'
                        f'{breadth.get("紅K", 0):,} / {breadth.get("黑K", 0):,}</div>'
                        f'<div class="tv-caption">收盤高於開盤為紅K</div></div>',
                        unsafe_allow_html=True)
                with c3:
                    st.markdown(
                        f'<div class="tv-card"><div class="tv-label">下跌家數</div>'
                        f'<div class="tv-value" style="color:#3fae7a;">{breadth.get("下跌", 0):,}</div>'
                        f'<div class="tv-caption">含跌停 {breadth.get("跌停", 0):,} 檔</div></div>',
                        unsafe_allow_html=True)

                st.caption(
                    f"資料來源：證交所／櫃買中心官方 OpenAPI。"
                    f"漲跌家數為{mp_market_label}一般股票最新收盤行情統計。"
                )
