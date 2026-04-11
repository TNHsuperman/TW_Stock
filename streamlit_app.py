import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import urllib3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板", layout="wide")

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False
if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = 0

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


# ============================================================
# 2. 股票清單
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_info_map():
    stock_info_map = {}
    stocks_list = []
    urls = [
        ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
        ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")
    ]
    for url, market in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15, verify=False)
            df_list = pd.read_html(StringIO(resp.text), flavor='lxml')
            df = df_list[0]
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            for _, row in df.iterrows():
                val = row['有價證券代號及名稱']
                industry = row['產業別']
                if val and '　' in str(val):
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        stocks_list.append((code, market))
                        stock_info_map[code] = {
                            "name": name,
                            "market": market,
                            "industry": industry if pd.notna(industry) else "其他"
                        }
        except Exception:
            continue
    return stocks_list, stock_info_map


# ============================================================
# 3. 核心：TWSE / TPEX 批次月報 API
#    一次請求 → 全市場整月的每日資料
# ============================================================

def parse_price(s):
    """安全轉換價格字串，處理逗號、-- 等異常值。"""
    try:
        v = float(str(s).replace(',', '').strip())
        return v if v > 0 else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_twse_monthly_all(year: int, month: int) -> pd.DataFrame:
    """
    TWSE「每日收盤行情（全部）」API：
    https://www.twse.com.tw/exchangeReport/MI_INDEX
    一次回傳整月所有上市股票的每日成交資料。

    回傳 DataFrame：index=Date, columns=MultiIndex(code, OHLCV)
    這裡我們只取最後一個交易日的收盤與成交量，用於預篩。
    """
    date_str = f"{year}{month:02d}01"
    url = (f"https://www.twse.com.tw/exchangeReport/MI_INDEX"
           f"?response=json&date={date_str}&type=ALLBUT0999")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        data = resp.json()
        if data.get('stat') != 'OK':
            return pd.DataFrame()

        # MI_INDEX 回傳多張表，找含「證券代號」欄位的那張
        for key in data:
            if not isinstance(data.get(key), list):
                continue
            fields_key = key.replace('data', 'fields') if 'data' in key else None
            if not fields_key or fields_key not in data:
                continue
            fields = data[fields_key]
            if '證券代號' not in fields or '收盤價' not in fields:
                continue

            fi_code  = fields.index('證券代號')
            fi_vol   = fields.index('成交股數') if '成交股數' in fields else None
            fi_close = fields.index('收盤價')

            rows = []
            for row in data[key]:
                try:
                    code  = str(row[fi_code]).strip()
                    close = parse_price(row[fi_close])
                    vol   = parse_price(row[fi_vol]) if fi_vol is not None else None
                    if len(code) == 4 and code.isdigit() and close:
                        rows.append({
                            'code':     code,
                            'close':    close,
                            'volume_k': (vol / 1000) if vol else 0,
                        })
                except Exception:
                    continue
            if rows:
                return pd.DataFrame(rows).set_index('code')
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_tpex_monthly_all(year: int, month: int) -> pd.DataFrame:
    """
    TPEX 月成交資料（全部上櫃股票）。
    """
    yr_roc = year - 1911
    url = (f"https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/"
           f"stk_wn1430_result.php?l=zh-tw&d={yr_roc}/{month:02d}&se=AL")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        data = resp.json()
        rows = []
        for row in data.get('aaData', []):
            try:
                code  = str(row[0]).strip()
                close = parse_price(row[2])
                vol   = parse_price(row[8])
                if len(code) == 4 and code.isdigit() and close:
                    rows.append({'code': code, 'close': close, 'volume_k': vol or 0})
            except Exception:
                continue
        if rows:
            return pd.DataFrame(rows).set_index('code')
    except Exception:
        pass
    return pd.DataFrame()


# ============================================================
# 4. 個股歷史日線（逐月串接）
# ============================================================

def fetch_twse_history(code: str, months: int = 4) -> pd.DataFrame:
    """抓取單支上市股票歷史日線，用欄位名稱解析確保正確性。"""
    all_rows = []
    for i in range(months, -1, -1):
        target   = datetime.now().replace(day=1) - timedelta(days=i * 28)
        date_str = target.strftime('%Y%m01')
        url = (f"https://www.twse.com.tw/exchangeReport/STOCK_DAY"
               f"?response=json&stockNo={code}&date={date_str}")
        for attempt in range(2):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=12)
                if resp.status_code != 200:
                    break
                data = resp.json()
                if data.get('stat') != 'OK':
                    break
                fields = data.get('fields', [])
                # 用欄位名稱定位，找不到才用預設索引
                idx = {
                    'date':  fields.index('日期')   if '日期'   in fields else 0,
                    'vol':   fields.index('成交股數') if '成交股數' in fields else 1,
                    'open':  fields.index('開盤價') if '開盤價'  in fields else 3,
                    'high':  fields.index('最高價') if '最高價'  in fields else 4,
                    'low':   fields.index('最低價') if '最低價'  in fields else 5,
                    'close': fields.index('收盤價') if '收盤價'  in fields else 6,
                }
                for row in data.get('data', []):
                    try:
                        parts = str(row[idx['date']]).split('/')
                        if len(parts) != 3:
                            continue
                        o = parse_price(row[idx['open']])
                        h = parse_price(row[idx['high']])
                        l = parse_price(row[idx['low']])
                        c = parse_price(row[idx['close']])
                        v = parse_price(row[idx['vol']])
                        if all(x is not None for x in [o, h, l, c, v]):
                            all_rows.append({
                                'Date':   f"{int(parts[0])+1911}-{parts[1]}-{parts[2]}",
                                'Open':  o, 'High': h, 'Low': l, 'Close': c, 'Volume': v
                            })
                    except Exception:
                        continue
                break
            except Exception:
                if attempt == 0:
                    time.sleep(0.3)
        time.sleep(0.08)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').drop_duplicates('Date').set_index('Date')


def fetch_tpex_history(code: str, months: int = 4) -> pd.DataFrame:
    """抓取單支上櫃股票歷史日線。"""
    all_rows = []
    for i in range(months, -1, -1):
        target   = datetime.now().replace(day=1) - timedelta(days=i * 28)
        yr_roc   = target.year - 1911
        date_str = f"{yr_roc}/{target.month:02d}"
        url = (f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/"
               f"st43_result.php?l=zh-tw&d={date_str}&stkno={code}&s=0,asc")
        for attempt in range(2):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=12)
                if resp.status_code != 200:
                    break
                data = resp.json()
                for row in data.get('aaData', []):
                    try:
                        parts = str(row[0]).split('/')
                        if len(parts) != 3:
                            continue
                        o = parse_price(row[2])
                        h = parse_price(row[3])
                        l = parse_price(row[4])
                        c = parse_price(row[5])
                        v = parse_price(row[1])
                        if all(x is not None for x in [o, h, l, c, v]):
                            all_rows.append({
                                'Date':   f"{int(parts[0])+1911}-{parts[1]}-{parts[2]}",
                                'Open':  o, 'High': h, 'Low': l, 'Close': c, 'Volume': v
                            })
                    except Exception:
                        continue
                break
            except Exception:
                if attempt == 0:
                    time.sleep(0.3)
        time.sleep(0.08)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').drop_duplicates('Date').set_index('Date')


def fetch_history(code: str, market: str, months: int = 4) -> pd.DataFrame:
    return fetch_twse_history(code, months) if market == "TW" else fetch_tpex_history(code, months)


# ============================================================
# 5. 分析
# ============================================================

def analyze(df: pd.DataFrame, strategy: str) -> dict | None:
    if len(df) < 60:
        return None
    close = float(df['Close'].iloc[-1])
    m30   = float(df['Close'].rolling(30).mean().iloc[-1])
    m45   = float(df['Close'].rolling(45).mean().iloc[-1])
    m60   = float(df['Close'].rolling(60).mean().iloc[-1])
    if any(np.isnan(v) for v in [close, m30, m45, m60]):
        return None
    bias_30 = ((close - m30) / m30) * 100
    if strategy == "均線多頭回測":
        if m30 > m45 > m60 and close > m30 and bias_30 < 2.0:
            return {"close": close, "bias_30": bias_30}
    elif strategy == "均線糾結偵測":
        spread = (max(m30, m45, m60) - min(m30, m45, m60)) / min(m30, m45, m60)
        if spread <= 0.02 and abs(bias_30) < 2.0:
            return {"close": close, "bias_30": bias_30}
    return None


def fetch_and_analyze(code: str, market: str, strategy: str) -> dict | None:
    try:
        df = fetch_history(code, market, months=4)
        if df.empty:
            return None
        result = analyze(df, strategy)
        if result:
            result.update({"code": code, "market": market})
        return result
    except Exception:
        return None


# ============================================================
# 6. 掃描主流程
# ============================================================

def run_scan(all_stocks, info_map, strategy, min_vol, progress_bar, status_text):
    now = datetime.now()
    stock_dict = {code: market for code, market in all_stocks}

    # ── 第一階段：批次月報預篩 ──
    status_text.text("⚡ 第一階段：抓取全市場月報進行預篩...")
    progress_bar.progress(0.03)

    tw_snap  = get_twse_monthly_all(now.year, now.month)
    two_snap = get_tpex_monthly_all(now.year, now.month)

    # 若本月資料不足（月初），嘗試上個月
    if tw_snap.empty:
        prev = now.replace(day=1) - timedelta(days=1)
        tw_snap = get_twse_monthly_all(prev.year, prev.month)
    if two_snap.empty:
        prev = now.replace(day=1) - timedelta(days=1)
        two_snap = get_tpex_monthly_all(prev.year, prev.month)

    snapshot = pd.concat([tw_snap, two_snap]) if not tw_snap.empty or not two_snap.empty else pd.DataFrame()

    if not snapshot.empty and 'volume_k' in snapshot.columns:
        snapshot['volume_k'] = pd.to_numeric(snapshot['volume_k'], errors='coerce').fillna(0)
        passed     = snapshot[snapshot['volume_k'] >= min_vol]
        candidates = [(c, stock_dict[c]) for c in passed.index if c in stock_dict]
        status_text.text(
            f"✅ 預篩完成：{len(all_stocks)} 支 → {len(candidates)} 支候選"
            f"（成交量 ≥ {min_vol} 張）"
        )
    else:
        candidates = all_stocks
        status_text.text(f"⚠️ 月報預篩失敗，改為掃描全部 {len(candidates)} 支...")

    progress_bar.progress(0.08)

    # ── 第二階段：歷史精算 ──
    total     = len(candidates)
    completed = 0
    results   = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_and_analyze, c, m, strategy): c
            for c, m in candidates
        }
        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0 or completed == total:
                pct = 0.08 + 0.92 * (completed / max(total, 1))
                progress_bar.progress(min(pct, 1.0))
                status_text.text(
                    f"🔍 精算進度：{completed} / {total}  |  已找到 {len(results)} 支"
                )
            try:
                result = future.result()
                if result:
                    code = result["code"]
                    info = info_map.get(code, {"name": "未知", "industry": "其他"})
                    results.append({
                        "代碼":    code,
                        "名稱":    info["name"],
                        "市場":    result["market"],
                        "類股":    info["industry"],
                        "收盤":    round(result["close"], 2),
                        "乖離(%)": round(result["bias_30"], 2),
                        "_code":   code,
                        "_market": result["market"],
                    })
            except Exception:
                continue

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values("乖離(%)").reset_index(drop=True)
    return res_df


# ============================================================
# 7. 側邊欄
# ============================================================

st.sidebar.header("⚙️ 策略參數設定")
strategy_option  = st.sidebar.radio("選擇選股策略：", ("均線多頭回測", "均線糾結偵測"))
indicator_choice = st.sidebar.selectbox("查看確認指標：", ["都不顯示", "RSI (強弱指標)", "MACD (趨勢指標)"])
min_volume       = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)
st.sidebar.markdown("---")
st.sidebar.caption("📡 TWSE / TPEX 官方 API\n\n⚡ 月報批次預篩 → 歷史日線精算")


# ============================================================
# 8. 掃描按鈕
# ============================================================

if not st.session_state['is_scanning']:
    if st.button("🔍 開始全市場掃描", use_container_width=True):
        st.session_state['is_scanning'] = True
        st.rerun()
else:
    all_stocks, info_map = get_stock_info_map()
    progress_bar = st.progress(0)
    status_text  = st.empty()
    res_df = run_scan(all_stocks, info_map, strategy_option, min_volume, progress_bar, status_text)
    status_text.text(f"🎉 完成！找到 {len(res_df)} 支符合條件標的。")
    st.session_state['scan_results'] = res_df
    st.session_state['is_scanning']  = False
    st.rerun()


# ============================================================
# 9. 結果顯示
# ============================================================

if not st.session_state['scan_results'].empty:
    df_raw = st.session_state['scan_results']
    selected_industry = st.selectbox("🎯 篩選類股：", ["全部"] + sorted(df_raw["類股"].unique().tolist()))
    df_filtered = (df_raw if selected_industry == "全部"
                   else df_raw[df_raw["類股"] == selected_industry]).reset_index(drop=True)
    display_cols = ["代碼", "名稱", "市場", "類股", "收盤", "乖離(%)"]

    if len(df_filtered) > 0:
        st.write(f"📊 符合條件標的：{len(df_filtered)} 支")
        event = st.dataframe(
            df_filtered[display_cols], hide_index=True, use_container_width=True,
            on_select="rerun", selection_mode="single-row", key="stock_table"
        )
        if event.selection and event.selection.rows:
            st.session_state['selected_index'] = event.selection.rows[0]

        st.write("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("⬅️ 上一支", use_container_width=True):
                st.session_state['selected_index'] = (st.session_state['selected_index'] - 1) % len(df_filtered)
        with c2:
            st.markdown(
                f"<h3 style='text-align:center;color:#9400D3;'>"
                f"{st.session_state['selected_index']+1} / {len(df_filtered)}</h3>",
                unsafe_allow_html=True)
        with c3:
            if st.button("下一支 ➡️", use_container_width=True):
                st.session_state['selected_index'] = (st.session_state['selected_index'] + 1) % len(df_filtered)

        idx = min(st.session_state['selected_index'], len(df_filtered) - 1)
        sel = df_filtered.iloc[idx]

        with st.spinner(f'載入中... {sel["名稱"]} ({sel["代碼"]})'):
            df_p = fetch_history(sel["_code"], sel["_market"], months=10)
            if df_p.empty:
                st.error("無法載入該股票資料，請稍後再試。")
            else:
                df_p = df_p.dropna(subset=['Close'])
                df_p['Date_Str'] = df_p.index.strftime('%Y-%m-%d')
                close_s      = df_p['Close']
                df_p['30MA'] = close_s.rolling(30).mean()
                df_p['45MA'] = close_s.rolling(45).mean()
                df_p['60MA'] = close_s.rolling(60).mean()

                n_rows      = 3 if indicator_choice != "都不顯示" else 2
                row_heights = [0.6, 0.15, 0.25] if n_rows == 3 else [0.75, 0.25]
                fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.03, row_heights=row_heights)

                fig.add_trace(go.Candlestick(
                    x=df_p['Date_Str'], open=df_p['Open'], high=df_p['High'],
                    low=df_p['Low'], close=df_p['Close'], name="K線"), row=1, col=1)

                for ma, color in zip(['30MA','45MA','60MA'], ['#FFA500','#2E8B57','#4169E1']):
                    fig.add_trace(go.Scatter(
                        x=df_p['Date_Str'], y=df_p[ma],
                        line=dict(color=color, width=1.5), name=ma), row=1, col=1)

                v_clrs = ['#ef5350' if c >= o else '#26a69a'
                          for c, o in zip(df_p['Close'], df_p['Open'])]
                fig.add_trace(go.Bar(
                    x=df_p['Date_Str'], y=df_p['Volume'],
                    marker_color=v_clrs, name="成交量"), row=2, col=1)

                if indicator_choice == "RSI (強弱指標)":
                    d   = close_s.diff()
                    rsi = 100 - (100 / (1 + d.where(d>0,0).rolling(14).mean() /
                                        (-d.where(d<0,0).rolling(14).mean() + 1e-9)))
                    fig.add_trace(go.Scatter(x=df_p['Date_Str'], y=rsi,
                        line=dict(color='purple', width=1.2), name="RSI"), row=3, col=1)
                    fig.add_hline(y=70, line_dash="dot", line_color="red",   row=3, col=1)
                    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

                elif indicator_choice == "MACD (趨勢指標)":
                    dif = close_s.ewm(span=12).mean() - close_s.ewm(span=26).mean()
                    h_s = dif - dif.ewm(span=9).mean()
                    fig.add_trace(go.Bar(x=df_p['Date_Str'], y=h_s,
                        marker_color=['#ef5350' if v>=0 else '#26a69a' for v in h_s],
                        name="MACD柱"), row=3, col=1)

                fig.update_xaxes(type='category', showgrid=True, gridcolor='rgba(200,200,200,0.2)')
                fig.update_layout(
                    title=f"<b>{sel['名稱']} ({sel['代碼']})</b>",
                    xaxis_rangeslider_visible=False, height=700, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("目前篩選條件下無標的。")
else:
    st.info("💡 點擊「開始全市場掃描」按鈕。⚡ 月報批次預篩大幅縮短掃描時間。")
