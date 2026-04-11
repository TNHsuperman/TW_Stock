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
if 'debug_log' not in st.session_state:
    st.session_state['debug_log'] = []

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
# 3. 診斷函數：測試單支股票能否成功下載
# ============================================================

def diagnose_api() -> dict:
    """
    測試各個 API endpoint 是否可以從 Streamlit Cloud 連線，
    回傳診斷結果供使用者參考。
    """
    results = {}

    # 測試 TWSE STOCK_DAY（2330 台積電，最近一個月）
    try:
        date_str = datetime.now().strftime('%Y%m01')
        url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&stockNo=2330&date={date_str}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        data = resp.json()
        stat = data.get('stat', 'unknown')
        row_count = len(data.get('data', []))
        results['TWSE STOCK_DAY'] = f"✅ HTTP {resp.status_code}, stat={stat}, rows={row_count}"
    except Exception as e:
        results['TWSE STOCK_DAY'] = f"❌ {type(e).__name__}: {str(e)[:80]}"

    # 測試 TPEX
    try:
        yr_roc = datetime.now().year - 1911
        mo = datetime.now().month
        url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d={yr_roc}/{mo:02d}&stkno=6488&s=0,asc"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        data = resp.json()
        row_count = len(data.get('aaData', []))
        results['TPEX STOCK_DAY'] = f"✅ HTTP {resp.status_code}, rows={row_count}"
    except Exception as e:
        results['TPEX STOCK_DAY'] = f"❌ {type(e).__name__}: {str(e)[:80]}"

    # 測試 TWSE ISIN（股票清單來源）
    try:
        url = 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2'
        resp = requests.get(url, headers=HEADERS, timeout=10, verify=False)
        results['TWSE ISIN'] = f"✅ HTTP {resp.status_code}, bytes={len(resp.content)}"
    except Exception as e:
        results['TWSE ISIN'] = f"❌ {type(e).__name__}: {str(e)[:80]}"

    return results


# ============================================================
# 4. 歷史資料抓取
# ============================================================

def parse_price(s):
    try:
        v = float(str(s).replace(',', '').strip())
        return v if v > 0 else None
    except Exception:
        return None


def fetch_twse_history(code: str, months: int = 4) -> tuple[pd.DataFrame, str]:
    """
    回傳 (DataFrame, 錯誤訊息)，方便除錯。
    """
    all_rows = []
    last_error = ""

    for i in range(months, -1, -1):
        target   = datetime.now().replace(day=1) - timedelta(days=i * 28)
        date_str = target.strftime('%Y%m01')
        url = (f"https://www.twse.com.tw/exchangeReport/STOCK_DAY"
               f"?response=json&stockNo={code}&date={date_str}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                time.sleep(0.1)
                continue
            data = resp.json()
            if data.get('stat') != 'OK':
                last_error = f"stat={data.get('stat')}"
                time.sleep(0.1)
                continue
            fields = data.get('fields', [])
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
                            'Date':  f"{int(parts[0])+1911}-{parts[1]}-{parts[2]}",
                            'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': v
                        })
                except Exception:
                    continue
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:60]}"
        time.sleep(0.08)

    if not all_rows:
        return pd.DataFrame(), last_error

    df = pd.DataFrame(all_rows)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').drop_duplicates('Date').set_index('Date'), ""


def fetch_tpex_history(code: str, months: int = 4) -> tuple[pd.DataFrame, str]:
    all_rows = []
    last_error = ""

    for i in range(months, -1, -1):
        target   = datetime.now().replace(day=1) - timedelta(days=i * 28)
        yr_roc   = target.year - 1911
        date_str = f"{yr_roc}/{target.month:02d}"
        url = (f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/"
               f"st43_result.php?l=zh-tw&d={date_str}&stkno={code}&s=0,asc")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                time.sleep(0.1)
                continue
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
                            'Date':  f"{int(parts[0])+1911}-{parts[1]}-{parts[2]}",
                            'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': v
                        })
                except Exception:
                    continue
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:60]}"
        time.sleep(0.08)

    if not all_rows:
        return pd.DataFrame(), last_error

    df = pd.DataFrame(all_rows)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').drop_duplicates('Date').set_index('Date'), ""


def fetch_history(code: str, market: str, months: int = 4) -> pd.DataFrame:
    if market == "TW":
        df, _ = fetch_twse_history(code, months)
    else:
        df, _ = fetch_tpex_history(code, months)
    return df


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
    total     = len(all_stocks)
    completed = 0
    results   = []
    empty_count = 0

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(fetch_and_analyze, c, m, strategy): (c, m)
            for c, m in all_stocks
        }
        for future in as_completed(futures):
            completed += 1
            if completed % 30 == 0 or completed == total:
                pct = completed / max(total, 1)
                progress_bar.progress(min(pct, 1.0))
                status_text.text(
                    f"🔍 進度：{completed} / {total}  |  找到 {len(results)} 支"
                    + (f"  |  ⚠️ {empty_count} 支無資料" if empty_count > 20 else "")
                )
            try:
                result = future.result()
                if result is None:
                    empty_count += 1
                else:
                    code = result["code"]
                    info = info_map.get(code, {"name": "未知", "industry": "其他"})
                    # 成交量篩選（在分析後才過濾，因為預篩已移除）
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
    return res_df, empty_count


# ============================================================
# 7. 側邊欄
# ============================================================

st.sidebar.header("⚙️ 策略參數設定")
strategy_option  = st.sidebar.radio("選擇選股策略：", ("均線多頭回測", "均線糾結偵測"))
indicator_choice = st.sidebar.selectbox("查看確認指標：", ["都不顯示", "RSI (強弱指標)", "MACD (趨勢指標)"])
min_volume       = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)

st.sidebar.markdown("---")
# 診斷工具
if st.sidebar.button("🔧 診斷 API 連線"):
    with st.sidebar:
        with st.spinner("測試中..."):
            diag = diagnose_api()
        for k, v in diag.items():
            st.sidebar.text(f"{k}:\n{v}")

st.sidebar.caption("📡 TWSE / TPEX 官方 API")


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

    res_df, empty_count = run_scan(
        all_stocks, info_map, strategy_option, min_volume,
        progress_bar, status_text
    )

    if empty_count > len(all_stocks) * 0.8:
        # 超過 80% 的股票回傳空資料，代表 API 被封鎖
        status_text.warning(
            f"⚠️ {empty_count} 支股票無法取得資料（佔 {empty_count/len(all_stocks)*100:.0f}%）。\n\n"
            "可能原因：TWSE/TPEX 封鎖了 Streamlit Cloud 的 IP。\n"
            "請點選左側「🔧 診斷 API 連線」確認連線狀態。"
        )
    else:
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
                    d = close_s.diff()
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
    st.info("💡 點擊「開始全市場掃描」按鈕。\n\n若持續出現 0 支結果，請先點側邊欄的「🔧 診斷 API 連線」確認連線狀態。")
