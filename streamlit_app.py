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
if 'yf_session' not in st.session_state:
    st.session_state['yf_session'] = None  # 快取 Yahoo Finance session（cookie+crumb）


# ============================================================
# 2. 股票清單
# ============================================================

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

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
                        suffix = ".TW" if market == "TW" else ".TWO"
                        ticker = f"{code}{suffix}"
                        stocks_list.append((ticker, market))
                        stock_info_map[ticker] = {
                            "name": name,
                            "code": code,
                            "market": market,
                            "industry": industry if pd.notna(industry) else "其他"
                        }
        except Exception:
            continue
    return stocks_list, stock_info_map


# ============================================================
# 3. Yahoo Finance Cookie + Crumb（繞過 IP 封鎖的關鍵）
#
#   Yahoo Finance 新版 API 需要：
#   1. 先取得 Cookie（訪問首頁）
#   2. 再用 Cookie 取得 Crumb（一次性驗證碼）
#   3. 每次 API 請求帶上 Cookie 和 Crumb
#
#   Cookie + Crumb 只需取得一次，後續所有請求共用。
#   這樣就不會被當成機器人封鎖。
# ============================================================

@st.cache_data(ttl=3600)  # Cookie 快取 1 小時
def get_yf_cookie_and_crumb() -> tuple[dict, str]:
    """
    取得 Yahoo Finance 的 Cookie 和 Crumb。
    Cookie 有效期約數小時，快取 1 小時確保不過期。
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/124.0.0.0 Safari/537.36'),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    })

    # 步驟 1：訪問 Yahoo Finance 首頁取得 Cookie
    try:
        resp = session.get('https://fc.yahoo.com', timeout=10)
    except Exception:
        pass
    try:
        resp = session.get('https://finance.yahoo.com', timeout=10)
    except Exception:
        pass

    cookies = dict(session.cookies)

    # 步驟 2：用 Cookie 換取 Crumb
    crumb = ""
    try:
        resp = session.get(
            'https://query1.finance.yahoo.com/v1/test/csrfToken',
            timeout=10
        )
        crumb = resp.text.strip()
    except Exception:
        pass

    if not crumb:
        try:
            resp = session.get(
                'https://query2.finance.yahoo.com/v1/test/csrfToken',
                timeout=10
            )
            crumb = resp.text.strip()
        except Exception:
            pass

    return cookies, crumb


def fetch_yf_history(ticker: str, days: int = 130) -> pd.DataFrame:
    """
    直接呼叫 Yahoo Finance v8 CSV API 下載歷史日線。
    帶上 Cookie + Crumb 避免被封鎖。

    相較於 yfinance 套件的優點：
    - 不依賴套件版本，直接控制 HTTP 請求
    - 可以共用 Cookie，減少被封鎖的機率
    - 輕量，不需要初始化 Ticker 物件
    """
    cookies, crumb = get_yf_cookie_and_crumb()

    p2 = int(datetime.now().timestamp())
    p1 = int((datetime.now() - timedelta(days=days)).timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1":  p1,
        "period2":  p2,
        "interval": "1d",
        "events":   "history",
        "crumb":    crumb,
    }
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/124.0.0.0 Safari/537.36'),
        'Accept': '*/*',
        'Referer': 'https://finance.yahoo.com',
    }

    try:
        resp = requests.get(url, params=params, headers=headers,
                            cookies=cookies, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        result = data.get('chart', {}).get('result', [])
        if not result:
            return pd.DataFrame()

        r         = result[0]
        timestamps = r.get('timestamp', [])
        quote      = r.get('indicators', {}).get('quote', [{}])[0]

        if not timestamps:
            return pd.DataFrame()

        df = pd.DataFrame({
            'Date':   pd.to_datetime(timestamps, unit='s').normalize(),
            'Open':   quote.get('open',   [None] * len(timestamps)),
            'High':   quote.get('high',   [None] * len(timestamps)),
            'Low':    quote.get('low',    [None] * len(timestamps)),
            'Close':  quote.get('close',  [None] * len(timestamps)),
            'Volume': quote.get('volume', [None] * len(timestamps)),
        })

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['Close'])
        return df.sort_values('Date').drop_duplicates('Date').set_index('Date')

    except Exception:
        return pd.DataFrame()


# ============================================================
# 4. 分析
# ============================================================

def analyze(df: pd.DataFrame, strategy: str, min_vol: float) -> dict | None:
    if len(df) < 60:
        return None
    # Volume 單位為股，除 1000 換算為張
    if (df['Volume'].tail(5).mean() / 1000) < min_vol:
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


def fetch_and_analyze(ticker: str, market: str, strategy: str,
                      min_vol: float) -> dict | None:
    try:
        df = fetch_yf_history(ticker, days=130)
        if df.empty:
            return None
        result = analyze(df, strategy, min_vol)
        if result:
            result.update({"ticker": ticker, "market": market})
        return result
    except Exception:
        return None


# ============================================================
# 5. 掃描主流程
# ============================================================

def run_scan(all_stocks, info_map, strategy, min_vol, progress_bar, status_text):
    """
    Yahoo Finance v8 API + Cookie/Crumb + ThreadPoolExecutor。
    無請求次數限制，MAX_WORKERS=15 在 Streamlit Cloud 實測穩定。
    """
    MAX_WORKERS = 15
    total     = len(all_stocks)
    completed = 0
    results   = []
    empty_cnt = 0

    # 預先取得 Cookie（共用，不用每次重新取）
    status_text.text("🍪 取得 Yahoo Finance 授權中...")
    get_yf_cookie_and_crumb()  # 觸發快取

    status_text.text(f"🔍 掃描 {total} 支（{MAX_WORKERS} 條執行緒並行）...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_and_analyze, t, m, strategy, min_vol): t
            for t, m in all_stocks
        }
        for future in as_completed(futures):
            completed += 1
            if completed % 30 == 0 or completed == total:
                progress_bar.progress(min(completed / total, 1.0))
                status_text.text(
                    f"🔍 進度：{completed} / {total}  |  找到 {len(results)} 支"
                    + (f"  ⚠️ {empty_cnt} 支無資料" if empty_cnt > 200 else "")
                )
            try:
                result = future.result()
                if result is None:
                    empty_cnt += 1
                else:
                    ticker = result["ticker"]
                    info   = info_map.get(ticker, {"name": "未知", "industry": "其他", "code": ticker})
                    results.append({
                        "代碼":    info.get("code", ticker.split('.')[0]),
                        "名稱":    info["name"],
                        "市場":    result["market"],
                        "類股":    info["industry"],
                        "收盤":    round(result["close"], 2),
                        "乖離(%)": round(result["bias_30"], 2),
                        "_ticker": ticker,
                        "_market": result["market"],
                    })
            except Exception:
                continue

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values("乖離(%)").reset_index(drop=True)
    return res_df, empty_cnt


# ============================================================
# 6. 側邊欄
# ============================================================

st.sidebar.header("⚙️ 策略參數設定")
strategy_option  = st.sidebar.radio("選擇選股策略：", ("均線多頭回測", "均線糾結偵測"))
indicator_choice = st.sidebar.selectbox("查看確認指標：",
                                        ["都不顯示", "RSI (強弱指標)", "MACD (趨勢指標)"])
min_volume = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)
st.sidebar.markdown("---")

# 診斷按鈕
if st.sidebar.button("🔧 診斷 Yahoo Finance 連線"):
    with st.sidebar:
        with st.spinner("測試中..."):
            cookies, crumb = get_yf_cookie_and_crumb()
            st.sidebar.write(f"Cookie 數量: {len(cookies)}")
            st.sidebar.write(f"Crumb: {crumb[:20] + '...' if len(crumb) > 20 else crumb or '(空)'}")
            df_test = fetch_yf_history("2330.TW", days=10)
            if df_test.empty:
                st.sidebar.error("❌ 無法取得 2330.TW 資料")
            else:
                st.sidebar.success(f"✅ 2330.TW 成功，{len(df_test)} 筆，最新收盤：{df_test['Close'].iloc[-1]:.1f}")

st.sidebar.caption("📡 資料來源：Yahoo Finance v8 API（Cookie/Crumb 驗證）")


# ============================================================
# 7. 掃描按鈕
# ============================================================

if not st.session_state['is_scanning']:
    if st.button("🔍 開始全市場掃描", use_container_width=True):
        st.session_state['is_scanning'] = True
        st.rerun()
else:
    all_stocks, info_map = get_stock_info_map()
    progress_bar = st.progress(0)
    status_text  = st.empty()

    res_df, empty_cnt = run_scan(
        all_stocks, info_map, strategy_option, min_volume,
        progress_bar, status_text
    )

    if empty_cnt > len(all_stocks) * 0.8:
        status_text.warning(
            f"⚠️ {empty_cnt} 支無資料。Cookie/Crumb 可能過期，"
            "請重新整理頁面後再試。"
        )
    else:
        status_text.text(f"🎉 完成！找到 {len(res_df)} 支符合條件標的。")

    st.session_state['scan_results'] = res_df
    st.session_state['is_scanning']  = False
    st.rerun()


# ============================================================
# 8. 結果顯示
# ============================================================

if not st.session_state['scan_results'].empty:
    df_raw = st.session_state['scan_results']
    selected_industry = st.selectbox(
        "🎯 篩選類股：",
        ["全部"] + sorted(df_raw["類股"].unique().tolist())
    )
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
                st.session_state['selected_index'] = (
                    st.session_state['selected_index'] - 1) % len(df_filtered)
        with c2:
            st.markdown(
                f"<h3 style='text-align:center;color:#9400D3;'>"
                f"{st.session_state['selected_index']+1} / {len(df_filtered)}</h3>",
                unsafe_allow_html=True)
        with c3:
            if st.button("下一支 ➡️", use_container_width=True):
                st.session_state['selected_index'] = (
                    st.session_state['selected_index'] + 1) % len(df_filtered)

        idx = min(st.session_state['selected_index'], len(df_filtered) - 1)
        sel = df_filtered.iloc[idx]

        with st.spinner(f'載入中... {sel["名稱"]} ({sel["代碼"]})'):
            df_p = fetch_yf_history(sel["_ticker"], days=240)

            if df_p.empty:
                st.error("無法載入資料，請稍後再試。")
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

                fig.update_xaxes(type='category', showgrid=True,
                                 gridcolor='rgba(200,200,200,0.2)')
                fig.update_layout(
                    title=f"<b>{sel['名稱']} ({sel['代碼']})</b>",
                    xaxis_rangeslider_visible=False, height=700, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("目前篩選條件下無標的。")
else:
    st.info("💡 點擊「開始全市場掃描」按鈕。\n\n先點左側「🔧 診斷 Yahoo Finance 連線」確認連線正常。")
