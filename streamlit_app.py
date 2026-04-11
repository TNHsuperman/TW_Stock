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

# ============================================================
# 1. 系統初始化
# ============================================================

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板", layout="wide")

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False
if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = 0


# ============================================================
# 2. 資料來源：改用台灣本地 API（不會被限速）
# ============================================================

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

@st.cache_data(ttl=86400)
def get_stock_info_map():
    """
    從台灣證交所（上市）與櫃買中心（上櫃）爬取股票代碼、名稱、產業別。
    """
    stock_info_map = {}
    stocks_list = []
    urls = [
        ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
        ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")
    ]
    for url, market in urls:
        try:
            response = requests.get(url, headers=HEADERS, timeout=15, verify=False)
            df_list = pd.read_html(StringIO(response.text), flavor='lxml')
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


def fetch_twse_history(code: str, days: int = 160) -> pd.DataFrame:
    """
    從台灣證交所官方 API 抓取上市股票近期日線資料。
    API 每次回傳單月資料，需循環抓取所需月數。

    TWSE API: https://www.twse.com.tw/exchangeReport/STOCK_DAY
    參數: stockNo=股票代碼, date=查詢月份(YYYYMMDD)
    """
    months_needed = (days // 20) + 2  # 每月約 20 個交易日，多抓 2 個月緩衝
    all_rows = []

    for i in range(months_needed - 1, -1, -1):
        target = datetime.now().replace(day=1) - timedelta(days=i * 30)
        date_str = target.strftime('%Y%m01')
        url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&stockNo={code}&date={date_str}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            data = resp.json()
            if data.get('stat') != 'OK' or 'data' not in data:
                continue
            for row in data['data']:
                # TWSE 回傳欄位：日期,成交股數,成交金額,開盤價,最高價,最低價,收盤價,漲跌價差,成交筆數
                try:
                    # 民國年轉西元年（例：113/04/01 → 2024-04-01）
                    parts = row[0].split('/')
                    year = int(parts[0]) + 1911
                    date = f"{year}-{parts[1]}-{parts[2]}"
                    all_rows.append({
                        'Date':   date,
                        'Volume': float(row[1].replace(',', '')),
                        'Open':   float(row[3].replace(',', '')),
                        'High':   float(row[4].replace(',', '')),
                        'Low':    float(row[5].replace(',', '')),
                        'Close':  float(row[6].replace(',', '')),
                    })
                except Exception:
                    continue
            time.sleep(0.3)  # 避免對 TWSE API 請求過快
        except Exception:
            continue

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').drop_duplicates('Date').set_index('Date')
    return df.tail(days)


def fetch_tpex_history(code: str, days: int = 160) -> pd.DataFrame:
    """
    從證券櫃檯買賣中心 API 抓取上櫃股票近期日線資料。
    API 每次回傳單月資料，格式與 TWSE 類似。

    TPEX API: https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php
    """
    months_needed = (days // 20) + 2
    all_rows = []

    for i in range(months_needed - 1, -1, -1):
        target = datetime.now().replace(day=1) - timedelta(days=i * 30)
        # TPEX 使用民國年
        year_roc = target.year - 1911
        date_str = f"{year_roc}/{target.month:02d}"
        url = (
            f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/"
            f"st43_result.php?l=zh-tw&d={date_str}&stkno={code}&s=0,asc"
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            data = resp.json()
            if not data.get('iTotalRecords', 0):
                continue
            for row in data.get('aaData', []):
                try:
                    # TPEX 回傳欄位：日期,成交股數,成交金額,開盤,最高,最低,收盤,...
                    parts = row[0].split('/')
                    year = int(parts[0]) + 1911
                    date = f"{year}-{parts[1]}-{parts[2]}"
                    all_rows.append({
                        'Date':   date,
                        'Volume': float(str(row[1]).replace(',', '')),
                        'Open':   float(str(row[2]).replace(',', '')),
                        'High':   float(str(row[3]).replace(',', '')),
                        'Low':    float(str(row[4]).replace(',', '')),
                        'Close':  float(str(row[5]).replace(',', '')),
                    })
                except Exception:
                    continue
            time.sleep(0.3)
        except Exception:
            continue

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').drop_duplicates('Date').set_index('Date')
    return df.tail(days)


def fetch_history(code: str, market: str, days: int = 160) -> pd.DataFrame:
    """根據市場別選擇對應的 API 抓取歷史資料。"""
    if market == "TW":
        return fetch_twse_history(code, days)
    else:
        return fetch_tpex_history(code, days)


# ============================================================
# 3. 單支下載 + 分析
# ============================================================

def fetch_and_analyze(code: str, market: str, strategy: str, min_vol: float):
    """
    抓取單支股票歷史資料並執行策略分析。
    資料來源為 TWSE / TPEX 官方 API，不依賴 Yahoo Finance，無限速問題。
    """
    try:
        df = fetch_history(code, market, days=160)

        if df.empty or len(df) < 60:
            return None
        if (df['Volume'].tail(5).mean() / 1000) < min_vol:
            return None

        close = float(df['Close'].iloc[-1])
        m30   = float(df['Close'].rolling(30).mean().iloc[-1])
        m45   = float(df['Close'].rolling(45).mean().iloc[-1])
        m60   = float(df['Close'].rolling(60).mean().iloc[-1])

        if any(np.isnan(v) for v in [close, m30, m45, m60]):
            return None

        bias_30 = ((close - m30) / m30) * 100

        keep = False
        if strategy == "均線多頭回測":
            if m30 > m45 > m60 and close > m30 and bias_30 < 2.0:
                keep = True
        elif strategy == "均線糾結偵測":
            spread = (max(m30, m45, m60) - min(m30, m45, m60)) / min(m30, m45, m60)
            if spread <= 0.02 and abs(bias_30) < 2.0:
                keep = True

        if not keep:
            return None

        return {"code": code, "market": market, "close": close, "bias_30": bias_30}

    except Exception:
        return None


# ============================================================
# 4. 掃描主流程
# ============================================================

def run_scan(all_stocks, info_map, strategy, min_vol, progress_bar, status_text):
    """
    TWSE/TPEX API 有每秒請求限制，因此 max_workers 設為 3。
    每支下載後有 0.3 秒延遲（在 fetch 函數內），避免被擋。
    """
    MAX_WORKERS = 3  # TWSE/TPEX API 限速，不能開太多
    total = len(all_stocks)
    completed = 0
    results = []

    status_text.text(f"🔍 掃描 {total} 支股票中（台灣本地 API，無限速問題）...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_and_analyze, code, market, strategy, min_vol): code
            for code, market in all_stocks
        }
        for future in as_completed(futures):
            completed += 1
            if completed % 30 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"🔍 掃描進度：{completed} / {total}")
            try:
                result = future.result()
                if result:
                    code = result["code"]
                    info = info_map.get(code, {"name": "未知", "industry": "其他"})
                    suffix = ".TW" if result["market"] == "TW" else ".TWO"
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
# 5. 側邊欄 UI
# ============================================================

st.sidebar.header("⚙️ 策略參數設定")
strategy_option  = st.sidebar.radio("選擇選股策略：", ("均線多頭回測", "均線糾結偵測"))
indicator_choice = st.sidebar.selectbox("查看確認指標：", ["都不顯示", "RSI (強弱指標)", "MACD (趨勢指標)"])
min_volume       = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)
st.sidebar.markdown("---")
st.sidebar.caption(
    "📡 **資料來源**\n\n"
    "台灣證交所 (TWSE) 官方 API\n\n"
    "證券櫃檯買賣中心 (TPEX) 官方 API\n\n"
    "✅ 不依賴 Yahoo Finance，無限速問題。"
)


# ============================================================
# 6. 掃描按鈕
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
# 7. 結果顯示
# ============================================================

if not st.session_state['scan_results'].empty:
    df_raw = st.session_state['scan_results']

    selected_industry = st.selectbox("🎯 篩選類股：", ["全部"] + sorted(df_raw["類股"].unique().tolist()))
    df_filtered = df_raw if selected_industry == "全部" else df_raw[df_raw["類股"] == selected_industry]
    df_filtered = df_filtered.reset_index(drop=True)

    # 顯示欄位（隱藏內部用的 _code / _market）
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
                unsafe_allow_html=True
            )
        with c3:
            if st.button("下一支 ➡️", use_container_width=True):
                st.session_state['selected_index'] = (st.session_state['selected_index'] + 1) % len(df_filtered)

        idx = min(st.session_state['selected_index'], len(df_filtered) - 1)
        sel = df_filtered.iloc[idx]

        # ── K 線圖：使用 TWSE/TPEX API ──
        with st.spinner(f'載入中... {sel["名稱"]} ({sel["代碼"]})'):

            df_p = fetch_history(sel["_code"], sel["_market"], days=240)

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
                    x=df_p['Date_Str'],
                    open=df_p['Open'], high=df_p['High'],
                    low=df_p['Low'],   close=df_p['Close'], name="K線"
                ), row=1, col=1)

                for ma, color in zip(['30MA','45MA','60MA'], ['#FFA500','#2E8B57','#4169E1']):
                    fig.add_trace(go.Scatter(
                        x=df_p['Date_Str'], y=df_p[ma],
                        line=dict(color=color, width=1.5), name=ma
                    ), row=1, col=1)

                v_clrs = ['#ef5350' if c >= o else '#26a69a'
                          for c, o in zip(df_p['Close'], df_p['Open'])]
                fig.add_trace(go.Bar(
                    x=df_p['Date_Str'], y=df_p['Volume'],
                    marker_color=v_clrs, name="成交量"
                ), row=2, col=1)

                if indicator_choice == "RSI (強弱指標)":
                    d   = close_s.diff()
                    rsi = 100 - (100 / (
                        1 + d.where(d>0, 0).rolling(14).mean() /
                           (-d.where(d<0, 0).rolling(14).mean() + 1e-9)
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_p['Date_Str'], y=rsi,
                        line=dict(color='purple', width=1.2), name="RSI"
                    ), row=3, col=1)
                    fig.add_hline(y=70, line_dash="dot", line_color="red",   row=3, col=1)
                    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

                elif indicator_choice == "MACD (趨勢指標)":
                    dif = close_s.ewm(span=12).mean() - close_s.ewm(span=26).mean()
                    h_s = dif - dif.ewm(span=9).mean()
                    fig.add_trace(go.Bar(
                        x=df_p['Date_Str'], y=h_s,
                        marker_color=['#ef5350' if v>=0 else '#26a69a' for v in h_s],
                        name="MACD柱"
                    ), row=3, col=1)

                fig.update_xaxes(type='category', showgrid=True, gridcolor='rgba(200,200,200,0.2)')
                fig.update_layout(
                    title=f"<b>{sel['名稱']} ({sel['代碼']})</b>",
                    xaxis_rangeslider_visible=False, height=700, template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("目前篩選條件下無標的。")
else:
    st.info("💡 點擊「開始全市場掃描」按鈕開始。資料來源為台灣本地官方 API，無 Yahoo Finance 限速問題。")
