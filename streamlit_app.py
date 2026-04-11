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

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


# ============================================================
# 2. 股票清單抓取
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


# ============================================================
# 3. 快速預篩：一次 API 拿全市場當日收盤與成交量
# ============================================================

@st.cache_data(ttl=3600)  # 快取 1 小時（盤中資料會變動）
def get_market_snapshot() -> pd.DataFrame:
    """
    速度優化關鍵：用單一 API 請求取得全市場最新收盤價與成交量。

    舊做法：每支股票各自抓 → N 次請求
    新做法：一次拿全市場快照 → 1~2 次請求，立刻過濾掉成交量不足的股票
           剩下需要計算均線的股票數量大幅減少（通常從 ~1800 支降至 ~400 支）

    資料來源：
      上市：TWSE 每日收盤行情 API（MI_INDEX）
      上櫃：TPEX 每日行情 API
    """
    rows = {}

    # ── 上市（TWSE）──
    try:
        today = datetime.now().strftime('%Y%m%d')
        url = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={today}&type=ALLBUT0999"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        data = resp.json()
        # MI_INDEX 回傳多個表，找含有股票代碼的那張（欄位數 >= 16）
        for table_key in data:
            if not isinstance(data[table_key], list):
                continue
            for row in data[table_key]:
                if isinstance(row, list) and len(row) >= 16:
                    try:
                        code  = str(row[0]).strip()
                        vol   = float(str(row[2]).replace(',', ''))   # 成交股數
                        close = float(str(row[8]).replace(',', ''))   # 收盤價
                        if len(code) == 4 and code.isdigit():
                            rows[code] = {"close": close, "volume_k": vol / 1000, "market": "TW"}
                    except Exception:
                        continue
    except Exception:
        pass

    # ── 上櫃（TPEX）──
    try:
        yr = datetime.now().year - 1911
        mo = datetime.now().month
        url = f"https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php?l=zh-tw&d={yr}/{mo:02d}&se=AL"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        data = resp.json()
        for row in data.get('aaData', []):
            try:
                code  = str(row[0]).strip()
                close = float(str(row[2]).replace(',', ''))
                vol   = float(str(row[8]).replace(',', ''))  # 成交張數
                if len(code) == 4 and code.isdigit():
                    rows[code] = {"close": close, "volume_k": vol, "market": "TWO"}
            except Exception:
                continue
    except Exception:
        pass

    return pd.DataFrame(rows).T  # index = 股票代碼


# ============================================================
# 4. 歷史資料抓取（只對通過預篩的股票執行）
# ============================================================

def fetch_history_months(code: str, market: str, months: int = 4) -> pd.DataFrame:
    """
    抓取指定月數的歷史日線資料。
    掃描用 months=4（約 80 交易日，足夠計算 60MA）。
    K 線圖用 months=10（約 200 交易日）。

    相較舊版改進：
      - months 從 8 降至 4（掃描時），請求次數減半
      - sleep 從 0.3s 降至 0.05s（TWSE/TPEX 實測可承受）
    """
    all_rows = []

    for i in range(months - 1, -1, -1):
        target   = datetime.now().replace(day=1) - timedelta(days=i * 30)
        date_str = target.strftime('%Y%m01')

        if market == "TW":
            url = (f"https://www.twse.com.tw/exchangeReport/STOCK_DAY"
                   f"?response=json&stockNo={code}&date={date_str}")
            try:
                resp = requests.get(url, headers=HEADERS, timeout=10)
                data = resp.json()
                if data.get('stat') != 'OK':
                    continue
                for row in data.get('data', []):
                    try:
                        parts = row[0].split('/')
                        date  = f"{int(parts[0])+1911}-{parts[1]}-{parts[2]}"
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
            except Exception:
                continue

        else:  # TWO
            yr_roc   = target.year - 1911
            date_str = f"{yr_roc}/{target.month:02d}"
            url = (f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/"
                   f"st43_result.php?l=zh-tw&d={date_str}&stkno={code}&s=0,asc")
            try:
                resp = requests.get(url, headers=HEADERS, timeout=10)
                data = resp.json()
                for row in data.get('aaData', []):
                    try:
                        parts = row[0].split('/')
                        date  = f"{int(parts[0])+1911}-{parts[1]}-{parts[2]}"
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
            except Exception:
                continue

        time.sleep(0.05)  # 降低至 0.05s，TWSE/TPEX 實測可承受

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').drop_duplicates('Date').set_index('Date')


# ============================================================
# 5. 分析邏輯
# ============================================================

def analyze(df: pd.DataFrame, strategy: str) -> dict | None:
    """對已下載的 DataFrame 執行均線計算與策略判斷。"""
    if len(df) < 60:
        return None

    close   = float(df['Close'].iloc[-1])
    m30     = float(df['Close'].rolling(30).mean().iloc[-1])
    m45     = float(df['Close'].rolling(45).mean().iloc[-1])
    m60     = float(df['Close'].rolling(60).mean().iloc[-1])

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
    """下載 4 個月歷史資料並執行分析。"""
    try:
        df = fetch_history_months(code, market, months=4)
        if df.empty:
            return None
        result = analyze(df, strategy)
        if result:
            result["code"]   = code
            result["market"] = market
        return result
    except Exception:
        return None


# ============================================================
# 6. 掃描主流程（兩階段：快照預篩 → 歷史資料精算）
# ============================================================

def run_scan(all_stocks, info_map, strategy, min_vol, progress_bar, status_text):
    """
    兩階段掃描，大幅縮短總時間：

    【第一階段】快照預篩（幾乎瞬間完成）
      - 用單一 API 拿全市場當日收盤與成交量
      - 過濾掉成交量不足的股票
      - 通常從 ~1800 支降至 ~300~500 支

    【第二階段】歷史資料精算（只處理通過預篩的股票）
      - 每支股票抓 4 個月日線（夠計算 60MA）
      - max_workers=8 並行，sleep=0.05s
      - 支數大幅減少，總時間從 30 分鐘降至約 5~8 分鐘
    """

    # ── 第一階段：快照預篩 ──
    status_text.text("⚡ 第一階段：取得全市場快照進行預篩...")
    progress_bar.progress(0.05)

    snapshot = get_market_snapshot()

    # 整理 all_stocks 為 dict 方便查找
    stock_dict = {code: market for code, market in all_stocks}

    # 過濾：快照中存在 + 成交量達標
    if not snapshot.empty and 'volume_k' in snapshot.columns:
        snapshot['volume_k'] = pd.to_numeric(snapshot['volume_k'], errors='coerce')
        passed = snapshot[snapshot['volume_k'] >= min_vol]
        candidates = [(code, stock_dict[code]) for code in passed.index if code in stock_dict]
    else:
        # 快照失敗（例如非交易日），回退到全部掃描
        candidates = all_stocks

    total = len(candidates)
    status_text.text(f"✅ 預篩完成：{len(all_stocks)} 支 → {total} 支候選（成交量 ≥ {min_vol} 張）")
    progress_bar.progress(0.1)

    # ── 第二階段：歷史資料精算 ──
    completed = 0
    results   = []
    MAX_WORKERS = 8

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_and_analyze, code, market, strategy): code
            for code, market in candidates
        }
        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0 or completed == total:
                pct = 0.1 + 0.9 * (completed / max(total, 1))
                progress_bar.progress(pct)
                status_text.text(f"🔍 精算進度：{completed} / {total}（已找到 {len(results)} 支）")
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
# 7. 側邊欄 UI
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
    "⚡ 兩階段掃描：快照預篩 → 歷史精算\n\n"
    "全市場約需 **5~8 分鐘**。"
)


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
    df_filtered = df_raw if selected_industry == "全部" else df_raw[df_raw["類股"] == selected_industry]
    df_filtered = df_filtered.reset_index(drop=True)

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

        with st.spinner(f'載入中... {sel["名稱"]} ({sel["代碼"]})'):
            df_p = fetch_history_months(sel["_code"], sel["_market"], months=10)

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
    st.info("💡 點擊「開始全市場掃描」按鈕。⚡ 兩階段掃描：快照預篩大幅縮短等待時間。")
