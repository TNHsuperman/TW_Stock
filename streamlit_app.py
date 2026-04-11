import streamlit as st
import pandas as pd
import requests
import numpy as np
from io import StringIO
import urllib3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 1. 系統初始化
# ============================================================

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="台股智慧選股儀表板（多執行緒版）", layout="wide")

# 初始化 Session State
if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False
if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = 0


# ============================================================
# 2. 股票清單抓取（同步，每日快取）
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_info_map():
    """
    從台灣證交所（上市）與櫃買中心（上櫃）爬取股票代碼、名稱、產業別。
    回傳：
        stocks_list   : Yahoo Finance ticker 列表（如 '2330.TW'）
        stock_info_map: dict，key 為 ticker，value 為 {"name":..., "industry":...}
    """
    stock_info_map = {}
    stocks_list = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    urls = [
        ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', ".TW"),
        ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', ".TWO")
    ]
    for url, suffix in urls:
        try:
            response = requests.get(url, headers=headers, timeout=15, verify=False)
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
                        ticker = f"{code}{suffix}"
                        stocks_list.append(ticker)
                        stock_info_map[ticker] = {
                            "name": name,
                            "industry": industry if pd.notna(industry) else "其他"
                        }
        except Exception:
            continue
    return stocks_list, stock_info_map


# ============================================================
# 3. 多執行緒下載核心
# ============================================================

# Yahoo Finance CSV 下載網址模板（直接呼叫 API，不經 yfinance 封裝）
YF_URL = (
    "https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
    "?period1={p1}&period2={p2}&interval=1d&events=history"
)

# 共用的請求標頭，模擬瀏覽器避免被 Yahoo 封鎖
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

def fetch_one(ticker: str, p1: int, p2: int) -> tuple:
    """
    下載單一股票的歷史日線 CSV 並解析為 DataFrame。
    此函數會在獨立的執行緒中執行。

    參數：
        ticker : Yahoo Finance 股票代碼（如 '2330.TW'）
        p1/p2  : 資料起迄的 Unix timestamp

    回傳：
        (ticker, DataFrame) 或 (ticker, None)（下載或解析失敗時）
    """
    url = YF_URL.format(ticker=ticker, p1=p1, p2=p2)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10, verify=False)
        if resp.status_code != 200:
            # HTTP 非 200（如 404 找不到、429 被限速）視為失敗
            return ticker, None

        df = pd.read_csv(StringIO(resp.text), parse_dates=["Date"], index_col="Date")
        df = df.apply(pd.to_numeric, errors='coerce')  # 非數值轉 NaN
        df = df.dropna(subset=["Close"])
        return ticker, df

    except Exception:
        return ticker, None


def fetch_all_threaded(
    tickers: list,
    days: int = 160,
    max_workers: int = 60
) -> dict:
    """
    使用 ThreadPoolExecutor 並行下載所有股票的歷史資料。

    為什麼用多執行緒而非 asyncio？
    → Streamlit 本身有自己的 event loop，直接呼叫 asyncio.run() 會造成
      「event loop is already running」衝突導致程式卡死。
    → ThreadPoolExecutor 完全在同步環境運作，不會與 Streamlit 衝突，
      對 I/O 密集型任務（網路下載）效果與 asyncio 相近。

    參數：
        tickers     : 股票代碼列表
        days        : 下載天數（含假日）
        max_workers : 最大同時執行緒數（預設 60）

    回傳：
        dict，key 為 ticker，value 為 DataFrame（失敗的不包含）
    """
    p2 = int(datetime.now().timestamp())
    p1 = int((datetime.now() - timedelta(days=days)).timestamp())

    results = {}

    # ThreadPoolExecutor 自動管理執行緒池，離開 with 區塊後自動清理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有下載任務，取得 future → ticker 的對應關係
        future_to_ticker = {
            executor.submit(fetch_one, t, p1, p2): t
            for t in tickers
        }
        # as_completed 會在每個 future 完成時立即回傳（不等全部完成）
        for future in as_completed(future_to_ticker):
            try:
                ticker, df = future.result()
                if df is not None and not df.empty:
                    results[ticker] = df
            except Exception:
                continue  # 單一執行緒失敗不影響其他

    return results


# ============================================================
# 4. 技術指標計算與選股邏輯
# ============================================================

def analyze_ticker(ticker: str, df: pd.DataFrame, strategy: str, min_vol: float) -> dict | None:
    """
    對單一股票執行技術指標計算與策略篩選。

    回傳符合條件的結果 dict，或 None（不符合）。
    """
    try:
        if len(df) < 60:
            return None

        # 近 5 日平均成交量（張）低於門檻則跳過
        if (df['Volume'].tail(5).mean() / 1000) < min_vol:
            return None

        close = float(df['Close'].iloc[-1])
        m30 = float(df['Close'].rolling(30).mean().iloc[-1])
        m45 = float(df['Close'].rolling(45).mean().iloc[-1])
        m60 = float(df['Close'].rolling(60).mean().iloc[-1])

        # 任一均線為 NaN 則跳過（資料頭部不足計算）
        if any(np.isnan(v) for v in [m30, m45, m60]):
            return None

        # 對 30MA 乖離率（%）
        bias_30 = ((close - m30) / m30) * 100

        keep = False
        if strategy == "均線多頭回測":
            # 多頭排列，且股價剛回測貼近 30MA（乖離 < 2%）
            if m30 > m45 > m60 and close > m30 and bias_30 < 2.0:
                keep = True
        elif strategy == "均線糾結偵測":
            # 三條均線極度靠近（壓縮蓄勢），乖離 < 2%
            ma_spread = (max(m30, m45, m60) - min(m30, m45, m60)) / min(m30, m45, m60)
            if ma_spread <= 0.02 and abs(bias_30) < 2.0:
                keep = True

        if not keep:
            return None

        return {"close": close, "bias_30": bias_30}

    except Exception:
        return None


# ============================================================
# 5. 掃描主流程
# ============================================================

def run_scan(all_stocks, info_map, strategy, min_vol, progress_bar, status_text):
    """
    執行完整掃描流程：
      1. 多執行緒並行下載所有股票歷史資料
      2. 逐支股票執行技術分析與篩選
      3. 回傳符合條件的 DataFrame
    """
    total = len(all_stocks)
    status_text.text(f"⬇️  多執行緒下載 {total} 支股票資料中（最多 60 條同時連線）...")

    # ── 步驟 1：並行下載 ──
    all_data = fetch_all_threaded(all_stocks, days=160, max_workers=60)
    downloaded = len(all_data)

    status_text.text(f"✅  下載完成（成功 {downloaded} / {total} 支），開始分析...")
    progress_bar.progress(0.5)  # 下載完成視為 50% 進度

    # ── 步驟 2：逐支分析 ──
    results = []
    for i, (ticker, df) in enumerate(all_data.items()):
        result = analyze_ticker(ticker, df, strategy, min_vol)
        if result:
            stock_data = info_map.get(ticker, {"name": "未知", "industry": "其他"})
            results.append({
                "ID":      ticker,
                "代碼":    ticker.split('.')[0],
                "名稱":    stock_data["name"],
                "類股":    stock_data["industry"],
                "收盤":    round(result["close"], 2),
                "乖離(%)": round(result["bias_30"], 2),
            })
        # 分析進度從 50% 跑到 100%
        progress_bar.progress(0.5 + 0.5 * (i + 1) / max(downloaded, 1))

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by="乖離(%)", ascending=True).reset_index(drop=True)

    return res_df


# ============================================================
# 6. 側邊欄 UI
# ============================================================

st.sidebar.header("⚙️ 策略參數設定")

strategy_option = st.sidebar.radio(
    "選擇選股策略：",
    ("均線多頭回測", "均線糾結偵測")
)

indicator_choice = st.sidebar.selectbox(
    "查看確認指標：",
    ["都不顯示", "RSI (強弱指標)", "MACD (趨勢指標)"]
)

min_volume = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚡ **多執行緒版**\n\n"
    "使用 `ThreadPoolExecutor` 並行下載，\n"
    "同時最多 60 條連線，\n"
    "避免 Streamlit 的 asyncio 衝突問題。\n"
    "全市場下載時間約 **20~40 秒**。"
)


# ============================================================
# 7. 掃描按鈕與觸發邏輯
# ============================================================

if not st.session_state['is_scanning']:
    if st.button("🔍 開始全市場掃描", use_container_width=True):
        st.session_state['is_scanning'] = True
        st.rerun()

else:
    all_stocks, info_map = get_stock_info_map()
    progress_bar = st.progress(0)
    status_text = st.empty()

    res_df = run_scan(
        all_stocks, info_map,
        strategy_option, min_volume,
        progress_bar, status_text
    )

    status_text.text(f"🎉 掃描完成！共找到 {len(res_df)} 支符合條件標的。")
    st.session_state['scan_results'] = res_df
    st.session_state['is_scanning'] = False
    st.rerun()


# ============================================================
# 8. 結果顯示：清單篩選、股票切換、K 線圖繪製
# ============================================================

if not st.session_state['scan_results'].empty:
    df_raw = st.session_state['scan_results']

    selected_industry = st.selectbox(
        "🎯 篩選類股：",
        ["全部"] + sorted(df_raw["類股"].unique().tolist())
    )
    df_filtered = df_raw if selected_industry == "全部" else df_raw[df_raw["類股"] == selected_industry]
    df_filtered = df_filtered.reset_index(drop=True)

    if len(df_filtered) > 0:
        st.write(f"📊 符合條件標的：{len(df_filtered)} 支")

        event = st.dataframe(
            df_filtered,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="stock_table"
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
                f"<h3 style='text-align: center; color: #9400D3;'>"
                f"{st.session_state['selected_index'] + 1} / {len(df_filtered)}</h3>",
                unsafe_allow_html=True
            )
        with c3:
            if st.button("下一支 ➡️", use_container_width=True):
                st.session_state['selected_index'] = (st.session_state['selected_index'] + 1) % len(df_filtered)

        idx = min(st.session_state['selected_index'], len(df_filtered) - 1)
        row = df_filtered.iloc[idx]

        # ── K 線圖：單支股票下載 ──
        with st.spinner(f'載入中... {row["名稱"]} ({row["ID"]})'):

            # 單支股票複用 fetch_all_threaded（傳入只有一支的列表）
            chart_data = fetch_all_threaded([row['ID']], days=240, max_workers=1)
            df_p = chart_data.get(row['ID'])

            if df_p is None or df_p.empty:
                st.error("無法載入該股票資料，請稍後再試。")
            else:
                df_p = df_p.dropna(subset=['Close'])
                df_p['Date_Str'] = df_p.index.strftime('%Y-%m-%d')
                df_p['30MA'] = df_p['Close'].rolling(30).mean()
                df_p['45MA'] = df_p['Close'].rolling(45).mean()
                df_p['60MA'] = df_p['Close'].rolling(60).mean()

                n_rows = 3 if indicator_choice != "都不顯示" else 2
                row_heights = [0.6, 0.15, 0.25] if n_rows == 3 else [0.75, 0.25]

                fig = make_subplots(
                    rows=n_rows, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=row_heights
                )

                # K 線
                fig.add_trace(go.Candlestick(
                    x=df_p['Date_Str'],
                    open=df_p['Open'], high=df_p['High'],
                    low=df_p['Low'],   close=df_p['Close'],
                    name="K線"
                ), row=1, col=1)

                # 均線
                for ma, color in zip(['30MA', '45MA', '60MA'], ['#FFA500', '#2E8B57', '#4169E1']):
                    fig.add_trace(go.Scatter(
                        x=df_p['Date_Str'], y=df_p[ma],
                        line=dict(color=color, width=1.5), name=ma
                    ), row=1, col=1)

                # 成交量
                v_clrs = ['#ef5350' if c >= o else '#26a69a'
                          for c, o in zip(df_p['Close'], df_p['Open'])]
                fig.add_trace(go.Bar(
                    x=df_p['Date_Str'], y=df_p['Volume'],
                    marker_color=v_clrs, name="成交量"
                ), row=2, col=1)

                # RSI
                if indicator_choice == "RSI (強弱指標)":
                    d = df_p['Close'].diff()
                    rsi = 100 - (100 / (
                        1 + (d.where(d > 0, 0)).rolling(14).mean() /
                        ((-d.where(d < 0, 0)).rolling(14).mean() + 1e-9)
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_p['Date_Str'], y=rsi,
                        line=dict(color='purple', width=1.2), name="RSI"
                    ), row=3, col=1)
                    fig.add_hline(y=70, line_dash="dot", line_color="red",   row=3, col=1)
                    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

                # MACD
                elif indicator_choice == "MACD (趨勢指標)":
                    m_s = df_p['Close'].ewm(span=12).mean() - df_p['Close'].ewm(span=26).mean()
                    h_s = m_s - m_s.ewm(span=9).mean()
                    fig.add_trace(go.Bar(
                        x=df_p['Date_Str'], y=h_s,
                        marker_color=['#ef5350' if v >= 0 else '#26a69a' for v in h_s],
                        name="MACD柱"
                    ), row=3, col=1)

                fig.update_xaxes(
                    type='category',
                    showgrid=True,
                    gridcolor='rgba(200,200,200,0.2)'
                )
                fig.update_layout(
                    title=f"<b>{row['名稱']} ({row['ID']})</b>",
                    xaxis_rangeslider_visible=False,
                    height=700,
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("目前篩選條件下無標的。")

else:
    st.info("💡 提示：點擊「開始全市場掃描」按鈕。多執行緒版下載速度約為原版的 3～5 倍。")
