import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import urllib3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# 2. 股票清單抓取
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
# 3. 單支股票下載 + 分析（在執行緒中執行）
# ============================================================

def fetch_and_analyze(ticker: str, strategy: str, min_vol: float) -> dict | None:
    """
    下載單支股票資料並立即執行策略分析。
    此函數在獨立執行緒中執行，每支股票獨立處理。

    將下載與分析合併在同一函數的好處：
    - 下載完立刻分析，不需等所有股票都下載完才開始
    - 分析完丟棄原始資料，記憶體用量極低

    回傳符合條件的結果 dict，或 None（不符合 / 下載失敗）
    """
    try:
        # 用 yfinance 下載單支，穩定性高
        df = yf.download(ticker, period="150d", progress=False, auto_adjust=True)

        # 處理 MultiIndex 欄位（yfinance 某些版本會產生）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=['Close'])

        if len(df) < 60:
            return None

        # 近 5 日均量過低則跳過（Volume 為股，除 1000 換算為張）
        if (df['Volume'].tail(5).mean() / 1000) < min_vol:
            return None

        close = float(df['Close'].iloc[-1])
        m30 = float(df['Close'].rolling(30).mean().iloc[-1])
        m45 = float(df['Close'].rolling(45).mean().iloc[-1])
        m60 = float(df['Close'].rolling(60).mean().iloc[-1])

        if any(np.isnan(v) for v in [m30, m45, m60]):
            return None

        bias_30 = ((close - m30) / m30) * 100

        keep = False
        if strategy == "均線多頭回測":
            if m30 > m45 > m60 and close > m30 and bias_30 < 2.0:
                keep = True
        elif strategy == "均線糾結偵測":
            ma_spread = (max(m30, m45, m60) - min(m30, m45, m60)) / min(m30, m45, m60)
            if ma_spread <= 0.02 and abs(bias_30) < 2.0:
                keep = True

        if not keep:
            return None

        return {"ticker": ticker, "close": close, "bias_30": bias_30}

    except Exception:
        return None


# ============================================================
# 4. 掃描主流程
# ============================================================

def run_scan(all_stocks, info_map, strategy, min_vol, progress_bar, status_text):
    """
    用 ThreadPoolExecutor 並行掃描全市場。

    為什麼 max_workers=8？
    - Streamlit Cloud 是共享容器，CPU / 網路資源有限
    - yfinance 每次下載需建立 HTTPS 連線，太多並行會被 Yahoo 限速（429）
    - 8 條執行緒在 Cloud 環境最穩定，速度約為單執行緒的 5~6 倍
    - 本機執行可調高至 16~20
    """
    MAX_WORKERS = 8  # ← 本機可調高至 16，Cloud 建議維持 8
    total = len(all_stocks)
    completed = 0
    results = []

    status_text.text(f"🔍 開始掃描 {total} 支股票（{MAX_WORKERS} 條執行緒並行）...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(fetch_and_analyze, t, strategy, min_vol): t
            for t in all_stocks
        }

        # as_completed：每完成一支立即處理，不等全部跑完
        for future in as_completed(future_to_ticker):
            completed += 1

            # 每 50 支更新一次進度，避免過於頻繁的 UI 刷新拖慢速度
            if completed % 50 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"🔍 掃描進度：{completed} / {total}")

            try:
                result = future.result()
                if result:
                    ticker = result["ticker"]
                    stock_data = info_map.get(ticker, {"name": "未知", "industry": "其他"})
                    results.append({
                        "ID":      ticker,
                        "代碼":    ticker.split('.')[0],
                        "名稱":    stock_data["name"],
                        "類股":    stock_data["industry"],
                        "收盤":    round(result["close"], 2),
                        "乖離(%)": round(result["bias_30"], 2),
                    })
            except Exception:
                continue

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by="乖離(%)", ascending=True).reset_index(drop=True)

    return res_df


# ============================================================
# 5. 側邊欄 UI
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
    "使用 `yfinance` 單支下載 ×\n"
    "`ThreadPoolExecutor` 8 條並行，\n"
    "相容 Streamlit Cloud 環境。\n\n"
    "全市場約需 **3~5 分鐘**。"
)


# ============================================================
# 6. 掃描按鈕與觸發邏輯
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
# 7. 結果顯示：清單篩選、股票切換、K 線圖繪製
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

        # ── K 線圖 ──
        with st.spinner(f'載入中... {row["名稱"]} ({row["ID"]})'):
            df_p = yf.download(row['ID'], period="8mo", progress=False, auto_adjust=True)

            if isinstance(df_p.columns, pd.MultiIndex):
                df_p.columns = df_p.columns.get_level_values(0)

            df_p = df_p.dropna(subset=['Close'])

            if df_p.empty:
                st.error("無法載入該股票資料，請稍後再試。")
            else:
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

                fig.add_trace(go.Candlestick(
                    x=df_p['Date_Str'],
                    open=df_p['Open'], high=df_p['High'],
                    low=df_p['Low'],   close=df_p['Close'],
                    name="K線"
                ), row=1, col=1)

                for ma, color in zip(['30MA', '45MA', '60MA'], ['#FFA500', '#2E8B57', '#4169E1']):
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
    st.info("💡 提示：點擊「開始全市場掃描」按鈕。Streamlit Cloud 環境約需 3~5 分鐘完成掃描。")
