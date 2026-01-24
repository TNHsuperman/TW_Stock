import sys
import warnings
import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import urllib3
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 系統初始化與安全設定 ---
if 'warnings' not in sys.modules:
    sys.modules['warnings'] = warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="台股快速選股", layout="centered")

# --- 2. 資料抓取與緩存 ---
@st.cache_data(ttl=86400)
def get_stock_info_map():
    stock_info_map = {}
    stocks_list = []
    headers = {'User-Agent': 'Mozilla/5.0'}
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
            for index, row in df.iterrows():
                val = row['有價證券代號及名稱']
                industry = row['產業別']
                if val and '　' in str(val):
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        ticker = f"{code}{suffix}"
                        stocks_list.append(ticker)
                        stock_info_map[ticker] = {"name": name, "industry": industry if pd.notna(industry) else "其他"}
        except:
            continue
    return stocks_list, stock_info_map

# --- 3. UI 策略與指標設定 ---
st.sidebar.header("⚙️ 策略參數")
strategy_option = st.sidebar.radio("選擇選股策略：", ("均線多頭回測", "均線糾結偵測"))
indicator_choice = st.sidebar.selectbox("查看確認指標：", ["都不顯示", "RSI (強弱指標)", "MACD (趨勢指標)"])

min_volume = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)
use_filter = st.sidebar.checkbox("僅顯示轉強標的 (RSI > 50 或 MACD 柱狀體 > 0)")

# --- 4. 核心掃描邏輯 ---
if st.button(f"🔍 開始全市場掃描", use_container_width=True):
    all_stocks, info_map = get_stock_info_map()
    results = []
    progress_bar = st.progress(0)
    
    batch_size = 100
    for i in range(0, len(all_stocks), batch_size):
        batch = all_stocks[i:i + batch_size]
        try:
            data = yf.download(batch, period="150d", group_by='ticker', progress=False)
            for ticker in batch:
                try:
                    df = data[ticker] if len(batch) > 1 else data
                    df = df.dropna(subset=['Close'])
                    if len(df) < 60: continue
                    
                    avg_vol = df['Volume'].tail(5).mean() / 1000
                    if avg_vol < min_volume: continue

                    close = float(df['Close'].iloc[-1])
                    m30 = df['Close'].rolling(30).mean().iloc[-1]
                    m45 = df['Close'].rolling(45).mean().iloc[-1]
                    m60 = df['Close'].rolling(60).mean().iloc[-1]
                    
                    # 計算 RSI
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rsi = (100 - (100 / (1 + gain/loss))).iloc[-1]

                    # 計算 MACD
                    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                    macd_val = exp1 - exp2
                    signal_val = macd_val.ewm(span=9, adjust=False).mean()
                    hist_val = (macd_val - signal_val).iloc[-1]
                    
                    keep = False
                    if strategy_option == "均線多頭回測":
                        if m30 > m45 > m60 and close > m30 and (close - m30) / m30 <= 0.02:
                            keep = True
                    elif strategy_option == "均線糾結偵測":
                        ma_spread = (max(m30, m45, m60) - min(m30, m45, m60)) / min(m30, m45, m60)
                        if ma_spread <= 0.015 and abs(close - m30) / m30 <= 0.02:
                            keep = True
                    
                    if keep and use_filter:
                        if indicator_choice == "RSI (強弱指標)" and rsi < 50: keep = False
                        if indicator_choice == "MACD (趨勢指標)" and hist_val < 0: keep = False

                    if keep:
                        stock_data = info_map.get(ticker, {"name": "未知", "industry": "其他"})
                        results.append({
                            "ID": ticker, "代碼": ticker.split('.')[0], "名稱": stock_data["name"],
                            "類股": stock_data["industry"], "收盤": round(close, 2),
                            "RSI": round(rsi, 1), "MACD柱": round(hist_val, 2)
                        })
                except: continue
        except: continue
        progress_bar.progress(min((i + batch_size) / len(all_stocks), 1.0))
    
    st.session_state['scan_results'] = pd.DataFrame(results)
    st.session_state['selected_index'] = 0  # 重設切換索引

# --- 5. 顯示與切換邏輯 ---
if 'scan_results' in st.session_state and not st.session_state['scan_results'].empty:
    df_raw = st.session_state['scan_results']
    selected_industry = st.selectbox("🎯 篩選類股：", ["全部"] + sorted(df_raw["類股"].unique().tolist()))
    df_filtered = df_raw if selected_industry == "全部" else df_raw[df_raw["類股"] == selected_industry]
    df_filtered = df_filtered.reset_index(drop=True)

    # 左右切換按鈕區域
    st.write("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    if 'selected_index' not in st.session_state:
        st.session_state['selected_index'] = 0

    with col1:
        if st.button("⬅️ 上一支", use_container_width=True):
            st.session_state['selected_index'] = (st.session_state['selected_index'] - 1) % len(df_filtered)
    with col2:
        st.write(f"<center><b>{st.session_state['selected_index'] + 1} / {len(df_filtered)}</b></center>", unsafe_allow_html=True)
    with col3:
        if st.button("下一支 ➡️", use_container_width=True):
            st.session_state['selected_index'] = (st.session_state['selected_index'] + 1) % len(df_filtered)

    # 繪圖邏輯
    row = df_filtered.iloc[st.session_state['selected_index']]
    ticker_id = row['ID']
    
    with st.spinner(f'載入 {row["名稱"]} ({ticker_id})...'):
        df_plot = yf.download(ticker_id, period="8mo", progress=False)
        if isinstance(df_plot.columns, pd.MultiIndex): df_plot.columns = df_plot.columns.get_level_values(0)
        
        df_plot['30MA'] = df_plot['Close'].rolling(30).mean()
        df_plot['45MA'] = df_plot['Close'].rolling(45).mean()
        df_plot['60MA'] = df_plot['Close'].rolling(60).mean()
        
        # 設定副圖數量
        num_rows = 3 if indicator_choice != "都不顯示" else 2
        row_heights = [0.5, 0.2, 0.3] if num_rows == 3 else [0.7, 0.3]
        
        fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)
        
        # Row 1: K線
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="K線"), row=1, col=1)
        for ma, color in zip(['30MA', '45MA', '60MA'], ['#FFA500', '#2E8B57', '#4169E1']):
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[ma], line=dict(color=color, width=1.2), name=ma), row=1, col=1)
        
        # Row 2: 成交量
        v_colors = ['red' if c >= o else 'green' for c, o in zip(df_plot['Close'], df_plot['Open'])]
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=v_colors, name="成交量"), row=2, col=1)
        
        # Row 3: 條件指標
        if indicator_choice == "RSI (強弱指標)":
            delta = df_plot['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi_series = 100 - (100 / (1 + gain/loss))
            fig.add_trace(go.Scatter(x=df_plot.index, y=rsi_series, line=dict(color='purple'), name="RSI(14)"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
        elif indicator_choice == "MACD (趨勢指標)":
            exp1 = df_plot['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df_plot['Close'].ewm(span=26, adjust=False).mean()
            macd_s = exp1 - exp2
            signal_s = macd_s.ewm(span=9, adjust=False).mean()
            hist_s = macd_s - signal_s
            h_colors = ['red' if v >= 0 else 'green' for v in hist_s]
            fig.add_trace(go.Bar(x=df_plot.index, y=hist_s, marker_color=h_colors, name="MACD柱狀體"), row=3, col=1)

        # 核心：移除未開盤時間 (週六、週日與非交易時段)
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]), # 移除週末
                # 若需要移除特定連續假期，可在這裡加入 dict(values=["2026-02-14", ...])
            ]
        )

        fig.update_layout(title=f"<b>{row['名稱']} ({ticker_id})</b>", xaxis_rangeslider_visible=False, height=600, template="plotly_white", dragmode='pan', margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # 列表放在下方作為輔助選擇
    st.write("---")
    st.write("📊 篩選清單總覽")
    st.dataframe(df_filtered, hide_index=True, use_container_width=True)

elif 'scan_results' in st.session_state:
    st.warning("查無標的，請試著放寬過濾條件。")
