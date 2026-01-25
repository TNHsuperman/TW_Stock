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

# --- 1. 系統初始化 ---
if 'warnings' not in sys.modules:
    sys.modules['warnings'] = warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="台股智慧選股儀表板", layout="wide")

# 初始化 Session State
if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False
if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = 0

# --- 2. 資料抓取函數 ---
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
        except: continue
    return stocks_list, stock_info_map

# --- 3. UI 設定 ---
st.sidebar.header("⚙️ 策略參數設定")
strategy_option = st.sidebar.radio("選擇選股策略：", ("均線多頭回測", "均線糾結偵測"))
indicator_choice = st.sidebar.selectbox("查看確認指標：", ["都不顯示", "RSI (強弱指標)", "MACD (趨勢指標)"])
min_volume = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)
use_filter = st.sidebar.checkbox("僅顯示轉強標的 (RSI > 50 或 MACD > 0)", value=True)

# --- 4. 掃描邏輯 ---
btn_label = "🔍 開始全市場掃描" if not st.session_state['is_scanning'] else "⏳ 正在掃描中..."
if st.button(btn_label, use_container_width=True, disabled=st.session_state['is_scanning']):
    st.session_state['is_scanning'] = True
    st.rerun()

if st.session_state['is_scanning']:
    all_stocks, info_map = get_stock_info_map()
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batch_size = 80
    for i in range(0, len(all_stocks), batch_size):
        batch = all_stocks[i:i + batch_size]
        status_text.text(f"正在分析第 {i} 支股票...")
        try:
            data = yf.download(batch, period="150d", group_by='ticker', progress=False)
            for ticker in batch:
                try:
                    df = data[ticker].dropna(subset=['Close']) if len(batch) > 1 else data.dropna(subset=['Close'])
                    if len(df) < 60: continue
                    if (df['Volume'].tail(5).mean() / 1000) < min_volume: continue

                    # 指標計算
                    close = df['Close'].iloc[-1]
                    m30, m45, m60 = df['Close'].rolling(30).mean().iloc[-1], df['Close'].rolling(45).mean().iloc[-1], df['Close'].rolling(60).mean().iloc[-1]
                    
                    keep = False
                    if strategy_option == "均線多頭回測":
                        if m30 > m45 > m60 and close > m30 and (close - m30) / m30 <= 0.02: keep = True
                    elif strategy_option == "均線糾結偵測":
                        ma_spread = (max(m30, m45, m60) - min(m30, m45, m60)) / min(m30, m45, m60)
                        if ma_spread <= 0.015 and abs(close - m30) / m30 <= 0.02: keep = True
                    
                    if keep:
                        stock_data = info_map.get(ticker, {"name": "未知", "industry": "其他"})
                        results.append({"ID": ticker, "代碼": ticker.split('.')[0], "名稱": stock_data["name"], "類股": stock_data["industry"], "收盤": round(close, 2)})
                except: continue
        except: continue
        progress_bar.progress(min((i + batch_size) / len(all_stocks), 1.0))
    
    st.session_state['scan_results'] = pd.DataFrame(results)
    st.session_state['is_scanning'] = False
    st.rerun()

# --- 5. 結果顯示與繪圖 ---
if not st.session_state['scan_results'].empty:
    df_raw = st.session_state['scan_results']
    selected_industry = st.selectbox("🎯 篩選類股：", ["全部"] + sorted(df_raw["類股"].unique().tolist()))
    df_filtered = df_raw if selected_industry == "全部" else df_raw[df_raw["類股"] == selected_industry]
    df_filtered = df_filtered.reset_index(drop=True)

    st.write(f"📊 找到 {len(df_filtered)} 支符合條件的股票")
    
    # 選擇股票
    event = st.dataframe(df_filtered, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row", key="stock_table")
    
    # 更新選中索引
    if event.selection and event.selection.rows:
        st.session_state['selected_index'] = event.selection.rows[0]

    # 防呆
    idx = min(st.session_state['selected_index'], len(df_filtered)-1)
    row = df_filtered.iloc[idx]
    
    with st.spinner(f'載入 {row["名稱"]} ({row["ID"]})...'):
        df_p = yf.download(row['ID'], period="8mo", progress=False)
        if isinstance(df_p.columns, pd.MultiIndex): df_p.columns = df_p.columns.get_level_values(0)
        
        # 建立均線與買入訊號邏輯
        df_p['30MA'] = df_p['Close'].rolling(30).mean()
        df_p['45MA'] = df_p['Close'].rolling(45).mean()
        df_p['60MA'] = df_p['Close'].rolling(60).mean()
        
        # --- 自動買入訊號偵測 ---
        # 定義：當日收盤大於 30MA 且 (收盤-30MA)/30MA < 2% (即靠近均線支撐)
        df_p['Signal'] = (df_p['Close'] > df_p['30MA']) & ((df_p['Close'] - df_p['30MA']) / df_p['30MA'] <= 0.02)
        if strategy_option == "均線多頭回測":
            df_p['Signal'] &= (df_p['30MA'] > df_p['45MA']) & (df_p['45MA'] > df_p['60MA'])
        
        # 繪圖
        n_rows = 3 if indicator_choice != "都不顯示" else 2
        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.2, 0.3] if n_rows==3 else [0.7, 0.3])
        
        # 1. K線圖
        fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['30MA'], line=dict(color='#FFA500', width=1.5), name="30MA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['60MA'], line=dict(color='#4169E1', width=1.5), name="60MA"), row=1, col=1)

        # 2. 加入買入訊號標註 (Annotation)
        # 我們只標註最近 60 天內的訊號，避免圖表太亂
        sig_dates = df_p[df_p['Signal']].tail(5).index 
        for sig_date in sig_dates:
            fig.add_annotation(
                x=sig_date, y=df_p.loc[sig_date, 'Low'] * 0.98,
                text="買入", showarrow=True, arrowhead=2, arrowcolor="#9400D3",
                arrowsize=1.5, arrowwidth=2, ay=-30, bgcolor="#9400D3", font=dict(color="white"), row=1, col=1
            )

        # 3. 成交量
        v_clrs = ['red' if c >= o else 'green' for c, o in zip(df_p['Close'], df_p['Open'])]
        fig.add_trace(go.Bar(x=df_p.index, y=df_p['Volume'], marker_color=v_clrs, name="成交量"), row=2, col=1)

        # 4. 技術指標
        if indicator_choice == "RSI (強弱指標)":
            d = df_p['Close'].diff()
            rsi = 100 - (100 / (1 + (d.where(d > 0, 0)).rolling(14).mean()/( ( -d.where(d < 0, 0)).rolling(14).mean() + 1e-9)))
            fig.add_trace(go.Scatter(x=df_p.index, y=rsi, line=dict(color='purple'), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
        elif indicator_choice == "MACD (趨勢指標)":
            m_s = df_p['Close'].ewm(span=12).mean() - df_p['Close'].ewm(span=26).mean()
            h_s = m_s - m_s.ewm(span=9).mean()
            fig.add_trace(go.Bar(x=df_p.index, y=h_s, marker_color=['red' if v>=0 else 'green' for v in h_s], name="MACD柱"), row=3, col=1)

        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_layout(title=f"<b>{row['名稱']} ({row['ID']}) - 買入訊號分析</b>", xaxis_rangeslider_visible=False, height=700, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

elif not st.session_state['is_scanning']:
    st.info("請點擊上方按鈕開始掃描市場標的。")
