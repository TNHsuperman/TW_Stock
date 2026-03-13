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
import time

# --- 1. 系統初始化 ---
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
                        stock_info_map[ticker] = {"name": name, "industry": industry if pd.notna(industry) else "其他"}
        except Exception as e:
            continue
    return stocks_list, stock_info_map

# --- 3. UI 設定 ---
st.sidebar.header("⚙️ 策略參數設定")
strategy_option = st.sidebar.radio("選擇選股策略：", ("均線多頭回測", "均線糾結偵測"))
indicator_choice = st.sidebar.selectbox("查看確認指標：", ["都不顯示", "RSI (強弱指標)", "MACD (趨勢指標)"])
min_volume = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)

# --- 4. 掃描邏輯 ---
if not st.session_state['is_scanning']:
    if st.button("🔍 開始全市場掃描", use_container_width=True):
        st.session_state['is_scanning'] = True
        st.rerun()
else:
    all_stocks, info_map = get_stock_info_map()
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 限制掃描數量測試用，正式版可移除切片 [:500]
    target_stocks = all_stocks
    
    batch_size = 50 # 降低批次數量以提高穩定性
    for i in range(0, len(target_stocks), batch_size):
        batch = target_stocks[i:i + batch_size]
        status_text.text(f"掃描進度: {i} / {len(target_stocks)}")
        try:
            # 修正 yfinance 下載
            data = yf.download(batch, period="150d", group_by='ticker', progress=False, threads=True)
            
            for ticker in batch:
                try:
                    # 處理單一或多個股票返回的 DataFrame 結構
                    df = data[ticker] if len(batch) > 1 else data
                    df = df.dropna(subset=['Close'])
                    
                    if len(df) < 60: continue
                    if (df['Volume'].tail(5).mean() / 1000) < min_volume: continue

                    close = float(df['Close'].iloc[-1])
                    m30 = df['Close'].rolling(30).mean().iloc[-1]
                    m45 = df['Close'].rolling(45).mean().iloc[-1]
                    m60 = df['Close'].rolling(60).mean().iloc[-1]
                    bias_30 = ((close - m30) / m30) * 100
                    
                    keep = False
                    if strategy_option == "均線多頭回測":
                        if m30 > m45 > m60 and close > m30 and bias_30 < 2.0: 
                            keep = True
                    elif strategy_option == "均線糾結偵測":
                        ma_list = [m30, m45, m60]
                        ma_spread = (max(ma_list) - min(ma_list)) / min(ma_list)
                        if ma_spread <= 0.02 and abs(bias_30) < 2.0: 
                            keep = True
                    
                    if keep:
                        stock_data = info_map.get(ticker, {"name": "未知", "industry": "其他"})
                        results.append({
                            "ID": ticker, "代碼": ticker.split('.')[0], "名稱": stock_data["name"], 
                            "類股": stock_data["industry"], "收盤": round(close, 2), "乖離(%)": round(bias_30, 2)
                        })
                except: continue
        except: continue
        progress_bar.progress(min((i + batch_size) / len(target_stocks), 1.0))
    
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by="乖離(%)", ascending=True)
    st.session_state['scan_results'] = res_df
    st.session_state['is_scanning'] = False
    st.rerun()

# --- 5. 顯示與導覽邏輯 ---
if not st.session_state['scan_results'].empty:
    df_raw = st.session_state['scan_results']
    selected_industry = st.selectbox("🎯 篩選類股：", ["全部"] + sorted(df_raw["類股"].unique().tolist()))
    df_filtered = df_raw if selected_industry == "全部" else df_raw[df_raw["類股"] == selected_industry]
    df_filtered = df_filtered.reset_index(drop=True)

    if len(df_filtered) > 0:
        st.write(f"📊 符合條件標的：{len(df_filtered)} 支")
        
        # 顯示表格
        event = st.dataframe(df_filtered, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row", key="stock_table")
        
        # 同步選中索引
        if event.selection and event.selection.rows:
            st.session_state['selected_index'] = event.selection.rows[0]

        st.write("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("⬅️ 上一支", use_container_width=True):
                st.session_state['selected_index'] = (st.session_state['selected_index'] - 1) % len(df_filtered)
        with c2:
            st.markdown(f"<h3 style='text-align: center; color: #9400D3;'>{st.session_state['selected_index'] + 1} / {len(df_filtered)}</h3>", unsafe_allow_html=True)
        with c3:
            if st.button("下一支 ➡️", use_container_width=True):
                st.session_state['selected_index'] = (st.session_state['selected_index'] + 1) % len(df_filtered)

        # 繪圖
        idx = min(st.session_state['selected_index'], len(df_filtered)-1)
        row = df_filtered.iloc[idx]
        
        with st.spinner(f'載入中... {row["名稱"]} ({row["ID"]})'):
            df_p = yf.download(row['ID'], period="8mo", progress=False)
            if isinstance(df_p.columns, pd.MultiIndex): df_p.columns = df_p.columns.get_level_values(0)
            df_p = df_p.dropna(subset=['Close'])
            df_p['Date_Str'] = df_p.index.strftime('%Y-%m-%d')
            
            # 技術指標計算
            df_p['30MA'] = df_p['Close'].rolling(30).mean()
            df_p['45MA'] = df_p['Close'].rolling(45).mean()
            df_p['60MA'] = df_p['Close'].rolling(60).mean()
            
            # 繪圖 logic...
            n_rows = 3 if indicator_choice != "都不顯示" else 2
            fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25] if n_rows==3 else [0.75, 0.25])
            fig.add_trace(go.Candlestick(x=df_p['Date_Str'], open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="K線"), row=1, col=1)
            
            for ma, color in zip(['30MA', '45MA', '60MA'], ['#FFA500', '#2E8B57', '#4169E1']):
                fig.add_trace(go.Scatter(x=df_p['Date_Str'], y=df_p[ma], line=dict(color=color, width=1.5), name=ma), row=1, col=1)
            
            v_clrs = ['#ef5350' if c >= o else '#26a69a' for c, o in zip(df_p['Close'], df_p['Open'])]
            fig.add_trace(go.Bar(x=df_p['Date_Str'], y=df_p['Volume'], marker_color=v_clrs, name="成交量"), row=2, col=1)

            if indicator_choice == "RSI (強弱指標)":
                d = df_p['Close'].diff()
                rsi = 100 - (100 / (1 + (d.where(d > 0, 0)).rolling(14).mean()/( ( -d.where(d < 0, 0)).rolling(14).mean() + 1e-9)))
                fig.add_trace(go.Scatter(x=df_p['Date_Str'], y=rsi, line=dict(color='purple', width=1.2), name="RSI"), row=3, col=1)
            elif indicator_choice == "MACD (趨勢指標)":
                m_s = df_p['Close'].ewm(span=12).mean() - df_p['Close'].ewm(span=26).mean()
                h_s = m_s - m_s.ewm(span=9).mean()
                fig.add_trace(go.Bar(x=df_p['Date_Str'], y=h_s, marker_color=['#ef5350' if v>=0 else '#26a69a' for v in h_s], name="MACD柱"), row=3, col=1)

            fig.update_xaxes(type='category', showgrid=True, gridcolor='rgba(200,200,200,0.2)')
            fig.update_layout(title=f"<b>{row['名稱']} ({row['ID']})</b>", xaxis_rangeslider_visible=False, height=700, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("目前篩選條件下無標的。")
else:
    st.info("💡 提示：點擊「開始全市場掃描」按鈕。請注意，掃描台股全市場可能需要 1-2 分鐘。")
