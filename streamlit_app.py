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
if 'warnings' not in sys.modules:
    sys.modules['warnings'] = warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="台股智慧選股 - 終極穩定版", layout="wide")

# 初始化 Session State
if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = 0
if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False

# --- 2. 工具函數 ---
def fix_yfinance_format(df, ticker=None):
    """處理 yfinance MultiIndex 問題"""
    temp_df = df.copy()
    if isinstance(temp_df.columns, pd.MultiIndex):
        if ticker and ticker in temp_df.columns.get_level_values(1):
            temp_df = temp_df.xs(ticker, axis=1, level=1)
        else:
            temp_df.columns = temp_df.columns.get_level_values(0)
    return temp_df

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
            for _, row in df.iterrows():
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
st.sidebar.header("⚙️ 策略參數")
strategy_option = st.sidebar.radio(
    "選擇選股策略：", 
    ("均線多頭回測", "均線糾結偵測", "成交量倍增 (量能爆發)", "強勢突破 (糾結+量增)")
)
indicator_choice = st.sidebar.selectbox("查看確認指標：", ["都不顯示", "RSI (強弱指標)", "MACD (趨勢指標)"])
min_volume = st.sidebar.slider("日均成交量門檻 (張)", 0, 2000, 500, step=100)
use_filter = st.sidebar.checkbox("僅顯示轉強標的 (RSI > 50 或 MACD 柱狀體 > 0)")

# --- 4. 掃描邏輯 ---
# 點擊按鈕僅改變狀態，不直接執行長任務
if not st.session_state['is_scanning']:
    if st.button("🔍 開始全市場掃描", use_container_width=True):
        st.session_state['is_scanning'] = True
        st.rerun()
else:
    st.info("⏳ 正在掃描全市場股票，請勿重新整理網頁...")
    try:
        all_stocks, info_map = get_stock_info_map()
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 嚴格限制：小 Batch 且禁用 yfinance 執行緒
        batch_size = 15 
        for i in range(0, len(all_stocks), batch_size):
            batch = all_stocks[i:i + batch_size]
            status_text.text(f"掃描進度：{i} / {len(all_stocks)}")
            try:
                # 核心修正：threads=False 解決 RuntimeError
                data = yf.download(batch, period="100d", group_by='ticker', progress=False, auto_adjust=True, threads=False)
                
                for ticker in batch:
                    try:
                        df = fix_yfinance_format(data, ticker)
                        if df.empty or len(df) < 60: continue
                        
                        # 交易量門檻過濾
                        current_vol = df['Volume'].iloc[-1] / 1000
                        avg_vol_5d = df['Volume'].iloc[-6:-1].mean() / 1000 
                        if (df['Volume'].tail(5).mean() / 1000) < min_volume: continue

                        # 技術指標計算
                        close = float(df['Close'].iloc[-1])
                        open_p = float(df['Open'].iloc[-1])
                        m30 = df['Close'].rolling(30).mean().iloc[-1]
                        m45 = df['Close'].rolling(45).mean().iloc[-1]
                        m60 = df['Close'].rolling(60).mean().iloc[-1]
                        
                        keep = False
                        ma_list = [m30, m45, m60]
                        ma_spread = (max(ma_list) - min(ma_list)) / min(ma_list)
                        vol_ratio = current_vol / avg_vol_5d if avg_vol_5d > 0 else 0

                        # 策略判斷
                        if strategy_option == "均線多頭回測":
                            if m30 > m45 > m60 and close > m30 and (close - m30) / m30 <= 0.02: keep = True
                        elif strategy_option == "均線糾結偵測":
                            if ma_spread <= 0.015 and abs(close - m30) / m30 <= 0.02: keep = True
                        elif strategy_option == "成交量倍增 (量能爆發)":
                            if vol_ratio >= 2.0 and close > open_p: keep = True
                        elif strategy_option == "強勢突破 (糾結+量增)":
                            if ma_spread <= 0.02 and vol_ratio >= 2.0 and close > max(ma_list): keep = True
                        
                        if keep and use_filter:
                            delta = df['Close'].diff()
                            rsi = (100 - (100 / (1 + (delta.where(delta > 0, 0)).rolling(14).mean() / (-delta.where(delta < 0, 0)).rolling(14).mean()))).iloc[-1]
                            exp1, exp2 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
                            hist_v = (exp1 - exp2 - (exp1 - exp2).ewm(span=9).mean()).iloc[-1]
                            if indicator_choice == "RSI (強弱指標)" and rsi < 50: keep = False
                            if indicator_choice == "MACD (趨勢指標)" and hist_v < 0: keep = False

                        if keep:
                            stock_data = info_map.get(ticker, {"name": "未知", "industry": "其他"})
                            results.append({
                                "ID": ticker, "代碼": ticker.split('.')[0], 
                                "名稱": stock_data["name"], "類股": stock_data["industry"], 
                                "收盤": round(close, 2), "量增倍數": round(vol_ratio, 1)
                            })
                    except: continue
            except: continue
            progress_bar.progress(min((i + batch_size) / len(all_stocks), 1.0))
            time.sleep(0.2) # 給伺服器喘息空間
        
        st.session_state['scan_results'] = pd.DataFrame(results)
        st.session_state['selected_index'] = 0
        st.session_state['is_scanning'] = False
        st.rerun() # 掃描完畢後切換回顯示模式
    except Exception as e:
        st.error(f"掃描發生錯誤: {e}")
        st.session_state['is_scanning'] = False

# --- 5. 顯示結果 ---
if not st.session_state['scan_results'].empty and not st.session_state['is_scanning']:
    df_filtered = st.session_state['scan_results']
    
    # 類股篩選
    industries = ["全部"] + sorted(df_filtered["類股"].unique().tolist())
    selected_ind = st.selectbox("🎯 篩選類股：", industries)
    if selected_ind != "全部":
        display_df = df_filtered[df_filtered["類股"] == selected_ind].reset_index(drop=True)
    else:
        display_df = df_filtered.reset_index(drop=True)

    st.write(f"📊 篩選清單 (共 {len(display_df)} 檔)")
    
    # 表格顯示
    event = st.dataframe(
        display_df, 
        hide_index=True, 
        use_container_width=True, 
        on_select="rerun", 
        selection_mode="single-row", 
        key="stock_table"
    )

    if event.selection and event.selection.rows:
        st.session_state['selected_index'] = event.selection.rows[0]

    # 切換控制
    st.write("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("⬅️ 上一支", use_container_width=True):
            st.session_state['selected_index'] = (st.session_state['selected_index'] - 1) % len(display_df)
    with c2:
        idx_display = st.session_state['selected_index']
        st.markdown(f"<h3 style='text-align: center;'>{idx_display + 1} / {len(display_df)}</h3>", unsafe_allow_html=True)
    with c3:
        if st.button("下一支 ➡️", use_container_width=True):
            st.session_state['selected_index'] = (st.session_state['selected_index'] + 1) % len(display_df)

    # 繪圖邏輯
    if not display_df.empty:
        row = display_df.iloc[st.session_state['selected_index']]
        ticker_id = row['ID']
        
        with st.spinner(f'載入 {row["名稱"]} 技術圖表...'):
            df_p = yf.download(ticker_id, period="8mo", progress=False, auto_adjust=True, threads=False)
            df_p = fix_yfinance_format(df_p)
            
            # 計算均線
            df_p['30MA'] = df_p['Close'].rolling(30).mean()
            df_p['45MA'] = df_p['Close'].rolling(45).mean()
            df_p['60MA'] = df_p['Close'].rolling(60).mean()
            df_p['V_MA5'] = df_p['Volume'].rolling(5).mean()
            
            n_rows = 3 if indicator_choice != "都不顯示" else 2
            fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                               row_heights=[0.5, 0.2, 0.3] if n_rows==3 else [0.7, 0.3])
            
            # K線與均線
            fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="K線"), row=1, col=1)
            for ma, clr in zip(['30MA', '45MA', '60MA'], ['#FFA500', '#2E8B57', '#4169E1']):
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p[ma], line=dict(color=clr, width=1.2), name=ma), row=1, col=1)
            
            # 成交量
            v_clrs = ['red' if c >= o else 'green' for c, o in zip(df_p['Close'], df_p['Open'])]
            fig.add_trace(go.Bar(x=df_p.index, y=df_p['Volume'], marker_color=v_clrs, name="成交量"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p['V_MA5'], line=dict(color='gray', width=1), name="5日均量"), row=2, col=1)
            
            # 指標
            if indicator_choice == "RSI (強弱指標)":
                d = df_p['Close'].diff()
                rsi_s = 100 - (100 / (1 + (d.where(d > 0, 0)).rolling(14).mean()/( -d.where(d < 0, 0)).rolling(14).mean()))
                fig.add_trace(go.Scatter(x=df_p.index, y=rsi_s, line=dict(color='purple'), name="RSI"), row=3, col=1)
            elif indicator_choice == "MACD (趨勢指標)":
                m_s = df_p['Close'].ewm(span=12).mean() - df_p['Close'].ewm(span=26).mean()
                h_s = m_s - m_s.ewm(span=9).mean()
                fig.add_trace(go.Bar(x=df_p.index, y=h_s, marker_color=['red' if v>=0 else 'green' for v in h_s], name="MACD柱"), row=3, col=1)

            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            fig.update_layout(title=f"<b>{row['名稱']} ({ticker_id}) - 量增倍數: {row['量增倍數']}</b>", 
                              xaxis_rangeslider_visible=False, height=650, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

elif not st.session_state['scan_results'].empty:
    st.warning("查無標的。")
