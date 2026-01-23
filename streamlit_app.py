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

# 手機版優化：不強制 wide mode，讓元件自動堆疊
st.set_page_config(page_title="台股選股", layout="centered")

# 自定義 CSS 讓手機版表格更易讀
st.markdown("""
    <style>
    .stDataFrame { width: 100%; }
    [data-testid="stSidebar"] { width: 250px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 台股多策略選股")

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
        except: continue
    return stocks_list, stock_info_map

# --- 3. UI 策略選擇 (手機側邊欄) ---
st.sidebar.header("⚙️ 策略參數")
strategy_option = st.sidebar.radio(
    "選擇選股策略：",
    ("均線多頭回測", "均線糾結偵測")
)

# 增加成交量濾網 (手機版必備，避免流動性風險)
min_volume = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)

if strategy_option == "均線多頭回測":
    st.sidebar.info("💡 30>45>60MA 且股價靠近 30MA (2%內)")
else:
    st.sidebar.info("💡 三線間距 < 1.5%，預期變盤")

# --- 4. 核心掃描邏輯 ---
if st.button(f"🔍 開始全市場掃描", use_container_width=True):
    all_stocks, info_map = get_stock_info_map()
    results = []
    progress_bar = st.progress(0)
    
    # 為了手機效能，使用較大的 batch 減少請求次數
    batch_size = 100
    for i in range(0, len(all_stocks), batch_size):
        batch = all_stocks[i:i + batch_size]
        try:
            data = yf.download(batch, period="150d", group_by='ticker', progress=False)
        except: continue
        
        for ticker in batch:
            try:
                df = data[ticker] if len(batch) > 1 else data
                df = df.dropna(subset=['Close'])
                if len(df) < 60: continue
                
                # 成交量過濾
                avg_vol = df['Volume'].tail(5).mean() / 1000 # 換算成張數
                if avg_vol < min_volume: continue

                close = float(df['Close'].iloc[-1])
                m30 = df['Close'].rolling(30).mean().iloc[-1]
                m45 = df['Close'].rolling(45).mean().iloc[-1]
                m60 = df['Close'].rolling(60).mean().iloc[-1]
                
                ma_list = [m30, m45, m60]
                ma_max, ma_min = max(ma_list), min(ma_list)
                ma_spread = (ma_max - ma_min) / ma_min
                
                keep = False
                score_val = 0
                
                if strategy_option == "均線多頭回測":
                    if m30 > m45 > m60 and close > m30:
                        bias = (close - m30) / m30
                        if bias <= 0.02:
                            keep = True
                            score_val = round(bias * 100, 2)
                
                elif strategy_option == "均線糾結偵測":
                    if ma_spread <= 0.015:
                        price_to_ma = abs(close - m30) / m30
                        if price_to_ma <= 0.02:
                            keep = True
                            score_val = round(ma_spread * 100, 2)
                
                if keep:
                    stock_data = info_map.get(ticker, {"name": "未知", "industry": "其他"})
                    results.append({
                        "ID": ticker, "代碼": ticker.split('.')[0], "名稱": stock_data["name"],
                        "類股": stock_data["industry"], "收盤": round(close, 2),
                        "指標": score_val
                    })
            except: continue
        progress_bar.progress(min((i + batch_size) / len(all_stocks), 1.0))
    
    st.session_state['scan_results'] = pd.DataFrame(results)
    st.session_state['current_strategy'] = strategy_option

# --- 5. 顯示結果與繪圖 ---
if 'scan_results' in st.session_state and not st.session_state['scan_results'].empty:
    df_raw = st.session_state['scan_results']
    
    # 手機版類股篩選置中
    all_industries = ["全部"] + sorted(df_raw["類股"].unique().tolist())
    selected_industry = st.selectbox("🎯 篩選類股：", all_industries)
    
    df_filtered = df_raw if selected_industry == "全部" else df_raw[df_raw["類股"] == selected_industry]
    df_filtered = df_filtered.sort_values("指標")

    score_label = "乖離%" if st.session_state['current_strategy'] == "均線多頭回測" else "糾結%"
    
    st.info("📱 點擊下方清單查看 K 線圖")
    event = st.dataframe(
        df_filtered,
        column_config={
            "ID": None,
            "指標": st.column_config.NumberColumn(score_label, format="%.2f%%"),
            "類股": st.column_config.TextColumn("類股", width="small")
        },
        hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row"
    )

    if event.selection.rows:
        selected_index = event.selection.rows[0]
        row = df_filtered.iloc[selected_index]
        ticker_id, stock_name = row['ID'], row['名稱']
        
        with st.spinner('繪製中...'):
            df_plot = yf.download(ticker_id, period="8mo", progress=False)
            if isinstance(df_plot.columns, pd.MultiIndex): df_plot.columns = df_plot.columns.get_level_values(0)
            df_plot['30MA'] = df_plot['Close'].rolling(30).mean()
            df_plot['45MA'] = df_plot['Close'].rolling(45).mean()
            df_plot['60MA'] = df_plot['Close'].rolling(60).mean()
            df_plot = df_plot.tail(60) # 手機版看 60 天即可，解析度較好
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="K線"), row=1, col=1)
            
            for ma, color in zip(['30MA', '45MA', '60MA'], ['#FFA500', '#2E8B57', '#4169E1']):
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[ma], line=dict(color=color, width=1.5), name=ma), row=1, col=1)
            
            v_colors = ['red' if c >= o else 'green' for c, o in zip(df_plot['Close'], df_plot['Open'])]
            fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=v_colors, name="成交量"), row=2, col=1)
            
            # 手機版互動優化
            fig.update_layout(
                title=f"{stock_name} ({ticker_id})",
                xaxis_rangeslider_visible=False,
                height=500, # 降低高度適配手機
                margin=dict(l=10, r=10, t=40, b=10),
                template="plotly_white",
                hovermode='x unified',
                dragmode='pan' # 預設平移模式，方便觸控
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

elif 'scan_results' in st.session_state:
    st.warning("查無標的，請試著調低成交量門檻。")