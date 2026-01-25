import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. 初始化 ---
st.set_page_config(page_title="台股智慧選股 - 防崩潰穩定版", layout="wide")

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False

# --- 2. 資料獲取 (帶快取) ---
@st.cache_data(ttl=86400)
def get_stock_list():
    """從證交所抓取清單"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    stocks = []
    info_map = {}
    urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', ".TW"),
            ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', ".TWO")]
    for url, suffix in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            df = pd.read_html(StringIO(r.text))[0]
            df.columns = df.iloc[0]
            for _, row in df.iloc[1:].iterrows():
                val = str(row['有價證券代號及名稱'])
                if '　' in val:
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        ticker = f"{code}{suffix}"
                        stocks.append(ticker)
                        info_map[ticker] = {"name": name, "industry": row['產業別']}
        except: continue
    return stocks, info_map

# --- 3. 側邊欄 UI ---
st.sidebar.header("📊 選股條件設定")
strategy = st.sidebar.selectbox("策略選擇", ["強勢突破 (糾結+量增)", "成交量倍增", "均線多頭回測"])
min_vol = st.sidebar.slider("成交量門檻 (張)", 100, 2000, 500)
scan_limit = st.sidebar.number_input("掃描上限 (建議先測 100 檔)", 10, 2000, 100)

# --- 4. 掃描執行 (單線程安全模式) ---
if st.sidebar.button("🚀 開始安全掃描"):
    st.session_state['is_scanning'] = True
    st.session_state['scan_results'] = pd.DataFrame()

if st.session_state['is_scanning']:
    all_tickers, info_map = get_stock_list()
    target_tickers = all_tickers[:scan_limit] # 限制掃描數量防止線程溢出
    
    results = []
    prog = st.progress(0)
    status = st.empty()
    
    for idx, ticker in enumerate(target_tickers):
        try:
            # 關鍵：單支下載，禁用多線程
            df = yf.download(ticker, period="100d", progress=False, threads=False, auto_adjust=True)
            
            if df is None or len(df) < 60:
                continue
                
            # 處理可能的多重索引 (yfinance v0.2.40+)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 計算指標
            close = df['Close'].iloc[-1]
            vol = df['Volume'].iloc[-1] / 1000
            ma5v = df['Volume'].iloc[-6:-1].mean() / 1000
            m30, m45, m60 = df['Close'].rolling(30).mean().iloc[-1], df['Close'].rolling(45).mean().iloc[-1], df['Close'].rolling(60).mean().iloc[-1]
            
            # 判定邏輯
            keep = False
            if strategy == "強勢突破 (糾結+量增)":
                spread = (max(m30, m45, m60) - min(m30, m45, m60)) / min(m30, m45, m60)
                if spread < 0.02 and vol > ma5v * 2 and close > max(m30, m45, m60):
                    keep = True
            elif strategy == "成交量倍增":
                if vol > ma5v * 2 and close > df['Open'].iloc[-1]:
                    keep = True
            
            if keep and vol >= min_vol:
                results.append({
                    "代碼": ticker, "名稱": info_map[ticker]['name'],
                    "類股": info_map[ticker]['industry'], "收盤": round(close, 2),
                    "量增倍數": round(vol/ma5v, 1)
                })
        except Exception as e:
            pass # 忽略單支錯誤
            
        prog.progress((idx + 1) / len(target_tickers))
        status.text(f"🔍 檢查中 ({idx+1}/{len(target_tickers)}): {ticker}")
        time.sleep(0.05) # 極短暫休息，避免線程堆積
        
    st.session_state['scan_results'] = pd.DataFrame(results)
    st.session_state['is_scanning'] = False
    st.rerun()

# --- 5. 結果顯示 ---
if not st.session_state['scan_results'].empty:
    df = st.session_state['scan_results']
    st.success(f"✅ 掃描完成！找到 {len(df)} 檔符合條件")
    
    # 使用單純的表格顯示，避免 selection 觸發過多互動
    st.table(df)

    # 繪製選中股票的圖表 (可選)
    selected_stock = st.selectbox("查看詳細走勢", df['代碼'].tolist())
    if selected_stock:
        data = yf.download(selected_stock, period="6mo", threads=False)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="成交量"), row=2, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
else:
    if not st.session_state['is_scanning']:
        st.info("請從左側點擊按鈕開始掃描。建議先將掃描上限設定為 100 測試。")
