import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
import random
import time
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# 1. 基礎設定與環境初始化
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板 v9.0", layout="wide")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
]

def get_headers():
    return {'User-Agent': random.choice(USER_AGENTS)}

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False
if 'current_idx' not in st.session_state:
    st.session_state['current_idx'] = 0

def get_tw_now():
    return datetime.now(timezone(timedelta(hours=8)))

# ============================================================
# 2. 數據清洗與深度資訊抓取
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_market_list():
    stocks = []
    try:
        urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
                ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
        for url, mkt in urls:
            r = requests.get(url, headers=get_headers(), timeout=10, verify=False)
            df_isin = pd.read_html(StringIO(r.text))[0]
            df_isin.columns = df_isin.iloc[0]
            for _, row in df_isin.iloc[1:].iterrows():
                val = str(row['有價證券代號及名稱'])
                if '　' in val:
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        stocks.append({"ticker": f"{code}.{mkt}", "name": name, "industry": row['產業別'], "code": code})
    except: pass
    return stocks

def clean_percent(text):
    if not text or text == "N/A": return np.nan
    try: return float(text.replace('%', '').replace(',', ''))
    except: return np.nan

def fetch_deep_info(ticker: str) -> dict:
    code = ticker.split('.')[0]
    res = {"pe": np.nan, "mom": np.nan, "yoy": np.nan}
    try:
        yt = yf.Ticker(ticker)
        pe = yt.info.get('trailingPE')
        if pe: res["pe"] = float(pe)
    except: pass
    try:
        rev_url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
        rev_resp = requests.get(rev_url, headers=get_headers(), timeout=5)
        soup = BeautifulSoup(rev_resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.get_text()]
            if len(percents) >= 2:
                res["mom"] = clean_percent(percents[0])
                res["yoy"] = clean_percent(percents[1])
    except: pass
    return res

# ============================================================
# 3. 技術分析、繪圖與新聞抓取邏輯
# ============================================================

def run_strategy_check(s, bias_limit, vol_limit):
    now_ts = int(get_tw_now().timestamp())
    start_ts = int((get_tw_now() - timedelta(days=250)).timestamp()) 
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}", 
                         params={"period1":start_ts, "period2":now_ts, "interval":"1d"}, headers=get_headers(), timeout=10)
        data = r.json()['chart']['result'][0]
        c_series = pd.Series(data['indicators']['quote'][0]['close']).ffill().dropna()
        v_series = pd.Series(data['indicators']['quote'][0]['volume']).ffill().dropna()
        if len(c_series) < 65: return None
        
        vol_today = v_series.iloc[-1]
        vol_change = ((vol_today - v_series.iloc[-2]) / v_series.iloc[-2]) * 100 if len(v_series)>1 else 0
        curr_vol_int = int(vol_today / 1000)
        avg_vol_5d = v_series.tail(5).mean() / 1000
        if avg_vol_5d < vol_limit: return None
        
        ma30 = c_series.rolling(30).mean().iloc[-1]
        ma45 = c_series.rolling(45).mean().iloc[-1]
        ma60 = c_series.rolling(60).mean().iloc[-1]
        curr_price = c_series.iloc[-1]
        bias_30 = ((curr_price - ma30) / ma30) * 100
        
        if (ma30 > ma45 > ma60) and (0 <= bias_30 <= bias_limit):
            return {**s, "收盤": round(curr_price, 2), "乖離30MA(%)": round(bias_30, 2), "成交量(張)": curr_vol_int, "量變動(%)": round(vol_change, 2)}
    except: return None
    return None

def draw_k_line(ticker, name):
    yt = yf.Ticker(ticker)
    df = yt.history(period="1y") 
    if df.empty or len(df) < 60: return None
    df['MA30'] = df['Close'].rolling(30).mean()
    df['MA45'] = df['Close'].rolling(45).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df = df.tail(180).copy()
    colors = ['#ef5350' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#26a69a' for i in range(len(df))]
    df.index = df.index.strftime('%Y-%m-%d')
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                                 name='K線', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA30'], line=dict(color='orange', width=1.5), name='30MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA45'], line=dict(color='blue', width=1.5), name='45MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='purple', width=1.5), name='60MA'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors), row=2, col=1)
    fig.update_layout(title=f"{name} ({ticker}) 180日K線與均線圖", xaxis_rangeslider_visible=False, height=600, template='plotly_dark')
    fig.update_xaxes(type='category') 
    return fig

def get_stock_news(ticker):
    """抓取股票相關新聞並進行簡單利多利空分析"""
    try:
        yt = yf.Ticker(ticker)
        news = yt.news[:5] # 抓取最新 5 則
        if not news: return None
        
        pos_words = ["成長", "新高", "利多", "噴發", "買進", "展望佳", "獲利", "創高", "轉盈", "法說", "漲"]
        neg_words = ["衰退", "減少", "利空", "調降", "跌", "虧損", "賣出", "縮減", "保守", "裁員", "崩"]
        
        results = []
        for n in news:
            title = n.get('title', '')
            link = n.get('link', '')
            sentiment = "💡 資訊"
            color = "#888888"
            
            if any(w in title for w in pos_words):
                sentiment = "📈 利多"
                color = "#ef5350"
            elif any(w in title for w in neg_words):
                sentiment = "📉 利空"
                color = "#26a69a"
                
            results.append({"title": title, "link": link, "sentiment": sentiment, "color": color})
        return results
    except: return None

# ============================================================
# 4. Streamlit UI 介面
# ============================================================

st.sidebar.header("🎯 策略設定")
user_bias = st.sidebar.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0, step=0.1)
user_vol = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if st.button("🚀 開始全市場智慧掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning = True
    st.session_state.current_idx = 0 
    st.rerun()

if st.session_state.is_scanning:
    status = st.empty()
    bar = st.progress(0)
    stocks_list = get_stock_market_list()
    initial_hits = []
    status.text(f"🔍 正在掃描全市場...")
    with ThreadPoolExecutor(max_workers=30) as ex:
        futures = {ex.submit(run_strategy_check, s, user_bias, user_vol): s for s in stocks_list}
        for i, f in enumerate(as_completed(futures), 1):
            if i % 100 == 0: bar.progress(i / len(stocks_list))
            res = f.result()
            if res: initial_hits.append(res)
            
    if initial_hits:
        status.text(f"📊 正在抓取財報數據...")
        final_list = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            f_deep = {ex.submit(fetch_deep_info, r['ticker']): r for r in initial_hits}
            for j, f in enumerate(as_completed(f_deep), 1):
                status.text(f"進度: {j} / {len(initial_hits)}")
                deep_res = f.result()
                final_list.append({**f_deep[f], "本益比": deep_res["pe"], "營收月增": deep_res["mom"], "營收年增": deep_res["yoy"]})
        st.session_state.scan_results = pd.DataFrame(final_list)
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("查無條件標的。")
    st.session_state.is_scanning = False
    st.rerun()

# ============================================================
# 5. 結果顯示 (維持原始表格呈現)
# ============================================================

if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results.copy()
    
    col_msg, col_dl = st.columns([3, 1])
    with col_msg:
        st.success(f"✅ 掃描完成！找到 {len(df)} 支標的")
    with col_dl:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        tw_date = get_tw_now().strftime("%Y%m%d")
        st.download_button(label="📥 下載選股清單 (CSV)", data=csv, file_name=f'tw_stock_scan_{tw_date}.csv', mime='text/csv', use_container_width=True)
    
    show_cols = ["code", "name", "收盤", "乖離30MA(%)", "成交量(張)", "量變動(%)", "本益比", "營收月增", "營收年增", "industry"]
    available_cols = [c for c in show_cols if c in df.columns]
    df_display = df[available_cols].rename(columns={"code":"代碼","name":"名稱","industry":"類股"})

    def color_tw_style(val):
        if pd.isna(val): return ''
        color = '#ef5350' if val > 0 else '#26a69a' if val < 0 else 'white'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df_display.style.map(color_tw_style, subset=[c for c in ['量變動(%)', '營收月增', '營收年增'] if c in df_display.columns]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "代碼": st.column_config.TextColumn("代碼"),
            "名稱": st.column_config.TextColumn("名稱"),
            "收盤": st.column_config.NumberColumn("價格", format="%.2f"),
            "乖離30MA(%)": st.column_config.ProgressColumn(
                "30MA 乖離",
                help=f"數值越小(越接近0%)代表股價越貼近支撐線。目前上限設定為 {user_bias}%",
                format="%.2f%%",
                min_value=0,
                max_value=user_bias,
            ),
            "量變動(%)": st.column_config.NumberColumn("量變動", format="%.1f%%"),
            "營收月增": st.column_config.NumberColumn("營收月增", format="%.1f%%"),
            "營收年增": st.column_config.NumberColumn("營收年增", format="%.1f%%"),
            "本益比": st.column_config.NumberColumn("PE", format="%.1f"),
            "成交量(張)": st.column_config.NumberColumn("成交量", format="%d 📦"),
            "類股": st.column_config.TextColumn("產業別")
        }
    )
    
    st.caption(f"💡 註1：進度條滿格代表乖離率接近你的上限值 ({user_bias}%)；條狀越短代表股價越貼近 30MA。")
    st.caption(f"💡 註2：營收增長與量變動如果為正數，會以紅色粗體顯示。")
    st.caption(f"💡 數據更新時間：{get_tw_now().strftime('%Y-%m-%d %H:%M:%S')} (台灣時間)")

    # ============================================================
    # 6. K線圖與切換區 (新增新聞功能)
    # ============================================================
    st.divider()
    
    total_found = len(df)
    c_idx = st.session_state.current_idx
    
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col1:
        if st.button("⬅️ 上一支", use_container_width=True) and c_idx > 0:
            st.session_state.current_idx -= 1
            st.rerun()
    with btn_col2:
        st.markdown(f"<center>第 {c_idx + 1} / {total_found} 支：<b>{df.iloc[c_idx]['code']} {df.iloc[c_idx]['name']}</b></center>", unsafe_allow_html=True)
    with btn_col3:
        if st.button("下一支 ➡️", use_container_width=True) and c_idx < total_found - 1:
            st.session_state.current_idx += 1
            st.rerun()
    
    current_stock = df.iloc[c_idx]
    
    # 繪製 K 線
    k_fig = draw_k_line(current_stock['ticker'], current_stock['name'])
    if k_fig:
        st.plotly_chart(k_fig, use_container_width=True)
    
    # 新增：新聞分析區塊
    st.subheader(f"📰 {current_stock['name']} 即時新聞分析")
    news_list = get_stock_news(current_stock['ticker'])
    
    if news_list:
        for n in news_list:
            st.markdown(f"""
            <div style="padding:10px; border-bottom:1px solid #444;">
                <span style="color:{n['color']}; font-weight:bold; border:1px solid {n['color']}; padding:2px 6px; border-radius:4px; margin-right:10px;">
                    {n['sentiment']}
                </span>
                <a href="{n['link']}" target="_blank" style="text-decoration:none; color:#ddd; font-size:16px;">{n['title']}</a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("暫無此標的之近期相關新聞。")

else:
    if not st.session_state.is_scanning:
        st.info("💡 調整左側參數後，點擊按鈕執行智慧選股。")
