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
st.set_page_config(page_title="台股智慧選股儀表板 v10.1", layout="wide")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
]

def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8',
        'Referer': 'https://tw.stock.yahoo.com/',
        'Connection': 'keep-alive'
    }

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False
if 'current_idx' not in st.session_state:
    st.session_state['current_idx'] = 0

def get_tw_now():
    return datetime.now(timezone(timedelta(hours=8)))

# ============================================================
# 2. 數據抓取邏輯
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_market_list():
    stocks = []
    try:
        urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
                ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
        for url, mkt in urls:
            r = requests.get(url, headers=get_headers(), timeout=15, verify=False)
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
        rev_resp = requests.get(rev_url, headers=get_headers(), timeout=10)
        soup = BeautifulSoup(rev_resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.get_text()]
            if len(percents) >= 2:
                res["mom"] = float(percents[0].replace('%','').replace(',',''))
                res["yoy"] = float(percents[1].replace('%','').replace(',',''))
    except: pass
    return res

# ============================================================
# 3. 繪圖與新聞功能
# ============================================================

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
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA30'], line=dict(color='orange', width=1.5), name='30MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA45'], line=dict(color='blue', width=1.5), name='45MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='purple', width=1.5), name='60MA'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors), row=2, col=1)
    fig.update_layout(title=f"{name} ({ticker}) 180日K線", xaxis_rangeslider_visible=False, height=600, template='plotly_dark')
    fig.update_xaxes(type='category') 
    return fig

def get_tw_stock_news(code):
    try:
        news_url = f"https://tw.stock.yahoo.com/quote/{code}/news"
        resp = requests.get(news_url, headers=get_headers(), timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        news_links = soup.find_all('a', href=True)
        pos_words = ["成長", "新高", "利多", "噴發", "買進", "展望佳", "獲利", "創高", "轉盈", "法說", "漲", "配息", "訂單", "亮眼"]
        neg_words = ["衰退", "減少", "利空", "調降", "跌", "虧損", "賣出", "縮減", "保守", "淡季", "壓力", "下修"]
        results = []
        seen = set()
        for link in news_links:
            href = link.get('href')
            if '/news/' in href:
                title = link.get_text(strip=True)
                if len(title) < 8 or title in seen: continue
                full_link = href if href.startswith('http') else "https://tw.stock.yahoo.com" + href
                sentiment, color = ("📈 利多", "#ef5350") if any(w in title for w in pos_words) else (("📉 利空", "#26a69a") if any(w in title for w in neg_words) else ("💡 資訊", "#888888"))
                results.append({"title": title, "link": full_link, "sentiment": sentiment, "color": color, "publisher": "Yahoo股市"})
                seen.add(title)
                if len(results) >= 8: break
        return results
    except: return None

# ============================================================
# 4. 掃描與分析
# ============================================================

st.sidebar.header("🎯 策略設定")
user_bias = st.sidebar.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0, step=0.1)
user_vol = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

if st.button("🚀 開始全市場智慧掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning = True
    st.rerun()

if st.session_state.is_scanning:
    status = st.empty()
    bar = st.progress(0)
    stocks_list = get_stock_market_list()
    initial_hits = []
    
    def check_logic(s):
        now_ts = int(get_tw_now().timestamp())
        start_ts = int((get_tw_now() - timedelta(days=250)).timestamp()) 
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}", params={"period1":start_ts, "period2":now_ts, "interval":"1d"}, headers=get_headers(), timeout=10)
            data = r.json()['chart']['result'][0]
            c = pd.Series(data['indicators']['quote'][0]['close']).ffill().dropna()
            v = pd.Series(data['indicators']['quote'][0]['volume']).ffill().dropna()
            if len(c) < 65 or (v.tail(5).mean()/1000) < user_vol: return None
            m30, m45, m60 = c.rolling(30).mean().iloc[-1], c.rolling(45).mean().iloc[-1], c.rolling(60).mean().iloc[-1]
            bias = ((c.iloc[-1] - m30) / m30) * 100
            if (m30 > m45 > m60) and (0 <= bias <= user_bias):
                return {**s, "收盤": round(c.iloc[-1], 2), "乖離30MA(%)": round(bias, 2), "成交量(張)": int(v.iloc[-1]/1000), "量變動(%)": round(((v.iloc[-1]-v.iloc[-2])/v.iloc[-2])*100, 2)}
        except: pass
        return None

    with ThreadPoolExecutor(max_workers=30) as ex:
        futures = {ex.submit(check_logic, s): s for s in stocks_list}
        for i, f in enumerate(as_completed(futures), 1):
            if i % 100 == 0: bar.progress(i / len(stocks_list))
            res = f.result()
            if res: initial_hits.append(res)
    
    if initial_hits:
        final_list = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            f_deep = {ex.submit(fetch_deep_info, r['ticker']): r for r in initial_hits}
            for f in as_completed(f_deep):
                d = f.result()
                final_list.append({**f_deep[f], "本益比": d["pe"], "營收月增": d["mom"], "營收年增": d["yoy"]})
        st.session_state.scan_results = pd.DataFrame(final_list)
        st.session_state.current_idx = 0
    st.session_state.is_scanning = False
    st.rerun()

# ============================================================
# 5. 表格與點擊聯動 (修正後的關鍵部分)
# ============================================================

if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results.copy()
    col_msg, col_dl = st.columns([3, 1])
    with col_msg: st.success(f"✅ 找到 {len(df)} 支標的")
    with col_dl:
        st.download_button(label="📥 下載清單 (CSV)", data=df.to_csv(index=False).encode('utf-8-sig'), file_name='stock.csv', use_container_width=True)

    # 為了支援 on_select，我們不使用 .style (Styler)，直接傳入 DataFrame
    show_cols = ["code", "name", "收盤", "乖離30MA(%)", "成交量(張)", "量變動(%)", "本益比", "營收月增", "營收年增", "industry"]
    df_display = df[show_cols].rename(columns={"code":"代碼","name":"名稱","industry":"類股"})

    # 點擊表格事件
    event = st.dataframe(
        df_display, # 直接傳入，不加 .style.map
        use_container_width=True, 
        hide_index=True, 
        on_select="rerun", 
        selection_mode="single_row",
        column_config={
            "乖離30MA(%)": st.column_config.ProgressColumn("30MA 乖離", format="%.2f%%", min_value=0, max_value=user_bias),
            "量變動(%)": st.column_config.NumberColumn(format="%.1f%%"),
            "營收月增": st.column_config.NumberColumn(format="%.1f%%"),
            "營收年增": st.column_config.NumberColumn(format="%.1f%%"),
        }
    )

    if event and event.selection.rows:
        st.session_state.current_idx = event.selection.rows[0]

    st.caption(f"💡 註：點擊表格任一列可切換 K線與新聞。更新時間：{get_tw_now().strftime('%H:%M:%S')} (台灣時間)")

    # ============================================================
    # 6. K線圖與迴圈切換
    # ============================================================
    st.divider()
    total = len(df)
    c_idx = st.session_state.current_idx
    
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col1:
        if st.button("⬅️ 上一支", use_container_width=True):
            st.session_state.current_idx = (c_idx - 1) % total
            st.rerun()
    with btn_col2:
        st.markdown(f"<center>第 {c_idx + 1} / {total} 支：<b>{df.iloc[c_idx]['code']} {df.iloc[c_idx]['name']}</b></center>", unsafe_allow_html=True)
    with btn_col3:
        if st.button("下一支 ➡️", use_container_width=True):
            st.session_state.current_idx = (c_idx + 1) % total
            st.rerun()
    
    current = df.iloc[c_idx]
    k_fig = draw_k_line(current['ticker'], current['name'])
    if k_fig: st.plotly_chart(k_fig, use_container_width=True)
    
    st.subheader(f"📰 {current['name']} 即時中文新聞")
    news_list = get_tw_stock_news(current['code'])
    if news_list:
        for n in news_list:
            st.markdown(f'<div style="padding:12px; border-bottom:1px solid #444; background:rgba(255,255,255,0.02); border-radius:10px; margin-bottom:8px;">'
                        f'<span style="color:{n["color"]}; font-weight:bold; border:1px solid {n["color"]}; padding:2px 8px; border-radius:15px; font-size:12px;">{n["sentiment"]}</span> '
                        f'<span style="color:#aaa; font-size:12px;"> {n["publisher"]}</span><br>'
                        f'<a href="{n["link"]}" target="_blank" style="text-decoration:none; color:#ffffff; font-size:16px;">{n["title"]}</a></div>', unsafe_allow_html=True)
else:
    if not st.session_state.is_scanning: st.info("💡 調整參數後開始掃描。")
