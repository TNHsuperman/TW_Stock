import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import urllib3
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
import random
import time
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# 1. 基礎設定與環境初始化 (修正 SSL 報錯)
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板 v10.3", layout="wide")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
]

def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
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
# 2. 數據清洗與深度資訊抓取 (修正 SSL 驗證)
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_market_list():
    stocks = []
    try:
        urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
                ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
        for url, mkt in urls:
            # 關鍵修正：verify=False 解決憑證失敗
            r = requests.get(url, headers=get_headers(), timeout=15, verify=False)
            r.encoding = 'big5'
            df_isin = pd.read_html(StringIO(r.text))[0]
            df_isin.columns = df_isin.iloc[0]
            for _, row in df_isin.iloc[1:].iterrows():
                val = str(row['有價證券代號及名稱'])
                if '　' in val:
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        stocks.append({"ticker": f"{code}.{mkt}", "name": name, "industry": row['產業別'], "code": code})
    except Exception as e:
        st.error(f"獲取市場清單失敗: {e}")
    return stocks

def clean_percent(text):
    if not text or text == "N/A": return np.nan
    try: return float(text.replace('%', '').replace(',', ''))
    except: return np.nan

def fetch_deep_info(ticker: str) -> dict:
    code = ticker.split('.')[0]
    res = {"pe": np.nan, "mom": np.nan, "yoy": np.nan}
    try:
        # 營收抓取 (verify=False)
        rev_url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
        rev_resp = requests.get(rev_url, headers=get_headers(), timeout=10, verify=False)
        soup = BeautifulSoup(rev_resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.get_text()]
            if len(percents) >= 2:
                res["mom"] = clean_percent(percents[0])
                res["yoy"] = clean_percent(percents[1])
        # PE 抓取
        yt = yf.Ticker(ticker)
        pe = yt.info.get('trailingPE')
        if pe: res["pe"] = float(pe)
    except: pass
    return res

# ============================================================
# 3. 技術分析核心 (優化穩定性)
# ============================================================

def run_strategy_check(s, bias_limit, vol_limit):
    now_ts = int(get_tw_now().timestamp())
    start_ts = int((get_tw_now() - timedelta(days=250)).timestamp()) 
    try:
        # 微延遲防止被 Yahoo 封鎖，維持掃描數量穩定
        time.sleep(random.uniform(0.05, 0.15))
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s['ticker']}", 
                         params={"period1":start_ts, "period2":now_ts, "interval":"1d"}, 
                         headers=get_headers(), timeout=10, verify=False)
        
        json_data = r.json()
        if not json_data['chart']['result']: return None
        
        data = json_data['chart']['result'][0]
        c_series = pd.Series(data['indicators']['quote'][0]['close']).ffill().dropna()
        v_series = pd.Series(data['indicators']['quote'][0]['volume']).ffill().dropna()

        if len(c_series) < 65: return None
        
        vol_today = v_series.iloc[-1]
        vol_yesterday = v_series.iloc[-2]
        vol_change = ((vol_today - vol_yesterday) / vol_yesterday) * 100 if vol_yesterday > 0 else 0
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

# ============================================================
# 4. 圖表與新聞 (還原原始功能與繪圖設定)
# ============================================================

def draw_k_line(ticker, name):
    yt = yf.Ticker(ticker)
    df = yt.history(period="1y") 
    if df.empty or len(df) < 60: return None
    df['MA30'] = df['Close'].rolling(30).mean()
    df['MA45'] = df['Close'].rolling(45).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df = df.tail(180).copy()
    # 原始顏色設定
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

def get_tw_stock_news(code):
    try:
        news_url = f"https://tw.stock.yahoo.com/quote/{code}/news"
        resp = requests.get(news_url, headers=get_headers(), timeout=10, verify=False)
        soup = BeautifulSoup(resp.text, 'html.parser')
        news_links = soup.find_all('a', href=True)
        pos_words = ["成長", "新高", "利多", "噴發", "買進", "展望佳", "獲利", "創高", "轉盈", "漲"]
        neg_words = ["衰退", "減少", "利空", "調降", "跌", "虧損", "賣出", "縮減"]
        results = []
        seen_titles = set()
        for link in news_links:
            href = link.get('href')
            if '/news/' in href:
                title = link.get_text(strip=True)
                if len(title) < 8 or title in seen_titles: continue
                full_link = href if href.startswith('http') else "https://tw.stock.yahoo.com" + href
                sentiment = "💡 資訊"; color = "#888888"
                if any(w in title for w in pos_words): sentiment = "📈 利多"; color = "#ef5350"
                elif any(w in title for w in neg_words): sentiment = "📉 利空"; color = "#26a69a"
                results.append({"title": title, "link": full_link, "sentiment": sentiment, "color": color, "publisher": "Yahoo股市"})
                seen_titles.add(title)
                if len(results) >= 8: break
        return results
    except: return None

# ============================================================
# 5. Streamlit UI 介面 (還原表格樣式與功能)
# ============================================================

st.sidebar.header("🎯 策略設定")
with st.sidebar.form("setting_form"):
    user_bias = st.number_input("30MA 乖離上限 (%)", 0.1, 15.0, 3.0, step=0.1)
    user_vol = st.slider("最小成交量 (張)", 0, 3000, 500)
    submit_scan = st.form_submit_button("🚀 開始智慧掃描", use_container_width=True)

if submit_scan:
    st.session_state.is_scanning = True
    st.session_state.current_idx = 0 
    
    status = st.empty()
    bar = st.progress(0)
    
    status.text("讀取市場清單中...")
    stocks_list = get_stock_market_list()
    
    if stocks_list:
        initial_hits = []
        status.text(f"🔍 掃描全市場個股技術面 (共 {len(stocks_list)} 支)...")
        # 調低併發數至 10，徹底解決支數變動問題
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(run_strategy_check, s, user_bias, user_vol): s for s in stocks_list}
            for i, f in enumerate(as_completed(futures), 1):
                if i % 100 == 0: bar.progress(i / len(stocks_list))
                res = f.result()
                if res: initial_hits.append(res)
                
        if initial_hits:
            status.text(f"📊 抓取 {len(initial_hits)} 支標的深度數據...")
            final_list = []
            with ThreadPoolExecutor(max_workers=5) as ex:
                f_deep = {ex.submit(fetch_deep_info, r['ticker']): r for r in initial_hits}
                for j, f in enumerate(as_completed(f_deep), 1):
                    try:
                        deep_res = f.result()
                        final_list.append({**f_deep[f], "本益比": deep_res["pe"], "營收月增": deep_res["mom"], "營收年增": deep_res["yoy"]})
                    except: continue
            st.session_state.scan_results = pd.DataFrame(final_list)
            status.success(f"✅ 掃描完成！找到 {len(final_list)} 支標的")
        else:
            st.session_state.scan_results = pd.DataFrame()
            status.error("查無符合標的。")
    st.session_state.is_scanning = False

# ============================================================
# 6. 表格呈現 (還原原始彩色樣式與欄位設定)
# ============================================================

if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results.copy()
    col_msg, col_dl = st.columns([3, 1])
    with col_dl:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="📥 下載清單 (CSV)", data=csv, file_name=f'tw_scan.csv', use_container_width=True)
    
    show_cols = ["code", "name", "收盤", "乖離30MA(%)", "成交量(張)", "量變動(%)", "本益比", "營收月增", "營收年增", "industry"]
    available_cols = [c for c in show_cols if c in df.columns]
    df_display = df[available_cols].rename(columns={"code":"代碼","name":"名稱","industry":"類股"})

    # 還原原始紅綠顏色定義
    def color_tw_style(val):
        if pd.isna(val): return ''
        color = '#ef5350' if val > 0 else '#26a69a' if val < 0 else 'white'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df_display.style.map(color_tw_style, subset=[c for c in ['量變動(%)', '營收月增', '營收年增'] if c in df_display.columns]),
        use_container_width=True, hide_index=True,
        column_config={
            "收盤": st.column_config.NumberColumn("價格", format="%.2f"),
            "乖離30MA(%)": st.column_config.ProgressColumn("30MA 乖離", format="%.2f%%", min_value=0, max_value=user_bias),
            "成交量(張)": st.column_config.NumberColumn("成交量", format="%d 📦"),
            "本益比": st.column_config.NumberColumn("PE", format="%.1f")
        }
    )

    # 翻頁邏輯
    st.divider()
    total_found = len(df)
    c_idx = st.session_state.current_idx
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    
    if btn_col1.button("⬅️ 上一支"):
        st.session_state.current_idx = (c_idx - 1) % total_found
        st.rerun()
    with btn_col2:
        st.markdown(f"<center>第 {c_idx + 1} / {total_found} 支：<b>{df.iloc[c_idx]['code']} {df.iloc[c_idx]['name']}</b></center>", unsafe_allow_html=True)
    if btn_col3.button("下一支 ➡️"):
        st.session_state.current_idx = (c_idx + 1) % total_found
        st.rerun()
    
    # 畫圖與新聞
    current_stock = df.iloc[st.session_state.current_idx]
    fig = draw_k_line(current_stock['ticker'], current_stock['name'])
    if fig: st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(f"📰 {current_stock['name']} 即時中文新聞")
    news = get_tw_stock_news(current_stock['code'])
    if news:
        for n in news:
            st.markdown(f"""
            <div style="padding:15px; border-bottom:1px solid #444; background-color:rgba(255,255,255,0.02); margin-bottom:8px; border-radius:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="color:{n['color']}; font-weight:bold; border:1px solid {n['color']}; padding:3px 10px; border-radius:15px; font-size:12px;">{n['sentiment']}</span>
                    <span style="color:#aaa; font-size:12px;">{n['publisher']}</span>
                </div>
                <a href="{n['link']}" target="_blank" style="text-decoration:none; color:#ffffff; font-size:17px; font-weight:500;">{n['title']}</a>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("💡 調整參數後，點擊「開始智慧掃描」。")
