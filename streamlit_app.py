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
import re

# ============================================================
# 1. 基礎設定與環境初始化
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板 v11.0 - Goodinfo 強化版", layout="wide")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
]

def get_headers(referer="https://goodinfo.com.tw/tw/index.asp"):
    """強化 Headers 以騙過 Goodinfo 反爬蟲"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': referer,
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1'
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
# 2. 數據清洗與深度資訊抓取 (Goodinfo 強制解析版)
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
    """最強 PE 抓取：優先從 Goodinfo 爬取，Yahoo 備援"""
    code = ticker.split('.')[0]
    res = {"pe": np.nan, "mom": np.nan, "yoy": np.nan}
    
    # --- 步驟 1: 優先攻擊 Goodinfo 抓取 PE (紅框處) ---
    try:
        # Goodinfo 網址結構
        gi_url = f"https://goodinfo.com.tw/tw/StockDetail.asp?STOCK_ID={code}"
        # 注意：Goodinfo 非常看重 Referer 與 Cookie
        gi_resp = requests.get(gi_url, headers=get_headers(), timeout=12)
        gi_resp.encoding = 'utf-8' # 強制編碼轉為 UTF-8
        
        if gi_resp.status_code == 200:
            soup = BeautifulSoup(gi_resp.text, 'html.parser')
            # 尋找圖中紅框位元。Goodinfo 的結構通常是表格位元 (td)
            # 尋找文本為 "PER" 或 "本益比" 的 cell
            target_td = soup.find('td', string="PER")
            if not target_td:
                target_td = soup.find('td', string="本益比")
            
            if target_td:
                # 獲取它相鄰的下一個單元格數值
                pe_val_node = target_td.find_next_sibling('td')
                if pe_val_node:
                    pe_text = pe_val_node.get_text(strip=True)
                    if pe_text and pe_text != '-' and pe_text != 'N/A':
                        res["pe"] = float(pe_text)
    except Exception as e:
        pass # Goodinfo 失敗則進入備援

    # --- 步驟 2: Yahoo 備援抓取 PE 與 營收 ---
    try:
        if np.isnan(res["pe"]):
            # 如果 Goodinfo 沒抓到，嘗試 Yahoo 概況頁面
            y_url = f"https://tw.stock.yahoo.com/quote/{code}"
            y_resp = requests.get(y_url, headers=get_headers("https://tw.stock.yahoo.com/"), timeout=10)
            y_soup = BeautifulSoup(y_resp.text, 'html.parser')
            # 查找 PE 標籤
            pe_label = y_soup.find(text=re.compile("本益比"))
            if pe_label:
                val = pe_label.find_next(class_=re.compile("Fw\(b\)"))
                if val and val.get_text() != '-':
                    res["pe"] = float(val.get_text().replace(',', ''))

        # 抓取營收
        rev_url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
        rev_resp = requests.get(rev_url, headers=get_headers("https://tw.stock.yahoo.com/"), timeout=10)
        r_soup = BeautifulSoup(rev_resp.text, 'html.parser')
        row = r_soup.select_one(r'li.List\(n\)')
        if row:
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.get_text()]
            if len(percents) >= 2:
                try: res["mom"] = float(percents[0].replace('%', ''))
                except: pass
                try: res["yoy"] = float(percents[1].replace('%', ''))
                except: pass
    except: pass

    return res

# ============================================================
# 3. 技術分析與繪圖 (保留原本所有邏輯)
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
    fig.update_layout(title=f"{name} ({ticker}) 180日K線圖", xaxis_rangeslider_visible=False, height=600, template='plotly_dark')
    fig.update_xaxes(type='category') 
    return fig

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
    status.text(f"🔍 掃描全市場技術面...")
    with ThreadPoolExecutor(max_workers=30) as ex:
        futures = {ex.submit(run_strategy_check, s, user_bias, user_vol): s for s in stocks_list}
        for i, f in enumerate(as_completed(futures), 1):
            if i % 100 == 0: bar.progress(i / len(stocks_list))
            res = f.result()
            if res: initial_hits.append(res)
            
    if initial_hits:
        status.text(f"📊 強制爬取 Goodinfo 本益比數據 (進度：0/{len(initial_hits)})")
        final_list = []
        # Goodinfo 限制嚴格，這裡降低並發數並增加隨機延遲
        with ThreadPoolExecutor(max_workers=5) as ex:
            f_deep = {ex.submit(fetch_deep_info, r['ticker']): r for r in initial_hits}
            for j, f in enumerate(as_completed(f_deep), 1):
                status.text(f"深度抓取進度: {j} / {len(initial_hits)}")
                deep_res = f.result()
                final_list.append({**f_deep[f], "本益比": deep_res["pe"], "營收月增": deep_res["mom"], "營收年增": deep_res["yoy"]})
                time.sleep(random.uniform(0.5, 1.2)) # 必須延遲避免被鎖 IP
        st.session_state.scan_results = pd.DataFrame(final_list)
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("查無條件標的。")
    st.session_state.is_scanning = False
    st.rerun()

# ============================================================
# 5. 結果顯示 (表格呈現)
# ============================================================

if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results.copy()
    st.success(f"✅ 掃描完成！找到 {len(df)} 支符合技術面標的")
    
    # PE 資料呈現優化
    df_display = df.rename(columns={"code":"代碼","name":"名稱","industry":"類股"})

    def color_tw_style(val):
        if pd.isna(val) or val == 0: return ''
        color = '#ef5350' if val > 0 else '#26a69a' if val < 0 else 'white'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df_display.style.map(color_tw_style, subset=[c for c in ['量變動(%)', '營收月增', '營收年增'] if c in df_display.columns]),
        use_container_width=True, hide_index=True,
        column_config={
            "本益比": st.column_config.NumberColumn("PE (Goodinfo)", format="%.2f"),
            "收盤": st.column_config.NumberColumn("價格", format="%.2f"),
            "乖離30MA(%)": st.column_config.NumberColumn("30MA乖離", format="%.2f%%"),
            "量變動(%)": st.column_config.NumberColumn("量變動", format="%.1f%%")
        }
    )
    
    # 底部 K 線輪播邏輯保持不變...
    st.divider()
    c_idx = st.session_state.current_idx
    total_found = len(df)
    
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col1:
        if st.button("⬅️ 上一支", use_container_width=True):
            st.session_state.current_idx = (c_idx - 1) % total_found
            st.rerun()
    with btn_col2:
        st.markdown(f"<center>第 {c_idx+1} 支：<b>{df.iloc[c_idx]['code']} {df.iloc[c_idx]['name']}</b></center>", unsafe_allow_html=True)
    with btn_col3:
        if st.button("下一支 ➡️", use_container_width=True):
            st.session_state.current_idx = (c_idx + 1) % total_found
            st.rerun()
    
    current_stock = df.iloc[c_idx]
    k_fig = draw_k_line(current_stock['ticker'], current_stock['name'])
    if k_fig: st.plotly_chart(k_fig, use_container_width=True)
else:
    if not st.session_state.is_scanning: st.info("💡 請執行智慧掃描。")
