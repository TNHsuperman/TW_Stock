import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import urllib3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
from bs4 import BeautifulSoup
import re

# ============================================================
# 1. 基礎設定
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板 v2.0", layout="wide")

# 初始化 session state
for key in ['scan_results', 'is_scanning', 'selected_index', 'yf_session']:
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame() if key == 'scan_results' else (False if key == 'is_scanning' else 0)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7'
}

# ============================================================
# 2. 核心抓取函數 (修正 N/A 關鍵)
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_info_map():
    stock_info_map = {}
    stocks_list = []
    urls = [
        ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
        ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")
    ]
    for url, market in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15, verify=False)
            df_list = pd.read_html(StringIO(resp.text), flavor='lxml')
            df = df_list[0]
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            for _, row in df.iterrows():
                val = row['有價證券代號及名稱']
                industry = row['產業別']
                if val and '　' in str(val):
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        suffix = ".TW" if market == "TW" else ".TWO"
                        ticker = f"{code}{suffix}"
                        stocks_list.append((ticker, market))
                        stock_info_map[ticker] = {
                            "name": name, "code": code, "market": market,
                            "industry": industry if pd.notna(industry) else "其他"
                        }
        except: continue
    return stocks_list, stock_info_map

def fetch_yf_pe(ticker: str) -> str:
    """ 修正版：從 Yahoo 奇摩股市 HTML 結構精準抓取本益比 """
    code = ticker.split('.')[0]
    url = f"https://tw.stock.yahoo.com/quote/{code}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 尋找「本益比」標籤後的數值
        target = soup.find('span', string="本益比")
        if target:
            val = target.find_next_sibling('span')
            if val: return val.get_text(strip=True)
        # 備用方案：搜尋特定元件
        main_info = soup.select_one('main li:has(span:-soup-contains("本益比")) span:nth-child(2)')
        return main_info.get_text(strip=True) if main_info else "N/A"
    except: return "N/A"

def fetch_revenue_growth(ticker: str) -> tuple[str, str]:
    """ 修正版：直接從 Yahoo 營收頁面抓取 MoM/YoY """
    code = ticker.split('.')[0]
    url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 抓取表格中最新一列
        row = soup.select_one('li.List\(n\)') 
        if row:
            # 根據目前 Yahoo 結構，月增與年增通常在最後幾個 span
            spans = row.find_all('span')
            if len(spans) >= 5:
                mom = spans[3].get_text(strip=True) # 月增
                yoy = spans[4].get_text(strip=True) # 年增
                return mom, yoy
    except: pass
    return "N/A", "N/A"

# ============================================================
# 3. Yahoo Finance 歷史資料與分析
# ============================================================

def fetch_yf_history(ticker: str, days: int = 130) -> pd.DataFrame:
    p2 = int(datetime.now().timestamp())
    p1 = int((datetime.now() - timedelta(days=days)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period1": p1, "period2": p2, "interval": "1d", "events": "history"}
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        data = resp.json()
        result = data['chart']['result'][0]
        df = pd.DataFrame({
            'Date': pd.to_datetime(result['timestamp'], unit='s').normalize(),
            'Open': result['indicators']['quote'][0]['open'],
            'High': result['indicators']['quote'][0]['high'],
            'Low': result['indicators']['quote'][0]['low'],
            'Close': result['indicators']['quote'][0]['close'],
            'Volume': result['indicators']['quote'][0]['volume'],
        }).dropna(subset=['Close']).set_index('Date')
        return df
    except: return pd.DataFrame()

def analyze(df: pd.DataFrame, strategy: str, min_vol: float) -> dict | None:
    if len(df) < 60: return None
    if (df['Volume'].tail(5).mean() / 1000) < min_vol: return None
    
    close = df['Close'].iloc[-1]
    ma30, ma45, ma60 = df['Close'].rolling(30).mean().iloc[-1], df['Close'].rolling(45).mean().iloc[-1], df['Close'].rolling(60).mean().iloc[-1]
    if any(np.isnan(v) for v in [close, ma30, ma45, ma60]): return None
    
    bias_30 = ((close - ma30) / ma30) * 100
    if strategy == "均線多頭回測":
        if ma30 > ma45 > ma60 and close > ma30 and bias_30 < 2.5:
            return {"close": close, "bias_30": bias_30}
    elif strategy == "均線糾結偵測":
        spread = (max(ma30, ma45, ma60) - min(ma30, ma45, ma60)) / min(ma30, ma45, ma60)
        if spread <= 0.025 and abs(bias_30) < 2.5:
            return {"close": close, "bias_30": bias_30}
    return None

# ============================================================
# 4. AI 題材分析 (Claude)
# ============================================================

def fetch_hot_themes(ticker: str, stock_name: str) -> str:
    try:
        # 1. 抓取新聞標題
        news_url = "https://query1.finance.yahoo.com/v1/finance/search"
        resp = requests.get(news_url, params={"q": ticker, "newsCount": 5}, headers=HEADERS, timeout=8)
        news = resp.json().get('news', [])
        titles = "\n".join([n.get('title', '') for n in news])
        if not titles: return ""

        # 2. 調用 Claude (請確保 st.secrets["CLAUDE_API_KEY"] 已設定)
        api_key = st.secrets.get("CLAUDE_API_KEY", "")
        if not api_key: return ""

        payload = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": f"分析「{stock_name}」近期新聞題材，僅回傳標籤如「AI伺服器、散熱」，無則回傳空字串。新聞：\n{titles}"}]
        }
        res = requests.post("https://api.anthropic.com/v1/messages", 
                            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                            json=payload, timeout=15)
        return res.json()['content'][0]['text'].strip()
    except: return ""

# ============================================================
# 5. 並行處理與 UI 邏輯
# ============================================================

def enrich_one(row_data: dict) -> dict:
    ticker, name = row_data['_ticker'], row_data['名稱']
    pe = fetch_yf_pe(ticker)
    mom, yoy = fetch_revenue_growth(ticker)
    theme = fetch_hot_themes(ticker, name)
    return {"_ticker": ticker, "本益比": pe, "營收月增": mom, "營收年增": yoy, "熱門題材": theme}

def enrich_results(res_df: pd.DataFrame, status_text) -> pd.DataFrame:
    total = len(res_df)
    rows_input = res_df.to_dict('records')
    enrich_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor: # 降低 worker 避免被封鎖
        futures = {executor.submit(enrich_one, row): row['_ticker'] for row in rows_input}
        for i, future in enumerate(as_completed(futures), 1):
            status_text.text(f"📊 補充基本面：{i}/{total} 完成")
            res = future.result()
            enrich_map[res['_ticker']] = res
    
    for col in ["本益比", "營收月增", "營收年增", "熱門題材"]:
        res_df[col] = res_df['_ticker'].map(lambda t: enrich_map.get(t, {}).get(col, "N/A"))
    return res_df

# ============================================================
# 6. Streamlit 介面佈局 (保持原有邏輯並微調)
# ============================================================

st.sidebar.header("⚙️ 策略參數設定")
strategy_option = st.sidebar.radio("選擇策略：", ("均線多頭回測", "均線糾結偵測"))
min_volume = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)
indicator_choice = st.sidebar.selectbox("輔助指標：", ["都不顯示", "RSI (強弱指標)", "MACD (趨勢指標)"])

if not st.session_state['is_scanning']:
    if st.button("🔍 開始全市場掃描", use_container_width=True):
        st.session_state['is_scanning'] = True
        st.rerun()
else:
    all_stocks, info_map = get_stock_info_map()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(fetch_yf_history, t): (t, m) for t, m in all_stocks}
        for i, future in enumerate(as_completed(futures), 1):
            t, m = futures[future]
            if i % 50 == 0: progress_bar.progress(i / len(all_stocks))
            df = future.result()
            if not df.empty:
                res = analyze(df, strategy_option, min_volume)
                if res:
                    info = info_map.get(t, {"name": "未知", "industry": "其他", "code": t})
                    results.append({
                        "代碼": info["code"], "名稱": info["name"], "市場": m, "類股": info["industry"],
                        "收盤": round(res["close"], 2), "乖離(%)": round(res["bias_30"], 2), "_ticker": t
                    })
    
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = enrich_results(res_df, status_text)
    st.session_state['scan_results'] = res_df
    st.session_state['is_scanning'] = False
    st.rerun()

# 結果顯示表格
if not st.session_state['scan_results'].empty:
    df_disp = st.session_state['scan_results']
    st.write(f"🎉 找到 {len(df_disp)} 支符合標的")
    event = st.dataframe(df_disp.drop(columns=['_ticker']), hide_index=True, use_container_width=True, 
                         on_select="rerun", selection_mode="single-row")
    
    # 這裡放 Plotly 繪圖邏輯 (與你原版一致即可)...
    if event.selection and event.selection.rows:
        sel_idx = event.selection.rows[0]
        sel = df_disp.iloc[sel_idx]
        st.subheader(f"📈 {sel['名稱']} ({sel['代碼']}) 技術圖表")
        # [此處插入原有的 Plotly 繪圖代碼]
