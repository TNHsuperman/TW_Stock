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
from bs4 import BeautifulSoup
import re
import time

# ============================================================
# 1. 基礎設定與 Header
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板 v3.0", layout="wide")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer': 'https://tw.stock.yahoo.com/'
}

# 熱門題材關鍵字資料庫（用於 Google 爬蟲比對）
THEME_KEYWORDS = [
    "AI", "伺服器", "散熱", "半導體", "台積電供應鏈", "低軌衛星", "重電", "儲能", "綠能", 
    "電動車", "CPO", "矽光子", "機器人", "軍工", "摺疊機", "Wi-Fi 7", "生技", "水資源",
    "面板級封裝", "GB200", "ASIC", "車用電子", "光通訊"
]

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False

# ============================================================
# 2. 核心抓取邏輯 (修正 N/A 與 營收單位問題)
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
    """ 修正版：使用文字匹配避開動態 Class 變動 """
    code = ticker.split('.')[0]
    url = f"https://tw.stock.yahoo.com/quote/{code}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=7)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 尋找包含「本益比」字樣的標籤
        label = soup.find('span', string=re.compile("本益比"))
        if label:
            val = label.find_next_sibling('span')
            if val: return val.get_text(strip=True)
        # 備用方案：解析所有 li
        for li in soup.find_all('li'):
            if "本益比" in li.get_text():
                spans = li.find_all('span')
                if len(spans) >= 2: return spans[-1].get_text(strip=True)
    except: pass
    return "N/A"

def fetch_revenue_growth(ticker: str) -> tuple[str, str]:
    """ 修正版：精準抓取帶有 % 的月增與年增欄位 """
    code = ticker.split('.')[0]
    url = f"https://tw.stock.yahoo.com/quote/{code}/revenue"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=7)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 找到包含數據的列表列 (使用原始字串避開警告)
        row = soup.select_one(r'li.List\(n\)')
        if row:
            spans = row.find_all('span')
            # 篩選出純數字帶 % 的字串
            percents = [s.get_text(strip=True) for s in spans if '%' in s.get_text()]
            if len(percents) >= 2:
                return percents[0], percents[1] # [月增%, 年增%]
    except: pass
    return "N/A", "N/A"

def fetch_google_themes(ticker: str, name: str) -> str:
    """ Google 搜尋爬蟲：查詢熱門題材 """
    code = ticker.split('.')[0]
    query = f"{name} {code} 題材 概念股"
    search_url = f"https://www.google.com/search?q={query}"
    try:
        # 使用獨立 Header 避免 Google 擋爬蟲
        g_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        resp = requests.get(search_url, headers=g_headers, timeout=5)
        found = [kw for kw in THEME_KEYWORDS if kw in resp.text]
        if found:
            return "、".join(list(dict.fromkeys(found))[:3])
    except: pass
    return ""

# ============================================================
# 3. 分析與並行處理
# ============================================================

def fetch_yf_history(ticker: str, days: int = 130) -> pd.DataFrame:
    p2, p1 = int(time.time()), int((datetime.now() - timedelta(days=days)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    try:
        resp = requests.get(url, params={"period1":p1, "period2":p2, "interval":"1d"}, headers=HEADERS, timeout=10)
        r = resp.json()['chart']['result'][0]
        df = pd.DataFrame({
            'Date': pd.to_datetime(r['timestamp'], unit='s').normalize(),
            'Open': r['indicators']['quote'][0]['open'],
            'High': r['indicators']['quote'][0]['high'],
            'Low': r['indicators']['quote'][0]['low'],
            'Close': r['indicators']['quote'][0]['close'],
            'Volume': r['indicators']['quote'][0]['volume'],
        }).dropna(subset=['Close']).set_index('Date')
        return df
    except: return pd.DataFrame()

def analyze(df: pd.DataFrame, strategy: str, min_vol: float) -> dict | None:
    if len(df) < 60: return None
    if (df['Volume'].tail(5).mean() / 1000) < min_vol: return None
    c = df['Close'].iloc[-1]
    m30 = df['Close'].rolling(30).mean().iloc[-1]
    m60 = df['Close'].rolling(60).mean().iloc[-1]
    bias = ((c - m30) / m30) * 100
    if strategy == "均線多頭回測" and c > m30 > m60 and bias < 2.5:
        return {"close": c, "bias": bias}
    if strategy == "均線糾結偵測":
        spread = (max(m30, m60) - min(m30, m60)) / min(m30, m60)
        if spread < 0.02 and abs(bias) < 2.5: return {"close": c, "bias": bias}
    return None

def enrich_one(row_data: dict) -> dict:
    t, n = row_data['_ticker'], row_data['名稱']
    time.sleep(np.random.uniform(0.5, 1.2)) # 禮貌延遲
    pe = fetch_yf_pe(t)
    mom, yoy = fetch_revenue_growth(t)
    theme = fetch_google_themes(t, n)
    return {"_ticker": t, "本益比": pe, "營收月增": mom, "營收年增": yoy, "熱門題材": theme}

# ============================================================
# 4. Streamlit UI
# ============================================================

st.sidebar.header("⚙️ 策略設定")
strat = st.sidebar.radio("策略：", ["均線多頭回測", "均線糾結偵測"])
vol_limit = st.sidebar.slider("成交量(張):", 0, 2000, 500)

if not st.session_state['is_scanning']:
    if st.button("🔍 開始掃描全市場", use_container_width=True):
        st.session_state['is_scanning'] = True
        st.rerun()
else:
    stocks, info_map = get_stock_info_map()
    bar = st.progress(0)
    status = st.empty()
    results = []
    
    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(fetch_yf_history, t): (t, m) for t, m in stocks}
        for i, f in enumerate(as_completed(futures), 1):
            if i % 100 == 0: bar.progress(i/len(stocks))
            t, m = futures[f]
            df = f.result()
            if not df.empty:
                res = analyze(df, strat, vol_limit)
                if res:
                    inf = info_map.get(t, {"name":"?", "industry":"?", "code":t})
                    results.append({"代碼":inf["code"],"名稱":inf["name"],"市場":m,"類股":inf["industry"],"收盤":res["close"],"乖離(%)":round(res["bias"],2),"_ticker":t})

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        status.text(f"📊 正在深度分析 {len(res_df)} 支標的基本面與題材...")
        enriched = []
        with ThreadPoolExecutor(max_workers=3) as ex: # 查詢題材不宜過快
            f_enrich = [ex.submit(enrich_one, r) for r in res_df.to_dict('records')]
            for f in as_completed(f_enrich):
                enriched.append(f.result())
        e_df = pd.DataFrame(enriched)
        res_df = res_df.merge(e_df, on='_ticker')

    st.session_state['scan_results'] = res_df
    st.session_state['is_scanning'] = False
    st.rerun()

# 顯示結果
if not st.session_state['scan_results'].empty:
    df = st.session_state['scan_results']
    st.success(f"✅ 掃描完成，找到 {len(df)} 支標的")
    st.dataframe(df.drop(columns=['_ticker']), use_container_width=True, hide_index=True)
else:
    st.info("💡 點擊上方按鈕開始掃描。")
