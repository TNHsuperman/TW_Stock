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
import json
import re
import time

# ============================================================
# 1. 基礎設定與初始化
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股進階選股儀表板 v4.0", layout="wide")

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer': 'https://tw.stock.yahoo.com/'
}

# ============================================================
# 2. 核心功能：本益比與營收 (JSON 抓取法)
# ============================================================

def fetch_yf_data(ticker: str) -> dict:
    """ 解決本益比 N/A：直接從網頁 JSON 數據提取 """
    code = ticker.split('.')[0]
    res = {"pe": "N/A", "mom": "N/A", "yoy": "N/A"}
    try:
        # 抓取本益比
        resp = requests.get(f"https://tw.stock.yahoo.com/quote/{code}", headers=HEADERS, timeout=10)
        # 尋找 Yahoo 隱藏的資料 JSON
        match = re.search(r'"priceToEarnings":\{"raw":([\d\.]+)', resp.text)
        if match:
            res["pe"] = match.group(1)
        
        # 抓取營收百分比
        rev_resp = requests.get(f"https://tw.stock.yahoo.com/quote/{code}/revenue", headers=HEADERS, timeout=10)
        soup = BeautifulSoup(rev_resp.text, 'html.parser')
        row = soup.select_one(r'li.List\(n\)')
        if row:
            percents = [s.get_text(strip=True) for s in row.find_all('span') if '%' in s.text]
            if len(percents) >= 2:
                res["mom"], res["yoy"] = percents[0], percents[1]
    except: pass
    return res

def fetch_dynamic_themes(ticker: str, name: str) -> str:
    """ 動態 Google 爬蟲分析熱門詞頻 (不 HardCode) """
    code = ticker.split('.')[0]
    query = f"{name} {code} 題材 概念股"
    try:
        g_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        resp = requests.get(f"https://www.google.com/search?q={query}", headers=g_headers, timeout=5)
        text = resp.text
        # 簡單的名詞篩選（2-4個中文字）
        potential = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)
        exclude = [name, code, "搜尋", "結果", "資訊", "股價", "目前", "投資", "廣告"]
        tags = [t for t in potential if t not in exclude and len(t) >= 2]
        from collections import Counter
        common = [w for w, c in Counter(tags).most_common(10) if c >= 2]
        return "、".join(common[:3])
    except: return ""

# ============================================================
# 3. 股票清單與掃描
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_list():
    """ 抓取上市櫃股票清單 """
    data = []
    urls = [('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', "TW"),
            ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', "TWO")]
    for url, market in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15, verify=False)
            df = pd.read_html(StringIO(r.text))[0]
            df.columns = df.iloc[0]
            for _, row in df.iloc[1:].iterrows():
                val = row['有價證券代號及名稱']
                if val and '　' in str(val):
                    code, name = val.split('　')
                    if len(code) == 4 and code.isdigit():
                        data.append({"ticker": f"{code}.{market}", "name": name, "industry": row['產業別'], "market": market, "code": code})
        except: continue
    return data

def fetch_yf_history(ticker: str) -> pd.DataFrame:
    p2, p1 = int(time.time()), int((datetime.now() - timedelta(days=150)).timestamp())
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}", 
                         params={"period1":p1, "period2":p2, "interval":"1d"}, headers=HEADERS, timeout=10)
        json_data = r.json()['chart']['result'][0]
        df = pd.DataFrame({
            'Date': pd.to_datetime(json_data['timestamp'], unit='s').normalize(),
            'Close': json_data['indicators']['quote'][0]['close'],
            'Volume': json_data['indicators']['quote'][0]['volume']
        }).dropna().set_index('Date')
        return df
    except: return pd.DataFrame()

# ============================================================
# 4. 側邊欄與 UI 觸發
# ============================================================

st.sidebar.header("🎯 篩選參數")
strat = st.sidebar.radio("選股策略", ["均線多頭回測", "均線糾結偵測"])
max_bias = st.sidebar.number_input("自定義乖離上限 (%)", 0.1, 10.0, 2.5, 0.1)
min_vol = st.sidebar.slider("最小成交量 (張)", 0, 3000, 500)

# 修正：按鈕改用 callback 或直接邏輯判斷
if st.button("🚀 開始全市場掃描", use_container_width=True, disabled=st.session_state.is_scanning):
    st.session_state.is_scanning = True
    st.rerun()

if st.session_state.is_scanning:
    status = st.empty()
    bar = st.progress(0)
    
    status.text("📡 正在載入證交所股票清單...")
    stocks = get_list_data = get_stock_list()
    
    results = []
    status.text(f"🔍 掃描中 (共 {len(stocks)} 支)...")
    
    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(fetch_yf_history, s['ticker']): s for s in stocks}
        for i, f in enumerate(as_completed(futures), 1):
            if i % 100 == 0: bar.progress(i / len(stocks))
            s = futures[f]
            df = f.result()
            if len(df) > 60:
                # 成交量過濾
                if (df['Volume'].tail(5).mean() / 1000) >= min_vol:
                    c = df['Close'].iloc[-1]
                    ma30 = df['Close'].rolling(30).mean().iloc[-1]
                    bias = ((c - ma30) / ma30) * 100
                    
                    # 策略過濾：使用使用者輸入的 max_bias
                    if strat == "均線多頭回測" and c > ma30 and 0 < bias <= max_bias:
                        results.append({**s, "收盤": round(c, 2), "乖離(%)": round(bias, 2)})
                    elif strat == "均線糾結偵測" and abs(bias) <= max_bias:
                        results.append({**s, "收盤": round(c, 2), "乖離(%)": round(bias, 2)})
    
    if results:
        status.text(f"📊 找到 {len(results)} 支符合初步條件，正在深度分析基本面與題材...")
        final_data = []
        with ThreadPoolExecutor(max_workers=3) as ex: # 深度查詢不能太快
            # 傳遞 results 進去
            enrich_futures = {ex.submit(lambda r: {**r, **fetch_yf_data(r['ticker']), "熱門題材": fetch_dynamic_themes(r['ticker'], r['name'])} , r): r for r in results}
            for j, f in enumerate(as_completed(enrich_futures), 1):
                status.text(f"深度分析中: {j} / {len(results)}")
                final_data.append(f.result())
        
        # 整理最終表格
        res_df = pd.DataFrame(final_data)
        res_df = res_df.rename(columns={"pe": "本益比", "mom": "營收月增", "yoy": "營收年增"})
        st.session_state.scan_results = res_df
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.warning("查無符合條件之標的。")

    st.session_state.is_scanning = False
    st.rerun()

# ============================================================
# 5. 結果呈現
# ============================================================
if not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    # 再次確認乖離率過濾（確保動態修改有效）
    display_df = df[df['乖離(%)'] <= max_bias].copy()
    
    st.subheader(f"✅ 掃描完成！找到 {len(display_df)} 支標的")
    cols = ["code", "name", "market", "industry", "收盤", "乖離(%)", "本益比", "營收月增", "營收年增", "熱門題材"]
    st.dataframe(display_df[cols].rename(columns={"code":"代碼","name":"名稱","market":"市場","industry":"類股"}), 
                 use_container_width=True, hide_index=True)
else:
    if not st.session_state.is_scanning:
        st.info("💡 準備就緒，請點擊按鈕開始掃描。")
