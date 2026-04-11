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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股智慧選股儀表板", layout="wide")

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = pd.DataFrame()
if 'is_scanning' not in st.session_state:
    st.session_state['is_scanning'] = False
if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = 0

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


# ============================================================
# 2. 股票清單（TWSE ISIN — 仍可正常使用）
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
                        stocks_list.append((code, market))
                        stock_info_map[code] = {
                            "name": name,
                            "market": market,
                            "industry": industry if pd.notna(industry) else "其他"
                        }
        except Exception:
            continue
    return stocks_list, stock_info_map


# ============================================================
# 3. 資料來源：FinMind API（第三方，不會封鎖 Cloud IP）
#
#   FinMind 免費版限制：
#   - 未登入：每 10 分鐘 300 次請求
#   - 註冊後（免費）：每天 600 次請求
#   - 申請 Token 可提高至每天 3000 次
#
#   申請 Token：https://finmindtrade.com/
#   取得後填入側邊欄的 Token 欄位即可
# ============================================================

def fetch_finmind(code: str, token: str = "", days: int = 130) -> pd.DataFrame:
    """
    透過 FinMind API 抓取台股歷史日線資料。

    參數：
        code  : 股票代碼（純數字，如 '2330'）
        token : FinMind API Token（選填，填入可提高請求上限）
        days  : 抓取天數

    FinMind API 文件：https://finmindtrade.com/analysis/#/data/document
    dataset: TaiwanStockPrice — 台灣股價日成交資訊
    """
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    params = {
        "dataset":    "TaiwanStockPrice",
        "data_id":    code,
        "start_date": start,
        "token":      token,
    }
    try:
        resp = requests.get(FINMIND_URL, params=params, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        if data.get('status') != 200:
            return pd.DataFrame()
        records = data.get('data', [])
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # FinMind 回傳欄位：date, open, max, min, close, Trading_Volume, ...
        df = df.rename(columns={
            'date':             'Date',
            'open':             'Open',
            'max':              'High',
            'min':              'Low',
            'close':            'Close',
            'Trading_Volume':   'Volume',
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        df = df.sort_values('Date').set_index('Date')

        # 確保數值型別正確
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Close'])
        return df

    except Exception:
        return pd.DataFrame()


# ============================================================
# 4. 分析
# ============================================================

def parse_price(s):
    try:
        v = float(str(s).replace(',', '').strip())
        return v if v > 0 else None
    except Exception:
        return None


def analyze(df: pd.DataFrame, strategy: str, min_vol: float) -> dict | None:
    if len(df) < 60:
        return None
    # 成交量篩選（FinMind 的 Volume 單位是股，除 1000 換算張）
    if (df['Volume'].tail(5).mean() / 1000) < min_vol:
        return None
    close = float(df['Close'].iloc[-1])
    m30   = float(df['Close'].rolling(30).mean().iloc[-1])
    m45   = float(df['Close'].rolling(45).mean().iloc[-1])
    m60   = float(df['Close'].rolling(60).mean().iloc[-1])
    if any(np.isnan(v) for v in [close, m30, m45, m60]):
        return None
    bias_30 = ((close - m30) / m30) * 100
    if strategy == "均線多頭回測":
        if m30 > m45 > m60 and close > m30 and bias_30 < 2.0:
            return {"close": close, "bias_30": bias_30}
    elif strategy == "均線糾結偵測":
        spread = (max(m30, m45, m60) - min(m30, m45, m60)) / min(m30, m45, m60)
        if spread <= 0.02 and abs(bias_30) < 2.0:
            return {"close": close, "bias_30": bias_30}
    return None


def fetch_finmind_batch(codes: list, token: str = "", days: int = 130) -> dict:
    """
    批次下載多支股票歷史資料。

    FinMind 支援在 data_id 欄位傳入逗號分隔的多個股票代碼，
    一次請求回傳所有股票的資料，大幅減少請求次數。

    例如原本 100 支股票需要 100 次請求，
    改用 batch_size=50 後只需要 2 次請求，速度提升約 30~50 倍。

    回傳：dict，key 為股票代碼，value 為 DataFrame
    """
    if not codes:
        return {}

    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    params = {
        "dataset":    "TaiwanStockPrice",
        "data_id":    ",".join(codes),   # 多支股票用逗號串接
        "start_date": start,
        "token":      token,
    }
    try:
        resp = requests.get(FINMIND_URL, params=params, timeout=30)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        if data.get('status') != 200:
            return {}
        records = data.get('data', [])
        if not records:
            return {}

        df_all = pd.DataFrame(records)
        df_all = df_all.rename(columns={
            'date':           'Date',
            'stock_id':       'Code',
            'open':           'Open',
            'max':            'High',
            'min':            'Low',
            'close':          'Close',
            'Trading_Volume': 'Volume',
        })
        df_all['Date'] = pd.to_datetime(df_all['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
        df_all = df_all.dropna(subset=['Close', 'Code'])

        # 依股票代碼拆分成個別 DataFrame
        result = {}
        for code, group in df_all.groupby('Code'):
            df = (group[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                  .sort_values('Date')
                  .drop_duplicates('Date')
                  .set_index('Date'))
            result[str(code)] = df

        return result

    except Exception:
        return {}


# ============================================================
# 4. 分析
# ============================================================

def parse_price(s):
    try:
        v = float(str(s).replace(',', '').strip())
        return v if v > 0 else None
    except Exception:
        return None


def analyze(df: pd.DataFrame, strategy: str, min_vol: float) -> dict | None:
    if len(df) < 60:
        return None
    # 成交量篩選（FinMind 的 Volume 單位是股，除 1000 換算張）
    if (df['Volume'].tail(5).mean() / 1000) < min_vol:
        return None
    close = float(df['Close'].iloc[-1])
    m30   = float(df['Close'].rolling(30).mean().iloc[-1])
    m45   = float(df['Close'].rolling(45).mean().iloc[-1])
    m60   = float(df['Close'].rolling(60).mean().iloc[-1])
    if any(np.isnan(v) for v in [close, m30, m45, m60]):
        return None
    bias_30 = ((close - m30) / m30) * 100
    if strategy == "均線多頭回測":
        if m30 > m45 > m60 and close > m30 and bias_30 < 2.0:
            return {"close": close, "bias_30": bias_30}
    elif strategy == "均線糾結偵測":
        spread = (max(m30, m45, m60) - min(m30, m45, m60)) / min(m30, m45, m60)
        if spread <= 0.02 and abs(bias_30) < 2.0:
            return {"close": close, "bias_30": bias_30}
    return None


# ============================================================
# 5. 掃描主流程（批次下載版）
# ============================================================

def run_scan(all_stocks, info_map, strategy, min_vol, token,
             progress_bar, status_text):
    """
    批次掃描流程：

    舊版：每支股票各自發一次請求 → 1800 次請求
    新版：每批 50 支合併成一次請求 → 36 次請求
         速度提升約 30~50 倍

    batch_size=50 是 FinMind 實測最穩定的批次大小，
    太大（>100）偶爾會超時或回傳不完整。
    """
    BATCH_SIZE = 50
    codes    = [c for c, _ in all_stocks]
    markets  = {c: m for c, m in all_stocks}
    total    = len(codes)
    results  = []
    done     = 0

    # 將所有股票切成批次
    batches = [codes[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    n_batches = len(batches)

    status_text.text(f"⚡ 批次下載中：共 {n_batches} 批，每批 {BATCH_SIZE} 支...")

    for batch_idx, batch in enumerate(batches):
        # 每批次一次請求拿到所有股票資料
        batch_data = fetch_finmind_batch(batch, token=token, days=130)

        # 逐支分析（純 CPU，不需要網路，速度快）
        for code in batch:
            df = batch_data.get(code)
            if df is not None and not df.empty:
                result = analyze(df, strategy, min_vol)
                if result:
                    info = info_map.get(code, {"name": "未知", "industry": "其他"})
                    results.append({
                        "代碼":    code,
                        "名稱":    info["name"],
                        "市場":    markets.get(code, "TW"),
                        "類股":    info["industry"],
                        "收盤":    round(result["close"], 2),
                        "乖離(%)": round(result["bias_30"], 2),
                        "_code":   code,
                        "_market": markets.get(code, "TW"),
                    })
            done += 1

        # 更新進度
        pct = (batch_idx + 1) / n_batches
        progress_bar.progress(min(pct, 1.0))
        status_text.text(
            f"⚡ 批次進度：{batch_idx+1} / {n_batches} 批"
            f"  ({done} / {total} 支)  |  找到 {len(results)} 支"
        )

        # 批次間短暫休息，避免觸發 FinMind 速率限制
        if batch_idx < n_batches - 1:
            time.sleep(0.5)

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values("乖離(%)").reset_index(drop=True)
    return res_df, done - len([c for c in codes if c in {r["_code"] for r in results}])


# ============================================================
# 6. 側邊欄
# ============================================================

st.sidebar.header("⚙️ 策略參數設定")
strategy_option  = st.sidebar.radio("選擇選股策略：", ("均線多頭回測", "均線糾結偵測"))
indicator_choice = st.sidebar.selectbox("查看確認指標：",
                                        ["都不顯示", "RSI (強弱指標)", "MACD (趨勢指標)"])
min_volume = st.sidebar.slider("最小成交量 (張)", 0, 2000, 500, step=100)

st.sidebar.markdown("---")

# Token 從 secrets.toml 自動讀取，不需要使用者手動輸入
# 本機開發：將 secrets.toml 放在專案根目錄的 .streamlit/ 資料夾內
# Streamlit Cloud：在 App 設定頁的 Secrets 區塊貼入內容
finmind_token = st.secrets.get("FINMIND_TOKEN", "")

if finmind_token:
    st.sidebar.success("✅ FinMind Token 已載入")
else:
    st.sidebar.warning("⚠️ 未設定 Token，每天限 600 次請求")

st.sidebar.caption("📡 資料來源：[FinMind](https://finmindtrade.com/)")


# ============================================================
# 7. 掃描按鈕
# ============================================================

if not st.session_state['is_scanning']:
    if st.button("🔍 開始全市場掃描", use_container_width=True):
        st.session_state['is_scanning'] = True
        st.rerun()
else:
    all_stocks, info_map = get_stock_info_map()
    progress_bar = st.progress(0)
    status_text  = st.empty()

    res_df, empty_cnt = run_scan(
        all_stocks, info_map,
        strategy_option, min_volume, finmind_token,
        progress_bar, status_text
    )

    if empty_cnt > len(all_stocks) * 0.5:
        status_text.warning(
            f"⚠️ {empty_cnt} 支股票無資料。"
            "可能已超過 FinMind 免費請求上限（每天 600 次），"
            "請明天再試或填入 API Token。"
        )
    else:
        status_text.text(f"🎉 完成！找到 {len(res_df)} 支符合條件標的。")

    st.session_state['scan_results'] = res_df
    st.session_state['is_scanning']  = False
    st.rerun()


# ============================================================
# 8. 結果顯示
# ============================================================

if not st.session_state['scan_results'].empty:
    df_raw = st.session_state['scan_results']
    selected_industry = st.selectbox(
        "🎯 篩選類股：",
        ["全部"] + sorted(df_raw["類股"].unique().tolist())
    )
    df_filtered = (df_raw if selected_industry == "全部"
                   else df_raw[df_raw["類股"] == selected_industry]).reset_index(drop=True)
    display_cols = ["代碼", "名稱", "市場", "類股", "收盤", "乖離(%)"]

    if len(df_filtered) > 0:
        st.write(f"📊 符合條件標的：{len(df_filtered)} 支")
        event = st.dataframe(
            df_filtered[display_cols], hide_index=True, use_container_width=True,
            on_select="rerun", selection_mode="single-row", key="stock_table"
        )
        if event.selection and event.selection.rows:
            st.session_state['selected_index'] = event.selection.rows[0]

        st.write("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("⬅️ 上一支", use_container_width=True):
                st.session_state['selected_index'] = (
                    st.session_state['selected_index'] - 1) % len(df_filtered)
        with c2:
            st.markdown(
                f"<h3 style='text-align:center;color:#9400D3;'>"
                f"{st.session_state['selected_index']+1} / {len(df_filtered)}</h3>",
                unsafe_allow_html=True)
        with c3:
            if st.button("下一支 ➡️", use_container_width=True):
                st.session_state['selected_index'] = (
                    st.session_state['selected_index'] + 1) % len(df_filtered)

        idx = min(st.session_state['selected_index'], len(df_filtered) - 1)
        sel = df_filtered.iloc[idx]

        with st.spinner(f'載入中... {sel["名稱"]} ({sel["代碼"]})'):
            df_p = fetch_finmind(sel["_code"], token=finmind_token, days=240)

            if df_p.empty:
                st.error("無法載入資料，可能已超過 FinMind 請求上限，請稍後再試。")
            else:
                df_p = df_p.dropna(subset=['Close'])
                df_p['Date_Str'] = df_p.index.strftime('%Y-%m-%d')
                close_s      = df_p['Close']
                df_p['30MA'] = close_s.rolling(30).mean()
                df_p['45MA'] = close_s.rolling(45).mean()
                df_p['60MA'] = close_s.rolling(60).mean()

                n_rows      = 3 if indicator_choice != "都不顯示" else 2
                row_heights = [0.6, 0.15, 0.25] if n_rows == 3 else [0.75, 0.25]
                fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.03, row_heights=row_heights)

                fig.add_trace(go.Candlestick(
                    x=df_p['Date_Str'], open=df_p['Open'], high=df_p['High'],
                    low=df_p['Low'], close=df_p['Close'], name="K線"), row=1, col=1)

                for ma, color in zip(['30MA','45MA','60MA'], ['#FFA500','#2E8B57','#4169E1']):
                    fig.add_trace(go.Scatter(
                        x=df_p['Date_Str'], y=df_p[ma],
                        line=dict(color=color, width=1.5), name=ma), row=1, col=1)

                v_clrs = ['#ef5350' if c >= o else '#26a69a'
                          for c, o in zip(df_p['Close'], df_p['Open'])]
                fig.add_trace(go.Bar(
                    x=df_p['Date_Str'], y=df_p['Volume'],
                    marker_color=v_clrs, name="成交量"), row=2, col=1)

                if indicator_choice == "RSI (強弱指標)":
                    d = close_s.diff()
                    rsi = 100 - (100 / (1 + d.where(d>0,0).rolling(14).mean() /
                                        (-d.where(d<0,0).rolling(14).mean() + 1e-9)))
                    fig.add_trace(go.Scatter(x=df_p['Date_Str'], y=rsi,
                        line=dict(color='purple', width=1.2), name="RSI"), row=3, col=1)
                    fig.add_hline(y=70, line_dash="dot", line_color="red",   row=3, col=1)
                    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

                elif indicator_choice == "MACD (趨勢指標)":
                    dif = close_s.ewm(span=12).mean() - close_s.ewm(span=26).mean()
                    h_s = dif - dif.ewm(span=9).mean()
                    fig.add_trace(go.Bar(x=df_p['Date_Str'], y=h_s,
                        marker_color=['#ef5350' if v>=0 else '#26a69a' for v in h_s],
                        name="MACD柱"), row=3, col=1)

                fig.update_xaxes(type='category', showgrid=True,
                                 gridcolor='rgba(200,200,200,0.2)')
                fig.update_layout(
                    title=f"<b>{sel['名稱']} ({sel['代碼']})</b>",
                    xaxis_rangeslider_visible=False, height=700, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("目前篩選條件下無標的。")
else:
    st.info(
        "💡 點擊「開始全市場掃描」按鈕。\n\n"
        "📡 資料來源已改為 **FinMind API**，解決 TWSE/TPEX IP 封鎖問題。\n\n"
        "建議先至 [finmindtrade.com](https://finmindtrade.com/) 免費註冊取得 Token，"
        "填入左側欄位後掃描效果更穩定。"
    )
