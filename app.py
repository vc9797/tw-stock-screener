import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="台股實戰策略篩選器", layout="wide")

# 初始化 Session State 用於追蹤 API 錯誤
if "api_last_error" not in st.session_state:
    st.session_state["api_last_error"] = None

# ==========================================
# 0. FinMind API 數據下載核心函數
# ==========================================
def fetch_finmind_data(dataset, stock_id, start_date, token=""):
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": start_date,
    }
    if token:
        parameter["token"] = token
    
    try:
        resp = requests.get(url, params=parameter, timeout=10)
        if resp.status_code == 200:
            resp_json = resp.json()
            if resp_json.get("msg") == "success":
                return pd.DataFrame(resp_json["data"])
            else:
                st.session_state["api_last_error"] = f"API 回傳限制: {resp_json.get('msg', '未知錯誤')}"
        elif resp.status_code == 429:
            st.session_state["api_last_error"] = "⚠️ 觸發 API 每小時限制 (429 Too Many Requests)！請於左側輸入免費申請的 FinMind Token。"
    except Exception as e:
        st.session_state["api_last_error"] = f"網路請求失敗: {str(e)}"
    return pd.DataFrame()

# ==========================================
# 1. 核心選股邏輯演算法（修正 ROE 計算與防錯機制）
# ==========================================
def scan_single_stock(stock_id, token, settings):
    today_str = datetime.today().strftime('%Y-%m-%d')
    start_3m = (datetime.today() - timedelta(days=120)).strftime('%Y-%m-%d')
    start_1y = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # 預設變數初始化
    recent_3m_growth = 0.0
    latest_roe = 0.0
    sitc_10d = 0
    consecutive_sell = 0
    latest_pe = 20.0
    
    # --- A. 技術面數據 (日K) ---
    df_price = fetch_finmind_data("TaiwanStockPrice", stock_id, start_3m, token)
    if df_price.empty or 'close' not in df_price.columns or len(df_price) < 20:
        return None
    
    latest_price = df_price['close'].iloc[-1]
    
    # 條件: 股價剛突破整理平台（排除已暴漲股）
    price_60d = df_price['close'].tail(60)
    min_60d = price_60d.min()
    max_gain_60d = ((price_60d.max() - min_60d) / min_60d) * 100 if min_60d > 0 else 0.0
    
    # 如果過濾開關開啟，才進行最高漲幅過濾
    if settings['filter_max_gain'] and max_gain_60d > settings['max_60d_gain']:
        return None
        
    price_20d = df_price['close'].tail(20)
    is_breakout = latest_price >= price_20d.max() * 0.98  # 容許 2% 誤差
    if settings['filter_breakout'] and not is_breakout:
        return None

    # --- B. 籌碼面數據 (法人買賣超) ---
    df_chip = fetch_finmind_data("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_3m, token)
    if not df_chip.empty and all(col in df_chip.columns for col in ['name', 'buy', 'sell']):
        # 投信近10日買賣超
        sitc_data = df_chip[df_chip['name'] == 'Investment_Trust'].tail(10)
        sitc_10d = sitc_data['buy'].sum() - sitc_data['sell'].sum()
        if sitc_10d < settings['min_sitc_buy']:
            return None
            
        # 外資連續大賣天數
        foreign_data = df_chip[df_chip['name'] == 'Foreign_Investor'].tail(10)
        foreign_net = foreign_data['buy'] - foreign_data['sell']
        for val in reversed(foreign_net.values):
            if val < -1000:  # 單日大賣超 1000 張以上算大賣
                consecutive_sell += 1
            else:
                break
        if settings['filter_foreign_sell'] and consecutive_sell > settings['max_foreign_sell_days']:
            return None
    else:
        # 籌碼數據缺失時，若設定了大於0的硬性指標則排除
        if settings['min_sitc_buy'] > 0:
            return None

    # --- C. 估值數據 (本益比) ---
    df_per = fetch_finmind_data("TaiwanStockPER", stock_id, start_3m, token)
    if not df_per.empty and 'PER' in df_per.columns:
        latest_pe = df_per['PER'].iloc[-1]

    # --- D. 基本面數據 (營收) ---
    df_rev = fetch_finmind_data("TaiwanStockMonthRevenue", stock_id, start_1y, token)
    if not df_rev.empty and 'revenue_year_growth' in df_rev.columns and len(df_rev) >= 3:
        recent_3m_growth = df_rev['revenue_year_growth'].tail(3).mean()
        if recent_3m_growth < settings['min_rev_yoy']:
            return None
    else:
        return None  # 營收為基本門檻，缺失直接排除

    # --- E. 財務報表數據 (EPS / ROE) ---
    df_finance = fetch_finmind_data("TaiwanStockFinancialStatements", stock_id, start_1y, token)
    if not df_finance.empty and all(col in df_finance.columns for col in ['type', 'value', 'date']):
        # 1. EPS 成長檢查
        df_eps = df_finance[df_finance['type'] == 'EPS'].sort_values('date')
        if len(df_eps) >= 4:
            eps_values = df_eps['value'].tail(4).values
            is_eps_growing = eps_values[-1] > eps_values[-4] if settings['filter_eps'] else True
            if not is_eps_growing:
                return None
        elif settings['filter_eps']:
            return None
        
        # 2. ROE 計算修正 (TTM 稅後淨利 / 最新股東權益)
        df_net = df_finance[df_finance['type'].isin(['NetIncome', 'NetIncomeAfterTax'])].sort_values('date')
        df_eq = df_finance[df_finance['type'] == 'Equity'].sort_values('date')
        
        if not df_net.empty and not df_eq.empty:
            ttm_net = df_net['value'].tail(4).sum()  # 最近四季淨利加總
            latest_eq = df_eq['value'].iloc[-1]       # 最新一季股東權益
            if latest_eq > 0:
                latest_roe = (ttm_net / latest_eq) * 100
        
        if latest_roe < settings['min_roe']:
            return None
    else:
        if settings['filter_eps'] or settings['min_roe'] > 0:
            return None

    return {
        "股號": stock_id,
        "現價": latest_price,
        "近3月均營收年增(%)": round(recent_3m_growth, 2),
        "計算後ROE(%)": round(latest_roe, 2),
        "投信10日淨買(張)": int(sitc_10d),
        "外資連大賣天數": consecutive_sell,
        "近60日最高漲幅(%)": round(max_gain_60d, 1),
        "目前本益比": round(latest_pe, 1)
    }

# ==========================================
# 2. 網頁 UI 介面設計
# ==========================================
st.title("🚀 實戰級台股多頭策略篩選器")
st.subheader("數據源：FinMind 真實盤後 API 接口")

st.sidebar.header("🔑 API 金鑰配置")
api_token = st.sidebar.text_input("請輸入 FinMind Token (極度建議輸入，可免除每小時限制):", type="password")
st.sidebar.markdown("[👉 按此免費註冊並取得 Token](https://api.finmindtrade.com/) (1分鐘搞定)")

st.sidebar.header("🎯 策略參數微調")
min_rev_yoy = st.sidebar.slider("近 3 個月營收年增率 > (%)", -10, 30, 15)
min_roe = st.sidebar.slider("ROE > (%)", 0, 30, 15)
filter_eps = st.sidebar.checkbox("要求近四季 EPS 成長", value=True)
min_sitc_buy = st.sidebar.number_input("投信近 10 日偏買超大於 (張)", value=100)

st.sidebar.subheader("⚠️ 容忍度調整 (往右滑較寬鬆)")
filter_foreign_sell = st.sidebar.checkbox("限制外資連續大賣", value=True)
max_foreign_sell_days = st.sidebar.slider("外資連續大賣天數上限 (天)", 1, 10, 5)

filter_max_gain = st.sidebar.checkbox("過濾已暴漲股票", value=True)
max_60d_gain = st.sidebar.slider("近 60 日最高漲幅上限 (%)", 30, 150, 60)

filter_breakout = st.sidebar.checkbox("要求股價剛突破/在平台高點 (20日高點 2% 內)", value=True)

settings = {
    'min_rev_yoy': min_rev_yoy,
    'min_roe': min_roe,
    'filter_eps': filter_eps,
    'min_sitc_buy': min_sitc_buy,
    'filter_foreign_sell': filter_foreign_sell,
    'max_foreign_sell_days': max_foreign_sell_days,
    'filter_max_gain': filter_max_gain,
    'max_60d_gain': max_60d_gain,
    'filter_breakout': filter_breakout
}

st.markdown("### 🔍 步驟 1: 選擇掃描池範疇")
stock_pool_type = st.radio("請選擇篩選範圍：", ["台灣50成份股精選", "自訂群組掃描"])

if stock_pool_type == "台灣50成份股精選":
    target_stocks = ["2330", "2317", "2454", "2308", "2382", "3231", "2603", "2609", "2881", "2882", "2357", "3711", "2412"]
else:
    custom_input = st.text_input("請輸入欲掃描的台股代碼（用逗號隔開）：", "2330,2317,2454,2382,2603")
    target_stocks = [s.strip() for s in custom_input.split(",")]

# 重置錯誤訊息
st.session_state["api_last_error"] = None

if st.button("🔥 開始全方位真實數據篩選", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    total = len(target_stocks)
    for idx, sid in enumerate(target_stocks):
        status_text.text(f"正在分析股票：{sid} ({idx+1}/{total})...")
        res = scan_single_stock(sid, api_token, settings)
        if res:
            results.append(res)
        progress_bar.progress((idx + 1) / total)
        
    status_text.text("📊 篩選完成！")
    
    # 顯示 API 限制警告
    if st.session_state["api_last_error"]:
        st.error(st.session_state["api_last_error"])
        st.info("💡 由於 FinMind 限制未登入用戶每小時僅能存取 30 次，請到 FinMind 官網申請免費的個人 API Token 並填入左側，即可完美流暢使用！")
    
    st.markdown("### 🏆 策略篩選結果")
    if results:
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)
        st.success(f"🎉 成功尋找到 {len(df_res)} 檔符合條件的黃金潛力股！")
    else:
        st.warning("😓 當前市場數據中，暫時沒有股票同時滿足您設定的條件。建議從側邊欄「放寬」營收、取消勾選「要求股價剛突破整理平台」再試一次！")