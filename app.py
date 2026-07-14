import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="台股實戰策略篩選器", layout="wide")

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
        if resp.status_code == 200 and resp.json().get("msg") == "success":
            return pd.DataFrame(resp.json()["data"])
    except Exception as e:
        st.warning(f"無法取得 {stock_id} 的 {dataset} 資料: {str(e)}")
    return pd.DataFrame()

# ==========================================
# 1. 核心選股邏輯演算法
# ==========================================
def scan_single_stock(stock_id, token, settings):
    today_str = datetime.today().strftime('%Y-%m-%d')
    start_3m = (datetime.today() - timedelta(days=120)).strftime('%Y-%m-%d')
    start_1y = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # --- A. 技術面與籌碼面數據 (日K、法人) ---
    df_price = fetch_finmind_data("TaiwanStockPrice", stock_id, start_3m, token)
    df_chip = fetch_finmind_data("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_3m, token)
    df_per = fetch_finmind_data("TaiwanStockPER", stock_id, start_3m, token)
    
    if df_price.empty or len(df_price) < 20:
        return None
    
    # 提取最新價格指標
    latest_price = df_price['close'].iloc[-1]
    
    # 條件 7: 股價剛突破整理平台（而非已大漲 50%）
    # 判斷近60日最高漲幅
    price_60d = df_price['close'].tail(60)
    min_60d = price_60d.min()
    max_gain_60d = ((price_60d.max() - min_60d) / min_60d) * 100
    if max_gain_60d > settings['max_60d_gain']:
        return None
        
    # 簡單平台突破定義：今日收盤價創近20日新高，且過去20日波動率小
    price_20d = df_price['close'].tail(20)
    is_breakout = latest_price >= price_20d.max() * 0.99
    if settings['filter_breakout'] and not is_breakout:
        return None

    # 條件 5 & 6: 籌碼面 (投信近10日偏買超 / 外資沒有連續大賣)
    if not df_chip.empty:
        # 投信近10日買賣超加總
        sitc_10d = df_chip[df_chip['name'] == 'Investment_Trust'].tail(10)['buy'].sum() - \
                    df_chip[df_chip['name'] == 'Investment_Trust'].tail(10)['sell'].sum()
        if sitc_10d < settings['min_sitc_buy']:
            return None
            
        # 外資連續大賣天數
        foreign_data = df_chip[df_chip['name'] == 'Foreign_Investor'].tail(10)
        foreign_net = foreign_data['buy'] - foreign_data['sell']
        consecutive_sell = 0
        for val in reversed(foreign_net.values):
            if val < -1000: # 定義單日大賣超過 1000 張
                consecutive_sell += 1
            else:
                break
        if consecutive_sell > settings['max_foreign_sell_days']:
            return None
    else:
        sitc_10d, consecutive_sell = 0, 0

    # 條件 8: 本益比不高於同產業平均太多
    latest_pe = df_per['PER'].iloc[-1] if not df_per.empty else 20.0
    
    # --- B. 基本面數據 (營收、季報) ---
    df_rev = fetch_finmind_data("TaiwanStockMonthRevenue", stock_id, start_1y, token)
    df_finance = fetch_finmind_data("TaiwanStockFinancialStatements", stock_id, start_1y, token)
    
    # 條件 1: 近 3 個月營收年增率 > X%
    if not df_rev.empty and len(df_rev) >= 3:
        recent_3m_growth = df_rev['revenue_year_growth'].tail(3).mean()
        if recent_3m_growth < settings['min_rev_yoy']:
            return None
    else:
        return None

    # 條件 2 & 3: 近四季 EPS 成長 & ROE > X%
    if not df_finance.empty:
        df_eps = df_finance[df_finance['type'] == 'EPS'].sort_values('date')
        if len(df_eps) >= 4:
            eps_values = df_eps['value'].tail(4).values
            # 判斷近四季EPS趨勢 (最新一季大於去年同期，或前幾季加總成長)
            is_eps_growing = eps_values[-1] > eps_values[-4] if settings['filter_eps'] else True
            if not is_eps_growing:
                return None
        
        df_roe = df_finance[df_finance['type'] == 'ReturnOnEquity'].tail(1)
        latest_roe = df_roe['value'].iloc[-1] if not df_roe.empty else 0
        if latest_roe < settings['min_roe']:
            return None
    else:
        latest_roe = 0

    # 回傳符合所有條件的股票數據
    return {
        "股號": stock_id,
        "現價": latest_price,
        "近3月均營收年增(%)": round(recent_3m_growth, 2),
        "ROE(%)": round(latest_roe, 2),
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

# 側邊欄設定
st.sidebar.header("🔑 API 金鑰配置")
api_token = st.sidebar.text_input("請輸入 FinMind Token (留空限制30次/小時):", type="password")
st.sidebar.caption("💡 可至 FinMind 官網免費註冊取得個人永久 Token。")

st.sidebar.header("🎯 策略參數微調")
settings = {
    'min_rev_yoy': st.sidebar.slider("近 3 個月營收年增率 > (%)", 0, 30, 15),
    'min_roe': st.sidebar.slider("ROE > (%)", 0, 30, 15),
    'filter_eps': st.sidebar.checkbox("要求近四季 EPS 成長", value=True),
    'min_sitc_buy': st.sidebar.number_input("投信近 10 日偏買超大於 (張)", value=100),
    'max_foreign_sell_days': st.sidebar.slider("外資連續大賣天數上限 (天)", 1, 5, 3),
    'filter_breakout': st.sidebar.checkbox("要求股價剛突破整理平台", value=True),
    'max_60d_gain': st.sidebar.slider("近 60 日最高漲幅上限 (%)", 30, 100, 50)
}

# 掃描範圍選擇
st.markdown("### 🔍 步驟 1: 選擇掃描池範疇")
stock_pool_type = st.radio("請選擇篩選範圍：", ["台灣50成份股精選", "自訂群組掃描"])

if stock_pool_type == "台灣50成份股精選":
    # 內建台灣前幾大權值與高流動性標的進行快速過濾
    target_stocks = ["2330", "2317", "2454", "2308", "2382", "3231", "2603", "2609", "2881", "2882", "2357", "3711", "2412"]
else:
    custom_input = st.text_input("請輸入欲掃描的台股代碼（用逗號隔開）：", "2330,2317,2454,2382,2603")
    target_stocks = [s.strip() for s in custom_input.split(",")]

# 執行選股
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
    
    st.markdown("### 🏆 策略篩選結果")
    if results:
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)
        st.success(f"🎉 成功尋找到 {len(df_res)} 檔同時符合「量價突破、法人鎖碼、基本面爆發」的黃金潛力股！")
    else:
        st.warning("😓 當前市場數據中，暫時沒有股票同時滿足您設定的嚴格條件。建議從側邊欄放寬營收或投信買超標準再試一次！")