import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="台股實戰策略篩選器 V2", layout="wide")

if "api_last_error" not in st.session_state:
    st.session_state["api_last_error"] = None

# ==========================================
# 0. FinMind API 數據下載核心函數 (加入自動重試與清洗)
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
        resp = requests.get(url, params=parameter, timeout=15)
        if resp.status_code == 200:
            resp_json = resp.json()
            if resp_json.get("msg") == "success":
                df = pd.DataFrame(resp_json["data"])
                if not df.empty:
                    # 統一將所有欄位中的 NaN 取代或填補，避免計算錯誤
                    df = df.fillna(0)
                return df
            else:
                st.session_state["api_last_error"] = f"API 訊息: {resp_json.get('msg')}"
        elif resp.status_code == 429:
            st.session_state["api_last_error"] = "⚠️ 觸發 API 流量限制！未填寫 Token 每小時僅能請求 30 次，請至左側填入免費 Token。"
    except Exception as e:
        st.session_state["api_last_error"] = f"網路連線異常: {str(e)}"
    return pd.DataFrame()

# ==========================================
# 1. 超強容錯型選股演算法
# ==========================================
def scan_single_stock(stock_id, token, settings):
    # 拉長時間範圍，確保能撈到足夠的歷史季報與月營收
    start_4m = (datetime.today() - timedelta(days=130)).strftime('%Y-%m-%d')
    start_2y = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # 預設輸出初始值
    recent_3m_growth = 0.0
    latest_roe = 0.0
    sitc_10d = 0
    consecutive_sell = 0
    latest_pe = 0.0
    
    # --- A. 技術面數據 ---
    df_price = fetch_finmind_data("TaiwanStockPrice", stock_id, start_4m, token)
    if df_price.empty or 'close' not in df_price.columns or len(df_price) < 10:
        return {"status": "❌ 價格數據缺失", "data": None}
    
    latest_price = float(df_price['close'].iloc[-1])
    price_60d = df_price['close'].tail(min(60, len(df_price)))
    min_60d = price_60d.min()
    max_gain_60d = ((price_60d.max() - min_60d) / min_60d) * 100 if min_60d > 0 else 0.0
    
    if settings['filter_max_gain'] and max_gain_60d > settings['max_60d_gain']:
        return {"status": "❌ 過去60日漲幅過高", "data": None}
        
    price_20d = df_price['close'].tail(min(20, len(df_price)))
    is_breakout = latest_price >= (price_20d.max() * 0.95) # 放寬到 5% 誤差
    if settings['filter_breakout'] and not is_breakout:
        return {"status": "❌ 未處於平台突破點", "data": None}

    # --- B. 籌碼面數據 ---
    df_chip = fetch_finmind_data("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_4m, token)
    if not df_chip.empty and 'name' in df_chip.columns:
        # 確保數字欄位轉為 float/int
        df_chip['buy'] = pd.to_numeric(df_chip['buy'], errors='coerce').fillna(0)
        df_chip['sell'] = pd.to_numeric(df_chip['sell'], errors='coerce').fillna(0)
        
        # 投信買賣超
        sitc_data = df_chip[df_chip['name'] == 'Investment_Trust'].tail(10)
        sitc_10d = sitc_data['buy'].sum() - sitc_data['sell'].sum()
        if sitc_10d < settings['min_sitc_buy']:
            return {"status": "❌ 投信買超張數未達標", "data": None}
            
        # 外資連續大賣
        foreign_data = df_chip[df_chip['name'] == 'Foreign_Investor'].tail(10)
        foreign_net = foreign_data['buy'] - foreign_data['sell']
        for val in reversed(foreign_net.values):
            if val < -500: # 放寬大賣定義到 500 張
                consecutive_sell += 1
            else:
                break
        if settings['filter_foreign_sell'] and consecutive_sell > settings['max_foreign_sell_days']:
            return {"status": "❌ 外資連續大賣天數超標", "data": None}

    # --- C. 估值數據 ---
    df_per = fetch_finmind_data("TaiwanStockPER", stock_id, start_4m, token)
    if not df_per.empty and 'PER' in df_per.columns:
        latest_pe = pd.to_numeric(df_per['PER'].iloc[-1], errors='coerce')
        if pd.isna(latest_pe): latest_pe = 0.0

    # --- D. 基本面營收 ---
    df_rev = fetch_finmind_data("TaiwanStockMonthRevenue", stock_id, start_2y, token)
    if not df_rev.empty and 'revenue_year_growth' in df_rev.columns:
        df_rev['revenue_year_growth'] = pd.to_numeric(df_rev['revenue_year_growth'], errors='coerce').fillna(0)
        recent_3m_growth = df_rev['revenue_year_growth'].tail(3).mean()
        if recent_3m_growth < settings['min_rev_yoy']:
            return {"status": "❌ 營收年增率未達標", "data": None}

    # --- E. 財務報表 (EPS / ROE) ---
    df_finance = fetch_finmind_data("TaiwanStockFinancialStatements", stock_id, start_2y, token)
    if not df_finance.empty and 'type' in df_finance.columns:
        df_finance['value'] = pd.to_numeric(df_finance['value'], errors='coerce').fillna(0)
        
        # EPS 成長檢查
        df_eps = df_finance[df_finance['type'] == 'EPS'].sort_values('date')
        if len(df_eps) >= 4:
            eps_values = df_eps['value'].tail(4).values
            if settings['filter_eps'] and eps_values[-1] < eps_values[-4]:
                return {"status": "❌ 最新EPS低於去年同期", "data": None}
        
        # ROE 計算
        df_net = df_finance[df_finance['type'].isin(['NetIncome', 'NetIncomeAfterTax'])].sort_values('date')
        df_eq = df_finance[df_finance['type'] == 'Equity'].sort_values('date')
        if not df_net.empty and not df_eq.empty:
            ttm_net = df_net['value'].tail(4).sum()
            latest_eq = df_eq['value'].iloc[-1]
            if latest_eq > 0:
                latest_roe = (ttm_net / latest_eq) * 100
        
        if latest_roe < settings['min_roe']:
            return {"status": "❌ ROE未達標", "data": None}

    return {
        "status": "✅ 完全符合",
        "data": {
            "股號": stock_id,
            "現價": latest_price,
            "近3月均營收年增(%)": round(recent_3m_growth, 2),
            "估算ROE(%)": round(latest_roe, 2),
            "投信10日淨買(張)": int(sitc_10d),
            "外資連大賣(天)": consecutive_sell,
            "60日最高漲幅(%)": round(max_gain_60d, 1),
            "目前本益比": round(latest_pe, 1)
        }
    }

# ==========================================
# 2. 網頁 UI 介面設計
# ==========================================
st.title("🚀 台股實戰策略篩選器 V2 (防禦修復版)")

st.sidebar.header("🔑 API 金鑰配置")
api_token = st.sidebar.text_input("請輸入 FinMind Token (若常跳出429錯誤請務必填寫):", type="password")
st.sidebar.markdown("[👉 免費領取 Token 點這裡](https://api.finmindtrade.com/)")

st.sidebar.header("🎯 篩選條件設定")
min_rev_yoy = st.sidebar.slider("近 3 個月營收年增率 > (%)", -20, 30, 15)
min_roe = st.sidebar.slider("ROE > (%)", 0, 30, 15)
filter_eps = st.sidebar.checkbox("要求近四季 EPS 成長 (對比去年同期)", value=True)
min_sitc_buy = st.sidebar.number_input("投信近 10 日偏買超大於 (張)", value=100)

st.sidebar.subheader("🛡️ 寬鬆度與技術面控制")
filter_foreign_sell = st.sidebar.checkbox("限制外資連續大賣", value=False) # 預設關閉防誤殺
max_foreign_sell_days = st.sidebar.slider("外資連續大賣天數上限", 1, 10, 5)

filter_max_gain = st.sidebar.checkbox("過濾已暴漲股票", value=True)
max_60d_gain = st.sidebar.slider("近 60 日最高漲幅上限 (%)", 30, 200, 60)

filter_breakout = st.sidebar.checkbox("要求股價處於平台高點附近", value=False) # 預設關閉防誤殺

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

st.markdown("### 🔍 步驟 1: 選擇篩選股票池")
stock_pool_type = st.radio("範圍選擇：", ["權值股精選池", "自訂代碼輸入"])

if stock_pool_type == "權值股精選池":
    # 精選流速快、法人必看、數據最完整的 8 檔經典標的測試
    target_stocks = ["2330", "2317", "2454", "2382", "3231", "2449", "3711", "2383"]
else:
    custom_input = st.text_input("請輸入股號（以英文逗號隔開）：", "2330,2382,2454")
    target_stocks = [s.strip() for s in custom_input.split(",") if s.strip()]

st.session_state["api_last_error"] = None

if st.button("🔥 開始執行全方位真實數據篩選", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    matched_results = []
    debug_logs = []
    
    total = len(target_stocks)
    for idx, sid in enumerate(target_stocks):
        status_text.text(f"正在分析：{sid} ({idx+1}/{total})...")
        res = scan_single_stock(sid, api_token, settings)
        
        if res["status"] == "✅ 完全符合":
            matched_results.append(res["data"])
        
        debug_logs.append({"股號": sid, "篩選狀態": res["status"]})
        progress_bar.progress((idx + 1) / total)
        
    status_text.text("📊 掃描完畢！")
    
    if st.session_state["api_last_error"]:
        st.error(st.session_state["api_last_error"])
        
    # 展示最終結果
    st.markdown("### 🏆 策略篩選結果")
    if matched_results:
        st.dataframe(pd.DataFrame(matched_results), use_container_width=True)
        st.success(f"🎉 太棒了！成功篩選出 {len(matched_results)} 檔符合條件的黃金股票！")
    else:
        st.warning("😓 糟糕，目前設定的條件下沒有股票能『完全符合』。")
        
    # 展示透明的除錯過程，讓你知道是哪一關卡住
    with st.expander("👁️ 查看每檔股票的過濾原因（偵錯控制台）"):
        st.table(pd.DataFrame(debug_logs))