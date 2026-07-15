import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="台股策略篩選器 V3 診斷版", layout="wide")

# 初始化 Session State
if "api_last_error" not in st.session_state:
    st.session_state["api_last_error"] = None

# ==========================================
# 0. FinMind API 核心請求函數 (含瀏覽器偽裝與深度偵錯)
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
    
    # 🚨 關鍵修正：加入 User-Agent 偽裝成一般 Chrome 瀏覽器，防止被 Cloudflare 直接封鎖
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        resp = requests.get(url, params=parameter, headers=headers, timeout=15)
        if resp.status_code == 200:
            try:
                resp_json = resp.json()
                if resp_json.get("msg") == "success":
                    return pd.DataFrame(resp_json["data"])
                else:
                    st.session_state["api_last_error"] = f"🛑 API 拒絕請求，原因: {resp_json.get('msg')}"
            except Exception:
                st.session_state["api_last_error"] = f"⚠️ 解析 JSON 失敗！伺服器可能回傳了阻擋網頁，開頭為: {resp.text[:100]}"
        elif resp.status_code == 429:
            st.session_state["api_last_error"] = "⚠️ 觸發 API 流量超限 (429 Too Many Requests)！Streamlit 共用 IP 已被耗盡，請務必填寫個人 Token。"
        elif resp.status_code == 403:
            st.session_state["api_last_error"] = "🚫 存取被拒 (403 Forbidden)！您的請求被 Cloudflare 防爬蟲防火牆攔截。"
        else:
            st.session_state["api_last_error"] = f"❌ 連線失敗，伺服器回傳 HTTP 狀態碼: {resp.status_code}"
    except Exception as e:
        st.session_state["api_last_error"] = f"💥 網路連線異常/逾時: {str(e)}"
        
    return pd.DataFrame()

# ==========================================
# 1. 選股核心過濾邏輯
# ==========================================
def scan_single_stock(stock_id, token, settings):
    start_4m = (datetime.today() - timedelta(days=130)).strftime('%Y-%m-%d')
    start_2y = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    recent_3m_growth = 0.0
    latest_roe = 0.0
    sitc_10d = 0
    consecutive_sell = 0
    latest_pe = 0.0
    
    # --- A. 技術面數據 ---
    df_price = fetch_finmind_data("TaiwanStockPrice", stock_id, start_4m, token)
    if df_price.empty or 'close' not in df_price.columns or len(df_price) < 5:
        return {"status": "❌ 價格數據缺失 (API未回傳或被擋)", "data": None}
    
    latest_price = float(df_price['close'].iloc[-1])
    price_60d = df_price['close'].tail(min(60, len(df_price)))
    min_60d = price_60d.min()
    max_gain_60d = ((price_60d.max() - min_60d) / min_60d) * 100 if min_60d > 0 else 0.0
    
    if settings['filter_max_gain'] and max_gain_60d > settings['max_60d_gain']:
        return {"status": "❌ 過去60日漲幅過高", "data": None}
        
    price_20d = df_price['close'].tail(min(20, len(df_price)))
    is_breakout = latest_price >= (price_20d.max() * 0.95)
    if settings['filter_breakout'] and not is_breakout:
        return {"status": "❌ 未處於平台突破點", "data": None}

    # --- B. 籌碼面數據 ---
    df_chip = fetch_finmind_data("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_4m, token)
    if not df_chip.empty and 'name' in df_chip.columns:
        df_chip['buy'] = pd.to_numeric(df_chip['buy'], errors='coerce').fillna(0)
        df_chip['sell'] = pd.to_numeric(df_chip['sell'], errors='coerce').fillna(0)
        
        sitc_data = df_chip[df_chip['name'] == 'Investment_Trust'].tail(10)
        sitc_10d = sitc_data['buy'].sum() - sitc_data['sell'].sum()
        if sitc_10d < settings['min_sitc_buy']:
            return {"status": "❌ 投信買超未達標", "data": None}
            
        foreign_data = df_chip[df_chip['name'] == 'Foreign_Investor'].tail(10)
        foreign_net = foreign_data['buy'] - foreign_data['sell']
        for val in reversed(foreign_net.values):
            if val < -500:
                consecutive_sell += 1
            else:
                break
        if settings['filter_foreign_sell'] and consecutive_sell > settings['max_foreign_sell_days']:
            return {"status": "❌ 外資連續大賣超標", "data": None}

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

    # --- E. 財務報表 ---
    df_finance = fetch_finmind_data("TaiwanStockFinancialStatements", stock_id, start_2y, token)
    if not df_finance.empty and 'type' in df_finance.columns:
        df_finance['value'] = pd.to_numeric(df_finance['value'], errors='coerce').fillna(0)
        
        df_eps = df_finance[df_finance['type'] == 'EPS'].sort_values('date')
        if len(df_eps) >= 4 and settings['filter_eps']:
            eps_values = df_eps['value'].tail(4).values
            if eps_values[-1] < eps_values[-4]:
                return {"status": "❌ EPS未見成長", "data": None}
        
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
            "股號": stock_id, "現價": latest_price,
            "近3月均營收年增(%)": round(recent_3m_growth, 2), "估算ROE(%)": round(latest_roe, 2),
            "投信10日淨買(張)": int(sitc_10d), "外資連大賣(天)": consecutive_sell,
            "60日最高漲幅(%)": round(max_gain_60d, 1), "目前本益比": round(latest_pe, 1)
        }
    }

# ==========================================
# 2. 網頁 UI 介面
# ==========================================
st.title("🚀 台股策略篩選器 V3 (防禦與聯網診斷版)")

# 診斷區塊：放在最醒目的地方
st.markdown("### 🛠️ 聯網與 API 健康度診斷面板")
col_diag1, col_diag2 = st.columns([1, 3])

with col_diag1:
    test_click = st.button("🔍 點我執行 API 直連測試")

with col_diag2:
    if test_click:
        st.session_state["api_last_error"] = None
        # 用台積電 2330 進行單次直連測試
        test_url = "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=2330&start_date=2026-06-01"
        if st.sidebar.text_input: # 如果有填 token 補上
            pass
        
        st.write("正在嘗試與 FinMind 伺服器進行握手測試...")
        try:
            h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            t_resp = requests.get(test_url, headers=h, timeout=10)
            st.info(f"系統回應狀態碼: {t_resp.status_code}")
            if t_resp.status_code == 200:
                json_data = t_resp.json()
                st.success(f"伺服器訊息 (msg): {json_data.get('msg')}")
                if json_data.get("msg") == "success" and json_data.get("data"):
                    st.json(json_data["data"][0]) # 印出第一筆資料格式
                else:
                    st.warning("⚠️ 雖然連線成功，但 API 沒有回傳有效的股票資料陣列。")
            else:
                st.error(f"連線失敗！回傳的內容為: {t_resp.text[:200]}")
        except Exception as ex:
            st.error(f"連線完全斷開，錯誤原因: {str(ex)}")

st.divider()

# 側邊欄配置
st.sidebar.header("🔑 API 金鑰配置")
api_token = st.sidebar.text_input("請輸入 FinMind Token (強烈建議至官網免費申請):", type="password")
st.sidebar.markdown("[👉 免費註冊領取 Token](https://api.finmindtrade.com/)")

st.sidebar.header("🎯 篩選條件設定")
min_rev_yoy = st.sidebar.slider("近 3 個月營收年增率 > (%)", -20, 30, -20) # 預設拉到最小
min_roe = st.sidebar.slider("ROE > (%)", 0, 30, 0) # 預設拉到最小
filter_eps = st.sidebar.checkbox("要求近四季 EPS 成長", value=False) # 預設關閉
min_sitc_buy = st.sidebar.number_input("投信近 10 日偏買超大於 (張)", value=-99999) # 預設拉到最小

st.sidebar.subheader("🛡️ 寬鬆度與技術面過濾")
filter_foreign_sell = st.sidebar.checkbox("限制外資連續大賣", value=False)
max_foreign_sell_days = st.sidebar.slider("外資連續大賣天數上限", 1, 15, 15)

filter_max_gain = st.sidebar.checkbox("過濾已暴漲股票", value=False)
max_60d_gain = st.sidebar.slider("近 60 日最高漲幅上限 (%)", 30, 300, 300)

filter_breakout = st.sidebar.checkbox("要求股價處於平台高點附近", value=False)

settings = {
    'min_rev_yoy': min_rev_yoy, 'min_roe': min_roe, 'filter_eps': filter_eps,
    'min_sitc_buy': min_sitc_buy, 'filter_foreign_sell': filter_foreign_sell,
    'max_foreign_sell_days': max_foreign_sell_days, 'filter_max_gain': filter_max_gain,
    'max_60d_gain': max_60d_gain, 'filter_breakout': filter_breakout
}

st.markdown("### 🔍 步驟 1: 選擇篩選股票池")
stock_pool_type = st.radio("範圍選擇：", ["權值股精選池", "自訂代碼輸入"])

if stock_pool_type == "權值股精選池":
    target_stocks = ["2330", "2317", "2454", "2382", "3231", "2449", "3711", "2383"]
else:
    custom_input = st.text_input("請輸入股號（以英文逗號隔開）：", "2330,2317,2454")
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
    
    # 顯示整體 API 偵錯紅框
    if st.session_state["api_last_error"]:
        st.error(st.session_state["api_last_error"])
        st.info("💡 提示：如果上方出現 429 或 403 錯誤，代表共用 IP 被擋，此時『必須』去 FinMind 官網註冊一個免費帳號，並在左側輸入你的 Token，就能立刻解鎖！")
        
    st.markdown("### 🏆 策略篩選結果")
    if matched_results:
        st.dataframe(pd.DataFrame(matched_results), use_container_width=True)
        st.success(f"🎉 成功篩選出 {len(matched_results)} 檔股票！")
    else:
        st.warning("😓 目前沒有股票能完全滿足條件。")
        
    with st.expander("👁️ 查看每檔股票的過濾原因（偵錯控制台）"):
        st.table(pd.DataFrame(debug_logs))