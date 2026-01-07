import streamlit as st
import psycopg2
import pandas as pd
import os

# 設定頁面標題
st.set_page_config(page_title="AI 股票預測系統", layout="wide")

st.title("股票預測與分析系統")

# 資料庫連線設定
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "stock-db"),
        database=os.getenv("DB_NAME", "stock_ai_hub"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
        port=os.getenv("DB_PORT", "5432")
    )
    return conn

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "current_symbol" not in st.session_state:
    st.session_state.current_symbol = ""

# 側邊欄輸入
st.sidebar.header("查詢設定")

symbol_input = st.sidebar.text_input("請輸入股票代號", value="TSLA").upper()

if st.sidebar.button("執行預測"):
    if not symbol_input:
        st.error("請輸入股票代號")
    else:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # 步驟 1: 抓取股票資料
                    st.sidebar.info(f"正在抓取 {symbol_input} 的最新資料...")
                    try:
                        cur.execute("SELECT fetch_stock_data(%s)", (symbol_input,))
                        conn.commit()
                        st.sidebar.success("資料抓取與更新成功！")
                    except Exception as e:
                        st.sidebar.error(f"抓取資料失敗: {e}")
                        conn.rollback()

                    # 步驟 2: 執行預測查詢 (MSSQL Rule-Based Strategy Model 2)
                    st.sidebar.info("AI 正在進行分析與預測...")
                    query = """
                    SELECT 
                        f.symbol as "股票代號",
                        f.close_price as "今日收盤價",
                        f.ema_10 as "10日EMA",
                        f.ema_30 as "30日EMA",
                        res.recommendation as "AI建議",
                        res.reason as "分析理由"
                    FROM (
                        SELECT 
                            symbol,
                            close_price,
                            ema_10,
                            ema_30,
                            prev_sma_10 as prev_ema_10,
                            prev_sma_30 as prev_ema_30,
                            prev_close
                        FROM v_stock_features
                        WHERE symbol = %s
                        ORDER BY trade_date DESC
                        LIMIT 1
                    ) f
                    CROSS JOIN LATERAL get_mssql_style_strategy(
                        f.ema_10, 
                        f.ema_30, 
                        f.prev_ema_10, 
                        f.prev_ema_30,
                        f.close_price
                    ) res;
                    """
                    
                    df = pd.read_sql(query, conn, params=(symbol_input,))
                    
                    if not df.empty:
                        st.session_state.analysis_result = df
                        st.session_state.current_symbol = symbol_input
                        # 強制重新執行一次以渲染畫面
                        st.rerun()
                    else:
                        st.warning("查無資料，請確認股票代號是否正確，或資料庫是否有足夠的歷史數據。")
                        st.session_state.analysis_result = None

        except Exception as e:
            st.error(f"連線或查詢發生錯誤: {e}")

# -----------------
# 主畫面渲染邏輯 (基於 Session State)
# -----------------
if st.session_state.analysis_result is not None:
    df = st.session_state.analysis_result
    symbol = st.session_state.current_symbol
    
    st.subheader("策略分析結果")
    
    # 使用 metrics 展示關鍵指標
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("股票代號", df.iloc[0]["股票代號"])
    with col2:
        st.metric("今日收盤價", f"${df.iloc[0]['今日收盤價']:.2f}")
    with col3:
        signal = df.iloc[0]["AI建議"]
        st.metric("AI建議", signal)

    # 顯示均線數據
    col4, col5 = st.columns(2)
    with col4:
            st.metric("10日 EMA (短期趨勢)", f"${df.iloc[0]['10日EMA']:.2f}")
    with col5:
            st.metric("30日 EMA (中期趨勢)", f"${df.iloc[0]['30日EMA']:.2f}")

    # 顯示分析理由
    reason = df.iloc[0]["分析理由"]
    st.subheader(f"分析理由: {reason}")
    
    if "BUY" in signal:
        st.success(f"目前信號: {signal} - {reason}")
    elif "SELL" in signal:
        st.error(f"目前信號: {signal} - {reason}")
    else:
        st.warning(f"目前信號: {signal} - {reason}")
        
    # 顯示完整表格
    st.table(df)

    # 查看原始數據集
    with st.expander("查看詳細歷史數據"):
        try:
            with get_db_connection() as conn:
                raw_query = """
                    SELECT 
                        trade_date as "日期",
                        close_price as "收盤價",
                        ema_10 as "10日 EMA",
                        ema_30 as "30日 EMA",
                        prev_close as "昨日收盤",
                        prev_sma_10 as "昨日 EMA10"
                    FROM v_stock_features
                    WHERE symbol = %s
                    ORDER BY trade_date DESC
                """
                raw_df = pd.read_sql(raw_query, conn, params=(symbol,))
                st.dataframe(raw_df)
        except Exception as e:
            st.error(f"無法讀取原始數據: {e}")

    # 新增：歷史回測按鈕 (Backtesting)
    st.markdown("---")
    st.subheader("歷史回測模擬")
    st.markdown(f"按下按鈕後，AI 將會重跑 **{symbol}** 過去 2 年的數據，模擬依照Model 1、2及Model 3策略交易的績效。")
    
    if st.button("執行回測模擬"):
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur_backtest:
                    with st.spinner(f"正在模擬 {symbol} 的歷史交易紀錄..."):
                        cur_backtest.execute("SELECT run_backtest(%s)", (symbol,))
                        conn.commit()
                        
                        # 讀取回測結果
                        trades_query = "SELECT buy_date, buy_price, sell_date, sell_price, profit_pct, strategy_model FROM stored_trades WHERE symbol = %s ORDER BY strategy_model, buy_date DESC"
                        trades_df = pd.read_sql(trades_query, conn, params=(symbol,))
                        
                        if not trades_df.empty:
                            # 建立三個分頁，分別顯示不同模型的結果
                            tab1, tab2, tab3 = st.tabs(["Model 1", "Model 2", "Model 3"])
                            
                            # Helper function to display metrics
                            def display_metrics(df_sub):
                                if df_sub.empty:
                                    st.info("此策略在選定期間內無交易訊號。")
                                    return
                                    
                                total = len(df_sub)
                                avg_p = df_sub['profit_pct'].mean()
                                wins = len(df_sub[df_sub['profit_pct'] > 0])
                                win_r = wins / total * 100
                                
                                c1, c2, c3 = st.columns(3)
                                c1.metric("總交易次數", total)
                                c2.metric("平均獲利", f"{avg_p:.2f}%")
                                c3.metric("勝率", f"{win_r:.2f}% ({wins}/{total})")
                                st.dataframe(df_sub)

                            with tab1:
                                st.markdown("**Model 1 (Standard):** 僅基於 EMA 10 與 EMA 30 的交叉進行買賣。交易次數較多，但雜訊也較多。")
                                df_m1 = trades_df[trades_df['strategy_model'] == "Model 1 (Standard)"]
                                display_metrics(df_m1)
                                
                            with tab2:
                                st.markdown("**Model 2 (Confirmed):** 在交叉基礎上，加入「收盤價 > 10日EMA」確認條件。確保趨勢確立後才進場。")
                                df_m2 = trades_df[trades_df['strategy_model'] == "Model 2 (Confirmed)"]
                                display_metrics(df_m2)
                                
                            with tab3:
                                st.markdown("**Model 3 (AI Enhanced):** 結合線性回歸 (Linear Regression) 機器學習模型。只有當 EMA 黃金交叉 **且** AI 預測明日股價上漲時才買入，雙重確認降低風險。")
                                df_m3 = trades_df[trades_df['strategy_model'] == "Model 3 (AI Enhanced)"]
                                display_metrics(df_m3)

                        else:
                            st.warning("模擬完成，但過去這段時間沒有觸發任何符合策略的交易訊號 (可能是盤整或條件過於嚴格)。")
                        
                        # ==========================================
                        #  Gap 3: 補上績效匯總報表 (Summary Report)
                        # ==========================================
                        st.markdown("---")
                        st.subheader("綜合績效報告")
                        st.markdown("下方資料來自資料庫視圖 ，自動計算勝率與獲利能力：")
                        
                        summary_sql = """
                            SELECT 
                                strategy_model as "策略模型",
                                total_trades as "總交易次數",
                                win_rate_pct as "勝率 (%%)",
                                avg_profit_pct as "平均獲利 (%%)",
                                max_profit_pct as "最大單筆獲利 (%%)"
                            FROM v_strategy_performance 
                            WHERE symbol = %s
                            ORDER BY win_rate_pct DESC
                        """
                        # 注意: 這裡使用 %% 跳脫百分比符號，避免 Python 格式化錯誤
                        
                        summary_df = pd.read_sql(summary_sql, conn, params=(symbol,))
                        st.table(summary_df)

        except Exception as e:
            st.error(f"回測執行失敗: {e}")

st.markdown("---")
st.markdown("*本系統使用 Docker 部署，後端連接 PostgreSQL 資料庫進行即時運算*")
