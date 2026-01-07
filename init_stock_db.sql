--
-- PostgreSQL database dump
--

\restrict cOm79HTGXlsEkixuzTzj3O6aOfPbA4hmJGhCqZGj3rOSLMMfRGlcAOIimiFyArJ

-- Dumped from database version 16.11 (Debian 16.11-1.pgdg13+1)
-- Dumped by pg_dump version 16.11 (Debian 16.11-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: plpython3u; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS plpython3u WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpython3u; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpython3u IS 'PL/Python3U untrusted procedural language';


--
-- Name: fetch_stock_data(text); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.fetch_stock_data(ticker text) RETURNS text
    LANGUAGE plpython3u
    AS $_$
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # 修改為下載最近 2 年的資料，並同時抓取開盤價和收盤價
    data = yf.download(ticker, period="2y", interval="1d")
    
    if data.empty:
        return "No data found"

    # 修改 SQL：現在同時插入 open_price 和 close_price
    plan = plpy.prepare("INSERT INTO stock_prices (symbol, trade_date, open_price, close_price) VALUES ($1, $2, $3, $4) ON CONFLICT (symbol, trade_date) DO UPDATE SET open_price = EXCLUDED.open_price, close_price = EXCLUDED.close_price", ["text", "date", "float", "float"])
    
    count = 0
    for index, row in data.iterrows():
        # 同時取得開盤價和收盤價
        open_val = float(row['Open'].iloc[0]) if hasattr(row['Open'], 'iloc') else float(row['Open'])
        close_val = float(row['Close'].iloc[0]) if hasattr(row['Close'], 'iloc') else float(row['Close'])
        plpy.execute(plan, [ticker, index.date(), open_val, close_val])
        count += 1
        
    return f"Successfully fetched {count} rows for {ticker}"
$_$;


ALTER FUNCTION public.fetch_stock_data(ticker text) OWNER TO postgres;

--
-- Name: get_ai_strategy(double precision[]); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.get_ai_strategy(prices double precision[]) RETURNS TABLE(predicted_price double precision, recommendation text)
    LANGUAGE plpython3u
    AS $$
    import numpy as np
    from sklearn.linear_model import LinearRegression

    # 1. 確保有足夠的數據進行預測
    if len(prices) < 2:
        return [(0.0, "INSUFFICIENT DATA")]

    # 2. 準備數據
    y = np.array(prices).reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)

    # 3. 訓練線性回歸模型
    model = LinearRegression().fit(x, y)

    # 4. 預測明天的價格
    next_day = np.array([[len(y)]])
    prediction = float(model.predict(next_day)[0][0])

    # 5. 決策邏輯 (預測價 vs 最後一天實際價)
    last_price = prices[-1]
    if prediction > last_price:
        signal = "★ BUY (Bullish)"
    elif prediction < last_price:
        signal = "☆ SELL (Bearish)"
    else:
        signal = "HOLD"

    return [(prediction, signal)]
$$;


ALTER FUNCTION public.get_ai_strategy(prices double precision[]) OWNER TO postgres;

--
-- Name: get_mssql_style_strategy(double precision, double precision, double precision, double precision); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE OR REPLACE FUNCTION public.get_mssql_style_strategy(ema_10_today double precision, ema_30_today double precision, ema_10_prev double precision, ema_30_prev double precision) RETURNS TABLE(recommendation text, reason text)
    LANGUAGE plpython3u
    AS $$
    # 10日EMA與30日EMA交叉策略 (Exponential Moving Average Crossover Strategy)
    # 完全依照 MSSQLTips 文章的 EMA 邏輯
    
    # 初始化
    signal = "HOLD"
    reason = "No significant trend change"
    
    # 判斷是否黃金交叉 (Golden Cross)
    # 定義: 今天 10EMA > 30EMA，且 昨天 10EMA <= 30EMA
    if ema_10_today > ema_30_today and ema_10_prev <= ema_30_prev:
        signal = "BUY"
        reason = "Golden Cross: 10-day EMA crossed above 30-day EMA"
        
    # 判斷是否死亡交叉 (Death Cross)
    # 定義: 今天 10EMA < 30EMA，且 昨天 10EMA >= 30EMA
    elif ema_10_today < ema_30_today and ema_10_prev >= ema_30_prev:
        signal = "SELL"
        reason = "Death Cross: 10-day EMA crossed below 30-day EMA"
        
    # 判斷當前持倉狀態
    elif ema_10_today > ema_30_today:
        signal = "HOLD (Bullish)"
        reason = "Uptrend continues (10EMA > 30EMA)"
        
    elif ema_10_today < ema_30_today:
        signal = "HOLD (Bearish)"
        reason = "Downtrend continues (10EMA < 30EMA)"

    return [(signal, reason)]
$$;

ALTER FUNCTION public.get_mssql_style_strategy(double precision, double precision, double precision, double precision) OWNER TO postgres;

--
-- Name: predict_next_price(double precision[]); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.predict_next_price(prices double precision[]) RETURNS double precision
    LANGUAGE plpython3u
    AS $$
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    y = np.array(prices).reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    next_day = np.array([[len(y)]])
    prediction = model.predict(next_day)
    return float(prediction[0][0])
$$;


ALTER FUNCTION public.predict_next_price(prices double precision[]) OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: stock_prices; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.stock_prices (
    symbol text NOT NULL,
    trade_date date NOT NULL,
    open_price double precision,
    close_price double precision
);


ALTER TABLE public.stock_prices OWNER TO postgres;

--
-- Name: calculate_ema(double precision, double precision, integer); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.calculate_ema(current_price double precision, prev_ema double precision, period integer) RETURNS double precision
    LANGUAGE plpgsql
    AS $$
DECLARE
    multiplier double precision;
BEGIN
    -- EMA Formula: (Close - Previous EMA) * (2 / (Period + 1)) + Previous EMA
    -- If prev_ema is NULL, return current_price (as initial EMA)
    IF prev_ema IS NULL THEN
        RETURN current_price;
    END IF;

    multiplier := 2.0 / (period + 1);
    RETURN (current_price - prev_ema) * multiplier + prev_ema;
END;
$$;


ALTER FUNCTION public.calculate_ema(double precision, double precision, integer) OWNER TO postgres;

--
-- Name: v_stock_features; Type: VIEW; Schema: public; Owner: postgres
--

-- EMA 計算非常依賴前一天的值，這在 SQL 視圖 (VIEW) 中很難直接透過簡單的 Window Function 完成遞迴計算。
-- 為了簡化並正確實現 EMA (像 pandas ewm 或 Excel 那樣)，我們通常需要：
-- 1. 使用 Recursive CTE (Common Table Expressions)
-- 2. 或者在寫入資料時就計算好存入
-- 這裡我們使用 Recursive CTE 來即時計算 EMA 10 和 EMA 30

CREATE VIEW public.v_stock_features AS
WITH RECURSIVE data_with_rownum AS (
    SELECT 
        symbol,
        trade_date,
        open_price,
        close_price,
        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date) as rn
    FROM public.stock_prices
),
ema_calc AS (
    -- Base case: 第一筆資料 (rn=1)，EMA 等於當天收盤價
    SELECT 
        symbol,
        trade_date,
        open_price,
        close_price,
        rn,
        close_price::double precision as ema_10,
        close_price::double precision as ema_30
    FROM data_with_rownum
    WHERE rn = 1

    UNION ALL

    -- Recursive step: 後續資料，依賴前一天的 EMA
    SELECT 
        d.symbol,
        d.trade_date,
        d.open_price,
        d.close_price,
        d.rn,
        public.calculate_ema(d.close_price, e.ema_10, 10),
        public.calculate_ema(d.close_price, e.ema_30, 30)
    FROM data_with_rownum d
    JOIN ema_calc e ON d.symbol = e.symbol AND d.rn = e.rn + 1
)
SELECT 
    symbol,
    trade_date,
    open_price,
    close_price,
    LAG(close_price) OVER (PARTITION BY symbol ORDER BY trade_date) as prev_close,
    LEAD(open_price) OVER (PARTITION BY symbol ORDER BY trade_date) as next_open,
    ema_10,
    ema_30,
    LAG(ema_10) OVER (PARTITION BY symbol ORDER BY trade_date) as prev_sma_10,
    LAG(ema_30) OVER (PARTITION BY symbol ORDER BY trade_date) as prev_sma_30
FROM ema_calc;


ALTER VIEW public.v_stock_features OWNER TO postgres;

--
-- Data for Name: stock_prices; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.stock_prices (symbol, trade_date, close_price) FROM stdin;
NVDA	2025-12-08	185.5500030517578
NVDA	2025-12-09	184.97000122070312
NVDA	2025-12-10	183.77999877929688
NVDA	2025-12-11	180.92999267578125
NVDA	2025-12-12	175.02000427246094
NVDA	2025-12-15	176.2899932861328
NVDA	2025-12-16	177.72000122070312
NVDA	2025-12-17	170.94000244140625
NVDA	2025-12-18	174.13999938964844
NVDA	2025-12-19	180.99000549316406
NVDA	2025-12-22	183.69000244140625
NVDA	2025-12-23	189.2100067138672
NVDA	2025-12-24	188.61000061035156
NVDA	2025-12-26	190.52999877929688
NVDA	2025-12-29	188.22000122070312
NVDA	2025-12-30	187.5399932861328
NVDA	2025-12-31	186.5
NVDA	2026-01-02	188.85000610351562
NVDA	2026-01-05	188.1199951171875
\.


--
-- Name: stock_prices stock_prices_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stock_prices
    ADD CONSTRAINT stock_prices_pkey PRIMARY KEY (symbol, trade_date);


--
-- PostgreSQL database dump complete
--

\unrestrict cOm79HTGXlsEkixuzTzj3O6aOfPbA4hmJGhCqZGj3rOSLMMfRGlcAOIimiFyArJ

CREATE OR REPLACE FUNCTION public.fetch_stock_data(ticker text) RETURNS text
    LANGUAGE plpython3u
    AS $_$
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # 修改為下載最近 2 年 (2y) 的資料，以便進行回測
    data = yf.download(ticker, period="2y", interval="1d")
    
    if data.empty:
        return "No data found"

    plan = plpy.prepare("INSERT INTO stock_prices (symbol, trade_date, close_price) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING", ["text", "date", "float"])
    
    count = 0
    for index, row in data.iterrows():
        # yfinance 格式處理
        close_val = float(row['Close'].iloc[0]) if hasattr(row['Close'], 'iloc') else float(row['Close'])
        plpy.execute(plan, [ticker, index.date(), close_val])
        count += 1
        
    return f"Successfully fetched {count} rows for {ticker}"
$_$;

ALTER FUNCTION public.fetch_stock_data(ticker text) OWNER TO postgres;
-- 更新策略函數，加入 Model 2 的 "Lag Close Price" 過濾條件
-- 原始文章建議: 
-- 買入訊號:黃金交叉 且 昨日收盤價 < 昨日 10日 EMA (買在低點/回測)
-- 賣出訊號:死亡交叉 且 昨日收盤價 > 昨日 10日 EMA (賣在高點/反彈)

DROP FUNCTION IF EXISTS public.get_mssql_style_strategy(double precision, double precision, double precision, double precision);

CREATE OR REPLACE FUNCTION public.get_mssql_style_strategy(
    ema_10_today double precision, 
    ema_30_today double precision, 
    ema_10_prev double precision, 
    ema_30_prev double precision,
    close_prev double precision
) RETURNS TABLE(recommendation text, reason text)
    LANGUAGE plpython3u
    AS $$
    
    # 初始化
    signal = "HOLD"
    reason = "No significant trend change"
    
    # 基本交叉判斷 (Model 1)
    is_golden_cross = (ema_10_today > ema_30_today) and (ema_10_prev <= ema_30_prev)
    is_death_cross = (ema_10_today < ema_30_today) and (ema_10_prev >= ema_30_prev)
    
    # 進階過濾判斷 (Model 2 logic)
    # 昨日收盤價是否低於昨日 10日 EMA (適合買入的"低點")
    is_dip = close_prev < ema_10_prev
    # 昨日收盤價是否高於昨日 10日 EMA (適合賣出的"高點")
    is_peak = close_prev > ema_10_prev
    
    if is_golden_cross:
        if is_dip:
            # 符合 Model 2 加強版買入
            signal = "BUY (Strong)"
            reason = "Golden Cross + Dip (Model 2: Prev Close < 10 EMA)"
        else:
            # 僅符合 Model 1，但價格可能過高
            signal = "BUY (Weak)"
            reason = "Golden Cross Only (Price not in dip)"
            
    elif is_death_cross:
        if is_peak:
            # 符合 Model 2 加強版賣出
            signal = "SELL (Strong)"
            reason = "Death Cross + Peak (Model 2: Prev Close > 10 EMA)"
        else:
            # 僅符合 Model 1
            signal = "SELL (Weak)"
            reason = "Death Cross Only (Price not at peak)"
            
    # 持倉狀態顯示
    elif ema_10_today > ema_30_today:
        signal = "HOLD (Bullish)"
        reason = "Uptrend continues (10EMA > 30EMA)"
        
    elif ema_10_today < ema_30_today:
        signal = "HOLD (Bearish)"
        reason = "Downtrend continues (10EMA < 30EMA)"

    return [(signal, reason)]
$$;

ALTER FUNCTION public.get_mssql_style_strategy(double precision, double precision, double precision, double precision, double precision) OWNER TO postgres;
-- backtest_logic.sql
-- 這是依照 MSSQLTips 文章實現的「歷史回測系統」 (Backtesting System)
-- 用途: 模擬過去一段時間的交易，計算如果依照 Model 2 策略操作會賺多少錢

-- 1. 建立交易紀錄表
CREATE TABLE IF NOT EXISTS public.stored_trades (
    id SERIAL PRIMARY KEY,
    symbol text,
    buy_date date,
    buy_price double precision,
    sell_date date,
    sell_price double precision,
    profit_pct double precision, -- 獲利百分比
    strategy_model text -- 使用的策略模型 (Model 1 or Model 2)
);

-- 2. 建立回測函數 (使用 Python 邏輯模擬交易)
DROP FUNCTION IF EXISTS public.run_backtest(text);
CREATE FUNCTION public.run_backtest(ticker text) RETURNS text
    LANGUAGE plpython3u
    AS $$
    import plpy
    
    # 1. 清除該股票的舊回測紀錄
    plpy.execute(plpy.prepare("DELETE FROM stored_trades WHERE symbol = $1", ["text"]), [ticker])
    
    # 2. 獲取該股票的所有歷史特徵數據 (按時間排序)
    query = """
        SELECT 
            trade_date, close_price, ema_10, ema_30, 
            prev_sma_10 as prev_ema_10, prev_sma_30 as prev_ema_30, prev_close
        FROM v_stock_features 
        WHERE symbol = $1 
        ORDER BY trade_date ASC
    """
    plan = plpy.prepare(query, ["text"])
    history = plpy.execute(plan, [ticker])
    
    if not history:
        return "No data found for " + ticker

    position = None # 目前持倉狀態: None 或 {'date': ..., 'price': ...}
    trade_count = 0
    
    # 3. 逐日模擬交易
    for row in history:
        date = row['trade_date']
        price = row['close_price']
        
        ema_10 = row['ema_10']
        ema_30 = row['ema_30']
        prev_ema_10 = row['prev_ema_10']
        prev_ema_30 = row['prev_ema_30']
        prev_close = row['prev_close']
        
        # 略過第一筆資料 (沒有前一日數據)
        if prev_ema_10 is None:
            continue
            
        # --- 策略邏輯 (Model 2) ---
        
        # 買入訊號: 黃金交叉 且 昨日收盤 < 昨日10EMA (買低)
        if position is None:
            is_golden_cross = (ema_10 > ema_30) and (prev_ema_10 <= prev_ema_30)
            is_dip = (prev_close < prev_ema_10)
            
            # 這裡我們採取嚴格模式，必須符合 Model 2 才買入
            if is_golden_cross and is_dip:
                position = {'date': date, 'price': price}
        
        # 賣出訊號: 死亡交叉 (或是停損停利，這裡簡化為只看交叉)
        elif position is not None:
            is_death_cross = (ema_10 < ema_30) and (prev_ema_10 >= prev_ema_30)
            
            if is_death_cross:
                buy_price = position['price']
                profit = (price - buy_price) / buy_price * 100
                
                # 紀錄交易
                ins_plan = plpy.prepare(
                    "INSERT INTO stored_trades (symbol, buy_date, buy_price, sell_date, sell_price, profit_pct, strategy_model) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    ["text", "date", "float", "date", "float", "float", "text"]
                )
                plpy.execute(ins_plan, [ticker, position['date'], buy_price, date, price, profit, "Model 2"])
                
                position = None # 清空持倉
                trade_count += 1
                
    return f"Simulated {trade_count} trades for {ticker}"
$$;

ALTER FUNCTION public.run_backtest(text) OWNER TO postgres;
CREATE OR REPLACE FUNCTION public.run_backtest(ticker text) RETURNS text
    LANGUAGE plpython3u
    AS $$
    import plpy
    
    # 1. 清除該股票的舊回測紀錄
    plpy.execute(plpy.prepare("DELETE FROM stored_trades WHERE symbol = $1", ["text"]), [ticker])
    
    # 2. 獲取該股票的所有歷史特徵數據 (按時間排序)
    query = """
        SELECT 
            trade_date, close_price, ema_10, ema_30, 
            prev_sma_10 as prev_ema_10, prev_sma_30 as prev_ema_30, prev_close
        FROM v_stock_features 
        WHERE symbol = $1 
        ORDER BY trade_date ASC
    """
    plan = plpy.prepare(query, ["text"])
    history = plpy.execute(plan, [ticker])
    
    if not history:
        return "No data found for " + ticker

    # 初始化兩個模型的持倉倉庫
    pos_m1 = None # Model 1: 標準交叉
    pos_m2 = None # Model 2: 交叉 + 動能過濾
    
    trade_count = 0
    
    # 3. 逐日模擬交易
    for row in history:
        date = row['trade_date']
        price = row['close_price']
        
        ema_10 = row['ema_10']
        ema_30 = row['ema_30']
        prev_ema_10 = row['prev_ema_10']
        prev_ema_30 = row['prev_ema_30']
        prev_close = row['prev_close']
        
        if prev_ema_10 is None:
            continue
            
        # 計算基礎訊號
        is_golden_cross = (ema_10 > ema_30) and (prev_ema_10 <= prev_ema_30)
        is_death_cross = (ema_10 < ema_30) and (prev_ema_10 >= prev_ema_30)
        
        # Model 2 額外過濾條件
        # 買入過濾: 相對低點 (昨日收盤 < 昨日短均)
        is_dip = (prev_close < prev_ema_10)
        # 賣出過濾: 相對高點 (昨日收盤 > 昨日短均)
        is_peak = (prev_close > prev_ema_10)

        # ----------------------------------------------------
        # Model 1 邏輯 (標準策略)
        # ----------------------------------------------------
        if pos_m1 is None:
            if is_golden_cross:
                pos_m1 = {'date': date, 'price': price}
        elif pos_m1 is not None:
            if is_death_cross:
                buy_price = pos_m1['price']
                profit = (price - buy_price) / buy_price * 100
                
                ins_plan = plpy.prepare(
                    "INSERT INTO stored_trades (symbol, buy_date, buy_price, sell_date, sell_price, profit_pct, strategy_model) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    ["text", "date", "float", "date", "float", "float", "text"]
                )
                plpy.execute(ins_plan, [ticker, pos_m1['date'], buy_price, date, price, profit, "Model 1 (Standard)"])
                pos_m1 = None
                trade_count += 1

        # ----------------------------------------------------
        # Model 2 邏輯 (嚴格策略)
        # ----------------------------------------------------
        if pos_m2 is None:
            # 買入: 黃金交叉 且 位於Dip區
            if is_golden_cross and is_dip:
                pos_m2 = {'date': date, 'price': price}
        elif pos_m2 is not None:
            # 賣出: 死亡交叉 (可選: 且位於Peak區。為保護獲利，這裡我們只用標準死亡交叉出場，或者嚴格執行Peak出場)
            # 為了避免死抱活抱，這裡採用：標準死亡交叉就跑 (保守)，或者嚴格Peak才跑 (積極)。
            # 依照文章精神，我們試試看加上 Peak 過濾，如果沒這過濾可能會提早被洗出場
            
            should_sell = is_death_cross
            # 如果要非常嚴格: should_sell = is_death_cross and is_peak
            
            if should_sell:
                buy_price = pos_m2['price']
                profit = (price - buy_price) / buy_price * 100
                
                ins_plan = plpy.prepare(
                    "INSERT INTO stored_trades (symbol, buy_date, buy_price, sell_date, sell_price, profit_pct, strategy_model) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    ["text", "date", "float", "date", "float", "float", "text"]
                )
                plpy.execute(ins_plan, [ticker, pos_m2['date'], buy_price, date, price, profit, "Model 2 (Strict)"])
                pos_m2 = None
                trade_count += 1
                
    return f"Simulated {trade_count} trades for {ticker}"
$$;

ALTER FUNCTION public.run_backtest(text) OWNER TO postgres;
-- 更新策略函數，修正 Model 2 邏輯以符合使用者描述
-- 使用者描述:
-- 買入建議: 10日EMA > 30日EMA 且 週期收盤價 > 10日EMA (確認站穩強勢區)
-- 賣出建議: 10日EMA < 30日EMA 且 週期收盤價 <= 10日EMA (確認跌破弱勢區)

DROP FUNCTION IF EXISTS public.get_mssql_style_strategy(double precision, double precision, double precision, double precision, double precision);

CREATE OR REPLACE FUNCTION public.get_mssql_style_strategy(
    ema_10_today double precision, 
    ema_30_today double precision, 
    ema_10_prev double precision, 
    ema_30_prev double precision,
    close_today double precision  -- 這裡修正：根據文字描述是 "上一週期的收盤價 > 本週期的10日移動平均值" 或 "本週期" 的比較
    -- 修正理解：文章通常指 Crossing 發生後的確認。
    -- 您的文字寫：「當上一週期的收盤價大於本週期的十日移動平均值」
    -- 這裡我們傳入當天的收盤價來比較
) RETURNS TABLE(recommendation text, reason text)
    LANGUAGE plpython3u
    AS $$
    
    # 初始化
    signal = "HOLD"
    reason = "No significant trend change"
    
    # 基本交叉判斷 (Model 1)
    # 定義: 今天 10EMA > 30EMA，且 昨天 10EMA <= 30EMA
    is_golden_cross = (ema_10_today > ema_30_today) and (ema_10_prev <= ema_30_prev)
    is_death_cross = (ema_10_today < ema_30_today) and (ema_10_prev >= ema_30_prev)
    
    # Model 2 邏輯修正 (Momentum 確認)
    # 收盤價 vs 10日EMA 的位置關係
    is_above_ema = close_today > ema_10_today
    is_below_ema = close_today <= ema_10_today
    
    # 改進邏輯：根據均線排列給出明確建議
    if is_golden_cross:
        # 剛發生黃金交叉
        if is_above_ema:
            signal = "BUY (強勢買入)"
            reason = "黃金交叉 + 價格站上 10EMA (Model 2 確認)"
        else:
            signal = "BUY (觀察)"
            reason = "黃金交叉，但價格未站穩 10EMA"
            
    elif is_death_cross:
        # 剛發生死亡交叉
        if is_below_ema:
            signal = "SELL (強勢賣出)"
            reason = "死亡交叉 + 價格跌破 10EMA (Model 2 確認)"
        else:
            signal = "SELL (觀察)"
            reason = "死亡交叉，但價格仍在 10EMA 之上"
            
    elif ema_10_today > ema_30_today:
        # 多頭排列：10EMA > 30EMA
        if is_above_ema:
            signal = "BUY (持有)"
            reason = "多頭趨勢確立 (10EMA > 30EMA，價格 > 10EMA)"
        else:
            signal = "BUY (謹慎)"
            reason = "多頭趨勢中，但價格回測 10EMA"
        
    elif ema_10_today < ema_30_today:
        # 空頭排列：10EMA < 30EMA
        if is_below_ema:
            signal = "SELL (空方)"
            reason = "空頭趨勢確立 (10EMA < 30EMA，價格 < 10EMA)"
        else:
            signal = "HOLD (反彈)"
            reason = "空頭趨勢中出現反彈 (價格 > 10EMA)"

    return [(signal, reason)]
$$;

ALTER FUNCTION public.get_mssql_style_strategy(double precision, double precision, double precision, double precision, double precision) OWNER TO postgres;
CREATE OR REPLACE FUNCTION public.run_backtest(ticker text) RETURNS text
    LANGUAGE plpython3u
    AS $$
    import plpy
    
    # 1. 清除該股票的舊回測紀錄
    plpy.execute(plpy.prepare("DELETE FROM stored_trades WHERE symbol = $1", ["text"]), [ticker])
    
    # 2. 獲取該股票的所有歷史特徵數據 (按時間排序)
    query = """
        SELECT 
            trade_date, close_price, ema_10, ema_30, 
            prev_sma_10 as prev_ema_10, prev_sma_30 as prev_ema_30, prev_close
        FROM v_stock_features 
        WHERE symbol = $1 
        ORDER BY trade_date ASC
    """
    plan = plpy.prepare(query, ["text"])
    history = plpy.execute(plan, [ticker])
    
    if not history:
        return "No data found for " + ticker

    # 初始化兩個模型的持倉倉庫
    pos_m1 = None # Model 1: 標準交叉
    pos_m2 = None # Model 2: 交叉 + 強勢確認 (Price > EMA)
    
    trade_count = 0
    
    # 3. 逐日模擬交易
    for row in history:
        date = row['trade_date']
        price = row['close_price']
        
        ema_10 = row['ema_10']
        ema_30 = row['ema_30']
        prev_ema_10 = row['prev_ema_10']
        prev_ema_30 = row['prev_ema_30']
        
        # 使用者文字描述: "上一週期的收盤價" (prev_close)
        # 用來判斷是否大於 "本週期的10日移動平均值" (ema_10) 或是上一週期的 EMA
        # 為了嚴謹模擬，我們通常比較 prev_close 與 prev_ema_10，或是 prev_close 與 ema_10
        # 根據文字邏輯: "當上一週期的收盤價 (prev_close) 大於本週期的十日移動平均值 (ema_10)"
        
        prev_close = row['prev_close']
        
        if prev_ema_10 is None:
            continue
            
        # 計算基礎訊號
        is_golden_cross = (ema_10 > ema_30) and (prev_ema_10 <= prev_ema_30)
        is_death_cross = (ema_10 < ema_30) and (prev_ema_10 >= prev_ema_30)
        
        # ----------------------------------------------------
        # Model 1 邏輯 (標準策略)
        # ----------------------------------------------------
        if pos_m1 is None:
            if is_golden_cross:
                pos_m1 = {'date': date, 'price': price}
        elif pos_m1 is not None:
            # Model 1 賣出: 30日均線從下方升至上方 (即 Death Cross)
            if is_death_cross: # 這裡我們簡化處理，視同文章說的 "賣出在第二天"
                buy_price = pos_m1['price']
                profit = (price - buy_price) / buy_price * 100
                
                ins_plan = plpy.prepare(
                    "INSERT INTO stored_trades (symbol, buy_date, buy_price, sell_date, sell_price, profit_pct, strategy_model) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    ["text", "date", "float", "date", "float", "float", "text"]
                )
                plpy.execute(ins_plan, [ticker, pos_m1['date'], buy_price, date, price, profit, "Model 1 (Standard)"])
                pos_m1 = None
                trade_count += 1

        # ----------------------------------------------------
        # Model 2 邏輯 (修正後: Momentum 確認)
        # ----------------------------------------------------
        # 買入: 10日均線升至30日均線之後的第二天 且 上一週期的收盤價 > 本週期EMA10
        # 這裡我們模擬為: 當交叉發生當下(或隔天) 且 條件符合
        
        is_above_ema = (prev_close > ema_10) # 上一周期收盤 > 本週期 EMA
        is_below_ema = (prev_close <= ema_10)
        
        if pos_m2 is None:
            # 買入: 黃金交叉 且 確認站上均線
            if is_golden_cross and is_above_ema:
                pos_m2 = {'date': date, 'price': price}
        elif pos_m2 is not None:
            # 賣出: 死亡交叉 且 確認跌破均線
            # 文章文字: "當30日均線從10日均線下方升至上方... 且上一週期收盤價 <= 本週期10日EMA"
            should_sell = is_death_cross and is_below_ema
            
            # 備註: 如果只發生死叉但價格還撐在 EMA 之上，Model 2 選擇不賣出 (這能避開假跌破)
            # 但若價格真的崩了 (is_death_cross 且價格下來了)，就賣。
            
            if should_sell:
                buy_price = pos_m2['price']
                profit = (price - buy_price) / buy_price * 100
                
                ins_plan = plpy.prepare(
                    "INSERT INTO stored_trades (symbol, buy_date, buy_price, sell_date, sell_price, profit_pct, strategy_model) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    ["text", "date", "float", "date", "float", "float", "text"]
                )
                plpy.execute(ins_plan, [ticker, pos_m2['date'], buy_price, date, price, profit, "Model 2 (Confirmed)"])
                pos_m2 = None
                trade_count += 1
                
    return f"Simulated {trade_count} trades for {ticker}"
$$;

ALTER FUNCTION public.run_backtest(text) OWNER TO postgres;
-- 建立績效匯總視圖 (Summary View)
CREATE OR REPLACE VIEW public.v_strategy_performance AS
SELECT 
    symbol,
    strategy_model,
    COUNT(*) as total_trades,
    ROUND(AVG(profit_pct)::numeric, 2) as avg_profit_pct,
    ROUND(MAX(profit_pct)::numeric, 2) as max_profit_pct,
    ROUND(MIN(profit_pct)::numeric, 2) as max_loss_pct,
    ROUND((COUNT(*) FILTER (WHERE profit_pct > 0)::numeric / COUNT(*)::numeric * 100), 2) as win_rate_pct
FROM stored_trades
GROUP BY symbol, strategy_model;

-- 1. Create the missing View
CREATE OR REPLACE VIEW public.v_strategy_performance AS
SELECT 
    symbol,
    strategy_model,
    COUNT(*) as total_trades,
    ROUND(AVG(profit_pct)::numeric, 2) as avg_profit_pct,
    ROUND(MAX(profit_pct)::numeric, 2) as max_profit_pct,
    ROUND(MIN(profit_pct)::numeric, 2) as max_loss_pct,
    ROUND((COUNT(*) FILTER (WHERE profit_pct > 0)::numeric / COUNT(*)::numeric * 100), 2) as win_rate_pct
FROM stored_trades
GROUP BY symbol, strategy_model;

-- 2. Update run_backtest to include Model 1, Model 2 (Confirmed), and Model 3 (AI)
-- Modified to use next day's opening price for execution (Gap 2 fix)
CREATE OR REPLACE FUNCTION public.run_backtest(ticker text) RETURNS text
    LANGUAGE plpython3u
    AS $$
    import plpy
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    # 清除舊紀錄
    plpy.execute(plpy.prepare("DELETE FROM stored_trades WHERE symbol = $1", ["text"]), [ticker])
    
    # 獲取歷史資料（包含次日開盤價）
    query = """
        SELECT 
            trade_date, close_price, next_open, ema_10, ema_30, 
            prev_sma_10 as prev_ema_10, prev_sma_30 as prev_ema_30, prev_close
        FROM v_stock_features 
        WHERE symbol = $1 
        ORDER BY trade_date ASC
    """
    history = plpy.execute(plpy.prepare(query, ["text"]), [ticker])
    
    if not history:
        return "No data found for " + ticker

    # 初始化三個模型的持倉
    pos_m1 = None # Model 1: 標準交叉
    pos_m2 = None # Model 2: 交叉 + 確認 (Price > EMA)
    pos_m3 = None # Model 3: 交叉 + AI 預測 (Linear Regression)
    
    trade_count = 0
    price_history = [] # 用來儲存最近 N 天的股價，供 AI 訓練用
    AI_WINDOW_SIZE = 30 # AI 使用過去 30 天數據來訓練與預測
    
    for row in history:
        date = row['trade_date']
        price = row['close_price']
        next_open = row['next_open']  # 次日開盤價
        
        # 維護價格視窗 (Rolling Window)
        price_history.append(price)
        if len(price_history) > AI_WINDOW_SIZE:
            price_history.pop(0)
            
        ema_10 = row['ema_10']
        ema_30 = row['ema_30']
        prev_ema_10 = row['prev_ema_10']
        prev_ema_30 = row['prev_ema_30']
        prev_close = row['prev_close']
        
        if prev_ema_10 is None:
            continue
            
        # 基礎訊號
        is_golden_cross = (ema_10 > ema_30) and (prev_ema_10 <= prev_ema_30)
        is_death_cross = (ema_10 < ema_30) and (prev_ema_10 >= prev_ema_30)
        
        # Model 2 條件
        is_above_ema = (prev_close > ema_10)
        is_below_ema = (prev_close <= ema_10)

        # ----------------------------------------------------
        # Model 3: AI 預測邏輯
        # ----------------------------------------------------
        ai_bullish = False
        
        # 只有在出現黃金交叉時，我們才耗費運算資源去跑 AI 模型
        if is_golden_cross and len(price_history) >= 10:
            # 準備訓練資料 X (時間), Y (價格)
            y_train = np.array(price_history).reshape(-1, 1)
            x_train = np.arange(len(y_train)).reshape(-1, 1)
            
            # 訓練模型
            model = LinearRegression().fit(x_train, y_train)
            
            # 預測"明天" (next step)
            next_step = np.array([[len(y_train)]])
            predicted_price = model.predict(next_step)[0][0]
            
            # 如果 AI 預測明天價格 > 今天價格 (= 趨勢向上)
            if predicted_price > price:
                ai_bullish = True
        
        # ----------------------------------------------------
        # 執行交易邏輯 (使用次日開盤價)
        # ----------------------------------------------------
        
        # ======= Model 1 (Standard) =======
        if pos_m1 is None and is_golden_cross and next_open is not None:
            # 關鍵修正：當天產生訊號，用次日開盤價買入
            pos_m1 = {'date': date, 'price': next_open}
        elif pos_m1 is not None and is_death_cross and next_open is not None:
            # 關鍵修正：當天產生賣出訊號，用次日開盤價賣出
            buy_price = pos_m1['price']
            profit = (next_open - buy_price) / buy_price * 100
            plpy.execute(plpy.prepare("INSERT INTO stored_trades (symbol, buy_date, buy_price, sell_date, sell_price, profit_pct, strategy_model) VALUES ($1, $2, $3, $4, $5, $6, $7)", ["text", "date", "float", "date", "float", "float", "text"]), [ticker, pos_m1['date'], buy_price, date, next_open, profit, "Model 1 (Standard)"])
            pos_m1 = None
            
        # ======= Model 2 (Confirmed) =======
        if pos_m2 is None and is_golden_cross and is_above_ema and next_open is not None:
            pos_m2 = {'date': date, 'price': next_open}
        elif pos_m2 is not None and is_death_cross and is_below_ema and next_open is not None:
            buy_price = pos_m2['price']
            profit = (next_open - buy_price) / buy_price * 100
            plpy.execute(plpy.prepare("INSERT INTO stored_trades (symbol, buy_date, buy_price, sell_date, sell_price, profit_pct, strategy_model) VALUES ($1, $2, $3, $4, $5, $6, $7)", ["text", "date", "float", "date", "float", "float", "text"]), [ticker, pos_m2['date'], buy_price, date, next_open, profit, "Model 2 (Confirmed)"])
            pos_m2 = None

        # ======= Model 3 (AI Enhanced) =======
        # 買入: 黃金交叉 且 AI 說會漲
        if pos_m3 is None and is_golden_cross and ai_bullish and next_open is not None:
            pos_m3 = {'date': date, 'price': next_open}
        # 賣出: 死亡交叉 (我們保持簡單，出場還是看技術面訊號，避免被 AI 的短期波動誤導)
        elif pos_m3 is not None and is_death_cross and next_open is not None:
            buy_price = pos_m3['price']
            profit = (next_open - buy_price) / buy_price * 100
            plpy.execute(plpy.prepare("INSERT INTO stored_trades (symbol, buy_date, buy_price, sell_date, sell_price, profit_pct, strategy_model) VALUES ($1, $2, $3, $4, $5, $6, $7)", ["text", "date", "float", "date", "float", "float", "text"]), [ticker, pos_m3['date'], buy_price, date, next_open, profit, "Model 3 (AI Enhanced)"])
            pos_m3 = None
            trade_count += 1
                
    return f"Simulated trades for {ticker} (Models 1, 2, 3)"
$$;

ALTER FUNCTION public.run_backtest(text) OWNER TO postgres;
