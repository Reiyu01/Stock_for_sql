# 1. 使用官方 PostgreSQL 16 映像檔
FROM postgres:16

# 2. 安裝 PL/Python 擴充功能與 Python 環境
RUN apt-get update && apt-get install -y \
    postgresql-plpython3-16 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. 安裝 AI 與 數據抓取需要的 Python 套件
# 補充了 yfinance，因為您的 SQL 函數中有用到它
RUN pip3 install --no-cache-dir --break-system-packages \
    numpy \
    pandas \
    scikit-learn \
    yfinance

# 4. 【關鍵步驟】將您的 SQL 備份檔放入初始化資料夾
# PostgreSQL 映像檔啟動時，會自動執行此資料夾下的所有 .sql 檔案
COPY init_stock_db.sql /docker-entrypoint-initdb.d/
