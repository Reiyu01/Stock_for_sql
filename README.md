# AI 股票預測與分析系統



#### 前置需求
- Docker Desktop (已安裝並運行)
- 無需安裝 Python 或 PostgreSQL

#### 啟動步驟
```powershell
# 1. 解壓縮後進入專案資料夾
cd sql_stock

# 2. 啟動所有服務（首次會自動下載映像並建立資料庫）
docker-compose up -d --build

# 3. 等待約 30 秒後，開啟瀏覽器訪問
http://localhost:8501
```

#### 停止服務
```powershell
docker-compose down
```

#### 完全重置（清空資料庫）
```powershell
docker-compose down -v
docker-compose up -d --build
```

### 3. 使用說明
1. 在側邊欄輸入股票代號（如 TSLA, AAPL, NVDA）
2. 點擊「執行預測」查看當前交易訊號
3. 點擊「執行回測模擬」查看過去 2 年的策略績效
4. 比較 Model 1 (標準策略) 和 Model 2 (確認策略) 的表現

### 4. 技術架構
- **資料庫**: PostgreSQL 16 + PL/Python3u
- **AI 引擎**: Scikit-learn (Linear Regression)
- **前端**: Streamlit
- **部署**: Docker Compose

---
開發者：UCL | 日期：2026-01-07
