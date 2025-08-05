# 🚀 快速開始指南

本指南將幫助您在 5 分鐘內啟動 AIOps RAG 系統。

## 📋 前置需求

- Docker 和 Docker Compose (推薦)
- Python 3.8+ (本地開發)
- Gemini API Key ([取得 API Key](https://makersuite.google.com/app/apikey))

## 🐳 使用 Docker 快速啟動（推薦）

### 1. Clone 專案

```bash
git clone https://github.com/[your-org]/aiops-rag-system.git
cd aiops-rag-system
```

### 2. 設定環境變數

```bash
# 複製環境變數範本
cp .env.example .env

# 編輯 .env 檔案，設定您的 Gemini API Key
# 最少需要設定：GEMINI_API_KEY=your-api-key-here
```

### 3. 啟動所有服務

```bash
# 啟動所有服務（第一次會需要較長時間建置映像）
docker-compose up -d

# 檢查服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f app
```

### 4. 驗證服務

```bash
# 健康檢查
curl http://localhost:8000/health

# 應該返回：
# {"status": "healthy"}
```

### 5. 測試 RAG API

```bash
# 生成維運報告
curl -X POST http://localhost:8000/api/v1/rag/report \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何解決 Kubernetes Pod OOMKilled 問題",
    "context": {
      "system": "kubernetes",
      "environment": "production"
    }
  }'
```

## 💻 本地開發環境

### 1. 安裝依賴

```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
```

### 2. 啟動 OpenSearch（使用 Docker）

```bash
# 只啟動 OpenSearch 和 Redis
docker-compose up -d opensearch redis
```

### 3. 設定環境變數

```bash
# 載入環境變數
export $(cat .env | xargs)

# 或使用 python-dotenv
python -c "from dotenv import load_dotenv; load_dotenv()"
```

### 4. 啟動應用程式

```bash
# 開發模式（含自動重載）
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 或直接執行
python main.py
```

## 🌐 存取服務

啟動後，您可以存取以下服務：

| 服務 | URL | 說明 |
|------|-----|------|
| API 文檔 | http://localhost:8000/docs | Swagger UI |
| Grafana | http://localhost:3000 | 監控儀表板 (admin/admin) |
| Prometheus | http://localhost:9090 | 指標收集 |
| Jaeger UI | http://localhost:16686 | 分散式追蹤 |
| OpenSearch | http://localhost:5601 | 向量資料庫儀表板 |

## 🧪 執行測試

```bash
# 執行所有測試
pytest

# 執行測試並查看覆蓋率
pytest --cov=app --cov-report=html

# 只執行特定測試
pytest tests/test_rag_service.py -v
```

## 🔧 常見問題

### 1. Gemini API Key 錯誤

確保您的 API Key 正確設定在 `.env` 檔案中：
```bash
GEMINI_API_KEY=your-actual-api-key-here
```

### 2. OpenSearch 連線失敗

檢查 OpenSearch 是否正常運行：
```bash
curl http://localhost:9200/_cluster/health
```

### 3. 記憶體不足

調整 Docker Compose 中的記憶體限制：
```yaml
services:
  opensearch:
    environment:
      - OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m
```

## 📚 下一步

- 閱讀[系統架構](./architecture/system-design.md)了解詳細設計
- 查看[API 文檔](./api/endpoints.md)了解所有端點
- 探索[範例程式碼](../examples/)學習進階用法

## 🆘 需要幫助？

- 查看[完整文檔](./README.md)
- 提交 [Issue](https://github.com/[your-org]/aiops-rag-system/issues)
- 參考[常見問題](./faq.md)

---

🎉 恭喜！您已成功啟動 AIOps RAG 系統。開始探索強大的 AI 維運功能吧！