# AIOps 智慧維運報告 RAG 系統

一個基於 HyDE (假設性文件嵌入) 和摘要精煉技術的智慧維運報告生成系統，透過 RAG 架構自動分析監控數據並生成專業的維運報告。

## 🆕 最新更新 - LangChain LCEL 重構

我們已使用 **LangChain 表達式語言 (LCEL)** 完成系統重構，大幅提升程式碼的可讀性和擴展性：

- 🔗 **LCEL 管道式流程** - 使用聲明式語法定義 RAG 流程
- 🎯 **統一模型管理** - 標準化的 LangChain 模型介面
- 🗄️ **向量資料庫抽象** - 輕鬆切換不同的向量資料庫
- ⚡ **保持向後相容** - 原有 API 介面完全不變

👉 **[查看 LangChain 重構報告](./docs/langchain_refactoring_report.md)**

## 🏗️ 系統架構概覽

本系統採用多層架構設計，包含數據採集、AI 處理、向量檢索和報告生成等核心組件。

**核心特色**：
- 🤖 **智慧分析**: 基於 HyDE + 摘要精煉的 RAG 架構
- 📊 **多維監控**: 支援主機、網路、服務層級指標整合  
- 🔍 **向量檢索**: 使用 OpenSearch k-NN 進行相似度搜尋
- ⚡ **效能優化**: 85% API 成本節省，70%+ 快取命中率
- 🐳 **容器化**: Docker Compose 一鍵部署
- 🔗 **LangChain 整合**: 使用 LCEL 實現優雅的 RAG 流程

**技術棧**：FastAPI + LangChain + OpenSearch + Gemini API + Prometheus + Grafana

## ✨ 主要功能

- **🤖 智慧維運報告生成**：基於 HyDE + 摘要精煉的 RAG 架構
- **📊 多維度監控整合**：支援主機、網路、服務層級指標
- **🔍 向量檢索**：使用 OpenSearch k-NN 進行相似度搜尋
- **🚀 自動化部署**：GitHub Actions CI/CD Pipeline
- **📈 效能監控**：Grafana 儀表板和快取狀態監控
- **🔗 LangChain LCEL**：聲明式的 RAG 流程定義

## 🚀 快速開始

### 前置需求

- Docker & Docker Compose
- Gemini API Key
- Python 3.9+（本地開發）

### 1. 快速部署

```bash
# Clone 專案
git clone https://github.com/your-username/aiops-rag-system.git
cd aiops-rag-system

# 設定環境變數
cp .env.example .env
# 編輯 .env 檔案，填入您的 Gemini API Key

# 啟動所有服務
docker-compose up -d

# 初始化 OpenSearch
python scripts/init_opensearch.py
```

### 2. 測試 API

```bash
curl -X POST http://localhost:8000/api/v1/generate_report \
  -H "Content-Type: application/json" \
  -d '{
    "monitoring_data": {
      "主機": "web-prod-03",
      "採集時間": "2025-01-26T10:30:00Z",
      "CPU使用率": "75%",
      "RAM使用率": "95%",
      "磁碟I/O等待": "5%"
    }
  }'
```

### 3. 使用 LangChain 組件

```python
from src.services.langchain import RAGChainService, model_manager

# 使用新的 LangChain RAG 服務
rag_service = RAGChainService()
report = await rag_service.generate_report(monitoring_data)

# 直接使用模型管理器
model = model_manager.pro_model
response = await model.ainvoke("你的提示詞")
```

### 4. 訪問監控介面

| 服務 | 網址 | 帳密 |
|------|------|------|
| **API 文檔** | http://localhost:8000/docs | - |
| **Grafana** | http://localhost:3000 | admin/admin |
| **Prometheus** | http://localhost:9090 | - |
| **OpenSearch** | http://localhost:5601 | admin/admin |

## 📚 詳細文檔

### 📋 文檔導航

- **📖 [完整文檔目錄](./docs/README.md)** - 所有技術文檔的入口
- **🆕 [LangChain 重構報告](./docs/langchain_refactoring_report.md)** - LCEL 重構詳細說明
- **🏗️ [系統架構設計](./docs/architecture/system-design.md)** - 詳細的架構說明和組件介紹  
- **💻 [開發環境設置](./docs/development/local-setup.md)** - 本地開發環境配置
- **🚀 [Docker 部署指南](./docs/deployment/docker-guide.md)** - 生產環境部署
- **📡 [API 端點參考](./docs/api/endpoints.md)** - 完整的 API 文檔

### 🔧 開發和優化

- **⚡ [系統優化指南](./docs/development/optimization-guide.md)** - 詳細的優化原理和方法
- **📊 [優化實作總結](./docs/development/OPTIMIZATION_SUMMARY.md)** - 優化效果和成本分析

## 🔧 核心組件

| 組件 | 功能 | 端口 |
|------|------|------|
| **FastAPI** | RESTful API 服務 | 8000 |
| **LangChain** | RAG 流程管理 | - |
| **OpenSearch** | 向量資料庫 | 9200 |
| **Prometheus** | 監控數據收集 | 9090 |
| **Grafana** | 監控儀表板 | 3000 |
| **Gemini API** | AI 模型服務 | - |

## 📊 效能指標

經過優化後的系統效能：

- **API 成本節省**: 85% （透過快取和批次處理）
- **快取命中率**: 70%+ （HyDE 和嵌入向量快取）
- **響應時間**: < 5 秒 (95th percentile)
- **API 呼叫減少**: 從 8000 次降至 1200 次（1000 個請求）

## 💻 開發指南

### 本地開發

```bash
# 設定虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安裝依賴
pip install -r requirements.txt

# 設定 PYTHONPATH
export PYTHONPATH=$PWD

# 啟動開發伺服器
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 執行測試

```bash
# 執行所有測試
pytest tests/ -v

# 測試覆蓋率
pytest tests/ --cov=src --cov-report=html
```

### 使用 LangChain 範例

```python
# 查看完整範例
python examples/langchain_rag_example.py
```

## 🛠️ 故障排除

### 常見問題

1. **OpenSearch 連線失敗**
   - 確認 Docker 服務已啟動
   - 檢查 `.env` 檔案設定

2. **Gemini API 錯誤**
   - 確認 API Key 正確
   - 檢查 API 配額

3. **模組導入錯誤**
   ```bash
   export PYTHONPATH=$PWD
   ```

更多故障排除資訊請參閱 [Docker 部署指南](./docs/deployment/docker-guide.md#故障排除)。

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！請確保：

1. 遵循現有的程式碼風格
2. 添加適當的測試
3. 更新相關文件
4. 遵循 [Conventional Commits](https://www.conventionalcommits.org/) 規範

## 📝 授權

本專案採用 MIT 授權條款。詳細內容請參閱 [LICENSE](LICENSE) 檔案。

## 📞 聯絡資訊

- 📧 **Email**: your-email@example.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/aiops-rag-system/issues)
- 💬 **討論**: [GitHub Discussions](https://github.com/your-username/aiops-rag-system/discussions)

---

⭐ 如果這個專案對您有幫助，請給我們一顆星星！