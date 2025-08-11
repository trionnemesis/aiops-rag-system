# 📚 AIOps RAG 系統文檔目錄

歡迎查閱 AIOps 智慧維運報告 RAG 系統的完整文檔！

## 🌟 系統特色

- **LangChain LCEL + LangGraph**: 聲明式 RAG 鏈與 DAG 控制流程
- **HyDE + RAG-Fusion**: 多策略文檔檢索和內容生成
- **LangExtract**: 結構化資訊提取與智慧元數據管理
- **KNN 向量搜尋**: HNSW 演算法優化，支援多種搜尋策略
  - 四種搜尋策略：純向量、混合、多向量、重新排序
  - 完整的 Pydantic 強型別驗證
  - 即時效能監控與指標追蹤
- **完整可觀測性**: 結構化日誌、分散式追蹤、度量指標收集
- **強型別化架構**: Pydantic v2 BaseModel 確保資料品質與型別安全

## 📑 文檔分類

### 🏗️ 系統架構
- [系統設計](./architecture/system-design.md) - 整體架構和核心組件說明
- [LangGraph RAG 整合](./README_LANGGRAPH_INTEGRATION.md) - LangGraph DAG 實作指南
- [LangExtract 整合指南](./langextract-integration.md) - 結構化資訊提取服務整合 🆕
- [可觀測性指南](./observability.md) - 結構化日誌、追蹤、指標詳細說明 ⚡

### 💻 開發指南
- [本地環境設置](./development/local-setup.md) - 開發環境配置指南
- [測試架構指南](./development/test-architecture.md) - LangGraph 測試策略與實踐 🆕
- [錯誤處理最佳實踐](./development/error-handling.md) - 錯誤處理機制詳解
- [重試與錯誤處理](./retry_and_error_handling.md) - 重試機制和錯誤處理策略 🆕
- [狀態持久化指南](./state_persistence_guide.md) - LangGraph 狀態管理與持久化 🆕
- [效能優化指南](./development/optimization-guide.md) - RAG 系統優化策略與實作細節
- [系統優化說明](./development/optimizations.md) - 提示工程與監控優化
- [向量檢索效能優化](./vector-performance-optimization.md) - 向量搜尋效能監控與優化 ⚡

### 🚀 部署指南
- [Docker 部署](./deployment/docker-guide.md) - 容器化部署完整指南
- [GitHub Actions 變更](./github-actions-changes.md) - CI/CD 配置更新說明

### 📡 API 文檔
- [端點參考](./api/endpoints.md) - 詳細的 API 端點說明
- [KNN 搜尋 API](./api/knn-search-api.md) - KNN 向量搜尋介面文檔

### 🔗 整合與遷移
- [LangChain 重構報告](./langchain_refactoring_report.md) - LangChain LCEL 重構詳細說明
- [LangChain 遷移指南](./langchain_migration_guide.md) - 從原實作遷移到 LCEL 指南

### 📝 其他資源
- [更新日誌](./CHANGELOG.md) - 版本更新和功能變更記錄

## 🎯 快速導航

### 新手入門
1. 🚀 從 [快速開始指南](./quick-start.md) 開始，5分鐘內啟動系統
2. 先閱讀 [系統設計](./architecture/system-design.md) 了解整體架構
3. 按照 [本地環境設置](./development/local-setup.md) 配置開發環境
4. 查看 [端點參考](./api/endpoints.md) 開始使用 API

### 進階開發
1. 學習 [測試架構指南](./development/test-architecture.md) 掌握 LangGraph 測試策略
2. 掌握 [錯誤處理最佳實踐](./development/error-handling.md) 提升程式碼品質
3. 實作 [重試與錯誤處理](./retry_and_error_handling.md) 增強系統穩定性
4. 配置 [狀態持久化](./state_persistence_guide.md) 實現工作流程狀態管理
5. 研究 [效能優化指南](./development/optimization-guide.md) 優化系統效能
6. 參考 [LangChain 重構報告](./langchain_refactoring_report.md) 了解架構演進

### 系統部署
1. 使用 [Docker 部署](./deployment/docker-guide.md) 快速部署系統
2. 配置 [GitHub Actions](./github-actions-changes.md) 實現自動化 CI/CD

### 最新功能
1. 🔥 探索 [LangGraph RAG 整合](./README_LANGGRAPH_INTEGRATION.md) 學習 DAG 控制流程
2. 🆕 了解 [LangExtract 整合指南](./langextract-integration.md) 實現結構化資訊提取
3. ⚡ 查看 [向量檢索效能優化](./vector-performance-optimization.md) 提升搜尋效能
4. 📊 配置 [可觀測性指南](./observability.md) 實現完整系統監控
5. 💾 實作 [狀態持久化](./state_persistence_guide.md) 管理工作流程狀態

## 📈 系統指標

- **測試覆蓋率**: 85%+
- **API 成本降低**: 85%
- **快取命中率**: 70%+
- **回應時間**: < 5秒 (P95)
- **向量搜尋延遲**: < 200ms (P95)
- **系統可用性**: 99.9%+

## 📝 文檔規範

- 所有文檔使用 Markdown 格式
- 包含清晰的標題層級結構
- 提供程式碼範例和實際案例
- 保持內容更新與準確性
- 使用 emoji 增強可讀性

## 🤝 貢獻文檔

歡迎改進和補充文檔！請遵循以下原則：
1. 保持格式一致性
2. 提供實用的範例
3. 確保技術準確性
4. 更新相關索引連結
5. 加入適當的 emoji 標記

---

💡 **提示**: 使用 `Ctrl+F` 快速搜尋您需要的內容！

📊 **最後更新**: 2024年1月