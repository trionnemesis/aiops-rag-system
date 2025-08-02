# 📚 AIOps RAG 系統文檔目錄

歡迎查閱 AIOps 智慧維運報告 RAG 系統的完整文檔！

## 📑 文檔分類

### 🏗️ 系統架構
- [系統設計](./architecture/system-design.md) - 整體架構和核心組件說明
- [LangGraph RAG 整合](./README_LANGGRAPH_INTEGRATION.md) - LangGraph DAG 實作指南
- [LangExtract 整合指南](./langextract-integration.md) - 結構化資訊提取服務整合 🆕

### 💻 開發指南
- [本地環境設置](./development/local-setup.md) - 開發環境配置指南
- [錯誤處理最佳實踐](./development/error-handling.md) - 錯誤處理機制詳解
- [效能優化指南](./development/optimization-guide.md) - RAG 系統優化策略
- [系統優化說明](./development/optimizations.md) - 優化實作細節
- [優化總結](./development/OPTIMIZATION_SUMMARY.md) - 優化成果總覽
- [向量檢索效能優化](./vector-performance-optimization.md) - 向量搜尋效能監控與優化

### 🚀 部署指南
- [Docker 部署](./deployment/docker-guide.md) - 容器化部署完整指南

### 📡 API 文檔
- [端點參考](./api/endpoints.md) - 詳細的 API 端點說明
- [KNN 搜尋 API](./api/knn-search-api.md) - KNN 向量搜尋介面文檔

### 🔗 整合與遷移
- [LangChain 重構報告](./langchain_refactoring_report.md) - LangChain LCEL 重構詳細說明
- [LangChain 遷移指南](./langchain_migration_guide.md) - 從原實作遷移到 LCEL 指南
- [GitHub Actions 變更](./github-actions-changes.md) - CI/CD 配置更新說明

## 🎯 快速導航

### 新手入門
1. 先閱讀 [系統設計](./architecture/system-design.md) 了解整體架構
2. 按照 [本地環境設置](./development/local-setup.md) 配置開發環境
3. 查看 [端點參考](./api/endpoints.md) 開始使用 API

### 進階開發
1. 學習 [錯誤處理最佳實踐](./development/error-handling.md) 提升程式碼品質
2. 研究 [效能優化指南](./development/optimization-guide.md) 優化系統效能
3. 參考 [LangChain 重構報告](./langchain_refactoring_report.md) 了解架構演進

### 系統部署
1. 使用 [Docker 部署](./deployment/docker-guide.md) 快速部署系統
2. 配置 [GitHub Actions](./github-actions-changes.md) 實現自動化 CI/CD

### 最新功能
1. 🔥 探索 [LangGraph RAG 整合](./README_LANGGRAPH_INTEGRATION.md) 學習 DAG 控制流程
2. 🆕 了解 [LangExtract 整合指南](./langextract-integration.md) 實現結構化資訊提取
3. ⚡ 查看 [向量檢索效能優化](./vector-performance-optimization.md) 提升搜尋效能

## 📝 文檔規範

- 所有文檔使用 Markdown 格式
- 包含清晰的標題層級結構
- 提供程式碼範例和實際案例
- 保持內容更新與準確性

## 🤝 貢獻文檔

歡迎改進和補充文檔！請遵循以下原則：
1. 保持格式一致性
2. 提供實用的範例
3. 確保技術準確性
4. 更新相關索引連結

---

💡 **提示**: 使用 `Ctrl+F` 快速搜尋您需要的內容！