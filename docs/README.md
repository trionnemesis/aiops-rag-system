# 📚 文檔目錄

本目錄包含 AIOps 智慧維運報告 RAG 系統的完整技術文檔。

## 🆕 最新文檔

### 🔗 [LangChain 重構](./langchain/)
- [重構報告](./langchain_refactoring_report.md) - LangChain LCEL 重構詳細說明
- [遷移指南](./langchain_migration_guide.md) - 從原實作遷移到 LangChain 版本

## 📁 文檔結構

### 🏗️ [系統架構](./architecture/)
- [系統設計](./architecture/system-design.md) - 整體架構和核心組件說明

### 💻 [開發指南](./development/)
- [本地環境設置](./development/local-setup.md) - 開發環境配置指南
- [優化指南](./development/optimization-guide.md) - RAG 系統優化措施
- [優化總結](./development/OPTIMIZATION_SUMMARY.md) - 優化實作總結

### 🚀 [部署指南](./deployment/)
- [Docker 部署](./deployment/docker-guide.md) - 容器化部署完整指南

### 📡 [API 文檔](./api/)
- [端點參考](./api/endpoints.md) - 詳細的 API 端點說明

## 🔗 快速導航

### 新手入門
1. 閱讀 [系統設計](./architecture/system-design.md) 了解整體架構
2. 按照 [本地環境設置](./development/local-setup.md) 配置開發環境
3. 參考 [API 端點參考](./api/endpoints.md) 開始 API 調用
4. 🆕 了解 [LangChain 重構](./langchain_refactoring_report.md) 掌握最新架構

### 部署上線
1. 閱讀 [Docker 部署指南](./deployment/docker-guide.md)
2. 配置生產環境參數
3. 執行部署和監控

### 性能優化
1. 了解 [優化指南](./development/optimization-guide.md) 中的優化原理
2. 查看 [優化總結](./development/OPTIMIZATION_SUMMARY.md) 了解實作細節
3. 監控系統性能指標

### LangChain 升級
1. 閱讀 [LangChain 重構報告](./langchain_refactoring_report.md) 了解新架構
2. 參考 [遷移指南](./langchain_migration_guide.md) 進行升級
3. 查看 `examples/langchain_rag_example.py` 學習使用方式

## 📋 文檔維護

### 貢獻指南
- 文檔遵循 Markdown 格式
- 新增功能時請同步更新相關文檔
- 圖表使用 Mermaid 語法
- 程式碼範例要包含註解

### 版本管理
- 重大變更會更新版本號
- 向後相容性變更會在文檔中標註
- 廢棄功能會有明確的遷移指南

### 反饋與改進
如發現文檔有任何問題，請：
1. 提交 GitHub Issue
2. 發送 Pull Request
3. 聯絡維護團隊

---

📝 **文檔最後更新**: 2025-01-28  
🔄 **更新頻率**: 跟隨代碼版本更新  
👥 **維護者**: 開發團隊