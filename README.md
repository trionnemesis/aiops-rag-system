# AIOps 智慧維運報告 RAG 系統

## 系統架構圖

```mermaid
graph TB
    subgraph "數據採集層"
        A[Node Exporter] --> B[Prometheus]
        C[App Metrics] --> B
    end
    
    subgraph "可視化層"
        B --> D[Grafana]
    end
    
    subgraph "API 服務層"
        E[FastAPI Server]
        B --> E
    end
    
    subgraph "向量資料庫"
        F[OpenSearch 2.x<br/>with k-NN Plugin]
    end
    
    subgraph "AI 處理層"
        G[Gemini API]
        H[HyDE Generator]
        I[Summary Refiner]
        J[Report Generator]
    end
    
    subgraph "RAG Pipeline"
        E --> H
        H --> F
        F --> I
        I --> J
        J --> E
    end
    
    K[Client] --> E
    E --> K
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#f96,stroke:#333,stroke-width:2px
    style D fill:#9f9,stroke:#333,stroke-width:2px
    style E fill:#69f,stroke:#333,stroke-width:2px
    style F fill:#f69,stroke:#333,stroke-width:2px
    style G fill:#ff9,stroke:#333,stroke-width:2px
    ```
    功能特色

智慧維運報告生成：基於 HyDE + 摘要精煉的 RAG 架構
多維度監控整合：支援主機、網路、服務層級指標
向量檢索：使用 OpenSearch k-NN 進行相似度搜尋
自動化部署：GitHub Actions CI/CD Pipeline
容器化架構：Docker Compose 一鍵部署

快速開始
前置需求

Docker & Docker Compose
Gemini API Key
Python 3.9+

環境設置

Clone 專案
git clone https://github.com/your-username/aiops-rag-system.git
cd aiops-rag-system
設定環境變數
cp .env.example .env
# 編輯 .env 檔案，填入您的 Gemini API Key
啟動服務
docker-compose up -d
初始化 OpenSearch
python scripts/init_opensearch.py
API 使用
發送監控數據以生成維運報告：
curl -X POST http://localhost:8000/api/v1/generate_report \
  -H "Content-Type: application/json" \
  -d '{
    "主機": "web-prod-03",
    "採集時間": "2025-07-26T22:30:00Z",
    "CPU使用率": "75%",
    "RAM使用率": "95%",
    "磁碟I/O等待": "5%",
    "網路流出量": "350 Mbps",
    "作業系統Port流量": {
      "Port 80/443 流入連線數": 2500
    },
    "服務指標": {
      "Apache活躍工作程序": 250,
      "Nginx日誌錯誤率": {
        "502 Bad Gateway 錯誤 (每分鐘)": 45
      }
    }
  }'
  系統架構說明
1. 數據流程

監控數據收集：Prometheus 從各個 Exporter 收集指標
API 接收請求：FastAPI 接收包含監控數據的 POST 請求
HyDE 生成：使用 Gemini Flash 生成假設性事件描述
向量檢索：在 OpenSearch 中檢索相關歷史事件
摘要精煉：壓縮檢索結果，提取關鍵資訊
報告生成：使用 Gemini Pro 生成最終維運報告

2. 核心組件

Prometheus: 時序資料庫，儲存監控指標
Grafana: 監控數據可視化
FastAPI: RESTful API 服務
OpenSearch: 向量資料庫，支援 k-NN 檢索
Gemini API: Google AI 模型服務

部署指南
本地開發
# 安裝依賴
pip install -r requirements.txt

# 啟動開發伺服器
uvicorn src.main:app --reload
生產部署
使用 GitHub Actions 自動部署到您的伺服器或雲端平台。
API 文件
啟動服務後，訪問 http://localhost:8000/docs 查看完整的 API 文件。
監控面板

Grafana: http://localhost:3000 (admin/admin)
Prometheus: http://localhost:9090
OpenSearch Dashboards: http://localhost:5601