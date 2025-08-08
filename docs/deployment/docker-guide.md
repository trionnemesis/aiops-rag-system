# Docker 部署指南

## 概述

本指南說明如何使用 Docker 和 Docker Compose 部署 AIOps RAG 系統。

## 前置需求

- Docker Engine 20.10+
- Docker Compose v2.0+
- 至少 4GB 可用記憶體
- 至少 10GB 可用磁碟空間

## 快速部署

### 1. 準備環境

```bash
# Clone 專案
git clone https://github.com/your-username/aiops-rag-system.git
cd aiops-rag-system

# 設定環境變數
cp .env.example .env
nano .env  # 編輯必要的環境變數
```

### 2. 啟動所有服務

```bash
# 建置並啟動所有服務
docker-compose up -d --build

# 檢查服務狀態
docker-compose ps

# 檢查健康狀態（所有服務都已配置健康檢查）
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Health}}"
```

> **注意**: 系統已配置完整的健康檢查和重啟策略：
> - 所有服務都有 `restart: on-failure` 政策
> - 主要服務（app、opensearch、redis、prometheus）包含健康檢查
> - 服務依賴使用 `condition: service_healthy` 確保啟動順序

### 3. 初始化系統

```bash
# 等待 OpenSearch 完全啟動（約 2-3 分鐘）
docker-compose logs -f opensearch

# 初始化 OpenSearch 索引
python scripts/init_opensearch.py
```

## 服務清單

| 服務名稱 | 連接埠 | 說明 |
|----------|--------|------|
| **api** | 8000 | FastAPI 應用服務 |
| **opensearch** | 9200, 9600 | 向量資料庫 |
| **opensearch-dashboards** | 5601 | OpenSearch 管理介面 |
| **prometheus** | 9090 | 監控數據收集 |
| **grafana** | 3000 | 監控儀表板 |
| **node-exporter** | 9100 | 主機指標收集 |

## 詳細設定

### Docker Compose 檔案結構

```yaml
# docker-compose.yml 主要服務
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENSEARCH_HOST=opensearch
    depends_on:
      - opensearch
      
  opensearch:
    image: opensearchproject/opensearch:2.12.0
    environment:
      - discovery.type=single-node
      - OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g
    
  # ... 其他服務
```

### 環境變數設定

必要的環境變數（`.env` 檔案）：

```bash
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# OpenSearch 設定
OPENSEARCH_HOST=opensearch
OPENSEARCH_PORT=9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=admin

# 應用設定
APP_ENV=production
LOG_LEVEL=INFO
```

## 生產環境部署

### 1. 安全性設定

```bash
# 修改預設密碼
# 編輯 docker-compose.yml 中的密碼設定
nano docker-compose.yml
```

重要的安全設定：
- 更改 OpenSearch 預設密碼
- 設定防火牆規則
- 使用 HTTPS（建議配置 Nginx 反向代理）
- 定期更新容器映像

### 2. 效能調優

#### OpenSearch 記憶體設定

```yaml
# docker-compose.yml
opensearch:
  environment:
    - OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g  # 調整記憶體
```

#### API 服務擴展

```yaml
# 多個 API 實例
api:
  deploy:
    replicas: 3  # 啟動 3 個實例
```

### 3. 資料持久化

確保重要資料持久化：

```yaml
volumes:
  opensearch_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
```

## 監控和日誌

### 1. 檢查服務狀態

```bash
# 查看所有服務狀態
docker-compose ps

# 查看特定服務日誌
docker-compose logs -f api
docker-compose logs -f opensearch

# 查看資源使用情況
docker stats
```

### 2. 健康檢查

```bash
# API 服務健康檢查
curl http://localhost:8000/health

# OpenSearch 健康檢查
curl http://localhost:9200/_cluster/health

# Prometheus 檢查
curl http://localhost:9090/-/healthy
```

### 3. 監控儀表板

訪問以下網址查看監控資料：

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **OpenSearch Dashboards**: http://localhost:5601 (admin/admin)

## 故障排除

### 1. 常見問題

#### OpenSearch 啟動失敗

```bash
# 檢查 OpenSearch 日誌
docker-compose logs opensearch

# 常見原因：記憶體不足
# 解決方案：調整 OPENSEARCH_JAVA_OPTS 或增加系統記憶體
```

#### API 服務連線錯誤

```bash
# 檢查網路連線
docker-compose exec api ping opensearch

# 檢查環境變數
docker-compose exec api env | grep OPENSEARCH
```

#### 容器無法啟動

```bash
# 清理舊容器和映像
docker-compose down -v
docker system prune -a

# 重新建置
docker-compose up -d --build
```

### 2. 效能問題

#### 記憶體不足

```bash
# 檢查記憶體使用
docker stats

# 調整 Java 堆記憶體設定
# 修改 docker-compose.yml 中的 OPENSEARCH_JAVA_OPTS
```

#### 磁碟空間不足

```bash
# 檢查磁碟使用
df -h
docker system df

# 清理無用的映像和容器
docker system prune -a
```

## 備份和復原

### 1. 資料備份

```bash
# 備份 OpenSearch 資料
docker-compose exec opensearch curl -X PUT "localhost:9200/_snapshot/backup_repo" -H 'Content-Type: application/json' -d'
{
  "type": "fs",
  "settings": {
    "location": "/backup"
  }
}'

# 建立快照
docker-compose exec opensearch curl -X PUT "localhost:9200/_snapshot/backup_repo/snapshot_1"
```

### 2. 設定檔備份

```bash
# 備份重要設定檔
tar -czf backup_$(date +%Y%m%d).tar.gz .env docker-compose.yml configs/
```

## 升級指南

### 1. 應用升級

```bash
# 1. 備份現有資料
docker-compose exec opensearch curl -X PUT "localhost:9200/_snapshot/backup_repo/pre_upgrade"

# 2. 停止服務
docker-compose down

# 3. 更新程式碼
git pull origin main

# 4. 重新建置並啟動
docker-compose up -d --build

# 5. 驗證服務
curl http://localhost:8000/health
```

### 2. OpenSearch 升級

```bash
# 1. 停止服務
docker-compose stop opensearch

# 2. 修改 docker-compose.yml 中的版本
# opensearchproject/opensearch:2.12.0 -> opensearchproject/opensearch:2.13.0

# 3. 啟動新版本
docker-compose up -d opensearch

# 4. 檢查叢集狀態
curl http://localhost:9200/_cluster/health
```

## 最佳實踐

### 1. 安全性

- 定期更新容器映像
- 使用非 root 使用者執行服務
- 配置防火牆和網路隔離
- 啟用 HTTPS 和身份認證

### 2. 效能

- 根據負載調整服務副本數
- 監控資源使用情況
- 定期清理日誌和暫存檔案
- 使用 SSD 儲存向量資料

### 3. 維護

- 設定自動備份計畫
- 監控服務健康狀態
- 建立告警機制
- 定期執行安全性掃描