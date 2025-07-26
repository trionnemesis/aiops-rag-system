# API 端點參考

## 概述

本文檔詳細說明 AIOps RAG 系統的所有 API 端點。

## 基礎資訊

- **Base URL**: `http://localhost:8000`
- **API 版本**: v1
- **內容類型**: `application/json`
- **認證**: 目前不需要認證（開發環境）

## API 端點清單

### 1. 健康檢查

#### `GET /health`

檢查 API 服務健康狀態。

**請求範例**:
```bash
curl http://localhost:8000/health
```

**響應範例**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-26T10:30:00Z",
  "services": {
    "opensearch": "connected",
    "gemini": "available"
  }
}
```

### 2. 報告生成

#### `POST /api/v1/generate_report`

基於監控數據生成智慧維運報告。

**請求參數**:

| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `monitoring_data` | object | 是 | 監控數據物件 |

**監控數據結構**:
```json
{
  "monitoring_data": {
    "主機": "string",           // 主機名稱
    "採集時間": "string",       // ISO 8601 格式時間
    "CPU使用率": "string",      // CPU 使用率百分比
    "RAM使用率": "string",      // 記憶體使用率百分比
    "磁碟I/O等待": "string",    // 磁碟 I/O 等待百分比
    "網路流出量": "string",     // 網路流出量
    "作業系統Port流量": {
      "Port 80/443 流入連線數": "number"
    },
    "服務指標": {
      "Apache活躍工作程序": "number",
      "Nginx日誌錯誤率": {
        "502 Bad Gateway 錯誤 (每分鐘)": "number"
      }
    }
  }
}
```

**請求範例**:
```bash
curl -X POST http://localhost:8000/api/v1/generate_report \
  -H "Content-Type: application/json" \
  -d '{
    "monitoring_data": {
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
    }
  }'
```

**響應範例**:
```json
{
  "status": "success",
  "report": {
    "summary": "主機 web-prod-03 出現高記憶體使用率警告...",
    "timestamp": "2025-01-26T10:30:00Z",
    "severity": "warning",
    "recommendations": [
      "建議立即檢查記憶體洩漏問題",
      "考慮重啟相關服務程序",
      "監控後續記憶體使用趨勢"
    ],
    "technical_details": {
      "affected_services": ["Apache", "Nginx"],
      "root_cause_analysis": "記憶體使用率達到 95%，可能影響服務穩定性"
    }
  },
  "processing_time": "2.34s",
  "cache_hit": false
}
```

### 3. 快取管理

#### `GET /api/v1/cache/info`

取得快取統計資訊。

**請求範例**:
```bash
curl http://localhost:8000/api/v1/cache/info
```

**響應範例**:
```json
{
  "status": "success",
  "cache_info": {
    "hyde_cache": {
      "hits": 45,
      "misses": 10,
      "maxsize": 50,
      "currsize": 35,
      "hit_rate": "81.82%"
    },
    "embedding_cache": {
      "hits": 120,
      "misses": 30,
      "maxsize": 100,
      "currsize": 75,
      "hit_rate": "80.00%"
    }
  }
}
```

#### `POST /api/v1/cache/clear`

清除所有快取。

**請求範例**:
```bash
curl -X POST http://localhost:8000/api/v1/cache/clear
```

**響應範例**:
```json
{
  "status": "success",
  "message": "All caches have been cleared",
  "cleared_caches": ["hyde_cache", "embedding_cache"]
}
```

### 4. 系統資訊

#### `GET /api/v1/info`

取得系統資訊和版本。

**請求範例**:
```bash
curl http://localhost:8000/api/v1/info
```

**響應範例**:
```json
{
  "status": "success",
  "system_info": {
    "version": "1.0.0",
    "environment": "development",
    "uptime": "2h 30m 45s",
    "python_version": "3.9.18",
    "fastapi_version": "0.104.1"
  },
  "features": {
    "hyde_optimization": true,
    "cache_enabled": true,
    "vector_search": true
  }
}
```

## 錯誤處理

### HTTP 狀態碼

| 狀態碼 | 說明 |
|--------|------|
| 200 | 成功 |
| 400 | 請求參數錯誤 |
| 500 | 伺服器內部錯誤 |
| 503 | 服務暫時無法使用 |

### 錯誤響應格式

```json
{
  "status": "error",
  "error": {
    "code": "INVALID_REQUEST",
    "message": "監控數據格式不正確",
    "details": "必填欄位 '主機' 缺失"
  },
  "timestamp": "2025-01-26T10:30:00Z"
}
```

### 常見錯誤

#### 1. 無效的監控數據

**錯誤**:
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "監控數據驗證失敗",
    "details": "CPU使用率格式不正確，應為百分比字串"
  }
}
```

#### 2. Gemini API 錯誤

**錯誤**:
```json
{
  "status": "error",
  "error": {
    "code": "GEMINI_API_ERROR",
    "message": "Gemini API 呼叫失敗",
    "details": "API 配額已用盡或網路連線問題"
  }
}
```

#### 3. OpenSearch 連線錯誤

**錯誤**:
```json
{
  "status": "error",
  "error": {
    "code": "OPENSEARCH_CONNECTION_ERROR",
    "message": "無法連線到 OpenSearch",
    "details": "檢查 OpenSearch 服務是否正常運行"
  }
}
```

## 請求限制

### 速率限制

- 每分鐘最多 60 個請求
- 每小時最多 1000 個請求
- 超過限制將返回 429 Too Many Requests

### 請求大小限制

- 最大請求體大小：10MB
- 監控資料項目數量：最多 100 個

## 互動式文檔

啟動服務後，可以通過以下網址訪問互動式 API 文檔：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## SDK 和程式碼範例

### Python 範例

```python
import httpx
import asyncio

async def generate_report(monitoring_data):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/generate_report",
            json={"monitoring_data": monitoring_data}
        )
        return response.json()

# 使用範例
monitoring_data = {
    "主機": "web-prod-01",
    "採集時間": "2025-01-26T10:30:00Z",
    "CPU使用率": "80%",
    "RAM使用率": "90%"
}

result = asyncio.run(generate_report(monitoring_data))
print(result["report"]["summary"])
```

### JavaScript 範例

```javascript
const generateReport = async (monitoringData) => {
  const response = await fetch('http://localhost:8000/api/v1/generate_report', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ monitoring_data: monitoringData })
  });
  
  return await response.json();
};

// 使用範例
const monitoringData = {
  "主機": "web-prod-01",
  "採集時間": "2025-01-26T10:30:00Z",
  "CPU使用率": "80%",
  "RAM使用率": "90%"
};

generateReport(monitoringData)
  .then(result => console.log(result.report.summary));
```

## 最佳實踐

### 1. 請求優化

- 重複的監控數據會觸發快取，提升響應速度
- 批次處理多個監控事件可降低 API 呼叫次數
- 定期檢查快取命中率，調整快取策略

### 2. 錯誤處理

- 實作重試機制處理暫時性錯誤
- 記錄 API 響應用於除錯
- 監控 API 回應時間和錯誤率

### 3. 安全性

- 在生產環境中啟用 HTTPS
- 實作 API 金鑰認證
- 設定適當的 CORS 政策