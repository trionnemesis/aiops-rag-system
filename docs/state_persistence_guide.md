# 狀態持久化與恢復指南

## 概述

LangGraph 支援狀態持久化功能，讓您可以：
- 在長時間運行的任務中保存執行狀態
- 從中斷點恢復執行，而不需要重新開始
- 追蹤對話歷史和執行軌跡

## 設置步驟

### 1. 啟動 Redis 服務

```bash
# 使用 docker-compose 啟動所有服務（包含 Redis）
docker-compose up -d

# 或只啟動 Redis
docker-compose up -d redis
```

### 2. 環境變數設定

確保設定了 `REDIS_URL` 環境變數：

```bash
export REDIS_URL=redis://localhost:6379
```

或在 `.env` 檔案中：

```
REDIS_URL=redis://redis:6379  # Docker 環境
```

### 3. 使用 Thread ID

在調用 graph 時，必須提供 `thread_id` 來識別對話：

```python
result = graph_app.invoke(
    initial_state,
    config={
        "configurable": {
            "thread_id": "unique-thread-id",  # 必須提供
            "run_id": "optional-run-id"
        }
    }
)
```

## 使用範例

### 基本使用

```python
from app.graph.build import build_graph

# 建構 graph（會自動使用 Redis 如果 REDIS_URL 已設定）
app = build_graph(
    llm=llm,
    retriever=retriever,
    policy={"max_retries": 3}
)

# 執行並保存狀態
thread_id = f"user-{user_id}-{timestamp}"
result = app.invoke(
    {"query": "複雜的查詢..."},
    config={"configurable": {"thread_id": thread_id}}
)
```

### 從中斷點恢復

```python
# 如果執行中斷，使用相同的 thread_id 可以恢復
try:
    result = app.invoke(
        {"query": "原始查詢"},
        config={"configurable": {"thread_id": thread_id}}
    )
except Exception as e:
    print(f"執行中斷: {e}")
    
# 稍後恢復執行
result = app.invoke(
    {"query": "原始查詢", "_resume": True},
    config={"configurable": {"thread_id": thread_id}}
)
```

### 檢查保存的狀態

```python
import redis

r = redis.from_url(os.getenv("REDIS_URL"))

# 列出所有 threads
for key in r.scan_iter("checkpoint:*"):
    print(key.decode())

# 獲取特定 thread 的狀態
thread_state = r.get(f"checkpoint:{thread_id}")
```

## API 整合

在 FastAPI 路由中已自動整合：

```python
@router.post("/api/rag/query")
async def rag_query(req: RAGRequest):
    # 自動使用 request_id 作為 thread_id 的一部分
    cfg = {
        "configurable": {
            "thread_id": f"thread-{request_id}",
            "run_id": request_id
        }
    }
    
    result = graph_app.invoke(initial_state, config=cfg)
```

## 最佳實踐

### 1. Thread ID 設計

- 使用有意義的 ID：`f"user-{user_id}-{session_id}"`
- 避免使用隨機 UUID（除非您有外部映射）
- 考慮包含時間戳以便管理

### 2. 狀態清理

```python
# 定期清理舊的 checkpoints
def cleanup_old_checkpoints(days=7):
    cutoff = time.time() - (days * 86400)
    for key in r.scan_iter("checkpoint:*"):
        # 檢查時間戳並刪除舊的
        pass
```

### 3. 錯誤處理

```python
def safe_invoke_with_retry(app, state, thread_id, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return app.invoke(
                state,
                config={"configurable": {"thread_id": thread_id}}
            )
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 ** attempt)  # 指數退避
```

## 監控與除錯

### 查看 Redis 中的資料

```bash
# 連接到 Redis CLI
docker exec -it aiops-redis redis-cli

# 查看所有 checkpoint keys
KEYS checkpoint:*

# 查看特定 checkpoint
GET checkpoint:thread-123

# 查看 checkpoint 大小
MEMORY USAGE checkpoint:thread-123
```

### 效能考量

1. **Checkpoint 大小**：大型狀態會影響效能
2. **頻率**：每個節點執行後都會保存狀態
3. **清理策略**：定期清理不需要的 checkpoints

## 故障排除

### Redis 連接失敗

如果看到 "Failed to connect to Redis, falling back to MemorySaver"：

1. 檢查 Redis 是否運行：`docker ps | grep redis`
2. 檢查 REDIS_URL 是否正確
3. 檢查網路連接

### 狀態恢復失敗

1. 確保使用相同的 thread_id
2. 檢查狀態是否已過期或被清理
3. 驗證 Redis 中是否有對應的 checkpoint

## 進階功能

### 自定義 Checkpoint 策略

```python
from langgraph.checkpoint.redis import RedisSaver

# 自定義 TTL
class CustomRedisSaver(RedisSaver):
    def put(self, config, checkpoint, metadata):
        super().put(config, checkpoint, metadata)
        # 設定 30 天過期
        key = self._get_key(config)
        self.client.expire(key, 30 * 86400)
```

### 狀態版本控制

```python
# 保存多個版本的狀態
thread_id_versioned = f"{thread_id}:v{version}"
```

## 總結

狀態持久化功能讓 LangGraph 應用更加健壯，特別適合：
- 長時間運行的複雜任務
- 需要故障恢復的關鍵應用
- 多輪對話的上下文保持
- 執行軌跡的審計和分析