# 開發環境設置指南

## 前置需求

- Python 3.9+
- Docker & Docker Compose
- Git
- Gemini API Key

## 本地開發環境設置

### 1. Clone 專案

```bash
git clone https://github.com/your-username/aiops-rag-system.git
cd aiops-rag-system
```

### 2. 設定 Python 虛擬環境

```bash
# 創建虛擬環境
python -m venv venv

# 啟動虛擬環境
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
```

### 3. 環境變數設定

```bash
# 複製環境變數範本
cp .env.example .env

# 編輯 .env 檔案，填入必要的設定
nano .env
```

必要的環境變數：
- `GEMINI_API_KEY`: Google Gemini API 金鑰
- `OPENSEARCH_HOST`: OpenSearch 主機位址
- `OPENSEARCH_PORT`: OpenSearch 連接埠
- `OPENSEARCH_USERNAME`: OpenSearch 使用者名稱
- `OPENSEARCH_PASSWORD`: OpenSearch 密碼

### 4. 啟動開發服務

```bash
# 設定 PYTHONPATH
export PYTHONPATH=$PWD

# 啟動 OpenSearch 和 Prometheus
docker-compose up -d opensearch prometheus grafana

# 啟動開發伺服器
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 初始化資料庫

```bash
# 初始化 OpenSearch 索引
python scripts/init_opensearch.py
```

## 開發工具

### 程式碼格式化

```bash
# 安裝開發工具
pip install black isort flake8

# 格式化程式碼
black src/ tests/
isort src/ tests/

# 檢查程式碼品質
flake8 src/ tests/ --max-line-length=127
```

### 執行測試

```bash
# 執行所有測試
pytest tests/ -v

# 執行特定測試文件
pytest tests/test_api.py -v

# 執行測試並產生覆蓋率報告
pytest tests/ --cov=src --cov-report=html
```

### 除錯設定

#### VS Code 設定

在 `.vscode/launch.json` 中加入：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FastAPI Debug",
            "type": "python",
            "request": "launch",
            "program": "-m",
            "args": ["uvicorn", "src.main:app", "--reload"],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        }
    ]
}
```

## 常見問題

### 1. 模組導入錯誤

```bash
# 確保設定 PYTHONPATH
export PYTHONPATH=$PWD
```

### 2. OpenSearch 連線失敗

- 確認 Docker 服務已啟動
- 檢查 `.env` 檔案中的連線設定
- 確認防火牆設定

### 3. Gemini API 錯誤

- 確認 API Key 正確且有效
- 檢查 API 配額和使用限制
- 確認網路連線正常

### 4. 依賴安裝問題

```bash
# 升級 pip
pip install --upgrade pip

# 清除快取重新安裝
pip cache purge
pip install -r requirements.txt --force-reinstall
```

## 開發流程

### 1. 功能開發

1. 從 `main` 分支建立新的功能分支
2. 實作功能並編寫測試
3. 確保所有測試通過
4. 執行程式碼品質檢查
5. 提交 Pull Request

### 2. 提交規範

遵循 Conventional Commits 規範：

```
type(scope): description

feat(api): add new report generation endpoint
fix(rag): resolve embedding cache issue
docs(readme): update installation guide
test(api): add integration tests for report endpoint
```

### 3. 測試策略

- 單元測試：測試個別函數和類別
- 整合測試：測試 API 端點
- 效能測試：測試快取和優化效果

## 效能調優

### 1. 監控工具

- **應用監控**: 透過 `/api/v1/cache/info` 查看快取狀態
- **系統監控**: 使用 Grafana 儀表板
- **日誌分析**: 檢查應用日誌和錯誤訊息

### 2. 快取優化

```python
# 調整快取參數
@alru_cache(maxsize=100, ttl=3600)  # 根據需求調整
```

### 3. 資料庫優化

- 定期檢查 OpenSearch 索引效能
- 調整分片和副本設定
- 監控查詢效能