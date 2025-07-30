from fastapi import FastAPI
from app.api.routes import router

# FastAPI 啟動（不需更動）
app = FastAPI(title="AIOps RAG Service")

# 掛載原有的路由
app.include_router(router, prefix="/api/v1")

# 健康檢查
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)