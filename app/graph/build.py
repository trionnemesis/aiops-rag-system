from __future__ import annotations
from typing import Dict, Any, Callable, Optional, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import os
import redis
from langgraph.checkpoint.redis import RedisSaver
from .state import RAGState
from .nodes import extract_node, plan_node, retrieve_node, synthesize_node, validate_node, error_handler_node

# ---- 簡易 RRF（最小版）：只用排名融合，不看分數 ----
def simple_rrf_fuse(runs: List[List], k: int = 8, c: int = 60):
    # runs: [[doc1, doc2, ...], [docA, docB, ...], ...]
    scores = {}
    for run in runs:
        for rank, doc in enumerate(run, start=1):
            doc_id = doc.metadata.get("id") or doc.metadata.get("_id") or hash(doc.page_content)
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (c + rank)
    # 重新依分數排序，取前 k
    order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    keep_ids = set(x[0] for x in order[:k])
    out = []
    seen = set()
    for run in runs:
        for doc in run:
            doc_id = doc.metadata.get("id") or doc.metadata.get("_id") or hash(doc.page_content)
            if doc_id in keep_ids and doc_id not in seen:
                out.append(doc)
                seen.add(doc_id)
            if len(out) >= k:
                return out
    return out

def default_build_context(docs, max_chars: int = 6000):
    out = []
    used = 0
    for i, d in enumerate(docs, start=1):
        chunk = d.page_content.strip()
        meta = d.metadata or {}
        title = meta.get("title") or meta.get("source") or meta.get("_id") or f"doc{i}"
        piece = f"[{title}]\n{chunk}\n"
        if used + len(piece) > max_chars:
            break
        out.append(piece)
        used += len(piece)
    return "\n".join(out)

def build_graph(
    *,
    llm,
    retriever,
    extract_service=None,  # 新增 LangExtract 服務
    bm25_search_fn: Optional[Callable[[str, int], list]] = None,
    rrf_fuse_fn: Optional[Callable] = simple_rrf_fuse,
    build_context_fn: Callable = default_build_context,
    policy: Dict[str, Any] = None,
):
    """
    將既有 LCEL 組成 LangGraph。僅回傳 graph 與可執行的 app（含 checkpoint）。
    你可以把 llm/retriever/BM25 函式從現有程式注入進來。
    
    新增：
    - extract_service: LangExtractService 實例，用於結構化資訊提取
    """
    policy = policy or {}
    graph = StateGraph(RAGState)

    # 添加提取節點（如果有提供 extract_service）
    if extract_service:
        graph.add_node("extract", lambda s: extract_node(s, extract_service=extract_service, policy=policy))
    
    graph.add_node("plan", lambda s: plan_node(s, llm=llm, policy=policy))
    graph.add_node("retrieve", lambda s: retrieve_node(
        s, retriever=retriever, bm25_search_fn=bm25_search_fn,
        rrf_fuse_fn=rrf_fuse_fn, policy=policy
    ))
    graph.add_node("synthesize", lambda s: synthesize_node(
        s, llm=llm, build_context_fn=build_context_fn, policy=policy
    ))
    graph.add_node("validate", lambda s: validate_node(s, policy=policy))
    graph.add_node("error_handler", lambda s: error_handler_node(s, policy=policy))

    # 定義條件函數來檢查是否有錯誤
    def check_error(state):
        """檢查狀態中是否有錯誤"""
        if state.get("error"):
            return "error_handler"
        return "continue"
    
    # 設定流程
    if extract_service:
        graph.add_edge(START, "extract")
        # extract 可能失敗，添加條件邊
        graph.add_conditional_edges(
            "extract",
            check_error,
            {
                "continue": "plan",
                "error_handler": "error_handler"
            }
        )
    else:
        graph.add_edge(START, "plan")
    
    # plan 可能失敗，添加條件邊
    graph.add_conditional_edges(
        "plan",
        check_error,
        {
            "continue": "retrieve",
            "error_handler": "error_handler"
        }
    )
    
    # retrieve 可能失敗，添加條件邊
    graph.add_conditional_edges(
        "retrieve",
        check_error,
        {
            "continue": "synthesize",
            "error_handler": "error_handler"
        }
    )
    
    # synthesize 已經有內建錯誤處理，但仍然添加條件邊以防其他錯誤
    graph.add_conditional_edges(
        "synthesize",
        check_error,
        {
            "continue": "validate",
            "error_handler": "error_handler"
        }
    )
    
    graph.add_edge("validate", END)
    graph.add_edge("error_handler", END)

    # 根據環境變數決定使用 Redis 或記憶體 checkpoint
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            # 建立 Redis 連接
            redis_client = redis.from_url(redis_url)
            # 測試連接
            redis_client.ping()
            checkpointer = RedisSaver(redis_client)
            print(f"Using Redis checkpoint at {redis_url}")
        except Exception as e:
            print(f"Failed to connect to Redis: {e}, falling back to MemorySaver")
            checkpointer = MemorySaver()
    else:
        checkpointer = MemorySaver()
        print("Using in-memory checkpoint")
    
    app = graph.compile(checkpointer=checkpointer)
    return app