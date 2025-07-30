from __future__ import annotations
from typing import Dict, Any, Callable, Optional, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import RAGState
from .nodes import plan_node, retrieve_node, synthesize_node, validate_node

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
    bm25_search_fn: Optional[Callable[[str, int], list]] = None,
    rrf_fuse_fn: Optional[Callable] = simple_rrf_fuse,
    build_context_fn: Callable = default_build_context,
    policy: Dict[str, Any] = None,
):
    """
    將既有 LCEL 組成 LangGraph。僅回傳 graph 與可執行的 app（含 checkpoint）。
    你可以把 llm/retriever/BM25 函式從現有程式注入進來。
    """
    policy = policy or {}
    graph = StateGraph(RAGState)

    graph.add_node("plan", lambda s: plan_node(s, llm=llm, policy=policy))
    graph.add_node("retrieve", lambda s: retrieve_node(
        s, retriever=retriever, bm25_search_fn=bm25_search_fn,
        rrf_fuse_fn=rrf_fuse_fn, policy=policy
    ))
    graph.add_node("synthesize", lambda s: synthesize_node(
        s, llm=llm, build_context_fn=build_context_fn, policy=policy
    ))
    graph.add_node("validate", lambda s: validate_node(s, policy=policy))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "validate")
    graph.add_edge("validate", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    return app