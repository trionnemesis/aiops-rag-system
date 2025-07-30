from __future__ import annotations
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

# ==== 可插拔的相依，由外部注入 ====
# llm: BaseLanguageModel
# retriever: 任意 LangChain retriever（OpenSearchVectorSearch retriever）
# bm25_search_fn: Optional[callable] -> (query: str, top_k: int) -> List[Document]
# rrf_fuse_fn: Optional[callable] -> (runs: List[List[Document]], k: int) -> List[Document]
# build_context_fn: (docs: List[Document], max_chars: int) -> str
# policy: dict（成本護欄、是否啟用 HyDE/RRF、top_k、timeouts ...）

def _unique_by_id(docs: List[Document], key: str = "id") -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        doc_id = d.metadata.get(key) or d.metadata.get("_id") or hash(d.page_content)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(d)
    return out

def plan_node(state, llm: BaseLanguageModel, policy: Dict[str, Any], **kwargs):
    """決策 fast/deep 與產生 queries（可含 HyDE、多查詢）"""
    q = state["query"]
    use_hyde = policy.get("use_hyde", False)
    multi_query = policy.get("use_multi_query", False)
    max_alts = policy.get("multi_query_alts", 2)

    queries = [q]

    # 簡單啟發式：過長/過短或含模糊詞 → deep 路徑
    ambiguous = len(q) < 8 or any(t in q for t in ["為什麼", "怎麼", "原因", "異常", "不穩定"])
    route = "deep" if (use_hyde or multi_query or ambiguous) else "fast"

    if route == "deep" and use_hyde:
        # HyDE：請用你原專案的 prompt，這裡示意
        pseudo = llm.invoke(f"依下列查詢產生一段可能的說明作為檢索用：\n查詢：{q}\n只回主體內容")
        if hasattr(pseudo, "content"):
            pseudo = pseudo.content
        queries.append(pseudo.strip()[:400])

    if route == "deep" and multi_query:
        # 多查詢：請用你現有邏輯/Prompt，這裡極簡示意
        mq = llm.invoke(f"為下列問題產生 {max_alts} 個等義查詢，每行一個：{q}")
        text = getattr(mq, "content", str(mq))
        for line in text.splitlines():
            line = line.strip(" -•\t")
            if line and line.lower() != q.lower():
                queries.append(line)
            if len(queries) >= 1 + max_alts + (2 if use_hyde else 0):
                break

    state["route"] = route
    state["queries"] = queries
    state.setdefault("metrics", {})["queries"] = len(queries)
    return state

def retrieve_node(state,
                  retriever,
                  bm25_search_fn=None,
                  rrf_fuse_fn=None,
                  policy: Dict[str, Any] = None,
                  **kwargs):
    """檢索（向量 + 選配 BM25 + 選配 RRF）"""
    policy = policy or {}
    top_k = policy.get("top_k", 8)
    enable_rrf = policy.get("use_rrf", False) and bm25_search_fn is not None and rrf_fuse_fn is not None

    queries: List[str] = state["queries"]
    runs = []

    # 向量檢索（對每個 query 各取 top_k，再合併去重）
    vec_docs_all = []
    for q in queries:
        docs = retriever.get_relevant_documents(q)
        vec_docs_all.extend(docs[:top_k])
    vec_docs_all = _unique_by_id(vec_docs_all)

    if enable_rrf:
        # BM25 跑一次（可按需對每個 q 跑，這裡最小可行跑主查詢）
        bm25_docs = bm25_search_fn(queries[0], top_k=top_k)
        runs = [vec_docs_all[:top_k], bm25_docs[:top_k]]
        fused = rrf_fuse_fn(runs, k=top_k)
        docs_final = _unique_by_id(fused)
    else:
        docs_final = vec_docs_all[:top_k]

    state["docs"] = docs_final
    m = state.setdefault("metrics", {})
    m["docs"] = len(docs_final)
    m["rrf_on"] = enable_rrf
    return state

def synthesize_node(state,
                    llm: BaseLanguageModel,
                    build_context_fn,
                    policy: Dict[str, Any] = None,
                    **kwargs):
    """將檢索結果組上下文並生成答案；失敗時回退簡版"""
    policy = policy or {}
    max_ctx = policy.get("max_ctx_chars", 6000)
    strict_cite = policy.get("strict_citation", True)
    answer_fallback = policy.get("fallback_text", "（系統忙碌，先提供精簡結論，稍後再補全報告）")

    docs = state.get("docs", [])
    q = state["query"]

    try:
        context = build_context_fn(docs, max_chars=max_ctx)
        state["context"] = context

        prompt = (
            "你是 AIOps 報告助理，請根據【資料來源】回答使用者問題。\n"
            "要求：\n"
            "1) 僅引用資料來源可得的內容；2) 給出條列、結構化回答；3) 標註來源標題或ID。\n\n"
            f"【問題】\n{q}\n\n【資料來源】\n{context}\n\n【回答】"
        )
        result = llm.invoke(prompt)
        answer = getattr(result, "content", str(result)).strip()
        if strict_cite and ("來源" not in answer and "[" not in answer):
            # 粗略檢查是否有引用痕跡；你可替換成更嚴謹規則
            answer += "\n\n（提示：來源標註不足）"

        state["answer"] = answer
        return state
    except Exception as e:
        state["error"] = f"synthesize_error: {e}"
        state["answer"] = answer_fallback
        return state

def validate_node(state, policy: Dict[str, Any] = None, **kwargs):
    """最小驗證：若沒有檢索到文件或答案過短，標示警告"""
    policy = policy or {}
    min_docs = policy.get("min_docs", 2)
    min_len = policy.get("min_answer_len", 40)

    warnings = []
    if len(state.get("docs", [])) < min_docs:
        warnings.append("low_docs")
    if len(state.get("answer", "")) < min_len:
        warnings.append("short_answer")

    if warnings:
        state.setdefault("metrics", {})["warnings"] = warnings
    return state