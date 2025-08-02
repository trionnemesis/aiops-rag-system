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
# extract_service: LangExtractService 實例

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

def extract_node(state, extract_service=None, policy: Dict[str, Any] = None, **kwargs):
    """提取結構化資訊節點：從原始文本中提取 AIOps 實體"""
    policy = policy or {}
    use_llm_extract = policy.get("use_llm_extract", True)
    
    # 如果沒有原始文本，跳過提取
    raw_texts = state.get("raw_texts", [])
    if not raw_texts or not extract_service:
        state["extracted_data"] = []
        return state
    
    # 批量提取
    extracted_results = extract_service.batch_extract(raw_texts, use_llm=use_llm_extract)
    
    # 轉換為元數據格式
    extracted_data = []
    for result in extracted_results:
        metadata = extract_service.extract_to_metadata(result.raw_text, use_llm=False)
        # 附加原始提取結果供後續使用
        metadata['_raw_extracted'] = result.entities.dict()
        metadata['_extraction_confidence'] = result.confidence
        extracted_data.append(metadata)
    
    state["extracted_data"] = extracted_data
    state.setdefault("metrics", {})["extracted_count"] = len(extracted_data)
    
    return state

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
    enable_metadata_filter = policy.get("use_metadata_filter", True)

    queries: List[str] = state["queries"]
    extracted_data = state.get("extracted_data", [])
    runs = []

    # 如果有提取的元數據，構建過濾條件
    metadata_filters = {}
    if enable_metadata_filter and extracted_data:
        # 從提取的資料中收集關鍵過濾條件
        for data in extracted_data:
            raw_extracted = data.get("_raw_extracted", {})
            # 優先過濾高信心度的提取結果
            if data.get("_extraction_confidence", 0) > 0.7:
                if raw_extracted.get("hostname"):
                    metadata_filters["extracted_hostname"] = raw_extracted["hostname"]
                if raw_extracted.get("service_name"):
                    metadata_filters["extracted_service_name"] = raw_extracted["service_name"]
                if raw_extracted.get("error_code"):
                    metadata_filters["extracted_error_code"] = raw_extracted["error_code"]

    # 向量檢索（對每個 query 各取 top_k，再合併去重）
    vec_docs_all = []
    for q in queries:
        # 如果 retriever 支援元數據過濾，則應用過濾
        if hasattr(retriever, "search_kwargs") and metadata_filters:
            # 暫存原始設定
            original_kwargs = retriever.search_kwargs.copy()
            # 添加過濾條件
            retriever.search_kwargs["filter"] = metadata_filters
            docs = retriever.get_relevant_documents(q)
            # 還原設定
            retriever.search_kwargs = original_kwargs
        else:
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
    m["metadata_filters"] = metadata_filters
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
    extracted_data = state.get("extracted_data", [])

    try:
        context = build_context_fn(docs, max_chars=max_ctx)
        state["context"] = context
        
        # 準備結構化資料摘要
        structured_summary = ""
        if extracted_data:
            # 收集高信心度的關鍵資訊
            key_info = []
            for data in extracted_data:
                if data.get("_extraction_confidence", 0) > 0.7:
                    raw = data.get("_raw_extracted", {})
                    if raw.get("hostname"):
                        key_info.append(f"主機: {raw['hostname']}")
                    if raw.get("service_name"):
                        key_info.append(f"服務: {raw['service_name']}")
                    if raw.get("error_code"):
                        key_info.append(f"錯誤碼: {raw['error_code']}")
                    if raw.get("cpu_usage") is not None:
                        key_info.append(f"CPU使用率: {raw['cpu_usage']}%")
                    if raw.get("memory_usage") is not None:
                        key_info.append(f"記憶體使用率: {raw['memory_usage']}%")
            
            if key_info:
                structured_summary = "\n【提取的關鍵資訊】\n" + "\n".join(f"• {info}" for info in key_info[:10])

        prompt = (
            "你是 AIOps 報告助理，請根據【資料來源】和【提取的關鍵資訊】回答使用者問題。\n"
            "要求：\n"
            "1) 優先使用提取的結構化資訊\n"
            "2) 結合資料來源提供深入分析\n"
            "3) 給出條列、結構化回答\n"
            "4) 標註來源標題或ID\n\n"
            f"【問題】\n{q}\n{structured_summary}\n\n【資料來源】\n{context}\n\n【回答】"
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