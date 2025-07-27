"""
LangChain 提示詞管理器
使用 LangChain 的 PromptTemplate 統一管理所有提示詞
"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from typing import Dict, Any


class PromptManager:
    """統一的提示詞管理器"""
    
    def __init__(self):
        self._prompts: Dict[str, BasePromptTemplate] = {}
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """初始化所有提示詞模板"""
        
        # HyDE 生成提示詞
        self._prompts["hyde_generation"] = ChatPromptTemplate.from_template(
            """# [Prompt 1: Hypothetical Event Generation]

你是一位資深的 SRE (網站可靠性工程師)，擁有豐富的故障排除經驗。

你的任務是根據以下 JSON 格式的多維度監控數據，生成一段「假設性的事件摘要」。

請用流暢、專業的中文描述這台主機最可能發生的情況。在描述中，請著重分析各項指標之間的關聯性，並推斷可能的根本原因。

# 監控數據:
{monitoring_data}

請生成假設性事件摘要："""
        )
        
        # 文檔摘要提示詞
        self._prompts["summary_refinement"] = ChatPromptTemplate.from_template(
            """# [Prompt 2: Summarization-Refinement]

你是一個高效的資訊摘要助理。

當前的監控情況是： {monitoring_data}

你的任務是閱讀以下從公司知識庫中檢索到的文件，並摘要出與當前監控情況最相關的「根本原因」與「解決方案」。

摘要必須簡潔，不超過 150 字，以便後續分析。

# 知識庫文件:
{context}

請提供摘要："""
        )
        
        # 最終報告生成提示詞
        self._prompts["final_report"] = ChatPromptTemplate.from_template(
            """# [Prompt 3: Final Report Generation]

你是一位頂尖的 AIOps 智慧維運專家，擁有超過 10 年的生產環境故障排除經驗。你的任務是為客戶撰寫一份清晰、專業且具備深刻洞見的維運報告。

作為專家，你必須從以下三個維度進行分析：
1. **業務影響**：問題如何影響服務可用性、用戶體驗和收入
2. **技術根因**：深入分析技術層面的根本原因
3. **短期與長期解決方案**：提供立即可執行的緊急措施和永久性解決方案

你的語言必須專業，避免使用「可能」、「大概」、「也許」等模糊詞彙。直接指出問題的嚴重性，並提供精確的數據支撐。

# 1. 原始監控數據 (JSON):
{monitoring_data}

# 2. 相關歷史經驗與知識庫內容:
{context}

---
# 輸出格式要求:
你的輸出必須嚴格遵循以下格式。在「具體建議」中，必須包含至少一個 [緊急處理] 步驟。

洞見分析
[從業務影響、技術根因、問題嚴重性三個維度進行分析。每個維度都要有具體數據支撐。]

具體建議
[緊急處理]: [立即執行的步驟，需在 5 分鐘內完成]
[中期優化]: [1-7 天內實施的優化措施]
[永久措施]: [長期架構改進方案]

---
# 範例 (請勿模仿此案例的內容，僅學習風格):
**不好的分析**: CPU 可能過高，建議檢查一下系統。
**好的分析**: CPU 使用率達到 95%，已觸發 P0 級告警閾值（>90%），這將直接導致 API 回應延遲超過 3 秒，影響支付成功率下降 15%。根據歷史數據，類似情況曾導致每小時 $50,000 的業務損失。

**不好的建議**: 重啟服務看看。
**好的建議**: 
[緊急處理]: 執行 `kubectl scale deployment api-server --replicas=10` 立即擴容，分散 CPU 負載
[中期優化]: 分析慢查詢日誌，優化 TOP 5 高耗時 SQL，預計降低 40% CPU 使用
[永久措施]: 實施自動擴縮容策略，設置 HPA 在 CPU > 70% 時自動擴容

請按照以上格式輸出："""
        )
        
        # RAG 查詢提示詞（用於整合的 RAG 鏈）
        self._prompts["rag_query"] = ChatPromptTemplate.from_template(
            """基於以下檢索到的相關文檔內容，回答關於監控數據的問題。

監控數據：{question}

相關文檔：
{context}

請提供專業的分析和建議："""
        )
        
        # 多查詢生成提示詞 (RAG-Fusion)
        self._prompts["multi_query_generation"] = ChatPromptTemplate.from_template(
            """你是一位經驗豐富的 SRE 工程師。請根據以下監控數據，生成 3 個不同角度的檢索問題，用於在知識庫中尋找相關的解決方案和經驗。

每個問題應該：
1. 從不同的技術角度切入（例如：根因分析、性能優化、故障恢復）
2. 包含具體的技術關鍵詞
3. 明確且具有針對性

監控數據：
{monitoring_data}

請生成 3 個檢索問題，每個問題一行，不要編號："""
        )
    
    def get_prompt(self, prompt_name: str) -> BasePromptTemplate:
        """獲取指定名稱的提示詞模板
        
        Args:
            prompt_name: 提示詞名稱
            
        Returns:
            對應的提示詞模板
        """
        if prompt_name not in self._prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        return self._prompts[prompt_name]
    
    def add_custom_prompt(self, name: str, template: str, 
                         input_variables: list = None) -> None:
        """添加自定義提示詞模板
        
        Args:
            name: 提示詞名稱
            template: 模板字符串
            input_variables: 輸入變量列表
        """
        if input_variables:
            self._prompts[name] = PromptTemplate(
                template=template,
                input_variables=input_variables
            )
        else:
            self._prompts[name] = ChatPromptTemplate.from_template(template)
    
    def list_prompts(self) -> list:
        """列出所有可用的提示詞名稱"""
        return list(self._prompts.keys())
    
    def update_prompt(self, prompt_name: str, new_template: str) -> None:
        """更新現有提示詞模板
        
        Args:
            prompt_name: 提示詞名稱
            new_template: 新的模板字符串
        """
        if prompt_name in self._prompts:
            # 保留原有的輸入變量
            old_prompt = self._prompts[prompt_name]
            if hasattr(old_prompt, 'input_variables'):
                self._prompts[prompt_name] = PromptTemplate(
                    template=new_template,
                    input_variables=old_prompt.input_variables
                )
            else:
                self._prompts[prompt_name] = ChatPromptTemplate.from_template(new_template)


# 全局單例實例
prompt_manager = PromptManager()