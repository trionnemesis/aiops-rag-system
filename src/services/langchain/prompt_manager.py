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

你是一位頂尖的 AIOps 智慧維運專家，你的任務是為客戶撰寫一份清晰、專業且具備深刻洞見的維運報告。

請整合以下所有資訊，生成兩段內容：「洞見分析」與「具體建議」。

在「洞見分析」中，請說明監控數據反映出的核心問題，並結合歷史經驗指出問題的嚴重性。
在「具體建議」中，請提供明確、可操作的步驟，幫助客戶解決問題或預防風險。

你的語言必須專業，避免使用模糊的詞彙，並直接提出可行的行動方案。

# 1. 原始監控數據 (JSON):
{monitoring_data}

# 2. 相關歷史經驗與知識庫內容:
{context}

請按照以下格式輸出：

洞見分析
[你的分析內容]

具體建議
[你的建議內容]"""
        )
        
        # RAG 查詢提示詞（用於整合的 RAG 鏈）
        self._prompts["rag_query"] = ChatPromptTemplate.from_template(
            """基於以下檢索到的相關文檔內容，回答關於監控數據的問題。

監控數據：{question}

相關文檔：
{context}

請提供專業的分析和建議："""
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