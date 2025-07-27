import os
import google.generativeai as genai
from typing import List, Dict, Any
import asyncio
from src.config import settings
import numpy as np

class GeminiService:
    def __init__(self):
        # Check if we're in testing mode
        is_testing = os.environ.get('TESTING', '').lower() == 'true'
        api_key = "test-api-key" if is_testing else settings.gemini_api_key
        
        # Only configure genai if we have a real API key
        if not is_testing and api_key:
            genai.configure(api_key=api_key)
            self.flash_model = genai.GenerativeModel(settings.gemini_flash_model)
            self.pro_model = genai.GenerativeModel(settings.gemini_pro_model)
        else:
            # In test mode, we'll use None and handle in methods
            self.flash_model = None
            self.pro_model = None
        
    async def generate_hyde(self, prompt: str) -> str:
        """使用 Gemini Flash 生成假設性事件描述"""
        try:
            if self.flash_model is None:
                # Return a test response
                return "Test HyDE response"
            
            response = await asyncio.to_thread(
                self.flash_model.generate_content,
                prompt
            )
            return response.text
        except Exception as e:
            raise Exception(f"HyDE generation failed: {str(e)}")
    
    async def summarize_document(self, prompt: str) -> str:
        """使用 Gemini Flash 進行文檔摘要"""
        try:
            if self.flash_model is None:
                # Return a test response
                return "Test summary"
            
            response = await asyncio.to_thread(
                self.flash_model.generate_content,
                prompt
            )
            return response.text[:settings.max_summary_length]
        except Exception as e:
            raise Exception(f"Document summarization failed: {str(e)}")
    
    async def generate_final_report(self, prompt: str) -> Dict[str, str]:
        """使用 Gemini Pro 生成最終報告"""
        try:
            if self.pro_model is None:
                # Return a test response
                return {
                    "insight_analysis": "Test insight analysis",
                    "recommendations": "Test recommendations"
                }
            
            response = await asyncio.to_thread(
                self.pro_model.generate_content,
                prompt
            )
            
            # 解析報告內容
            content = response.text
            
            # 簡單的解析邏輯，實際可能需要更複雜的處理
            parts = content.split("具體建議")
            insight = parts[0].replace("洞見分析", "").strip()
            recommendations = parts[1].strip() if len(parts) > 1 else ""
            
            return {
                "insight_analysis": insight,
                "recommendations": recommendations
            }
        except Exception as e:
            raise Exception(f"Final report generation failed: {str(e)}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        try:
            if os.environ.get('TESTING', '').lower() == 'true':
                # Return a test embedding vector
                return [0.1] * settings.opensearch_embedding_dim
            
            # 使用 Gemini 的嵌入模型
            model = 'models/embedding-001'
            result = await asyncio.to_thread(
                genai.embed_content,
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")