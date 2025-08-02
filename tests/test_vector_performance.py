"""
向量檢索效能測試套件
包含標準查詢集、效能評估指標和基線測試
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from src.services.knn_search_service import KNNSearchService, KNNSearchParams, SearchStrategy
from src.services.opensearch_service import OpenSearchService
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class VectorPerformanceTestSuite:
    """向量檢索效能測試套件"""
    
    # 標準查詢集合
    STANDARD_QUERIES = {
        "short_queries": [
            "Python教程",
            "機器學習",
            "深度學習",
            "資料庫",
            "API設計",
        ],
        "medium_queries": [
            "如何使用Python進行數據分析",
            "深度學習模型的訓練技巧",
            "微服務架構設計最佳實踐",
            "資料庫效能優化方法",
            "RESTful API設計原則",
        ],
        "long_queries": [
            "在Python中如何實現高效的並行處理來提升大規模數據分析的效能",
            "使用深度學習進行自然語言處理時如何選擇合適的模型架構和超參數",
            "在微服務架構中如何實現服務間的安全通信和負載均衡",
            "關係型資料庫和NoSQL資料庫在不同場景下的選擇標準和優缺點分析",
            "如何設計一個可擴展的RESTful API系統並確保高可用性和一致性",
        ],
        "fuzzy_queries": [
            "Pyton程式設計",  # 拼寫錯誤
            "機器學習 deep lerning",  # 混合語言和拼寫錯誤
            "資料庫 performnce",  # 拼寫錯誤
            "API設計 best practice",  # 混合語言
            "深度學習 transfomer",  # 拼寫錯誤
        ]
    }
    
    def __init__(self):
        self.knn_service = KNNSearchService()
        self.opensearch_service = OpenSearchService()
        self.results = []
        
    async def measure_query_latency(self, query: str, params: KNNSearchParams = None) -> Tuple[float, List[Any]]:
        """測量單次查詢延遲"""
        start_time = time.time()
        results = await self.knn_service.knn_search(query, params)
        latency = (time.time() - start_time) * 1000  # 轉換為毫秒
        return latency, results
    
    async def measure_batch_performance(self, queries: List[str], params: KNNSearchParams = None) -> Dict[str, Any]:
        """測量批次查詢效能"""
        latencies = []
        all_results = []
        
        for query in queries:
            latency, results = await self.measure_query_latency(query, params)
            latencies.append(latency)
            all_results.append(results)
        
        # 計算效能指標
        latencies_sorted = sorted(latencies)
        metrics = {
            "avg_latency": np.mean(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "p50_latency": np.percentile(latencies_sorted, 50),
            "p95_latency": np.percentile(latencies_sorted, 95),
            "p99_latency": np.percentile(latencies_sorted, 99),
            "std_dev": np.std(latencies),
            "query_count": len(queries),
            "total_time": sum(latencies),
        }
        
        return metrics
    
    async def evaluate_recall_precision(self, query: str, ground_truth: List[str], k: int = 10) -> Dict[str, float]:
        """評估召回率和準確率"""
        params = KNNSearchParams(k=k)
        results = await self.knn_service.knn_search(query, params)
        
        retrieved_ids = [r.doc_id for r in results]
        
        # 計算召回率和準確率
        true_positives = len(set(retrieved_ids) & set(ground_truth))
        
        recall = true_positives / len(ground_truth) if ground_truth else 0
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
        
        # 計算 F1 分數
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score,
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(ground_truth),
            "true_positives": true_positives
        }
    
    async def run_performance_baseline(self) -> pd.DataFrame:
        """執行效能基線測試"""
        results = []
        
        # 測試不同查詢類型
        for query_type, queries in self.STANDARD_QUERIES.items():
            logger.info(f"測試查詢類型: {query_type}")
            
            # 測試不同的 k 值
            for k in [5, 10, 20, 50]:
                params = KNNSearchParams(k=k)
                metrics = await self.measure_batch_performance(queries, params)
                
                result = {
                    "query_type": query_type,
                    "k": k,
                    "timestamp": datetime.now(),
                    **metrics
                }
                results.append(result)
                
        # 測試不同的搜尋策略
        strategies = [SearchStrategy.KNN_ONLY, SearchStrategy.HYBRID]
        for strategy in strategies:
            logger.info(f"測試搜尋策略: {strategy.value}")
            
            all_queries = []
            for queries in self.STANDARD_QUERIES.values():
                all_queries.extend(queries)
            
            params = KNNSearchParams(k=10)
            latencies = []
            
            for query in all_queries:
                start_time = time.time()
                await self.knn_service.knn_search(query, params, strategy)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
            
            result = {
                "query_type": "all",
                "strategy": strategy.value,
                "k": 10,
                "timestamp": datetime.now(),
                "avg_latency": np.mean(latencies),
                "p95_latency": np.percentile(latencies, 95),
                "p99_latency": np.percentile(latencies, 99),
                "query_count": len(all_queries)
            }
            results.append(result)
        
        # 轉換為 DataFrame
        df = pd.DataFrame(results)
        return df
    
    async def test_ef_search_impact(self) -> pd.DataFrame:
        """測試 ef_search 參數對效能的影響"""
        results = []
        ef_search_values = [50, 100, 200, 500, 1000]
        
        for ef_search in ef_search_values:
            logger.info(f"測試 ef_search={ef_search}")
            
            # 更新索引設定
            update_body = {
                "index": {
                    "knn.algo_param.ef_search": ef_search
                }
            }
            
            try:
                self.opensearch_service.client.indices.put_settings(
                    index=self.opensearch_service.index_name,
                    body=update_body
                )
                
                # 等待設定生效
                await asyncio.sleep(1)
                
                # 執行測試
                all_queries = []
                for queries in self.STANDARD_QUERIES.values():
                    all_queries.extend(queries[:2])  # 每類取2個查詢
                
                params = KNNSearchParams(k=10)
                metrics = await self.measure_batch_performance(all_queries, params)
                
                result = {
                    "ef_search": ef_search,
                    "timestamp": datetime.now(),
                    **metrics
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"更新 ef_search 失敗: {e}")
        
        # 還原預設值
        update_body = {"index": {"knn.algo_param.ef_search": 100}}
        self.opensearch_service.client.indices.put_settings(
            index=self.opensearch_service.index_name,
            body=update_body
        )
        
        df = pd.DataFrame(results)
        return df


# 測試函數
@pytest.mark.asyncio
async def test_performance_baseline():
    """執行效能基線測試"""
    test_suite = VectorPerformanceTestSuite()
    df = await test_suite.run_performance_baseline()
    
    # 保存結果
    df.to_csv("vector_performance_baseline.csv", index=False)
    
    # 輸出摘要
    print("\n=== 效能基線測試結果 ===")
    print(f"平均延遲: {df['avg_latency'].mean():.2f} ms")
    print(f"P95 延遲: {df['p95_latency'].mean():.2f} ms")
    print(f"P99 延遲: {df['p99_latency'].mean():.2f} ms")
    
    # 按查詢類型分組統計
    print("\n按查詢類型統計:")
    query_stats = df.groupby('query_type')[['avg_latency', 'p95_latency']].mean()
    print(query_stats)


@pytest.mark.asyncio
async def test_ef_search_impact():
    """測試 ef_search 參數影響"""
    test_suite = VectorPerformanceTestSuite()
    df = await test_suite.test_ef_search_impact()
    
    # 保存結果
    df.to_csv("ef_search_impact.csv", index=False)
    
    # 輸出結果
    print("\n=== ef_search 參數影響測試 ===")
    for _, row in df.iterrows():
        print(f"ef_search={row['ef_search']}: "
              f"avg={row['avg_latency']:.2f}ms, "
              f"p95={row['p95_latency']:.2f}ms")


@pytest.mark.asyncio
async def test_recall_precision():
    """測試召回率和準確率"""
    test_suite = VectorPerformanceTestSuite()
    
    # 模擬相關文檔 ID（實際應用中應該有人工標註的相關文檔）
    test_cases = [
        {
            "query": "Python數據分析教程",
            "ground_truth": ["doc_001", "doc_002", "doc_003", "doc_004", "doc_005"]
        },
        {
            "query": "深度學習模型訓練",
            "ground_truth": ["doc_010", "doc_011", "doc_012", "doc_013", "doc_014"]
        }
    ]
    
    results = []
    for test_case in test_cases:
        metrics = await test_suite.evaluate_recall_precision(
            test_case["query"],
            test_case["ground_truth"],
            k=10
        )
        
        result = {
            "query": test_case["query"],
            **metrics
        }
        results.append(result)
        
        print(f"\n查詢: {test_case['query']}")
        print(f"召回率: {metrics['recall']:.2%}")
        print(f"準確率: {metrics['precision']:.2%}")
        print(f"F1分數: {metrics['f1_score']:.2%}")
    
    # 保存結果
    pd.DataFrame(results).to_csv("recall_precision_results.csv", index=False)


if __name__ == "__main__":
    # 可直接執行測試
    asyncio.run(test_performance_baseline())