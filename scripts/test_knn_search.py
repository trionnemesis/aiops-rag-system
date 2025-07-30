#!/usr/bin/env python3
"""
測試 KNN 向量搜尋功能
驗證各種搜尋策略和參數設定
"""

import asyncio
import json
import sys
import os
from typing import List, Dict, Any
import time
from colorama import init, Fore, Style

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.knn_search_service import (
    KNNSearchService, 
    KNNSearchParams, 
    SearchStrategy
)
from scripts.build_knn_index import KNNIndexBuilder
import argparse

# 初始化 colorama
init()

# 測試查詢集
TEST_QUERIES = [
    {
        "query": "Apache 記憶體使用率過高",
        "expected_tags": ["apache", "memory"],
        "description": "測試 Apache 相關問題搜尋"
    },
    {
        "query": "MySQL 慢查詢優化",
        "expected_tags": ["mysql", "slow-query"],
        "description": "測試 MySQL 效能問題搜尋"
    },
    {
        "query": "高併發導致服務無回應",
        "expected_tags": ["high-traffic", "performance"],
        "description": "測試高併發問題搜尋"
    },
    {
        "query": "資料庫索引建立最佳實踐",
        "expected_tags": ["mysql", "performance"],
        "description": "測試最佳實踐搜尋"
    }
]


def print_success(message: str):
    """打印成功訊息"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_error(message: str):
    """打印錯誤訊息"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")


def print_info(message: str):
    """打印資訊訊息"""
    print(f"{Fore.CYAN}ℹ {message}{Style.RESET_ALL}")


def print_result(result: Dict[str, Any], index: int):
    """打印搜尋結果"""
    print(f"\n{Fore.YELLOW}結果 {index + 1}:{Style.RESET_ALL}")
    print(f"  文件 ID: {result.doc_id}")
    print(f"  標題: {result.title}")
    print(f"  分數: {result.score:.4f}")
    print(f"  標籤: {', '.join(result.metadata.get('tags', []))}")
    
    if result.highlights:
        print(f"  高亮片段:")
        for highlight in result.highlights[:2]:
            print(f"    - {highlight}")


async def test_knn_search_strategies():
    """測試不同的搜尋策略"""
    print(f"\n{Fore.MAGENTA}=== 測試 KNN 搜尋策略 ==={Style.RESET_ALL}\n")
    
    # 初始化搜尋服務
    search_service = KNNSearchService()
    
    strategies = [
        (SearchStrategy.KNN_ONLY, "純向量搜尋"),
        (SearchStrategy.HYBRID, "混合搜尋 (向量 + BM25)"),
        (SearchStrategy.MULTI_VECTOR, "多向量搜尋"),
        (SearchStrategy.RERANK, "重新排序搜尋")
    ]
    
    query = "Apache 效能優化"
    
    for strategy, description in strategies:
        print(f"\n{Fore.BLUE}測試策略: {description}{Style.RESET_ALL}")
        
        try:
            start_time = time.time()
            
            # 執行搜尋
            results = await search_service.knn_search(
                query_text=query,
                params=KNNSearchParams(k=3),
                strategy=strategy
            )
            
            elapsed_time = time.time() - start_time
            
            print_success(f"搜尋完成 (耗時: {elapsed_time:.2f}秒)")
            print(f"找到 {len(results)} 個結果")
            
            # 顯示前 2 個結果
            for i, result in enumerate(results[:2]):
                print_result(result, i)
                
        except Exception as e:
            print_error(f"搜尋失敗: {str(e)}")


async def test_search_parameters():
    """測試不同的搜尋參數"""
    print(f"\n{Fore.MAGENTA}=== 測試搜尋參數 ==={Style.RESET_ALL}\n")
    
    search_service = KNNSearchService()
    query = "資料庫效能問題"
    
    # 測試不同的 k 值
    k_values = [1, 5, 10]
    
    for k in k_values:
        print(f"\n{Fore.BLUE}測試 k={k}{Style.RESET_ALL}")
        
        try:
            params = KNNSearchParams(
                k=k,
                num_candidates=k * 20,  # HNSW 候選數
                min_score=0.5  # 最低分數門檻
            )
            
            results = await search_service.knn_search(
                query_text=query,
                params=params,
                strategy=SearchStrategy.HYBRID
            )
            
            print_success(f"找到 {len(results)} 個結果")
            
            # 顯示分數分佈
            if results:
                scores = [r.score for r in results]
                print(f"  分數範圍: {min(scores):.4f} - {max(scores):.4f}")
                print(f"  平均分數: {sum(scores)/len(scores):.4f}")
                
        except Exception as e:
            print_error(f"測試失敗: {str(e)}")


async def test_filter_search():
    """測試帶過濾條件的搜尋"""
    print(f"\n{Fore.MAGENTA}=== 測試過濾搜尋 ==={Style.RESET_ALL}\n")
    
    search_service = KNNSearchService()
    
    # 測試標籤過濾
    filters = [
        {"term": {"tags": "mysql"}},
        {"term": {"category": "incident_report"}},
        {"bool": {"must": [
            {"term": {"tags": "performance"}},
            {"term": {"tags": "mysql"}}
        ]}}
    ]
    
    query = "效能問題"
    
    for i, filter_dict in enumerate(filters):
        print(f"\n{Fore.BLUE}測試過濾條件 {i + 1}: {json.dumps(filter_dict, ensure_ascii=False)}{Style.RESET_ALL}")
        
        try:
            params = KNNSearchParams(
                k=5,
                filter=filter_dict
            )
            
            results = await search_service.knn_search(
                query_text=query,
                params=params,
                strategy=SearchStrategy.KNN_ONLY
            )
            
            print_success(f"找到 {len(results)} 個符合條件的結果")
            
            if results:
                print(f"  第一個結果: {results[0].title}")
                
        except Exception as e:
            print_error(f"過濾搜尋失敗: {str(e)}")


async def test_search_accuracy():
    """測試搜尋準確度"""
    print(f"\n{Fore.MAGENTA}=== 測試搜尋準確度 ==={Style.RESET_ALL}\n")
    
    search_service = KNNSearchService()
    
    total_tests = len(TEST_QUERIES)
    passed_tests = 0
    
    for test_case in TEST_QUERIES:
        print(f"\n{Fore.BLUE}測試: {test_case['description']}{Style.RESET_ALL}")
        print(f"查詢: {test_case['query']}")
        print(f"預期標籤: {', '.join(test_case['expected_tags'])}")
        
        try:
            results = await search_service.knn_search(
                query_text=test_case['query'],
                params=KNNSearchParams(k=3),
                strategy=SearchStrategy.HYBRID
            )
            
            if results:
                # 檢查第一個結果是否包含預期標籤
                first_result_tags = results[0].metadata.get('tags', [])
                matched_tags = set(test_case['expected_tags']) & set(first_result_tags)
                
                if matched_tags:
                    print_success(f"找到相關結果: {results[0].title}")
                    print(f"  匹配標籤: {', '.join(matched_tags)}")
                    passed_tests += 1
                else:
                    print_error(f"結果不包含預期標籤")
                    print(f"  實際標籤: {', '.join(first_result_tags)}")
            else:
                print_error("未找到任何結果")
                
        except Exception as e:
            print_error(f"測試失敗: {str(e)}")
    
    # 顯示總結
    print(f"\n{Fore.MAGENTA}=== 準確度測試總結 ==={Style.RESET_ALL}")
    print(f"總測試數: {total_tests}")
    print(f"通過測試: {passed_tests}")
    print(f"準確率: {(passed_tests/total_tests)*100:.1f}%")


async def test_performance():
    """測試搜尋效能"""
    print(f"\n{Fore.MAGENTA}=== 測試搜尋效能 ==={Style.RESET_ALL}\n")
    
    search_service = KNNSearchService()
    
    # 準備多個查詢
    queries = [
        "Apache 效能優化",
        "MySQL 慢查詢",
        "記憶體使用率過高",
        "磁碟 I/O 瓶頸",
        "高併發處理"
    ]
    
    strategies = [
        (SearchStrategy.KNN_ONLY, "純向量"),
        (SearchStrategy.HYBRID, "混合"),
        (SearchStrategy.MULTI_VECTOR, "多向量")
    ]
    
    for strategy, name in strategies:
        print(f"\n{Fore.BLUE}測試 {name} 搜尋效能{Style.RESET_ALL}")
        
        total_time = 0
        query_times = []
        
        for query in queries:
            start_time = time.time()
            
            try:
                await search_service.knn_search(
                    query_text=query,
                    params=KNNSearchParams(k=5),
                    strategy=strategy
                )
                
                elapsed = time.time() - start_time
                query_times.append(elapsed)
                total_time += elapsed
                
            except Exception as e:
                print_error(f"查詢失敗: {query} - {str(e)}")
        
        if query_times:
            avg_time = total_time / len(query_times)
            print_success(f"完成 {len(query_times)} 個查詢")
            print(f"  平均耗時: {avg_time:.3f}秒")
            print(f"  最快: {min(query_times):.3f}秒")
            print(f"  最慢: {max(query_times):.3f}秒")


async def test_explain_search():
    """測試搜尋解釋功能"""
    print(f"\n{Fore.MAGENTA}=== 測試搜尋解釋 ==={Style.RESET_ALL}\n")
    
    search_service = KNNSearchService()
    
    # 先執行搜尋
    query = "Apache 記憶體問題"
    results = await search_service.knn_search(
        query_text=query,
        params=KNNSearchParams(k=1),
        strategy=SearchStrategy.KNN_ONLY
    )
    
    if results:
        doc_id = results[0].doc_id
        print_info(f"解釋文件 {doc_id} 的評分")
        
        try:
            explanation = await search_service.explain_search(query, doc_id)
            print(json.dumps(explanation, indent=2, ensure_ascii=False))
            
        except Exception as e:
            print_error(f"解釋失敗: {str(e)}")


async def main():
    """主測試程式"""
    parser = argparse.ArgumentParser(description="測試 KNN 搜尋功能")
    parser.add_argument("--test", choices=["all", "strategies", "parameters", "filter", "accuracy", "performance", "explain"],
                       default="all", help="選擇測試類型")
    
    args = parser.parse_args()
    
    print(f"{Fore.GREEN}開始 KNN 搜尋功能測試{Style.RESET_ALL}")
    
    tests = {
        "strategies": test_knn_search_strategies,
        "parameters": test_search_parameters,
        "filter": test_filter_search,
        "accuracy": test_search_accuracy,
        "performance": test_performance,
        "explain": test_explain_search
    }
    
    if args.test == "all":
        for test_func in tests.values():
            await test_func()
    else:
        await tests[args.test]()
    
    print(f"\n{Fore.GREEN}測試完成！{Style.RESET_ALL}")


if __name__ == "__main__":
    asyncio.run(main())