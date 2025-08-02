"""
向量檢索壓力測試
使用 Locust 進行負載測試
"""

from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import StatsCSVFileWriter
import json
import random
import time
from typing import List, Dict
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorSearchUser(HttpUser):
    """向量檢索負載測試用戶"""
    
    wait_time = between(0.5, 2.0)  # 請求間隔時間
    
    # 預定義的測試查詢
    queries = [
        # 短查詢
        "Python教程", "機器學習", "深度學習", "資料庫", "API設計",
        "演算法", "數據結構", "網路安全", "雲計算", "微服務",
        
        # 中等長度查詢
        "如何使用Python進行數據分析",
        "深度學習模型的訓練技巧",
        "微服務架構設計最佳實踐",
        "資料庫效能優化方法",
        "RESTful API設計原則",
        "容器化技術Docker使用指南",
        "分散式系統設計要點",
        "前端框架React開發教程",
        
        # 長查詢
        "在Python中如何實現高效的並行處理來提升大規模數據分析的效能",
        "使用深度學習進行自然語言處理時如何選擇合適的模型架構和超參數",
        "在微服務架構中如何實現服務間的安全通信和負載均衡",
        "關係型資料庫和NoSQL資料庫在不同場景下的選擇標準和優缺點分析",
    ]
    
    def on_start(self):
        """用戶開始時執行"""
        self.query_count = 0
        self.start_time = time.time()
        
    @task(3)
    def search_knn_only(self):
        """純向量搜尋測試"""
        query = random.choice(self.queries)
        
        payload = {
            "query": query,
            "k": random.choice([5, 10, 20]),
            "strategy": "knn_only"
        }
        
        with self.client.post(
            "/api/v1/search/vector",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data and len(data["results"]) > 0:
                    response.success()
                else:
                    response.failure("No results returned")
            else:
                response.failure(f"Status code: {response.status_code}")
                
        self.query_count += 1
    
    @task(2)
    def search_hybrid(self):
        """混合搜尋測試"""
        query = random.choice(self.queries)
        
        payload = {
            "query": query,
            "k": 10,
            "strategy": "hybrid"
        }
        
        with self.client.post(
            "/api/v1/search/vector",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")
                
        self.query_count += 1
    
    @task(1)
    def search_with_filter(self):
        """帶過濾條件的搜尋測試"""
        query = random.choice(self.queries)
        
        payload = {
            "query": query,
            "k": 10,
            "filter": {
                "tags": random.choice(["python", "machine-learning", "database", "api"])
            }
        }
        
        with self.client.post(
            "/api/v1/search/vector",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
                
        self.query_count += 1
    
    def on_stop(self):
        """用戶停止時執行"""
        duration = time.time() - self.start_time
        qps = self.query_count / duration if duration > 0 else 0
        logger.info(f"User completed {self.query_count} queries in {duration:.2f}s (QPS: {qps:.2f})")


class VectorSearchLoadTest:
    """向量搜尋負載測試管理器"""
    
    def __init__(self, host: str = "http://localhost:8000"):
        self.host = host
        self.results = []
        
    def run_stepped_load_test(self, 
                            initial_users: int = 1,
                            step_users: int = 5,
                            step_time: int = 60,
                            max_users: int = 50,
                            duration: int = 300):
        """執行階梯式負載測試
        
        Args:
            initial_users: 初始用戶數
            step_users: 每階段增加的用戶數
            step_time: 每階段持續時間（秒）
            max_users: 最大用戶數
            duration: 總測試時間（秒）
        """
        from locust import LoadTestShape
        
        class SteppedLoadShape(LoadTestShape):
            """階梯式負載形狀"""
            
            def tick(self):
                run_time = self.get_run_time()
                
                if run_time < duration:
                    current_step = int(run_time / step_time)
                    users = min(initial_users + current_step * step_users, max_users)
                    return (users, users)
                
                return None
        
        # 運行測試
        self._run_test(SteppedLoadShape)
    
    def run_spike_test(self, 
                      base_users: int = 10,
                      spike_users: int = 100,
                      spike_time: int = 60,
                      total_time: int = 300):
        """執行尖峰負載測試"""
        from locust import LoadTestShape
        
        class SpikeLoadShape(LoadTestShape):
            """尖峰負載形狀"""
            
            def tick(self):
                run_time = self.get_run_time()
                
                if run_time < spike_time:
                    return (base_users, base_users)
                elif run_time < spike_time * 2:
                    return (spike_users, spike_users)
                elif run_time < total_time:
                    return (base_users, base_users)
                
                return None
        
        self._run_test(SpikeLoadShape)
    
    def _run_test(self, shape_class=None):
        """執行測試"""
        # 這裡需要實際的 Locust 運行邏輯
        # 通常通過命令行或 API 啟動
        pass


# Pytest 整合測試
import pytest
import subprocess
import os
import signal
import psutil


class TestVectorLoadPerformance:
    """向量檢索負載效能測試"""
    
    @pytest.fixture
    def locust_process(self):
        """啟動 Locust 進程"""
        # 準備 CSV 輸出
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_prefix = f"locust_results_{timestamp}"
        
        # 啟動 Locust
        cmd = [
            "locust",
            "-f", __file__,
            "--headless",
            "--host", "http://localhost:8000",
            "--csv", csv_prefix,
            "--csv-full-history"
        ]
        
        process = subprocess.Popen(cmd)
        yield process
        
        # 清理
        process.terminate()
        process.wait()
    
    @pytest.mark.load
    def test_steady_load(self, locust_process):
        """穩定負載測試"""
        # 10個用戶，持續5分鐘
        cmd = [
            "locust", "-f", __file__,
            "--headless",
            "--users", "10",
            "--spawn-rate", "2",
            "--run-time", "5m",
            "--host", "http://localhost:8000",
            "--csv", "steady_load_test"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Locust failed: {result.stderr}"
        
        # 分析結果
        self._analyze_results("steady_load_test_stats.csv")
    
    @pytest.mark.load
    def test_increasing_load(self):
        """遞增負載測試"""
        # 使用階梯式增加用戶
        for users in [10, 20, 50, 100]:
            cmd = [
                "locust", "-f", __file__,
                "--headless",
                "--users", str(users),
                "--spawn-rate", "5",
                "--run-time", "2m",
                "--host", "http://localhost:8000",
                "--csv", f"increasing_load_{users}users"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0
            
            # 短暫休息
            time.sleep(30)
    
    @pytest.mark.load
    def test_spike_load(self):
        """尖峰負載測試"""
        # 突然增加到100個用戶
        cmd = [
            "locust", "-f", __file__,
            "--headless",
            "--users", "100",
            "--spawn-rate", "50",
            "--run-time", "2m",
            "--host", "http://localhost:8000",
            "--csv", "spike_load_test"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
    
    def _analyze_results(self, csv_file: str):
        """分析測試結果"""
        import pandas as pd
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            # 計算關鍵指標
            avg_response_time = df["Average Response Time"].mean()
            p95_response_time = df["95%"].mean()
            p99_response_time = df["99%"].mean()
            failure_rate = df["Failure Count"].sum() / df["Request Count"].sum() * 100
            
            print(f"\n=== 負載測試結果分析 ===")
            print(f"平均響應時間: {avg_response_time:.2f} ms")
            print(f"P95 響應時間: {p95_response_time:.2f} ms")
            print(f"P99 響應時間: {p99_response_time:.2f} ms")
            print(f"失敗率: {failure_rate:.2f}%")
            
            # 檢查效能基準
            assert avg_response_time < 200, "平均響應時間超過 200ms"
            assert p95_response_time < 500, "P95 響應時間超過 500ms"
            assert failure_rate < 1, "失敗率超過 1%"


# 獨立運行腳本
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 運行 pytest 測試
        pytest.main([__file__, "-v", "-m", "load"])
    else:
        # 作為 Locust 文件運行
        from locust import main as locust_main
        locust_main.main()