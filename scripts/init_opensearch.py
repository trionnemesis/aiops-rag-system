#!/usr/bin/env python3
"""
初始化 OpenSearch 並載入範例知識庫文件
"""

import asyncio
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.opensearch_service import OpenSearchService
from src.services.gemini_service import GeminiService

# 範例知識庫文件
SAMPLE_DOCUMENTS = [
    {
        "event_id": "E-2024-03-15",
        "title": "前台網站服務無回應事件",
        "tags": ["apache", "performance", "high-traffic", "memory", "maxclients"],
        "content": """# 事件報告: E-2024-03-15 - 前台網站服務無回應事件

## 事件標籤: apache, performance, high-traffic, memory, maxclients

## 摘要
在晚間 8 點的高峰時段，多個監控項告警，大量用戶回報網站加載緩慢或超時。經查，主要原因是 Apache 的 `KeepAliveTimeout` 設置過長，導致高併發下工作程序被耗盡，進而引發記憶體耗盡。

## 觀察到的症狀
1. Zabbix 監控顯示 Apache `MaxClients` 參數達到上限值 256。
2. 主機的記憶體使用率持續在 90% 以上，並出現少量 Swap 使用。
3. 前端的 Nginx 反向代理出現大量 502 (Bad Gateway) 錯誤。
4. CPU 使用率雖高，但並未達到 100%，I/O Wait 也處於正常水平。

## 根本原因分析 (RCA)
經查，主要原因是 Apache 的 `KeepAliveTimeout` 設置過長 (設定為 15秒)。在高併發下，大量已完成請求的 HTTP 連線長時間佔用工作程序不釋放，導致工作程序被快速耗盡。同時，每個閒置的 Apache 程序仍佔用約 20-30MB 的記憶體，大量閒置程序堆積，最終導致系統記憶體資源被完全佔用。

## 解決方案與行動項
1. **[緊急處理]** 將 Apache 設定檔中的 `KeepAliveTimeout` 從 15秒 緊急調整為 2秒，並重啟服務。服務在重啟後恢復正常。
2. **[中期優化]** 適度增加 `MaxClients` 的值至 300，並持續監控記憶體用量。
3. **[永久措施]** 為 Apache 伺服器啟用 `mod_status` 模組，並納入 Prometheus 監控，以便後續能即時追蹤 Apache 內部工作狀態。"""
    },
    {
        "event_id": "E-2024-05-22",
        "title": "MySQL 資料庫效能下降事件",
        "tags": ["mysql", "performance", "slow-query", "disk-io"],
        "content": """# 事件報告: E-2024-05-22 - MySQL 資料庫效能下降事件

## 事件標籤: mysql, performance, slow-query, disk-io

## 摘要
資料庫伺服器在上午 10 點左右開始出現效能下降，應用程式回報大量超時錯誤。經查，主要原因是缺少適當的索引導致全表掃描，加上磁碟 I/O 瓶頸。

## 觀察到的症狀
1. MySQL 慢查詢日誌顯示每分鐘超過 80 筆慢查詢。
2. 磁碟 I/O 等待時間高達 45%。
3. 資料庫活躍線程數達到 150。
4. 記憶體使用率達 92%，但 CPU 使用率僅 30%。

## 根本原因分析 (RCA)
1. 新部署的應用程式版本包含一個未優化的查詢，對 orders 表執行全表掃描。
2. orders 表資料量超過 1000 萬筆，缺少 created_date 欄位的索引。
3. 磁碟為傳統 HDD，無法應付突增的隨機讀取需求。

## 解決方案與行動項
1. **[緊急處理]** 為 created_date 欄位新增索引，查詢時間從 30 秒降至 0.1 秒。
2. **[中期優化]** 啟用 MySQL 查詢快取，並調整 innodb_buffer_pool_size。
3. **[永久措施]** 計劃將資料庫遷移至 SSD 存儲，並實施查詢審核機制。"""
    }
]

async def init_opensearch():
    """初始化 OpenSearch 並載入範例文件"""
    print("Initializing OpenSearch...")
    
    opensearch = OpenSearchService()
    gemini = GeminiService()
    
    # 創建索引
    await opensearch.create_index()
    print("Index created successfully")
    
    # 載入範例文件
    for doc in SAMPLE_DOCUMENTS:
        print(f"Processing document: {doc['event_id']}")
        
        # 生成嵌入向量
        embedding = await gemini.generate_embedding(doc['content'])
        
        # 索引文件
        await opensearch.index_document(
            doc_id=doc['event_id'],
            content=doc['content'],
            embedding=embedding,
            title=doc['title'],
            tags=doc['tags']
        )
        
        print(f"Document {doc['event_id']} indexed successfully")
    
    print("OpenSearch initialization completed!")

if __name__ == "__main__":
    asyncio.run(init_opensearch())