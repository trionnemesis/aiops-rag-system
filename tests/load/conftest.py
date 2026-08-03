"""壓測（Locust）專用設定

這裡的測試需要一個實際在跑的服務，且 import locust 會觸發 gevent 的
monkey.patch_all()。在一般 pytest 執行中，ssl 早已被其他測試模組匯入，
此時再 monkey-patch 會造成 RecursionError，讓整個 collection 失敗。

因此預設跳過收集；要跑壓測請明確指定：

    RUN_LOAD_TESTS=1 pytest tests/load
"""

import os

if not os.getenv("RUN_LOAD_TESTS"):
    collect_ignore_glob = ["*"]
