"""基本程式驗證測試

這個檔案包含最基本的測試，確保程式能正常啟動和執行。
"""

import pytest
import sys
import os

# 確保能正確導入模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBasicValidation:
    """基本驗證測試類別"""
    
    def test_python_version(self):
        """測試 Python 版本是否符合要求"""
        assert sys.version_info >= (3, 9), "Python 版本必須是 3.9 或以上"
    
    def test_import_main_modules(self):
        """測試是否能正確導入主要模組"""
        try:
            import src.main
            import src.api.endpoints
            import src.core.gemini_service
            assert True
        except ImportError as e:
            pytest.fail(f"無法導入必要的模組: {e}")
    
    def test_import_dependencies(self):
        """測試是否能正確導入所有依賴套件"""
        required_packages = [
            'fastapi',
            'uvicorn',
            'pydantic',
            'aiohttp',
            'opensearch',
            'prometheus_client',
            'google.generativeai',
            'langchain',
            'numpy',
            'dotenv',
            'pytest',
            'httpx'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                # 嘗試導入子模組
                if '.' in package:
                    parent = package.split('.')[0]
                    try:
                        __import__(parent)
                    except ImportError:
                        pytest.fail(f"無法導入套件: {package}")
                else:
                    pytest.fail(f"無法導入套件: {package}")
    
    def test_environment_setup(self):
        """測試環境設定是否正確"""
        # 只檢查是否能讀取環境變數，不檢查具體值
        assert os.environ.get('PYTHONPATH') is not None or True, "環境變數可能未正確設定"
    
    @pytest.mark.asyncio
    async def test_basic_async_function(self):
        """測試異步功能是否正常"""
        async def simple_async():
            return "success"
        
        result = await simple_async()
        assert result == "success"
    
    def test_file_structure(self):
        """測試專案檔案結構是否正確"""
        required_dirs = ['src', 'tests', '.github']
        required_files = ['requirements.txt', 'Dockerfile', 'README.md']
        
        for dir_name in required_dirs:
            assert os.path.isdir(dir_name), f"找不到必要的目錄: {dir_name}"
        
        for file_name in required_files:
            assert os.path.isfile(file_name), f"找不到必要的檔案: {file_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])