#!/usr/bin/env python3
"""
Verify that updated dependencies are compatible with the codebase
"""
import subprocess
import sys
import importlib
from typing import List, Tuple

def check_imports() -> List[Tuple[str, bool, str]]:
    """Check if all required modules can be imported"""
    modules_to_check = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic_settings",
        "multipart",
        "aiohttp",
        "opensearchpy",
        "prometheus_client",
        "google.generativeai",
        "dotenv",
        "numpy",
        "langchain",
        "langchain_google_genai",
        "langchain_community",
        "langchain_core",
        "tiktoken",
        "starlette",
    ]
    
    results = []
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            results.append((module, True, "OK"))
        except ImportError as e:
            results.append((module, False, str(e)))
    
    return results

def check_langchain_compatibility():
    """Check specific LangChain compatibility"""
    try:
        # Test critical imports
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel
        from langchain_core.output_parsers import StrOutputParser
        from langchain_google_genai import GoogleGenerativeAI
        from langchain_community.vectorstores import OpenSearchVectorSearch
        from langchain_core.vectorstores import VectorStore
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings
        
        print("✅ All critical LangChain imports successful")
        return True
    except ImportError as e:
        print(f"❌ LangChain import error: {e}")
        return False

def run_security_audit():
    """Run pip-audit to check for vulnerabilities"""
    try:
        result = subprocess.run(
            ["pip-audit", "--desc"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ No vulnerabilities found by pip-audit")
            return True
        else:
            print("❌ Vulnerabilities found:")
            print(result.stdout)
            return False
    except FileNotFoundError:
        print("⚠️  pip-audit not installed. Run: pip install pip-audit")
        return False

def main():
    print("=== Dependency Compatibility Verification ===\n")
    
    # Check imports
    print("1. Checking module imports...")
    import_results = check_imports()
    all_imports_ok = all(result[1] for result in import_results)
    
    for module, success, message in import_results:
        status = "✅" if success else "❌"
        print(f"  {status} {module}: {message}")
    
    print(f"\nImport check: {'PASSED' if all_imports_ok else 'FAILED'}\n")
    
    # Check LangChain compatibility
    print("2. Checking LangChain compatibility...")
    langchain_ok = check_langchain_compatibility()
    print(f"LangChain check: {'PASSED' if langchain_ok else 'FAILED'}\n")
    
    # Run security audit
    print("3. Running security audit...")
    security_ok = run_security_audit()
    print(f"Security check: {'PASSED' if security_ok else 'FAILED'}\n")
    
    # Summary
    print("=== Summary ===")
    all_ok = all_imports_ok and langchain_ok and security_ok
    
    if all_ok:
        print("✅ All checks passed! Dependencies are compatible.")
        sys.exit(0)
    else:
        print("❌ Some checks failed. Please review the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()