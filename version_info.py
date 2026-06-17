"""
当前环境 LangChain 版本信息
============================
查看已安装的 LangChain 及其组件的版本号，用于环境验证。
"""

import langchain
import langchain_community
import sys

print("=" * 50)
print("📦 LangChain 环境版本信息")
print("=" * 50)
print(f"  LangChain version:         {langchain.__version__}")
print(f"  LangChain Community ver:   {langchain_community.__version__}")
print(f"  Python version:            {sys.version}")
print(f"  LangChain 安装路径:        {langchain.__file__}")
