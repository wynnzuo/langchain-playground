"""
Ollama 本地模型调用
====================
使用 langchain_ollama.ChatOllama 对接本地运行的 Ollama 服务。
无需 API Key，依赖本地 Ollama 服务（默认 localhost:11434）。
"""

from langchain_ollama import ChatOllama

# 创建 Ollama LLM 实例（需先本地启动 Ollama 并拉取模型）
llm = ChatOllama(model="llama2", temperature=0.0)

# 调用本地模型
result = llm.invoke("你是谁？")
print(result.content)
