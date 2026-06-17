"""
init_chat_model 基础调用
=========================
使用 LangChain 的 init_chat_model 快捷方法创建聊天模型并调用。
自动从 .env 文件加载 API Key 和 Base URL。
"""

import os

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# 从 .env 文件加载环境变量（API Key 等）
load_dotenv()

# 创建 LLM 实例 — init_chat_model 根据参数自动选择合适的模型类
llm = init_chat_model(
    "deepseek-v4-flash",
    model_provider="openai",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# 调用模型
result = llm.invoke("你是谁？")
print(result.content)
