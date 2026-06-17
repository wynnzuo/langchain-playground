"""
init_chat_model + Message 对象
===============================
通过 SystemMessage / HumanMessage 构建多轮对话上下文，
再传给 init_chat_model 调用。
"""

import os

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage

# 加载环境变量
load_dotenv()

# 创建 LLM 实例
llm = init_chat_model(
    "deepseek-v4-flash",
    model_provider="openai",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# 构建消息列表：系统指令 + 用户问题
messages = [
    SystemMessage(content="你是一个有帮助的助手。"),
    HumanMessage(content="你是谁？"),
]

# 传入消息列表（而非单字符串）让模型理解上下文
result = llm.invoke(messages)
print(f"type of result: {type(result)}")
print(result.content)
