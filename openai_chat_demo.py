"""
ChatOpenAI 直接调用
====================
使用 langchain_openai.ChatOpenAI 直接调用大模型，
比 init_chat_model 更显式，适合需要精细控制参数的场景。
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
load_dotenv()

# 直接创建 ChatOpenAI 实例（自动从环境变量读取 API Key 和 Base URL）
llm = ChatOpenAI(model="deepseek-v4-flash", temperature=0.0)

# 调用模型
result = llm.invoke("你是谁？")
print(result.content)
