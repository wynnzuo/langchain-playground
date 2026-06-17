"""
Agent + Tool 调用示例（法国信息查询）
========================================
演示最基础的 Agent 使用：直接向 LLM 提问。
后续可扩展为带工具调用的 ReAct Agent 模式。
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
load_dotenv()

# 创建 LLM 实例
llm = ChatOpenAI(model="deepseek-v4-flash", temperature=0.0)


if __name__ == "__main__":
    print("Asking: What is the capital of France?")
    question = "What is the capital of France?"
    response = llm.invoke(question)
    print(response.content)
