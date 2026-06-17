"""
JsonOutputParser 使用示例
===========================
使用 JsonOutputParser 让模型返回结构化的 JSON 数据，
适用于需要解析模型输出为字典/列表的场景。
"""

import os
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

# ==========================================
# 1. 构建 Prompt（注入格式指令）
# ==========================================
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    ("human", "{question},{format_instructions}"),
])

# 创建 JSON 输出解析器并获取格式指令
json_parser = JsonOutputParser()
format_instructions = json_parser.get_format_instructions()

messages = chat_prompt_template.format_messages(
    question="什么是人工智能？",
    format_instructions=format_instructions,
)
logger.info(f"构造的消息: {messages}")

# ==========================================
# 2. 调用模型
# ==========================================
llm = init_chat_model(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

result = llm.invoke(messages)
logger.info(f"模型原始响应: {result}")
logger.info(f"模型响应内容: {result.content}")

# ==========================================
# 3. 解析为 JSON
# ==========================================
try:
    parsed_result = json_parser.invoke(result)
    logger.info(f"解析结果: {parsed_result}")
except Exception as e:
    logger.error(f"解析失败: {e}")
