"""
翻译 LCEL 链
==============
将自定义翻译链（Prompt → LLM → JSON 解析器）组装为 LCEL 管道，
演示最简单的 chain = prompt | llm | parser 模式。
"""

import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 加载环境变量
load_dotenv()

# ==========================================
# 1. 创建 JSON 输出解析器
# ==========================================
json_output_parser = JsonOutputParser()
format_instructions = json_output_parser.get_format_instructions()

# ==========================================
# 2. 构建 Prompt 模板
# ==========================================
prompt_template = PromptTemplate.from_template(
    "You are a helpful assistant that translates {input_language} to {output_language}.\n\n"
    "{sentence}\n\n"
    "Format instructions: {format_instructions}"
)

# ==========================================
# 3. 初始化模型
# ==========================================
llm = init_chat_model(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# ==========================================
# 4. 组装 LCEL 链并执行
# ==========================================
# LCEL 链：prompt → llm → json_output_parser
chain = prompt_template | llm | json_output_parser

response = chain.invoke({
    "input_language": "English",
    "output_language": "French",
    "format_instructions": format_instructions,
    "sentence": "Hello, how are you?",
})
print(response)
