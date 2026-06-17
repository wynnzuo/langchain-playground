"""
PromptTemplate 使用示例
========================
演示 PromptTemplate 的三种调用方式：
  1. format() — 直接格式化
  2. partial_variables — 预填充部分变量
  3. invoke() — 通过 LCEL 接口调用
"""

from datetime import datetime
import os

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

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

# ==========================================
# 方式 1：format() — 字符串格式化
# ==========================================
print("=" * 60)
print("1️⃣  format() — 字符串模板直接格式化")
print("=" * 60)

prompt = PromptTemplate(
    template="你是一个有帮助的助手。请回答以下问题：{question}",
    input_variables=["question"],
)
prompt_text = prompt.format(question="你是谁？")
print(prompt_text)
print("\n---\n")

result = llm.invoke(prompt_text)
print(result.content)

# ==========================================
# 方式 2：partial_variables — 预填充变量
# ==========================================
print("\n" + "=" * 60)
print("2️⃣  partial_variables — 预填充时间变量")
print("=" * 60)

prompt1 = PromptTemplate.from_template(
    "你是一个有帮助的助手。现在的时间是:{time}，请回答以下问题：{question}",
    partial_variables={"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
)
prompt_text1 = prompt1.format(question="你是谁？")
print(prompt_text1)

# ==========================================
# 方式 3：invoke() — LCEL 接口
# ==========================================
print("\n" + "=" * 60)
print("3️⃣  invoke() — 通过 LCEL 接口调用（返回 StringPromptValue）")
print("=" * 60)

prompt2 = PromptTemplate.from_template(
    "你是一个有帮助的助手。现在的时间是:{time}，请回答以下问题：{question}",
    partial_variables={"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
)
prompt_text2 = prompt2.invoke({"question": "你是谁？"})
print(f"返回类型: {type(prompt_text2)}")
print(prompt_text2.to_string())
