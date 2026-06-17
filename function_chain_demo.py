"""
LCEL 函数链（RunnableLambda / RunnablePassthrough）示例
========================================================
将自定义 Python 函数嵌入 LCEL 链中，实现数据清洗、校验、转换等操作。
演示四种组合模式：管道连接、透传赋值、链式 LCEL、并行执行。
"""

import os
import re
import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

llm = init_chat_model(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)


# ==========================================
# 1. RunnableLambda — 自定义函数包装
# ==========================================
print("=" * 60)
print("1️⃣  RunnableLambda — 将函数变成链的一环")
print("=" * 60)


def clean_text(text: str) -> str:
    """清洗文本：去首尾空格、合并多个空格"""
    return re.sub(r"\s+", " ", text.strip())


def count_words(text: str) -> dict:
    """统计文本信息"""
    words = text.split()
    return {
        "original": text,
        "word_count": len(words),
        "char_count": len(text),
    }


# 用 RunnableLambda 把普通函数包成链组件
clean_step = RunnableLambda(clean_text)
count_step = RunnableLambda(count_words)

# 组装：输入 → 清洗 → 统计
clean_and_count = clean_step | count_step

result = clean_and_count.invoke("   Hello    world!   This   is   LangChain.   ")
print("输入: '   Hello    world!   This   is   LangChain.   '")
print(f"输出: {result}")


# ==========================================
# 2. RunnablePassthrough — 透传 / 添加字段
# ==========================================
print("\n" + "=" * 60)
print("2️⃣  RunnablePassthrough — 透传数据或注入新字段")
print("=" * 60)

# .assign() 在不丢失原有数据的情况下添加新字段
add_timestamp = RunnablePassthrough.assign(
    length=lambda x: len(x["text"]),
    first_word=lambda x: x["text"].split()[0] if x["text"] else "",
)

data = {"text": "LangChain is powerful", "language": "English"}
result = add_timestamp.invoke(data)
print(f"输入: {data}")
print(f"输出: {result}")


# ==========================================
# 3. 函数链 + LLM 组合 — 真实场景
# ==========================================
print("\n" + "=" * 60)
print("3️⃣  函数链 + LLM — 文本预处理 → LLM → 后处理")
print("=" * 60)


# --- 前置处理 ---
def validate_input(data: dict) -> dict:
    """校验并预处理输入"""
    text = data.get("text", "")
    if not text.strip():
        raise ValueError("输入文本不能为空")
    return {"text": text[:500], "language": data.get("language", "中文")}  # 截断


def build_prompt(data: dict) -> dict:
    """根据语言构建不同的 prompt"""
    language = data["language"]
    text = data["text"]
    if language == "English":
        return {"question": f"Please summarize the following text:\n\n{text}"}
    else:
        return {"question": f"请用一句话总结以下内容：\n\n{text}"}


# --- 后置处理 ---
def format_response(text: str) -> str:
    """格式化输出"""
    return f"📝 总结结果：\n{text.strip()}"


# 组装完整链
summary_chain = (
    RunnableLambda(validate_input)
    | RunnableLambda(build_prompt)
    | PromptTemplate.from_template("{question}")
    | llm
    | StrOutputParser()
    | RunnableLambda(format_response)
)

test_cases = [
    {
        "text": "LangChain is a framework for developing applications powered by large language models. "
                "It enables applications that are context-aware and reason.",
        "language": "English",
    },
    {
        "text": "人工智能正在深刻改变各行各业，从医疗诊断到自动驾驶，AI 的应用范围越来越广。",
        "language": "中文",
    },
]

for case in test_cases:
    print(f"\n输入: {case['text'][:40]}...")
    result = summary_chain.invoke(case)
    print(result)


# ==========================================
# 4. 多种函数组合方式对比
# ==========================================
print("\n" + "=" * 60)
print("4️⃣  多种函数组合方式对比")
print("=" * 60)


def double(x: int) -> int:
    return x * 2


def add_one(x: int) -> int:
    return x + 1


def to_string(x: int) -> str:
    return f"结果是: {x}"


# 方式一：管道连接（顺序执行）
pipeline = RunnableLambda(double) | RunnableLambda(add_one) | RunnableLambda(to_string)
print(f"管道连接: 3 → *2 → +1 → str  =  {pipeline.invoke(3)}")

# 方式二：并行执行
parallel = RunnableParallel(
    double=RunnableLambda(double),
    add_one=RunnableLambda(add_one),
)
print(f"并行执行: 5 → {{double: *2, add_one: +1}}  =  {parallel.invoke(5)}")

# 方式三：RunnablePassthrough.assign 链式添加字段
chain = (
    RunnablePassthrough.assign(
        doubled=lambda x: x["value"] * 2,
    ).assign(
        added=lambda x: x["doubled"] + 1,
    )
)
print(f"链式添加: {{'value': 3}} → 先 double → 再 +1  =  {chain.invoke({'value': 3})}")
