"""
LCEL 分支链（RunnableBranch）示例
===================================
根据输入条件路由到不同的处理链，类似 if/else 但以链式方式组合。
还演示 RunnableParallel 并行执行以及两者的组合使用。
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# 初始化 LLM
llm = init_chat_model(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)


# ==========================================
# 1. RunnableBranch — 条件分支
# ==========================================
print("=" * 60)
print("1️⃣  RunnableBranch 条件分支")
print("=" * 60)

# 语言检测链
detect_language_prompt = PromptTemplate.from_template(
    "分析以下句子是哪种语言，只返回语言名称（如 English、中文、日本語 等）。\n\n句子：{sentence}"
)
detect_chain = detect_language_prompt | llm | StrOutputParser()

# 不同语言的翻译链
translate_to_chinese = (
    PromptTemplate.from_template("将以下句子翻译成中文：\n\n{sentence}")
    | llm
    | StrOutputParser()
)

translate_to_english = (
    PromptTemplate.from_template("将以下句子翻译成英语：\n\n{sentence}")
    | llm
    | StrOutputParser()
)

translate_to_japanese = (
    PromptTemplate.from_template("将以下句子翻译成日语：\n\n{sentence}")
    | llm
    | StrOutputParser()
)

# 默认链（其他语言统一翻成中文）
translate_default = (
    PromptTemplate.from_template("将以下句子翻译成中文：\n\n{sentence}")
    | llm
    | StrOutputParser()
)

# 构建分支：根据语言选择对应的翻译链
branch = RunnableBranch(
    (lambda x: "中文" in x["language"], translate_to_english),
    (lambda x: "English" in x["language"], translate_to_chinese),
    (lambda x: "日本" in x["language"], translate_to_japanese),
    translate_default,  # 默认分支
)

# 组装完整链：先检测语言 → 再路由到对应翻译
full_chain = (
    RunnableParallel(language=detect_chain, sentence=lambda x: x["sentence"])
    | branch
)

test_sentences = [
    "Hello, how are you?",
    "今天天气真不错",
    "こんにちは、元気ですか？",
    "Bonjour, tout le monde!",
]

for s in test_sentences:
    print(f"\n输入: {s}")
    result = full_chain.invoke({"sentence": s})
    print(f"输出: {result}")


# ==========================================
# 2. RunnableParallel — 并行分支
# ==========================================
print("\n" + "=" * 60)
print("2️⃣  RunnableParallel 并行分支")
print("=" * 60)

summary_prompt = PromptTemplate.from_template("用一句话总结：{sentence}")
keywords_prompt = PromptTemplate.from_template("提取关键词（逗号分隔）：{sentence}")
sentiment_prompt = PromptTemplate.from_template("判断情感倾向（正面/负面/中性）：{sentence}")

parallel_chain = RunnableParallel(
    summary=summary_prompt | llm | StrOutputParser(),
    keywords=keywords_prompt | llm | StrOutputParser(),
    sentiment=sentiment_prompt | llm | StrOutputParser(),
)

text = "LangChain is an amazing framework that makes it easy to build LLM applications!"
result = parallel_chain.invoke({"sentence": text})

print(f"\n输入: {text}")
print(f"摘要:   {result['summary']}")
print(f"关键词: {result['keywords']}")
print(f"情感:   {result['sentiment']}")

# 可视化链结构
parallel_chain.get_graph().print_ascii()


# ==========================================
# 3. RunnableParallel + RunnableBranch 组合
# ==========================================
print("\n" + "=" * 60)
print("3️⃣  组合使用：并行分析 + 条件路由")
print("=" * 60)


class AnalysisResult(BaseModel):
    """文本分析结果"""
    summary: str = Field(description="内容摘要")
    language: str = Field(description="检测到的语言")
    sentiment: str = Field(description="情感倾向")
    word_count: int = Field(description="单词/字符数")


analysis_chain = RunnableParallel(
    summary=PromptTemplate.from_template("用一句话总结：{sentence}") | llm | StrOutputParser(),
    language=detect_chain,
    sentiment=sentiment_prompt | llm | StrOutputParser(),
    word_count=lambda x: len(x["sentence"]),
)

input_text = "I love learning new things every day!"
result = analysis_chain.invoke({"sentence": input_text})
print(f"\n输入: {input_text}")
print(f"语言:   {result['language']}")
print(f"摘要:   {result['summary']}")
print(f"情感:   {result['sentiment']}")
print(f"字数:   {result['word_count']}")
