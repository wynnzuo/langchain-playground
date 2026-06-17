"""
PydanticOutputParser 使用示例
==============================
调用大模型并返回结构化数据，用 Pydantic 模型进行校验和解析。
适用于需要强类型约束、字段验证的场景。
"""

import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()


# ==========================================
# 1. 定义 Pydantic 模型（输出的数据结构）
# ==========================================
class MovieReview(BaseModel):
    """一部电影的评论分析结果"""
    title: str = Field(description="电影的中文标题")
    year: int = Field(description="上映年份")
    rating: float = Field(description="评分（满分 10 分）", ge=0, le=10)
    summary: str = Field(description="简短影评（50 字以内）")
    tags: list[str] = Field(description="电影标签，如 科幻、动作、剧情")


# ==========================================
# 2. 创建 Parser 并获取格式指令
# ==========================================
parser = PydanticOutputParser(pydantic_object=MovieReview)
format_instructions = parser.get_format_instructions()

print("=" * 60)
print("格式指令（会注入到 Prompt 中）:")
print("-" * 60)
print(format_instructions)

# ==========================================
# 3. 构建 Prompt
# ==========================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的电影评论助手。请严格按照要求的格式输出。"),
    ("human", "请分析电影：{movie_name}\n\n{format_instructions}"),
])

messages = prompt.format_messages(
    movie_name="《盗梦空间》",
    format_instructions=format_instructions,
)

print("=" * 60)
print("\n发送给模型的 Prompt:")
print("-" * 60)
for msg in messages:
    print(f"[{msg.type}]\n{msg.content}\n")

# ==========================================
# 4. 调用大模型
# ==========================================
llm = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

print("=" * 60)
print("正在调用模型...")
result = llm.invoke(messages)
print("\n模型原始输出:")
print("-" * 60)
print(result.content)

# ==========================================
# 5. 用 PydanticOutputParser 解析结果
# ==========================================
print("\n" + "=" * 60)
print("PydanticOutputParser 解析结果:")
print("-" * 60)

try:
    parsed: MovieReview = parser.invoke(result)
    print(f"电影: {parsed.title} ({parsed.year})")
    print(f"评分: {parsed.rating}/10")
    print(f"影评: {parsed.summary}")
    print(f"标签: {', '.join(parsed.tags)}")
    print(f"\n类型: {type(parsed)}")
    print(f"模型字段验证通过 ✅")
except Exception as e:
    print(f"解析失败: {e}")

# ==========================================
# 6. 也能手动访问原始字典
# ==========================================
print("\n" + "=" * 60)
print("原始字典:")
print("-" * 60)
print(json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2))
