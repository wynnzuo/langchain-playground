"""
ChatPromptTemplate 使用示例
=============================
演示 ChatPromptTemplate 的多种构建方式：
  1. 直接传 tuple 列表
  2. from_messages() + dict
  3. **kwargs 解包
  4. MessagesPlaceholder 占位符
  5. "placeholder" 类型
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 方式 1：tuple 列表 — 最简洁
# ==========================================
print("=" * 60)
print("1️⃣  tuple 列表构建 ChatPromptTemplate")
print("=" * 60)

chat_prompt = ChatPromptTemplate(
    [
        ("system", "你是一个有帮助的助手。现在的时间是:{time}，请回答以下问题：{question}"),
        ("human", "{question}"),
        ("ai", "{answer}"),
    ]
)
messages = chat_prompt.format_messages(
    time="2023-10-01 12:00:00",
    question="什么是人工智能？",
    answer="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
)
print(messages)

# ==========================================
# 方式 2：from_messages() + dict 格式
# ==========================================
print("\n" + "=" * 60)
print("2️⃣  from_messages() + dict 构建")
print("=" * 60)

chat_prompt2 = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "你是一个有帮助的助手。现在的时间是:{time}，请回答以下问题：{question}"},
        {"role": "human", "content": "{question}"},
        {"role": "ai", "content": "{answer}"},
    ]
)
messages = chat_prompt2.format_messages(
    time="2023-10-01 12:00:00",
    question="什么是人工智能？",
    answer="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
)
print(messages)

# ==========================================
# 方式 3：**kwargs 解包
# ==========================================
print("\n" + "=" * 60)
print("3️⃣  **kwargs 解包传参")
print("=" * 60)

chat_prompt3 = ChatPromptTemplate(
    [
        ("system", "你是一个有帮助的助手。"),
        ("human", "{question}"),
        ("ai", "{answer}"),
    ]
)
messages = chat_prompt3.format_messages(
    **{
        "question": "什么是人工智能？",
        "answer": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
    }
)
print(messages)

# ==========================================
# 方式 4：MessagesPlaceholder — 插槽占位
# ==========================================
print("\n" + "=" * 60)
print("4️⃣  MessagesPlaceholder — 动态插入历史消息")
print("=" * 60)

chat_prompt4 = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有帮助的助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
messages = chat_prompt4.format_messages(
    history=[
        SystemMessage(content="你是一个有帮助的助手。"),
        HumanMessage(content="什么是人工智能？"),
        SystemMessage(content="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"),
    ],
    question="什么是人工智能？",
)
print(messages)

# ==========================================
# 方式 5："placeholder" 类型 — MessagesPlaceholder 的简写
# ==========================================
print("\n" + "=" * 60)
print("5️⃣  'placeholder' 类型（MessagesPlaceholder 简写）")
print("=" * 60)

chat_prompt5 = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有帮助的助手。"),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ]
)
messages = chat_prompt5.format_messages(
    history=[
        SystemMessage(content="你是一个有帮助的助手。"),
        HumanMessage(content="什么是人工智能？"),
        SystemMessage(content="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"),
    ],
    question="什么是人工智能？",
)
print(messages)
