"""
RunnableWithMessageHistory 使用示例
======================================
将 InMemoryChatMessageHistory 与 LCEL 链组合，实现带记忆的多轮对话。

注意：RunnableWithMessageHistory 已废弃，LangChain 推荐改用
LangGraph 的持久化方案。但对简单的内存对话场景仍然可用。
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage

load_dotenv()

llm = init_chat_model(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)


# ==========================================
# 1. 基础：带记忆的对话链
# ==========================================
print("=" * 60)
print("1️⃣  RunnableWithMessageHistory 基础用法")
print("=" * 60)

# 用一个 dict 存储所有会话，同一个 session_id 返回同一个 history
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手，请用中文回答。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain_with_history = RunnableWithMessageHistory(
    runnable=prompt | llm,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("\n第一轮对话:")
resp = chain_with_history.invoke(
    {"input": "我叫张三"},
    config={"configurable": {"session_id": "demo"}},
)
print(f"用户: 我叫张三")
print(f"助手: {resp.content}")

print("\n第二轮（验证记忆）:")
resp = chain_with_history.invoke(
    {"input": "我叫什么名字？"},
    config={"configurable": {"session_id": "demo"}},
)
print(f"用户: 我叫什么名字？")
print(f"助手: {resp.content}")

print("\n第三轮:")
resp = chain_with_history.invoke(
    {"input": "帮我造个句，用上我的名字"},
    config={"configurable": {"session_id": "demo"}},
)
print(f"用户: 帮我造个句，用上我的名字")
print(f"助手: {resp.content}")


# ==========================================
# 2. 多会话隔离
# ==========================================
print("\n" + "=" * 60)
print("2️⃣  多会话隔离")
print("=" * 60)

resp_a1 = chain_with_history.invoke(
    {"input": "我喜欢篮球"},
    config={"configurable": {"session_id": "user_a"}},
)
resp_a2 = chain_with_history.invoke(
    {"input": "我喜欢什么运动？"},
    config={"configurable": {"session_id": "user_a"}},
)

resp_b1 = chain_with_history.invoke(
    {"input": "我喜欢画画"},
    config={"configurable": {"session_id": "user_b"}},
)
resp_b2 = chain_with_history.invoke(
    {"input": "我喜欢什么运动？"},
    config={"configurable": {"session_id": "user_b"}},
)

print(f"用户 A: 我喜欢什么运动？→ {resp_a2.content}")
print(f"用户 B: 我喜欢什么运动？→ {resp_b2.content}")


# ==========================================
# 3. 限制历史长度（滑动窗口）
# ==========================================
print("\n" + "=" * 60)
print("3️⃣  限制历史长度 — 手动裁剪")
print("=" * 60)


def trim_history(session_id: str, max_pairs: int = 2):
    """裁剪历史，只保留最近的 max_pairs 轮"""
    hist = store.get(session_id)
    if not hist:
        return
    # 获取原始消息
    msgs = hist.messages
    system_msgs = [m for m in msgs if m.type == "system"]
    non_system = msgs[len(system_msgs):]
    if len(non_system) > max_pairs * 2:
        # 重建消息列表：system + 最近 N 轮
        hist.messages = system_msgs + non_system[-(max_pairs * 2):]


prompt_trimmed = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain_trimmed = RunnableWithMessageHistory(
    runnable=prompt_trimmed | llm,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("\n连续对话，每次调用后裁剪到最近 2 轮:")
session = "trim_session"
for i in range(5):
    resp = chain_trimmed.invoke(
        {"input": f"这是第 {i+1} 轮说的内容"},
        config={"configurable": {"session_id": session}},
    )
    # 调用后裁剪历史
    trim_history(session, max_pairs=2)
    hist_len = len(store[session].messages)
    print(f"  第 {i+1} 轮 → {resp.content[:30]}... (历史保留 {hist_len} 条)")


# ==========================================
# 4. 纯手动管理记忆（不用 RunnableWithMessageHistory）
# ==========================================
print("\n" + "=" * 60)
print("4️⃣  纯手动管理记忆 — 完全掌控")
print("=" * 60)

manual_history = InMemoryChatMessageHistory()
manual_history.add_message(SystemMessage(content="你是一个数学助手，用中文回答。"))

questions = [
    "1+1等于几？",
    "加上2呢？",
    "再加上3呢？",
]

for q in questions:
    print(f"\n用户: {q}")
    manual_history.add_user_message(q)

    resp = llm.invoke(manual_history.messages)
    print(f"助手: {resp.content}")

    manual_history.add_ai_message(resp.content)
    # 手动裁剪：只保留 system + 最近 3 轮 (6 条 non-system)
    msgs = manual_history.messages
    system = [m for m in msgs if m.type == "system"]
    others = msgs[len(system):]
    manual_history.messages = system + others[-6:]

print(f"\n最终消息数: {len(manual_history.messages)}（保留最近 3 轮）")
