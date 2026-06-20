"""
LangGraph ContextSchema Demo
==============================
演示 context_schema 的用法：将运行时配置（用户ID、语言、模型选择等）
与 State 分离，节点通过 Runtime 参数访问。

优势:
  - 配置数据不污染 State
  - 每个 invoke 可传入不同配置
  - 类型安全（TypedDict）
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime


# ============================================================
# 1. 定义 State 和 Context
# ============================================================

class ChatState(TypedDict):
    messages: list[str]

class ChatContext(TypedDict):
    user_name: str
    language: str          # "中文" / "English"


# ============================================================
# 2. 节点 — 通过 runtime 参数访问 context
# ============================================================

def greet(state: ChatState, runtime: Runtime[ChatContext]) -> dict:
    """读取 context 中的配置，生成个性化回复"""
    ctx = runtime.context
    name = ctx["user_name"]
    lang = ctx["language"]

    if lang == "中文":
        msg = f"你好，{name}！欢迎使用 LangGraph。"
    else:
        msg = f"Hello, {name}! Welcome to LangGraph."

    print(f"  [context] user={name}, lang={lang}")
    return {"messages": state["messages"] + [msg]}


# ============================================================
# 3. 构建图 — 传入 context_schema
# ============================================================

builder = StateGraph(ChatState, context_schema=ChatContext)
builder.add_node("greet", greet)
builder.add_edge(START, "greet")
builder.add_edge("greet", END)

graph = builder.compile()


# ============================================================
# 4. 运行 — 每次 invoke 传入不同 context
# ============================================================

if __name__ == "__main__":
    # 第一次：中文，用户 Wynn
    r1 = graph.invoke({"messages": []}, context={"user_name": "Wynn", "language": "中文"})
    print(f"  result: {r1['messages'][-1]}\n")

    # 第二次：English，用户 Alice
    r2 = graph.invoke({"messages": []}, context={"user_name": "Alice", "language": "English"})
    print(f"  result: {r2['messages'][-1]}\n")

    # 第三次：中文，用户 Bob（无 state 残留）
    r3 = graph.invoke({"messages": []}, context={"user_name": "Bob", "language": "中文"})
    print(f"  result: {r3['messages'][-1]}")
