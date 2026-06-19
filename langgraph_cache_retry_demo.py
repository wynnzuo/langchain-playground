"""
LangGraph 缓存策略 + 重试策略 Demo
====================================
演示如何在 LangGraph Agent 中使用框架自带的：
  - RedisCache + set_llm_cache()  — LLM 响应缓存（相同提问直接命中）
  - Runnable.with_retry()          — 工具调用失败自动重试（指数退避）

全部使用 langchain/langgraph 内置 API，无第三方重试库。

运行要求: Redis 服务在 localhost:6379
"""

import os
import time
import random
from typing import Literal

import redis as redis_client
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# ============================================================
# 0. Redis 连接
# ============================================================
r = redis_client.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    decode_responses=True,
    socket_connect_timeout=3,
)

try:
    r.ping()
    print(f"✅ Redis 已连接")
except redis_client.ConnectionError:
    print("❌ Redis 连接失败，请确保 Redis 服务在运行")
    exit(1)

# ============================================================
# 1. LLM 缓存 — langchain 内置 RedisCache
# ============================================================
# set_llm_cache(RedisCache(redis_client, ttl=N))
# 会自动缓存所有 llm.invoke() 的响应，相同 prompt+参数命中缓存
llm_cache = RedisCache(r, ttl=600)
set_llm_cache(llm_cache)
print("✅ LLM 缓存已启用 (RedisCache, TTL=600s)")

# ============================================================
# 2. LLM
# ============================================================
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# ============================================================
# 3. 工具定义 — 模拟 30% 概率故障，测试重试
# ============================================================
FAIL_RATE = 0.3

@tool
def get_city_info(city: str) -> str:
    """查询指定城市的人口和面积信息。"""
    if random.random() < FAIL_RATE:
        msg = f"⚠️  查询 {city} 时网络超时"
        print(f"   {msg}")
        raise ConnectionError(msg)

    time.sleep(0.3)  # 模拟耗时，方便观察缓存效果
    db = {
        "北京": "人口 2189 万，面积 16410 km²",
        "上海": "人口 2475 万，面积 6340 km²",
        "广州": "人口 1873 万，面积 7434 km²",
        "深圳": "人口 1768 万，面积 1997 km²",
        "成都": "人口 2126 万，面积 14335 km²",
    }
    info = db.get(city)
    if not info:
        return f"抱歉，没有 {city} 的数据"
    result = f"{city}: {info}"
    print(f"   ✅ {result}")
    return result


@tool
def get_calculator_result(expression: str) -> str:
    """计算数学表达式。"""
    if random.random() < FAIL_RATE:
        print("   ⚠️  计算服务暂时不可用")
        raise ConnectionError("计算服务暂时不可用")

    try:
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed)
        out = f"{expression} = {result}"
        print(f"   ✅ {out}")
        return out
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time() -> str:
    """获取当前服务器时间。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [get_city_info, get_calculator_result, get_current_time]
llm_with_tools = llm.bind_tools(tools)

# ============================================================
# 4. 构建 LangGraph — 用 with_retry() 实现重试
# ============================================================
# ToolNode 本身是 Runnable，直接用 .with_retry() 包装
# 不需要手写重试循环或引入第三方库
tool_node = ToolNode(tools).with_retry(
    stop_after_attempt=3,
    retry_if_exception_type=(ConnectionError,),
)


def agent_node(state: MessagesState) -> dict:
    """Agent 节点"""
    response = llm_with_tools.invoke(state["messages"])
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"   [Agent] → 调用 {tc['name']}({tc['args']})")
    else:
        print(f"   [Agent] → {response.content[:60]}...")
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["tools", END]:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)          # ← 带 with_retry 的 ToolNode

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

graph = builder.compile()


# ============================================================
# 5. 运行演示
# ============================================================
def print_sep(title: str):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


if __name__ == "__main__":
    # 清理 LLM 缓存（RedisCache.clear() 方法遍历删除所有缓存键）
    llm_cache.clear()
    print("🧹 Redis 缓存已清理\n")

    # ────────── Demo A: LLM 缓存 ──────────
    print_sep("DEMO A: LLM 缓存 — 相同提问第二次命中缓存")
    question = "北京和上海哪个城市人口更多？"

    for i in range(2):
        print(f"\n  {'─── 第 1 次调用（Miss）' if i == 0 else '─── 第 2 次调用（应命中缓存）'} ───")
        start = time.time()
        result = graph.invoke({"messages": [HumanMessage(content=question)]})
        elapsed = time.time() - start
        print(f"  🤖 AI: {result['messages'][-1].content[:60]}...")
        print(f"  ⏱  {elapsed:.2f}s")

    # ────────── Demo B: with_retry 重试 ──────────
    print_sep("DEMO B: 工具重试 — 故障时自动重试")
    print(f"  get_city_info 有 {FAIL_RATE*100:.0f}% 概率抛出 ConnectionError")
    print("  ToolNode.with_retry(stop_after_attempt=3) 自动重试\n")

    # 跑 4 次，展示重试效果
    for i in range(4):
        print(f"  ─── 第 {i+1} 次 ───")
        result = graph.invoke({"messages": [HumanMessage(content="查一下广州的人口和面积")]})
        print(f"  🤖 AI: {result['messages'][-1].content[:60]}...\n")

    # ────────── Demo C: 换问题 → Miss ──────────
    print_sep("DEMO C: 不同问题缓存不命中")
    question2 = "帮我算一下 (15 + 7) * 3"
    print(f"  😊 用户: {question2}")

    start = time.time()
    result = graph.invoke({"messages": [HumanMessage(content=question2)]})
    t = time.time() - start
    print(f"  🤖 AI: {result['messages'][-1].content[:60]}...")
    print(f"  ⏱  {t:.2f}s（第一次，未命中）\n")

    # 再问一次 → 命中
    print("  ─── 再问一次（命中 LLM 缓存） ───")
    start = time.time()
    result = graph.invoke({"messages": [HumanMessage(content=question2)]})
    t = time.time() - start
    print(f"  🤖 AI: {result['messages'][-1].content[:60]}...")
    print(f"  ⏱  {t:.2f}s（命中缓存）")

    print()
    print("=" * 65)
    print("✅  演示完毕！")
    print("=" * 65)

    # ────────── 可选清理 ──────────
    if input("\n是否清理 Redis 缓存? (y/N): ").lower() == "y":
        r.flushdb()
