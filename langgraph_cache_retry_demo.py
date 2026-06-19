"""
LangGraph 缓存策略 + 重试策略 Demo
====================================
演示如何在 LangGraph Agent 中使用：
  - Redis LLM 缓存（相同提问直接命中缓存，不再调用 API）
  - 工具结果缓存（相同参数的工具调用返回缓存值）
  - tenacity 重试（工具调用失败时自动重试）

架构:
  用户提问
    ↓
  ┌─ Agent LLM ───────────────────────┐
  │  langchain 内置 RedisCache(ttl)    │  ← 缓存 LLM 响应
  │  命中缓存 → 免 API 调用            │
  └──────────┬─────────────────────────┘
             ↓ 工具调用请求
  ┌─ ToolNode ─────────────────────────┐
  │  1. 查 Redis 工具缓存 (hash)       │  ← 缓存工具结果
  │  2. 未命中 → tenacity 重试调用工具  │  ← 失败自动重试
  │  3. 结果写回 Redis 缓存             │
  └──────────┬─────────────────────────┘
             ↓ 工具结果
  ┌─ Agent LLM (再次) ─────────────────┐
  │  用工具结果组织最终回答              │
  └────────────────────────────────────┘

运行要求: Redis 服务在 localhost:6379
"""

import os
import json
import time
import random
from typing import Literal

import redis as redis_client_module
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# ============================================================
# 0. Redis 连接 & 基础配置
# ============================================================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

r = redis_client_module.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True,         # 自动解码为字符串
    socket_connect_timeout=3,
)

# 验证 Redis 连接
try:
    r.ping()
    print(f"✅ Redis 已连接 ({REDIS_HOST}:{REDIS_PORT}/{REDIS_DB})")
except redis_client_module.ConnectionError:
    print("❌ Redis 连接失败，请确保 Redis 服务在运行")
    print("   brew services start redis   # macOS")
    print("   sudo systemctl start redis  # Linux")
    exit(1)

# ============================================================
# 1. 全局 LLM 缓存 — RedisCache
# ============================================================
# RedisCache(redis_, ttl=seconds) 自动缓存所有 llm.invoke() 的响应
# 相同 prompt + 相同参数 → 命中缓存 → 免 API 调用
llm_cache = RedisCache(r, ttl=3600)       # 缓存 1 小时
set_llm_cache(llm_cache)
print("✅ LLM 缓存已启用 (RedisCache, TTL=3600s)")

# ============================================================
# 2. 基础 LLM
# ============================================================
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# ============================================================
# 3. 工具定义 — 包含缓存 + 模拟失败
# ============================================================
TOOL_CACHE_PREFIX = "tool_cache:"       # Redis key 前缀

# --- 工具级别的 Redis 缓存 ---
def cache_tool_result(tool_name: str, args_str: str, result: str) -> None:
    """将工具调用结果存入 Redis Hash: tool_cache:<name> → {args: result}"""
    key = f"{TOOL_CACHE_PREFIX}{tool_name}"
    r.hset(key, args_str, result)
    r.expire(key, 300)                  # 工具结果缓存 5 分钟
    print(f"   📀 工具结果已缓存: {tool_name}({args_str})")


def get_cached_tool_result(tool_name: str, args_str: str) -> str | None:
    """从 Redis 读取缓存的工具结果"""
    key = f"{TOOL_CACHE_PREFIX}{tool_name}"
    cached = r.hget(key, args_str)
    if cached:
        print(f"   💥 工具结果缓存命中! {tool_name}({args_str})")
    return cached


# --- tenacity 重试装饰器 ---
# 配置: 最多重试 3 次, 指数退避 (1s → 2s → 4s)
def retry_with_backoff():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(ConnectionError),
        before_sleep=lambda retry_state: print(
            f"   🔄 重试第 {retry_state.attempt_number} 次..."
            if retry_state.attempt_number else ""
        ),
    )


# --- 工具1: 城市信息查询（模拟可能失败） ---
@tool
def get_city_info(city: str) -> str:
    """查询指定城市的人口和面积信息。"""
    args_str = json.dumps({"city": city}, ensure_ascii=False, sort_keys=True)

    # 1. 查工具缓存
    cached = get_cached_tool_result("get_city_info", args_str)
    if cached:
        return cached

    # 2. 模拟 30% 概率网络故障（演示重试）
    if random.random() < 0.3:
        print(f"   ⚠️  get_city_info({city}) 模拟网络故障!")
        raise ConnectionError(f"查询 {city} 时网络连接超时")

    # 3. 实际业务逻辑
    db = {
        "北京": {"population": "2189万", "area": "16410 km²"},
        "上海": {"population": "2475万", "area": "6340 km²"},
        "广州": {"population": "1873万", "area": "7434 km²"},
        "深圳": {"population": "1768万", "area": "1997 km²"},
        "成都": {"population": "2126万", "area": "14335 km²"},
        "杭州": {"population": "1237万", "area": "16853 km²"},
    }
    info = db.get(city)
    if not info:
        return f"抱歉，没有 {city} 的数据"

    # 4. 模拟处理耗时（看得清缓存效果）
    time.sleep(0.5)
    result = f"{city}: 人口 {info['population']}, 面积 {info['area']}"
    print(f"   ✅ get_city_info({city}) = {result}")

    # 5. 写回缓存
    cache_tool_result("get_city_info", args_str, result)
    return result


# --- 工具2: 汇率查询（带重试） ---
@tool
def get_calculator_result(expression: str) -> str:
    """计算数学表达式。"""
    args_str = json.dumps({"expression": expression}, ensure_ascii=False, sort_keys=True)

    cached = get_cached_tool_result("get_calculator_result", args_str)
    if cached:
        return cached

    # 模拟 20% 计算服务故障
    if random.random() < 0.2:
        print(f"   ⚠️ get_calculator_result 模拟计算服务故障!")
        raise ConnectionError("计算服务暂时不可用")

    try:
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed)
        result_str = f"{expression} = {result}"
        print(f"   ✅ {result_str}")

        cache_tool_result("get_calculator_result", args_str, result_str)
        return result_str
    except Exception as e:
        return f"计算错误: {e}"


# --- 工具3: 模拟稳定工具（不缓存、不失败） ---
@tool
def get_current_time() -> str:
    """获取当前服务器时间。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [get_city_info, get_calculator_result, get_current_time]
llm_with_tools = llm.bind_tools(tools)

# ============================================================
# 4. 构建 LangGraph — 带重试的 ToolNode 包装
# ============================================================

# 自定义带重试的工具执行
def tool_node_with_retry(state: MessagesState) -> dict:
    """包装 ToolNode，对每个工具调用添加 tenacity 重试"""
    # 使用标准 ToolNode 先处理
    tool_node = ToolNode(tools)

    # 获取工具调用
    last_msg = state["messages"][-1]
    if not getattr(last_msg, "tool_calls", None):
        return {}

    # 用 tenacity 重试装饰器来包裹调用
    @retry_with_backoff()
    def execute_with_retry():
        return tool_node.invoke(state)

    try:
        result = execute_with_retry()
        return result
    except Exception as e:
        print(f"   ❌ 工具调用最终失败（已重试 3 次）: {e}")
        # 返回错误消息
        return {"messages": [
            {"role": "tool", "content": f"工具调用失败: {e}", "tool_call_id": last_msg.tool_calls[0]["id"]}
        ]}


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


# 构建图
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node_with_retry)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

graph = builder.compile()


# ============================================================
# 5. 清理上次缓存（避免干扰演示效果）
# ============================================================
def reset_cache():
    """清理工具缓存和 LLM 缓存，便于演示"""
    # 清理工具缓存
    for key in r.scan_iter(f"{TOOL_CACHE_PREFIX}*"):
        r.delete(key)
    # 清理 LLM 缓存
    for key in r.scan_iter("llm_cache:*"):
        r.delete(key)
    print("🧹 Redis 缓存已清理\n")


# ============================================================
# 6. 运行演示
# ============================================================
def print_separator(title: str):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


if __name__ == "__main__":
    reset_cache()

    # ---- Demo A: LLM 缓存演示 ----
    print_separator("DEMO A: LLM 缓存 — 相同提问命中缓存，免 API 调用")

    question = "北京和上海哪个城市人口更多？"
    print(f"\n  😊 用户: {question}")

    print("  ─── 第 1 次调用（Miss，调用 API） ───")
    start = time.time()
    r1 = graph.invoke({"messages": [HumanMessage(content=question)]})
    t1 = time.time() - start
    print(f"  🤖 AI: {r1['messages'][-1].content[:60]}...")
    print(f"  ⏱  耗时: {t1:.2f}s")

    print("\n  ─── 第 2 次调用（应该命中 LLM Cache） ───")
    start = time.time()
    r2 = graph.invoke({"messages": [HumanMessage(content=question)]})
    t2 = time.time() - start
    print(f"  🤖 AI: {r2['messages'][-1].content[:60]}...")
    print(f"  ⏱  耗时: {t2:.2f}s")

    cache_hit_1 = t2 < t1 * 0.5  # 第二次明显更快 → 缓存命中
    print(f"  {'💥 LLM 缓存命中!' if cache_hit_1 else '⚠️  未命中（可能模型不同）'} "
          f"(第1次={t1:.2f}s, 第2次={t2:.2f}s)")

    # ---- Demo B: 工具缓存演示 ----
    print_separator("DEMO B: 工具结果缓存 — 相同参数直接返回")

    question2 = "查一下深圳的人口和面积"
    print(f"\n  😊 用户: {question2}")

    print("  ─── 第 1 次调用（工具结果未缓存，要真正执行） ───")
    r3 = graph.invoke({"messages": [HumanMessage(content=question2)]})
    print(f"  🤖 AI: {r3['messages'][-1].content[:60]}...")

    print("\n  ─── 第 2 次调用（工具结果应命中缓存） ───")
    r4 = graph.invoke({"messages": [HumanMessage(content=question2)]})
    print(f"  🤖 AI: {r4['messages'][-1].content[:60]}...")

    # ---- Demo C: 重试策略演示 ----
    print_separator("DEMO C: tenacity 重试 — 失败自动重试")

    print("  get_city_info 有 30% 概率模拟网络故障")
    print("  tenacity 配置: 最多重试 3 次, 指数退避 (1s/2s/4s)\n")

    for i in range(3):
        print(f"  ─── 测试 #{i+1}: 查询上海 ───")
        r5 = graph.invoke({"messages": [HumanMessage(content="上海的人口和面积是多少？")]})
        last = r5['messages'][-1]
        if last.type == "ai" and last.content:
            print(f"  🤖 AI: {last.content[:60]}...")
        elif last.type == "tool":
            print(f"  🔧 工具结果: {last.content}")
        print()

    # ---- Demo D: 综合测试 — 显示缓存统计数据 ----
    print_separator("DEMO D: 缓存统计总览")

    tool_cache_keys = list(r.scan_iter(f"{TOOL_CACHE_PREFIX}*"))
    # LLM 缓存键是哈希值，无法通过 key pattern 区分，用 dbsize 评估
    total_cache_entries = r.dbsize()

    print(f"  📊 工具缓存条目数:   {sum(r.hlen(k) for k in tool_cache_keys)} "
          f"({len(tool_cache_keys)} 个 key)")
    print(f"  📊 Redis DB 总写入数: {total_cache_entries} keys "
          f"(含 LLM 缓存哈希键)")
    print()

    # 演示全局缓存命中效果
    print("  💡 换个问题验证 LLM 缓存不命中:")
    question3 = "帮我算一下 (23 + 45) * 2"
    print(f"  😊 用户: {question3}")

    start = time.time()
    r6 = graph.invoke({"messages": [HumanMessage(content=question3)]})
    t6 = time.time() - start
    print(f"  🤖 AI: {r6['messages'][-1].content[:60]}...")
    print(f"  ⏱  耗时: {t6:.2f}s (第一次，Miss)")

    print("\n  ─── 再次提问（命中计算器缓存 + LLM 缓存） ───")
    start = time.time()
    r7 = graph.invoke({"messages": [HumanMessage(content=question3)]})
    t7 = time.time() - start
    print(f"  🤖 AI: {r7['messages'][-1].content[:60]}...")
    print(f"  ⏱  耗时: {t7:.2f}s (命中! {t7:.2f}s vs {t6:.2f}s)")

    print("\n" + "=" * 65)
    print("✅  演示完毕！")
    print("=" * 65)

    # ---- 清理缓存（可选） ----
    # 默认不清理，保留缓存供下次运行验证
    if input("\n是否清理 Redis 缓存? (y/N): ").lower() == "y":
        reset_cache()
