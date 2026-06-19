"""
LangGraph Send Demo
====================
演示 Send API 的"动态扇出"模式：将一个任务拆成多个子任务并行执行，
然后汇总结果（Map-Reduce 模式）。

适用场景:
  - 同时搜索多个城市天气
  - 对多份文档分别做分析
  - 并行处理多条数据后合并
"""

import os
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import Annotated, TypedDict, Any
from typing_extensions import TypedDict
import operator

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# ============================================================
# 1. 定义状态
# ============================================================

class OverallState(TypedDict):
    """全局状态"""
    cities: list[str]               # 要查询的城市列表（输入）
    results: Annotated[list[str], operator.add]  # 各个城市的查询结果（自动合并）
    summary: str                    # 最终汇总

class CityState(TypedDict):
    """单个城市的子任务状态"""
    city: str

# ============================================================
# 2. 节点定义
# ============================================================

def city_analyst(state: CityState) -> dict:
    """
    子任务节点：分析单个城市。
    这个节点会被 Send 多次调用，每次处理一个城市，多个副本并行执行。
    """
    city = state["city"]
    print(f"   🔍 正在分析 {city}... (pid: {os.getpid()})")

    # 模拟耗时分析
    time.sleep(1)

    prompt = f"请用一句话介绍城市 {city} 的特色（50字以内）。"
    response = llm.invoke(prompt)

    result = f"🏙️  {city}: {response.content.strip()}"
    print(f"   ✅ {city} 分析完成")
    return {"results": [result]}        # operator.add 自动合并到总列表


def distributor(state: OverallState) -> dict:
    """
    分发节点：决定要并行处理哪些任务。
    用 Send() 返回多个子任务，框架会自动并行执行。
    """
    print(f"\n📤 分发 {len(state['cities'])} 个城市进行分析")
    # 关键：返回 Send 列表
    return {"results": []}


def summary_writer(state: OverallState) -> dict:
    """汇总节点：把并行结果合并成一段总结"""
    results = state["results"]
    print(f"\n📥 收到 {len(results)} 个分析结果，正在汇总...")

    text = "\n".join(results)
    prompt = f"根据以下城市分析，写一段连贯的旅行推荐：\n\n{text}"
    response = llm.invoke(prompt)

    return {"summary": response.content.strip()}


# ============================================================
# 3. 条件边：根据城市数量动态分发
# ============================================================

def route_cities(state: OverallState) -> list[Send]:
    """
    条件边的路由函数。
    不返回节点名称，而是返回 Send(node, arg) 列表。
    每个 Send 创建一条独立的执行分支，框架会并行运行所有分支。
    """
    return [Send("city_analyst", {"city": c}) for c in state["cities"]]


# ============================================================
# 4. 构建图
# ============================================================

builder = StateGraph(OverallState)

# 节点
builder.add_node("distributor", distributor)
builder.add_node("city_analyst", city_analyst)
builder.add_node("summary_writer", summary_writer)

# 边
builder.add_edge(START, "distributor")
builder.add_conditional_edges("distributor", route_cities)  # ← 关键：返回 Send 列表
builder.add_edge("city_analyst", "summary_writer")         # 所有分支执行完后再汇总
builder.add_edge("summary_writer", END)


# ============================================================
# 5. 运行
# ============================================================

if __name__ == "__main__":
    graph = builder.compile()

    cities = ["北京", "上海", "广州", "深圳", "成都", "杭州"]

    print("=" * 60)
    print("🚀 LangGraph Send Demo — 并行分析多个城市")
    print("=" * 60)
    print(f"\n📋 待分析城市: {', '.join(cities)}")
    print(f"   共 {len(cities)} 个城市，将并行执行\n")

    start = time.time()
    result = graph.invoke({"cities": cities})
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"✅ 全部完成！耗时: {elapsed:.1f}s（若不并行需约 {len(cities)}s）")
    print(f"{'=' * 60}\n")

    print("📊 各城市分析:")
    for r in result["results"]:
        print(f"  {r}")

    print(f"\n📝 汇总推荐:")
    print(f"  {result['summary']}")
