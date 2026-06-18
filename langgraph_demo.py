"""
LangGraph Demo
==============
演示 LangGraph 的核心概念：StateGraph、条件边、Agent+Tool 循环、
多 Agent 协作、MemorySaver 持久化。

概念速览:
  - StateGraph:  带状态（全局 State）的图，节点读写同一份状态
  - 节点 (Node):  一个函数，接收 State，返回要更新的字段
  - 边 (Edge):    节点间的连接
  - 条件边:       根据 State 内容决定走哪条边
  - MemorySaver:  持久化对话状态，支持多轮交互
"""

import os
import json
import random
from typing import Literal, Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessageGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ──────────────────────────────────────────────────────────────
# 全局 LLM（项目统一使用 DeepSeek）
# ──────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# ============================================================
# DEMO 1: 基础 StateGraph — 最简单的"状态机"
# ============================================================
def demo_basic_state_graph():
    """
    场景: 输入一个数字，依次经过三个节点：
      输入 → 翻倍 → 加一 → 转字符串
    每个节点读取 State 中的字段，写入新字段。
    """
    print("\n" + "=" * 65)
    print("📘 DEMO 1: 基础 StateGraph — 节点依次读写全局状态")
    print("=" * 65)

    # ---- 1. 定义状态类型 ----
    class BasicState(TypedDict):
        value: int           # 初始输入值
        doubled: int         # 翻倍后
        added: int           # 加一后
        message: str         # 最终字符串

    # ---- 2. 定义节点函数 ----
    # 节点函数签名: (state) -> dict  # dict 中的 key 会更新到全局 state

    def double_value(state: BasicState) -> dict:
        """翻倍"""
        doubled = state["value"] * 2
        print(f"   [double_value] {state['value']} × 2 = {doubled}")
        return {"doubled": doubled}

    def add_one(state: BasicState) -> dict:
        """加一"""
        added = state["doubled"] + 1
        print(f"   [add_one] {state['doubled']} + 1 = {added}")
        return {"added": added}

    def to_message(state: BasicState) -> dict:
        """转成字符串"""
        msg = f"最终结果: {state['added']}"
        print(f"   [to_message] → '{msg}'")
        return {"message": msg}

    # ---- 3. 构建图 ----
    builder = StateGraph(BasicState)                  # 创建一个状态图

    builder.add_node("double", double_value)          # 注册节点（名称 → 函数）
    builder.add_node("add_one", add_one)
    builder.add_node("to_msg", to_message)

    builder.add_edge(START, "double")                 # 入口 → 第一个节点
    builder.add_edge("double", "add_one")             # 节点间顺序连接
    builder.add_edge("add_one", "to_msg")
    builder.add_edge("to_msg", END)                   # 最后一个节点 → 出口

    graph = builder.compile()                         # 编译成可执行图

    # ---- 4. 执行 ----
    result = graph.invoke({"value": 5})               # 传入初始 state
    print(f"\n   结果: {result}")


# ============================================================
# DEMO 2: 条件边 — 根据状态走不同分支
# ============================================================
def demo_conditional_edge():
    """
    场景: 输入一个分数，根据分数走不同分支：
      ≥ 90 → "优秀" 分支
      ≥ 60 → "及格" 分支
      < 60 → "不及格" 分支
    """
    print("\n" + "=" * 65)
    print("📘 DEMO 2: 条件边 — 根据 State 内容路由到不同节点")
    print("=" * 65)

    class GradeState(TypedDict):
        score: int
        grade: str
        suggestion: str

    def check_grade(state: GradeState) -> dict:
        """根据分数判定等级"""
        s = state["score"]
        if s >= 90:
            grade = "优秀"
        elif s >= 60:
            grade = "及格"
        else:
            grade = "不及格"
        print(f"   [check_grade] {s}分 → {grade}")
        return {"grade": grade}

    def give_excellent_suggestion(state: GradeState) -> dict:
        s = state["score"]
        suggestion = (
            f"太棒了！{s}分属于优秀水平，继续保持！"
            if s >= 95
            else f"{s}分很不错！还可以再往上冲一冲满分💪"
        )
        print(f"   [优秀分支] {suggestion}")
        return {"suggestion": suggestion}

    def give_pass_suggestion(state: GradeState) -> dict:
        suggestion = f"{state['score']}分，及格了，但还可以更好哦！"
        print(f"   [及格分支] {suggestion}")
        return {"suggestion": suggestion}

    def give_fail_suggestion(state: GradeState) -> dict:
        suggestion = f"{state['score']}分… 需要努力了！加油！"
        print(f"   [不及格分支] {suggestion}")
        return {"suggestion": suggestion}

    # ---- 条件路由函数 ----
    # 接收 state，返回要去的目的地节点名称
    def route_by_grade(state: GradeState) -> Literal["excellent", "pass", "fail"]:
        if state["grade"] == "优秀":
            return "excellent"
        elif state["grade"] == "及格":
            return "pass"
        else:
            return "fail"

    # ---- 构建图 ----
    builder = StateGraph(GradeState)
    builder.add_node("check", check_grade)
    builder.add_node("excellent", give_excellent_suggestion)
    builder.add_node("pass", give_pass_suggestion)
    builder.add_node("fail", give_fail_suggestion)

    builder.add_edge(START, "check")

    # 条件边: 从 check 出发, 根据 route_by_grade 的返回值选择下一个节点
    builder.add_conditional_edges("check", route_by_grade)

    builder.add_edge("excellent", END)
    builder.add_edge("pass", END)
    builder.add_edge("fail", END)

    graph = builder.compile()

    # ---- 分别测试三个分支 ----
    for score in [95, 72, 43]:
        print(f"\n  ▶ 输入分数: {score}")
        result = graph.invoke({"score": score})
        print(f"    结果: {result['grade']} — {result['suggestion']}")


# ============================================================
# DEMO 3: MessageGraph / MessagesState — 聊天对话
# ============================================================
def demo_message_graph():
    """
    场景: 多轮对话，利用 MessagesState 管理聊天消息列表。
    节点可以读取/追加消息，用 add_messages reducer 自动合并。
    """
    print("\n" + "=" * 65)
    print("📘 DEMO 3: MessagesState — 管理聊天消息")
    print("=" * 65)

    # ---- 1. 定义节点 ----
    def chat_node(state: MessagesState) -> dict:
        """用 LLM 回复用户的上一条消息"""
        messages = state["messages"]
        # 取最后一条用户消息来回复
        last_msg = messages[-1]
        print(f"   [chat_node] 收到: {last_msg.content[:40]}{'...' if len(last_msg.content) > 40 else ''}")

        response = llm.invoke(messages)
        print(f"   [chat_node] 回复: {response.content[:40]}...")
        return {"messages": [response]}        # add_messages 会自动追加

    # ---- 2. 构建图 ----
    builder = StateGraph(MessagesState)                # MessagesState = {messages: list[BaseMessage]}
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)

    graph = builder.compile()

    # ---- 3. 执行多轮对话 ----
    config = {"configurable": {"thread_id": "demo3"}}  # thread_id 标识对话会话

    print("\n  🤖 开始多轮对话:\n")
    for user_input in ["你好！", "用中文介绍一下 LangGraph 是什么？", "刚才你提到了 Agent，能展开说说吗？"]:
        print(f"  😊 用户: {user_input}")
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,          # 传递 config 以便跨轮次保持状态
        )
        print(f"  🤖 AI: {result['messages'][-1].content}\n")


# ============================================================
# DEMO 4: Agent + Tool — ReAct 循环（最实用场景）
# ============================================================
def demo_agent_with_tools():
    """
    场景: AI Agent 可以自主决定是否调用工具。
    流程:
      LLM 思考 → 需要工具? → [是] 调用工具 → 继续思考 → ...
                         → [否] 输出最终回答

    这里用 ToolNode + 条件边实现标准的 ReAct 模式。
    """
    print("\n" + "=" * 65)
    print("📘 DEMO 4: Agent + Tool — ReAct 循环（手动构建）")
    print("=" * 65)

    # ---- 1. 定义工具 ----
    @tool
    def roll_dice(sides: int) -> str:
        """掷一个指定面数的骰子，返回点数。"""
        result = random.randint(1, sides)
        return f"🎲 掷出了 {sides} 面骰子: {result} 点"

    @tool
    def get_current_time() -> str:
        """获取当前服务器时间。"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式，如 '2 + 3 * 4'。"""
        try:
            # 安全地计算表达式
            allowed_names = {"__builtins__": {}}
            result = eval(expression, allowed_names)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    tools = [roll_dice, get_current_time, calculator]
    tools_by_name = {t.name: t for t in tools}

    # 用 ToolNode 统一管理工具调用
    tool_node = ToolNode(tools)

    # ---- 2. 绑定工具到 LLM ----
    llm_with_tools = llm.bind_tools(tools)

    # ---- 3. 定义节点 ----
    def agent_node(state: MessagesState) -> dict:
        """Agent 节点: LLM 决定是否调用工具"""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)

        # 打印思考过程
        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"   [Agent] 调用工具: {tc['name']}({tc['args']})")
        else:
            print(f"   [Agent] 最终回答: {response.content[:50]}...")

        return {"messages": [response]}

    # ---- 4. 条件路由: Agent 输出包含 tool_calls 就去 ToolNode，否则结束 ----
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("   → 需要调用工具，路由到 tools 节点")
            return "tools"
        print("   → 无需调用工具，结束")
        return END

    # ---- 5. 构建图 ----
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue)
    builder.add_edge("tools", "agent")           # 工具调用完后回到 agent 继续

    graph = builder.compile()

    # ---- 6. 测试 ----
    queries = [
        "现在几点了？",
        "帮我计算 (15 + 7) * 3 等于多少",
        "掷一个 12 面骰子",
        "你好，今天天气怎么样？",   # 没有天气工具，看 LLM 如何回应
    ]

    for query in queries:
        print(f"\n  😊 用户: {query}")
        result = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
        )
        # 打印最终回答
        final_msg = result["messages"][-1]
        print(f"  🤖 AI: {final_msg.content}")


# ============================================================
# DEMO 5: 多 Agent 协作 — 翻译 + 审校流水线
# ============================================================
def demo_multi_agent_pipeline():
    """
    场景: 两个 Agent 接力完成工作：
      Agent A (Translator):  将英文翻译成中文
      Agent B (Reviewer):    审校翻译质量，给出改进建议

    演示一个 Agent 的输出作为另一个 Agent 的输入。
    """
    print("\n" + "=" * 65)
    print("📘 DEMO 5: 多 Agent 协作 — 翻译 + 审校流水线")
    print("=" * 65)

    class PipelineState(TypedDict):
        source_text: str           # 原始文本
        translated: str            # 翻译结果
        review: str                # 审校意见
        final_version: str         # 最终版本

    def translator(state: PipelineState) -> dict:
        """翻译节点"""
        prompt = (
            f"请将以下英文翻译成中文，保留专业术语原样:\n\n{state['source_text']}\n\n"
            "只返回翻译结果，不要附加解释。"
        )
        response = llm.invoke(prompt)
        translated = response.content.strip()
        print(f"   [Translator] 翻译完成 ({len(translated)} 字)")
        return {"translated": translated}

    def reviewer(state: PipelineState) -> dict:
        """审校节点"""
        prompt = (
            f"以下是某段英文的中文翻译，请审校其准确性和流畅度。\n\n"
            f"原文: {state['source_text']}\n\n"
            f"译文: {state['translated']}\n\n"
            "请指出问题并提出改进建议。"
        )
        response = llm.invoke(prompt)
        review = response.content.strip()
        print(f"   [Reviewer] 审校完成 ({len(review)} 字)")
        return {"review": review}

    def finalizer(state: PipelineState) -> dict:
        """根据审校意见输出最终版本"""
        prompt = (
            f"根据以下审校意见，修改翻译。\n\n"
            f"原文: {state['source_text']}\n\n"
            f"原译文: {state['translated']}\n\n"
            f"审校意见: {state['review']}\n\n"
            "输出修改后的最终翻译版本。"
        )
        response = llm.invoke(prompt)
        final = response.content.strip()
        print(f"   [Finalizer] 最终版本完成")
        return {"final_version": final}

    builder = StateGraph(PipelineState)
    builder.add_node("translator", translator)
    builder.add_node("reviewer", reviewer)
    builder.add_node("finalizer", finalizer)

    builder.add_edge(START, "translator")
    builder.add_edge("translator", "reviewer")
    builder.add_edge("reviewer", "finalizer")
    builder.add_edge("finalizer", END)

    graph = builder.compile()

    # ---- 测试 ----
    test_text = (
        "LangGraph is a framework for building stateful, multi-agent applications "
        "with LLMs. It extends LangChain's capabilities by introducing a graph-based "
        "execution model, where nodes represent computation steps and edges define "
        "the control flow between them."
    )

    print(f"\n  📝 原文: {test_text[:50]}...\n")
    result = graph.invoke({"source_text": test_text})

    print(f"\n  📖 翻译:")
    print(f"  {result['translated']}")
    print(f"\n  📋 审校:")
    print(f"  {result['review']}")
    print(f"\n  ✅ 最终版本:")
    print(f"  {result['final_version']}")


# ============================================================
# DEMO 6: MemorySaver — 持久化对话状态（跨轮次记忆）
# ============================================================
def demo_memory_saver():
    """
    场景: 带记忆的 Chat Agent。多轮对话中 Agent 能记住之前说过的话。
    使用 MemorySaver(checkpointer) 实现状态持久化。
    """
    print("\n" + "=" * 65)
    print("📘 DEMO 6: MemorySaver — 跨轮次对话记忆")
    print("=" * 65)

    @tool
    def roll_dice(sides: int) -> str:
        """掷一个指定面数的骰子，返回点数。"""
        result = random.randint(1, sides)
        return f"🎲 掷出了 {sides} 面骰子: {result} 点"

    tools = [roll_dice]
    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: MessagesState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"   [Agent] 调用: {tc['name']}({tc['args']})")
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["tools", END]:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    # ---- 创建带 checkpointer 的图 ----
    memory = MemorySaver()

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue)
    builder.add_edge("tools", "agent")

    # compile 时传入 checkpointer
    graph = builder.compile(checkpointer=memory)

    # ---- 多轮对话，验证记忆 ----
    thread_config = {"configurable": {"thread_id": "memory-demo"}}

    print("\n  ⏳ 第一轮: 掷骰子")
    graph.invoke(
        {"messages": [HumanMessage(content="掷一个 20 面骰子")]},
        config=thread_config,
    )

    print("\n  ⏳ 第二轮: 问刚才掷骰子的结果")
    result = graph.invoke(
        {"messages": [HumanMessage(content="我刚才掷骰子得到了几点？")]},
        config=thread_config,   # 同一个 thread_id → 能访问历史
    )
    print(f"  🤖 AI: {result['messages'][-1].content}")

    print("\n  ⏳ 第三轮: 换个新会话（不共享记忆）")
    thread_config2 = {"configurable": {"thread_id": "memory-demo-2"}}
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="我刚才掷骰子得到了几点？")]},
        config=thread_config2,  # 不同 thread_id → 不记得
    )
    print(f"  🤖 AI: {result2['messages'][-1].content}")
    print("   （换了会话，记忆丢失，Agent 无法回答）")


# ============================================================
# DEMO 7: 使用 prebuilt create_react_agent（快速上手）
# ============================================================
def demo_create_react_agent():
    """
    场景: 用 LangGraph 提供的 create_react_agent 快速创建一个 ReAct Agent。
    这是"把 DEMO 4 的手动构建简化到一行"的版本。
    """
    print("\n" + "=" * 65)
    print("📘 DEMO 7: create_react_agent — 一行创建 Agent")
    print("=" * 65)

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气信息。"""
        data = {
            "北京": "28°C, 晴, 湿度 45%",
            "上海": "32°C, 多云, 湿度 70%",
            "广州": "35°C, 雷阵雨, 湿度 80%",
            "深圳": "33°C, 多云转阴, 湿度 78%",
        }
        info = data.get(city, f"{city}: 未知城市，无法查询")
        return info

    tools = [get_weather]

    # ---- 一行创建 Agent ----
    agent = create_react_agent(llm, tools)  # ← 这就是全部

    # ---- 使用 ----
    print("\n  😊 用户: 北京今天天气怎么样？")
    result = agent.invoke(
        {"messages": [HumanMessage(content="北京今天天气怎么样？")]},
    )
    # 提取 AI 回复（跳过工具调用消息）
    for msg in result["messages"]:
        if msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
            print(f"  🤖 AI: {msg.content}")

    print("\n  😊 用户: 上海呢？")
    result = agent.invoke(
        {"messages": [HumanMessage(content="上海呢？")]},
        # ⚠️ 注意: 默认没有 memory, 所以"上海呢？"的上下文可能丢失
        # 如果需要记忆，可以加 checkpointer
    )
    for msg in result["messages"]:
        if msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
            print(f"  🤖 AI: {msg.content}")


# ============================================================
# 运行所有 Demo
# ============================================================
if __name__ == "__main__":
    DEMOS = [
        ("基础 StateGraph", demo_basic_state_graph),
        ("条件边路由", demo_conditional_edge),
        ("MessagesState 聊天", demo_message_graph),
        ("Agent + Tool (ReAct)", demo_agent_with_tools),
        ("多 Agent 协作", demo_multi_agent_pipeline),
        ("MemorySaver 持久化", demo_memory_saver),
        ("create_react_agent 快速上手", demo_create_react_agent),
    ]

    print("=" * 65)
    print("🚀  LangGraph Demo Collection")
    print("=" * 65)
    print(f"\n共 {len(DEMOS)} 个 Demo，可选择运行:")

    for i, (title, _) in enumerate(DEMOS, 1):
        print(f"  {i}. {title}")
    print(f"  0. 全部运行")
    print(f"  输入数字选择 (0-{len(DEMOS)}), 或直接回车运行全部: ", end="")

    try:
        choice = input().strip()
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(DEMOS):
                DEMOS = [DEMOS[choice - 1]]
            elif choice != 0:
                print(f"无效选择，运行全部。")
        else:
            pass  # 运行全部
    except (EOFError, KeyboardInterrupt):
        pass

    for title, fn in DEMOS:
        try:
            fn()
        except Exception as e:
            print(f"\n⚠️  [{title}] 出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 65)
    print("✅  所有 Demo 运行完毕！")
    print("=" * 65)
