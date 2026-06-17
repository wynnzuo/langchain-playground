"""
天气查询 Agent Demo
=====================
使用 LangGraph 的 create_react_agent 创建一个
能查询天气的 AI 助手，演示 Agent + Tool 调用模式。

工具:
  - get_weather: 查询指定城市的天气（模拟数据，可替换为真实 API）

流程:
  用户提问 → LLM 判断是否需用工具 → 调用工具获取数据 → 组织回答
"""

import os
import json
import random
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()


# ==========================================
# 1. 定义天气查询工具
# ==========================================
@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息。

    Args:
        city: 城市名称，如 "北京"、"上海"、"Tokyo"、"New York"
    """
    # 模拟天气数据（实际使用时可替换为真实 API 调用）
    weather_data = {
        "北京": {"temp": 28, "humidity": 45, "condition": "晴", "wind": "3级"},
        "上海": {"temp": 32, "humidity": 70, "condition": "多云", "wind": "4级"},
        "广州": {"temp": 35, "humidity": 80, "condition": "雷阵雨", "wind": "3级"},
        "深圳": {"temp": 33, "humidity": 78, "condition": "多云转阴", "wind": "3级"},
        "成都": {"temp": 26, "humidity": 60, "condition": "阴", "wind": "2级"},
        "杭州": {"temp": 30, "humidity": 65, "condition": "晴", "wind": "3级"},
        "武汉": {"temp": 34, "humidity": 72, "condition": "多云", "wind": "2级"},
        "西安": {"temp": 31, "humidity": 50, "condition": "晴", "wind": "3级"},
    }

    # 海外城市用随机数据
    if city not in weather_data:
        weather_data[city] = {
            "temp": random.randint(5, 38),
            "humidity": random.randint(20, 90),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Windy", "Clear"]),
            "wind": f"{random.randint(1, 6)}级",
        }

    info = weather_data[city]
    return json.dumps({
        "city": city,
        "temperature": f"{info['temp']}°C",
        "humidity": f"{info['humidity']}%",
        "condition": info["condition"],
        "wind": info["wind"],
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }, ensure_ascii=False)


# ==========================================
# 2. 初始化 LLM
# ==========================================
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# ==========================================
# 3. 创建 Agent
# ==========================================
tools = [get_weather]
agent = create_agent(llm, tools)


# ==========================================
# 4. 交互循环
# ==========================================
def main():
    print("=" * 60)
    print("🌤️  天气查询 Agent（输入 q 退出）")
    print("  示例: 北京今天天气怎么样？")
    print("        上海和深圳哪个更热？")
    print("        Tokyo 明天会下雨吗？")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
            if user_input.lower() in ("q", "quit", "exit"):
                break
            if not user_input:
                continue

            print("\n🤖 Agent 思考中...")

            # 调用 agent
            response = agent.invoke({"messages": [("human", user_input)]})

            # 提取最终回答
            for msg in response["messages"]:
                if msg.type == "ai" and msg.content:
                    # 打印工具调用信息
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"\n🔧 调用工具: {tc['name']}({tc['args']})")
                    # 打印最终回复
                    if not getattr(msg, "tool_calls", None):
                        print(f"\n🤖 {msg.content}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")

    print("\n👋 再见！")


if __name__ == "__main__":
    main()
