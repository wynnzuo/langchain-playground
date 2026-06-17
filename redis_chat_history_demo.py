"""
Redis 聊天记忆示例
====================
用 Redis 持久化对话历史，重启程序后记忆不丢失。
通过 RunnableWithMessageHistory + RedisChatMessageHistory 实现。

前置条件:
  - Redis Stack 运行中（默认 localhost:6379）
  - 安装 redis 模块: pip install redis
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

load_dotenv()

llm = init_chat_model(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# ==========================================
# 配置 Prompt
# ==========================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手，请用中文回答。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

# ==========================================
# 用 Redis 存储聊天历史
# ==========================================
SESSION_ID = "redis_demo_session"

chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=lambda session_id: RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379/0",
    ),
    input_messages_key="input",
    history_messages_key="history",
)

print("=" * 60)
print("Redis 聊天记忆 Demo")
print("按 Ctrl+C 退出，重新运行后记忆仍在")
print("=" * 60)

while True:
    try:
        user_input = input("\n你: ")
        if user_input.lower() in ("exit", "quit", "q"):
            break

        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": SESSION_ID}},
        )
        print(f"助手: {response.content}")

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"错误: {e}")

print("\n对话结束，历史已保存到 Redis")
print(f"查看: docker exec redis-stack redis-cli KEYS '*{SESSION_ID}*'")
