import os

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()  # 从 .env 文件加载环境变量

# 创建一个 ChatOpenAI 实例，指定模型和温度
llm = init_chat_model(
    "deepseek-v4-flash",
    model_provider="openai",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)
messages = [
    SystemMessage(content="你是一个有帮助的助手。"),
    HumanMessage(content="你是谁？"),
]
result = llm.invoke(messages)
print(f"type of result: {type(result)}")
print(result.content)
