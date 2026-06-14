import os

from langchain.chat_models import BaseChatModel, init_chat_model
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
load_dotenv()  # 从 .env 文件加载环境变量


def init_chat_model_example() -> BaseChatModel:
    """示例：如何使用 init_chat_model 创建一个聊天模型实例"""
    # 创建一个 ChatOpenAI 实例，指定模型和温度
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
    llm = init_chat_model(
        "deepseek-v4-flash",
        model_provider="openai",
        temperature=0.0,
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    return llm


def main():
    try:
        result = init_chat_model_example().stream("你是谁？")
        for chunk in result:
            print(chunk.content, end="", flush=True)
    except Exception as e:
        logger.error(f"Error invoking the chat model: {e}")


if __name__ == "__main__":
    main()
