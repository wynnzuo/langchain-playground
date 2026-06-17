"""
init_chat_model 流式输出
=========================
使用 init_chat_model 的 stream() 方法实现流式输出，
逐 token 显示 LLM 回复，体验打字机效果。
"""

import os
import logging

from langchain.chat_models import BaseChatModel, init_chat_model
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

# 加载环境变量
load_dotenv()


def create_llm() -> BaseChatModel:
    """创建并返回聊天模型实例"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
    return init_chat_model(
        "deepseek-v4-flash",
        model_provider="openai",
        temperature=0.0,
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_BASE"),
    )


def main():
    try:
        # stream() 逐 chunk 返回，每次输出一个 token
        result = create_llm().stream("你是谁？")
        for chunk in result:
            print(chunk.content, end="", flush=True)
        print()  # 换行
    except Exception as e:
        logger.error(f"调用流式接口失败: {e}")


if __name__ == "__main__":
    main()
