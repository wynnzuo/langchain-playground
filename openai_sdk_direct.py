"""
OpenAI SDK 直连
==================
不使用 LangChain，直接通过 OpenAI SDK 调用兼容接口。
适合简单场景或需要底层控制的场景。
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

# 创建 OpenAI 客户端（对接任意 OpenAI 兼容 API）
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# 直接调用 Chat Completion API
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "你是谁？"}],
    temperature=0.0,
)

print(response.choices[0].message.content)
