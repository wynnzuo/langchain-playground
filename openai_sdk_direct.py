import os
from openai import OpenAI
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
)
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "你是谁？"}],
    temperature=0.0,
)
print(response.choices[0].message.content)
