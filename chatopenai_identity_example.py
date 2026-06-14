from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
load_dotenv()

llm = ChatOpenAI(model="deepseek-v4-flash", temperature=0.0)
result = llm.invoke("你是谁？")
print(result.content)
