from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
load_dotenv()

llm = ChatOpenAI(model="deepseek-v4-flash", temperature=0.0)


if __name__ == "__main__":
    print("Asking: What is the capital of France?")
    question = "What is the capital of France?"
    response = llm.invoke(question)
    print(response.content)
