from datetime import datetime
import os

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

load_dotenv()  # 从 .env 文件加载环境变量

# 创建一个 ChatOpenAI 实例，指定模型和温度
llm = init_chat_model(
    "deepseek-v4-flash",
    model_provider="openai",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)
prompt = PromptTemplate(
    template="你是一个有帮助的助手。请回答以下问题：{question}",
    input_variables=["question"],
)

prompt_text = prompt.format(question="你是谁？")
print(prompt_text)
print("\n---\n")
result = llm.invoke(prompt_text)


print(result.content)
print("\n---\n")

prompt1 = PromptTemplate.from_template(
    "你是一个有帮助的助手。现在的时间是:{time}，请回答以下问题：{question}",
    partial_variables={"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
)
prompt_text1 = prompt1.format(question="你是谁？")
print(prompt_text1)
print("\n---\n")

prompt2 = PromptTemplate.from_template(
    "你是一个有帮助的助手。现在的时间是:{time}，请回答以下问题：{question}",
    partial_variables={"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
)
prompt_text2 = prompt2.invoke({"question": "你是谁？"})
print(type(prompt_text2))
print(prompt_text2.to_string())
print("\n---\n")
