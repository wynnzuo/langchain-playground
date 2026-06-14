from langchain_core.prompts import ChatPromptTemplate
import logging
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
import os

load_dotenv()  # 从 .env 文件加载环境变量

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有帮助的助手。"),
        ("human", "{question},{format_instructions}"),
    ]
)
json_parser = JsonOutputParser()
formatted_instructions = json_parser.get_format_instructions()

messages = chat_prompt_template.format_messages(
    question="什么是人工智能？", format_instructions=formatted_instructions
)
logger.info(f"Formatted messages: {messages}")

llm = init_chat_model(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

result = llm.invoke(messages)
logger.info(f"Raw model response: {result}")
logger.info(f"Model response: {result.content}")


try:
    parsed_result = json_parser.invoke(result)
    logger.info(f"Parsed result: {parsed_result}")
except Exception as e:
    logger.error(f"Error parsing result: {e}")
