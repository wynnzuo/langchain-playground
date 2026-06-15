from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os

load_dotenv()

json_output_parser = JsonOutputParser()
format_instructions = json_output_parser.get_format_instructions()

prompt_template = PromptTemplate.from_template(
    "You are a helpful assistant that translates {input_language} to {output_language}.\n\n"
    "{sentence}\n\n"
    "Format instructions: {format_instructions}"
)

llm = init_chat_model(
    model="deepseek-v4-flash",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# LCEL 链：prompt → llm → parser
chain = prompt_template | llm | json_output_parser

# 必须 invoke 才能执行
response = chain.invoke(
    {
        "input_language": "English",
        "output_language": "French",
        "format_instructions": format_instructions,
        "sentence": "Hello, how are you?"
    }
)
print(response)