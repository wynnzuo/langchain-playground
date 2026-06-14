from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

chat_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "你是一个有帮助的助手。现在的时间是:{time}，请回答以下问题：{question}",
        ),
        ("human", "{question}"),
        ("ai", "{answer}"),
    ]
)
messages = chat_prompt.format_messages(
    time="2023-10-01 12:00:00",
    question="什么是人工智能？",
    answer="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
)
print(messages)
print("\n---\n")

chat_prompt2 = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": "你是一个有帮助的助手。现在的时间是:{time}，请回答以下问题：{question}",
        },
        {"role": "human", "content": "{question}"},
        {"role": "ai", "content": "{answer}"},
    ]
)
messages = chat_prompt2.format_messages(
    time="2023-10-01 12:00:00",
    question="什么是人工智能？",
    answer="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
)
print(messages)
print("\n---\n")

chat_prompt3 = ChatPromptTemplate(
    [
        SystemMessage(content="{answer}"),
        HumanMessage(content="{question}"),
    ]
)
messages = chat_prompt3.format_messages(
    **{
        "question": "什么是人工智能？",
        "answer": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
    }
)
print(messages)
