from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama2", temperature=0.0)
result = llm.invoke("你是谁？")
print(result.content)