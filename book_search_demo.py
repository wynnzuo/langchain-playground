"""
LLM + Web Search 查书 Demo
============================
通过书名搜索出版信息，使用:
  1. Tavily Search — 联网搜索（中文内容友好，每月 1000 次免费）
  2. LLM + PydanticOutputParser — 结构化提取

流程: 用户输入书名 → 联网搜索 → LLM 提取 → 结构化 JSON

准备工作:
  注册 https://tavily.com 获取 API Key，并写入 .env 文件:
  TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
"""

import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()


# ==========================================
# 1. 定义输出数据结构
# ==========================================
class BookInfo(BaseModel):
    """图书出版信息"""
    title: str = Field(description="书名")
    author: list[str] = Field(description="作者列表")
    publisher: str = Field(description="出版社")
    isbn: str = Field(description="ISBN 号（标准书号）")
    publish_date: str = Field(description="出版日期，如 '2024年1月'")
    pages: int | None = Field(default=None, description="总页数")
    price: str | None = Field(default=None, description="定价")
    summary: str = Field(description="内容简介（100字以内）")


# ==========================================
# 2. 初始化搜索工具
# ==========================================
search = TavilySearchResults(
    max_results=5,
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
)

# ==========================================
# 3. 初始化 LLM
# ==========================================
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.1,          # 低温度 → 更精确
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)


# ==========================================
# 4. 构建提取链 (LLM + PydanticOutputParser)
# ==========================================
parser = PydanticOutputParser(pydantic_object=BookInfo)

extract_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个图书信息提取助手。从搜索结果中提取图书的出版信息。\n"
     "注意:\n"
     "  - 作者可能有多个，用列表列出\n"
     "  - 如果某个字段找不到，用 '未知' 填充\n"
     "  - 内容简介控制在 100 字以内\n"
     "  - 严格按要求的 JSON 格式输出\n\n"
     "{format_instructions}"),
    ("human", "请提取下面搜索结果中「{book_name}」的出版信息：\n\n{search_results}"),
])

# LCEL 链：模板 → LLM → 解析器
chain = extract_prompt | llm | parser


# ==========================================
# 5. 核心函数：查书
# ==========================================
def search_book(book_name: str) -> BookInfo | None:
    """
    通过书名查出版信息。

    1. 先用 Tavily 搜索书名 + 出版关键词
    2. 再将搜索结果交给 LLM 提取结构化数据
    """
    print(f"\n📖 正在查询: 「{book_name}」")

    # --- 第 1 步：联网搜索 ---
    print("🔍 第 1 步：联网搜索中...")
    query = f"{book_name} 出版信息 作者 ISBN 出版社 定价"
    raw_results = search.invoke({"query": query})

    # Tavily 返回的是 list[dict]，拼接成纯文本
    texts = []
    for r in raw_results:
        title = r.get("title", "")
        content = r.get("content", "")
        texts.append(f"[{title}]\n{content}")
    combined = "\n\n".join(texts)

    print(f"✅ 搜索完成（{len(combined)} 字符，{len(raw_results)} 条结果）")
    preview = combined[:300] + ("..." if len(combined) > 300 else "")
    print(f"📄 搜索结果预览:\n{preview}\n")

    # --- 第 2 步：LLM 结构化提取 ---
    print("🧠 第 2 步：LLM 结构化提取中...")
    try:
        book: BookInfo = chain.invoke({
            "book_name": book_name,
            "search_results": combined[:5000],
            "format_instructions": parser.get_format_instructions(),
        })
        print(f"✅ 提取完成\n")
        return book
    except Exception as e:
        print(f"❌ 结构化提取失败: {e}")
        return None


# ==========================================
# 5. 格式化输出
# ==========================================
def print_book_info(book: BookInfo):
    """友好的控制台输出"""
    line = "─" * 50
    print(line)
    print(f"  📚 {book.title}")
    print(line)
    print(f"  ✍️  作者　　: {', '.join(book.author)}")
    print(f"  🏢 出版社　: {book.publisher}")
    print(f"  🔖 ISBN　　: {book.isbn}")
    print(f"  📅 出版日期 : {book.publish_date}")
    if book.pages:
        print(f"  📄 页数　　: {book.pages} 页")
    if book.price:
        print(f"  💰 定价　　: {book.price}")
    print(f"  📝 简介　　: {book.summary}")
    print(line)

    # 同时输出 JSON
    print("\n💻 JSON 格式:")
    print(json.dumps(book.model_dump(), ensure_ascii=False, indent=2))


# ==========================================
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("  LLM + Web Search 查书工具")
    print("  输入书名 → 联网搜索 → 结构化输出")
    print("=" * 60)

    # 预置例子 or 交互式输入
    examples = ["三体", "百年孤独", "活着", "平凡的世界", "深入理解计算机系统"]

    print("\n预设书目：")
    for i, name in enumerate(examples, 1):
        print(f"  [{i}] {name}")
    print("  [0] 自定义书名")

    try:
        choice = int(input("\n请选择 [0-5]: ").strip())
        if choice == 0:
            book_name = input("请输入书名: ").strip()
        elif 1 <= choice <= len(examples):
            book_name = examples[choice - 1]
        else:
            book_name = examples[0]
    except (ValueError, EOFError):
        book_name = examples[0]

    book = search_book(book_name)
    if book:
        print_book_info(book)
    else:
        print("\n⚠️  查询失败，请重试")

    print("\n✨ 提示: 如果信息不准确，可能是因为搜索结果不完整。")
    print("   也可以尝试搜「书名+ISBN」获取更精确的结果。")
    print("")

    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  注意: 你还没有配置 TAVILY_API_KEY")
        print("   注册 https://tavily.com 获取 Key，然后添加到 .env 文件：")
        print("   TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx")
