"""
MCP + LangChain 查书 Demo
============================
通过 MCP 搜索服务查书，用 langchain-mcp-adapters 将
MCP 工具转为 LangChain BaseTool，与 LCEL 链无缝集成。

流程:
  Python → MCP Client → one-search-mcp → 互联网 → LLM 提取 → JSON

前置:
  npm install -g one-search-mcp

配置搜索引擎（写 .env）:
  SEARCH_PROVIDER=tavily          # 推荐
  SEARCH_API_KEY=tvly-xxxxx       # 或用已有的 TAVILY_API_KEY

  # SEARCH_PROVIDER=local         # 本地浏览器搜索（无需 Key）
"""

import os
import json
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


# ==========================================
# 1. 输出数据结构
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
# 2. MCP 连接
# ==========================================
@asynccontextmanager
async def connect_mcp(command: str, args: list[str], env: dict | None = None):
    """连接 MCP Server，返回已初始化的 ClientSession。"""
    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


# ==========================================
# 3. LLM 结构化提取链
# ==========================================
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

parser = PydanticOutputParser(pydantic_object=BookInfo)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个图书信息提取助手。从搜索结果中提取图书出版信息。\n"
     "注意:\n"
     "  - 作者有多个就用列表\n"
     "  - 未知字段填 '未知'\n"
     "  - 简介 100 字以内\n"
     "  - 严格按要求的 JSON 格式输出\n\n"
     "{format_instructions}"),
    ("human", "提取下面的「{book_name}」出版信息：\n\n{search_results}"),
])

chain = prompt | llm | parser


# ==========================================
# 4. 查书
# ==========================================
async def search_book(book_name: str, session: ClientSession) -> BookInfo | None:
    """用 MCP 工具搜索 → LLM 提取结构化信息。"""
    print(f"\n📖 正在查询: 「{book_name}」")

    # load_mcp_tools 将 MCP 工具转为 LangChain BaseTool
    tools = await load_mcp_tools(session)
    search_tool = next((t for t in tools if "search" in t.name), None)
    if not search_tool:
        print(f"❌ 未找到搜索工具。可用: {[t.name for t in tools]}")
        return None

    print(f"🔧 使用工具: {search_tool.name}")

    # 两次搜索，提高命中率
    queries = [
        f"{book_name} ISBN 出版日期 出版社 定价 页数",
        f"{book_name} 书籍 出版信息 作者",
    ]
    raw = ""
    for i, q in enumerate(queries):
        print(f"🔍 搜索 {i+1}: 「{q}」")
        result = await search_tool.ainvoke({"query": q, "limit": 5})
        # load_mcp_tools 返回 MCP 原始 list[dict]，提取 text 字段
        if isinstance(result, list):
            text = "\n".join(
                item["text"] for item in result if isinstance(item, dict) and item.get("type") == "text"
            )
        else:
            text = str(result)
        raw += f"\n=== {i+1} ===\n{text}"
        if len(text) > 500:
            break

    t = str(raw)
    print(f"✅ 完成（{len(t)} 字符）")
    if len(t) > 200:
        print(f"📄 预览: {t[:200]}...\n")

    # LLM 提取
    print("🧠 LLM 提取中...")
    try:
        book = await chain.ainvoke({
            "book_name": book_name,
            "search_results": t[:5000],
            "format_instructions": parser.get_format_instructions(),
        })
        print("✅ 完成\n")
        return book
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        return None


# ==========================================
# 5. 输出
# ==========================================
def print_book(book: BookInfo):
    L = "─" * 50
    print(L)
    print(f"  📚 {book.title}")
    print(L)
    print(f"  ✍️  作者　　: {', '.join(book.author)}")
    print(f"  🏢 出版社　: {book.publisher}")
    print(f"  🔖 ISBN　　: {book.isbn}")
    print(f"  📅 出版日期 : {book.publish_date}")
    if book.pages:
        print(f"  📄 页数　　: {book.pages} 页")
    if book.price:
        print(f"  💰 定价　　: {book.price}")
    print(f"  📝 简介　　: {book.summary}")
    print(L)
    print("💻 JSON:", json.dumps(book.model_dump(), ensure_ascii=False, indent=2))


# ==========================================
# 6. 主入口
# ==========================================
async def main():
    # 配置
    provider = os.getenv("SEARCH_PROVIDER", "tavily")
    api_key = os.getenv("SEARCH_API_KEY") or os.getenv("TAVILY_API_KEY") or ""
    mcp_env = {"SEARCH_PROVIDER": provider, "SEARCH_API_KEY": api_key}

    print("=" * 60)
    print(f"  MCP & langchain-mcp-adapters 查书工具")
    print(f"  搜索后端: {provider}")
    print("=" * 60)

    if provider in {"tavily", "bing", "google"} and not api_key:
        print(f"\n⚠️  未配置 {provider} 的 API Key")
        return
    if provider == "local":
        print("\nℹ️  本地搜索模式，需安装 Chrome/Chromium")

    # 选书
    books = {"1": "三体", "2": "百年孤独", "3": "活着", "4": "深入理解计算机系统"}
    print("\n📚 预设:")
    for k, v in books.items():
        print(f"  [{k}] {v}")
    print("  [0] 自定义")
    try:
        c = input("\n请选择 [0-4]: ").strip()
        name = input("请输入书名: ").strip() if c == "0" else books.get(c, "三体")
    except (EOFError, KeyboardInterrupt):
        name = "三体"

    # 执行
    try:
        async with connect_mcp("npx", ["one-search-mcp"], env=mcp_env) as session:
            book = await search_book(name, session)
            print_book(book) if book else print("\n⚠️  查询失败")
    except FileNotFoundError:
        print("❌ 未找到 npx。请安装 Node.js")
    except Exception as e:
        print(f"❌ 失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
