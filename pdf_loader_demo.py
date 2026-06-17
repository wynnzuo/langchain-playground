"""
PDF 文档加载与预处理 Demo
===========================
演示 LangChain 文档加载器的工作流:
  PDF → PyPDFLoader 逐页加载 → RecursiveCharacterTextSplitter 分块
为后续 RAG / 向量化准备数据。
"""

import os
from pprint import pprint

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = "/Users/wynn/Downloads/有钱人和你想的不一样.pdf"
MAX_PREVIEW_PAGES = 3  # 预览前几页


def main():
    # ─── 1. 加载 PDF（逐页读取）───
    print("=" * 60)
    print(f"📄 加载 PDF：{PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()  # list[Document]，每页一个 Document
    print(f"\n✅ 共加载 {len(docs)} 页")

    # 基本信息
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"📊 总字符数: {total_chars:,}")
    print(f"📏 平均每页: {total_chars // len(docs):,} 字符")
    print(f"📋 元数据字段: {list(docs[0].metadata.keys())}")

    # ─── 2. 预览前几页内容 ───
    print(f"\n{'─' * 60}")
    print(f"🔍 前 {MAX_PREVIEW_PAGES} 页预览\n")
    for i, doc in enumerate(docs[:MAX_PREVIEW_PAGES]):
        page_num = doc.metadata.get("page", doc.metadata.get("page_number", i)) + 1
        text = doc.page_content.strip()
        print(f"─── 第 {page_num} 页 (共 {len(text)} 字符) ───")
        print(text[:300])
        if len(text) > 300:
            print(f"... (省略 {len(text) - 300} 字符)")
        print()

    # ─── 3. 文本分块（为后续 RAG / 向量化做准备）───
    print(f"{'─' * 60}")
    print("✂️  文本分块 (RecursiveCharacterTextSplitter)\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 每块约 500 字符
        chunk_overlap=50,    # 块间重叠 50 字符（保持上下文连贯）
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    print(f"  分块前: {len(docs)} 个 Document（每页一个）")
    print(f"  分块后: {len(chunks)} 个 Chunk")
    print(f"  平均每块: {sum(len(c.page_content) for c in chunks) // len(chunks)} 字符\n")

    # 展示前 3 个块
    for i, chunk in enumerate(chunks[:3]):
        print(f"  ─── Chunk #{i+1} (来自第 {chunk.metadata.get('page', 0)+1} 页) ───")
        print(f"  {chunk.page_content[:200]}...")
        print()

    # ─── 4. 页面元数据一览 ───
    print(f"{'─' * 60}")
    print("📌 元数据示例（第 1 页）")
    pprint(docs[0].metadata)


if __name__ == "__main__":
    main()
