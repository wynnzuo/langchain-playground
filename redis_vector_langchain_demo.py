"""
向量化文本 → 存入 Redis → 向量搜索 + 混合检索  (LangChain 简化版)
====================================================================

与 redis_vector_demo.py 对比，LangChain 帮你省掉了:
  - Embedding API 的调用封装  → OpenAIEmbeddings
  - float32 <-> bytes 的序列化  → 内部自动处理
  - FT.CREATE 索引创建语句    → Redis vectorstore 自动建
  - pipeline / hset 批量写入  → add_texts / from_texts
  - Query 语法拼接            → similarity_search_with_score
  - 结果解析（score bytes 转 float）→ 返回干净的对象
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Redis as RedisVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_BASE = os.getenv("DASHSCOPE_API_BASE")
MODEL = "text-embedding-v4"
INDEX_NAME = "idx:docs"


def main():
    # ─── 1. Embedding 模型（传入 base_url 即可对接百炼 / 任意 OpenAI 兼容API）───
    embeddings = OpenAIEmbeddings(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        check_embedding_ctx_length=False,  # DashScope 只接受原始文本，不用 tiktoken 分词
    )
    print("✅ Embedding 模型就绪")

    # ─── 2. 清理旧索引 ───
    RedisVectorStore.drop_index(
        index_name=INDEX_NAME,
        redis_url="redis://localhost:6379",
        delete_documents=True,
    )

    # ─── 3. 写入文本（自动建索引、自动序列化向量）───
    docs = [
        "Redis Stack 是一个将内存数据库与向量搜索、JSON、时序等能力集成的产品",
        "向量搜索通过 RediSearch 模块实现，支持 FLAT 和 HNSW 两种算法",
        "HNSW 是一种分层可导航小世界图算法，适合大规模向量近似搜索",
        "FLAT 是暴力搜索，精度 100%，适合中小规模数据集",
        "Cosine 距离度量计算两个向量夹角的余弦值，值越大越相似",
        "百炼平台提供 text-embedding-v4 等 Embedding 模型，输出 1024 维向量",
        "Embedding 模型将文本映射到语义空间，相似文本的向量距离更近",
        "RAG (Retrieval-Augmented Generation) 是向量搜索在 LLM 中的经典应用",
        "Redis Stack 还支持 RediJSON、TimeSeries、BloomFilter 等模块",
        "RediSearch 支持索引 Hash 或 JSON 文档类型的任意字段",
    ]

    # 构造元数据（替代原始代码中的 id TAG 字段）
    metadatas = [{"id": str(i)} for i in range(len(docs))]

    vectorstore = RedisVectorStore.from_texts(
        texts=docs,
        embedding=embeddings,
        metadatas=metadatas,
        redis_url="redis://localhost:6379",
        index_name=INDEX_NAME,
    )
    print(f"✅ 写入 {len(docs)} 篇文档\n")

    # ─── 4. 纯向量 KNN 搜索（返回 (document, score) 元组）───
    query_text = "如何在 Redis 里做相似度搜索？"
    results = vectorstore.similarity_search_with_score(query_text, k=5)

    print(f"【纯向量 KNN】查询「{query_text}」→ 命中 {len(results)}\n")
    for i, (doc, score) in enumerate(results):
        # score 是 cosine 距离；1 - score 为相似度
        sim = 1 - score
        print(f"  #{i+1}  sim={sim:.4f}  |  {doc.page_content}")

    # ─── 5. 混合检索（TEXT 全文过滤 + 向量排序）───
    print(f"\n{'─' * 50}")
    print("【混合检索】全文含 \"Redis\" → 再向量排序")

    # 全文搜索用 %（LIKE 匹配），不是 ==（精确匹配）
    from langchain_community.vectorstores.redis import RedisFilter

    hybrid = vectorstore.similarity_search_with_score(
        query_text,
        k=5,
        filter=RedisFilter.text("content") % "Redis",
    )
    for i, (doc, score) in enumerate(hybrid):
        sim = 1 - score
        print(f"  #{i+1}  sim={sim:.4f}  |  {doc.page_content}")

    # ─── 6. 需要操作原生 redis-py 时也能拿到原始连接 ───
    first_key = next(vectorstore.client.scan_iter(match="doc:idx:docs:*"))
    raw = vectorstore.client.hget(first_key, "content_vector")
    print(f"\n{'─' * 50}")
    print(f"{first_key.decode()} content_vector 字节数: {len(raw)} → {len(raw) // 4} 维")


if __name__ == "__main__":
    main()
