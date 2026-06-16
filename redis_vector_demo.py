"""
向量化文本 → 存入 Redis Stack (RediSearch) → 向量搜索 + 混合检索  Demo

前置条件:
  - Redis Stack 运行中 (docker run ... redis/redis-stack-server)
  - 环境变量 DASHSCOPE_API_KEY / DASHSCOPE_API_BASE (百炼模型服务)
"""
import os
import struct
from typing import cast

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_BASE = os.getenv("DASHSCOPE_API_BASE")
MODEL = "text-embedding-v4"
DIM = 1024  # text-embedding-v4 的输出维度


# ──────────────────────── 工具函数 ────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """调用百炼 Embedding API → N×DIM 向量列表"""
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    resp = client.embeddings.create(model=MODEL, input=texts)
    sorted_data = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in sorted_data]


def vec_to_bytes(vec: list[float]) -> bytes:
    """float32 列表 → 小端序 bytes (RediSearch 要求的格式)"""
    return struct.pack(f"{len(vec)}f", *vec)


def bytes_to_vec(data: bytes) -> list[float]:
    """Redis bytes → float32 列表"""
    return list(struct.unpack(f"{len(data) // 4}f", data))


# ──────────────────────── 主流程 ────────────────────────

def main():
    import redis
    from redis.commands.search.query import Query

    r = redis.Redis(host="localhost", port=6379, decode_responses=False)
    INDEX = "idx:docs"
    PREFIX = "doc:"

    # ─── 1. 清理 ───
    try:
        r.ft(INDEX).dropindex()
    except redis.ResponseError:
        pass
    for key in r.scan_iter(match=f"{PREFIX}*"):
        r.delete(key)

    # ─── 2. 创建 RediSearch 索引 (含 VECTOR 字段) ───
    r.execute_command(
        "FT.CREATE", INDEX,
        "ON", "HASH",
        "PREFIX", "1", PREFIX,
        "SCHEMA",
        "id",   "TAG",       # 标识符
        "text", "TEXT",      # 原始文本（英文全文检索；中文不适用）
        "embedding", "VECTOR", "FLAT", "6",
            "TYPE", "FLOAT32",
            "DIM", str(DIM),
            "DISTANCE_METRIC", "COSINE",
    )
    print(f"✅ 索引 [{INDEX}] 就绪 (FLAT + COSINE, DIM={DIM})")

    # ─── 3. 嵌入 + 批量写入 ───
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

    vectors = embed_texts(docs)
    with r.pipeline() as pipe:
        for i, (text, vec) in enumerate(zip(docs, vectors)):
            key = f"{PREFIX}{i}"
            pipe.hset(key, mapping={
                "id": str(i),
                "text": text,
                "embedding": vec_to_bytes(vec),
            })
        pipe.execute()
    print(f"✅ 写入 {len(docs)} 篇文档\n")

    # ─── 4. 纯向量 KNN 搜索 ───
    query_text = "如何在 Redis 里做相似度搜索？"
    query_vec = embed_texts([query_text])[0]

    q_knn = (
        Query("*=>[KNN 5 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("id", "text", "score")
        .dialect(2)
        .paging(0, 5)
    )
    result = r.ft(INDEX).search(q_knn, query_params={"vec": vec_to_bytes(query_vec)})

    print(f"【纯向量 KNN】查询「{query_text}」→ 命中 {result.total}\n")
    for i, doc in enumerate(result.docs):
        score = float(cast(bytes, doc.score))
        sim = 1 - score
        print(f"  #{i+1}  sim={sim:.4f}  |  {doc.text}")

    # ─── 5. 混合检索 (TEXT 全文过滤 + 向量重排序) ───
    # 注: RediSearch 中文分词器有限制，中文全文检索用 TAG 更可靠
    print(f"\n{'─' * 50}")
    print("【混合检索】全文含 \"Redis\" → 再向量排序")
    q_hybrid = (
        Query("@text:Redis =>[KNN 5 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("id", "text", "score")
        .dialect(2)
        .paging(0, 5)
    )
    hybrid = r.ft(INDEX).search(q_hybrid, query_params={"vec": vec_to_bytes(query_vec)})
    for i, doc in enumerate(hybrid.docs):
        score = float(cast(bytes, doc.score))
        sim = 1 - score
        print(f"  #{i+1}  sim={sim:.4f}  |  {doc.text}")

    # ─── 6. 存进去的向量什么样 ───
    print("\n" + "─" * 50)
    raw = r.hget("doc:0", "embedding")
    vec0 = bytes_to_vec(cast(bytes, raw))
    print(f"doc:0 向量前 5 维: {vec0[:5]}")

    # 可选: 在浏览器打开 http://localhost:8001 进入 RedisInsight 查看


if __name__ == "__main__":
    main()
