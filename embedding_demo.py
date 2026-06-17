"""
通义百炼 Embedding 三种调用方式 + 余弦相似度 Demo
===================================================
演示三种 Embedding API 调用方式：
  1. langchain_community.DashScopeEmbeddings
  2. OpenAI 兼容接口（openai SDK）
  3. DashScope SDK 原生调用（含多模态 Embedding）
"""

import json
import os

import dashscope
import numpy as np
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from openai import OpenAI

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_BASE = os.getenv("DASHSCOPE_API_BASE")
MODEL = "text-embedding-v4"

# ---------- 测试文本 ----------
QUERY_TEXT = "This is a test document."
DOC_TEXTS = [
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!",
]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度（值越大越相似）"""
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def print_similarity_matrix(vectors: list[list[float]], labels: list[str]):
    """打印文本间的余弦相似度矩阵，便于直观比较语义距离"""
    print(f"\n{'':<22}", end="")
    for t in labels:
        print(f"{t[:14]:<14}", end="")
    print()
    for i in range(len(labels)):
        print(f"{labels[i]:<22}", end="")
        for j in range(len(labels)):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"{sim:<14.4f}", end="")
        print()


# ==================== 1. langchain_community DashScopeEmbeddings ====================
print("=" * 60)
print("[1] langchain_community.DashScopeEmbeddings")
print("=" * 60)

emb = DashScopeEmbeddings(model=MODEL, dashscope_api_key=API_KEY)

# 单个查询文本的 Embedding
qv = emb.embed_query(QUERY_TEXT)
print(f"查询向量维度: {len(qv)}")

# 批量文档 Embedding
dvs = emb.embed_documents(DOC_TEXTS)
print(f"文档向量数量: {len(dvs)}, 维度: {len(dvs[0])}")

# 打印文档间的相似度矩阵
print_similarity_matrix(dvs, DOC_TEXTS)

# 查询文本与各文档的相似度
print("\n[1] 查询「Hello, how are you?」与各文档的相似度:")
qv2 = emb.embed_query("Hello, how are you?")
for t, v in zip(DOC_TEXTS, dvs):
    print(f"  vs 「{t}」 → {cosine_similarity(qv2, v):.4f}")

# ==================== 2. OpenAI 兼容接口 ====================
print("\n" + "=" * 60)
print("[2] OpenAI 兼容接口")
print("=" * 60)

client = OpenAI(api_key=API_KEY, base_url=API_BASE)
resp = client.embeddings.create(model=MODEL, input=QUERY_TEXT)
vec = resp.data[0].embedding
print(f"向量维度: {len(vec)}, 前5维: {vec[:5]}")

# ==================== 3. DashScope SDK 多模态 Embedding ====================
print("\n" + "=" * 60)
print("[3] DashScope SDK — 多模态 Embedding")
print("=" * 60)

# 支持文本、图片、视频输入:
#   input=[{"image": "https://...png"}]
#   input=[{"video": "https://...mp4"}]
resp = dashscope.MultiModalEmbedding.call(
    api_key=API_KEY,
    model="tongyi-embedding-vision-plus",
    input=[{"text": QUERY_TEXT}],
)
print(json.dumps(resp.output, indent=4, ensure_ascii=False))
