# 🎮 LangChain Playground

LangChain 框架的示例集合，涵盖模型调用、Prompt 模板、输出解析、Agent、记忆、向量搜索、文档加载等场景。

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.3-important)](https://github.com/langchain-ai/langchain)

---

## 🚀 快速开始

```bash
# 1. 克隆
git clone https://github.com/wynnzuo/langchain-playground.git
cd langchain-playground

# 2. 安装依赖（建议使用 Conda 环境 langchain-env）
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 4. 运行示例
python init_chat_model_demo.py
```

---

## 📂 项目结构

### 🛜 模型调用

| 文件 | 说明 |
|------|------|
| `init_chat_model_demo.py` | `init_chat_model` 快速调用大模型 |
| `init_chat_model_messages.py` | 通过 Message 对象传递对话历史 |
| `init_chat_model_stream.py` | 流式输出（逐 token 打印） |
| `openai_chat_demo.py` | `ChatOpenAI` 直接调用 |
| `openai_sdk_direct.py` | 用 OpenAI SDK 直连（绕过 LangChain） |
| `ollama_chat_demo.py` | 对接本地 Ollama 模型 |

### 📝 Prompt 模板

| 文件 | 说明 |
|------|------|
| `prompt_template_demo.py` | `PromptTemplate` 的 `format` / `partial` / `invoke` 三种用法 |
| `chat_prompt_template_demo.py` | `ChatPromptTemplate` 五种构建方式（tuple / dict / placeholder 等） |

### 🔧 输出解析

| 文件 | 说明 |
|------|------|
| `json_output_parser_demo.py` | `JsonOutputParser` 让模型返回 JSON |
| `pydantic_output_parser_demo.py` | `PydanticOutputParser` 解析为 Pydantic 模型 |
| `typeddict_vs_basemodel_demo.py` | `TypedDict` 与 `BaseModel` 对比 |

### ⛓️ LCEL 链式编排

| 文件 | 说明 |
|------|------|
| `branching_demo.py` | `RunnableBranch` 条件分支 + `RunnableParallel` 并行 |
| `function_chain_demo.py` | `RunnableLambda` / `RunnablePassthrough` 函数链 |
| `translation_demo.py` | 最简单的 `prompt \| llm \| parser` 翻译链 |

### 🧠 记忆 & 对话历史

| 文件 | 说明 |
|------|------|
| `chat_history_demo.py` | `InMemoryChatMessageHistory` 多轮记忆 + 滑动窗口 |
| `redis_chat_history_demo.py` | `RedisChatMessageHistory` 持久化记忆 |

### 🔎 搜索 & Agent

| 文件 | 说明 |
|------|------|
| `langgraph_demo.py` | **LangGraph 合集** — StateGraph / 条件边 / ReAct / 多 Agent 协作 / MemorySaver |
| `weather_agent_demo.py` | LangChain Agent 天气查询 |
| `book_search_demo.py` | `Tavily` 联网搜索 + LLM 提取图书信息 |
| `book_search_mcp_demo.py` | `MCP` + `langchain-mcp-adapters` 查书 |
| `agent_france_demo.py` | 基础 Agent 调用示例 |

### 📄 文档加载 & 分块

| 文件 | 说明 |
|------|------|
| `pdf_loader_demo.py` | `PyPDFLoader` 加载 PDF → `RecursiveCharacterTextSplitter` 分块 |

### 🧬 Embedding & 向量搜索

| 文件 | 说明 |
|------|------|
| `embedding_demo.py` | 三种 Embedding 方式（DashScope / OpenAI SDK / DashScope SDK） |
| `redis_vector_demo.py` | 手写版：Embed → FT.CREATE → KNN 搜索 → 混合检索 |
| `redis_vector_langchain_demo.py` | LangChain 简化版（自动建索引 + 序列化） |

### 🛠️ 工具

| 文件 | 说明 |
|------|------|
| `version_info.py` | 查看当前 LangChain 版本信息 |
| `redis_check.py` | 检查 Redis 模块是否安装 |

---

## ⚙️ 环境变量

项目使用三个 API 服务，在 `.env` 中配置。（参考 `.env.example`）

| 变量 | 用途 | 获取 |
|------|------|------|
| `OPENAI_API_KEY` | LLM 调用（DeepSeek / OpenAI） | [DeepSeek](https://platform.deepseek.com) |
| `OPENAI_API_BASE` | API 端点地址 | `https://api.deepseek.com` |
| `DASHSCOPE_API_KEY` | Embedding / 向量化 | [百炼](https://bailian.console.aliyun.com) |
| `DASHSCOPE_API_BASE` | 百炼兼容端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `TAVILY_API_KEY` | 联网搜索（查书 Demo） | [Tavily](https://tavily.com) |

> 项目默认使用 DeepSeek 兼容 OpenAI 接口的 API，也支持其他 OpenAI 兼容服务，只需修改 `base_url` 和 `api_key`。

---

## 📦 依赖

| 组件 | 版本 | 说明 |
|------|------|------|
| `langchain` | 1.3.x | 核心框架 |
| `langchain-openai` | 1.3.x | OpenAI / DeepSeek 驱动 |
| `langchain-ollama` | 1.1.x | 本地 Ollama 模型 |
| `langchain-community` | 0.4.x | 社区组件（Tavily、Redis 等） |
| `langchain-mcp-adapters` | 0.3.x | MCP 工具适配 |
| `openai` | 2.41.x | OpenAI SDK |
| `pydantic` | 2.13.x | 数据验证 |
| `redis` | 5.3.x | 向量搜索 & 聊天记忆 |
| `pypdf` | 6.13.x | PDF 解析 |

基于 Conda 环境 `langchain-env`，完整依赖见 `requirements.txt`。
