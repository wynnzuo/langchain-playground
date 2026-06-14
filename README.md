# 🎮 LangChain Playground

LangChain 框架的示例集合，涵盖 Prompt 模板、输出解析、Agent、流式调用、多模型对接等场景。

## 📂 项目结构

### 模型调用
| 文件 | 说明 |
|------|------|
| `init_chat_model_example.py` | 使用 `init_chat_model` 快速调用大模型 |
| `init_chat_model_messages.py` | 通过 Message 对象传递对话历史 |
| `init_chat_model_stream.py` | 流式输出示例 |
| `chatopenai_identity_example.py` | `ChatOpenAI` 直接调用（简化版） |
| `openai_sdk_direct.py` | 使用 OpenAI SDK 直连（绕过 LangChain） |
| `ollama_chat_example.py` | 对接本地 Ollama 模型 |

### Prompt 模板
| 文件 | 说明 |
|------|------|
| `prompt_template_example.py` | `PromptTemplate` 的 `format`、`partial_variables`、`invoke` 用法 |
| `chat_prompt_template_example.py` | `ChatPromptTemplate` 的多种构建方式（tuple、dict、Message 对象）及 `MessagesPlaceholder` |

### 输出解析
| 文件 | 说明 |
|------|------|
| `json_output_parser_example.py` | 使用 `JsonOutputParser` 让模型返回 JSON |
| `pydantic_output_parser_demo.py` | 使用 `PydanticOutputParser` 解析为 Pydantic 模型 |

### 其他
| 文件 | 说明 |
|------|------|
| `typeddict_base_model_example.py` | `TypedDict` 与 Pydantic `BaseModel` 对比 |
| `langchain_agent_france_example.py` | Agent + Tool 调用示例 |
| `langchain_version_info.py` | 查看当前环境版本信息 |

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/wynnzuo/langchain-playground.git
cd langchain-playground

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API key 和 base URL

# 4. 运行示例
python init_chat_model_example.py
```

## ⚙️ 环境变量

在 `.env` 文件中配置：

```ini
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.deepseek.com
```

项目使用 DeepSeek 兼容 OpenAI 接口的 API，也支持其他 OpenAI 兼容服务，只需修改 `base_url` 和 `api_key`。

## 📦 依赖

基于 Conda 环境 `langchain-env`，LangChain v1 系列。详见 `requirements.txt`。
