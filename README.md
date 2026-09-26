# Wiki RAG - 个人知识库系统

[![CI](https://github.com/helloworldtang/wiki-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/helloworldtang/wiki-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

> Karpathy 范式的活 Wiki + 轻量向量检索：不依赖向量数据库，核心代码一个包，跑通 RAG 全链路。

## 它解决什么问题

个人笔记越攒越多，"存了但想不起来"。本项目用 LLM 把原始笔记编译成结构化 Wiki，再用向量检索让提问直达答案——**全程不依赖向量数据库和 RAG 框架**，适合想搞懂 RAG 每一步原理的人阅读和魔改。

## 工作原理

```
                 ┌──────────────────────────────────────────────┐
                 │                compile（编译）                │
 raw/note.md ──▶ │ LLM改写为结构化Wiki ──▶ wiki/note.md          │
                 │ 切chunk(≤512字) ──▶ Ollama批量embedding       │
                 │        ──▶ storage/vector_index.json          │
                 └──────────────────────────────────────────────┘

                 ┌──────────────────────────────────────────────┐
 "Python装饰器   │                query（问答）                  │
  是什么？" ───▶ │ 问题embedding ──▶ 余弦相似度矩阵 ──▶ Top-5    │
                 │ chunk ──▶ 拼prompt ──▶ DeepSeek ──▶ 回答      │
                 └──────────────────────────────────────────────┘
```

**数据流三层：**

| 目录 | 作用 | 谁写它 |
|---|---|---|
| `raw/` | 原始笔记（Markdown） | 你 / `add` 命令 |
| `wiki/` | LLM 编译后的结构化文章 + `index.md` 目录页 | `compile` 命令 |
| `storage/` | `meta.json`（编译缓存）+ `vector_index.json`（向量索引） | `compile` 命令 |

**几个关键设计（都有代码注释对应）：**

- **文件级增量**：wiki 文件 hash 未变就复用旧 embedding，重复 `compile` 的嵌入调用次数为 0
- **chunk 切分**：按段落切，超长段落按句子贪心打包到 512 字（`CHUNK_SIZE`）
- **矩阵化检索**：全部 embedding 堆成矩阵归一化，一次矩阵乘拿到所有相似度，不是逐条循环
- **中文友好降级**：Ollama 挂掉时自动降级为字符 bigram 关键词匹配（中文没有空格分词，bigram 是零依赖的轻量方案）
- **未命中不硬答**：检索不到相关内容时直接说"没找到"，不把全库塞进 prompt

## 快速开始

```bash
# 1. 克隆 + 安装（需要 uv：https://docs.astral.sh/uv/）
git clone https://github.com/helloworldtang/wiki-rag.git
cd wiki-rag
uv sync

# 2. 配置密钥（Embedding 始终走本地，LLM 默认走 DeepSeek API）
cp .env.example .env       # 填入 DEEPSEEK_API_KEY
ollama pull nomic-embed-text

# 3. 跑起来：仓库自带 raw/ 示例笔记，可直接体验全链路
uv run python -m wiki_rag compile          # 编译 + 构建索引（首次需调用 LLM，稍慢）
uv run python -m wiki_rag query "Python装饰器是什么"

# 4. 添加自己的知识
uv run python -m wiki_rag add notes.md             # 文件（LLM自动拆分多主题）
uv run python -m wiki_rag add --no-split notes.md  # 文件（原样保存，不经过LLM）
uv run python -m wiki_rag add --text "一段混了多个主题的笔记..."
cat note.md | uv run python -m wiki_rag add -      # 管道
uv run python -m wiki_rag compile                  # 重新编译（增量，只处理新内容）
```

完全离线方案：设置 `WIKI_USE_LOCAL_LLM=1`，LLM 也走本地 Ollama（质量会下降，见下表）。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | （无） | DeepSeek API 密钥，不设则必须开本地模式 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址，走中转时改 |
| `WIKI_LLM_MODEL` | `deepseek-chat` | LLM 模型名 |
| `WIKI_EMBED_MODEL` | `nomic-embed-text:latest` | Embedding 模型（Ollama） |
| `WIKI_USE_LOCAL_LLM` | （关） | 设为 `1` 时 LLM 走本地 Ollama |
| `WIKI_LOCAL_MODEL` | `deepseek-r1:1.5b` | 本地 LLM 模型名 |

均可写入 `.env`（已 gitignore，不会误提交密钥）。

## 技术栈

- **LLM**: DeepSeek API（默认）/ 本地 Ollama
- **Embedding**: Ollama（nomic-embed-text, 768维）
- **向量检索**: numpy 余弦相似度，矩阵化一次计算
- **存储**: JSON 文件（MVP级，刻意不用数据库）

## 常见问题

| 症状 | 原因与解法 |
|---|---|
| `connection refused` / 向量检索失败 | Ollama 没启动：`ollama serve`，然后 `ollama pull nomic-embed-text` |
| `DEEPSEEK_API_KEY 未设置` | 复制 `.env.example` 为 `.env` 并填入密钥 |
| `embedding失败 ... model not found` | `ollama pull nomic-embed-text`，或改 `.env` 里的 `WIKI_EMBED_MODEL` |
| 首次 `compile` 很慢 | 正常：每篇 raw 都要过一次 LLM + embedding，第二次起增量跳过 |
| 回答说"没有相关信息" | 检索未命中，属于设计行为（不硬答）；先 `compile` 再试 |

## 设计理念

**Karpathy范式：** 不用向量数据库和传统RAG栈，用LLM自己维护的"活Wiki"作为知识载体。

**为什么不用纯Karpathy：** 纯Wiki适合100篇以内的个人知识。当知识量增大，需要一个检索层。本项目的混合方案取两者之长。

**为什么不用 LangChain / LlamaIndex：** 教学项目的核心价值是**每一行都能读懂**。框架封装掉的恰恰是值得理解的部分——chunking、embedding、检索、prompt 组装这里全部手写，总共不到 400 行核心代码，且每个函数都有测试覆盖。

## 扩展方向

想动手实践？这些是难度递增的好题目：

1. **混合检索**：向量 + BM25 双路召回，用 RRF 融合（中文可用 jieba 分词替换 bigram）
2. **换存储**：`storage/` 的 JSON 换成 FAISS / sqlite-vec，接口就 `build_vector_index` 和 `query` 两个函数
3. **Rerank**：Top-5 召回后加一层 bge-reranker 精排，提升命中率
4. **评测**：写一组问题+标准答案，用检索命中率量化每次改动的效果

## 开发

```bash
uv sync                  # 安装依赖（含 dev 组）
uv run pytest tests/ -v  # 21 个测试
uvx ruff check src tests # lint
```

提交前跑一遍上面两条命令即可，CI（`.github/workflows/ci.yml`）也会在 Python 3.10/3.12/3.13 上做同样检查。

## License

[MIT](LICENSE)
