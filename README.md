# Wiki RAG - 个人知识库系统

> Karpathy 范式的活 Wiki + 轻量向量检索：不依赖向量数据库，一个 Python 文件跑通全链路

## 架构设计

```
raw/           → 原始知识素材（Markdown笔记）
wiki/          → LLM编译后的结构化Wiki文章
storage/       → 向量索引（JSON格式，文件级增量）
wiki/index.md  → 全局索引目录
```

**核心流程：**
1. `add` — 添加原始素材，LLM 自动拆分多主题
2. `compile` — raw笔记 → LLM编译为结构化wiki文章 + 构建索引
3. `query` — 用户提问 → 向量检索 → LLM回答

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 配置 LLM（默认 DeepSeek API）
export DEEPSEEK_API_KEY=sk-xxx
# 也可切换到本地 Ollama：
#   export WIKI_USE_LOCAL_LLM=1 && ollama pull deepseek-r1:1.5b

# 3. Embedding 始终走本地 Ollama
ollama pull nomic-embed-text

# 4. 编译raw → wiki
uv run python -m src.wiki_rag compile

# 5. 查询
uv run python -m src.wiki_rag query "Python装饰器是什么"

# 6. 添加新知识（文件或直接文本，自动拆分多主题）
uv run python -m src.wiki_rag add notes.md
uv run python -m src.wiki_rag add --text "一段包含多个主题的笔记..."
uv run python -m src.wiki_rag compile  # 重新编译
```

## 技术栈

- **LLM**: DeepSeek API（默认）/ 本地 Ollama（`WIKI_USE_LOCAL_LLM=1`）
- **Embedding**: Ollama（nomic-embed-text, 768维）
- **向量检索**: numpy 余弦相似度，矩阵化一次计算
- **存储**: JSON文件（MVP级，可扩展为FAISS/Chroma）

## 设计理念

**Karpathy范式：** 不用向量数据库和传统RAG栈，用LLM自己维护的"活Wiki"作为知识载体。

**混合架构：**
- Wiki层：结构化的Markdown知识库（人可读、LLM可维护）
- RAG检索层：向量检索，支持大规模精准查询

**为什么不用纯Karpathy：** 纯Wiki适合100篇以内的个人知识。当知识量增大，需要一个检索层。我们的混合方案取两者之长。

## 测试

```bash
uv run pytest tests/ -v
```

## 仓库

https://github.com/helloworldtang/wiki-rag
