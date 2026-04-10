# Wiki RAG - 个人知识库系统

> Karpathy范式的LlamaIndex实现，用Ollama本地模型搭建可闭环的MVP

## 架构设计

```
raw/           → 原始知识素材（Markdown笔记）
wiki/          → LLM编译后的结构化Wiki文章  
storage/       → 向量索引（JSON格式）
wiki/index.md  → 全局索引目录
```

**核心流程：**
1. `compile` — raw笔记 → LLM编译为结构化wiki文章
2. `build_index` — wiki文章 → 全局索引 + 向量索引
3. `query` — 用户提问 → 向量检索 → LLM回答

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 确保Ollama在运行，需要以下模型
ollama pull deepseek-r1:1.5b     # LLM
ollama pull nomic-embed-text      # Embedding

# 3. 编译raw → wiki
uv run python -m src.wiki_rag compile

# 4. 查询
uv run python -m src.wiki_rag query "Python装饰器是什么"

# 5. 添加新知识
uv run python -m src.wiki_rag add "新主题" "笔记内容..."
uv run python -m src.wiki_rag compile  # 重新编译
```

## 技术栈

- **LLM**: Ollama (deepseek-r1:1.5b)
- **Embedding**: Ollama (nomic-embed-text, 768维)
- **向量检索**: 余弦相似度（numpy）
- **存储**: JSON文件（MVP级，可扩展为FAISS/Chroma）
- **框架**: LlamaIndex + 直接ollama SDK

## 设计理念

**Karpathy范式：** 不用向量数据库和传统RAG栈，用LLM自己维护的"活Wiki"作为知识载体。

**混合架构：**
- Wiki层：结构化的Markdown知识库（人可读、LLM可维护）
- RAG检索层：向量检索，支持大规模精准查询

**为什么不用纯Karpathy：** 纯Wiki适合100篇以内的个人知识。当知识量增大，需要一个检索层。我们的混合方案取两者之长。

## 测试

```bash
uv run pytest tests/ -v
# 13 passed
```

## 仓库

https://github.com/helloworldtang/wiki-rag
