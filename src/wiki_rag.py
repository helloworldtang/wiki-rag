"""
Wiki RAG 系统 - Karpathy范式的LlamaIndex实现

架构设计（三层）：
1. raw/    — 原始知识素材（Markdown笔记）
2. wiki/   — LLM编译后的结构化Wiki文章
3. index.md — 全局索引，所有Wiki文章的目录

核心流程：
- compile: raw → wiki（LLM摘要+结构化）
- build_index: wiki → index.md（生成全局索引）
- query: 用户提问 → RAG检索wiki → LLM回答
- add: 新增raw素材 → 自动compile → 更新索引

混合架构：
- Karpathy Wiki层：结构化的Markdown知识库
- RAG检索层：向量检索，支持大规模查询
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional
import numpy as np

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# ============================================================
# 配置
# ============================================================

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "raw"
WIKI_DIR = BASE_DIR / "wiki"
INDEX_FILE = BASE_DIR / "wiki" / "index.md"
STORAGE_DIR = BASE_DIR / "storage"
META_FILE = BASE_DIR / "storage" / "meta.json"

# Ollama 模型配置
LLM_MODEL = os.getenv("WIKI_LLM_MODEL", "deepseek-r1:1.5b")
EMBED_MODEL = os.getenv("WIKI_EMBED_MODEL", "nomic-embed-text:latest")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def setup_settings():
    """初始化LlamaIndex全局设置，使用Ollama本地模型"""
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0, context_window=4096, num_ctx=4096)
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url="http://localhost:11434",
    )
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP


# ============================================================
# Wiki 编译器：raw → wiki
# ============================================================

COMPILE_PROMPT = """你是一个知识整理专家。请将以下原始笔记编译为一篇结构清晰的Wiki文章。

要求：
1. 保留所有核心技术要点，不遗漏关键信息
2. 使用清晰的标题层级（h1主题，h2子主题）
3. 关键术语加粗
4. 如果有代码示例，保留核心代码
5. 在文章末尾添加一个「一句话总结」
6. 生成3-5个标签（#tag格式）

原始笔记：
---
{content}
---

请输出编译后的Wiki文章（Markdown格式）："""


def file_hash(filepath: Path) -> str:
    """计算文件内容的MD5哈希"""
    return hashlib.md5(filepath.read_bytes()).hexdigest()


def load_meta() -> dict:
    """加载元数据（记录哪些raw文件已编译）"""
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return {"compiled": {}}  # {filename: {hash: str, wiki_file: str}}


def save_meta(meta: dict):
    """保存元数据"""
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def compile_raw_to_wiki(raw_file: Path, force: bool = False) -> Optional[Path]:
    """
    将一个raw文件编译为wiki文章。
    如果文件内容未变化且已编译，跳过。
    返回wiki文件路径，如果跳过返回None。
    """
    meta = load_meta()
    filename = raw_file.name
    current_hash = file_hash(raw_file)

    # 检查是否需要重新编译
    if not force and filename in meta["compiled"]:
        if meta["compiled"][filename]["hash"] == current_hash:
            wiki_path = WIKI_DIR / meta["compiled"][filename]["wiki_file"]
            if wiki_path.exists():
                return None  # 无变化，跳过

    print(f"  📝 编译: {filename} ...")

    # 调用LLM编译（直接使用ollama SDK，避免LlamaIndex Ollama集成问题）
    content = raw_file.read_text(encoding="utf-8")
    import ollama as _ollama
    client = _ollama.Client()
    response = client.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": COMPILE_PROMPT.format(content=content)}],
        stream=False,
    )
    wiki_content = response["message"]["content"]

    # 生成wiki文件名
    stem = raw_file.stem
    wiki_file = f"{stem}.md"
    wiki_path = WIKI_DIR / wiki_file

    # 写入wiki
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    wiki_path.write_text(wiki_content, encoding="utf-8")

    # 更新元数据
    meta["compiled"][filename] = {
        "hash": current_hash,
        "wiki_file": wiki_file,
    }
    save_meta(meta)

    print(f"  ✅ 完成: {wiki_file}")
    return wiki_path


# ============================================================
# 索引构建器：wiki → index.md + 向量索引
# ============================================================

def build_wiki_index() -> Path:
    """
    从所有wiki文章生成 index.md（全局目录）
    """
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    wiki_files = sorted(WIKI_DIR.glob("*.md"))

    lines = ["# 📚 个人Wiki知识库索引\n"]
    lines.append(f"> 共 {len(wiki_files)} 篇文章 | 自动生成\n")
    lines.append("## 文章目录\n")

    for wf in wiki_files:
        if wf.name == "index.md":
            continue
        title = wf.stem.replace("-", " ").replace("_", " ").title()
        # 读取文件第一行作为摘要
        first_line = ""
        with open(wf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    first_line = line[:80]
                    break
        lines.append(f"- **{title}** — {first_line}  ")
        lines.append(f"  `wiki/{wf.name}`\n")

    index_content = "\n".join(lines)
    INDEX_FILE.write_text(index_content, encoding="utf-8")
    print(f"  📑 索引已更新: {len(wiki_files)-1} 篇文章")
    return INDEX_FILE


def build_vector_index():
    """
    从wiki文章构建向量索引。
    直接使用ollama SDK生成embedding，避免LlamaIndex Ollama集成的502问题。
    """
    print("  🔨 构建向量索引 ...")
    import ollama as _ollama
    import numpy as np

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    wiki_files = [f for f in WIKI_DIR.glob("*.md") if f.name != "index.md"]
    if not wiki_files:
        print("  ⚠️  没有wiki文章，跳过索引构建")
        return None

    client = _ollama.Client()
    chunks = []  # [{text, metadata, embedding}]

    for wf in wiki_files:
        content = wf.read_text(encoding="utf-8")
        title = wf.stem.replace("-", " ").replace("_", " ").title()
        # 简单分块：按段落分割
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            if len(para) < 20:
                continue
            try:
                resp = client.embeddings(model=EMBED_MODEL, prompt=para)
                emb = resp["embedding"]
                chunks.append({
                    "text": para,
                    "metadata": {"filename": wf.name, "title": title, "chunk": i},
                    "embedding": emb,
                })
            except Exception as e:
                print(f"  ⚠️  embedding失败 {wf.name} chunk {i}: {e}")
                continue

    if not chunks:
        print("  ⚠️  没有有效的embedding")
        return None

    # 保存索引（JSON格式，简单但够用）
    index_data = {
        "model": EMBED_MODEL,
        "dim": len(chunks[0]["embedding"]),
        "chunks": chunks,
    }
    index_file = STORAGE_DIR / "vector_index.json"
    index_file.write_text(json.dumps(index_data, ensure_ascii=False), encoding="utf-8")
    print(f"  ✅ 向量索引构建完成: {len(chunks)} 个chunk, {len(wiki_files)} 篇文章")
    return index_data


# ============================================================
# 查询引擎
# ============================================================

def create_query_engine():
    """
    创建查询引擎。从本地索引加载，进行向量检索 + LLM生成回答。
    """
    index_file = STORAGE_DIR / "vector_index.json"
    if not index_file.exists():
        return None

    import ollama as _ollama
    import numpy as np

    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    return {"data": index_data, "client": _ollama.Client()}


def _cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def query(question: str, query_engine=None) -> str:
    """
    查询Wiki知识库：embedding检索 → 拼接上下文 → LLM回答
    """
    if query_engine is None:
        setup_settings()
        query_engine = create_query_engine()
        if query_engine is None:
            return "❌ 知识库为空，请先运行 compile"

    client = query_engine["client"]
    data = query_engine["data"]

    # 1. 对问题生成embedding
    q_emb = client.embeddings(model=data["model"], prompt=question)["embedding"]

    # 2. 检索最相关的chunk
    scored = []
    for chunk in data["chunks"]:
        sim = _cosine_similarity(q_emb, chunk["embedding"])
        scored.append((sim, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)

    top_chunks = scored[:3]
    context = "\n\n".join(
        f"[{c['metadata']['title']} - chunk {c['metadata']['chunk']}]\n{c['text']}"
        for _, c in top_chunks
    )

    # 3. LLM生成回答
    prompt = f"""基于以下知识库内容回答问题。如果知识库中没有相关信息，请直接说明。

知识库内容：
{context}

问题：{question}

请用中文回答："""

    resp = client.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return resp["message"]["content"]


# ============================================================
# CLI 入口
# ============================================================

def cmd_compile(force: bool = False):
    """编译所有raw文件为wiki文章"""
    setup_settings()
    print("🔨 编译 raw → wiki ...")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = list(RAW_DIR.glob("*.md"))

    if not raw_files:
        print("  ⚠️  raw/ 目录为空")
        return

    compiled_count = 0
    for rf in raw_files:
        result = compile_raw_to_wiki(rf, force=force)
        if result is not None:
            compiled_count += 1

    print(f"\n📊 编译完成: {compiled_count} 篇新编译, {len(raw_files)-compiled_count} 篇无变化")

    # 更新索引
    build_wiki_index()
    build_vector_index()


def cmd_query(question: str):
    """查询知识库"""
    setup_settings()
    print(f"🔍 查询: {question}\n")
    answer = query(question)
    print(f"💡 回答:\n{answer}\n")


def cmd_add(title: str, content: str):
    """新增知识条目"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filename = title.lower().replace(" ", "-").replace("/", "-") + ".md"
    filepath = RAW_DIR / filename
    filepath.write_text(f"# {title}\n\n{content}", encoding="utf-8")
    print(f"✅ 已添加: {filename}")
    print("💡 运行 `python -m src.wiki_rag compile` 编译新条目")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m src.wiki_rag [compile|query|add] [args]")
        print("  compile [--force]  编译raw → wiki")
        print("  query <question>   查询知识库")
        print("  add <title> <content>  添加新条目")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "compile":
        force = "--force" in sys.argv
        cmd_compile(force=force)
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("请输入查询内容")
            sys.exit(1)
        cmd_query(" ".join(sys.argv[2:]))
    elif cmd == "add":
        if len(sys.argv) < 4:
            print("用法: add <title> <content>")
            sys.exit(1)
        cmd_add(sys.argv[2], " ".join(sys.argv[3:]))
    else:
        print(f"未知命令: {cmd}")
