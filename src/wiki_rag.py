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
- add: 新增raw素材 → 自动拆分多主题 → compile → 更新索引

模型策略：
- 默认使用DeepSeek API（线上模型，回答质量高）
- Embedding使用Ollama本地模型（nomic-embed-text）
- 可通过环境变量切换模型
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional
import numpy as np

# ============================================================
# 配置
# ============================================================

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "raw"
WIKI_DIR = BASE_DIR / "wiki"
INDEX_FILE = BASE_DIR / "wiki" / "index.md"
STORAGE_DIR = BASE_DIR / "storage"
META_FILE = BASE_DIR / "storage" / "meta.json"

# 模型配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("WIKI_LLM_MODEL", "deepseek-chat")
EMBED_MODEL = os.getenv("WIKI_EMBED_MODEL", "nomic-embed-text:latest")
USE_LOCAL_LLM = os.getenv("WIKI_USE_LOCAL_LLM", "").lower() in ("1", "true", "yes")

CHUNK_SIZE = 512


def get_llm_client():
    """获取LLM客户端。默认使用DeepSeek API，可通过环境变量切换到本地Ollama"""
    if USE_LOCAL_LLM:
        import ollama
        return {"type": "ollama", "client": ollama.Client(), "model": os.getenv("WIKI_LOCAL_MODEL", "deepseek-r1:1.5b")}
    
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY 未设置。请设置环境变量或设置 WIKI_USE_LOCAL_LLM=1 使用本地模型")
    
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return {"type": "openai", "client": client, "model": LLM_MODEL}


def llm_chat(prompt: str, system: str = "你是一个知识管理专家。") -> str:
    """统一的LLM调用接口"""
    llm = get_llm_client()
    if llm["type"] == "openai":
        resp = llm["client"].chat.completions.create(
            model=llm["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    else:
        resp = llm["client"].chat(
            model=llm["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return resp["message"]["content"]


def get_embed_client():
    """获取Embedding客户端（始终使用本地Ollama）"""
    import ollama
    return ollama.Client()


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
    return hashlib.md5(filepath.read_bytes()).hexdigest()


def load_meta() -> dict:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return {"compiled": {}}


def save_meta(meta: dict):
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def compile_raw_to_wiki(raw_file: Path, force: bool = False) -> Optional[Path]:
    meta = load_meta()
    filename = raw_file.name
    current_hash = file_hash(raw_file)

    if not force and filename in meta["compiled"]:
        if meta["compiled"][filename]["hash"] == current_hash:
            wiki_path = WIKI_DIR / meta["compiled"][filename]["wiki_file"]
            if wiki_path.exists():
                return None

    print(f"  📝 编译: {filename} ...")

    content = raw_file.read_text(encoding="utf-8")
    wiki_content = llm_chat(COMPILE_PROMPT.format(content=content))

    stem = raw_file.stem
    wiki_file = f"{stem}.md"
    wiki_path = WIKI_DIR / wiki_file

    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    wiki_path.write_text(wiki_content, encoding="utf-8")

    meta["compiled"][filename] = {"hash": current_hash, "wiki_file": wiki_file}
    save_meta(meta)

    print(f"  ✅ 完成: {wiki_file}")
    return wiki_path


# ============================================================
# 智能添加：支持多主题自动拆分
# ============================================================

SPLIT_PROMPT = """你是一个知识分类专家。用户会给你一段文本（可能是Markdown格式的长文）。

请分析这段文本，识别其中的所有独立主题，然后为每个主题生成：
1. 一个简洁的标题（中文，10字以内）
2. 对应的内容（保留原文相关段落，保持Markdown格式）

如果文本只有一个主题，就输出一个。
如果文本有多个主题，分别输出。

请用以下JSON格式输出（不要用markdown代码块包裹）：
[
  {{"title": "主题标题1", "content": "对应内容（Markdown格式）"}},
  {{"title": "主题标题2", "content": "对应内容（Markdown格式）"}}
]

原始文本：
---
{text}
---"""


def cmd_add_smart(text: str):
    """智能添加：LLM自动拆分多主题，生成多个raw文件"""
    print("🧠 分析文本，识别主题 ...")
    
    response = llm_chat(SPLIT_PROMPT.format(text=text))
    
    # 解析JSON（兼容LLM输出可能带的markdown代码块）
    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[1] if "\n" in response else response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    
    try:
        topics = json.loads(response)
    except json.JSONDecodeError:
        # JSON解析失败，作为单主题处理
        topics = [{"title": "未分类知识", "content": text}]
    
    if not isinstance(topics, list):
        topics = [topics]
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    
    for topic in topics:
        title = topic.get("title", "未分类")
        content = topic.get("content", "")
        
        # 生成文件名
        filename = title.lower().replace(" ", "-").replace("/", "-")
        # 去掉特殊字符
        filename = "".join(c for c in filename if c.isalnum() or c in "-_") + ".md"
        filepath = RAW_DIR / filename
        
        # 如果文件已存在，追加内容
        if filepath.exists():
            existing = filepath.read_text(encoding="utf-8")
            filepath.write_text(existing + "\n\n---\n\n" + content, encoding="utf-8")
            print(f"  📎 追加到: {filename}")
        else:
            filepath.write_text(f"# {title}\n\n{content}", encoding="utf-8")
            print(f"  ✅ 新建: {filename}")
        created.append(filename)
    
    print(f"\n📊 识别 {len(topics)} 个主题: {', '.join(t.get('title', '?') for t in topics)}")
    print(f"💡 运行 `uv run python -m src.wiki_rag compile` 编译新条目")
    return created


# ============================================================
# 索引构建器
# ============================================================

def build_wiki_index() -> Path:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    wiki_files = sorted(WIKI_DIR.glob("*.md"))

    lines = ["# 📚 个人Wiki知识库索引\n"]
    lines.append(f"> 共 {len(wiki_files)} 篇文章 | 自动生成\n")
    lines.append("## 文章目录\n")

    for wf in wiki_files:
        if wf.name == "index.md":
            continue
        title = wf.stem.replace("-", " ").replace("_", " ").title()
        first_line = ""
        with open(wf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    first_line = line[:80]
                    break
        lines.append(f"- **{title}** — {first_line}  ")
        lines.append(f"  `wiki/{wf.name}`\n")

    INDEX_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📑 索引已更新: {len(wiki_files)-1} 篇文章")
    return INDEX_FILE


def build_vector_index():
    """从wiki文章构建向量索引（使用Ollama本地Embedding）"""
    print("  🔨 构建向量索引 ...")

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    wiki_files = [f for f in WIKI_DIR.glob("*.md") if f.name != "index.md"]
    
    if not wiki_files:
        print("  ⚠️  没有wiki文章，跳过索引构建")
        return None

    client = get_embed_client()
    chunks = []

    for wf in wiki_files:
        content = wf.read_text(encoding="utf-8")
        title = wf.stem.replace("-", " ").replace("_", " ").title()
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        
        for i, para in enumerate(paragraphs):
            if len(para) < 20:
                continue
            try:
                resp = client.embeddings(model=EMBED_MODEL, prompt=para)
                chunks.append({
                    "text": para,
                    "metadata": {"filename": wf.name, "title": title, "chunk": i},
                    "embedding": resp["embedding"],
                })
            except Exception as e:
                print(f"  ⚠️  embedding失败 {wf.name} chunk {i}: {e}")

    if not chunks:
        print("  ⚠️  没有有效的embedding")
        return None

    index_data = {"model": EMBED_MODEL, "dim": len(chunks[0]["embedding"]), "chunks": chunks}
    (STORAGE_DIR / "vector_index.json").write_text(
        json.dumps(index_data, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  ✅ 向量索引构建完成: {len(chunks)} 个chunk, {len(wiki_files)} 篇文章")
    return index_data


# ============================================================
# 查询引擎
# ============================================================

def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return np.dot(a, b) / norm if norm > 0 else 0


def query(question: str) -> str:
    """查询Wiki知识库：关键词匹配/向量检索 → LLM回答（使用DeepSeek）"""
    index_file = STORAGE_DIR / "vector_index.json"
    if not index_file.exists():
        return "❌ 知识库为空，请先运行 compile"

    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    
    # 尝试向量检索，失败则用关键词匹配
    top_chunks = []
    try:
        embed_client = get_embed_client()
        q_emb = embed_client.embeddings(model=index_data["model"], prompt=question)["embedding"]
        scored = [(sim, c) for c in index_data["chunks"] if (sim := _cosine_similarity(q_emb, c["embedding"])) is not None]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored[:5]
    except Exception as e:
        print(f"  ⚠️  向量检索失败({e})，使用关键词匹配")
        # 关键词匹配 fallback：搜索子串匹配
        scored_kw = []
        # 提取问题中的关键词（去掉常见停用词）
        stopwords = {"的", "是", "了", "在", "有", "和", "与", "或", "不", "也", "都", "吗", "什么", "怎么", "如何", "为什么", "哪", "哪些", "可以", "能", "会", "请", "问"}
        q_keywords = [w for w in question.replace("？", "").replace("?", "").split() if w not in stopwords and len(w) > 1]
        # 如果分词后为空，直接用原始问题搜索
        if not q_keywords:
            q_keywords = [question]
        for chunk in index_data["chunks"]:
            text_lower = chunk["text"].lower()
            hits = sum(1 for kw in q_keywords if kw.lower() in text_lower)
            if hits > 0:
                scored_kw.append((hits, chunk))
        scored_kw.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [(0, c) for _, c in scored_kw[:5]]

    if not top_chunks:
        # 没有匹配到chunk，直接传所有wiki内容给LLM
        print("  ℹ️  无精确匹配，使用全量上下文")
        all_wiki = []
        for wf in sorted(WIKI_DIR.glob("*.md")):
            if wf.name == "index.md": continue
            all_wiki.append(f"[{wf.stem}]\n{wf.read_text(encoding='utf-8')}")
        context = "\n\n".join(all_wiki) if all_wiki else "(空)"
    else:
        context = "\n\n".join(
            f"[{c['metadata']['title']}]\n{c['text']}" for _, c in top_chunks
        )

    prompt = f"""基于以下知识库内容回答问题。如果知识库中没有相关信息，请直接说明。

知识库内容：
{context}

问题：{question}

请用中文详细回答："""

    return llm_chat(prompt, system="你是一个知识助手。基于提供的知识库内容准确回答问题。")


# ============================================================
# CLI 入口
# ============================================================

def cmd_compile(force: bool = False):
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
    build_wiki_index()
    build_vector_index()


def cmd_query(question: str):
    print(f"🔍 查询: {question}\n")
    answer = query(question)
    print(f"💡 回答:\n{answer}\n")


def cmd_add(args: list[str]):
    """添加知识。支持：文件路径、直接文本、管道输入"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    if not args:
        print("用法:")
        print("  add <文件路径>         从文件添加")
        print("  add --text <内容>      直接添加文本（自动拆分多主题）")
        return
    
    if args[0] == "--text":
        # 直接文本模式：智能拆分多主题
        text = " ".join(args[1:]) if len(args) > 1 else ""
        if not text:
            # 尝试从stdin读取
            import sys
            if not sys.stdin.isatty():
                text = sys.stdin.read()
        if not text:
            print("请输入内容")
            return
        cmd_add_smart(text)
    else:
        # 文件模式
        filepath = Path(args[0])
        if not filepath.exists():
            print(f"文件不存在: {filepath}")
            return
        content = filepath.read_text(encoding="utf-8")
        cmd_add_smart(content)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m src.wiki_rag [compile|query|add] [args]")
        print("  compile [--force]     编译raw → wiki")
        print("  query <question>      查询知识库")
        print("  add <文件路径>         从文件添加（自动拆分多主题）")
        print("  add --text <内容>     直接添加文本（自动拆分多主题）")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "compile":
        cmd_compile(force="--force" in sys.argv)
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("请输入查询内容")
            sys.exit(1)
        cmd_query(" ".join(sys.argv[2:]))
    elif cmd == "add":
        cmd_add(sys.argv[2:])
    else:
        print(f"未知命令: {cmd}")
