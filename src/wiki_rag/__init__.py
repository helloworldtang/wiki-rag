"""
Wiki RAG 系统 - Karpathy范式的个人知识库RAG（DeepSeek + Ollama本地Embedding）

架构设计（三层）：
1. raw/    — 原始知识素材（Markdown笔记）
2. wiki/   — LLM编译后的结构化Wiki文章
3. index.md — 全局索引，所有Wiki文章的目录

核心流程：
- add: 新增raw素材（可选LLM自动拆分多主题）
- compile: raw → wiki（LLM摘要+结构化）+ 构建索引
- query: 用户提问 → 向量检索wiki → LLM回答

模型策略：
- 默认使用DeepSeek API（线上模型，回答质量高）
- Embedding使用Ollama本地模型（nomic-embed-text）
- 可通过环境变量或 .env 文件切换模型（见 .env.example）
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

# ============================================================
# 配置
# ============================================================

# src/wiki_rag/__init__.py 向上两级 = 仓库根目录
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "raw"
WIKI_DIR = BASE_DIR / "wiki"
INDEX_FILE = BASE_DIR / "wiki" / "index.md"
STORAGE_DIR = BASE_DIR / "storage"
META_FILE = BASE_DIR / "storage" / "meta.json"


def _load_dotenv(path: Path) -> None:
    """极简.env加载（不引入依赖）：只支持 KEY=VALUE 和 # 注释，已存在的环境变量不覆盖"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(BASE_DIR / ".env")

# 模型配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("WIKI_LLM_MODEL", "deepseek-chat")
EMBED_MODEL = os.getenv("WIKI_EMBED_MODEL", "nomic-embed-text:latest")
USE_LOCAL_LLM = os.getenv("WIKI_USE_LOCAL_LLM", "").lower() in ("1", "true", "yes")

CHUNK_SIZE = 512
TOP_K = 5


def get_llm_client() -> dict[str, Any]:
    """获取LLM客户端。默认使用DeepSeek API，可通过环境变量切换到本地Ollama"""
    if USE_LOCAL_LLM:
        import ollama
        return {"type": "ollama", "client": ollama.Client(), "model": os.getenv("WIKI_LOCAL_MODEL", "deepseek-r1:1.5b")}

    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "DEEPSEEK_API_KEY 未设置。请在 .env 中配置或设置环境变量，"
            "或设置 WIKI_USE_LOCAL_LLM=1 使用本地模型"
        )

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
        return str(resp.choices[0].message.content)
    else:
        resp = llm["client"].chat(
            model=llm["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return str(resp["message"]["content"])


def get_embed_client() -> Any:
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


def load_meta() -> dict[str, Any]:
    if META_FILE.exists():
        data = json.loads(META_FILE.read_text())
        if isinstance(data, dict):
            return data
    return {"compiled": {}}


def save_meta(meta: dict[str, Any]) -> None:
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def compile_raw_to_wiki(raw_file: Path, force: bool = False, meta: dict[str, Any] | None = None) -> Path | None:
    """编译单个raw文件。未变化（hash一致且wiki存在）返回None，否则返回wiki文件路径"""
    if meta is None:
        meta = load_meta()
    filename = raw_file.name
    current_hash = file_hash(raw_file)

    if not force and filename in meta["compiled"] and meta["compiled"][filename]["hash"] == current_hash:
        wiki_path = WIKI_DIR / meta["compiled"][filename]["wiki_file"]
        if wiki_path.exists():
            return None

    print(f"  📝 编译: {filename} ...")

    content = raw_file.read_text(encoding="utf-8")
    wiki_content = llm_chat(COMPILE_PROMPT.format(content=content))

    wiki_file = f"{raw_file.stem}.md"
    wiki_path = WIKI_DIR / wiki_file

    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    wiki_path.write_text(wiki_content, encoding="utf-8")

    meta["compiled"][filename] = {"hash": current_hash, "wiki_file": wiki_file}
    save_meta(meta)

    print(f"  ✅ 完成: {wiki_file}")
    return wiki_path


# ============================================================
# 添加知识：多主题自动拆分 / 原样直存
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


def _filename_from_title(title: str) -> str:
    """标题 → 安全文件名（保留中文/字母/数字/-_，空标题用hash兜底）"""
    stem = title.lower().replace(" ", "-").replace("/", "-")
    stem = "".join(c for c in stem if c.isalnum() or c in "-_")
    if not stem:
        stem = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
    return stem + ".md"


def _write_raw(filename: str, title: str, content: str) -> None:
    """写入raw文件。已存在则追加（并提示可能混主题）"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RAW_DIR / filename
    if filepath.exists():
        existing = filepath.read_text(encoding="utf-8")
        filepath.write_text(existing + "\n\n---\n\n" + content, encoding="utf-8")
        print(f"  📎 追加到: {filename}（文件已存在，两个主题将混在同一文件，建议检查）")
    else:
        filepath.write_text(f"# {title}\n\n{content}", encoding="utf-8")
        print(f"  ✅ 新建: {filename}")


def cmd_add_smart(text: str) -> list[str]:
    """智能添加：LLM自动拆分多主题，生成多个raw文件"""
    print("🧠 分析文本，识别主题 ...")

    response = llm_chat(SPLIT_PROMPT.format(text=text))

    # 解析JSON（兼容LLM输出可能带的markdown代码块）
    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[1] if "\n" in response else response[3:]
    response = response.removesuffix("```")
    response = response.strip()

    try:
        topics = json.loads(response)
    except json.JSONDecodeError:
        print("  ⚠️  主题拆分失败（LLM输出无法解析为JSON），整段作为单文件保存")
        topics = [{"title": "未分类知识", "content": text}]

    if not isinstance(topics, list):
        topics = [topics]

    created = []
    for topic in topics:
        title = (topic.get("title") or "").strip()
        content = topic.get("content", "")
        # 空标题兜底：用内容hash命名，避免生成空名文件或互相覆盖
        if not title:
            title = f"未命名-{hashlib.md5(content.encode('utf-8')).hexdigest()[:6]}"
        filename = _filename_from_title(title)
        _write_raw(filename, title, content)
        created.append(filename)

    print(f"\n📊 识别 {len(topics)} 个主题: {', '.join(t.get('title', '?') for t in topics)}")
    print("💡 运行 `python -m wiki_rag compile` 编译新条目")
    return created


def cmd_add_direct(text: str) -> str:
    """原样添加：不经过LLM拆分，整篇存为一个raw文件（文件名取首个一级标题）"""
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = (m.group(1).strip() if m else "") or f"未命名-{hashlib.md5(text.encode('utf-8')).hexdigest()[:6]}"
    filename = _filename_from_title(title)
    _write_raw(filename, title, text.strip())
    print("💡 运行 `python -m wiki_rag compile` 编译新条目")
    return filename


# ============================================================
# 索引构建器
# ============================================================

def build_wiki_index() -> Path:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    # index.md是本函数自己生成的目录页，不计入文章数
    wiki_files = sorted(f for f in WIKI_DIR.glob("*.md") if f.name != "index.md")

    lines = ["# 📚 个人Wiki知识库索引\n"]
    lines.append(f"> 共 {len(wiki_files)} 篇文章 | 自动生成\n")
    lines.append("## 文章目录\n")

    for wf in wiki_files:
        title = wf.stem.replace("-", " ").replace("_", " ").title()
        first_line = ""
        with open(wf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    first_line = line[:80]
                    break
        lines.append(f"- **{title}** — {first_line or '(无描述)'}  ")
        lines.append(f"  `wiki/{wf.name}`\n")

    INDEX_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📑 索引已更新: {len(wiki_files)} 篇文章")
    return INDEX_FILE


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[。！？!?\.])\s*", text) if s.strip()]


def _chunk_text(content: str) -> list[str]:
    """切分检索chunk：按段落，超过CHUNK_SIZE的段落按句子贪心打包"""
    chunks = []
    for para in (p.strip() for p in content.split("\n\n")):
        if len(para) < 20:
            continue
        if len(para) <= CHUNK_SIZE:
            chunks.append(para)
            continue
        buf = ""
        for sent in _split_sentences(para):
            while len(sent) > CHUNK_SIZE:  # 单句超长，硬切
                chunks.append(sent[:CHUNK_SIZE])
                sent = sent[CHUNK_SIZE:]
            if buf and len(buf) + len(sent) + 1 > CHUNK_SIZE:
                chunks.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf.strip():
            chunks.append(buf.strip())
    return chunks


def build_vector_index() -> dict[str, Any] | None:
    """从wiki文章构建向量索引（Ollama本地Embedding，文件级增量）"""
    print("  🔨 构建向量索引 ...")

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    wiki_files = [f for f in WIKI_DIR.glob("*.md") if f.name != "index.md"]

    if not wiki_files:
        print("  ⚠️  没有wiki文章，跳过索引构建")
        return None

    index_file = STORAGE_DIR / "vector_index.json"
    old_files = {}
    if index_file.exists():
        try:
            old = json.loads(index_file.read_text(encoding="utf-8"))
            if old.get("model") == EMBED_MODEL:
                old_files = old.get("files", {})
        except (json.JSONDecodeError, KeyError):
            pass  # 旧索引损坏，全量重建

    client = get_embed_client()
    files = {}
    reused, embedded = 0, 0

    for wf in wiki_files:
        h = file_hash(wf)
        cached = old_files.get(wf.name)
        if cached and cached.get("hash") == h:
            files[wf.name] = cached
            reused += 1
            continue

        title = wf.stem.replace("-", " ").replace("_", " ").title()
        texts = _chunk_text(wf.read_text(encoding="utf-8"))
        if not texts:
            continue
        try:
            # 批量嵌入：一个文件一次调用
            embeddings = client.embed(model=EMBED_MODEL, input=texts)["embeddings"]
        except Exception as e:
            print(f"  ⚠️  embedding失败 {wf.name}: {e}")
            continue

        files[wf.name] = {
            "hash": h,
            "chunks": [
                {"text": t, "embedding": emb, "metadata": {"filename": wf.name, "title": title, "chunk": i}}
                for i, (t, emb) in enumerate(zip(texts, embeddings, strict=True))
            ],
        }
        embedded += 1

    if not files:
        print("  ⚠️  没有有效的embedding")
        return None

    dim = len(next(iter(files.values()))["chunks"][0]["embedding"])
    index_data = {"model": EMBED_MODEL, "dim": dim, "files": files}
    index_file.write_text(json.dumps(index_data, ensure_ascii=False), encoding="utf-8")
    print(f"  ✅ 向量索引构建完成: {len(files)} 篇文章（新嵌入{embedded}, 复用{reused}）")
    return index_data


# ============================================================
# 查询引擎
# ============================================================

def _flatten_chunks(index_data: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for f in index_data["files"].values() for c in f["chunks"]]


def _extract_bigrams(text: str) -> set[str]:
    """提取字符二元组。中文没有空格分词，bigram是最轻量的匹配方案"""
    grams = set()
    for run in re.findall(r"[0-9a-z一-鿿]+", text.lower()):
        if len(run) == 1:
            grams.add(run)
        else:
            grams.update(run[i:i + 2] for i in range(len(run) - 1))
    return grams


def query(question: str) -> str:
    """查询Wiki知识库：向量检索（失败降级关键词bigram匹配）→ LLM回答"""
    index_file = STORAGE_DIR / "vector_index.json"
    if not index_file.exists():
        return "❌ 知识库为空，请先运行 compile"

    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    chunks = _flatten_chunks(index_data)
    if not chunks:
        return "❌ 知识库为空，请先运行 compile"

    top_chunks: list[tuple[float, dict[str, Any]]] = []
    try:
        q_emb = get_embed_client().embeddings(model=index_data["model"], prompt=question)["embedding"]
        # 全部chunk堆成矩阵，归一化后一次矩阵乘
        emb_matrix = np.array([c["embedding"] for c in chunks], dtype=np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 零向量不归一化，直接跳过
        emb_matrix = emb_matrix / norms
        q = np.array(q_emb, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = emb_matrix @ q
        top_chunks = [(float(sims[i]), chunks[i]) for i in np.argsort(-sims)[:TOP_K]]
    except Exception as e:
        print(f"  ⚠️  向量检索失败({e})，使用关键词匹配")
        q_grams = _extract_bigrams(question)
        scored = []
        for chunk in chunks:
            hits = len(q_grams & _extract_bigrams(chunk["text"]))
            if hits > 0:
                scored.append((hits, chunk))
        if not scored:
            return "❓ 知识库中没有找到与问题相关的内容"
        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [(float(s), c) for s, c in scored[:TOP_K]]

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
# 命令实现（CLI入口见 __main__.py）
# ============================================================

def cmd_compile(force: bool = False) -> None:
    print("🔨 编译 raw → wiki ...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = list(RAW_DIR.glob("*.md"))

    if not raw_files:
        print("  ⚠️  raw/ 目录为空")
        return

    compiled_count = 0
    meta = load_meta()
    for rf in raw_files:
        result = compile_raw_to_wiki(rf, force=force, meta=meta)
        if result is not None:
            compiled_count += 1

    print(f"\n📊 编译完成: {compiled_count} 篇新编译, {len(raw_files)-compiled_count} 篇无变化")
    build_wiki_index()
    build_vector_index()


def cmd_query(question: str) -> None:
    print(f"🔍 查询: {question}\n")
    answer = query(question)
    print(f"💡 回答:\n{answer}\n")
