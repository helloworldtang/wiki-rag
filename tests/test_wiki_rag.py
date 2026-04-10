"""Wiki RAG 系统测试"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 项目根目录
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "raw"
WIKI_DIR = BASE_DIR / "wiki"
STORAGE_DIR = BASE_DIR / "storage"
META_FILE = STORAGE_DIR / "meta.json"


class TestFileHash:
    """文件哈希计算"""

    def test_hash_consistency(self, tmp_path):
        from src.wiki_rag import file_hash
        f = tmp_path / "test.md"
        f.write_text("hello world")
        h1 = file_hash(f)
        h2 = file_hash(f)
        assert h1 == h2
        assert len(h1) == 32  # MD5 hex长度

    def test_hash_changes_with_content(self, tmp_path):
        from src.wiki_rag import file_hash
        f = tmp_path / "test.md"
        f.write_text("hello")
        h1 = file_hash(f)
        f.write_text("world")
        h2 = file_hash(f)
        assert h1 != h2


class TestMetaData:
    """元数据管理"""

    def test_load_empty_meta(self, tmp_path, monkeypatch):
        from src.wiki_rag import load_meta
        monkeypatch.setattr("src.wiki_rag.META_FILE", tmp_path / "meta.json")
        meta = load_meta()
        assert "compiled" in meta
        assert meta["compiled"] == {}

    def test_save_and_load_meta(self, tmp_path, monkeypatch):
        from src.wiki_rag import save_meta, load_meta
        monkeypatch.setattr("src.wiki_rag.META_FILE", tmp_path / "meta.json")
        meta = {"compiled": {"test.md": {"hash": "abc123", "wiki_file": "test.md"}}}
        save_meta(meta)
        loaded = load_meta()
        assert loaded == meta


class TestCompile:
    """编译逻辑"""

    def test_compile_skips_unchanged_file(self, tmp_path, monkeypatch):
        from src.wiki_rag import compile_raw_to_wiki

        # 准备
        raw_file = tmp_path / "raw" / "test.md"
        raw_file.parent.mkdir()
        raw_file.write_text("# Test\ncontent here")

        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()

        # 模拟已编译
        from src.wiki_rag import file_hash
        meta_file = tmp_path / "meta.json"
        meta = {"compiled": {"test.md": {"hash": file_hash(raw_file), "wiki_file": "test.md"}}}
        meta_file.write_text(json.dumps(meta))

        # 创建已存在的wiki文件
        (wiki_dir / "test.md").write_text("compiled content")

        monkeypatch.setattr("src.wiki_rag.META_FILE", meta_file)
        monkeypatch.setattr("src.wiki_rag.WIKI_DIR", wiki_dir)

        result = compile_raw_to_wiki(raw_file)
        assert result is None  # 跳过

    def test_compile_calls_llm_for_new_file(self, tmp_path, monkeypatch):
        from src.wiki_rag import compile_raw_to_wiki
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr("src.wiki_rag.DEEPSEEK_API_KEY", "test-key")

        raw_file = tmp_path / "raw" / "new-topic.md"
        raw_file.parent.mkdir()
        raw_file.write_text("# New Topic\nSome content")

        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()

        meta_file = tmp_path / "meta.json"
        meta_file.write_text('{"compiled": {}}')

        monkeypatch.setattr("src.wiki_rag.META_FILE", meta_file)
        monkeypatch.setattr("src.wiki_rag.WIKI_DIR", wiki_dir)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="# New Topic\nCompiled content\n\n## 总结\nDone"))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock(return_value=mock_client)

        with patch("openai.OpenAI", mock_openai):
            result = compile_raw_to_wiki(raw_file)

        assert result is not None
        assert result.name == "new-topic.md"
        assert (wiki_dir / "new-topic.md").exists()
        assert "Compiled content" in (wiki_dir / "new-topic.md").read_text()

    def test_compile_force_recompiles(self, tmp_path, monkeypatch):
        from src.wiki_rag import compile_raw_to_wiki, file_hash
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr("src.wiki_rag.DEEPSEEK_API_KEY", "test-key")

        raw_file = tmp_path / "raw" / "test.md"
        raw_file.parent.mkdir()
        raw_file.write_text("# Test\ncontent")

        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "test.md").write_text("old content")

        meta_file = tmp_path / "meta.json"
        meta = {"compiled": {"test.md": {"hash": file_hash(raw_file), "wiki_file": "test.md"}}}
        meta_file.write_text(json.dumps(meta))

        monkeypatch.setattr("src.wiki_rag.META_FILE", meta_file)
        monkeypatch.setattr("src.wiki_rag.WIKI_DIR", wiki_dir)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="# Test\nNew compiled content"))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock(return_value=mock_client)

        with patch("openai.OpenAI", mock_openai):
            result = compile_raw_to_wiki(raw_file, force=True)

        assert result is not None
        assert "New compiled" in (wiki_dir / "test.md").read_text()


class TestBuildIndex:
    """索引构建"""

    def test_build_wiki_index(self, tmp_path, monkeypatch):
        from src.wiki_rag import build_wiki_index

        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "python-decorators.md").write_text("# Python装饰器\n装饰器是...")
        (wiki_dir / "docker-network.md").write_text("# Docker网络\n网络模式...")

        index_file = wiki_dir / "index.md"

        monkeypatch.setattr("src.wiki_rag.WIKI_DIR", wiki_dir)
        monkeypatch.setattr("src.wiki_rag.INDEX_FILE", index_file)

        result = build_wiki_index()
        assert result == index_file
        assert index_file.exists()
        content = index_file.read_text()
        assert "python-decorators" in content
        assert "docker-network" in content
        assert "2 篇文章" in content

    def test_build_wiki_index_empty(self, tmp_path, monkeypatch):
        from src.wiki_rag import build_wiki_index

        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        index_file = wiki_dir / "index.md"

        monkeypatch.setattr("src.wiki_rag.WIKI_DIR", wiki_dir)
        monkeypatch.setattr("src.wiki_rag.INDEX_FILE", index_file)

        build_wiki_index()
        content = index_file.read_text()
        assert "0 篇文章" in content


class TestQuery:
    """查询逻辑"""

    def test_query_with_empty_index(self, tmp_path, monkeypatch):
        from src.wiki_rag import query
        monkeypatch.setattr("src.wiki_rag.STORAGE_DIR", tmp_path / "nonexistent")
        result = query("test question")
        assert "知识库为空" in result


class TestAdd:
    """添加条目"""

    def test_add_smart_creates_files(self, tmp_path, monkeypatch):
        from src.wiki_rag import cmd_add_smart
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr("src.wiki_rag.DEEPSEEK_API_KEY", "test-key")

        raw_dir = tmp_path / "raw"
        monkeypatch.setattr("src.wiki_rag.RAW_DIR", raw_dir)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(
            content='[{"title": "主题A", "content": "内容A"}, {"title": "主题B", "content": "内容B"}]'
        ))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock(return_value=mock_client)

        with patch("openai.OpenAI", mock_openai):
            created = cmd_add_smart("一段包含多个主题的文本")

        assert len(created) == 2
        assert (raw_dir / "主题a.md").exists() or len(list(raw_dir.glob("*.md"))) == 2


class TestRawDataExists:
    """验证测试数据完整"""

    def test_raw_files_exist(self):
        raw_files = list(RAW_DIR.glob("*.md"))
        assert len(raw_files) >= 5, f"至少需要5个raw文件，当前{len(raw_files)}个"

    def test_each_raw_has_content(self):
        for rf in RAW_DIR.glob("*.md"):
            content = rf.read_text()
            assert len(content) > 50, f"{rf.name} 内容太少"
            assert content.startswith("#"), f"{rf.name} 应以 # 开头"
