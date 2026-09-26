"""CLI入口：python -m wiki_rag <command>"""

import argparse
import sys
from pathlib import Path

from . import cmd_add_direct, cmd_add_smart, cmd_compile, cmd_query


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="wiki_rag",
        description="个人Wiki知识库RAG：add收集素材 → compile编译+索引 → query问答",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_compile = sub.add_parser("compile", help="编译 raw/ → wiki/ 并构建索引")
    p_compile.add_argument("--force", action="store_true", help="忽略缓存，全部重新编译")

    p_query = sub.add_parser("query", help="向知识库提问")
    p_query.add_argument("question", help="问题内容")

    p_add = sub.add_parser("add", help="添加知识（默认LLM自动拆分多主题）")
    p_add.add_argument("input", help="Markdown文件路径、'-'（读stdin），或配合 --text 时的文本内容")
    p_add.add_argument("--text", action="store_true", help="把 INPUT 当作文本内容而非文件路径")
    p_add.add_argument("--no-split", action="store_true", help="跳过LLM拆分，整篇原样存入 raw/")

    args = parser.parse_args(argv)

    if args.command == "compile":
        cmd_compile(force=args.force)
    elif args.command == "query":
        cmd_query(args.question)
    elif args.command == "add":
        text = args.input
        if not args.text:
            if text == "-":  # 管道输入: cat note.md | python -m wiki_rag add -
                text = sys.stdin.read()
            else:
                path = Path(text)
                if not path.exists():
                    sys.exit(f"❌ 文件不存在: {path}")
                text = path.read_text(encoding="utf-8")
        if not text.strip():
            sys.exit("❌ 内容为空")

        if args.no_split:
            cmd_add_direct(text)
        else:
            cmd_add_smart(text)


if __name__ == "__main__":
    main()
