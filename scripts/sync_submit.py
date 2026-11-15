"""Собрать solution.py из канонических модулей backend/app.

Платформа получает только solution.py, поэтому импортировать backend из
сабмита нельзя: канонический код вставляется генерацией между маркерами
GENERATED. Правки делаются в backend/app, затем запускается этот скрипт.
Тест backend/tests/test_prompt_sync.py падает, если генерацию забыли.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
TARGET = ROOT / "submit" / "solution.py"

sys.path.insert(0, str(BACKEND))

BLOCKS = {
    "parser": ("app/rag/parser.py", None),
    "textsel": ("app/rag/textsel.py", None),
}


def module_body(relative: str, names: list[str] | None = None) -> str:
    """Исходник модуля без докстринга, __future__ и блока __main__."""
    path = BACKEND / relative
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    if names:
        keep, lines = [], source.splitlines()
        for node in tree.body:
            name = getattr(node, "name", None)
            if name in names or isinstance(node, (ast.Import, ast.ImportFrom)):
                keep.append("\n".join(lines[node.lineno - 1: node.end_lineno]))
        return "\n\n".join(keep)

    lines = source.splitlines()
    drop = set()
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            drop.update(range(node.lineno, node.end_lineno + 1))
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            drop.update(range(node.lineno, node.end_lineno + 1))
        elif isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
            drop.update(range(node.lineno, node.end_lineno + 1))
    return "\n".join(line for number, line in enumerate(lines, 1) if number not in drop).strip()


def replace_block(source: str, name: str, body: str) -> str:
    start_marker = f"# >>> GENERATED:{name}"
    end_marker = f"# <<< GENERATED:{name}"
    start = source.index(start_marker)
    end = source.index(end_marker) + len(end_marker)
    return (source[:start]
            + f"{start_marker}\n{body}\n{end_marker}"
            + source[end:])


def render_prompt(prompt: str) -> str:
    escaped = prompt.replace("\\", "\\\\").replace('"', '\\"')
    lines = escaped.split("\n")
    parts = [f'    "{line}\\n"\n' for line in lines[:-1]]
    parts.append(f'    "{lines[-1]}"\n')
    return "SYSTEM_PROMPT = (\n" + "".join(parts) + "\n)\n"


def main() -> None:
    from app.llm.prompt import SYSTEM_PROMPT

    source = TARGET.read_text(encoding="utf-8")
    original = source

    for name, (relative, names) in BLOCKS.items():
        source = replace_block(source, name, module_body(relative, names))

    start = source.index("SYSTEM_PROMPT = (\n")
    end = source.index("\n)\n", start) + len("\n)\n")
    source = source[:start] + render_prompt(SYSTEM_PROMPT) + source[end:]

    if source == original:
        print("solution.py уже совпадает с backend — правок не нужно")
        return
    TARGET.write_text(source, encoding="utf-8")
    print(f"solution.py обновлён из backend: блоки {', '.join(BLOCKS)}, SYSTEM_PROMPT")


if __name__ == "__main__":
    main()
