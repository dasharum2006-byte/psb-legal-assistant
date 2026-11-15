"""Сгенерировать SYSTEM_PROMPT в сабмите из канонического backend-промпта.

Единый источник правды — backend/app/llm/prompt.py. Сабмит должен быть
одним файлом (платформа получает только solution.py), поэтому импортировать
backend он не может: текст вставляется генерацией. Дрейф ловит
backend/tests/test_prompt_sync.py.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.llm.prompt import SYSTEM_PROMPT  # noqa: E402

TARGET = ROOT / "submit" / "solution.py"
START = "SYSTEM_PROMPT = (\n"
END = "\n)\n"


def render(prompt: str) -> str:
    """Собрать python-литерал: каждая строка отдельной склеиваемой строкой."""
    escaped = prompt.replace("\\", "\\\\").replace('"', '\\"')
    lines = escaped.split("\n")
    parts = [f'    "{line}\\n"\n' for line in lines[:-1]]
    parts.append(f'    "{lines[-1]}"\n')
    return START + "".join(parts) + END


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    start = source.index(START)
    end = source.index(END, start) + len(END)
    updated = source[:start] + render(SYSTEM_PROMPT) + source[end:]
    if updated == source:
        print("промпт уже совпадает с backend — правок не нужно")
        return
    TARGET.write_text(updated, encoding="utf-8")
    print(f"SYSTEM_PROMPT обновлён в {TARGET.relative_to(ROOT)} "
          f"({len(SYSTEM_PROMPT)} символов из backend)")


if __name__ == "__main__":
    main()
