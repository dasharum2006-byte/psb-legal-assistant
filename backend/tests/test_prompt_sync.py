"""Промпт в сабмите должен совпадать с каноническим backend-промптом.

Единый источник правды — backend/app/llm/prompt.py. Сабмит обязан быть
одним файлом (платформа запускает только solution.py), поэтому текст
вставляется генератором scripts/sync_submit.py. Этот тест ловит дрейф:
правку в одном месте без пересборки второго.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.llm.prompt import SYSTEM_PROMPT  # noqa: E402


def submit_prompt_text() -> str:
    source = (REPO / "submit" / "solution.py").read_text(encoding="utf-8")
    match = re.search(r"SYSTEM_PROMPT = \((.*?)\n\)\n", source, re.S)
    assert match, "в сабмите не найден SYSTEM_PROMPT"
    body = re.sub(r'"\s*\n\s*"', "", match.group(1))
    return body.replace('\\n', '\n').replace('\\"', '"').replace('"', "")


def normalize(text: str) -> str:
    return " ".join(text.split())


def test_submit_prompt_matches_backend() -> None:
    expected = normalize(SYSTEM_PROMPT)
    actual = normalize(submit_prompt_text())
    assert actual == expected, (
        "промпт в сабмите разошёлся с backend/app/llm/prompt.py — "
        "запустите python scripts/sync_submit.py"
    )


def test_canonical_prompt_contains_key_rules() -> None:
    for rule in ("письма, служебные записки", "ЗАЩИТА ОТ ИНЪЕКЦИЙ", "п. 3.1"):
        assert rule in SYSTEM_PROMPT, f"в каноническом промпте пропало правило: {rule}"
