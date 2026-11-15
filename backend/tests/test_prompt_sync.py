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


# ── генерация блоков из backend/app ──────────────────────────────

import importlib.util  # noqa: E402


def load_sync_module():
    path = REPO / "scripts" / "sync_submit.py"
    spec = importlib.util.spec_from_file_location("sync_submit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_blocks_match_backend() -> None:
    """Блоки parser и textsel в сабмите должны совпадать с backend.

    Правка в backend/app без запуска scripts/sync_submit.py — дрейф,
    из-за которого баг чинится в одном месте и живёт в другом.
    """
    sync = load_sync_module()
    source = (REPO / "submit" / "solution.py").read_text(encoding="utf-8")
    for name, (relative, names) in sync.BLOCKS.items():
        expected = sync.module_body(relative, names)
        first_line = expected.strip().splitlines()[0]
        assert first_line in source, f"блок {name} не найден в solution.py"
        assert expected.strip() in source, (
            f"блок {name} разошёлся с {relative} — запустите python scripts/sync_submit.py"
        )


def test_no_duplicate_parser_in_submit() -> None:
    """Копии парсера в сабмите быть не должно: только сгенерированный блок."""
    source = (REPO / "submit" / "solution.py").read_text(encoding="utf-8")
    assert source.count("def load_documents(") == 1
    assert source.count("def read_docx(") == 1
    assert source.count("def has_amount(") == 1
