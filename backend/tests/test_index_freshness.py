"""Индекс должен быть собран тем же кодом нарезки, что сейчас в репозитории.

Если чанкер поменяли и забыли пересобрать — тест падает. Ровно та ошибка,
из-за которой прототип работал на чанках суточной давности.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data/index/meta.json"


def current_fingerprint() -> str:
    fingerprint = ""
    for name in ("chunker.py", "parser.py"):
        code = ROOT / "backend/app/rag" / name
        fingerprint += hashlib.sha256(code.read_bytes()).hexdigest()
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def test_index_matches_current_code() -> None:
    if not META.exists():
        pytest.skip("индекс не собран — запустите python scripts/build_index.py")

    meta = json.loads(META.read_text(encoding="utf-8"))
    assert meta["code_fingerprint"] == current_fingerprint(), (
        "код нарезки изменился после сборки индекса: "
        "пересоберите его командой python scripts/build_index.py"
    )


def test_index_has_chunks_for_all_documents() -> None:
    if not META.exists():
        pytest.skip("индекс не собран")

    meta = json.loads(META.read_text(encoding="utf-8"))
    assert meta["documents"] >= 21, "в индексе не все документы базы знаний"
    assert meta["chunks"] > 200, "подозрительно мало чанков"
