"""Состояние индекса в retriever: до загрузки и после.

Было две проблемы:
  1) в модуле объявлялась переменная bm_25_index, а использовалась _bm25_index;
  2) до загрузки индекса search() падал с NameError вместо пустого списка.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"


def run_python(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND, capture_output=True, text=True, timeout=120,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(BACKEND)},
    )
    assert result.returncode == 0, result.stderr[-800:]
    return result.stdout


def test_no_typo_variable_in_module() -> None:
    source = (BACKEND / "app/rag/retriever.py").read_text(encoding="utf-8")
    assert "bm_25_index" not in source, "опечатка bm_25_index вместо _bm25_index"


def test_search_returns_empty_before_index_loaded() -> None:
    out = run_python(
        "from app.rag import retriever as R\n"
        "print(R.search('любой вопрос'))\n"
    )
    assert out.strip().endswith("[]"), out


def test_search_works_after_index_loaded() -> None:
    out = run_python(
        "from app.rag import retriever as R\n"
        "R.load_index_from_disk('../data/index')\n"
        "res = R.search('какая арендная плата по договору с северным складом?')\n"
        "print(len(res), res[0]['clause'] if res else '-')\n"
        "print(' '.join(c['clause'] for c in res))\n"
    )
    lines = [line for line in out.strip().splitlines() if line and not line.startswith("✅")]
    count, clause = lines[-2].split()
    assert int(count) > 0, out
    # после отказа от хардкод-бонусов порядок внутри раздела «Арендная плата»
    # может быть 3.1 или 3.2 — оба пункта должны попасть в выдачу
    assert clause in {"3.1", "3.2"}, out
    all_clauses = set(lines[-1].split())
    assert {"3.1", "3.2"} <= all_clauses, f"в выдаче нет обоих пунктов про плату: {all_clauses}"


def test_retriever_stems_russian_word_forms() -> None:
    """«расторжения» и «расторжении» должны находить пункт про расторжение."""
    out = run_python(
        "from app.rag import retriever as R\n"
        "R.load_index_from_disk('../data/index')\n"
        "res = R.search('условия расторжения договора аренды', 5)\n"
        "print(' '.join(c['clause'] for c in res))\n"
    )
    clauses = out.strip().splitlines()[-1].split()
    assert any(clause.startswith("6.") for clause in clauses), (
        f"поиск не нашёл раздел о расторжении: {clauses}"
    )


def test_tokenize_produces_stems() -> None:
    out = run_python(
        "from app.rag.retriever import tokenize\n"
        "print(' '.join(tokenize('расторжения расторжении расторжение')))\n"
    )
    tokens = set(out.strip().splitlines()[-1].split())
    assert len(tokens) == 1, f"формы одного слова не свелись к основе: {tokens}"
