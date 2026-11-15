"""Соседние чанки: полнота без шума.

Замер на демо-наборе (18 вопросов, живая модель):
  без соседей   F1 47.6  сходство 22.1  числа 7/15
  топ-1 сосед   F1 47.6  сходство 23.9  числа 8/15
  топ-2 соседа  F1 47.5  сходство 25.2  числа 7/15
F1 в пределах шума, сходство лучше, изредка сосед приносит лишнюю цифру —
поэтому по умолчанию подклеиваем только к топ-1.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.rag import retriever  # noqa: E402
from app.rag.orchestrator import NEIGHBOUR_TOP, add_neighbours  # noqa: E402


def run_python(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_next_chunk_stays_in_same_document() -> None:
    out = run_python(
        "from app.rag import retriever as R\n"
        "R.load_index_from_disk('../data/index')\n"
        "res = R.search('арендная плата по договору с северным складом', 3)\n"
        "nxt = R.next_chunk(res[0], 1)\n"
        "print(len(nxt), nxt[0]['doc_id'] == res[0]['doc_id'] if nxt else 'none')\n"
    )
    count, same = out.splitlines()[-1].split()
    assert int(count) >= 1
    assert same == "True"


def test_add_neighbours_respects_budget() -> None:
    out = run_python(
        "from app.rag import retriever as R\n"
        "from app.rag.orchestrator import add_neighbours, _adapt\n"
        "R.load_index_from_disk('../data/index')\n"
        "res = R.search('арендная плата по договору с северным складом', 3)\n"
        "picked = [_adapt(c) for c in res[:1]]\n"
        "tiny = add_neighbours(picked, R, budget=len(picked[0]['text']))\n"
        "print(len(picked), len(tiny))\n"
    )
    before, after = out.splitlines()[-1].split()
    assert before == after, "сосед не должен влезать при исчерпанном бюджете"


def test_add_neighbours_does_not_duplicate_clauses() -> None:
    out = run_python(
        "from app.rag import retriever as R\n"
        "from app.rag.orchestrator import add_neighbours, _adapt\n"
        "R.load_index_from_disk('../data/index')\n"
        "res = R.search('арендная плата', 5)\n"
        "picked = [_adapt(c) for c in res[:2]]\n"
        "added = add_neighbours(picked, R)\n"
        "keys = [(c['doc_id'], c['clause']) for c in added]\n"
        "print(len(keys) == len(set(keys)))\n"
    )
    assert out.splitlines()[-1].strip() == "True"


def test_default_is_conservative() -> None:
    assert NEIGHBOUR_TOP == 1, "по умолчанию сосед только к лучшему чанку"
