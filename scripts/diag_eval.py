"""Замер поиска на перефразированных вопросах: что реально теряется.

Считаем recall@12 на уровне документа и пункта, а также сравниваем
варианты токенизации — чтобы видеть вклад алгоритма, а не догадываться.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submit"))

import solution  # noqa: E402

questions = json.loads(Path("data/diag_questions.json").read_text(encoding="utf-8"))
documents = solution.load_documents(Path("data/knowledge-base.zip"))
chunks = solution.chunk_documents(documents)


def naive_tokens(text: str) -> list[str]:
    low = text.lower().replace("ё", "е")
    return [w for w in re.sub(r"[^а-яa-z0-9\s]", " ", low).split() if len(w) > 1]


class NaiveRetriever(solution.Retriever):
    """Тот же ранжировщик, но без стемминга — как было изначально."""

    def __init__(self, items: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = items
        self.bm25 = BM25Okapi([naive_tokens(c["searchable"]) for c in items])

    def search(self, question: str, k: int = solution.TOP_K):
        scores = self.bm25.get_scores(naive_tokens(question))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])[:k]
        return [self.chunks[i] for i in order]


def evaluate(engine, label: str) -> None:
    doc_hits = clause_hits = 0
    by_type = defaultdict(lambda: [0, 0, 0])
    fails = []
    for item in questions:
        found = engine.search(item["question"], 12)
        doc_ok = any(c["doc"] == item["doc"] for c in found)
        clause_ok = any(c["doc"] == item["doc"] and c["clause"] == item["clause"] for c in found)
        doc_hits += doc_ok
        clause_hits += clause_ok
        kind = item.get("doc_type") or "other"
        by_type[kind][0] += doc_ok
        by_type[kind][1] += clause_ok
        by_type[kind][2] += 1
        if not clause_ok:
            fails.append((kind, item["question"][:62], item["doc"][:26], item["clause"],
                          found[0]["doc"][:24] if found else "-"))
    n = len(questions)
    print(f"{label:18} документ {doc_hits:2}/{n} = {100 * doc_hits / n:3.0f}%   "
          f"пункт {clause_hits:2}/{n} = {100 * clause_hits / n:3.0f}%")
    for kind, (dh, ch, cnt) in sorted(by_type.items()):
        print(f"    {kind:12} док {dh:2}/{cnt:2}  пункт {ch:2}/{cnt:2}")
    return fails


print(f"вопросов: {len(questions)}\n")
fails = evaluate(solution.Retriever(chunks), "стемминг")
evaluate(NaiveRetriever(chunks), "без стемминга")

print("\nпромахи на уровне пункта (стемминг):")
for kind, q, doc, clause, top in fails[:12]:
    print(f"  [{kind}] «{q}»\n      нужно: {doc} п.{clause}   топ: {top}")
