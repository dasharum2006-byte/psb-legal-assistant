"""Диагностика поиска на всей базе, а не на 18 демо-вопросах.

Из каждого чанка делаем «вопрос» (первые значимые слова текста) и смотрим,
находит ли поисковик этот же чанк. Так видно, где именно теряется документ:
законы, договоры, письма.
"""
from __future__ import annotations

import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submit"))

import solution  # noqa: E402

random.seed(42)
KB = Path("data/knowledge-base.zip")
documents = solution.load_documents(KB)
chunks = solution.chunk_documents(documents)
retriever = solution.Retriever(chunks)

STOP = {"который", "которая", "данного", "настоящего", "договора", "документа", "также", "если"}


def make_query(chunk: dict) -> str:
    words = [w for w in re.findall(r"[а-яa-z]{5,}", chunk["text"].lower()) if w not in STOP]
    if len(words) < 4:
        words = re.findall(r"[а-яa-z]{4,}", chunk["text"].lower())
    picked = random.sample(words, min(6, len(words)))
    return " ".join(picked)


sample = random.sample(chunks, min(150, len(chunks)))
by_type = defaultdict(lambda: [0, 0])
misses = defaultdict(list)
for chunk in sample:
    query = make_query(chunk)
    found = retriever.search(query, 12)
    hit = any(c["doc"] == chunk["doc"] and c["clause"] == chunk["clause"] for c in found)
    kind = chunk.get("doc_type") or "other"
    by_type[kind][0] += hit
    by_type[kind][1] += 1
    if not hit and len(misses[kind]) < 2:
        misses[kind].append((query[:60], chunk["doc"][:34], chunk["clause"], found[0]["doc"][:28] if found else "-"))

total_hit = sum(v[0] for v in by_type.values())
total = sum(v[1] for v in by_type.values())
print(f"recall@12 по всей базе: {total_hit}/{total} = {100 * total_hit / total:.0f}%\n")
for kind, (hit, n) in sorted(by_type.items(), key=lambda kv: kv[1][0] / kv[1][1]):
    print(f"  {kind:12} {hit:3}/{n:3} = {100 * hit / n:3.0f}%")
print("\nпримеры промахов:")
for kind, items in misses.items():
    for query, doc, clause, top in items[:2]:
        print(f"  [{kind}] «{query}» → нужно {doc} {clause}, топ: {top}")
