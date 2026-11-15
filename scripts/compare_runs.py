"""Сравнить два прогона сабмита по метрикам против эталона."""
from __future__ import annotations

import csv
import difflib
import re
import sys
from pathlib import Path


def norm(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    return " ".join(text.split())


def f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return 0.0
    overlap = sum(min(p.count(t), g.count(t)) for t in set(p))
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def numbers(text: str) -> set[str]:
    return {re.sub(r"\s+", "", n) for n in re.findall(r"\d[\d\s]{2,}", text)}


gold = {r["id"]: r["answer"] for r in csv.DictReader(open("data/demo_output.csv", encoding="utf-8-sig"))}
runs = {name: {r["id"]: r["answer"] for r in csv.DictReader(open(name, encoding="utf-8-sig"))}
        for name in sys.argv[1:]}

print(f"{'прогон':26} {'F1':>6} {'сходство':>9} {'числа':>7} {'длина':>6}")
for name, pred in runs.items():
    ids = sorted(set(gold) & set(pred), key=int)
    f1_avg = sum(f1(pred[i], gold[i]) for i in ids) / len(ids)
    sim = sum(difflib.SequenceMatcher(None, norm(pred[i]), norm(gold[i])).ratio() for i in ids) / len(ids)
    num_ok = sum(bool(numbers(gold[i])) and numbers(gold[i]) <= numbers(pred[i]) for i in ids)
    num_all = sum(bool(numbers(gold[i])) for i in ids)
    avg_len = sum(len(pred[i]) for i in ids) / len(ids)
    print(f"{name:26} {100 * f1_avg:5.1f}% {100 * sim:8.1f}% {num_ok:3}/{num_all:<3} {avg_len:6.0f}")
