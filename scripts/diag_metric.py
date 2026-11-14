"""Какая метрика даёт ~17%? Считаем разные варианты на 18 демо-вопросах.

Нужно понять, что именно измеряет платформа: точное совпадение, F1 по словам
или совпадение ключевых фактов. Иначе оптимизируем вслепую.
"""
from __future__ import annotations

import csv
import difflib
import re
from pathlib import Path

GOLD = Path("data/demo_output.csv")
PRED = Path("submission_real.csv")


def norm(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return norm(text).split()


def token_f1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return 0.0
    common = {}
    for token in p:
        common[token] = min(p.count(token), g.count(token))
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def numbers(text: str) -> set[str]:
    return {re.sub(r"\s+", "", n) for n in re.findall(r"\d[\d\s]{2,}", text)}


gold = {row["id"]: row["answer"] for row in csv.DictReader(GOLD.open(encoding="utf-8-sig"))}
pred = {row["id"]: row["answer"] for row in csv.DictReader(PRED.open(encoding="utf-8-sig"))}
ids = sorted(set(gold) & set(pred), key=lambda x: int(x))

exact = sum(norm(pred[i]) == norm(gold[i]) for i in ids)
f1 = sum(token_f1(pred[i], gold[i]) for i in ids) / len(ids)
ratio = sum(difflib.SequenceMatcher(None, norm(pred[i]), norm(gold[i])).ratio() for i in ids) / len(ids)
num_hit = sum(bool(numbers(gold[i])) and numbers(gold[i]) <= numbers(pred[i]) for i in ids)
num_total = sum(bool(numbers(gold[i])) for i in ids)

print(f"вопросов: {len(ids)}\n")
print(f"точное совпадение строк : {100 * exact / len(ids):5.1f}%")
print(f"F1 по словам (SQuAD)    : {100 * f1:5.1f}%")
print(f"сходство строк (difflib): {100 * ratio:5.1f}%")
print(f"все числа попали в ответ: {100 * num_hit / max(num_total, 1):5.1f}%  ({num_hit}/{num_total})")
print(f"\nплатформа показывает    :  17.2%")

print("\nпокрытие чисел эталона (по одному):")
for i in ids:
    g, p = numbers(gold[i]), numbers(pred[i])
    if not g:
        continue
    missing = sorted(g - p)
    print(f"  {i:>2}. есть {len(g) - len(missing)}/{len(g)}"
          + (f"  потеряно: {missing[:4]}" if missing else "  ✅")
          + f"   | эталон: {gold[i][:54]}")
