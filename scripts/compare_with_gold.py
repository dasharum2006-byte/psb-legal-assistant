"""Diff ответов на demo-вопросах с data/demo_output.csv.

Показывает не только «ответ не равен эталону» (LLM всегда сформулирует
иначе), а смысловые расхождения:
  * ушли в отказ вместо ответа или наоборот;
  * сослались не на тот пункт/статью;
  * потеряли ключевые суммы/проценты.

Запуск: python scripts/compare_with_gold.py [submission.csv]
"""
from __future__ import annotations

import csv
import re
import sys
from difflib import SequenceMatcher

DOC_NUMBER_RE = re.compile(
    r"(?<![а-яa-z0-9])([а-я]{1,4})[-‑](\d{1,4})(?:/(\d{4}))?", re.I
)
REFUSAL_MARKERS = (
    "нет документов",
    "нет данных",
    "не могу ответить",
    "не отвечаю",
    "не хватает",
    "ответить по документам не могу",
    "в базе знаний нет",
    "не содержится в базе",
)


def refs(text: str) -> set[str]:
    """Ссылки: пункты, статьи, номера документов, годы редакций."""
    found: set[str] = set()
    for match in re.finditer(r"п\.?\s*(\d+(?:\.\d+)+)", text):
        found.add(f"п.{match.group(1)}")
    for match in re.finditer(r"(?:ст\.|стать[а-яё]*)\s*(\d+(?:\.\d+)?)", text, re.I):
        found.add(f"ст.{match.group(1)}")
    for match in re.finditer(r"ред\.?\s*(20\d{2})", text):
        found.add(f"ред.{match.group(1)}")
    for match in DOC_NUMBER_RE.finditer(text):
        letter, digits, year = match.groups()
        found.add(f"№{letter.lower()}-{digits}" + (f"/{year}" if year else ""))
    return found


def amounts(text: str) -> set[str]:
    """Крупные числа, проценты и суммы в рублях — то, что нельзя терять.

    Не берём даты («10.01.2023») и номера пунктов («3.1»): иначе diff
    показывает ложные расхождения по годам и пунктам.
    """
    found: set[str] = set()
    # 450 000, 10 000 000, 15000000 — но не части даты через точки
    for match in re.finditer(r"(?<!\d)(?:\d{1,3}(?:[ \u00a0]\d{3})+|\d{4,})(?!\d)", text):
        clean = re.sub(r"\D", "", match.group(0))
        if not re.fullmatch(r"20\d{2}", clean):
            found.add(clean)
    for match in re.finditer(r"(\d{1,3}(?:[ \u00a0]\d{3})*)\s*(?:руб|₽)", text, re.I):
        clean = re.sub(r"\D", "", match.group(1))
        if len(clean) >= 3:
            found.add(clean)
    for match in re.finditer(r"(\d+(?:[.,]\d+)?)\s*%", text):
        found.add(match.group(1).replace(",", ".") + "%")
    return found


def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def words(text: str) -> set[str]:
    return set(re.findall(r"[а-яa-z]{5,}", text.lower().replace("ё", "е")))


def short(text: str, limit: int = 110) -> str:
    return " ".join(text.split())[:limit]


def verdict_for(gold: str, ours: str) -> tuple[str, list[str]]:
    g_ref, o_ref = refs(gold), refs(ours)
    g_num, o_num = amounts(gold), amounts(ours)

    if is_refusal(gold):
        return ("REFUSE_OK", []) if is_refusal(ours) else ("REFUSE_BAD", ["ответ вместо отказа"])
    if is_refusal(ours):
        return "MISSED", ["ушел в отказ вместо ответа"]

    details: list[str] = []
    missing_refs = sorted(g_ref - o_ref)
    missing_nums = sorted(g_num - o_num)
    if missing_refs:
        details.append("нет ссылок: " + ", ".join(missing_refs[:5]))
    if missing_nums:
        details.append("нет чисел: " + ", ".join(missing_nums[:5]))

    if g_ref:
        if not (g_ref & o_ref):
            return "WRONG", details or ["ни одна ссылка не совпала"]
        if missing_refs:
            return "PARTIAL", details
        if missing_nums:
            return "PARTIAL", details
        return "OK", []

    if g_num:
        if g_num <= o_num:
            return "OK", []
        if g_num & o_num:
            return "PARTIAL", details
        return "WRONG", details

    ratio = SequenceMatcher(None, " ".join(sorted(words(gold))), " ".join(sorted(words(ours)))).ratio()
    return ("OK" if ratio >= 0.55 else "PARTIAL"), details


def main() -> None:
    ours_path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    gold = {
        row["id"]: row["answer"]
        for row in csv.DictReader(open("data/demo_output.csv", encoding="utf-8-sig"))
    }
    ours = {
        row["id"]: row["answer"]
        for row in csv.DictReader(open(ours_path, encoding="utf-8-sig"))
    }

    stats = {"OK": 0, "PARTIAL": 0, "WRONG": 0, "MISSED": 0, "REFUSE_OK": 0, "REFUSE_BAD": 0}
    problems = 0
    for qid in sorted(gold, key=int):
        verdict, details = verdict_for(gold[qid], ours.get(qid, ""))
        stats[verdict] += 1
        if verdict not in ("OK", "REFUSE_OK"):
            problems += 1
        note = (" | " + "; ".join(details)) if details else ""
        print(f"{qid:>2} {verdict:10}{note}")
        print(f"   эталон: {short(gold[qid])}")
        print(f"   наше:   {short(ours.get(qid, ''))}")

    print()
    print("итог:", stats)
    print(f"расхождений: {problems}/18")


if __name__ == "__main__":
    main()
