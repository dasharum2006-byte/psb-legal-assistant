"""Head-to-head сравнение двух вариантов ответов локальным LLM-судьёй.

В отличие от ab_judge.py (наши ответы против baseline организаторов), здесь
сравниваются два наших варианта: например, чанкер сабмита против чанкера
бэкенда. Судья вызывается дважды на вопрос (A/B и B/A), чтобы убрать
позиционное смещение.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import urllib.request
from pathlib import Path

BASE = (os.getenv("LLM_BASE_URL") or "").rstrip("/")
KEY = os.getenv("LLM_API_KEY") or ""
MODEL = os.getenv("LLM_MODEL") or "deepseek/deepseek-v4.1-flash"
DATA = Path(os.getenv("DATA_DIR", "data"))

SYSTEM = (
    "Ты строгий юридический эксперт. Сравни два ответа на вопрос с эталоном. "
    "Ответь ровно одним словом: A, B или TIE. "
    "A — если ответ A точнее и полнее; B — если ответ B; TIE — если равноценны."
)


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def chat(user: str) -> str:
    data = post("/chat/completions", {
        "model": MODEL, "temperature": 0.0, "max_tokens": 800,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    })
    return (data["choices"][0]["message"].get("content") or "").strip()


def verdict(question: str, golden: str, a: str, b: str) -> str:
    user = (f"ВОПРОС: {question}\n\nЭТАЛОН: {golden}\n\nОТВЕТ A: {a}\n\nОТВЕТ B: {b}\n\n"
            "Кто ближе к эталону? Ответь: A, B или TIE.")
    answer = chat(user).upper()
    for token in ("TIE", "A", "B"):
        if token in answer:
            return token
    return "TIE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    questions = {r["id"]: r["question"] if "question" in r else r.get("text", "")
                 for r in csv.DictReader(open(DATA / "demo_questions.csv", encoding="utf-8-sig"))}
    golden = {r["id"]: r["answer"] for r in csv.DictReader(open(DATA / "demo_output.csv", encoding="utf-8-sig"))}
    a = {r["id"]: r["answer"] for r in csv.DictReader(open(args.file_a, encoding="utf-8-sig"))}
    b = {r["id"]: r["answer"] for r in csv.DictReader(open(args.file_b, encoding="utf-8-sig"))}

    ids = sorted(set(a) & set(b) & set(golden), key=int)
    if args.limit:
        ids = ids[: args.limit]

    wins_a = wins_b = ties = 0
    for qid in ids:
        question = questions[qid].split("Пользователь:")[-1].strip()
        first = verdict(question, golden[qid], a[qid], b[qid])
        second = verdict(question, golden[qid], b[qid], a[qid])
        # во втором прогоне A и B поменяны местами
        second = {"A": "B", "B": "A", "TIE": "TIE"}[second]
        score = {"A": 0, "B": 0, "TIE": 0}
        score[first] += 1
        score[second] += 1
        winner = "A" if score["A"] > score["B"] else "B" if score["B"] > score["A"] else "TIE"
        wins_a += winner == "A"
        wins_b += winner == "B"
        ties += winner == "TIE"
        print(f"  {qid:>2}: {winner:3}  (1-й прогон {first}, 2-й {second})", flush=True)

    total = len(ids)
    print(f"\nA={args.file_a}: побед {wins_a}, B={args.file_b}: побед {wins_b}, ничьих {ties} из {total}")
    print(f"доля побед A: {100 * wins_a / total:.0f}%  B: {100 * wins_b / total:.0f}%  "
          f"ничьи: {100 * ties / total:.0f}%")


if __name__ == "__main__":
    main()
