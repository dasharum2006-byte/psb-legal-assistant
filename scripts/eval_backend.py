"""Прогнать оркестратор бэкенда по демо-вопросам и сохранить ответы в CSV.

Нужно для сравнения вариантов (например, с соседними чанками и без) по
метрикам против эталона, а не по ощущениям.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.rag import retriever  # noqa: E402
from app.rag.orchestrator import ask  # noqa: E402


def split_dialog(question: str) -> tuple[str, list[dict]]:
    """«Пользователь: … Ассистент: … Пользователь: …» -> последний вопрос + история."""
    history, last = [], question
    if "Пользователь:" in question:
        parts = [p.strip() for p in question.split("Пользователь:") if p.strip()]
        for part in parts[:-1]:
            text, _, answer = part.partition("Ассистент:")
            history.append({"role": "user", "content": text.strip()})
            if answer.strip():
                history.append({"role": "assistant", "content": answer.strip()})
        last = parts[-1]
    return last, history


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "backend_answers.csv"
    index_dir = os.getenv("INDEX_DIR", str(ROOT / "data/index"))
    retriever.load_index_from_disk(index_dir)
    rows = list(csv.DictReader(open(ROOT / "data/demo_questions.csv", encoding="utf-8-sig")))
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for row in rows:
            last, history = split_dialog(row["question"])
            result = ask(last, history, retriever)
            writer.writerow([row["id"], (result.get("answer") or "").replace("\n", " ")])
            print(f"  {row['id']:>2}: отказ={result.get('refusal')} "
                  f"источников={len(result.get('sources', []))}", flush=True)


if __name__ == "__main__":
    main()
