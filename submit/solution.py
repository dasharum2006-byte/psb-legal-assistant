"""Точка входа для автоматической проверки ПСБ.

Пока заглушка: читает questions.csv и создаёт submission.csv с ответами-заглушками.
Дальше сюда подключается реальный RAG-пайплайн.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    questions_path = data_dir / "questions.csv"

    with questions_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for row in rows:
            writer.writerow([row["id"], "Ответ пока не реализован."])

    print(f"Создан submission.csv: {len(rows)} ответов")


if __name__ == "__main__":
    main()
