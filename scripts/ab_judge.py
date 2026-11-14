"""Локальный LLM-судья: воспроизводит методику оценки платформы ПСБ.

Что делает:
  1. строит ответы «baseline» так же примитивно, как в примере организаторов
     (куски по 1200 символов, поиск по словам, простой промпт);
  2. берёт наши ответы из submission.csv;
  3. сравнивает A и B дважды (A=baseline/B=наше и наоборот) с golden-ответом,
     выставляет 0 / 0.25 / 0.5 / 0.75 / 1;
  4. считает Match Rate и Win Rate и итоговый Score = 0.7*MR + 0.3*WR.

Запуск (нужен ключ OpenRouter или адрес платформенного прокси):
  LLM_BASE_URL=... LLM_API_KEY=... python3 scripts/ab_judge.py

Флаги:
  --answers path   файл с нашими ответами (по умолчанию submission.csv)
  --limit N        проверить первые N вопросов
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import urllib.request
from pathlib import Path

BASE = (os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL") or "").rstrip("/")
KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
MODEL = os.getenv("LLM_MODEL") or os.getenv("GEN_MODEL") or "deepseek/deepseek-v4.1-flash"

DATA = Path(os.getenv("DATA_DIR", "data"))
SNIPPET = 1200
WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{4,}")


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def chat(system: str, user: str, temperature: float = 0.0, max_tokens: int = 800) -> str:
    data = post("/chat/completions", {
        "model": MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    })
    return (data["choices"][0]["message"].get("content") or "").strip()


# ─────────── baseline: как в примере организаторов ───────────

def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower().replace("ё", "е")))


def load_kb() -> list[tuple[str, str]]:
    import zipfile
    archive_path = None
    for name in ("kb.zip", "knowledge-base.zip", "knowledge_base.zip"):
        candidate = DATA / name
        if candidate.exists():
            archive_path = candidate
            break
    if archive_path is None:
        raise SystemExit("нет архива базы знаний в DATA_DIR")

    documents = []
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith((".txt", ".md")):
                continue
            text = archive.read(name).decode("utf-8-sig", errors="replace")
            if text.strip():
                documents.append((Path(name).stem, text.strip()))
    return documents


def snippet(text: str) -> str:
    piece = text[:SNIPPET]
    if len(text) > SNIPPET:
        boundary = piece.rfind("\n")
        if boundary > SNIPPET // 2:
            piece = piece[:boundary]
    return piece.strip()


def baseline_answer(question: str, documents: list[tuple[str, str]], top_k: int = 4) -> str:
    query = words(question)
    candidates = sorted(
        documents,
        key=lambda doc: len(query & words(doc[0] + " " + doc[1])),
        reverse=True,
    )[:top_k]
    context = "\n\n".join(name + "\n" + snippet(text) for name, text in candidates)
    return chat(
        "Ответь на вопрос по приложенным фрагментам. Не выдумывай юридические нормы. "
        "Если ответа нет, сообщи об этом.",
        "Фрагменты:\n" + context + "\n\nВопрос:\n" + question,
    )


# ─────────── судья ───────────

JUDGE_SYSTEM = (
    "Ты оцениваешь два ответа юридического ассистента: A и B. "
    "Есть эталонный ответ (golden) и база знаний, по которой отвечали. "
    "Сравни B с A по точности, полноте и наличию корректных ссылок на пункты. "
    "Выдуманный факт (номер пункта, сумма, срок, которых нет в документах) — грубая ошибка. "
    "Верни ТОЛЬКО JSON: {\"score\": 0|0.25|0.5|0.75|1} и поле \"why\" (до 15 слов).\n"
    "Шкала: 1.00 — B значительно лучше A; 0.75 — немного лучше; 0.50 — сопоставимы; "
    "0.25 — B немного хуже; 0.00 — B значительно хуже."
)

SCORE_RE = re.compile(r'"score"\s*:\s*([01](?:\.\d+)?)')


def judge(question: str, golden: str, answer_a: str, answer_b: str) -> float:
    user = (
        f"Вопрос: {question}\n\n"
        f"Эталонный ответ (golden): {golden}\n\n"
        f"Ответ A:\n{answer_a}\n\n"
        f"Ответ B:\n{answer_b}\n"
    )
    try:
        raw = chat(JUDGE_SYSTEM, user, temperature=0.0, max_tokens=200)
        match = SCORE_RE.search(raw)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
    except Exception as exc:
        print(f"судья упал: {type(exc).__name__}", file=sys.stderr)
    return 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", default="submission.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--baseline", choices=("naive", "golden"), default="naive",
                        help="с чем сравнивать: 'naive' — как в примере организаторов, "
                             "'golden' — эталонные ответы (жёсткая планка)")
    args = parser.parse_args()

    if not BASE or not KEY:
        raise SystemExit("нужны LLM_BASE_URL и LLM_API_KEY")

    questions = list(csv.DictReader(open(DATA / "demo_questions.csv", encoding="utf-8-sig")))
    golden = {r["id"]: r["answer"] for r in csv.DictReader(open(DATA / "demo_output.csv", encoding="utf-8-sig"))}
    ours = {r["id"]: r["answer"] for r in csv.DictReader(open(args.answers, encoding="utf-8-sig"))}
    if args.limit:
        questions = questions[: args.limit]

    documents = load_kb()
    print(f"вопросов: {len(questions)}, документов: {len(documents)}", flush=True)

    scores = []
    for row in questions:
        qid = row["id"]
        question = row["question"].split("Пользователь:")[-1].strip()
        if args.baseline == "golden":
            base = golden.get(qid, "")
        else:
            base = baseline_answer(question, documents)
        mine = ours.get(qid, "")

        # два раунда, чтобы убрать влияние порядка
        forward = judge(question, golden.get(qid, ""), base, mine)
        backward = judge(question, golden.get(qid, ""), mine, base)
        final = (forward + (1.0 - backward)) / 2.0
        scores.append(final)
        print(f"{qid:>2} score={final:.2f}  (A=B:{forward:.2f}, B=A:{1 - backward:.2f})  "
              f"{question[:44]}", flush=True)

    match_rate = statistics.mean(scores) if scores else 0.0
    wins = sum(1 for s in scores if s > 0.5)
    losses = sum(1 for s in scores if s < 0.5)
    win_rate = wins / (wins + losses) if (wins + losses) else 0.5
    total = 0.7 * match_rate + 0.3 * win_rate

    print()
    print(f"Match Rate = {match_rate:.3f}   Win Rate = {win_rate:.3f}   "
          f"(побед {wins}, поражений {losses})")
    print(f"ИТОГОВЫЙ Score = {total * 100:.2f}%   ← на платформе сейчас ~17.9%")


if __name__ == "__main__":
    main()
