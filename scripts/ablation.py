"""Абляция: где именно ломается пайплайн — поиск, отбор или генерация.

Три режима на одних и тех же вопросах:
  1. pipeline — как в сабмите: поиск → отбор → LLM;
  2. oracle   — в контекст принудительно кладём ПРАВИЛЬНЫЙ пункт + топ-3 поиска;
  3. no_llm   — без модели: экстрактивная цитата.

Если oracle решает заметно больше, чем pipeline — виноват поиск/отбор.
Если oracle почти не помогает — виноват промпт/модель/нарезка.

Запуск:
  LLM_BASE_URL=... LLM_API_KEY=... DATA_DIR=data python3 scripts/ablation.py
"""
from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(ROOT / "data")

# Правильный источник для каждого демо-вопроса (проверено по demo_output.csv)
GOLD = {
    "1": ("а-08/2024", {"3.1", "3.2"}),
    "2": ("статья 70", None),
    "3": (None, None),
    "4": ("а-08/2024", {"3.1"}),
    "5": ("ред. 2023", {"2.4"}),
    "6": ("п-17/2024", {"6.1", "3.3"}),
    "7": ("а-08/2024", {"4.3"}),
    "8": ("п-17/2024", {"6.2"}),
    "9": ("пт-05/2023", {"4.1", "4.2"}),
    "10": ("ред. 2025", {"1.3"}),
    "11": ("записка", None),
    "12": ("статья 70", None),
    "13": ("статья 9", None),
    "14": ("статья 59", None),
    "15": ("а-08/2024", {"6.3"}),
    "16": ("п-17/2024", {"2.1"}),
    "17": ("пт-05/2023", {"6.1"}),
    "18": ("ред. 2025", {"3.2"}),
}


def load_solution():
    spec = importlib.util.spec_from_file_location("sol", ROOT / "submit/solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matches(chunk: dict, gold: tuple) -> bool:
    fragment, clauses = gold
    if fragment is None:
        return False
    if fragment not in chunk["doc"].lower():
        return False
    return clauses is None or chunk["clause"] in clauses


def norm_numbers(text: str) -> set[str]:
    return {re.sub(r"\D", "", n) for n in re.findall(r"\d[\d\s]{2,}", text) if len(re.sub(r"\D", "", n)) >= 3}


def norm_refs(text: str) -> set[str]:
    refs = set(re.findall(r"п\.\s*(\d+(?:\.\d+)+)", text))
    refs |= {"ст." + n for n in re.findall(r"ст\.\s*(\d+)", text)}
    return refs


def similarity(answer: str, golden: str) -> str:
    """Грубая оценка: совпали числа и ссылки или нет (для сравнения режимов)."""
    if not answer:
        return "пусто"
    ans_nums, gold_nums = norm_numbers(answer), norm_numbers(golden)
    ans_refs, gold_refs = norm_refs(answer), norm_refs(golden)
    if gold_nums and not ans_nums & gold_nums and not ans_refs & gold_refs:
        return "мимо"
    if ans_refs & gold_refs and (not gold_nums or ans_nums & gold_nums):
        return "точно"
    return "частично"


def main() -> None:
    sol = load_solution()
    documents = sol.load_documents(sol.find_kb())
    chunks = sol.chunk_documents(documents)
    retriever = sol.Retriever(chunks)
    client = sol.make_client() if sol.api_available() else None

    questions = list(csv.DictReader(open(DATA / "demo_questions.csv", encoding="utf-8-sig")))
    golden = {r["id"]: r["answer"] for r in csv.DictReader(open(DATA / "demo_output.csv", encoding="utf-8-sig"))}

    stats = {"pipeline": {}, "oracle": {}, "no_llm": {}}
    gold_in_candidates = 0
    gold_in_selected = 0

    print(f"{'№':>2} {'в кандид.':>9} {'pipeline':>10} {'oracle':>10} {'no_llm':>9}  вопрос")
    for row in questions:
        qid = row["id"]
        question = row["question"].split("Пользователь:")[-1].strip()
        gold = GOLD[qid]
        search_text = sol.expand_query(question)
        candidates = [c for c in retriever.search(search_text) if c["score"] >= sol.MIN_SCORE]

        gold_chunk = next((c for c in candidates if matches(c, gold)), None)
        gold_any = next((c for c in chunks if matches(c, gold)), None)
        if gold[0] is not None:
            gold_in_candidates += bool(gold_chunk)

        # 1. обычный пайплайн
        pipeline_answer = sol.answer_question(client, question, retriever)
        pipeline_verdict = "отказ" if pipeline_answer == sol.REFUSAL else similarity(pipeline_answer, golden[qid])

        # 2. oracle: правильный пункт первым в контексте
        import os as _os
        if _os.getenv("SKIP_ORACLE") == "1":
            oracle_answer = pipeline_answer
        elif gold_any is not None and client is not None:
            context = [gold_any] + [c for c in candidates if c["clause"] != gold_any["clause"]][:3]
            text = sol.ask_llm(client, question, context)
            if text:
                text = sol._ensure_citation(text, context)
                oracle_answer = text if sol._grounded(text, context) else sol.extractive_answer(question, context)
            else:
                oracle_answer = sol.extractive_answer(question, context)
        else:
            oracle_answer = pipeline_answer
        oracle_verdict = "отказ" if oracle_answer == sol.REFUSAL else similarity(oracle_answer, golden[qid])

        # 3. без модели
        no_llm_answer = sol.extractive_answer(question, candidates) if candidates else sol.REFUSAL
        no_llm_verdict = "отказ" if no_llm_answer == sol.REFUSAL else similarity(no_llm_answer, golden[qid])

        for mode, verdict in (("pipeline", pipeline_verdict), ("oracle", oracle_verdict), ("no_llm", no_llm_verdict)):
            stats[mode][verdict] = stats[mode].get(verdict, 0) + 1

        print(f"{qid:>2} {'да' if gold_chunk else 'НЕТ':>9} {pipeline_verdict:>10} {oracle_verdict:>10} {no_llm_verdict:>9}  {question[:42]}")

    if client is not None:
        client.close()

    print()
    print("=== ИТОГ ===")
    print(f"правильный пункт среди кандидатов поиска: {gold_in_candidates} из 17")
    for mode in ("no_llm", "pipeline", "oracle"):
        row = stats[mode]
        print(f"{mode:9}: точно {row.get('точно', 0):>2}  частично {row.get('частично', 0):>2}  "
              f"мимо {row.get('мимо', 0):>2}  отказ {row.get('отказ', 0):>2}")


if __name__ == "__main__":
    main()
