"""Сгенерировать перефразированные вопросы из чанков (кэш на диск).

Нужно, чтобы мерить поиск честно: пользователь не повторяет слова документа.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submit"))

import httpx  # noqa: E402

import solution  # noqa: E402

CACHE = Path("data/diag_questions.json")
MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-v4.1-flash")
KEY = os.getenv("LLM_API_KEY", "")
URL = (os.getenv("LLM_BASE_URL", "") or "").rstrip("/") + "/chat/completions"
random.seed(7)

if CACHE.exists():
    print(f"кэш: {CACHE} ({len(json.loads(CACHE.read_text()))} вопросов)")
    raise SystemExit

documents = solution.load_documents(Path("data/knowledge-base.zip"))
chunks = solution.chunk_documents(documents)
sample, seen_docs = [], set()
for chunk in random.sample(chunks, len(chunks)):
    if chunk["doc"] in seen_docs and len(sample) < 40:
        continue
    seen_docs.add(chunk["doc"])
    sample.append(chunk)
    if len(sample) >= 40:
        break

PROMPT = (
    "По фрагменту документа составь ОДИН короткий вопрос юриста от лица сотрудника "
    "(до 100 знаков). Не повторяй дословно формулировки фрагмента, перескажи своими словами. "
    "Только вопрос, без пояснений.\n\nФРАГМЕНТ:\n{text}"
)

out = []
client = httpx.Client(timeout=40)
for i, chunk in enumerate(sample, 1):
    payload = {"model": MODEL, "temperature": 0.3, "max_tokens": 800,
               "messages": [{"role": "user", "content": PROMPT.format(text=chunk["text"][:1200])}]}
    try:
        r = client.post(URL, headers={"Authorization": f"Bearer {KEY}"}, json=payload)
        question = r.json()["choices"][0]["message"]["content"].strip().splitlines()[0]
        question = re.sub(r"^[\d.\-—)\s]+", "", question)[:160]
    except Exception as exc:
        print(f"  {i}: ошибка {type(exc).__name__}: {exc}", file=sys.stderr)
        continue
    out.append({"question": question, "doc": chunk["doc"], "number": chunk.get("number"),
                "clause": chunk["clause"], "doc_type": chunk.get("doc_type")})
    print(f"  {i:2}. [{chunk.get('doc_type')}] {question[:80]}")

CACHE.parent.mkdir(parents=True, exist_ok=True)
CACHE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nсохранено: {len(out)} вопросов -> {CACHE}")
