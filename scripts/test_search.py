import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.rag.retriever import load_index_from_disk, search

# Загружаем индекс, который мы только что построили
load_index_from_disk()

# Тестовые вопросы (возьми пару из demo_questions.csv)
questions = [
    "Какая арендная плата по договору с ООО «Северный Склад»?",
    "Какой срок оплаты по договору поставки?",
    "Какие документы нужны для проверки контрагента?"
]

for q in questions:
    print(f"\n--- Вопрос: {q} ---")
    results = search(q, k=2)
    for r in results:
        print(f"[{r.get('clause')}] {r['text'][:100]}... (score: {r['score']:.2f})")
