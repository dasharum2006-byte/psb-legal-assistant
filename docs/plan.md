# План пайплайна (ветка `feat/ai-orchestrator`)

Документ фиксирует контракты между модулями. Все пишут против заглушек и этого
документа — ждать друг друга не нужно.

## Схема запроса

```
POST /api/chat
      │
      ▼
api/chat.py ──► rag/orchestrator.ask()
                       │
       ┌───────────────┼─────────────────┬──────────────────┐
       ▼               ▼                 ▼                  ▼
 analyze_query()   retriever.search()  build_context()   llm.generate()
   (я)                 (Ярик)              (я)             (Даня)
                                                            │
                                                            ▼
                                                   llm/verifier.py (Даня)
                                                            │
                                                            ▼
                                        {answer, sources, refusal, missing_info}
```

## Контракты

### `retriever.search(query: str, k: int = TOP_K) -> list[dict]`

Владелец: Ярик (`backend/app/rag/retriever.py`)

```python
[
    {"doc": "Трудовой кодекс РФ", "clause": "ст. 136", "text": "...", "score": 0.83},
]
```

- `score` — чем больше, тем релевантнее; сортировка по убыванию.
- Пустой список допустим — оркестратор сам решит, отказать или нет.

### `llm_client.generate(prompt: str) -> str`

Владелец: Даня (`backend/app/llm/client.py`)

- Возвращает только текст ответа, без разбора JSON-обвязки.
- Модель/ключ берутся из `.env` (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`).

### `llm_verifier.verify(answer: str, chunks: list[dict]) -> dict`

Владелец: Даня (`backend/app/llm/verifier.py`)

```python
{"ok": False, "refusal": True, "missing_info": ["нет данных о размере платы"]}
```

- Проверяет, что каждая ссылка (`doc` + `clause`) существует в переданных чанках.
- Если опоры нет — `refusal=True`, в `missing_info` пишем, чего не хватило.

## Функции оркестратора (мои)

| Функция | Что делает |
|---|---|
| `analyze_query(question)` | нормализация: даты, номера договоров, названия сторон |
| `build_context(chunks, max_chars=6000)` | склейка чанков в один контекст с бюджетом символов |
| `ask(question, history, retriever)` | главный вход: разбор → поиск → контекст → LLM → проверка → ответ/отказ |

Целевой ответ `ask()` (совпадает со схемой `ChatResponse` в `backend/app/api/chat.py`):

```python
{
    "answer": "...",
    "sources": [{"doc": "...", "clause": "...", "text": "...", "score": 0.83}],
    "refusal": False,
    "missing_info": [],
}
```

## Порядок интеграции

1. Ярик — поиск (`search()` возвращает чанки из `data/knowledge-base.zip`).
2. Даня — ответ через LLM + проверка цитат.
3. Я соединяю всё в `ask()`.
4. `api/chat.py` начинает отвечать не заглушкой (фронт оживает).
5. 4-й человек — eval-скрипт и демо (см. ниже).

## Eval harness (задача 4-го человека, `feat/eval-harness`)

Файлы: `scripts/eval_demo.py`, `data/eval_results.csv`. Никого не пересекает.

1. Прочитать `data/demo_questions.csv`.
2. По каждому вопросу отправить `POST /api/chat`.
3. Получить `answer`, `sources`, `refusal`.
4. Записать результат в таблицу.
5. Пометить проблемные кейсы:
   - пустой ответ;
   - `sources` пуст, хотя ответа-отказа нет;
   - выдуманный документ (пункта нет в базе);
   - нет отказа, хотя данных в базе нет;
   - расхождение с ключевыми фактами из `data/demo_output.csv`.

Пример вывода:

```text
id | question | sources_count | refusal | status
 1 | ...      | 2             | false   | OK
 2 | ...      | 0             | true    | OK
 3 | ...      | 1             | false   | CHECK
```

Скрипт — это панель регрессий: меняем промпт/чанкинг/поиск, прогоняем, видим
стало лучше или хуже. Имеет смысл, когда пайплайн уже отвечает не заглушкой.
