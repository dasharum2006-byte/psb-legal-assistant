# psb-legal-assistant

RAG-ассистент для юристов: поиск по корпоративным нормативным документам и законодательству, ответы со ссылками на пункты/статьи и обоснованные отказы. Кейс ПСБ на Hack&Change 2026.

## Стек и архитектура

- **backend** — FastAPI (`backend/app`), RAG в `backend/app/rag`, LLM и верификатор в `backend/app/llm`
- **frontend** — React + Vite (`frontend/src`)
- **данные** — `data/knowledge-base.zip`, вопросы и эталонные ответы в `data/`
- **сабмит** — автономный модуль `submit/solution.py` (только стандартная библиотека + `httpx` и `rank-bm25`), синхронизируемый из `backend/app`

### Как устроен RAG-пайплайн

1. **Парсинг и метаданные (`parser.py`):** чтение `.txt`, `.md` и `.docx` (разбор XML), извлечение типа документа (`law`, `contract`, `regulation`, `amendment`, `memo`, `letter`), регистрационного номера (`№ А-08/2024`, `ЮО-7`) и статуса действия редакции (`active` / `repealed`).
2. **Структурный чанкинг (`chunker.py`):** нарезка законов по статьям (`ст. 70`), договоров и регламентов по многоуровневым пунктам (`3.1`, `3.1.2`) с привязкой к заголовку раздела, а писем и записок — по абзацам (`абз. 1`).
3. **Гибридный поиск и бустинг (`retriever.py`, `metadata.py`, `semantic.py`):**
   - BM25 со стеммингом русских словоформ и расширением запроса доменными синонимами;
   - точный бустинг по номерам документов (включая упоминания без `№`), разведение контрагентов в кавычках (`«Альфа-Трейд»` ≠ `«Альфа-Трейдинг»`) и штраф устаревшим редакциям (`repealed`), если вопрос не про прошлый период;
   - векторный поиск и переупорядочивание через `/embeddings` (`baai/bge-m3`) с потокобезопасным кэшем + финальный отбор через `/rerank` (`cohere/rerank-5-fast`);
   - если платформенные эндпоинты недоступны, поиск автоматически работает на BM25 с экстрактивным fallback.
4. **Проверка достоверности (`verifier.py` / `_grounded`):** обязательные ссылки на пункты/статьи, пост-проверка цифр, сумм, процентов и пунктов по контексту, защита от prompt-инъекций и честный отказ при отсутствии данных в базе.

---

## Быстрый старт

### Настройка окружения

```bash
cp .env.example .env        # вписать LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
```

Основные параметры в `.env`:

```env
LLM_BASE_URL=https://example.com
LLM_API_KEY=your_api_key_here
LLM_MODEL=deepseek-chat
EMB_MODEL=baai/bge-m3
RERANKER_MODEL=cohere/rerank-5-fast
DATA_DIR=data
```

### Запуск через Docker Compose

```bash
docker compose up --build   # backend :8000, frontend :3000
```

### Локальный запуск (без Docker)

```bash
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload
cd frontend && npm install && npm run dev
```

**Важно:** после правок в `backend/app/rag/parser.py` или `chunker.py` пересоберите индекс, иначе сервер будет работать на старых чанках:

```bash
.venv/bin/python scripts/build_index.py data/knowledge-base.zip
```

Сервер проверяет отпечаток кода нарезки: если индекс собран другим кодом, в `/api/health` появится `index_stale` с просьбой пересобрать, а тест `backend/tests/test_index_freshness.py` упадёт.

Проверка здоровья сервиса:
```bash
curl localhost:8000/api/health
```

---

## Сабмит и тестирование

Платформа принимает автономный `submit/solution.py`, поэтому канонические модули (`parser.py`, `metadata.py`, `textsel.py`, `SYSTEM_PROMPT`) автоматически переносятся из `backend/app` в `submit/solution.py`:

```bash
# 1. Синхронизировать код из backend/app в submit/solution.py
.venv/bin/python scripts/sync_submit.py

# 2. Запустить все тесты (бэкенд + сабмит + синхронизация)
.venv/bin/pytest tests backend/tests

# 3. Прогнать предполётную проверку на демо-вопросах и собрать submission.zip
bash scripts/pre_submit_check.sh
```

---

## Ветки и процесс

`main` защищена, напрямую не пушим. Своя зона → своя ветка → PR.

```bash
git checkout main && git pull
git checkout -b feat/<зона>
# ... работа ...
git fetch origin && git rebase origin/main
git push -u origin feat/<зона>
```

Имена: `feat/<kebab-case>`. Коммиты: `feat(rag): ...`, `fix(api): ...`.

## Кто чем владеет

Чтобы не ловить конфликты — в чужие файлы не лезем, вопросы через PR-комментарии.

| Путь | Ветка | Владелец |
| - | - | - |
| `backend/app/rag/orchestrator.py`, `docs/` | `feat/ai-orchestrator` | Даша |
| `backend/app/rag/parser.py`, `chunker.py`, `retriever.py` | `feat/rag-retrieval` | Ярик |
| `backend/app/llm/client.py`, `verifier.py` | `feat/llm-trust` | Даня |
| `scripts/eval_demo.py`, `data/eval_*` | `feat/eval-harness` | Денис |
| `backend/app/api/*`, `frontend/*` | — | Даша (общие, изменения через PR) |
| И главный босс проекта | Нишевый Начальник💅🏻 | Илья |

Контракты между модулями (`retriever.search`, `llm.generate`, схема ответа) — в [`docs/plan.md`](docs/plan.md). Пишем против заглушек, друг друга не ждём.

## Что нельзя коммитить

`.env` (реальные ключи), `data/knowledge-base.zip`, временные артефакты прогонов `submission*.csv` / `backend_*.csv` (см. `.gitignore`).
