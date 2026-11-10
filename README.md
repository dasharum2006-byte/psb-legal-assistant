# psb-legal-assistant

RAG-ассистент для юристов: поиск по нормативным документам, ответы со ссылками на пункты и обоснованные отказы. Кейс ПСБ на Hack&Change 2026.

## Стек

- **backend** — FastAPI (`backend/app`), RAG в `backend/app/rag`, LLM в `backend/app/llm`
- **frontend** — React + Vite (`frontend/src`)
- **данные** — `data/knowledge-base.zip`, вопросы и эталонные ответы в `data/`

## Быстрый старт

```bash
cp .env.example .env        # вписать LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
docker compose up --build   # backend :8000, frontend :3000
```

Без Docker:

```bash
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload
cd frontend && npm install && npm run dev
```

Проверка: `curl localhost:8000/api/health`

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
|---|---|---|
| `backend/app/rag/orchestrator.py`, `docs/` | `feat/ai-orchestrator` | Даша |
| `backend/app/rag/parser.py`, `chunker.py`, `retriever.py` | `feat/rag-retrieval` | Ярик |
| `backend/app/llm/client.py`, `verifier.py` | `feat/llm-trust` | Даня |
| `scripts/eval_demo.py`, `data/eval_*` | `feat/eval-harness` | 4-й участник |
| `backend/app/api/*`, `frontend/*` | — | Даша (общие, изменения через PR) |
| И главный босс проекта | Нишевый Начальник💅🏻 | Илья |

Контракты между модулями (`retriever.search`, `llm.generate`, схема ответа) —
в [`docs/plan.md`](docs/plan.md). Пишем против заглушек, друг друга не ждём.

## Что нельзя коммитить

`.env` (реальные ключи), `data/knowledge-base.zip` (см. `.gitignore`).
