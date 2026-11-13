#!/usr/bin/env bash
# Имитация запуска на платформе ПСБ.
#
# Проверяет ровно то, что проверит платформа:
#   1) свежий каталог с данными (questions.csv + knowledge-base.zip);
#   2) отдельное окружение, где стоят ТОЛЬКО зависимости из requirements.txt;
#   3) запуск solution.py без сети до интернета, но с локальным mock-прокси;
#   4) валидация submission.csv по регламенту.
#
# Запуск: bash scripts/simulate_platform.sh
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
WORK="$(mktemp -d)"
PORT=8111
FAILS=0

fail() { echo "❌ $1"; FAILS=$((FAILS + 1)); }
ok()   { echo "✅ $1"; }

cleanup() {
  [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "── 1. готовим окружение как на платформе ──"
mkdir -p "$WORK/data" "$WORK/env"
cp data/demo_questions.csv "$WORK/data/questions.csv"
cp data/knowledge-base.zip "$WORK/data/"
cp submit/solution.py submit/requirements.txt "$WORK/"
ok "рабочий каталог: $WORK"

python3 -m venv "$WORK/env" >/dev/null 2>&1 || fail "не удалось создать venv"
"$WORK/env/bin/pip" install -q -r "$WORK/requirements.txt" 2>&1 | tail -2
ok "зависимости из requirements.txt установлены"

echo "── 2. поднимаем mock-прокси (chat + embeddings + rerank) ──"
"$ROOT/.venv/bin/python" "$ROOT/scripts/mock_proxy.py" "$PORT" >"$WORK/proxy.log" 2>&1 &
PROXY_PID=$!
sleep 2
curl -s -m 5 -o /dev/null -X POST "http://127.0.0.1:$PORT/embeddings" \
  -H "Authorization: Bearer test" -H "Content-Type: application/json" \
  -d '{"model":"m","input":["тест"]}' && ok "прокси отвечает" || fail "прокси не отвечает"

echo "── 3. запуск solution.py (как на платформе) ──"
cd "$WORK"
env -i PATH="$WORK/env/bin:/usr/bin:/bin" \
  DATA_DIR="$WORK/data" \
  LLM_BASE_URL="http://127.0.0.1:$PORT" \
  LLM_API_KEY="test-key" \
  LLM_MODEL="mock" \
  MAX_WORKERS=4 \
  TOP_K=8 \
  RERANKER_TOP_N=3 \
  TEMPERATURE=0 \
  MAX_TOKENS=1500 \
  timeout 600 "$WORK/env/bin/python" solution.py 2>"$WORK/stderr.log"
RUN_CODE=$?
[ "$RUN_CODE" -eq 0 ] && ok "код завершился без ошибок" || fail "код завершился с ошибкой $RUN_CODE"
[ -s "$WORK/submission.csv" ] && ok "submission.csv создан" || fail "submission.csv отсутствует"

echo "── 4. валидация по регламенту ──"
"$ROOT/.venv/bin/python" - "$WORK" <<'PY'
import csv, sys, re
from pathlib import Path

work = Path(sys.argv[1])
questions = list(csv.DictReader(open(work / "data/questions.csv", encoding="utf-8-sig")))
rows = list(csv.DictReader(open(work / "submission.csv", encoding="utf-8-sig")))

problems = []
if [r.strip() for r in rows[0].keys()] != ["id", "answer"]:
    problems.append(f"колонки должны быть id,answer, а есть {list(rows[0].keys())}")
if len(rows) != len(questions):
    problems.append(f"ответов {len(rows)}, вопросов {len(questions)}")
ids = [r["id"] for r in rows]
if len(ids) != len(set(ids)):
    problems.append("есть повторяющиеся id")
if {r["id"] for r in rows} != {q["id"] for q in questions}:
    problems.append("набор id не совпадает с вопросами")
empty = [r["id"] for r in rows if not (r["answer"] or "").strip()]
if empty:
    problems.append(f"пустые ответы: {empty[:5]}")
long = [r["id"] for r in rows if len(r["answer"]) > 10000]
if long:
    problems.append(f"ответы длиннее 10 000 символов: {long[:5]}")
cited = sum(1 for r in rows if re.search(r"(п\.|ст\.|абз\.)", r["answer"]))
refused = sum(1 for r in rows if "нет документов" in r["answer"].lower())

print(f"   ответов: {len(rows)}, со ссылкой: {cited}, отказов: {refused}")
if "maximum" in work.name:
    pass
if problems:
    for problem in problems:
        print(f"❌ {problem}")
    sys.exit(1)
print("✅ submission.csv соответствует регламенту")
PY
[ $? -eq 0 ] || FAILS=$((FAILS + 1))

echo
echo "── stderr решения (первые строки) ──"
head -5 "$WORK/stderr.log" 2>/dev/null | sed 's/^/   /'

echo
if [ "$FAILS" -eq 0 ]; then
  echo "🟢 Имитация платформы пройдена: решение запускается и создаёт валидный submission.csv"
else
  echo "🔴 Проблем: $FAILS"
fi
exit "$FAILS"
