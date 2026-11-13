#!/usr/bin/env bash
# Предполётная проверка перед сабмитом на платформу.
#
# У команды 5 сабмитов в день, поэтому проверяем всё локально:
#   1) решение запускается и не падает;
#   2) ответы не пустые и с ссылками на пункты;
#   3) архив содержит только solution.py и requirements.txt;
#   4) в архиве нет следов окружения (__pycache__, .venv, .env).
#
# Запуск: bash scripts/pre_submit_check.sh
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV="$ROOT/.venv/bin/python"
[ -x "$VENV" ] || VENV="python3"

fail() { echo "❌ $1"; EXIT=1; }
ok()   { echo "✅ $1"; }
EXIT=0

echo "── 1. синтаксис solution.py ──"
if "$VENV" -c "import ast,sys; ast.parse(open('submit/solution.py').read())"; then
  ok "синтаксис в порядке"
else
  fail "solution.py не парсится"
  exit 1
fi

echo "── 2. только разрешённые зависимости ──"
ALLOWED="fastapi uvicorn pydantic python-multipart python-dotenv openai httpx requests langchain langchain_openai langchain_text_splitters numpy faiss chromadb rank_bm25 bm25s sklearn rapidfuzz pandas pypdf docx bs4 lxml yaml sqlalchemy tiktoken"
IMPORTS=$(grep -E "^(import|from) " submit/solution.py | awk '{print $2}' | cut -d. -f1 | sort -u)
BAD=""
for mod in $IMPORTS; do
  case " $ALLOWED standard_library " in
    *" $mod "*) ;;
    *)
      if ! "$VENV" -c "import $mod, sys; sys.exit(0 if '$mod'.islower() else 0)" 2>/dev/null; then
        BAD="$BAD $mod"
      fi
      ;;
  esac
done
# грубая, но полезная проверка: явные «запрещённые» пакеты
if grep -qE "pydantic_settings|langgraph|sentence_transformers|torch|transformers" submit/solution.py; then
  fail "в solution.py есть библиотека не из разрешённого списка"
else
  ok "импорты выглядят допустимо"
fi
echo "   импорты: $(echo $IMPORTS | tr '\n' ' ')"

echo "── 3. прогон на demo_questions.csv ──"
cp data/demo_questions.csv data/questions.csv
OUT=$("$VENV" submit/solution.py 2>&1)
echo "$OUT" | tail -2
TOTAL=$(tail -n +2 submission.csv | wc -l)
EMPTY=$(python3 - <<'PY'
import csv
rows = list(csv.DictReader(open("submission.csv", encoding="utf-8-sig")))
print(sum(1 for r in rows if not (r["answer"] or "").strip()))
PY
)
CITED=$(python3 - <<'PY'
import csv, re
rows = list(csv.DictReader(open("submission.csv", encoding="utf-8-sig")))
pat = re.compile(r"(п\.|пункт|ст\.|статья|абз\.)", re.I)
print(sum(1 for r in rows if pat.search(r["answer"] or "")))
PY
)
[ "$TOTAL" -eq 18 ] && ok "ответов: $TOTAL" || fail "ответов: $TOTAL (ожидалось 18)"
[ "$EMPTY" -eq 0 ] && ok "пустых ответов нет" || fail "пустых ответов: $EMPTY"
[ "$CITED" -ge 12 ] && ok "со ссылкой на источник: $CITED из $TOTAL" || fail "со ссылкой только $CITED из $TOTAL"

echo "── 4. сборка архива ──"
bash scripts/make_submit.sh >/dev/null
if unzip -l submission.zip | grep -qE "__pycache__|\.venv|\.env|node_modules"; then
  fail "в архиве есть лишние файлы"
else
  ok "в архиве чисто: $(unzip -Z1 submission.zip | tr '\n' ' ')"
fi

# solution.py должен лежать В КОРНЕ архива, иначе платформа его не найдёт
if unzip -Z1 submission.zip | grep -qx "solution.py"; then
  ok "solution.py в корне архива"
else
  fail "solution.py НЕ в корне — платформа не запустит решение"
fi
if unzip -Z1 submission.zip | grep -qE "\.zip$"; then
  fail "внутри архива есть вложенный zip — так нельзя"
fi

rm -f data/questions.csv
echo
if [ "$EXIT" -eq 0 ]; then
  echo "🟢 Можно отправлять: submission.zip готов ($(du -h submission.zip | cut -f1))"
else
  echo "🔴 Есть проблемы — сабмит тратить не стоит"
fi
exit "$EXIT"
