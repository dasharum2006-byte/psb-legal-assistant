# Данные

В папке лежат материалы для локальной проверки:

- `knowledge-base.zip` — база знаний (не коммитится, см. `.gitignore`);
- `demo_questions.csv` — примеры вопросов;
- `demo_output.csv` — пример ответов.

Для автопроверки `submit/solution.py` читает `questions.csv` и `knowledge-base.zip` из `DATA_DIR`, который передаёт платформа.
