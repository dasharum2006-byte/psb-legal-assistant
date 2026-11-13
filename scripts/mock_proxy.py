"""Mock OpenAI-совместимого прокси для локальной проверки сабмита.

Повторяет три эндпоинта платформы ПСБ:
  POST /chat/completions  — «модель» возвращает ответ по первому фрагменту;
  POST /embeddings        — детерминированные векторы по словам;
  POST /rerank            — сортировка документов по пересечению слов.

Нужен, чтобы проверить все ветки solution.py (эмбеддинги, реранкер,
генерация, разбор ответа) без ключей и без интернета.

Запуск:  python3 scripts/mock_proxy.py 8111
"""
from __future__ import annotations

import json
import math
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

WORD = re.compile(r"[0-9a-zA-ZА-Яа-яЁё]{3,}")
DIM = 64


def words(text: str) -> set[str]:
    return {w.lower().replace("ё", "е") for w in WORD.findall(text)}


def vector(text: str) -> list[float]:
    """Простой детерминированный вектор: хэш слов в 64 измерения."""
    values = [0.0] * DIM
    for word in words(text):
        values[hash(word) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):          # тише в консоли
        pass

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:             # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send({"error": "bad json"}, 400)
            return

        if self.headers.get("Authorization", "").replace("Bearer ", "") == "":
            self._send({"error": "no key"}, 401)
            return

        if self.path.endswith("/embeddings"):
            inputs = payload.get("input") or []
            self._send({
                "data": [
                    {"index": index, "embedding": vector(text)}
                    for index, text in enumerate(inputs)
                ]
            })
            return

        if self.path.endswith("/rerank"):
            query = words(payload.get("query", ""))
            documents = payload.get("documents") or []
            scored = []
            for index, document in enumerate(documents):
                overlap = len(query & words(str(document)))
                scored.append((overlap, index))
            scored.sort(reverse=True)
            top_n = int(payload.get("top_n") or len(documents))
            self._send({
                "results": [
                    {"index": index, "relevance_score": float(overlap)}
                    for overlap, index in scored[:top_n]
                ]
            })
            return

        if self.path.endswith("/chat/completions"):
            messages = payload.get("messages") or []
            user = messages[-1]["content"] if messages else ""
            # имитация «модели»: цитируем первый фрагмент, который увидели
            clause = re.search(r"п\.\s*([\d.]+)", user)
            answer = (
                f"По предоставленным документам ответ такой: см. пункт {clause.group(1)}."
                if clause else
                "В предоставленной базе знаний нет документов по этому вопросу."
            )
            self._send({
                "choices": [{
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }]
            })
            return

        self._send({"error": "unknown endpoint"}, 404)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8111
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"mock proxy: http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
