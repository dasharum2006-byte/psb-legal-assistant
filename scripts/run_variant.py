"""Прогнать сабмит с другим чанкером (для A/B-замера).

Использование: python scripts/run_variant.py --chunker backend|submit
Ответы пишутся в submission.csv текущего каталога (как на платформе).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submit"))
sys.path.insert(0, str(ROOT / "backend"))

import solution  # noqa: E402


def to_submit_schema(chunks: list[dict]) -> list[dict]:
    """Чанки бэкенда -> формат, который ждёт конвейер сабмита."""
    return [
        {
            "doc": chunk["title"],
            "number": chunk.get("number"),
            "doc_type": chunk.get("doc_type"),
            "clause": chunk["clause"],
            "section": chunk.get("section", ""),
            "text": chunk["text"],
            "repealed": bool(chunk.get("repealed")),
            "searchable": chunk.get("searchable")
            or f"{chunk['title']} {chunk.get('section', '')} {chunk['clause']} {chunk['text']}",
        }
        for chunk in chunks
    ]


def main() -> None:
    choice = "submit"
    if "--chunker" in sys.argv:
        choice = sys.argv[sys.argv.index("--chunker") + 1]

    if choice == "backend":
        from app.rag.chunker import chunk_documents as backend_chunk

        original = solution.chunk_documents

        def patched(documents: list[dict]) -> list[dict]:
            return to_submit_schema(backend_chunk(documents))

        solution.chunk_documents = patched
        print(f"[VARIANT] чанкер бэкенда (было бы {len(original([])) if False else 'submit'})", flush=True)

    solution.main()


if __name__ == "__main__":
    main()
