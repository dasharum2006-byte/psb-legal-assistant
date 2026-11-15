import sys
import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# Добавляем путь к корню проекта, чтобы работали импорты из backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from app.rag.parser import load_documents
from app.rag.chunker import chunk_documents
from app.rag.retriever import build_index

def main(kb_path):
    print(f"1. Читаем документы из {kb_path}...")
    docs = load_documents(kb_path)
    print(f"   Прочитано документов: {len(docs)}")
    # Ручная сборка должна падать громко: пустой индекс — почти всегда
    # не тот путь к архиву, а не пустая база.
    if not docs:
        print(f"❌ в {kb_path} нет читаемых документов", file=sys.stderr)
        sys.exit(1)
    
    print("2. Режем документы на чанки...")
    chunks = chunk_documents(docs)
    print(f"   Получено чанков: {len(chunks)}")
    if not chunks:
        print("❌ документы прочитаны, но чанков нет", file=sys.stderr)
        sys.exit(1)
    
    print("3. Строим индекс BM25...")
    # Вызываем функцию из retriever.py, она сохранит индекс в памяти
    build_index(chunks)
    
    # Создаем папку data/index, если ее нет
    os.makedirs("data/index", exist_ok=True)
    
    # Сохраняем чанки на диск (в формате JSON Lines)
    chunks_path = "data/index/chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    # Отпечаток кода нарезки: по нему сервер поймёт, что индекс устарел
    fingerprint = ""
    for name in ("backend/app/rag/chunker.py", "backend/app/rag/parser.py"):
        path = Path(name)
        if path.exists():
            fingerprint += hashlib.sha256(path.read_bytes()).hexdigest()

    meta = {
        "chunks": len(chunks),
        "documents": len(docs),
        "code_fingerprint": hashlib.sha256(fingerprint.encode()).hexdigest()[:16],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(kb_path),
    }
    with open("data/index/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Готово! Индекс сохранен в {chunks_path}")
    print(f"   meta: {meta['chunks']} чанков, отпечаток кода {meta['code_fingerprint']}")

if __name__ == "__main__":
    # Путь к архиву можно передать аргументом; без аргумента берём стандартный.
    # Раньше аргумент молча игнорировался и всегда читался data/knowledge-base.zip.
    kb = sys.argv[1] if len(sys.argv) > 1 else "data/knowledge-base.zip"
    main(kb)
