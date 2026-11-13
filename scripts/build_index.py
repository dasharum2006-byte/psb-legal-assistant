import sys
import os
import json

# Добавляем путь к корню проекта, чтобы работали импорты из backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.rag.parser import load_documents
from backend.app.rag.chunker import chunk_documents
from backend.app.rag.retriever import build_index

def main(kb_path):
    print(f"1. Читаем документы из {kb_path}...")
    docs = load_documents(kb_path)
    print(f"   Прочитано документов: {len(docs)}")
    
    print("2. Режем документы на чанки...")
    chunks = chunk_documents(docs)
    print(f"   Получено чанков: {len(chunks)}")
    
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
            
    print(f"✅ Готово! Индекс сохранен в {chunks_path}")

if __name__ == "__main__":
    # Путь к архиву. Если запускаешь из корня проекта, путь такой:
    main("data/knowledge-base.zip")
