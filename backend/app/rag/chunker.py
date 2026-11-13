"""Нарезка документов на чанки.

TODO:
- резать по пунктам договоров и статьям законов;
- сохранять doc_id, clause, date, revision, status.
"""
import re
from backend.app.rag.parser import load_documents
def chunk_documents(docs):
    chunks = []
    for doc in docs:
        text = doc["text"]
        doc_id = doc["doc_id"]
        title = doc["title"]
        doc_type = doc["doc_type"]

        if doc_type in ['contract', 'regulation', 'amendment']:
            pattern =  r'(?m)^\s*(\d+\.\d+)[\.\)]\s*'
            parts = re.split(pattern, text)
            for i in range(1, len(parts), 2):
                clause = parts[i]
                chunk_text = parts[i + 1]
                if len(chunk_text.strip()) > 10:
                    chunk = {
                        "doc_id":doc_id,
                        "title":title,
                        "doc_type":doc_type,
                        "clause": f"п. {clause}",
                        "text": chunk_text.strip(),
                        "status": doc.get("status"),
                        "valid_from":doc.get("date"),
                        "valid_to":None,
                        "suspicious_instruction": False,
                        "searchable": f"{title} {clause} {chunk_text.strip()}"
                    }
                    chunks.append(chunk)
        elif(doc_type == "law"):
            pattern = r'(?m)^\s*(Статья\s+\d+(?:\.\d+)?)'
            parts = re.split(pattern, text)
            for i in range(1, len(parts), 2):
                clause = parts[i]
                chunk_text = parts[i + 1]
                if len(chunk_text.strip()) > 10:
                    chunk = {
                        "doc_id":doc_id,
                        "title":title,
                        "doc_type":doc_type,
                        "clause": clause,
                        "text": chunk_text.strip(),
                        "status": doc.get("status"),
                        "valid_from": doc.get("date"),
                        "valid_to": None,
                        "suspicious_instruction": False,
                        "searchable": f"{title} {clause} {chunk_text.strip()}"
                    }
                    chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    docs = load_documents("../../../data/knowledge-base.zip")
    print(f"Прочитано документов: {len(docs)}")
    
    # Вызываем функцию, которая уже есть в этом файле
    chunks = chunk_documents(docs) 
    print(f"Получено чанков: {len(chunks)}")
    
    for c in chunks[:3]:
        print(f"[{c['clause']}] {c['text'][:50]}...")

