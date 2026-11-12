"""Парсинг документов базы знаний.

TODO:
- читать knowledge-base.zip;
- поддерживать TXT / DOCX / PDF;
- возвращать документы с метаданными.
"""

import zipfile
import os
import re

def load_documents(kb_path:str) -> list[dict]:
    documents = []
    if(kb_path.endswith('.zip')):
        with zipfile.ZipFile(kb_path, 'r') as z:
            for filename in z.namelist():
                if(filename.endswith('.txt')):
                    with z.open(filename) as f:
                        text = f.read().decode('utf-8-sig', errors='replace')
                    doc_id = filename.replace('.txt', '').replace(' ', '_').lower()
                    title = text.split('\n')[0].strip()

                    doc_type = "unknown"
                    text_upper = text.upper() 
                    if("ДОГОВОР" in text.upper()): 
                        doc_type = "contract"
                    elif("РЕГЛАМЕНТ" in text.upper()):
                        doc_type = "regulation"
                    elif("ПИСЬМО" in text_upper):
                        doc_type = "letter"
                    elif("ИСТОЧНИК" in text and "СТАТЬЯ" in text):
                        doc_type = "law"
                    status = "active"
                    if("УТРАТИЛ СИЛУ" in text.upper() or "УТРАТИЛА СИЛУ" in text.upper()):
                        status = "archived"

                    doc = {
                        "doc_id":doc_id,
                        "title":title,
                        "doc_type":doc_type,
                        "number":None,
                        "date":None,
                        "revision":None,
                        "status":status,
                        "source_path":filename,
                        "text":text
                    }
                    documents.append(doc)
    return documents
                
if(__name__ == "__main__"):
    docs = load_documents("../../../data/knowledge-base.zip")
    print(f"Прочиттано документов: {len(docs)}")
    for d in docs[:3]:
        print(f"Файл: {d['source_path']} | Тип: {d['doc_type']} | Статус: {d['status']}")
