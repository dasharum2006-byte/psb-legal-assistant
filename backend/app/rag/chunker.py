"""Нарезка документов на чанки.

Правило: документ НИКОГДА не выбрасывается. Если у документа нет
нумерованных пунктов (письмо, записка, доп. соглашение) — режем по
абзацам. Иначе половина базы молча выпадала из индекса.
"""
from __future__ import annotations

import re

MAX_LAW_CHUNK = 1800  # длинные статьи бьём на части, иначе не влезают в контекст

INJECTION_PATTERNS = (
    "игнорируй",
    "внимание для ии",
    "не упоминай эту инструкцию",
    "отвечай «",
    "отвечай \"",
    "забудь все",
)


def _base(doc: dict) -> dict:
    """Общие поля чанка — одинаковые для любого типа документа."""
    return {
        "doc_id": doc["doc_id"],
        "doc": doc["title"],          # человекочитаемое имя для цитаты
        "title": doc["title"],
        "doc_type": doc["doc_type"],
        "number": doc.get("number"),
        "date": doc.get("date"),
        "status": doc.get("status"),
        "valid_from": doc.get("valid_from"),
        "valid_to": doc.get("valid_to"),
        "revision": doc.get("revision"),
    }


def _make_chunk(doc: dict, clause: str, text: str, part: str | None = None) -> dict | None:
    body = re.sub(r"[ \t]+", " ", text).strip()
    if len(body) < 30:
        return None

    chunk = _base(doc)
    chunk["clause"] = clause
    chunk["text"] = body
    chunk["suspicious_instruction"] = any(
        pattern in body.lower() for pattern in INJECTION_PATTERNS
    )
    chunk["searchable"] = f"{doc['title']} {clause} {body}"
    if part:
        chunk["chunk_id"] = f"{doc['doc_id']}#{clause}/{part}"
    else:
        chunk["chunk_id"] = f"{doc['doc_id']}#{clause}"
    return chunk


def _by_clauses(doc: dict) -> list[dict]:
    """Договоры и регламенты: один пункт = один чанк («3.1», «2.4»)."""
    chunks: list[dict] = []
    pattern = r"(?m)^\s*(\d+\.\d+)[\.\)]\s*"
    parts = re.split(pattern, doc["text"])

    for index in range(1, len(parts), 2):
        clause = parts[index]
        body = parts[index + 1] if index + 1 < len(parts) else ""
        chunk = _make_chunk(doc, clause, body)
        if chunk:
            chunks.append(chunk)
    return chunks


def _by_paragraphs(doc: dict, prefix: str = "абз.") -> list[dict]:
    """Всё, где нет нумерованных пунктов: письма, записки, соглашения."""
    chunks: list[dict] = []
    blocks = [block for block in re.split(r"\n\s*\n", doc["text"]) if block.strip()]

    number = 0
    for block in blocks:
        # шапка garant.ru в законах не нужна как отдельный чанк
        if block.strip().lower().startswith(("источник:", "url:", "дата выгрузки:")):
            continue
        number += 1
        chunk = _make_chunk(doc, f"{prefix} {number}", block)
        if chunk:
            chunks.append(chunk)
    return chunks


def _by_articles(doc: dict) -> list[dict]:
    """Законы: статья = чанк; слишком длинные статьи режем по абзацам."""
    chunks: list[dict] = []
    pattern = r"(?m)^\s*(Статья\s+\d+(?:\.\d+)?)"
    parts = re.split(pattern, doc["text"])

    for index in range(1, len(parts), 2):
        article = parts[index].strip().replace("Статья", "ст.")
        body = parts[index + 1] if index + 1 < len(parts) else ""

        if len(body) <= MAX_LAW_CHUNK:
            chunk = _make_chunk(doc, article, body)
            if chunk:
                chunks.append(chunk)
            continue

        part_number = 0
        for block in re.split(r"\n\s*\n", body):
            if len(block.strip()) < 30:
                continue
            part_number += 1
            chunk = _make_chunk(doc, article, block, part=str(part_number))
            if chunk:
                chunks.append(chunk)
    return chunks


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Разложить документы на чанки. Ни один документ не теряется."""
    chunks: list[dict] = []
    lost: list[str] = []

    for doc in docs:
        doc_type = doc.get("doc_type")

        if doc_type in ("contract", "regulation"):
            produced = _by_clauses(doc)
            if not produced:                      # нет пунктов — не теряем документ
                produced = _by_paragraphs(doc)
        elif doc_type == "law":
            produced = _by_articles(doc)
        else:                                     # amendment, letter, memo, other
            produced = _by_paragraphs(doc)

        if not produced:
            lost.append(doc["title"])
        chunks.extend(produced)

    if lost:
        print(f"⚠️ документы без чанков ({len(lost)}): {lost}")
    return chunks


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "backend")
    from app.rag.parser import load_documents

    docs = load_documents("data/knowledge-base.zip")
    chunks = chunk_documents(docs)

    by_doc: dict[str, int] = {}
    for chunk in chunks:
        by_doc[chunk["doc"]] = by_doc.get(chunk["doc"], 0) + 1

    print(f"документов: {len(docs)}, из них с чанками: {len(by_doc)}")
    print(f"чанков: {len(chunks)}")
    for title, count in sorted(by_doc.items(), key=lambda item: -item[1]):
        print(f"  {count:4}  {title[:60]}")
