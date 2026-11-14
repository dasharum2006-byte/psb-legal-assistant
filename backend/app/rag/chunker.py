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


def _make_chunk(
    doc: dict,
    clause: str,
    text: str,
    part: str | None = None,
    section: str = "",
) -> dict | None:
    body = re.sub(r"[ \t]+", " ", text).strip()
    if len(body) < 30:
        return None

    chunk = _base(doc)
    chunk["clause"] = clause
    chunk["text"] = body
    chunk["suspicious_instruction"] = any(
        pattern in body.lower() for pattern in INJECTION_PATTERNS
    )
    chunk["section"] = section
    chunk["searchable"] = f"{doc['title']} {section} {clause} {body}".strip()
    if part:
        chunk["chunk_id"] = f"{doc['doc_id']}#{clause}/{part}"
    else:
        chunk["chunk_id"] = f"{doc['doc_id']}#{clause}"
    return chunk


HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[А-ЯЁ][^\n]{2,90}$")


def _strip_headings(body: str) -> str:
    """Убрать строки-заголовки разделов из тела пункта.

    Без этого пункт 5.1 заканчивается заголовком «6. Изменение и
    расторжение договора», и он попадает в поиск и в промпт.
    """
    lines = body.splitlines()
    while lines and HEADING_RE.match(lines[0]):
        lines.pop(0)
    while lines and HEADING_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines)


def _by_clauses(doc: dict) -> list[dict]:
    """Договоры и регламенты: один пункт = один чанк («3.1», «2.4»)."""
    chunks: list[dict] = []
    text = doc["text"]
    sections = [
        (match.start(), f"{match.group(1)}. {match.group(2).strip()}")
        for match in re.finditer(r"(?m)^\s*(\d+)\.\s+([А-ЯЁ][^\n]{2,60})$", text)
    ]

    # многоуровневые пункты (3.1, 3.1.2); если их нет — одноуровневые (1., 2.)
    matches = list(re.finditer(r"(?m)^\s*(\d+(?:\.\d+)+)[\.\)]?\s+", text))
    single_level = False
    if not matches:
        matches = list(re.finditer(r"(?m)^\s*(\d+)[\.\)]\s+", text))
        single_level = bool(matches)

    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = _strip_headings(text[match.end():end])
        if len(body.strip()) < (120 if single_level else 30):
            continue

        section = ""
        for start, name in sections:
            if start <= match.start():
                section = name
            else:
                break

        chunk = _make_chunk(doc, match.group(1), body, section=section)
        if chunk:
            chunks.append(chunk)
    return chunks


def _by_paragraphs(doc: dict, prefix: str = "абз.") -> list[dict]:
    """Всё, где нет нумерованных пунктов: письма, записки, соглашения."""
    chunks: list[dict] = []
    blocks = [block for block in re.split(r"\n\s*\n", doc["text"]) if block.strip()]

    number = 0
    for block in blocks:
        block = _strip_headings(block)
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
            # абзац может быть длиннее лимита (в законах часто нет пустых
            # строк) — режем дополнительно по предложениям
            for piece in _split_by_sentences(_strip_headings(block), MAX_LAW_CHUNK):
                part_number += 1
                chunk = _make_chunk(doc, article, piece, part=str(part_number))
                if chunk:
                    chunks.append(chunk)
    return chunks


def _split_by_sentences(block: str, limit: int) -> list[str]:
    """Разрезать длинный блок по границам предложений."""
    block = block.strip()
    if len(block) <= limit:
        return [block]

    sentences = [s.strip() for s in re.split(r"(?<=[.;!?])\s+", block) if s.strip()]
    pieces, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return [piece for piece in pieces if len(piece) >= 30]


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
