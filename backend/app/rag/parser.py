"""Парсинг документов базы знаний.

Читает zip с .txt файлами и возвращает документы с метаданными:
тип, номер, дата, статус редакции и период действия.

Метаданные важны для поиска: по ним оркестратор отсекает устаревшие
редакции (вопрос «12 млн в августе 2024» требует редакцию 2023 года).
"""
from __future__ import annotations

import os
import re
import zipfile

MONTHS = {
    "января": 1, "январь": 1, "январе": 1,
    "февраля": 2, "февраль": 2, "феврале": 2,
    "марта": 3, "март": 3, "марте": 3,
    "апреля": 4, "апрель": 4, "апреле": 4,
    "мая": 5, "май": 5, "мае": 5,
    "июня": 6, "июнь": 6, "июне": 6,
    "июля": 7, "июль": 7, "июле": 7,
    "августа": 8, "август": 8, "августе": 8,
    "сентября": 9, "сентябрь": 9, "сентябре": 9,
    "октября": 10, "октябрь": 10, "октябре": 10,
    "ноября": 11, "ноябрь": 11, "ноябре": 11,
    "декабря": 12, "декабрь": 12, "декабре": 12,
}


def _first_date(text: str) -> str | None:
    """Первая дата в тексте -> ISO. Поддерживает 12.02.2024 и 12 февраля 2024."""
    low = text.lower().replace("ё", "е")

    exact = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", low)
    if exact:
        day, month, year = (int(part) for part in exact.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"

    words = re.search(r"\b(\d{1,2})\s+([а-я]+)\s+(20\d{2})", low)
    if words and words.group(2) in MONTHS:
        day = int(words.group(1))
        month = MONTHS[words.group(2)]
        year = int(words.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _document_type(filename: str, text: str) -> str:
    """Тип документа: сначала по папке в имени файла, потом по заголовку."""
    name = filename.lower()
    head = text[:600].lower()

    if name.startswith("законы__") or "источник:" in head:
        return "law"
    if "дополнительное соглашение" in head or "доп соглашение" in name:
        return "amendment"
    if "служебная записка" in head or "записка" in name:
        return "memo"
    if "письмо" in head or "письмо" in name:
        return "letter"
    if "регламент" in head:
        return "regulation"
    if "договор" in head:
        return "contract"
    return "other"


def _validity(text: str) -> tuple[str | None, str | None, str]:
    """Период действия редакции и статус.

    Возвращает (valid_from, valid_to, status).
    """
    low = text.lower().replace("ё", "е")
    valid_from = valid_to = None

    repealed = re.search(r"редакция утратила силу с (\d{2})\.(\d{2})\.(\d{4})", low)
    if repealed:
        valid_to = f"{repealed.group(3)}-{repealed.group(2)}-{repealed.group(1)}"

    active_since = re.search(r"действует с (\d{2})\.(\d{2})\.(\d{4})", low)
    if active_since:
        valid_from = f"{active_since.group(3)}-{active_since.group(2)}-{active_since.group(1)}"

    if valid_to and not valid_from:
        status = "repealed"
    else:
        status = "active"
    return valid_from, valid_to, status


def _title(text: str) -> str:
    """Заголовок: ищем строку с названием документа, а не шапку про отмену."""
    lines = [line.strip() for line in text.splitlines()[:15] if line.strip()]

    # 1) закон: строка вида "Статья 70. Испытание при приёме на работу"
    for line in lines:
        if re.match(r"^Статья\s+\d+(?:\.\d+)?\s*\.", line):
            return line

    # 2) шапка гаранта: "Источник: Статья 70. Испытание при приёме на работу..."
    for line in lines:
        if line.lower().startswith("источник:"):
            return line.split(":", 1)[1].strip()

    # 3) документ: строка с ключевым словом типа
    keywords = ("договор", "регламент", "письмо", "записка", "соглашение", "исх. №")
    for line in lines:
        low = line.lower()
        if low.startswith(("редакция утратила", "в связи с")):
            continue
        if any(word in low for word in keywords):
            return line

    # 4) иначе первая не-шапка
    for line in lines:
        low = line.lower()
        if not low.startswith(("редакция утратила", "url:", "дата выгрузки")):
            return line

    return lines[0] if lines else ""


def _number(title: str) -> str | None:
    match = re.search(r"№\s*([А-ЯA-Z0-9][А-ЯA-Z0-9./\-]*)", title)
    return match.group(1).rstrip(".") if match else None


def load_documents(kb_path: str) -> list[dict]:
    """Прочитать документы из zip (или папки) с метаданными."""
    documents: list[dict] = []

    if kb_path.endswith(".zip"):
        with zipfile.ZipFile(kb_path, "r") as archive:
            for filename in archive.namelist():
                if not filename.endswith(".txt"):
                    continue
                with archive.open(filename) as handle:
                    text = handle.read().decode("utf-8-sig", errors="replace")
                documents.append(_make_document(filename, text))
    else:
        for name in sorted(os.listdir(kb_path)):
            if not name.endswith(".txt"):
                continue
            with open(os.path.join(kb_path, name), encoding="utf-8-sig", errors="replace") as handle:
                documents.append(_make_document(name, handle.read()))

    return documents


def _revision(filename: str, text: str, valid_from: str | None, valid_to: str | None) -> str | None:
    """Год редакции.

    Порядок: год из имени файла («ред 2023») -> год начала действия.
    Год окончания действия НЕ подходит: у редакции 2023 он равен 2025,
    и обе редакции получили бы одинаковый год.
    """
    name = filename.lower()
    found = re.search(r"ред[^0-9]{0,3}(20\d{2})", name)
    if found:
        return found.group(1)
    found = re.search(r"(20\d{2})", name)
    if found:
        return found.group(1)
    if valid_from:
        return valid_from[:4]
    found = re.search(r"редакция от \d{2}\.\d{2}\.(20\d{2})", text.lower())
    return found.group(1) if found else None


def _make_document(filename: str, text: str) -> dict:
    valid_from, valid_to, status = _validity(text)
    revision = _revision(filename, text, valid_from, valid_to)

    title = _title(text)
    doc_type = _document_type(filename, text)
    # у двух редакций одного регламента заголовок одинаковый — различаем годом
    if doc_type == "regulation" and revision and "(ред." not in title.lower():
        title = f"{title} (ред. {revision})"

    return {
        "doc_id": filename.replace(".txt", "").replace(" ", "_").lower(),
        "title": title,
        "doc_type": doc_type,
        "number": _number(title),
        "date": _first_date(text),
        "revision": revision,
        "status": status,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "source_path": filename,
        "text": text,
    }


if __name__ == "__main__":
    docs = load_documents("data/knowledge-base.zip")
    print(f"документов: {len(docs)}")
    for doc in docs:
        print(f"  {doc['doc_type']:11} {doc['status']:9} {doc['valid_to'] or '-':11} {doc['title'][:55]}")
