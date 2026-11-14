"""Архив базы знаний: вложенные папки и служебный мусор.

Платформа может отдать zip с папками внутри («законы/ТК.txt»), а macOS
добавляет служебные файлы (__MACOSX, ._имя). Ни то, ни другое не должно
ломать разбор или попадать в базу как документ.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.chunker import chunk_documents  # noqa: E402
from app.rag.parser import load_documents  # noqa: E402

LAW = (
    "Источник: Статья 70. Испытание при приёме на работу\n"
    "1. При заключении трудового договора может быть предусмотрено условие об испытании работника.\n"
    "2. Срок испытания не может превышать трёх месяцев, а для главных бухгалтеров — шести месяцев.\n"
)
CONTRACT = (
    "ДОГОВОР АРЕНДЫ № А-99/2025\n"
    "1. Предмет договора\n"
    "1.1. Арендодатель передаёт Арендатору помещение общей площадью сто квадратных метров.\n"
)


def make_zip(tmp_path: Path) -> Path:
    path = tmp_path / "kb.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("законы/ТК_ст70.txt", LAW)
        archive.writestr("документы компании/Договор аренды.txt", CONTRACT)
        archive.writestr("__MACOSX/._Договор.txt", "мусор")
        archive.writestr("документы компании/._Договор аренды.txt", "скрытый файл")
        archive.writestr("документы компании/.DS_Store", "мусор")
    return path


def test_nested_folders_are_parsed(tmp_path: Path) -> None:
    docs = load_documents(str(make_zip(tmp_path)))
    assert len(docs) == 2, [doc["title"] for doc in docs]

    types = {doc["doc_type"] for doc in docs}
    assert types == {"law", "contract"}


def test_junk_files_are_skipped(tmp_path: Path) -> None:
    docs = load_documents(str(make_zip(tmp_path)))
    titles = " ".join(doc["title"] for doc in docs)
    assert "мусор" not in titles
    assert "скрытый файл" not in titles


def test_doc_id_has_no_path_separators(tmp_path: Path) -> None:
    docs = load_documents(str(make_zip(tmp_path)))
    for doc in docs:
        assert "/" not in doc["doc_id"] and "\\\\" not in doc["doc_id"]
        assert " " not in doc["doc_id"]


def test_chunks_from_nested_documents(tmp_path: Path) -> None:
    chunks = chunk_documents(load_documents(str(make_zip(tmp_path))))
    assert chunks
    assert any(chunk["clause"] == "ст. 70" for chunk in chunks)
    assert any(chunk["clause"] == "1.1" for chunk in chunks)
