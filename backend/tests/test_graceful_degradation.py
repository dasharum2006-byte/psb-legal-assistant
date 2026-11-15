"""Отказоустойчивая загрузка: библиотека не падает, CLI падает громко.

Пустая или битая база знаний не должна превращаться в отсутствие
submission.csv — лучше валидный файл с отказами.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(REPO / "submit"))

from app.rag import parser, retriever  # noqa: E402
from app.rag.orchestrator import ask  # noqa: E402


def make_zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def test_parser_returns_empty_list_for_junk_only_archive(tmp_path: Path) -> None:
    path = make_zip(tmp_path / "kb.zip", {
        "__MACOSX/._doc.txt": b"junk",
        ".DS_Store": b"junk",
        "thumbs.db": b"junk",
    })
    assert parser.load_documents(str(path)) == []


def test_parser_skips_broken_file_but_keeps_others(tmp_path: Path, monkeypatch) -> None:
    path = make_zip(tmp_path / "kb.zip", {
        "good.txt": "Договор аренды. 1.1. Плата 100 рублей.".encode(),
        "broken.txt": b"\xff\xfe\x00broken",
    })

    original = parser._make_document

    def flaky(filename, text):
        if filename == "broken.txt":
            raise ValueError("имитация битого файла")
        return original(filename, text)

    monkeypatch.setattr(parser, "_make_document", flaky)
    documents = parser.load_documents(str(path))
    assert [doc["doc_id"] for doc in documents] == ["good"]


def test_empty_index_search_returns_empty_list() -> None:
    retriever.build_index([])
    assert retriever.search("любой вопрос", 8) == []


def test_orchestrator_refuses_on_empty_index() -> None:
    class EmptyRetriever:
        def search(self, query, k=40, filters=None):
            return []

    result = ask("какая арендная плата?", [], EmptyRetriever())
    assert result["refusal"] is True
    assert result["answer"]
    assert result["missing_info"]


def test_submit_load_documents_returns_empty_list(tmp_path: Path) -> None:
    import solution

    path = make_zip(tmp_path / "kb.zip", {"__MACOSX/._x.txt": b"junk", ".DS_Store": b"junk"})
    assert solution.load_documents(path) == []


def test_submit_retriever_handles_empty_base() -> None:
    import solution

    empty = solution.Retriever([])
    assert empty.search("какая арендная плата?", 8) == []


def test_build_index_cli_fails_loudly_on_empty_archive(tmp_path: Path) -> None:
    """Скрипт ручной сборки должен падать, а не писать пустой индекс."""
    path = make_zip(tmp_path / "kb.zip", {".DS_Store": b"junk"})
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/build_index.py"), str(path)],
        capture_output=True, text=True, timeout=120, cwd=REPO,
    )
    assert result.returncode == 1
    assert "нет читаемых документов" in result.stderr
