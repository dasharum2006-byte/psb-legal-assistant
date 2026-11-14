"""Нарезка: трёхуровневые пункты, одноуровневые, разделы, законы.

Раньше регулярка ловила только двухуровневую нумерацию: подпункт «3.1.2»
приклеивался к пункту 3.1, а пункты вида «1.» и заголовки разделов терялись.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location("sol", ROOT / "submit/solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)


def _doc(text: str, doc_type: str = "contract", title: str = "ДОГОВОР № 1") -> dict:
    return {"file": "x.txt", "title": title, "number": "1", "doc_type": doc_type,
            "repealed": False, "text": text}


def clauses(chunks: list[dict]) -> set[str]:
    return {chunk["clause"] for chunk in chunks}


def test_three_level_clause_is_separate_chunk() -> None:
    text = (
        "ДОГОВОР № 1\n"
        "1. Предмет договора\n"
        "1.1. Поставщик передаёт товар покупателю в течение десяти дней.\n"
        "1.1.1. Срок может быть продлён по письменному соглашению сторон договора.\n"
        "1.2. Оплата производится в течение тридцати дней с даты поставки товара.\n"
    )
    chunks = sol.chunk_documents([_doc(text)])
    assert "1.1.1" in clauses(chunks), "подпункт 1.1.1 должен быть отдельным чанком"
    assert "1.1" in clauses(chunks)


def test_single_level_numbering_supported() -> None:
    text = (
        "ДОГОВОР № 2\n"
        "1. Исполнитель обязуется оказать услуги по разработке программного обеспечения "
        "в соответствии с техническим заданием заказчика и в установленные сроки.\n"
        "2. Заказчик обязуется принять и оплатить оказанные услуги в течение пяти дней.\n"
    )
    chunks = sol.chunk_documents([_doc(text)])
    assert {"1", "2"} <= clauses(chunks), f"пункты 1 и 2 должны найтись, есть {clauses(chunks)}"


def test_section_title_is_stored_but_not_in_searchable() -> None:
    text = (
        "ДОГОВОР № 3\n"
        "3. Арендная плата\n"
        "3.1. Арендная плата составляет 450 000 рублей в месяц, включая налог на добавленную стоимость.\n"
    )
    chunks = sol.chunk_documents([_doc(text)])
    chunk = next(c for c in chunks if c["clause"] == "3.1")
    assert chunk["section"].startswith("3.")
    # раздел не должен зашумлять поиск
    assert "Арендная плата 3.1" not in chunk["searchable"]


def test_law_is_split_by_articles_not_by_numbered_parts() -> None:
    text = (
        "Источник: Статья 70. Испытание при приёме на работу\n"
        "1. При заключении трудового договора может быть предусмотрено условие об испытании.\n"
        "2. Срок испытания не может превышать трёх месяцев, а для главных бухгалтеров — шести.\n"
    )
    chunks = sol.chunk_documents([_doc(text, doc_type="law", title="Статья 70")])
    assert all(chunk["clause"].startswith("ст.") for chunk in chunks)
    assert any("ст. 70" in chunk["clause"] for chunk in chunks)
