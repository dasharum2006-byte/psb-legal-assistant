"""Таймауты должны быть согласованы: клиент LLM укладывается в REQUEST_TIMEOUT.

Иначе API отдаёт 504 раньше, чем клиент успевает повторить запрос
(было: клиент 3 попытки × 30 с = 93 с против REQUEST_TIMEOUT = 30 с).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402


def test_llm_budget_fits_request_timeout() -> None:
    client = LLMClient()
    backoff = sum(2 ** (attempt - 1) for attempt in range(1, client.max_retries))
    worst_case = client.timeout * client.max_retries + backoff

    assert worst_case < settings.REQUEST_TIMEOUT, (
        f"худший случай клиента {worst_case} с не влезает в REQUEST_TIMEOUT "
        f"{settings.REQUEST_TIMEOUT} с — API отдаст 504"
    )


def test_request_timeout_is_reasonable() -> None:
    assert settings.REQUEST_TIMEOUT >= 45
