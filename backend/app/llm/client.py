import os
import httpx
from typing import Optional, Dict, Any


class LLMClientError(Exception):
    """Базовое исключение для ошибок клиента LLM."""
    pass


class LLMClientTimeoutError(LLMClientError):
    """Превышено время ожидания ответа от LLM."""
    pass


class LLMClientAPIError(LLMClientError):
    """Ошибка при вызове LLM API."""
    pass


class LLMClient:
    def __init__(self):
        self.api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "30.0"))

        if not self.api_key:
            # Предупреждение или ошибка, если ключ отсутствует
            pass

    def generate(
            self,
            prompt: str,
            system: Optional[str] = None,
            temperature: float = 0.0,
            max_tokens: int = 1024
    ) -> str:
        """
        Синхронный метод генерации ответа через OpenAI-совместимый API с помощью httpx.
        """
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Безопасность: НИКОГДА не логируем self.api_key, prompt и system текст (персональные/юридические данные)

        max_retries = 3
        backoff_factor = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)

                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt < max_retries:
                            import time
                            time.sleep(backoff_factor * (2 ** (attempt - 1)))
                            continue

                    response.raise_for_status()
                    data = response.json()

                    # Извлекаем контент ответа
                    choices = data.get("choices", [])
                    if not choices:
                        raise LLMClientAPIError("Пустой ответ от LLM: отсутствуют choices.")

                    content = choices[0].get("message", {}).get("content", "")
                    return content.strip()

            except httpx.TimeoutException as e:
                if attempt == max_retries:
                    raise LLMClientTimeoutError("Таймаут при обращении к LLM API.") from e
            except httpx.HTTPStatusError as e:
                if attempt == max_retries:
                    raise LLMClientAPIError(f"HTTP ошибка LLM API: {e.response.status_code}") from e
            except Exception as e:
                if attempt == max_retries:
                    raise LLMClientError(f"Ошибка вызова LLM: {str(e)}") from e

        return ""