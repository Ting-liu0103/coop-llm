"""模型後端的統一介面。換模型只要換實作,賽局與代理人程式不用動。"""
from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


@dataclass
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


class LLMClient(ABC):
    """所有後端都實作 complete()。meta 只給 mock 用,真實後端必須忽略它。"""

    name: str = "llm"

    @abstractmethod
    async def complete(self, system: str, messages: list[dict], *, temperature: float | None = 0.7,
                       max_tokens: int = 400, meta: dict | None = None) -> LLMResponse:
        ...


async def with_retries(fn: Callable[[], Awaitable[T]], *, attempts: int = 5, base_delay: float = 1.0,
                       retry_on: tuple[type[BaseException], ...] = (Exception,)) -> T:
    """指數退避重試,給 API 暫時性錯誤(限速、逾時)用。最後一次失敗會把例外丟出去。"""
    for i in range(attempts):
        try:
            return await fn()
        except retry_on:
            if i == attempts - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** i) * (0.5 + random.random()))
    raise RuntimeError("unreachable")
