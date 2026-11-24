"""模型後端的統一介面。換模型只要換實作,賽局與代理人程式不用動。"""
from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from collections import deque
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


class RateLimiter:
    """限制每個時間窗內的請求數(rpm)與同時進行的請求數(max_concurrent)。

    `async with limiter:` 包住「每一次」實際送出的請求(含重試),這樣重試也會算進額度。
    """

    def __init__(self, rpm: int | None = None, max_concurrent: int | None = None, window: float = 60.0):
        self.rpm, self.window = rpm, window
        self._times: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    async def __aenter__(self):
        if self._sem:
            await self._sem.acquire()
        if self.rpm:
            async with self._lock:  # 一次只讓一個等待者排隊,先到先得
                while True:
                    now = time.monotonic()
                    while self._times and now - self._times[0] >= self.window:
                        self._times.popleft()
                    if len(self._times) < self.rpm:
                        self._times.append(now)
                        break
                    await asyncio.sleep(self.window - (now - self._times[0]) + 0.01)
        return self

    async def __aexit__(self, *exc):
        if self._sem:
            self._sem.release()


async def with_retries(fn: Callable[[], Awaitable[T]], *, attempts: int = 5, base_delay: float = 1.0,
                       retry_on: tuple[type[BaseException], ...] = (Exception,),
                       should_retry: Callable[[BaseException], bool] | None = None,
                       delay_for: Callable[[BaseException], float | None] | None = None) -> T:
    """指數退避重試,給 API 暫時性錯誤(限速、逾時)用。最後一次失敗會把例外丟出去。

    should_retry(e) 回傳 False 的錯誤(例如金鑰錯誤、額度用完)立刻丟出,不白白重試。
    delay_for(e) 回傳秒數時(例如伺服器給的 Retry-After)優先使用。
    """
    for i in range(attempts):
        try:
            return await fn()
        except retry_on as e:
            if i == attempts - 1 or (should_retry is not None and not should_retry(e)):
                raise
            hint = delay_for(e) if delay_for else None
            await asyncio.sleep(hint if hint is not None else base_delay * (2 ** i) * (0.5 + random.random()))
    raise RuntimeError("unreachable")
