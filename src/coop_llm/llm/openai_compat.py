"""OpenAI 相容後端。

同一份程式可以接 OpenAI 的 API,也可以接本機或雲端的開源模型服務
(只要該服務提供 OpenAI 相容的 /v1/chat/completions,並設定 base_url)。
`openai` 套件只在真的使用時才匯入,所以沒安裝也不影響 mock 與測試。
"""
from __future__ import annotations

import os
import time

from .base import LLMClient, LLMResponse, RateLimiter, with_retries


def _error_code(e: BaseException) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            return str(err.get("code") or err.get("type") or "")
    return ""


def is_transient(e: BaseException) -> bool:
    """只重試暫時性錯誤:限速、排隊逾時、連線問題、5xx。金鑰錯誤、額度用完、請求本身有問題都不重試。"""
    status = getattr(e, "status_code", None)
    if status is None:  # 沒有 HTTP 狀態 = 連線/逾時類
        return e.__class__.__name__ in ("APIConnectionError", "APITimeoutError")
    if status == 429:
        return _error_code(e) != "insufficient_quota"  # 本期額度用完:等重置才會好,重試只是浪費
    return status >= 500


def retry_after(e: BaseException) -> float | None:
    resp = getattr(e, "response", None)
    try:
        v = resp.headers.get("retry-after") if resp is not None else None
        return min(float(v), 120.0) if v is not None else None
    except (TypeError, ValueError):
        return None


class OpenAICompatClient(LLMClient):
    def __init__(self, model: str, *, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 name: str | None = None, extra_body: dict | None = None, rpm: int | None = None,
                 max_concurrent: int | None = None, max_tokens_param: str = "max_tokens"):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("請先 pip install openai") from e
        # 本機服務常常不需要 key,給一個佔位值即可
        api_key = os.environ.get(api_key_env) or "EMPTY"
        # max_retries=0:所有重試都經過我們的限速器與退避,不讓 SDK 偷偷多送請求
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.limiter = RateLimiter(rpm=rpm, max_concurrent=max_concurrent)
        self.max_tokens_param = max_tokens_param
        self.model = model
        self.name = name or model
        self.extra_body = extra_body

    async def complete(self, system, messages, *, temperature=0.7, max_tokens=400, meta=None) -> LLMResponse:
        msgs = [{"role": "system", "content": system}, *messages]
        kwargs = {"model": self.model, "messages": msgs, self.max_tokens_param: max_tokens}
        if temperature is not None:  # 部分模型不接受自訂 temperature,設成 None 就不傳
            kwargs["temperature"] = temperature
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        async def call():
            async with self.limiter:
                t0 = time.perf_counter()
                r = await self._client.chat.completions.create(**kwargs)
                return r, (time.perf_counter() - t0) * 1000

        r, ms = await with_retries(call, attempts=6, should_retry=is_transient, delay_for=retry_after)
        u = r.usage
        return LLMResponse(r.choices[0].message.content or "", getattr(u, "prompt_tokens", 0) or 0,
                           getattr(u, "completion_tokens", 0) or 0, ms)
