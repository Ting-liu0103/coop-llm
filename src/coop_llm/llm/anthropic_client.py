"""Anthropic Messages API 後端。`anthropic` 套件只在真的使用時才匯入。"""
from __future__ import annotations

import os
import time

from .base import LLMClient, LLMResponse, with_retries


class AnthropicClient(LLMClient):
    def __init__(self, model: str, *, api_key_env: str = "ANTHROPIC_API_KEY", name: str | None = None):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("請先 pip install anthropic") from e
        self._client = AsyncAnthropic(api_key=os.environ.get(api_key_env))
        self.model = model
        self.name = name or model

    async def complete(self, system, messages, *, temperature=0.7, max_tokens=400, meta=None) -> LLMResponse:
        kwargs = dict(model=self.model, system=system, messages=messages, max_tokens=max_tokens)
        if temperature is not None:  # 部分模型不接受自訂 temperature,設成 None 就不傳
            kwargs["temperature"] = temperature

        async def call():
            t0 = time.perf_counter()
            r = await self._client.messages.create(**kwargs)
            return r, (time.perf_counter() - t0) * 1000

        r, ms = await with_retries(call)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return LLMResponse(text, r.usage.input_tokens, r.usage.output_tokens, ms)
