"""OpenAI 相容後端。

同一份程式可以接 OpenAI 的 API,也可以接本機或雲端的開源模型服務
(只要該服務提供 OpenAI 相容的 /v1/chat/completions,並設定 base_url)。
`openai` 套件只在真的使用時才匯入,所以沒安裝也不影響 mock 與測試。
"""
from __future__ import annotations

import os
import time

from .base import LLMClient, LLMResponse, with_retries


class OpenAICompatClient(LLMClient):
    def __init__(self, model: str, *, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 name: str | None = None, extra_body: dict | None = None):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("請先 pip install openai") from e
        # 本機服務常常不需要 key,給一個佔位值即可
        api_key = os.environ.get(api_key_env) or "EMPTY"
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.name = name or model
        self.extra_body = extra_body

    async def complete(self, system, messages, *, temperature=0.7, max_tokens=400, meta=None) -> LLMResponse:
        msgs = [{"role": "system", "content": system}, *messages]
        kwargs = dict(model=self.model, messages=msgs, max_tokens=max_tokens)
        if temperature is not None:  # 部分模型不接受自訂 temperature,設成 None 就不傳
            kwargs["temperature"] = temperature
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        async def call():
            t0 = time.perf_counter()
            r = await self._client.chat.completions.create(**kwargs)
            return r, (time.perf_counter() - t0) * 1000

        r, ms = await with_retries(call)
        u = r.usage
        return LLMResponse(r.choices[0].message.content or "", getattr(u, "prompt_tokens", 0) or 0,
                           getattr(u, "completion_tokens", 0) or 0, ms)
