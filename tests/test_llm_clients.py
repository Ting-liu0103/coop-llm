"""限速器、重試與 OpenAI 相容後端(不連網路)。"""
import asyncio
import time

import httpx
import openai
import pytest

from coop_llm.llm import RateLimiter, make_client, with_retries
from coop_llm.llm.openai_compat import is_transient, retry_after


def run(c):
    return asyncio.run(c)


def api_error(cls, status, body=None, headers=None):
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    resp = httpx.Response(status, request=req, headers=headers or {})
    return cls("err", response=resp, body=body)


def test_rate_limiter_caps_requests_per_window():
    async def go():
        lim, t0, stamps = RateLimiter(rpm=2, window=0.3), time.monotonic(), []
        for _ in range(5):
            async with lim:
                stamps.append(time.monotonic() - t0)
        return stamps
    s = run(go())
    assert s[1] < 0.2 and s[2] >= 0.29 and s[4] >= 0.58   # 每 0.3 秒最多 2 個


def test_rate_limiter_caps_concurrency():
    live, peak = 0, 0

    async def worker(lim):
        nonlocal live, peak
        async with lim:
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.02)
            live -= 1

    async def go():
        lim = RateLimiter(max_concurrent=2)
        await asyncio.gather(*(worker(lim) for _ in range(8)))
    run(go())
    assert peak == 2


def test_with_retries_respects_should_retry_and_delay_hint():
    calls = []

    async def bad():
        calls.append(1)
        raise ValueError("fatal")
    with pytest.raises(ValueError):
        run(with_retries(bad, should_retry=lambda e: False, base_delay=0.01))
    assert len(calls) == 1  # 不該重試的錯誤只送一次

    n = []

    async def flaky():
        n.append(1)
        if len(n) < 3:
            raise RuntimeError("temp")
        return "ok"
    t0 = time.monotonic()
    assert run(with_retries(flaky, delay_for=lambda e: 0.05)) == "ok"
    assert len(n) == 3 and 0.1 <= time.monotonic() - t0 < 1.0


def test_transient_classification():
    assert is_transient(api_error(openai.RateLimitError, 429, {"error": {"code": "rate_limit_exceeded"}}))
    assert is_transient(api_error(openai.RateLimitError, 429, {"error": {"code": "queue_timeout"}}))
    assert not is_transient(api_error(openai.RateLimitError, 429, {"error": {"code": "insufficient_quota"}}))
    assert not is_transient(api_error(openai.AuthenticationError, 401, {"error": {"code": "invalid_api_key"}}))
    assert not is_transient(api_error(openai.PermissionDeniedError, 403, {"error": {"code": "model_not_allowed"}}))
    assert is_transient(api_error(openai.InternalServerError, 503))
    assert is_transient(api_error(openai.InternalServerError, 502))
    assert is_transient(openai.APIConnectionError(request=httpx.Request("POST", "https://x")))


def test_retry_after_header():
    e = api_error(openai.RateLimitError, 429, None, {"retry-after": "7"})
    assert retry_after(e) == 7.0
    assert retry_after(api_error(openai.RateLimitError, 429)) is None
    assert retry_after(api_error(openai.RateLimitError, 429, None, {"retry-after": "9999"})) == 120.0


def test_client_builds_with_mercury_style_config(monkeypatch):
    monkeypatch.setenv("MERCURY_API_KEY", "sk-mercury-test")
    c = make_client({"name": "m", "provider": "openai_compat", "model": "gpt-oss-120b",
                     "base_url": "https://air.cgu.edu.tw/mercuryapi/v1", "api_key_env": "MERCURY_API_KEY",
                     "rpm": 27, "max_concurrent": 2})
    assert c.limiter.rpm == 27 and c._client.max_retries == 0 and c.max_tokens_param == "max_tokens"


def test_client_complete_goes_through_limiter_and_parses(monkeypatch):
    monkeypatch.setenv("K", "x")
    c = make_client({"name": "m", "provider": "openai_compat", "model": "mm", "api_key_env": "K",
                     "max_concurrent": 1, "max_tokens_param": "max_completion_tokens"})
    seen = {}

    class Fake:
        class chat:
            class completions:
                @staticmethod
                async def create(**kw):
                    seen.update(kw)
                    msg = type("M", (), {"content": '{"thought":"t","action":3}'})
                    ch = type("C", (), {"message": msg})
                    us = type("U", (), {"prompt_tokens": 11, "completion_tokens": 5})
                    return type("R", (), {"choices": [ch], "usage": us})
    c._client = Fake
    r = run(c.complete("sys", [{"role": "user", "content": "hi"}], max_tokens=123))
    assert r.text.startswith("{") and (r.tokens_in, r.tokens_out) == (11, 5)
    assert seen["max_completion_tokens"] == 123 and "max_tokens" not in seen
