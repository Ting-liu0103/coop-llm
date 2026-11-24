import asyncio

import pytest

from coop_llm.agents import Agent, extract_json
from coop_llm.agents.prompts import PERSONAS, REASONING
from coop_llm.games import IPD, PGG
from coop_llm.llm.base import LLMResponse
from coop_llm.llm.mock import MockLLM


def run(coro):
    return asyncio.run(coro)


# ---- JSON 解析 ----------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ('{"thought": "ok", "action": 5}', {"thought": "ok", "action": 5}),
    ('好的,我的回覆:\n```json\n{"thought": "t", "action": 7}\n```', {"thought": "t", "action": 7}),
    ('前綴 {"action": 3} 後綴', {"action": 3}),
    ('{"thought": "含大括號 {x}", "action": 9}', {"thought": "含大括號 {x}", "action": 9}),
])
def test_extract_json_ok(text, expected):
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "沒有 JSON", "{壞掉的 json", "[1,2,3]"])
def test_extract_json_none(text):
    assert extract_json(text) is None


# ---- Agent --------------------------------------------------------------
def make_agent(llm, game=None, **kw):
    game = game or PGG(seed=1)
    return Agent("A1", 0, kw.pop("persona", "neutral"), kw.pop("reasoning", "plain"), llm, game, **kw), game


def test_act_parses_valid_json():
    a, g = make_agent(MockLLM("cooperate"))
    r = run(a.act([], [], 1))
    assert r.parse_ok and r.action == 20 and r.attempts == 1


def test_act_retries_then_succeeds():
    class Once(MockLLM):
        async def complete(self, system, messages, **kw):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse("不是 JSON", 10, 5, 1)
            return LLMResponse('{"thought": "t", "action": 4}', 10, 5, 1)

    a, _ = make_agent(Once())
    r = run(a.act([], [], 1))
    assert r.parse_ok and r.action == 4 and r.attempts == 2
    assert r.tokens_in == 20  # 兩次呼叫的 token 都有累計


def test_act_out_of_range_action_is_retried():
    class Bad(MockLLM):
        async def complete(self, system, messages, **kw):
            self.calls += 1
            return LLMResponse('{"thought": "t", "action": 99}', 1, 1, 1)

    a, g = make_agent(Bad(), max_retries=2)
    r = run(a.act([], [], 1))
    assert not r.parse_ok and r.attempts == 3
    assert r.action == g.default_action()  # 沒有上一回合動作 → 預設值


def test_fallback_uses_last_action_when_available():
    class Seq(MockLLM):
        async def complete(self, system, messages, **kw):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse('{"thought": "t", "action": 7}', 1, 1, 1)
            return LLMResponse("壞掉", 1, 1, 1)

    a, g = make_agent(Seq(), max_retries=1)
    assert run(a.act([], [], 1)).action == 7
    r2 = run(a.act([], [], 2))
    assert not r2.parse_ok and r2.action == 7  # 用上一回合的動作


def test_flaky_mock_marks_parse_ok_after_repair():
    a, _ = make_agent(MockLLM("flaky"))
    r = run(a.act([], [], 1))
    assert r.parse_ok and r.attempts == 2


def test_prompt_contains_persona_and_reasoning_text():
    seen = {}

    class Spy(MockLLM):
        async def complete(self, system, messages, **kw):
            seen["system"], seen["user"] = system, messages[0]["content"]
            return await super().complete(system, messages, **kw)

    a, _ = make_agent(Spy("cooperate"), persona="selfish", reasoning="universalization")
    run(a.act([], [], 1))
    assert PERSONAS["zh"]["selfish"] in seen["system"]
    assert REASONING["zh"]["universalization"] in seen["user"]
    assert "第 1 回合" in seen["user"] and "共 10 回合" in seen["user"]


def test_prompt_never_leaks_other_persona_or_game_name():
    seen = []

    class Spy(MockLLM):
        async def complete(self, system, messages, **kw):
            seen.append(system + messages[0]["content"])
            return await super().complete(system, messages, **kw)

    a, _ = make_agent(Spy("cooperate"), game=IPD(seed=2), persona="neutral")
    run(a.act([], [], 1))
    blob = "".join(seen)
    assert PERSONAS["zh"]["selfish"] not in blob and "囚徒" not in blob


def test_memory_window_keeps_last_3_rounds_verbatim():
    g = PGG(seed=1, rounds=10)
    for _ in range(5):
        g.step([5, 6, 7, 8])
    seen = {}

    class Spy(MockLLM):
        async def complete(self, system, messages, **kw):
            seen["user"] = messages[0]["content"]
            return await super().complete(system, messages, **kw)

    a, _ = make_agent(Spy("cooperate"), game=g)
    run(a.act(g.history, [], 6))
    u = seen["user"]
    assert "摘要" in u  # 第 1–2 回合被壓縮
    assert "第 3 回合:" in u and "第 5 回合:" in u
    assert "第 1 回合:" not in u and "第 2 回合:" not in u


def test_speak_truncates_and_handles_garbage():
    class Talker(MockLLM):
        async def complete(self, system, messages, **kw):
            return LLMResponse('{"message": "' + "好" * 100 + '"}', 5, 5, 1)

    a, _ = make_agent(Talker(), chat_chars=60)
    r = run(a.speak([], [], 1))
    assert r.parse_ok and len(r.message) == 60

    a2, _ = make_agent(MockLLM("flaky"))
    r2 = run(a2.speak([], [], 1))
    assert r2.message == "我們一起合作吧。" or not r2.parse_ok  # flaky 的 chat 回覆可能是亂碼
