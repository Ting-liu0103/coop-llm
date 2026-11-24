"""LLM 代理人:組 prompt、呼叫模型、解析 JSON、失敗時重試。"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..games.base import Game, RoundRecord
from ..llm.base import LLMClient
from . import prompts
from .memory import build_chat_text, build_memory

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """從模型回覆裡取出第一個能解析的 JSON 物件(容忍前後有多餘文字或 code fence)。"""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if m:
        s = m.group(0)
        # 貪婪匹配可能多抓,從最後一個 } 往前逐步縮短
        while s:
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            j = s.rfind("}", 0, len(s) - 1)
            if j == -1:
                break
            s = s[: j + 1]
    return None


@dataclass
class ActResult:
    thought: str
    action: Any
    parse_ok: bool
    raw_text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    attempts: int
    prompt_hash: str


@dataclass
class ChatResult:
    message: str
    parse_ok: bool
    tokens_in: int
    tokens_out: int
    latency_ms: float


class Agent:
    def __init__(self, agent_id: str, seat: int, persona: str, reasoning: str, llm: LLMClient, game: Game,
                 *, lang: str = "zh", temperature: float | None = 0.7, max_tokens: int = 400,
                 max_retries: int = 2, chat_chars: int = 60, keep_last: int = 3):
        if persona not in prompts.PERSONAS[lang]:
            raise ValueError(f"未知人格 {persona!r}")
        if reasoning not in prompts.REASONING[lang]:
            raise ValueError(f"未知推理策略 {reasoning!r}")
        self.agent_id, self.seat = agent_id, seat
        self.persona, self.reasoning = persona, reasoning
        self.llm, self.game, self.lang = llm, game, lang
        self.temperature, self.max_tokens = temperature, max_tokens
        self.max_retries, self.chat_chars, self.keep_last = max_retries, chat_chars, keep_last
        self.system = prompts.build_system(agent_id, game.rules_text(lang), persona, lang)
        self._last_action: Any = None

    # ---- 共用 -------------------------------------------------------------
    def _context(self, records: list[RoundRecord], chat_log: list[dict], round_no: int) -> dict:
        return dict(
            round_no=round_no, rounds=self.game.rounds, state=self.game.public_state(self.lang),
            memory=build_memory(self.game, records, self.seat, self.lang, self.keep_last),
            chat=build_chat_text(chat_log, round_no, self.keep_last), lang=self.lang)

    def _others_last(self, records: list[RoundRecord]) -> float | None:
        if not records:
            return None
        last = records[-1].coop
        others = [c for i, c in enumerate(last) if i != self.seat]
        return sum(others) / len(others)

    # ---- 行動 -------------------------------------------------------------
    async def act(self, records: list[RoundRecord], chat_log: list[dict], round_no: int) -> ActResult:
        ctx = self._context(records, chat_log, round_no)
        user = prompts.build_round(reasoning=self.reasoning, action_prompt=self.game.action_prompt(self.lang), **ctx)
        phash = hashlib.sha1((self.system + user).encode()).hexdigest()[:10]
        messages = [{"role": "user", "content": user}]
        meta = {"kind": "action", "space": self.game.action_space(), "others_last": self._others_last(records)}

        tin = tout = 0
        lat = 0.0
        raw = ""
        for attempt in range(1, self.max_retries + 2):
            resp = await self.llm.complete(self.system, messages, temperature=self.temperature,
                                           max_tokens=self.max_tokens, meta=meta)
            tin, tout, lat, raw = tin + resp.tokens_in, tout + resp.tokens_out, lat + resp.latency_ms, resp.text
            obj = extract_json(resp.text)
            if obj is not None:
                action = self.game.parse_action(obj.get("action"))
                if action is not None:
                    self._last_action = action
                    return ActResult(str(obj.get("thought", "")), action, True, raw, tin, tout, lat, attempt, phash)
            # 解析失敗:把壞回覆與修正指令接在後面,再試一次
            messages = [{"role": "user", "content": user}, {"role": "assistant", "content": resp.text or "(空)"},
                        {"role": "user", "content": prompts.build_repair(self.game.action_prompt(self.lang), self.lang)}]
            meta = {**meta, "repair": True}

        # 重試用完:用上一回合動作(沒有的話用預設值),並記錄 parse_ok=False,不要默默吞掉
        fallback = self._last_action if self._last_action is not None else self.game.default_action()
        return ActResult("", fallback, False, raw, tin, tout, lat, self.max_retries + 1, phash)

    # ---- 聊天 -------------------------------------------------------------
    async def speak(self, records: list[RoundRecord], chat_log: list[dict], round_no: int) -> ChatResult:
        ctx = self._context(records, chat_log, round_no)
        user = prompts.build_chat(chars=self.chat_chars, **ctx)
        meta = {"kind": "chat", "space": self.game.action_space()}
        resp = await self.llm.complete(self.system, [{"role": "user", "content": user}],
                                       temperature=self.temperature, max_tokens=self.max_tokens, meta=meta)
        obj = extract_json(resp.text)
        msg = str(obj.get("message", "")).strip() if obj else ""
        ok = bool(msg)
        if not ok:  # 聊天失敗不重試,視為沉默,但會被記錄
            msg = ""
        return ChatResult(msg[: self.chat_chars], ok, resp.tokens_in, resp.tokens_out, resp.latency_ms)
