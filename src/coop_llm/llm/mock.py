"""不需要網路與 API key 的假模型,用來跑單元測試與 dry-run 全流程。

policy:
  random      每個動作在合法範圍內隨機
  cooperate   永遠合作
  defect      永遠背叛
  conditional 第一回合合作,之後模仿(接近)上一回合其他人的行為(由 meta['others_last'] 提供)
  flaky       前幾次回覆故意輸出無法解析的文字,用來測試重試與 parse_ok=False
"""
from __future__ import annotations

import json
import random

from .base import LLMClient, LLMResponse


class MockLLM(LLMClient):
    def __init__(self, policy: str = "random", seed: int = 0, flaky_every: int = 0, name: str = "mock"):
        self.policy = policy
        self.rng = random.Random(seed)
        self.flaky_every = flaky_every
        self.calls = 0
        self.name = name

    async def complete(self, system, messages, *, temperature=0.7, max_tokens=400, meta=None) -> LLMResponse:
        self.calls += 1
        meta = meta or {}
        if self.policy == "flaky" or (self.flaky_every and self.calls % self.flaky_every == 0):
            if not meta.get("repair"):
                return LLMResponse("我覺得這很難決定,讓我想想……", 50, 10, 1.0)
        if meta.get("kind") == "chat":
            return LLMResponse(json.dumps({"message": "我們一起合作吧。"}, ensure_ascii=False), 80, 12, 1.0)

        space = meta.get("space") or {}
        coop, defect = space.get("cooperate"), space.get("defect")
        if self.policy in ("cooperate", "flaky"):
            action = coop
        elif self.policy == "defect":
            action = defect
        elif self.policy == "conditional":
            last = meta.get("others_last")  # 0~1 的合作分數,None 代表第一回合
            if last is None or self.rng.random() < last:
                action = coop
            else:
                action = defect
        else:  # random
            if space.get("kind") == "choice":
                action = self.rng.choice(space["options"])
            else:
                action = self.rng.randint(space["low"], space["high"])
        text = json.dumps({"thought": f"(mock:{self.policy})", "action": action}, ensure_ascii=False)
        return LLMResponse(text, 300, 20, 1.0)
