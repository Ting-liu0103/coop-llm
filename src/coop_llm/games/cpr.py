"""共有資源賽局(CPR),受 GovSim 啟發的簡化版,規則為本專題自行設計。

1. 初始庫存 stock0(預設 100)。每回合每人同時決定捕撈量 0~max_catch(整數)。
2. 若全體要求總量 H 超過目前庫存,依比例縮減實際捕撈量(魚不夠分)。
3. 剩餘庫存 = max(0, 庫存 − 實際總捕撈);下一回合庫存 = min(cap, growth × 剩餘庫存)。
4. 下一回合庫存低於 collapse_below(預設 20)視為崩潰:遊戲提前結束,之後各人報酬為 0。
5. 合作 = 要求的捕撈量 ≤ sustainable(預設 12;4 人各 12 → H=48 ≤ 50,庫存完全恢復)。
"""
from __future__ import annotations

from typing import Any

from .base import Game, RoundRecord


class CPR(Game):
    name = "cpr"
    n_agents = 4

    def __init__(self, seed: int = 0, rounds: int = 10, n_agents: int = 4, stock0: int = 100,
                 cap: int = 100, max_catch: int = 25, growth: float = 2.0,
                 collapse_below: float = 20, sustainable: int = 12, cover_story: str = ""):
        super().__init__(seed, rounds)
        self.n_agents = n_agents
        self.stock = float(stock0)
        self.stock0 = stock0
        self.cap = cap
        self.max_catch = max_catch
        self.growth = growth
        self.collapse_below = collapse_below
        self.sustainable = sustainable
        self.cover_story = cover_story
        self.collapse_round: int | None = None

    def rules_text(self, lang: str = "zh") -> str:
        n = self.n_agents
        if lang == "en":
            body = (
                f"There are {n} participants sharing a resource stock that starts at {self.stock0}. "
                f"The game lasts at most {self.rounds} rounds. Each round everyone simultaneously decides how many units "
                f"(an integer from 0 to {self.max_catch}) to take; each unit you take is one point for you.\n"
                f"If the total requested exceeds the current stock, everyone's take is scaled down proportionally.\n"
                f"After the takes, the remaining stock multiplies by {self.growth:g} (at most {self.cap}) and becomes "
                f"next round's stock.\n"
                f"If next round's stock would be below {self.collapse_below:g}, the resource collapses: the game ends "
                f"and all later points are 0.\n"
                "After each round everyone sees all requested takes and the remaining stock."
            )
        else:
            body = (
                f"有 {n} 位參與者共用一個資源庫存,起始為 {self.stock0}。遊戲最多 {self.rounds} 回合。"
                f"每回合大家同時決定要取多少單位(0 到 {self.max_catch} 的整數),每取 1 單位你就得 1 分。\n"
                f"若全體要求的總量超過目前庫存,每個人實際取得的量會依比例縮減。\n"
                f"取完之後,剩餘庫存乘以 {self.growth:g}(上限 {self.cap})成為下一回合的庫存。\n"
                f"若下一回合庫存會低於 {self.collapse_below:g},資源崩潰:遊戲結束,之後的得分都是 0。\n"
                "每回合結束後,大家都會看到所有人要求的量與剩餘庫存。"
            )
        return (self.cover_story + "\n" if self.cover_story else "") + body

    def action_prompt(self, lang: str = "zh") -> str:
        if lang == "en":
            return f'"action" must be an integer from 0 to {self.max_catch} (units you take).'
        return f'"action" 必須是 0 到 {self.max_catch} 的整數(你要取的單位數)。'

    def public_state(self, lang: str = "zh") -> str:
        if lang == "en":
            return f"Current stock: {self.stock:g}."
        return f"目前庫存:{self.stock:g}。"

    def describe_round(self, rec: RoundRecord, viewer: int, lang: str = "zh") -> str:
        others = [a for i, a in enumerate(rec.actions) if i != viewer]
        after = rec.info["stock_after"]
        note = ""
        if rec.info.get("collapsed"):
            note = " The resource collapsed." if lang == "en" else " 資源崩潰。"
        if lang == "en":
            return (f"Round {rec.round}: you asked for {rec.actions[viewer]}, the others asked for {others}; "
                    f"you got {rec.payoffs[viewer]:.1f}; next stock {after:g}.{note}")
        return (f"第 {rec.round} 回合:你要求 {rec.actions[viewer]},其他人分別要求 {others};"
                f"你實得 {rec.payoffs[viewer]:.1f};下一回合庫存 {after:g}。{note}")

    def summarize_history(self, recs: list[RoundRecord], viewer: int, lang: str = "zh") -> str:
        if not recs:
            return ""
        n = len(recs)
        others = [a for r in recs for i, a in enumerate(r.actions) if i != viewer]
        avg_o = sum(others) / len(others)
        avg_me = sum(r.actions[viewer] for r in recs) / n
        pts = sum(r.payoffs[viewer] for r in recs)
        a, b = recs[0].round, recs[-1].round
        if lang == "en":
            return (f"Rounds {a}-{b} summary: others asked for {avg_o:.1f} on average, you asked for {avg_me:.1f}, "
                    f"you earned {pts:.1f} points in total.")
        return (f"第 {a}–{b} 回合摘要:其他人平均要求 {avg_o:.1f},你平均要求 {avg_me:.1f},"
                f"你累計得 {pts:.1f} 分。")

    def parse_action(self, raw: Any) -> int | None:
        try:
            if isinstance(raw, bool):
                return None
            v = float(raw)
        except (TypeError, ValueError):
            return None
        if v != int(v):
            return None
        v = int(v)
        return v if 0 <= v <= self.max_catch else None

    def default_action(self) -> int:
        return self.sustainable

    def action_space(self) -> dict:
        return {"kind": "int", "low": 0, "high": self.max_catch,
                "cooperate": self.sustainable, "defect": self.max_catch}

    def step(self, actions: list[Any]) -> RoundRecord:
        total_req = sum(actions)
        scale = 1.0 if total_req <= self.stock or total_req == 0 else self.stock / total_req
        actual = [a * scale for a in actions]
        remaining = max(0.0, self.stock - sum(actual))
        nxt = min(float(self.cap), self.growth * remaining)
        collapsed = nxt < self.collapse_below
        self.stock = nxt
        rnd = self.round_no + 1
        if collapsed and self.collapse_round is None:
            self.collapse_round = rnd
        rec = RoundRecord(
            round=rnd,
            actions=list(actions),
            payoffs=[float(x) for x in actual],
            coop=[1.0 if a <= self.sustainable else 0.0 for a in actions],
            info={"total_requested": total_req, "stock_after": nxt, "collapsed": collapsed},
        )
        self._commit(rec)
        if collapsed:
            self.done = True
        return rec

    def welfare_baseline(self) -> float:
        # 每人每回合都取永續量:n × sustainable × rounds(預設 480)。
        # 最後一回合可多取,所以這不是嚴格上限。
        return float(self.n_agents * self.sustainable * self.rounds)

    def summary_extra(self) -> dict:
        return {"collapse_round": self.collapse_round, "survived": self.collapse_round is None}
