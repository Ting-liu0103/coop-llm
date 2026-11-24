"""公共財賽局(PGG)。

每回合每人有 endowment 點,可捐 0~endowment(整數)。
報酬 = endowment − 自己捐的 + multiplier × 全體捐獻總和 ÷ n。
預設 n=4、endowment=20、multiplier=1.6 → 每人回報率 0.4 (<1),全體回報 1.6 (>1),困境成立。
"""
from __future__ import annotations

from typing import Any

from .base import Game, RoundRecord


class PGG(Game):
    name = "pgg"
    n_agents = 4

    def __init__(self, seed: int = 0, rounds: int = 10, n_agents: int = 4,
                 endowment: int = 20, multiplier: float = 1.6, cover_story: str = ""):
        super().__init__(seed, rounds)
        self.n_agents = n_agents
        self.endowment = endowment
        self.multiplier = multiplier
        self.cover_story = cover_story

    @property
    def mpcr(self) -> float:
        return self.multiplier / self.n_agents

    def rules_text(self, lang: str = "zh") -> str:
        n, e, m = self.n_agents, self.endowment, self.multiplier
        if lang == "en":
            body = (
                f"There are {n} participants and {self.rounds} rounds. Each round everyone receives {e} points "
                f"and simultaneously decides how many points (an integer from 0 to {e}) to put into a shared fund.\n"
                f"The fund is multiplied by {m:g} and split equally among all {n} participants.\n"
                f"Your points this round = {e} - (your contribution) + {m:g} x (total contributions) / {n}.\n"
                "After each round everyone sees all contributions and their own points."
            )
        else:
            body = (
                f"共有 {n} 位參與者、{self.rounds} 回合。每回合每人拿到 {e} 點,同時決定要把多少點"
                f"(0 到 {e} 的整數)放進共同基金。\n"
                f"基金總額乘以 {m:g} 之後,平分給全部 {n} 位參與者。\n"
                f"你本回合的得分 = {e} − 你放入的點數 + {m:g} × 全體放入總和 ÷ {n}。\n"
                "每回合結束後,大家都會看到所有人放入的點數和自己的得分。"
            )
        return (self.cover_story + "\n" if self.cover_story else "") + body

    def action_prompt(self, lang: str = "zh") -> str:
        if lang == "en":
            return f'"action" must be an integer from 0 to {self.endowment} (points you put into the fund).'
        return f'"action" 必須是 0 到 {self.endowment} 的整數(你放入基金的點數)。'

    def describe_round(self, rec: RoundRecord, viewer: int, lang: str = "zh") -> str:
        others = [a for i, a in enumerate(rec.actions) if i != viewer]
        if lang == "en":
            return (f"Round {rec.round}: you put in {rec.actions[viewer]}, the others put in {others}, "
                    f"you got {rec.payoffs[viewer]:.1f} points.")
        return (f"第 {rec.round} 回合:你放入 {rec.actions[viewer]},其他人分別放入 {others},"
                f"你得 {rec.payoffs[viewer]:.1f} 分。")

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
            return (f"Rounds {a}-{b} summary: others put in {avg_o:.1f} on average, you put in {avg_me:.1f} "
                    f"on average, you earned {pts:.1f} points in total.")
        return (f"第 {a}–{b} 回合摘要:其他人平均放入 {avg_o:.1f},你平均放入 {avg_me:.1f},"
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
        return v if 0 <= v <= self.endowment else None

    def default_action(self) -> int:
        return self.endowment // 2

    def action_space(self) -> dict:
        return {"kind": "int", "low": 0, "high": self.endowment,
                "cooperate": self.endowment, "defect": 0}

    def step(self, actions: list[Any]) -> RoundRecord:
        total = sum(actions)
        payoffs = [self.endowment - a + self.multiplier * total / self.n_agents for a in actions]
        rec = RoundRecord(
            round=self.round_no + 1,
            actions=list(actions),
            payoffs=[float(p) for p in payoffs],
            coop=[a / self.endowment for a in actions],
            info={"total_contribution": total},
        )
        return self._commit(rec)

    def welfare_baseline(self) -> float:
        # 全員都捐滿:每人每回合 multiplier × endowment(預設 32)
        return float(self.n_agents * self.multiplier * self.endowment * self.rounds)  # 1280
