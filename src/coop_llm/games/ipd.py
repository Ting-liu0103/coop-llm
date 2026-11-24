"""重複囚徒困境(IPD)。

偽裝包裝:規則文字不出現「囚徒困境」,兩個選項只叫 X 與 Y。
哪一個字母代表「合作」由 seed 決定,避免「第一個選項」偏誤(設計書:打亂選項順序)。
報酬:雙合作 3,3;雙背叛 1,1;單方背叛者 5、被背叛者 0。
"""
from __future__ import annotations

import random
from typing import Any

from .base import Game, RoundRecord

T, R, P, S = 5, 3, 1, 0  # 誘惑、合作、懲罰、被騙


class IPD(Game):
    name = "ipd"
    n_agents = 2

    def __init__(self, seed: int = 0, rounds: int = 10, shuffle_labels: bool = True,
                 cover_story: str = ""):
        super().__init__(seed, rounds)
        labels = ["X", "Y"]
        if shuffle_labels:
            random.Random(seed).shuffle(labels)
        self.coop_label, self.defect_label = labels
        self.cover_story = cover_story

    # ---- 文字 -------------------------------------------------------------
    def _payoff(self, a: str, b: str) -> int:
        c = self.coop_label
        if a == c and b == c:
            return R
        if a != c and b != c:
            return P
        return S if a == c else T

    def rules_text(self, lang: str = "zh") -> str:
        c, d = self.coop_label, self.defect_label
        if lang == "en":
            body = (
                f"You and one other participant play {self.rounds} rounds. Each round you both "
                f"choose option {c} or option {d} at the same time.\n"
                f"Points per round: both choose {c}: {R} each. Both choose {d}: {P} each. "
                f"You choose {d} and the other chooses {c}: you get {T}, the other gets {S}. "
                f"You choose {c} and the other chooses {d}: you get {S}, the other gets {T}.\n"
                "After each round you both see both choices and your points."
            )
        else:
            body = (
                f"你和另一位參與者一共進行 {self.rounds} 回合。每回合你們同時選擇選項 {c} 或選項 {d}。\n"
                f"每回合的得分:雙方都選 {c},各得 {R} 分。雙方都選 {d},各得 {P} 分。"
                f"你選 {d}、對方選 {c},你得 {T} 分、對方得 {S} 分。"
                f"你選 {c}、對方選 {d},你得 {S} 分、對方得 {T} 分。\n"
                "每回合結束後,你們都會看到雙方的選擇與自己的得分。"
            )
        return (self.cover_story + "\n" if self.cover_story else "") + body

    def action_prompt(self, lang: str = "zh") -> str:
        c, d = self.coop_label, self.defect_label
        if lang == "en":
            return f'"action" must be the string "{c}" or "{d}".'
        return f'"action" 必須是字串 "{c}" 或 "{d}"。'

    def describe_round(self, rec: RoundRecord, viewer: int, lang: str = "zh") -> str:
        me, other = rec.actions[viewer], rec.actions[1 - viewer]
        if lang == "en":
            return f"Round {rec.round}: you chose {me}, the other chose {other}, you got {rec.payoffs[viewer]:g} points."
        return f"第 {rec.round} 回合:你選 {me},對方選 {other},你得 {rec.payoffs[viewer]:g} 分。"

    def summarize_history(self, recs: list[RoundRecord], viewer: int, lang: str = "zh") -> str:
        if not recs:
            return ""
        n = len(recs)
        other_c = sum(r.coop[1 - viewer] for r in recs) / n
        my_c = sum(r.coop[viewer] for r in recs) / n
        pts = sum(r.payoffs[viewer] for r in recs)
        a, b = recs[0].round, recs[-1].round
        if lang == "en":
            return (f"Rounds {a}-{b} summary: the other chose {self.coop_label} in {other_c:.0%} of rounds, "
                    f"you chose {self.coop_label} in {my_c:.0%}, you earned {pts:g} points in total.")
        return (f"第 {a}–{b} 回合摘要:對方選 {self.coop_label} 的比例 {other_c:.0%},"
                f"你選 {self.coop_label} 的比例 {my_c:.0%},你累計得 {pts:g} 分。")

    # ---- 動作 -------------------------------------------------------------
    def parse_action(self, raw: Any) -> str | None:
        if isinstance(raw, str):
            s = raw.strip().strip("\"'").upper()
            if s in ("X", "Y"):
                return s
        return None

    def default_action(self) -> str:
        return self.coop_label

    def action_space(self) -> dict:
        return {"kind": "choice", "options": ["X", "Y"],
                "cooperate": self.coop_label, "defect": self.defect_label}

    # ---- 推進 -------------------------------------------------------------
    def step(self, actions: list[Any]) -> RoundRecord:
        a, b = actions
        rec = RoundRecord(
            round=self.round_no + 1,
            actions=[a, b],
            payoffs=[float(self._payoff(a, b)), float(self._payoff(b, a))],
            coop=[1.0 if x == self.coop_label else 0.0 for x in (a, b)],
        )
        return self._commit(rec)

    def welfare_baseline(self) -> float:
        return float(self.n_agents * R * self.rounds)  # 60
