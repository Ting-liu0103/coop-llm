"""腳本化的對手(confederate)。

探針實驗(probe)裡只有 1 位 LLM 代理人(focal),其餘座位由程式腳本扮演。好處:
  * 對手的行為完全受控,所以「LLM 的反應」是因果上乾淨的,不是自我對弈的複雜互動。
  * 每場只需要 1 個 LLM 代理人,呼叫次數比自我對弈少很多。
  * 能算出「對好人合作、對壞人不合作」這種自我對弈量不到的指標。

腳本名稱 = 行為 [+ "_promise"]:
  allc       永遠合作                      alld       永遠背叛
  tft        第 1 回合合作,之後鏡射其他人上一回合的平均合作分數
  betrayer   前 betray_round-1 回合合作,之後永遠背叛(假朋友)
  *_promise  同上,但每回合行動前在聊天頻道公開承諾「我這回合會合作」(說話與做事分開操作)
"""
from __future__ import annotations

import random
from typing import Any

from .games import make_game
from .games.base import Game, RoundRecord

BEHAVIORS = ("allc", "alld", "tft", "betrayer")
FRIENDLY = ("allc", "allc_promise")   # 算「對好人合作」的腳本
HOSTILE = ("alld", "alld_promise")    # 算「對壞人合作」的腳本


def parse_script(name: str) -> tuple[str, bool]:
    """'alld_promise' -> ('alld', True)。"""
    promise = name.endswith("_promise")
    behavior = name[: -len("_promise")] if promise else name
    if behavior not in BEHAVIORS:
        raise ValueError(f"未知腳本 {name!r};行為必須是 {BEHAVIORS},可加 _promise 後綴")
    return behavior, promise


def _action_from_score(space: dict, score: float) -> Any:
    """合作分數(0~1)→ 該賽局的合法動作。選項型只有兩個值,>=0.5 視為合作。"""
    coop, defect = space["cooperate"], space["defect"]
    if space["kind"] == "choice":
        return coop if score >= 0.5 else defect
    return int(round(defect + (coop - defect) * score))


PROMISES = {
    "zh": {"ipd": "我這回合會選 {coop}。", "pgg": "我這回合會投入全部 {coop} 元。",
           "cpr": "我這回合只取 {coop},大家一起維持資源。"},
    "en": {"ipd": "I will pick {coop} this round.", "pgg": "I will contribute all {coop} this round.",
           "cpr": "I will take only {coop} this round so we keep the resource going."},
}


class ScriptedPartner:
    """一個腳本化座位。act() 與 message() 只看公開歷史,不偷看 LLM 的想法。"""

    def __init__(self, seat: int, script: str, *, betray_round: int = 4, lang: str = "zh"):
        self.seat = seat
        self.agent_id = f"S{seat + 1}"
        self.script = script
        self.behavior, self.promise = parse_script(script)
        self.betray_round = betray_round
        self.lang = lang

    def score(self, history: list[RoundRecord], round_no: int) -> float:
        b = self.behavior
        if b == "allc":
            return 1.0
        if b == "alld":
            return 0.0
        if b == "betrayer":
            return 1.0 if round_no < self.betray_round else 0.0
        # tft:第一回合合作;之後鏡射其他人(含 focal)上一回合的平均合作分數
        if not history:
            return 1.0
        last = history[-1].coop
        others = [c for i, c in enumerate(last) if i != self.seat]
        return sum(others) / len(others)

    def act(self, game: Game, round_no: int) -> Any:
        return _action_from_score(game.action_space(), self.score(game.history, round_no))

    def message(self, game: Game, round_no: int) -> str:
        """承諾永遠是「我會合作」,不管之後做不做到。沒有 _promise 的腳本保持沉默。"""
        if not self.promise:
            return ""
        coop = game.action_space()["cooperate"]
        return PROMISES[self.lang][game.name].format(coop=coop)


def make_partners(game: Game, script: str, *, focal_seat: int = 0, betray_round: int = 4,
                  lang: str = "zh") -> list[ScriptedPartner]:
    return [ScriptedPartner(i, script, betray_round=betray_round, lang=lang)
            for i in range(game.n_agents) if i != focal_seat]


def anchor_payoffs(game_name: str, seed: int, script: str, game_params: dict | None = None, *,
                   focal_seat: int = 0, betray_round: int = 4) -> dict[str, float]:
    """focal 永遠合作 / 永遠背叛 時,對同一個腳本能拿到的總報酬。

    腳本與賽局都是確定性的,所以這是精確值。用來把報酬正規化成 payoff_index:
    0 = 兩種固定策略中較差的、1 = 較好的。LLM 若做得比兩者都好(例如先合作再報復),
    指數可以超過 1;若兩者相等(沒有差異)則為 NaN。
    """
    out = {}
    for label, s in (("allc", 1.0), ("alld", 0.0)):
        g = make_game(game_name, seed, **(game_params or {}))
        partners = make_partners(g, script, focal_seat=focal_seat, betray_round=betray_round)
        total = 0.0
        while not g.done:
            rnd = g.round_no + 1
            acts: list[Any] = [None] * g.n_agents
            acts[focal_seat] = _action_from_score(g.action_space(), s)
            for p in partners:
                acts[p.seat] = p.act(g, rnd)
            total += g.step(acts).payoffs[focal_seat]
        out[label] = total
    return out


def payoff_index(total: float, anchors: dict[str, float]) -> float:
    lo, hi = min(anchors.values()), max(anchors.values())
    if hi - lo < 1e-9:
        return float("nan")
    return (total - lo) / (hi - lo)


def shuffled_speakers(n: int, rng: random.Random) -> list[int]:
    order = list(range(n))
    rng.shuffle(order)
    return order
