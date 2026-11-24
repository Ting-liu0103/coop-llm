"""從 JSONL 日誌算設計書裡的指標。統計單位是「一場對局」。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_logs(log_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """回傳 (round 列, chat 列, summary 列)。.part 檔(未完成的場次)會被忽略。"""
    rounds, chats, sums = [], [], []
    for f in sorted(Path(log_dir).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            {"round": rounds, "chat": chats, "summary": sums}[r["type"]].append(r)
    return pd.DataFrame(rounds), pd.DataFrame(chats), pd.DataFrame(sums)


def _event_rate(events: list[bool]) -> float:
    return float(np.mean(events)) if events else float("nan")


def dynamics_for_game(g: pd.DataFrame, defect_threshold: float = 0.5) -> dict:
    """單場的信任動態指標。g 為這場的 round 列。

    * retaliation 報復率:t-1 回合其他人平均合作分數比 t-2 回合下降時,自己 t 回合的合作分數也比 t-1 下降的比例
    * forgiveness 寬恕率:t-1 回合其他人平均合作分數回升時,自己 t 回合也回升的比例
    * first_defection 首次背叛回合:第一個「有人合作分數 < defect_threshold」的回合(沒有則為 NaN)
    * endgame_gap 終局差:最後一回合平均合作分數減去前面各回合的平均
    """
    piv = g.pivot(index="round", columns="agent_id", values="coop").sort_index()
    n = piv.shape[1]
    ret, forg = [], []
    rounds = list(piv.index)
    for ag in piv.columns:
        others = piv.drop(columns=ag).mean(axis=1)
        for k in range(2, len(rounds)):
            t, t1, t2 = rounds[k], rounds[k - 1], rounds[k - 2]
            d_others = others[t1] - others[t2]
            d_self = piv.loc[t, ag] - piv.loc[t1, ag]
            if d_others < 0:
                ret.append(d_self < 0)
            elif d_others > 0:
                forg.append(d_self > 0)
    below = piv[(piv < defect_threshold).any(axis=1)]
    first_def = float(below.index[0]) if len(below) else float("nan")
    team = piv.mean(axis=1)
    gap = float(team.iloc[-1] - team.iloc[:-1].mean()) if len(team) > 1 else float("nan")
    return {"retaliation": _event_rate(ret), "forgiveness": _event_rate(forg),
            "first_defection": first_def, "endgame_gap": gap, "n_agents": n}


def game_level(rounds: pd.DataFrame, summary: pd.DataFrame, defect_threshold: float = 0.5) -> pd.DataFrame:
    """每場一列:合作率、報酬、福利比、存活、信任動態、解析失敗數。"""
    if "script" in summary.columns:  # 探針場次用 probe_game_level,這裡只處理自我對弈
        summary = summary[summary["script"].isna()]
        rounds = rounds[rounds["script"].isna()] if "script" in rounds.columns else rounds
    keys = ["game_id", "game", "condition", "model", "rep"]
    s = summary[keys + ["coop_rate", "welfare_ratio", "rounds_played", "n_parse_fail"]
                + [c for c in ("collapse_round", "survived") if c in summary.columns]].copy()
    pay = rounds.groupby("game_id")["payoff"].sum().rename("payoff_sum")
    n_ag = rounds.groupby("game_id")["agent_id"].nunique().rename("n_agents")
    s = s.merge(pay, on="game_id").merge(n_ag, on="game_id")
    s["mean_payoff"] = s["payoff_sum"] / s["n_agents"]  # 全隊平均的 10 回合總分
    dyn = pd.DataFrame({gid: dynamics_for_game(g, defect_threshold)
                        for gid, g in rounds.groupby("game_id")}).T
    dyn.index.name = "game_id"
    return s.merge(dyn.drop(columns=["n_agents"]).reset_index(), on="game_id")


def coop_by_round(rounds: pd.DataFrame) -> pd.DataFrame:
    """每個 (game, model, condition, round) 的合作分數:先對場內代理人取平均,再對場取平均與標準誤。"""
    per_game = (rounds.groupby(["game", "model", "condition", "game_id", "round"])["coop"].mean().reset_index())
    agg = per_game.groupby(["game", "model", "condition", "round"])["coop"].agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    return agg


def parse_failure_report(rounds: pd.DataFrame) -> pd.DataFrame:
    """每個 (model, condition) 的解析失敗率。分析時要回報,失敗場次不要默默丟掉。"""
    g = rounds.groupby(["model", "condition"])["parse_ok"].agg(["size", "sum"]).reset_index()
    g["fail_rate"] = 1 - g["sum"] / g["size"]
    return g.rename(columns={"size": "n_calls"}).drop(columns=["sum"])
