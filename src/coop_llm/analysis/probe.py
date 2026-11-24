"""探針實驗的指標。核心問題:介入(人格/溝通/推理)讓代理人「更會分辨」,還是只是「更愛合作」?

每場(一個 focal LLM 對一組腳本對手)算:
  coop_r1       第 1 回合的合作分數(還沒看到任何行為,只反映先驗;聊天條件下反映對承諾的信任)
  coop_from_r2  第 2 回合起的平均合作分數(看過對手行為之後)
  payoff_index  報酬正規化(0=永遠合作/永遠背叛中較差者,1=較好者)
  adapt_lag     只對 betrayer:對手開始背叛後,focal 隔幾回合才把合作降到 <=0.5(沒降 = 設限值,adapted=False)

成對指標(同 game/model/condition/rep,同 seed):
  DI(辨別力)= 對好人的合作 − 對壞人的合作
    好人 = allc(聊天條件為 allc_promise);壞人 = alld(聊天條件為 alld_promise)
    聊天條件下兩者說的話完全一樣,只有行為不同,所以 DI 量的是「看行為還是聽承諾」。
  DI 高 = 懂得分辨;兩者都高 = 天真地全面合作;兩者都低 = 一律防衛。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..confederates import FRIENDLY, HOSTILE

KEYS = ["game", "model", "condition", "rep"]


def _is_probe(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["script"].notna()] if "script" in df.columns else df.iloc[0:0]


def _adapt(g: pd.DataFrame, betray_round: int) -> tuple[float, bool]:
    after = g[g["round"] > betray_round].sort_values("round")
    if after.empty:
        return float("nan"), False
    low = after[after["coop"] <= 0.5]
    if len(low):
        return float(low["round"].iloc[0] - betray_round), True
    return float(after["round"].iloc[-1] - betray_round + 1), False  # 設限


def probe_game_level(rounds: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rounds, summary = _is_probe(rounds), _is_probe(summary)
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for gid, g in rounds.groupby("game_id"):
        g = g.sort_values("round")
        late = g[g["round"] >= 2]
        s = summary[summary["game_id"] == gid].iloc[0]
        lag, adapted = (float("nan"), False)
        if s["script"].startswith("betrayer"):
            lag, adapted = _adapt(g, int(s["betray_round"]))
        rows.append(dict(
            game_id=gid, game=s["game"], model=s["model"], condition=s["condition"], rep=int(s["rep"]),
            script=s["script"], coop_rate=s["coop_rate"],
            coop_r1=float(g["coop"].iloc[0]),
            coop_from_r2=float(late["coop"].mean()) if len(late) else float("nan"),
            payoff_index=s["payoff_index"], focal_payoff=s["focal_payoff"],
            adapt_lag=lag, adapted=adapted, rounds_played=int(s["rounds_played"]),
            n_parse_fail=int(s["n_parse_fail"])))
    return pd.DataFrame(rows)


def paired_di(pg: pd.DataFrame) -> pd.DataFrame:
    """每個 (game, model, condition, rep) 一列:對好人/壞人的合作與辨別力 DI。"""
    out = []
    for key, g in pg.groupby(KEYS):
        fr = g[g["script"].isin(FRIENDLY)]
        ho = g[g["script"].isin(HOSTILE)]
        if fr.empty or ho.empty:
            continue
        f, h = fr.iloc[0], ho.iloc[0]
        out.append(dict(zip(KEYS, key), coop_friendly=f["coop_from_r2"], coop_hostile=h["coop_from_r2"],
                        di=f["coop_from_r2"] - h["coop_from_r2"],
                        coop_r1=np.nanmean([f["coop_r1"], h["coop_r1"]]),
                        exploit_protection=h["payoff_index"], promise=f["script"].endswith("_promise")))
    return pd.DataFrame(out)


def plot_quality_map(pairs: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """每個 (game, model) 一張圖。x=對壞人的合作(天真度),y=對好人的合作。每個點是一個條件。
    越靠左上越好(對好人合作、對壞人不合作);右上是天真地全面合作;左下是一律防衛。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for (game, model), d in pairs.groupby(["game", "model"]):
        agg = d.groupby("condition")[["coop_hostile", "coop_friendly"]].agg(["mean", "sem"])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], ls=":", c="gray", lw=1)
        for cond, r in agg.iterrows():
            ax.errorbar(r[("coop_hostile", "mean")], r[("coop_friendly", "mean")],
                        xerr=1.96 * np.nan_to_num(r[("coop_hostile", "sem")]),
                        yerr=1.96 * np.nan_to_num(r[("coop_friendly", "sem")]), fmt="o", ms=5, capsize=2)
            ax.annotate(cond, (r[("coop_hostile", "mean")], r[("coop_friendly", "mean")]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("coop vs hostile partner (naivety)")
        ax.set_ylabel("coop vs friendly partner")
        ax.set_title(f"{game} · {model}  (top-left = discerning)")
        fig.tight_layout()
        p = out_dir / f"quality_map_{game}_{model}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths
