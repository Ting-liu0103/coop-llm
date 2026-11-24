"""主要圖表:每個賽局、每個模型一張「合作率 × 回合」折線圖,每條線一個條件,附 95% 信賴帶(平均 ± 1.96 SE)。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import coop_by_round


def plot_coop_by_round(rounds: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = coop_by_round(rounds)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for (game, model), d in agg.groupby(["game", "model"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        for cond, c in d.groupby("condition"):
            c = c.sort_values("round")
            ax.plot(c["round"], c["mean"], marker="o", ms=3, label=cond)
            ax.fill_between(c["round"], c["mean"] - 1.96 * c["se"].fillna(0),
                            c["mean"] + 1.96 * c["se"].fillna(0), alpha=0.15)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("round")
        ax.set_ylabel("cooperation score")
        ax.set_title(f"{game} · {model}")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        p = out_dir / f"coop_by_round_{game}_{model}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths
