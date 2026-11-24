"""設計書的統計方法:場級分數 → 對基準做排列檢定 → bootstrap 信賴區間 → Cliff's delta → Holm 校正。

只用 numpy / pandas,不依賴 scipy。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def permutation_test(a, b, n_perm: int = 10_000, seed: int = 0) -> float:
    """雙尾排列檢定,檢定統計量為均值差。回傳 p 值(含 +1 校正,不會是 0)。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:n].mean() - pooled[n:].mean()) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def bootstrap_diff_ci(a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """均值差 (a − b) 的百分位 bootstrap 信賴區間。回傳 (差, 下界, 上界)。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def cliffs_delta(a, b) -> float:
    """Cliff's delta ∈ [-1, 1]:P(a>b) − P(a<b)。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def holm(pvals) -> np.ndarray:
    """Holm–Bonferroni 校正後的 p 值,順序與輸入相同。"""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def compare_to_baseline(games: pd.DataFrame, metric: str, baseline: str = "C0",
                        n_perm: int = 10_000, n_boot: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """對每個 (game, model) 家族,把每個條件和基準比較。Holm 校正在同一家族內做。

    games: metrics.game_level() 的輸出(每場一列)。回傳每個比較一列。
    """
    out = []
    for (game, model), fam in games.groupby(["game", "model"]):
        base = fam.loc[fam["condition"] == baseline, metric].dropna().to_numpy()
        rows = []
        for cond, g in fam.groupby("condition"):
            if cond == baseline:
                continue
            x = g[metric].dropna().to_numpy()
            if len(x) < 2 or len(base) < 2:
                continue
            diff, lo, hi = bootstrap_diff_ci(x, base, n_boot=n_boot, seed=seed)
            rows.append(dict(game=game, model=model, metric=metric, condition=cond, baseline=baseline,
                             n_cond=len(x), n_base=len(base), diff=diff, ci_low=lo, ci_high=hi,
                             p_perm=permutation_test(x, base, n_perm=n_perm, seed=seed),
                             cliffs_delta=cliffs_delta(x, base)))
        if rows:
            p_adj = holm([r["p_perm"] for r in rows])
            for r, pa in zip(rows, p_adj):
                r["p_holm"] = float(pa)
            out.extend(rows)
    return pd.DataFrame(out)


def supported(results: pd.DataFrame, condition: str, alpha: float = 0.05,
              min_models: int = 1, min_games: int = 2) -> bool:
    """設計書的事先判準:Holm 校正後 p < alpha、方向一致,且達到「至少 min_models 個模型、
    至少 min_games 個賽局」才算支持。方向以多數(同號)為準。"""
    r = results[(results["condition"] == condition) & (results["p_holm"] < alpha)]
    if r.empty:
        return False
    sign = np.sign(r["diff"])
    if (sign != sign.iloc[0]).any():
        return False
    # 至少 min_games 個賽局在同一個模型中顯著,且這樣的模型至少 min_models 個
    per_model = r.groupby("model")["game"].nunique()
    return int((per_model >= min_games).sum()) >= min_models
