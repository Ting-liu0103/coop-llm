import numpy as np
import pandas as pd
import pytest

from coop_llm.analysis.metrics import dynamics_for_game
from coop_llm.analysis.stats import (bootstrap_diff_ci, cliffs_delta, compare_to_baseline, holm,
                                     permutation_test, supported)


def test_cliffs_delta_extremes():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_holm_known_values():
    # 教科書例子:p=(0.01,0.04,0.03) → 0.03, 0.06, 0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.9]) == pytest.approx([0.9])
    assert (holm([0.5, 0.5, 0.5]) <= 1).all()


def test_permutation_detects_difference_and_not_noise():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 0.2, 20), rng.normal(0.0, 0.2, 20)
    assert permutation_test(a, b, n_perm=2000) < 0.01
    c, d = rng.normal(0, 1, 20), rng.normal(0, 1, 20)
    assert permutation_test(c, d, n_perm=2000) > 0.05


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(1)
    a, b = rng.normal(1.0, 0.3, 40), rng.normal(0.5, 0.3, 40)
    diff, lo, hi = bootstrap_diff_ci(a, b, n_boot=2000)
    assert lo < 0.5 < hi and lo < diff < hi


def _games_df(effect=0.4, n=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for game in ("ipd", "pgg"):
        for cond, mu in (("C0", 0.3), ("C4", 0.3 + effect), ("C5", 0.3)):
            for i in range(n):
                rows.append(dict(game=game, model="m", condition=cond, coop_rate=float(np.clip(rng.normal(mu, 0.1), 0, 1))))
    return pd.DataFrame(rows)


def test_compare_to_baseline_flags_only_the_real_effect():
    res = compare_to_baseline(_games_df(), "coop_rate", n_perm=1000, n_boot=1000)
    assert set(res["condition"]) == {"C4", "C5"} and len(res) == 4
    c4 = res[res.condition == "C4"]
    assert (c4["p_holm"] < 0.05).all() and (c4["diff"] > 0).all()
    assert (res[res.condition == "C5"]["p_holm"] > 0.05).all()


def test_supported_requires_two_games():
    res = compare_to_baseline(_games_df(), "coop_rate", n_perm=1000, n_boot=1000)
    assert supported(res, "C4") is True
    assert supported(res, "C5") is False
    assert supported(res[res.game == "ipd"], "C4") is False  # 只有一個賽局達標,不夠


def test_dynamics_retaliation_and_forgiveness():
    # 兩位代理人:A1 完全跟隨 A2 的上一回合;A2 在第 3 回合背叛、第 5 回合回來
    rows = []
    a2 = {1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 1}
    a1 = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1}
    for r in range(1, 7):
        rows += [dict(round=r, agent_id="A1", coop=float(a1[r])), dict(round=r, agent_id="A2", coop=float(a2[r]))]
    d = dynamics_for_game(pd.DataFrame(rows))
    assert d["first_defection"] == 3
    assert d["retaliation"] > 0 and d["forgiveness"] > 0
    assert d["endgame_gap"] == pytest.approx(1.0 - np.mean([1, 1, 0.5, 0, 0.5]))
