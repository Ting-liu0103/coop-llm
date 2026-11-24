"""探針模式(1 位 LLM + 腳本對手)的測試。"""
import asyncio
from pathlib import Path

import pytest

from coop_llm.analysis import load_logs, paired_di, probe_game_level
from coop_llm.confederates import ScriptedPartner, anchor_payoffs, parse_script, payoff_index
from coop_llm.games import CPR, IPD, PGG
from coop_llm.llm.mock import MockLLM
from coop_llm.runner import Condition, play_probe_game, run_experiment, stable_seed

BASE = Condition("P0", confederates=["allc", "alld", "tft", "betrayer"])
CHAT = Condition("P5", communication="chat", confederates=["allc_promise", "alld_promise"])
MODELS = [{"name": "m_cond", "provider": "mock", "policy": "conditional"},
          {"name": "m_coop", "provider": "mock", "policy": "cooperate"}]


def run(c):
    return asyncio.run(c)


# ---- 腳本本身 ------------------------------------------------------------
def test_parse_script():
    assert parse_script("allc") == ("allc", False)
    assert parse_script("alld_promise") == ("alld", True)
    with pytest.raises(ValueError):
        parse_script("grudger")


def test_yaml_rejects_bad_script(tmp_path: Path):
    f = tmp_path / "c.yaml"
    f.write_text("name: X\nconfederates: [allc, nope]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Condition.from_yaml(f)


def test_partner_scores():
    g = PGG(seed=1)
    allc, alld = ScriptedPartner(1, "allc"), ScriptedPartner(1, "alld")
    bet = ScriptedPartner(1, "betrayer", betray_round=4)
    assert [allc.score([], r) for r in (1, 5)] == [1.0, 1.0]
    assert [alld.score([], r) for r in (1, 5)] == [0.0, 0.0]
    assert [bet.score([], r) for r in (1, 3, 4, 9)] == [1.0, 1.0, 0.0, 0.0]
    tft = ScriptedPartner(1, "tft")
    assert tft.score([], 1) == 1.0
    g.step([0, 20, 20, 20])  # 其他人(座位 0,2,3)平均合作 = 2/3
    assert tft.score(g.history, 2) == pytest.approx(2 / 3)


@pytest.mark.parametrize("game,coop,defect", [(PGG(seed=1), 20, 0), (CPR(), 12, 25)])
def test_scripted_actions_map_to_game_anchors(game, coop, defect):
    assert ScriptedPartner(1, "allc").act(game, 1) == coop
    assert ScriptedPartner(1, "alld").act(game, 1) == defect


def test_ipd_partner_uses_labels():
    g = IPD(seed=3)
    assert ScriptedPartner(1, "allc").act(g, 1) == g.coop_label
    assert ScriptedPartner(1, "alld").act(g, 1) == g.defect_label


def test_promise_message_only_for_promise_scripts():
    g = PGG(seed=1)
    assert ScriptedPartner(1, "allc").message(g, 1) == ""
    assert "20" in ScriptedPartner(1, "alld_promise").message(g, 1)  # 承諾合作,哪怕行為是背叛
    assert len(ScriptedPartner(1, "allc_promise").message(g, 1)) <= 60


def test_anchor_payoffs_hand_computed():
    # PGG:3 位全不投,focal 投 20 → 每回合 1.6*20/4 = 8,共 80;focal 不投 → 20×10 = 200
    a = anchor_payoffs("pgg", 1, "alld")
    assert a == {"allc": pytest.approx(80), "alld": pytest.approx(200)}
    # 3 位全投 20:focal 投 20 → 32×10;不投 → (20 + 1.6*60/4)×10 = 440
    b = anchor_payoffs("pgg", 1, "allc")
    assert b == {"allc": pytest.approx(320), "alld": pytest.approx(440)}
    assert payoff_index(200, a) == 1.0 and payoff_index(80, a) == 0.0
    assert payoff_index(5, {"allc": 3, "alld": 3}) != payoff_index(5, {"allc": 3, "alld": 3})  # NaN


# ---- 一場探針 ------------------------------------------------------------
@pytest.mark.parametrize("game,n", [("ipd", 2), ("pgg", 4), ("cpr", 4)])
def test_probe_log_structure(game, n):
    rows = run(play_probe_game(game, BASE, "allc", "m", MockLLM("cooperate"), 0))
    rr = [r for r in rows if r["type"] == "round"]
    assert all(r["agent_id"] == "A1" for r in rr)            # 只記 focal
    assert len(rr[0]["others_actions"]) == n - 1
    assert rows[-1]["type"] == "summary" and rows[-1]["script"] == "allc"
    assert {"payoff_index", "anchor_allc", "anchor_alld", "focal_payoff"} <= set(rows[-1])


def test_cooperator_vs_defectors_is_exploited():
    r = run(play_probe_game("pgg", BASE, "alld", "m", MockLLM("cooperate"), 0))[-1]
    d = run(play_probe_game("pgg", BASE, "alld", "m", MockLLM("defect"), 0))[-1]
    assert r["payoff_index"] == pytest.approx(0.0) and d["payoff_index"] == pytest.approx(1.0)


def test_betrayer_acts_change_at_betray_round():
    rows = run(play_probe_game("pgg", BASE, "betrayer", "m", MockLLM("cooperate"), 0))
    rr = [r for r in rows if r["type"] == "round"]
    assert rr[2]["others_actions"] == [20, 20, 20] and rr[3]["others_actions"] == [0, 0, 0]


def test_chat_rows_scripted_vs_none():
    rows = run(play_probe_game("pgg", CHAT, "alld_promise", "m", MockLLM("conditional"), 0))
    chats = [r for r in rows if r["type"] == "chat"]
    assert sum(r["scripted"] for r in chats) == 3 * 10 and sum(not r["scripted"] for r in chats) == 10
    quiet = run(play_probe_game("pgg", BASE, "alld", "m", MockLLM("conditional"), 0))
    assert not [r for r in quiet if r["type"] == "chat"]


def test_seed_shared_across_scripts_and_conditions():
    a = run(play_probe_game("ipd", BASE, "allc", "m", MockLLM("cooperate"), 4))[0]
    b = run(play_probe_game("ipd", Condition("P3", reasoning="social_cot", confederates=["alld"]),
                            "alld", "m", MockLLM("cooperate"), 4))[0]
    assert a["seed"] == b["seed"] == stable_seed("ipd", "m", 4)


# ---- 整批 + 分析 ---------------------------------------------------------
def test_run_experiment_counts_and_resume(tmp_path: Path):
    st = run(run_experiment(BASE, MODELS, ["ipd", "pgg"], 3, tmp_path, dry_run=True, verbose=False))
    assert st == {"done": 4 * 2 * 2 * 3, "skipped": 0, "failed": 0}  # 腳本 × 賽局 × 模型 × 重複
    assert not list(tmp_path.glob("*.part"))
    assert (tmp_path / "pgg-P0-betrayer-m_cond-000.jsonl").exists()
    st2 = run(run_experiment(BASE, MODELS, ["ipd", "pgg"], 3, tmp_path, dry_run=True, verbose=False))
    assert st2["done"] == 0 and st2["skipped"] == 48


def _collect(tmp_path: Path):
    for c in (BASE, CHAT):
        run(run_experiment(c, MODELS, ["ipd", "pgg", "cpr"], 4, tmp_path, dry_run=True, verbose=False))
    rounds, chats, summary = load_logs(tmp_path)
    pg = probe_game_level(rounds, summary)
    return pg, paired_di(pg)


def test_discrimination_separates_conditional_from_naive(tmp_path: Path):
    pg, pairs = _collect(tmp_path)
    cond = pairs[(pairs.model == "m_cond") & (pairs.condition == "P0")]
    coop = pairs[(pairs.model == "m_coop") & (pairs.condition == "P0")]
    assert cond["di"].mean() > 0.5          # 條件式:對好人合作、對壞人不合作
    assert coop["di"].abs().max() < 1e-9    # 天真全合作:DI = 0
    assert coop["coop_hostile"].mean() == pytest.approx(1.0)


def test_adaptation_lag_and_censoring(tmp_path: Path):
    pg, _ = _collect(tmp_path)
    bet = pg[(pg.script == "betrayer") & (pg.game == "pgg")]
    lag_c = bet[bet.model == "m_cond"]["adapt_lag"]
    assert (lag_c >= 1).all() and bet[bet.model == "m_cond"]["adapted"].all()
    assert not bet[bet.model == "m_coop"]["adapted"].any()   # 永遠合作者永遠沒適應(設限)
    assert (bet[bet.model == "m_coop"]["adapt_lag"] == 7).all()  # 10 回合 - 第 4 回合 + 1


def test_game_level_ignores_probe_rows(tmp_path: Path):
    from coop_llm.analysis import game_level
    run(run_experiment(BASE, MODELS[:1], ["pgg"], 2, tmp_path, dry_run=True, verbose=False))
    run(run_experiment(Condition("C0"), MODELS[:1], ["pgg"], 2, tmp_path, dry_run=True, verbose=False))
    rounds, _c, summary = load_logs(tmp_path)
    g = game_level(rounds, summary)
    assert set(g["condition"]) == {"C0"} and len(g) == 2
