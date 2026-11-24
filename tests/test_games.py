"""賽局規則測試:對應設計書「從一開始就要做的三件事」第 1 點。跑大量實驗前先讓這些通過。"""
import pytest

from coop_llm.games import CPR, IPD, PGG, make_game


# ---- PGG --------------------------------------------------------------
def test_pgg_all_zero_gives_20_each():
    g = PGG(seed=1)
    rec = g.step([0, 0, 0, 0])
    assert rec.payoffs == [20.0] * 4


def test_pgg_all_full_gives_32_each():
    g = PGG(seed=1)
    rec = g.step([20, 20, 20, 20])
    assert rec.payoffs == pytest.approx([32.0] * 4)


def test_pgg_dilemma_holds():
    g = PGG()
    assert g.mpcr < 1 < g.multiplier  # 個人回報率 <1,全體回報率 >1


def test_pgg_free_rider_beats_contributor():
    g = PGG()
    rec = g.step([0, 20, 20, 20])
    assert rec.payoffs[0] > rec.payoffs[1]


def test_pgg_parse_action():
    g = PGG()
    assert g.parse_action(12) == 12
    assert g.parse_action("12") == 12
    assert g.parse_action(12.0) == 12
    assert g.parse_action(12.5) is None
    assert g.parse_action(-1) is None
    assert g.parse_action(21) is None
    assert g.parse_action("abc") is None
    assert g.parse_action(True) is None


def test_pgg_welfare_baseline_1280():
    assert PGG().welfare_baseline() == 1280


# ---- CPR --------------------------------------------------------------
def test_cpr_total_48_recovers_to_100():
    g = CPR()
    rec = g.step([12, 12, 12, 12])
    assert g.stock == 100
    assert not rec.info["collapsed"]
    assert rec.payoffs == [12.0] * 4


def test_cpr_total_100_collapses():
    g = CPR()
    rec = g.step([25, 25, 25, 25])
    assert rec.info["collapsed"]
    assert g.done
    assert g.collapse_round == 1
    assert g.summary_extra() == {"collapse_round": 1, "survived": False}


def test_cpr_total_over_50_declines_gradually():
    g = CPR()
    g.step([15, 15, 15, 15])  # H=60 → 剩 40 → 下一回合 80
    assert g.stock == 80
    assert not g.done


def test_cpr_scales_when_request_exceeds_stock():
    g = CPR(stock0=30, cap=100)
    rec = g.step([20, 20, 20, 20])  # 要求 80 > 庫存 30,按比例縮到總共 30
    assert sum(rec.payoffs) == pytest.approx(30)
    assert all(p == pytest.approx(7.5) for p in rec.payoffs)
    assert g.stock == 0 and rec.info["collapsed"]


def test_cpr_cooperation_flag_is_sustainable_quota():
    g = CPR()
    rec = g.step([12, 13, 0, 25])
    assert rec.coop == [1.0, 0.0, 1.0, 0.0]


def test_cpr_baseline_480():
    assert CPR().welfare_baseline() == 480


def test_cpr_ends_after_rounds():
    g = CPR(rounds=3)
    for _ in range(3):
        g.step([12] * 4)
    assert g.done and g.round_no == 3 and g.collapse_round is None


# ---- IPD --------------------------------------------------------------
def test_ipd_payoffs_all_four_cells():
    g = IPD(seed=3)
    c, d = g.coop_label, g.defect_label
    assert g.step([c, c]).payoffs == [3.0, 3.0]
    assert g.step([d, d]).payoffs == [1.0, 1.0]
    assert g.step([d, c]).payoffs == [5.0, 0.0]
    assert g.step([c, d]).payoffs == [0.0, 5.0]


def test_ipd_labels_are_shuffled_but_deterministic():
    labels = {IPD(seed=s).coop_label for s in range(40)}
    assert labels == {"X", "Y"}  # 兩種都會出現
    assert IPD(seed=7).coop_label == IPD(seed=7).coop_label  # 同 seed 同結果


def test_ipd_rules_do_not_name_the_classic_game():
    txt = IPD(seed=1).rules_text("zh") + IPD(seed=1).rules_text("en")
    assert "囚徒" not in txt and "dilemma" not in txt.lower() and "prisoner" not in txt.lower()


def test_ipd_parse():
    g = IPD()
    assert g.parse_action("x") == "X"
    assert g.parse_action(' "Y" ') == "Y"
    assert g.parse_action("Z") is None
    assert g.parse_action(1) is None


def test_ipd_baseline_60():
    assert IPD().welfare_baseline() == 60


# ---- 共用 -------------------------------------------------------------
def test_make_game_unknown():
    with pytest.raises(ValueError):
        make_game("nope", 0)


@pytest.mark.parametrize("name", ["ipd", "pgg", "cpr"])
def test_text_methods_work_in_both_languages(name):
    g = make_game(name, 5, rounds=4)
    first = [g.default_action()] * g.n_agents
    g.step(first)
    g.step(first)
    for lang in ("zh", "en"):
        assert g.rules_text(lang) and g.action_prompt(lang)
        assert g.describe_round(g.history[0], 0, lang)
        assert g.summarize_history(g.history, 0, lang)
        assert isinstance(g.public_state(lang), str)
