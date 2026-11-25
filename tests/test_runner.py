import asyncio
import json
from pathlib import Path

import pytest

from coop_llm.analysis import game_level, load_logs
from coop_llm.runner import (Condition, game_id_of, personas_for, play_game, run_experiment, stable_seed,
                             write_game_log)
from coop_llm.llm.mock import MockLLM

MODELS = [{"name": "m_rand", "provider": "mock", "policy": "random"},
          {"name": "m_cond", "provider": "mock", "policy": "conditional"}]


def run(c):
    return asyncio.run(c)


def test_seed_is_independent_of_condition_and_stable():
    assert stable_seed("pgg", "m", 3) == stable_seed("pgg", "m", 3)
    assert stable_seed("pgg", "m", 3) != stable_seed("pgg", "m", 4)


def test_personas_for_mixed():
    c = Condition("C3", persona="mixed")
    assert personas_for(c, 4) == ["selfish", "neutral", "neutral", "neutral"]
    assert personas_for(c, 2) == ["selfish", "neutral"]
    assert personas_for(Condition("C2", persona="altruistic"), 3) == ["altruistic"] * 3


@pytest.mark.parametrize("game,agents", [("ipd", 2), ("pgg", 4), ("cpr", 4)])
def test_play_game_row_structure(game, agents):
    rows = run(play_game(game, Condition("C0"), "m", MockLLM("cooperate"), 0))
    rr = [r for r in rows if r["type"] == "round"]
    s = rows[-1]
    assert s["type"] == "summary"
    assert len(rr) == agents * s["rounds_played"] == agents * 10
    assert {"game_id", "condition", "model", "seed", "round", "agent_id", "persona", "thought", "action",
            "payoff", "coop", "parse_ok", "tokens_in", "tokens_out"} <= set(rr[0])
    assert s["coop_rate"] == pytest.approx(1.0)
    assert s["welfare_ratio"] == pytest.approx(1.0)  # 全部合作 = 基準福利
    assert s["n_parse_fail"] == 0


def test_defectors_collapse_cpr_early():
    rows = run(play_game("cpr", Condition("C0"), "m", MockLLM("defect"), 0))
    s = rows[-1]
    assert s["collapse_round"] == 1 and s["survived"] is False and s["rounds_played"] == 1


def test_chat_condition_adds_chat_rows_before_actions():
    rows = run(play_game("pgg", Condition("C4", communication="chat"), "m", MockLLM("cooperate"), 0))
    chats = [r for r in rows if r["type"] == "chat"]
    assert len(chats) == 4 * 10
    first_round_idx = next(i for i, r in enumerate(rows) if r["type"] == "round")
    assert all(rows.index(c) < first_round_idx for c in chats if c["round"] == 1)


def test_flaky_failures_are_logged_not_dropped():
    class AlwaysBad(MockLLM):
        async def complete(self, *a, **k):
            from coop_llm.llm.base import LLMResponse
            return LLMResponse("壞掉", 1, 1, 1)

    rows = run(play_game("pgg", Condition("C0"), "m", AlwaysBad(), 0, max_retries=1))
    s = rows[-1]
    assert s["n_parse_fail"] == 4 * 10
    assert all(r["parse_ok"] is False for r in rows if r["type"] == "round")


def test_same_seed_same_labels_across_conditions():
    a = run(play_game("ipd", Condition("C0"), "m", MockLLM("cooperate"), 5))
    b = run(play_game("ipd", Condition("C5", reasoning="social_cot"), "m", MockLLM("cooperate"), 5))
    assert a[0]["seed"] == b[0]["seed"] and a[0]["action"] == b[0]["action"]


def test_write_is_atomic_and_resume_skips(tmp_path: Path):
    stats = run(run_experiment(Condition("C0"), MODELS, ["ipd", "pgg"], 3, tmp_path, dry_run=True, verbose=False))
    assert stats == {"done": 12, "skipped": 0, "failed": 0}
    assert not list(tmp_path.glob("*.part"))
    stats2 = run(run_experiment(Condition("C0"), MODELS, ["ipd", "pgg"], 3, tmp_path, dry_run=True, verbose=False))
    assert stats2 == {"done": 0, "skipped": 12, "failed": 0}
    # 追加重複次數只跑新的
    stats3 = run(run_experiment(Condition("C0"), MODELS, ["ipd", "pgg"], 5, tmp_path, dry_run=True, verbose=False))
    assert stats3["done"] == 8 and stats3["skipped"] == 12


def test_part_file_is_not_treated_as_done(tmp_path: Path):
    gid = game_id_of("ipd", "C0", "m_rand", 0)
    (tmp_path / f"{gid}.jsonl.part").write_text("{}")  # 模擬中途當掉
    stats = run(run_experiment(Condition("C0"), MODELS[:1], ["ipd"], 1, tmp_path, dry_run=True, verbose=False))
    assert stats["done"] == 1 and (tmp_path / f"{gid}.jsonl").exists()


def test_one_failing_game_does_not_stop_the_batch(tmp_path: Path, monkeypatch):
    import coop_llm.runner as R
    real = R.play_game

    async def flaky(game, cond, model, llm, rep, **kw):
        if rep == 1:
            raise RuntimeError("API 掛了")
        return await real(game, cond, model, llm, rep, **kw)

    monkeypatch.setattr(R, "play_game", flaky)
    stats = run(R.run_experiment(Condition("C0"), MODELS[:1], ["ipd"], 3, tmp_path, dry_run=True, verbose=False))
    assert stats == {"done": 2, "skipped": 0, "failed": 1}


def test_logs_load_and_game_level_metrics(tmp_path: Path):
    for cond in (Condition("C0"), Condition("C1", persona="selfish")):
        run(run_experiment(cond, MODELS, ["ipd", "pgg", "cpr"], 4, tmp_path, dry_run=True, verbose=False))
    rounds, chats, summary = load_logs(tmp_path)
    assert chats.empty and summary["game_id"].nunique() == 2 * 2 * 3 * 4
    g = game_level(rounds, summary)
    assert len(g) == 48
    assert g[["coop_rate", "welfare_ratio", "mean_payoff"]].notna().all().all()
    assert g["coop_rate"].between(0, 1).all()


def test_failure_reason_is_written_to_failures_txt(tmp_path: Path, monkeypatch):
    import coop_llm.runner as R

    async def boom(*a, **k):
        raise RuntimeError("503 server_is_overloaded")

    monkeypatch.setattr(R, "play_game", boom)
    stats = run(R.run_experiment(Condition("C0"), MODELS[:1], ["ipd"], 2, tmp_path, dry_run=True, verbose=False))
    assert stats["failed"] == 2
    txt = (tmp_path / "_failures.txt").read_text(encoding="utf-8")
    assert txt.count("server_is_overloaded") == 2 and "ipd-C0-m_rand-000" in txt
    assert not list(tmp_path.glob("*.jsonl"))  # 失敗紀錄不會被當成對局資料
