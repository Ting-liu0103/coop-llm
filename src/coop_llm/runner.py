"""實驗執行器:讀設定檔 → 逐場跑賽局 → 每場寫一個 JSONL。

    python -m coop_llm.runner --condition configs/conditions/c0_baseline.yaml \\
        --models configs/models.yaml --reps 20 --out logs --dry-run

設計重點:
  * 斷點續跑:日誌檔 {game_id}.jsonl 已存在就跳過。寫入時先寫 .part,整場完成才改名,
    所以中途當掉的場次不會被誤認為完成。
  * 同條件、同賽局、同模型、同 rep 在不同條件下用同一個 seed(座位、選項標籤、發言順序一致),
    讓條件之間的差異更乾淨。seed 與條件名稱無關,只由 (game, model, rep) 決定。
  * 場與場之間並行(受 --concurrency 限制)。同一場內:每回合先依序聊天(若有),
    再讓所有代理人同時行動。回合之間嚴格依序。
  * 失敗不吞掉:解析失敗會記錄 parse_ok=false 與重試次數。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .agents import Agent
from .confederates import anchor_payoffs, make_partners, parse_script, payoff_index, shuffled_speakers
from .games import make_game
from .llm import LLMClient, make_client

PERSONA_CHOICES = ("neutral", "selfish", "altruistic", "mixed")


@dataclass
class Condition:
    name: str
    persona: str = "neutral"          # neutral | selfish | altruistic | mixed
    communication: str = "none"       # none | chat
    reasoning: str = "plain"          # plain | social_cot | universalization
    lang: str = "zh"
    game_params: dict | None = None   # 例如 {"pgg": {"rounds": 10}}
    confederates: list[str] | None = None  # 有值 = 探針模式:1 位 LLM + 腳本對手(見 confederates.py)
    betray_round: int = 4             # betrayer 腳本從第幾回合開始背叛

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Condition":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        c = cls(**d)
        if c.persona not in PERSONA_CHOICES:
            raise ValueError(f"persona 必須是 {PERSONA_CHOICES}")
        if c.communication not in ("none", "chat"):
            raise ValueError("communication 必須是 none 或 chat")
        for sc in c.confederates or []:
            parse_script(sc)  # 名稱寫錯會在這裡就報錯,不用等跑了一半
        if c.confederates and c.persona == "mixed":
            raise ValueError("探針模式只有 1 位 LLM,persona 不能是 mixed")
        return c


def personas_for(condition: Condition, n_agents: int) -> list[str]:
    if condition.persona == "mixed":  # 1 位自私 + 其餘中性(IPD 為自私 vs. 中性)
        return ["selfish"] + ["neutral"] * (n_agents - 1)
    return [condition.persona] * n_agents


def stable_seed(game: str, model: str, rep: int) -> int:
    return zlib.crc32(f"{game}|{model}|{rep}".encode()) & 0x7FFFFFFF


def game_id_of(game: str, condition: str, model: str, rep: int, script: str | None = None) -> str:
    if script:  # 探針模式:腳本名稱放在條件之後,不含連字號,方便用 split("-") 解析
        return f"{game}-{condition}-{script}-{model}-{rep:03d}"
    return f"{game}-{condition}-{model}-{rep:03d}"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


async def play_game(game_name: str, condition: Condition, model_name: str, llm: LLMClient, rep: int,
                    *, temperature: float | None = 0.7, max_tokens: int = 400, max_retries: int = 2) -> list[dict]:
    """跑完一整場,回傳所有日誌列(round / chat / summary)。不做任何檔案 I/O。"""
    seed = stable_seed(game_name, model_name, rep)
    params = (condition.game_params or {}).get(game_name, {})
    game = make_game(game_name, seed, **params)
    gid = game_id_of(game_name, condition.name, model_name, rep)
    rng = random.Random(seed)

    personas = personas_for(condition, game.n_agents)
    agents = [Agent(f"A{i + 1}", i, personas[i], condition.reasoning, llm, game, lang=condition.lang,
                    temperature=temperature, max_tokens=max_tokens, max_retries=max_retries)
              for i in range(game.n_agents)]
    base = dict(game_id=gid, game=game_name, condition=condition.name, model=model_name, seed=seed, rep=rep,
                comm=condition.communication, reasoning=condition.reasoning)

    rows: list[dict] = []
    chat_log: list[dict] = []
    while not game.done:
        rnd = game.round_no + 1
        if condition.communication == "chat":
            order = list(range(game.n_agents))
            rng.shuffle(order)  # 每回合發言順序隨機,後發言者看得到前面的話
            for i in order:
                a = agents[i]
                res = await a.speak(game.history, chat_log, rnd)
                chat_log.append({"round": rnd, "speaker": a.agent_id, "message": res.message})
                rows.append({**base, "type": "chat", "round": rnd, "agent_id": a.agent_id, "persona": a.persona,
                             "message": res.message, "parse_ok": res.parse_ok, "tokens_in": res.tokens_in,
                             "tokens_out": res.tokens_out, "latency_ms": round(res.latency_ms, 1), "ts": _now()})
        results = await asyncio.gather(*(a.act(game.history, chat_log, rnd) for a in agents))
        rec = game.step([r.action for r in results])
        for a, r in zip(agents, results):
            rows.append({**base, "type": "round", "round": rnd, "agent_id": a.agent_id, "persona": a.persona,
                         "thought": r.thought, "action": r.action, "payoff": rec.payoffs[a.seat],
                         "coop": rec.coop[a.seat], "parse_ok": r.parse_ok, "attempts": r.attempts,
                         "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
                         "latency_ms": round(r.latency_ms, 1), "prompt_hash": r.prompt_hash,
                         "info": rec.info, "ts": _now()})

    total = [sum(r.payoffs[i] for r in game.history) for i in range(game.n_agents)]
    all_coop = [c for r in game.history for c in r.coop]
    rows.append({**base, "type": "summary", "rounds_played": game.round_no, "rounds_planned": game.rounds,
                 "total_payoffs": total, "welfare": sum(total), "welfare_baseline": game.welfare_baseline(),
                 "welfare_ratio": sum(total) / game.welfare_baseline(), "coop_rate": sum(all_coop) / len(all_coop),
                 "n_parse_fail": sum(1 for x in rows if x["type"] == "round" and not x["parse_ok"]),
                 **game.summary_extra(), "ts": _now()})
    return rows


async def play_probe_game(game_name: str, condition: Condition, script: str, model_name: str, llm: LLMClient,
                          rep: int, *, temperature: float | None = 0.7, max_tokens: int = 400,
                          max_retries: int = 2) -> list[dict]:
    """探針模式:座位 0 是 LLM(focal),其餘座位是腳本對手。日誌只記 focal 的 round 列,
    對手的動作放在 others_actions / others_coop 欄位;承諾訊息記成 type=chat 且 scripted=true。"""
    seed = stable_seed(game_name, model_name, rep)  # 與條件、腳本無關:不同腳本/條件共用同一個 seed
    params = (condition.game_params or {}).get(game_name, {})
    game = make_game(game_name, seed, **params)
    gid = game_id_of(game_name, condition.name, model_name, rep, script)
    rng = random.Random(seed)
    if condition.persona == "mixed":
        raise ValueError("探針模式只有 1 位 LLM,persona 不能是 mixed")

    focal = Agent("A1", 0, condition.persona, condition.reasoning, llm, game, lang=condition.lang,
                  temperature=temperature, max_tokens=max_tokens, max_retries=max_retries)
    partners = make_partners(game, script, focal_seat=0, betray_round=condition.betray_round, lang=condition.lang)
    base = dict(game_id=gid, game=game_name, condition=condition.name, model=model_name, seed=seed, rep=rep,
                comm=condition.communication, reasoning=condition.reasoning, script=script,
                betray_round=condition.betray_round)
    anchors = anchor_payoffs(game_name, seed, script, params, betray_round=condition.betray_round)

    rows: list[dict] = []
    chat_log: list[dict] = []
    while not game.done:
        rnd = game.round_no + 1
        if condition.communication == "chat":
            for i in shuffled_speakers(game.n_agents, rng):  # 發言順序隨機,後發言者看得到前面的話
                if i == 0:
                    res = await focal.speak(game.history, chat_log, rnd)
                    chat_log.append({"round": rnd, "speaker": focal.agent_id, "message": res.message})
                    rows.append({**base, "type": "chat", "round": rnd, "agent_id": focal.agent_id,
                                 "persona": focal.persona, "scripted": False, "message": res.message,
                                 "parse_ok": res.parse_ok, "tokens_in": res.tokens_in,
                                 "tokens_out": res.tokens_out, "latency_ms": round(res.latency_ms, 1), "ts": _now()})
                else:
                    p = next(x for x in partners if x.seat == i)
                    msg = p.message(game, rnd)
                    if msg:
                        chat_log.append({"round": rnd, "speaker": p.agent_id, "message": msg})
                        rows.append({**base, "type": "chat", "round": rnd, "agent_id": p.agent_id, "persona": None,
                                     "scripted": True, "message": msg, "parse_ok": True, "tokens_in": 0,
                                     "tokens_out": 0, "latency_ms": 0.0, "ts": _now()})
        res = await focal.act(game.history, chat_log, rnd)
        acts: list = [None] * game.n_agents
        acts[0] = res.action
        for p in partners:
            acts[p.seat] = p.act(game, rnd)
        rec = game.step(acts)
        rows.append({**base, "type": "round", "round": rnd, "agent_id": focal.agent_id, "persona": focal.persona,
                     "thought": res.thought, "action": res.action, "payoff": rec.payoffs[0], "coop": rec.coop[0],
                     "others_actions": rec.actions[1:], "others_coop": rec.coop[1:],
                     "parse_ok": res.parse_ok, "attempts": res.attempts, "tokens_in": res.tokens_in,
                     "tokens_out": res.tokens_out, "latency_ms": round(res.latency_ms, 1),
                     "prompt_hash": res.prompt_hash, "info": rec.info, "ts": _now()})

    focal_total = sum(r.payoffs[0] for r in game.history)
    group_total = sum(sum(r.payoffs) for r in game.history)
    coops = [r.coop[0] for r in game.history]
    rows.append({**base, "type": "summary", "rounds_played": game.round_no, "rounds_planned": game.rounds,
                 "coop_rate": sum(coops) / len(coops), "focal_payoff": focal_total,
                 "anchor_allc": anchors["allc"], "anchor_alld": anchors["alld"],
                 "payoff_index": payoff_index(focal_total, anchors),
                 "welfare": group_total, "welfare_baseline": game.welfare_baseline(),
                 "welfare_ratio": group_total / game.welfare_baseline(),
                 "n_parse_fail": sum(1 for x in rows if x["type"] == "round" and not x["parse_ok"]),
                 **game.summary_extra(), "ts": _now()})
    return rows


def write_game_log(path: Path, rows: list[dict]) -> None:
    """先寫 .part 再改名,整場完成才算數。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with part.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    part.replace(path)


async def run_experiment(condition: Condition, models: list[dict], games: list[str], reps: int, out_dir: Path,
                         *, start_rep: int = 0, concurrency: int = 8, dry_run: bool = False,
                         temperature: float | None = 0.7, max_tokens: int = 400, verbose: bool = True) -> dict:
    sem = asyncio.Semaphore(concurrency)
    clients: dict[str, LLMClient] = {m["name"]: make_client(m, dry_run=dry_run) for m in models}
    stats = {"done": 0, "skipped": 0, "failed": 0}

    async def one(game: str, model: dict, rep: int, script: str | None = None) -> None:
        gid = game_id_of(game, condition.name, model["name"], rep, script)
        path = out_dir / f"{gid}.jsonl"
        if path.exists():  # 斷點續跑
            stats["skipped"] += 1
            return
        async with sem:
            try:
                kw = dict(temperature=model.get("temperature", temperature),
                          max_tokens=model.get("max_tokens", max_tokens))
                if script is None:
                    rows = await play_game(game, condition, model["name"], clients[model["name"]], rep, **kw)
                else:
                    rows = await play_probe_game(game, condition, script, model["name"],
                                                 clients[model["name"]], rep, **kw)
            except Exception as e:  # 單場失敗不拖垮整批;下次重跑會再補這一場
                stats["failed"] += 1
                msg = f"{type(e).__name__}: {str(e)[:300]}"
                if verbose:
                    print(f"[失敗] {gid}: {msg}")
                try:  # 把失敗原因留在日誌資料夾(.txt,不會被當成對局資料讀進分析)
                    with open(out_dir / "_failures.txt", "a", encoding="utf-8") as fh:
                        fh.write(f"{_now()}\t{gid}\t{getattr(e, 'status_code', '')}\t{msg}\n")
                except OSError:
                    pass
                return
            write_game_log(path, rows)
            stats["done"] += 1
            if verbose:
                s = rows[-1]
                print(f"[完成] {gid} coop={s['coop_rate']:.2f} welfare={s['welfare_ratio']:.2f}")

    scripts: list[str | None] = list(condition.confederates) if condition.confederates else [None]
    jobs = [one(g, m, r, sc) for m in models for g in games for sc in scripts
            for r in range(start_rep, start_rep + reps)]
    await asyncio.gather(*jobs)
    return stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="跑一個條件的賽局實驗")
    p.add_argument("--condition", required=True, help="條件 YAML,例如 configs/conditions/c0_baseline.yaml")
    p.add_argument("--models", required=True, help="models.yaml")
    p.add_argument("--games", default="ipd,pgg,cpr")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--start-rep", type=int, default=0)
    p.add_argument("--out", default="logs")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--only-model", default=None, help="只跑 models.yaml 裡這個 name")
    p.add_argument("--dry-run", action="store_true", help="用 mock 模型,不連網路")
    args = p.parse_args(argv)

    cond = Condition.from_yaml(args.condition)
    models = yaml.safe_load(Path(args.models).read_text(encoding="utf-8"))["models"]
    if args.only_model:
        models = [m for m in models if m["name"] == args.only_model]
        if not models:
            raise SystemExit(f"models.yaml 裡沒有 {args.only_model!r}")
    stats = asyncio.run(run_experiment(cond, models, args.games.split(","), args.reps, Path(args.out),
                                       start_rep=args.start_rep, concurrency=args.concurrency,
                                       dry_run=args.dry_run))
    print(f"完成 {stats['done']} 場,跳過 {stats['skipped']} 場,失敗 {stats['failed']} 場")


if __name__ == "__main__":
    main()
