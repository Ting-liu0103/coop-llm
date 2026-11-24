"""一鍵分析:python -m coop_llm.analyze --logs logs --out results

輸出:
  results/game_level.csv        每場一列的指標
  results/comparisons_<metric>.csv   每個條件對 C0 的檢定結果(含 Holm 校正)
  results/parse_failures.csv    解析失敗率
  results/coop_by_round_*.png   主要圖表
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import compare_to_baseline, game_level, load_logs, parse_failure_report
from .analysis.plots import plot_coop_by_round

METRICS = ["coop_rate", "mean_payoff", "welfare_ratio", "retaliation", "forgiveness"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs")
    p.add_argument("--out", default="results")
    p.add_argument("--baseline", default="C0")
    p.add_argument("--n-perm", type=int, default=10_000)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    rounds, _chats, summary = load_logs(args.logs)
    if rounds.empty:
        raise SystemExit(f"{args.logs} 裡沒有完成的日誌")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    games = game_level(rounds, summary)
    games.to_csv(out / "game_level.csv", index=False)
    parse_failure_report(rounds).to_csv(out / "parse_failures.csv", index=False)
    for m in METRICS:
        if m in games.columns and args.baseline in set(games["condition"]):
            compare_to_baseline(games, m, args.baseline, n_perm=args.n_perm, n_boot=args.n_boot) \
                .to_csv(out / f"comparisons_{m}.csv", index=False)
    if not args.no_plots:
        for pth in plot_coop_by_round(rounds, out):
            print("圖:", pth)
    print(f"已輸出到 {out}/(共 {games['game_id'].nunique()} 場)")


if __name__ == "__main__":
    main()
