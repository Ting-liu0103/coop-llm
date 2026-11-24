"""探針實驗一鍵分析:python -m coop_llm.analyze_probe --logs logs/probe --out results/probe

輸出:
  probe_game_level.csv     每場一列(coop_r1, coop_from_r2, payoff_index, adapt_lag)
  di_pairs.csv             每個 (game,model,condition,rep) 一列:對好人/壞人的合作與辨別力 DI
  comparisons_<metric>.csv 每個條件對基準的檢定(置換 + bootstrap CI + Cliff's delta + Holm)
  quality_map_*.png        x=對壞人合作(天真度)、y=對好人合作;左上最好
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import compare_to_baseline, load_logs, paired_di, plot_quality_map, probe_game_level

PAIR_METRICS = ["di", "coop_friendly", "coop_hostile", "coop_r1", "exploit_protection"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs/probe")
    p.add_argument("--out", default="results/probe")
    p.add_argument("--baseline", default="P0")
    p.add_argument("--n-perm", type=int, default=10_000)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    rounds, _chats, summary = load_logs(args.logs)
    pg = probe_game_level(rounds, summary)
    if pg.empty:
        raise SystemExit(f"{args.logs} 裡沒有探針日誌(summary 需有 script 欄位)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pg.to_csv(out / "probe_game_level.csv", index=False)
    pairs = paired_di(pg)
    pairs.to_csv(out / "di_pairs.csv", index=False)

    if args.baseline in set(pairs["condition"]):
        for m in PAIR_METRICS:
            compare_to_baseline(pairs, m, args.baseline, n_perm=args.n_perm, n_boot=args.n_boot) \
                .to_csv(out / f"comparisons_{m}.csv", index=False)
    else:
        print(f"注意:資料裡沒有基準條件 {args.baseline!r},略過檢定")
    bet = pg[pg["script"].str.startswith("betrayer")]
    if not bet.empty and args.baseline in set(bet["condition"]):
        compare_to_baseline(bet, "adapt_lag", args.baseline, n_perm=args.n_perm, n_boot=args.n_boot) \
            .to_csv(out / "comparisons_adapt_lag.csv", index=False)
    if not args.no_plots:
        for pth in plot_quality_map(pairs, out):
            print("圖:", pth)
    print(f"已輸出到 {out}/(共 {pg['game_id'].nunique()} 場,配對 {len(pairs)} 組)")


if __name__ == "__main__":
    main()
