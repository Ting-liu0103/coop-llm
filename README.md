# coop-llm:LLM 代理人在社會困境賽局中的合作實驗(程式骨架)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ting-liu0103/coop-llm/blob/main/notebooks/colab_run.ipynb)

三個賽局(迭代囚徒困境 IPD、公共財 PGG、共享資源 CPR)+ 代理人(人格 / 溝通 / 推理策略)+ 實驗執行器 + 分析工具。
**預設全程可用 mock 模型離線跑通**,接上真實模型只需改 `configs/models.yaml`。

## 安裝與測試
```bash
pip install -e ".[dev,plots]"          # 要用真實模型再加 openai 或 anthropic
python -m pytest -q                    # 應全數通過(78 項)
```


## 研究切入點:合作的「品質」,不只是合作的「量」(2026-11 修訂)
老師指出「把人格、溝通、推理放在一起」不算新東西。查證後,這些因子各自都有人做過(見 `docs/related_work.md`)。
因此本專案的核心改成一個前人沒有統一回答的問題:

> 人格、溝通、推理這些介入,是讓代理人**更懂得分辨好人與壞人**,還是只是讓它**更愛合作**(因而更容易被剝削)?

做法是**探針實驗**:每場只有 1 位 LLM,其餘座位由腳本對手扮演(永遠合作 / 永遠背叛 / 鏡射 / 假朋友,
以及「公開承諾合作但實際行為不同」的版本)。核心指標是**辨別力 DI = 對好人的合作 − 對壞人的合作**。
聊天條件下「好人」與「壞人」說一模一樣的承諾,只有行為不同,所以 DI 量的是「看行為還是聽承諾」。
原本的多代理人自我對弈(C0–C9)保留為輔助實驗,用來檢查探針的結論在真實互動中是否仍成立。

```bash
for f in configs/probe/*.yaml; do
  python -m coop_llm.runner --condition $f --models configs/models_mock.yaml --reps 6 --out logs/probe --dry-run
done
python -m coop_llm.analyze_probe --logs logs/probe --out results/probe
```
| 條件 | 設定 | 腳本對手 |
|---|---|---|
| P0 | 基準(中性、無聊天、一般 prompt) | allc, alld, tft, betrayer |
| P1 / P2 | 自私 / 利他人格 | 同上 |
| P3 / P4 | Social CoT / Universalization | 同上 |
| P5 | 回合前聊天 | allc_promise, alld_promise, betrayer_promise |

指標(`analysis/probe.py`):`coop_r1`(第 1 回合,反映先驗/對承諾的信任)、`coop_from_r2`、`payoff_index`
(對照「永遠合作/永遠背叛」兩個錨點的正規化報酬,可精確計算)、`adapt_lag`(被背叛後幾回合才停止合作)、DI。
估計呼叫量:每格 120 場(3 賽局 × 2 模型 × 20 重複),23 格,上限約 3.1 萬次呼叫(含聊天),少於自我對弈的 5.6 萬次。

**限制(誠實聲明)**:腳本對手不會適應,因此探針量的是「LLM 對固定行為模式的反應」,不是完整的社會互動;
這正是它因果上乾淨的原因,也是需要保留自我對弈輔助實驗的原因。

## 快速試跑(不花錢、不連網)
```bash
python -m coop_llm.runner --condition configs/conditions/c0_baseline.yaml \
    --models configs/models_mock.yaml --reps 3 --out logs/dry --dry-run
python -m coop_llm.runner --condition configs/conditions/c1_selfish.yaml \
    --models configs/models_mock.yaml --reps 3 --out logs/dry --dry-run
python -m coop_llm.analyze --logs logs/dry --out results/dry
```

## 接上真實模型
1. 編輯 `configs/models.yaml`:填 `model` 型號、`api_key_env`(環境變數名稱);本地模型(Ollama/vLLM)設 `base_url`。
2. `export OPENAI_API_KEY=...`(**金鑰只放環境變數,不要寫進檔案或進版控**)。
3. 先小量 pilot,再放大:
```bash
python -m coop_llm.runner --condition configs/conditions/c0_baseline.yaml \
    --models configs/models.yaml --games ipd pgg cpr --reps 2 --out logs/pilot
```
常用參數:`--only-model NAME`、`--start-rep N`(追加重複)、`--concurrency K`(同時進行的場數)。

## 目錄
| 路徑 | 內容 |
|---|---|
| `src/coop_llm/games/` | `ipd.py` `pgg.py` `cpr.py`:純規則,不呼叫 LLM;`make_game(name, seed)` |
| `src/coop_llm/agents/` | `prompts.py`(中英提示詞、人格、推理策略)、`memory.py`(近 3 回合原文 + 程式產生的統計摘要)、`agent.py`(解析 JSON、重試、後備) |
| `src/coop_llm/llm/` | 統一介面 `LLMClient`;`openai_compat`、`anthropic_client`、`mock` |
| `src/coop_llm/runner.py` | 條件、執行、日誌、續跑(自我對弈 `play_game`、探針 `play_probe_game`) |
| `src/coop_llm/confederates.py` | 腳本對手、承諾訊息、錨點報酬 |
| `src/coop_llm/analysis/` + `analyze.py` | 指標、置換檢定、bootstrap CI、Cliff's delta、Holm 校正、圖 |
| `configs/probe/` | 探針實驗條件 P0–P5(主要實驗) |
| `configs/conditions/` | C0–C9 自我對弈條件(輔助實驗) |
| `tests/` | 規則、代理人、執行器、統計的測試 |

## 實驗條件
C0 基線|C1 自私|C2 利他|C3 混合(1 自私 + 其餘中性)|C4 溝通|C5 Social CoT|C6 Universalization|C7–C9 交互作用(可選)。
新增條件:複製一份 YAML,改 `name / persona / communication / reasoning / lang / game_params` 即可。

## 日誌格式
每場一個 `logs/<game>-<cond>-<model>-<rep>.jsonl`,三種列:
- `chat`:溝通階段的每則訊息
- `round`:每回合每位代理人一列 —— `thought, action, payoff, coop, parse_ok, tokens_in, tokens_out`
- `summary`:整場摘要(`coop_rate, welfare_ratio, rounds_played, n_parse_fail, collapse_round, survived`)

寫入先寫 `.part` 再改名,所以當掉不會留下半場資料;重跑同一指令會自動跳過已完成的場次。

## 設計決定與注意事項
- **共同隨機數**:seed = crc32(賽局|模型|重複編號),**與條件無關**,所以各條件的 IPD 標籤(X/Y 誰是合作)一致,比較更乾淨。
- **解析失敗不會被默默丟掉**:重試 2 次後退回「上一回合動作 / 預設值」,並記 `parse_ok=False`;`analyze` 會輸出 `parse_failures.csv`。若某條件失敗率偏高,應在報告中說明。
- **同一回合內,各代理人的決策是同時(併發)送出**,不是依序;回合與回合之間仍是依序。這比設計書「同一場內依序」的字面寫法更貼近同時行動的賽局定義(也避免後行動者看到前者的動作)。若要嚴格依序,改 `runner.play_game` 的 `asyncio.gather` 即可 —— 請確認這符合你的設計意圖。
- 分析單位是**場(game)**,不是回合或代理人;兩個模型分開報告。
- IPD 以 X/Y 選項偽裝,提示詞不出現「囚徒困境」等字眼,避免模型直接背誦既有答案。
- `temperature` 預設 0.7;部分模型不接受自訂溫度,在 `models.yaml` 設 `temperature: null`。

## 尚未驗證(誠實聲明)
- 測試只覆蓋規則、解析、日誌與統計;**尚未對真實 API 實跑**,提供者的參數差異(如 token 欄位、JSON 模式)可能需要小調整。
- `models.yaml` 的型號是占位字串。
- mock 策略只用來驗證管線,其結果沒有任何科學意義。
