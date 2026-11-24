"""代理人的記憶:最近 keep_last 回合逐字保留,更早的回合壓縮成統計摘要。

摘要由賽局程式計算,不是 LLM 寫的,所以所有代理人的摘要品質一致。
每場從空白開始,不跨場記憶。
"""
from __future__ import annotations

from ..games.base import Game, RoundRecord


def build_memory(game: Game, records: list[RoundRecord], viewer: int, lang: str = "zh",
                 keep_last: int = 3) -> str:
    if not records:
        return ""
    early, recent = records[:-keep_last], records[-keep_last:]
    parts = []
    if early:
        parts.append(game.summarize_history(early, viewer, lang))
    parts.extend(game.describe_round(r, viewer, lang) for r in recent)
    return "\n".join(p for p in parts if p)


def build_chat_text(chat_log: list[dict], round_no: int, keep_last: int = 3) -> str:
    """chat_log 每項:{round, speaker, message}。只顯示最近 keep_last 回合(含目前回合)。"""
    lo = round_no - keep_last + 1
    lines = [f"[{c['round']}] {c['speaker']}: {c['message']}" for c in chat_log if c["round"] >= lo]
    return "\n".join(lines)
