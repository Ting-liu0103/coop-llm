"""賽局基底類別。

每個賽局負責:規則文字、動作解析與驗證、報酬計算、回合描述(給代理人看的歷史)。
賽局本身不碰 LLM,方便寫單元測試。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoundRecord:
    """一個回合的公開紀錄。index 為座位編號(0 起算)。"""

    round: int  # 1 起算
    actions: list[Any]
    payoffs: list[float]
    coop: list[float]  # 每位代理人本回合的合作分數,範圍 0~1
    info: dict = field(default_factory=dict)  # 公開的額外資訊,例如剩餘庫存


class Game(ABC):
    name: str = "game"
    n_agents: int = 2

    def __init__(self, seed: int = 0, rounds: int = 10):
        self.seed = seed
        self.rounds = rounds
        self.round_no = 0  # 已完成的回合數
        self.done = False
        self.history: list[RoundRecord] = []

    # ---- 給代理人看的文字 -------------------------------------------------
    @abstractmethod
    def rules_text(self, lang: str = "zh") -> str:
        """完整規則與報酬公式(不出現經典賽局名稱,見設計書的偽裝包裝)。"""

    @abstractmethod
    def action_prompt(self, lang: str = "zh") -> str:
        """告訴代理人 action 欄位該填什麼。"""

    def public_state(self, lang: str = "zh") -> str:
        """目前的公開狀態(例如庫存)。預設沒有。"""
        return ""

    @abstractmethod
    def describe_round(self, rec: RoundRecord, viewer: int, lang: str = "zh") -> str:
        """一個回合的逐字描述,從 viewer 的視角。"""

    @abstractmethod
    def summarize_history(self, recs: list[RoundRecord], viewer: int, lang: str = "zh") -> str:
        """較早回合的統計摘要。由程式產生,不用 LLM,確保所有代理人摘要品質相同。"""

    # ---- 動作 -------------------------------------------------------------
    @abstractmethod
    def parse_action(self, raw: Any) -> Any | None:
        """把模型輸出轉成合法動作,不合法回傳 None。"""

    @abstractmethod
    def default_action(self) -> Any:
        """解析失敗且沒有上一回合動作時使用的預設行動(會被記錄 parse_ok=False)。"""

    @abstractmethod
    def action_space(self) -> dict:
        """給 mock 模型與分析用的動作空間描述,含 cooperate / defect 錨點。"""

    # ---- 推進 -------------------------------------------------------------
    @abstractmethod
    def step(self, actions: list[Any]) -> RoundRecord:
        """推進一個回合,更新內部狀態與 self.done,並把紀錄存入 self.history。"""

    @abstractmethod
    def welfare_baseline(self) -> float:
        """基準總報酬(全隊、全部回合),用來算社會福利比。"""

    # ---- 共用 -------------------------------------------------------------
    def _commit(self, rec: RoundRecord) -> RoundRecord:
        self.round_no = rec.round
        self.history.append(rec)
        if self.round_no >= self.rounds:
            self.done = True
        return rec

    def summary_extra(self) -> dict:
        """寫進 summary 列的賽局專屬欄位(例如崩潰回合)。"""
        return {}
