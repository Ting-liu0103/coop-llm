from .base import Game, RoundRecord
from .cpr import CPR
from .ipd import IPD
from .pgg import PGG

GAMES = {"ipd": IPD, "pgg": PGG, "cpr": CPR}


def make_game(name: str, seed: int, **params) -> Game:
    """依名稱建立賽局。params 會傳給建構子(rounds、endowment 等)。"""
    try:
        return GAMES[name](seed=seed, **params)
    except KeyError:
        raise ValueError(f"未知賽局 {name!r},可用:{sorted(GAMES)}") from None


__all__ = ["Game", "RoundRecord", "IPD", "PGG", "CPR", "GAMES", "make_game"]
