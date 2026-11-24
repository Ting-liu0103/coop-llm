"""所有實驗用到的 prompt 文字都集中在這裡,方便改寫與做穩健性檢查(換句話說的版本)。

三個實驗因子只改動很小的片段,其餘文字完全相同:
  persona   人格段落(進系統 prompt)
  reasoning 推理指令(進每回合訊息)
  chat      是否有回合前聊天(另一個獨立的呼叫)
推理指令與人格措辭是依 Akata et al. 與 GovSim 的想法自行改寫的設計,不是論文原文。
"""
from __future__ import annotations

PERSONAS = {
    "zh": {
        "neutral": "你的目標是盡量獲得最多的點數。",
        "selfish": "你的目標是盡量獲得最多的點數。你只關心自己的點數,不在意其他人得到多少。",
        "altruistic": "你的目標是盡量獲得最多的點數。你很在意團體整體的成果,密切關心所有人的總分。",
    },
    "en": {
        "neutral": "Your goal is to earn as many points as possible.",
        "selfish": "Your goal is to earn as many points as possible. You only care about your own points "
                   "and do not care how many points others get.",
        "altruistic": "Your goal is to earn as many points as possible. You care a lot about the group's overall "
                      "outcome and closely watch everyone's total points.",
    },
}

REASONING = {
    "zh": {
        "plain": "請簡短說明你的理由,再給行動。",
        "social_cot": "先預測其他人這回合最可能怎麼做,再根據預測選行動。",
        "universalization": "先想:如果每一位參與者都照你這樣做,結果會怎麼樣?再選行動。",
    },
    "en": {
        "plain": "Briefly explain your reasoning, then give your action.",
        "social_cot": "First predict what the others will most likely do this round, then choose your action based on that prediction.",
        "universalization": "First think: if every participant did exactly what you do, what would happen? Then choose your action.",
    },
}

SYSTEM = {
    "zh": "你是參與者 {name}。你正在參與一個多回合的決策遊戲。\n\n{rules}\n\n{persona}\n\n每回合你要決定一個行動。",
    "en": "You are participant {name}. You are taking part in a multi-round decision game.\n\n{rules}\n\n{persona}\n\nEach round you must decide on one action.",
}

ROUND = {
    "zh": ("現在是第 {round} 回合(共 {rounds} 回合)。\n{state}{memory}{chat}\n{reasoning}\n"
           "只回覆一個 JSON 物件,格式:{{\"thought\": \"<你的簡短理由>\", \"action\": <你的行動>}}。\n{action_prompt}"),
    "en": ("This is round {round} of {rounds}.\n{state}{memory}{chat}\n{reasoning}\n"
           "Reply with a single JSON object of the form: {{\"thought\": \"<your brief reasoning>\", \"action\": <your action>}}.\n{action_prompt}"),
}

CHAT = {
    "zh": ("現在是第 {round} 回合(共 {rounds} 回合)開始前的聊天時間。\n{state}{memory}{chat}\n"
           "請對其他參與者說一句話,不超過 {chars} 字。這句話不會改變遊戲規則,只是溝通。\n"
           "只回覆一個 JSON 物件,格式:{{\"message\": \"<你要說的話>\"}}。"),
    "en": ("Chat time before round {round} of {rounds}.\n{state}{memory}{chat}\n"
           "Say one sentence to the other participants, at most {chars} characters. It does not change the game rules.\n"
           "Reply with a single JSON object of the form: {{\"message\": \"<what you say>\"}}."),
}

REPAIR = {
    "zh": "你上一則回覆無法解析。請只回覆符合格式的 JSON 物件,不要加其他文字。{action_prompt}",
    "en": "Your previous reply could not be parsed. Reply with only a JSON object in the required format. {action_prompt}",
}

LABELS = {
    "zh": {"history": "【過去的回合】\n", "chat": "【最近的聊天】\n", "no_history": "(這是第一回合,還沒有歷史。)\n"},
    "en": {"history": "[Previous rounds]\n", "chat": "[Recent chat]\n", "no_history": "(This is the first round; no history yet.)\n"},
}


def build_system(name: str, rules: str, persona: str, lang: str = "zh") -> str:
    return SYSTEM[lang].format(name=name, rules=rules, persona=PERSONAS[lang][persona])


def build_round(*, round_no: int, rounds: int, state: str, memory: str, chat: str, reasoning: str,
                action_prompt: str, lang: str = "zh") -> str:
    return ROUND[lang].format(
        round=round_no, rounds=rounds,
        state=(state + "\n") if state else "",
        memory=_mem(memory, lang), chat=_chat(chat, lang),
        reasoning=REASONING[lang][reasoning], action_prompt=action_prompt)


def build_chat(*, round_no: int, rounds: int, state: str, memory: str, chat: str, chars: int,
               lang: str = "zh") -> str:
    return CHAT[lang].format(round=round_no, rounds=rounds, state=(state + "\n") if state else "",
                             memory=_mem(memory, lang), chat=_chat(chat, lang), chars=chars)


def build_repair(action_prompt: str, lang: str = "zh") -> str:
    return REPAIR[lang].format(action_prompt=action_prompt)


def _mem(memory: str, lang: str) -> str:
    lab = LABELS[lang]
    return lab["history"] + memory + "\n" if memory else lab["no_history"]


def _chat(chat: str, lang: str) -> str:
    return LABELS[lang]["chat"] + chat + "\n" if chat else ""
