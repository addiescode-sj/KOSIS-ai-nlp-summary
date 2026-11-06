"""Korean particles settled by the word they follow (T1-28 S2b). Moved from kosis_chat/judge/messages.py with the
bodies unchanged: the judge's refusal sentences, compute's refusal sentences (through messages.refusal_message), the
template sentences (answer/template.py) and the rendered answer (answer/render.josa) all use them. judge/messages.py
re-exports every name.
"""

from __future__ import annotations

import re

# Final consonant of the sound a word ends with: digits and unit symbols as they are read (3 = 삼, % = 퍼센트).
_READ_LAST = {"0": "영", "1": "일", "2": "이", "3": "삼", "4": "사", "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구",
              "%": "트", "p": "트", "㎢": "터"}
_PAIRS = {"이": ("이", "가"), "은": ("은", "는"), "을": ("을", "를"), "과": ("과", "와"), "으로": ("으로", "로")}


def josa(word: str, particle: str) -> str:
    """word + the particle form its last sound takes: josa('서울특별시', '이') -> '서울특별시가'."""
    with_final, without = _PAIRS[particle]
    last = next((ch for ch in reversed(word.strip()) if not ch.isspace() and ch not in ")'\""), "")
    last = _READ_LAST.get(last, last)
    code = ord(last) - 0xAC00 if last else -1
    if not 0 <= code < 11172:
        return word + without
    final = code % 28
    if particle == "으로" and final == 8:  # ㄹ: 서울로
        return word + without
    return word + (with_final if final else without)


_PLACEHOLDER_JOSA = re.compile(r"(\S+?)(은\(는\)|와\(과\)|이\(가\)|을\(를\))")
_FORM = {"은(는)": "은", "와(과)": "과", "이(가)": "이", "을(를)": "을"}


def with_josa(text: str) -> str:
    """'조출생률'은(는) -> '조출생률'은: settle placeholder particles once the word is known (T1-14)."""
    return _PLACEHOLDER_JOSA.sub(lambda m: josa(m.group(1), _FORM[m.group(2)]), text)
