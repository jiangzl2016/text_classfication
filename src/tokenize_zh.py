"""Chinese tokenizer wrapper for use with sklearn's TfidfVectorizer."""
from __future__ import annotations

import re

import jieba

jieba.initialize()

_NON_CJK_ALNUM = re.compile(r"^[\W\d_]+$", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens = jieba.lcut(text, cut_all=False)
    out: list[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if _NON_CJK_ALNUM.fullmatch(tok):
            continue
        if len(tok) == 1 and not _is_cjk(tok):
            continue
        out.append(tok)
    return out


def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"
