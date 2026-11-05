"""Step 8: tokenizer + halka stemmer (BM25 aur fake embedder ke liye)."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been by with as at from that this it its "
    "under shall may can what which who how when where does do did if not any all such than then "
    "there their these those into per".split()
)


def stem(w: str) -> str:
    """Halka suffix-stripper (Porter nahi, heuristic). Plural / -ing / -ed / trailing -e ko ek jaisa karta hai.
    issued, issue, issues -> issu | lending, lend -> lend | banks -> bank | policies -> policy"""
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("es") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def tokenize(text: str, use_stem: bool = True, drop_stop: bool = True) -> list:
    toks = _TOKEN.findall((text or "").lower())
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    if use_stem:
        toks = [stem(t) for t in toks]
    return toks