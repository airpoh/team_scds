from __future__ import annotations
from typing import List
import numpy as np, re

LINK = re.compile(r"(http[s]?://|www\.|bit\.ly|wa\.me|t\.me|telegram|instagram\.com/[\w_]+)", re.I)
PHONE = re.compile(r"(\+?\d[\d\-\s]{7,})")
COUPON = re.compile(r"(promo|coupon|discount|code|voucher)", re.I)
REPEAT = re.compile(r"(.)\1{3,}")
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")

def spam_scores(texts: List[str]) -> np.ndarray:
    vals = []
    for t in texts:
        t = t or ""
        s = 0
        s += 1 if LINK.search(t) else 0
        s += 1 if PHONE.search(t) else 0
        s += 1 if COUPON.search(t) else 0
        s += 1 if REPEAT.search(t) else 0
        s += 1 if len(EMOJI.findall(t)) >= 5 else 0
        s += 1 if len(t) < 5 else 0
        vals.append(min(s/5.0, 1.0))
    return np.array(vals, dtype=float)

def civility_scores(texts: List[str]) -> np.ndarray:
    BAD = re.compile(r"\b(damn|shit|idiot|stupid)\b", re.I)
    return np.array([0.2 if BAD.search((t or "")) else 1.0 for t in texts], dtype=float)
