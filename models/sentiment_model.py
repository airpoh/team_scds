from __future__ import annotations
from typing import List, Dict
import re

# Try HF transformers; fall back to rules if not available
try:
    from transformers import pipeline
    _hf = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    _has_hf = True
except Exception:
    _has_hf = False

_POS = re.compile(r"\b(love|great|amazing|happy|recommend|works? well|stunning|best)\b", re.I)
_NEG = re.compile(r"\b(terrible|awful|bad|worse|refund|broken|disappointed|worst|greasy|itchy)\b", re.I)

def _rule_predict(texts: List[str]) -> List[Dict[str, float]]:
    out = []
    for t in texts:
        t = t or ""
        if _NEG.search(t): out.append({"label":"negative","score":0.85})
        elif _POS.search(t): out.append({"label":"positive","score":0.80})
        else: out.append({"label":"neutral","score":0.50})
    return out

def predict_sentiment(texts: List[str]) -> List[Dict[str, float]]:
    if _has_hf:
        try:
            raw = _hf(texts, truncation=True)
            res = []
            for r in raw:
                lab = r["label"].lower()
                if "pos" in lab: lab = "positive"
                elif "neg" in lab: lab = "negative"
                else: lab = "neutral"
                res.append({"label": lab, "score": float(r["score"])})
            return res
        except Exception:
            return _rule_predict(texts)
    return _rule_predict(texts)
