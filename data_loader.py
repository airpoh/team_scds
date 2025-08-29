from __future__ import annotations
import pandas as pd
import numpy as np
from typing import IO
from models.sentiment_model import predict_sentiment
from models.spam_model import spam_scores, civility_scores

def load_csv(file: IO) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _ensure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "comment" not in df.columns:
        for alt in ["comment_text","text","review"]:
            if alt in df.columns:
                df.rename(columns={alt: "comment"}, inplace=True)
                break
    for c in ["post_text","like_count","share_count","save_count"]:
        if c not in df.columns: df[c] = "" if c=="post_text" else 0
    df["post_text"] = df["post_text"].fillna("").astype(str)
    df["comment"]   = df["comment"].fillna("").astype(str)
    return df

def process_dataframe(df_raw: pd.DataFrame, tau: float = 0.6, spam_cap: float = 1.0) -> pd.DataFrame:
    df = _ensure(df_raw)

    # Models
    sent = predict_sentiment(df["comment"].tolist())
    df["sentiment"] = [s["label"] for s in sent]
    df["sentiment_score"] = [s["score"] for s in sent]
    df["spamness"] = spam_scores(df["comment"].tolist())
    df["civility"] = civility_scores(df["comment"].tolist())

    # Lightweight constructiveness/specificity heuristics (no extra model)
    df["constructiveness"] = df["comment"].str.contains(
        r"should|recommend|try|because|improve|better|i used|i tried|after|since",
        case=False, regex=True
    ).astype(float) * 0.7
    df["specificity"] = (df["comment"].str.len()/250.0).clip(0,1)

    # CQS (weights can be tuned later)
    w = dict(relevance=0.0, constructiveness=0.30, specificity=0.25, civility=0.20, spamness=0.25)
    df["cqs"] = (
        w["constructiveness"]*df["constructiveness"]
        + w["specificity"]*df["specificity"]
        + w["civility"]*df["civility"]
        - w["spamness"]*df["spamness"]
    ).clip(0,1)

    # Filters/flags
    df = df[df["spamness"] <= spam_cap].copy()
    df["is_quality"] = df["cqs"] >= tau
    return df.reset_index(drop=True)
