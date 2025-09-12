# components/kpis.py
import streamlit as st
import pandas as pd

def _numseries(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """
    Return a numeric Series from the first existing candidate column.
    If none exist, return a zero Series aligned to df.
    """
    for name in candidates:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
    return pd.Series([0] * len(df), index=df.index, dtype="float64")

def render(df: pd.DataFrame):
    st.subheader("📊 KPIs")

    comments = int(len(df))
    likes_s  = _numseries(df, ["like_count", "likes"])
    shares_s = _numseries(df, ["share_count", "shares"])
    saves_s  = _numseries(df, ["save_count", "saves"])

    likes, shares, saves = int(likes_s.sum()), int(shares_s.sum()), int(saves_s.sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comments", f"{comments:,}")
    c2.metric("Likes",    f"{likes:,}")
    c3.metric("Shares",   f"{shares:,}")
    c4.metric("Saves",    f"{saves:,}")

    extras = []
    if "is_quality" in df.columns and comments:
        extras.append(f"Quality ratio: {df['is_quality'].mean():.1%}")
    if "spamness" in df.columns and comments:
        extras.append(f"Avg spamness: {df['spamness'].mean():.2f}")
    if "sentiment" in df.columns and comments:
        pos = (df["sentiment"].astype(str).str.upper() == "POSITIVE").mean()
        extras.append(f"Positive sentiment: {pos:.1%}")
    if extras:
        st.caption(" | ".join(extras))