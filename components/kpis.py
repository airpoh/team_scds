import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    likes  = df.get("like_count", 0).sum()
    shares = df.get("share_count", 0).sum()
    saves  = df.get("save_count", 0).sum()
    soe = likes + shares + saves + len(df)           # total engagement proxy
    soqe = float(df["cqs"].sum() / max(soe, 1))      # Share of Quality Engagement
    qcr  = float(df["is_quality"].mean()) if len(df) else 0.0
    avg  = float(df["cqs"].mean()) if len(df) else 0.0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("SoE", f"{soe:,.0f}")
    c2.metric("SoQE", f"{soqe:.3f}")
    c3.metric("QCR", f"{qcr:.2%}")
    c4.metric("Avg CQS", f"{avg:.3f}")
