import streamlit as st
import pandas as pd

from state import init_state
from data_loader import load_csv, process_dataframe

from components.header import render as Header
from components.kpis import render as KPIs
from components.sentiment_spam import render as SentimentSpam

st.set_page_config(page_title="CommentSense (Lite)", layout="wide")
init_state()

# Header
Header("CommentSense · Lite Prototype")

# Sidebar
with st.sidebar:
    st.header("Upload & Controls")
    file = st.file_uploader("CSV with at least a 'comment' column", type=["csv"])
    st.divider()
    st.session_state.tau = st.slider("Quality Threshold (τ)", 0.0, 1.0, st.session_state.tau, 0.05)
    st.session_state.spam_max = st.slider("Max Spamness", 0.0, 1.0, st.session_state.spam_max, 0.05)

# Data
if file:
    df_raw = load_csv(file)
else:
    df_raw = pd.DataFrame({
        "post_text": [
            "New niacinamide serum launched this week!",
            "Floral eau de parfum with citrus notes.",
            "Long-wear matte foundation."
        ],
        "comment": [
            "I tried it 2 weeks—better than my old brand!",
            "DM me for discount code http://bit.ly/SALE",
            "Is it ok for oily skin? SPF would help."
        ],
        "like_count":[100,80,50], "share_count":[10,8,5], "save_count":[20,12,7],
    })

with st.spinner("Scoring comments…"):
    df = process_dataframe(df_raw, tau=st.session_state.tau, spam_cap=st.session_state.spam_max)

# Components
KPIs(df)
SentimentSpam(df)

st.subheader("Comments Explorer")
st.dataframe(
    df[["post_text","comment","sentiment","sentiment_score","spamness","cqs","is_quality"]],
    use_container_width=True
)

st.download_button(
    "Download enriched CSV",
    df.to_csv(index=False).encode("utf-8"),
    "commentsense_enriched.csv",
    "text/csv",
)
