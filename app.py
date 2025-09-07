# app.py
from __future__ import annotations

import math
import streamlit as st
import pandas as pd

# ---- Project imports ----
from state import init_state
from data_loader import load_upload, process_dataframe  # CSV/ZIP loader + scoring

from components.header import render as Header
from components.kpis import render as KPIs
from components.sentiment_spam import render as SentimentSpam

from models.alvin_utils import process_alvin
from components.categories_topics import render as CategoriesTopics
from components.personas import render as Personas
from components.influencers import render as Influencers


# ===================== Tunables / Safe Defaults =====================
# Rows used for the Alvin track (categories/personas/influencers) — keeps graph fast
ROW_CAP_FOR_ALVIN = 30_000

# Rows used for interactive scoring in the UI to avoid OOM on laptops
ROWS_FOR_SCORING = 250_000

# Graph fast mode
FAST_MODE = True

# Heavy NLP toggles (set True if you really want them and have RAM/GPU)
USE_ZEROSHOT = True         # facebook/bart-large-mnli (very big)
USE_PERSONA_CLUSTER = True  # SBERT + HDBSCAN
USE_BERTOPIC = True         # Topic model

# ===================== Page / State =====================
st.set_page_config(page_title="CommentSense (Lite)", layout="wide")
init_state()

# ===================== Header =====================
Header("CommentSense · Lite Prototype")
st.caption("SoE + quality comments with sentiment & spam insights")

# ===================== Sidebar (minimal) =====================
with st.sidebar:
    st.header("Upload & Controls")
    file_comments = st.file_uploader(
        "Comments CSV or ZIP (YouTube schema or a 'comment'/'text' column)",
        type=["csv", "zip"],
        key="uploader_comments",
    )
    st.divider()
    st.session_state.tau = st.slider("Quality Threshold (τ)", 0.0, 1.0, st.session_state.tau, 0.05)
    st.session_state.spam_max = st.slider("Max Spamness", 0.0, 1.0, st.session_state.spam_max, 0.05)

    st.divider()
    # Optional, RAM-safe full scoring to Parquet (chunks). Needs pyarrow installed.
    def _batch_score_parquet(df_raw: pd.DataFrame, out_path: str,
                             tau: float, spam_cap: float, chunk: int = 100_000):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception:
            st.error("pyarrow not installed. `pip install pyarrow` to enable batch scoring.")
            return None

        prog = st.progress(0.0, text="Batch scoring…")
        writer = None
        total = len(df_raw)
        for i in range(0, total, chunk):
            part_raw = df_raw.iloc[i:i+chunk].copy()
            # row_cap=None to ensure we score the whole chunk
            part = process_dataframe(part_raw, tau=tau, spam_cap=spam_cap, row_cap=None)
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            del part, part_raw, table
            prog.progress(min(1.0, (i + chunk) / total))
        if writer:
            writer.close()
        prog.progress(1.0, text="Done.")
        return out_path

    if st.button("🔄 Batch score full file to Parquet (RAM-safe)"):
        if file_comments is None:
            st.warning("Upload a CSV/ZIP first.")
        else:
            st.info("Reading upload (this may take a moment)…")
            df_for_batch = load_upload(file_comments, merge_all=True)
            out_file = "commentsense_full.parquet"
            path = _batch_score_parquet(
                df_for_batch, out_file,
                tau=st.session_state.tau,
                spam_cap=st.session_state.spam_max,
                chunk=100_000,
            )
            if path:
                st.success(f"Saved: {path}")

# ===================== Data Load =====================
def _demo_df() -> pd.DataFrame:
    return pd.DataFrame({
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
        "like_count": [100, 80, 50],
        "share_count": [10, 8, 5],
        "save_count": [20, 12, 7],
    })

if file_comments:
    # If ZIP: merge all CSVs inside; keeps UX simple
    df_raw = load_upload(file_comments, merge_all=True)
    st.caption(f"Loaded **{len(df_raw):,}** rows from `{file_comments.name}`.")
else:
    df_raw = _demo_df()
    st.caption("Demo data loaded (upload a CSV/ZIP to analyze your own).")

# ===================== Helper (no caching of giant frames) =====================
def _score(df_raw: pd.DataFrame, tau: float, spam_cap: float) -> pd.DataFrame:
    # Keep interactive scoring RAM-friendly by capping rows
    df_in = df_raw.head(ROWS_FOR_SCORING).copy() if len(df_raw) > ROWS_FOR_SCORING else df_raw
    return process_dataframe(df_in, tau=tau, spam_cap=spam_cap)

@st.cache_data(show_spinner=False)
def _alvin_cached(df_raw: pd.DataFrame) -> dict:
    # Alvin track uses a smaller slice to keep graph & clustering snappy
    df_in = df_raw.head(ROW_CAP_FOR_ALVIN).copy()
    return process_alvin(
        df_in,
        fast_mode=FAST_MODE,
        use_zeroshot=USE_ZEROSHOT,
        use_persona_cluster=USE_PERSONA_CLUSTER,
        use_bertopic=USE_BERTOPIC,
        user_meta=None,
    )

# ===================== Scoring (sentiment/spam/quality) =====================
with st.spinner("Scoring comments…"):
    df = _score(
        df_raw,
        tau=st.session_state.tau,
        spam_cap=st.session_state.spam_max,
    )

if len(df_raw) > ROWS_FOR_SCORING:
    st.caption(f"⚠️ Loaded {len(df_raw):,} rows; UI shows first {ROWS_FOR_SCORING:,} scored rows to avoid OOM.")

# ===================== Core Components =====================
KPIs(df)
SentimentSpam(df)

st.subheader("Comments Explorer")

def _cols_exist(dataframe: pd.DataFrame, wanted: list[str]) -> list[str]:
    return [c for c in wanted if c in dataframe.columns]

explore_cols = _cols_exist(df, [
    "post_text", "comment", "sentiment", "sentiment_score",
    "spamness", "cqs", "is_quality", "likes", "like_count", "ts", "user_id", "post_id"
])
if not explore_cols:
    explore_cols = list(df.columns)[:10]

st.dataframe(df[explore_cols], width="stretch")

st.download_button(
    label="Download enriched comments CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="commentsense_enriched.csv",
    mime="text/csv",
)

# ===================== Alvin's Track: Categories / Personas / Influencers =====================
with st.spinner("Building categories/personas/influencers…"):
    alvin = _alvin_cached(df_raw)

cats_out        = alvin["cats_out"]
personas_df     = alvin["personas_df"]
influencers_df  = alvin["influencers_df"]
communities_df  = alvin.get("communities_df")

if len(df_raw) > ROW_CAP_FOR_ALVIN:
    st.caption(f"⚡ Processed first **{ROW_CAP_FOR_ALVIN:,}** rows for Categories/Personas/Influencers for speed.")

tab1, tab2, tab3 = st.tabs(["🧩 Categories & Topics", "🧑‍🤝‍🧑 Personas", "📣 Influencers"])

with tab1:
    CategoriesTopics(cats_out)

with tab2:
    Personas(personas_df)

with tab3:
    # Influencers component accepts (influencers_df, communities_df)
    Influencers(influencers_df, communities_df)
