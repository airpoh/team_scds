# data_loader.py
from __future__ import annotations

import io
import re
import zipfile
from typing import IO, Iterable, Callable, List

import numpy as np
import pandas as pd

# Your models
from models.sentiment_model import predict_sentiment
from models.spam_model import spam_scores, civility_scores

# ----------------------- constants -----------------------
MENTION_RE = re.compile(r"@[\w\._-]+", re.I)


# ===================== CSV / ZIP utils =====================
def _read_csv_robust(src, **kwargs) -> pd.DataFrame:
    """
    Robust CSV reader that works across pandas versions and messy files.
    Tries:
      1) on_bad_lines="skip", encoding_errors="ignore" (pandas >= 2.0)
      2) on_bad_lines="skip" (pandas >= 1.3)
      3) error_bad_lines=False, warn_bad_lines=False (older pandas)
    """
    try:
        return pd.read_csv(src, on_bad_lines="skip", encoding_errors="ignore", **kwargs)
    except TypeError:
        try:
            return pd.read_csv(src, on_bad_lines="skip", **kwargs)
        except TypeError:
            return pd.read_csv(src, error_bad_lines=False, warn_bad_lines=False, **kwargs)  # type: ignore[arg-type]


def _read_bytes(uploaded_file) -> bytes:
    # Streamlit's UploadedFile has .getvalue(); fall back to .read()
    return uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()


def list_zip_csvs(uploaded_file) -> list[str]:
    """List CSV members inside a .zip upload."""
    data = _read_bytes(uploaded_file)
    bio = io.BytesIO(data)
    if not zipfile.is_zipfile(bio):
        return []
    with zipfile.ZipFile(bio) as z:
        return [n for n in z.namelist() if n.lower().endswith(".csv") and not n.endswith("/")]


def load_upload(uploaded_file, zip_member: str | None = None, merge_all: bool = False) -> pd.DataFrame:
    """
    Load either a regular CSV or a CSV from within a .zip upload.
    If merge_all=True, concatenates all CSVs in the zip and adds `source_file`.
    """
    data = _read_bytes(uploaded_file)
    bio = io.BytesIO(data)

    if zipfile.is_zipfile(bio):
        with zipfile.ZipFile(bio) as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.endswith("/")]
            if not csv_names:
                raise ValueError("No CSV files found inside the zip.")
            if merge_all:
                frames = []
                for name in csv_names:
                    with z.open(name) as f:
                        df_part = _read_csv_robust(f)
                        df_part["source_file"] = name
                        frames.append(df_part)
                df = pd.concat(frames, ignore_index=True)
            else:
                member = zip_member or csv_names[0]
                if member not in csv_names:
                    raise ValueError(f"CSV '{member}' not found in zip. Found: {csv_names}")
                with z.open(member) as f:
                    df = _read_csv_robust(f)
    else:
        # regular CSV
        df = _read_csv_robust(io.BytesIO(data))

    # normalize column names like the rest of the pipeline expects
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_csv(file: IO) -> pd.DataFrame:
    """Read a plain CSV and lowercase columns (keeps flexible dtypes)."""
    df = _read_csv_robust(file, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


# ===================== helpers =====================
def _guess_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column by exact lower-case match first, then substring fallback."""
    cols = list(df.columns)
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for cand in candidates:
        for c in cols:
            if cand in str(c).lower():
                return c
    return None


def _batched(seq: Iterable, n: int) -> Iterable[list]:
    """Yield successive lists of size n from seq."""
    seq_list = list(seq)
    for i in range(0, len(seq_list), n):
        yield seq_list[i : i + n]


def _predict_safely(
    fn: Callable[[List[str]], List],
    texts: List[str],
    batch_try: tuple[int, ...] = (4096, 2048, 1024, 512, 256, 128),
):
    """
    Call a list->list predictor in descending batch sizes to avoid OOM.
    Retries on MemoryError or typical OOM messages; re-raises other errors.
    """
    last_err = None
    for bs in batch_try:
        try:
            out = []
            for i in range(0, len(texts), bs):
                out.extend(fn(texts[i : i + bs]))
            if len(out) != len(texts):
                raise RuntimeError("Prediction output length mismatch.")
            return out
        except Exception as e:
            msg = str(e).lower()
            if isinstance(e, MemoryError) or ("out of memory" in msg) or ("cuda" in msg and "memory" in msg):
                last_err = e
                continue  # try smaller batch
            raise  # not a memory issue
    raise last_err if last_err else RuntimeError("Prediction failed; all batch sizes exhausted.")


def _coerce_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Force numeric counters to ints (handles stringy numbers)."""
    for c in ("like_count", "likes", "share_count", "save_count"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


# ===================== schema normalizer =====================
def _ensure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw schemas to a common set of columns used downstream.

    YouTube export (lower-cased by loader) supported out-of-the-box:
      textoriginal -> comment
      commentid    -> comment_id
      videoid      -> post_id
      authorid     -> user_id
      parentcommentid -> reply_to_id (optional)
      likecount    -> likes (and like_count)
      publishedat  -> ts (UTC datetime)
    """
    out = df.copy()
    cols = set(out.columns)

    # ---- YouTube-style schema ----
    if {"textoriginal", "commentid", "videoid", "authorid"}.issubset(cols):
        out["comment"] = out["textoriginal"].astype(str).fillna("")
        out["comment_id"] = out["commentid"].astype(str)
        out["post_id"] = out["videoid"].astype(str)
        out["user_id"] = out["authorid"].astype(str)

        out["reply_to_id"] = out["parentcommentid"].astype(str) if "parentcommentid" in cols else np.nan

        # likes + alias like_count
        out["likes"] = pd.to_numeric(out.get("likecount", 0), errors="coerce").fillna(0).astype(int)
        out["like_count"] = out["likes"]

        # published timestamp
        out["ts"] = pd.to_datetime(out.get("publishedat", pd.NaT), errors="coerce", utc=True)

        # derive mentions from text
        out["mentions"] = (
            out["comment"].str.findall(MENTION_RE).map(lambda xs: ",".join(xs) if isinstance(xs, list) else "")
        )

        # ensure other counters exist
        for c in ("share_count", "save_count"):
            if c not in out.columns:
                out[c] = 0

        # ensure optional post_text exists for UI consistency
        if "post_text" not in out.columns:
            out["post_text"] = ""

        out = _coerce_counts(out)
        return out

    # ---- Generic fallback ----
    text_col = _guess_col(out, ["comment", "comment_text", "text", "message", "body", "content", "review", "caption"])
    if text_col is None:
        raise KeyError(
            "No text column found. Include 'textOriginal' (YouTube) or one of: "
            "'comment', 'text', 'message', 'body', 'content', 'review', 'caption'."
        )
    out["comment"] = out[text_col].astype(str).fillna("")

    post_text_col = _guess_col(out, ["post_text", "title", "caption"])
    out["post_text"] = out[post_text_col].astype(str).fillna("") if post_text_col else ""

    user_col = _guess_col(out, ["user_id", "user", "author", "username", "handle", "account", "authorid"])
    out["user_id"] = out[user_col].astype(str).fillna("") if user_col else out.index.map(lambda i: f"user_{i}")

    cid_col = _guess_col(out, ["comment_id", "commentid", "id"])
    if cid_col:
        out["comment_id"] = out[cid_col].astype(str)

    pid_col = _guess_col(out, ["post_id", "videoid", "thread_id", "tweet_id"])
    if pid_col:
        out["post_id"] = out[pid_col].astype(str)

    rid_col = _guess_col(out, ["reply_to_id", "parent_id", "in_reply_to", "parentcommentid"])
    out["reply_to_id"] = out[rid_col].astype(str) if rid_col else np.nan

    ts_col = _guess_col(out, ["ts", "timestamp", "created_at", "date", "time", "publishedat"])
    if ts_col:
        out["ts"] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)

    out["mentions"] = (
        out["comment"].str.findall(MENTION_RE).map(lambda xs: ",".join(xs) if isinstance(xs, list) else "")
    )

    # counters
    if "like_count" not in out.columns:
        if "likes" in out.columns:
            out["like_count"] = pd.to_numeric(out["likes"], errors="coerce").fillna(0).astype(int)
        else:
            out["like_count"] = 0
    for c in ("share_count", "save_count"):
        if c not in out.columns:
            out[c] = 0

    out = _coerce_counts(out)
    return out


# data_loader.py  — drop-in replacement for process_dataframe(...)
from concurrent.futures import ThreadPoolExecutor

def _dedupe_list(seq: List[str]) -> tuple[list[str], list[int]]:
    """Return (unique_items, index_map_to_unique)."""
    seen = {}
    uniq = []
    idx_map = []
    for s in seq:
        key = s
        if key in seen:
            idx_map.append(seen[key])
        else:
            seen[key] = len(uniq)
            uniq.append(s)
            idx_map.append(seen[key])
    return uniq, idx_map

def process_dataframe(
    df_raw: pd.DataFrame,
    tau: float = 0.6,
    spam_cap: float = 1.0,
    row_cap: int | None = 200_000,
) -> pd.DataFrame:
    """
    Run models + heuristics, then compute Comment Quality Score (CQS).
    Identical behavior, but faster via dedup + parallel scoring.
    """
    df = _ensure(df_raw)

    # ---- Safety cap for huge files (prevents OOM) ----
    if row_cap is not None and len(df) > row_cap:
        df = df.head(row_cap).copy()

    df["comment"] = df["comment"].astype(str).fillna("")
    texts = df["comment"].tolist()

    # ---- De-duplicate identical comments to avoid repeat inference ----
    uniq_texts, idx_map = _dedupe_list(texts)

    # ---- Run sentiment & spam in parallel (PyTorch releases GIL; CPU gains) ----
    def _run_sent():
        out = _predict_safely(predict_sentiment, uniq_texts, batch_try=(8192, 4096, 2048, 1024, 512))
        labels = [o["label"] for o in out]
        scores = [o["score"] for o in out]
        # map back
        return [labels[i] for i in idx_map], [scores[i] for i in idx_map]

    def _run_spam():
        spam_u = _predict_safely(spam_scores, uniq_texts, batch_try=(8192, 4096, 2048, 1024, 512))
        civ_u  = _predict_safely(civility_scores, uniq_texts, batch_try=(8192, 4096, 2048, 1024, 512))
        return [spam_u[i] for i in idx_map], [civ_u[i] for i in idx_map]

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(_run_sent)
        f2 = ex.submit(_run_spam)
        (sent_labels, sent_scores) = f1.result()
        (spam_all, civ_all) = f2.result()

    df["sentiment"] = sent_labels
    df["sentiment_score"] = sent_scores
    df["spamness"] = spam_all
    df["civility"] = civ_all

    # --- Heuristics (unchanged) ---
    df["constructiveness"] = df["comment"].str.contains(
        r"should|recommend|try|because|improve|better|i used|i tried|after|since",
        case=False, regex=True
    ).astype(float) * 0.7
    df["specificity"] = (df["comment"].str.len() / 250.0).clip(0, 1)

    # --- Composite score (unchanged) ---
    w = dict(constructiveness=0.30, specificity=0.25, civility=0.20, spamness=0.25)
    df["cqs"] = (
        w["constructiveness"] * df["constructiveness"]
        + w["specificity"]      * df["specificity"]
        + w["civility"]         * df["civility"]
        - w["spamness"]         * df["spamness"]
    ).clip(0, 1)

    # --- Filters / flags (unchanged) ---
    df = df[df["spamness"] <= spam_cap].copy()
    df["is_quality"] = df["cqs"] >= tau
    return df.reset_index(drop=True)
