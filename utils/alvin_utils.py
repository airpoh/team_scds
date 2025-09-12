"""
Utility functions for CommentSense pipeline data processing

This module provides shared utility functions for:
- Column normalization and data cleaning
- Text preprocessing and mention extraction
- Device detection and configuration helpers
"""

from __future__ import annotations

import re
from typing import List
import pandas as pd
import numpy as np

# ---------------- Device Detection ----------------
try:
    import torch
    _CUDA = torch.cuda.is_available()
    TORCH_DEVICE = "cuda" if _CUDA else "cpu"
    HF_DEVICE = 0 if _CUDA else -1          # for transformers pipeline
    HF_DTYPE = torch.float16 if _CUDA else None
except Exception:  # torch not installed
    torch = None
    _CUDA = False
    TORCH_DEVICE = "cpu"
    HF_DEVICE = -1
    HF_DTYPE = None

# ---------------- Text Processing Patterns ----------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
HANDLE_RE = re.compile(r"@[\w\._-]+", re.I)

# ===================== Column Guessing Utilities =====================
def _guess_col(df: pd.DataFrame, opts: List[str]) -> str | None:
    """
    Guess which column name matches from a list of options
    
    Args:
        df: DataFrame to search
        opts: List of column name options to try
    
    Returns:
        Matching column name or None if not found
    """
    # Try exact matches first
    for opt in opts:
        for col in df.columns:
            if opt.lower() == col.lower():
                return col
    
    # Try substring matches
    for opt in opts:
        for col in df.columns:
            if opt.lower() in col.lower():
                return col
    
    return None

# ===================== Text Cleaning =====================
def clean_text(s: str) -> str:
    """
    Clean and normalize text content
    
    Args:
        s: Input text string
    
    Returns:
        Cleaned text string
    """
    s = "" if pd.isna(s) else str(s)
    s = URL_RE.sub("", s)  # Remove URLs
    s = re.sub(r"\s+", " ", s)  # Normalize whitespace
    return s.strip()

def extract_inline_mentions(s: str) -> List[str]:
    """
    Extract @mentions from text content
    
    Args:
        s: Input text string
    
    Returns:
        List of mentioned handles (without @)
    """
    if not isinstance(s, str):
        return []
    return [handle[1:] if handle.startswith("@") else handle 
            for handle in HANDLE_RE.findall(s)]

# ===================== Column Normalization =====================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column names and content to standard format
    
    This function attempts to map various column naming conventions
    to a standardized format expected by the analysis modules.
    
    Args:
        df: Input DataFrame with raw column names
    
    Returns:
        DataFrame with normalized columns:
        - text: Main text content
        - comment_id: Unique comment identifier
        - post_id: Parent post identifier
        - user_id: User identifier
        - reply_to_id: Parent comment for replies
        - mentions: Comma-separated mentioned users
        - likes: Like/reaction count
        - ts: Timestamp
    """
    out = df.copy()

    # Text content column
    text_col = _guess_col(out, [
        "text", "textoriginal", "comment", "comment_text", 
        "message", "body", "content", "review", "caption"
    ])
    out["text"] = out[text_col].astype(str) if text_col else ""

    # Comment ID
    cid = _guess_col(out, ["comment_id", "commentid", "id"])
    if cid:
        out["comment_id"] = out[cid].astype(str)

    # Post ID  
    pid = _guess_col(out, [
        "post_id", "videoid", "video_id", "thread_id", "tweet_id"
    ])
    if pid:
        out["post_id"] = out[pid].astype(str)

    # User ID
    uid = _guess_col(out, [
        "user_id", "authorid", "author_id", "user", 
        "author", "username", "handle", "account"
    ])
    if uid:
        out["user_id"] = out[uid].astype(str)
    else:
        # Generate user IDs from index if not available
        out["user_id"] = out.index.map(lambda i: f"user_{i}").astype(str)

    # Reply-to relationship
    rid = _guess_col(out, [
        "reply_to_id", "parent_id", "in_reply_to", "parentcommentid"
    ])
    out["reply_to_id"] = out[rid].astype(str) if rid else np.nan

    # Mentions
    ment = _guess_col(out, [
        "mentions", "mentioned_users", "tags", "entities_mentions"
    ])
    if ment:
        out["mentions"] = out[ment].astype(str)
    else:
        # Extract mentions from text content
        out["mentions"] = (out["text"]
                          .str.findall(HANDLE_RE)
                          .map(lambda xs: ",".join(xs) if isinstance(xs, list) else ""))

    # Likes/reactions
    lk = _guess_col(out, ["likes", "like_count"])
    out["likes"] = (pd.to_numeric(out[lk], errors="coerce").fillna(0).astype(int) 
                   if lk else 0)

    # Timestamp parsing (with error handling for memory efficiency)
    ts = _guess_col(out, [
        "ts", "timestamp", "publishedat", "created_at", "date", "time"
    ])
    if ts:
        try:
            # cache=False uses less intermediate memory
            out["ts"] = pd.to_datetime(out[ts], errors="coerce", utc=True, cache=False)
        except Exception:
            # If RAM is tight, keep as string instead of failing
            out["ts"] = out[ts].astype(str)

    return out

# ===================== Device Information =====================
def get_device_info() -> dict:
    """
    Get information about available computational devices
    
    Returns:
        Dictionary with device configuration
    """
    return {
        "cuda_available": _CUDA,
        "torch_device": TORCH_DEVICE,
        "hf_device": HF_DEVICE,
        "hf_dtype": str(HF_DTYPE) if HF_DTYPE else None,
        "torch_available": torch is not None
    }

# ===================== Legacy Compatibility =====================
def process_alvin(df_raw: pd.DataFrame, **kwargs) -> dict:
    """
    Legacy compatibility function that redirects to modular analysis
    
    This function is kept for backward compatibility but now delegates
    to the new modular analysis system.
    
    Args:
        df_raw: Input DataFrame
        **kwargs: Analysis options
    
    Returns:
        Dictionary with analysis results
    """
    import warnings
    warnings.warn(
        "process_alvin is deprecated. Use the new modular analysis system in /modules/",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import the new modular components
    try:
        from modules.network_analysis import NetworkAnalyzer
        analyzer = NetworkAnalyzer()
        return analyzer.analyze_network(df_raw, **kwargs)
    except ImportError:
        raise ImportError(
            "New modular analysis system not available. "
            "Please use modules in /modules/ directory directly."
        )