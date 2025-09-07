# models/alvin_utils.py
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------- Device ----------------
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

# ---------------- Graph & communities ----------------
import networkx as nx
try:
    import community as community_louvain  # python-louvain
except Exception:
    community_louvain = None

# ---------------- Optional models ----------------
# transformers (zero-shot)
try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

# sentence-transformers + HDBSCAN (personas / BERTopic embeddings)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    import hdbscan
except Exception:
    hdbscan = None

# BERTopic (optional)
try:
    from bertopic import BERTopic
except Exception:
    BERTopic = None

from sklearn.feature_extraction.text import TfidfVectorizer

# ===================== Regex / Lexicons =====================
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
HANDLE_RE = re.compile(r"@[\w\._-]+", re.I)

CATEGORIES = ["skincare", "makeup", "fragrance", "haircare", "body care"]
CATEGORY_KW = {
    "skincare": ["serum","toner","cleanser","moisturizer","skin","retinol","hyaluronic","acne","spf","sunscreen"],
    "makeup":   ["lipstick","foundation","blush","mascara","eyeshadow","concealer","shade","palette","primer"],
    "fragrance":["perfume","fragrance","eau de","scent","parfum","toilette","cologne"],
    "haircare": ["shampoo","conditioner","hair","curl","frizz","keratin","scalp","split ends"],
    "body care":["body","lotion","hand cream","shower","soap","butter","scrub"],
}
SUBTOPIC_KW = {
    "skincare": {
        "hydration": ["hydrate","moisture","dewy","hyaluronic"],
        "anti-aging":["wrinkle","retinol","firming","fine lines"],
        "acne":      ["acne","pimple","blemish","salicylic"],
        "sunscreen": ["sunscreen","spf","sunblock","uva","uvb"],
    },
    "makeup": {
        "lipstick":   ["lipstick","lip","matte","gloss","tint","balm"],
        "foundation": ["foundation","coverage","shade","concealer","base"],
        "mascara":    ["mascara","lash","volume","curl"],
        "eyeshadow":  ["eyeshadow","palette","pigment"],
    },
    "fragrance": {
        "floral": ["rose","jasmine","floral","peony"],
        "woody":  ["sandalwood","cedar","woody","oud"],
        "citrus": ["citrus","lemon","bergamot","orange"],
        "sweet":  ["vanilla","gourmand","sweet"],
    },
    "haircare": {
        "anti-frizz":["frizz","smooth","sleek"],
        "repair":    ["repair","damage","split ends","keratin"],
        "volume":    ["volume","volumizing","thickening"],
    },
    "body care": {
        "lotion":["lotion","body cream","butter"],
        "wash":  ["body wash","shower gel","soap"],
        "scrub": ["scrub","exfoliate","exfoliation"],
    },
}

# ===================== Small Helpers =====================
def _guess_col(df: pd.DataFrame, opts: List[str]) -> str | None:
    # exact, then substring
    for opt in opts:
        for c in df.columns:
            if opt.lower() == c.lower():
                return c
    for opt in opts:
        for c in df.columns:
            if opt.lower() in c.lower():
                return c
    return None

def clean_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = URL_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_inline_mentions(s: str) -> list[str]:
    if not isinstance(s, str): return []
    return HANDLE_RE.findall(s)

# ===================== Normalization =====================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    text_col = _guess_col(out, ["text","textoriginal","comment","comment_text","message","body","content","review","caption"])
    out["text"] = out[text_col].astype(str) if text_col else ""

    cid = _guess_col(out, ["comment_id","commentid","id"])
    if cid: out["comment_id"] = out[cid].astype(str)
    pid = _guess_col(out, ["post_id","videoid","video_id","thread_id","tweet_id"])
    if pid: out["post_id"] = out[pid].astype(str)
    uid = _guess_col(out, ["user_id","authorid","author_id","user","author","username","handle","account"])
    out["user_id"] = out[uid].astype(str) if uid else out.get("user_id", out.index.map(lambda i: f"user_{i}")).astype(str)

    rid = _guess_col(out, ["reply_to_id","parent_id","in_reply_to","parentcommentid"])
    out["reply_to_id"] = out[rid].astype(str) if rid else np.nan

    ment = _guess_col(out, ["mentions","mentioned_users","tags","entities_mentions"])
    if ment:
        out["mentions"] = out[ment].astype(str)
    else:
        out["mentions"] = out["text"].str.findall(HANDLE_RE).map(lambda xs: ",".join(xs) if isinstance(xs, list) else "")

    lk = _guess_col(out, ["likes","like_count"])
    out["likes"] = pd.to_numeric(out[lk], errors="coerce").fillna(0).astype(int) if lk else 0

    # ---- SAFE TIMESTAMP PARSE ----
    ts = _guess_col(out, ["ts","timestamp","publishedat","created_at","date","time"])
    if ts:
        try:
            # cache=False uses less intermediate memory
            out["ts"] = pd.to_datetime(out[ts], errors="coerce", utc=True, cache=False)
        except Exception:
            # If RAM is tight, keep as string instead of failing
            out["ts"] = out[ts].astype(str)

    return out

# ===================== Category / Subtopic (heuristics) =====================
def assign_category(text: str) -> str:
    t = (text or "").lower()
    counts = {c: 0 for c in CATEGORIES}
    for c, kws in CATEGORY_KW.items():
        for w in kws:
            if w in t:
                counts[c] += 1
    cat = max(counts, key=counts.get)
    return cat if counts[cat] > 0 else "general"

def assign_subtopic(text: str, cat: str) -> str:
    t = (text or "").lower()
    subs = SUBTOPIC_KW.get(cat, {})
    best, score = "general", 0
    for name, kws in subs.items():
        s = sum(1 for w in kws if w in t)
        if s > score:
            best, score = name, s
    return best

def category_confidence(text: str, cat: str) -> float:
    t = (text or "").lower()
    kws = CATEGORY_KW.get(cat, [])
    hits = sum(1 for w in kws if w in t)
    denom = max(3, len(kws)//3 or 1)
    return min(1.0, hits/denom) if kws else 0.0

# ===================== Zero-shot categories (GPU-aware, batched, deduped) =====================
_ZS_PIPE = None  # cached transformers pipeline

def _zero_shot_categories(texts: list[str], labels: list[str], batch: int = 64) -> tuple[list[str], list[float]]:
    if hf_pipeline is None:
        raise ImportError("transformers not installed")
    global _ZS_PIPE
    if _ZS_PIPE is None:
        _ZS_PIPE = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=HF_DEVICE,                   # 0 for CUDA, -1 for CPU
            torch_dtype=HF_DTYPE,              # fp16 on GPU
        )
    # de-duplicate inputs to reduce compute
    uniq, map_idx, seen = [], [], {}
    for t in texts:
        k = t
        if k in seen:
            map_idx.append(seen[k])
        else:
            seen[k] = len(uniq)
            uniq.append(t)
            map_idx.append(seen[k])

    pred_u, conf_u = [], []
    for i in range(0, len(uniq), batch):
        chunk = uniq[i:i+batch]
        out = _ZS_PIPE(chunk, candidate_labels=labels, multi_label=False)
        if isinstance(out, dict): out = [out]
        for r in out:
            pred_u.append(r["labels"][0].lower())
            conf_u.append(float(r["scores"][0]))

    pred = [pred_u[i] for i in map_idx]
    conf = [conf_u[i] for i in map_idx]
    return pred, conf

# ===================== Influencer Graph =====================
def _norm_handle(h: str) -> str:
    h = str(h).strip()
    return h[1:] if h.startswith("@") else h

def _build_edges(df: pd.DataFrame) -> List[Tuple[str, str]]:
    edges: list[tuple[str, str]] = []

    # reply edges
    if "reply_to_id" in df.columns and "comment_id" in df.columns:
        cid_to_uid = dict(zip(df["comment_id"].astype(str), df["user_id"].astype(str)))
        rep = df.dropna(subset=["reply_to_id"])
        for _, r in rep.iterrows():
            src = str(r["user_id"])
            tgt = cid_to_uid.get(str(r["reply_to_id"]))
            if src and tgt and src != tgt:
                edges.append((src, tgt))

    # mention edges
    for _, r in df.iterrows():
        src = str(r.get("user_id"))
        if not src or src == "nan":
            continue
        ms = set()
        mcol = r.get("mentions")
        if isinstance(mcol, str) and mcol.strip():
            ms.update([_norm_handle(x) for x in mcol.split(",") if x])
        for x in extract_inline_mentions(r.get("text", "")):
            ms.add(_norm_handle(x))
        for m in ms:
            if m and m != src:
                edges.append((src, m))
    return edges

def _graph_metrics(edges: List[Tuple[str, str]], user_engagement: Dict[str, float]):
    if not edges:
        return {}, {}, {}

    G = nx.DiGraph()
    for a, b in edges:
        if G.has_edge(a, b):
            G[a][b]["weight"] += 1
        else:
            G.add_edge(a, b, weight=1)

    personalization = {n: max(0.1, float(user_engagement.get(n, 1.0))) for n in G.nodes()}
    pr = nx.pagerank(G, alpha=0.85, weight="weight", personalization=personalization)

    # betweenness (approximate when large)
    n = G.number_of_nodes()
    k = None
    if n > 600:
        k = min(200, max(20, int(0.05 * n)))
    btw = nx.betweenness_centrality(G, k=k, seed=42, weight="weight")

    # louvain on undirected projection
    UG = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        if UG.has_edge(u, v):
            UG[u][v]["weight"] += w
        else:
            UG.add_edge(u, v, weight=w)

    if community_louvain and UG.number_of_edges():
        part = community_louvain.best_partition(UG, weight="weight", random_state=42)
    else:
        part = {n: i for i, n in enumerate(UG.nodes())}
    return pr, btw, part

# ===================== Personas (Heuristic) =====================
def _heuristic_personas(df: pd.DataFrame, cats_out: pd.DataFrame) -> pd.DataFrame:
    cat_counts = cats_out.groupby(["user_id","category"]).size().reset_index(name="cnt")
    cat_counts["r"] = cat_counts.groupby("user_id")["cnt"].rank("dense", ascending=False)
    dom = cat_counts[cat_counts["r"] == 1][["user_id","category"]].rename(columns={"category":"dominant_category"})

    if "sentiment" in df.columns:
        sentiment_profile = df.groupby("user_id").agg({
            "sentiment": lambda x: x.mode().iloc[0] if len(x.mode()) else "neutral",
            "sentiment_score": "mean"
        }).reset_index().rename(columns={"sentiment":"dominant_sentiment","sentiment_score":"avg_sentiment_score"})
    else:
        sentiment_profile = pd.DataFrame({"user_id": dom["user_id"].unique(),
                                          "dominant_sentiment":"neutral",
                                          "avg_sentiment_score":0.5})

    if "likes" in df.columns:
        engagement_profile = df.groupby("user_id").agg({"likes":["mean","sum"]}).round(2)
        engagement_profile.columns = ["avg_likes","total_likes"]
        engagement_profile = engagement_profile.reset_index()
        counts = df.groupby("user_id").size().reset_index(name="comment_count")
        engagement_profile = engagement_profile.merge(counts, on="user_id", how="left")
        engagement_profile["engagement_tier"] = pd.cut(
            engagement_profile["avg_likes"], bins=[-1,1,10,float("inf")], labels=["Low","Medium","High"]
        )
    else:
        engagement_profile = df.groupby("user_id").size().reset_index(name="comment_count")
        engagement_profile["engagement_tier"] = "Medium"
        engagement_profile["avg_likes"] = 0
        engagement_profile["total_likes"] = 0

    base = (dom.merge(sentiment_profile, on="user_id", how="left")
              .merge(engagement_profile, on="user_id", how="left"))

    sub_counts = cats_out.groupby(["user_id","subtopic"]).size().reset_index(name="cnt")
    sub_counts["r"] = sub_counts.groupby("user_id")["cnt"].rank("dense", ascending=False)
    top_sub = sub_counts[sub_counts["r"] == 1][["user_id","subtopic"]].rename(columns={"subtopic":"top_subtopic"})
    base = base.merge(top_sub, on="user_id", how="left")

    def _label(row):
        cat = str(row.get("dominant_category","General")).title()
        sent = row.get("dominant_sentiment","neutral")
        tier = str(row.get("engagement_tier","Medium"))
        if sent == "positive" and tier == "High": return f"{cat} Advocate"
        if sent == "positive": return f"{cat} Enthusiast"
        if sent == "negative": return f"{cat} Critic"
        return f"{cat} Explorer"

    base["persona_label"] = base.apply(_label, axis=1)
    cols = ["user_id","persona_label","dominant_category","top_subtopic",
            "dominant_sentiment","engagement_tier","comment_count","avg_likes","total_likes"]
    keep = [c for c in cols if c in base.columns]
    return base[keep].sort_values("comment_count", ascending=False)

# ===================== SBERT + HDBSCAN Personas =====================
_SBERT = None  # cached SentenceTransformer

def _get_sbert(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _SBERT
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not installed")
    if _SBERT is None:
        _SBERT = SentenceTransformer(model_name, device=TORCH_DEVICE)
    return _SBERT

def _embed_docs(docs: list[str], batch: int = 256):
    model = _get_sbert()
    bs = 512 if _CUDA else batch
    return model.encode(docs, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)

def _build_user_docs(df: pd.DataFrame, text_col: str,
                     max_per_user: int = 120, max_chars: int = 20000,
                     top_users: int | None = 4000) -> tuple[list[str], list[str], pd.DataFrame]:
    stats = df.groupby("user_id", dropna=True).size().reset_index(name="comment_count")
    if top_users:
        keep = set(stats.sort_values("comment_count", ascending=False).head(top_users)["user_id"].astype(str))
        df = df[df["user_id"].astype(str).isin(keep)]
        stats = stats[stats["user_id"].astype(str).isin(keep)]

    uids, docs = [], []
    for uid, grp in df.groupby("user_id"):
        s = grp[text_col].astype(str).head(max_per_user)
        doc = " ".join(s.tolist())[:max_chars]
        if not doc.strip(): 
            continue
        uids.append(str(uid)); docs.append(doc)
    stats["user_id"] = stats["user_id"].astype(str)
    return uids, docs, stats

def _hdbscan_labels(X: np.ndarray, min_cluster_size: int = 20, min_samples: int | None = None):
    if hdbscan is None:
        raise ImportError("hdbscan not installed")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                metric="euclidean", cluster_selection_method="eom")
    return clusterer.fit_predict(X)

def _cluster_terms(uids: list[str], docs: list[str], labels: np.ndarray, top_k: int = 6) -> pd.DataFrame:
    if len(uids) == 0:
        return pd.DataFrame(columns=["persona_cluster","cluster_terms","cluster_size"])
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=8000)
    tfidf = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    rows = []
    for lab in sorted(set(labels)):
        if lab < 0: 
            continue
        idx = np.where(labels == lab)[0]
        if len(idx) == 0: 
            continue
        mean_tfidf = tfidf[idx].mean(axis=0).A1
        top_ids = mean_tfidf.argsort()[::-1][:top_k]
        terms = ", ".join(vocab[top_ids])
        rows.append({"persona_cluster": int(lab), "cluster_terms": terms, "cluster_size": int(len(idx))})
    return pd.DataFrame(rows)

def _cluster_examples(df: pd.DataFrame, labels_map: dict[str,int], text_col: str, k: int = 2) -> pd.DataFrame:
    if not labels_map or text_col not in df.columns:
        return pd.DataFrame(columns=["persona_cluster","cluster_examples"])
    tmp = df.copy()
    tmp["persona_cluster"] = tmp["user_id"].astype(str).map(labels_map)
    tmp = tmp[tmp["persona_cluster"].notna() & (tmp["persona_cluster"] >= 0)]
    if "cqs" not in tmp.columns: tmp["cqs"] = 0.0
    if "likes" not in tmp.columns: tmp["likes"] = 0
    rows = []
    for cid, g in tmp.groupby("persona_cluster"):
        gg = g.copy()
        gg["__len"] = gg[text_col].astype(str).str.len()
        gg = gg.sort_values(["likes","cqs","__len"], ascending=[False,False,False]).head(k)
        sample = " | ".join((gg[text_col].astype(str).str.slice(0,160) + "…").tolist())
        rows.append({"persona_cluster": int(cid), "cluster_examples": sample})
    return pd.DataFrame(rows)

def _cluster_personas(df: pd.DataFrame, cats_out: pd.DataFrame, text_col: str,
                      fast_mode: bool = True, top_users: int = 4000) -> pd.DataFrame:
    uids, docs, _ = _build_user_docs(df, text_col, top_users=top_users)
    if len(uids) < 10:
        return _heuristic_personas(df, cats_out)
    try:
        X = _embed_docs(docs, batch=128 if fast_mode else 256)
        labels = _hdbscan_labels(X, min_cluster_size=10 if fast_mode else 20)
    except Exception:
        return _heuristic_personas(df, cats_out)

    terms_df = _cluster_terms(uids, docs, labels, top_k=6)
    label_map = {u: int(l) for u, l in zip(uids, labels)}
    ex_df = _cluster_examples(df, label_map, text_col, k=2)

    base = _heuristic_personas(df, cats_out)
    base["persona_cluster"] = base["user_id"].astype(str).map(label_map).fillna(-1).astype(int)
    base = base.merge(terms_df, on="persona_cluster", how="left")
    base = base.merge(ex_df, on="persona_cluster", how="left")

    def _mk_name(row):
        if row.get("persona_cluster",-1) < 0 or not isinstance(row.get("cluster_terms"), str):
            return f"{str(row.get('dominant_category','General')).title()} Explorer"
        head = ", ".join(row["cluster_terms"].split(", ")[:2])
        return f"{str(row.get('dominant_category','General')).title()} — {head}"

    base["persona_label"] = base.apply(_mk_name, axis=1)
    cols = ["user_id","persona_label","persona_cluster","cluster_terms","cluster_examples",
            "dominant_category","top_subtopic","dominant_sentiment","engagement_tier",
            "comment_count","avg_likes","total_likes"]
    keep = [c for c in cols if c in base.columns]
    return base[keep].sort_values(["persona_cluster","comment_count"], ascending=[True,False])

# ===================== Community Health & Summary =====================
def _community_health_metrics(df: pd.DataFrame, influencers_df: pd.DataFrame) -> pd.DataFrame:
    uid_to_cid = dict(zip(influencers_df["user_id"], influencers_df["community_id"]))
    tmp = df.copy()
    tmp["community_id"] = tmp["user_id"].map(lambda u: uid_to_cid.get(str(u), None))
    tmp = tmp.dropna(subset=["community_id"])

    rows = []
    for cid, g in tmp.groupby("community_id"):
        metrics = {"community_id": str(cid)}
        metrics["total_comments"] = len(g)
        metrics["unique_users"] = g["user_id"].nunique()
        metrics["avg_comments_per_user"] = metrics["total_comments"] / max(1, metrics["unique_users"])
        if "sentiment" in g.columns:
            dist = g["sentiment"].value_counts(normalize=True)
            metrics["positive_ratio"] = dist.get("positive", 0)
            metrics["negative_ratio"] = dist.get("negative", 0)
            metrics["sentiment_health"] = metrics["positive_ratio"] - metrics["negative_ratio"]
        else:
            metrics["positive_ratio"] = 0.33; metrics["negative_ratio"] = 0.33; metrics["sentiment_health"] = 0.0
        if "likes" in g.columns:
            metrics["avg_likes"] = g["likes"].mean(); metrics["total_likes"] = g["likes"].sum()
        else:
            metrics["avg_likes"] = 0; metrics["total_likes"] = 0
        if "spamness" in g.columns:
            metrics["avg_spamness"] = g["spamness"].mean(); metrics["spam_health"] = 1 - metrics["avg_spamness"]
        else:
            metrics["avg_spamness"] = 0.1; metrics["spam_health"] = 0.9
        metrics["avg_quality"] = g["cqs"].mean() if "cqs" in g.columns else 0.5
        rows.append(metrics)
    return pd.DataFrame(rows).fillna(0)

def _summarize_communities(df: pd.DataFrame, influencers: pd.DataFrame) -> pd.DataFrame:
    base = influencers.copy()
    base["community_id"] = base["community_id"].astype(str)
    users_per = base.groupby("community_id")["user_id"].nunique().rename("users").reset_index()
    comm_comments = base.groupby("community_id")["comments_count"].sum().rename("total_comments").reset_index()
    summary = users_per.merge(comm_comments, on="community_id", how="left")

    def top_users_str(cid):
        block = base[base["community_id"] == cid].sort_values("rank_score", ascending=False).head(5)
        return ", ".join(block["user_id"].astype(str).tolist())
    summary["top_users"] = summary["community_id"].map(top_users_str)

    if "cqs" not in df.columns: df["cqs"] = 0.0
    if "likes" not in df.columns: df["likes"] = 0
    uid_to_cid = dict(zip(base["user_id"], base["community_id"]))
    tmp = df.copy()
    tmp["__cid"] = tmp["user_id"].map(lambda u: uid_to_cid.get(str(u), None))

    exemplars = []
    for cid, grp in tmp.dropna(subset=["__cid"]).groupby("__cid"):
        g = grp.copy()
        comment_col = "comment" if "comment" in g.columns else "text"
        if comment_col in g.columns:
            g["len"] = g[comment_col].astype(str).str.len()
            g = g.sort_values(["likes","cqs","len"], ascending=[False,False,False]).head(3)
            sample = " | ".join((g[comment_col].astype(str).str.slice(0,180) + "…").tolist())
        else:
            sample = ""
        exemplars.append({"community_id": str(cid), "sample_comments": sample})
    ex_df = pd.DataFrame(exemplars)
    return summary.merge(ex_df, on="community_id", how="left").fillna({"sample_comments": ""}) \
                  .sort_values(["users","total_comments"], ascending=False)

# ===================== Public API =====================
def process_alvin(
    df_raw: pd.DataFrame,
    fast_mode: bool = True,
    use_zeroshot: bool = False,
    use_persona_cluster: bool = False,
    use_bertopic: bool = False,
    user_meta: pd.DataFrame | None = None,
):
    """
    Returns dict with: df, cats_out, personas_df, influencers_df, communities_df
    """
    df = normalize_columns(df_raw)

    text_col = "text" if "text" in df.columns else None
    df["text_clean"] = df[text_col].map(clean_text) if text_col else ""

    # ---- Categories ----
    if use_zeroshot and text_col:
        try:
            cats, conf = _zero_shot_categories(df["text_clean"].tolist(), CATEGORIES)
            df["category"] = cats
            df["category_score"] = conf
        except Exception:
            df["category"] = df["text_clean"].map(assign_category)
            df["category_score"] = [category_confidence(t, c) for t, c in zip(df["text_clean"], df["category"])]
    else:
        df["category"] = df["text_clean"].map(assign_category)
        df["category_score"] = [category_confidence(t, c) for t, c in zip(df["text_clean"], df["category"])]

    # ---- Subtopics (heuristics) ----
    df["subtopic"] = [assign_subtopic(t, c) for t, c in zip(df["text_clean"], df["category"])]

    # ---- Optional BERTopic (uses same SBERT model if available) ----
    if use_bertopic and BERTopic is not None and text_col:
        try:
            embs = None
            if SentenceTransformer is not None:
                embs = _embed_docs(df["text_clean"].tolist(), batch=256)
            tm = BERTopic(verbose=False, calculate_probabilities=False, embedding_model=_SBERT if _SBERT else None)
            topics, _ = tm.fit_transform(df["text_clean"].tolist(), embeddings=embs)
            df["topic"] = topics
        except Exception:
            df["topic"] = -1

    # ---- cats_out payload ----
    base_cols = ["comment_id","post_id","user_id"]
    available_cols = [c for c in base_cols if c in df.columns]
    cats_out = df[available_cols + ["category","category_score","subtopic"]].copy()
    if "topic" in df.columns:
        cats_out["topic"] = df["topic"]
    cats_out["text"] = df[text_col] if text_col else ""

    # ---- Personas ----
    if use_persona_cluster and text_col and SentenceTransformer is not None and hdbscan is not None:
        personas_df = _cluster_personas(df, cats_out, text_col=text_col, fast_mode=fast_mode, top_users=4000)
    else:
        personas_df = _heuristic_personas(df, cats_out)

    # ---- Influencers / communities ----
    edges = _build_edges(df)
    user_engagement = df.groupby("user_id")["likes"].sum().to_dict() if "likes" in df.columns \
                      else df.groupby("user_id").size().to_dict()
    pr, btw, part = _graph_metrics(edges, user_engagement)

    mentions_made = defaultdict(int); mentions_recv = defaultdict(int)
    for a, b in edges:
        mentions_made[a] += 1; mentions_recv[b] += 1

    all_nodes = set(df["user_id"].astype(str).dropna().tolist())
    for a, b in edges:
        all_nodes.add(a); all_nodes.add(b)

    comments_per_user = df.groupby("user_id").size().to_dict()

    influencers_df = pd.DataFrame({"user_id": sorted(list(all_nodes))})
    influencers_df["mentions_made"] = influencers_df["user_id"].map(lambda u: mentions_made.get(u, 0))
    influencers_df["mentions_received"] = influencers_df["user_id"].map(lambda u: mentions_recv.get(u, 0))
    influencers_df["comments_count"] = influencers_df["user_id"].map(lambda u: comments_per_user.get(u, 0))
    influencers_df["pagerank"] = influencers_df["user_id"].map(lambda u: pr.get(u, 0.0))
    influencers_df["betweenness"] = influencers_df["user_id"].map(lambda u: btw.get(u, 0.0))
    influencers_df["community_id"] = influencers_df["user_id"].map(lambda u: part.get(u, str(u)))
    influencers_df["engagement_score"] = influencers_df["user_id"].map(lambda u: user_engagement.get(u, 0))

    # attach optional meta (followers / verified) for tiers
    if isinstance(user_meta, pd.DataFrame) and "user_id" in user_meta.columns:
        meta = user_meta.copy()
        meta["user_id"] = meta["user_id"].astype(str)
        influencers_df = influencers_df.merge(meta, on="user_id", how="left")
        if "followers" in influencers_df.columns:
            def _tier(x):
                try:
                    x = int(x)
                except Exception:
                    return "n/a"
                if x >= 1_000_000: return "mega"
                if x >=   100_000: return "macro"
                if x >=    10_000: return "micro"
                return "nano"
            influencers_df["tier"] = influencers_df["followers"].map(_tier)

    influencers_df["rank_score"] = (
        influencers_df["pagerank"].rank(pct=True) * 0.4 +
        influencers_df["betweenness"].rank(pct=True) * 0.2 +
        influencers_df["mentions_received"].rank(pct=True) * 0.2 +
        influencers_df["engagement_score"].rank(pct=True) * 0.2
    ).fillna(0.0)
    influencers_df = influencers_df.sort_values("rank_score", ascending=False)

    communities_df = _summarize_communities(df, influencers_df)
    community_health_df = _community_health_metrics(df, influencers_df)
    communities_enhanced = communities_df.merge(community_health_df, on="community_id", how="left").fillna(0)

    return {
        "df": df,
        "cats_out": cats_out,
        "personas_df": personas_df,
        "influencers_df": influencers_df,
        "communities_df": communities_enhanced,
    }
