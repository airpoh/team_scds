"""
Voice-of-Customer Personas Module for CommentSense Pipeline

This module provides:
- User persona clustering using Sentence Transformers + HDBSCAN
- Heuristic persona assignment based on behavior patterns
- User document aggregation and embedding generation
- TF-IDF cluster term extraction
- Domain-specific persona labeling for beauty/cosmetics
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Import configuration
try:
    import json
    import os
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'alvin_models_config.json')
    with open(config_path, 'r') as f:
        CONFIG = json.load(f)
except Exception:
    CONFIG = {}

logger = logging.getLogger(__name__)

# Disable meta device for HuggingFace transformers to prevent meta tensor issues
os.environ['PYTORCH_DISABLE_META_DEVICE'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# ---------------- Device Setup ----------------
try:
    import torch
    import platform
    
    # For Apple Silicon, force CPU only (PyTorch not fully compatible with macOS)
    if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
        TORCH_DEVICE = "cpu"
        logger.info("Apple Silicon detected - using CPU only (PyTorch macOS compatibility)")
    else:
        # For Windows/Linux users - enable CUDA if available
        _CUDA = torch.cuda.is_available()
        TORCH_DEVICE = "cuda" if _CUDA else "cpu"
        if _CUDA:
            logger.info("CUDA available, using GPU acceleration")
        else:
            logger.info("No GPU available, using CPU")
except Exception:
    torch = None
    TORCH_DEVICE = "cpu"

# ---------------- Optional Dependencies ----------------
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import hdbscan
except Exception:
    hdbscan = None

from sklearn.feature_extraction.text import TfidfVectorizer

class PersonaClustering:
    """
    Voice-of-Customer Personas using Sentence Transformers + HDBSCAN clustering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the persona clustering module"""
        
        self.config = config or CONFIG
        self._sbert_model = None  # Cached sentence transformer
        
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not available, will use heuristic personas only")
        if hdbscan is None:
            logger.warning("hdbscan not available, will use heuristic personas only")
            
        logger.info("Persona Clustering module initialized")
    
    def _get_sentence_transformer(self):
        """Get or create sentence transformer model (shared or individual)"""
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed")
        
        if self._sbert_model is None:
            try:
                # Try to use shared model first
                import sys
                from pathlib import Path
                
                # Add parent directory to path to import shared manager
                parent_dir = Path(__file__).parent.parent
                if str(parent_dir) not in sys.path:
                    sys.path.append(str(parent_dir))
                
                from commentsense_pipeline import get_shared_model_manager
                
                shared_manager = get_shared_model_manager()
                model_config = self.config.get("models", {}).get("sentence_transformer", {})
                model_name = model_config.get("primary", "sentence-transformers/all-MiniLM-L6-v2")
                
                self._sbert_model = shared_manager.get_shared_model("sentence_transformer", model_name)
                
                if self._sbert_model is not None:
                    logger.info(f"Using shared sentence transformer: {model_name}")
                    return self._sbert_model
                else:
                    logger.warning("Failed to get shared sentence transformer, loading individual model")
                    
            except Exception as e:
                logger.warning(f"Failed to use shared sentence transformer: {e}, loading individual model")
            
            # Fallback to individual model loading
            model_config = self.config.get("models", {}).get("sentence_transformer", {})
            model_name = model_config.get("primary", "sentence-transformers/all-MiniLM-L6-v2")
            
            try:
                # Try loading with specified device
                self._sbert_model = SentenceTransformer(model_name, device=TORCH_DEVICE)
                logger.info(f"Loaded individual sentence transformer: {model_name} on {TORCH_DEVICE}")
            except Exception as e:
                if "meta tensor" in str(e).lower() or "cannot copy out of meta tensor" in str(e).lower():
                    logger.warning(f"Meta tensor error with {TORCH_DEVICE}, forcing CPU: {e}")
                    # Force CPU loading for meta tensor issues
                    self._sbert_model = SentenceTransformer(model_name, device="cpu")
                    logger.info(f"Loaded individual sentence transformer: {model_name} on CPU (fallback)")
                else:
                    raise e
        
        return self._sbert_model
    
    def build_user_documents(self, 
                           df: pd.DataFrame,
                           text_column: str = "text",
                           max_per_user: int = 120,
                           max_chars: int = 20000,
                           top_users: Optional[int] = 4000) -> Tuple[List[str], List[str], pd.DataFrame]:
        """
        Aggregate user texts into documents for clustering
        
        Args:
            df: DataFrame with user comments
            text_column: Column containing text
            max_per_user: Maximum comments per user
            max_chars: Maximum characters per user document
            top_users: Only use top N most active users (None for all)
        
        Returns:
            Tuple of (user_ids, user_documents, user_stats)
        """
        if "user_id" not in df.columns:
            raise ValueError("DataFrame must have 'user_id' column")
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found")
        
        # Get user activity stats
        stats = df.groupby("user_id", dropna=True).size().reset_index(name="comment_count")
        
        # Filter to top users if specified
        if top_users:
            keep_users = set(stats.sort_values("comment_count", ascending=False)
                           .head(top_users)["user_id"].astype(str))
            df_filtered = df[df["user_id"].astype(str).isin(keep_users)]
            stats = stats[stats["user_id"].astype(str).isin(keep_users)]
        else:
            df_filtered = df
        
        # Aggregate user texts
        user_ids, user_docs = [], []
        
        for user_id, group in df_filtered.groupby("user_id"):
            texts = group[text_column].astype(str).head(max_per_user).tolist()
            user_doc = " ".join(texts)[:max_chars]
            
            if user_doc.strip():
                user_ids.append(str(user_id))
                user_docs.append(user_doc)
        
        stats["user_id"] = stats["user_id"].astype(str)
        return user_ids, user_docs, stats
    
    def embed_documents(self, documents: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generate embeddings for user documents
        
        Args:
            documents: List of user documents
            batch_size: Batch size for encoding
        
        Returns:
            Normalized embeddings matrix
        """
        model = self._get_sentence_transformer()
        
        if batch_size is None:
            batch_size = 512 if _CUDA else 256
        
        embeddings = model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def cluster_embeddings(self, 
                          embeddings: np.ndarray,
                          min_cluster_size: int = 20,
                          min_samples: Optional[int] = None) -> np.ndarray:
        """
        Cluster embeddings using HDBSCAN
        
        Args:
            embeddings: Normalized embeddings matrix
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples (None for auto)
        
        Returns:
            Cluster labels (-1 for noise)
        """
        if hdbscan is None:
            raise ImportError("hdbscan not installed")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        
        labels = clusterer.fit_predict(embeddings)
        return labels
    
    def extract_cluster_terms(self, 
                             user_ids: List[str],
                             user_docs: List[str],
                             cluster_labels: np.ndarray,
                             top_k: int = 6) -> pd.DataFrame:
        """
        Extract representative terms for each cluster using TF-IDF
        
        Args:
            user_ids: List of user IDs
            user_docs: List of user documents
            cluster_labels: Cluster assignments
            top_k: Number of top terms per cluster
        
        Returns:
            DataFrame with cluster terms
        """
        if len(user_ids) == 0:
            return pd.DataFrame(columns=["persona_cluster", "cluster_terms", "cluster_size"])
        
        # Fit TF-IDF
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=8000
        )
        tfidf_matrix = vectorizer.fit_transform(user_docs)
        vocabulary = np.array(vectorizer.get_feature_names_out())
        
        # Extract terms for each cluster
        cluster_info = []
        
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id < 0:  # Skip noise cluster
                continue
                
            cluster_mask = cluster_labels == cluster_id
            if not cluster_mask.any():
                continue
            
            # Calculate mean TF-IDF for cluster
            cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0).A1
            top_indices = cluster_tfidf.argsort()[::-1][:top_k]
            top_terms = ", ".join(vocabulary[top_indices])
            
            cluster_info.append({
                "persona_cluster": int(cluster_id),
                "cluster_terms": top_terms,
                "cluster_size": int(cluster_mask.sum())
            })
        
        return pd.DataFrame(cluster_info)
    
    def extract_cluster_examples(self,
                                df: pd.DataFrame,
                                user_cluster_map: Dict[str, int],
                                text_column: str = "text",
                                examples_per_cluster: int = 2) -> pd.DataFrame:
        """
        Extract example comments for each cluster
        
        Args:
            df: Original DataFrame with comments
            user_cluster_map: Mapping from user_id to cluster_id
            text_column: Column containing comment text
            examples_per_cluster: Number of examples per cluster
        
        Returns:
            DataFrame with cluster examples
        """
        if not user_cluster_map or text_column not in df.columns:
            return pd.DataFrame(columns=["persona_cluster", "cluster_examples"])
        
        # Add cluster assignments to DataFrame
        temp_df = df.copy()
        temp_df["persona_cluster"] = temp_df["user_id"].astype(str).map(user_cluster_map)
        temp_df = temp_df[temp_df["persona_cluster"].notna() & (temp_df["persona_cluster"] >= 0)]
        
        # Add default quality metrics if not present
        if "cqs" not in temp_df.columns:
            temp_df["cqs"] = 0.0
        if "likes" not in temp_df.columns:
            temp_df["likes"] = 0
        
        # Extract top examples for each cluster
        cluster_examples = []
        
        for cluster_id, group in temp_df.groupby("persona_cluster"):
            # Sort by engagement and quality metrics
            group_scored = group.copy()
            group_scored["text_length"] = group_scored[text_column].astype(str).str.len()
            
            top_examples = (group_scored
                          .sort_values(["likes", "cqs", "text_length"], ascending=[False, False, False])
                          .head(examples_per_cluster))
            
            # Create sample text (truncated)
            example_texts = (top_examples[text_column]
                           .astype(str)
                           .str.slice(0, 160)
                           .map(lambda x: x + "…"))
            
            sample_text = " | ".join(example_texts.tolist())
            
            cluster_examples.append({
                "persona_cluster": int(cluster_id),
                "cluster_examples": sample_text
            })
        
        return pd.DataFrame(cluster_examples)
    
    def create_heuristic_personas(self, df: pd.DataFrame, cats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create personas using heuristic rules (fallback method)
        
        Args:
            df: Original DataFrame with user data
            cats_df: DataFrame with category classifications
        
        Returns:
            DataFrame with persona assignments
        """
        if "user_id" not in cats_df.columns:
            raise ValueError("cats_df must have 'user_id' column")
        
        # Dominant category per user
        category_counts = cats_df.groupby(["user_id", "category"]).size().reset_index(name="count")
        category_counts["rank"] = category_counts.groupby("user_id")["count"].rank("dense", ascending=False)
        dominant_categories = (category_counts[category_counts["rank"] == 1]
                             [["user_id", "category"]]
                             .rename(columns={"category": "dominant_category"}))
        
        # Sentiment profile (if available)
        if "sentiment" in df.columns:
            sentiment_profile = df.groupby("user_id").agg({
                "sentiment": lambda x: x.mode().iloc[0] if len(x.mode()) else "neutral",
                "sentiment_score": "mean"
            }).reset_index().rename(columns={
                "sentiment": "dominant_sentiment",
                "sentiment_score": "avg_sentiment_score"
            })
        else:
            users = dominant_categories["user_id"].unique()
            sentiment_profile = pd.DataFrame({
                "user_id": users,
                "dominant_sentiment": "neutral",
                "avg_sentiment_score": 0.5
            })
        
        # Engagement profile
        if "likes" in df.columns:
            engagement_stats = df.groupby("user_id").agg({
                "likes": ["mean", "sum"]
            }).round(2)
            engagement_stats.columns = ["avg_likes", "total_likes"]
            engagement_stats = engagement_stats.reset_index()
            
            comment_counts = df.groupby("user_id").size().reset_index(name="comment_count")
            engagement_profile = engagement_stats.merge(comment_counts, on="user_id", how="left")
            
            # Engagement tiers
            engagement_profile["engagement_tier"] = pd.cut(
                engagement_profile["avg_likes"],
                bins=[-1, 1, 10, float("inf")],
                labels=["Low", "Medium", "High"]
            )
        else:
            engagement_profile = df.groupby("user_id").size().reset_index(name="comment_count")
            engagement_profile["engagement_tier"] = "Medium"
            engagement_profile["avg_likes"] = 0
            engagement_profile["total_likes"] = 0
        
        # Merge profiles
        personas = (dominant_categories
                   .merge(sentiment_profile, on="user_id", how="left")
                   .merge(engagement_profile, on="user_id", how="left"))
        
        # Top subtopic per user
        if "subtopic" in cats_df.columns:
            subtopic_counts = cats_df.groupby(["user_id", "subtopic"]).size().reset_index(name="count")
            subtopic_counts["rank"] = subtopic_counts.groupby("user_id")["count"].rank("dense", ascending=False)
            top_subtopics = (subtopic_counts[subtopic_counts["rank"] == 1]
                           [["user_id", "subtopic"]]
                           .rename(columns={"subtopic": "top_subtopic"}))
            personas = personas.merge(top_subtopics, on="user_id", how="left")
        
        # Generate persona labels
        def create_persona_label(row):
            category = str(row.get("dominant_category", "General")).title()
            sentiment = row.get("dominant_sentiment", "neutral")
            tier = str(row.get("engagement_tier", "Medium"))
            
            if sentiment == "positive" and tier == "High":
                return f"{category} Advocate"
            elif sentiment == "positive":
                return f"{category} Enthusiast"
            elif sentiment == "negative":
                return f"{category} Critic"
            else:
                return f"{category} Explorer"
        
        personas["persona_label"] = personas.apply(create_persona_label, axis=1)
        
        # Select and order columns
        columns = [
            "user_id", "persona_label", "dominant_category", "top_subtopic",
            "dominant_sentiment", "engagement_tier", "comment_count", 
            "avg_likes", "total_likes"
        ]
        available_columns = [c for c in columns if c in personas.columns]
        
        return personas[available_columns].sort_values("comment_count", ascending=False)
    
    def create_cluster_personas(self,
                               df: pd.DataFrame,
                               cats_df: pd.DataFrame,
                               text_column: str = "text",
                               fast_mode: bool = True,
                               top_users: int = 4000) -> pd.DataFrame:
        """
        Create personas using Sentence Transformers + HDBSCAN clustering
        
        Args:
            df: Original DataFrame with user comments
            cats_df: DataFrame with category classifications
            text_column: Column containing comment text
            fast_mode: Use faster but less accurate settings
            top_users: Limit to top N most active users
        
        Returns:
            DataFrame with persona assignments including cluster information
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found")
        
        # Build user documents
        try:
            user_ids, user_docs, user_stats = self.build_user_documents(
                df, text_column, top_users=top_users
            )
        except Exception as e:
            logger.warning(f"Failed to build user documents: {e}")
            return self.create_heuristic_personas(df, cats_df)
        
        if len(user_ids) < 10:
            logger.info("Too few users for clustering, using heuristic personas")
            return self.create_heuristic_personas(df, cats_df)
        
        try:
            # Generate embeddings
            embeddings = self.embed_documents(
                user_docs,
                batch_size=128 if fast_mode else 256
            )
            
            # Cluster embeddings
            clustering_config = self.config.get("clustering", {})
            if fast_mode:
                min_cluster_size = clustering_config.get("fast_mode", {}).get("min_cluster_size", 10)
            else:
                min_cluster_size = clustering_config.get("hdbscan", {}).get("min_cluster_size", 20)
            
            cluster_labels = self.cluster_embeddings(embeddings, min_cluster_size=min_cluster_size)
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return self.create_heuristic_personas(df, cats_df)
        
        # Extract cluster information
        cluster_terms_df = self.extract_cluster_terms(user_ids, user_docs, cluster_labels)
        user_cluster_map = {uid: int(label) for uid, label in zip(user_ids, cluster_labels)}
        cluster_examples_df = self.extract_cluster_examples(df, user_cluster_map, text_column)
        
        # Create base personas using heuristics
        base_personas = self.create_heuristic_personas(df, cats_df)
        
        # Add cluster information
        base_personas["persona_cluster"] = (base_personas["user_id"]
                                          .astype(str)
                                          .map(user_cluster_map)
                                          .fillna(-1)
                                          .astype(int))
        
        # Merge cluster terms and examples
        base_personas = base_personas.merge(cluster_terms_df, on="persona_cluster", how="left")
        base_personas = base_personas.merge(cluster_examples_df, on="persona_cluster", how="left")
        
        # Update persona labels with cluster information
        def create_cluster_persona_label(row):
            if row.get("persona_cluster", -1) < 0 or not isinstance(row.get("cluster_terms"), str):
                category = str(row.get("dominant_category", "General")).title()
                return f"{category} Explorer"
            
            # Use first two cluster terms
            terms = row["cluster_terms"].split(", ")[:2]
            category = str(row.get("dominant_category", "General")).title()
            return f"{category} — {', '.join(terms)}"
        
        base_personas["persona_label"] = base_personas.apply(create_cluster_persona_label, axis=1)
        
        # Select and order columns
        columns = [
            "user_id", "persona_label", "persona_cluster", "cluster_terms", "cluster_examples",
            "dominant_category", "top_subtopic", "dominant_sentiment", "engagement_tier",
            "comment_count", "avg_likes", "total_likes"
        ]
        available_columns = [c for c in columns if c in base_personas.columns]
        
        result = base_personas[available_columns].sort_values(
            ["persona_cluster", "comment_count"], 
            ascending=[True, False]
        )
        
        logger.info(f"Created {len(result)} personas with {len(cluster_terms_df)} clusters")
        return result
    
    def analyze_personas(self,
                        df: pd.DataFrame,
                        cats_df: pd.DataFrame,
                        text_column: str = "text",
                        use_clustering: bool = True,
                        fast_mode: bool = True) -> pd.DataFrame:
        """
        Main method to analyze user personas
        
        Args:
            df: DataFrame with user comments
            cats_df: DataFrame with category classifications
            text_column: Column containing comment text
            use_clustering: Whether to use ML clustering (requires sentence-transformers + hdbscan)
            fast_mode: Use faster settings for clustering
        
        Returns:
            DataFrame with persona analysis results
        """
        if use_clustering and SentenceTransformer is not None and hdbscan is not None:
            return self.create_cluster_personas(df, cats_df, text_column, fast_mode=fast_mode)
        else:
            if use_clustering:
                logger.warning("Clustering dependencies not available, using heuristic personas")
            return self.create_heuristic_personas(df, cats_df)

def create_persona_clustering(config: Optional[Dict[str, Any]] = None) -> PersonaClustering:
    """Factory function to create persona clustering module"""
    return PersonaClustering(config)