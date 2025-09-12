"""
Category + Sub-Topic Classification Module for CommentSense Pipeline

This module provides:
- Zero-shot classification using BART-large-mnli
- BERTopic fallback for topic modeling
- Heuristic category assignment with confidence scoring
- Beauty/cosmetics domain-specific categories
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import re

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
        HF_DEVICE = -1
        HF_DTYPE = torch.float32  # Use float32 for stability
        logger.info("Apple Silicon detected - using CPU only (PyTorch macOS compatibility)")
    else:
        # For Windows/Linux users - enable CUDA if available
        _CUDA = torch.cuda.is_available()
        TORCH_DEVICE = "cuda" if _CUDA else "cpu"
        HF_DEVICE = 0 if _CUDA else -1
        HF_DTYPE = torch.float16 if _CUDA else torch.float32
        if _CUDA:
            logger.info("CUDA available, using GPU acceleration")
        else:
            logger.info("No GPU available, using CPU")
except Exception:
    torch = None
    TORCH_DEVICE = "cpu"
    HF_DEVICE = -1
    HF_DTYPE = torch.float32

# ---------------- Optional Dependencies ----------------
try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from bertopic import BERTopic
except Exception:
    BERTopic = None

# ---------------- Domain Categories ----------------
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

# Text cleaning utilities
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)

def clean_text(s: str) -> str:
    """Clean and normalize text"""
    s = "" if pd.isna(s) else str(s)
    s = URL_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

class CategoryClassifier:
    """
    Category + Sub-Topic Classification using zero-shot BART and BERTopic
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the category classifier"""
        
        self.config = config or CONFIG
        self._zs_pipe = None  # Cached zero-shot pipeline
        self._sbert_model = None  # Cached sentence transformer
        
        logger.info("Category Classification module initialized")
    
    def _get_zero_shot_pipeline(self):
        """Get or create zero-shot classification pipeline"""
        if hf_pipeline is None:
            raise ImportError("transformers not installed")
        
        if self._zs_pipe is None:
            model_config = self.config.get("models", {}).get("zero_shot_classifier", {})
            model_name = model_config.get("primary", "facebook/bart-large-mnli")
            
            # Load with meta tensor prevention
            with torch.no_grad():
                torch.set_default_device('cpu')
                
                self._zs_pipe = hf_pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=HF_DEVICE,
                    model_kwargs={
                        'torch_dtype': HF_DTYPE or torch.float32,
                        'device_map': None,  # Disable device mapping to prevent meta tensors
                        'low_cpu_mem_usage': False,  # Disable to prevent meta tensors
                    }
                )
            logger.info(f"Loaded zero-shot classifier: {model_name}")
        
        return self._zs_pipe
    
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
                self._sbert_model = SentenceTransformer(model_name, device=TORCH_DEVICE)
            except Exception as e:
                if "meta tensor" in str(e).lower() or "cannot copy out of meta tensor" in str(e).lower():
                    logger.warning(f"Meta tensor error with {TORCH_DEVICE}, forcing CPU: {e}")
                    self._sbert_model = SentenceTransformer(model_name, device="cpu")
                    logger.info(f"Loaded sentence transformer: {model_name} on CPU (fallback)")
                else:
                    raise e
            logger.info(f"Loaded individual sentence transformer: {model_name}")
        
        return self._sbert_model
    
    def assign_category_heuristic(self, text: str) -> str:
        """Assign category using keyword heuristics"""
        t = (text or "").lower()
        counts = {c: 0 for c in CATEGORIES}
        for c, kws in CATEGORY_KW.items():
            for w in kws:
                if w in t:
                    counts[c] += 1
        cat = max(counts, key=counts.get)
        return cat if counts[cat] > 0 else "general"
    
    def assign_subtopic(self, text: str, category: str) -> str:
        """Assign subtopic within a category"""
        t = (text or "").lower()
        subs = SUBTOPIC_KW.get(category, {})
        best, score = "general", 0
        for name, kws in subs.items():
            s = sum(1 for w in kws if w in t)
            if s > score:
                best, score = name, s
        return best
    
    def category_confidence(self, text: str, category: str) -> float:
        """Calculate confidence score for category assignment"""
        t = (text or "").lower()
        kws = CATEGORY_KW.get(category, [])
        hits = sum(1 for w in kws if w in t)
        denom = max(3, len(kws)//3 or 1)
        return min(1.0, hits/denom) if kws else 0.0
    
    def classify_zero_shot(self, texts: List[str], batch_size: int = 64) -> Tuple[List[str], List[float]]:
        """
        Zero-shot classification using BART-large-mnli
        Returns categories and confidence scores
        """
        pipe = self._get_zero_shot_pipeline()
        
        # Deduplicate for efficiency
        unique_texts, map_idx, seen = [], [], {}
        for t in texts:
            if t in seen:
                map_idx.append(seen[t])
            else:
                seen[t] = len(unique_texts)
                unique_texts.append(t)
                map_idx.append(seen[t])
        
        pred_u, conf_u = [], []
        
        # Process in batches
        for i in range(0, len(unique_texts), batch_size):
            chunk = unique_texts[i:i+batch_size]
            results = pipe(chunk, candidate_labels=CATEGORIES, multi_label=False)
            
            if isinstance(results, dict):
                results = [results]
            
            for r in results:
                pred_u.append(r["labels"][0].lower())
                conf_u.append(float(r["scores"][0]))
        
        # Map back to original order
        predictions = [pred_u[i] for i in map_idx]
        confidences = [conf_u[i] for i in map_idx]
        
        return predictions, confidences
    
    def classify_bertopic(self, texts: List[str]) -> List[int]:
        """
        Topic modeling using BERTopic
        Returns topic IDs
        """
        if BERTopic is None:
            raise ImportError("bertopic not installed")
        
        try:
            # Use pre-computed embeddings if available
            embeddings = None
            if SentenceTransformer is not None:
                sbert = self._get_sentence_transformer()
                batch_size = 512 if _CUDA else 256
                embeddings = sbert.encode(texts, batch_size=batch_size, 
                                        normalize_embeddings=True, show_progress_bar=False)
            
            # Configure BERTopic
            bertopic_config = self.config.get("models", {}).get("bertopic", {})
            topic_model = BERTopic(
                verbose=bertopic_config.get("verbose", False),
                calculate_probabilities=bertopic_config.get("calculate_probabilities", False),
                embedding_model=self._sbert_model if self._sbert_model else None
            )
            
            topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
            return topics.tolist()
            
        except Exception as e:
            logger.warning(f"BERTopic failed: {e}")
            return [-1] * len(texts)
    
    def analyze_categories(self, 
                          df: pd.DataFrame,
                          text_column: str = "text",
                          use_zero_shot: bool = True,
                          use_bertopic: bool = False) -> pd.DataFrame:
        """
        Analyze categories and subtopics for a DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to classify
            use_zero_shot: Whether to use zero-shot classification
            use_bertopic: Whether to add BERTopic topics
        
        Returns:
            DataFrame with category analysis results
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Clean text
        texts = df[text_column].astype(str).map(clean_text).tolist()
        
        # Category classification
        if use_zero_shot:
            try:
                categories, confidences = self.classify_zero_shot(texts)
                logger.info("Used zero-shot classification")
            except Exception as e:
                logger.warning(f"Zero-shot failed: {e}, falling back to heuristics")
                categories = [self.assign_category_heuristic(t) for t in texts]
                confidences = [self.category_confidence(t, c) for t, c in zip(texts, categories)]
        else:
            categories = [self.assign_category_heuristic(t) for t in texts]
            confidences = [self.category_confidence(t, c) for t, c in zip(texts, categories)]
        
        # Subtopic assignment
        subtopics = [self.assign_subtopic(t, c) for t, c in zip(texts, categories)]
        
        # Create results DataFrame
        result_cols = ["comment_id", "post_id", "user_id"] if all(c in df.columns for c in ["comment_id", "post_id", "user_id"]) else []
        available_cols = [c for c in result_cols if c in df.columns]
        
        results = df[available_cols].copy() if available_cols else pd.DataFrame(index=df.index)
        results["text"] = df[text_column]
        results["category"] = categories
        results["category_score"] = confidences
        results["subtopic"] = subtopics
        
        # Optional BERTopic
        if use_bertopic:
            try:
                topics = self.classify_bertopic(texts)
                results["topic"] = topics
                logger.info("Added BERTopic topics")
            except Exception as e:
                logger.warning(f"BERTopic failed: {e}")
                results["topic"] = -1
        
        logger.info(f"Processed {len(results)} texts for category classification")
        return results

def create_category_classifier(config: Optional[Dict[str, Any]] = None) -> CategoryClassifier:
    """Factory function to create category classifier"""
    return CategoryClassifier(config)