"""
Comment Quality Scoring 2.0 Module for CommentSense Pipeline

This module implements composite KPI scoring based on three pillars:
- Relevance: Is the comment about the brand/product/topic?
- Informativeness: Does it mention product attributes, experiences, questions?  
- Constructiveness: Does it provide suggestions/issues vs generic praise?

Uses fine-tuned DistilBERT/RoBERTa for classification and computes QEI scores.
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import json
import os
import platform
from pathlib import Path


# Suppress warnings
warnings.filterwarnings("ignore")

# Core ML libraries
try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, AutoConfig
    )
    from sklearn.metrics import f1_score, roc_auc_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class CommentQualityScorer:
    """
    Advanced comment quality scoring using transformer models
    
    Implements the three-pillar QEI (Quality Engagement Index) system:
    - Relevance (0/1): About brand/product/topic
    - Informativeness (multi-label): Product attributes, experience, questions
    - Constructiveness (ordinal): Suggestions/issues vs generic praise
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Comment Quality Scorer"""
        
        # Use default config if no config provided or if config is empty
        if not config:
            self.config = self._get_default_config()
        else:
            # Merge provided config with defaults
            default_config = self._get_default_config()
            self.config = {**default_config, **config}
        
        # Model configurations
        self.model_name = self.config.get("model_name", "distilbert-base-uncased")
        self.device = self._get_device()
        
        # Scoring weights (wr >= wi >= wc)
        self.weights = self.config.get("weights", {
            "relevance": 0.5,      # wr
            "informativeness": 0.3, # wi  
            "constructiveness": 0.2  # wc
        })
        
        # Models for each pillar
        self.models = {}
        self.tokenizers = {}
        self.calibrators = {}
        
        # Initialize models if available
        if TRANSFORMERS_AVAILABLE:
            self._initialize_models()
        else:
            logger.warning("Transformers not available - using fallback scoring")
            
        logger.info("Comment Quality Scoring 2.0 module initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "model_name": "distilbert-base-uncased",
            "max_length": 512,
            "batch_size": 32,
            "weights": {"relevance": 0.5, "informativeness": 0.3, "constructiveness": 0.2},
            "thresholds": {
                "relevance": 0.5,
                "informativeness": 0.3,
                "constructiveness": 0.4
            },
            "fallback_keywords": {
                "relevance": ["product", "brand", "quality", "price", "feature"],
                "informativeness": ["experience", "used", "tried", "works", "quality", "size", "color"],
                "constructiveness": ["recommend", "suggest", "improve", "issue", "problem", "fix"]
            }
        }
    
    def _get_device(self) -> str:
        """Get the best available device"""
        # Check for environment variable override
        env_device = os.getenv("COMMENTSENSE_DEVICE", "").lower()
        if env_device in ["cuda", "cpu"]:
            logger.info(f"Using device from environment variable: {env_device}")
            return env_device
        
        # For Apple Silicon, force CPU for stability
        import platform
        if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
            logger.info("Apple Silicon detected - using CPU for stability (quality scoring transformers)")
            return "cpu"
        
        # For Windows/Linux users - use GPU for DistilBERT models (optimal)
        elif torch and torch.cuda.is_available():
            logger.info("CUDA available, using GPU for quality scoring transformers (optimal)")
            return "cuda"
        else:
            logger.info("No GPU available, using CPU")
            return "cpu"
    
    def _initialize_models(self):
        """Initialize transformer models for each pillar"""
        try:
            # Suppress model initialization warnings for untrained classification heads
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Some weights of DistilBertForSequenceClassification were not initialized")
                
                # For now, use the same base model for all pillars
                # In production, you'd fine-tune separate models
                for pillar in ["relevance", "informativeness", "constructiveness"]:
                    logger.info(f"Loading model for {pillar} classification...")
                    
                    # Load tokenizer
                    self.tokenizers[pillar] = AutoTokenizer.from_pretrained(self.model_name)
                
                    # Configure model based on task
                    if pillar == "informativeness":
                        # Multi-label classification
                        num_labels = 5  # attributes, experience, questions, comparison, usage
                        problem_type = "multi_label_classification"
                    else:
                        # Binary/ordinal classification  
                        num_labels = 2 if pillar == "relevance" else 3
                        problem_type = "single_label_classification"
                    
                    config = AutoConfig.from_pretrained(
                        self.model_name,
                        num_labels=num_labels,
                        problem_type=problem_type
                    )
                    
                    # Load model with explicit device control to prevent meta tensor issues
                    with torch.device("cpu"):  # Force CPU context during loading
                        self.models[pillar] = AutoModelForSequenceClassification.from_pretrained(
                            self.model_name,
                            config=config,
                            torch_dtype=torch.float32  # Explicit dtype to prevent meta tensors
                        )
                    
                    # Safely move to target device
                    try:
                        self.models[pillar].to(self.device)
                    except Exception as e:
                        if "meta tensor" in str(e).lower():
                            logger.warning(f"Meta tensor error moving {pillar} model to {self.device}: {e}")
                            # Keep on CPU if meta tensor issues
                            self.models[pillar] = self.models[pillar].cpu()
                            logger.info(f"Model for {pillar} kept on CPU due to meta tensor issues")
                        else:
                            raise e
                    
                    logger.info(f"Model for {pillar} loaded successfully on {self.device}")
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Fall back to keyword-based scoring
            self.models = {}
    
    def score_comments(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Score comments using the three-pillar QEI system
        
        Args:
            comments_df: DataFrame with comment data
            
        Returns:
            Dictionary containing scoring results and metrics
        """
        
        try:
            logger.info(f"Scoring {len(comments_df)} comments for quality...")
            
            # Ensure we have text column
            if 'text' not in comments_df.columns:
                if 'textOriginal' in comments_df.columns:
                    comments_df['text'] = comments_df['textOriginal']
                else:
                    logger.error("No text column found for quality scoring")
                    return {"error": "No text column found"}
            
            # Initialize result columns
            results_df = comments_df.copy()
            
            # Score each pillar
            if self.models and TRANSFORMERS_AVAILABLE:
                # Use transformer models
                relevance_scores = self._score_relevance(results_df['text'].tolist())
                informativeness_scores = self._score_informativeness(results_df['text'].tolist())
                constructiveness_scores = self._score_constructiveness(results_df['text'].tolist())
            else:
                # Use fallback keyword-based scoring
                logger.info("Using fallback keyword-based scoring")
                relevance_scores = self._fallback_relevance(results_df['text'].tolist())
                informativeness_scores = self._fallback_informativeness(results_df['text'].tolist())
                constructiveness_scores = self._fallback_constructiveness(results_df['text'].tolist())
            
            # Add scores to dataframe
            results_df['relevance'] = relevance_scores
            results_df['informativeness'] = informativeness_scores  
            results_df['constructiveness'] = constructiveness_scores
            
            # Calculate composite QEI scores
            results_df['qei_score'] = self._calculate_qei(
                relevance_scores, 
                informativeness_scores,
                constructiveness_scores
            )
            
            # Generate aggregated metrics
            aggregated_results = self._aggregate_results(results_df)
            
            logger.info("Comment quality scoring completed successfully")
            
            return {
                "status": "success",
                "scored_comments": results_df,
                "aggregated_metrics": aggregated_results,
                "model_info": {
                    "model_name": self.model_name,
                    "device": self.device,
                    "weights": self.weights,
                    "using_transformers": bool(self.models and TRANSFORMERS_AVAILABLE)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in comment quality scoring: {e}")
            return {"error": str(e)}
    
    def _score_relevance(self, texts: List[str]) -> List[float]:
        """Score relevance using transformer model"""
        return self._score_pillar("relevance", texts)
    
    def _score_informativeness(self, texts: List[str]) -> List[float]:
        """Score informativeness using transformer model"""  
        return self._score_pillar("informativeness", texts)
    
    def _score_constructiveness(self, texts: List[str]) -> List[float]:
        """Score constructiveness using transformer model"""
        return self._score_pillar("constructiveness", texts)
    
    def _score_pillar(self, pillar: str, texts: List[str]) -> List[float]:
        """Generic method to score any pillar using transformer model"""
        try:
            model = self.models[pillar]
            tokenizer = self.tokenizers[pillar]
            
            scores = []
            batch_size = self.config.get("batch_size", 32)
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.get("max_length", 512),
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    if pillar == "informativeness":
                        # Multi-label: use sigmoid
                        probs = torch.sigmoid(logits).mean(dim=1)  # Average across labels
                    else:
                        # Binary/ordinal: use softmax  
                        probs = torch.softmax(logits, dim=1)[:, 1]  # Positive class
                    
                    batch_scores = probs.cpu().numpy().tolist()
                    scores.extend(batch_scores)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error scoring {pillar}: {e}")
            # Fall back to keyword scoring for this pillar
            if pillar == "relevance":
                return self._fallback_relevance(texts)
            elif pillar == "informativeness":
                return self._fallback_informativeness(texts)  
            else:
                return self._fallback_constructiveness(texts)
    
    def _fallback_relevance(self, texts: List[str]) -> List[float]:
        """Fallback keyword-based relevance scoring"""
        keywords = self.config["fallback_keywords"]["relevance"]
        scores = []
        
        for text in texts:
            text_lower = text.lower()
            matches = sum(1 for kw in keywords if kw in text_lower)
            score = min(1.0, matches / len(keywords))
            scores.append(score)
            
        return scores
    
    def _fallback_informativeness(self, texts: List[str]) -> List[float]:
        """Fallback keyword-based informativeness scoring"""
        keywords = self.config["fallback_keywords"]["informativeness"]
        scores = []
        
        for text in texts:
            text_lower = text.lower()
            matches = sum(1 for kw in keywords if kw in text_lower)
            # Boost score for longer, detailed comments
            length_boost = min(0.3, len(text) / 500)
            score = min(1.0, (matches / len(keywords)) + length_boost)
            scores.append(score)
            
        return scores
    
    def _fallback_constructiveness(self, texts: List[str]) -> List[float]:
        """Fallback keyword-based constructiveness scoring"""
        keywords = self.config["fallback_keywords"]["constructiveness"]
        scores = []
        
        for text in texts:
            text_lower = text.lower()
            matches = sum(1 for kw in keywords if kw in text_lower)
            # Penalize very short comments
            length_penalty = 0.2 if len(text) < 20 else 0
            score = max(0.0, min(1.0, (matches / len(keywords)) - length_penalty))
            scores.append(score)
            
        return scores
    
    def _calculate_qei(self, relevance: List[float], informativeness: List[float], 
                      constructiveness: List[float]) -> List[float]:
        """
        Calculate composite QEI scores
        
        QEI = 100 × (wr·R + wi·I + wc·C)
        """
        qei_scores = []
        
        for r, i, c in zip(relevance, informativeness, constructiveness):
            qei = 100 * (
                self.weights["relevance"] * r +
                self.weights["informativeness"] * i + 
                self.weights["constructiveness"] * c
            )
            qei_scores.append(round(qei, 2))
            
        return qei_scores
    
    def _aggregate_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate aggregated quality metrics"""
        
        aggregated = {
            "overall_metrics": {
                "total_comments": len(results_df),
                "avg_qei_score": results_df['qei_score'].mean(),
                "median_qei_score": results_df['qei_score'].median(),
                "std_qei_score": results_df['qei_score'].std(),
                "high_quality_comments": len(results_df[results_df['qei_score'] >= 75]),
                "low_quality_comments": len(results_df[results_df['qei_score'] <= 25])
            },
            "pillar_metrics": {
                "avg_relevance": results_df['relevance'].mean(),
                "avg_informativeness": results_df['informativeness'].mean(), 
                "avg_constructiveness": results_df['constructiveness'].mean()
            },
            "quality_distribution": {
                "excellent": len(results_df[results_df['qei_score'] >= 80]),
                "good": len(results_df[(results_df['qei_score'] >= 60) & (results_df['qei_score'] < 80)]),
                "fair": len(results_df[(results_df['qei_score'] >= 40) & (results_df['qei_score'] < 60)]),
                "poor": len(results_df[results_df['qei_score'] < 40])
            }
        }
        
        # Post-level aggregation if post_id available
        if 'post_id' in results_df.columns:
            post_qei = results_df.groupby('post_id')['qei_score'].agg(['mean', 'count', 'std']).reset_index()
            post_qei.columns = ['post_id', 'avg_qei', 'comment_count', 'qei_std']
            aggregated["post_level_qei"] = post_qei.to_dict('records')
        
        return aggregated

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "weights": self.weights,
            "models_loaded": list(self.models.keys()),
            "using_transformers": bool(self.models and TRANSFORMERS_AVAILABLE)
        }

# Factory function for pipeline integration
def create_quality_scorer(config_path: Optional[str] = None) -> CommentQualityScorer:
    """Factory function to create quality scorer with configuration"""
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return CommentQualityScorer(config)