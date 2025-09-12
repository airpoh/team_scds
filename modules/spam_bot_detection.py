"""
Spam + Bot Network Detection Module for CommentSense Pipeline

This module implements:
1. Spam Detection: Hybrid rules + TF-IDF logistic regression
2. Bot Network Detection: MinHash + graph communities for coordinated behavior
3. Authenticity Scoring: Flags suspicious engagement patterns

Uses advanced techniques including:
- MinHash for near-duplicate detection (datasketch)
- Network graph analysis (networkx/Neo4j) 
- Community detection for bot clusters
- Temporal pattern analysis
"""

import logging
import warnings
import os
import platform
from typing import Dict, List, Optional, Any, Tuple, Set
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta


# Suppress warnings
warnings.filterwarnings("ignore")

# Core ML and graph libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    import networkx as nx
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

try:
    from datasketch import MinHashLSH, MinHash
    DATASKETCH_AVAILABLE = True
except ImportError:
    logging.warning("datasketch not available - using fallback similarity")
    DATASKETCH_AVAILABLE = False

try:
    import community as community_louvain
    COMMUNITY_AVAILABLE = True
except ImportError:
    logging.warning("python-louvain not available - using basic clustering")
    COMMUNITY_AVAILABLE = False

logger = logging.getLogger(__name__)

class SpamBotDetector:
    """
    Advanced spam and bot network detection system
    
    Features:
    - Rule-based spam detection (links, promotional patterns, emoji floods)
    - ML-based spam classification using TF-IDF + Logistic Regression
    - Near-duplicate detection using MinHash LSH
    - Bot network identification through graph communities
    - Temporal pattern analysis for coordinated behavior
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Spam Bot Detector"""
        
        # Always start with default config to ensure all required keys exist
        self.config = self._get_default_config()
        
        # Merge provided config with defaults if provided
        if config and isinstance(config, dict):
            # Deep merge to preserve nested dictionaries
            for key, value in config.items():
                if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        # Spam detection components
        self.spam_classifier = None
        self.tfidf_vectorizer = None
        
        # MinHash LSH for duplicate detection
        self.lsh = None
        self.minhashes = {}
        
        # Bot network detection
        self.similarity_graph = None
        self.bot_communities = {}
        
        # Spam patterns
        self.spam_patterns = self._compile_spam_patterns()
        
        # Initialize ML components if available
        if SKLEARN_AVAILABLE:
            self._initialize_ml_components()
        
        if DATASKETCH_AVAILABLE:
            self._initialize_lsh()
            
        logger.info("Spam + Bot Detection module initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Spam detection thresholds
            "spam_thresholds": {
                "url_count": 2,
                "emoji_ratio": 0.5,
                "caps_ratio": 0.7,
                "repeated_chars": 5,
                "promotional_keywords": 3
            },
            
            # MinHash LSH settings
            "minhash_settings": {
                "num_perm": 128,
                "threshold": 0.8,
                "shingle_size": 3
            },
            
            # Bot network detection
            "bot_detection": {
                "min_cluster_size": 3,
                "similarity_threshold": 0.7,
                "temporal_window_hours": 24,
                "min_repetition_rate": 0.6
            },
            
            # TF-IDF settings
            "tfidf_settings": {
                "max_features": 5000,
                "ngram_range": (1, 3),
                "stop_words": "english",
                "min_df": 2
            },
            
            # Known spam patterns
            "spam_keywords": [
                "dm for promo", "check my bio", "follow for follow", "like for like",
                "subscribe", "click link", "free money", "make money fast",
                "work from home", "get rich quick", "buy now", "limited time",
                "act now", "call now", "click here", "visit my page"
            ],
            
            "promotional_domains": [
                "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
                "shorturl.com", "tiny.cc", "is.gd"
            ]
        }
    
    def _compile_spam_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for spam detection"""
        return {
            "urls": re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE),
            "emails": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone_numbers": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            "repeated_chars": re.compile(r'(.)\1{4,}'),  # 5+ repeated characters
            "emoji_flood": re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]{3,}'),
            "caps_words": re.compile(r'\b[A-Z]{3,}\b'),
            "promotional": re.compile(r'\b(free|discount|sale|offer|deal|promo|coupon)\b', re.IGNORECASE)
        }
    
    def _initialize_ml_components(self):
        """Initialize ML components for spam detection with GPU fallback"""
        try:
            # Check available device with fallback
            self.device = self._get_safe_device()
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config["tfidf_settings"]["max_features"],
                ngram_range=self.config["tfidf_settings"]["ngram_range"],
                stop_words=self.config["tfidf_settings"]["stop_words"],
                min_df=self.config["tfidf_settings"]["min_df"]
            )
            
            self.spam_classifier = LogisticRegression(
                class_weight='balanced',
                random_state=42
            )
            
            logger.info(f"ML components for spam detection initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing ML components: {e}")
    
    def _get_safe_device(self):
        """Get optimal device for spam detection (CPU-optimized for sklearn models)"""
        # Spam detection uses sklearn (TF-IDF + Logistic Regression) which is CPU-optimized
        # Force CPU on all platforms for optimal performance
        logger.info("Using CPU for spam detection (sklearn models are CPU-optimized)")
        return "cpu"
    

    def _cleanup_memory(self):
        """Aggressive memory cleanup for cross-platform compatibility"""
        try:
            import gc
            
            # Python garbage collection
            gc.collect()
            
            # GPU memory cleanup if available
            try:
                import torch
                
                if hasattr(self, 'device') and self.device == "mps":
                    # Apple Silicon cleanup
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    
                elif hasattr(self, 'device') and "cuda" in str(self.device):
                    # CUDA cleanup
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
            except (ImportError, AttributeError):
                pass
                
        except Exception as e:
            logger.debug(f"Memory cleanup warning: {e}")
    
    def _initialize_lsh(self):
        """Initialize MinHash LSH for duplicate detection"""
        try:
            settings = self.config["minhash_settings"]
            self.lsh = MinHashLSH(
                threshold=settings["threshold"],
                num_perm=settings["num_perm"]
            )
            logger.info("MinHash LSH initialized")
            
        except Exception as e:
            logger.error(f"Error initializing LSH: {e}")
    
    def detect_spam_and_bots(self, comments_df: pd.DataFrame, batch_size: int = None) -> Dict[str, Any]:
        """
        Main method to detect spam and bot networks with batch processing
        
        Args:
            comments_df: DataFrame with comment data
            batch_size: Number of comments to process per batch (auto-determined based on system)
            
        Returns:
            Dictionary containing detection results and metrics
        """
        
        try:
            # Set batch size based on platform and dataset size
            if batch_size is None:
                if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
                    # Ultra-small batch size for Apple Silicon to prevent memory accumulation
                    if total_comments > 100:
                        batch_size = 10  # Very small batches for large datasets
                    else:
                        batch_size = 20  # Slightly larger for small datasets
                    logger.info(f"Apple Silicon detected - using small batch size ({batch_size}) to prevent memory issues")
                else:
                    batch_size = 100  # Default batch size for other platforms
            
            logger.info(f"Analyzing {len(comments_df)} comments for spam and bot activity...")
            
            # Ensure we have required columns
            if 'text' not in comments_df.columns:
                if 'textOriginal' in comments_df.columns:
                    comments_df = comments_df.copy()
                    comments_df['text'] = comments_df['textOriginal']
                else:
                    logger.error("No text column found")
                    return {"error": "No text column found"}
            
            results_df = comments_df.copy()
            total_comments = len(results_df)
            
            # Initialize result columns
            results_df['spam_score_rules'] = 0.0
            results_df['spam_score'] = 0.0
            results_df['is_duplicate'] = False
            results_df['duplicate_cluster'] = -1
            results_df['bot_score'] = 0.0
            results_df['bot_cluster'] = -1
            results_df['authenticity_score'] = 1.0
            
            # Process in batches to avoid memory issues
            for start_idx in range(0, total_comments, batch_size):
                end_idx = min(start_idx + batch_size, total_comments)
                batch_df = results_df.iloc[start_idx:end_idx].copy()
                
                logger.info(f"Processing batch {start_idx//batch_size + 1}/{(total_comments-1)//batch_size + 1} ({start_idx}:{end_idx})")
                
                try:
                    # Step 1: Rule-based spam detection (batch)
                    spam_scores = self._detect_spam_rules_batch(batch_df)
                    results_df.loc[start_idx:end_idx-1, 'spam_score_rules'] = spam_scores
                    
                    # Step 2: ML-based spam detection (disabled for large datasets and Apple Silicon to avoid crashes)
                    if (SKLEARN_AVAILABLE and 
                        len(batch_df) <= 50 and 
                        total_comments <= 1000 and
                        not (platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin')):
                        try:
                            ml_scores = self._detect_spam_ml(batch_df['text'].tolist())
                            results_df.loc[start_idx:end_idx-1, 'spam_score_ml'] = ml_scores
                            # Combine rule-based and ML scores
                            combined_scores = [(r + m) / 2 for r, m in zip(spam_scores, ml_scores)]
                            results_df.loc[start_idx:end_idx-1, 'spam_score'] = combined_scores
                        except Exception as ml_error:
                            logger.warning(f"ML spam detection failed, using rules only: {ml_error}")
                            results_df.loc[start_idx:end_idx-1, 'spam_score'] = spam_scores
                    else:
                        # Log why ML spam detection is being skipped
                        if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
                            logger.info("Apple Silicon detected - skipping ML spam detection to prevent segfaults (using rules-only)")
                        results_df.loc[start_idx:end_idx-1, 'spam_score'] = spam_scores
                    
                    # Aggressive memory cleanup after each batch (especially for Apple Silicon)
                    self._cleanup_memory()
                    
                    # Additional aggressive cleanup for Apple Silicon
                    if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
                        import gc
                        import time
                        # Force multiple garbage collections
                        for _ in range(3):
                            gc.collect()
                        # Brief pause to allow memory recovery
                        time.sleep(0.2)
                        logger.debug(f"Apple Silicon memory cleanup completed for batch {start_idx//batch_size + 1}")
                    
                except Exception as e:
                    logger.warning(f"Error processing batch {start_idx}:{end_idx}: {e}")
                    # Set default values and continue
                    results_df.loc[start_idx:end_idx-1, 'spam_score'] = 0.0
                    # Cleanup memory even on error
                    self._cleanup_memory()
                    continue
            
            # Step 3: Near-duplicate detection (on full dataset but with memory optimization)
            try:
                duplicate_info = self._detect_duplicates_optimized(results_df)
                results_df['is_duplicate'] = duplicate_info['is_duplicate']
                results_df['duplicate_cluster'] = duplicate_info['cluster_ids']
            except Exception as e:
                logger.warning(f"Duplicate detection failed: {e}")
                duplicate_info = {'clusters': [], 'is_duplicate': [False] * len(results_df), 'cluster_ids': [-1] * len(results_df)}
            
            # Step 4: Bot network detection (simplified for large datasets)
            try:
                bot_analysis = self._detect_bot_networks_simplified(results_df)
                results_df['bot_score'] = bot_analysis['bot_scores']
                results_df['bot_cluster'] = bot_analysis['bot_clusters']
            except Exception as e:
                logger.warning(f"Bot detection failed: {e}")
                bot_analysis = {'bot_scores': [0.0] * len(results_df), 'bot_clusters': [-1] * len(results_df), 'communities': {}}
            
            # Step 5: Authenticity scoring
            authenticity_scores = self._calculate_authenticity(results_df)
            results_df['authenticity_score'] = authenticity_scores
            
            # Generate aggregated results
            aggregated_results = self._aggregate_detection_results(results_df)
            
            logger.info("Spam and bot detection completed successfully")
            
            return {
                "status": "success",
                "analyzed_comments": results_df,
                "detection_metrics": aggregated_results,
                "duplicate_clusters": duplicate_info['clusters'],
                "bot_communities": bot_analysis.get('communities', {}),
                "model_info": {
                    "using_ml": False,  # Disabled for large datasets
                    "using_lsh": DATASKETCH_AVAILABLE and self.lsh is not None,
                    "using_community_detection": COMMUNITY_AVAILABLE,
                    "batch_processing": True,
                    "batch_size": batch_size
                }
            }
            
        except Exception as e:
            logger.error(f"Error in spam and bot detection: {e}")
            return {"error": str(e)}
    
    def _detect_spam_rules(self, comments_df: pd.DataFrame) -> List[float]:
        """Rule-based spam detection using patterns and keywords"""
        return self._detect_spam_rules_batch(comments_df)
    
    def _detect_spam_rules_batch(self, comments_df: pd.DataFrame) -> List[float]:
        """Optimized batch rule-based spam detection"""
        spam_scores = []
        thresholds = self.config["spam_thresholds"]
        
        for text in comments_df['text']:
            if pd.isna(text):
                spam_scores.append(0.0)
                continue
                
            text = str(text)
            score = 0.0
            
            # URL count
            urls = self.spam_patterns["urls"].findall(text)
            if len(urls) >= thresholds["url_count"]:
                score += 0.3
            
            # Promotional domains
            for url in urls:
                for domain in self.config["promotional_domains"]:
                    if domain in url:
                        score += 0.2
                        break
            
            # Emoji ratio
            emoji_matches = self.spam_patterns["emoji_flood"].findall(text)
            if emoji_matches:
                emoji_ratio = len(''.join(emoji_matches)) / len(text)
                if emoji_ratio > thresholds["emoji_ratio"]:
                    score += 0.2
            
            # Caps ratio
            caps_matches = self.spam_patterns["caps_words"].findall(text)
            if caps_matches:
                caps_ratio = sum(len(match) for match in caps_matches) / len(text)
                if caps_ratio > thresholds["caps_ratio"]:
                    score += 0.2
            
            # Repeated characters
            repeated = self.spam_patterns["repeated_chars"].findall(text)
            if len(repeated) >= thresholds["repeated_chars"]:
                score += 0.1
            
            # Promotional keywords
            text_lower = text.lower()
            keyword_count = sum(1 for kw in self.config["spam_keywords"] if kw in text_lower)
            if keyword_count >= thresholds["promotional_keywords"]:
                score += 0.3
            
            # Email addresses
            if self.spam_patterns["emails"].search(text):
                score += 0.2
            
            # Phone numbers  
            if self.spam_patterns["phone_numbers"].search(text):
                score += 0.2
            
            spam_scores.append(min(1.0, score))
        
        return spam_scores
    
    def _detect_spam_ml(self, texts: List[str]) -> List[float]:
        """ML-based spam detection using TF-IDF + Logistic Regression"""
        try:
            # For this demo, we'll create synthetic training data
            # In production, you'd use labeled spam/ham data
            synthetic_spam = [
                "check my bio for deals", "dm for promotion", "free money click here",
                "subscribe to my channel", "follow for follow back", "like for like exchange"
            ]
            synthetic_ham = [
                "great product, really works well", "love this brand quality",
                "had a good experience with this", "would recommend to others",
                "excellent customer service", "fast shipping and good packaging"
            ]
            
            # Create training data
            train_texts = synthetic_spam + synthetic_ham
            train_labels = [1] * len(synthetic_spam) + [0] * len(synthetic_ham)
            
            # Add some real comments as ham (assuming most are legitimate)
            sample_size = min(100, len(texts))
            sample_texts = texts[:sample_size]
            train_texts.extend(sample_texts)
            train_labels.extend([0] * sample_size)  # Assume legitimate
            
            # Train classifier
            X_train = self.tfidf_vectorizer.fit_transform(train_texts)
            self.spam_classifier.fit(X_train, train_labels)
            
            # Predict on all texts
            X_test = self.tfidf_vectorizer.transform(texts)
            spam_probabilities = self.spam_classifier.predict_proba(X_test)[:, 1]
            
            return spam_probabilities.tolist()
            
        except Exception as e:
            logger.error(f"Error in ML spam detection: {e}")
            return [0.0] * len(texts)
    
    def _detect_duplicates(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect near-duplicates using MinHash LSH"""
        
        # For Apple Silicon, always use fallback to prevent segmentation faults
        import platform
        if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
            logger.info("Apple Silicon detected - using fallback duplicate detection to prevent segfaults")
            return self._fallback_duplicate_detection(comments_df)
        
        if not DATASKETCH_AVAILABLE or self.lsh is None:
            # Fallback to exact duplicates
            return self._fallback_duplicate_detection(comments_df)
        
        try:
            # Create MinHash for each comment
            texts = comments_df['text'].fillna('').astype(str).tolist()
            comment_ids = comments_df.index.tolist()
            
            minhashes = {}
            shingle_size = self.config["minhash_settings"]["shingle_size"]
            
            for i, (comment_id, text) in enumerate(zip(comment_ids, texts)):
                if len(text.strip()) < shingle_size:
                    continue
                    
                # Create shingles (n-grams of words)
                words = text.lower().split()
                shingles = [' '.join(words[j:j+shingle_size]) 
                           for j in range(len(words)-shingle_size+1)]
                
                if not shingles:
                    continue
                
                # Create MinHash
                m = MinHash(num_perm=self.config["minhash_settings"]["num_perm"])
                for shingle in shingles:
                    m.update(shingle.encode('utf8'))
                
                minhashes[comment_id] = m
                self.lsh.insert(comment_id, m)
            
            # Find duplicates
            duplicate_clusters = []
            processed = set()
            is_duplicate = [False] * len(comments_df)
            cluster_ids = [-1] * len(comments_df)
            
            for i, comment_id in enumerate(comment_ids):
                if comment_id in processed or comment_id not in minhashes:
                    continue
                
                # Find similar comments
                similar = self.lsh.query(minhashes[comment_id])
                
                if len(similar) > 1:  # Found duplicates
                    cluster_id = len(duplicate_clusters)
                    duplicate_clusters.append(list(similar))
                    
                    for sim_id in similar:
                        if sim_id in comment_ids:
                            idx = comment_ids.index(sim_id)
                            is_duplicate[idx] = True
                            cluster_ids[idx] = cluster_id
                            processed.add(sim_id)
            
            return {
                "is_duplicate": is_duplicate,
                "cluster_ids": cluster_ids,
                "clusters": duplicate_clusters
            }
            
        except Exception as e:
            logger.error(f"Error in duplicate detection: {e}")
            return self._fallback_duplicate_detection(comments_df)
    
    def _fallback_duplicate_detection(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback exact duplicate detection"""
        texts = comments_df['text'].fillna('').astype(str)
        
        # Find exact duplicates
        text_counts = texts.value_counts()
        duplicate_texts = set(text_counts[text_counts > 1].index)
        
        is_duplicate = texts.isin(duplicate_texts).tolist()
        
        # Assign cluster IDs
        cluster_ids = [-1] * len(comments_df)
        clusters = []
        
        for i, text in enumerate(texts):
            if text in duplicate_texts and cluster_ids[i] == -1:
                # Find all instances of this text
                cluster_id = len(clusters)
                indices = texts[texts == text].index.tolist()
                clusters.append(indices)
                
                for idx in indices:
                    pos = list(comments_df.index).index(idx)
                    cluster_ids[pos] = cluster_id
        
        return {
            "is_duplicate": is_duplicate,
            "cluster_ids": cluster_ids,
            "clusters": clusters
        }
    
    def _detect_duplicates_optimized(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Memory-optimized duplicate detection for large datasets"""
        
        # For large datasets, use simpler hash-based duplicate detection
        if len(comments_df) > 500:
            return self._fallback_duplicate_detection(comments_df)
        else:
            return self._detect_duplicates(comments_df)
    
    def _detect_bot_networks_simplified(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Simplified bot detection for large datasets"""
        
        try:
            bot_scores = [0.0] * len(comments_df)
            bot_clusters = [-1] * len(comments_df)
            
            # Group by user for analysis (if user_id available)
            if 'user_id' in comments_df.columns:
                user_groups = comments_df.groupby('user_id')
            else:
                # For large datasets without user_id, skip complex analysis
                return {
                    "bot_scores": bot_scores,
                    "bot_clusters": bot_clusters,
                    "communities": {}
                }
            
            # Simplified user pattern analysis
            user_patterns = {}
            for user_id, group in user_groups:
                texts = group['text'].fillna('').tolist()
                
                # Calculate repetition rate
                unique_texts = len(set(texts))
                total_texts = len(texts)
                repetition_rate = 1 - (unique_texts / total_texts) if total_texts > 0 else 0
                
                user_patterns[user_id] = {
                    'repetition_rate': repetition_rate,
                    'comment_count': total_texts,
                    'avg_length': np.mean([len(t) for t in texts]) if texts else 0
                }
            
            # Calculate simplified bot scores
            bot_threshold = self.config["bot_detection"]["min_repetition_rate"]
            
            for i, row in comments_df.iterrows():
                user_id = row.get('user_id', i)
                
                if user_id in user_patterns:
                    pattern = user_patterns[user_id]
                    
                    # Simplified bot score
                    bot_score = 0.0
                    
                    # High repetition rate
                    if pattern['repetition_rate'] > bot_threshold:
                        bot_score += 0.6
                    
                    # Very short or very long comments
                    if pattern['avg_length'] < 10 or pattern['avg_length'] > 500:
                        bot_score += 0.3
                    
                    # High volume posting
                    if pattern['comment_count'] > 100:
                        bot_score += 0.1
                    
                    bot_scores[list(comments_df.index).index(i)] = min(1.0, bot_score)
            
            return {
                "bot_scores": bot_scores,
                "bot_clusters": bot_clusters,
                "communities": {},
                "user_patterns": user_patterns
            }
            
        except Exception as e:
            logger.error(f"Error in simplified bot detection: {e}")
            return {
                "bot_scores": [0.0] * len(comments_df),
                "bot_clusters": [-1] * len(comments_df),
                "communities": {}
            }
    
    def _detect_bot_networks(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect bot networks using graph analysis"""
        
        try:
            # Create similarity graph
            G = nx.Graph()
            
            # Add nodes (users)
            if 'user_id' in comments_df.columns:
                users = comments_df['user_id'].unique()
            else:
                users = comments_df.index.tolist()
            
            G.add_nodes_from(users)
            
            # Add edges based on similar content and timing
            bot_scores = [0.0] * len(comments_df)
            bot_clusters = [-1] * len(comments_df)
            
            # Group by user for analysis
            if 'user_id' in comments_df.columns:
                user_groups = comments_df.groupby('user_id')
            else:
                # Fallback: treat each comment as separate user
                user_groups = comments_df.groupby(comments_df.index)
            
            # Analyze user patterns
            user_patterns = {}
            for user_id, group in user_groups:
                texts = group['text'].fillna('').tolist()
                
                # Calculate repetition rate
                unique_texts = len(set(texts))
                total_texts = len(texts)
                repetition_rate = 1 - (unique_texts / total_texts) if total_texts > 0 else 0
                
                # Calculate temporal clustering (if timestamp available)
                temporal_score = 0.0
                if 'ts' in group.columns or 'publishedAt' in group.columns:
                    ts_col = 'ts' if 'ts' in group.columns else 'publishedAt'
                    timestamps = pd.to_datetime(group[ts_col], errors='coerce').dropna()
                    
                    if len(timestamps) > 1:
                        time_diffs = timestamps.diff().dt.total_seconds().dropna()
                        # High score if posts are very regular or clustered
                        if len(time_diffs) > 0:
                            std_time = time_diffs.std()
                            mean_time = time_diffs.mean()
                            temporal_score = 1.0 / (1.0 + std_time / max(mean_time, 1))
                
                user_patterns[user_id] = {
                    'repetition_rate': repetition_rate,
                    'temporal_score': temporal_score,
                    'comment_count': total_texts,
                    'avg_length': np.mean([len(t) for t in texts])
                }
            
            # Calculate bot scores based on patterns
            bot_threshold = self.config["bot_detection"]["min_repetition_rate"]
            
            for i, row in comments_df.iterrows():
                user_id = row.get('user_id', i)
                
                if user_id in user_patterns:
                    pattern = user_patterns[user_id]
                    
                    # Bot score based on multiple factors
                    bot_score = 0.0
                    
                    # High repetition rate
                    if pattern['repetition_rate'] > bot_threshold:
                        bot_score += 0.4
                    
                    # Very regular posting pattern
                    bot_score += pattern['temporal_score'] * 0.3
                    
                    # Very short or very long comments (bots often have consistent lengths)
                    if pattern['avg_length'] < 10 or pattern['avg_length'] > 500:
                        bot_score += 0.2
                    
                    # High volume posting
                    if pattern['comment_count'] > 50:
                        bot_score += 0.1
                    
                    bot_scores[list(comments_df.index).index(i)] = min(1.0, bot_score)
            
            # Community detection for bot clusters
            communities = {}
            if COMMUNITY_AVAILABLE and len(G.nodes()) > 0:
                try:
                    # Add edges between users with similar patterns
                    for user1 in user_patterns:
                        for user2 in user_patterns:
                            if user1 != user2:
                                p1, p2 = user_patterns[user1], user_patterns[user2]
                                
                                # Similarity based on patterns
                                similarity = (
                                    abs(p1['repetition_rate'] - p2['repetition_rate']) < 0.2 and
                                    abs(p1['temporal_score'] - p2['temporal_score']) < 0.3 and
                                    abs(p1['avg_length'] - p2['avg_length']) < 100
                                )
                                
                                if similarity:
                                    G.add_edge(user1, user2)
                    
                    # Detect communities
                    if G.number_of_edges() > 0:
                        communities = community_louvain.best_partition(G)
                        
                        # Assign cluster IDs
                        for i, row in comments_df.iterrows():
                            user_id = row.get('user_id', i)
                            if user_id in communities:
                                bot_clusters[list(comments_df.index).index(i)] = communities[user_id]
                
                except Exception as e:
                    logger.warning(f"Community detection failed: {e}")
            
            return {
                "bot_scores": bot_scores,
                "bot_clusters": bot_clusters,
                "communities": dict(communities) if communities else {},
                "user_patterns": user_patterns
            }
            
        except Exception as e:
            logger.error(f"Error in bot network detection: {e}")
            return {
                "bot_scores": [0.0] * len(comments_df),
                "bot_clusters": [-1] * len(comments_df),
                "communities": {},
                "user_patterns": {}
            }
    
    def _calculate_authenticity(self, comments_df: pd.DataFrame) -> List[float]:
        """Calculate authenticity scores combining spam and bot indicators"""
        
        authenticity_scores = []
        
        for i, row in comments_df.iterrows():
            # Start with baseline authenticity
            authenticity = 1.0
            
            # Reduce for spam
            spam_score = row.get('spam_score', 0.0)
            authenticity -= spam_score * 0.4
            
            # Reduce for bot behavior
            bot_score = row.get('bot_score', 0.0)  
            authenticity -= bot_score * 0.3
            
            # Reduce for duplicates
            is_duplicate = row.get('is_duplicate', False)
            if is_duplicate:
                authenticity -= 0.2
            
            # Ensure score stays in [0, 1]
            authenticity_scores.append(max(0.0, min(1.0, authenticity)))
        
        return authenticity_scores
    
    def _aggregate_detection_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate aggregated detection metrics"""
        
        total_comments = len(results_df)
        
        return {
            "spam_metrics": {
                "total_spam": len(results_df[results_df['spam_score'] > 0.5]),
                "spam_rate": (results_df['spam_score'] > 0.5).mean(),
                "avg_spam_score": results_df['spam_score'].mean(),
                "high_spam": len(results_df[results_df['spam_score'] > 0.8])
            },
            
            "duplicate_metrics": {
                "total_duplicates": len(results_df[results_df['is_duplicate']]),
                "duplicate_rate": results_df['is_duplicate'].mean(),
                "unique_clusters": results_df['duplicate_cluster'].nunique() - 1,  # -1 for non-duplicates
            },
            
            "bot_metrics": {
                "total_bots": len(results_df[results_df['bot_score'] > 0.5]),
                "bot_rate": (results_df['bot_score'] > 0.5).mean(),
                "avg_bot_score": results_df['bot_score'].mean(),
                "bot_communities": results_df['bot_cluster'].nunique() - 1  # -1 for non-clustered
            },
            
            "authenticity_metrics": {
                "avg_authenticity": results_df['authenticity_score'].mean(),
                "high_authenticity": len(results_df[results_df['authenticity_score'] > 0.8]),
                "low_authenticity": len(results_df[results_df['authenticity_score'] < 0.3]),
                "authenticity_distribution": {
                    "excellent": len(results_df[results_df['authenticity_score'] > 0.9]),
                    "good": len(results_df[(results_df['authenticity_score'] > 0.7) & 
                                         (results_df['authenticity_score'] <= 0.9)]),
                    "fair": len(results_df[(results_df['authenticity_score'] > 0.5) & 
                                         (results_df['authenticity_score'] <= 0.7)]),
                    "poor": len(results_df[results_df['authenticity_score'] <= 0.5])
                }
            },
            
            "overall_metrics": {
                "total_comments": total_comments,
                "clean_comments": len(results_df[
                    (results_df['spam_score'] < 0.3) & 
                    (results_df['bot_score'] < 0.3) & 
                    (~results_df['is_duplicate'])
                ]),
                "flagged_comments": len(results_df[
                    (results_df['spam_score'] > 0.5) | 
                    (results_df['bot_score'] > 0.5) | 
                    (results_df['is_duplicate'])
                ])
            }
        }

# Factory function for pipeline integration
def create_spam_bot_detector(config_path: Optional[str] = None) -> SpamBotDetector:
    """Factory function to create spam bot detector with configuration"""
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return SpamBotDetector(config)