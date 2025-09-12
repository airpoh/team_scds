"""
Predictive Analytics Module for CommentSense Pipeline

This module implements:
1. Fake Engagement Insurance: Authenticity score flagging suspicious SoE bumps
2. Predictive SoE Impact: Forecast next-week SoE from QEI trends + influencer data
3. Advanced anomaly detection using Isolation Forest
4. Time series forecasting with XGBoost/LightGBM + SHAP explanations

Features:
- Isolation Forest for engagement anomaly detection
- XGBoost/LightGBM for SoE prediction with SHAP explainability
- Temporal pattern analysis for fake engagement detection
- Feature engineering from QEI, categories, influencer metrics
"""

import logging
import warnings
import os
import platform
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import atexit
import gc

# Disable meta device and MPS for Apple Silicon to prevent memory leaks and segmentation faults
if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
    os.environ['PYTORCH_DISABLE_META_DEVICE'] = '1'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    
# Suppress warnings including multiprocessing resource warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# Core ML libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logging.warning("XGBoost not available")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logging.warning("LightGBM not available")
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logging.warning("SHAP not available - no model explanations")
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

class PredictiveAnalyzer:
    """
    Advanced predictive analytics for engagement and authenticity
    
    Features:
    - Fake Engagement Insurance: Detect suspicious engagement spikes
    - Predictive SoE Impact: Forecast engagement metrics from quality indicators
    - Isolation Forest anomaly detection for authenticity scoring
    - XGBoost/LightGBM with SHAP explanations for interpretability
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Predictive Analyzer"""
        
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
        
        # Models
        self.anomaly_detector = None
        self.soe_predictor = None
        self.feature_scaler = None
        self.label_encoders = {}
        
        # SHAP explainer
        self.shap_explainer = None
        
        # Feature engineering components
        self.feature_columns = []
        self.temporal_features = []
        
        # Device management for memory leak prevention
        self.system_info = self._get_system_info()
        self.primary_device, self.fallback_device = self._setup_devices()
        
        # Initialize ML components
        if SKLEARN_AVAILABLE:
            self._initialize_anomaly_detector()
        
        # Register cleanup function for proper resource management
        atexit.register(self._cleanup_resources)
        
        logger.info(f"Predictive Analytics module initialized on {self.primary_device}")
        logger.info(f"System info: {self.system_info}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Anomaly detection settings
            "anomaly_detection": {
                "contamination": 0.1,  # Expected fraction of anomalies
                "n_estimators": 100,
                "random_state": 42,
                "bootstrap": False
            },
            
            # SoE prediction settings
            "soe_prediction": {
                "model_type": "xgboost",  # xgboost, lightgbm, or sklearn
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "test_size": 0.2
            },
            
            # Feature engineering
            "feature_engineering": {
                "temporal_windows": [1, 3, 7, 14],  # days for rolling features
                "lag_features": [1, 2, 3, 7],  # lag periods for time series
                "aggregation_functions": ["mean", "std", "min", "max", "median"]
            },
            
            # Thresholds
            "thresholds": {
                "fake_engagement": -0.5,  # Isolation forest threshold
                "high_risk_authenticity": 0.3,
                "suspicious_spike_factor": 3.0,  # 3x normal engagement
                "min_data_points": 10  # Minimum data for reliable predictions
            },
            
            # SHAP settings
            "shap_settings": {
                "max_samples": 1000,  # For SHAP explanations
                "feature_importance_top_k": 10
            }
        }
    
    def _initialize_anomaly_detector(self):
        """Initialize Isolation Forest for anomaly detection"""
        try:
            settings = self.config["anomaly_detection"]
            self.anomaly_detector = IsolationForest(
                contamination=settings["contamination"],
                n_estimators=settings["n_estimators"],
                random_state=settings["random_state"],
                bootstrap=settings["bootstrap"]
            )
            
            self.feature_scaler = StandardScaler()
            
            logger.info("Anomaly detector initialized")
            
        except Exception as e:
            logger.error(f"Error initializing anomaly detector: {e}")
    
    def _get_system_info(self) -> dict:
        """Get system information for device selection"""
        try:
            import torch
            return {
                'platform': platform.system(),
                'machine': platform.machine(),
                'is_apple_silicon': platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin',
                'cuda_available': torch.cuda.is_available() if hasattr(torch, 'cuda') else False,
                'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            }
        except ImportError:
            return {
                'platform': platform.system(),
                'machine': platform.machine(),
                'is_apple_silicon': platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin',
                'cuda_available': False,
                'mps_available': False,
            }
    
    def _setup_devices(self) -> tuple:
        """Setup optimal devices for predictive analytics (CPU-optimized for small datasets)"""
        try:
            import torch
            
            # Predictive analytics uses XGBoost/LightGBM which are CPU-optimized for small datasets
            # Force CPU on all platforms for optimal performance
            fallback_device = torch.device("cpu")
            logger.info("Using CPU for predictive analytics (XGBoost/LightGBM are CPU-optimized for small datasets)")
            return fallback_device, fallback_device
            
        except ImportError:
            logger.warning("PyTorch not available, using CPU-only mode")
            return "cpu", "cpu"
    
    def _clear_memory_cache(self):
        """Clear device memory cache and run garbage collection"""
        try:
            import torch
            
            if hasattr(self, 'primary_device') and hasattr(self.primary_device, 'type'):
                if self.primary_device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif self.primary_device.type == "mps":
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
            
            # Always run garbage collection
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error clearing memory cache: {e}")
    
    def analyze_engagement_patterns(self, comments_df: pd.DataFrame, 
                                   quality_results: Optional[Dict[str, Any]] = None,
                                   spam_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main method for predictive analytics
        
        Args:
            comments_df: DataFrame with comment data
            quality_results: Results from quality scoring module  
            spam_results: Results from spam/bot detection module
            
        Returns:
            Dictionary containing predictive analytics results
        """
        
        try:
            logger.info(f"Analyzing engagement patterns for {len(comments_df)} comments...")
            
            # Clear memory cache before processing
            self._clear_memory_cache()
            
            # Limit data size to prevent memory issues and segmentation faults
            max_comments = 5000  # Conservative limit for Apple Silicon
            if len(comments_df) > max_comments:
                logger.info(f"Limiting analysis to {max_comments} comments to prevent memory issues")
                comments_df = comments_df.head(max_comments)
            
            # Prepare features for analysis
            features_df = self._prepare_features(comments_df, quality_results, spam_results)
            
            if len(features_df) < self.config["thresholds"]["min_data_points"]:
                logger.warning("Insufficient data for reliable predictions")
                return {"error": "Insufficient data for predictions"}
            
            # Clear memory after feature preparation
            self._clear_memory_cache()
            
            results = {}
            
            # Step 1: Fake Engagement Insurance (Anomaly Detection)
            if SKLEARN_AVAILABLE and self.anomaly_detector is not None:
                authenticity_analysis = self._detect_fake_engagement(features_df)
                results["fake_engagement_insurance"] = authenticity_analysis
            
            # Step 2: Predictive SoE Impact (Forecasting)
            if XGBOOST_AVAILABLE or LIGHTGBM_AVAILABLE:
                soe_predictions = self._predict_soe_impact(features_df)
                results["soe_predictions"] = soe_predictions
            
            # Step 3: Risk Assessment
            risk_analysis = self._assess_engagement_risks(features_df, results)
            results["risk_assessment"] = risk_analysis
            
            # Step 4: Recommendations
            recommendations = self._generate_recommendations(results)
            results["recommendations"] = recommendations
            
            logger.info("Predictive analytics completed successfully")
            
            # Final memory cleanup
            self._clear_memory_cache()
            
            return {
                "status": "success",
                "analytics_results": results,
                "feature_summary": self._summarize_features(features_df),
                "model_info": {
                    "anomaly_detector": SKLEARN_AVAILABLE and self.anomaly_detector is not None,
                    "soe_predictor": XGBOOST_AVAILABLE or LIGHTGBM_AVAILABLE,
                    "shap_available": SHAP_AVAILABLE,
                    "features_engineered": len(self.feature_columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in predictive analytics: {e}")
            # Clear memory even on error
            self._clear_memory_cache()
            return {"error": str(e)}
    
    def _prepare_features(self, comments_df: pd.DataFrame,
                         quality_results: Optional[Dict[str, Any]] = None,
                         spam_results: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Prepare features for predictive modeling"""
        
        try:
            features_df = comments_df.copy()
            
            # Basic engagement metrics
            if 'likes' not in features_df.columns:
                if 'likeCount' in features_df.columns:
                    features_df['likes'] = features_df['likeCount']
                else:
                    features_df['likes'] = 0
            
            features_df['likes'] = pd.to_numeric(features_df['likes'], errors='coerce').fillna(0)
            
            # Text features
            if 'text' not in features_df.columns and 'textOriginal' in features_df.columns:
                features_df['text'] = features_df['textOriginal']
            
            features_df['text_length'] = features_df['text'].fillna('').astype(str).str.len()
            features_df['word_count'] = features_df['text'].fillna('').astype(str).str.split().str.len()
            
            # Quality features (from quality scoring module)
            if quality_results and quality_results.get('status') == 'success':
                if 'scored_comments' in quality_results:
                    # Direct DataFrame access
                    quality_df = quality_results['scored_comments']
                    if hasattr(quality_df, 'columns'):
                        for col in ['qei_score', 'relevance', 'informativeness', 'constructiveness']:
                            if col in quality_df.columns:
                                features_df[col] = quality_df[col]
                    else:
                        # Handle string representation case - use aggregated metrics
                        self._apply_aggregated_quality_features(features_df, quality_results)
                elif 'aggregated_metrics' in quality_results:
                    # Use aggregated metrics when DataFrame not available
                    self._apply_aggregated_quality_features(features_df, quality_results)
                else:
                    # Fallback quality features
                    features_df['qei_score'] = 50.0  # Neutral quality
                    features_df['relevance'] = 0.5
                    features_df['informativeness'] = 0.5
                    features_df['constructiveness'] = 0.5
            else:
                # Check if we have aggregated quality metrics
                if quality_results and 'aggregated_metrics' in quality_results:
                    features_df = self._apply_aggregated_quality_features(features_df, quality_results['aggregated_metrics'])
                else:
                    # Fallback quality features
                    features_df['qei_score'] = 50.0  # Neutral quality
                    features_df['relevance'] = 0.5
                    features_df['informativeness'] = 0.5
                    features_df['constructiveness'] = 0.5
            
            # Authenticity features (from spam/bot detection)
            if spam_results and 'analyzed_comments' in spam_results:
                spam_df = spam_results['analyzed_comments']
                if hasattr(spam_df, 'columns'):  # It's a DataFrame
                    for col in ['spam_score', 'bot_score', 'authenticity_score']:
                        if col in spam_df.columns:
                            features_df[col] = spam_df[col]
                else:
                    # Handle when analyzed_comments is stored as string representation
                    # Use aggregated metrics if available
                    if 'aggregated_metrics' in spam_results:
                        spam_metrics = spam_results['aggregated_metrics']
                        features_df['spam_score'] = spam_metrics.get('spam_ratio', 0.1)
                        features_df['bot_score'] = spam_metrics.get('bot_ratio', 0.1)
                        features_df['authenticity_score'] = 1.0 - spam_metrics.get('spam_ratio', 0.1)
                    else:
                        # Fallback
                        features_df['spam_score'] = 0.1
                        features_df['bot_score'] = 0.1
                        features_df['authenticity_score'] = 0.8
            else:
                # Fallback authenticity features
                features_df['spam_score'] = 0.1
                features_df['bot_score'] = 0.1
                features_df['authenticity_score'] = 0.8
            
            # Temporal features
            if 'ts' in features_df.columns or 'publishedAt' in features_df.columns:
                ts_col = 'ts' if 'ts' in features_df.columns else 'publishedAt'
                # Try multiple datetime formats
                try:
                    features_df['timestamp'] = pd.to_datetime(features_df[ts_col], errors='coerce', infer_datetime_format=True)
                    # If that fails, try common formats
                    if features_df['timestamp'].isna().all():
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                            try:
                                features_df['timestamp'] = pd.to_datetime(features_df[ts_col], format=fmt, errors='coerce')
                                if not features_df['timestamp'].isna().all():
                                    break
                            except:
                                continue
                except Exception as e:
                    logger.warning(f"Could not parse timestamps from {ts_col}: {e}")
                    features_df['timestamp'] = pd.NaT
                
                # Extract temporal components
                features_df['hour'] = features_df['timestamp'].dt.hour
                features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
                features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6])
                
                # Time since start of dataset
                min_time = features_df['timestamp'].min()
                features_df['days_since_start'] = (features_df['timestamp'] - min_time).dt.total_seconds() / 86400
                
                # Rolling features (if enough data)
                if len(features_df) > 50 and not features_df['timestamp'].isna().all():
                    features_df = features_df.sort_values('timestamp').dropna(subset=['timestamp'])
                    for window in self.config["feature_engineering"]["temporal_windows"]:
                        if len(features_df) > window:
                            try:
                                # Use simple numeric rolling instead of time-based rolling
                                if 'likes' in features_df.columns:
                                    features_df[f'likes_rolling_{window}d'] = features_df['likes'].rolling(
                                        window=window, min_periods=1).mean()
                                
                                # Only create qei_score rolling if column exists and has valid data
                                if 'qei_score' in features_df.columns and not features_df['qei_score'].isna().all():
                                    features_df[f'qei_rolling_{window}d'] = features_df['qei_score'].rolling(
                                        window=window, min_periods=1).mean()
                                else:
                                    # Create a fallback neutral rolling score
                                    features_df[f'qei_rolling_{window}d'] = 50.0
                                    
                            except Exception as e:
                                logger.warning(f"Could not create rolling features for window {window}: {e}")
                                continue
            
            # User features (if available)
            if 'user_id' in features_df.columns:
                try:
                    # Build aggregation dict based on available columns
                    agg_dict = {}
                    if 'likes' in features_df.columns:
                        agg_dict['likes'] = ['count', 'mean', 'std']
                    if 'qei_score' in features_df.columns:
                        agg_dict['qei_score'] = 'mean'
                    if 'authenticity_score' in features_df.columns:
                        agg_dict['authenticity_score'] = 'mean'
                    
                    if agg_dict:  # Only proceed if we have columns to aggregate
                        user_stats = features_df.groupby('user_id').agg(agg_dict).round(3)
                        
                        # Create appropriate column names based on what was aggregated
                        new_cols = []
                        if 'likes' in agg_dict:
                            new_cols.extend(['user_comment_count', 'user_avg_likes', 'user_likes_std'])
                        if 'qei_score' in agg_dict:
                            new_cols.append('user_avg_qei')
                        if 'authenticity_score' in agg_dict:
                            new_cols.append('user_avg_authenticity')
                        
                        user_stats.columns = new_cols
                        features_df = features_df.merge(user_stats, left_on='user_id', right_index=True, how='left')
                except Exception as e:
                    logger.warning(f"Could not create user features: {e}")
                    pass
            
            # Category features (if available from network analysis)
            if 'category' in features_df.columns:
                # One-hot encode categories
                category_dummies = pd.get_dummies(features_df['category'], prefix='category')
                features_df = pd.concat([features_df, category_dummies], axis=1)
            
            # Fill missing values
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            features_df[numeric_columns] = features_df[numeric_columns].fillna(
                features_df[numeric_columns].median()
            )
            
            # Store feature columns for later use
            self.feature_columns = [col for col in features_df.columns 
                                  if col not in ['text', 'timestamp', 'user_id', 'comment_id', 'post_id']]
            
            logger.info(f"Engineered {len(self.feature_columns)} features for predictive modeling")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return comments_df
    
    def _apply_aggregated_quality_features(self, features_df: pd.DataFrame, aggregated_metrics: Dict[str, Any]) -> pd.DataFrame:
        """Apply aggregated quality metrics to the features dataframe"""
        try:
            # Extract aggregated quality metrics and apply them uniformly
            qei_score = aggregated_metrics.get('average_qei_score', 50.0)
            relevance = aggregated_metrics.get('average_relevance', 0.5)
            informativeness = aggregated_metrics.get('average_informativeness', 0.5)
            constructiveness = aggregated_metrics.get('average_constructiveness', 0.5)
            
            # Apply these values to all rows
            features_df['qei_score'] = qei_score
            features_df['relevance'] = relevance
            features_df['informativeness'] = informativeness
            features_df['constructiveness'] = constructiveness
            
            logger.info(f"Applied aggregated quality features - QEI: {qei_score:.2f}, Relevance: {relevance:.3f}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error applying aggregated quality features: {e}")
            # Fallback to neutral values
            features_df['qei_score'] = 50.0
            features_df['relevance'] = 0.5
            features_df['informativeness'] = 0.5
            features_df['constructiveness'] = 0.5
            return features_df
    
    def _detect_fake_engagement(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect fake engagement using Isolation Forest with Apple Silicon safety guards"""
        
        try:
            # Apple Silicon stability check - disable Isolation Forest completely 
            if self.system_info.get('is_apple_silicon', False):
                logger.info("Apple Silicon detected - using fallback fake engagement detection to prevent segmentation faults")
                return self._fallback_fake_engagement_detection(features_df)
            
            # For non-Apple Silicon systems, proceed with ML-based detection
            # Very aggressive safety check for stability
            max_samples = 100  # Much smaller limit for stability
            if len(features_df) > max_samples:
                logger.warning(f"Large dataset ({len(features_df)} samples), sampling {max_samples} for anomaly detection")
                features_sample = features_df.sample(n=max_samples, random_state=42)
            else:
                features_sample = features_df
            
            # Additional safety: require minimum features
            if len(features_sample) < 10:
                logger.warning("Too few samples for reliable anomaly detection, using fallback")
                return self._fallback_fake_engagement_detection(features_df)
            
            # Select features for anomaly detection
            anomaly_features = [
                'likes', 'qei_score', 'spam_score', 'bot_score', 'authenticity_score',
                'text_length', 'word_count'
            ]
            
            # Add temporal features if available
            temporal_cols = [col for col in features_sample.columns if 'rolling' in col or 'user_' in col]
            anomaly_features.extend(temporal_cols[:5])  # Limit to avoid overfitting
            
            # Filter features that exist
            anomaly_features = [col for col in anomaly_features if col in features_sample.columns]
            
            if len(anomaly_features) == 0:
                logger.warning("No valid features found for anomaly detection")
                return self._fallback_fake_engagement_detection(features_df)
            
            X = features_sample[anomaly_features].copy()
            
            # Clean data: handle missing values and infinities
            X = X.fillna(0)  # Fill NaN with 0
            X = X.replace([np.inf, -np.inf], 0)  # Replace infinities with 0
            
            # Additional safety check for valid numeric data
            if not np.isfinite(X.values).all():
                logger.warning("Non-finite values detected after cleaning, using fallback")
                return self._fallback_fake_engagement_detection(features_df)
            
            # Scale features with safety checks
            try:
                X_scaled = self.feature_scaler.fit_transform(X)
            except Exception as scale_error:
                logger.warning(f"Feature scaling failed: {scale_error}, using fallback")
                return self._fallback_fake_engagement_detection(features_df)
            
            # Detect anomalies with extra safety for non-Apple Silicon systems
            try:
                # Force very conservative settings for Isolation Forest
                # Use a smaller n_estimators to reduce memory pressure
                conservative_detector = IsolationForest(
                    contamination=0.1,
                    n_estimators=10,  # Much smaller than default 100
                    max_samples=min(50, len(X_scaled)),  # Very small max samples
                    random_state=42,
                    bootstrap=False,
                    n_jobs=1  # Single thread to avoid multiprocessing issues
                )
                
                anomaly_labels = conservative_detector.fit_predict(X_scaled)
                anomaly_scores = conservative_detector.decision_function(X_scaled)
                
            except Exception as anomaly_error:
                logger.warning(f"Anomaly detection failed even with conservative settings: {anomaly_error}")
                logger.warning("Disabling Isolation Forest due to stability issues, using fallback")
                return self._fallback_fake_engagement_detection(features_df)
            
            # Convert to authenticity scores (higher = more authentic)
            fake_engagement_scores = 1 / (1 + np.exp(-anomaly_scores))  # Sigmoid normalization
            
            # Identify suspicious spikes
            threshold = self.config["thresholds"]["fake_engagement"]
            suspicious_indices = np.where(anomaly_scores < threshold)[0]
            
            # Analyze engagement spikes
            spike_analysis = self._analyze_engagement_spikes(features_df)
            
            return {
                "fake_engagement_scores": fake_engagement_scores.tolist(),
                "anomaly_labels": (anomaly_labels == -1).tolist(),  # True = anomaly
                "suspicious_count": len(suspicious_indices),
                "suspicious_rate": len(suspicious_indices) / len(features_df),
                "spike_analysis": spike_analysis,
                "features_used": anomaly_features,
                "threshold_used": threshold,
                "method": "isolation_forest"
            }
            
        except Exception as e:
            logger.error(f"Error in fake engagement detection: {e}")
            return {"error": str(e)}
    
    def _fallback_fake_engagement_detection(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Simple fallback fake engagement detection without ML"""
        
        try:
            logger.info("Using fallback fake engagement detection")
            
            # Simple rule-based detection
            total_comments = len(features_df)
            
            if total_comments == 0:
                return {"error": "No data for analysis"}
            
            # Basic metrics using available columns
            high_risk_count = 0
            authenticity_scores = []
            
            for _, row in features_df.iterrows():
                # Simple authenticity score based on available metrics
                authenticity = 0.7  # Base authenticity
                
                # Adjust based on spam score if available
                if 'spam_score' in row and not pd.isna(row['spam_score']):
                    authenticity -= row['spam_score'] * 0.3
                
                # Adjust based on bot score if available 
                if 'bot_score' in row and not pd.isna(row['bot_score']):
                    authenticity -= row['bot_score'] * 0.2
                
                # Adjust based on text quality
                if 'text_length' in row and not pd.isna(row['text_length']):
                    if row['text_length'] < 10:  # Very short comments
                        authenticity -= 0.1
                
                authenticity = max(0.0, min(1.0, authenticity))  # Clamp to [0,1]
                authenticity_scores.append(authenticity)
                
                if authenticity < 0.3:
                    high_risk_count += 1
            
            # Calculate summary metrics
            avg_authenticity = np.mean(authenticity_scores) if authenticity_scores else 0.5
            high_risk_rate = high_risk_count / total_comments
            
            return {
                "average_authenticity": avg_authenticity,
                "high_risk_comments": high_risk_count,
                "high_risk_rate": high_risk_rate,
                "total_analyzed": total_comments,
                "method": "fallback_rule_based",
                "risk_distribution": {
                    "low_risk": sum(1 for s in authenticity_scores if s >= 0.7),
                    "medium_risk": sum(1 for s in authenticity_scores if 0.3 <= s < 0.7),
                    "high_risk": sum(1 for s in authenticity_scores if s < 0.3)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in fallback fake engagement detection: {e}")
            return {"error": str(e)}
    
    def _analyze_engagement_spikes(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze engagement spikes for suspicious patterns"""
        
        try:
            spikes = []
            
            if 'timestamp' in features_df.columns and len(features_df) > 10:
                df_sorted = features_df.sort_values('timestamp')
                
                # Calculate rolling average
                window = min(7, len(df_sorted) // 3)
                rolling_avg = df_sorted['likes'].rolling(window=window, center=True).mean()
                
                # Identify spikes (engagement > 3x rolling average)
                spike_factor = self.config["thresholds"]["suspicious_spike_factor"]
                spike_threshold = rolling_avg * spike_factor
                
                spike_mask = df_sorted['likes'] > spike_threshold
                spike_data = df_sorted[spike_mask]
                
                if len(spike_data) > 0:
                    spikes = {
                        "total_spikes": len(spike_data),
                        "spike_rate": len(spike_data) / len(df_sorted),
                        "avg_spike_magnitude": (spike_data['likes'] / rolling_avg[spike_mask]).mean(),
                        "spike_timestamps": spike_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()[:10],
                        "suspicious_pattern": len(spike_data) > len(df_sorted) * 0.1  # >10% spikes
                    }
                else:
                    spikes = {"total_spikes": 0, "spike_rate": 0.0, "suspicious_pattern": False}
            
            return spikes
            
        except Exception as e:
            logger.error(f"Error analyzing engagement spikes: {e}")
            return {"error": str(e)}
    
    def _predict_soe_impact(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Predict Share of Engagement (SoE) impact using ML models"""
        
        try:
            # Create SoE target variable (likes per comment as proxy)
            if len(features_df) < 20:
                logger.warning("Insufficient data for SoE prediction")
                return {"error": "Insufficient data for SoE prediction"}
            
            # Calculate SoE-like metric
            features_df['soe_metric'] = features_df['likes'] / (features_df['likes'].mean() + 1)
            
            # Prepare features for prediction
            prediction_features = [
                'qei_score', 'relevance', 'informativeness', 'constructiveness',
                'spam_score', 'bot_score', 'authenticity_score',
                'text_length', 'word_count'
            ]
            
            # Add temporal features if available
            temporal_cols = [col for col in features_df.columns if 'rolling' in col or col in ['hour', 'day_of_week']]
            prediction_features.extend(temporal_cols[:5])
            
            # Filter existing features
            prediction_features = [col for col in prediction_features if col in features_df.columns]
            
            X = features_df[prediction_features].copy()
            y = features_df['soe_metric'].copy()
            
            # Split data
            settings = self.config["soe_prediction"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=settings["test_size"], random_state=settings["random_state"]
            )
            
            # Train model
            if XGBOOST_AVAILABLE and settings["model_type"] == "xgboost":
                model = xgb.XGBRegressor(
                    n_estimators=settings["n_estimators"],
                    max_depth=settings["max_depth"],
                    learning_rate=settings["learning_rate"],
                    random_state=settings["random_state"]
                )
            elif LIGHTGBM_AVAILABLE and settings["model_type"] == "lightgbm":
                model = lgb.LGBMRegressor(
                    n_estimators=settings["n_estimators"],
                    max_depth=settings["max_depth"],
                    learning_rate=settings["learning_rate"],
                    random_state=settings["random_state"],
                    verbose=-1
                )
            else:
                # Fallback to sklearn
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=settings["n_estimators"],
                    max_depth=settings["max_depth"],
                    random_state=settings["random_state"]
                )
            
            model.fit(X_train, y_train)
            self.soe_predictor = model
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(prediction_features, model.feature_importances_))
                sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            else:
                sorted_importance = {}
            
            # SHAP explanations (if available)
            shap_values = None
            if SHAP_AVAILABLE:
                try:
                    if settings["model_type"] in ["xgboost", "lightgbm"]:
                        explainer = shap.Explainer(model)
                        shap_values = explainer(X_test[:min(100, len(X_test))])
                        self.shap_explainer = explainer
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
            
            # Generate future predictions (next week scenario)
            future_scenarios = self._generate_future_scenarios(X.mean(), prediction_features)
            
            return {
                "model_performance": {
                    "mse": float(mse),
                    "mae": float(mae),
                    "r2_score": float(r2),
                    "train_size": len(X_train),
                    "test_size": len(X_test)
                },
                "feature_importance": sorted_importance,
                "predictions": {
                    "predicted_soe": y_pred.tolist(),
                    "actual_soe": y_test.tolist()
                },
                "future_scenarios": future_scenarios,
                "model_type": settings["model_type"],
                "features_used": prediction_features,
                "shap_available": shap_values is not None
            }
            
        except Exception as e:
            logger.error(f"Error in SoE prediction: {e}")
            return {"error": str(e)}
    
    def _generate_future_scenarios(self, baseline_features: pd.Series, feature_names: List[str]) -> Dict[str, Any]:
        """Generate future SoE scenarios based on feature changes"""
        
        scenarios = {}
        
        try:
            if self.soe_predictor is None:
                return {"error": "No trained model available"}
            
            # Baseline prediction
            baseline_pred = self.soe_predictor.predict(baseline_features.values.reshape(1, -1))[0]
            scenarios["baseline"] = float(baseline_pred)
            
            # Scenario 1: Improved quality (+20% QEI)
            improved_quality = baseline_features.copy()
            if 'qei_score' in improved_quality.index:
                improved_quality['qei_score'] *= 1.2
            pred_improved = self.soe_predictor.predict(improved_quality.values.reshape(1, -1))[0]
            scenarios["improved_quality"] = {
                "predicted_soe": float(pred_improved),
                "change_pct": float((pred_improved - baseline_pred) / baseline_pred * 100)
            }
            
            # Scenario 2: Reduced spam (-50% spam/bot scores)
            reduced_spam = baseline_features.copy()
            for col in ['spam_score', 'bot_score']:
                if col in reduced_spam.index:
                    reduced_spam[col] *= 0.5
            if 'authenticity_score' in reduced_spam.index:
                reduced_spam['authenticity_score'] = min(1.0, reduced_spam['authenticity_score'] * 1.2)
            pred_clean = self.soe_predictor.predict(reduced_spam.values.reshape(1, -1))[0]
            scenarios["reduced_spam"] = {
                "predicted_soe": float(pred_clean),
                "change_pct": float((pred_clean - baseline_pred) / baseline_pred * 100)
            }
            
            # Scenario 3: Higher engagement content (+30% relevance/informativeness)
            higher_engagement = baseline_features.copy()
            for col in ['relevance', 'informativeness', 'constructiveness']:
                if col in higher_engagement.index:
                    higher_engagement[col] = min(1.0, higher_engagement[col] * 1.3)
            pred_engaging = self.soe_predictor.predict(higher_engagement.values.reshape(1, -1))[0]
            scenarios["higher_engagement"] = {
                "predicted_soe": float(pred_engaging),
                "change_pct": float((pred_engaging - baseline_pred) / baseline_pred * 100)
            }
            
        except Exception as e:
            logger.error(f"Error generating future scenarios: {e}")
            scenarios["error"] = str(e)
        
        return scenarios
    
    def _assess_engagement_risks(self, features_df: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall engagement risks"""
        
        risk_factors = []
        risk_score = 0.0
        
        try:
            # Risk from fake engagement
            if "fake_engagement_insurance" in results:
                fake_rate = results["fake_engagement_insurance"].get("suspicious_rate", 0)
                if fake_rate > 0.2:  # >20% suspicious
                    risk_factors.append(f"High fake engagement rate: {fake_rate:.1%}")
                    risk_score += 0.3
                
                spike_analysis = results["fake_engagement_insurance"].get("spike_analysis", {})
                if spike_analysis.get("suspicious_pattern", False):
                    risk_factors.append("Suspicious engagement spike pattern detected")
                    risk_score += 0.2
            
            # Risk from poor quality
            if 'qei_score' in features_df.columns:
                avg_qei = features_df['qei_score'].mean()
                if avg_qei < 40:
                    risk_factors.append(f"Low average quality score: {avg_qei:.1f}")
                    risk_score += 0.2
            else:
                avg_qei = 50.0  # Default neutral score
            
            # Risk from spam/bots
            if 'authenticity_score' in features_df.columns:
                avg_authenticity = features_df['authenticity_score'].mean()
                if avg_authenticity < 0.5:
                    risk_factors.append(f"Low authenticity score: {avg_authenticity:.2f}")
                    risk_score += 0.3
            else:
                avg_authenticity = 0.8  # Default high authenticity score
            
            # Risk from poor engagement prediction
            if "soe_predictions" in results:
                model_r2 = results["soe_predictions"]["model_performance"].get("r2_score", 0)
                if model_r2 < 0.3:
                    risk_factors.append(f"Poor engagement predictability (R² = {model_r2:.2f})")
                    risk_score += 0.1
            
            # Risk level
            if risk_score < 0.3:
                risk_level = "LOW"
            elif risk_score < 0.6:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            return {
                "risk_level": risk_level,
                "risk_score": float(min(1.0, risk_score)),
                "risk_factors": risk_factors,
                "metrics": {
                    "avg_qei_score": float(avg_qei),
                    "avg_authenticity": float(avg_authenticity),
                    "total_comments": len(features_df)
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing risks: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        try:
            # Fake engagement recommendations
            if "fake_engagement_insurance" in results:
                suspicious_rate = results["fake_engagement_insurance"].get("suspicious_rate", 0)
                if suspicious_rate > 0.15:
                    recommendations.append(
                        f"⚠️ HIGH PRIORITY: {suspicious_rate:.1%} of engagement appears suspicious. "
                        "Implement stronger bot detection and content moderation."
                    )
                
                spike_analysis = results["fake_engagement_insurance"].get("spike_analysis", {})
                if spike_analysis.get("suspicious_pattern", False):
                    recommendations.append(
                        "📈 Investigate unusual engagement spikes - may indicate coordinated inauthentic behavior."
                    )
            
            # Quality improvement recommendations
            if "soe_predictions" in results:
                feature_importance = results["soe_predictions"].get("feature_importance", {})
                
                # Find most important quality factors
                top_features = list(feature_importance.keys())[:3]
                if 'qei_score' in top_features or 'relevance' in top_features:
                    recommendations.append(
                        "💡 Focus on content quality - QEI and relevance are key drivers of engagement."
                    )
                
                if 'authenticity_score' in top_features:
                    recommendations.append(
                        "🔒 Authenticity is a major factor - strengthen spam/bot detection measures."
                    )
                
                # Future scenario recommendations
                scenarios = results["soe_predictions"].get("future_scenarios", {})
                improved_quality = scenarios.get("improved_quality", {})
                if improved_quality.get("change_pct", 0) > 10:
                    recommendations.append(
                        f"📊 Improving content quality could increase engagement by {improved_quality['change_pct']:.1f}%"
                    )
                
                reduced_spam = scenarios.get("reduced_spam", {})
                if reduced_spam.get("change_pct", 0) > 5:
                    recommendations.append(
                        f"🧹 Reducing spam could boost engagement by {reduced_spam['change_pct']:.1f}%"
                    )
            
            # Risk-based recommendations
            if "risk_assessment" in results:
                risk_level = results["risk_assessment"].get("risk_level", "UNKNOWN")
                
                if risk_level == "HIGH":
                    recommendations.append(
                        "🚨 URGENT: High engagement risk detected. Review authenticity measures immediately."
                    )
                elif risk_level == "MEDIUM":
                    recommendations.append(
                        "⚠️ Moderate risk - monitor engagement patterns closely and consider preventive measures."
                    )
                else:
                    recommendations.append(
                        "✅ Low risk profile - maintain current quality and authenticity standards."
                    )
            
            # Default recommendations if none generated
            if not recommendations:
                recommendations = [
                    "📈 Continue monitoring engagement patterns for anomalies",
                    "💯 Focus on high-quality, authentic content to maximize SoE",
                    "🔍 Regularly assess authenticity scores and investigate suspicious activity"
                ]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations = ["❌ Error generating recommendations - review analysis manually"]
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _summarize_features(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize engineered features"""
        
        return {
            "total_features": len(self.feature_columns),
            "feature_categories": {
                "quality_features": len([f for f in self.feature_columns if any(q in f for q in ['qei', 'relevance', 'informative', 'construct'])]),
                "authenticity_features": len([f for f in self.feature_columns if any(a in f for a in ['spam', 'bot', 'authentic'])]),
                "temporal_features": len([f for f in self.feature_columns if 'rolling' in f or f in ['hour', 'day_of_week']]),
                "text_features": len([f for f in self.feature_columns if any(t in f for t in ['text_length', 'word_count'])]),
                "user_features": len([f for f in self.feature_columns if 'user_' in f])
            },
            "data_quality": {
                "total_comments": len(features_df),
                "missing_values": features_df[self.feature_columns].isnull().sum().sum(),
                "avg_qei_score": features_df.get('qei_score', pd.Series([0])).mean(),
                "avg_authenticity": features_df.get('authenticity_score', pd.Series([0])).mean()
            }
        }
    
    def _cleanup_resources(self):
        """Clean up resources to prevent memory leaks"""
        try:
            # Explicitly delete ML models to free memory
            if hasattr(self, 'anomaly_detector') and self.anomaly_detector is not None:
                del self.anomaly_detector
                self.anomaly_detector = None
                
            if hasattr(self, 'soe_predictor') and self.soe_predictor is not None:
                del self.soe_predictor
                self.soe_predictor = None
                
            if hasattr(self, 'feature_scaler') and self.feature_scaler is not None:
                del self.feature_scaler
                self.feature_scaler = None
                
            if hasattr(self, 'shap_explainer') and self.shap_explainer is not None:
                del self.shap_explainer
                self.shap_explainer = None
                
            # Clear dictionaries and lists
            if hasattr(self, 'label_encoders'):
                self.label_encoders.clear()
            if hasattr(self, 'feature_columns'):
                self.feature_columns.clear()
            if hasattr(self, 'temporal_features'):
                self.temporal_features.clear()
                
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            # Don't raise exceptions during cleanup
            logger.debug(f"Error during resource cleanup: {e}")

# Factory function for pipeline integration
def create_predictive_analyzer(config_path: Optional[str] = None) -> PredictiveAnalyzer:
    """Factory function to create predictive analyzer with configuration"""
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return PredictiveAnalyzer(config)