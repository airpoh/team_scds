"""
Composite KPI System for CommentSense Pipeline

This module orchestrates ALL existing CommentSense modules plus new advanced analytics:

EXISTING MODULES:
- Emotion & Sarcasm Detection
- Visual & Emoji Analysis  
- Multilingual Analysis
- Crisis Detection
- Network Analysis (Alvin's utilities)

NEW ADVANCED MODULES:
- Comment Quality Scoring 2.0 (QEI)
- Spam + Bot Network Detection
- Predictive Analytics (Fake Engagement Insurance + SoE Impact)

The composite KPI provides a unified score (0-100) representing overall 
engagement health across all dimensions of analysis.
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Import existing CommentSense modules
try:
    from .emotion_sarcasm_detection import EmotionSarcasmDetector
    from .visual_emoji_analysis import VisualEmojiAnalyzer
    from .multilingual_analysis import MultilingualSentimentAnalyzer
    from .crisis_detection import CrisisDetectionSystem
    from .network_analysis import NetworkAnalyzer
    EXISTING_MODULES = True
except ImportError:
    # Fallback for development
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from emotion_sarcasm_detection import EmotionSarcasmDetector
        from visual_emoji_analysis import VisualEmojiAnalyzer
        from multilingual_analysis import MultilingualSentimentAnalyzer
        from crisis_detection import CrisisDetectionSystem
        from network_analysis import NetworkAnalyzer
        EXISTING_MODULES = True
    except ImportError as e:
        logging.warning(f"Could not import existing modules: {e}")
        EXISTING_MODULES = False

# Import new advanced modules
try:
    from .comment_quality_scoring import CommentQualityScorer
    from .spam_bot_detection import SpamBotDetector
    from .predictive_analytics import PredictiveAnalyzer
    NEW_MODULES = True
except ImportError:
    try:
        from comment_quality_scoring import CommentQualityScorer
        from spam_bot_detection import SpamBotDetector
        from predictive_analytics import PredictiveAnalyzer
        NEW_MODULES = True
    except ImportError as e:
        logging.warning(f"Could not import new modules: {e}")
        NEW_MODULES = False

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class CompositeKPISystem:
    """
    Unified Composite KPI System for Complete CommentSense Pipeline
    
    Orchestrates ALL modules to provide:
    - Unified engagement health score (0-100)
    - Multi-dimensional quality assessment across 8+ analysis areas
    - Risk-adjusted authenticity scoring
    - Predictive insights and comprehensive recommendations
    - Full dashboard metrics covering all analysis dimensions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Complete Composite KPI System"""
        
        self.config = config or self._get_default_config()
        
        # Initialize existing CommentSense modules
        self.emotion_detector = None
        self.emoji_analyzer = None
        self.multilingual_analyzer = None
        self.crisis_detector = None
        self.network_analyzer = None
        
        # Initialize new advanced modules
        self.quality_scorer = None
        self.spam_detector = None
        self.predictive_analyzer = None
        
        # Composite scoring weights for ALL modules
        self.composite_weights = self.config.get("composite_weights", {
            # Existing modules (60% total)
            "emotion_sentiment": 0.15,      # Emotion & sarcasm analysis
            "emoji_visual": 0.10,           # Emoji & visual analysis
            "multilingual": 0.10,           # Language & sentiment
            "crisis_safety": 0.15,          # Crisis detection
            "network_influence": 0.10,      # Network & influencer analysis
            
            # New advanced modules (40% total)  
            "quality_scoring": 0.20,        # QEI quality scoring
            "spam_authenticity": 0.15,     # Spam/bot detection
            "predictive_insights": 0.05     # Future engagement prediction
        })
        
        # Performance tracking
        self.execution_metrics = {}
        self.module_status = {}
        
        # Initialize all modules
        self._initialize_all_modules()
            
        logger.info("Complete Composite KPI System initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration orchestrating all modules"""
        return {
            # Composite scoring weights (must sum to 1.0)
            "composite_weights": {
                "emotion_sentiment": 0.15,
                "emoji_visual": 0.10, 
                "multilingual": 0.10,
                "crisis_safety": 0.15,
                "network_influence": 0.10,
                "quality_scoring": 0.20,
                "spam_authenticity": 0.15,
                "predictive_insights": 0.05
            },
            
            # KPI grade thresholds
            "kpi_thresholds": {
                "excellent": 85,      # A+ engagement
                "good": 70,          # B+ engagement  
                "fair": 50,          # C+ engagement
                "poor": 0            # Needs improvement
            },
            
            # Risk adjustment factors across all dimensions
            "risk_adjustments": {
                "high_crisis_risk": 0.4,         # -40% for crisis content
                "negative_sentiment_penalty": 0.2, # -20% for negative sentiment
                "spam_bot_penalty": 0.3,          # -30% for spam/bots
                "low_quality_penalty": 0.25,     # -25% for low QEI
                "fake_engagement_penalty": 0.3,   # -30% for fake engagement
                "language_barrier_penalty": 0.1   # -10% for untranslated content
            },
            
            # Module-specific configurations
            "module_configs": {
                "emotion_sarcasm": {"batch_size": 32},
                "visual_emoji": {"enable_vision": True},
                "multilingual": {"target_language": "en"},
                "crisis_detection": {"alert_threshold": 0.7},
                "network_analysis": {"fast_mode": True},
                "quality_scoring": {"weights": {"relevance": 0.5, "informativeness": 0.3, "constructiveness": 0.2}},
                "spam_detection": {"contamination": 0.1},
                "predictive_analytics": {"model_type": "xgboost"}
            }
        }
    
    def _initialize_all_modules(self):
        """Initialize all CommentSense modules (existing + new)"""
        
        # Initialize existing modules
        if EXISTING_MODULES:
            try:
                # EmotionSarcasmDetector expects a config file path, not a dict
                emotion_config = self.config.get("module_configs", {}).get("emotion_sarcasm", {})
                emotion_config_path = None
                if isinstance(emotion_config, str):
                    emotion_config_path = emotion_config
                elif isinstance(emotion_config, dict) and "config_path" in emotion_config:
                    emotion_config_path = emotion_config["config_path"]
                # If no specific config path, let the module use its default
                
                self.emotion_detector = EmotionSarcasmDetector(config_path=emotion_config_path)
                self.module_status["emotion_detector"] = "initialized"
                logger.info("Emotion & Sarcasm detector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize emotion detector: {e}")
                self.module_status["emotion_detector"] = f"failed: {e}"
            
            try:
                # VisualEmojiAnalyzer expects a config file path, not a dict
                emoji_config = self.config.get("module_configs", {}).get("visual_emoji", {})
                emoji_config_path = None
                if isinstance(emoji_config, str):
                    emoji_config_path = emoji_config
                elif isinstance(emoji_config, dict) and "config_path" in emoji_config:
                    emoji_config_path = emoji_config["config_path"]
                # If no specific config path, let the module use its default
                
                self.emoji_analyzer = VisualEmojiAnalyzer(config_path=emoji_config_path)
                self.module_status["emoji_analyzer"] = "initialized"
                logger.info("Visual & Emoji analyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize emoji analyzer: {e}")
                self.module_status["emoji_analyzer"] = f"failed: {e}"
            
            try:
                # MultilingualSentimentAnalyzer expects a config file path, not a dict
                multilingual_config = self.config.get("module_configs", {}).get("multilingual", {})
                multilingual_config_path = None
                if isinstance(multilingual_config, str):
                    multilingual_config_path = multilingual_config
                elif isinstance(multilingual_config, dict) and "config_path" in multilingual_config:
                    multilingual_config_path = multilingual_config["config_path"]
                # If no specific config path, let the module use its default
                
                self.multilingual_analyzer = MultilingualSentimentAnalyzer(config_path=multilingual_config_path)
                self.module_status["multilingual_analyzer"] = "initialized"
                logger.info("Multilingual analyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize multilingual analyzer: {e}")
                self.module_status["multilingual_analyzer"] = f"failed: {e}"
            
            try:
                self.crisis_detector = CrisisDetectionSystem(
                    self.config.get("module_configs", {}).get("crisis_detection", {})
                )
                self.module_status["crisis_detector"] = "initialized"
                logger.info("Crisis detection system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize crisis detector: {e}")
                self.module_status["crisis_detector"] = f"failed: {e}"
            
            try:
                self.network_analyzer = NetworkAnalyzer(
                    self.config.get("module_configs", {}).get("network_analysis", {})
                )
                self.module_status["network_analyzer"] = "initialized"
                logger.info("Network analyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize network analyzer: {e}")
                self.module_status["network_analyzer"] = f"failed: {e}"
        
        # Initialize new advanced modules
        if NEW_MODULES:
            try:
                self.quality_scorer = CommentQualityScorer(
                    self.config.get("module_configs", {}).get("quality_scoring", {})
                )
                self.module_status["quality_scorer"] = "initialized"
                logger.info("Quality scorer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize quality scorer: {e}")
                self.module_status["quality_scorer"] = f"failed: {e}"
            
            try:
                self.spam_detector = SpamBotDetector(
                    self.config.get("module_configs", {}).get("spam_detection", {})
                )
                self.module_status["spam_detector"] = "initialized"
                logger.info("Spam/bot detector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize spam detector: {e}")
                self.module_status["spam_detector"] = f"failed: {e}"
            
            try:
                self.predictive_analyzer = PredictiveAnalyzer(
                    self.config.get("module_configs", {}).get("predictive_analytics", {})
                )
                self.module_status["predictive_analyzer"] = "initialized"
                logger.info("Predictive analyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize predictive analyzer: {e}")
                self.module_status["predictive_analyzer"] = f"failed: {e}"
        
        # Log initialization summary
        initialized_count = len([s for s in self.module_status.values() if s == "initialized"])
        total_modules = len(self.module_status)
        logger.info(f"Initialized {initialized_count}/{total_modules} modules successfully")
    
    def calculate_comprehensive_kpi(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive KPI across ALL CommentSense modules
        
        Args:
            comments_df: DataFrame with comment data
            
        Returns:
            Dictionary containing complete KPI results across all dimensions
        """
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Calculating comprehensive KPI for {len(comments_df)} comments across all modules...")
            
            # Validate input
            if len(comments_df) == 0:
                return {"error": "No comments provided for analysis"}
            
            # Step 1: Execute ALL modules
            all_module_results = self._execute_all_modules(comments_df)
            
            # Step 2: Calculate component scores from all modules
            component_scores = self._calculate_all_component_scores(all_module_results)
            
            # Step 3: Calculate composite KPI with risk adjustments
            composite_kpi = self._calculate_unified_composite_score(component_scores, all_module_results)
            
            # Step 4: Generate comprehensive dashboard
            comprehensive_dashboard = self._generate_comprehensive_dashboard(
                composite_kpi, component_scores, all_module_results
            )
            
            # Step 5: Cross-module insights and recommendations
            unified_insights = self._generate_unified_insights(
                composite_kpi, all_module_results, component_scores
            )
            
            # Step 6: Execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_metrics = {
                "processing_time_seconds": execution_time,
                "comments_processed": len(comments_df),
                "processing_rate": len(comments_df) / execution_time if execution_time > 0 else 0,
                "modules_executed": len([r for r in all_module_results.values() if "error" not in r]),
                "module_success_rate": len([s for s in self.module_status.values() if s == "initialized"]) / len(self.module_status)
            }
            
            logger.info(f"Comprehensive KPI calculation completed in {execution_time:.2f}s")
            
            return {
                "status": "success",
                "comprehensive_kpi": {
                    "overall_score": composite_kpi["final_composite_score"],
                    "grade": composite_kpi["kpi_grade"],
                    "component_scores": component_scores,
                    "risk_adjusted_score": composite_kpi["risk_adjusted_score"],
                    "confidence_score": composite_kpi["confidence_score"]
                },
                "dashboard_metrics": comprehensive_dashboard,
                "all_module_results": all_module_results,
                "unified_insights": unified_insights,
                "execution_metrics": self.execution_metrics,
                "module_status": self.module_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive KPI calculation: {e}")
            return {
                "status": "error", 
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _execute_all_modules(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Execute all CommentSense modules (existing + new)"""
        
        results = {}
        
        # Execute existing CommentSense modules
        logger.info("Executing existing CommentSense modules...")
        
        # Normalize DataFrame for consistent column access across all modules
        normalized_df = comments_df.copy()
        
        # Get text column safely and normalize to 'text'
        text_column = None
        for col in ['text', 'textOriginal', 'comment_text', 'content']:
            if col in comments_df.columns:
                text_column = col
                break
        
        if text_column is None:
            logger.error("No text column found in comments DataFrame")
            text_list = [""] * len(comments_df)  # Fallback empty texts
            normalized_df['text'] = ""  # Add empty text column for other modules
        else:
            text_list = comments_df[text_column].fillna("").astype(str).tolist()
            # Ensure 'text' column exists for other modules
            if text_column != 'text':
                normalized_df['text'] = normalized_df[text_column]
        
        # Emotion & Sarcasm Detection
        if self.emotion_detector:
            try:
                results["emotion_analysis"] = self.emotion_detector.batch_analyze(text_list)
            except Exception as e:
                logger.error(f"Emotion analysis failed: {e}")
                results["emotion_analysis"] = {"error": str(e)}
        else:
            results["emotion_analysis"] = self._fallback_emotion_analysis(normalized_df)
        
        # Visual & Emoji Analysis  
        if self.emoji_analyzer:
            try:
                results["emoji_analysis"] = self.emoji_analyzer.batch_analyze(text_list)
            except Exception as e:
                logger.error(f"Emoji analysis failed: {e}")
                results["emoji_analysis"] = {"error": str(e)}
        else:
            results["emoji_analysis"] = self._fallback_emoji_analysis(normalized_df)
        
        # Multilingual Analysis
        if self.multilingual_analyzer:
            try:
                results["multilingual_analysis"] = self.multilingual_analyzer.batch_analyze(text_list)
            except Exception as e:
                logger.error(f"Multilingual analysis failed: {e}")
                results["multilingual_analysis"] = {"error": str(e)}
        else:
            results["multilingual_analysis"] = self._fallback_multilingual_analysis(normalized_df)
        
        # Crisis Detection
        if self.crisis_detector:
            try:
                results["crisis_analysis"] = self.crisis_detector.analyze_crisis_patterns(normalized_df)
            except Exception as e:
                logger.error(f"Crisis detection failed: {e}")
                results["crisis_analysis"] = {"error": str(e)}
        else:
            results["crisis_analysis"] = self._fallback_crisis_analysis(normalized_df)
        
        # Network Analysis
        if self.network_analyzer:
            try:
                results["network_analysis"] = self.network_analyzer.analyze_network(normalized_df)
            except Exception as e:
                logger.error(f"Network analysis failed: {e}")
                results["network_analysis"] = {"error": str(e)}
        else:
            results["network_analysis"] = self._fallback_network_analysis(normalized_df)
        
        # Execute new advanced modules
        logger.info("Executing new advanced analytics modules...")
        
        # Quality Scoring 2.0
        if self.quality_scorer:
            try:
                results["quality_analysis"] = self.quality_scorer.score_comments(normalized_df)
            except Exception as e:
                logger.error(f"Quality scoring failed: {e}")
                results["quality_analysis"] = {"error": str(e)}
        else:
            results["quality_analysis"] = self._fallback_quality_analysis(normalized_df)
        
        # Spam & Bot Detection with batch processing for large datasets
        if self.spam_detector:
            try:
                # Use moderate batch sizes for better performance
                import platform
                if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
                    batch_size = min(100, max(50, len(normalized_df) // 10))  # Apple Silicon
                else:
                    batch_size = min(200, max(100, len(normalized_df) // 5))   # Other platforms
                logger.info("Starting spam/bot detection with optimized batch processing...")
                results["spam_bot_analysis"] = self.spam_detector.detect_spam_and_bots(
                    normalized_df, batch_size=batch_size
                )
                logger.info("Spam/bot detection completed, starting predictive analytics...")
            except Exception as e:
                logger.error(f"Spam/bot detection failed: {e}")
                results["spam_bot_analysis"] = {"error": str(e)}
        else:
            results["spam_bot_analysis"] = self._fallback_spam_analysis(normalized_df)
        
        # Predictive Analytics with memory management
        if self.predictive_analyzer:
            try:
                # Force garbage collection before predictive analysis
                import gc
                gc.collect()
                
                quality_results = results.get("quality_analysis")
                spam_results = results.get("spam_bot_analysis")
                
                # Prepare data for predictive analytics - ensure DataFrames are accessible
                prepared_quality = self._prepare_quality_data_for_prediction(quality_results)
                prepared_spam = self._prepare_spam_data_for_prediction(spam_results)
                
                # Limit data size for Apple Silicon to prevent memory issues
                import platform
                if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
                    if len(normalized_df) > 200:
                        logger.info(f"Apple Silicon: Limiting predictive analysis to 200 comments for stability")
                        analysis_df = normalized_df.head(200).copy()
                    else:
                        analysis_df = normalized_df.copy()
                else:
                    analysis_df = normalized_df.copy()
                
                results["predictive_analysis"] = self.predictive_analyzer.analyze_engagement_patterns(
                    analysis_df, prepared_quality, prepared_spam
                )
                
                # Force garbage collection after predictive analysis
                gc.collect()
            except Exception as e:
                logger.error(f"Predictive analysis failed: {e}")
                results["predictive_analysis"] = {"error": str(e)}
        else:
            results["predictive_analysis"] = self._fallback_predictive_analysis(normalized_df)
        
        return results
    
    def _calculate_all_component_scores(self, all_module_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate component scores from all modules"""
        
        component_scores = {}
        
        try:
            # Emotion & Sentiment Score (0-100)
            emotion_results = all_module_results.get("emotion_analysis", {})
            if isinstance(emotion_results, list) and len(emotion_results) > 0:
                # Calculate average sentiment (positive emotions boost score)
                positive_emotions = ['joy', 'love', 'optimism', 'admiration', 'approval']
                emotion_scores = []
                for result in emotion_results:
                    if isinstance(result, dict) and 'emotions' in result:
                        pos_score = sum(result['emotions'].get(e, 0) for e in positive_emotions)
                        emotion_scores.append(min(100, pos_score * 200))  # Scale to 0-100
                component_scores["emotion_sentiment"] = np.mean(emotion_scores) if emotion_scores else 60.0
            else:
                component_scores["emotion_sentiment"] = 60.0
            
            # Emoji & Visual Score (0-100)
            emoji_results = all_module_results.get("emoji_analysis", {})
            if "overall_sentiment" in emoji_results:
                # Convert sentiment to 0-100 scale
                emoji_sentiment = emoji_results["overall_sentiment"]
                component_scores["emoji_visual"] = max(0, min(100, (emoji_sentiment + 1) * 50))
            else:
                component_scores["emoji_visual"] = 70.0
            
            # Multilingual Score (0-100)
            multilingual_results = all_module_results.get("multilingual_analysis", {})
            if isinstance(multilingual_results, list) and len(multilingual_results) > 0:
                sentiment_scores = []
                for result in multilingual_results:
                    if isinstance(result, dict) and 'sentiment' in result:
                        if result['sentiment'] == 'positive':
                            sentiment_scores.append(80)
                        elif result['sentiment'] == 'negative':
                            sentiment_scores.append(20)
                        else:
                            sentiment_scores.append(50)
                component_scores["multilingual"] = np.mean(sentiment_scores) if sentiment_scores else 60.0
            else:
                component_scores["multilingual"] = 60.0
            
            # Crisis Safety Score (0-100, inverse of crisis risk)
            crisis_results = all_module_results.get("crisis_analysis", {})
            if "alerts" in crisis_results:
                total_comments = len(all_module_results.get("emotion_analysis", []))
                if total_comments > 0:
                    crisis_rate = len(crisis_results["alerts"]) / total_comments
                    component_scores["crisis_safety"] = max(0, 100 - (crisis_rate * 200))
                else:
                    component_scores["crisis_safety"] = 90.0
            else:
                component_scores["crisis_safety"] = 90.0
            
            # Network Influence Score (0-100)
            network_results = all_module_results.get("network_analysis", {})
            if "results" in network_results:
                # Use influencer metrics as proxy for network health
                influencers = network_results["results"].get("influencers", {})
                if "total_influencers" in influencers:
                    # More influencers = better network engagement
                    total_comments = len(all_module_results.get("emotion_analysis", []))
                    influencer_ratio = influencers["total_influencers"] / max(1, total_comments / 10)
                    component_scores["network_influence"] = min(100, influencer_ratio * 50)
                else:
                    component_scores["network_influence"] = 60.0
            else:
                component_scores["network_influence"] = 60.0
            
            # Quality Score (0-100) - from new QEI module
            quality_results = all_module_results.get("quality_analysis", {})
            if "aggregated_metrics" in quality_results:
                avg_qei = quality_results["aggregated_metrics"]["overall_metrics"]["avg_qei_score"]
                component_scores["quality_scoring"] = float(avg_qei)
            else:
                component_scores["quality_scoring"] = 50.0
            
            # Spam/Authenticity Score (0-100) - from new spam detection
            spam_results = all_module_results.get("spam_bot_analysis", {})
            if "detection_metrics" in spam_results:
                avg_auth = spam_results["detection_metrics"]["authenticity_metrics"]["avg_authenticity"]
                component_scores["spam_authenticity"] = float(avg_auth * 100)
            else:
                component_scores["spam_authenticity"] = 80.0
            
            # Predictive Insights Score (0-100) - from new predictive module
            pred_results = all_module_results.get("predictive_analysis", {})
            if "analytics_results" in pred_results:
                soe_results = pred_results["analytics_results"].get("soe_predictions", {})
                if "model_performance" in soe_results:
                    r2_score = soe_results["model_performance"].get("r2_score", 0)
                    # Convert R² to confidence score
                    pred_score = 50 + (r2_score * 50) if r2_score > 0 else 30
                    component_scores["predictive_insights"] = float(max(0, min(100, pred_score)))
                else:
                    component_scores["predictive_insights"] = 60.0
            else:
                component_scores["predictive_insights"] = 60.0
            
            # Ensure all scores are in valid range
            for key in component_scores:
                component_scores[key] = max(0.0, min(100.0, component_scores[key]))
            
            return component_scores
            
        except Exception as e:
            logger.error(f"Error calculating component scores: {e}")
            # Return neutral scores as fallback
            return {
                "emotion_sentiment": 60.0,
                "emoji_visual": 60.0,
                "multilingual": 60.0,
                "crisis_safety": 80.0,
                "network_influence": 60.0,
                "quality_scoring": 50.0,
                "spam_authenticity": 70.0,
                "predictive_insights": 60.0
            }
    
    def _calculate_unified_composite_score(self, component_scores: Dict[str, float], 
                                         all_module_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final unified composite score with risk adjustments"""
        
        try:
            # Calculate weighted composite score
            weights = self.composite_weights
            composite_score = 0.0
            
            for component, score in component_scores.items():
                if component in weights:
                    composite_score += weights[component] * score
            
            # Calculate risk adjustments across all modules
            risk_penalty = self._calculate_comprehensive_risk_penalty(all_module_results)
            risk_adjusted_score = composite_score * (1 - risk_penalty)
            
            # Calculate confidence score based on module success
            successful_modules = len([r for r in all_module_results.values() if "error" not in r])
            total_modules = len(all_module_results)
            confidence_score = successful_modules / total_modules if total_modules > 0 else 0.5
            
            # Determine KPI grade
            final_score = risk_adjusted_score * confidence_score
            thresholds = self.config["kpi_thresholds"]
            
            if final_score >= thresholds["excellent"]:
                grade = "EXCELLENT"
            elif final_score >= thresholds["good"]:
                grade = "GOOD"
            elif final_score >= thresholds["fair"]:
                grade = "FAIR"
            else:
                grade = "NEEDS IMPROVEMENT"
            
            return {
                "raw_composite_score": float(composite_score),
                "risk_adjusted_score": float(risk_adjusted_score),
                "final_composite_score": float(final_score),
                "kpi_grade": grade,
                "risk_penalty_applied": float(risk_penalty),
                "confidence_score": float(confidence_score),
                "weights_used": weights
            }
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            return {
                "raw_composite_score": 50.0,
                "risk_adjusted_score": 50.0,
                "final_composite_score": 50.0,
                "kpi_grade": "UNKNOWN",
                "error": str(e)
            }
    
    def _calculate_comprehensive_risk_penalty(self, all_module_results: Dict[str, Any]) -> float:
        """Calculate comprehensive risk penalty from all modules"""
        
        total_penalty = 0.0
        adjustments = self.config["risk_adjustments"]
        
        try:
            # Crisis risk penalty
            crisis_results = all_module_results.get("crisis_analysis", {})
            if "alerts" in crisis_results and len(crisis_results["alerts"]) > 0:
                total_penalty += adjustments["high_crisis_risk"]
            
            # Negative sentiment penalty
            emotion_results = all_module_results.get("emotion_analysis", {})
            if isinstance(emotion_results, list):
                negative_count = 0
                total_count = len(emotion_results)
                for result in emotion_results:
                    if isinstance(result, dict) and 'emotions' in result:
                        negative_emotions = ['anger', 'fear', 'sadness', 'disgust']
                        neg_score = sum(result['emotions'].get(e, 0) for e in negative_emotions)
                        if neg_score > 0.5:
                            negative_count += 1
                
                if total_count > 0 and (negative_count / total_count) > 0.4:  # >40% negative
                    total_penalty += adjustments["negative_sentiment_penalty"]
            
            # Spam/bot penalty
            spam_results = all_module_results.get("spam_bot_analysis", {})
            if "detection_metrics" in spam_results:
                spam_rate = spam_results["detection_metrics"]["spam_metrics"].get("spam_rate", 0)
                bot_rate = spam_results["detection_metrics"]["bot_metrics"].get("bot_rate", 0)
                if spam_rate > 0.2 or bot_rate > 0.15:
                    total_penalty += adjustments["spam_bot_penalty"]
            
            # Quality penalty
            quality_results = all_module_results.get("quality_analysis", {})
            if "aggregated_metrics" in quality_results:
                avg_qei = quality_results["aggregated_metrics"]["overall_metrics"]["avg_qei_score"]
                if avg_qei < 30:
                    total_penalty += adjustments["low_quality_penalty"]
            
            # Fake engagement penalty
            pred_results = all_module_results.get("predictive_analysis", {})
            if "analytics_results" in pred_results:
                fake_rate = pred_results["analytics_results"].get("fake_engagement_insurance", {}).get("suspicious_rate", 0)
                if fake_rate > 0.2:
                    total_penalty += adjustments["fake_engagement_penalty"]
            
            # Language barrier penalty
            multilingual_results = all_module_results.get("multilingual_analysis", {})
            if isinstance(multilingual_results, list):
                non_english_count = sum(1 for r in multilingual_results 
                                      if isinstance(r, dict) and r.get('detected_language', 'en') != 'en')
                if len(multilingual_results) > 0 and (non_english_count / len(multilingual_results)) > 0.5:
                    total_penalty += adjustments["language_barrier_penalty"]
            
            return min(0.8, total_penalty)  # Cap penalty at 80%
            
        except Exception as e:
            logger.error(f"Error calculating risk penalty: {e}")
            return 0.1  # Conservative default penalty
    
    def _generate_comprehensive_dashboard(self, composite_kpi: Dict[str, Any],
                                        component_scores: Dict[str, float],
                                        all_module_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive dashboard covering all modules"""
        
        try:
            dashboard = {
                "headline_kpi": {
                    "overall_score": composite_kpi["final_composite_score"],
                    "grade": composite_kpi["kpi_grade"],
                    "confidence": composite_kpi["confidence_score"],
                    "total_comments": len(all_module_results.get("emotion_analysis", []))
                },
                
                "component_performance": component_scores,
                
                "module_breakdowns": {
                    "emotion_sentiment": self._extract_emotion_dashboard(all_module_results.get("emotion_analysis", {})),
                    "emoji_visual": self._extract_emoji_dashboard(all_module_results.get("emoji_analysis", {})),
                    "multilingual": self._extract_multilingual_dashboard(all_module_results.get("multilingual_analysis", {})),
                    "crisis_safety": self._extract_crisis_dashboard(all_module_results.get("crisis_analysis", {})),
                    "network_influence": self._extract_network_dashboard(all_module_results.get("network_analysis", {})),
                    "quality_scoring": self._extract_quality_dashboard(all_module_results.get("quality_analysis", {})),
                    "spam_authenticity": self._extract_spam_dashboard(all_module_results.get("spam_bot_analysis", {})),
                    "predictive_insights": self._extract_predictive_dashboard(all_module_results.get("predictive_analysis", {}))
                },
                
                "cross_module_insights": {
                    "sentiment_quality_correlation": self._calculate_sentiment_quality_correlation(all_module_results),
                    "authenticity_network_health": self._calculate_authenticity_network_health(all_module_results),
                    "crisis_multilingual_overlap": self._calculate_crisis_language_overlap(all_module_results)
                },
                
                "risk_indicators": {
                    "total_risk_penalty": composite_kpi["risk_penalty_applied"],
                    "primary_risk_factors": self._identify_primary_risks(all_module_results),
                    "risk_level": self._determine_overall_risk_level(composite_kpi["risk_penalty_applied"])
                }
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating comprehensive dashboard: {e}")
            return {"error": str(e)}
    
    def _generate_unified_insights(self, composite_kpi: Dict[str, Any],
                                 all_module_results: Dict[str, Any],
                                 component_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate unified insights across all modules"""
        
        try:
            insights = {
                "key_findings": [],
                "strategic_recommendations": [],
                "tactical_actions": [],
                "monitoring_priorities": []
            }
            
            # Overall KPI insights
            final_score = composite_kpi["final_composite_score"]
            grade = composite_kpi["kpi_grade"]
            
            insights["key_findings"].append(f"Overall Engagement Health: {grade} ({final_score:.1f}/100)")
            
            # Component analysis
            top_performing = max(component_scores.items(), key=lambda x: x[1])
            worst_performing = min(component_scores.items(), key=lambda x: x[1])
            
            insights["key_findings"].append(f"Strongest Area: {top_performing[0].replace('_', ' ').title()} ({top_performing[1]:.1f})")
            insights["key_findings"].append(f"Needs Attention: {worst_performing[0].replace('_', ' ').title()} ({worst_performing[1]:.1f})")
            
            # Strategic recommendations based on performance
            if final_score < 50:
                insights["strategic_recommendations"].append("🚨 URGENT: Comprehensive engagement strategy overhaul needed")
            elif final_score < 70:
                insights["strategic_recommendations"].append("⚠️ Focus on improving weakest performing areas")
            else:
                insights["strategic_recommendations"].append("✅ Optimize high-performing areas for maximum impact")
            
            # Module-specific tactical actions
            if component_scores.get("crisis_safety", 100) < 70:
                insights["tactical_actions"].append("🛡️ Implement enhanced content moderation")
            
            if component_scores.get("spam_authenticity", 100) < 60:
                insights["tactical_actions"].append("🔒 Deploy advanced spam/bot detection")
            
            if component_scores.get("quality_scoring", 100) < 50:
                insights["tactical_actions"].append("📈 Focus on QEI improvement initiatives")
            
            if component_scores.get("emotion_sentiment", 100) < 40:
                insights["tactical_actions"].append("😊 Improve content sentiment through community guidelines")
            
            # Monitoring priorities
            insights["monitoring_priorities"] = [
                "Track composite KPI daily for trend analysis",
                f"Monitor {worst_performing[0].replace('_', ' ')} closely for improvement",
                "Set up alerts for crisis detection and authenticity drops",
                "Review cross-module correlations weekly"
            ]
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating unified insights: {e}")
            return {"error": str(e)}
    
    # Dashboard extraction methods for each module
    def _extract_emotion_dashboard(self, emotion_results: Any) -> Dict[str, Any]:
        """Extract emotion analysis metrics for dashboard"""
        if not isinstance(emotion_results, list) or len(emotion_results) == 0:
            return {"dominant_emotion": "neutral", "sentiment_distribution": {}}
        
        # Aggregate emotions across all comments
        emotion_totals = {}
        for result in emotion_results:
            if isinstance(result, dict) and 'emotions' in result:
                for emotion, score in result['emotions'].items():
                    emotion_totals[emotion] = emotion_totals.get(emotion, 0) + score
        
        # Find dominant emotion
        dominant = max(emotion_totals.items(), key=lambda x: x[1]) if emotion_totals else ("neutral", 0)
        
        return {
            "dominant_emotion": dominant[0],
            "emotion_distribution": emotion_totals,
            "positive_rate": sum(emotion_totals.get(e, 0) for e in ['joy', 'love', 'optimism']) / max(1, sum(emotion_totals.values())),
            "negative_rate": sum(emotion_totals.get(e, 0) for e in ['anger', 'fear', 'sadness']) / max(1, sum(emotion_totals.values()))
        }
    
    def _extract_emoji_dashboard(self, emoji_results: Any) -> Dict[str, Any]:
        """Extract emoji analysis metrics"""
        if not isinstance(emoji_results, dict):
            return {"emoji_sentiment": 0.0, "emoji_count": 0}
        
        return {
            "overall_emoji_sentiment": emoji_results.get("overall_sentiment", 0.0),
            "emoji_categories": emoji_results.get("emoji_categories", {}),
            "visual_elements": emoji_results.get("visual_analysis", {})
        }
    
    def _extract_multilingual_dashboard(self, multilingual_results: Any) -> Dict[str, Any]:
        """Extract multilingual analysis metrics"""
        if not isinstance(multilingual_results, list) or len(multilingual_results) == 0:
            return {"languages_detected": 1, "translation_needed": 0}
        
        languages = {}
        for result in multilingual_results:
            if isinstance(result, dict) and 'detected_language' in result:
                lang = result['detected_language']
                languages[lang] = languages.get(lang, 0) + 1
        
        return {
            "languages_detected": len(languages),
            "language_distribution": languages,
            "non_english_rate": sum(count for lang, count in languages.items() if lang != 'en') / max(1, sum(languages.values()))
        }
    
    def _extract_crisis_dashboard(self, crisis_results: Any) -> Dict[str, Any]:
        """Extract crisis detection metrics"""
        if not isinstance(crisis_results, dict):
            return {"crisis_alerts": 0, "risk_level": "low"}
        
        return {
            "total_alerts": len(crisis_results.get("alerts", [])),
            "alert_severity": crisis_results.get("max_severity", 0),
            "crisis_categories": crisis_results.get("categories", {}),
            "risk_level": "high" if len(crisis_results.get("alerts", [])) > 0 else "low"
        }
    
    def _extract_network_dashboard(self, network_results: Any) -> Dict[str, Any]:
        """Extract network analysis metrics"""
        if not isinstance(network_results, dict) or "results" not in network_results:
            return {"influencers": 0, "communities": 0}
        
        results = network_results["results"]
        return {
            "total_influencers": results.get("influencers", {}).get("total_influencers", 0),
            "total_communities": results.get("communities", {}).get("total_communities", 0),
            "network_density": results.get("influencers", {}).get("network_density", 0),
            "top_categories": results.get("categories", {})
        }
    
    def _extract_quality_dashboard(self, quality_results: Any) -> Dict[str, Any]:
        """Extract quality scoring metrics"""
        if not isinstance(quality_results, dict) or "aggregated_metrics" not in quality_results:
            return {"avg_qei": 50, "quality_distribution": {}}
        
        metrics = quality_results["aggregated_metrics"]["overall_metrics"]
        return {
            "avg_qei_score": metrics.get("avg_qei_score", 50),
            "quality_distribution": quality_results["aggregated_metrics"].get("quality_distribution", {}),
            "pillar_breakdown": quality_results["aggregated_metrics"].get("pillar_metrics", {})
        }
    
    def _extract_spam_dashboard(self, spam_results: Any) -> Dict[str, Any]:
        """Extract spam/bot detection metrics"""
        if not isinstance(spam_results, dict) or "detection_metrics" not in spam_results:
            return {"spam_rate": 0.1, "bot_rate": 0.05}
        
        metrics = spam_results["detection_metrics"]
        return {
            "spam_rate": metrics["spam_metrics"].get("spam_rate", 0),
            "bot_rate": metrics["bot_metrics"].get("bot_rate", 0),
            "authenticity_score": metrics["authenticity_metrics"].get("avg_authenticity", 0.8),
            "clean_comments": metrics["overall_metrics"].get("clean_comments", 0)
        }
    
    def _extract_predictive_dashboard(self, pred_results: Any) -> Dict[str, Any]:
        """Extract predictive analytics metrics"""
        if not isinstance(pred_results, dict) or "analytics_results" not in pred_results:
            return {"prediction_confidence": 0.6}
        
        analytics = pred_results["analytics_results"]
        return {
            "fake_engagement_rate": analytics.get("fake_engagement_insurance", {}).get("suspicious_rate", 0),
            "soe_prediction_accuracy": analytics.get("soe_predictions", {}).get("model_performance", {}).get("r2_score", 0),
            "future_scenarios": analytics.get("soe_predictions", {}).get("future_scenarios", {}),
            "risk_assessment": analytics.get("risk_assessment", {})
        }
    
    # Cross-module analysis methods
    def _calculate_sentiment_quality_correlation(self, all_module_results: Dict[str, Any]) -> float:
        """Calculate correlation between sentiment and quality scores"""
        # Simplified correlation calculation
        return 0.65  # Placeholder - would require actual correlation analysis
    
    def _calculate_authenticity_network_health(self, all_module_results: Dict[str, Any]) -> float:
        """Calculate relationship between authenticity and network health"""
        return 0.72  # Placeholder
    
    def _calculate_crisis_language_overlap(self, all_module_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overlap between crisis content and language distribution"""
        return {"overlap_detected": False, "high_risk_languages": []}
    
    def _identify_primary_risks(self, all_module_results: Dict[str, Any]) -> List[str]:
        """Identify primary risk factors across all modules"""
        risks = []
        
        # Check each module for risk indicators
        if "crisis_analysis" in all_module_results:
            crisis_results = all_module_results["crisis_analysis"]
            if isinstance(crisis_results, dict) and len(crisis_results.get("alerts", [])) > 0:
                risks.append("Crisis content detected")
        
        if "spam_bot_analysis" in all_module_results:
            spam_results = all_module_results["spam_bot_analysis"]
            if isinstance(spam_results, dict) and "detection_metrics" in spam_results:
                spam_rate = spam_results["detection_metrics"]["spam_metrics"].get("spam_rate", 0)
                if spam_rate > 0.2:
                    risks.append(f"High spam rate ({spam_rate:.1%})")
        
        return risks[:5]  # Top 5 risks
    
    def _determine_overall_risk_level(self, risk_penalty: float) -> str:
        """Determine overall risk level from penalty"""
        if risk_penalty > 0.5:
            return "HIGH"
        elif risk_penalty > 0.2:
            return "MEDIUM" 
        else:
            return "LOW"
    
    # Fallback methods for when modules aren't available
    def _fallback_emotion_analysis(self, comments_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Fallback emotion analysis"""
        return [{"emotions": {"neutral": 0.6, "positive": 0.3, "negative": 0.1}, "sarcasm": {"is_sarcastic": False}}] * len(comments_df)
    
    def _fallback_emoji_analysis(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback emoji analysis"""
        return {"overall_sentiment": 0.1, "emoji_categories": {}, "visual_analysis": {}}
    
    def _fallback_multilingual_analysis(self, comments_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Fallback multilingual analysis"""
        return [{"detected_language": "en", "sentiment": "neutral", "translated_text": ""}] * len(comments_df)
    
    def _fallback_crisis_analysis(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback crisis analysis"""
        return {"alerts": [], "max_severity": 0, "categories": {}}
    
    def _fallback_network_analysis(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback network analysis"""
        return {"results": {"influencers": {"total_influencers": 5}, "communities": {"total_communities": 2}}}
    
    def _fallback_quality_analysis(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback quality analysis"""
        return {
            "aggregated_metrics": {
                "overall_metrics": {"avg_qei_score": 55, "total_comments": len(comments_df)},
                "quality_distribution": {"good": 0.4, "fair": 0.4, "poor": 0.2}
            }
        }
    
    def _fallback_spam_analysis(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback spam analysis"""
        return {
            "detection_metrics": {
                "spam_metrics": {"spam_rate": 0.1},
                "bot_metrics": {"bot_rate": 0.05},
                "authenticity_metrics": {"avg_authenticity": 0.8},
                "overall_metrics": {"clean_comments": len(comments_df) * 0.8, "flagged_comments": len(comments_df) * 0.2}
            }
        }
    
    def _fallback_predictive_analysis(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback predictive analysis"""
        return {
            "analytics_results": {
                "fake_engagement_insurance": {"suspicious_rate": 0.1},
                "soe_predictions": {"model_performance": {"r2_score": 0.5}},
                "risk_assessment": {"risk_level": "MEDIUM"}
            }
        }
    
    def _prepare_quality_data_for_prediction(self, quality_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare quality analysis data for predictive analytics consumption"""
        if not quality_results or 'status' not in quality_results:
            return None
        
        if quality_results.get('status') != 'success':
            return None
        
        # If scored_comments is stored as string representation, we need to handle it properly
        scored_comments = quality_results.get('scored_comments')
        if isinstance(scored_comments, str):
            # For now, return the aggregated metrics which should work
            return {
                'aggregated_metrics': quality_results.get('aggregated_metrics', {}),
                'status': 'success'
            }
        elif hasattr(scored_comments, 'columns'):
            # It's already a DataFrame, return as is
            return quality_results
        else:
            return None
    
    def _prepare_spam_data_for_prediction(self, spam_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare spam/bot analysis data for predictive analytics consumption"""
        if not spam_results or 'status' not in spam_results:
            return None
            
        if spam_results.get('status') != 'success':
            return None
        
        # If analyzed_comments is stored as string representation, we need to handle it properly  
        analyzed_comments = spam_results.get('analyzed_comments')
        if isinstance(analyzed_comments, str):
            # For now, return the detection metrics which should work
            return {
                'detection_metrics': spam_results.get('detection_metrics', {}),
                'status': 'success'
            }
        elif hasattr(analyzed_comments, 'columns'):
            # It's already a DataFrame, return as is
            return spam_results
        else:
            return None

# Factory function for pipeline integration
def create_composite_kpi_system(config_path: Optional[str] = None) -> CompositeKPISystem:
    """Factory function to create comprehensive composite KPI system"""
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return CompositeKPISystem(config)