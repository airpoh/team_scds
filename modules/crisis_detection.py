"""
Crisis Detection (Early Warning) System

Real-time detection of sudden negative sentiment spikes and risky content patterns.
Uses change-point detection algorithms to identify anomalous patterns in comment streams.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from .common import load_config

    
# Core libraries
try:
    import ruptures as rpt
    from scipy import stats
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    logging.warning(f"Optional dependency not available: {e}")
    rpt = None

# Optional efficient keyword search
try:
    import ahocorasick
    AHOCORASICK_AVAILABLE = True
except ImportError:
    AHOCORASICK_AVAILABLE = False
    logging.info("ahocorasick not available, falling back to basic string search")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CrisisAlert:
    """Data class for crisis alerts"""
    timestamp: datetime
    alert_type: str
    severity: float
    description: str
    affected_comment_ids: List[str]  # Store IDs instead of full text for privacy
    confidence: float
    video_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def get_affected_comments_count(self) -> int:
        """Get number of affected comments"""
        return len(self.affected_comment_ids)

class SentimentMonitor:
    """Monitors sentiment patterns and detects anomalies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['sentiment_monitoring']
        self.window_size = self.config.get('window_size', 50)
        self.change_point_model = self.config.get('change_point_model', 'rbf')
        self.min_size = self.config.get('min_segment_size', 10)
        
        # Sliding window for real-time monitoring
        self.sentiment_buffer = deque(maxlen=self.window_size * 2)
        self.timestamp_buffer = deque(maxlen=self.window_size * 2)
        
    def detect_sentiment_anomalies(self, sentiments: List[float], 
                                 timestamps: List[datetime]) -> List[CrisisAlert]:
        """Detect sudden changes in sentiment patterns"""
        alerts = []
        
        if rpt is None or len(sentiments) < self.min_size:
            return alerts
            
        try:
            # Convert sentiments to numpy array
            signal = np.array(sentiments)
            
            # Multiple change point detection algorithms
            algorithms = [
                ('rbf', rpt.Binseg(model='rbf')),
                ('l2', rpt.Binseg(model='l2')),
                ('normal', rpt.Binseg(model='normal'))
            ]
            
            for algo_name, algo in algorithms:
                try:
                    algo.fit(signal.reshape(-1, 1))
                    change_points = algo.predict(n_bkps=5)
                    
                    # Analyze segments between change points
                    prev_cp = 0
                    for cp in change_points:
                        if cp - prev_cp >= self.min_size:
                            segment_sentiments = signal[prev_cp:cp]
                            segment_timestamps = timestamps[prev_cp:cp]
                            
                            # Check for crisis patterns
                            crisis_alert = self._analyze_segment(
                                segment_sentiments, segment_timestamps, algo_name, 
                                comment_ids=[f"segment_{prev_cp}_{i}" for i in range(len(segment_sentiments))]
                            )
                            if crisis_alert:
                                alerts.append(crisis_alert)
                        
                        prev_cp = cp
                        
                except Exception as e:
                    logger.warning(f"Error in {algo_name} algorithm: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in sentiment anomaly detection: {e}")
            
        return alerts
    
    def _analyze_segment(self, sentiments: np.ndarray, 
                        timestamps: List[datetime], 
                        algorithm: str,
                        comment_ids: List[str] = None) -> Optional[CrisisAlert]:
        """Analyze a segment for crisis patterns"""
        
        mean_sentiment = np.mean(sentiments)
        std_sentiment = np.std(sentiments)
        trend = np.polyfit(range(len(sentiments)), sentiments, 1)[0]
        
        # Crisis detection criteria
        thresholds = self.config['thresholds']
        
        is_crisis = False
        alert_type = ""
        severity = 0.0
        description = ""
        
        # Sudden negative drop
        if mean_sentiment < thresholds['negative_threshold'] and abs(trend) > thresholds['trend_threshold']:
            is_crisis = True
            alert_type = "sentiment_drop"
            severity = min(1.0, abs(mean_sentiment) + abs(trend))
            description = f"Sudden negative sentiment drop detected (mean: {mean_sentiment:.3f}, trend: {trend:.3f})"
        
        # High volatility
        elif std_sentiment > thresholds['volatility_threshold']:
            is_crisis = True
            alert_type = "high_volatility"
            severity = min(1.0, std_sentiment)
            description = f"High sentiment volatility detected (std: {std_sentiment:.3f})"
        
        # Extreme negative values
        elif np.min(sentiments) < thresholds['extreme_negative']:
            is_crisis = True
            alert_type = "extreme_negative"
            severity = min(1.0, abs(np.min(sentiments)))
            description = f"Extreme negative sentiment detected (min: {np.min(sentiments):.3f})"
            
        if is_crisis:
            return CrisisAlert(
                timestamp=timestamps[-1] if timestamps else datetime.now(),
                alert_type=alert_type,
                severity=severity,
                description=description,
                affected_comment_ids=comment_ids or [],
                confidence=0.8,
                metadata={
                    'algorithm': algorithm,
                    'segment_size': len(sentiments),
                    'mean_sentiment': float(mean_sentiment),
                    'std_sentiment': float(std_sentiment),
                    'trend': float(trend)
                }
            )
            
        return None

class AlertManager(ABC):
    """Abstract base class for alert handling"""
    
    @abstractmethod
    def send_alert(self, alert: CrisisAlert) -> bool:
        """Send an alert through specific channel"""
        pass

class ConsoleAlertManager(AlertManager):
    """Console-based alert manager with rate limiting"""
    
    def __init__(self):
        self.sent_alerts = set()  # Track sent alerts to prevent duplicates
        self.last_alert_time = {}  # Track last time each type was sent
        self.min_interval = 300  # Minimum 5 minutes between identical alerts
    
    def send_alert(self, alert: CrisisAlert) -> bool:
        try:
            # Create a unique key for this alert
            alert_key = f"{alert.alert_type}_{alert.severity:.1f}"
            current_time = datetime.now()
            
            # Check if we've sent this type of alert recently
            if alert_key in self.last_alert_time:
                time_since_last = (current_time - self.last_alert_time[alert_key]).total_seconds()
                if time_since_last < self.min_interval:
                    return False  # Skip this alert
            
            # Log instead of print to reduce spam
            logger.warning(f"CRISIS ALERT [{alert.alert_type}]: {alert.description}")
            logger.info(f"Severity: {alert.severity:.3f} | Confidence: {alert.confidence:.3f}")
            
            # Update tracking
            self.last_alert_time[alert_key] = current_time
            return True
            
        except Exception as e:
            logger.error(f"Failed to send console alert: {e}")
            return False

class KeywordRiskDetector:
    """Detects risky keywords and content patterns"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['keyword_detection']
        self.risk_keywords = self._load_risk_keywords()
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.get('max_features', 1000),
            ngram_range=tuple(self.config.get('ngram_range', [1, 2]))
        )
        self._setup_keyword_automaton()
        
    def _load_risk_keywords(self) -> Dict[str, List[str]]:
        """Load risk keywords from config"""
        try:
            return {
                'violence': self.config['keywords']['violence'],
                'hate_speech': self.config['keywords']['hate_speech'],
                'self_harm': self.config['keywords']['self_harm'],
                'harassment': self.config['keywords']['harassment'],
                'extremism': self.config['keywords']['extremism'],
                'toxic_behavior': self.config['keywords'].get('toxic_behavior', [])
            }
        except KeyError as e:
            logger.error(f"Missing keyword category in config: {e}")
            return {}
    
    def _setup_keyword_automaton(self):
        """Setup Aho-Corasick automaton for efficient keyword search"""
        self.automaton = None
        if AHOCORASICK_AVAILABLE:
            try:
                self.automaton = ahocorasick.Automaton()
                keyword_id = 0
                
                for category, keywords in self.risk_keywords.items():
                    for keyword in keywords:
                        self.automaton.add_word(keyword.lower(), (keyword_id, category, keyword))
                        keyword_id += 1
                
                self.automaton.make_automaton()
                logger.info("Aho-Corasick automaton initialized for efficient keyword search")
            except Exception as e:
                logger.warning(f"Failed to setup automaton: {e}")
                self.automaton = None
    
    def detect_risk_patterns(self, comments: List[str], 
                           timestamps: List[datetime],
                           comment_ids: List[str] = None) -> List[CrisisAlert]:
        """Detect risky keyword patterns and clusters"""
        alerts = []
        
        if not comments:
            return alerts
        
        if comment_ids is None:
            comment_ids = [f"comment_{i}" for i in range(len(comments))]
            
        try:
            # Keyword-based detection
            keyword_alerts = self._detect_keyword_spikes(comments, timestamps, comment_ids)
            alerts.extend(keyword_alerts)
            
            # Cluster-based detection using TF-IDF
            cluster_alerts = self._detect_content_clusters(comments, timestamps, comment_ids)
            alerts.extend(cluster_alerts)
            
        except Exception as e:
            logger.error(f"Error in risk pattern detection: {e}")
            
        return alerts
    
    def _detect_keyword_spikes(self, comments: List[str], 
                              timestamps: List[datetime],
                              comment_ids: List[str] = None) -> List[CrisisAlert]:
        """Detect sudden spikes in risky keywords using efficient search"""
        alerts = []
        
        if comment_ids is None:
            comment_ids = [f"comment_{i}" for i in range(len(comments))]
        
        # Use efficient keyword counting
        if self.automaton:
            category_counts_per_comment = self._count_keywords_efficient(comments)
        else:
            category_counts_per_comment = self._count_keywords_basic(comments)
        
        for category in self.risk_keywords.keys():
            category_counts = []
            
            # Count keywords in sliding windows
            window_size = self.config.get('keyword_window_size', 20)
            for i in range(len(comments) - window_size + 1):
                window_count = sum(
                    category_counts_per_comment[j].get(category, 0)
                    for j in range(i, i + window_size)
                )
                category_counts.append(window_count)
            
            if not category_counts or len(category_counts) < 5:
                continue
                
            # Detect spikes using z-score
            try:
                z_scores = np.abs(stats.zscore(category_counts))
                spike_threshold = self.config['thresholds']['keyword_spike_zscore']
                
                spike_indices = np.where(z_scores > spike_threshold)[0]
                for idx in spike_indices:
                    affected_ids = comment_ids[idx:idx + window_size]
                    alert = CrisisAlert(
                        timestamp=timestamps[min(idx + window_size - 1, len(timestamps) - 1)],
                        alert_type=f"keyword_spike_{category}",
                        severity=min(1.0, z_scores[idx] / 10.0),
                        description=f"Spike in {category} keywords detected (z-score: {z_scores[idx]:.2f})",
                        affected_comment_ids=affected_ids,
                        confidence=0.7,
                        metadata={
                            'category': category,
                            'keyword_count': int(category_counts[idx]),
                            'z_score': float(z_scores[idx])
                        }
                    )
                    alerts.append(alert)
            except Exception as e:
                logger.warning(f"Error in spike detection for {category}: {e}")
                continue
        
        return alerts
    
    def _count_keywords_efficient(self, comments: List[str]) -> List[Dict[str, int]]:
        """Count keywords efficiently using Aho-Corasick"""
        results = [defaultdict(int) for _ in comments]
        
        for i, comment in enumerate(comments):
            comment_lower = comment.lower()
            for end_index, (keyword_id, category, keyword) in self.automaton.iter(comment_lower):
                results[i][category] += 1
        
        return results
    
    def _count_keywords_basic(self, comments: List[str]) -> List[Dict[str, int]]:
        """Count keywords using basic string search"""
        results = [defaultdict(int) for _ in comments]
        
        for i, comment in enumerate(comments):
            comment_lower = comment.lower()
            for category, keywords in self.risk_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in comment_lower:
                        results[i][category] += 1
        
        return results
    
    def _detect_content_clusters(self, comments: List[str], 
                                timestamps: List[datetime],
                                comment_ids: List[str]) -> List[CrisisAlert]:
        """Detect clusters of similar problematic content"""
        alerts = []
        
        try:
            if len(comments) < 10:
                return alerts
                
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(comments)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find highly similar comment clusters
            similarity_threshold = self.config['thresholds']['similarity_threshold']
            cluster_size_threshold = self.config['thresholds']['min_cluster_size']
            
            visited = set()
            for i in range(len(comments)):
                if i in visited:
                    continue
                    
                # Find similar comments
                similar_indices = np.where(similarity_matrix[i] > similarity_threshold)[0]
                
                if len(similar_indices) >= cluster_size_threshold:
                    # Mark as visited
                    visited.update(similar_indices)
                    
                    # Check if cluster contains risky content
                    cluster_comments = [comments[idx] for idx in similar_indices]
                    cluster_comment_ids = [comment_ids[idx] for idx in similar_indices]
                    risk_score = self._calculate_cluster_risk(cluster_comments)
                    
                    if risk_score > self.config['thresholds']['cluster_risk_threshold']:
                        alert = CrisisAlert(
                            timestamp=timestamps[similar_indices[-1]],
                            alert_type="content_cluster",
                            severity=min(1.0, risk_score),
                            description=f"Risky content cluster detected ({len(similar_indices)} similar comments)",
                            affected_comment_ids=cluster_comment_ids,
                            confidence=0.6,
                            metadata={
                                'cluster_size': len(similar_indices),
                                'risk_score': float(risk_score),
                                'similarity_threshold': similarity_threshold
                            }
                        )
                        alerts.append(alert)
                        
        except Exception as e:
            logger.warning(f"Error in content cluster detection: {e}")
            
        return alerts
    
    def _calculate_cluster_risk(self, comments: List[str]) -> float:
        """Calculate risk score for a cluster of comments"""
        risk_score = 0.0
        total_keywords = 0
        
        for comment in comments:
            comment_lower = comment.lower()
            for category, keywords in self.risk_keywords.items():
                category_weight = self.config['category_weights'].get(category, 1.0)
                for keyword in keywords:
                    if keyword.lower() in comment_lower:
                        risk_score += category_weight
                        total_keywords += 1
        
        # Normalize by cluster size
        if len(comments) > 0:
            risk_score = (risk_score / len(comments)) * min(1.0, total_keywords / 10.0)
            
        return risk_score

class CrisisDetectionSystem:
    """Main crisis detection system combining all detection methods"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Use the enhanced load_config from common module
        if isinstance(config_path, dict):
            # Config passed as dict directly
            self.config = load_config(config_path, self._get_default_config_dict())
            self.config_path = None
        else:
            # Config path passed as string/Path or None
            self.config_path = config_path or Path(__file__).parent.parent / "config" / "crisis_config.json"
            self.config = load_config(self.config_path, self._get_default_config_dict())
        
        # Initialize components
        self.sentiment_monitor = SentimentMonitor(self.config)
        self.keyword_detector = KeywordRiskDetector(self.config)
        
        # Alert management
        self.alert_history = deque(maxlen=self.config['system']['max_alert_history'])
        self.alert_cooldown = {}
        self.alert_manager = ConsoleAlertManager()  # Default alert manager
        
    def _get_default_config_dict(self) -> Dict[str, Any]:
        """Get default configuration as dictionary"""
        return {
            "sentiment_monitoring": {
                "window_size": 50,
                "change_point_model": "rbf", 
                "min_segment_size": 10,
                "thresholds": {
                    "negative_threshold": -0.6,
                    "trend_threshold": 0.1,
                    "volatility_threshold": 0.4,
                    "extreme_negative": -0.8
                }
            },
            "keyword_detection": {
                "max_features": 1000,
                "ngram_range": [1, 2],
                "keyword_window_size": 20,
                "keywords": {
                    "violence": ["kill", "die", "hurt", "attack", "fight"],
                    "hate_speech": ["hate", "stupid", "idiot", "worthless"], 
                    "self_harm": ["suicide", "kill myself", "end it"],
                    "harassment": ["harass", "bully", "threaten", "stalk"],
                    "extremism": ["radical", "extreme", "fanatic"],
                    "toxic_behavior": ["toxic", "cancel", "brigade"]
                },
                "category_weights": {
                    "violence": 3.0,
                    "hate_speech": 2.0, 
                    "self_harm": 4.0,
                    "harassment": 2.5,
                    "extremism": 3.5,
                    "toxic_behavior": 1.8
                },
                "thresholds": {
                    "keyword_spike_zscore": 2.0,
                    "similarity_threshold": 0.7,
                    "min_cluster_size": 3,
                    "cluster_risk_threshold": 0.5
                }
            },
            "system": {
                "max_alert_history": 1000,
                "alert_cooldown_minutes": 30,
                "min_severity_threshold": 0.3,
                "enable_clustering": True,
                "enable_trend_analysis": True,
                "data_retention_days": 30
            }
        }
    
    
    
    def batch_analyze(self, comments: List[str], batch_size: int = None) -> Dict[str, Any]:
        """
        Analyze multiple comments for crisis patterns with batch processing support.
        
        This method provides interface consistency with other analysis modules.
        
        Args:
            comments: List of comment texts to analyze
            batch_size: Number of comments to process per batch (unused for crisis detection)
            
        Returns:
            Dictionary containing crisis analysis results
        """
        # Convert comments list to DataFrame format expected by analyze_crisis_patterns
        comments_df = pd.DataFrame({
            'textOriginal': comments,
            'publishedAt': [datetime.now() - timedelta(minutes=i) for i in range(len(comments))]
        })
        
        return self.analyze_crisis_patterns(comments_df)
    
    def analyze_crisis_patterns(self, comments_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze crisis patterns in comment data"""
        
        if comments_data.empty:
            return {"alerts": [], "summary": {"total_alerts": 0}}
        
        # Prepare data
        comments = comments_data['textOriginal'].fillna('').tolist()
        
        # Handle timestamps
        if 'publishedAt' in comments_data.columns:
            timestamps = pd.to_datetime(comments_data['publishedAt'], errors='coerce')
            timestamps = timestamps.fillna(datetime.now()).tolist()
        else:
            # Create synthetic timestamps
            timestamps = [datetime.now() - timedelta(minutes=i) for i in range(len(comments))]
        
        # Get sentiment scores (placeholder - would integrate with sentiment module)
        sentiments = self._get_sentiment_scores(comments)
        
        all_alerts = []
        
        try:
            # Sentiment-based detection
            sentiment_alerts = self.sentiment_monitor.detect_sentiment_anomalies(
                sentiments, timestamps
            )
            all_alerts.extend(sentiment_alerts)
            
            # Generate comment IDs for privacy
            comment_ids = [f"comment_{i}_{hash(comment) % 10000}" for i, comment in enumerate(comments)]
            
            # Keyword-based detection
            keyword_alerts = self.keyword_detector.detect_risk_patterns(
                comments, timestamps, comment_ids
            )
            all_alerts.extend(keyword_alerts)
            
            # Filter and deduplicate alerts
            filtered_alerts = self._filter_alerts(all_alerts)
            
            # Send alerts through alert manager
            for alert in filtered_alerts:
                try:
                    self.alert_manager.send_alert(alert)
                except Exception as e:
                    logger.error(f"Failed to send alert: {e}")
            
            # Add to history
            self.alert_history.extend(filtered_alerts)
            
        except Exception as e:
            logger.error(f"Error in crisis pattern analysis: {e}")
            filtered_alerts = []
        
        # Generate summary
        summary = self._generate_summary(filtered_alerts, comments_data)
        
        return {
            "alerts": [self._alert_to_dict(alert) for alert in filtered_alerts],
            "summary": summary,
            "total_comments_analyzed": len(comments),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _get_sentiment_scores(self, comments: List[str]) -> List[float]:
        """Get sentiment scores for comments (placeholder for integration)"""
        # This would integrate with the emotion_sarcasm_detection module
        # For now, return random scores for demonstration
        np.random.seed(42)
        return np.random.normal(0.0, 0.3, len(comments)).tolist()
    
    def _filter_alerts(self, alerts: List[CrisisAlert]) -> List[CrisisAlert]:
        """Filter alerts based on severity and cooldown"""
        filtered = []
        current_time = datetime.now()
        min_severity = self.config['system']['min_severity_threshold']
        cooldown_minutes = self.config['system']['alert_cooldown_minutes']
        
        for alert in alerts:
            # Check severity threshold
            if alert.severity < min_severity:
                continue
            
            # Check cooldown with more granular key to prevent spam
            cooldown_key = f"{alert.alert_type}_{alert.severity:.1f}"
            if cooldown_key in self.alert_cooldown:
                last_alert_time = self.alert_cooldown[cooldown_key]
                if (current_time - last_alert_time).total_seconds() < cooldown_minutes * 60:
                    continue
            
            # Add to filtered alerts and update cooldown
            filtered.append(alert)
            self.alert_cooldown[cooldown_key] = current_time
            
        return sorted(filtered, key=lambda x: x.severity, reverse=True)
    
    def _alert_to_dict(self, alert: CrisisAlert) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "timestamp": alert.timestamp.isoformat(),
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "description": alert.description,
            "confidence": alert.confidence,
            "affected_comments_count": alert.get_affected_comments_count(),
            "video_id": alert.video_id,
            "metadata": alert.metadata or {}
        }
    
    def _generate_summary(self, alerts: List[CrisisAlert], 
                         comments_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate analysis summary"""
        summary = {
            "total_alerts": len(alerts),
            "high_severity_alerts": len([a for a in alerts if a.severity > 0.7]),
            "alert_types": defaultdict(int),
            "max_severity": 0.0,
            "risk_level": "low"
        }
        
        for alert in alerts:
            summary["alert_types"][alert.alert_type] += 1
            summary["max_severity"] = max(summary["max_severity"], alert.severity)
        
        # Determine overall risk level
        if summary["max_severity"] > 0.8 or summary["high_severity_alerts"] > 5:
            summary["risk_level"] = "high"
        elif summary["max_severity"] > 0.5 or summary["total_alerts"] > 2:
            summary["risk_level"] = "medium"
        
        return dict(summary)

def main():
    """Test the crisis detection system"""
    
    # Create sample data
    sample_comments = [
        "This video is great!",
        "I hate this so much",
        "This is terrible, I want to die",
        "Amazing content, love it!",
        "This makes me angry",
        "Worst video ever, complete garbage",
        "Beautiful work, thank you!",
        "I'm so depressed after watching this",
        "Fantastic job!",
        "This is disgusting and awful"
    ]
    
    df = pd.DataFrame({
        'textOriginal': sample_comments,
        'publishedAt': [datetime.now() - timedelta(minutes=i) for i in range(len(sample_comments))]
    })
    
    # Test crisis detection
    detector = CrisisDetectionSystem()
    results = detector.analyze_crisis_patterns(df)
    
    print("Crisis Detection Results:")
    print(f"Total alerts: {results['summary']['total_alerts']}")
    print(f"Risk level: {results['summary']['risk_level']}")
    
    for alert in results['alerts']:
        print(f"\nAlert: {alert['alert_type']}")
        print(f"Severity: {alert['severity']:.3f}")
        print(f"Description: {alert['description']}")

if __name__ == "__main__":
    main()