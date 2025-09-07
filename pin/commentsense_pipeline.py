"""
CommentSense: Advanced YouTube Comment Analysis Pipeline

Integrates all analysis modules to provide comprehensive comment insights:
- Emotion & Sarcasm Detection
- Visual & Emoji Signals Analysis  
- Cross-Language Comment Analysis
- Crisis Detection (Early Warning)
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
warnings.filterwarnings("ignore")

# Data handling
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None
    logging.error("pandas and numpy are required for the pipeline")

# Import all modules
from modules.emotion_sarcasm_detection import EmotionSarcasmDetector
from modules.visual_emoji_analysis import VisualEmojiAnalyzer
from modules.multilingual_analysis import MultilingualSentimentAnalyzer
from modules.crisis_detection import CrisisDetectionSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Worker functions for parallel processing (must be at module level for pickling)
def run_emotion_sarcasm_worker(comments_list):
    """Worker function for emotion sarcasm detection"""
    try:
        from modules.emotion_sarcasm_detection import EmotionSarcasmDetector
        detector = EmotionSarcasmDetector()
        return detector.batch_analyze(comments_list)
    except Exception as e:
        logger.error(f"Error in emotion_sarcasm worker: {e}")
        return {"error": str(e)}

def run_visual_emoji_worker(comments_list):
    """Worker function for visual emoji analysis"""
    try:
        from modules.visual_emoji_analysis import VisualEmojiAnalyzer
        analyzer = VisualEmojiAnalyzer()
        return analyzer.batch_analyze(comments_list)
    except Exception as e:
        logger.error(f"Error in visual_emoji worker: {e}")
        return {"error": str(e)}

def run_multilingual_worker(comments_df_dict):
    """Worker function for multilingual analysis"""
    try:
        import pandas as pd
        from modules.multilingual_analysis import MultilingualSentimentAnalyzer
        analyzer = MultilingualSentimentAnalyzer()
        comments_df = pd.DataFrame(comments_df_dict)
        # Use the text column for multilingual analysis
        texts = comments_df['textOriginal'].fillna('').tolist()
        return analyzer.batch_analyze(texts)
    except Exception as e:
        logger.error(f"Error in multilingual worker: {e}")
        return {"error": str(e)}

def run_crisis_detection_worker(comments_df_dict):
    """Worker function for crisis detection"""
    try:
        import pandas as pd
        from modules.crisis_detection import CrisisDetectionSystem
        detector = CrisisDetectionSystem()
        comments_df = pd.DataFrame(comments_df_dict)
        return detector.analyze_crisis_patterns(comments_df)
    except Exception as e:
        logger.error(f"Error in crisis_detection worker: {e}")
        return {"error": str(e)}

class CommentSensePipeline:
    """Main pipeline integrating all analysis modules"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the CommentSense pipeline"""
        
        self.config_path = config_path or Path(__file__).parent / "config" / "pipeline_config.json"
        self.config = self._load_config()
        
        # Initialize all analysis modules
        self.modules = {}
        self._initialize_modules()
        
        # Pipeline state
        self.processing_stats = {
            'total_comments_processed': 0,
            'processing_time': 0.0,
            'last_processed': None,
            'errors': []
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return {
            "modules": {
                "emotion_sarcasm": {"enabled": True},
                "visual_emoji": {"enabled": True},
                "multilingual": {"enabled": True},
                "crisis_detection": {"enabled": True}
            },
            "processing": {
                "batch_size": 100,
                "max_workers": 4,
                "enable_parallel": True,
                "timeout_seconds": 300
            },
            "output": {
                "save_results": True,
                "results_directory": "results",
                "format": "json",
                "include_raw_data": False
            },
            "dashboard": {
                "enabled": True,
                "port": 8501,
                "theme": "dark"
            }
        }
    
    def _initialize_modules(self):
        """Initialize analysis modules based on configuration"""
        
        module_config = self.config.get('modules', {})
        
        try:
            # Emotion & Sarcasm Detection
            if module_config.get('emotion_sarcasm', {}).get('enabled', True):
                self.modules['emotion_sarcasm'] = EmotionSarcasmDetector()
                logger.info("Emotion & Sarcasm Detection module initialized")
            
            # Visual & Emoji Analysis
            if module_config.get('visual_emoji', {}).get('enabled', True):
                self.modules['visual_emoji'] = VisualEmojiAnalyzer()
                logger.info("Visual & Emoji Analysis module initialized")
            
            # Multilingual Analysis
            if module_config.get('multilingual', {}).get('enabled', True):
                self.modules['multilingual'] = MultilingualSentimentAnalyzer()
                logger.info("Multilingual Analysis module initialized")
            
            # Crisis Detection
            if module_config.get('crisis_detection', {}).get('enabled', True):
                self.modules['crisis_detection'] = CrisisDetectionSystem()
                logger.info("Crisis Detection module initialized")
                
        except Exception as e:
            logger.error(f"Error initializing modules: {e}")
            self.processing_stats['errors'].append(f"Module initialization error: {e}")
    
    def analyze_comments(self, comments_df: pd.DataFrame, 
                        videos_df: Optional[pd.DataFrame] = None,
                        config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze comments using all enabled modules
        
        Args:
            comments_df: DataFrame with comment data
            videos_df: Optional DataFrame with video metadata
            
        Returns:
            Dictionary containing results from all modules
        """
        
        if comments_df.empty:
            return {"error": "No comments to analyze"}
        
        start_time = datetime.now()
        results = {
            "pipeline_info": {
                "timestamp": start_time.isoformat(),
                "total_comments": len(comments_df),
                "enabled_modules": list(self.modules.keys())
            },
            "module_results": {},
            "aggregated_insights": {}
        }
        
        # Apply configuration override if provided
        effective_config = self.config.copy()
        if config_override:
            effective_config = self._merge_config_override(effective_config, config_override)
        
        # Process comments in batches if configured
        batch_size = effective_config.get('processing', {}).get('batch_size', 100)
        enable_parallel = effective_config.get('processing', {}).get('enable_parallel', True)
        
        try:
            if enable_parallel and len(comments_df) > batch_size:
                results['module_results'] = self._process_parallel(comments_df, videos_df)
            else:
                results['module_results'] = self._process_sequential(comments_df, videos_df)
            
            # Generate aggregated insights
            results['aggregated_insights'] = self._generate_insights(
                results['module_results'], comments_df, videos_df
            )
            
            # Update processing stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_stats.update({
                'total_comments_processed': self.processing_stats['total_comments_processed'] + len(comments_df),
                'processing_time': self.processing_stats['processing_time'] + processing_time,
                'last_processed': datetime.now().isoformat()
            })
            
            results['pipeline_info']['processing_time'] = processing_time
            
        except Exception as e:
            logger.error(f"Error in comment analysis: {e}")
            results['error'] = str(e)
            self.processing_stats['errors'].append(f"Analysis error: {e}")
        
        # Save results if configured
        if self.config.get('output', {}).get('save_results', True):
            self._save_results(results)
        
        return results
    
    def _process_sequential(self, comments_df: pd.DataFrame, 
                          videos_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Process comments sequentially through all modules"""
        
        module_results = {}
        
        # Emotion & Sarcasm Detection
        if 'emotion_sarcasm' in self.modules:
            try:
                logger.info("Running emotion & sarcasm detection...")
                emotion_results = self.modules['emotion_sarcasm'].batch_analyze(
                    comments_df['textOriginal'].fillna('').tolist()
                )
                module_results['emotion_sarcasm'] = emotion_results
            except Exception as e:
                logger.error(f"Emotion & Sarcasm module error: {e}")
                module_results['emotion_sarcasm'] = {"error": str(e)}
        
        # Visual & Emoji Analysis
        if 'visual_emoji' in self.modules:
            try:
                logger.info("Running visual & emoji analysis...")
                emoji_results = self.modules['visual_emoji'].batch_analyze(
                    comments_df['textOriginal'].fillna('').tolist()
                )
                module_results['visual_emoji'] = emoji_results
            except Exception as e:
                logger.error(f"Visual & Emoji module error: {e}")
                module_results['visual_emoji'] = {"error": str(e)}
        
        # Multilingual Analysis
        if 'multilingual' in self.modules:
            try:
                logger.info("Running multilingual analysis...")
                multilingual_results = self.modules['multilingual'].batch_analyze(
                    comments_df['textOriginal'].fillna('').tolist()
                )
                module_results['multilingual'] = multilingual_results
            except Exception as e:
                logger.error(f"Multilingual module error: {e}")
                module_results['multilingual'] = {"error": str(e)}
        
        # Crisis Detection
        if 'crisis_detection' in self.modules:
            try:
                logger.info("Running crisis detection...")
                crisis_results = self.modules['crisis_detection'].analyze_crisis_patterns(
                    comments_df
                )
                module_results['crisis_detection'] = crisis_results
            except Exception as e:
                logger.error(f"Crisis Detection module error: {e}")
                module_results['crisis_detection'] = {"error": str(e)}
        
        return module_results
    
    def _process_parallel(self, comments_df: pd.DataFrame, 
                         videos_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Process comments in parallel using ThreadPoolExecutor"""
        
        max_workers = self.config.get('processing', {}).get('max_workers', 4)
        module_results = {}
        
        # Prepare data for worker functions
        comments_list = comments_df['textOriginal'].fillna('').tolist()
        comments_df_dict = comments_df.to_dict('records')
        
        # Use ThreadPoolExecutor for better compatibility (avoids pickle issues)
        # ProcessPoolExecutor requires more complex data serialization
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            # Submit tasks using module-level worker functions
            if 'emotion_sarcasm' in self.modules:
                futures['emotion_sarcasm'] = executor.submit(run_emotion_sarcasm_worker, comments_list)
            
            if 'visual_emoji' in self.modules:
                futures['visual_emoji'] = executor.submit(run_visual_emoji_worker, comments_list)
            
            if 'multilingual' in self.modules:
                futures['multilingual'] = executor.submit(run_multilingual_worker, comments_df_dict)
            
            if 'crisis_detection' in self.modules:
                futures['crisis_detection'] = executor.submit(run_crisis_detection_worker, comments_df_dict)
            
            # Collect results
            for module_name, future in futures.items():
                try:
                    module_results[module_name] = future.result(
                        timeout=self.config.get('processing', {}).get('timeout_seconds', 300)
                    )
                except Exception as e:
                    logger.error(f"{module_name} module error: {e}")
                    module_results[module_name] = {"error": str(e)}
        
        return module_results
    
    def _generate_insights(self, module_results: Dict[str, Any], 
                          comments_df: pd.DataFrame,
                          videos_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Generate aggregated insights from all module results"""
        
        insights = {
            "overall_sentiment": self._calculate_overall_sentiment(module_results),
            "dominant_emotions": self._extract_dominant_emotions(module_results),
            "language_distribution": self._get_language_distribution(module_results),
            "crisis_summary": self._summarize_crisis_alerts(module_results),
            "emoji_sentiment": self._summarize_emoji_sentiment(module_results),
            "recommendations": self._generate_recommendations(module_results)
        }
        
        return insights
    
    def _calculate_overall_sentiment(self, module_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall sentiment from multiple sources"""
        
        sentiments = []
        
        # From emotion module
        if 'emotion_sarcasm' in module_results and isinstance(module_results['emotion_sarcasm'], list):
            emotion_data = module_results['emotion_sarcasm']
            for result in emotion_data:
                if isinstance(result, dict) and 'emotions' in result:
                    emotion_scores = result['emotions']
                    if isinstance(emotion_scores, dict):
                        positive_emotions = ['joy', 'love', 'surprise', 'optimism']
                        negative_emotions = ['anger', 'fear', 'sadness', 'disgust', 'pessimism']
                        
                        pos_score = sum(emotion_scores.get(e, 0) for e in positive_emotions)
                        neg_score = sum(emotion_scores.get(e, 0) for e in negative_emotions)
                        
                        if pos_score + neg_score > 0:
                            sentiment = (pos_score - neg_score) / (pos_score + neg_score)
                            sentiments.append(sentiment)
        
        # From multilingual module
        if 'multilingual' in module_results and isinstance(module_results['multilingual'], list):
            multilingual_data = module_results['multilingual']
            for result in multilingual_data:
                if isinstance(result, dict) and 'sentiment' in result:
                    # Convert sentiment labels to scores: positive=+1, negative=-1, neutral=0
                    sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
                    sentiment_label = result['sentiment']
                    if sentiment_label in sentiment_mapping:
                        sentiments.append(sentiment_mapping[sentiment_label])
                elif isinstance(result, dict) and 'scores' in result:
                    # Use score differences if available
                    scores = result['scores']
                    if 'positive' in scores and 'negative' in scores:
                        sentiment_score = scores['positive'] - scores['negative']
                        sentiments.append(sentiment_score)
        
        # From emoji module
        if 'visual_emoji' in module_results and 'overall_sentiment' in module_results['visual_emoji']:
            emoji_sentiment = module_results['visual_emoji']['overall_sentiment']
            if emoji_sentiment is not None:
                sentiments.append(emoji_sentiment)
        
        if sentiments:
            return {
                "mean": float(np.mean(sentiments)),
                "std": float(np.std(sentiments)),
                "min": float(np.min(sentiments)),
                "max": float(np.max(sentiments))
            }
        
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    def _extract_dominant_emotions(self, module_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract dominant emotions across all comments"""
        
        emotion_totals = {}
        
        if 'emotion_sarcasm' in module_results and isinstance(module_results['emotion_sarcasm'], list):
            emotions_data = module_results['emotion_sarcasm']
            
            for result in emotions_data:
                if isinstance(result, dict) and 'emotions' in result:
                    emotion_scores = result['emotions']
                    if isinstance(emotion_scores, dict):
                        for emotion, score in emotion_scores.items():
                            emotion_totals[emotion] = emotion_totals.get(emotion, 0) + score
        
        # Normalize by number of comments
        if emotion_totals:
            num_comments = len(module_results['emotion_sarcasm'])
            return {emotion: score / num_comments for emotion, score in emotion_totals.items()}
        
        return {}
    
    def _get_language_distribution(self, module_results: Dict[str, Any]) -> Dict[str, int]:
        """Get language distribution from multilingual analysis"""
        
        if 'multilingual' in module_results and isinstance(module_results['multilingual'], list):
            multilingual_data = module_results['multilingual']
            lang_counts = {}
            
            for result in multilingual_data:
                if isinstance(result, dict) and 'language' in result:
                    lang = result['language']
                    if lang:
                        lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
            return lang_counts
        
        return {}
    
    def _summarize_crisis_alerts(self, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize crisis detection alerts"""
        
        if 'crisis_detection' in module_results and 'summary' in module_results['crisis_detection']:
            return module_results['crisis_detection']['summary']
        
        return {"total_alerts": 0, "risk_level": "low"}
    
    def _summarize_emoji_sentiment(self, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize emoji sentiment analysis"""
        
        if 'visual_emoji' in module_results and isinstance(module_results['visual_emoji'], list):
            visual_emoji_data = module_results['visual_emoji']
            
            total_sentiment = 0.0
            total_positive = 0
            total_negative = 0
            count = 0
            
            for result in visual_emoji_data:
                if isinstance(result, dict) and 'emoji_analysis' in result:
                    emoji_data = result['emoji_analysis']
                    if isinstance(emoji_data, dict):
                        # Aggregate sentiment scores
                        if 'emoji_sentiment_score' in emoji_data:
                            total_sentiment += emoji_data['emoji_sentiment_score']
                            count += 1
                        
                        # Count emoji sentiment distribution
                        if 'emoji_distribution' in emoji_data:
                            dist = emoji_data['emoji_distribution']
                            if isinstance(dist, dict):
                                total_positive += dist.get('positive', 0)
                                total_negative += dist.get('negative', 0)
            
            return {
                "overall_sentiment": total_sentiment / count if count > 0 else 0.0,
                "positive_emojis_count": total_positive,
                "negative_emojis_count": total_negative
            }
        
        return {"overall_sentiment": 0.0, "positive_emojis_count": 0, "negative_emojis_count": 0}
    
    def _generate_recommendations(self, module_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Check crisis alerts
        crisis_summary = self._summarize_crisis_alerts(module_results)
        if crisis_summary['total_alerts'] > 0:
            recommendations.append(f" {crisis_summary['total_alerts']} crisis alerts detected. Review immediately.")
        
        if crisis_summary['risk_level'] == 'high':
            recommendations.append(" High risk level detected. Consider content moderation.")
        
        # Check overall sentiment
        overall_sentiment = self._calculate_overall_sentiment(module_results)
        if overall_sentiment['mean'] < -0.3:
            recommendations.append(" Overall negative sentiment detected. Engage with community.")
        
        # Check language diversity
        lang_dist = self._get_language_distribution(module_results)
        if len(lang_dist) > 1:
            recommendations.append(f" Comments in {len(lang_dist)} languages detected. Consider multilingual responses.")
        
        # Check sarcasm levels
        if 'emotion_sarcasm' in module_results and isinstance(module_results['emotion_sarcasm'], list):
            emotion_data = module_results['emotion_sarcasm']
            high_sarcasm = 0
            for result in emotion_data:
                if isinstance(result, dict) and 'is_sarcastic' in result:
                    if result['is_sarcastic']:
                        high_sarcasm += 1
                elif isinstance(result, dict) and 'sarcasm_confidence' in result:
                    if result['sarcasm_confidence'] > 0.7:
                        high_sarcasm += 1
            if high_sarcasm > 0:
                recommendations.append(f" {high_sarcasm} highly sarcastic comments detected. Context may be important.")
        
        if not recommendations:
            recommendations.append(" No immediate concerns detected. Continue monitoring.")
        
        return recommendations
    
    def _save_results(self, results: Dict[str, Any]):
        """Save analysis results to file"""
        
        try:
            output_config = self.config.get('output', {})
            results_dir = Path(output_config.get('results_directory', 'results'))
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"commentsense_analysis_{timestamp}.json"
            filepath = results_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get pipeline processing statistics"""
        return self.processing_stats.copy()
    
    def _merge_config_override(self, base_config: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration override with base configuration"""
        import copy
        result = copy.deepcopy(base_config)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config_override(result[key], value)
            else:
                result[key] = value
                
        return result

def load_datasets(comments_files: List[str], videos_file: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load and combine comment datasets"""
    
    if not pd:
        raise ImportError("pandas is required to load datasets")
    
    all_comments = []
    
    for file_path in comments_files:
        try:
            df = pd.read_csv(file_path)
            all_comments.append(df)
            logger.info(f"Loaded {len(df)} comments from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    if not all_comments:
        raise ValueError("No comment files could be loaded")
    
    # Combine all comment dataframes
    comments_df = pd.concat(all_comments, ignore_index=True)
    
    # Load videos if provided
    videos_df = None
    if videos_file:
        try:
            videos_df = pd.read_csv(videos_file)
            logger.info(f"Loaded {len(videos_df)} videos from {videos_file}")
        except Exception as e:
            logger.error(f"Error loading videos file: {e}")
    
    return comments_df, videos_df

def main():
    """Main function for testing the pipeline"""
    
    # Define data files
    comment_files = [
        "dataset/comments1.csv",
        "dataset/comments2.csv", 
        "dataset/comments3.csv",
        "dataset/comments4.csv",
        "dataset/comments5.csv"
    ]
    videos_file = "dataset/videos.csv"
    
    try:
        # Load datasets
        print("Loading datasets...")
        comments_df, videos_df = load_datasets(comment_files, videos_file)
        print(f"Total comments loaded: {len(comments_df)}")
        
        # Initialize pipeline
        print("Initializing CommentSense pipeline...")
        pipeline = CommentSensePipeline()
        
        # Analyze first 100 comments for testing
        test_df = comments_df.head(100)
        print(f"Analyzing {len(test_df)} comments...")
        
        results = pipeline.analyze_comments(test_df, videos_df)
        
        # Print summary
        print("\n=== CommentSense Analysis Results ===")
        print(f"Processing time: {results['pipeline_info']['processing_time']:.2f} seconds")
        print(f"Enabled modules: {', '.join(results['pipeline_info']['enabled_modules'])}")
        
        insights = results['aggregated_insights']
        print(f"\nOverall sentiment: {insights['overall_sentiment']['mean']:.3f}")
        print(f"Dominant emotions: {list(insights['dominant_emotions'].keys())[:3]}")
        print(f"Languages detected: {list(insights['language_distribution'].keys())}")
        print(f"Crisis alerts: {insights['crisis_summary']['total_alerts']}")
        
        print("\nRecommendations:")
        for rec in insights['recommendations']:
            print(f"  {rec}")
        
        print("\n=== Analysis Complete ===")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()