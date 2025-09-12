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
import gc
warnings.filterwarnings("ignore")

# Shared Model Manager for memory optimization
class SharedModelManager:
    """
    Centralized manager for sharing transformer models across modules
    Based on HuggingFace Transformers best practices for memory optimization
    """
    
    def __init__(self):
        self._models = {}
        self._tokenizers = {}
        self._device = self._setup_device()
        
    def _setup_device(self):
        """Setup optimal device configuration"""
        import platform
        try:
            import torch
        except ImportError:
            return "cpu"
        
        # For Apple Silicon, force CPU only
        if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
            return torch.device("cpu")
        
        # For Windows/Linux, use CUDA if available
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        return torch.device("cpu")
    
    def get_shared_model(self, model_type: str, model_name: str = None):
        """Get or create a shared model instance"""
        
        if model_type in self._models:
            logger.info(f"Reusing shared model: {model_type}")
            return self._models[model_type]
        
        try:
            import torch
            from transformers import pipeline
            
            # Force garbage collection before loading
            gc.collect()
            
            # Configure model loading with memory optimization
            model_kwargs = {
                'torch_dtype': torch.float32,
                'device_map': None,
                'low_cpu_mem_usage': False,
            }
            
            logger.info(f"Loading shared model: {model_type}")
            
            if model_type == "sentence_transformer":
                # Shared sentence transformer for multiple modules
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(
                        model_name or "sentence-transformers/all-MiniLM-L6-v2", 
                        device=str(self._device)
                    )
                    self._models[model_type] = model
                    logger.info(f"Shared SentenceTransformer loaded on {self._device}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load SentenceTransformer: {e}")
                    return None
                    
            elif model_type == "multilingual_sentiment":
                # Shared multilingual sentiment model
                model_name = model_name or "cardiffnlp/twitter-xlm-roberta-base-sentiment"
                model = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=0 if self._device.type == "cuda" else -1,
                    return_all_scores=True,
                    model_kwargs=model_kwargs
                )
                self._models[model_type] = model
                logger.info(f"Shared multilingual sentiment model loaded on {self._device}")
                
            elif model_type == "zero_shot_classifier":
                # Shared zero-shot classifier
                model_name = model_name or "facebook/bart-large-mnli"
                model = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=0 if self._device.type == "cuda" else -1,
                    model_kwargs=model_kwargs
                )
                self._models[model_type] = model
                logger.info(f"Shared zero-shot classifier loaded on {self._device}")
                
            else:
                logger.warning(f"Unknown shared model type: {model_type}")
                return None
            
            # Force garbage collection after loading
            gc.collect()
            
            return self._models[model_type]
            
        except Exception as e:
            logger.error(f"Failed to load shared model {model_type}: {e}")
            return None
    
    def clear_unused_models(self):
        """Clear models that are not frequently used"""
        logger.info("Clearing unused shared models...")
        self._models.clear()
        self._tokenizers.clear()
        gc.collect()

# Global shared model manager instance
_shared_model_manager = None

def get_shared_model_manager():
    """Get the global shared model manager"""
    global _shared_model_manager
    if _shared_model_manager is None:
        _shared_model_manager = SharedModelManager()
    return _shared_model_manager

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
from modules.network_analysis import NetworkAnalyzer
from modules.persona_clustering import PersonaClustering

# Import new advanced modules
try:
    from modules.composite_kpi_system import CompositeKPISystem
    COMPOSITE_KPI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Composite KPI System not available: {e}")
    COMPOSITE_KPI_AVAILABLE = False

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

def run_network_analysis_worker_wrapper(comments_df_dict):
    """Worker function for network analysis"""
    try:
        import pandas as pd
        from modules.network_analysis import NetworkAnalyzer
        analyzer = NetworkAnalyzer()
        comments_df = pd.DataFrame(comments_df_dict)
        return analyzer.analyze_network(comments_df)
    except Exception as e:
        logger.error(f"Error in network_analysis worker: {e}")
        return {"error": str(e)}

class CommentSensePipeline:
    """Main pipeline integrating all analysis modules"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the CommentSense pipeline"""
        
        self.config_path = config_path or Path(__file__).parent / "config" / "pipeline_config.json"
        self.config = self._load_config()
        
        # Pipeline state - initialize before modules to avoid AttributeError
        self.processing_stats = {
            'total_comments_processed': 0,
            'processing_time': 0.0,
            'last_processed': None,
            'errors': []
        }
        
        # Initialize analysis modules registry (lazy loading for memory optimization)
        self.modules = {}
        self.shared_manager = get_shared_model_manager()
        logger.info("Pipeline initialized with lazy module loading for memory optimization")
        # Note: Modules will be initialized only when needed
    
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
                "crisis_detection": {"enabled": True},
                "network_analysis": {"enabled": True}
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
    
    def _get_module(self, module_name: str):
        """Lazy load a module when needed (with shared models)"""
        
        if module_name in self.modules:
            return self.modules[module_name]
        
        module_config = self.config.get('modules', {})
        
        # Check if module is enabled
        if not module_config.get(module_name, {}).get('enabled', True):
            logger.info(f"Module {module_name} is disabled")
            return None
        
        try:
            logger.info(f"Lazy loading module: {module_name}")
            
            # Force garbage collection before loading
            gc.collect()
            
            if module_name == 'emotion_sarcasm':
                module = EmotionSarcasmDetector()
            elif module_name == 'visual_emoji':
                module = VisualEmojiAnalyzer()
            elif module_name == 'multilingual':
                module = MultilingualSentimentAnalyzer(use_shared_models=True)
            elif module_name == 'crisis_detection':
                module = CrisisDetectionSystem()
            elif module_name == 'network_analysis':
                module = NetworkAnalyzer()
            else:
                logger.error(f"Unknown module: {module_name}")
                return None
            
            self.modules[module_name] = module
            logger.info(f"Module {module_name} loaded successfully")
            
            # Force garbage collection after loading
            gc.collect()
            
            return module
            
        except Exception as e:
            logger.error(f"Error loading module {module_name}: {e}")
            self.processing_stats['errors'].append(f"Module {module_name} loading error: {e}")
            return None
    
    def _initialize_modules(self):
        """Initialize module registry (deprecated - using lazy loading now)"""
        logger.info("Using lazy module loading - modules will be loaded on demand")
    
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
            # For Apple Silicon, force sequential processing to prevent segmentation faults
            import platform
            if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
                logger.info("Apple Silicon detected - using sequential processing to prevent segmentation faults")
                results['module_results'] = self._process_sequential(comments_df, videos_df)
            elif enable_parallel and len(comments_df) > batch_size:
                results['module_results'] = self._process_parallel(comments_df, videos_df)
            else:
                results['module_results'] = self._process_sequential(comments_df, videos_df)
            
            # Generate aggregated insights
            results['aggregated_insights'] = self._generate_insights(
                results['module_results'], comments_df, videos_df
            )
            
            # Generate Composite KPI (if available)
            if COMPOSITE_KPI_AVAILABLE:
                try:
                    composite_kpi_system = CompositeKPISystem()
                    composite_results = composite_kpi_system.calculate_comprehensive_kpi(comments_df)
                    results['composite_kpi'] = composite_results
                    logger.info(f"Composite KPI calculated: {composite_results.get('comprehensive_kpi', {}).get('overall_score', 'N/A')}")
                except Exception as e:
                    logger.error(f"Composite KPI calculation failed: {e}")
                    results['composite_kpi'] = {"error": str(e)}
            
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
        
        # For Apple Silicon, limit data size to prevent memory issues
        import platform
        if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
            max_comments = 1000  # Conservative limit for Apple Silicon
            if len(comments_df) > max_comments:
                logger.info(f"Apple Silicon: Limiting analysis to {max_comments} comments to prevent memory issues")
                comments_df = comments_df.head(max_comments)
        
        # Emotion & Sarcasm Detection (lazy loaded)
        emotion_module = self._get_module('emotion_sarcasm')
        if emotion_module is not None:
            try:
                logger.info("Running emotion & sarcasm detection...")
                emotion_results = emotion_module.batch_analyze(
                    comments_df['textOriginal'].fillna('').tolist()
                )
                module_results['emotion_sarcasm'] = emotion_results
                
                # Force garbage collection after each module (Apple Silicon memory management)
                import gc
                gc.collect()
                
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
        
        # Network Analysis
        if 'network_analysis' in self.modules:
            try:
                logger.info("Running network analysis...")
                network_results = self.modules['network_analysis'].analyze_network(
                    comments_df
                )
                module_results['network_analysis'] = network_results
            except Exception as e:
                logger.error(f"Network Analysis module error: {e}")
                module_results['network_analysis'] = {"error": str(e)}
        
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
            
            if 'network_analysis' in self.modules:
                futures['network_analysis'] = executor.submit(run_network_analysis_worker_wrapper, comments_df_dict)
            
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
            "network_analysis": self._summarize_network_analysis(module_results),
            "combined_insights": self._generate_combined_insights(module_results),
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
    
    def _summarize_network_analysis(self, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize network analysis results"""
        
        if 'network_analysis' in module_results and isinstance(module_results['network_analysis'], dict):
            network_data = module_results['network_analysis']
            
            if 'error' in network_data:
                return {"error": network_data['error']}
            
            # Check if we have the DataFrame-based results structure
            if 'influencers_df' in network_data and 'communities_df' in network_data:
                try:
                    # Parse the string representations of DataFrames
                    influencers_info = str(network_data.get('influencers_df', ''))
                    communities_info = str(network_data.get('communities_df', ''))
                    personas_info = str(network_data.get('personas_df', ''))
                    categories_info = str(network_data.get('cats_out', ''))
                    
                    # Extract basic stats from the string representations
                    total_influencers = 0
                    total_communities = 0 
                    total_personas = 0
                    total_categories = 0
                    
                    # Extract counts from the DataFrame string representations
                    if 'rows x' in influencers_info:
                        # Extract number like "[1017 rows x 9 columns]"
                        import re
                        match = re.search(r'\[(\d+) rows x \d+ columns\]', influencers_info)
                        if match:
                            total_influencers = int(match.group(1))
                    
                    if 'rows x' in communities_info:
                        match = re.search(r'\[(\d+) rows x \d+ columns\]', communities_info)
                        if match:
                            total_communities = int(match.group(1))
                    
                    if 'rows x' in personas_info:
                        match = re.search(r'\[(\d+) rows x \d+ columns\]', personas_info)
                        if match:
                            total_personas = int(match.group(1))
                    
                    # For categories, check if it has data
                    if categories_info and 'rows x' in categories_info:
                        total_categories = 1  # At least one category if data exists
                    
                    summary = {
                        "status": "success",
                        "key_stats": {
                            "total_influencers": total_influencers,
                            "total_communities": total_communities,
                            "total_personas": total_personas,
                            "category_diversity": total_categories
                        }
                    }
                    
                    return summary
                    
                except Exception as e:
                    logger.error(f"Error parsing network analysis results: {e}")
                    return {"status": "error", "error": str(e)}
            
            # Fallback for other result formats
            summary = {
                "status": network_data.get('status', 'unknown'),
                "metrics": network_data.get('metrics', {}),
                "summary": network_data.get('summary', {})
            }
            
            # Extract key metrics for easy access
            results = network_data.get('results', {})
            if results:
                summary["key_stats"] = {
                    "total_influencers": results.get('influencers', {}).get('total_influencers', 0),
                    "total_communities": results.get('communities', {}).get('total_communities', 0),
                    "total_personas": results.get('personas', {}).get('total_personas', 0),
                    "category_diversity": results.get('categories', {}).get('total_categories', 0)
                }
            
            return summary
        
        return {"status": "not_available"}
    
    def _generate_combined_insights(self, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights that combine multiple module results"""
        
        combined = {
            "cross_module_correlations": {},
            "unified_recommendations": [],
            "risk_assessment": {},
            "engagement_insights": {}
        }
        
        try:
            # Combine sentiment analysis from multiple sources
            sentiment_sources = []
            
            # CommentSense sentiment
            overall_sentiment = self._calculate_overall_sentiment(module_results)
            if overall_sentiment.get('mean', 0) != 0:
                sentiment_sources.append(('emotion_analysis', overall_sentiment['mean']))
            
            # Emoji sentiment
            emoji_sentiment = self._summarize_emoji_sentiment(module_results)
            if emoji_sentiment.get('overall_sentiment', 0) != 0:
                sentiment_sources.append(('emoji_analysis', emoji_sentiment['overall_sentiment']))
            
            if sentiment_sources:
                combined["cross_module_correlations"]["sentiment_consensus"] = {
                    "sources": len(sentiment_sources),
                    "average_sentiment": sum(score for _, score in sentiment_sources) / len(sentiment_sources),
                    "sentiment_agreement": self._calculate_sentiment_agreement(sentiment_sources)
                }
            
            # Risk assessment combining crisis detection and network analysis
            crisis_summary = self._summarize_crisis_alerts(module_results)
            network_summary = self._summarize_network_analysis(module_results)
            
            risk_level = "low"
            risk_factors = []
            
            if crisis_summary.get('total_alerts', 0) > 0:
                risk_factors.append(f"Crisis alerts: {crisis_summary['total_alerts']}")
                if crisis_summary.get('risk_level') == 'high':
                    risk_level = "high"
            
            if network_summary.get('status') == 'success':
                network_key_stats = network_summary.get('key_stats', {})
                high_influence_users = network_key_stats.get('high_influence_users', 0)
                if high_influence_users > 0:
                    risk_factors.append(f"High influence users: {high_influence_users}")
            
            combined["risk_assessment"] = {
                "overall_risk_level": risk_level,
                "risk_factors": risk_factors,
                "mitigation_needed": len(risk_factors) > 0
            }
            
            # Engagement insights
            if network_summary.get('status') == 'success':
                key_stats = network_summary.get('key_stats', {})
                combined["engagement_insights"] = {
                    "community_engagement": key_stats.get('total_communities', 0) > 1,
                    "influencer_presence": key_stats.get('total_influencers', 0) > 0,
                    "persona_diversity": key_stats.get('total_personas', 0) > 1,
                    "content_categorization": key_stats.get('category_diversity', 0) > 1
                }
            
        except Exception as e:
            logger.warning(f"Error generating combined insights: {e}")
            combined["error"] = str(e)
        
        return combined
    
    def _calculate_sentiment_agreement(self, sentiment_sources: List[Tuple[str, float]]) -> str:
        """Calculate agreement level between different sentiment analysis sources"""
        if len(sentiment_sources) < 2:
            return "single_source"
        
        scores = [score for _, score in sentiment_sources]
        variance = np.var(scores)
        
        if variance < 0.1:
            return "high_agreement"
        elif variance < 0.3:
            return "moderate_agreement"
        else:
            return "low_agreement"
    
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
        
        # Check network analysis insights
        if 'network_analysis' in module_results:
            network_data = module_results['network_analysis']
            if isinstance(network_data, dict) and network_data.get('status') == 'success':
                results = network_data.get('results', {})
                
                # Influencer recommendations
                if 'influencers' in results:
                    influencer_data = results['influencers']
                    total_influencers = influencer_data.get('total_influencers', 0)
                    high_influence_users = influencer_data.get('high_influence_users', 0)
                    
                    if high_influence_users > 0:
                        recommendations.append(f" {high_influence_users} high-influence users identified. Consider targeted engagement.")
                    elif total_influencers > 10:
                        recommendations.append(f" {total_influencers} influencers detected. Monitor for engagement opportunities.")
                
                # Community recommendations
                if 'communities' in results:
                    community_data = results['communities']
                    total_communities = community_data.get('total_communities', 0)
                    
                    if total_communities > 1:
                        recommendations.append(f" {total_communities} distinct communities detected. Tailor messaging by community.")
                
                # Category diversity recommendations
                if 'categories' in results:
                    category_data = results['categories']
                    total_categories = category_data.get('total_categories', 0)
                    
                    if total_categories > 3:
                        recommendations.append(f" Content spans {total_categories} categories. Consider category-specific strategies.")
                
                # Persona recommendations
                if 'personas' in results:
                    persona_data = results['personas']
                    total_personas = persona_data.get('total_personas', 0)
                    
                    if total_personas > 5:
                        recommendations.append(f"🧑‍🤝‍🧑 {total_personas} user personas identified. Customize content for different user types.")
        
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