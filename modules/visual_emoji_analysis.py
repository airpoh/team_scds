"""
CommentSense - Visual & Emoji Signals Analysis Module
====================================================

This module implements visual and emoji signal analysis for the CommentSense system.
Analyzes emoji sentiment and processes images for additional context clues.

Features:
- Comprehensive emoji sentiment analysis using custom lexicons
- Image analysis with Vision Transformer models (optional)
- Multi-modal sentiment fusion
- Emoji contribution scoring
"""

import re
import json
import logging
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


# Core libraries
try:
    import emoji
    from transformers import pipeline, ViTFeatureExtractor, ViTForImageClassification
    from PIL import Image
    import requests
    import torch
    HAS_VISION_MODELS = True
except ImportError as e:
    logging.warning(f"Some vision dependencies not available: {e}")
    HAS_VISION_MODELS = False

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmojiSentimentAnalyzer:
    """
    Comprehensive emoji sentiment analysis with custom lexicons and scoring.
    """
    
    def __init__(self, config_path: str = None, base_dir: str = None):
        """
        Initialize emoji analyzer with custom sentiment lexicon.
        
        Args:
            config_path: Path to configuration JSON file
            base_dir: Base directory for relative path resolution
        """
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.config = self._load_config(config_path)
        self.emoji_sentiment_lexicon = self._build_emoji_lexicon()
        self.emoji_pattern = self._compile_emoji_pattern()
        self._validate_config()
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from JSON file."""
        if config_path is None:
            config_path = self.base_dir / "config" / "emoji_config.json"
        else:
            config_path = Path(config_path)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Emoji configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _validate_config(self):
        """Validate configuration for duplicate emojis across categories."""
        positive_emojis = set(self.config.get("positive_emojis", {}).keys())
        negative_emojis = set(self.config.get("negative_emojis", {}).keys())
        neutral_emojis = set(self.config.get("neutral_emojis", {}).keys())
        
        # Check for overlaps
        pos_neg_overlap = positive_emojis & negative_emojis
        pos_neu_overlap = positive_emojis & neutral_emojis
        neg_neu_overlap = negative_emojis & neutral_emojis
        
        if pos_neg_overlap:
            logger.warning(f"Emojis found in both positive and negative categories: {pos_neg_overlap}")
        if pos_neu_overlap:
            logger.warning(f"Emojis found in both positive and neutral categories: {pos_neu_overlap}")
        if neg_neu_overlap:
            logger.warning(f"Emojis found in both negative and neutral categories: {neg_neu_overlap}")
    
    def _get_default_config(self) -> Dict:
        """Return default emoji analysis configuration."""
        return {
            "positive_emojis": {
                "😀": 0.8, "😃": 0.8, "😄": 0.9, "😁": 0.8, "😆": 0.9,
                "😊": 0.9, "☺️": 0.8, "🙂": 0.7, "😉": 0.8, "😍": 0.95,
                "🥰": 0.95, "😘": 0.9, "😗": 0.7, "😙": 0.7, "😚": 0.8,
                "🤗": 0.9, "😋": 0.8, "😛": 0.7, "😝": 0.8, "😜": 0.8,
                "🤪": 0.7, "🤩": 0.95, "🥳": 0.9, "😎": 0.8, "🤓": 0.6,
                "🥸": 0.6, "😇": 0.9, "🥺": 0.6, "🤤": 0.6, "😴": 0.5,
                "❤️": 1.0, "🧡": 0.9, "💛": 0.9, "💚": 0.9, "💙": 0.9,
                "💜": 0.9, "🤍": 0.8, "🖤": 0.3, "🤎": 0.5, "💕": 0.95,
                "💖": 0.95, "💗": 0.9, "💘": 0.9, "💝": 0.9, "💞": 0.95,
                "💟": 0.9, "♥️": 0.9, "💌": 0.8, "💎": 0.8, "💍": 0.8,
                "👍": 0.9, "👌": 0.8, "✌️": 0.8, "🤞": 0.7, "🤟": 0.8,
                "🤘": 0.8, "🤙": 0.8, "👏": 0.9, "🙌": 0.9, "👐": 0.7,
                "🔥": 0.9, "⭐": 0.8, "🌟": 0.9, "✨": 0.9, "🎉": 0.95,
                "🎊": 0.9, "🎈": 0.8, "🎁": 0.9, "🏆": 0.9, "🥇": 0.9,
                "🎯": 0.8, "💯": 0.95, "✅": 0.8, "☀️": 0.8, "🌞": 0.8,
                "🌈": 0.9, "🦄": 0.9, "🍕": 0.7, "🍰": 0.8, "🎂": 0.8,
                "🍾": 0.8, "🥂": 0.8, "💪": 0.8, "🤝": 0.8, "🙏": 0.8
            },
            "negative_emojis": {
                "😞": -0.8, "😔": -0.7, "😟": -0.7, "😕": -0.6, "🙁": -0.6,
                "☹️": -0.7, "😣": -0.7, "😖": -0.7, "😫": -0.8, "😩": -0.8,
                "🥺": -0.3, "😢": -0.9, "😭": -0.95, "😤": -0.7, "😠": -0.9,
                "😡": -0.95, "🤬": -1.0, "🤯": -0.8, "😳": -0.5, "🥵": -0.6,
                "🥶": -0.6, "😱": -0.8, "😨": -0.8, "😰": -0.8, "😥": -0.7,
                "😓": -0.7, "🤗": 0.5, "🤔": -0.2, "🤐": -0.4, "🤨": -0.3,
                "😐": -0.2, "😑": -0.3, "😶": -0.2, "🙄": -0.6, "😏": -0.3,
                "😒": -0.6, "🤥": -0.7, "😪": -0.5, "😵": -0.8, "🤒": -0.8,
                "🤕": -0.8, "🤢": -0.9, "🤮": -0.95, "🤧": -0.6, "🥴": -0.6,
                "😵‍💫": -0.8, "🤐": -0.4, "💔": -1.0, "💀": -0.8, "☠️": -0.9,
                "👎": -0.9, "👊": -0.7, "✊": -0.5, "👹": -0.9, "👺": -0.9,
                "🤡": -0.3, "👻": -0.4, "💩": -0.8, "🔴": -0.4, "💥": -0.6,
                "💢": -0.8, "🚫": -0.7, "⛔": -0.7, "❌": -0.8, "❗": -0.5,
                "⚠️": -0.6, "🆘": -0.9, "🔞": -0.6, "📵": -0.5, "🚷": -0.6
            },
            "neutral_emojis": {
                "😐": 0.0, "😑": 0.0, "😶": 0.0, "🤔": 0.0, "🤷": 0.0,
                "🤷‍♀️": 0.0, "🤷‍♂️": 0.0, "👍": 0.5, "👎": -0.5, "👌": 0.3,
                "🤝": 0.2, "👋": 0.1, "✋": 0.0, "🖐️": 0.0, "👊": -0.2,
                "✊": 0.0, "🤚": 0.0, "👐": 0.1, "🙌": 0.3, "👏": 0.4
            },
            "intensity_multipliers": {
                "skin_tone_modifiers": 1.0,
                "repetition_boost": 0.1,  # Additional score per repeated emoji
                "max_repetition_boost": 0.5,  # Maximum additional score
                "context_nearby_boost": 0.2  # Boost when multiple positive/negative emojis nearby
            }
        }
    
    def _build_emoji_lexicon(self) -> Dict[str, float]:
        """Build comprehensive emoji sentiment lexicon."""
        lexicon = {}
        
        # Add configured emojis
        for emoji_dict, sentiment_type in [
            (self.config["positive_emojis"], "positive"),
            (self.config["negative_emojis"], "negative"),
            (self.config["neutral_emojis"], "neutral")
        ]:
            lexicon.update(emoji_dict)
        
        return lexicon
    
    def _compile_emoji_pattern(self) -> re.Pattern:
        """Compile regex pattern to match all emojis."""
        try:
            # Use emoji library's pattern if available
            import emoji
            return re.compile(emoji.get_emoji_regexp())
        except:
            # Fallback pattern for basic emoji detection
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+",
                flags=re.UNICODE
            )
            return emoji_pattern
    
    def extract_emojis(self, text: str) -> List[str]:
        """
        Extract all emojis from text.
        
        Args:
            text: Input text
            
        Returns:
            List of emoji characters found
        """
        if not text:
            return []
        
        try:
            # Use emoji library for better detection if available
            if 'emoji' in globals():
                try:
                    # Try new emoji library API
                    return [match['emoji'] for match in emoji.analyze(text, non_emoji=False)]
                except (KeyError, TypeError):
                    try:
                        # Try older API
                        return [match.emoji for match in emoji.analyze(text, non_emoji=False)]
                    except (AttributeError, TypeError):
                        # Fallback to regex if emoji.analyze fails
                        return self.emoji_pattern.findall(text)
            else:
                # Fallback regex extraction
                return self.emoji_pattern.findall(text)
        except Exception as e:
            logger.warning(f"Error extracting emojis: {e}")
            return []
    
    def analyze_emoji_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze emoji sentiment in text with detailed scoring.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with emoji analysis results
        """
        emojis = self.extract_emojis(text)
        
        if not emojis:
            return {
                'emoji_count': 0,
                'unique_emoji_count': 0,
                'emoji_sentiment_score': 0.0,
                'emoji_sentiment': 'neutral',
                'emoji_intensity': 0.0,
                'emoji_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'dominant_emoji_sentiment': 'neutral',
                'emoji_list': [],
                'emoji_scores': {}
            }
        
        # Count emojis and calculate sentiment
        emoji_counts = {}
        total_score = 0.0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        emoji_scores = {}
        
        for emoji_char in emojis:
            emoji_counts[emoji_char] = emoji_counts.get(emoji_char, 0) + 1
            
            # Get base sentiment score
            base_score = self.emoji_sentiment_lexicon.get(emoji_char, 0.0)
            
            # Apply intensity multipliers
            repetition_count = emoji_counts[emoji_char]
            repetition_boost = min(
                (repetition_count - 1) * self.config["intensity_multipliers"]["repetition_boost"],
                self.config["intensity_multipliers"]["max_repetition_boost"]
            )
            
            final_score = base_score + (repetition_boost if base_score != 0 else 0)
            emoji_scores[emoji_char] = final_score
            total_score += final_score
            
            # Count by sentiment type
            if final_score > 0.1:
                positive_count += 1
            elif final_score < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate overall metrics
        emoji_count = len(emojis)
        unique_count = len(emoji_counts)
        
        # Normalize score by emoji count to prevent overwhelming with quantity
        normalized_score = total_score / emoji_count if emoji_count > 0 else 0.0
        
        # Determine overall emoji sentiment
        if normalized_score > 0.2:
            emoji_sentiment = 'positive'
        elif normalized_score < -0.2:
            emoji_sentiment = 'negative'
        else:
            emoji_sentiment = 'neutral'
        
        # Calculate intensity (0-1 scale)
        intensity = min(abs(normalized_score), 1.0)
        
        # Determine dominant sentiment
        if positive_count > negative_count and positive_count > neutral_count:
            dominant = 'positive'
        elif negative_count > positive_count and negative_count > neutral_count:
            dominant = 'negative'
        else:
            dominant = 'neutral'
        
        return {
            'emoji_count': emoji_count,
            'unique_emoji_count': unique_count,
            'emoji_sentiment_score': round(normalized_score, 3),
            'emoji_sentiment': emoji_sentiment,
            'emoji_intensity': round(intensity, 3),
            'emoji_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            },
            'dominant_emoji_sentiment': dominant,
            'emoji_list': list(set(emojis)),
            'emoji_scores': emoji_scores,
            'most_frequent_emojis': sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }


class ImageAnalyzer:
    """
    Optional image analysis using Vision Transformer models for additional context.
    """
    
    def __init__(self, device: str = "auto", config: Dict = None):
        """
        Initialize image analyzer.
        
        Args:
            device: Device to run models on
            config: Configuration dictionary with image analysis settings
        """
        self.device = self._setup_device(device)
        self.config = config or {}
        self.model = None
        self.feature_extractor = None
        self.enabled = HAS_VISION_MODELS
        
        # Get image analysis indicators from config
        self.positive_indicators = self.config.get("image_analysis", {}).get("positive_indicators", [])
        self.negative_indicators = self.config.get("image_analysis", {}).get("negative_indicators", [])
        
        if self.enabled:
            self._load_models()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup device for inference."""
        if not HAS_VISION_MODELS:
            return torch.device("cpu")
            
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_models(self):
        """Load Vision Transformer models."""
        if not self.enabled:
            logger.warning("Vision models not available, image analysis disabled")
            return
        
        try:
            logger.info("Loading Vision Transformer model...")
            
            model_name = "google/vit-base-patch16-224"
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            self.model = ViTForImageClassification.from_pretrained(model_name)
            
            if self.device.type != "cpu":
                self.model = self.model.to(self.device)
            
            logger.info("Vision model loaded successfully!")
            
        except Exception as e:
            logger.warning(f"Could not load vision models: {e}")
            self.enabled = False
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image for sentiment-relevant features.
        
        Args:
            image_path: Path to image file or URL
            
        Returns:
            Dictionary with image analysis results
        """
        if not self.enabled:
            return {
                'image_analyzed': False,
                'reason': 'Vision models not available',
                'predictions': [],
                'sentiment_indicators': {}
            }
        
        try:
            # Load image
            if image_path.startswith(('http://', 'https://')):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract features and predict
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            if self.device.type != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get top predictions
            top_predictions = []
            for i in range(min(5, predictions.shape[1])):
                score = predictions[0][i].item()
                label = self.model.config.id2label[i]
                top_predictions.append({'label': label, 'score': score})
            
            # Sort by confidence
            top_predictions.sort(key=lambda x: x['score'], reverse=True)
            
            # Analyze for sentiment indicators
            sentiment_indicators = self._analyze_image_sentiment(top_predictions)
            
            return {
                'image_analyzed': True,
                'predictions': top_predictions,
                'sentiment_indicators': sentiment_indicators
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing image: {e}")
            return {
                'image_analyzed': False,
                'reason': str(e),
                'predictions': [],
                'sentiment_indicators': {}
            }
    
    def _analyze_image_sentiment(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze image predictions for sentiment-relevant indicators.
        
        Args:
            predictions: List of model predictions
            
        Returns:
            Sentiment indicators dictionary
        """
        # Use configured indicators or defaults
        positive_indicators = self.positive_indicators or [
            'golden retriever', 'labrador retriever', 'beagle', 'pug', 'corgi',
            'sunflower', 'daisy', 'rose', 'tulip', 'cherry', 'strawberry',
            'rainbow', 'sunset', 'beach', 'ocean', 'mountain', 'forest',
            'cake', 'pizza', 'ice cream', 'chocolate', 'candy',
            'smile', 'laugh', 'celebration', 'party', 'wedding'
        ]
        
        negative_indicators = self.negative_indicators or [
            'storm', 'rain', 'thunder', 'lightning', 'tornado', 'hurricane',
            'fire', 'smoke', 'explosion', 'crash', 'accident',
            'cemetery', 'grave', 'funeral', 'hospital', 'ambulance',
            'spider', 'snake', 'rat', 'cockroach', 'wasp',
            'garbage', 'trash', 'pollution', 'waste', 'rotten'
        ]
        
        # Analyze top predictions
        positive_score = 0.0
        negative_score = 0.0
        
        for pred in predictions[:3]:  # Check top 3 predictions
            label = pred['label'].lower()
            score = pred['score']
            
            # Check for positive indicators
            for indicator in positive_indicators:
                if indicator in label:
                    positive_score += score
                    break
            
            # Check for negative indicators  
            for indicator in negative_indicators:
                if indicator in label:
                    negative_score += score
                    break
        
        # Determine overall image sentiment
        if positive_score > negative_score and positive_score > 0.1:
            image_sentiment = 'positive'
        elif negative_score > positive_score and negative_score > 0.1:
            image_sentiment = 'negative'
        else:
            image_sentiment = 'neutral'
        
        return {
            'image_sentiment': image_sentiment,
            'positive_score': round(positive_score, 3),
            'negative_score': round(negative_score, 3),
            'confidence': round(max(positive_score, negative_score), 3)
        }


class VisualEmojiAnalyzer:
    """
    Combined visual and emoji analysis for comprehensive multi-modal sentiment analysis.
    """
    
    def __init__(self, config_path: str = None, device: str = "auto", base_dir: str = None):
        """
        Initialize combined analyzer.
        
        Args:
            config_path: Path to configuration file
            device: Device for model inference
            base_dir: Base directory for relative path resolution
        """
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.emoji_analyzer = EmojiSentimentAnalyzer(config_path, str(self.base_dir))
        
        # Get config for image analyzer
        config = self.emoji_analyzer.config
        self.image_analyzer = ImageAnalyzer(device, config)
        
        # Get fusion weights from config
        self.fusion_weights = config.get('fusion_weights', {
            'emoji': 0.7,  # Emoji analysis weight
            'image': 0.3   # Image analysis weight
        })
    
    def analyze_comment(self, text: str, image_path: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive visual and emoji analysis.
        
        Args:
            text: Comment text to analyze
            image_path: Optional path to associated image
            
        Returns:
            Combined analysis results
        """
        # Analyze emoji sentiment
        emoji_results = self.emoji_analyzer.analyze_emoji_sentiment(text)
        
        # Analyze image if provided
        image_results = {}
        if image_path:
            image_results = self.image_analyzer.analyze_image(image_path)
        
        # Fuse results
        fused_sentiment = self._fuse_multimodal_sentiment(emoji_results, image_results)
        
        return {
            'text': text,
            'emoji_analysis': emoji_results,
            'image_analysis': image_results,
            'fused_sentiment': fused_sentiment,
            'timestamp': datetime.now().isoformat()
        }
    
    def _fuse_multimodal_sentiment(self, emoji_results: Dict, image_results: Dict) -> Dict[str, Any]:
        """
        Fuse emoji and image analysis results for final sentiment.
        
        Args:
            emoji_results: Results from emoji analysis
            image_results: Results from image analysis
            
        Returns:
            Fused sentiment analysis
        """
        # Get emoji sentiment score
        emoji_score = emoji_results.get('emoji_sentiment_score', 0.0)
        emoji_sentiment = emoji_results.get('emoji_sentiment', 'neutral')
        emoji_confidence = emoji_results.get('emoji_intensity', 0.0)
        
        # Get image sentiment if available
        image_sentiment = 'neutral'
        image_score = 0.0
        image_confidence = 0.0
        
        if image_results.get('image_analyzed', False):
            indicators = image_results.get('sentiment_indicators', {})
            image_sentiment = indicators.get('image_sentiment', 'neutral')
            pos_score = indicators.get('positive_score', 0.0)
            neg_score = indicators.get('negative_score', 0.0)
            image_score = pos_score - neg_score
            image_confidence = indicators.get('confidence', 0.0)
        
        # Weighted fusion
        if image_results.get('image_analyzed', False):
            # Both emoji and image available
            fused_score = (
                self.fusion_weights['emoji'] * emoji_score +
                self.fusion_weights['image'] * image_score
            )
            fused_confidence = (
                self.fusion_weights['emoji'] * emoji_confidence +
                self.fusion_weights['image'] * image_confidence
            )
        else:
            # Only emoji available
            fused_score = emoji_score
            fused_confidence = emoji_confidence
        
        # Determine final sentiment
        if fused_score > 0.2:
            final_sentiment = 'positive'
        elif fused_score < -0.2:
            final_sentiment = 'negative'
        else:
            final_sentiment = 'neutral'
        
        return {
            'final_sentiment': final_sentiment,
            'fused_score': round(fused_score, 3),
            'confidence': round(fused_confidence, 3),
            'emoji_contribution': round(self.fusion_weights['emoji'] * emoji_score, 3),
            'image_contribution': round(self.fusion_weights['image'] * image_score, 3) if image_results.get('image_analyzed', False) else 0.0,
            'modalities_used': ['emoji'] + (['image'] if image_results.get('image_analyzed', False) else [])
        }
    
    def batch_analyze(self, comments: List[str], image_paths: List[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple comments with optional images.
        
        Args:
            comments: List of comment texts
            image_paths: Optional list of image paths (same length as comments)
            
        Returns:
            List of analysis results
        """
        if image_paths is None:
            image_paths = [None] * len(comments)
        
        results = []
        for comment, image_path in zip(comments, image_paths):
            result = self.analyze_comment(comment, image_path)
            results.append(result)
        
        return results
    
    def get_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from batch analysis results.
        
        Args:
            results: List of analysis results
            
        Returns:
            Summary statistics
        """
        if not results:
            return {}
        
        # Collect metrics
        total_comments = len(results)
        emoji_counts = []
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        emoji_used = 0
        images_analyzed = 0
        total_emoji_score = 0.0
        
        for result in results:
            emoji_analysis = result.get('emoji_analysis', {})
            fused_sentiment = result.get('fused_sentiment', {})
            
            # Count emojis
            emoji_count = emoji_analysis.get('emoji_count', 0)
            emoji_counts.append(emoji_count)
            if emoji_count > 0:
                emoji_used += 1
                total_emoji_score += emoji_analysis.get('emoji_sentiment_score', 0.0)
            
            # Count sentiments
            final_sentiment = fused_sentiment.get('final_sentiment', 'neutral')
            sentiment_counts[final_sentiment] += 1
            
            # Count images
            if result.get('image_analysis', {}).get('image_analyzed', False):
                images_analyzed += 1
        
        return {
            'total_comments': total_comments,
            'emoji_usage_rate': emoji_used / total_comments,
            'average_emojis_per_comment': np.mean(emoji_counts) if emoji_counts else 0,
            'image_analysis_rate': images_analyzed / total_comments,
            'sentiment_distribution': {k: v / total_comments for k, v in sentiment_counts.items()},
            'average_emoji_sentiment': total_emoji_score / emoji_used if emoji_used > 0 else 0.0,
            'multimodal_comments': images_analyzed
        }


def create_emoji_config():
    """Create default emoji configuration file."""
    config_dir = Path(__file__).parent.parent / "config"
    config_dir.mkdir(exist_ok=True)
    
    analyzer = EmojiSentimentAnalyzer()
    config_path = config_dir / "emoji_config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(analyzer.config, f, indent=2, ensure_ascii=False)
    
    print(f"Emoji configuration created at: {config_path}")


def main():
    """Main function to demonstrate visual and emoji analysis."""
    
    # Create config if it doesn't exist
    create_emoji_config()
    
    # Initialize analyzer
    print("Initializing Visual & Emoji Analyzer...")
    analyzer = VisualEmojiAnalyzer()
    
    # Test with sample comments
    test_comments = [
        "I love this product! 😍❤️🔥 Amazing quality and fast shipping! 👍✨",
        "Worst purchase ever 😡💔 Complete waste of money! 👎🤮",
        "It's okay, nothing special 😐 Could be better 🤷‍♀️",
        "OMG this is AMAZING!!! 🤩🎉💯 Best day ever! 🥳🙌",
        "Why would anyone buy this? 🤔💸 So disappointed 😞",
        "🌈✨ Beautiful colors and great design! 🎨💖 Highly recommend! ⭐⭐⭐⭐⭐"
    ]
    
    print(f"\nAnalyzing {len(test_comments)} sample comments...")
    results = analyzer.batch_analyze(test_comments)
    
    # Generate summary
    summary = analyzer.get_summary_stats(results)
    
    # Display results
    print("\n" + "="*60)
    print("VISUAL & EMOJI ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Total Comments Analyzed: {summary['total_comments']}")
    print(f"Emoji Usage Rate: {summary['emoji_usage_rate']:.1%}")
    print(f"Average Emojis per Comment: {summary['average_emojis_per_comment']:.1f}")
    print(f"Average Emoji Sentiment: {summary['average_emoji_sentiment']:.3f}")
    
    print("\nSentiment Distribution:")
    for sentiment, ratio in summary['sentiment_distribution'].items():
        print(f"  {sentiment.title()}: {ratio:.1%}")
    
    print("\nDetailed Analysis Examples:")
    print("-" * 40)
    
    for i, (comment, result) in enumerate(zip(test_comments, results)):
        emoji_analysis = result['emoji_analysis']
        fused_sentiment = result['fused_sentiment']
        
        print(f"\nComment {i+1}: {comment}")
        print(f"Emojis found: {emoji_analysis['emoji_count']} ({emoji_analysis['emoji_list']})")
        print(f"Emoji sentiment: {emoji_analysis['emoji_sentiment']} ({emoji_analysis['emoji_sentiment_score']:.3f})")
        print(f"Final sentiment: {fused_sentiment['final_sentiment']} ({fused_sentiment['fused_score']:.3f})")
    
    # Save results
    current_dir = Path(__file__).parent.parent
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_data = {
        'summary': summary,
        'detailed_results': results,
        'timestamp': datetime.now().isoformat(),
        'analyzer_config': analyzer.emoji_analyzer.config
    }
    
    output_file = results_dir / 'visual_emoji_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()