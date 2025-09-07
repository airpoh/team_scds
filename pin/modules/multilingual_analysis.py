"""
CommentSense - Cross-Language Comment Analysis Module
====================================================

This module implements multilingual sentiment and relevance analysis for the CommentSense system.
Supports analysis in multiple languages without English-only bias.

Features:
- Automatic language identification 
- Multilingual sentiment analysis using XLM-RoBERTa
- Relevance classification across languages
- Translation fallback for low-confidence predictions
- Regional sentiment comparisons
"""

import re
import json
import logging
import warnings
import os
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

# Core libraries
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, 
        AutoModelForMaskedLM, pipeline
    )
    HAS_TRANSFORMERS = True
except ImportError as e:
    logging.warning(f"Transformers not available: {e}")
    HAS_TRANSFORMERS = False

# Language detection
try:
    import langdetect
    from langdetect import detect, detect_langs, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    import fasttext
    HAS_FASTTEXT = False  # Disable by default as it's optional
except ImportError:
    HAS_FASTTEXT = False

# Translation
try:
    from deep_translator import LibreTranslator, GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', category=UserWarning, module='langdetect')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Robust language identification using multiple methods.
    """
    
    def __init__(self, use_fasttext: bool = False):
        """
        Initialize language detector.
        
        Args:
            use_fasttext: Whether to use FastText for language detection
        """
        self.use_fasttext = use_fasttext and HAS_FASTTEXT
        self.fasttext_model = None
        
        if self.use_fasttext:
            self._load_fasttext_model()
    
    def _load_fasttext_model(self):
        """Load FastText language identification model."""
        try:
            # Note: This requires downloading the FastText LID model
            # fasttext.util.download_model('en', if_exists='ignore')
            # self.fasttext_model = fasttext.load_model('lid.176.bin')
            logger.info("FastText model loading skipped (requires manual setup)")
            self.use_fasttext = False
        except Exception as e:
            logger.warning(f"Could not load FastText model: {e}")
            self.use_fasttext = False
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of input text with confidence scores.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with language detection results
        """
        if not text or not text.strip():
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'method': 'none',
                'alternatives': []
            }
        
        # Clean text for detection
        cleaned_text = self._clean_text_for_detection(text)
        
        if len(cleaned_text) < 3:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'method': 'too_short',
                'alternatives': []
            }
        
        # Try different detection methods
        if self.use_fasttext and self.fasttext_model:
            return self._detect_with_fasttext(cleaned_text)
        elif HAS_LANGDETECT:
            return self._detect_with_langdetect(cleaned_text)
        else:
            return self._detect_with_heuristics(cleaned_text)
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text to improve language detection accuracy."""
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)
        
        # Remove excessive punctuation and emojis
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF\u0100-\u017F\u0180-\u024F]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _detect_with_fasttext(self, text: str) -> Dict[str, Any]:
        """Detect language using FastText."""
        try:
            predictions = self.fasttext_model.predict(text, k=3)
            languages, scores = predictions
            
            # Parse language codes (remove __label__ prefix)
            parsed_languages = [lang.replace('__label__', '') for lang in languages]
            
            return {
                'language': parsed_languages[0],
                'confidence': float(scores[0]),
                'method': 'fasttext',
                'alternatives': list(zip(parsed_languages[1:], scores[1:]))
            }
        except Exception as e:
            logger.warning(f"FastText detection failed: {e}")
            return self._detect_with_langdetect(text)
    
    def _detect_with_langdetect(self, text: str) -> Dict[str, Any]:
        """Detect language using langdetect."""
        try:
            # Get primary language
            primary_lang = detect(text)
            
            # Get all possible languages with probabilities
            lang_probs = detect_langs(text)
            
            # Find confidence for primary language
            confidence = 0.0
            alternatives = []
            
            for lang_prob in lang_probs:
                if lang_prob.lang == primary_lang:
                    confidence = lang_prob.prob
                else:
                    alternatives.append((lang_prob.lang, lang_prob.prob))
            
            return {
                'language': primary_lang,
                'confidence': confidence,
                'method': 'langdetect',
                'alternatives': alternatives[:3]  # Top 3 alternatives
            }
            
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
            return self._detect_with_heuristics(text)
    
    def _detect_with_heuristics(self, text: str) -> Dict[str, Any]:
        """Fallback heuristic-based language detection."""
        # Simple character-based heuristics
        char_counts = defaultdict(int)
        total_chars = 0
        
        for char in text.lower():
            if char.isalpha():
                char_counts[char] += 1
                total_chars += 1
        
        if total_chars == 0:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'method': 'heuristic',
                'alternatives': []
            }
        
        # Language-specific character patterns
        language_patterns = {
            'en': set('abcdefghijklmnopqrstuvwxyz'),
            'es': set('abcdefghijklmnñopqrstuvwxyzáéíóúü'),
            'fr': set('abcdefghijklmnopqrstuvwxyzàâäéèêëïîôöùûüÿç'),
            'de': set('abcdefghijklmnopqrstuvwxyzäöüß'),
            'it': set('abcdefghijklmnopqrstuvwxyzàéèíîóòúù'),
            'pt': set('abcdefghijklmnopqrstuvwxyzáâãàéêíóôõúç'),
            'zh': set('中文汉字'),  # Simplified approach
            'ar': set('ابتثجحخدذرزسشصضطظعغفقكلمنهويء'),
            'ja': set('ひらがなカタカナ漢字'),  # Simplified
        }
        
        # Score each language
        scores = {}
        for lang, charset in language_patterns.items():
            matches = sum(char_counts[char] for char in charset if char in char_counts)
            scores[lang] = matches / total_chars if total_chars > 0 else 0.0
        
        # Get best match
        best_lang = max(scores, key=scores.get) if scores else 'unknown'
        confidence = scores.get(best_lang, 0.0)
        
        # If confidence is too low, default to English
        if confidence < 0.3:
            best_lang = 'en'
            confidence = 0.5
        
        return {
            'language': best_lang,
            'confidence': confidence,
            'method': 'heuristic',
            'alternatives': sorted([(k, v) for k, v in scores.items() if k != best_lang], 
                                 key=lambda x: x[1], reverse=True)[:3]
        }


class MultilingualSentimentAnalyzer:
    """
    Multilingual sentiment analysis using XLM-RoBERTa and translation fallback.
    """
    
    def __init__(self, config_path: str = None, device: str = "auto"):
        """
        Initialize multilingual sentiment analyzer.
        
        Args:
            config_path: Path to configuration file
            device: Device for model inference
        """
        self.device = self._setup_device(device)
        self.config = self._load_config(config_path)
        self.sentiment_model = None
        self.relevance_model = None
        self.translator = None
        self.language_detector = LanguageDetector()
        
        self._load_models()
        self._setup_translator()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup device for inference."""
        if not HAS_TRANSFORMERS:
            return torch.device("cpu")
            
        if device == "auto":
            # Check for environment variable override
            env_device = os.getenv("COMMENTSENSE_DEVICE", "").lower()
            if env_device in ["cuda", "mps", "cpu"]:
                logger.info(f"Using device from environment variable: {env_device}")
                return torch.device(env_device)
            
            # Auto-detect best available device
            if torch.cuda.is_available():
                logger.info("CUDA available, using GPU acceleration")
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("MPS available, using Apple Silicon GPU acceleration")
                return torch.device("mps")
            else:
                logger.info("No GPU available, using CPU")
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from JSON file."""
        if config_path is None:
            current_dir = Path(__file__).parent
            config_path = current_dir.parent / "config" / "multilingual_config.json"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Multilingual configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "models": {
                "sentiment": {
                    "primary": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                    "fallback": "nlptown/bert-base-multilingual-uncased-sentiment"
                },
                "relevance": {
                    "primary": "microsoft/DialoGPT-medium",  # Placeholder for relevance model
                    "fallback": "distilbert-base-multilingual-cased"
                }
            },
            "supported_languages": [
                "en", "es", "fr", "de", "it", "pt", "nl", "pl", "tr", "ar",
                "zh", "ja", "ko", "hi", "ru", "sv", "da", "no", "fi"
            ],
            "translation": {
                "enabled": True,
                "confidence_threshold": 0.6,
                "fallback_language": "en"
            },
            "thresholds": {
                "sentiment_confidence": 0.5,
                "language_confidence": 0.3,
                "relevance_confidence": 0.5
            },
            "batch_size": 16
        }
    
    def _load_models(self):
        """Load multilingual models."""
        if not HAS_TRANSFORMERS:
            logger.error("Transformers not available, multilingual analysis disabled")
            return
        
        try:
            logger.info("Loading multilingual sentiment model...")
            
            # Load sentiment analysis model
            sentiment_model_name = self.config["models"]["sentiment"]["primary"]
            # Try loading directly on target device first
            try:
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model=sentiment_model_name,
                    device=0 if self.device.type == "cuda" else -1,
                    return_all_scores=True
                )
                logger.info(f"Sentiment model loaded successfully on {self.device}")
            except Exception as e:
                if "meta tensor" in str(e).lower():
                    logger.warning(f"Meta tensor error on {self.device}, falling back to CPU loading: {e}")
                    # Fallback: load on CPU then move to target device
                    self.sentiment_model = pipeline(
                        "sentiment-analysis",
                        model=sentiment_model_name,
                        device=-1,  # CPU
                        return_all_scores=True
                    )
                    if self.device.type != "cpu":
                        try:
                            self.sentiment_model.model = self.sentiment_model.model.to(self.device)
                            logger.info(f"Sentiment model moved to {self.device} successfully")
                        except Exception as device_error:
                            logger.warning(f"Could not move sentiment model to {self.device}: {device_error}")
                            logger.info("Using sentiment model on CPU")
                else:
                    raise e
            
            logger.info("Multilingual sentiment model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback models if primary models fail."""
        try:
            logger.info("Loading fallback multilingual models...")
            
            fallback_model = self.config["models"]["sentiment"]["fallback"]
            # Try loading directly on target device first
            try:
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model=fallback_model,
                    device=0 if self.device.type == "cuda" else -1,
                    return_all_scores=True
                )
                logger.info(f"Fallback sentiment model loaded successfully on {self.device}")
            except Exception as e:
                if "meta tensor" in str(e).lower():
                    logger.warning(f"Meta tensor error on {self.device}, falling back to CPU loading: {e}")
                    # Fallback: load on CPU then move to target device
                    self.sentiment_model = pipeline(
                        "sentiment-analysis",
                        model=fallback_model,
                        device=-1,  # CPU
                        return_all_scores=True
                    )
                    if self.device.type != "cpu":
                        try:
                            self.sentiment_model.model = self.sentiment_model.model.to(self.device)
                            logger.info(f"Fallback sentiment model moved to {self.device} successfully")
                        except Exception as device_error:
                            logger.warning(f"Could not move fallback sentiment model to {self.device}: {device_error}")
                            logger.info("Using fallback sentiment model on CPU")
                else:
                    raise e
            
            logger.info("Fallback models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load fallback models: {e}")
            raise RuntimeError("Could not load any multilingual models")
    
    def _setup_translator(self):
        """Setup translation service."""
        if not HAS_TRANSLATOR or not self.config["translation"]["enabled"]:
            logger.info("Translation disabled or not available")
            return
        
        try:
            # Use Google Translator as primary
            self.translator = GoogleTranslator(
                source='auto', 
                target=self.config["translation"]["fallback_language"]
            )
            logger.info("Translation service initialized")
        except Exception as e:
            logger.warning(f"Could not initialize translator: {e}")
            self.translator = None
    
    def analyze_sentiment(self, text: str, language: str = None) -> Dict[str, Any]:
        """
        Analyze sentiment of multilingual text.
        
        Args:
            text: Input text to analyze
            language: Optional language code (auto-detected if None)
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
                'language': 'unknown',
                'translated': False,
                'original_text': text,
                'processed_text': text
            }
        
        # Detect language if not provided
        if language is None:
            lang_result = self.language_detector.detect_language(text)
            language = lang_result['language']
            lang_confidence = lang_result['confidence']
        else:
            lang_confidence = 1.0
        
        # Determine if translation is needed
        processed_text = text
        translated = False
        
        if (self.translator and 
            language not in ['en', 'unknown'] and 
            lang_confidence < self.config["thresholds"]["language_confidence"]):
            
            # Translate for better analysis
            try:
                processed_text = self.translator.translate(text)
                translated = True
                logger.debug(f"Translated text from {language}: {text[:50]}... -> {processed_text[:50]}...")
            except Exception as e:
                logger.warning(f"Translation failed: {e}")
                processed_text = text
        
        # Analyze sentiment
        try:
            results = self.sentiment_model(processed_text)
            
            # Process results
            if isinstance(results[0], list):
                # Model returns all scores
                scores = {item['label'].lower(): item['score'] for item in results[0]}
            else:
                # Model returns single prediction
                scores = {results[0]['label'].lower(): results[0]['score']}
            
            # Normalize scores to standard format
            normalized_scores = self._normalize_sentiment_scores(scores)
            
            # Determine primary sentiment
            primary_sentiment = max(normalized_scores, key=normalized_scores.get)
            confidence = normalized_scores[primary_sentiment]
            
            return {
                'sentiment': primary_sentiment,
                'confidence': confidence,
                'scores': normalized_scores,
                'language': language,
                'language_confidence': lang_confidence,
                'translated': translated,
                'original_text': text,
                'processed_text': processed_text
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
                'language': language,
                'language_confidence': lang_confidence,
                'translated': translated,
                'original_text': text,
                'processed_text': processed_text
            }
    
    def _normalize_sentiment_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize sentiment scores to standard positive/negative/neutral format."""
        # Map various model outputs to standard format
        normalized = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        for label, score in scores.items():
            label_lower = label.lower()
            
            if label_lower in ['positive', 'pos', 'label_2', '2']:
                normalized['positive'] = score
            elif label_lower in ['negative', 'neg', 'label_0', '0']:
                normalized['negative'] = score
            elif label_lower in ['neutral', 'neu', 'label_1', '1']:
                normalized['neutral'] = score
        
        # Ensure scores sum to 1.0
        total = sum(normalized.values())
        if total > 0:
            normalized = {k: v/total for k, v in normalized.items()}
        else:
            normalized = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        return normalized
    
    def batch_analyze(self, texts: List[str], languages: List[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            languages: Optional list of language codes
            
        Returns:
            List of sentiment analysis results
        """
        if languages is None:
            languages = [None] * len(texts)
        
        results = []
        batch_size = self.config.get("batch_size", 16)
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_langs = languages[i:i + batch_size]
            
            batch_results = []
            for text, lang in zip(batch_texts, batch_langs):
                result = self.analyze_sentiment(text, lang)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if i % (batch_size * 5) == 0:
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} multilingual texts")
        
        return results
    
    def get_language_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get language distribution from analysis results.
        
        Args:
            results: List of analysis results
            
        Returns:
            Language distribution statistics
        """
        if not results:
            return {}
        
        language_counts = defaultdict(int)
        sentiment_by_language = defaultdict(lambda: defaultdict(int))
        translation_count = 0
        total_confidence = 0.0
        
        for result in results:
            language = result.get('language', 'unknown')
            sentiment = result.get('sentiment', 'neutral')
            
            language_counts[language] += 1
            sentiment_by_language[language][sentiment] += 1
            
            if result.get('translated', False):
                translation_count += 1
            
            total_confidence += result.get('language_confidence', 0.0)
        
        total_texts = len(results)
        
        # Calculate percentages
        language_percentages = {
            lang: count / total_texts 
            for lang, count in language_counts.items()
        }
        
        # Calculate sentiment distribution by language
        sentiment_by_lang_pct = {}
        for lang, sentiments in sentiment_by_language.items():
            total_lang = language_counts[lang]
            sentiment_by_lang_pct[lang] = {
                sentiment: count / total_lang 
                for sentiment, count in sentiments.items()
            }
        
        return {
            'total_texts': total_texts,
            'language_distribution': language_percentages,
            'sentiment_by_language': sentiment_by_lang_pct,
            'translation_rate': translation_count / total_texts,
            'average_language_confidence': total_confidence / total_texts,
            'top_languages': sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }


def create_multilingual_config():
    """Create default multilingual configuration file."""
    config_dir = Path(__file__).parent.parent / "config"
    config_dir.mkdir(exist_ok=True)
    
    analyzer = MultilingualSentimentAnalyzer()
    config_path = config_dir / "multilingual_config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(analyzer.config, f, indent=2, ensure_ascii=False)
    
    print(f"Multilingual configuration created at: {config_path}")


def main():
    """Main function to demonstrate multilingual analysis."""
    
    # Create config if it doesn't exist
    create_multilingual_config()
    
    # Initialize analyzer
    print("Initializing Multilingual Analyzer...")
    analyzer = MultilingualSentimentAnalyzer()
    
    # Test with multilingual comments
    test_comments = [
        # English
        "This product is absolutely amazing! Love it so much!",
        "Terrible quality, waste of money ",
        
        # Spanish
        "¡Me encanta este producto! Excelente calidad y muy buen precio ",
        "No me gustó nada, muy malo ",
        
        # French
        "C'est fantastique! Je le recommande vivement ",
        "Très déçu de cet achat, qualité horrible ",
        
        # German
        "Sehr gut! Bin sehr zufrieden mit diesem Produkt ",
        "Schlechte Qualität, kann ich nicht empfehlen ",
        
        # Italian
        "Ottimo prodotto, molto soddisfatto! ",
        "Non vale i soldi spesi, pessimo ",
        
        # Portuguese
        "Produto incrível! Recomendo a todos! ",
        "Qualidade terrível, não comprem ",
    ]
    
    print(f"\nAnalyzing {len(test_comments)} multilingual comments...")
    results = analyzer.batch_analyze(test_comments)
    
    # Generate language distribution
    lang_stats = analyzer.get_language_distribution(results)
    
    # Display results
    print("\n" + "="*60)
    print("MULTILINGUAL SENTIMENT ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Total Comments Analyzed: {lang_stats['total_texts']}")
    print(f"Translation Rate: {lang_stats['translation_rate']:.1%}")
    print(f"Average Language Confidence: {lang_stats['average_language_confidence']:.3f}")
    
    print("\nLanguage Distribution:")
    for lang, percentage in sorted(lang_stats['language_distribution'].items(), 
                                 key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {percentage:.1%}")
    
    print("\nSentiment by Language:")
    for lang, sentiments in lang_stats['sentiment_by_language'].items():
        print(f"  {lang}:")
        for sentiment, pct in sentiments.items():
            print(f"    {sentiment}: {pct:.1%}")
    
    print("\nDetailed Analysis Examples:")
    print("-" * 40)
    
    for i, (comment, result) in enumerate(zip(test_comments, results)):
        if i >= 8:  # Show first 8 examples
            break
        
        print(f"\nComment {i+1}: {comment}")
        print(f"Language: {result['language']} (confidence: {result['language_confidence']:.3f})")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        if result['translated']:
            print(f"Translated: {result['processed_text']}")
    
    # Save results
    current_dir = Path(__file__).parent.parent
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_data = {
        'language_statistics': lang_stats,
        'detailed_results': results,
        'timestamp': datetime.now().isoformat(),
        'analyzer_config': analyzer.config
    }
    
    output_file = results_dir / 'multilingual_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()