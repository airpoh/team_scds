"""
CommentSense - Emotion and Sarcasm Detection Module
===================================================

This module implements emotion and sarcasm detection for the CommentSense system.
Supports fine-grained emotion recognition and sarcasm detection in social media comments.

Features:
- Multi-emotion classification using GoEmotions model
- Sarcasm detection using fine-tuned BERT
- Post-processing to adjust sentiment when sarcasm is detected
- True batch processing for improved performance
- Configurable models and thresholds via config file
"""

import torch
import pandas as pd
import numpy as np
from transformers import pipeline
from typing import List, Dict, Tuple, Optional, Union
import logging
import warnings
import json
import os
import platform
from datetime import datetime
from pathlib import Path


# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionSarcasmDetector:
    """
    Advanced emotion and sarcasm detection for social media comments.
    
    Combines GoEmotions for emotion classification and fine-tuned models
    for sarcasm detection with intelligent post-processing.
    """
    
    def __init__(self, config_path: str = None, device: str = "auto"):
        """
        Initialize the emotion and sarcasm detection models.
        
        Args:
            config_path: Path to configuration JSON file
            device: Device to run models on ('auto', 'cpu', 'cuda', 'mps')
        """
        self.device = self._setup_device(device)
        self.config = self._load_config(config_path)
        self.emotion_model = None
        self.sarcasm_model = None
        
        self._load_models()
    
    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device for model inference with optimal allocation."""
        if device == "auto":
            import platform
            
            if torch.cuda.is_available():
                # Windows/Linux with CUDA - Use GPU for transformer models (optimal)
                logger.info("CUDA available, using GPU for transformer models (optimal)")
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Apple Silicon - Force CPU to prevent segmentation faults
                if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
                    logger.info("Apple Silicon detected - using CPU for stability (transformer models)")
                    return "cpu"
                else:
                    logger.info("MPS available, using GPU acceleration")
                    return "mps"
            else:
                logger.info("Using CPU for transformer models")
                return "cpu"
        return device
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from JSON file."""
        if config_path is None:
            # Default to config directory relative to this file
            current_dir = Path(__file__).parent
            config_path = current_dir.parent / "config" / "emotion_sarcasm_config.json"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            # Return default config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration if config file is not available."""
        return {
            "models": {
                "emotion": {
                    "primary": "j-hartmann/emotion-english-distilroberta-base",
                    "fallback": "SamLowe/roberta-base-go_emotions"
                },
                "sarcasm": {
                    "primary": "cardiffnlp/twitter-roberta-base-irony", 
                    "fallback": "helinivan/english-sarcasm-detector"
                }
            },
            "thresholds": {
                "sarcasm_confidence": 0.6,
                "emotion_confidence": 0.1
            },
            "labels": {
                "sarcasm_positive": ["SARCASM", "irony", "IRONY", "1", "sarcastic"],
                "sarcasm_negative": ["not_sarcasm", "literal", "0", "NOT_SARCASM"]
            },
            "emotion_mapping": {
                "admiration": "positive", "amusement": "positive", "anger": "negative",
                "annoyance": "negative", "approval": "positive", "caring": "positive",
                "confusion": "neutral", "curiosity": "neutral", "desire": "neutral",
                "disappointment": "negative", "disapproval": "negative", "disgust": "negative",
                "embarrassment": "negative", "excitement": "positive", "fear": "negative",
                "gratitude": "positive", "grief": "negative", "joy": "positive",
                "love": "positive", "nervousness": "negative", "optimism": "positive",
                "pride": "positive", "realization": "neutral", "relief": "positive",
                "remorse": "negative", "sadness": "negative", "surprise": "neutral",
                "neutral": "neutral"
            },
            "batch_size": 32,
            "max_length": 512
        }
    
    def _get_device_id(self) -> int:
        """Get device ID for pipeline initialization."""
        if self.device == "cuda":
            return 0
        elif self.device == "mps":
            return 0  # MPS uses device 0
        else:
            return -1  # CPU
    
    def _load_models(self):
        """Load the emotion and sarcasm detection models."""
        try:
            device_id = self._get_device_id()
            logger.info(f"Loading emotion detection model...")
            
            # Load emotion model
            self.emotion_model = pipeline(
                "text-classification",
                model=self.config["models"]["emotion"]["primary"],
                device=device_id,
                return_all_scores=True,
                max_length=self.config.get("max_length", 512),
                truncation=True
            )
            logger.info(f"Emotion model loaded successfully")
            
            logger.info(f"Loading sarcasm detection model...")
            
            # Load sarcasm model
            self.sarcasm_model = pipeline(
                "text-classification", 
                model=self.config["models"]["sarcasm"]["primary"],
                device=device_id,
                return_all_scores=True,
                max_length=self.config.get("max_length", 512),
                truncation=True
            )
            logger.info(f"Sarcasm model loaded successfully")
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading primary models: {e}")
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback models if primary models fail."""
        try:
            logger.info(f"Loading fallback models...")
            device_id = self._get_device_id()
            
            # Load fallback emotion model
            self.emotion_model = pipeline(
                "text-classification",
                model=self.config["models"]["emotion"]["fallback"],
                device=device_id,
                return_all_scores=True,
                max_length=self.config.get("max_length", 512),
                truncation=True
            )
            logger.info(f"Fallback emotion model loaded successfully")
            
            # Load fallback sarcasm model
            self.sarcasm_model = pipeline(
                "text-classification",
                model=self.config["models"]["sarcasm"]["fallback"],
                device=device_id,
                return_all_scores=True,
                max_length=self.config.get("max_length", 512),
                truncation=True
            )
            logger.info(f"Fallback sarcasm model loaded successfully")
            
            logger.info("Fallback models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load fallback models: {e}")
            raise RuntimeError("Could not load any emotion/sarcasm models")
    
    def detect_emotions_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Detect emotions in multiple texts using batch processing.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of dictionaries mapping emotion labels to confidence scores
        """
        try:
            if not texts:
                return []
            
            # Filter out empty texts and keep track of indices
            valid_texts = []
            valid_indices = []
            results = []
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text.strip())
                    valid_indices.append(i)
                
            if not valid_texts:
                return [{"neutral": 1.0} for _ in texts]
            
            # Get emotion predictions for all valid texts at once
            batch_results = self.emotion_model(valid_texts)
            
            # Initialize results list with neutral emotions for all texts
            results = [{"neutral": 1.0} for _ in texts]
            
            # Process results for valid texts
            for idx, batch_result in enumerate(batch_results):
                original_idx = valid_indices[idx]
                
                emotions = {}
                if isinstance(batch_result, list):
                    # Handle models that return all scores
                    for result in batch_result:
                        emotions[result['label']] = result['score']
                else:
                    # Handle models that return single prediction
                    emotions = {batch_result['label']: batch_result['score']}
                
                # Filter low confidence predictions
                filtered_emotions = {}
                threshold = self.config["thresholds"]["emotion_confidence"]
                for emotion, score in emotions.items():
                    if score > threshold:
                        filtered_emotions[emotion] = float(score)
                
                results[original_idx] = filtered_emotions if filtered_emotions else {"neutral": 1.0}
            
            return results
            
        except Exception as e:
            logger.warning(f"Error in batch emotion detection: {e}")
            return [{"neutral": 1.0} for _ in texts]
    
    def detect_sarcasm_batch(self, texts: List[str]) -> List[Tuple[bool, float]]:
        """
        Detect sarcasm in multiple texts using batch processing.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of tuples (is_sarcastic, confidence_score)
        """
        try:
            if not texts:
                return []
                
            # Filter out empty texts and keep track of indices
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text.strip())
                    valid_indices.append(i)
                    
            if not valid_texts:
                return [(False, 0.0) for _ in texts]
            
            # Get sarcasm predictions for all valid texts at once
            batch_results = self.sarcasm_model(valid_texts)
            
            # Initialize results list
            results = [(False, 0.0) for _ in texts]
            
            # Process results for valid texts
            sarcasm_positive_labels = self.config["labels"]["sarcasm_positive"]
            sarcasm_negative_labels = self.config["labels"]["sarcasm_negative"]
            threshold = self.config["thresholds"]["sarcasm_confidence"]
            
            for idx, batch_result in enumerate(batch_results):
                original_idx = valid_indices[idx]
                sarcasm_score = 0.0
                
                if isinstance(batch_result, list):
                    # Handle models that return all scores
                    sarcasm_scores = {r['label']: r['score'] for r in batch_result}
                    
                    # Look for positive sarcasm labels
                    for label in sarcasm_positive_labels:
                        if label in sarcasm_scores:
                            sarcasm_score = max(sarcasm_score, sarcasm_scores[label])
                    
                    # If no positive label found, try inverting negative labels
                    if sarcasm_score == 0.0:
                        for label in sarcasm_negative_labels:
                            if label in sarcasm_scores:
                                sarcasm_score = 1 - sarcasm_scores[label]
                                break
                else:
                    # Handle single prediction
                    if batch_result['label'] in sarcasm_positive_labels:
                        sarcasm_score = batch_result['score']
                    elif batch_result['label'] in sarcasm_negative_labels:
                        sarcasm_score = 1 - batch_result['score']
                    else:
                        # Default fallback
                        sarcasm_score = batch_result['score']
                
                is_sarcastic = sarcasm_score > threshold
                results[original_idx] = (is_sarcastic, float(sarcasm_score))
            
            return results
            
        except Exception as e:
            logger.warning(f"Error in batch sarcasm detection: {e}")
            return [(False, 0.0) for _ in texts]
    
    def detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in a single text (wrapper for batch processing).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping emotion labels to confidence scores
        """
        results = self.detect_emotions_batch([text])
        return results[0] if results else {"neutral": 1.0}
    
    def detect_sarcasm(self, text: str) -> Tuple[bool, float]:
        """
        Detect sarcasm in a single text (wrapper for batch processing).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (is_sarcastic, confidence_score)
        """
        results = self.detect_sarcasm_batch([text])
        return results[0] if results else (False, 0.0)
    
    def batch_analyze(self, texts: List[str], batch_size: int = None) -> List[Dict]:
        """
        Analyze multiple comments in batches for efficiency with true batch processing.
        
        Args:
            texts: List of text comments to analyze
            batch_size: Number of texts to process at once (uses config default if None)
            
        Returns:
            List of analysis results for each text
        """
        if batch_size is None:
            batch_size = self.config.get("batch_size", 32)
            
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Process entire batch at once
            batch_emotions = self.detect_emotions_batch(batch_texts)
            batch_sarcasm = self.detect_sarcasm_batch(batch_texts)
            
            # Combine results
            batch_results = []
            for j, text in enumerate(batch_texts):
                result = self._combine_analysis_results(
                    text, batch_emotions[j], batch_sarcasm[j]
                )
                batch_results.append(result)
            
            all_results.extend(batch_results)
            
            if i % (batch_size * 5) == 0:
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} comments")
        
        return all_results
    
    def _combine_analysis_results(self, text: str, emotions: Dict[str, float], sarcasm_result: Tuple[bool, float]) -> Dict:
        """
        Combine emotion and sarcasm results into a comprehensive analysis.
        
        Args:
            text: Original text
            emotions: Emotion detection results
            sarcasm_result: Sarcasm detection results
            
        Returns:
            Combined analysis dictionary
        """
        if not text or not text.strip():
            return {
                'emotions': {'neutral': 1.0},
                'is_sarcastic': False,
                'sarcasm_confidence': 0.0,
                'dominant_emotion': 'neutral',
                'sentiment': 'neutral',
                'adjusted_sentiment': 'neutral',
                'confidence': 0.0
            }
        
        is_sarcastic, sarcasm_confidence = sarcasm_result
        
        # Find dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        emotion_confidence = emotions[dominant_emotion]
        
        # Map emotion to sentiment using config
        sentiment = self.config["emotion_mapping"].get(dominant_emotion, 'neutral')
        
        # Adjust sentiment based on sarcasm
        adjusted_sentiment = self._adjust_for_sarcasm(sentiment, is_sarcastic, sarcasm_confidence)
        
        return {
            'emotions': emotions,
            'is_sarcastic': is_sarcastic,
            'sarcasm_confidence': sarcasm_confidence,
            'dominant_emotion': dominant_emotion,
            'sentiment': sentiment,
            'adjusted_sentiment': adjusted_sentiment,
            'confidence': emotion_confidence
        }
    
    def analyze_comment(self, text: str) -> Dict[str, Union[Dict, bool, float]]:
        """
        Perform comprehensive emotion and sarcasm analysis on a single comment.
        
        Args:
            text: Input comment text
            
        Returns:
            Dictionary containing emotion scores, sarcasm detection, and adjusted sentiment
        """
        # Use batch processing for consistency
        results = self.batch_analyze([text])
        return results[0] if results else {
            'emotions': {'neutral': 1.0},
            'is_sarcastic': False,
            'sarcasm_confidence': 0.0,
            'dominant_emotion': 'neutral',
            'sentiment': 'neutral',
            'adjusted_sentiment': 'neutral',
            'confidence': 0.0
        }
    
    def _adjust_for_sarcasm(self, sentiment: str, is_sarcastic: bool, sarcasm_confidence: float) -> str:
        """
        Adjust sentiment based on sarcasm detection.
        
        Args:
            sentiment: Original sentiment
            is_sarcastic: Whether sarcasm was detected
            sarcasm_confidence: Confidence of sarcasm detection
            
        Returns:
            Adjusted sentiment
        """
        if not is_sarcastic or sarcasm_confidence < 0.7:
            return sentiment
        
        # Invert sentiment for high-confidence sarcasm
        if sentiment == 'positive':
            return 'negative'
        elif sentiment == 'negative':
            return 'positive'
        else:
            return 'neutral'  # Keep neutral as is
    
    def get_emotion_summary(self, results: List[Dict]) -> Dict[str, float]:
        """
        Generate summary statistics from batch analysis results.
        
        Args:
            results: List of analysis results from batch_analyze
            
        Returns:
            Dictionary with emotion distribution and other metrics
        """
        if not results:
            return {}
        
        # Aggregate emotions
        emotion_counts = {}
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        sarcasm_count = 0
        total_confidence = 0
        
        for result in results:
            # Count dominant emotions
            dominant_emotion = result['dominant_emotion']
            emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
            
            # Count sentiments
            sentiment = result['adjusted_sentiment']
            sentiment_counts[sentiment] += 1
            
            # Count sarcasm
            if result['is_sarcastic']:
                sarcasm_count += 1
            
            # Sum confidence
            total_confidence += result['confidence']
        
        total_comments = len(results)
        
        return {
            'total_comments': total_comments,
            'emotion_distribution': {k: v/total_comments for k, v in emotion_counts.items()},
            'sentiment_distribution': {k: v/total_comments for k, v in sentiment_counts.items()},
            'sarcasm_rate': sarcasm_count / total_comments,
            'average_confidence': total_confidence / total_comments,
            'top_emotions': sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }


def load_and_process_comments(file_paths: List[str]) -> pd.DataFrame:
    """
    Load and combine comment datasets from multiple CSV files.
    
    Args:
        file_paths: List of paths to CSV files containing comments
        
    Returns:
        Combined DataFrame with processed comments
    """
    all_comments = []
    
    for file_path in file_paths:
        try:
            logger.info(f"Loading comments from {file_path}")
            df = pd.read_csv(file_path)
            
            # Ensure we have the required columns
            if 'textOriginal' in df.columns:
                df['comment_text'] = df['textOriginal']
            elif 'text' in df.columns:
                df['comment_text'] = df['text']
            else:
                logger.warning(f"No text column found in {file_path}")
                continue
            
            # Add metadata
            df['source_file'] = file_path
            df['comment_id'] = df.index if 'commentId' not in df.columns else df['commentId']
            
            all_comments.append(df)
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    if not all_comments:
        raise ValueError("No valid comment files could be loaded")
    
    # Combine all comments
    combined_df = pd.concat(all_comments, ignore_index=True)
    
    # Clean and filter comments
    combined_df = combined_df.dropna(subset=['comment_text'])
    combined_df = combined_df[combined_df['comment_text'].str.len() > 0]
    combined_df['comment_text'] = combined_df['comment_text'].str.strip()
    
    logger.info(f"Loaded {len(combined_df)} total comments")
    return combined_df


def main():
    """Main function to demonstrate the emotion and sarcasm detection module."""
    
    # Use relative paths
    current_dir = Path(__file__).parent.parent
    dataset_dir = current_dir / "dataset"
    results_dir = current_dir / "results"
    
    # Ensure results directory exists
    results_dir.mkdir(exist_ok=True)
    
    # Dataset file paths
    dataset_files = [
        dataset_dir / "comments1.csv",
        dataset_dir / "comments2.csv", 
        dataset_dir / "comments3.csv",
        dataset_dir / "comments4.csv",
        dataset_dir / "comments5.csv"
    ]
    
    # Filter existing files
    existing_files = [str(f) for f in dataset_files if f.exists()]
    if not existing_files:
        logger.error("No dataset files found!")
        return
    
    # Initialize detector
    print("Initializing Emotion & Sarcasm Detector...")
    detector = EmotionSarcasmDetector()
    
    # Load comments
    print("Loading comment datasets...")
    comments_df = load_and_process_comments(existing_files)
    
    # Sample analysis on first 100 comments for demonstration
    sample_size = min(100, len(comments_df))
    sample_comments = comments_df['comment_text'].head(sample_size).tolist()
    
    print(f"Analyzing {sample_size} sample comments...")
    results = detector.batch_analyze(sample_comments)
    
    # Generate summary
    summary = detector.get_emotion_summary(results)
    
    # Display results
    print("\n" + "="*50)
    print("EMOTION & SARCASM ANALYSIS RESULTS")
    print("="*50)
    
    print(f"Total Comments Analyzed: {summary['total_comments']}")
    print(f"Sarcasm Detection Rate: {summary['sarcasm_rate']:.2%}")
    print(f"Average Confidence: {summary['average_confidence']:.3f}")
    
    print("\nSentiment Distribution:")
    for sentiment, ratio in summary['sentiment_distribution'].items():
        print(f"  {sentiment.title()}: {ratio:.2%}")
    
    print("\nTop 5 Emotions:")
    for emotion, count in summary['top_emotions']:
        print(f"  {emotion.title()}: {count} comments")
    
    # Show some example analyses
    print("\nExample Analyses:")
    print("-" * 30)
    
    num_examples = min(5, len(sample_comments))
    for i in range(num_examples):
        comment = sample_comments[i]
        result = results[i]
        print(f"\nComment {i+1}: \"{comment[:100]}{'...' if len(comment) > 100 else ''}\"")
        print(f"Dominant Emotion: {result['dominant_emotion']} ({result['confidence']:.3f})")
        print(f"Sentiment: {result['sentiment']} → {result['adjusted_sentiment']}")
        print(f"Sarcastic: {result['is_sarcastic']} ({result['sarcasm_confidence']:.3f})")
    
    # Save results to JSON for further analysis
    output_data = {
        'summary': summary,
        'timestamp': datetime.now().isoformat(),
        'sample_results': results[:20],  # Save first 20 detailed results
        'config_used': detector.config,
        'device_used': detector.device
    }
    
    output_file = results_dir / 'emotion_sarcasm_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()