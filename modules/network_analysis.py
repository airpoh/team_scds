"""
Network Analysis Module for CommentSense Pipeline
Orchestrates category classification, persona clustering, and influencer detection

This module provides:
- Category + Sub-Topic Classification using zero-shot BART and BERTopic
- Voice-of-Customer Personas using Sentence Transformers + HDBSCAN  
- Influencer & Community Detection using PageRank + Louvain
- Unified network analysis pipeline
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Import the new modular components
try:
    from .category_classification import create_category_classifier
    from .persona_clustering import create_persona_clustering  
    from .influencer_community_detection import create_influencer_community_detector
except ImportError as e:
    logging.error(f"Could not import modular components: {e}")
    create_category_classifier = None
    create_persona_clustering = None
    create_influencer_community_detector = None

# Import utilities
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.alvin_utils import normalize_columns, clean_text
except ImportError as e:
    logging.warning(f"Could not import utilities: {e}")
    normalize_columns = None
    clean_text = None

logger = logging.getLogger(__name__)

class NetworkAnalyzer:
    """
    Network analysis module that orchestrates the modular analysis components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the network analyzer with modular components"""
        
        self.config = config or self._get_default_config()
        
        # Validate that required modular components are available
        if any(comp is None for comp in [create_category_classifier, create_persona_clustering, create_influencer_community_detector]):
            raise ImportError("Network analysis requires modular analysis components")
        
        # Initialize modular components
        self.category_classifier = create_category_classifier(config)
        self.persona_clustering = create_persona_clustering(config) 
        self.influencer_detector = create_influencer_community_detector(config)
        
        logger.info("Network Analysis module initialized with modular components")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for network analysis"""
        return {
            "fast_mode": True,
            "use_zeroshot": False,
            "use_persona_cluster": True,
            "use_bertopic": False,
            "min_interactions": 2,  # Minimum interactions to include a user in network
            "enable_categories": True,
            "enable_influencers": True,
            "enable_personas": True,
            "enable_communities": True
        }
    
    def analyze_network(self, comments_df: pd.DataFrame, user_metadata: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform network analysis using the new modular components
        
        Args:
            comments_df: DataFrame with comment data including user interactions
            user_metadata: Optional DataFrame with user follower counts, etc.
            
        Returns:
            Dictionary containing network analysis results
        """
        
        try:
            logger.info(f"Starting modular network analysis on {len(comments_df)} comments...")
            print(f"DEBUG: Network analysis called with {len(comments_df)} comments")
            
            # Prepare data for network analysis
            if normalize_columns is None:
                logger.error("normalize_columns utility not available")
                return {"error": "Data normalization utilities not available"}
            
            prepared_df = normalize_columns(comments_df)
            
            if len(prepared_df) == 0:
                logger.warning("No valid data for network analysis")
                return {"error": "No valid data for network analysis"}
            
            # Clean text data
            if clean_text and "text" in prepared_df.columns:
                prepared_df["text_clean"] = prepared_df["text"].map(clean_text)
            else:
                prepared_df["text_clean"] = prepared_df.get("text", "")
            
            # Step 1: Category Classification
            categories_result = None
            if self.config.get("enable_categories", True):
                try:
                    categories_result = self.category_classifier.analyze_categories(
                        prepared_df,
                        text_column="text_clean",
                        use_zero_shot=self.config.get("use_zeroshot", False),
                        use_bertopic=self.config.get("use_bertopic", False)
                    )
                    logger.info("Category classification completed")
                except Exception as e:
                    logger.warning(f"Category classification failed: {e}")
                    categories_result = pd.DataFrame()
            
            # Step 2: Persona Clustering
            personas_result = None
            if self.config.get("enable_personas", True) and categories_result is not None:
                try:
                    personas_result = self.persona_clustering.analyze_personas(
                        prepared_df,
                        categories_result if len(categories_result) > 0 else prepared_df,
                        text_column="text_clean",
                        use_clustering=self.config.get("use_persona_cluster", True),
                        fast_mode=self.config.get("fast_mode", True)
                    )
                    logger.info("Persona clustering completed")
                except Exception as e:
                    logger.warning(f"Persona clustering failed: {e}")
                    personas_result = pd.DataFrame()
            
            # Step 3: Influencer & Community Detection
            influencers_result = None
            communities_result = None
            if self.config.get("enable_influencers", True) and self.config.get("enable_communities", True):
                try:
                    influencers_result, communities_result = self.influencer_detector.analyze_influencers_and_communities(
                        prepared_df,
                        user_metadata=user_metadata
                    )
                    logger.info("Influencer and community detection completed")
                except Exception as e:
                    logger.warning(f"Influencer detection failed: {e}")
                    influencers_result = pd.DataFrame()
                    communities_result = pd.DataFrame()
            
            # Structure results
            structured_results = {
                "df": prepared_df,
                "cats_out": categories_result if categories_result is not None else pd.DataFrame(),
                "personas_df": personas_result if personas_result is not None else pd.DataFrame(),
                "influencers_df": influencers_result if influencers_result is not None else pd.DataFrame(),
                "communities_df": communities_result if communities_result is not None else pd.DataFrame()
            }
            
            logger.info("Modular network analysis completed successfully")
            return structured_results
            
        except Exception as e:
            logger.error(f"Error in network analysis: {e}")
            return {"error": str(e)}

def create_network_analyzer(config: Optional[Dict[str, Any]] = None) -> NetworkAnalyzer:
    """Factory function to create network analyzer"""
    return NetworkAnalyzer(config)
