"""
Influencer & Community Detection Module for CommentSense Pipeline

This module provides:
- PageRank-based influencer identification with personalized scoring
- Betweenness centrality analysis for bridge users
- Louvain community detection on social graphs
- Community health metrics and sentiment analysis
- Network graph construction from mentions and replies
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# Import configuration
try:
    import json
    import os
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'alvin_models_config.json')
    with open(config_path, 'r') as f:
        CONFIG = json.load(f)
except Exception:
    CONFIG = {}

logger = logging.getLogger(__name__)

# ---------------- Required Dependencies ----------------
import networkx as nx

try:
    import community as community_louvain  # python-louvain
except Exception:
    community_louvain = None
    logger.warning("python-louvain not available, community detection will use fallback")

# ---------------- Text Processing ----------------
HANDLE_RE = re.compile(r"@[\w\._-]+", re.I)

def normalize_handle(handle: str) -> str:
    """Normalize social media handle"""
    handle = str(handle).strip()
    return handle[1:] if handle.startswith("@") else handle

def extract_mentions(text: str) -> List[str]:
    """Extract @mentions from text"""
    if not isinstance(text, str):
        return []
    return [normalize_handle(m) for m in HANDLE_RE.findall(text)]

class InfluencerCommunityDetector:
    """
    Influencer & Community Detection using PageRank, Betweenness, and Louvain
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the influencer and community detector"""
        
        self.config = config or CONFIG
        
        if community_louvain is None:
            logger.warning("Community detection will use basic partitioning without Louvain")
        
        logger.info("Influencer & Community Detection module initialized")
    
    def build_social_graph(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Build social graph edges from mentions and replies
        
        Args:
            df: DataFrame with user interactions
        
        Returns:
            List of (source, target) edges
        """
        edges = []
        
        # Reply-based edges
        if "reply_to_id" in df.columns and "comment_id" in df.columns:
            # Build mapping from comment_id to user_id
            comment_to_user = dict(zip(
                df["comment_id"].astype(str),
                df["user_id"].astype(str)
            ))
            
            # Process replies
            reply_data = df.dropna(subset=["reply_to_id"])
            for _, row in reply_data.iterrows():
                source_user = str(row["user_id"])
                target_user = comment_to_user.get(str(row["reply_to_id"]))
                
                if source_user and target_user and source_user != target_user:
                    edges.append((source_user, target_user))
        
        # Mention-based edges
        for _, row in df.iterrows():
            source_user = str(row.get("user_id"))
            if not source_user or source_user == "nan":
                continue
            
            # Extract mentions from dedicated column
            mentions_set = set()
            mention_col = row.get("mentions")
            if isinstance(mention_col, str) and mention_col.strip():
                mention_handles = [normalize_handle(m) for m in mention_col.split(",") if m.strip()]
                mentions_set.update(mention_handles)
            
            # Extract mentions from text content
            text_mentions = extract_mentions(row.get("text", ""))
            mentions_set.update(text_mentions)
            
            # Create edges for all mentions
            for mentioned_user in mentions_set:
                if mentioned_user and mentioned_user != source_user:
                    edges.append((source_user, mentioned_user))
        
        logger.info(f"Built social graph with {len(edges)} edges")
        return edges
    
    def calculate_graph_metrics(self, 
                               edges: List[Tuple[str, str]], 
                               user_engagement: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
        """
        Calculate PageRank, betweenness centrality, and community assignments
        
        Args:
            edges: List of (source, target) edges
            user_engagement: Dictionary mapping user_id to engagement score
        
        Returns:
            Tuple of (pagerank_scores, betweenness_scores, community_assignments)
        """
        if not edges:
            return {}, {}, {}
        
        # Build directed graph with edge weights
        G = nx.DiGraph()
        for source, target in edges:
            if G.has_edge(source, target):
                G[source][target]["weight"] += 1
            else:
                G.add_edge(source, target, weight=1)
        
        logger.info(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # PageRank with personalized scores based on engagement
        personalization = {
            node: max(0.1, float(user_engagement.get(node, 1.0))) 
            for node in G.nodes()
        }
        
        pagerank_scores = nx.pagerank(
            G, 
            alpha=0.85, 
            weight="weight", 
            personalization=personalization
        )
        
        # Betweenness centrality (with approximation for large graphs)
        num_nodes = G.number_of_nodes()
        k = None
        if num_nodes > 600:
            k = min(200, max(20, int(0.05 * num_nodes)))
            logger.info(f"Using k-approximation for betweenness: k={k}")
        
        betweenness_scores = nx.betweenness_centrality(
            G, 
            k=k, 
            seed=42, 
            weight="weight"
        )
        
        # Community detection using Louvain on undirected projection
        UG = nx.Graph()
        for u, v, data in G.edges(data=True):
            weight = data.get("weight", 1)
            if UG.has_edge(u, v):
                UG[u][v]["weight"] += weight
            else:
                UG.add_edge(u, v, weight=weight)
        
        # Apply Louvain community detection
        if community_louvain and UG.number_of_edges() > 0:
            community_assignments = community_louvain.best_partition(
                UG, 
                weight="weight", 
                random_state=42
            )
            logger.info(f"Detected {len(set(community_assignments.values()))} communities")
        else:
            # Fallback: assign each node to its own community
            community_assignments = {node: str(i) for i, node in enumerate(UG.nodes())}
            logger.info("Used fallback community detection")
        
        # Convert community IDs to strings
        community_assignments = {
            node: str(comm_id) 
            for node, comm_id in community_assignments.items()
        }
        
        return pagerank_scores, betweenness_scores, community_assignments
    
    def calculate_community_health(self, 
                                  df: pd.DataFrame, 
                                  influencers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate health metrics for each community
        
        Args:
            df: Original DataFrame with comments
            influencers_df: DataFrame with user-community mappings
        
        Returns:
            DataFrame with community health metrics
        """
        # Map users to communities
        user_to_community = dict(zip(
            influencers_df["user_id"].astype(str),
            influencers_df["community_id"].astype(str)
        ))
        
        # Add community assignments to comments
        temp_df = df.copy()
        temp_df["community_id"] = temp_df["user_id"].astype(str).map(user_to_community)
        temp_df = temp_df.dropna(subset=["community_id"])
        
        if len(temp_df) == 0:
            return pd.DataFrame()
        
        # Calculate health metrics for each community
        health_metrics = []
        
        for community_id, group in temp_df.groupby("community_id"):
            metrics = {"community_id": str(community_id)}
            
            # Basic metrics
            metrics["total_comments"] = len(group)
            metrics["unique_users"] = group["user_id"].nunique()
            metrics["avg_comments_per_user"] = metrics["total_comments"] / max(1, metrics["unique_users"])
            
            # Sentiment health
            if "sentiment" in group.columns:
                sentiment_dist = group["sentiment"].value_counts(normalize=True)
                metrics["positive_ratio"] = sentiment_dist.get("positive", 0.0)
                metrics["negative_ratio"] = sentiment_dist.get("negative", 0.0)
                metrics["sentiment_health"] = metrics["positive_ratio"] - metrics["negative_ratio"]
            else:
                metrics["positive_ratio"] = 0.33
                metrics["negative_ratio"] = 0.33
                metrics["sentiment_health"] = 0.0
            
            # Engagement metrics
            if "likes" in group.columns:
                metrics["avg_likes"] = float(group["likes"].mean())
                metrics["total_likes"] = int(group["likes"].sum())
            else:
                metrics["avg_likes"] = 0.0
                metrics["total_likes"] = 0
            
            # Quality and spam metrics
            if "spamness" in group.columns:
                metrics["avg_spamness"] = float(group["spamness"].mean())
                metrics["spam_health"] = 1.0 - metrics["avg_spamness"]
            else:
                metrics["avg_spamness"] = 0.1
                metrics["spam_health"] = 0.9
            
            if "cqs" in group.columns:
                metrics["avg_quality"] = float(group["cqs"].mean())
            else:
                metrics["avg_quality"] = 0.5
            
            health_metrics.append(metrics)
        
        return pd.DataFrame(health_metrics).fillna(0)
    
    def summarize_communities(self, 
                             df: pd.DataFrame, 
                             influencers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create community summary with top users and example comments
        
        Args:
            df: Original DataFrame with comments
            influencers_df: DataFrame with influencer data
        
        Returns:
            DataFrame with community summaries
        """
        if len(influencers_df) == 0:
            return pd.DataFrame()
        
        # Ensure string community IDs
        influencers_df = influencers_df.copy()
        influencers_df["community_id"] = influencers_df["community_id"].astype(str)
        
        # Basic community stats
        users_per_community = (influencers_df
                              .groupby("community_id")["user_id"]
                              .nunique()
                              .rename("users")
                              .reset_index())
        
        comments_per_community = (influencers_df
                                .groupby("community_id")["comments_count"]
                                .sum()
                                .rename("total_comments")
                                .reset_index())
        
        summary = users_per_community.merge(comments_per_community, on="community_id", how="left")
        
        # Top users per community
        def get_top_users(community_id):
            community_users = (influencers_df[influencers_df["community_id"] == community_id]
                             .sort_values("rank_score", ascending=False)
                             .head(5))
            return ", ".join(community_users["user_id"].astype(str).tolist())
        
        summary["top_users"] = summary["community_id"].apply(get_top_users)
        
        # Example comments per community
        if "cqs" not in df.columns:
            df["cqs"] = 0.0
        if "likes" not in df.columns:
            df["likes"] = 0
        
        # Map users to communities
        user_to_community = dict(zip(
            influencers_df["user_id"].astype(str),
            influencers_df["community_id"].astype(str)
        ))
        
        temp_df = df.copy()
        temp_df["_community_id"] = temp_df["user_id"].astype(str).map(user_to_community)
        
        example_comments = []
        
        for community_id, group in temp_df.dropna(subset=["_community_id"]).groupby("_community_id"):
            # Determine comment column
            comment_col = "comment" if "comment" in group.columns else "text"
            
            if comment_col in group.columns:
                # Score and rank comments
                scored_group = group.copy()
                scored_group["text_length"] = scored_group[comment_col].astype(str).str.len()
                
                top_comments = (scored_group
                              .sort_values(["likes", "cqs", "text_length"], ascending=[False, False, False])
                              .head(3))
                
                # Create sample text
                sample_texts = (top_comments[comment_col]
                              .astype(str)
                              .str.slice(0, 180)
                              .map(lambda x: x + "…"))
                
                sample_text = " | ".join(sample_texts.tolist())
            else:
                sample_text = ""
            
            example_comments.append({
                "community_id": str(community_id),
                "sample_comments": sample_text
            })
        
        examples_df = pd.DataFrame(example_comments)
        
        # Merge and sort
        result = (summary
                 .merge(examples_df, on="community_id", how="left")
                 .fillna({"sample_comments": ""})
                 .sort_values(["users", "total_comments"], ascending=False))
        
        return result
    
    def analyze_influencers_and_communities(self,
                                          df: pd.DataFrame,
                                          user_metadata: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main method to analyze influencers and communities
        
        Args:
            df: DataFrame with user comments and interactions
            user_metadata: Optional DataFrame with follower counts, verification status, etc.
        
        Returns:
            Tuple of (influencers_df, communities_df)
        """
        if "user_id" not in df.columns:
            raise ValueError("DataFrame must have 'user_id' column")
        
        # Build social graph
        edges = self.build_social_graph(df)
        
        if not edges:
            logger.warning("No edges found, returning empty results")
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate user engagement scores
        if "likes" in df.columns:
            user_engagement = df.groupby("user_id")["likes"].sum().to_dict()
        else:
            user_engagement = df.groupby("user_id").size().to_dict()
        
        # Calculate graph metrics
        pagerank_scores, betweenness_scores, community_assignments = self.calculate_graph_metrics(
            edges, user_engagement
        )
        
        # Count mentions
        mentions_made = defaultdict(int)
        mentions_received = defaultdict(int)
        
        for source, target in edges:
            mentions_made[source] += 1
            mentions_received[target] += 1
        
        # Get all users (nodes and edge participants)
        all_users = set(df["user_id"].astype(str).dropna().tolist())
        for source, target in edges:
            all_users.add(source)
            all_users.add(target)
        
        # Calculate comments per user
        comments_per_user = df.groupby("user_id").size().to_dict()
        
        # Build influencers DataFrame
        influencers_data = []
        
        for user_id in sorted(all_users):
            user_data = {
                "user_id": user_id,
                "mentions_made": mentions_made.get(user_id, 0),
                "mentions_received": mentions_received.get(user_id, 0),
                "comments_count": comments_per_user.get(user_id, 0),
                "pagerank": pagerank_scores.get(user_id, 0.0),
                "betweenness": betweenness_scores.get(user_id, 0.0),
                "community_id": community_assignments.get(user_id, str(user_id)),
                "engagement_score": user_engagement.get(user_id, 0)
            }
            influencers_data.append(user_data)
        
        influencers_df = pd.DataFrame(influencers_data)
        
        # Add user metadata if provided
        if isinstance(user_metadata, pd.DataFrame) and "user_id" in user_metadata.columns:
            metadata = user_metadata.copy()
            metadata["user_id"] = metadata["user_id"].astype(str)
            influencers_df = influencers_df.merge(metadata, on="user_id", how="left")
            
            # Add follower tiers
            if "followers" in influencers_df.columns:
                def categorize_follower_tier(follower_count):
                    try:
                        count = int(follower_count)
                    except (ValueError, TypeError):
                        return "n/a"
                    
                    if count >= 1_000_000:
                        return "mega"
                    elif count >= 100_000:
                        return "macro"
                    elif count >= 10_000:
                        return "micro"
                    else:
                        return "nano"
                
                influencers_df["tier"] = influencers_df["followers"].apply(categorize_follower_tier)
        
        # Calculate composite rank score
        influencers_df["rank_score"] = (
            influencers_df["pagerank"].rank(pct=True) * 0.4 +
            influencers_df["betweenness"].rank(pct=True) * 0.2 +
            influencers_df["mentions_received"].rank(pct=True) * 0.2 +
            influencers_df["engagement_score"].rank(pct=True) * 0.2
        ).fillna(0.0)
        
        # Sort by rank score
        influencers_df = influencers_df.sort_values("rank_score", ascending=False)
        
        # Create community summary
        communities_df = self.summarize_communities(df, influencers_df)
        
        # Add community health metrics
        if len(communities_df) > 0:
            community_health_df = self.calculate_community_health(df, influencers_df)
            communities_df = communities_df.merge(community_health_df, on="community_id", how="left").fillna(0)
        
        logger.info(f"Analyzed {len(influencers_df)} users in {len(communities_df)} communities")
        
        return influencers_df, communities_df

def create_influencer_community_detector(config: Optional[Dict[str, Any]] = None) -> InfluencerCommunityDetector:
    """Factory function to create influencer and community detector"""
    return InfluencerCommunityDetector(config)