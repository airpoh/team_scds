# CommentSense: Advanced YouTube Comment Analysis Pipeline

CommentSense is a comprehensive AI-powered system designed for analyzing YouTube comments at scale. It integrates multiple machine learning modules to provide deep insights into sentiment, user behavior, community dynamics, and content quality. The system is particularly optimized for social media comment analysis in multilingual environments.


## Table of Contents

- [Setup](#setup)
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Module Descriptions](#module-descriptions)


### Setup

1. Open VS Code terminal, clone CommentSense repository [git clone https://github.com/airpoh/team_scds.git]
2. Install Python 3.11 【https://www.python.org/downloads/release/python-3110/】
3. Create a virtual environment (Ctrl+Shift+P → Python: Create Environment → Conda) [conda create -n commentsense python=3.11 -y]
4. Activate the conda [conda activate commentsense]
5. Tell VS Code to use this venv (Ctrl+Shift+P → Python: Select Interpreter) [Anaconda3\envs\commentsense\python.exe (3.11)]
6. Install dependencies [python -m pip install --upgrade pip wheel, python -m pip install -r requirements.txt]
7. Create folder 'results' and folder 'dataset' in team_scds
8. Prepare your comment datasets in CSV format (e.g., comment1.csv ~ comment5.csv) inside dataset/
9. Configure the system using the provided configuration files
10. Run dashboard [streamlit run streamlit_dashboard.py]

## Overview

CommentSense processes comment datasets through a modular pipeline that analyzes various dimensions of user-generated content. The system is designed to handle large-scale comment analysis while providing actionable insights for content creators, marketers, and researchers.

The pipeline integrates state-of-the-art NLP models including transformer-based architectures (BERT, RoBERTa, XLM-R) with specialized analysis modules to extract meaningful patterns from social media conversations.

## Key Features

- **Multilingual Support**: Analyzes comments in multiple languages without English-only bias
- **Real-time Crisis Detection**: Identifies sudden sentiment shifts and potential PR issues
- **Advanced Sentiment Analysis**: Fine-grained emotion detection beyond basic positive/negative classification
- **User Persona Clustering**: Groups users based on behavioral patterns and content preferences
- **Network Analysis**: Identifies influencers and community structures within comment threads
- **Visual Content Analysis**: Processes emoji and visual signals for additional context
- **Quality Scoring**: Evaluates comment quality based on relevance, informativeness, and constructiveness
- **Spam Detection**: Identifies bot networks and spam content patterns
- **Predictive Analytics**: Forecasts engagement trends and potential issues

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CommentSense Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│  Input: Comments CSV Files + Videos Metadata (Optional)    │
├─────────────────────────────────────────────────────────────┤
│                   Analysis Modules                         │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Emotion &       │  │ Visual & Emoji  │                 │
│  │ Sarcasm         │  │ Analysis        │                 │
│  │ Detection       │  │                 │                 │
│  └─────────────────┘  └─────────────────┘                 │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Multilingual    │  │ Crisis          │                 │
│  │ Analysis        │  │ Detection       │                 │
│  └─────────────────┘  └─────────────────┘                 │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Network         │  │ Quality         │                 │
│  │ Analysis        │  │ Scoring         │                 │
│  └─────────────────┘  └─────────────────┘                 │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Spam/Bot        │  │ Predictive      │                 │
│  │ Detection       │  │ Analytics       │                 │
│  └─────────────────┘  └─────────────────┘                 │
├─────────────────────────────────────────────────────────────┤
│              Aggregated Insights & KPI System              │
├─────────────────────────────────────────────────────────────┤
│  Output: JSON Results + Streamlit Dashboard               │
└─────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### Core Analysis Modules

#### 1. Emotion and Sarcasm Detection (`modules/emotion_sarcasm_detection.py`)

**Purpose**: Provides fine-grained emotional analysis and sarcasm detection for understanding the true sentiment behind comments.

**Key Technologies**:
- GoEmotions model for multi-emotion classification
- Fine-tuned BERT for sarcasm detection
- Post-processing algorithms to adjust sentiment when sarcasm is detected

**Why It's Essential**: Basic sentiment analysis often misclassifies sarcastic comments. This module identifies 28 different emotions and detects sarcasm to provide more accurate sentiment understanding, crucial for brand reputation management.

**Output**: Emotion scores, sarcasm confidence, adjusted sentiment labels

#### 2. Visual and Emoji Analysis (`modules/visual_emoji_analysis.py`)

**Purpose**: Analyzes emoji usage patterns and visual signals to extract additional sentiment and contextual information.

**Key Technologies**:
- Custom emoji sentiment lexicons
- Vision Transformer models for image analysis (optional)
- Multi-modal sentiment fusion algorithms

**Why It's Essential**: Emojis carry significant emotional weight in social media communication. This module decodes emoji sentiment and analyzes visual content to capture non-textual communication patterns that traditional NLP might miss.

**Output**: Emoji sentiment scores, visual analysis results, emoji distribution metrics

#### 3. Multilingual Analysis (`modules/multilingual_analysis.py`)

**Purpose**: Enables sentiment analysis across multiple languages without requiring translation to English.

**Key Technologies**:
- XLM-RoBERTa for cross-lingual sentiment analysis
- Language detection using langdetect and FastText
- Translation fallback systems using LibreTranslator

**Why It's Essential**: Global brands need to understand sentiment across different languages and cultures. This module prevents English-only bias and provides culturally-aware sentiment analysis for international audiences.

**Output**: Language distribution, multilingual sentiment scores, cultural sentiment patterns

#### 4. Crisis Detection System (`modules/crisis_detection.py`)

**Purpose**: Provides early warning capabilities by detecting sudden negative sentiment spikes and identifying potential PR crises.

**Key Technologies**:
- Change-point detection algorithms (ruptures library)
- Real-time sentiment monitoring with sliding windows
- Risk pattern recognition using statistical analysis

**Why It's Essential**: Early crisis detection can save brands millions in reputation damage. This module monitors comment streams in real-time to identify emerging issues before they escalate into major problems.

**Output**: Crisis alerts, risk levels, temporal sentiment analysis, affected comment identification

#### 5. Network Analysis (`modules/network_analysis.py`)

**Purpose**: Orchestrates community detection, influencer identification, and user behavior analysis to understand social dynamics.

**Key Technologies**:
- PageRank algorithm for influencer scoring
- Louvain method for community detection
- Graph-based analysis of user interactions

**Why It's Essential**: Understanding who influences conversations and how communities form helps optimize engagement strategies and identify key stakeholders in brand discussions.

**Output**: Influencer rankings, community structures, interaction patterns, network metrics

#### 6. Category Classification (`modules/category_classification.py`)

**Purpose**: Classifies comments into domain-specific categories and sub-topics for organized analysis.

**Key Technologies**:
- Zero-shot classification using BART-large-mnli
- BERTopic for unsupervised topic modeling
- Domain-specific category hierarchies

**Why It's Essential**: Organizing comments by topic enables targeted analysis and helps brands understand which aspects of their products or services generate the most discussion.

**Output**: Category assignments, topic distributions, content organization metrics

#### 7. Persona Clustering (`modules/persona_clustering.py`)

**Purpose**: Groups users into distinct personas based on their commenting behavior and content preferences.

**Key Technologies**:
- Sentence Transformers for user profile embeddings
- HDBSCAN clustering for persona identification
- TF-IDF analysis for persona characterization

**Why It's Essential**: Understanding different user personas helps tailor marketing messages and content strategies to specific audience segments, improving engagement and conversion rates.

**Output**: User persona assignments, persona characteristics, behavioral patterns

### Advanced Analysis Modules

#### 8. Comment Quality Scoring (`modules/comment_quality_scoring.py`)

**Purpose**: Evaluates comment quality using a three-pillar QEI (Quality Engagement Index) system.

**Quality Dimensions**:
- **Relevance**: Is the comment about the brand/product/topic?
- **Informativeness**: Does it mention product attributes, experiences, questions?
- **Constructiveness**: Does it provide suggestions/issues vs generic praise?

**Key Technologies**:
- Fine-tuned DistilBERT/RoBERTa models
- Multi-label classification
- Calibrated confidence scoring

**Why It's Essential**: Not all comments are equal. High-quality comments provide actionable insights and drive meaningful engagement, while low-quality comments may indicate spam or superficial interaction.

#### 9. Spam and Bot Detection (`modules/spam_bot_detection.py`)

**Purpose**: Identifies spam content and coordinated bot networks that may artificially inflate or deflate engagement metrics.

**Key Technologies**:
- Behavioral pattern analysis
- Network analysis for bot detection
- Content similarity scoring for spam identification

**Why It's Essential**: Authentic engagement measurement requires filtering out artificial activity. Bot networks and spam can skew analytics and waste marketing resources.

#### 10. Predictive Analytics (`modules/predictive_analytics.py`)

**Purpose**: Forecasts engagement trends and predicts potential issues based on historical patterns.

**Key Technologies**:
- Time series analysis
- Machine learning models for trend prediction
- Risk assessment algorithms

**Why It's Essential**: Proactive management requires understanding future trends. This module helps predict viral content potential and upcoming reputation risks.

#### 11. Composite KPI System (`modules/composite_kpi_system.py`)

**Purpose**: Orchestrates all modules and provides a unified engagement health score (0-100).

**Key Features**:
- Integrates outputs from all analysis modules
- Weighted scoring based on business importance
- Comprehensive performance metrics

**Why It's Essential**: Decision-makers need a single, interpretable metric to gauge overall engagement health while maintaining access to detailed breakdowns when needed.
