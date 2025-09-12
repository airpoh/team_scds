"""
CommentSense Streamlit Dashboard

Interactive dashboard for YouTube comment analysis with real-time insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from pathlib import Path
import logging

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline
try:
    from commentsense_pipeline import CommentSensePipeline, load_datasets
    PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Pipeline import failed: {e}")
    PIPELINE_AVAILABLE = False

# Configure Streamlit
st.set_page_config(
    page_title="CommentSense Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    
    .alert-high {
        background-color: #ff4757;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        color: white;
    }
    
    .alert-medium {
        background-color: #ffa502;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        color: white;
    }
    
    .alert-low {
        background-color: #2ed573;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        color: white;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

class CommentSenseDashboard:
    """Main dashboard class with session state management"""
    
    def __init__(self):
        self._initialize_session_state()
        self._initialize_pipeline()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = None
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'comments_df' not in st.session_state:
            st.session_state.comments_df = None
        if 'videos_df' not in st.session_state:
            st.session_state.videos_df = None
        if 'analysis_running' not in st.session_state:
            st.session_state.analysis_running = False
        if 'last_analysis_time' not in st.session_state:
            st.session_state.last_analysis_time = None
        if 'cache_cleared' not in st.session_state:
            st.session_state.cache_cleared = False
        if 'current_results_file' not in st.session_state:
            st.session_state.current_results_file = None
    
    def _initialize_pipeline(self):
        """Initialize the pipeline if not already done"""
        if st.session_state.pipeline is None and PIPELINE_AVAILABLE:
            try:
                st.session_state.pipeline = CommentSensePipeline()
                st.success("CommentSense Pipeline loaded successfully!")
            except Exception as e:
                st.error(f"Failed to initialize pipeline: {e}")
        elif not PIPELINE_AVAILABLE:
            st.error("Pipeline not available. Please ensure all dependencies are installed.")
    
    def _safe_get_results(self, key: str, default=None):
        """Safely get data from results with proper error handling"""
        if not isinstance(st.session_state.results, dict):
            logger.error(f"_safe_get_results: results is not dict, type: {type(st.session_state.results)}")
            return default
        
        result = st.session_state.results.get(key, default)
        logger.info(f"_safe_get_results: key='{key}', result_type={type(result)}")
        return result
    
    def clear_cache(self):
        """Clear all cached data and reset session state"""
        logger.info("Clearing cache and resetting session state")
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Reinitialize
        self._initialize_session_state()
        
        # Clear Streamlit cache
        st.cache_data.clear()
        st.cache_resource.clear()
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header"> CommentSense Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("**Advanced YouTube Comment Analysis with AI-Powered Insights**")
        
        # Display all 8 comprehensive features in two rows
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <h4>Emotion Analysis</h4>
                <p>Advanced emotion recognition with sarcasm detection</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container">
                <h4>Multilingual</h4>
                <p>Support for 40+ languages with auto-translation</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container">
                <h4>Emoji Analysis</h4>
                <p>Visual sentiment analysis from emojis and images</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-container">
                <h4>Crisis Detection</h4>
                <p>Real-time alerts for negative sentiment spikes</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Second row of features
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.markdown("""
            <div class="metric-container">
                <h4>Network Analysis</h4>
                <p>Influencer detection, communities & user personas</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col6:
            st.markdown("""
            <div class="metric-container">
                <h4>Quality Analysis</h4>
                <p>Comment relevance, informativeness & constructiveness scoring</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col7:
            st.markdown("""
            <div class="metric-container">
                <h4>Spam & Bot Detection</h4>
                <p>Advanced spam filtering & bot account identification</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col8:
            st.markdown("""
            <div class="metric-container">
                <h4>Predictive Analytics</h4>
                <p>Engagement forecasting & risk assessment</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.markdown("## Analysis Controls")
        
        # Data source selection
        st.sidebar.markdown("### Data Source")
        
        # Show dataset info
        if st.session_state.comments_df is not None:
            total_comments = len(st.session_state.comments_df)
            st.sidebar.info(f"Dataset loaded: {total_comments:,} comments")
        
        data_source = st.sidebar.radio(
            "Choose data source:",
            ["Upload Files", "Load Previous Results"]
        )
        
        if data_source == "Upload Files":
            uploaded_files = st.sidebar.file_uploader(
                "Upload CSV files",
                type=['csv'],
                accept_multiple_files=True,
                help="Upload YouTube comment CSV files"
            )
            
            if uploaded_files:
                self.load_uploaded_data(uploaded_files)
        
        elif data_source == "Load Previous Results":
            self.load_previous_results()
        
        # Analysis options
        st.sidebar.markdown("### Analysis Options")
        
        # Analysis size options
        analysis_mode = st.sidebar.radio(
            "Analysis scope:",
            ["Sample Analysis", "Full Dataset Analysis", "Individual CSV Files"]
        )
        
        if analysis_mode == "Sample Analysis":
            self.max_comments = st.sidebar.slider(
                "Max comments to analyze",
                min_value=10,
                max_value=10000,
                value=100,
                step=10
            )
            
            # Show processing time estimate
            if self.max_comments > 1000:
                estimated_time = self.max_comments * 0.05  # ~0.05 seconds per comment
                st.sidebar.warning(f"Processing {self.max_comments:,} comments may take ~{estimated_time:.0f} seconds")
            elif self.max_comments > 500:
                estimated_time = self.max_comments * 0.03
                st.sidebar.info(f"ℹProcessing {self.max_comments:,} comments may take ~{estimated_time:.0f} seconds")
        elif analysis_mode == "Full Dataset Analysis":
            # Full dataset analysis
            if st.session_state.comments_df is not None:
                total_comments = len(st.session_state.comments_df)
                self.max_comments = total_comments
                estimated_hours = (total_comments * 0.05) / 3600
                st.sidebar.error(f"FULL DATASET ANALYSIS")
                st.sidebar.error(f"{total_comments:,} comments")
                st.sidebar.error(f"Estimated time: ~{estimated_hours:.1f} hours")
                st.sidebar.error(f"Memory usage: ~{total_comments * 0.001:.0f} MB")
                
                if st.sidebar.checkbox("I understand the risks and want to proceed"):
                    st.sidebar.success("Full analysis enabled")
                else:
                    st.sidebar.warning("Please confirm to enable full analysis")
                    self.max_comments = 0
            else:
                st.sidebar.warning("Please load dataset first")
                self.max_comments = 0
        
        elif analysis_mode == "Individual CSV Files":
            # Individual CSV file analysis
            st.sidebar.markdown("### Select CSV File to Analyze")
            
            csv_files = [
                "dataset/comments1.csv",
                "dataset/comments2.csv", 
                "dataset/comments3.csv",
                "dataset/comments4.csv",
                "dataset/comments5.csv"
            ]
            
            # Check which files exist
            existing_files = [f for f in csv_files if Path(f).exists()]
            
            if existing_files:
                selected_file = st.sidebar.selectbox(
                    "Choose CSV file:",
                    existing_files
                )
                
                # Show file info
                try:
                    df = pd.read_csv(selected_file)
                    file_comments = len(df)
                    st.sidebar.info(f"{Path(selected_file).name}: {file_comments:,} comments")
                    
                    # Analysis options for this file
                    self.max_comments = st.sidebar.slider(
                        f"Max comments from {Path(selected_file).name}",
                        min_value=10,
                        max_value=min(file_comments, 50000),
                        value=min(1000, file_comments),
                        step=100
                    )
                    
                    # Store selected file for analysis
                    st.session_state.selected_csv_file = selected_file
                    
                    # Show processing time estimate
                    if self.max_comments > 1000:
                        estimated_time = self.max_comments * 0.05
                        st.sidebar.warning(f"Processing {self.max_comments:,} comments may take ~{estimated_time:.0f} seconds")
                    
                    # Auto-load data if not already loaded or different file selected
                    if (st.session_state.comments_df is None or 
                        not hasattr(st.session_state, 'last_loaded_file') or 
                        st.session_state.last_loaded_file != selected_file):
                        self.load_csv_file(selected_file)
                        st.session_state.last_loaded_file = selected_file
                    
                except Exception as e:
                    st.sidebar.error(f"Error reading {selected_file}: {e}")
                    self.max_comments = 0
            else:
                st.sidebar.warning("No CSV files found in dataset/ directory")
                self.max_comments = 0
        
        # All modules enabled by default - checkboxes removed from UI
        self.enable_modules = {
            "emotion_sarcasm": True,
            "visual_emoji": True,
            "multilingual": True,
            "crisis_detection": True,
            "network_analysis": True
        }
        
        # Analysis button
        if st.sidebar.button("Run Analysis", type="primary"):
            if st.session_state.comments_df is not None:
                self.run_analysis()
            else:
                st.sidebar.error("Please load data first!")
        
    def load_uploaded_data(self, uploaded_files):
        """Load data from uploaded files"""
        try:
            dfs = []
            for file in uploaded_files:
                df = pd.read_csv(file)
                dfs.append(df)
                
            if dfs:
                st.session_state.comments_df = pd.concat(dfs, ignore_index=True)
                st.sidebar.success(f"Loaded {len(st.session_state.comments_df)} comments")
                
        except Exception as e:
            st.sidebar.error(f"Error loading files: {e}")
    
    def load_csv_file(self, file_path: str):
        """Load a specific CSV file"""
        try:
            st.session_state.comments_df, st.session_state.videos_df = load_datasets([file_path])
            st.sidebar.success(f"Loaded {len(st.session_state.comments_df)} comments from {Path(file_path).name}")
        except Exception as e:
            st.sidebar.error(f"Error loading {file_path}: {e}")
    
    def load_previous_results(self):
        """Load previously saved analysis results"""
        results_dir = Path("results")
        
        if results_dir.exists():
            result_files = list(results_dir.glob("*.json"))
            
            if result_files:
                selected_file = st.sidebar.selectbox(
                    "Select results file:",
                    [f.name for f in result_files]
                )
                
                # Load results button to prevent auto-reload issues
                if st.sidebar.button(f"Load {selected_file}", key=f"load_{selected_file}"):
                    try:
                        with open(results_dir / selected_file, 'r') as f:
                            loaded_results = json.load(f)
                            # Ensure results is a dictionary, not a list
                            if isinstance(loaded_results, dict):
                                st.session_state.results = loaded_results
                                st.session_state.current_results_file = selected_file
                                st.sidebar.success(f"Results loaded: {selected_file}")
                                # Don't use st.rerun() to prevent infinite loops
                            else:
                                st.sidebar.error(f"Invalid results format: expected dict, got {type(loaded_results)}")
                    except Exception as e:
                        st.sidebar.error(f"Error loading results: {e}")
                
                # Show currently loaded results
                if hasattr(st.session_state, 'current_results_file') and st.session_state.current_results_file:
                    if st.session_state.current_results_file == selected_file:
                        st.sidebar.info(f"Currently displaying: {selected_file}")
                    else:
                        st.sidebar.warning(f"Displaying: {st.session_state.current_results_file}, but {selected_file} is selected")
            else:
                st.sidebar.info("No saved results found")
        else:
            st.sidebar.info("Results directory not found")
    
    def run_analysis(self):
        """Run the analysis pipeline"""
        if not st.session_state.pipeline:
            st.error("Pipeline not available!")
            return
        
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Prepare data
            status_text.text("Preparing data...")
            progress_bar.progress(20)
            
            analysis_df = st.session_state.comments_df.head(self.max_comments).copy()
            
            # Configure modules based on user selection
            for module, enabled in self.enable_modules.items():
                if module in st.session_state.pipeline.config['modules']:
                    st.session_state.pipeline.config['modules'][module]['enabled'] = enabled
            
            # Run analysis
            status_text.text("Running analysis...")
            progress_bar.progress(60)
            
            results = st.session_state.pipeline.analyze_comments(analysis_df, st.session_state.videos_df)
            
            # Debug: Check results type
            logger.info(f"Analysis results type: {type(results)}")
            if isinstance(results, dict):
                logger.info(f"Results keys: {list(results.keys())}")
            else:
                logger.error(f"Unexpected results type: {type(results)}, value: {results}")
            
            # Clear any cached results before setting new ones
            if 'results' in st.session_state:
                del st.session_state.results
            
            st.session_state.results = results
            
            # Complete
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            # Clear progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success("Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            logger.error(f"Analysis error: {e}")
    
    def render_main_content(self):
        """Render main dashboard content"""
        if st.session_state.results is None:
            st.info(" **No results loaded yet**")
            st.markdown("""
            **To view analysis results:**
            1. Select "Load Previous Results" in the sidebar
            2. Choose a results file from the dropdown
            3. Click the "Load [filename]" button
            
            **To run new analysis:**
            1. Select data source in the sidebar
            2. Configure analysis options
            3. Click "Run Analysis" (Note: May cause memory issues with large datasets)
            """)
            return
        
        # Debug: Log results type and content
        logger.info(f"render_main_content - results type: {type(st.session_state.results)}")
        if isinstance(st.session_state.results, dict):
            logger.info(f"render_main_content - results keys: {list(st.session_state.results.keys())}")
        else:
            logger.error(f"render_main_content - results is not dict: {st.session_state.results}")
        
        # Validate results format
        if not isinstance(st.session_state.results, dict):
            st.error(f"Invalid results format: expected dict, got {type(st.session_state.results)}")
            st.info("Please reload the results or run a new analysis.")
            return
        
        # Show which results file is being displayed
        if hasattr(st.session_state, 'current_results_file') and st.session_state.current_results_file:
            st.success(f"**Displaying results from:** {st.session_state.current_results_file}")
        else:
            st.info("📊 **Displaying results from:** Live analysis")
        
        # Results overview
        self.render_overview()
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "Sentiment Analysis",
            "Emotion Detection", 
            "Multilingual Insights",
            "Crisis Alerts",
            "Network Analysis",
            "Influencers & Communities",
            "Composite KPI",
            "Quality Analysis", 
            "Spam & Bot Detection",
            "Detailed Results"
        ])
        
        with tab1:
            self.render_sentiment_analysis()
            
        with tab2:
            self.render_emotion_analysis()
            
        with tab3:
            self.render_multilingual_analysis()
            
        with tab4:
            self.render_crisis_alerts()
            
        with tab5:
            self.render_network_analysis()
            
        with tab6:
            self.render_influencers_communities()
            
        with tab7:
            self.render_composite_kpi()
            
        with tab8:
            self.render_quality_analysis()
            
        with tab9:
            self.render_spam_bot_detection()
            
        with tab10:
            self.render_detailed_results()
    
    def render_overview(self):
        """Render results overview"""
        st.markdown("## Analysis Overview")
        
        logger.info("render_overview: Starting to get insights and pipeline_info")
        insights = self._safe_get_results('aggregated_insights', {})
        pipeline_info = self._safe_get_results('pipeline_info', {})
        
        logger.info(f"render_overview: insights type: {type(insights)}")
        logger.info(f"render_overview: pipeline_info type: {type(pipeline_info)}")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            logger.info(f"render_overview: insights.get('overall_sentiment') type: {type(insights.get('overall_sentiment', {}))}")
            overall_sentiment = insights.get('overall_sentiment', {})
            logger.info(f"render_overview: overall_sentiment type: {type(overall_sentiment)}")
            
            if isinstance(overall_sentiment, dict):
                sentiment = overall_sentiment.get('mean', 0)
                st.metric(
                    "Overall Sentiment",
                    f"{sentiment:.3f}",
                    delta=f"±{overall_sentiment.get('std', 0):.3f}"
                )
            else:
                logger.error(f"render_overview: overall_sentiment is not dict: {overall_sentiment}")
                st.metric("Overall Sentiment", "N/A")
            
        with col2:
            total_comments = pipeline_info.get('total_comments', 0)
            st.metric("Comments Analyzed", f"{total_comments:,}")
            
        with col3:
            processing_time = pipeline_info.get('processing_time', 0)
            st.metric("Processing Time", f"{processing_time:.2f}s")
            
        with col4:
            crisis_alerts = insights.get('crisis_summary', {}).get('total_alerts', 0)
            st.metric("Crisis Alerts", crisis_alerts)
            
        with col5:
            # Network analysis metrics
            network_summary = insights.get('network_analysis', {})
            if isinstance(network_summary, dict) and 'key_stats' in network_summary:
                total_communities = network_summary['key_stats'].get('total_communities', 0)
                st.metric("Communities", total_communities)
            else:
                st.metric("Network Analysis", "N/A")
        
        # Risk level indicator
        risk_level = insights.get('crisis_summary', {}).get('risk_level', 'low')
        risk_colors = {'low': 'alert-low', 'medium': 'alert-medium', 'high': 'alert-high'}
        risk_color = risk_colors.get(risk_level, 'alert-low')
        
        st.markdown(f"""
        <div class="{risk_color}">
            <h4>Risk Level: {risk_level.upper()}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        recommendations = insights.get('recommendations', [])
        if recommendations:
            st.markdown("### Recommendations")
            for rec in recommendations:
                st.write(f"• {rec}")
    
    def render_sentiment_analysis(self):
        """Render sentiment analysis visualizations"""
        st.markdown("## Sentiment Analysis")
        
        logger.info("render_sentiment_analysis: Starting")
        insights = self._safe_get_results('aggregated_insights', {})
        logger.info(f"render_sentiment_analysis: insights type: {type(insights)}")
        overall_sentiment = insights.get('overall_sentiment', {})
        logger.info(f"render_sentiment_analysis: overall_sentiment type: {type(overall_sentiment)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            if overall_sentiment:
                fig = go.Figure(data=go.Scatter(
                    x=['Mean', 'Min', 'Max'],
                    y=[overall_sentiment.get('mean', 0), 
                       overall_sentiment.get('min', 0), 
                       overall_sentiment.get('max', 0)],
                    mode='markers+lines',
                    marker=dict(size=12, color=['blue', 'red', 'green'])
                ))
                fig.update_layout(title="Sentiment Range", yaxis_title="Sentiment Score")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Emoji sentiment
            emoji_sentiment = insights.get('emoji_sentiment', {})
            if emoji_sentiment:
                labels = ['Positive Emojis', 'Negative Emojis']
                values = [emoji_sentiment.get('positive_emojis_count', 0),
                         emoji_sentiment.get('negative_emojis_count', 0)]
                
                fig = px.pie(values=values, names=labels, title="Emoji Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
    
    def render_emotion_analysis(self):
        """Render emotion analysis visualizations"""
        st.markdown("## Emotion Detection")
        
        logger.info("render_emotion_analysis: Starting")
        insights = self._safe_get_results('aggregated_insights', {})
        logger.info(f"render_emotion_analysis: insights type: {type(insights)}")
        emotions = insights.get('dominant_emotions', {})
        logger.info(f"render_emotion_analysis: emotions type: {type(emotions)}")
        
        if emotions:
            # Emotion radar chart
            fig = go.Figure(data=go.Scatterpolar(
                r=list(emotions.values()),
                theta=list(emotions.keys()),
                fill='toself',
                name='Emotions'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(emotions.values()) * 1.1] if emotions.values() else [0, 1]
                    )),
                title="Dominant Emotions"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Emotion bar chart
            fig_bar = px.bar(
                x=list(emotions.keys()),
                y=list(emotions.values()),
                title="Emotion Intensity",
                labels={'x': 'Emotion', 'y': 'Intensity'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No emotion data available")
    
    def render_multilingual_analysis(self):
        """Render multilingual analysis"""
        st.markdown("## Multilingual Analysis")
        
        logger.info("render_multilingual_analysis: Starting")
        insights = self._safe_get_results('aggregated_insights', {})
        logger.info(f"render_multilingual_analysis: insights type: {type(insights)}")
        lang_dist = insights.get('language_distribution', {})
        logger.info(f"render_multilingual_analysis: lang_dist type: {type(lang_dist)}")
        
        if lang_dist:
            # Language distribution pie chart
            fig = px.pie(
                values=list(lang_dist.values()),
                names=list(lang_dist.keys()),
                title="Language Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Language bar chart
            fig_bar = px.bar(
                x=list(lang_dist.keys()),
                y=list(lang_dist.values()),
                title="Comments by Language",
                labels={'x': 'Language', 'y': 'Number of Comments'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No multilingual data available")
    
    def render_crisis_alerts(self):
        """Render crisis detection alerts"""
        st.markdown("## Crisis Detection")
        
        logger.info("render_crisis_alerts: Starting")
        module_results = self._safe_get_results('module_results', {})
        logger.info(f"render_crisis_alerts: module_results type: {type(module_results)}")
        crisis_results = module_results.get('crisis_detection', {})
        logger.info(f"render_crisis_alerts: crisis_results type: {type(crisis_results)}")
        
        if isinstance(crisis_results, dict) and 'alerts' in crisis_results:
            alerts = crisis_results['alerts']
            
            if alerts:
                st.warning(f"{len(alerts)} crisis alerts detected!")
                
                for i, alert in enumerate(alerts):
                    severity = alert.get('severity', 0)
                    alert_type = alert.get('alert_type', 'unknown')
                    description = alert.get('description', 'No description')
                    
                    severity_color = 'red' if severity > 0.7 else 'orange' if severity > 0.4 else 'yellow'
                    
                    st.markdown(f"""
                    <div style="border-left: 4px solid {severity_color}; padding: 10px; margin: 10px 0; background: #f8f9fa;">
                        <strong>Alert {i+1}: {alert_type}</strong><br>
                        Severity: {severity:.3f}<br>
                        {description}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No crisis alerts detected")
        elif isinstance(crisis_results, list):
            st.info("Crisis detection results in unexpected format - please re-run analysis")
        else:
            st.info("Crisis detection data not available")
    
    def render_network_analysis(self):
        """Render network analysis overview (emoji-free)"""
        st.markdown("## Network Analysis Overview")
    
        logger.info("render_network_analysis: Starting")
        insights = self._safe_get_results('aggregated_insights', {})
        network_summary = insights.get('network_analysis', {})
    
        if isinstance(network_summary, dict) and network_summary.get('status') == 'success':
            key_stats = network_summary.get('key_stats', {})
        
            # Network metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Influencers", key_stats.get('total_influencers', 0))
            with col2:
                st.metric("Communities", key_stats.get('total_communities', 0))
            with col3:
                st.metric("User Personas", key_stats.get('total_personas', 0))
            with col4:
                st.metric("Category Diversity", key_stats.get('category_diversity', 0))
        
            # Combined insights
            combined_insights = insights.get('combined_insights', {})
            if combined_insights:
                st.markdown("### Cross-Analysis Insights")
            
                # Risk assessment
                risk_assessment = combined_insights.get('risk_assessment', {})
                if risk_assessment:
                    risk_level = (risk_assessment.get('overall_risk_level', 'low') or 'low').lower()
                    
                    st.markdown(f"**Risk Level:** {risk_level.upper()}")
                
                    risk_factors = risk_assessment.get('risk_factors', [])
                    if risk_factors:
                        st.markdown("**Risk Factors:**")
                        for factor in risk_factors:
                            st.write(f"- {factor}")
            
                # Engagement insights
                engagement_insights = combined_insights.get('engagement_insights', {})
                if engagement_insights:
                    st.markdown("### Engagement Patterns")
                
                    engagement_indicators = []
                    if engagement_insights.get('community_engagement'):
                        engagement_indicators.append("Active community engagement detected")
                    if engagement_insights.get('influencer_presence'):
                        engagement_indicators.append("Key influencers identified")
                    if engagement_insights.get('persona_diversity'):
                        engagement_indicators.append("Diverse user personas present")
                    if engagement_insights.get('content_categorization'):
                        engagement_indicators.append("Multi-category content discussion")
                
                    for indicator in engagement_indicators:
                        st.write(f"- {indicator}")
        else:
            st.info("Network analysis data not available or failed to process")
            if isinstance(network_summary, dict) and 'error' in network_summary:
                st.error(f"Error: {network_summary['error']}")

    
    def render_influencers_communities(self):
        """Render detailed influencers and communities analysis"""
        st.markdown("## Influencers & Communities")
        
        logger.info("render_influencers_communities: Starting")
        module_results = self._safe_get_results('module_results', {})
        network_results = module_results.get('network_analysis', {})
        
        if isinstance(network_results, dict):
            if 'error' in network_results:
                st.error(f"Network analysis error: {network_results['error']}")
                return
                
            # Check if we have DataFrame results
            if 'influencers_df' in network_results or 'communities_df' in network_results:
                self._render_dataframe_network_results(network_results)
            elif network_results.get('status') == 'success':
                self._render_structured_network_results(network_results)
            else:
                st.info("Network analysis data not available or incomplete")
        else:
            st.info("No network analysis results found")
    
    def _render_dataframe_network_results(self, network_results):
        """Render network results when stored as DataFrame strings"""
        
        # Parse string representations to extract basic info
        influencers_info = str(network_results.get('influencers_df', ''))
        communities_info = str(network_results.get('communities_df', ''))
        personas_info = str(network_results.get('personas_df', ''))
        categories_info = str(network_results.get('cats_out', ''))
        
        # Extract counts
        import re
        
        # Influencers
        influencer_count = 0
        if 'rows x' in influencers_info:
            match = re.search(r'\[(\d+) rows x \d+ columns\]', influencers_info)
            if match:
                influencer_count = int(match.group(1))
        
        # Communities 
        community_count = 0
        if 'rows x' in communities_info:
            match = re.search(r'\[(\d+) rows x \d+ columns\]', communities_info)
            if match:
                community_count = int(match.group(1))
        
        # Personas
        persona_count = 0
        if 'rows x' in personas_info:
            match = re.search(r'\[(\d+) rows x \d+ columns\]', personas_info)
            if match:
                persona_count = int(match.group(1))
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(" Total Influencers", influencer_count)
        with col2:
            st.metric(" Communities Detected", community_count)
        with col3:
            st.metric(" User Personas", persona_count)
        
        # Show data previews if available
        if influencer_count > 0:
            st.markdown("### Influencers Overview")
            # Extract sample data from string representation
            if '\n' in influencers_info:
                lines = influencers_info.split('\n')
                sample_lines = [line for line in lines[:10] if line.strip() and 'user_id' in line or any(c.isdigit() for c in line)]
                if sample_lines:
                    st.text("Sample Influencer Data:")
                    for line in sample_lines[:5]:
                        if line.strip():
                            st.text(line[:100] + "..." if len(line) > 100 else line)
                else:
                    st.info(f"{influencer_count} influencers detected in the dataset")
        
        if community_count > 0:
            st.markdown("### Communities Overview")
            if '\n' in communities_info:
                lines = communities_info.split('\n')
                sample_lines = [line for line in lines[:10] if line.strip() and ('community_id' in line or any(c.isdigit() for c in line))]
                if sample_lines:
                    st.text("Sample Community Data:")
                    for line in sample_lines[:5]:
                        if line.strip():
                            st.text(line[:100] + "..." if len(line) > 100 else line)
                else:
                    st.info(f"{community_count} communities detected in the dataset")
        
        if persona_count > 0:
            st.markdown("### User Personas Overview")
            if '\n' in personas_info:
                lines = personas_info.split('\n')
                sample_lines = [line for line in lines[:10] if line.strip() and ('persona_label' in line or 'General' in line)]
                if sample_lines:
                    st.text("Sample Persona Data:")
                    for line in sample_lines[:5]:
                        if line.strip():
                            st.text(line[:100] + "..." if len(line) > 100 else line)
                else:
                    st.info(f"{persona_count} user personas identified")
        
        # Categories
        if categories_info and 'rows x' in categories_info:
            st.markdown("### Content Categories")
            if '\n' in categories_info:
                lines = categories_info.split('\n')
                category_lines = [line for line in lines if 'general' in line.lower() or 'category' in line.lower()]
                if category_lines:
                    st.text("Sample Category Data:")
                    for line in category_lines[:3]:
                        if line.strip():
                            st.text(line[:100] + "..." if len(line) > 100 else line)
                else:
                    st.info("Content categorization completed")
        
        # Summary insights
        st.markdown("### Network Analysis Summary")
        insights = []
        if influencer_count > 0:
            insights.append(f" Identified {influencer_count} influencers in the comment network")
        if community_count > 0:
            insights.append(f" Detected {community_count} distinct communities")
        if persona_count > 0:
            insights.append(f" Classified users into {persona_count} different personas")
        
        if insights:
            for insight in insights:
                st.success(insight)
        else:
            st.info("Network analysis completed but no specific patterns detected")
    
    def _render_structured_network_results(self, network_results):
        """Render structured network results (legacy format)"""
        results = network_results.get('results', {})
        
        # Influencers section
        if 'influencers' in results:
            st.markdown("### Top Influencers")
            influencer_data = results['influencers']
            top_influencers = influencer_data.get('top_influencers', [])
            
            if top_influencers:
                # Create DataFrame for display
                df_influencers = pd.DataFrame(top_influencers)
                if not df_influencers.empty:
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Influencers", influencer_data.get('total_influencers', 0))
                    with col2:
                        st.metric("High Influence Users", influencer_data.get('high_influence_users', 0))
                    with col3:
                        avg_pagerank = influencer_data.get('avg_pagerank', 0)
                        st.metric("Avg PageRank", f"{avg_pagerank:.4f}")
                    
                    # Influencer table
                    st.dataframe(df_influencers, use_container_width=True)
                    
                    # Visualization
                    if len(df_influencers) > 1:
                        fig = px.scatter(
                            df_influencers,
                            x='pagerank',
                            y='betweenness',
                            size='rank_score',
                            hover_data=['user_id'],
                            title="Influencer Network Position",
                            labels={
                                'pagerank': 'PageRank (Influence)',
                                'betweenness': 'Betweenness (Bridging)',
                                'rank_score': 'Overall Rank'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No influencer data available")
        
        # Communities section  
        if 'communities' in results:
            st.markdown("### Communities")
            community_data = results['communities']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Communities", community_data.get('total_communities', 0))
            with col2:
                st.metric("Largest Community", community_data.get('largest_community', 0))
            with col3:
                avg_size = community_data.get('avg_community_size', 0)
                st.metric("Avg Community Size", f"{avg_size:.1f}")
            
            # Community health
            community_health = community_data.get('community_health', {})
            if community_health:
                st.markdown("### Community Health Metrics")
                health_col1, health_col2, health_col3 = st.columns(3)
                
                with health_col1:
                    sentiment_health = community_health.get('avg_sentiment_health', 0)
                    st.metric("Sentiment Health", f"{sentiment_health:.2f}")
                with health_col2:
                    spam_health = community_health.get('avg_spam_health', 0)
                    st.metric("Spam Health", f"{spam_health:.2f}")
                with health_col3:
                    avg_quality = community_health.get('avg_quality', 0)
                    st.metric("Avg Quality", f"{avg_quality:.2f}")
        
        # Categories and Personas sections (similar pattern)
        if 'categories' in results:
            st.markdown("### Content Categories")
            category_data = results['categories']
            st.metric("Category Diversity", category_data.get('total_categories', 0))
        
        if 'personas' in results:
            st.markdown("### User Personas")
            persona_data = results['personas']
            st.metric("Total Personas", persona_data.get('total_personas', 0))
    
    def render_detailed_results(self):
        """Render detailed raw results"""
        st.markdown("## Detailed Results")
        
        logger.info("render_detailed_results: Starting")
        
        if st.checkbox("Show raw results JSON"):
            st.json(st.session_state.results)
        
        # Main section tabs
        tab1, tab2, tab3 = st.tabs(["Basic Module Results", "Advanced Module Results", "Complete Analysis"])
        
        with tab1:
            st.markdown("### Basic Module Results")
            module_results = self._safe_get_results('module_results', {})
            logger.info(f"render_detailed_results: module_results type: {type(module_results)}")
            
            if module_results:
                for module_name, module_data in module_results.items():
                    logger.info(f"render_detailed_results: Processing {module_name}, type: {type(module_data)}")
                    with st.expander(f"{module_name.replace('_', ' ').title()} Results"):
                        if isinstance(module_data, dict) and 'error' not in module_data:
                            st.json(module_data)
                        elif isinstance(module_data, dict) and 'error' in module_data:
                            st.error(f"Error in {module_name}: {module_data.get('error', 'Unknown error')}")
                        elif isinstance(module_data, list):
                            st.info(f"{module_name} contains {len(module_data)} results")
                            if len(module_data) > 0:
                                st.json(module_data[:5])  # Show first 5 items
                                if len(module_data) > 5:
                                    st.info(f"... and {len(module_data) - 5} more items")
                        else:
                            st.error(f"Unexpected data type in {module_name}: {type(module_data)}")
            else:
                st.info("No basic module results found")
        
        with tab2:
            st.markdown("### Advanced Module Results")
            composite_kpi = self._safe_get_results('composite_kpi', {})
            if composite_kpi and composite_kpi.get('status') == 'success':
                all_module_results = composite_kpi.get('all_module_results', {})
                
                if all_module_results:
                    for module_name, module_data in all_module_results.items():
                        with st.expander(f" {module_name.replace('_', ' ').title()} Analysis"):
                            if isinstance(module_data, dict):
                                if 'status' in module_data:
                                    status = module_data.get('status', 'unknown')
                                    if status == 'success':
                                        st.success(f" Status: {status}")
                                    else:
                                        st.error(f" Status: {status}")
                                
                                # Show key metrics first
                                if 'detection_metrics' in module_data or 'aggregated_metrics' in module_data:
                                    st.markdown("**Key Metrics:**")
                                    
                                if 'aggregated_metrics' in module_data:
                                    metrics = module_data['aggregated_metrics']
                                    st.json(metrics)
                                elif 'detection_metrics' in module_data:
                                    metrics = module_data['detection_metrics']
                                    st.json(metrics)
                                else:
                                    st.json(module_data)
                            else:
                                st.json(module_data)
                else:
                    st.info("No advanced module results found")
            else:
                st.info("Advanced analysis results not available")
        
        with tab3:
            st.markdown("### Complete Analysis Overview")
            
            # Pipeline info
            pipeline_info = self._safe_get_results('pipeline_info', {})
            if pipeline_info:
                st.markdown("#### Pipeline Information")
                st.json(pipeline_info)
            
            # Aggregated insights
            insights = self._safe_get_results('aggregated_insights', {})
            if insights:
                st.markdown("#### Aggregated Insights")
                st.json(insights)
            
            # Composite KPI summary
            if composite_kpi:
                st.markdown("#### Composite KPI Summary")
                kpi_summary = {
                    'status': composite_kpi.get('status'),
                    'timestamp': composite_kpi.get('timestamp'),
                    'comprehensive_kpi': composite_kpi.get('comprehensive_kpi', {}),
                    'dashboard_metrics': composite_kpi.get('dashboard_metrics', {})
                }
                st.json(kpi_summary)
        
        # Download results
        st.markdown("---")
        if st.button(" Download Complete Results"):
            results_json = json.dumps(st.session_state.results, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=results_json,
                file_name=f"commentsense_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def render_composite_kpi(self):
        """Render composite KPI analysis (emoji-free)"""
        st.markdown("## Composite KPI System")
    
        composite_kpi = self._safe_get_results('composite_kpi', {})
    
        if composite_kpi and composite_kpi.get('status') == 'success':
            kpi_data = composite_kpi.get('comprehensive_kpi', {})
        
            # Main KPI Score
            overall_score = float(kpi_data.get('overall_score', 0) or 0)
            grade = kpi_data.get('grade', 'N/A')
            confidence = float(kpi_data.get('confidence_score', 0) or 0)
        
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall KPI Score", f"{overall_score:.1f}/100", help="Comprehensive engagement health score")
            with col2:
                # No emoji/color badges, just the grade text
                st.metric("Grade", f"{grade}")
            with col3:
                st.metric("Confidence", f"{confidence:.2f}")
        
            # Component Scores
            component_scores = kpi_data.get('component_scores', {})
            if component_scores:
                st.markdown("### Component Performance")
            
                # Two columns for layout
                col1, col2 = st.columns(2)
            
                components = [
                    ("Emotion & Sentiment", "emotion_sentiment"),
                    ("Emoji & Visual", "emoji_visual"),
                    ("Multilingual Analysis", "multilingual"),
                    ("Crisis Safety", "crisis_safety"),
                    ("Network Influence", "network_influence"),
                    ("Quality Scoring", "quality_scoring"),
                    ("Spam Authenticity", "spam_authenticity"),
                    ("Predictive Insights", "predictive_insights"),
                ]
            
                for i, (name, key) in enumerate(components):
                    score = float(component_scores.get(key, 0) or 0)
                    col = col1 if i % 2 == 0 else col2
                    with col:
                        st.markdown(f"**{name}**")  # removed "emoji + name"
                        st.progress(min(max(score, 0), 100) / 100.0)
                        st.write(f"Score: {score:.1f}/100")
                        st.write("")
        
            # Dashboard metrics
            dashboard_metrics = composite_kpi.get('dashboard_metrics', {})
            if dashboard_metrics:
                headline_kpi = dashboard_metrics.get('headline_kpi', {})
                st.markdown("### Key Performance Indicators")
            
                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                with kpi_col1:
                    st.metric("Overall Score", f"{headline_kpi.get('overall_score', 0):.1f}")
                with kpi_col2:
                    st.metric("Grade", headline_kpi.get('grade', 'N/A'))
                with kpi_col3:
                    st.metric("Confidence", f"{headline_kpi.get('confidence', 0):.2f}")
                with kpi_col4:
                    st.metric("Comments Analyzed", f"{headline_kpi.get('total_comments', 0):,}")
        else:
            st.info("Composite KPI data not available")
            if composite_kpi.get('error'):
                st.error(f"Error: {composite_kpi['error']}")
    
    def render_quality_analysis(self):
        """Render quality analysis results"""
        st.markdown("## Comment Quality Analysis")
        
        composite_kpi = self._safe_get_results('composite_kpi', {})
        
        if composite_kpi and composite_kpi.get('status') == 'success':
            # Check all_module_results for quality data
            all_module_results = composite_kpi.get('all_module_results', {})
            quality_data = all_module_results.get('quality_analysis', {})
            
            if quality_data and quality_data.get('status') == 'success':
                aggregated_metrics = quality_data.get('aggregated_metrics', {})
                overall_metrics = aggregated_metrics.get('overall_metrics', {})
                pillar_metrics = aggregated_metrics.get('pillar_metrics', {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_quality = overall_metrics.get('avg_qei_score', 0)
                    st.metric("Average QEI Score", f"{avg_quality:.2f}")
                    
                with col2:
                    total_scored = overall_metrics.get('total_comments', 0)
                    st.metric("Comments Scored", f"{total_scored:,}")
                    
                with col3:
                    high_quality = overall_metrics.get('high_quality_comments', 0)
                    st.metric("High Quality Comments", f"{high_quality:,}")
                
                # Pillar metrics
                st.markdown("### Quality Pillars")
                pillar_col1, pillar_col2, pillar_col3 = st.columns(3)
                
                with pillar_col1:
                    relevance = pillar_metrics.get('avg_relevance', 0)
                    st.metric("Relevance", f"{relevance:.3f}")
                with pillar_col2:
                    informativeness = pillar_metrics.get('avg_informativeness', 0)
                    st.metric("Informativeness", f"{informativeness:.3f}")
                with pillar_col3:
                    constructiveness = pillar_metrics.get('avg_constructiveness', 0)
                    st.metric("Constructiveness", f"{constructiveness:.3f}")
                
                # Quality distribution
                quality_dist = aggregated_metrics.get('quality_distribution', {})
                if quality_dist:
                    st.markdown("### Quality Score Distribution")
                    fig = px.bar(
                        x=list(quality_dist.keys()),
                        y=list(quality_dist.values()),
                        title="Distribution of Quality Scores",
                        labels={'x': 'Quality Range', 'y': 'Number of Comments'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Additional metrics
                st.markdown("### Statistical Overview")
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    median_score = overall_metrics.get('median_qei_score', 0)
                    st.metric("Median Score", f"{median_score:.2f}")
                with stat_col2:
                    std_score = overall_metrics.get('std_qei_score', 0)
                    st.metric("Std Deviation", f"{std_score:.3f}")
                with stat_col3:
                    low_quality = overall_metrics.get('low_quality_comments', 0)
                    st.metric("Low Quality", f"{low_quality:,}")
                    
            else:
                st.info("Quality analysis details not found in results")
                if quality_data.get('error'):
                    st.error(f"Error: {quality_data['error']}")
        else:
            st.info("Quality analysis data not available")
    
    def render_spam_bot_detection(self):
        """Render spam and bot detection results"""
        st.markdown("## Spam & Bot Detection")
        
        # Get spam/bot data from composite_kpi -> all_module_results
        composite_kpi = self._safe_get_results('composite_kpi', {})
        if composite_kpi and composite_kpi.get('status') == 'success':
            all_module_results = composite_kpi.get('all_module_results', {})
            spam_data = all_module_results.get('spam_bot_analysis', {})
        else:
            # Fallback to direct access
            all_results = self._safe_get_results('all_module_results', {})
            spam_data = all_results.get('spam_bot_analysis', {})
        
        # Check if there's an error
        if isinstance(spam_data, dict) and 'error' in spam_data:
            st.error(f"Spam detection error: {spam_data['error']}")
            st.info("This analysis module encountered a configuration issue. The fix has been applied for future runs.")
            return
        
        if spam_data and spam_data.get('status') == 'success':
            detection_metrics = spam_data.get('detection_metrics', {})
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                spam_metrics = detection_metrics.get('spam_metrics', {})
                spam_count = spam_metrics.get('total_spam', 0)
                st.metric("Spam Comments", f"{spam_count:,}")
                
            with col2:
                bot_metrics = detection_metrics.get('bot_metrics', {})
                bot_count = bot_metrics.get('total_bots', 0)
                st.metric("Bot Accounts", f"{bot_count:,}")
                
            with col3:
                authenticity_metrics = detection_metrics.get('authenticity_metrics', {})
                avg_auth = authenticity_metrics.get('avg_authenticity', 0)
                st.metric("Avg Authenticity", f"{avg_auth:.3f}")
                
            with col4:
                duplicate_metrics = detection_metrics.get('duplicate_metrics', {})
                duplicates = duplicate_metrics.get('total_duplicates', 0)
                st.metric("Duplicates", f"{duplicates:,}")
            
            # Detailed metrics
            st.markdown("### Detection Metrics")
            
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.markdown("**Spam Analysis:**")
                spam_rate = spam_metrics.get('spam_rate', 0) * 100
                avg_spam_score = spam_metrics.get('avg_spam_score', 0)
                st.write(f"• Spam Rate: {spam_rate:.2f}%")
                st.write(f"• Avg Spam Score: {avg_spam_score:.3f}")
                
                st.markdown("**Bot Analysis:**")
                bot_rate = bot_metrics.get('bot_rate', 0) * 100
                avg_bot_score = bot_metrics.get('avg_bot_score', 0)
                st.write(f"• Bot Rate: {bot_rate:.2f}%")
                st.write(f"• Avg Bot Score: {avg_bot_score:.3f}")
                
            with metric_col2:
                st.markdown("**Authenticity Analysis:**")
                high_auth = authenticity_metrics.get('high_authenticity', 0)
                low_auth = authenticity_metrics.get('low_authenticity', 0)
                st.write(f"• High Authenticity: {high_auth:,}")
                st.write(f"• Low Authenticity: {low_auth:,}")
                
                st.markdown("**Duplicate Analysis:**")
                duplicate_rate = duplicate_metrics.get('duplicate_rate', 0) * 100
                unique_clusters = duplicate_metrics.get('unique_clusters', 0)
                st.write(f"• Duplicate Rate: {duplicate_rate:.2f}%")
                st.write(f"• Unique Clusters: {unique_clusters:,}")
            
            # Authenticity distribution
            auth_dist = authenticity_metrics.get('authenticity_distribution', {})
            if auth_dist:
                st.markdown("### Authenticity Distribution")
                fig = px.bar(
                    x=list(auth_dist.keys()),
                    y=list(auth_dist.values()),
                    title="Distribution of Authenticity Scores",
                    labels={'x': 'Authenticity Level', 'y': 'Number of Comments'},
                    color=list(auth_dist.values()),
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Overall summary
            overall_metrics = detection_metrics.get('overall_metrics', {})
            if overall_metrics:
                st.markdown("### Summary")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    total_comments = overall_metrics.get('total_comments', 0)
                    st.metric("Total Comments", f"{total_comments:,}")
                with summary_col2:
                    clean_comments = overall_metrics.get('clean_comments', 0)
                    st.metric("Clean Comments", f"{clean_comments:,}")
                with summary_col3:
                    flagged_comments = overall_metrics.get('flagged_comments', 0)
                    st.metric("Flagged Comments", f"{flagged_comments:,}")
                    
        else:
            st.info("Spam and bot detection data not available")
            if spam_data and spam_data.get('error'):
                st.error(f"Error: {spam_data['error']}")
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        self.render_main_content()
        
        # Footer
        st.markdown("---")
        st.markdown("**CommentSense Dashboard** - Advanced YouTube Comment Analysis | Built with Streamlit")

def main():
    """Main function"""
    try:
        dashboard = CommentSenseDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard initialization failed: {e}")
        logger.error(f"Dashboard error: {e}")

if __name__ == "__main__":
    main()