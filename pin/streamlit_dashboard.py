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
    from pin.commentsense_pipeline import CommentSensePipeline, load_datasets
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
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <h4>Emotion Detection</h4>
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
        
        self.enable_modules = {
            "emotion_sarcasm": st.sidebar.checkbox("Emotion & Sarcasm", value=True),
            "visual_emoji": st.sidebar.checkbox("Visual & Emoji", value=True),
            "multilingual": st.sidebar.checkbox("Multilingual", value=True),
            "crisis_detection": st.sidebar.checkbox("Crisis Detection", value=True)
        }
        
        # Analysis button
        if st.sidebar.button("Run Analysis", type="primary"):
            if st.session_state.comments_df is not None:
                self.run_analysis()
            else:
                st.sidebar.error("Please load data first!")
        
        # Real-time monitoring
        st.sidebar.markdown("### Real-time Monitoring")
        self.auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
        
        if self.auto_refresh:
            refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 5, 60, 30)
            time.sleep(refresh_rate)
            st.experimental_rerun()
        
        # Cache management
        st.sidebar.markdown("### Cache Management")
        if st.sidebar.button("Clear Cache & Reset"):
            self.clear_cache()
            st.sidebar.success("Cache cleared!")
            st.experimental_rerun()
    
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
                
                if st.sidebar.button("Load Results"):
                    try:
                        with open(results_dir / selected_file, 'r') as f:
                            loaded_results = json.load(f)
                            # Ensure results is a dictionary, not a list
                            if isinstance(loaded_results, dict):
                                st.session_state.results = loaded_results
                                st.sidebar.success("Results loaded!")
                            else:
                                st.sidebar.error(f"Invalid results format: expected dict, got {type(loaded_results)}")
                    except Exception as e:
                        st.sidebar.error(f"Error loading results: {e}")
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
            st.info("Please load data and run analysis using the sidebar controls")
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
        
        # Results overview
        self.render_overview()
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Sentiment Analysis",
            "Emotion Detection", 
            "Multilingual Insights",
            "Crisis Alerts",
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
        col1, col2, col3, col4 = st.columns(4)
        
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
    
    def render_detailed_results(self):
        """Render detailed raw results"""
        st.markdown("## Detailed Results")
        
        logger.info("render_detailed_results: Starting")
        
        if st.checkbox("Show raw results JSON"):
            st.json(st.session_state.results)
        
        # Module-specific results
        module_results = self._safe_get_results('module_results', {})
        logger.info(f"render_detailed_results: module_results type: {type(module_results)}")
        
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
        
        # Download results
        if st.button("Download Results"):
            results_json = json.dumps(st.session_state.results, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=results_json,
                file_name=f"commentsense_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
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