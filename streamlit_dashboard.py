"""
CommentSense Streamlit Dashboard

Interactive dashboard for YouTube comment analysis with real-time insights
"""
# --- macOS stability & tokenizer hygiene (must be before torch/transformers import) ---
import os, platform, multiprocessing as mp
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # avoid hard MPS failures
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")         # keeps things calmer with NumPy/BLAS
if platform.system() == "Darwin" and platform.machine() in {"arm64","aarch64"}:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")       # be explicit: no CUDA on Mac

try:
    mp.set_start_method("spawn", force=True)                # avoids leaked semaphores on macOS
except RuntimeError:
    pass
# -------------------------------------------------------------------

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
        if not key:
            logger.warning("_safe_get_results called with empty key, returning default")
            return default if default is not None else {}
            
        if not isinstance(st.session_state.results, dict):
            logger.error(f"_safe_get_results: results is not dict, type: {type(st.session_state.results)}")
            return default
        
        try:
            result = st.session_state.results.get(key, default)
            logger.debug(f"_safe_get_results: key='{key}', result_type={type(result)}")  # Changed from info to debug
            return result
        except Exception as e:
            logger.error(f"Error in _safe_get_results for key '{key}': {e}")
            return default if default is not None else {}
    
    def _get_module_data(self, module_name: str, data_key: str = None):
        """Enhanced helper to get module data from various result sources"""
        logger.info(f"_get_module_data: Looking for {module_name}, data_key: {data_key}")
        
        # Try aggregated_insights first
        insights = self._safe_get_results('aggregated_insights', {})
        if data_key and data_key in insights:
            logger.info(f"_get_module_data: Found {data_key} in aggregated_insights")
            return insights[data_key]
        
        # Try module_results with exact name
        module_results = self._safe_get_results('module_results', {})
        if module_name in module_results:
            logger.info(f"_get_module_data: Found {module_name} in module_results")
            return module_results[module_name]
        
        # Try module_results with alternative names (based on actual result structure)
        alternative_names = {
            'category_classification': ['category_analysis', 'categories'],
            'persona_clustering': ['personas', 'persona_analysis'],
            'crisis_detection': ['crisis_analysis', 'crisis'],
            'spam_bot_detection': ['spam_bot_analysis', 'spam_analysis'],
            'multilingual_analysis': ['multilingual', 'language_analysis'],
            'quality_scoring': ['quality_analysis', 'comment_quality'],
            'network_analysis': ['network', 'influencer_analysis'],
            'emotion_detection': ['emotion_analysis', 'emotion_sarcasm'],
            'visual_emoji': ['emoji_analysis', 'visual_emoji_analysis'],
            'predictive_analytics': ['predictive_analysis', 'predictive']
        }
        
        if module_name in alternative_names:
            for alt_name in alternative_names[module_name]:
                if alt_name in module_results:
                    logger.info(f"_get_module_data: Found {alt_name} as alternative for {module_name}")
                    return module_results[alt_name]
        
        # Try composite_kpi all_module_results
        composite_kpi = self._safe_get_results('composite_kpi', {})
        if composite_kpi and composite_kpi.get('status') == 'success':
            all_modules = composite_kpi.get('all_module_results', {})
            
            # Try exact name
            if module_name in all_modules:
                logger.info(f"_get_module_data: Found {module_name} in composite_kpi")
                return all_modules[module_name]
            
            # Try alternative names
            if module_name in alternative_names:
                for alt_name in alternative_names[module_name]:
                    if alt_name in all_modules:
                        logger.info(f"_get_module_data: Found {alt_name} in composite_kpi as alternative for {module_name}")
                        return all_modules[alt_name]
        
        # Try direct result keys for backwards compatibility
        results = st.session_state.results or {}
        if module_name in results:
            logger.info(f"_get_module_data: Found {module_name} in direct results")
            return results[module_name]
        
        logger.warning(f"_get_module_data: No data found for {module_name}")
        return None
    
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
        
        # Display key analysis modules in priority order
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <h4>Quality Scoring 2.0</h4>
                <p>Comprehensive comment quality evaluation with composite KPIs</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container">
                <h4>Spam & Bot Detection</h4>
                <p>Advanced spam filtering & bot network identification</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container">
                <h4>Cross-Language Analysis</h4>
                <p>Support for 40+ languages with auto-translation</p>
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
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.markdown("""
            <div class="metric-container">
                <h4>Category Classification</h4>
                <p>Content categorization and sub-topic analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col6:
            st.markdown("""
            <div class="metric-container">
                <h4>Customer Personas</h4>
                <p>Voice-of-customer clustering and persona identification</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col7:
            st.markdown("""
            <div class="metric-container">
                <h4>Network Analysis</h4>
                <p>Influencer detection and community mapping</p>
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
                "dataset/comments5.csv",
                "videos.csv"
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
            # Note: Pipeline config doesn't have 'modules' key - modules are configured individually
            # This configuration step is optional since modules are enabled by default
            if 'modules' in st.session_state.pipeline.config:
                for module, enabled in self.enable_modules.items():
                    if module in st.session_state.pipeline.config['modules']:
                        st.session_state.pipeline.config['modules'][module]['enabled'] = enabled
            else:
                # Pipeline uses default module configurations
                logger.info("Pipeline uses default module configurations (modules config not found)")
            
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
        
        # Detailed analysis tabs - Reordered according to user preference
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Comment Quality Scoring 2.0",
            "Spam + Bot Network Detection",
            "Cross-Language Analysis", 
            "Crisis Detection",
            "Category Classification",
            "Voice-of-Customer Personas",
            "Influencer & Community Detection",
            "Detailed Results"
        ])
        
        with tab1:
            # Comment Quality Scoring 2.0 (Composite KPI)
            self.render_composite_kpi()
            self.render_quality_analysis()
            
        with tab2:
            # Spam + Bot Network Detection
            self.render_spam_bot_detection()
            
        with tab3:
            # Cross-Language Comment Analysis
            self.render_multilingual_analysis()
            
        with tab4:
            # Crisis Detection (Early Warning)
            self.render_crisis_alerts()
            
        with tab5:
            # Category + Sub-Topic Classification
            self.render_category_classification()
            
        with tab6:
            # Voice-of-Customer Personas
            self.render_persona_clustering()
            
        with tab7:
            # Influencer & Community Detection
            self.render_network_analysis()
            self.render_influencers_communities()
            
        with tab8:
            # Detailed Results
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
    
    def render_category_classification(self):
        """Render category classification results"""
        st.markdown("## Category + Sub-Topic Classification")
        
        # Try multiple sources for category data
        category_data = self._get_module_data('category_classification')
        
        if category_data is None or (isinstance(category_data, dict) and not category_data):
            # Try composite KPI results
            composite_kpi = self._safe_get_results('composite_kpi', {})
            all_modules = composite_kpi.get('all_module_results', {})
            category_data = all_modules.get('category_classification', {})
        
        # Handle DataFrame results directly from modules
        if isinstance(category_data, pd.DataFrame):
            # Process DataFrame directly
            st.success(f"Classified {len(category_data)} comments")
            
            # Display category summary
            if 'category' in category_data.columns:
                category_counts = category_data['category'].value_counts().to_dict()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Category pie chart
                    import plotly.express as px
                    fig = px.pie(
                        values=list(category_counts.values()),
                        names=list(category_counts.keys()),
                        title="Content Category Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Category bar chart
                    fig_bar = px.bar(
                        x=list(category_counts.keys()),
                        y=list(category_counts.values()),
                        title="Comments by Category",
                        labels={'x': 'Category', 'y': 'Number of Comments'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Show category breakdown
            st.markdown("### Category Analysis")
            display_cols = [col for col in ['text', 'category', 'confidence', 'subtopic'] if col in category_data.columns]
            if display_cols:
                st.dataframe(category_data[display_cols].head(20))
            else:
                st.dataframe(category_data.head(20))
                
        elif isinstance(category_data, list) and len(category_data) > 0:
            # Handle list of dicts format (new JSON serialization)
            st.success(f"Classified {len(category_data)} comments")
            
            # Convert to DataFrame for processing
            category_df = pd.DataFrame(category_data)
            
            # Display category summary
            if 'category' in category_df.columns:
                category_counts = category_df['category'].value_counts().to_dict()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Category pie chart
                    import plotly.express as px
                    fig = px.pie(
                        values=list(category_counts.values()),
                        names=list(category_counts.keys()),
                        title="Content Category Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Category bar chart
                    fig_bar = px.bar(
                        x=list(category_counts.keys()),
                        y=list(category_counts.values()),
                        title="Comments by Category",
                        labels={'x': 'Category', 'y': 'Number of Comments'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Subtopic analysis
                if 'subtopic' in category_df.columns:
                    st.markdown("### Subtopic Analysis")
                    
                    # Overall subtopic distribution
                    subtopic_counts = category_df['subtopic'].value_counts()
                    fig_sub = px.bar(
                        x=subtopic_counts.index,
                        y=subtopic_counts.values,
                        title="Overall Subtopic Distribution",
                        labels={'x': 'Subtopic', 'y': 'Count'}
                    )
                    st.plotly_chart(fig_sub, use_container_width=True)
                    
                    # Breakdown by category
                    st.markdown("### Subtopic Breakdown by Category")
                    for category in category_df['category'].unique():
                        cat_data = category_df[category_df['category'] == category]
                        subtopic_breakdown = cat_data['subtopic'].value_counts()
                        
                        with st.expander(f"{category.title()} ({len(cat_data)} comments)"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                for subtopic, count in subtopic_breakdown.items():
                                    percentage = (count / len(cat_data)) * 100
                                    st.write(f"• **{subtopic}**: {count} ({percentage:.1f}%)")
                            
                            with col2:
                                # Show examples for this category
                                if 'text' in cat_data.columns:
                                    examples = cat_data.head(3)['text'].tolist()
                                    st.markdown("**Examples:**")
                                    for i, example in enumerate(examples):
                                        st.write(f"{i+1}. {example[:100]}...")
            
            # Show category breakdown
            st.markdown("### Category Analysis")
            display_cols = [col for col in ['text', 'category', 'category_score', 'subtopic'] if col in category_df.columns]
            if display_cols:
                st.dataframe(category_df[display_cols].head(20))
            else:
                st.dataframe(category_df.head(20))
                
        elif isinstance(category_data, dict) and category_data.get('status') == 'success':
            results = category_data.get('results', {})
            
            # Category distribution
            category_counts = results.get('category_counts', {})
            if category_counts:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Category pie chart
                    import plotly.express as px
                    fig = px.pie(
                        values=list(category_counts.values()),
                        names=list(category_counts.keys()),
                        title="Content Category Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Category bar chart
                    fig_bar = px.bar(
                        x=list(category_counts.keys()),
                        y=list(category_counts.values()),
                        title="Comments by Category",
                        labels={'x': 'Category', 'y': 'Number of Comments'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Category metrics
                st.markdown("### Category Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_categories = len(category_counts)
                    st.metric("Total Categories", total_categories)
                
                with col2:
                    most_common_category = max(category_counts, key=category_counts.get) if category_counts else "N/A"
                    st.metric("Most Common", most_common_category)
                
                with col3:
                    total_classified = sum(category_counts.values())
                    st.metric("Comments Classified", f"{total_classified:,}")
                
                with col4:
                    avg_per_category = total_classified / total_categories if total_categories > 0 else 0
                    st.metric("Avg per Category", f"{avg_per_category:.0f}")
            
            # Sub-topic analysis
            subtopics = results.get('subtopics', {})
            if subtopics:
                st.markdown("### Sub-Topic Analysis")
                
                for category, topics in subtopics.items():
                    if topics:
                        with st.expander(f"{category} Sub-Topics"):
                            for topic, count in topics.items():
                                st.write(f"• **{topic}**: {count} comments")
            
            # Classification confidence
            confidence_stats = results.get('confidence_stats', {})
            if confidence_stats:
                st.markdown("### Classification Quality")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_confidence = confidence_stats.get('avg_confidence', 0)
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                
                with col2:
                    high_confidence = confidence_stats.get('high_confidence_count', 0)
                    st.metric("High Confidence", f"{high_confidence:,}")
                
                with col3:
                    low_confidence = confidence_stats.get('low_confidence_count', 0)
                    st.metric("Low Confidence", f"{low_confidence:,}")
        
        elif isinstance(category_data, dict) and 'error' in category_data:
            st.error(f"Category classification error: {category_data['error']}")
        else:
            st.warning("Category classification module is not currently enabled in the pipeline")
            st.info("This module would provide content categorization and sub-topic analysis. To enable it, please check the pipeline configuration.")
    
    def render_persona_clustering(self):
        """Render persona clustering results"""
        st.markdown("## Voice-of-Customer Personas")
        
        # Try multiple sources for persona data
        persona_data = self._get_module_data('persona_clustering')
        
        if persona_data is None or (isinstance(persona_data, dict) and not persona_data):
            # Try composite KPI results
            composite_kpi = self._safe_get_results('composite_kpi', {})
            all_modules = composite_kpi.get('all_module_results', {})
            persona_data = all_modules.get('persona_clustering', {})
        
        # Handle DataFrame results directly from modules
        if isinstance(persona_data, pd.DataFrame):
            # Process DataFrame directly
            st.success(f"Found {len(persona_data)} user personas")
            
            # Display personas summary
            if 'persona_label' in persona_data.columns:
                persona_counts = persona_data['persona_label'].value_counts().to_dict()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Persona pie chart
                    import plotly.express as px
                    fig = px.pie(
                        values=list(persona_counts.values()),
                        names=list(persona_counts.keys()),
                        title="User Persona Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Persona bar chart
                    fig_bar = px.bar(
                        x=list(persona_counts.keys()),
                        y=list(persona_counts.values()),
                        title="Users by Persona Type",
                        labels={'x': 'Persona', 'y': 'Number of Users'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Show top personas
            st.markdown("### Top User Personas")
            display_cols = [col for col in ['user_id', 'persona_label', 'dominant_category', 'top_subtopic', 'comment_count'] if col in persona_data.columns]
            if display_cols:
                st.dataframe(persona_data[display_cols].head(20))
            else:
                st.dataframe(persona_data.head(20))
                
        elif isinstance(persona_data, dict) and persona_data.get('status') == 'success':
            results = persona_data.get('results', {})
            
            # Persona distribution
            persona_counts = results.get('persona_counts', {})
            if persona_counts:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Persona pie chart
                    import plotly.express as px
                    fig = px.pie(
                        values=list(persona_counts.values()),
                        names=list(persona_counts.keys()),
                        title="User Persona Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Persona bar chart
                    fig_bar = px.bar(
                        x=list(persona_counts.keys()),
                        y=list(persona_counts.values()),
                        title="Users by Persona Type",
                        labels={'x': 'Persona', 'y': 'Number of Users'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Persona metrics
                st.markdown("### Persona Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_personas = len(persona_counts)
                    st.metric("Total Personas", total_personas)
                
                with col2:
                    dominant_persona = max(persona_counts, key=persona_counts.get) if persona_counts else "N/A"
                    st.metric("Dominant Persona", dominant_persona)
                
                with col3:
                    total_users = sum(persona_counts.values())
                    st.metric("Users Classified", f"{total_users:,}")
                
                with col4:
                    avg_per_persona = total_users / total_personas if total_personas > 0 else 0
                    st.metric("Avg per Persona", f"{avg_per_persona:.0f}")
            
            # Persona characteristics
            persona_details = results.get('persona_details', {})
            if persona_details:
                st.markdown("### Persona Characteristics")
                
                for persona, details in persona_details.items():
                    with st.expander(f"{persona} Persona Details"):
                        if isinstance(details, dict):
                            for key, value in details.items():
                                st.write(f"• **{key.replace('_', ' ').title()}**: {value}")
                        else:
                            st.write(f"• {details}")
            
            # Clustering quality
            clustering_stats = results.get('clustering_stats', {})
            if clustering_stats:
                st.markdown("### Clustering Quality")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    silhouette_score = clustering_stats.get('silhouette_score', 0)
                    st.metric("Silhouette Score", f"{silhouette_score:.3f}")
                
                with col2:
                    inertia = clustering_stats.get('inertia', 0)
                    st.metric("Inertia", f"{inertia:.0f}")
                
                with col3:
                    n_clusters = clustering_stats.get('n_clusters', 0)
                    st.metric("Optimal Clusters", n_clusters)
        
        elif isinstance(persona_data, dict) and 'error' in persona_data:
            st.error(f"Persona clustering error: {persona_data['error']}")
        else:
            st.warning("👥 Voice-of-Customer personas module is not currently enabled in the pipeline")
            st.info("This module would provide user clustering and persona identification. To enable it, please check the pipeline configuration.")

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
            if overall_sentiment and any(overall_sentiment.get(key) is not None for key in ['mean', 'min', 'max']):
                mean_val = overall_sentiment.get('mean', 0)
                min_val = overall_sentiment.get('min', 0) 
                max_val = overall_sentiment.get('max', 0)
                std_val = overall_sentiment.get('std', 0)
                
                # Create a more informative sentiment chart
                fig = go.Figure()
                
                # Add main sentiment values
                fig.add_trace(go.Scatter(
                    x=['Mean', 'Min', 'Max'],
                    y=[mean_val, min_val, max_val],
                    mode='markers+lines',
                    marker=dict(size=12, color=['blue', 'red', 'green']),
                    name='Sentiment Values'
                ))
                
                # Add standard deviation as error bar on mean
                fig.add_trace(go.Scatter(
                    x=['Mean'],
                    y=[mean_val],
                    error_y=dict(type='data', array=[std_val], visible=True),
                    mode='markers',
                    marker=dict(size=15, color='orange'),
                    name='Mean ± Std'
                ))
                
                fig.update_layout(
                    title=f"Sentiment Analysis (Mean: {mean_val:.3f})",
                    yaxis_title="Sentiment Score",
                    yaxis=dict(range=[-1.1, 1.1]),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available")
        
        with col2:
            # Sarcasm detection metrics
            sarcasm_stats = insights.get('sarcasm_stats', {})
            if sarcasm_stats:
                sarcastic_count = sarcasm_stats.get('sarcastic_comments', 0)
                total_comments = sarcasm_stats.get('total_comments', 1)
                sarcasm_ratio = sarcastic_count / total_comments if total_comments > 0 else 0
                
                st.metric(
                    "Sarcastic Comments", 
                    f"{sarcastic_count}",
                    delta=f"{sarcasm_ratio:.1%} of total",
                    help="Number and percentage of sarcastic comments detected"
                )
                
                if sarcasm_ratio > 0.1:  # More than 10% sarcastic
                    st.warning(f"High sarcasm detected ({sarcasm_ratio:.1%})")
                elif sarcasm_ratio > 0.05:  # More than 5% sarcastic
                    st.info(f"Moderate sarcasm detected ({sarcasm_ratio:.1%})")
                else:
                    st.success("Low sarcasm levels")
            else:
                # Try to get sarcasm data from module results
                module_results = self._safe_get_results('module_results', {})
                if 'emotion_sarcasm' in module_results:
                    emotion_list = module_results['emotion_sarcasm']
                    if emotion_list:
                        sarcastic_count = sum(1 for item in emotion_list if item.get('is_sarcastic', False))
                        total_count = len(emotion_list)
                        sarcasm_ratio = sarcastic_count / total_count if total_count > 0 else 0
                        
                        st.metric("Sarcastic Comments", f"{sarcastic_count}/{total_count}")
                        if sarcasm_ratio > 0.1:
                            st.warning(f"High sarcasm: {sarcasm_ratio:.1%}")
                        else:
                            st.success(f"Sarcasm level: {sarcasm_ratio:.1%}")
                else:
                    st.info("No sarcasm analysis data available")
    
    def render_emotion_analysis(self):
        """Render emotion analysis visualizations"""
        st.markdown("## Emotion Detection")
        
        logger.info("render_emotion_analysis: Starting")
        insights = self._safe_get_results('aggregated_insights', {})
        logger.info(f"render_emotion_analysis: insights type: {type(insights)}")
        emotions = insights.get('dominant_emotions', {})
        logger.info(f"render_emotion_analysis: emotions type: {type(emotions)}")
        
        if emotions and any(v > 0 for v in emotions.values()):
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Emotion radar chart
                emotion_names = list(emotions.keys())
                emotion_values = list(emotions.values())
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=emotion_values,
                    theta=emotion_names,
                    fill='toself',
                    name='Emotions',
                    line_color='rgb(51, 153, 255)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(emotion_values) * 1.1],
                            tickfont=dict(size=10)
                        )),
                    title="Emotion Radar Chart",
                    font=dict(size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Emotion bar chart with colors
                emotion_colors = {
                    'joy': '#FFD700',
                    'sadness': '#4682B4', 
                    'anger': '#DC143C',
                    'fear': '#9932CC',
                    'surprise': '#FF6347',
                    'disgust': '#228B22',
                    'neutral': '#708090'
                }
                
                colors = [emotion_colors.get(emotion, '#708090') for emotion in emotion_names]
                
                fig_bar = go.Figure(data=go.Bar(
                    x=emotion_names,
                    y=emotion_values,
                    marker_color=colors,
                    text=[f'{v:.3f}' for v in emotion_values],
                    textposition='auto'
                ))
                
                fig_bar.update_layout(
                    title="Emotion Intensity",
                    xaxis_title="Emotion",
                    yaxis_title="Intensity",
                    yaxis=dict(range=[0, max(emotion_values) * 1.1])
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Show dominant emotion info
            dominant_emotion = max(emotions, key=emotions.get)
            dominant_value = emotions[dominant_emotion]
            
            st.success(f"**Dominant Emotion**: {dominant_emotion.title()} ({dominant_value:.1%})")
            
        else:
            st.info("No emotion data available - analyzing raw emotion results...")
            
            # Try to get data from module results 
            module_results = self._safe_get_results('module_results', {})
            if 'emotion_sarcasm' in module_results:
                emotion_list = module_results['emotion_sarcasm']
                if emotion_list and len(emotion_list) > 0:
                    st.info(f"Found {len(emotion_list)} emotion analysis results")
                    
                    # Aggregate emotions from individual results
                    emotion_totals = {}
                    for result in emotion_list:
                        if 'emotions' in result:
                            for emotion, value in result['emotions'].items():
                                emotion_totals[emotion] = emotion_totals.get(emotion, 0) + value
                    
                    if emotion_totals:
                        # Normalize by number of results
                        emotion_averages = {k: v/len(emotion_list) for k, v in emotion_totals.items()}
                        st.bar_chart(emotion_averages)
                        st.success(f"Showing aggregated emotions from {len(emotion_list)} comments")
            else:
                st.warning("No emotion analysis results found in module data")
    
    def render_multilingual_analysis(self):
        """Render multilingual analysis"""
        st.markdown("## Multilingual Analysis")
        
        logger.info("render_multilingual_analysis: Starting")
        
        # Try to get multilingual data using enhanced retrieval
        multilingual_data = self._get_module_data('multilingual_analysis')
        
        # Also check aggregated insights
        insights = self._safe_get_results('aggregated_insights', {})
        lang_dist = insights.get('language_distribution', {})
        
        logger.info(f"render_multilingual_analysis: multilingual_data type: {type(multilingual_data)}")
        logger.info(f"render_multilingual_analysis: lang_dist type: {type(lang_dist)}")
        
        # Check if we have meaningful language distribution
        has_lang_data = lang_dist and any(v > 0 for v in lang_dist.values()) if lang_dist else False
        
        # If no aggregated language distribution, try to extract from multilingual module results
        if not has_lang_data:
            module_results = self._safe_get_results('module_results', {})
            if 'multilingual' in module_results and isinstance(module_results['multilingual'], list):
                # Extract languages from individual results
                temp_lang_counts = {}
                for result in module_results['multilingual']:
                    if isinstance(result, dict) and 'language' in result:
                        lang = result['language']
                        if lang and lang != 'unknown':
                            temp_lang_counts[lang] = temp_lang_counts.get(lang, 0) + 1
                
                if temp_lang_counts:
                    lang_dist = temp_lang_counts
                    has_lang_data = True
                    st.info("Language distribution extracted from multilingual analysis")
        
        if has_lang_data:
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
            
            # Show summary
            total_comments = sum(lang_dist.values())
            most_common_lang = max(lang_dist, key=lang_dist.get)
            st.success(f"**{total_comments}** total comments analyzed, **{most_common_lang}** is the most common language")
            
        else:
            st.warning("No language distribution available")
            st.info("This could happen if:")
            st.write("- All comments are very short (< 8 characters)")
            st.write("- All comments contain only symbols/emojis")
            st.write("- Language detection failed")
            
            # Show raw multilingual data for debugging
            module_results = self._safe_get_results('module_results', {})
            composite_kpi = self._safe_get_results('composite_kpi', {})
            all_modules = composite_kpi.get('all_module_results', {}) if composite_kpi else {}
            
            if 'multilingual' in module_results:
                with st.expander("Raw Multilingual Analysis Data (for debugging)"):
                    multilingual_data = module_results['multilingual']
                    if isinstance(multilingual_data, list) and len(multilingual_data) > 0:
                        st.write(f"Found {len(multilingual_data)} results:")
                        # Show first 3 results as examples
                        for i, result in enumerate(multilingual_data[:3]):
                            if isinstance(result, dict):
                                st.json({f"example_{i+1}": result})
            
            elif 'multilingual_analysis' in all_modules:
                with st.expander("Raw Multilingual Analysis Data (for debugging)"):
                    multilingual_data = all_modules['multilingual_analysis']
                    if isinstance(multilingual_data, list) and len(multilingual_data) > 0:
                        st.write(f"Found {len(multilingual_data)} results from composite KPI:")
                        # Show first 3 results as examples
                        for i, result in enumerate(multilingual_data[:3]):
                            if isinstance(result, dict):
                                st.json({f"example_{i+1}": result})
                        
                        # Try to extract language distribution manually
                        st.write("**Extracting language distribution:**")
                        manual_lang_counts = {}
                        for result in multilingual_data:
                            if isinstance(result, dict) and 'language' in result:
                                lang = result['language']
                                if lang:
                                    manual_lang_counts[lang] = manual_lang_counts.get(lang, 0) + 1
                        
                        if manual_lang_counts:
                            st.success(f"Found languages: {manual_lang_counts}")
                            # Create quick visualization
                            fig = px.bar(
                                x=list(manual_lang_counts.keys()),
                                y=list(manual_lang_counts.values()),
                                title="Language Distribution (Manual Extraction)",
                                labels={'x': 'Language', 'y': 'Number of Comments'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No language data found in multilingual results")
                    else:
                        st.json({"multilingual_results": multilingual_data})
            else:
                # Default assumption - most comments are probably English
                st.info("**Assumption**: Most comments appear to be in **English** (language detection not available)")
                pipeline_info = self._safe_get_results('pipeline_info', {})
                total_comments = pipeline_info.get('total_comments', 0)
                if total_comments > 0:
                    estimated_lang_dist = {'English': total_comments}
                    fig = px.bar(
                        x=['English (estimated)'],
                        y=[total_comments],
                        title="Estimated Language Distribution",
                        labels={'x': 'Language', 'y': 'Number of Comments'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_crisis_alerts(self):
        """Render crisis detection alerts"""
        st.markdown("## Crisis Detection")
        
        logger.info("render_crisis_alerts: Starting")
        
        # Try multiple sources for crisis data using correct module name
        crisis_results = self._get_module_data('crisis_detection')  # This will find crisis_analysis via alternative names
        
        if crisis_results is None or (isinstance(crisis_results, dict) and not crisis_results):
            # Direct check for crisis_analysis
            composite_kpi = self._safe_get_results('composite_kpi', {})
            all_modules = composite_kpi.get('all_module_results', {})
            crisis_results = all_modules.get('crisis_analysis', {})
        
        logger.info(f"render_crisis_alerts: crisis_results type: {type(crisis_results)}")
        
        if isinstance(crisis_results, dict) and 'alerts' in crisis_results:
            alerts = crisis_results['alerts']
            summary = crisis_results.get('summary', {})
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Alerts", len(alerts))
            with col2:
                high_severity_alerts = len([a for a in alerts if a.get('severity', 0) > 0.7])
                st.metric("High Severity", high_severity_alerts)
            with col3:
                risk_level = summary.get('risk_level', 'low')
                st.metric("Risk Level", risk_level.upper())
            with col4:
                total_analyzed = crisis_results.get('total_comments_analyzed', 0)
                st.metric("Comments Analyzed", f"{total_analyzed:,}")
            
            if alerts:
                st.warning(f"{len(alerts)} crisis alerts detected!")
                
                for i, alert in enumerate(alerts):
                    severity = alert.get('severity', 0)
                    alert_type = alert.get('alert_type', 'unknown')
                    description = alert.get('description', 'No description')
                    confidence = alert.get('confidence', 0)
                    
                    severity_color = 'red' if severity > 0.7 else 'orange' if severity > 0.4 else 'yellow'
                    
                    st.markdown(f"""
                    <div style="border-left: 4px solid {severity_color}; padding: 10px; margin: 10px 0; background: #f8f9fa;">
                        <strong>Alert {i+1}: {alert_type.replace('_', ' ').title()}</strong><br>
                        Severity: {severity:.3f} | Confidence: {confidence:.3f}<br>
                        {description}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No crisis alerts detected - All comments appear safe")
                
            # Show summary statistics if available
            if summary:
                st.markdown("### Analysis Summary")
                col1, col2 = st.columns(2)
                with col1:
                    if 'alert_types' in summary:
                        st.write("**Alert Types:**")
                        for alert_type, count in summary['alert_types'].items():
                            st.write(f"• {alert_type.replace('_', ' ').title()}: {count}")
                with col2:
                    max_severity = summary.get('max_severity', 0)
                    st.metric("Max Severity", f"{max_severity:.3f}")
                    
        elif isinstance(crisis_results, list):
            st.info("Crisis detection results in list format - please check data structure")
        else:
            st.info("Crisis detection data not available - module may not have run successfully")
    
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
                                # For list results, show summary and sample
                                st.markdown("**Sample Results:**")
                                st.json(module_data[:3])  # Show first 3 items
                                if len(module_data) > 3:
                                    st.info(f"... and {len(module_data) - 3} more items")
                                
                                # Show statistics if possible
                                if module_name == 'emotion_sarcasm':
                                    self._show_emotion_stats(module_data)
                                elif module_name == 'category_classification':
                                    self._show_category_stats(module_data)
                                elif module_name == 'persona_clustering':
                                    self._show_persona_stats(module_data)
                        elif hasattr(module_data, 'to_dict'):  # DataFrame
                            st.info(f"{module_name} DataFrame with {len(module_data)} rows")
                            st.dataframe(module_data.head(10))
                        else:
                            st.warning(f"Unexpected data type in {module_name}: {type(module_data)}")
                            try:
                                st.json(str(module_data)[:1000])  # Show first 1000 chars as string
                            except:
                                st.error(f"Cannot display {module_name} data")
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
        
        # Get spam/bot data using enhanced module data retrieval
        spam_data = self._get_module_data('spam_bot_detection')  # This will find spam_bot_analysis via alternative names
        
        if spam_data is None or (isinstance(spam_data, dict) and not spam_data):
            # Direct check for spam_bot_analysis
            composite_kpi = self._safe_get_results('composite_kpi', {})
            all_module_results = composite_kpi.get('all_module_results', {})
            spam_data = all_module_results.get('spam_bot_analysis', {})
        
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
    
    def _show_emotion_stats(self, emotion_data):
        """Show emotion statistics for detailed results"""
        if not emotion_data:
            return
            
        emotions = {}
        sentiments = {}
        sarcasm_count = 0
        
        for item in emotion_data:
            if isinstance(item, dict):
                # Count dominant emotions
                dominant = item.get('dominant_emotion', 'unknown')
                emotions[dominant] = emotions.get(dominant, 0) + 1
                
                # Count sentiments
                sentiment = item.get('sentiment', 'unknown')
                sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
                
                # Count sarcasm
                if item.get('is_sarcastic', False):
                    sarcasm_count += 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sarcastic Comments", sarcasm_count)
        with col2:
            most_common_emotion = max(emotions, key=emotions.get) if emotions else "N/A"
            st.metric("Most Common Emotion", most_common_emotion)
        with col3:
            most_common_sentiment = max(sentiments, key=sentiments.get) if sentiments else "N/A"
            st.metric("Most Common Sentiment", most_common_sentiment)
    
    def _show_category_stats(self, category_data):
        """Show category statistics for detailed results"""
        if not category_data:
            return
            
        categories = {}
        subtopics = {}
        
        for item in category_data:
            if isinstance(item, dict):
                category = item.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                
                subtopic = item.get('subtopic', 'unknown')
                subtopics[subtopic] = subtopics.get(subtopic, 0) + 1
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Categories", len(categories))
            if categories:
                st.write("**Top Categories:**")
                sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
                for cat, count in sorted_cats:
                    st.write(f"• {cat}: {count}")
        
        with col2:
            st.metric("Total Subtopics", len(subtopics))
            if subtopics:
                st.write("**Top Subtopics:**")
                sorted_subs = sorted(subtopics.items(), key=lambda x: x[1], reverse=True)[:3]
                for sub, count in sorted_subs:
                    st.write(f"• {sub}: {count}")
    
    def _show_persona_stats(self, persona_data):
        """Show persona statistics for detailed results"""
        if not persona_data:
            return
            
        clusters = {}
        
        for item in persona_data:
            if isinstance(item, dict):
                cluster = item.get('cluster', 'unknown')
                clusters[cluster] = clusters.get(cluster, 0) + 1
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Clusters", len(clusters))
        with col2:
            if clusters:
                largest_cluster = max(clusters, key=clusters.get)
                st.metric("Largest Cluster", f"{largest_cluster} ({clusters[largest_cluster]} users)")
    
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