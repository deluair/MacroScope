import streamlit as st
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is in PYTHONPATH so that `src` package can be resolved
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
for p in {PROJECT_ROOT.as_posix(), SRC_PATH.as_posix()}:
    if p not in sys.path:
        sys.path.append(p)
from src.dashboard.utils.data_processing import load_all_data
from src.dashboard.components.executive_summary import display_executive_summary
from src.dashboard.components.deep_dive_analytics import display_deep_dive_analytics
from src.dashboard.components.risk_assessment import display_risk_assessment
from src.dashboard.components.investment_intelligence import render_investment_intelligence
from src.dashboard.utils.report_generator import generate_markdown_report, generate_html_report

# Custom CSS for professional styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --background-color: #f8f9fa;
        --card-background: #ffffff;
        --text-color: #2c3e50;
        --border-color: #e1e8ed;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #2c5aa0 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white !important;
        text-align: center;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header .subtitle {
        color: #e8f4f8 !important;
        text-align: center;
        font-size: 1.2rem !important;
        margin-top: 0.5rem !important;
        font-weight: 300 !important;
    }
    
    /* Professional sidebar styling */
    .css-1d391kg {
        background-color: #2c3e50;
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
    }
    
    /* Card-like containers */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-live {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-synthetic {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    /* Professional metrics styling */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        color: #2c3e50;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Professional button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4 0%, #2c5aa0 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Data source selection styling */
    .data-source-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    
    /* Footer styling */
    .dashboard-footer {
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #e1e8ed;
        text-align: center;
        color: #6c757d;
        font-size: 0.875rem;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """
    Main function to run the Streamlit dashboard.
    """
    st.set_page_config(
        page_title="MacroScope - Economic Intelligence Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>📊 MacroScope</h1>
        <p class="subtitle">Advanced Economic Intelligence & Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # Data source selection with enhanced styling
        st.markdown("#### 📊 Data Source Configuration")
        use_real_data = st.radio(
            "Select data source:",
            ["🌐 Real-Time API Data", "📁 Synthetic Test Data"],
            index=0,
            help="Real API Data: Live economic data from FRED, Yahoo Finance, BLS\nSynthetic Data: Generated test data for demonstration"
        ) == "🌐 Real-Time API Data"
        
        # Status indicator
        if use_real_data:
            st.markdown("""
            <div class="status-indicator status-live">
                ✅ LIVE DATA ACTIVE
            </div>
            """, unsafe_allow_html=True)
            st.info("**Data Sources:**\n- Federal Reserve (FRED)\n- Yahoo Finance\n- Bureau of Labor Statistics")
        else:
            st.markdown("""
            <div class="status-indicator status-synthetic">
                🧪 TEST DATA MODE
            </div>
            """, unsafe_allow_html=True)
            st.info("Using synthetic data for demonstration purposes")
        
        st.markdown("---")
        
        # Dashboard info
        st.markdown("#### ℹ️ Dashboard Info")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"**Last Updated:** {current_time}")
        st.markdown(f"**Data Mode:** {'Live API' if use_real_data else 'Synthetic'}")
    
    # Load data with professional loading indicator
    with st.spinner("🔄 Loading economic intelligence data..."):
        data = load_all_data(use_real_data=use_real_data)

    if not data:
        st.error("❌ **Data Loading Failed** - Unable to retrieve economic data. Please check your connection and try again.")
        st.stop()
    
    # Professional data status display
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if use_real_data:
            st.markdown("""
            <div class="metric-card">
                <h4>🌐 Live Data Status</h4>
                <p>Successfully connected to real-time economic data sources</p>
                <small>Sources: FRED, Yahoo Finance, BLS</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h4>📁 Demo Data Status</h4>
                <p>Using synthetic data for demonstration</p>
                <small>Generated test datasets loaded successfully</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Data Sources", len(data), help="Number of active data sources")
    
    with col3:
        data_freshness = "Real-time" if use_real_data else "Static"
        st.metric("Data Type", data_freshness, help="Data refresh status")
    
    # Enhanced sidebar navigation
    with st.sidebar:
        st.markdown("#### 🧭 Navigation")
        page = st.radio(
            "Select Analysis View:",
            ("📊 Executive Summary", "🔍 Deep Dive Analytics", "⚠️ Risk Assessment", "🎯 Investment Intelligence"),
            help="Choose the analysis perspective you want to explore"
        )
        
        st.markdown("---")
        
        # Professional export section
        st.markdown("#### 📄 Export & Reports")
        with st.expander("Generate Reports", expanded=False):
            st.markdown("**Available Formats:**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📝 Markdown", use_container_width=True):
                    report_md = generate_markdown_report(data)
                    st.download_button(
                        label="⬇️ Download MD",
                        data=report_md,
                        file_name=f"macroscope_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown",
                        key="md_dl",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("🌐 HTML", use_container_width=True):
                    report_html = generate_html_report(data)
                    st.download_button(
                        label="⬇️ Download HTML",
                        data=report_html,
                        file_name=f"macroscope_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                        mime="text/html",
                        key="html_dl",
                        use_container_width=True
                    )
        
        # About section
        with st.expander("ℹ️ About MacroScope"):
            st.markdown("""
            **MacroScope** is an advanced economic intelligence platform that provides:
            
            - 📈 Real-time economic indicators
            - 🔍 Deep analytical insights
            - ⚠️ Risk assessment tools
            - 📊 Interactive visualizations
            
            Built with modern data science tools for professional economic analysis.
            """)
    
    # Main content area with tabs for better organization
    st.markdown("---")
    
    # Clean page mapping
    page_mapping = {
        "📊 Executive Summary": "Executive Summary",
        "🔍 Deep Dive Analytics": "Deep Dive Analytics", 
        "⚠️ Risk Assessment": "Risk Assessment",
        "🎯 Investment Intelligence": "Investment Intelligence"
    }
    
    selected_page = page_mapping[page]
    
    # Display the selected page with enhanced styling
    if selected_page == "Executive Summary":
        st.markdown("## 📊 Executive Summary")
        st.markdown("*High-level overview of key economic indicators and trends*")
        display_executive_summary(data)
    elif selected_page == "Deep Dive Analytics":
        st.markdown("## 🔍 Deep Dive Analytics")
        st.markdown("*Detailed analysis and advanced visualizations*")
        display_deep_dive_analytics(data)
    elif selected_page == "Risk Assessment":
        st.markdown("## ⚠️ Risk Assessment")
        st.markdown("*Economic risk analysis and scenario modeling*")
        display_risk_assessment(data)
    elif selected_page == "Investment Intelligence":
        render_investment_intelligence(data, use_real_data)
    
    # Professional footer
    st.markdown("""
    <div class="dashboard-footer">
        <p>MacroScope Economic Intelligence Platform | Built with Streamlit & Python</p>
        <p>Data sources: Federal Reserve Economic Data (FRED), Yahoo Finance, Bureau of Labor Statistics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
