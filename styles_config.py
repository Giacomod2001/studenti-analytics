# =============================================================================
# STYLES_CONFIG.PY - Full Dark Professional Theme
# Version 6.0 - Consistent Dark UI & Premium Tables
# =============================================================================

import streamlit as st

def inject_custom_css():
    """Injects full dark professional CSS."""
    st.markdown("""
<style>
/* =============================================================================
   RESET & BASICS
   ============================================================================= */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0E1117; /* Streamlit Dark BG */
    color: #FAFAFA;
}

/* Force Dark Background for Main Content Area */
.stApp {
    background-color: #0E1117;
}

/* =============================================================================
   SIDEBAR STYLING
   ============================================================================= */
section[data-testid="stSidebar"] {
    background-color: #161B22 !important; /* Slightly lighter than main for depth */
    border-right: 1px solid #30363D;
}

section[data-testid="stSidebar"] h1 {
    color: #58A6FF !important; /* GitHub/VSCode Blue */
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    border-bottom: 1px solid #30363D;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem !important;
}

section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #8B949E !important; /* Muted Text */
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em;
    margin-top: 1.5rem;
}

/* Navigation Links */
section[data-testid="stSidebar"] .stRadio label {
    color: #C9D1D9 !important;
    background: transparent !important;
    padding: 8px 12px !important;
    border-radius: 6px;
    margin-bottom: 2px !important;
    transition: all 0.2s;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: #21262D !important;
    color: #58A6FF !important;
}

section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background: #1F6FEB !important; /* Strong Blue */
    color: #FFFFFF !important;
}

/* =============================================================================
   MAIN CONTENT TYPOGRAPHY
   ============================================================================= */
h1 {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px;
}

h2 {
    color: #E6EDF3 !important;
    font-weight: 500 !important;
}

h3 {
    color: #C9D1D9 !important;
    font-weight: 500 !important;
}

p, li, .stMarkdown {
    color: #C9D1D9 !important;
    line-height: 1.6;
}

/* =============================================================================
   CARDS & CONTAINERS
   ============================================================================= */
div[data-testid="stMetric"] {
    background-color: #161B22 !important;
    border: 1px solid #30363D !important;
    border-radius: 6px !important;
    padding: 16px !important;
    transition: border-color 0.2s;
}

div[data-testid="stMetric"]:hover {
    border-color: #58A6FF !important;
}

div[data-testid="stMetric"] label {
    color: #8B949E !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #FAFAFA !important;
}

/* =============================================================================
   BUTTONS
   ============================================================================= */
/* Default Main Area Buttons (Action) */
.stMain .stButton > button {
    background-color: #da3633 !important; /* Professional Red */
    color: white !important;
    border: 1px solid rgba(240, 246, 252, 0.1) !important;
    border-radius: 6px;
    font-weight: 500;
}

.stMain .stButton > button:hover {
    background-color: #b62d2a !important;
    box-shadow: 0 0 8px rgba(218, 54, 51, 0.4);
}

/* Sidebar Navigation Buttons (Ghost Style) */
section[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    color: #C9D1D9 !important;
    border: 1px solid transparent !important;
    text-align: left !important;
    padding-left: 0px !important;
    font-weight: 400 !important;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #21262D !important;
    border: 1px solid #30363D !important;
    color: #58A6FF !important;
    padding-left: 10px !important; /* Slide effect */
}

section[data-testid="stSidebar"] .stButton > button:focus {
    color: #58A6FF !important;
    border-color: #58A6FF !important;
}

/* Secondary / Export Buttons */
.stDownloadButton > button {
    background-color: #21262D !important;
    color: #C9D1D9 !important;
    border: 1px solid #30363D !important;
}

.stDownloadButton > button:hover {
    background-color: #30363D !important;
    border-color: #8B949E !important;
    color: white !important;
}

/* =============================================================================
   TABLES (DATAFRAMES)
   ============================================================================= */
.stDataFrame {
    border: 1px solid #30363D;
    border-radius: 6px;
    background-color: #0D1117; 
}

/* Target streamlit dataframe internal structure if possible (limited via CSS) */
/* The uploaded image shows a clean dark table, which is default Streamlit dark theme behavior 
   if the config.toml theme is set correctly. We enforce dark container here. */

/* =============================================================================
   TABS
   ============================================================================= */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #30363D;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8B949E !important;
    border-bottom: 2px solid transparent !important;
}

.stTabs [aria-selected="true"] {
    color: #58A6FF !important;
    border-bottom: 2px solid #58A6FF !important;
}

/* =============================================================================
   ALERTS / INFO BOXES (Dark Mode)
   ============================================================================= */
.stSuccess { background: rgba(46, 160, 67, 0.15) !important; border: 1px solid rgba(46, 160, 67, 0.4) !important; color: #7EE787 !important; }
.stInfo { background: rgba(56, 139, 253, 0.15) !important; border: 1px solid rgba(56, 139, 253, 0.4) !important; color: #79C0FF !important; }
.stWarning { background: rgba(187, 128, 9, 0.15) !important; border: 1px solid rgba(187, 128, 9, 0.4) !important; color: #D2A8DA !important; } /* Adjusted for visibility */
.stError { background: rgba(248, 81, 73, 0.15) !important; border: 1px solid rgba(248, 81, 73, 0.4) !important; color: #FF7B72 !important; }

/* =============================================================================
   CUSTOM TEXT REPORT BOX
   ============================================================================= */
.report-box {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 20px;
    margin-top: 10px;
}

.report-header {
    color: #58A6FF;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 10px;
    border-bottom: 1px solid #30363D;
    padding-bottom: 5px;
}

</style>
""", unsafe_allow_html=True)

# No chart functions needed as we are using text reports only
