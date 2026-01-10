# =============================================================================
# STYLES_CONFIG.PY - High Contrast Corporate Theme
# Version 5.0 - Fixed Visibility \u0026 Usability
# =============================================================================

import streamlit as st

def inject_custom_css():
    """Injects high-contrast professional CSS."""
    st.markdown("""
<style>
/* =============================================================================
   RESET & BASICS
   ============================================================================= */
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif;
}

/* Force Light Background for Main Content Area */
.stApp {
    background-color: #F4F6F9;
}

/* =============================================================================
   SIDEBAR STYLING - DARK MODE FOR NAV (High Contrast)
   ============================================================================= */
section[data-testid="stSidebar"] {
    background-color: #1A202C !important; /* Dark Slate */
    border-right: 1px solid #2D3748;
}

/* Sidebar Text Color override */
section[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

/* Sidebar Title */
section[data-testid="stSidebar"] h1 {
    color: #FFFFFF !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    border-bottom: 1px solid #4A5568;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem !important;
}

section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #A0AEC0 !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em;
    margin-top: 1.5rem;
}

/* Radio Buttons (Navigation) */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0px !important;
}

section[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    padding: 10px 15px !important;
    font-size: 0.95rem !important;
    border-radius: 6px;
    margin-bottom: 2px !important;
    color: #CBD5E0 !important;
    transition: background 0.2s;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: #2D3748 !important;
    color: #FFFFFF !important;
}

/* Selected Tab in Sidebar */
section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background: #3182CE !important; /* Corporate Blue */
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

/* Divider */
section[data-testid="stSidebar"] hr {
    border-color: #4A5568 !important;
}

/* =============================================================================
   MAIN CONTENT STYLING
   ============================================================================= */
h1 {
    color: #1A202C !important;
    font-weight: 700 !important;
    font-size: 2.25rem !important;
    padding-bottom: 0.5rem;
}

h2 {
    color: #2D3748 !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
}

h3 {
    color: #4A5568 !important;
    font-weight: 600 !important;
    font-size: 1.25rem !important;
}

p, li, .stMarkdown {
    color: #2D3748 !important;
    line-height: 1.6;
}

/* Cards / Containers */
div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 20px !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
}

div[data-testid="stMetric"] label {
    color: #718096 !important; /* Muted Gray */
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #2B6CB0 !important; /* Blue Value */
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background-color: #3182CE !important;
    color: white !important;
    border-radius: 6px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 500;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.stButton > button:hover {
    background-color: #2C5282 !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

/* DataFrames */
.stDataFrame {
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    background: white;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 2px solid #E2E8F0;
    margin-bottom: 1rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #718096 !important;
    font-weight: 500;
    border-bottom: 2px solid transparent !important;
}

.stTabs [aria-selected="true"] {
    color: #3182CE !important;
    border-bottom: 2px solid #3182CE !important;
}

/* Alerts */
.stSuccess, .stInfo, .stWarning, .stError {
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid transparent;
}
.stSuccess { background: #F0FFF4 !important; border-color: #C6F6D5 !important; color: #22543D !important; }
.stInfo { background: #EBF8FF !important; border-color: #BEE3F8 !important; color: #2A4365 !important; }
.stWarning { background: #FFFAF0 !important; border-color: #FEEBC8 !important; color: #744210 !important; }
.stError { background: #FFF5F5 !important; border-color: #FED7D7 !important; color: #742A2A !important; }

/* Custom Classes */
.card-box {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}

</style>
""", unsafe_allow_html=True)

def apply_chart_theme(fig):
    """Applies a clean white corporate theme to Plotly charts."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Roboto, sans-serif", color="#2D3748"),
        title_font=dict(size=18, color="#1A202C", family="Roboto, sans-serif", weight=700),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=True, gridcolor="#EDF2F7", tickfont=dict(color="#718096")),
        yaxis=dict(showgrid=True, gridcolor="#EDF2F7", tickfont=dict(color="#718096"))
    )
    return fig
