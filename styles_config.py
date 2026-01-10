# =============================================================================
# STYLES_CONFIG.PY - Professional Corporate Design
# Version 4.0 - Clean, Minimalist, No Emojis
# =============================================================================

import streamlit as st

def inject_custom_css():
    """Injects professional CSS theme."""
    st.markdown("""
<style>
/* =============================================================================
   FONTS & GLOBAL VARS
   ============================================================================= */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary-color: #0F52BA; /* Sapphire Blue - Professional */
    --secondary-color: #E1E8ED;
    --background-color: #F5F7F9; /* Very light gray for corporate feel */
    --text-color: #2c3e50;
    --card-bg: white;
}

* {
    font-family: 'Inter', sans-serif !important;
}

/* =============================================================================
   SIDEBAR - Clean & Structured
   ============================================================================= */
section[data-testid="stSidebar"] {
    background-color: #FAFAFA;
    border-right: 1px solid #E0E0E0;
}

section[data-testid="stSidebar"] h1 {
    color: #1a237e !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #0F52BA;
    padding-bottom: 1rem;
    margin-bottom: 2rem !important;
}

/* Sidebar Navigation Items */
section[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    color: #455a64 !important;
    border: none !important;
    padding: 0.6rem 1rem !important;
    font-weight: 500 !important;
    border-left: 3px solid transparent !important;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: #F0F4F8 !important;
    color: #0F52BA !important;
    border-left: 3px solid #0F52BA !important;
}

section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: #E3F2FD !important;
    color: #0F52BA !important;
    font-weight: 600 !important;
    border-left: 3px solid #0F52BA !important;
}

/* =============================================================================
   MAIN AREA
   ============================================================================= */
.stApp {
    background-color: #FFFFFF;
}

h1, h2, h3 {
    color: #1a237e !important;
    font-weight: 600 !important;
}

h1 { font-size: 2.2rem !important; margin-bottom: 1rem !important; }
h2 { font-size: 1.6rem !important; margin-top: 2rem !important; }
h3 { font-size: 1.2rem !important; color: #455a64 !important; }

/* =============================================================================
   CARDS & CONTAINERS (KPIs, Features)
   ============================================================================= */
.css-1r6slb0, .css-12w0qpk { /* Streamlit metric containers placeholders */
    background: white;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    padding: 1.5rem !important;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    color: #78909c !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    color: #1a237e !important;
    font-weight: 700 !important;
}

/* =============================================================================
   BUTTONS
   ============================================================================= */
.stButton > button {
    background: #0F52BA !important;
    color: white !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 4px rgba(15, 82, 186, 0.2) !important;
    transition: background 0.2s ease !important;
}

.stButton > button:hover {
    background: #0a3d8f !important;
}

.stDownloadButton > button {
    background: white !important;
    border: 1px solid #0F52BA !important;
    color: #0F52BA !important;
}

.stDownloadButton > button:hover {
    background: #F0F4F8 !important;
}

/* =============================================================================
   TABS
   ============================================================================= */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #E0E0E0;
    gap: 2rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #78909c !important;
    padding-bottom: 0.5rem !important;
}

.stTabs [aria-selected="true"] {
    border-bottom: 2px solid #0F52BA !important;
    color: #0F52BA !important;
    font-weight: 600 !important;
}

/* =============================================================================
   ALERTS & INFO BOXES
   ============================================================================= */
.stInfo {
    background-color: #E3F2FD !important;
    color: #0d47a1 !important;
    border: none !important;
    border-left: 4px solid #1976D2 !important;
}

.stSuccess {
    background-color: #E8F5E9 !important;
    color: #1b5e20 !important;
    border: none !important;
    border-left: 4px solid #43A047 !important;
}

.stWarning {
    background-color: #FFF3E0 !important;
    color: #e65100 !important;
    border: none !important;
    border-left: 4px solid #FB8C00 !important;
}

.stError {
    background-color: #FFEBEE !important;
    color: #b71c1c !important;
    border: none !important;
    border-left: 4px solid #D32F2F !important;
}

/* =============================================================================
   DATAFRAME
   ============================================================================= */
.stDataFrame {
    border: 1px solid #E0E0E0 !important;
    border-radius: 4px !important;
}

/* =============================================================================
   CUSTOM CLASSES
   ============================================================================= */
.subtitle {
    font-size: 1.1rem;
    color: #546e7a;
    line-height: 1.6;
}

.card {
    background: white;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 1.5rem;
    height: 100%;
    transition: transform 0.2s;
}
.card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
.card-title {
    color: #0F52BA;
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
}
.card-text {
    color: #546e7a;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)

def apply_chart_theme(fig):
    """Applies professional light theme to charts."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12, color="#2c3e50"),
        title_font=dict(family="Inter, sans-serif", size=16, color="#1a237e", weight=600),
        xaxis=dict(showgrid=True, gridcolor="#F0F2F5", linecolor="#CFD8DC"),
        yaxis=dict(showgrid=True, gridcolor="#F0F2F5", linecolor="#CFD8DC"),
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig
