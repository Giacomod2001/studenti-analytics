# =============================================================================
# STYLES_CONFIG.PY - Premium Visual Design for Student Analytics
# Version 3.0 - Based on datamining_git template
# =============================================================================

import streamlit as st

def inject_custom_css():
    """Injects premium CSS theme with Google Fonts, glassmorphism, and animations."""
    st.markdown("""
<style>
/* =============================================================================
   GOOGLE FONTS
   ============================================================================= */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1, h2, h3, .stTitle {
    font-family: 'Outfit', sans-serif !important;
}

/* =============================================================================
   SIDEBAR STYLING - Premium Dark
   ============================================================================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
    border-right: 1px solid rgba(0, 160, 220, 0.2);
}

section[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.5rem;
}

/* Sidebar Title */
section[data-testid="stSidebar"] h1 {
    color: #00A0DC !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    padding-bottom: 1rem;
    border-bottom: 2px solid rgba(0, 160, 220, 0.3);
    text-shadow: 0 0 20px rgba(0, 160, 220, 0.3);
}

/* Sidebar Section Headers */
section[data-testid="stSidebar"] h3 {
    color: #8b949e !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
}

/* Radio buttons - Premium Style */
section[data-testid="stSidebar"] .stRadio > div > label {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(75, 85, 99, 0.3) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    margin: 0.25rem 0 !important;
    transition: all 0.3s ease !important;
    color: #e0e6ed !important;
}

section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(0, 160, 220, 0.15) !important;
    border-color: #00A0DC !important;
    transform: translateX(4px);
}

section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: linear-gradient(135deg, #0077B5 0%, #00A0DC 100%) !important;
    border-color: transparent !important;
    box-shadow: 0 4px 15px rgba(0, 160, 220, 0.3) !important;
}

/* =============================================================================
   MAIN CONTENT STYLING
   ============================================================================= */
   
/* Page background */
.stApp {
    background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
}

/* Main title */
h1 {
    color: #f0f6fc !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

h2 {
    color: #e0e6ed !important;
    font-weight: 600 !important;
    border-bottom: 2px solid rgba(0, 160, 220, 0.2);
    padding-bottom: 0.5rem;
}

h3 {
    color: #c9d1d9 !important;
    font-weight: 500 !important;
}

/* =============================================================================
   METRICS - KPI CARDS
   ============================================================================= */
[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(75, 85, 99, 0.3);
    border-radius: 16px;
    padding: 1.25rem !important;
    transition: all 0.3s ease;
}

[data-testid="stMetric"]:hover {
    border-color: #00A0DC;
    box-shadow: 0 0 20px rgba(0, 160, 220, 0.2);
    transform: translateY(-2px);
}

[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #00A0DC !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

[data-testid="stMetricDelta"] {
    font-size: 0.9rem !important;
}

/* =============================================================================
   BUTTONS - Premium with Animations
   ============================================================================= */
.stButton > button {
    background: linear-gradient(135deg, #0077B5 0%, #00A0DC 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 119, 181, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(0, 160, 220, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* Secondary button style */
.stDownloadButton > button {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid #00A0DC !important;
    color: #00A0DC !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0077B5 0%, #00A0DC 100%) !important;
    color: white !important;
}

/* =============================================================================
   TABS - Modern Style
   ============================================================================= */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(30, 41, 59, 0.5);
    padding: 6px;
    border-radius: 12px;
    gap: 6px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 10px 20px !important;
    color: #9ca3af !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f0f6fc !important;
    background: rgba(0, 160, 220, 0.15) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0077B5 0%, #00A0DC 100%) !important;
    color: white !important;
    box-shadow: 0 2px 10px rgba(0, 160, 220, 0.3) !important;
}

/* =============================================================================
   EXPANDERS - Glassmorphism
   ============================================================================= */
.streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(75, 85, 99, 0.3) !important;
    border-radius: 12px !important;
    color: #f0f6fc !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.streamlit-expanderHeader:hover {
    border-color: #00A0DC !important;
    background: rgba(30, 41, 59, 0.8) !important;
}

/* =============================================================================
   DATAFRAMES - Styled Tables
   ============================================================================= */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
}

.stDataFrame [data-testid="stDataFrameResizable"] {
    border: 1px solid rgba(75, 85, 99, 0.3) !important;
    border-radius: 12px !important;
}

/* =============================================================================
   ALERTS - Risk Colors
   ============================================================================= */
.stSuccess {
    background: rgba(16, 185, 129, 0.15) !important;
    border-left: 4px solid #10B981 !important;
    border-radius: 8px !important;
}

.stWarning {
    background: rgba(245, 158, 11, 0.15) !important;
    border-left: 4px solid #F59E0B !important;
    border-radius: 8px !important;
}

.stError {
    background: rgba(239, 68, 68, 0.15) !important;
    border-left: 4px solid #EF4444 !important;
    border-radius: 8px !important;
}

.stInfo {
    background: rgba(0, 160, 220, 0.15) !important;
    border-left: 4px solid #00A0DC !important;
    border-radius: 8px !important;
}

/* =============================================================================
   DIVIDERS
   ============================================================================= */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(0, 160, 220, 0.3), transparent) !important;
    margin: 2rem 0 !important;
}

/* =============================================================================
   SELECTBOX & INPUTS
   ============================================================================= */
.stSelectbox > div > div {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(75, 85, 99, 0.3) !important;
    border-radius: 10px !important;
    color: #f0f6fc !important;
}

.stTextInput > div > div > input {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(75, 85, 99, 0.3) !important;
    border-radius: 10px !important;
    color: #f0f6fc !important;
}

.stTextInput > div > div > input:focus {
    border-color: #00A0DC !important;
    box-shadow: 0 0 0 2px rgba(0, 160, 220, 0.2) !important;
}

/* =============================================================================
   SCROLLBAR
   ============================================================================= */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(30, 41, 59, 0.3);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(0, 160, 220, 0.5);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #00A0DC;
}

/* =============================================================================
   CAPTION
   ============================================================================= */
.stCaption {
    color: #6b7280 !important;
}

/* =============================================================================
   LANDING PAGE SPECIFIC
   ============================================================================= */
.hero-title {
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #00A0DC 0%, #0077B5 50%, #00D4AA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 1rem;
}

.hero-subtitle {
    font-size: 1.3rem;
    color: #8b949e;
    text-align: center;
    margin-bottom: 2rem;
}

.feature-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(75, 85, 99, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}

.feature-card:hover {
    border-color: #00A0DC;
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0, 160, 220, 0.2);
}

</style>
""", unsafe_allow_html=True)


def apply_chart_theme(fig):
    """
    Applies premium dark theme to Plotly charts.
    """
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter, sans-serif", size=12, color="#e0e6ed"),
        title_font=dict(family="Outfit, sans-serif", size=18, color="#f0f6fc"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(
            bgcolor="#1a1f2e",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor="#00A0DC",
            font_color="#f0f6fc"
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(color="#e0e6ed"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(75, 85, 99, 0.2)",
            linecolor="rgba(75, 85, 99, 0.4)",
            zeroline=False,
            title_font=dict(size=12, color="#9ca3af"),
            tickfont=dict(color="#9ca3af")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(75, 85, 99, 0.2)",
            linecolor="rgba(75, 85, 99, 0.4)",
            zeroline=False,
            title_font=dict(size=12, color="#9ca3af"),
            tickfont=dict(color="#9ca3af")
        )
    )
    
    return fig


def get_risk_color(risk_pct):
    """Returns color based on risk percentage."""
    if risk_pct < 30:
        return "#10B981"  # Green
    elif risk_pct < 70:
        return "#F59E0B"  # Yellow
    else:
        return "#EF4444"  # Red


def render_hero_section():
    """Renders the landing page hero section."""
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <h1 class="hero-title">🎓 Student Analytics</h1>
        <p class="hero-subtitle">AI-Powered Dropout Prediction & Retention Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
