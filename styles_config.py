# =============================================================================
# STYLES_CONFIG.PY - Professional Visual Design for Student Analytics
# Version 2.0 - Dark Theme with Transparent Charts
# =============================================================================

import streamlit as st

def inject_custom_css():
    """Injects premium CSS theme for the Student Analytics Dashboard."""
    st.markdown("""
<style>
/* =============================================================================
   SIDEBAR STYLING - Professional Look
   ============================================================================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f2e 0%, #0f1419 100%);
    padding-top: 2rem;
}

section[data-testid="stSidebar"] > div:first-child {
    padding: 1rem 1.5rem;
}

/* Sidebar Title */
section[data-testid="stSidebar"] h1 {
    color: #00A0DC !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #0077B5;
}

/* Sidebar Headers */
section[data-testid="stSidebar"] h3 {
    color: #8b949e !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 1.5rem !important;
    margin-bottom: 0.5rem !important;
}

/* Radio buttons in sidebar */
section[data-testid="stSidebar"] .stRadio > label {
    color: #f0f6fc !important;
    font-weight: 500;
}

section[data-testid="stSidebar"] .stRadio > div {
    gap: 0.25rem;
}

section[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 0.5rem 0.75rem !important;
    margin: 0.15rem 0;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(0, 119, 181, 0.15) !important;
    border-color: #0077B5;
}

section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: linear-gradient(135deg, #0077B5, #004471) !important;
    border-color: #0077B5;
}

/* Sidebar caption */
section[data-testid="stSidebar"] .stCaption {
    color: #6b7280 !important;
    font-size: 0.75rem;
}

/* =============================================================================
   MAIN CONTENT STYLING
   ============================================================================= */

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #00A0DC !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.85rem !important;
}

/* Headers */
h1 {
    color: #f0f6fc !important;
}

h2, h3 {
    color: #e0e6ed !important;
}

/* =============================================================================
   BUTTONS
   ============================================================================= */
   
.stButton > button {
    background: linear-gradient(135deg, #0077B5 0%, #004471 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 119, 181, 0.4) !important;
}

/* =============================================================================
   TABS
   ============================================================================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(30, 41, 59, 0.5);
    padding: 4px;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 16px !important;
    color: #9ca3af !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f0f6fc !important;
    background: rgba(0, 119, 181, 0.2) !important;
}

.stTabs [aria-selected="true"] {
    background: #0077B5 !important;
    color: white !important;
}

/* =============================================================================
   DOWNLOAD BUTTON
   ============================================================================= */

.stDownloadButton > button {
    background: #0077B5 !important;
    color: white !important;
}

/* =============================================================================
   EXPANDER
   ============================================================================= */

.streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.5) !important;
    border-radius: 8px !important;
    color: #f0f6fc !important;
}

</style>
""", unsafe_allow_html=True)


def apply_chart_theme(fig):
    """
    Applies a dark transparent theme to Plotly charts.
    """
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter, sans-serif", size=12, color="#e0e6ed"),
        title_font=dict(family="Inter, sans-serif", size=16, color="#f0f6fc"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(
            bgcolor="#1a1f2e",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor="#0077B5",
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
            gridcolor="rgba(75, 85, 99, 0.3)",
            linecolor="rgba(75, 85, 99, 0.5)",
            zeroline=False,
            title_font=dict(size=12, color="#9ca3af"),
            tickfont=dict(color="#9ca3af")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(75, 85, 99, 0.3)",
            linecolor="rgba(75, 85, 99, 0.5)",
            zeroline=False,
            title_font=dict(size=12, color="#9ca3af"),
            tickfont=dict(color="#9ca3af")
        )
    )
    
    return fig
