# =============================================================================
# STYLES_CONFIG.PY - Professional Visual Design System for Student Analytics
# Version 2.0 - Clean Light Theme with Blue Accents
# =============================================================================

import streamlit as st

def inject_custom_css():
    """Injects professional CSS theme for the Student Analytics Dashboard."""
    st.markdown("""
<style>
/* =============================================================================
   CSS VARIABLES - Design Tokens
   ============================================================================= */
:root {
    /* Professional Blue Palette */
    --primary-blue: #0077B5;
    --primary-dark: #004471;
    --primary-light: #00A0DC;
    --accent-green: #00C853;
    --accent-amber: #FFB300;
    --accent-red: #E53935;
    
    /* Transitions */
    --transition-fast: 0.15s ease;
    --transition-normal: 0.3s ease;
}

/* =============================================================================
   BUTTONS
   ============================================================================= */
   
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all var(--transition-normal) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 119, 181, 0.3) !important;
}

/* =============================================================================
   METRICS - Enhanced Values
   ============================================================================= */

[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--primary-blue) !important;
}

/* =============================================================================
   TABS STYLING
   ============================================================================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 10px 20px !important;
}

.stTabs [aria-selected="true"] {
    background: var(--primary-blue) !important;
    color: white !important;
}

/* =============================================================================
   DOWNLOAD BUTTON
   ============================================================================= */

.stDownloadButton > button {
    background: var(--primary-blue) !important;
    color: white !important;
    border-radius: 8px !important;
}

</style>
""", unsafe_allow_html=True)


def apply_chart_theme(fig):
    """
    Applies a clean, readable theme to Plotly charts.
    Uses light background with high-contrast colors for readability.
    """
    # Professional color palette for data
    colors = ['#0077B5', '#00A0DC', '#00C853', '#FFB300', '#E53935', '#9C27B0', '#607D8B']
    
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12, color="#333333"),
        title_font=dict(family="Inter, sans-serif", size=16, color="#333333", weight=600),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor="#0077B5",
            font_color="#333333"
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E0E0E0",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#E8E8E8",
            linecolor="#CCCCCC",
            zeroline=False,
            title_font=dict(size=12, color="#666666")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E8E8E8",
            linecolor="#CCCCCC",
            zeroline=False,
            title_font=dict(size=12, color="#666666")
        ),
        colorway=colors
    )
    
    # Update traces for better visibility
    fig.update_traces(
        marker=dict(line=dict(width=0)),
        selector=dict(type='bar')
    )
    
    return fig
