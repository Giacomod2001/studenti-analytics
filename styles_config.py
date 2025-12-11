import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─── COLOR PALETTES ──────────────────────────────────────────────────────────
# Premium, high-contrast palette for charts
COLORS = {
    "primary": "#6366f1",      # Indigo 500
    "secondary": "#ec4899",    # Pink 500
    "accent": "#8b5cf6",       # Violet 500
    "neutral": "#cbd5e1",      # Slate 300
    "background": "#ffffff",
    "text": "#1e293b",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
}

def inject_custom_css():
    """Injects premium CSS with animations, glassmorphism, and better cards."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        /* GLOBAL TYPOGRAPHY */
        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif;
            color: #1e293b;
        }
        
        /* APP BACKGROUND */
        .stApp {
            background-color: #f8fafc;
            background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
            background-size: 20px 20px;
        }
        
        /* SIDEBAR STYLING */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
            box-shadow: 4px 0 24px rgba(0,0,0,0.02);
        }
        
        /* CARD COMPONENT CLASS (Used in HTML) */
        .premium-card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            border: 1px solid #f1f5f9;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.01), 0 2px 4px -1px rgba(0, 0, 0, 0.01);
            transition: all 0.3s ease;
        }
        .premium-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
            border-color: #6366f1;
        }
        
        /* KPI METRIC CARDS */
        .kpi-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            text-align: center;
            transition: all 0.2s ease;
        }
        .kpi-card:hover {
            border-color: #6366f1;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
        }
        .kpi-value {
            font-size: 28px;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #4f46e5, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .kpi-label {
            font-size: 14px;
            color: #64748b;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* HEADERS */
        h1, h2, h3 {
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 8px;
            background-color: white;
            border: 1px solid #e2e8f0;
            padding: 0 20px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #6366f1 !important;
            color: white !important;
            border: none;
        }
        
        /* SPINNERS */
        .stSpinner > div {
            border-top-color: #6366f1 !important;
        }
        
        /* BUTTONS */
        .stButton button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .stButton button:hover {
            transform: scale(1.02);
        }
    </style>
    """, unsafe_allow_html=True)

def apply_chart_theme(fig):
    """
    Applies a premium, modern theme to Plotly charts.
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Outfit, sans-serif", size=12, color=COLORS["text"]),
        title_font=dict(family="Outfit, sans-serif", size=18, color=COLORS["text"], weight=700),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Outfit, sans-serif",
            bordercolor=COLORS["neutral"],
            font_color=COLORS["text"]
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=COLORS["neutral"],
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        # Clean Grid
        xaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            zeroline=False
        )
    )
    return fig
