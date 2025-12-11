import streamlit as st
import plotly.graph_objects as go

def inject_custom_css():
    """Injects global CSS for the application."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #f8f9fa;
        }
        
        [data-testid="stMetric"] {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border: 1px solid #e9ecef;
        }
        
        @media (prefers-color-scheme: dark) {
            [data-testid="stMetric"] {
                background-color: #262730;
                border: 1px solid #363940;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            .stApp {
                background-color: #0e1117;
            }
        }

        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
            border-right: 1px solid #e9ecef;
        }
        @media (prefers-color-scheme: dark) {
            [data-testid="stSidebar"] {
                background-color: #1a1c24;
                border-right: 1px solid #363940;
            }
        }

        h1, h2, h3 {
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        .block-container {
            padding-top: 2rem;
        }
        
        .stSpinner > div {
            border-top-color: #4F46E5 !important;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_chart_theme(fig):
    """
    Applies a minimal, clean theme to Plotly charts.
    Focus on readability with minimal colors.
    """
    fig.update_layout(
        template="simple_white",
        font=dict(family="Inter, sans-serif", size=11, color="#1f2937"),
        title_font=dict(family="Inter, sans-serif", size=16, color="#111827"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Inter, sans-serif",
            bordercolor="#e5e7eb"
        ),
        showlegend=True,
        legend=dict(
            bgcolor="white",
            bordercolor="#e5e7eb",
            borderwidth=1
        )
    )
    # Minimal grid lines
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='#f3f4f6',
        showline=True,
        linewidth=1,
        linecolor='#d1d5db'
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='#f3f4f6',
        showline=True,
        linewidth=1,
        linecolor='#d1d5db'
    )
    return fig
