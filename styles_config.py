import streamlit as st

def inject_custom_css():
    """
    Standard Streamlit styling. 
    No custom CSS injected to ensure maximum compatibility.
    """
    pass

def apply_chart_theme(fig):
    """
    Applies the standard Streamlit theme to charts.
    """
    fig.update_layout(template="streamlit")
    return fig
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
