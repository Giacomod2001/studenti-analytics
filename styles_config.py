# =============================================================================
# STYLES_CONFIG.PY - LinkedIn Premium Design System
# Version 2.0 - Unified with CareerMatch AI
# =============================================================================

import streamlit as st

def inject_custom_css():
    """
    Injects LinkedIn-inspired premium CSS with glassmorphism effects.
    """
    st.markdown("""
<style>
/* =============================================================================
   CSS VARIABLES - Design Tokens
   ============================================================================= */
:root {
    /* LinkedIn-Inspired Color Palette */
    --primary-blue: #0077B5;
    --primary-dark: #004471;
    --primary-light: #00A0DC;
    --accent-green: #00C853;
    --accent-amber: #FFB300;
    --accent-red: #E53935;
    
    /* Neutral Colors */
    --bg-dark: #0d1117;
    --bg-card: #161b22;
    --bg-elevated: #21262d;
    --text-primary: #f0f6fc;
    --text-secondary: #8b949e;
    --border-color: #30363d;
    
    /* Shadows */
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.15), 0 2px 4px rgba(0,0,0,0.12);
    --shadow-lg: 0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23);
    
    /* Transitions */
    --transition-fast: 0.15s ease;
    --transition-normal: 0.3s ease;
}

/* =============================================================================
   GLOBAL STYLES
   ============================================================================= */
   
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* =============================================================================
   TYPOGRAPHY
   ============================================================================= */
h1, h2, h3, h4, h5, h6 {
    font-weight: 600 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
}

h1 {
    background: linear-gradient(90deg, var(--primary-light), var(--primary-blue), var(--primary-dark));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* =============================================================================
   BUTTONS
   ============================================================================= */
   
.stButton > button {
    background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-md) !important;
    transition: all var(--transition-normal) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg), 0 0 20px rgba(0, 119, 181, 0.4) !important;
    background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary-blue) 100%) !important;
}

/* =============================================================================
   CARDS & CONTAINERS
   ============================================================================= */
   
.glass-card {
    background: rgba(22, 27, 34, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(48, 54, 61, 0.7);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: var(--shadow-md);
    transition: all var(--transition-normal);
    margin-bottom: 1.5rem;
}

.glass-card:hover {
    border-color: var(--primary-blue);
    box-shadow: var(--shadow-lg), 0 0 15px rgba(0, 119, 181, 0.2);
}

.metric-card {
    background: linear-gradient(135deg, rgba(0, 119, 181, 0.1) 0%, rgba(0, 68, 113, 0.1) 100%);
    border: 1px solid var(--primary-blue);
    border-radius: 16px;
    padding: 1.25rem;
    text-align: center;
    transition: all var(--transition-normal);
}

.metric-card:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px rgba(0, 119, 181, 0.3);
}

/* =============================================================================
   STATUS TAGS
   ============================================================================= */
   
.tag-critical {
    background: linear-gradient(135deg, #5c1f23 0%, #842029 100%);
    color: #ff8a8a;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 20px;
    margin: 4px;
    display: inline-block;
    border: 1px solid #8a3035;
}

.tag-monitor {
    background: linear-gradient(135deg, #5c4813 0%, #856404 100%);
    color: #ffd666;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 20px;
    margin: 4px;
    display: inline-block;
    border: 1px solid #7a5f10;
}

.tag-safe {
    background: linear-gradient(135deg, #1e4620 0%, #155724 100%);
    color: #75f083;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 20px;
    margin: 4px;
    display: inline-block;
    border: 1px solid #2d5a30;
}

/* =============================================================================
   SIDEBAR STYLING
   ============================================================================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--primary-light) !important;
    -webkit-text-fill-color: var(--primary-light) !important;
}

/* =============================================================================
   INPUT FIELDS
   ============================================================================= */
   
.stTextArea textarea,
.stTextInput input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-fast) !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: var(--primary-blue) !important;
    box-shadow: 0 0 0 3px rgba(0, 119, 181, 0.2) !important;
}

/* =============================================================================
   TABS STYLING
   ============================================================================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-elevated);
    padding: 6px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 12px !important;
    transition: all var(--transition-fast) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(0, 119, 181, 0.15) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--primary-blue) !important;
}

/* =============================================================================
   METRICS
   ============================================================================= */

[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, var(--primary-light), var(--primary-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* =============================================================================
   PROGRESS BAR
   ============================================================================= */

.stProgress > div > div {
    background: linear-gradient(90deg, var(--primary-blue), var(--primary-light), var(--accent-green)) !important;
    border-radius: 10px;
}

/* =============================================================================
   DIVIDER
   ============================================================================= */

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-color), var(--primary-blue), var(--border-color), transparent);
    margin: 3rem 0;
}

/* =============================================================================
   HERO SECTION
   ============================================================================= */

.hero-gradient {
    background: linear-gradient(135deg, 
        rgba(0, 119, 181, 0.15) 0%, 
        rgba(0, 68, 113, 0.1) 50%,
        rgba(0, 160, 220, 0.1) 100%
    );
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(0, 119, 181, 0.3);
    text-align: center;
}

/* =============================================================================
   REPORT BOX - Enhanced
   ============================================================================= */

.report-box {
    background: rgba(22, 27, 34, 0.9);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    box-shadow: var(--shadow-md);
}

.report-header {
    color: var(--primary-light);
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 8px;
}

/* =============================================================================
   ADA AI ASSISTANT UI
   ============================================================================= */

.sidebar-chat-container {
    margin-top: 2rem;
    padding: 1.5rem;
    background: rgba(0, 119, 181, 0.08);
    border: 1px solid rgba(0, 119, 181, 0.4);
    border-radius: 16px;
    box-shadow: inset 0 0 25px rgba(0, 119, 181, 0.08);
    margin-bottom: 1.5rem;
}

.sidebar-chat-header {
    font-size: 1rem;
    font-weight: 800;
    color: #00C9A7 !important;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(0, 201, 167, 0.2);
    padding-bottom: 8px;
}

.sidebar-chat-message {
    background: #1c2128;
    color: #f0f6fc;
    border: 1px solid #444c56;
    padding: 16px 22px;
    border-radius: 18px;
    font-size: 0.95rem;
    line-height: 1.6;
    margin-top: 1rem;
    box-shadow: var(--shadow-sm);
}

/* =============================================================================
   ANIMATIONS
   ============================================================================= */

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeInUp 0.5s ease-out forwards;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* =============================================================================
   RESPONSIVE DESIGN
   ============================================================================= */

@media (max-width: 768px) {
    .stButton > button {
        width: 100% !important;
    }
    .glass-card {
        padding: 1rem;
    }
}

/* =============================================================================
   ALERTS - Dark Theme
   ============================================================================= */
   
.stSuccess { background: rgba(46, 160, 67, 0.15) !important; border: 1px solid rgba(46, 160, 67, 0.4) !important; color: #7EE787 !important; }
.stInfo { background: rgba(56, 139, 253, 0.15) !important; border: 1px solid rgba(56, 139, 253, 0.4) !important; color: #79C0FF !important; }
.stWarning { background: rgba(187, 128, 9, 0.15) !important; border: 1px solid rgba(187, 128, 9, 0.4) !important; color: #E3B341 !important; }
.stError { background: rgba(248, 81, 73, 0.15) !important; border: 1px solid rgba(248, 81, 73, 0.4) !important; color: #FF7B72 !important; }

</style>
""", unsafe_allow_html=True)
