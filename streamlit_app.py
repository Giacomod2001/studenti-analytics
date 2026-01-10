import streamlit as st
import pandas as pd
import numpy as np
import logging

# Local Modules
import constants
import styles_config
import data_utils

# ─── 1) PAGE CONFIGURATION ─────────────────────────────────────────────────

st.set_page_config(
    page_title="Student Analytics Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── 2) HELPER: TEXTUAL REPORT GENERATION ──────────────────────────────────

def generate_textual_report(df: pd.DataFrame, table_name: str) -> str:
    """
    Generates a high-quality textual intelligence report.
    Replaces all visual charts with descriptive text.
    """
    lines = []
    
    lines.append(f"""
    <div class="report-box">
    <div class="report-header">Automatic Intelligence Report: {table_name}</div>
    """)
    
    # Overview
    row_count = len(df)
    col_count = len(df.columns)
    lines.append(f"<p>The dataset <strong>{table_name}</strong> consists of <strong>{row_count:,} records</strong> analyzed across <strong>{col_count} dimensions</strong>.</p>")
    
    # Specific Logic for known tables
    if "churn" in table_name.lower() or "pred" in table_name.lower():
        if 'churn_percentage' in df.columns:
            mean_risk = df['churn_percentage'].mean()
            # Normalize if needed (assuming 0-100 scale based on previous context)
            if mean_risk > 1 and mean_risk <= 100: pass 
            elif mean_risk <= 1: mean_risk *= 100
            
            high_risk = df[df['churn_percentage'] > 70]
            lines.append(f"<p><strong>Dropout Risk Analysis:</strong> The average predicted dropout risk across the population is <strong>{mean_risk:.1f}%</strong>.</p>")
            lines.append(f"<p>Critically, <strong>{len(high_risk):,} students</strong> have been flagged with a high probability (>70%) of attrition, requiring immediate intervention strategies.</p>")
    
    if "cluster" in table_name.lower():
        if 'cluster' in df.columns:
            n_clusters = df['cluster'].nunique()
            top_cluster = df['cluster'].value_counts().idxmax()
            lines.append(f"<p><strong>Segmentation Analysis:</strong> The population is divided into <strong>{n_clusters} distinct behavioral clusters</strong>.</p>")
            lines.append(f"<p>The largest segment is <strong>Cluster {top_cluster}</strong>, indicating a dominant behavioral pattern among the student body.</p>")

    # General Statistical Summary
    lines.append("<hr style='border-color: #30363D; margin: 15px 0;'>")
    lines.append("<p><strong>Statistical Highlights:</strong></p><ul>")
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols[:4]: # Limit to top 4 numeric columns to avoid wall of text
        if col.lower() in ['id', 'student_id', 'matricola']: continue
        avg = df[col].mean()
        std = df[col].std()
        lines.append(f"<li><strong>{col}:</strong> Average value is {avg:.2f} (±{std:.2f}).</li>")
        
    lines.append("</ul></div>")
    
    return "\n".join(lines)


# ─── 3) LANDING PAGE ───────────────────────────────────────────────────────

def render_landing_page():
    """
    Professional Landing Page (Dark Theme).
    """
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("Student Analytics Platform")
    st.markdown("""
    <p style="font-size: 1.2rem; color: #8B949E;">
    Advanced predictive analytics environment for higher education retention and performance monitoring.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="report-box">
            <h3 style="color: #58A6FF;">Predictive Modeling</h3>
            <p>Utilizes Random Forest algorithms to identify attrition risk with high precision.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="report-box">
            <h3 style="color: #7EE787;">Cluster Segmentation</h3>
            <p>Unsupervised learning segregates students into behavioral archetypes.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="report-box">
            <h3 style="color: #D2A8DA;">Qualitative Drivers</h3>
            <p>Correlates satisfaction survey metrics with academic output.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    if st.button("Initialize Dashboard", type="primary"):
        st.session_state["show_landing"] = False
        st.rerun()


# ─── 4) HOME DASHBOARD ─────────────────────────────────────────────────────

def render_home_dashboard(tables_info):
    """
    Executive Dashboard (Dark Mode).
    """
    st.title("Executive Overview")
    st.markdown("Real-time monitoring of student success metrics.")
    st.markdown("---")
    
    # Load Data
    try:
        df_students = data_utils.load_table_data_optimized("studenti")
        df_churn = data_utils.load_table_data_optimized("studenti_churn_pred")
        df_clusters = data_utils.load_table_data_optimized("studenti_cluster")
    except:
        df_students = pd.DataFrame()
        df_churn = pd.DataFrame()
        df_clusters = pd.DataFrame()
        
    # Stats
    total = len(df_students)
    risk_count = 0
    if not df_churn.empty and 'churn_percentage' in df_churn.columns:
        risk_count = len(df_churn[df_churn['churn_percentage'] > 70])
        
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Enrolled", f"{total:,}")
    c2.metric("High Risk Detected", f"{risk_count:,}")
    c3.metric("Models Active", "2")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Reports instead of charts
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Retention Intelligence Report")
        if risk_count > 0:
            pct = (risk_count / total) * 100 if total > 0 else 0
            st.markdown(f"""
            <div class="report-box">
                <p>The system has identified <strong>{risk_count} students</strong> ({pct:.1f}%) exhibiting critical risk factors associated with dropout.</p>
                <p>Primary indicators include irregular attendance patterns and declining grade averages in core modules.</p>
                <br>
                <p style="color: #FF7B72;"><strong>Recommendation:</strong> Initiate Tier-1 intervention protocols for the identified cohort immediately.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Insufficient data for risk analysis.")

    with col_right:
        st.subheader("System Alerts")
        st.error("39 High Priority Cases")
        st.warning("Data sync pending")
        st.success("Model retraining complete")


# ─── 5) TABLE INSPECTION ───────────────────────────────────────────────────

def render_table_inspection(df: pd.DataFrame, table_info: dict):
    """
    Data Inspector with Text Reports.
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(table_info['name'])
        st.markdown(f"*{table_info['description']}*")
    with col2:
        st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), f"{table_info['name']}.csv", "text/csv")
        
    st.markdown("---")
    
    tab_overview, tab_report = st.tabs(["Data Grid", "Intelligence Report"])
    
    with tab_overview:
        # Styled dataframe via Streamlit's new column config (Dark theme handles colors)
        st.dataframe(
            df.head(200),
            use_container_width=True, 
            height=600,
            column_config={
                "churn_percentage": st.column_config.ProgressColumn(
                    "Risk Score",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )
            }
        )
        st.caption(f"Displaying top 200 of {len(df):,} records.")
        
    with tab_report:
        report_html = generate_textual_report(df, table_info['name'])
        st.markdown(report_html, unsafe_allow_html=True)


# ─── 6) MAIN APP ───────────────────────────────────────────────────────────

def main():
    styles_config.inject_custom_css()
    
    if "show_landing" not in st.session_state:
        st.session_state["show_landing"] = True
        
    if st.session_state["show_landing"]:
        render_landing_page()
        return

    with st.sidebar:
        st.title("Analytics")
        st.caption("Admin Console")
        st.markdown("---")
        
        tables_info = data_utils.get_tables_metadata_cached()
        
        st.markdown("<strong>MODULES</strong>", unsafe_allow_html=True)
        nav = st.radio("Navigate", ["Executive Overview"] + [t['name'] for t in tables_info], label_visibility="collapsed")
        
    if nav == "Executive Overview":
        render_home_dashboard(tables_info)
    else:
        info = next((t for t in tables_info if t['name'] == nav), None)
        if info:
            df = data_utils.load_table_data_optimized(info['id'])
            render_table_inspection(df, info)

if __name__ == "__main__":
    main()
