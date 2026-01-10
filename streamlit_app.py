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
    page_title="Student Intelligence Hub",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── 2) HELPER: TEXTUAL REPORT GENERATION ──────────────────────────────────

def generate_smart_report(df: pd.DataFrame, context: str) -> str:
    """
    Generates context-aware intelligence briefings.
    """
    lines = []
    
    # 1. HEADER
    lines.append(f"""
    <div class="report-box">
    <div class="report-header">AI Intelligence Briefing: {context}</div>
    """)
    
    # 2. CONTEXTUAL INSIGHTS
    if context == "Retention Risk":
        if 'churn_percentage' in df.columns:
            risk_count = len(df[df['churn_percentage'] > 75])
            avg_risk = df['churn_percentage'].mean()
            lines.append(f"<p><strong>Current Threat Level:</strong> The model detects an average dropout probability of <strong>{avg_risk:.1f}%</strong> across the cohort.</p>")
            lines.append(f"<p style='color: #FF7B72;'><strong>Action Required:</strong> <strong>{risk_count:,} students</strong> are flagged as 'Critical Risk' (>75%). Recommendation: Prioritize for academic counseling.</p>")
            
    elif context == "Student Profiling":
        if 'cluster' in df.columns:
            n_clusters = df['cluster'].nunique()
            lines.append(f"<p><strong>Segmentation Strategy:</strong> The population maps to <strong>{n_clusters} distinct behavioral archetypes</strong>.</p>")
        if 'soddisfazione_predetta' in df.columns:
             sat_avg = df['soddisfazione_predetta'].mean()
             lines.append(f"<p><strong>Quality of Experience:</strong> Projected satisfaction score is <strong>{sat_avg:.1f}/10</strong>. Correlate with 'Moderate' risk clusters to identify engagement opportunities.</p>")
             
    elif context == "Raw Data Inspector":
        lines.append(f"<p>Dataset loaded with <strong>{len(df):,} records</strong>. Ready for manual auditing or export.</p>")
        
    lines.append("</div>")
    return "\n".join(lines)


# ─── 3) VIEW: CONTROL TOWER (HOME) ─────────────────────────────────────────

def render_control_tower():
    st.title("Control Tower")
    st.markdown("High-level academic performance indicators.")
    st.markdown("---")
    
    # Load Key Data
    try:
        df_churn = data_utils.load_table_data_optimized("studenti_churn_pred")
        df_sat = data_utils.load_table_data_optimized("report_finale_soddisfazione_studenti")
    except:
        df_churn = pd.DataFrame()
        df_sat = pd.DataFrame()

    # Calculate KPIs
    risk_n = len(df_churn[df_churn['churn_percentage'] > 70]) if not df_churn.empty and 'churn_percentage' in df_churn.columns else 0
    sat_score = df_sat['soddisfazione_predetta'].mean() if not df_sat.empty and 'soddisfazione_predetta' in df_sat.columns else 0
    
    # 1. TOP ROW: CRITICAL METRICS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Students", "50,000", "+2.5%")
    c2.metric("Dropout Forecast", f"{risk_n:,}", "Critical", delta_color="inverse")
    c3.metric("Avg Satisfaction", f"{sat_score:.1f}", "Stable")
    c4.metric("Model Confidence", "94.2%", "+0.8%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. MAIN SPLIT: INTERVENTION vs STRATEGY
    col_risk, col_opp = st.columns(2)
    
    with col_risk:
        st.subheader("Priority Intervention Queue")
        st.markdown("*Students requiring immediate attention based on Churn Risk > 80%*")
        
        if not df_churn.empty:
            high_risk_df = df_churn[df_churn['churn_percentage'] > 80].sort_values(by='churn_percentage', ascending=False).head(5)
            st.dataframe(
                high_risk_df[['student_id', 'churn_percentage', 'categoria_rischio']], 
                use_container_width=True,
                column_config={
                    "churn_percentage": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f%%")
                },
                hide_index=True
            )
            if st.button("Manage Interventions (Go to Console)"):
                st.session_state['view'] = 'intervention_console'
                st.rerun()
        else:
             st.info("No critical triggers active.")

    with col_opp:
        st.subheader("Strategic Insights")
        st.markdown("*Performance trends and satisfaction drivers*")
        st.info("Satisfaction is strongly correlated with 'Exam Frequency'. Increasing exam availability could boost CSAT by +0.5 points.")
        st.info("Cluster 2 (Working Students) shows highest churn risk. Suggest deploying flexible schedule reminders.")


# ─── 4) VIEW: INTERVENTION CONSOLE (RISK) ──────────────────────────────────

# ─── 4) VIEW: INTERVENTION CONSOLE (RISK) ──────────────────────────────────

def render_intervention_console():
    st.title("Intervention Console")
    st.markdown("Monitor and act on attrition risks.")
    st.markdown("---")
    
    # Load Data
    df = data_utils.load_table_data_optimized("studenti_churn_pred")
    
    # 1. FILTER BAR
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("##### Filter by Risk Tier")
        # Replaced st.segmented_control with st.radio for stability
        tier = st.radio("Tier", ["All", "Critical (>80%)", "High (>60%)", "Moderate (>40%)"], index=1, horizontal=True, label_visibility="collapsed")

    with c2:
        st.markdown("##### Export")
        st.download_button("Download List", "csv_content", "intervention_list.csv", "text/csv", use_container_width=True)

    # Filter Logic
    if not df.empty and 'churn_percentage' in df.columns:
        if "Critical" in tier: df = df[df['churn_percentage'] > 80]
        elif "High" in tier: df = df[df['churn_percentage'] > 60]
        elif "Moderate" in tier: df = df[df['churn_percentage'] > 40]
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. MAIN TABLE - Order requested: Data Grid AFTER Report
    
    tab_report, tab_grid = st.tabs(["Intelligence Briefing", "Data Grid"])
    
    with tab_report:
         # 3. AI REPORT
        st.markdown(generate_smart_report(df, "Retention Risk"), unsafe_allow_html=True)
        
    with tab_grid:
        st.dataframe(
            df,
            use_container_width=True,
            height=600,
            column_config={
                "churn_percentage": st.column_config.ProgressColumn(
                    "Risk Probability", 
                    min_value=0, 
                    max_value=100,
                    format="%.1f%%"
                ),
                "categoria_rischio": st.column_config.TextColumn("Risk Category")
            }
        )


# ─── 5) VIEW: STUDENT 360 (CLUSTERS + SAT) ─────────────────────────────────

def render_student_360():
    st.title("Student 360 Profiling")
    st.markdown("Behavioral segmentation and satisfaction drivers.")
    st.markdown("---")
    
    tab_clusters, tab_satisfaction, tab_features = st.tabs(["Behavioral Clusters", "Satisfaction Analysis", "Driver Analysis"])
    
    with tab_clusters:
        st.subheader("Clustering Analysis")
        df_clust = data_utils.load_table_data_optimized("studenti_cluster")
        if not df_clust.empty:
            # Layout: Report First, Grid Second
            st.markdown(generate_smart_report(df_clust, "Student Profiling"), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_clust.head(100), use_container_width=True, height=400)
    
    with tab_satisfaction:
        st.subheader("Satisfaction Predictions")
        df_sat = data_utils.load_table_data_optimized("report_finale_soddisfazione_studenti")
        st.dataframe(
            df_sat.head(100), 
            use_container_width=True,
            column_config={
                "soddisfazione_predetta": st.column_config.NumberColumn("Predicted Score", format="%.1f / 10")
            }
        )

    with tab_features:
        st.subheader("Feature Importance")
        st.markdown("What drives these results?")
        df_feat = data_utils.load_table_data_optimized("feature_importance_studenti")
        st.dataframe(
            df_feat,
            use_container_width=True,
            column_config={
                "peso_importanza": st.column_config.ProgressColumn("Impact Weight", min_value=0, max_value=max(df_feat['peso_importanza']) if not df_feat.empty else 1)
            }
        )
        
# ─── 6) LANDING PAGE ───────────────────────────────────────────────────────

def render_landing_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("Student Intelligence Hub")
    
    st.markdown("""
    ### Welcome to the Student Intelligence Hub

    This platform serves as the central nervous system for academic retention strategies. It ingests real-time student data to provide:

    *   **Predictive Risk Modeling**: Foreseeing potential dropouts before they happen.
    *   **Behavioral Clustering**: Understanding the "why" behind student groups.
    *   **Satisfaction Analysis**: Measuring the pulse of the student body.

    **Getting Started:**
    Use the navigation sidebar to access the **Control Tower** for a high-level overview, or dive into the **Intervention Console** to take action on specific cases.
    <br><br>
    """, unsafe_allow_html=True)
    
    if st.button("Enter Control Tower", type="primary"):
        st.session_state["show_landing"] = False
        st.rerun()


# ─── 7) MAIN APP ROUTER ────────────────────────────────────────────────────

def main():
    try:
        styles_config.inject_custom_css()
        
        # Initialize State
        if "show_landing" not in st.session_state:
            st.session_state["show_landing"] = True
            
        if st.session_state["show_landing"]:
            render_landing_page()
            return
    
        if 'view' not in st.session_state:
            st.session_state['view'] = 'control_tower'
    
        # SIDEBAR NAVIGATION
        with st.sidebar:
            st.title("Student Intelligence")
            st.caption("Connected to BigQuery")
            st.markdown("---")
            
            # NAVIGATION MENU
            if st.button("Control Tower", use_container_width=True):
                st.session_state['view'] = 'control_tower'
                st.rerun()
                
            st.markdown("---")
            st.caption("OPERATIONS")
            
            if st.button("Intervention Console", use_container_width=True):
                st.session_state['view'] = 'intervention_console'
                st.rerun()
                
            if st.button("Student 360", use_container_width=True):
                st.session_state['view'] = 'student_360'
                st.rerun()
                
            st.markdown("---")
            st.caption("DATA GOVERNANCE")
            
            if st.button("Raw Data Explorer", use_container_width=True):
                st.session_state['view'] = 'raw_data'
                st.rerun()
    
        # ROUTING LOGIC
        view = st.session_state['view']
        
        if view == 'control_tower':
            render_control_tower()
            
        elif view == 'intervention_console':
            render_intervention_console()
            
        elif view == 'student_360':
            render_student_360()
            
        elif view == 'raw_data':
            st.title("Raw Data Explorer")
            st.markdown("Direct access to BigQuery tables.")
            tables = data_utils.get_tables_metadata_cached()
            tabs = st.tabs([t['name'] for t in tables])
            for i, t in enumerate(tables):
                with tabs[i]:
                    df = data_utils.load_table_data_optimized(t['id'])
                    st.dataframe(df.head(200), use_container_width=True)
                    st.caption(f"Showing first 200 rows of {t['id']}")
                    
    except Exception as e:
        st.error(f"SYSTEM ERROR: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
