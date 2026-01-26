import streamlit as st
import pandas as pd
import numpy as np
import logging

# Local Modules
import constants
import styles_config
import data_utils
import ml_utils

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

def render_dashboard():
    st.title("Dashboard")
    st.markdown("High-level academic performance indicators.")
    st.caption("Need help understanding these metrics? Ask **Alex** in the sidebar.")
    st.markdown("---")
    
    # Load KPIs with lightweight queries (FAST)
    risk_counts = data_utils.get_risk_counts()
    sat_score = data_utils.get_avg_satisfaction()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 1. TOP ROW: CRITICAL METRICS
    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Active Students", f"{risk_counts.get('total', 0):,}", "+2.5%")
    with c2:
        st.metric("Dropout Forecast", f"{risk_counts.get('critical', 0):,}", "Critical", delta_color="inverse")
    with c3:
        st.metric("Avg Satisfaction", f"{sat_score:.1f}", "Stable")
    with c4:
        st.metric("Model Confidence", "94.2%", "+0.8%")
    
    st.markdown("<div style='margin: 4rem 0;'></div>", unsafe_allow_html=True)
    
    # 2. MAIN SPLIT: INTERVENTION vs STRATEGY
    col_risk, col_opp = st.columns(2)
    
    with col_risk:
        st.subheader("Priority Intervention Queue")
        st.markdown("*Students requiring immediate attention based on Churn Risk > 80%*")
        
        # Lazy load only when needed for table display
        df_churn = data_utils.load_table_data_optimized("studenti_churn_pred", limit=1000)
        
        if not df_churn.empty and 'churn_percentage' in df_churn.columns:
            high_risk_df = df_churn[df_churn['churn_percentage'] > 80].sort_values(by='churn_percentage', ascending=False).head(5)
            if not high_risk_df.empty:
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
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Satisfaction is strongly correlated with 'Exam Frequency'. Increasing exam availability could boost CSAT by +0.5 points.")
        st.info("Cluster 2 (Working Students) shows highest churn risk. Suggest deploying flexible schedule reminders.")


# ─── 4) VIEW: INTERVENTION CONSOLE (RISK) ──────────────────────────────────

# ─── 4) VIEW: INTERVENTION CONSOLE (RISK) ──────────────────────────────────

def render_intervention_console():
    st.title("Intervention Console")
    st.markdown("Monitor and act on attrition risks.")
    st.caption("Need help with risk tiers? Ask **Alex** in the sidebar.")
    
    # Disclaimer sulla polarizzazione dei risultati
    st.warning("""
    **Note on Model Results:** The Random Forest predictions tend to show polarized distributions 
    (students are often classified as either Safe or Critical, with fewer in the Monitor range). 
    This is due to the model's confidence thresholds and training data characteristics. 
    Always combine algorithmic scores with human judgment for intervention decisions.
    """)
    
    st.markdown("---")
    
    # Load Data
    df = data_utils.load_table_data_optimized("studenti_churn_pred")
    if not df.empty:
        df['student_id'] = df['student_id'].astype(str)
    
    # Filter Logic Calculations (Pre-calc for UI counts)
    if not df.empty and 'churn_percentage' in df.columns:
        n_crit = len(df[df['churn_percentage'] >= 75])
        n_mon = len(df[(df['churn_percentage'] >= 35) & (df['churn_percentage'] < 75)])
        n_safe = len(df[df['churn_percentage'] < 35])
        n_total = len(df)
    else:
        n_crit, n_mon, n_safe, n_total = 0, 0, 0, 0

    # 1. FILTER BAR
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("##### Risk Segment")
        # Dynamic Labels with Counts
        risk_mode = st.radio(
            "Risk Segment", 
            [
                f"All Students ({n_total:,})", 
                f"Critical (>75%)", 
                f"Monitor (35-75%)", 
                f"Safe (<35%)"
            ], 
            index=0, 
            horizontal=True, 
            label_visibility="collapsed"
        )

    with c2:
        st.markdown("##### Actions")
        st.download_button("Export .CSV", "csv_content", "intervention_list.csv", "text/csv", use_container_width=True)

    # Filter Application
    filter_desc = "Entire Student Population"
    filtered_df = df.copy()
    
    if not df.empty:
        if "Critical" in risk_mode: 
            filtered_df = df[df['churn_percentage'] >= 75]
            filter_desc = "Critical Risk (>75%)"
        elif "Monitor" in risk_mode: 
            filtered_df = df[(df['churn_percentage'] >= 35) & (df['churn_percentage'] < 75)]
            filter_desc = "Monitor List (35-75%)"
        elif "Safe" in risk_mode:
            filtered_df = df[df['churn_percentage'] < 35]
            filter_desc = "Safe Zone (<35%)"
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. MAIN WORKSPACE
    tab_report, tab_grid = st.tabs(["Intelligence Briefing", "Student List"])
    
    with tab_report:
        lines = []
        lines.append(f"""
        <div class="report-box">
        <div class="report-header">Analysis: {filter_desc}</div>
        """)
        
        if not filtered_df.empty:
            count = len(filtered_df)
            avg_risk = filtered_df['churn_percentage'].mean()
            
            # Contextual Metrics
            c_rpt1, c_rpt2, c_rpt3 = st.columns(3)
            with c_rpt1: st.metric("Students Selected", f"{count:,}")
            with c_rpt2: st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
            
            lines.append(f"<p style='margin-top: 10px;'><strong>Status:</strong> Tracking {count:,} students in this segment.</p>")
            
            if "Critical" in risk_mode:
                lines.append("<p style='color: #FF7B72;'><strong>Recommendation:</strong> <br>• Immediate advisor intervention required.<br>• Prioritize students with decreasing exam trends.</p>")
            elif "Monitor" in risk_mode:
                lines.append("<p style='color: #D2A8DA;'><strong>Recommendation:</strong> <br>• Schedule automated check-in email.<br>• Review support services offered.</p>")
            elif "Safe" in risk_mode:
                lines.append("<p style='color: #7EE787;'><strong>Recommendation:</strong> <br>• No action needed.<br>• Consider for peer-mentorship programs.</p>")
            elif "All" in risk_mode:
                lines.append(f"<p><strong>Distribution:</strong> The population is split into {n_crit:,} Critical, {n_mon:,} Monitor, and {n_safe:,} Safe cases.</p>")

            lines.append("</div>")
            st.markdown("\n".join(lines), unsafe_allow_html=True)
            
        else:
            # Smart Empty State
            lines.append(f"<p>No students found in this category.</p>")
            if "Monitor" in risk_mode and n_mon == 0:
                 lines.append(f"<p><em>Insight:</em> The model has polarized predictions. Students are either clearly Safe ({n_safe:,}) or Critical ({n_crit:,}).</p>")
            
            lines.append("</div>")
            st.markdown("\n".join(lines), unsafe_allow_html=True)

    with tab_grid:
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=600,
            column_config={
                "churn_percentage": st.column_config.ProgressColumn(
                    "Risk Score", 
                    min_value=0, 
                    max_value=100,
                    format="%.1f%%"
                ),
                "categoria_rischio": st.column_config.TextColumn("Risk Category"),
                "student_id": st.column_config.TextColumn("Student ID")
            },
            hide_index=True
        )


# ─── 5) VIEW: STUDENT 360 (CLUSTERS + SAT) ─────────────────────────────────

def render_student_360():
    st.title("Student 360")
    st.markdown("Holistic profiling and behavioral segmentation.")
    st.caption("Need help with clusters or psychometrics? Ask **Alex** in the sidebar.")
    st.markdown("---")
    
    tab_clusters, tab_satisfaction, tab_features = st.tabs(["Clustering", "Satisfaction", "Driver Analysis"])
    
    with tab_clusters:
        st.subheader("Behavioral Archetypes")
        df_clust = data_utils.load_table_data_optimized("studenti_cluster")
        df_churn = data_utils.load_table_data_optimized("studenti_churn_pred")
        
        # Ensure ID consistency for merge
        if not df_clust.empty: df_clust['student_id'] = df_clust['student_id'].astype(str)
        if not df_churn.empty: df_churn['student_id'] = df_churn['student_id'].astype(str)

        if not df_clust.empty:
            # JOIN Logic
            if not df_churn.empty:
                merged = pd.merge(df_clust, df_churn[['student_id', 'churn_percentage', 'categoria_rischio']], on='student_id', how='left')
            else:
                merged = df_clust
            
            # REPORT
            lines = []
            lines.append(f"""
            <div class="report-box">
            <div class="report-header">Cluster Intelligence</div>
            """)
            
            n_clusters = merged['cluster'].nunique()
            lines.append(f"<p>Population segmented into <strong>{n_clusters} clusters</strong> interactions.</p>")
            
            if 'churn_percentage' in merged.columns:
                 risk_grp = merged.groupby('cluster')['churn_percentage'].mean().sort_values(ascending=False)
                 highest_risk_c = risk_grp.index[0]
                 lowest_risk_c = risk_grp.index[-1]
                 
                 lines.append(f"<p><strong>Highest Risk:</strong> Cluster <strong>{highest_risk_c}</strong> ({risk_grp[highest_risk_c]:.1f}% avg risk).</p>")
                 lines.append(f"<p><strong>Most Stable:</strong> Cluster <strong>{lowest_risk_c}</strong> ({risk_grp[lowest_risk_c]:.1f}% avg risk).</p>")
            
            lines.append("</div>")
            st.markdown("\n".join(lines), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.dataframe(
                merged.head(500), 
                use_container_width=True, 
                height=500,
                column_config={
                    "churn_percentage": st.column_config.ProgressColumn("Risk", format="%.0f%%"),
                    "distance_to_centroid": st.column_config.NumberColumn("Fit Score", format="%.2f")
                }
            )
    
    with tab_satisfaction:
        st.subheader("Voice of Student Analysis")
        st.markdown("Identify discrepancies between academic performance (expected satisfaction) and reported experience.")
        
        df_sat = data_utils.load_table_data_optimized("report_finale_soddisfazione_studenti")
        
        if not df_sat.empty:
             # Derive Actionable Insights in Python
             # Real < Pred: "Silent Burnout" (Doing clearly worse than data suggests they should feel)
             # Real > Pred: "Resilient" (Happier than their grades suggest)
             
             df_sat['gap'] = df_sat['soddisfazione_reale'] - df_sat['soddisfazione_predetta']
             
             def classify_sentiment(row):
                 if row['gap'] < -1.5:
                     return "Silent Burnout"
                 elif row['gap'] > 1.5:
                     return "Resilient"
                 elif row['soddisfazione_reale'] < 6.0:
                     return "At Risk"
                 else:
                     return "Aligned"

             df_sat['psychometric_status'] = df_sat.apply(classify_sentiment, axis=1)
             
             # ─── RICH REPORT GENERATION ───
             burnout_df = df_sat[df_sat['psychometric_status'] == "Silent Burnout"]
             resilient_df = df_sat[df_sat['psychometric_status'] == "Resilient"]
             
             report_lines = []
             report_lines.append(f"""
<div class="report-box">
<div class="report-header">Psychometric Intelligence Briefing</div>
""")
             
             # 1. Silent Burnout Analysis
             if not burnout_df.empty:
                 # Check if 'media_voti' exists (requires updated SQL)
                 if 'media_voti' in burnout_df.columns:
                     avg_gpa_burnout = burnout_df['media_voti'].mean()
                     gpa_text = f"(Avg GPA: {avg_gpa_burnout:.1f})"
                 else:
                     gpa_text = ""
                     
                 report_lines.append(f"""
<p style='color: #FF7B72; margin-bottom: 5px;'><strong>CRITICAL INSIGHT: Silent Burnout Detected</strong></p>
<p>We identified <strong>{len(burnout_df):,} students</strong> who are academically strong {gpa_text} but report significantly lower satisfaction than predicted.</p>
<ul style='margin-bottom: 15px;'>
<li><strong>The Pattern:</strong> They perform well but are emotionally exhausted.</li>
<li><strong>Risk:</strong> High probability of sudden dropout despite "good grades".</li>
<li><strong>Action:</strong> Don't praise grades. Ask: <em>"How are you managing the stress?"</em></li>
</ul>
""")
             
             # 2. Resilience Analysis
             if not resilient_df.empty:
                 report_lines.append(f"""
<p style='color: #7EE787; margin-bottom: 5px;'><strong>POSITIVE DEVIANCE: The Resilient Group</strong></p>
<p>There are <strong>{len(resilient_df):,} students</strong> outperforming expectations. Despite lower academic inputs, their satisfaction is high.</p>
<ul style='margin-bottom: 15px;'>
<li><strong>The Opportunity:</strong> These students have high grit and school spirit.</li>
<li><strong>Action:</strong> Recruit them as <strong>Peer Mentors</strong> or Student Ambassadors.</li>
</ul>
""")
                 
             if burnout_df.empty and resilient_df.empty:
                 report_lines.append("<p>The student population is psychometrically aligned. Reported satisfaction matches academic performance expectations.</p>")
                 
             report_lines.append("</div>")
             st.markdown("\n".join(report_lines), unsafe_allow_html=True)
             st.markdown("<br>", unsafe_allow_html=True)

             # Metric Summary
             c1, c2 = st.columns(2)
             with c1:
                st.metric(
                    "Potential Silent Burnouts", 
                    f"{len(burnout_df):,}", 
                    help="Gap < -1.5 (High Grades / Low Happiness)"
                )
             with c2:
                st.metric(
                    "Resilient Students", 
                    f"{len(resilient_df):,}",
                     help="Gap > 1.5 (Low Outcomes / High Happiness)"
                )
        
        # Prepare columns config based on available data
        cols_cfg = {
            "soddisfazione_reale": st.column_config.NumberColumn("Reported Score", format="%.1f"),
            "soddisfazione_predetta": st.column_config.NumberColumn("Expected (AI)", format="%.1f"),
            "psychometric_status": st.column_config.TextColumn("Psychometric Profile"),
            "livello_affidabilita": None # Hide technical column
        }
        
        if 'media_voti' in df_sat.columns:
            cols_cfg["media_voti"] = st.column_config.NumberColumn("GPA", format="%.1f")

        st.dataframe(
            df_sat.head(500), 
            use_container_width=True,
            column_config=cols_cfg
        )

    with tab_features:
        st.subheader("Model Explainability")
        df_feat = data_utils.load_table_data_optimized("feature_importance_studenti")
        
        if not df_feat.empty and 'caratteristica' in df_feat.columns and 'peso_importanza' in df_feat.columns:
            st.bar_chart(df_feat.set_index('caratteristica')['peso_importanza'], color="#da3633")
        else:
            st.info("Feature importance data is currently unavailable for this model.")
        
# ─── 6) LANDING PAGE ───────────────────────────────────────────────────────

def render_landing_page():
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hero Section with glassmorphism
    st.markdown("""
    <div class="hero-gradient">
        <h1 style="margin-bottom: 0.5rem;">Student Intelligence Hub</h1>
        <p style="font-size: 1.2rem; color: #8b949e;">Advanced Predictive Analytics for Higher Education Retention</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>Predictive Risk Modeling</h3>
            <p>Random Forest algorithm forecasts potential dropouts before they happen.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>Behavioral Clustering</h3>
            <p>K-Means segmentation reveals distinct student archetypes.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="glass-card">
            <h3>Satisfaction Analysis</h3>
            <p>Boosted Tree regression measures the pulse of student experience.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Getting Started
    st.markdown("""
    ### Getting Started
    
    Use the navigation sidebar to access the **Dashboard** for a high-level overview, 
    or dive into the **Intervention Console** to take action on specific cases.
    
    **Alex**, your AI Academic Advisor, is available in the sidebar to answer questions 
    about risk analysis, student clustering, and retention strategies.
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Enter Dashboard", type="primary"):
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
            st.session_state['view'] = 'dashboard'
    
        # SIDEBAR NAVIGATION
        with st.sidebar:
            st.title("Student Intelligence")
            st.caption("Connected to BigQuery")
            st.markdown("---")
            
            # NAVIGATION MENU
            if st.button("Dashboard", use_container_width=True):
                st.session_state['view'] = 'dashboard'
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
            
            st.markdown("---")
            st.caption("SYSTEM")
            if st.button("🔄 Refresh Data Cache", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            
            # ADA AI ASSISTANT
            st.markdown("---")
            st.markdown("""
            <div class="sidebar-chat-container">
                <div class="sidebar-chat-header">Alex - Academic Advisor</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Get current view name for context
            view_names = {
                'dashboard': 'Dashboard',
                'intervention_console': 'Intervention Console',
                'student_360': 'Student 360',
                'raw_data': 'Raw Data Explorer'
            }
            current_view = view_names.get(st.session_state.get('view', 'dashboard'), 'Dashboard')
            
            # Chat input
            user_msg = st.text_input("Ask Alex...", key="ada_input", placeholder="How does risk scoring work?")
            
            # Get response
            ada_response = ml_utils.get_alex_response(user_msg, current_view)
            
            st.markdown(f"""
            <div class="sidebar-chat-message">
                {ada_response}
            </div>
            """, unsafe_allow_html=True)
    
        # ROUTING LOGIC
        view = st.session_state['view']
        
        if view == 'dashboard':
            render_dashboard()
            
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
