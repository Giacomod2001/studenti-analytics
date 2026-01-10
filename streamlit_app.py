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

    # ... (inside generate_smart_report) ...

    # 1. TOP ROW: CRITICAL METRICS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Students", "50,000", "+2.5%")
    c2.metric("Dropout Forecast", f"{risk_n:,}", "Critical", delta_color="inverse")
    c3.metric("Avg Satisfaction", f"{sat_score:.1f}", "Stable")
    c4.metric("Model Confidence", "94.2%", "+0.8%")
    
    # ... (inside render_control_tower) ...
    
    with col_risk:
        st.subheader("Priority Intervention Queue")
        
    # ...

    with col_opp:
        st.subheader("Strategic Insights")

    # ... (inside render_intervention_console) ...

    # 3. AI REPORT
    st.markdown(generate_smart_report(df, "Retention Risk"), unsafe_allow_html=True)
        
    with tab_grid:
        st.dataframe(
            df,
            # ...
                "churn_percentage": st.column_config.ProgressColumn(
                    "Risk Probability", 
                    min_value=0, 
                    max_value=100,
                    format="%.1f%%"
                ),
            # ...
        )

    # ... (inside render_student_360) ...
            column_config={
                "soddisfazione_predetta": st.column_config.NumberColumn("Predicted Score", format="%.1f / 10")
            }

    # ... (inside main sidebar) ...
    
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
                # USER REQUEST: Data Grid AFTER Report? Here we just have grid, but if report avail, report first.
                # Since raw data has minimal report logic, we keep grid.
                df = data_utils.load_table_data_optimized(t['id'])
                st.dataframe(df.head(200), use_container_width=True)
                st.caption(f"Showing first 200 rows of {t['id']}")

if __name__ == "__main__":
    main()
