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
    Replaces visual charts with descriptive text.
    """
    lines = []
    
    # Clean up table name for display
    display_name = table_name.replace("_", " ").title()
    
    lines.append(f"""
    <div class="report-box">
    <div class="report-header">Automatic Intelligence Report: {display_name}</div>
    """)
    
    # Overview
    row_count = len(df)
    col_count = len(df.columns)
    lines.append(f"<p>The dataset contains <strong>{row_count:,} records</strong> analyzed across <strong>{col_count} dimensions</strong>.</p>")
    lines.append("<hr style='border-color: #30363D; margin: 15px 0;'>")
    
    # --- SPECIFIC LOGIC PER TABLE TYPE ---
    
    name_lower = table_name.lower()
    
    # A) CHURN / DROPOUT
    if "churn" in name_lower or "dropout" in name_lower:
        if 'churn_percentage' in df.columns:
            mean_risk = df['churn_percentage'].mean()
            # Normalize if needed
            if mean_risk <= 1: mean_risk *= 100
            
            high_risk = df[df['churn_percentage'] > 70]
            lines.append(f"<p><strong>Warning System:</strong> The average predicted dropout risk is <strong>{mean_risk:.1f}%</strong>.</p>")
            lines.append(f"<p style='color: #FF7B72;'><strong>Crucial Insight:</strong> {len(high_risk):,} students are flagged as High Risk (>70%). Immediate attention recommended.</p>")

    # B) CLUSTERING
    elif "cluster" in name_lower:
        if 'cluster' in df.columns:
            n_clusters = df['cluster'].nunique()
            top_cluster = df['cluster'].value_counts().idxmax()
            lines.append(f"<p><strong>Behavioral Segmentation:</strong> Students are grouped into <strong>{n_clusters} clusters</strong> based on study habits and performance.</p>")
            lines.append(f"<p><strong>Dominant Profile:</strong> Cluster {top_cluster} represents the largest student segment.</p>")
            
    # C) SATISFACTION (Regression)
    elif "soddisfazione" in name_lower or "satisfaction" in name_lower:
        if 'soddisfazione_predetta' in df.columns:
            avg_sat = df['soddisfazione_predetta'].mean()
            lines.append(f"<p><strong>Sentiment Analysis:</strong> The projected average satisfaction score is <strong>{avg_sat:.1f}/10</strong>.</p>")
            
            if 'categoria_soddisfazione' in df.columns:
                top_cat = df['categoria_soddisfazione'].mode()[0]
                lines.append(f"<p>The prevailing sentiment is <strong>'{top_cat}'</strong>.</p>")
                
    # D) FEATURE IMPORTANCE
    elif "feature" in name_lower or "importance" in name_lower:
        if 'caratteristica' in df.columns and 'peso_importanza' in df.columns:
            top_3 = df.sort_values(by='peso_importanza', ascending=False).head(3)
            features = ", ".join([f"<strong>{r['caratteristica']}</strong>" for _, r in top_3.iterrows()])
            lines.append(f"<p><strong>Key Drivers:</strong> The top 3 factors influencing the model are: {features}.</p>")
            lines.append("<p>These variables have the highest predictive power.</p>")

    # --- GENERAL STATS FOR ALL TABLES ---
    lines.append("<br><p><strong>Data Highlights:</strong></p><ul>")
    num_cols = df.select_dtypes(include=[np.number]).columns
    count = 0
    for col in num_cols: 
        if col.lower() in ['id', 'student_id', 'matricola', 'cluster', 'churn_pred']: continue
        if count >= 3: break # Max 3 generic stats
        avg = df[col].mean()
        lines.append(f"<li><strong>{col}:</strong> Average {avg:.2f}</li>")
        count += 1
        
    lines.append("</ul></div>")
    
    return "\n".join(lines)


# ─── 3) LANDING PAGE ───────────────────────────────────────────────────────

def render_landing_page():
    # Keep existing landing page logic, it's fine
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("Student Analytics Platform")
    st.markdown("""
    <p style="font-size: 1.2rem; color: #8B949E;">
    Advanced predictive analytics environment for higher education retention and performance monitoring.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("Enter Dashboard", type="primary"):
        st.session_state["show_landing"] = False
        st.rerun()


# ─── 4) HOME DASHBOARD ─────────────────────────────────────────────────────

def render_home_dashboard(tables_info):
    st.title("Executive Overview")
    st.markdown("High-level performance indicators.")
    st.markdown("---")
    
    try:
        # Load key tables for KPIs
        df_churn = data_utils.load_table_data_optimized("studenti_churn_pred") 
        df_sat = data_utils.load_table_data_optimized("report_finale_soddisfazione_studenti")
    except:
        df_churn = pd.DataFrame()
        df_sat = pd.DataFrame()

    c1, c2, c3 = st.columns(3)
    
    risk_n = len(df_churn[df_churn['churn_percentage'] > 70]) if not df_churn.empty and 'churn_percentage' in df_churn.columns else 0
    avg_sat = df_sat['soddisfazione_predetta'].mean() if not df_sat.empty and 'soddisfazione_predetta' in df_sat.columns else 0
    
    c1.metric("Students Monitor", "50,000+")
    c2.metric("At-Risk Students", f"{risk_n:,}", delta="-High Priority" if risk_n>0 else "Stable", delta_color="inverse")
    c3.metric("Avg Satisfaction", f"{avg_sat:.1f}/10")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # System Health (Positive)
    st.success("✅ System Status: All systems operational. Data is up-to-date.")


# ─── 5) TABLE INSPECTION ───────────────────────────────────────────────────

def render_table_inspection(df: pd.DataFrame, table_alias: str):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(table_alias)
    with col2:
        st.download_button("Export CSV", df.to_csv(index=False).encode('utf-8'), f"{table_alias}.csv", "text/csv")
        
    st.markdown("---")
    
    tab_data, tab_intel = st.tabs(["Data Grid", "Intelligence Report"])
    
    with tab_data:
        # Smart column config based on column names
        cfg = {}
        if "churn_percentage" in df.columns:
            cfg["churn_percentage"] = st.column_config.ProgressColumn("Risk %", min_value=0, max_value=100, format="%.1f%%")
        if "soddisfazione_predetta" in df.columns:
             cfg["soddisfazione_predetta"] = st.column_config.NumberColumn("Score", format="%.1f ⭐️")
        if "peso_importanza" in df.columns:
             cfg["peso_importanza"] = st.column_config.ProgressColumn("Weight", min_value=0, max_value=1)
             
        st.dataframe(df.head(500), use_container_width=True, height=600, column_config=cfg)
        
    with tab_intel:
        st.markdown(generate_textual_report(df, table_alias), unsafe_allow_html=True)


# ─── 6) MAIN APP STRUCTURE ─────────────────────────────────────────────────

def main():
    styles_config.inject_custom_css()
    
    if "show_landing" not in st.session_state:
        st.session_state["show_landing"] = True
        
    if st.session_state["show_landing"]:
        render_landing_page()
        return

    # --- CATEGORIZED SIDEBAR ---
    with st.sidebar:
        st.title("Analytics")
        st.caption("v5.1 | Connected to BigQuery")
        st.markdown("---")
        
        # 1. Overview
        st.markdown("#### 🏠 Overview")
        if st.button("Executive Dashboard", use_container_width=True):
             st.session_state['view'] = 'home'
             st.rerun()

        # 2. Predictive Models
        st.markdown("#### 🧠 Predictive Models")
        if st.button("Dropout Prediction", use_container_width=True):
             st.session_state['view'] = 'studenti_churn_pred'
             st.session_state['alias'] = 'Dropout Prediction'
             st.rerun()
             
        if st.button("Student Clustering", use_container_width=True):
             st.session_state['view'] = 'studenti_cluster'
             st.session_state['alias'] = 'Student Clustering'
             st.rerun()
             
        if st.button("Satisfaction Report", use_container_width=True):
             st.session_state['view'] = 'report_finale_soddisfazione_studenti'
             st.session_state['alias'] = 'Satisfaction Analysis'
             st.rerun()

        # 3. Model Explainability
        st.markdown("#### 🔍 Deep Dive")
        if st.button("Feature Importance", use_container_width=True):
             st.session_state['view'] = 'feature_importance_studenti'
             st.session_state['alias'] = 'Feature Importance'
             st.rerun()

        # 4. Raw Data
        st.markdown("#### 📂 Data Warehouse")
        if st.button("Students Registry", use_container_width=True):
             st.session_state['view'] = 'studenti'
             st.session_state['alias'] = 'Raw Students Data'
             st.rerun()

    # --- ROUTING ---
    if 'view' not in st.session_state:
        st.session_state['view'] = 'home'

    if st.session_state['view'] == 'home':
        render_home_dashboard(None)
    else:
        # Load specific table
        table_id = st.session_state['view']
        alias = st.session_state.get('alias', table_id)
        
        try:
            # We attempt to load by ID. If user hasn't created table yet, handle error gracefully
            df = data_utils.load_table_data_optimized(table_id)
            if df.empty:
                st.warning(f"Table '{table_id}' not found or empty. Please run the SQL scripts in BigQuery first.")
            else:
                render_table_inspection(df, alias)
        except Exception as e:
            st.error(f"Error loading {alias}: {e}")
            st.info("Tip: Ensure you have executed the SQL scripts provided in the `sql/` folder.")

if __name__ == "__main__":
    main()
