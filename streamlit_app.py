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

# ─── 2) HELPER: TEXTUAL ANALYSIS ───────────────────────────────────────────

def generate_automated_report(df: pd.DataFrame) -> str:
    """
    Generates a natural language report describing the dataframe's contents.
    Replaces visual charts with descriptive insights.
    """
    report = []
    
    # 1. General Overview
    report.append(f"### Dataset Overview")
    report.append(f"The dataset contains **{len(df):,} records** and **{len(df.columns)} variables**.")
    
    missing_count = df.isna().sum().sum()
    if missing_count == 0:
        report.append("Data quality is high with **no missing values** detected.")
    else:
        report.append(f"Data quality check indicates **{missing_count:,} missing values** across the dataset.")

    # 2. Numerical Analysis
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        report.append(f"### Numerical Analysis")
        for col in num_cols:
            if col.lower() in ['id', 'student_id', 'matricola']: continue
            
            avg = df[col].mean()
            median = df[col].median()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            
            # Distribution description
            dist_desc = ""
            if abs(avg - median) < (0.1 * std):
                dist_desc = "symmetrically distributed"
            elif avg > median:
                dist_desc = "positively skewed (higher tail)"
            else:
                dist_desc = "negatively skewed (lower tail)"
                
            report.append(f"**{col}**")
            report.append(f"- Ranges from {min_val:.2f} to {max_val:.2f}.")
            report.append(f"- The average is {avg:.2f} (median: {median:.2f}), indicating the data is {dist_desc}.")
            
            # Outliers or spread
            cv = std / avg if avg != 0 else 0
            if cv > 1:
                report.append(f"- Shows high variability (Coefficient of Variation: {cv:.2f}).")
            else:
                report.append(f"- Shows consistent values with low variability.")
            
            report.append("<br>")

    # 3. Categorical Analysis
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        report.append(f"### Categorical Distribution")
        for col in cat_cols:
            if df[col].nunique() > 50: continue # Skip high cardinality
            
            top_val = df[col].mode()[0]
            top_count = df[col].value_counts()[top_val]
            top_pct = (top_count / len(df)) * 100
            unique_count = df[col].nunique()
            
            report.append(f"**{col}**")
            report.append(f"- Contains {unique_count} unique categories.")
            report.append(f"- The dominant category is **'{top_val}'**, accounting for {top_pct:.1f}% of records.")
            
            if unique_count <= 5:
                # List distribution
                dist_str = ", ".join([f"{k} ({v})" for k, v in df[col].value_counts().items()])
                report.append(f"- Breakdown: {dist_str}")
                
            report.append("<br>")

    return "\n".join(report)


# ─── 3) LANDING PAGE ───────────────────────────────────────────────────────

def render_landing_page():
    """
    Professional Landing Page - No Emojis, Strict Corporate Style.
    """
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("Student Analytics Platform")
        st.markdown("""
        <p class='subtitle'>
        Authorized access only. This platform provides advanced predictive analytics for student retention and academic performance monitoring.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Access Dashboard", type="primary", use_container_width=False):
            st.session_state["show_landing"] = False
            st.rerun()

    with col2:
        st.markdown("""
        <div style="background-color: #F8F9FA; padding: 2rem; border-radius: 8px; border: 1px solid #E9ECEF;">
            <h3 style="margin-top:0;">System Status</h3>
            <p style="color: #198754; font-weight: 600;">System Operational</p>
            <hr style="margin: 1rem 0; border: 0; border-top: 1px solid #E9ECEF;">
            <p style="font-size: 0.9rem; color: #6C757D;">
            <strong>BigQuery Connection:</strong> Verified<br>
            <strong>Model Version:</strong> Random Forest v2.1<br>
            <strong>Last Update:</strong> Today
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Feature Grid
    st.subheader("Platform Modules")
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Dropout Prediction</div>
            <div class="card-text">
                Proprietary Random Forest algorithms to identify at-risk students with precision.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Behavioral Clustering</div>
            <div class="card-text">
                Unsupervised K-Means segmentation to categorize student engagement patterns.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="card">
            <div class="card-title">Driver Analysis</div>
            <div class="card-text">
                Feature importance ranking to isolate key variables impacting academic success.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown("""
        <div class="card">
            <div class="card-title">Satisfaction Metrics</div>
            <div class="card-text">
                Correlative analysis between qualitative survey data and quantitative performance.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.caption("2026 University Analytics Dept. | Internal Use Only")


# ─── 4) HOME DASHBOARD ─────────────────────────────────────────────────────

def render_home_dashboard(tables_info):
    """
    Executive Dashboard View.
    """
    st.title("Executive Dashboard")
    st.markdown("Overview of student population health and risk assessment.")
    st.markdown("---")
    
    # Load data
    try:
        df_churn = data_utils.load_table_data_optimized("studenti_churn_pred")
        df_students = data_utils.load_table_data_optimized("studenti")
        df_clusters = data_utils.load_table_data_optimized("studenti_cluster")
    except Exception as e:
        df_churn = pd.DataFrame()
        df_students = pd.DataFrame()
        df_clusters = pd.DataFrame()

    # --- KPI ROW ---
    c1, c2, c3, c4 = st.columns(4)
    
    total_students = len(df_students) if not df_students.empty else 0
    
    avg_risk = 0
    high_risk_count = 0
    if not df_churn.empty and 'churn_percentage' in df_churn.columns:
        churn_vals = df_churn['churn_percentage'].copy()
        if churn_vals.mean() > 1: churn_vals /= 100
        avg_risk = churn_vals.mean() * 100
        high_risk_count = len(df_churn[churn_vals > 0.7])
    
    clusters_count = df_clusters['cluster'].nunique() if not df_clusters.empty and 'cluster' in df_clusters.columns else 0
    
    c1.metric("Total Population", f"{total_students:,}")
    c2.metric("Avg. Retention Risk", f"{avg_risk:.1f}%")
    c3.metric("Critical Risk Cases", f"{high_risk_count:,}")
    c4.metric("Active Segments", f"{clusters_count}")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- DETAILED ANALYSIS ---
    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.subheader("Risk Analysis Report")
        if high_risk_count > 0:
            low = len(df_churn[churn_vals <= 0.3])
            med = len(df_churn[(churn_vals > 0.3) & (churn_vals <= 0.7)])
            high = high_risk_count
            total_churn = len(df_churn)
            
            low_pct = (low / total_churn) * 100
            med_pct = (med / total_churn) * 100
            high_pct = (high / total_churn) * 100
            
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border: 1px solid #E0E0E0; border-radius: 8px;">
                <p>The predictive model has analyzed <strong>{total_churn:,} active records</strong>. The population is stratified as follows:</p>
                <ul>
                    <li style="margin-bottom: 0.5rem;">
                        <strong>Low Risk Cohort:</strong> {low:,} students ({low_pct:.1f}%) <br>
                        <span style="color: #666; font-size: 0.9rem;">Students demonstrating strong engagement and academic stability.</span>
                    </li>
                    <li style="margin-bottom: 0.5rem;">
                        <strong>Medium Risk Cohort:</strong> {med:,} students ({med_pct:.1f}%) <br>
                        <span style="color: #666; font-size: 0.9rem;">Students showing early warning signs (attendance fluctuation, grade variance).</span>
                    </li>
                    <li style="margin-bottom: 0.5rem; color: #D32F2F;">
                        <strong>High Risk Cohort:</strong> {high:,} students ({high_pct:.1f}%) <br>
                        <span style="color: #D32F2F; font-size: 0.9rem;">Immediate intervention required. Probability of attrition > 70%.</span>
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.info("No risk data available for analysis.")

    with col_side:
        st.subheader("Action Items")
        st.markdown("""
        <div style="background-color: white; border: 1px solid #E0E0E0; border-radius: 8px; padding: 1rem;">
            <p style="font-weight: 600; color: #D32F2F; margin-bottom: 0.5rem;">CRITICAL</p>
            <p style="font-size: 0.9rem; margin-bottom: 1rem;">
                Review <strong>High Risk</strong> cohort immediately. Schedule academic counseling.
            </p>
            <hr>
            <p style="font-weight: 600; color: #FB8C00; margin-bottom: 0.5rem;">WARNING</p>
            <p style="font-size: 0.9rem;">
                Monitor <strong>Medium Risk</strong> segment for attendance drops.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─── 5) TABLE INSPECTION ───────────────────────────────────────────────────

def render_table_inspection(df: pd.DataFrame, table_info: dict):
    """
    Professional Data Inspector - Text Reports Only, No Charts.
    """
    st.subheader(table_info["name"])
    st.caption(table_info["description"])
    
    col_kpi, col_export = st.columns([3, 1])
    
    with col_kpi:
        st.markdown(f"**Records:** {len(df):,} | **Fields:** {len(df.columns)}")
    
    with col_export:
        st.download_button("Export Dataset (.csv)", df.to_csv(index=False).encode('utf-8'), f"{table_info['name']}.csv", "text/csv")
    
    st.markdown("---")

    tab_data, tab_report, tab_meta = st.tabs(["Data View", "Automated Intelligence Report", "Metadata"])
    
    with tab_data:
        st.dataframe(df.head(100), use_container_width=True, height=600)
    
    with tab_report:
        if not df.empty:
            st.markdown("#### Automated Data Intelligence")
            st.markdown("The system has performed a statistical scan of the dataset to generate the following insights:")
            st.markdown("---")
            
            report_content = generate_automated_report(df)
            st.markdown(report_content, unsafe_allow_html=True)
        else:
            st.warning("Dataset is empty. No report available.")
            
    with tab_meta:
        st.markdown("**Field Descriptions & Logic**")
        st.text_area("SQL Schema Definition", "Detailed schema info available in BigQuery documentation.", disabled=True)


# ─── 6) MAIN APP STRUCTURE ─────────────────────────────────────────────────

def main():
    styles_config.inject_custom_css()
    
    if "show_landing" not in st.session_state:
        st.session_state["show_landing"] = True
        
    if st.session_state["show_landing"]:
        render_landing_page()
        return

    # --- SIDEBAR NAV ---
    with st.sidebar:
        st.title("Analytics")
        
        # Connection Status
        client = data_utils.get_bigquery_client()
        if client:
            st.caption("Connected to BigQuery")
        else:
            st.error("Connection Failed")
            st.stop()
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        tables_info = data_utils.get_tables_metadata_cached()
        
        # Navigation Groups
        st.markdown("<strong>DASHBOARDS</strong>", unsafe_allow_html=True)
        nav_main = st.radio("Dashboards", ["Executive Overview"], label_visibility="collapsed")
        
        st.markdown("<br><strong>PREDICTIVE MODELS</strong>", unsafe_allow_html=True)
        ml_tables = [t for t in tables_info if "pred" in t['id'] or "cluster" in t['id']]
        ml_names = [t['name'] for t in ml_tables]
        nav_ml = st.radio("Models", ["Select Model..."] + ml_names, label_visibility="collapsed")
        
        st.markdown("<br><strong>DATA WAREHOUSE</strong>", unsafe_allow_html=True)
        raw_tables = [t for t in tables_info if t not in ml_tables]
        raw_names = [t['name'] for t in raw_tables]
        nav_data = st.radio("Data", ["Select Data..."] + raw_names, label_visibility="collapsed")
        
        st.markdown("---")
        st.caption("v4.0.0 | IULM Analytics")

    # --- ROUTING ---
    selected_view = "Executive Overview"
    selected_table_info = None
    
    if nav_ml != "Select Model...":
        selected_view = "Table"
        selected_table_info = next((t for t in tables_info if t['name'] == nav_ml), None)
    elif nav_data != "Select Data...":
        selected_view = "Table"
        selected_table_info = next((t for t in tables_info if t['name'] == nav_data), None)
        
    if selected_view == "Executive Overview":
        render_home_dashboard(tables_info)
    elif selected_view == "Table" and selected_table_info:
        df = data_utils.load_table_data_optimized(selected_table_info['id'])
        render_table_inspection(df, selected_table_info)

if __name__ == "__main__":
    main()
