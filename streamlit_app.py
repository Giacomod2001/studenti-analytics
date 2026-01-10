import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging

# Local Modules
import constants
import styles_config
import data_utils

# ─── 1) PAGE CONFIGURATION ─────────────────────────────────────────────────

st.set_page_config(
    page_title="Student Analytics | AI-Powered Retention",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── 2) LANDING PAGE ───────────────────────────────────────────────────────

def render_landing_page():
    """
    Premium landing page with hero section and feature highlights.
    """
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem 2rem 2rem;">
        <h1 style="font-size: 3.5rem; font-weight: 700; 
                   background: linear-gradient(135deg, #00A0DC 0%, #0077B5 50%, #00D4AA 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   margin-bottom: 1rem;">
            🎓 Student Analytics
        </h1>
        <p style="font-size: 1.4rem; color: #8b949e; margin-bottom: 0.5rem;">
            AI-Powered Dropout Prediction & Retention Intelligence
        </p>
        <p style="font-size: 1rem; color: #6b7280; max-width: 600px; margin: 0 auto;">
            Data-driven platform for higher education. Predict dropout risk, 
            analyze student behavior, and design targeted retention strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Cards
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(75, 85, 99, 0.3);
                    border-radius: 16px; padding: 1.5rem; text-align: center; height: 200px;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
            <h4 style="color: #00A0DC; margin-bottom: 0.5rem;">Dropout Prediction</h4>
            <p style="color: #9ca3af; font-size: 0.9rem;">
                Random Forest model predicts dropout probability with high accuracy
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(75, 85, 99, 0.3);
                    border-radius: 16px; padding: 1.5rem; text-align: center; height: 200px;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
            <h4 style="color: #00A0DC; margin-bottom: 0.5rem;">Feature Importance</h4>
            <p style="color: #9ca3af; font-size: 0.9rem;">
                Explainable AI reveals key factors driving student outcomes
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(75, 85, 99, 0.3);
                    border-radius: 16px; padding: 1.5rem; text-align: center; height: 200px;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">👥</div>
            <h4 style="color: #00A0DC; margin-bottom: 0.5rem;">Student Clustering</h4>
            <p style="color: #9ca3af; font-size: 0.9rem;">
                K-Means groups students into behavioral profiles for targeted support
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(75, 85, 99, 0.3);
                    border-radius: 16px; padding: 1.5rem; text-align: center; height: 200px;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📈</div>
            <h4 style="color: #00A0DC; margin-bottom: 0.5rem;">Satisfaction Analysis</h4>
            <p style="color: #9ca3af; font-size: 0.9rem;">
                Boosted Trees correlate satisfaction with academic performance
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # CTA Button
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        if st.button("🚀 Enter Dashboard", use_container_width=True, type="primary"):
            st.session_state["show_landing"] = False
            st.rerun()
    
    st.markdown("---")
    
    # Tech Stack
    st.markdown("### 🛠️ Powered By")
    
    tech_cols = st.columns(6)
    techs = [
        ("🐍", "Python"),
        ("☁️", "BigQuery"),
        ("🎨", "Streamlit"),
        ("🤖", "Scikit-Learn"),
        ("📊", "Plotly"),
        ("🐼", "Pandas")
    ]
    
    for col, (icon, name) in zip(tech_cols, techs):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem;">{icon}</div>
                <p style="color: #9ca3af; font-size: 0.85rem; margin-top: 0.5rem;">{name}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #6b7280;">
        <p style="margin-bottom: 0.5rem;">
            Developed by <strong>Alessandro Geli, Giacomo Dellacqua, Paolo Junior Del Giudice, Ruben Scoletta, Luca Tallarico</strong>
        </p>
        <p style="font-size: 0.85rem;">
            Data Mining & Text Analytics | IULM University | 2024-2025
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── 3) HOME DASHBOARD ─────────────────────────────────────────────────────

def render_home_dashboard(tables_info):
    """
    Enhanced dashboard with premium KPI cards and visual insights.
    """
    st.markdown("""
    <h1 style="font-size: 2.2rem; margin-bottom: 0.25rem;">📊 Analytics Dashboard</h1>
    <p style="color: #8b949e; font-size: 1rem;">Real-time student insights powered by Machine Learning</p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load data
    try:
        df_churn = data_utils.load_table_data_optimized("studenti_churn_pred")
        df_students = data_utils.load_table_data_optimized("studenti")
        df_clusters = data_utils.load_table_data_optimized("studenti_cluster")
        df_features = data_utils.load_table_data_optimized("feature_importance_studenti")
    except:
        df_churn = pd.DataFrame()
        df_students = pd.DataFrame()
        df_clusters = pd.DataFrame()
        df_features = pd.DataFrame()
    
    # ==================== KEY METRICS ====================
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_students = len(df_students) if not df_students.empty else 0
    
    # Calculate dropout risk metrics
    if not df_churn.empty and 'churn_percentage' in df_churn.columns:
        churn_vals = df_churn['churn_percentage'].copy()
        if churn_vals.mean() > 1:
            churn_vals = churn_vals / 100
        avg_risk = churn_vals.mean() * 100
        high_risk_count = len(df_churn[churn_vals > 0.7])
        high_risk_pct = (high_risk_count / len(df_churn)) * 100 if len(df_churn) > 0 else 0
    else:
        avg_risk = 0
        high_risk_count = 0
        high_risk_pct = 0
    
    n_clusters = df_clusters['cluster'].nunique() if not df_clusters.empty and 'cluster' in df_clusters.columns else 0
    
    with col1:
        st.metric("Total Students", f"{total_students:,}", help="Total students in database")
    
    with col2:
        delta_color = "inverse" if avg_risk > 50 else "normal"
        st.metric("Avg Dropout Risk", f"{avg_risk:.1f}%", help="Average predicted dropout probability")
    
    with col3:
        st.metric("High Risk", f"{high_risk_count:,}", delta=f"{high_risk_pct:.0f}% of total", delta_color="inverse")
    
    with col4:
        st.metric("Segments", n_clusters, help="Behavioral clusters identified")
    
    st.markdown("---")
    
    # ==================== RISK BREAKDOWN ====================
    st.markdown("### 🎯 Risk Distribution")
    
    if not df_churn.empty and 'churn_percentage' in df_churn.columns:
        churn_vals = df_churn['churn_percentage'].copy()
        if churn_vals.mean() > 1:
            churn_vals = churn_vals / 100
        
        low_risk = len(df_churn[churn_vals <= 0.3])
        med_risk = len(df_churn[(churn_vals > 0.3) & (churn_vals <= 0.7)])
        high_risk = len(df_churn[churn_vals > 0.7])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"""
            #### ✅ Low Risk
            **{low_risk:,}** students  
            Dropout probability < 30%
            """)
        
        with col2:
            st.warning(f"""
            #### ⚠️ Medium Risk
            **{med_risk:,}** students  
            Dropout probability 30-70%
            """)
        
        with col3:
            st.error(f"""
            #### 🚨 High Risk
            **{high_risk:,}** students  
            Dropout probability > 70%
            """)
        
        st.markdown("---")
        
        # ==================== ACTIONS ====================
        st.markdown("### 💡 Recommended Actions")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.info(f"""
            **🔴 Immediate**  
            Review {high_risk} high-risk students and schedule counseling sessions.
            """)
        
        with col_b:
            st.info("""
            **🟡 Short-term**  
            Analyze Feature Importance to identify common risk factors.
            """)
        
        with col_c:
            st.info("""
            **🟢 Long-term**  
            Design retention programs based on Student Clustering profiles.
            """)
    else:
        st.info("📊 Dropout prediction data not available. Navigate to 'Dropout Prediction' table to view details.")
    
    # ==================== QUICK NAVIGATION ====================
    st.markdown("---")
    
    with st.expander("📁 Data Catalogue", expanded=False):
        cols = st.columns(4)
        for idx, t in enumerate(tables_info):
            with cols[idx % 4]:
                st.markdown(f"**{t['name']}**")
                st.caption(f"{t['rows']:,} rows")


# ─── 4) KEY INSIGHTS ───────────────────────────────────────────────────────

def render_key_insights(df: pd.DataFrame, table_id: str):
    """
    Renders text-based insights using correct column names.
    """
    if table_id == "studenti_churn_pred":
        if 'churn_percentage' in df.columns:
            mean_val = df['churn_percentage'].mean()
            if mean_val > 1:
                df['churn_percentage'] = df['churn_percentage'] / 100
                
            avg_churn = df['churn_percentage'].mean()
            high_risk = df[df['churn_percentage'] > 0.7].shape[0]
            total = len(df)
            pct_risk = (high_risk / total) * 100 if total > 0 else 0
            
            st.info(f"""
            **🔍 Key Insights:**
            - **Average Dropout Risk:** {avg_churn:.1%}
            - **Critical Students:** {high_risk} ({pct_risk:.1%} of total)
            - **Action:** {high_risk} students require immediate counseling intervention.
            """)
        else:
             st.warning("Insight not available: 'churn_percentage' column missing.")
    
    elif table_id == "feature_importance_studenti":
        importance_cols = [col for col in df.columns if 'importance' in col.lower() or 'peso' in col.lower() or 'percentuale' in col.lower()]
        feature_col = next((col for col in df.columns if 'feature' in col.lower() or 'caratteristica' in col.lower()), df.columns[0])
        
        if importance_cols:
            top_3 = df.sort_values(by=importance_cols[0], ascending=False).head(3)
            
            msg = "**🏆 Top 3 Drivers of Dropout:**\n"
            for _, row in top_3.iterrows():
                msg += f"- **{row[feature_col]}**: {row[importance_cols[0]]:.2f} impact\n"
            
            st.info(msg)
    
    elif table_id == "studenti_cluster":
        cluster_col = next((col for col in df.columns if 'cluster' in col.lower()), None)
        if cluster_col:
            top_cluster = df[cluster_col].value_counts().idxmax()
            counts = df[cluster_col].value_counts()
            
            st.info(f"""
            **👥 Segmentation Summary:**
            - **Dominant Group:** Cluster "{top_cluster}" ({counts[top_cluster]} students)
            - **Distribution:** {len(counts)} distinct behavioral profiles identified.
            """)


# ─── 5) TABLE INSPECTION ───────────────────────────────────────────────────

def render_table_inspection(df: pd.DataFrame, table_info: dict):
    """
    Enhanced table visualization with premium styling.
    """
    # Header
    col_head_1, col_head_2 = st.columns([3, 1])
    with col_head_1:
        st.markdown(f"# 📋 {table_info['name']}")
        st.markdown(f"*{table_info['description']}*")
    with col_head_2:
        st.download_button(
            label="⬇️ Export CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{table_info['name']}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Quick metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Columns", len(df.columns))
    missing_pct = round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2) if not df.empty else 0
    m3.metric("Missing", f"{missing_pct}%")
    mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2) if not df.empty else 0
    m4.metric("Memory", f"{mem_mb} MB")
    
    st.markdown("---")

    # Tabs
    tab_data, tab_stats, tab_info = st.tabs(["📊 Explore Data", "📈 Statistics & Charts", "ℹ️ Info & Origin"])
    
    with tab_data:
        with st.expander("🔍 Advanced Filters", expanded=False):
            col_f1, col_f2 = st.columns([1, 2])
            with col_f1:
                search = st.text_input("Search text", placeholder="Type to filter...")
            with col_f2:
                cols = st.multiselect("Visible columns", df.columns.tolist(), default=df.columns.tolist()[:8])
        
        df_view = df.copy()
        if search:
            mask = df_view.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
            df_view = df_view[mask]
        
        if cols:
            st.dataframe(df_view[cols].head(200), use_container_width=True, height=500)
            st.caption(f"Showing {min(200, len(df_view))} of {len(df_view)} filtered rows.")
        else:
            st.warning("Select at least one column.")

    with tab_stats:
        render_key_insights(df, table_info["id"])
        st.markdown("---")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col_viz_1, col_viz_2 = st.columns([1, 3])
        
        with col_viz_1:
            st.markdown("#### ⚙️ Configuration")
            chart_type = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Scatter", "Bar Chart", "Heatmap"], index=0)
            
            x_axis = st.selectbox("X Axis", df.columns)
            y_axis = st.selectbox("Y Axis", [None] + numeric_cols) if chart_type != "Heatmap" else None
            color_dim = st.selectbox("Color", [None] + df.columns.tolist()) if chart_type != "Heatmap" else None
            
        with col_viz_2:
            try:
                fig = None
                default_color = '#00A0DC'
                
                if chart_type == "Histogram":
                    fig = px.histogram(
                        df, x=x_axis, y=y_axis, color=color_dim, 
                        title=f"Distribution of {x_axis}",
                        color_discrete_sequence=[default_color] if not color_dim else None
                    )
                elif chart_type == "Box Plot":
                    fig = px.box(
                        df, x=x_axis, y=y_axis, color=color_dim, 
                        title=f"Box Plot: {x_axis}",
                        color_discrete_sequence=[default_color] if not color_dim else None
                    )
                elif chart_type == "Scatter":
                    fig = px.scatter(
                        df, x=x_axis, y=y_axis, color=color_dim, 
                        title=f"Scatter: {x_axis} vs {y_axis}",
                        color_discrete_sequence=[default_color] if not color_dim else None
                    )
                elif chart_type == "Bar Chart":
                    if len(df) > 1000 and y_axis:
                        df_agg = df.groupby(x_axis)[y_axis].mean().reset_index()
                        fig = px.bar(
                            df_agg, x=x_axis, y=y_axis, 
                            color=color_dim if color_dim in df_agg else None, 
                            title=f"Average {y_axis} by {x_axis}",
                            color_discrete_sequence=[default_color] if not color_dim or color_dim not in df_agg else None
                        )
                    else:
                        fig = px.bar(
                            df, x=x_axis, y=y_axis, color=color_dim, 
                            title=f"Bar Chart: {x_axis}",
                            color_discrete_sequence=[default_color] if not color_dim else None
                        )
                elif chart_type == "Heatmap":
                    if len(numeric_cols) > 1:
                        corr = df[numeric_cols].corr()
                        fig = px.imshow(
                            corr, 
                            text_auto='.2f', 
                            title="Correlation Matrix", 
                            color_continuous_scale='RdBu_r',
                            aspect="auto"
                        )
                    else:
                        st.info("Need at least 2 numerical columns for Heatmap.")

                if fig:
                    fig = styles_config.apply_chart_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating chart: {e}")

    with tab_info:
        st.markdown("### 📖 Origin and Description")
        origin_text = constants.TABLE_ORIGINS.get(table_info["id"], "No detailed information available.")
        st.markdown(origin_text)


# ─── 6) MAIN APP ───────────────────────────────────────────────────────────

def main():
    styles_config.inject_custom_css()
    
    # Initialize session state
    if "show_landing" not in st.session_state:
        st.session_state["show_landing"] = True
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("# 🎓 Student Analytics")
        st.caption("v3.0 | BigQuery Powered")
        
        st.divider()
        
        # Connection check
        client = data_utils.get_bigquery_client()
        if not client:
            st.error("❌ Unable to connect to BigQuery")
            st.stop()
        else:
            st.success("✅ Connected to BigQuery")
        
        # Load metadata
        tables_info = data_utils.get_tables_metadata_cached()
        
        if not tables_info:
            st.warning("No tables found")
            st.stop()
        
        st.divider()
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        
        # Organize tables by type
        ml_tables = [t for t in tables_info if "pred" in t['id'] or "cluster" in t['id'] or "importance" in t['id']]
        raw_tables = [t for t in tables_info if t not in ml_tables]
        
        options = ["🏠 Home", "🌟 Landing Page"]
        
        # Add ML tables
        if ml_tables:
            options.append("---")
            options.extend([f"🤖 {t['name']}" for t in ml_tables])
        
        # Add raw tables
        if raw_tables:
            options.append("---")
            options.extend([f"📁 {t['name']}" for t in raw_tables])
        
        # Remove separator markers for radio
        clean_options = [o for o in options if o != "---"]
        
        selection = st.radio("", clean_options, label_visibility="collapsed")
        
        st.divider()
        
        # Quick Stats
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        col1.metric("Tables", len(tables_info))
        total_rows = sum(t['rows'] for t in tables_info)
        col2.metric("Records", f"{total_rows//1000}K")
        
        st.divider()
        
        # Footer
        st.caption("Powered by BigQuery & Streamlit")
        st.caption("[GitHub](https://github.com/Giacomod2001/studenti-analytics)")
    
    # ==================== MAIN CONTENT ====================
    if selection == "🌟 Landing Page":
        render_landing_page()
    elif selection == "🏠 Home":
        render_home_dashboard(tables_info)
    else:
        # Extract actual table name (remove emoji prefix)
        table_name = selection.split(" ", 1)[1] if " " in selection else selection
        current_info = next((t for t in tables_info if t["name"] == table_name), None)
        
        if current_info:
            with st.spinner(f"Loading {table_name}..."):
                df = data_utils.load_table_data_optimized(current_info["id"])
                
            if not df.empty:
                render_table_inspection(df, current_info)
            else:
                st.warning(f"Table {table_name} is empty or unable to load.")


if __name__ == "__main__":
    main()
