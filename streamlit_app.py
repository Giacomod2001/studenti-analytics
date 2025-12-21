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
    page_title="Student Analytics Dashboard",
    page_icon="graduation_cap",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── 2) RENDERING FUNCTIONS ────────────────────────────────────────────────

def render_home_dashboard(tables_info):
    """
    Main dashboard with real KPIs and actionable insights.
    """
    st.title("Student Analytics Dashboard")
    st.caption("AI-Powered Dropout Prediction and Retention Intelligence")
    
    st.markdown("---")
    
    # Load actual data for insights
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
    st.header("Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_students = len(df_students) if not df_students.empty else 0
    
    # Dropout risk metrics
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
    
    col1.metric("Total Students", f"{total_students:,}")
    col2.metric("Average Dropout Risk", f"{avg_risk:.1f}%")
    col3.metric("High Risk Students", f"{high_risk_count:,}", delta=f"{high_risk_pct:.1f}% of total")
    col4.metric("Behavioral Segments", n_clusters)
    
    st.markdown("---")
    
    # ==================== RISK ANALYSIS ====================
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.header("Dropout Risk Distribution")
        
        if not df_churn.empty and 'churn_percentage' in df_churn.columns:
            churn_vals = df_churn['churn_percentage'].copy()
            if churn_vals.mean() > 1:
                churn_vals = churn_vals / 100
            
            df_risk = pd.DataFrame({'Risk Score': churn_vals * 100})
            df_risk['Risk Category'] = pd.cut(
                df_risk['Risk Score'], 
                bins=[0, 30, 70, 100], 
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            risk_counts = df_risk['Risk Category'].value_counts().reset_index()
            risk_counts.columns = ['Category', 'Count']
            
            fig = px.bar(
                risk_counts, x='Category', y='Count', color='Category',
                color_discrete_map={'Low Risk': '#00C853', 'Medium Risk': '#FFB300', 'High Risk': '#E53935'},
                title="Students by Risk Level"
            )
            fig = styles_config.apply_chart_theme(fig)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dropout prediction data not available.")
    
    with col_right:
        st.header("Risk Summary")
        
        if not df_churn.empty and 'churn_percentage' in df_churn.columns:
            churn_vals = df_churn['churn_percentage'].copy()
            if churn_vals.mean() > 1:
                churn_vals = churn_vals / 100
            low_risk = len(df_churn[churn_vals <= 0.3])
            med_risk = len(df_churn[(churn_vals > 0.3) & (churn_vals <= 0.7)])
            high_risk = len(df_churn[churn_vals > 0.7])
            
            st.success(f"**Low Risk:** {low_risk:,} students")
            st.warning(f"**Medium Risk:** {med_risk:,} students")
            st.error(f"**High Risk:** {high_risk:,} students")
            
            st.markdown("---")
            st.markdown("**Recommended Actions:**")
            st.markdown("- Review high-risk students")
            st.markdown("- Schedule counseling")
            st.markdown("- Analyze risk factors")
    
    st.markdown("---")
    
    # ==================== DATA CATALOGUE (Compact) ====================
    with st.expander("Data Catalogue", expanded=False):
        cols = st.columns(4)
        for idx, t in enumerate(tables_info):
            with cols[idx % 4]:
                st.markdown(f"**{t['name']}**")
                st.caption(f"{t['rows']:,} rows")


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
            **Key Insights:**
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
            
            msg = "**Top 3 Drivers of Dropout:**\n"
            for _, row in top_3.iterrows():
                msg += f"- **{row[feature_col]}**: {row[importance_cols[0]]:.2f} impact\n"
            
            st.info(msg)
    
    elif table_id == "studenti_cluster":
        cluster_col = next((col for col in df.columns if 'cluster' in col.lower()), None)
        if cluster_col:
            top_cluster = df[cluster_col].value_counts().idxmax()
            counts = df[cluster_col].value_counts()
            
            st.info(f"""
            **Segmentation Summary:**
            - **Dominant Group:** Cluster "{top_cluster}" ({counts[top_cluster]} students)
            - **Distribution:** {len(counts)} distinct behavioral profiles identified.
            """)


def render_table_inspection(df: pd.DataFrame, table_info: dict):
    """
    Detailed visualization of a single table.
    """
    # Header with metadata
    col_head_1, col_head_2 = st.columns([3, 1])
    with col_head_1:
        st.title(table_info['name'])
        st.markdown(f"*{table_info['description']}*")
    with col_head_2:
        st.download_button(
            label="Export CSV",
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
    m3.metric("Missing Values", f"{missing_pct}%")
    mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2) if not df.empty else 0
    m4.metric("Memory", f"{mem_mb} MB")
    
    st.markdown("---")

    # Tabs
    tab_data, tab_stats, tab_info = st.tabs(["Explore Data", "Statistics and Charts", "Info and Origin"])
    
    with tab_data:
        # Quick filters
        with st.expander("Advanced Filters", expanded=False):
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
        # Render Text Insights
        render_key_insights(df, table_info["id"])
        st.markdown("---")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col_viz_1, col_viz_2 = st.columns([1, 3])
        
        with col_viz_1:
            st.markdown("#### Configuration")
            chart_type = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Scatter", "Bar Chart", "Heatmap"], index=0)
            
            x_axis = st.selectbox("X Axis", df.columns)
            y_axis = st.selectbox("Y Axis", [None] + numeric_cols) if chart_type != "Heatmap" else None
            color_dim = st.selectbox("Color", [None] + df.columns.tolist()) if chart_type != "Heatmap" else None
            
        with col_viz_2:
            try:
                fig = None
                default_color = '#0077B5'
                
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
        st.markdown("### Origin and Description")
        origin_text = constants.TABLE_ORIGINS.get(table_info["id"], "No detailed information available.")
        st.markdown(origin_text)

# ─── 3) MAIN APP ────────────────────────────────────────────────────────────

def main():
    styles_config.inject_custom_css()
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        # Header
        st.title("Student Analytics")
        st.caption("v2.0 | BigQuery Powered")
        
        st.divider()
        
        # Connection check
        client = data_utils.get_bigquery_client()
        if not client:
            st.error("Unable to connect to BigQuery")
            st.stop()
        
        # Load metadata
        tables_info = data_utils.get_tables_metadata_cached()
        
        if not tables_info:
            st.warning("No tables found")
            st.stop()
        
        # Navigation
        st.markdown("### Navigation")
        
        # Organize tables by type
        ml_tables = [t for t in tables_info if "pred" in t['id'] or "cluster" in t['id'] or "importance" in t['id']]
        raw_tables = [t for t in tables_info if t not in ml_tables]
        
        options = ["Home Dashboard"]
        
        # Add ML tables first
        if ml_tables:
            options.extend([t['name'] for t in ml_tables])
        
        # Add raw tables
        if raw_tables:
            options.extend([t['name'] for t in raw_tables])
        
        selection = st.radio("", options, label_visibility="collapsed")
        
        st.divider()
        
        # Quick Stats
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        col1.metric("Tables", len(tables_info))
        total_rows = sum(t['rows'] for t in tables_info)
        col2.metric("Records", f"{total_rows//1000}K")
        
        st.divider()
        
        # Footer
        st.caption("Powered by BigQuery and Streamlit")
        st.caption("[View on GitHub](https://github.com/Giacomod2001/studenti-analytics)")
    
    # ==================== MAIN CONTENT ====================
    if selection == "Home Dashboard":
        render_home_dashboard(tables_info)
    else:
        current_info = next((t for t in tables_info if t["name"] == selection), None)
        
        if current_info:
            with st.spinner(f"Loading {selection}..."):
                df = data_utils.load_table_data_optimized(current_info["id"])
                
            if not df.empty:
                render_table_inspection(df, current_info)
            else:
                st.warning(f"Table {selection} is empty or unable to load.")

if __name__ == "__main__":
    main()

