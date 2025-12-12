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
    page_title="🎓 Student Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── 2) RENDERING FUNCTIONS ────────────────────────────────────────────────

def render_home_dashboard(tables_info):
    """
    Main dashboard with aggregated KPIs and Standard UI.
    """
    st.title("🎓 Student Analytics Dashboard")
    st.caption("AI-Powered Dropout Prediction & Retention Intelligence")
    
    st.markdown("---")
    
    # KPI Section - Standard Streamlit Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_rows = sum(t["rows"] for t in tables_info)
    total_size = sum(t["size_mb"] for t in tables_info)
    last_update = max([t["created"] for t in tables_info]) if tables_info else "N/A"
    if hasattr(last_update, 'strftime'):
        last_update = last_update.strftime("%d/%m")

    col1.metric("Datasets", len(tables_info))
    col2.metric("Total Records", f"{total_rows:,}")
    col3.metric("Size in Cloud", f"{total_size:.1f} MB")
    col4.metric("Last Update", last_update)
    
    st.markdown("---")
    
    st.header("📂 Data Warehouse Catalogue")
    st.info("Select a dataset below to explore insights, visualize trends, and export data.")
    
    # Grid layout for table cards
    cols = st.columns(3)
    for idx, t in enumerate(tables_info):
        with cols[idx % 3]:
            # Standard Streamlit Container
            origin_badge = "ML Generated" if "pred" in t['id'] or "cluster" in t['id'] else "Raw Data"
            
            with st.container(border=True):
                st.subheader(f"📄 {t['name']}")
                st.caption(f"**{origin_badge}**")
                st.write(t['description'])
                st.divider()
                c1, c2 = st.columns(2)
                c1.caption(f"**Rows:** {t['rows']:,}")
                c2.caption(f"**Size:** {t['size_mb']} MB")


def render_dropout_dashboard(df: pd.DataFrame):
    """
    Dedicated Dashboard for Dropout Prediction.
    Focus: Actionable Intelligence, Lists of At-Risk Students.
    """
    # fuzzy column detection - Prioritize 'prob' over 'pred'
    # We look for specific keywords in priority order
    risk_col = None
    candidates = df.columns.tolist()
    
    # Priority 1: Contains 'prob' (e.g. prob_churn, probability)
    for c in candidates:
        if 'prob' in c.lower():
            risk_col = c
            break
            
    # Priority 2: Contains 'score'
    if not risk_col:
        for c in candidates:
            if 'score' in c.lower():
                risk_col = c
                break
                
    # Priority 3: Fallback (but dangerous as it might be class labels)
    if not risk_col:
        for c in candidates:
            if 'pred' in c.lower() and df[c].dtype.kind in 'fi': # Only if float/int
                risk_col = c
                break
    
    if not risk_col:
        st.error(f"⚠️ Could not find a numeric risk column. Available columns: {df.columns.tolist()}")
        return
    else:
        # Standardize and Ensure Numeric
        try:
            df['prob_churn'] = pd.to_numeric(df[risk_col], errors='coerce')
        except Exception as e:
            st.error(f"Error converting column '{risk_col}' to numbers: {str(e)}")
            return

    # 1. TOP METRICS
    # Drop NaNs just for the stats to avoid errors
    valid_df = df.dropna(subset=['prob_churn'])
    
    avg_churn = valid_df['prob_churn'].mean()
    high_risk_df = df[df['prob_churn'] > 0.7].copy()
    high_risk_count = len(high_risk_df)
    
    st.markdown("### 🚨 Dropout Risk Monitor")
    
    col1, col2, col3 = st.columns(3)
    if not pd.isna(avg_churn):
        col1.metric("Average Risk Score", f"{avg_churn:.1%}", delta_color="inverse")
    else:
        col1.metric("Average Risk Score", "N/A")
        
    col2.metric("High Risk Students (>70%)", f"{high_risk_count}", f"{(high_risk_count/len(df)):.1%} of total", delta_color="inverse")
    col3.metric("Safe Students (<30%)", f"{len(df[df['prob_churn'] < 0.3])}", delta="Stable")
    
    st.markdown("---")
    
    # 2. ACTION LIST (The "Insight al volo")
    st.subheader("📋 Priority Action List")
    st.caption("Students with the highest probability of dropping out. Focus your retention efforts here.")
    
    # Sort by risk descending
    display_cols = [c for c in df.columns if c in ['student_id', 'nome', 'cognome', 'email', 'prob_churn', 'classe']]
    if not display_cols: display_cols = df.columns.tolist()
    
    # Format the risk column for better readability
    high_risk_df = high_risk_df.sort_values("prob_churn", ascending=False)
    
    # We create a style wrapper
    st.dataframe(
        high_risk_df[display_cols].head(50),
        use_container_width=True,
        hide_index=True,
        column_config={
            "prob_churn": st.column_config.ProgressColumn(
                "Risk Probability",
                help="Probability of student dropping out",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        }
    )
    
    # 3. FAST DISTRIBUTION INSIGHT
    st.markdown("### 📊 Portfolio Risk Overview")
    c1, c2 = st.columns([2, 1])
    with c1:
        # Simple breakdown
        risk_bins = pd.cut(df['prob_churn'], bins=[0, 0.3, 0.7, 1.0], labels=["Low Risk", "Medium Risk", "High Risk"])
        risk_counts = risk_bins.value_counts().sort_index()
        
        st.bar_chart(risk_counts, color=["#10b981", "#f59e0b", "#ef4444"])
    
    with c2:
        st.info("""
        **Strategy:**
        - **Red (High):** Call immediately.
        - **Yellow (Medium):** Schedule academic review.
        - **Green (Low):** Automated check-ins.
        """)

def render_key_insights(df: pd.DataFrame, table_id: str):
    """
    Renders text-based insights instead of charts.
    """
    if table_id == "studenti_churn_pred" and 'prob_churn' in df.columns:
        # Churn Insights
        avg_churn = df['prob_churn'].mean()
        high_risk = df[df['prob_churn'] > 0.7].shape[0]
        total = len(df)
        pct_risk = (high_risk / total) * 100 if total > 0 else 0
        
        st.info(f"""
        **💡 Key Insights:**
        - **Average Dropout Risk:** {avg_churn:.1%}
        - **Critical Students:** {high_risk} ({pct_risk:.1%} of total)
        - **Action:** {high_risk} students require immediate counseling intervention.
        """)
    
    elif table_id == "feature_importance_studenti":
        # Feature Importance Insights
        importance_cols = [col for col in df.columns if 'importance' in col.lower() or 'peso' in col.lower() or 'percentuale' in col.lower()]
        feature_col = next((col for col in df.columns if 'feature' in col.lower() or 'caratteristica' in col.lower()), df.columns[0])
        
        if importance_cols:
            top_3 = df.sort_values(by=importance_cols[0], ascending=False).head(3)
            
            msg = "**🔍 Top 3 Drivers of Dropout:**\n"
            for _, row in top_3.iterrows():
                msg += f"- **{row[feature_col]}**: {row[importance_cols[0]]:.2f} impact\n"
            
            st.info(msg)
    
    elif table_id == "studenti_cluster":
        # Cluster Insights
        cluster_col = next((col for col in df.columns if 'cluster' in col.lower()), None)
        if cluster_col:
            top_cluster = df[cluster_col].value_counts().idxmax()
            counts = df[cluster_col].value_counts()
            
            st.info(f"""
            **👥 Segmentation Summary:**
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
        st.title(f"📄 {table_info['name']}")
        st.markdown(f"*{table_info['description']}*")
    with col_head_2:
        st.download_button(
            label="📥 Export CSV",
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
    # For Churn Prediction, we want a special dashboard, not the generic tabs
    if table_info["id"] == "studenti_churn_pred":
        render_dropout_dashboard(df)
        return

    # Tabs
    tab_data, tab_stats, tab_info = st.tabs(["🔍 Explore Data", "📊 Statistics & Charts", "ℹ️ Info & Origin"])
    
    with tab_data:
        # Quick filters
        with st.expander("🔎 Advanced Filters", expanded=False):
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
            st.markdown("#### ⚙️ Configuration")
            chart_type = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Scatter", "Bar Chart", "Heatmap"], index=0)
            
            x_axis = st.selectbox("X Axis", df.columns)
            y_axis = st.selectbox("Y Axis", [None] + numeric_cols) if chart_type != "Heatmap" else None
            color_dim = st.selectbox("Color", [None] + df.columns.tolist()) if chart_type != "Heatmap" else None
            
        with col_viz_2:
            try:
                fig = None
                # Default neutral color
                default_color = '#6b7280'
                
                if chart_type == "Histogram":
                    fig = px.histogram(
                        df, x=x_axis, y=y_axis, color=color_dim, 
                        title=f"Distribution of {x_axis}",
                        color_discrete_sequence=[default_color] if not color_dim else None
                    )
                elif chart_type == "Box Plot":
                    fig = px.box(
                        df, x=x_axis, y=y_axis, color=color_dim, 
                        title=f"Box Plot {x_axis}",
                        color_discrete_sequence=[default_color] if not color_dim else None
                    )
                elif chart_type == "Scatter":
                    fig = px.scatter(
                        df, x=x_axis, y=y_axis, color=color_dim, 
                        title=f"Scatter {x_axis} vs {y_axis}",
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
                            title=f"Bar Chart {x_axis}",
                            color_discrete_sequence=[default_color] if not color_dim else None
                        )
                elif chart_type == "Heatmap":
                    if len(numeric_cols) > 1:
                        corr = df[numeric_cols].corr()
                        # Minimal heatmap with white-to-blue scale
                        fig = px.imshow(
                            corr, 
                            text_auto='.2f', 
                            title="Correlation Matrix", 
                            color_continuous_scale=['#ffffff', '#3b82f6'],
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

# ─── 3) MAIN APP ────────────────────────────────────────────────────────────

def main():
    styles_config.inject_custom_css()
    
    # Sidebar Navigation
    st.sidebar.title("🎓 Analytics")
    st.sidebar.caption("v2.0 | BigQuery Powered")
    
    # Connection status
    client = data_utils.get_bigquery_client()
    if not client:
        st.error("❌ Critical error: Unable to connect to BigQuery.")
        st.stop()
        
    # Load metadata (cached)
    with st.spinner("Loading catalogue..."):
        tables_info = data_utils.get_tables_metadata_cached()
    
    if not tables_info:
        st.warning("No tables found.")
        st.stop()
        
    # Navigation Menu
    st.sidebar.markdown("### 🧭 Navigation")
    options = ["🏠 Home Dashboard"] + [f"📄 {t['name']}" for t in tables_info]
    selection = st.sidebar.radio("", options, label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Settings"):
        if st.button("🔄 Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.info("Cache TTL: 10 min")
        
    # Routing
    if selection == "🏠 Home Dashboard":
        render_home_dashboard(tables_info)
    else:
        # Extract table name
        table_name = selection.split("📄 ")[1]
        current_info = next((t for t in tables_info if t["name"] == table_name), None)
        
        if current_info:
            with st.spinner(f"Loading data {table_name}..."):
                df = data_utils.load_table_data_optimized(current_info["id"])
                
            if not df.empty:
                render_table_inspection(df, current_info)
            else:
                st.warning(f"Table {table_name} is empty or unable to load.")

if __name__ == "__main__":
    main()
