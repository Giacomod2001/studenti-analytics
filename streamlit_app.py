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
    Main dashboard with aggregated KPIs and About section.
    """
    st.title("🎓 Student Analytics Dashboard")
    
    # About Section
    st.markdown("""
    <div style="background-color: #eef2ff; padding: 20px; border-radius: 10px; border-left: 5px solid #4f46e5; margin-bottom: 25px;">
        <h4 style="color: #4f46e5; margin-top: 0;">ℹ️ About this Platform</h4>
        <p style="margin-bottom: 0; color: #374151;">
            A functional MVP of a cloud-native ML platform to predict university dropout risk, 
            demonstrating data-driven retention strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_rows = sum(t["rows"] for t in tables_info)
    total_size = sum(t["size_mb"] for t in tables_info)
    last_update = max([t["created"] for t in tables_info]) if tables_info else "N/A"
    if hasattr(last_update, 'strftime'):
        last_update = last_update.strftime("%d/%m/%Y")

    col1.metric("Total Datasets", len(tables_info))
    col2.metric("Total Records", f"{total_rows:,}")
    col3.metric("Data Size", f"{total_size:.1f} MB")
    col4.metric("Last Update", last_update)
    
    st.markdown("---")
    st.subheader("📂 Data Catalogue")
    
    # Grid layout for table cards
    cols = st.columns(3)
    for idx, t in enumerate(tables_info):
        with cols[idx % 3]:
            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; height: 100%; background-color: white;">
                    <h4 style="margin-top: 0; color: #111827;">📄 {t['name']}</h4>
                    <p style="font-size: 0.9em; color: #6b7280; height: 40px; overflow: hidden; text-overflow: ellipsis;">{t['description']}</p>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                        <span style="background-color: #f3f4f6; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; color: #374151;">{t['rows']:,} rows</span>
                        <span style="font-size: 0.8em; color: #9ca3af;">{t['size_mb']} MB</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def create_specialized_chart(df: pd.DataFrame, table_id: str):
    """
    Creates minimal, clean specialized charts for ML tables.
    """
    NEUTRAL_COLOR = '#6b7280'  # Neutral gray
    ACCENT_COLOR = '#3b82f6'   # Clean blue
    
    if table_id == "studenti_churn_pred" and 'prob_churn' in df.columns:
        # Clean bar chart instead of filled histogram
        fig = go.Figure()
        
        # Create bins
        hist_data = np.histogram(df['prob_churn'], bins=20)
        
        fig.add_trace(go.Bar(
            x=hist_data[1][:-1],
            y=hist_data[0],
            marker=dict(
                color=NEUTRAL_COLOR,
                line=dict(color='white', width=2)
            ),
            name='Students'
        ))
        
        fig.update_layout(
            title="Dropout Probability Distribution",
            xaxis_title="Dropout Probability",
            yaxis_title="Number of Students",
            bargap=0.1
        )
        return styles_config.apply_chart_theme(fig)
    
    elif table_id == "feature_importance_studenti":
        # Clean horizontal bar chart with single color
        importance_cols = [col for col in df.columns if 'importance' in col.lower() or 'peso' in col.lower() or 'percentuale' in col.lower()]
        feature_col = next((col for col in df.columns if 'feature' in col.lower() or 'caratteristica' in col.lower()), df.columns[0])
        
        if importance_cols:
            df_sorted = df.sort_values(by=importance_cols[0], ascending=True).tail(15)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_sorted[feature_col],
                x=df_sorted[importance_cols[0]],
                orientation='h',
                marker=dict(
                    color=ACCENT_COLOR,
                    line=dict(color='white', width=1)
                ),
                name='Importance'
            ))
            
            fig.update_layout(
                title="Top 15 Features by Importance",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=500
            )
            return styles_config.apply_chart_theme(fig)
    
    elif table_id == "studenti_cluster":
        # Simple bar chart with neutral colors
        cluster_col = next((col for col in df.columns if 'cluster' in col.lower()), None)
        if cluster_col:
            cluster_counts = df[cluster_col].value_counts().sort_index().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cluster_counts['Cluster'].astype(str),
                y=cluster_counts['Count'],
                marker=dict(
                    color=NEUTRAL_COLOR,
                    line=dict(color='white', width=2)
                ),
                name='Students'
            ))
            
            fig.update_layout(
                title="Student Distribution by Cluster",
                xaxis_title="Cluster ID",
                yaxis_title="Number of Students"
            )
            return styles_config.apply_chart_theme(fig)
    
    return None


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
        # Check for specialized chart
        specialized_chart = create_specialized_chart(df, table_info["id"])
        
        if specialized_chart:
            st.plotly_chart(specialized_chart, use_container_width=True)
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
