import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
from google.oauth2 import service_account
import logging

# Custom Modules
import styles_config
import ml_utils
import constants

# ─── 1) CONFIGURATION ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Student Analytics Dashboard",
    page_icon="graduation_cap",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = "laboratorio-ai-460517"
DATASET_ID = "dataset"

# ─── 2) DATA MANAGEMENT ────────────────────────────────────────────────────

@st.cache_resource
def get_bigquery_client():
    try:
        credentials_dict = dict(st.secrets)
        if "private_key" in st.secrets:
             credentials_dict = {k: st.secrets.get(k) for k in [
                 "type", "project_id", "private_key_id", "private_key", 
                 "client_email", "client_id", "auth_uri", "token_uri", 
                 "auth_provider_x509_cert_url", "client_x509_cert_url", "universe_domain"
             ]}
        
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        return bigquery.Client(credentials=credentials, project=PROJECT_ID)
    except Exception as e:
        logger.error(f"Error initializing BQ client: {e}")
        return None

@st.cache_data(ttl=600, show_spinner=False)
def get_tables_metadata_cached():
    client = get_bigquery_client()
    if not client: return []
    try:
        dataset_ref = client.dataset(DATASET_ID)
        tables = list(client.list_tables(dataset_ref))
        
        return sorted([
            {
                "id": t.table_id,
                "name": constants.TABLE_DISPLAY_NAMES.get(t.table_id, t.table_id),
                "description": constants.TABLE_DESCRIPTIONS.get(t.table_id, "N/A"),
                "rows": (obj := client.get_table(dataset_ref.table(t.table_id))).num_rows,
                "size_mb": round(obj.num_bytes / (1024**2), 2) if obj.num_bytes else 0,
                "created": obj.created
            }
            for t in tables
        ], key=lambda x: x["id"])
    except Exception as e:
        logger.error(f"Error metadata: {e}")
        return []

@st.cache_data(ttl=600, show_spinner=False)
def load_table_data(table_id: str):
    client = get_bigquery_client()
    if not client: return pd.DataFrame()
    try:
        df = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_id}`").to_dataframe()
        # Optimization
        for col in df.select_dtypes(include=['object']):
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        return df
    except Exception as e:
        logger.error(f"Error loading {table_id}: {e}")
        return pd.DataFrame()

# ─── 3) VISUALIZATION ──────────────────────────────────────────────────────

def render_home(tables_info):
    st.title("Student Analytics Dashboard")
    
    st.markdown("""
    <div class="glass-card">
        <h4>About this Platform</h4>
        <p>A professional ML platform for predicting university dropout risk and analyzing student performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets", len(tables_info))
    c2.metric("Records", f"{sum(t['rows'] for t in tables_info):,}")
    c3.metric("Size", f"{sum(t['size_mb'] for t in tables_info):.1f} MB")
    
    last_update = max(t['created'] for t in tables_info) if tables_info else "N/A"
    c4.metric("Last Update", last_update.strftime("%d/%m/%Y") if hasattr(last_update, 'strftime') else "N/A")
    
    st.markdown("### Data Catalogue")
    cols = st.columns(3)
    for i, t in enumerate(tables_info):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="glass-card" style="height: 100%;">
                <h4>{t['name']}</h4>
                <p style="color: #8b949e; font-size: 0.9em;">{t['description']}</p>
                <div style="margin-top: 10px; font-size: 0.8em; color: #8b949e;">
                    {t['rows']:,} rows | {t['size_mb']} MB
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_analysis(df, table_info):
    st.title(table_info['name'])
    
    tab1, tab2, tab3 = st.tabs(["Explore", "Visuals", "Info"])
    
    with tab1:
        st.dataframe(df, use_container_width=True, height=600)
    
    with tab2:
        # Auto-viz
        num_cols = df.select_dtypes(include=np.number).columns
        if not num_cols.empty:
            c1, c2 = st.columns([1, 3])
            with c1:
                x_axis = st.selectbox("X Axis", df.columns)
                y_axis = st.selectbox("Y Axis", num_cols)
                chart_type = st.selectbox("Type", ["Bar", "Scatter", "Box"])
            
            with c2:
                if chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis)
                elif chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis)
                else:
                    fig = px.box(df, x=x_axis, y=y_axis)
                
                st.plotly_chart(styles_config.apply_chart_theme(fig), use_container_width=True)
        else:
            st.info("No numerical data for automatic visualization.")

    with tab3:
        st.markdown(constants.TABLE_ORIGINS.get(table_info['id'], "No info."))

def render_live_ml():
    st.title("Live Clustering Analysis")
    st.markdown("""
    <div class="glass-card">
        Train a new <strong>K-Means Clustering</strong> model in real-time.
        This uses the <code>ml_utils</code> pipeline with scaling, PCA, and Silhouette validation.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Train Model"):
        with st.spinner("Loading raw data..."):
            df = load_table_data("studenti")
        
        if df.empty:
            st.error("Could not load 'studenti' table.")
            return

        with st.spinner("Training model (~30s)..."):
            try:
                results = ml_utils.train_improved_clustering(df, n_clusters=4)
                
                # Metrics
                c1, c2 = st.columns(2)
                c1.metric("Silhouette Score (Quality)", f"{results['silhouette_score']:.3f}")
                c2.metric("Inertia", f"{results['inertia']:.0f}")
                
                # Visualization
                df_viz = pd.DataFrame(results['X_final'][:, :2], columns=['PC1', 'PC2'])
                df_viz['Cluster'] = results['labels'].astype(str)
                
                fig = px.scatter(df_viz, x='PC1', y='PC2', color='Cluster', 
                               title="Cluster Visualization (PCA Projection)")
                st.plotly_chart(styles_config.apply_chart_theme(fig), use_container_width=True)
                
                st.write("Feature Names used:", results['features_used'])
                
            except Exception as e:
                st.error(f"Training failed: {e}")

# ─── 4) MAIN ENTRYPOINT ────────────────────────────────────────────────────

def main():
    styles_config.inject_custom_css()
    
    st.sidebar.title("Analytics")
    
    # Check connection
    if not get_bigquery_client():
        st.error("Connection failed.")
        st.stop()
        
    tables = get_tables_metadata_cached()
    
    # Navigation
    menu = ["Home"] + [t['name'] for t in tables] + ["Live Analysis"]
    choice = st.sidebar.radio("Go to", menu, label_visibility="collapsed")
    
    if choice == "Home":
        render_home(tables)
    elif choice == "Live Analysis":
        render_live_ml()
    else:
        # Find table info by name
        info = next(t for t in tables if t['name'] == choice)
        with st.spinner(f"Loading {info['name']}..."):
            df = load_table_data(info['id'])
        render_analysis(df, info)

if __name__ == "__main__":
    main()
