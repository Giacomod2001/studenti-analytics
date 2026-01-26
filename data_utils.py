# =============================================================================
# DATA_UTILS.PY - Optimized BigQuery Data Loading
# Version 2.0 - Performance Optimized
# =============================================================================

import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import logging
import constants

logger = logging.getLogger(__name__)

# =============================================================================
# BIGQUERY CLIENT (Cached - never expires)
# =============================================================================

@st.cache_resource
def get_bigquery_client():
    """Initializes and caches the BigQuery client."""
    try:
        credentials_dict = {
            "type": st.secrets.get("type"),
            "project_id": st.secrets.get("project_id"),
            "private_key_id": st.secrets.get("private_key_id"),
            "private_key": st.secrets.get("private_key"),
            "client_email": st.secrets.get("client_email"),
            "client_id": st.secrets.get("client_id"),
            "auth_uri": st.secrets.get("auth_uri"),
            "token_uri": st.secrets.get("token_uri"),
            "auth_provider_x509_cert_url": st.secrets.get("auth_provider_x509_cert_url"),
            "client_x509_cert_url": st.secrets.get("client_x509_cert_url"),
            "universe_domain": st.secrets.get("universe_domain")
        }
        
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        client = bigquery.Client(credentials=credentials, project=constants.PROJECT_ID)
        return client
    except Exception as e:
        st.error(f"Error initializing BQ client: {e}")
        logger.error(f"Error initializing BQ client: {e}")
        return None

# =============================================================================
# TABLE METADATA (Cached 1 hour)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_tables_metadata_cached():
    """Retrieves table metadata - cached for 1 hour."""
    client = get_bigquery_client()
    if not client:
        return []

    try:
        dataset_ref = client.dataset(constants.DATASET_ID)
        tables_list = list(client.list_tables(dataset_ref))
        
        tables_info = []
        for table in tables_list:
            table_ref = dataset_ref.table(table.table_id)
            t_obj = client.get_table(table_ref)
            
            tables_info.append({
                "id": table.table_id,
                "name": constants.TABLE_DISPLAY_NAMES.get(table.table_id, table.table_id),
                "description": constants.TABLE_DESCRIPTIONS.get(table.table_id, "N/A"),
                "rows": t_obj.num_rows,
                "size_mb": round(t_obj.num_bytes / (1024 * 1024), 2) if t_obj.num_bytes else 0,
                "created": t_obj.created
            })
            
        return sorted(tables_info, key=lambda x: x["id"])
    except Exception as e:
        st.error(f"Error retrieving metadata: {e}")
        return []

# =============================================================================
# OPTIMIZED DATA LOADING (Cached 30 min, with LIMIT)
# =============================================================================

@st.cache_data(ttl=1800, show_spinner="Loading data...")
def load_table_data_optimized(table_id: str, limit: int = 10000):
    """
    Loads data with automatic LIMIT for performance.
    
    Args:
        table_id: BigQuery table name
        limit: Max rows to load (default 10,000)
    """
    client = get_bigquery_client()
    if not client:
        return pd.DataFrame()

    try:
        # Apply LIMIT for faster loading
        query = f"SELECT * FROM `{constants.PROJECT_ID}.{constants.DATASET_ID}.{table_id}` LIMIT {limit}"
        
        # Fast path: standard query
        df = client.query(query).to_dataframe(create_bqstorage_client=False)

        # Optimize memory with category types
        if not df.empty:
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
                
        return df
    except Exception as e:
        logger.error(f"Error loading {table_id}: {e}")
        return pd.DataFrame()

# =============================================================================
# LIGHTWEIGHT QUERIES (For Dashboard KPIs)
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_risk_counts():
    """Fast query for risk tier counts only."""
    client = get_bigquery_client()
    if not client:
        st.error("BigQuery client not initialized")
        return {"critical": 0, "monitor": 0, "safe": 0, "total": 0}
    
    try:
        query = f"""
        SELECT 
            COUNTIF(churn_percentage >= 75) as critical,
            COUNTIF(churn_percentage >= 35 AND churn_percentage < 75) as monitor,
            COUNTIF(churn_percentage < 35) as safe,
            COUNT(*) as total
        FROM `{constants.PROJECT_ID}.{constants.DATASET_ID}.studenti_churn_pred`
        """
        result = client.query(query).to_dataframe(create_bqstorage_client=False)
        return result.iloc[0].to_dict() if not result.empty else {"critical": 0, "monitor": 0, "safe": 0, "total": 0}
    except Exception as e:
        logger.error(f"get_risk_counts error: {e}")
        st.error(f"BigQuery error: {e}")
        return {"critical": 0, "monitor": 0, "safe": 0, "total": 0}

@st.cache_data(ttl=1800, show_spinner=False)
def get_avg_satisfaction():
    """Fast query for average satisfaction only."""
    client = get_bigquery_client()
    if not client:
        return 0.0
    
    try:
        query = f"""
        SELECT AVG(soddisfazione_predetta) as avg_sat
        FROM `{constants.PROJECT_ID}.{constants.DATASET_ID}.report_finale_soddisfazione_studenti`
        """
        result = client.query(query).to_dataframe(create_bqstorage_client=False)
        return float(result.iloc[0]['avg_sat']) if not result.empty else 0.0
    except Exception as e:
        logger.error(f"get_avg_satisfaction error: {e}")
        return 0.0
