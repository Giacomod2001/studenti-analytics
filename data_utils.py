import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import logging
import constants

logger = logging.getLogger(__name__)

@st.cache_resource
def get_bigquery_client():
    """
    Initializes and caches the BigQuery client.
    Uses cache_resource because the client is a non-serializable object (connection).
    """
    try:
        credentials_dict = dict(st.secrets)
        
        if "private_key" in st.secrets:
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

@st.cache_data(ttl=600, show_spinner=False)
def get_tables_metadata_cached():
    """
    Retrieves table metadata.
    """
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
        logger.error(f"Error metadata: {e}")
        return []

@st.cache_data(ttl=600, show_spinner=False)
def load_table_data_optimized(table_id: str):
    """
    Loads data optimizing types for Arrow/Streamlit.
    """
    client = get_bigquery_client()
    if not client:
        return pd.DataFrame()

    try:
        query = f"SELECT * FROM `{constants.PROJECT_ID}.{constants.DATASET_ID}.{table_id}`"
        
        # Attempt 1: BQ Storage API (fast)
        try:
            df = client.query(query).to_dataframe()
        except Exception as e_fast:
            logger.warning(f"Fast loading failed for {table_id}: {e_fast}")
            # Attempt 2: REST API standard
            try:
                df = client.query(query).to_dataframe(create_bqstorage_client=False)
            except Exception as e_rest:
                logger.warning(f"REST loading failed for {table_id}: {e_rest}")
                # Attempt 3: Manual (slow but safe, doesn't require db-dtypes)
                try:
                    job = client.query(query)
                    rows = [dict(row) for row in job.result()]
                    df = pd.DataFrame(rows)
                except Exception as e_manual:
                    raise e_manual

        # Type optimization to reduce memory and improve Arrow compatibility
        if not df.empty:
            for col in df.select_dtypes(include=['object']).columns:
                num_unique = df[col].nunique()
                num_total = len(df)
                if num_total > 0 and num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
                
        return df
    except Exception as e:
        st.error(f"Error loading data {table_id}: {e}")
        logger.error(f"Error loading data {table_id}: {e}")
        return pd.DataFrame()
