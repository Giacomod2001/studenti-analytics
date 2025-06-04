import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os

def create_bigquery_client():
    """Crea il client BigQuery usando le credenziali"""
    try:
        # Prova prima con st.secrets
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            project_id = st.secrets["gcp_service_account"]["project_id"]
        else:
            # Fallback al file JSON
            import json
            with open("credentials.json", "r") as f:
                credentials_info = json.load(f)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            project_id = credentials_info['project_id']
        
        return bigquery.Client(credentials=credentials, project=project_id)
    except Exception as e:
        st.error(f"Errore nella creazione del client BigQuery: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Student Analytics Dashboard",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Student Analytics Dashboard")
    
    # Test connessione
    client = create_bigquery_client()
    
    if client is None:
        st.error("❌ Impossibile connettersi a BigQuery")
        st.info("Verifica la configurazione delle credenziali")
        return
    
    st.success("✅ Connesso a BigQuery!")
    
    # Resto del codice dell'app...
    st.write("Dashboard funzionante!")

if __name__ == "__main__":
    main()
