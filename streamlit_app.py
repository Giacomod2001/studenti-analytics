import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from google.cloud import bigquery
from google.oauth2 import service_account
import traceback
import logging
from typing import Tuple, List, Dict, Optional

# ─── CONFIGURAZIONE ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎓 Student Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Costanti di configurazione
PROJECT_ID = "laboratorio-ai-460517"
DATASET_ID = "dataset"
CACHE_TTL = 600  # 10 minuti

# ─── CONFIGURAZIONE DATI ───────────────────────────────────────────────────────

TABLE_DESCRIPTIONS = {
    "studenti": "Dati anagrafici e performance degli studenti",
    "studenti_churn_pred": "Previsioni di abbandono scolastico con probabilità",
    "studenti_cluster": "Segmentazione degli studenti tramite clustering",
    "studenti_soddisfazione_btr": "Analisi della soddisfazione degli studenti",
    "feature_importance_studenti": "Importanza delle variabili nel modello predittivo",
    "report_finale_soddisfazione_studenti": "Report completo dell'analisi di soddisfazione",
    "student_churn_rf": "Dettagli del modello Random Forest per la previsione di abbandono",
    "student_kmeans": "Dettagli del modello K-means per clustering comportamentale"
}

TABLE_ORIGINS = {
    "studenti": """
**Origine**: Dati anagrafici e performance degli studenti dal gestionale universitario.
- Pulizia e normalizzazione dei dati
- Rimozione duplicati e uniformazione formati
- Calcolo feature derivate (media voti, numero esami)
""",
    "studenti_churn_pred": """
**Origine**: Previsioni di abbandono tramite modello Random Forest.
- Feature engineering su dati comportamentali
- Training/test split e cross-validation
- Calcolo probabilità di churn e confidence score
""",
    "student_churn_rf": """
**Origine**: Metriche e parametri del modello Random Forest.
- Performance metrics (accuracy, precision, recall)
- Parametri ottimali da hyperparameter tuning
- Validazione su hold-out set
""",
    "feature_importance_studenti": """
**Origine**: Importanza delle variabili dal modello Random Forest.
- Feature importance da scikit-learn
- Calcolo peso, guadagno informazione, copertura
- Categorizzazione qualitativa dell'importanza
""",
    "studenti_cluster": """
**Origine**: Clustering comportamentale tramite K-means (K=4).
- Selezione e standardizzazione feature numeriche
- Elbow method per scelta numero cluster
- Assegnazione cluster e calcolo distanze dai centroidi
""",
    "student_kmeans": """
**Origine**: Dettagli algoritmo K-means per clustering.
- Coordinate dei centroidi per ogni cluster
- Inertia per iterazione e verifica convergenza
- Metriche di qualità della suddivisione
""",
    "studenti_soddisfazione_btr": """
**Origine**: Stima soddisfazione tramite Boosted Tree (XGBoost).
- Elaborazione questionari Likert (1-5)
- Feature engineering su dati comportamentali
- Calcolo R², RMSE e intervalli di confidenza
""",
    "report_finale_soddisfazione_studenti": """
**Origine**: Report aggregato analisi soddisfazione.
- Grafici distribuzione punteggi
- Confronti per corso e cluster
- Suggerimenti operativi automatizzati
"""
}

# ─── FUNZIONI CORE ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL)
def init_bigquery_client() -> Tuple[Optional[bigquery.Client], str]:
    """Inizializza client BigQuery con gestione errori robusta."""
    try:
        required_keys = [
            "type", "project_id", "private_key_id", "private_key", "client_email",
            "client_id", "auth_uri", "token_uri", "auth_provider_x509_cert_url",
            "client_x509_cert_url", "universe_domain"
        ]
        
        # Verifica presenza chiavi necessarie
        missing_keys = [key for key in required_keys if key not in st.secrets]
        if missing_keys:
            return None, f"❌ Chiavi mancanti in secrets: {', '.join(missing_keys)}"
        
        credentials_dict = {key: st.secrets[key] for key in required_keys}
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

        # Test connessione
        client.query("SELECT 1 as test").result()
        logger.info("✅ Connessione BigQuery stabilita")
        return client, "✅ Connesso a BigQuery"

    except Exception as e:
        error_msg = f"❌ Errore connessione BigQuery: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return None, error_msg


@st.cache_data(ttl=CACHE_TTL)
def get_tables_info() -> Tuple[List[Dict], str]:
    """Recupera metadati di tutte le tabelle del dataset."""
    client, status = init_bigquery_client()
    if not client:
        return [], status

    try:
        dataset_ref = client.dataset(DATASET_ID)
        tables_list = list(client.list_tables(dataset_ref))
        
        tables_info = []
        for table in tables_list:
            table_obj = client.get_table(dataset_ref.table(table.table_id))
            tables_info.append({
                "id": table.table_id,
                "name": table.table_id,
                "description": TABLE_DESCRIPTIONS.get(table.table_id, f"Dati: {table.table_id}"),
                "rows": table_obj.num_rows,
                "size_mb": round(table_obj.num_bytes / (1024 * 1024), 2) if table_obj.num_bytes else 0,
                "created": table_obj.created.strftime("%Y-%m-%d") if table_obj.created else "N/A"
            })

        tables_info.sort(key=lambda x: x["id"])
        return tables_info, f"✅ Trovate {len(tables_info)} tabelle"

    except Exception as e:
        error_msg = f"❌ Errore recupero tabelle: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return [], error_msg


@st.cache_data(ttl=CACHE_TTL)
def load_table_data(table_id: str) -> Tuple[Optional[pd.DataFrame], str]:
    """Carica tutti i dati di una tabella con gestione errori robusta."""
    client, status = init_bigquery_client()
    if not client:
        return None, status

    try:
        query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_id}`"
        
        # Primo tentativo: conversione diretta
        try:
            df = client.query(query).to_dataframe()
        except Exception as e:
            # Fallback per problemi db-dtypes
            if "db-dtypes" in str(e).lower():
                logger.warning(f"Usando fallback per {table_id}")
                df = _load_with_fallback(client, query)
            else:
                raise e

        if df.empty:
            return df, f"⚠️ Tabella {table_id} vuota"
        
        return df, f"✅ Caricate {len(df):,} righe da {table_id}"

    except Exception as e:
        error_msg = f"❌ Errore caricamento {table_id}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return None, error_msg


def _load_with_fallback(client: bigquery.Client, query: str) -> pd.DataFrame:
    """Caricamento dati con conversione manuale per problemi db-dtypes."""
    query_job = client.query(query)
    results = query_job.result()
    
    rows = []
    for row in results:
        row_dict = {}
        for key, value in row.items():
            if hasattr(value, 'isoformat'):  # Date/datetime
                row_dict[key] = value.isoformat()
            elif isinstance(value, bytes):
                row_dict[key] = value.decode('utf-8', errors='ignore')
            else:
                row_dict[key] = value
        rows.append(row_dict)
    
    df = pd.DataFrame(rows)
    
    # Conversione automatica tipi
    for col in df.columns:
        if df[col].dtype == 'object':
            # Prova conversione numerica
            df[col] = pd.to_numeric(df[col], errors='ignore')
            # Prova conversione datetime se ancora object
            if df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col], errors='ignore')
    
    return df


# ─── FUNZIONI UI ───────────────────────────────────────────────────────────────

def render_sidebar(tables_info: List[Dict]) -> str:
    """Renderizza sidebar con selezione tabella e informazioni."""
    st.sidebar.title("🎓 Student Analytics Dashboard")
    st.sidebar.markdown("Esplora i dati del dataset con interfaccia intuitiva.")

    # Status connessione
    with st.sidebar.expander("🔌 Status Connessione", expanded=False):
        client, status = init_bigquery_client()
        if "✅" in status:
            st.success(status)
        else:
            st.error(status)

    # Spiegazione tabelle
    with st.sidebar.expander("📖 Spiegazione Tabelle", expanded=False):
        for table in tables_info:
            st.markdown(f"**{table['name']}**: {table['description']}")

    # Selezione tabella
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Seleziona Tabella")
    table_names = [t["name"] for t in tables_info]
    return st.sidebar.selectbox("Scegli tabella da analizzare:", options=table_names)


def render_overview_metrics(tables_info: List[Dict]):
    """Mostra metriche di overview del dataset."""
    if not tables_info:
        st.warning("⚠️ Nessuna tabella trovata nel dataset")
        return

    total_tables = len(tables_info)
    total_rows = sum(t["rows"] for t in tables_info)
    total_size = sum(t["size_mb"] for t in tables_info)

    col1, col2, col3 = st.columns(3)
    col1.metric("📋 Tabelle", total_tables)
    col2.metric("📊 Righe Totali", f"{total_rows:,}")
    col3.metric("💾 Dimensione", f"{total_size:.1f} MB")


def render_table_overview(tables_info: List[Dict]):
    """Mostra tabella riassuntiva delle tabelle."""
    df_overview = pd.DataFrame([
        {
            "Tabella": t["name"],
            "Descrizione": t["description"],
            "Righe": f"{t['rows']:,}",
            "Dimensione (MB)": f"{t['size_mb']:.2f}",
            "Creata": t["created"]
        }
        for t in tables_info
    ])
    st.dataframe(df_overview, use_container_width=True, height=350)


def render_data_origin(table_id: str):
    """Mostra origine e metodologia dei dati."""
    st.subheader("📖 Origine dei Dati")
    origin_text = TABLE_ORIGINS.get(table_id, """
**Origine**: Dati del progetto Student Analytics.
- Preprocessing standard applicato
- Consulta la sidebar per dettagli specifici
""")
    st.markdown(origin_text)
    st.markdown("---")


def render_data_metrics(df: pd.DataFrame, table_info: Dict):
    """Mostra metriche base del dataframe."""
    st.title(f"📈 Analisi: {table_info['description']}")
    st.subheader("📏 Metriche Generali")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Righe", f"{len(df):,}")
    col2.metric("Colonne", f"{len(df.columns)}")
    
    missing_pct = round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2)
    col3.metric("Dati Mancanti", f"{missing_pct}%")
    
    mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    col4.metric("Memoria", f"{mem_mb} MB")


def render_data_preview(df: pd.DataFrame):
    """Mostra anteprima e selezione colonne."""
    st.subheader("🔍 Anteprima Dati")
    
    # Anteprima
    st.dataframe(df.head(20), use_container_width=True, height=300)
    
    # Selezione colonne
    all_cols = df.columns.tolist()
    default_cols = all_cols[:min(8, len(all_cols))]
    selected_cols = st.multiselect(
        "Seleziona colonne da visualizzare:",
        options=all_cols,
        default=default_cols,
        help="Scegli le colonne più rilevanti per l'analisi"
    )

    if selected_cols:
        subset = df[selected_cols]
        st.dataframe(subset.head(20), use_container_width=True, height=250)
    
    return selected_cols


def render_visualizations(df: pd.DataFrame):
    """Crea visualizzazioni interattive dei dati."""
    st.subheader("📊 Visualizzazioni")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.info("ℹ️ Nessuna colonna numerica per i grafici.")
        return

    # Istogrammi
    with st.expander("📈 Distribuzioni", expanded=True):
        selected_nums = st.multiselect(
            "Variabili per istogrammi (max 3):",
            options=numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))],
            max_selections=3
        )
        
        if selected_nums:
            cols = st.columns(len(selected_nums))
            for i, col in enumerate(selected_nums):
                with cols[i]:
                    fig = px.histogram(df, x=col, title=f"Distribuzione {col}")
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

    # Correlazioni
    if len(numeric_cols) >= 2:
        with st.expander("🔗 Correlazioni", expanded=False):
            corr_cols = st.multiselect(
                "Variabili per matrice correlazioni:",
                options=numeric_cols,
                default=numeric_cols[:min(6, len(numeric_cols))],
                help="Seleziona fino a 6 variabili per una matrice leggibile"
            )
            
            if len(corr_cols) >= 2:
                corr = df[corr_cols].corr()
                fig = px.imshow(
                    corr,
                    labels=dict(color="Correlazione"),
                    color_continuous_scale="RdBu_r",
                    title="Matrice Correlazioni"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)


def render_data_export(df: pd.DataFrame, table_name: str):
    """Gestisce ricerca, filtri e export dei dati."""
    st.subheader("📋 Esplora e Scarica Dati")
    
    # Ricerca testuale
    search_term = st.text_input("🔎 Cerca nel dataset:")
    df_filtered = df.copy()
    
    if search_term:
        mask = df_filtered.astype(str).apply(
            lambda row: row.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        df_filtered = df_filtered[mask]
        st.info(f"Trovate {len(df_filtered):,} righe con '{search_term}'")

    # Visualizzazione dati filtrati
    st.dataframe(df_filtered, use_container_width=True, height=400)

    # Download
    if not df_filtered.empty:
        csv_data = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Scarica CSV",
            data=csv_data,
            file_name=f"{table_name}_export.csv",
            mime="text/csv",
            help="Download dei dati visualizzati in formato CSV"
        )


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    """Funzione principale dell'applicazione."""
    
    # Recupero dati tabelle
    with st.spinner("📡 Caricamento tabelle..."):
        tables_info, tables_status = get_tables_info()

    if not tables_info:
        st.error(f"❌ {tables_status}")
        st.stop()

    # Sidebar
    selected_table = render_sidebar(tables_info)
    current_table_info = next((t for t in tables_info if t["name"] == selected_table), None)

    # Caricamento dati tabella selezionata
    with st.spinner(f"⏳ Caricamento {selected_table}..."):
        df, load_status = load_table_data(selected_table)

    if df is None:
        st.error(f"❌ {load_status}")
        st.stop()
    
    st.success(load_status)

    # Contenuto principale
    render_data_origin(selected_table)
    render_data_metrics(df, current_table_info)
    
    st.markdown("---")
    render_data_preview(df)
    
    st.markdown("---")
    render_visualizations(df)
    
    st.markdown("---")
    render_data_export(df, selected_table)


if __name__ == "__main__":
    main()
