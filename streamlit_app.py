import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from google.cloud import bigquery
from google.oauth2 import service_account
import traceback
import logging

# ─── 1) CONFIGURAZIONE DELLA PAGINA E DEL LOG ─────────────────────────────────

st.set_page_config(
    page_title="🎓 Student Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = "laboratorio-ai-460517"
DATASET_ID = "dataset"

# ─── 2) MAPPA DELLE DESCRIZIONI E ORIGINI DEI DATI ───────────────────────────────

# Descrizioni "uman‐readable" di ogni tabella presente in BigQuery
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

# Spiegazione di come sono stati generati i dati per ogni tabella (pipeline e algoritmi)
TABLE_ORIGINS = {
    "studenti": """**Origine:**

La tabella `studenti` raccoglie le informazioni anagrafiche e le metriche di performance di ogni studente.
I dati di partenza provengono dal gestionale dell'università (registro studenti, voti, esami sostenuti, ecc.).
Prima di caricarli in BigQuery, è stato eseguito un processo di pulizia e normalizzazione:
- Rimozione di record duplicati
- Uniformazione dei formati di data e di stringa
- Calcolo di nuove feature (ad esempio, media voti, numero di esami sostenuti)
""",
    "studenti_churn_pred": """**Origine:**

Questa tabella contiene le previsioni di abbandono scolastico (churn) generate da un modello di Machine Learning di tipo **Random Forest**.
**Passaggi principali della pipeline:**
1. Caricamento e pulizia dei dati di base da `studenti` e tabelle correlate.
2. Feature engineering: selezione e trasformazione delle variabili più rilevanti (ad esempio: media dei voti, ore di studio, partecipazione a eventi).
3. Suddivisione del dataset in training e test set.
4. Addestramento del modello Random Forest (con ottimizzazione dei parametri tramite cross-validation).
5. Calcolo delle probabilità di churn per ogni studente (colonna `prob_churn`) e della classe predetta (churn sì/no).
6. Salvataggio dei risultati in questa tabella, insieme a livello di confidenza e label predetta.
""",
    "student_churn_rf": """**Origine:**

Questa tabella contiene i dettagli e le metriche del modello **Random Forest** usato per predire l'abbandono scolastico.
Ogni riga riporta:
- Una metrica di performance (es. accuracy, precision, recall) calcolata sul test set.
- I parametri ottimali utilizzati (numero di alberi, profondità massima, ecc.).
Viene generata durante la fase di validazione, dopo aver eseguito hyperparameter tuning e aver misurato le prestazioni su un hold-out set.
""",
    "feature_importance_studenti": """**Origine:**

Questa tabella mostra l'importanza delle variabili (feature importance) estratte dal modello di Random Forest `student_churn_rf`.
Per ogni caratteristica (`caratteristica`) sono presenti:
- `peso_importanza`: numero di volte in cui la feature è stata selezionata per una divisione nei vari alberi del modello.
- `guadagno_informazione`: somma dell'informazione guadagnata, che indica quanto la feature ha contribuito a ridurre l'impurità.
- `copertura`: numero totale di esempi nel dataset che hanno attraversato un nodo che usa quella feature.
- `percentuale_importanza`: peso normalizzato su scala [0,100].
- `categoria_importanza`: etichetta qualitativa (per esempio: "Molto Importante", "Moderatamente Importante", "Poco Importante").
Questa tabella viene generata prendendo i valori di `feature_importances_` di scikit-learn dal modello e salvandoli su BigQuery.
""",
    "studenti_cluster": """**Origine:**

La tabella `studenti_cluster` assegna ogni studente a un cluster, ottenuto tramite l'algoritmo **K-means**.
**Passaggi principali:**
1. Selezione di feature numeriche significative (es. ore di studio settimanali, media voti, numero di assenze).
2. Standardizzazione delle variabili (scaling) in modo che abbiano media = 0 e varianza = 1.
3. Addestramento di K-means con K = 4 (numero di cluster scelto via elbow method).
4. Calcolo del centroide per ogni cluster e assegnazione dell'etichetta `cluster_id` a ciascuno studente.
5. Salvataggio in questa tabella di `cluster_id`, delle coordinate dei centroidi e della distanza di ciascuno studente dal proprio centroide.
""",
    "student_kmeans": """**Origine:**

Questa tabella contiene i dettagli dell'algoritmo **K-means (K = 4)** utilizzato per il clustering degli studenti.
Include:
- Le coordinate dei centroidi di ciascun cluster.
- L'inertia (somma delle distanze al quadrato dei punti dal rispettivo centroide) per ogni iterazione (utile per verificare la convergenza).
Viene creata durante l'addestramento di K-means per analizzare la qualità della suddivisione.
""",
    "studenti_soddisfazione_btr": """**Origine:**

Questa tabella registra i risultati di un modello di regressione **Boosted Tree** (ad esempio XGBoost) usato per stimare il livello di soddisfazione degli studenti.
**Passaggi principali:**
1. Raccolta dei questionari di soddisfazione (scale Likert 1-5).
2. Pulizia e ricodifica delle risposte (ad esempio, trasformazione in valori numerici).
3. Creazione di feature descrittive (es. numero di eventi frequentati, media voti, reddito familiare).
4. Addestramento del modello di regressione Boosted Tree per predire il punteggio di soddisfazione.
5. Calcolo delle metriche di performance (R², RMSE) su un hold-out set.
6. Salvataggio dei risultati in questa tabella, con stima del punteggio, intervalli di confidenza e feature più influenti.
""",
    "report_finale_soddisfazione_studenti": """**Origine:**

Questo report riassume l'analisi di soddisfazione degli studenti, basata sui risultati di `studenti_soddisfazione_btr`.
Include:
- Grafici di distribuzione dei punteggi di soddisfazione.
- Confronto tra corsi di laurea e cluster di studenti.
- Suggerimenti operativi per migliorare l'esperienza studentesca.
Viene generato automaticamente tramite uno script Python che:
1. Crea diverse viste (view) in BigQuery.
2. Aggrega i dati in tabelle di sintesi.
3. Produce un PDF/HTML finale da condividere con il team di progetto.
"""
}

# ─── 3) GESTIONE DATI E CACHING OTTIMIZZATO ─────────────────────────────────────

@st.cache_resource
def get_bigquery_client():
    """
    Inizializza e cacha il client BigQuery.
    Usa cache_resource perché il client è un oggetto non serializzabile (connessione).
    """
    try:
        # Costruzione credenziali da st.secrets
        credentials_dict = dict(st.secrets)
        
        # Tentativo di ricostruire il dizionario credenziali se i secrets sono sparsi
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
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        return client
    except Exception as e:
        st.error(f"Errore inizializzazione client BQ: {e}")
        logger.error(f"Errore inizializzazione client BQ: {e}")
        return None

@st.cache_data(ttl=600, show_spinner=False)
def get_tables_metadata_cached():
    """
    Recupera i metadati delle tabelle.
    """
    client = get_bigquery_client()
    if not client:
        return []

    try:
        dataset_ref = client.dataset(DATASET_ID)
        tables_list = list(client.list_tables(dataset_ref))
        
        tables_info = []
        for table in tables_list:
            table_ref = dataset_ref.table(table.table_id)
            t_obj = client.get_table(table_ref)
            
            tables_info.append({
                "id": table.table_id,
                "name": table.table_id,
                "description": TABLE_DESCRIPTIONS.get(table.table_id, "N/A"),
                "rows": t_obj.num_rows,
                "size_mb": round(t_obj.num_bytes / (1024 * 1024), 2) if t_obj.num_bytes else 0,
                "created": t_obj.created
            })
            
        return sorted(tables_info, key=lambda x: x["id"])
    except Exception as e:
        st.error(f"Errore recupero metadati: {e}")
        logger.error(f"Errore metadati: {e}")
        return []

@st.cache_data(ttl=600, show_spinner=False)
def load_table_data_optimized(table_id: str):
    """
    Carica i dati ottimizzando i tipi per Arrow/Streamlit.
    """
    client = get_bigquery_client()
    if not client:
        return pd.DataFrame()

    try:
        query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_id}`"
        
        # Tentativo 1: BQ Storage API (veloce)
        try:
            df = client.query(query).to_dataframe()
        except Exception as e_fast:
            logger.warning(f"Fast loading fallito per {table_id}, provo fallback. Errore: {e_fast}")
            # Fallback: create_bqstorage_client=False per usare REST API standard
            df = client.query(query).to_dataframe(create_bqstorage_client=False)
        
        # Ottimizzazione tipi per ridurre memoria e migliorare compatibilità Arrow
        # Convertiamo colonne object che sembrano categorie
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df)
            if num_unique / num_total < 0.5: # Euristica
                df[col] = df[col].astype('category')
                
        return df
    except Exception as e:
        st.error(f"Errore caricamento dati {table_id}: {e}")
        logger.error(f"Errore caricamento dati {table_id}: {e}")
        return pd.DataFrame()

# ─── 4) UI & DESIGN SYSTEM ────────────────────────────────────────────────────

def inject_custom_css():
    st.markdown("""
    <style>
        /* Import Font Inter */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        /* Sfondo generale più pulito */
        .stApp {
            background-color: #f8f9fa; /* Light mode default, dark mode gestito da Streamlit */
        }
        
        /* Dark mode overrides automatici di Streamlit sono buoni, ma rifiniamo le card */
        [data-testid="stMetric"] {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border: 1px solid #e9ecef;
        }
        
        /* Adattamento dark mode per le card */
        @media (prefers-color-scheme: dark) {
            [data-testid="stMetric"] {
                background-color: #262730;
                border: 1px solid #363940;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            .stApp {
                background-color: #0e1117;
            }
        }

        /* Sidebar più elegante */
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
            border-right: 1px solid #e9ecef;
        }
        @media (prefers-color-scheme: dark) {
            [data-testid="stSidebar"] {
                background-color: #1a1c24;
                border-right: 1px solid #363940;
            }
        }

        /* Headers */
        h1, h2, h3 {
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        /* Rimuovere padding eccessivo in alto */
        .block-container {
            padding-top: 2rem;
        }
        
        /* Custom spinner */
        .stSpinner > div {
            border-top-color: #4F46E5 !important;
        }
    </style>
    """, unsafe_allow_html=True)


# ─── 6) FUNZIONI DI RENDERING / VISUALIZZAZIONE DEI DATI ─────────────────────────

def render_home_dashboard(tables_info):
    """
    Dashboard principale con KPI aggregati.
    """
    st.title("🎓 Student Analytics Dashboard")
    st.markdown("Benvenuto nella piattaforma di analisi dati studenti. Seleziona una sezione dalla sidebar per esplorare i dettagli.")
    
    # KPI Cards Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_rows = sum(t["rows"] for t in tables_info)
    total_size = sum(t["size_mb"] for t in tables_info)
    last_update = max([t["created"] for t in tables_info]) if tables_info else "N/A"
    if isinstance(last_update, str) and last_update != "N/A":
        pass # è già stringa
    elif last_update != "N/A":
        last_update = last_update.strftime("%d/%m/%Y")

    col1.metric("Dataset Totali", len(tables_info))
    col2.metric("Record Totali", f"{total_rows:,}")
    col3.metric("Dimensione Dati", f"{total_size:.1f} MB")
    col4.metric("Ultimo Aggiornamento", last_update)
    
    st.markdown("---")
    
    st.subheader("📂 Catalogo Dati Disponibili")
    
    # Grid layout per le card delle tabelle
    cols = st.columns(3)
    for idx, t in enumerate(tables_info):
        with cols[idx % 3]:
            with st.container():
                st.markdown(f"#### 📄 {t['name']}")
                st.caption(t['description'])
                st.markdown(f"**Righe:** {t['rows']:,} | **Size:** {t['size_mb']} MB")
                st.progress(min(1.0, t['rows'] / (total_rows if total_rows > 0 else 1)))


def render_table_inspection(df: pd.DataFrame, table_info: dict):
    """
    Visualizzazione dettagliata di una singola tabella.
    """
    # Header con metadati
    st.title(f"📄 {table_info['name']}")
    st.markdown(f"*{table_info['description']}*")
    
    # Metriche rapide della tabella
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Righe", f"{len(df):,}")
    m2.metric("Colonne", len(df.columns))
    missing_pct = round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2) if not df.empty else 0
    m3.metric("Missing Values", f"{missing_pct}%")
    mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2) if not df.empty else 0
    m4.metric("Memoria", f"{mem_mb} MB")
    
    st.markdown("---")

    # Tabs per organizzare la vista
    tab_data, tab_stats, tab_info = st.tabs(["🔍 Esplora Dati", "📊 Statistiche & Grafici", "ℹ️ Info & Origine"])
    
    with tab_data:
        st.subheader("Anteprima Dati")
        
        # Filtri rapidi
        with st.expander("🔎 Filtri Avanzati", expanded=False):
            search = st.text_input("Cerca testo in tutte le colonne", placeholder="Digita per filtrare...")
            cols = st.multiselect("Seleziona colonne", df.columns.tolist(), default=df.columns.tolist()[:8])
        
        df_view = df.copy()
        if search:
            mask = df_view.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
            df_view = df_view[mask]
        
        if cols:
            st.dataframe(df_view[cols].head(100), use_container_width=True, height=400)
        else:
            st.warning("Seleziona almeno una colonna.")
            
        st.caption(f"Mostrando {min(100, len(df_view))} di {len(df_view)} righe filtratte.")

    with tab_stats:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        col_viz_1, col_viz_2 = st.columns([1, 2])
        
        with col_viz_1:
            st.markdown("### Configurazione Grafico")
            chart_type = st.selectbox("Tipo Grafico", ["Istogramma", "Box Plot", "Scatter", "Bar Chart"])
            
            x_axis = st.selectbox("Asse X", df.columns)
            y_axis = st.selectbox("Asse Y (Opzionale)", [None] + numeric_cols)
            color_dim = st.selectbox("Colore (Opzionale)", [None] + df.columns.tolist())
            
        with col_viz_2:
            try:
                if chart_type == "Istogramma":
                    fig = px.histogram(df, x=x_axis, y=y_axis, color=color_dim, template="plotly_white")
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_dim, template="plotly_white")
                elif chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_dim, template="plotly_white")
                elif chart_type == "Bar Chart":
                    # Aggregazione automatica per bar chart se troppi dati
                    if len(df) > 1000 and y_axis:
                        df_agg = df.groupby(x_axis)[y_axis].mean().reset_index()
                        fig = px.bar(df_agg, x=x_axis, y=y_axis, color=color_dim if color_dim in df_agg else None, template="plotly_white")
                    else:
                        fig = px.bar(df, x=x_axis, y=y_axis, color=color_dim, template="plotly_white")
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Impossibile creare il grafico con questa configurazione: {e}")

    with tab_info:
        st.markdown("### 📖 Origine e Descrizione")
        origin_text = TABLE_ORIGINS.get(table_info["id"], "Nessuna informazione dettagliata disponibile.")
        st.markdown(origin_text)

# ─── 7) MAIN APP ──────────────────────────────────────────────────────────────

def main():
    inject_custom_css()
    
    # Sidebar Navigation
    st.sidebar.title("🎓 Analytics")
    
    # Stato connessione (nascosto se OK per pulizia, mostrato solo se errore o su richiesta)
    client = get_bigquery_client()
    if not client:
        st.error("❌ Errore critico: Impossibile connettersi a BigQuery.")
        st.stop()
        
    # Caricamento metadati (cached)
    with st.spinner("Caricamento catalogo..."):
        tables_info = get_tables_metadata_cached()
    
    if not tables_info:
        st.warning("Nessuna tabella trovata.")
        st.stop()
        
    # Menu Navigazione
    options = ["🏠 Home Dashboard"] + [f"📄 {t['name']}" for t in tables_info]
    selection = st.sidebar.radio("Navigazione", options)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Pulisci Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
        
    # Routing
    if selection == "🏠 Home Dashboard":
        render_home_dashboard(tables_info)
    else:
        # Estrai nome tabella
        table_name = selection.split("📄 ")[1]
        current_info = next((t for t in tables_info if t["name"] == table_name), None)
        
        if current_info:
            with st.spinner(f"Caricamento dati {table_name}..."):
                df = load_table_data_optimized(table_name)
                
            if not df.empty:
                render_table_inspection(df, current_info)
            else:
                st.warning(f"La tabella {table_name} è vuota o impossibile da caricare.")

if __name__ == "__main__":
    main()
