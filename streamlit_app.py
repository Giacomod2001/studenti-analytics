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

# ─── 3) FUNZIONE DI INIZIALIZZAZIONE DEL CLIENT BIGQUERY ─────────────────────────

def init_bigquery_client():
    """
    Inizializza il client BigQuery utilizzando le credenziali
    presenti in st.secrets. Ritorna (client, messaggio_status).
    In caso di errore, client=None e status contiene l'errore.
    """
    try:
        credentials_dict = {
            "type": st.secrets["type"],
            "project_id": st.secrets["project_id"],
            "private_key_id": st.secrets["private_key_id"],
            "private_key": st.secrets["private_key"],
            "client_email": st.secrets["client_email"],
            "client_id": st.secrets["client_id"],
            "auth_uri": st.secrets["auth_uri"],
            "token_uri": st.secrets["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["client_x509_cert_url"],
            "universe_domain": st.secrets["universe_domain"]
        }

        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

        # Test di connessione
        test_query = "SELECT 1 as check_connection"
        result = client.query(test_query).result()
        for row in result:
            if row.check_connection == 1:
                logger.info("✅ Connessione BigQuery OK")
                return client, "✅ Connessione BigQuery OK"

        return None, "❌ Test connessione fallito"

    except KeyError as ke:
        missing = str(ke)
        error_msg = f"❌ Configurazione mancante: st.secrets non contiene la chiave {missing}"
        logger.error(error_msg)
        return None, error_msg

    except Exception as e:
        detailed = traceback.format_exc()
        logger.error(f"Errore BigQuery: {detailed}")
        return None, f"❌ Errore BigQuery: {str(e)}"

# ─── 4) FUNZIONE PER RECUPERARE METADATI SULLE TABELLE ────────────────────────────

@st.cache_data(ttl=300)
def get_all_tables():
    """
    Recupera tutte le tabelle presenti nel dataset specificato.
    Ritorna (lista_tables, messaggio_status).
    Ogni elemento di lista_tables è un dict con:
        - id: l'ID (nome) della tabella
        - name: stesso dell'ID
        - description: descrizione testuale
        - rows: numero di righe
        - size_mb: dimensione in MB
        - created: data di creazione
    """
    client, status = init_bigquery_client()
    if not client:
        return [], status

    try:
        dataset_ref = client.dataset(DATASET_ID)
        dataset = client.get_dataset(dataset_ref)
        tables_list = list(client.list_tables(dataset_ref))

        tables_info = []
        for table in tables_list:
            table_ref = dataset_ref.table(table.table_id)
            table_obj = client.get_table(table_ref)
            tables_info.append({
                "id": table.table_id,
                "name": table.table_id,
                "description": TABLE_DESCRIPTIONS.get(table.table_id, f"Tabella dati: {table.table_id}"),
                "rows": table_obj.num_rows,
                "size_mb": round(table_obj.num_bytes / (1024 * 1024), 2) if table_obj.num_bytes else 0,
                "created": table_obj.created.strftime("%Y-%m-%d") if table_obj.created else "N/A"
            })

        tables_info = sorted(tables_info, key=lambda x: x["id"])
        return tables_info, f"✅ Trovate {len(tables_info)} tabelle"

    except Exception as e:
        detailed = traceback.format_exc()
        logger.error(f"Errore get_all_tables: {detailed}")
        return [], f"❌ Errore nel recupero tabelle: {str(e)}"

# ─── 5) FUNZIONE PER CARICARE TUTTI I DATI DI UNA TABELLA ─────────────────────────

@st.cache_data(ttl=300)
def load_full_table(table_id: str) -> (pd.DataFrame, str):
    """
    Carica l'intera tabella BigQuery (senza limiti) in un DataFrame pandas.
    Gestisce il problema db-dtypes con fallback a conversione manuale.
    Ritorna (df, status_message). Se df è None, status_message contiene l'errore.
    """
    client, status = init_bigquery_client()
    if not client:
        return None, status

    try:
        query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_id}`"
        
        # Primo tentativo: conversione diretta a DataFrame
        try:
            df = client.query(query).to_dataframe()
        except Exception as e:
            if "db-dtypes" in str(e).lower():
                # Fallback: carica i dati come righe e converti manualmente
                logger.warning(f"db-dtypes non disponibile, uso fallback per {table_id}")
                query_job = client.query(query)
                results = query_job.result()
                
                # Converti i risultati in lista di dizionari
                rows = []
                for row in results:
                    row_dict = {}
                    for key, value in row.items():
                        # Gestisci tipi speciali di BigQuery
                        if hasattr(value, 'isoformat'):  # DATE, DATETIME, TIMESTAMP
                            row_dict[key] = value.isoformat()
                        elif isinstance(value, bytes):  # BYTES
                            row_dict[key] = value.decode('utf-8', errors='ignore')
                        else:
                            row_dict[key] = value
                    rows.append(row_dict)
                
                # Crea DataFrame dai dizionari
                df = pd.DataFrame(rows)
                
                # Converti colonne numeriche dove possibile
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Prova a convertire in numerico
                        try:
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        except:
                            pass
                        
                        # Prova a convertire in datetime
                        if df[col].dtype == 'object':
                            try:
                                df[col] = pd.to_datetime(df[col], errors='ignore')
                            except:
                                pass
            else:
                raise e

        if df.empty:
            return df, f"⚠️ Tabella {table_id} è vuota"
        else:
            return df, f"✅ Caricate {len(df):,} righe da {table_id}"

    except Exception as e:
        detailed = traceback.format_exc()
        logger.error(f"Errore load_full_table: {detailed}")
        return None, f"❌ Errore nel caricamento {table_id}: {str(e)}"

# ─── 6) FUNZIONI DI RENDERING / VISUALIZZAZIONE DEI DATI ─────────────────────────

def render_overview(tables_info: list):
    """
    Mostra una panoramica generale delle tabelle:
    - Metriche totali: numero di tabelle, righe complessive, dimensione totale
    - Tabella dati con nome, descrizione, righe, dimensione, data creazione
    """
    st.header("📊 Panoramica del Dataset")

    if not tables_info:
        st.warning("⚠️ Nessuna tabella trovata nel dataset")
        return

    # Calcolo metriche
    total_tabelle = len(tables_info)
    total_righe = sum(t["rows"] for t in tables_info)
    total_dimensione = sum(t["size_mb"] for t in tables_info)
    tipi_diversi = len(set(t["description"] for t in tables_info))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Totale Tabelle", total_tabelle)
    col2.metric("📊 Totale Righe", f"{total_righe:,}")
    col3.metric("💾 Dimensione Totale", f"{total_dimensione:.1f} MB")
    col4.metric("🏷️ Tipi Distinti", tipi_diversi)

    st.markdown("---")
    st.subheader("🔍 Dettaglio Tabelle")

    df_overview = pd.DataFrame([
        {
            "Tabella": t["name"],
            "Descrizione": t["description"],
            "Righe": f"{t['rows']:,}",
            "Dimensione (MB)": f"{t['size_mb']:.2f}",
            "Data Creazione": t["created"]
        }
        for t in tables_info
    ])
    st.dataframe(df_overview, use_container_width=True, height=350)


def render_table_inspection(df: pd.DataFrame, table_info: dict):
    """
    Mostra:
    1. Spiegazione breve dell'origine dei dati e dell'algoritmo utilizzato
    2. Intestazione con nome tabella e descrizione
    3. Metriche base: numero righe, colonne, percentuale missing, memoria
    4. Anteprima (prime 20 righe)
    5. Selezione colonne da includere/visualizzare
    6. Grafici base: distribuzioni (istogrammi) e correlazioni (heatmap)
    7. Sezione "Dati Grezzi e Download" con filtro testuale
    """
    # ─── 6.1) "Origine dei Dati e Algoritmi Utilizzati" ────────────────────────────
    st.subheader("📖 Origine dei Dati e Algoritmi Utilizzati")
    origin_text = TABLE_ORIGINS.get(
        table_info["id"],
        """Questa tabella contiene dati prodotti durante le fasi di analisi e modellazione.
- Dati di partenza: informazioni sugli studenti (anagrafica, voti, esami, ecc.).
- Preprocessing: pulizia, normalizzazione e creazione di nuove feature.
- Modelli di ML: a seconda della tabella, si usano Random Forest (per churn), K-means (per clustering) 
  o modelli di Boosted Tree (per soddisfazione).
- Salvataggio: i risultati vengono memorizzati in BigQuery per permettere visualizzazione e download.

Consulta la sezione "Spiegazione Tabelle" nella sidebar per dettagli specifici su ogni tabella."""
    )
    st.markdown(origin_text)
    st.markdown("---")

    # ─── 6.2) INTESTAZIONE E METRICHE BASE SUI DATI ─────────────────────────────────
    st.title(f"📈 Analisi: {table_info['description']}")
    st.markdown("### 📏 Metriche Generali")

    col_r, col_c, col_m, col_mem = st.columns(4)
    col_r.metric("Righe Totali", f"{len(df):,}")
    col_c.metric("Colonne Totali", f"{len(df.columns)}")
    missing_pct = round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2)
    col_m.metric("Dati Mancanti (%)", f"{missing_pct}%")
    mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    col_mem.metric("Memoria Occupata", f"{mem_mb} MB")

    st.markdown("---")

    # ─── 6.3) ANTEPRIMA E SELEZIONE COLONNE ─────────────────────────────────────────
    st.subheader("🔍 Anteprima e Selezione Colonne")
    st.write("**Anteprima (prime 20 righe):**")
    st.dataframe(df.head(20), use_container_width=True, height=240)

    all_cols = df.columns.tolist()
    default_cols = all_cols[: min(10, len(all_cols))]
    selected_cols = st.multiselect(
        "Scegli le colonne da mostrare (default: prime 10)",
        options=all_cols,
        default=default_cols
    )

    if selected_cols:
        subset = df[selected_cols]
        st.write(f"**Visualizzazione colonne selezionate ({len(selected_cols)}):**")
        st.dataframe(subset.head(20), use_container_width=True, height=200)

    st.markdown("---")

    # ─── 6.4) GRAFICI BASE ─────────────────────────────────────────────────────────
    st.subheader("📊 Grafici Base")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.info("ℹ️ Nessuna colonna numerica disponibile per i grafici.")
    else:
        with st.expander("📈 Istogrammi Variabili Numeriche"):
            selected_num = st.multiselect(
                "Seleziona fino a 3 variabili numeriche da plottare:",
                options=numeric_cols,
                default=numeric_cols[: min(3, len(numeric_cols))],
                help="Seleziona 1–3 colonne numeriche per visualizzare gli istogrammi"
            )
            for col in selected_num:
                fig = px.histogram(df, x=col, title=f"Istogramma di {col}")
                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("")

        with st.expander("🔗 Matrice delle Correlazioni"):
            if len(numeric_cols) < 2:
                st.write("Serve almeno una coppia di variabili numeriche.")
            else:
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr,
                    labels=dict(x="Variabile", y="Variabile", color="Correlazione"),
                    x=numeric_cols,
                    y=numeric_cols,
                    color_continuous_scale="RdBu_r",
                    title="Heatmap Correlazioni"
                )
                fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=600)
                st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # ─── 6.5) DATI GREZZI E DOWNLOAD ───────────────────────────────────────────────
    st.subheader("📋 Dati Grezzi e Download")
    search_term = st.text_input("🔎 Cerca testo nei dati (su tutte le colonne):")
    df_search = df.copy()

    if search_term:
        mask = df_search.astype(str).apply(lambda row: row.str.contains(search_term, case=False, na=False)).any(axis=1)
        df_search = df_search[mask]
        st.info(f"Trovate {len(df_search):,} righe contenenti '{search_term}'")

    st.dataframe(df_search.reset_index(drop=True), use_container_width=True, height=300)

    csv = df_search.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Scarica i dati in CSV",
        data=csv,
        file_name=f"{table_info['name']}_export.csv",
        mime="text/csv",
        help="Scarica un file CSV con i dati visualizzati."
    )


# ─── 7) FUNZIONE PRINCIPALE (ENTRY POINT) ────────────────────────────────────────

def main():
    st.sidebar.title("🎓 Student Analytics Dashboard")
    st.sidebar.markdown("""
    Benvenuto! Questa applicazione ti permette di esplorare **tutti i dati** delle tabelle
    presenti nel dataset BigQuery, con un'interfaccia semplice e intuitiva.
    """)

    # ─── 7.1) CONNESSIONE A BIGQUERY ───────────────────────────────────────────────
    with st.sidebar.expander("🔌 Configurazione Connessione BigQuery", expanded=True):
        client, conn_status = init_bigquery_client()
        if "✅" in conn_status:
            st.sidebar.success(conn_status)
        else:
            st.sidebar.error(conn_status)
            st.error("❌ Impossibile connettersi a BigQuery. Verifica i segreti e i permessi.")
            st.stop()

    # ─── 7.2) RECUPERO METADATI SULLE TABELLE ───────────────────────────────────────
    with st.spinner("📡 Recupero tabelle..."):
        tables_info, tables_status = get_all_tables()

    if not tables_info:
        st.sidebar.error(tables_status)
        st.error("❌ Errore nel recupero delle tabelle. Controlla i log.")
        st.stop()
    else:
        st.sidebar.success(tables_status)

    # ─── 7.3) "Spiegazione Tabelle" NELLA SIDEBAR ─────────────────────────────────
    with st.sidebar.expander("📖 Spiegazione Tabelle", expanded=False):
        for t in tables_info:
            descr = TABLE_DESCRIPTIONS.get(t["id"], f"Tabella dati: {t['id']}")
            st.markdown(f"**{t['name']}**: {descr}")

    # ─── 7.4) SELEZIONE DELLA TABELLA ─────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Seleziona Tabella")
    table_names = [t["name"] for t in tables_info]
    sel_table = st.sidebar.selectbox("Scegli la tabella da analizzare:", options=table_names)

    current_info = next((t for t in tables_info if t["name"] == sel_table), None)

    # ─── 7.5) CARICAMENTO DEI DATI (TUTTI) ────────────────────────────────────────
    with st.spinner(f"⏳ Caricamento dati da {sel_table}..."):
        df, load_msg = load_full_table(sel_table)

    if df is None:
        st.sidebar.error(load_msg)
        st.error(f"❌ Errore: {load_msg}")
        st.stop()
    else:
        st.sidebar.success(load_msg)

    # ─── 7.6) PULSANTE DI REFRESH DELLA CACHE ─────────────────────────────────────
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Ricarica Dati (cancella cache)"):
        st.cache_data.clear()
        st.experimental_rerun()

    # ─── 7.7) RENDERING DELL'ANALISI PRINCIPALE ──────────────────────────────────
    render_table_inspection(df, current_info)


if __name__ == "__main__":
    main()
