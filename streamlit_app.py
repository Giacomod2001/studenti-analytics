import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import os
import traceback
import logging
from datetime import datetime

# Configurazione del logging per tracciare errori e operazioni importanti
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurazione delle costanti
PROJECT_ID = "laboratorio-ai-460517"
DATASET_ID = "dataset"
CACHE_TTL = 300  # Tempo di cache in secondi

# Configurazione della pagina Streamlit
st.set_page_config(
    page_title="🎓 Student Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_bigquery_client():
    """
    Inizializza il client BigQuery con gestione degli errori migliorata.
    Utilizza le credenziali dai segreti di Streamlit per autenticarsi.
    """
    try:
        # Leggi le credenziali dai segreti di Streamlit
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

        # Crea le credenziali
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)

        # Crea client BigQuery
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

        # Test connessione semplice
        test_query = "SELECT 1 as test_connection"
        result = client.query(test_query).result()

        # Verifica risultato
        for row in result:
            if row.test_connection == 1:
                logger.info("✅ Connessione BigQuery stabilita")
                return client, "✅ Connessione BigQuery OK"

        return None, "❌ Test connessione fallito"

    except Exception as e:
        error_msg = f"❌ Errore BigQuery: {str(e)}"
        logger.error(f"Errore dettagliato: {traceback.format_exc()}")
        return None, error_msg

@st.cache_data(ttl=CACHE_TTL)
def get_all_tables():
    """
    Recupera tutte le tabelle dal dataset specificato in BigQuery.
    Restituisce una lista di tabelle con informazioni dettagliate.
    """
    client, status = init_bigquery_client()

    if not client:
        return [], status

    try:
        # Verifica dataset
        dataset_ref = client.dataset(DATASET_ID)
        dataset = client.get_dataset(dataset_ref)

        # Lista tabelle
        tables_list = list(client.list_tables(dataset_ref))

        tables_info = []
        for table in tables_list:
            # Ottieni info dettagliate tabella
            table_ref = dataset_ref.table(table.table_id)
            table_obj = client.get_table(table_ref)

            table_info = {
                'id': table.table_id,
                'name': table.table_id,
                'type': classify_table_type(table.table_id),
                'description': get_table_description(table.table_id),
                'rows': table_obj.num_rows,
                'size_mb': round(table_obj.num_bytes / (1024 * 1024), 2) if table_obj.num_bytes else 0,
                'created': table_obj.created.strftime('%Y-%m-%d') if table_obj.created else 'N/A'
            }
            tables_info.append(table_info)

        return sorted(tables_info, key=lambda x: x['id']), f"✅ Trovate {len(tables_info)} tabelle"

    except Exception as e:
        error_msg = f"❌ Errore nel recupero tabelle: {str(e)}"
        logger.error(f"Errore get_all_tables: {traceback.format_exc()}")
        return [], error_msg

def classify_table_type(table_id):
    """
    Classifica il tipo di tabella in base al nome.
    Utilizza emoji per rendere più intuitiva la classificazione.
    """
    table_lower = table_id.lower()
    if 'churn' in table_lower:
        return '🔮 Predizione'
    elif 'cluster' in table_lower or 'kmeans' in table_lower:
        return '🎯 Clustering'
    elif 'soddisfazione' in table_lower:
        return '😊 Soddisfazione'
    elif 'feature' in table_lower:
        return '⚙️ Features'
    elif 'report' in table_lower:
        return '📊 Report'
    elif table_lower == 'studenti':
        return '👥 Dati Base'
    else:
        return '📋 Altro'

def get_table_description(table_id):
    """
    Ottiene una descrizione dettagliata della tabella in base al nome.
    """
    descriptions = {
        'studenti': 'Dati anagrafici e performance degli studenti',
        'studenti_churn_pred': 'Previsioni di abbandono scolastico con probabilità',
        'studenti_cluster': 'Segmentazione degli studenti tramite clustering',
        'studenti_soddisfazione_btr': 'Analisi della soddisfazione degli studenti',
        'feature_importance_studenti': 'Importanza delle variabili nel modello predittivo',
        'report_finale_soddisfazione_studenti': 'Report completo analisi soddisfazione',
        'student_churn_rf': 'Modello Random Forest per previsione abbandoni',
        'student_kmeans': 'Modello K-means per clustering comportamentale'
    }
    return descriptions.get(table_id, f'Tabella dati: {table_id}')

@st.cache_data(ttl=CACHE_TTL)
def load_table_data(table_id, limit=1000):
    """
    Carica i dati da una tabella specifica in BigQuery.
    Restituisce un DataFrame pandas con i dati e un messaggio di stato.
    """
    client, _ = init_bigquery_client()

    if not client:
        return None, "❌ Client BigQuery non disponibile"

    try:
        query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.{table_id}`
        LIMIT {limit}
        """

        df = client.query(query).to_dataframe()

        if df.empty:
            return df, f"⚠️ Tabella {table_id} vuota"

        return df, f"✅ Caricati {len(df):,} record da {table_id}"

    except Exception as e:
        error_msg = f"❌ Errore nel caricamento {table_id}: {str(e)}"
        logger.error(f"Errore load_table_data: {traceback.format_exc()}")
        return None, error_msg

def render_tables_overview(tables):
    """
    Renderizza una panoramica delle tabelle disponibili nel dataset.
    Mostra metriche generali e una tabella dettagliata con informazioni sulle tabelle.
    """
    st.header("📊 Tabelle Disponibili nel Dataset")

    if not tables:
        st.warning("⚠️ Nessuna tabella trovata nel dataset")
        return

    # Metriche generali
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📋 Totale Tabelle", len(tables))

    with col2:
        total_rows = sum(t['rows'] for t in tables if t['rows'])
        st.metric("📊 Totale Righe", f"{total_rows:,}")

    with col3:
        total_size = sum(t['size_mb'] for t in tables if t['size_mb'])
        st.metric("💾 Dimensione Tot.", f"{total_size:.1f} MB")

    with col4:
        types = set(t['type'] for t in tables)
        st.metric("🏷️ Tipi Diversi", len(types))

    # Tabella dettagliata
    st.subheader("🔍 Dettagli Tabelle")

    df_tables = pd.DataFrame([{
        'Tabella': t['name'],
        'Tipo': t['type'],
        'Descrizione': t['description'],
        'Righe': f"{t['rows']:,}" if t['rows'] else 'N/A',
        'Dimensione (MB)': f"{t['size_mb']:.2f}" if t['size_mb'] else 'N/A',
        'Creata': t['created']
    } for t in tables])

    st.dataframe(df_tables, use_container_width=True, height=400)

def render_data_analysis(df, table_info):
    """
    Renderizza un'analisi dettagliata dei dati.
    Mostra metriche principali, informazioni sulle colonne, distribuzioni, correlazioni e grafici.
    """
    st.header(f"📈 Analisi: {table_info['description']}")

    if df is None or df.empty:
        st.warning("⚠️ Nessun dato da analizzare")
        return

    # Metriche principali
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📏 Righe", f"{len(df):,}")

    with col2:
        st.metric("📊 Colonne", len(df.columns))

    with col3:
        missing_pct = round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2)
        st.metric("❓ Dati Mancanti", f"{missing_pct}%")

    with col4:
        memory_mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        st.metric("💾 Memoria", f"{memory_mb} MB")

    # Tabs per analisi dettagliata
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Info Colonne", "📊 Distribuzione", "🔗 Correlazioni", "📈 Grafici"])

    with tab1:
        render_columns_info(df)

    with tab2:
        render_distribution_analysis(df)

    with tab3:
        render_correlation_analysis(df)

    with tab4:
        render_visualizations(df)

def render_columns_info(df):
    """
    Renderizza informazioni dettagliate sulle colonne del DataFrame.
    Mostra il nome della colonna, il tipo di dato, il numero di valori nulli, il numero di valori unici e un esempio di valore.
    """
    st.subheader("📋 Informazioni Colonne")

    col_info = []
    for col in df.columns:
        col_data = {
            'Colonna': col,
            'Tipo': str(df[col].dtype),
            'Valori Nulli': f"{df[col].isnull().sum():,}",
            'Valori Unici': f"{df[col].nunique():,}",
            'Completezza %': f"{((len(df) - df[col].isnull().sum()) / len(df) * 100):.1f}%"
        }

        # Aggiungi esempio valore
        if not df[col].dropna().empty:
            example = str(df[col].dropna().iloc[0])
            col_data['Esempio'] = example[:50] + "..." if len(example) > 50 else example
        else:
            col_data['Esempio'] = 'N/A'

        col_info.append(col_data)

    df_info = pd.DataFrame(col_info)
    st.dataframe(df_info, use_container_width=True, height=400)

def render_distribution_analysis(df):
    """
    Renderizza un'analisi delle distribuzioni delle variabili numeriche e categoriche.
    Mostra statistiche numeriche e grafici a barre per le variabili categoriche.
    """
    st.subheader("📊 Analisi Distribuzioni")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Statistiche numeriche
    if numeric_cols:
        st.write("**📈 Statistiche Variabili Numeriche:**")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    # Top valori categorici
    if cat_cols:
        st.write("**📊 Top Valori Variabili Categoriche:**")

        selected_cat = st.selectbox("Seleziona variabile categorica:", cat_cols)
        if selected_cat:
            value_counts = df[selected_cat].value_counts().head(15)

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = px.bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    orientation='h',
                    title=f'Top 15 valori - {selected_cat}'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("**Conteggi:**")
                for val, count in value_counts.items():
                    pct = (count / len(df)) * 100
                    st.write(f"**{val}:** {count:,} ({pct:.1f}%)")

def render_correlation_analysis(df):
    """
    Renderizza un'analisi delle correlazioni tra le variabili numeriche.
    Mostra una matrice delle correlazioni e un elenco delle correlazioni più forti.
    """
    st.subheader("🔗 Analisi Correlazioni")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.info("ℹ️ Servono almeno 2 variabili numeriche per l'analisi delle correlazioni")
        return

    # Matrice correlazioni
    corr_matrix = df[numeric_cols].corr()

    # Heatmap
    fig = px.imshow(
        corr_matrix,
        title="Matrice delle Correlazioni",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Top correlazioni
    st.write("**🔝 Correlazioni più forti:**")

    # Estrai correlazioni (escludendo diagonale)
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            correlations.append({
                'Variabile 1': corr_matrix.columns[i],
                'Variabile 2': corr_matrix.columns[j],
                'Correlazione': corr_matrix.iloc[i, j]
            })

    # Ordina per valore assoluto
    correlations_df = pd.DataFrame(correlations)
    correlations_df['Abs_Corr'] = correlations_df['Correlazione'].abs()
    correlations_df = correlations_df.sort_values('Abs_Corr', ascending=False).head(10)

    st.dataframe(correlations_df[['Variabile 1', 'Variabile 2', 'Correlazione']], use_container_width=True)

def render_visualizations(df):
    """
    Renderizza grafici e visualizzazioni per le variabili numeriche e categoriche.
    Permette di selezionare più variabili per i grafici.
    """
    st.subheader("📈 Grafici e Visualizzazioni")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Istogrammi variabili numeriche
    if numeric_cols:
        st.write("**📊 Distribuzioni Numeriche:**")

        selected_numeric = st.multiselect("Seleziona variabili numeriche:", numeric_cols)
        if selected_numeric:
            for col in selected_numeric:
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.histogram(df, x=col, title=f'Istogramma - {col}')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.box(df, y=col, title=f'Box Plot - {col}')
                    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot se ci sono almeno 2 variabili numeriche
    if len(numeric_cols) >= 2:
        st.write("**🎯 Scatter Plot:**")

        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Variabile X:", numeric_cols)
        with col2:
            y_vars = st.multiselect("Variabili Y:", [col for col in numeric_cols if col != x_var])

        if x_var and y_vars:
            for y_var in y_vars:
                # Colore per variabile categorica se disponibile
                color_var = None
                if cat_cols:
                    color_var = st.selectbox("Colore per:", ['Nessuno'] + cat_cols)
                    if color_var == 'Nessuno':
                        color_var = None

                fig = px.scatter(df, x=x_var, y=y_var, color=color_var,
                               title=f'Scatter Plot: {x_var} vs {y_var}')
                st.plotly_chart(fig, use_container_width=True)

def render_raw_data_viewer(df):
    """
    Renderizza un visualizzatore per i dati grezzi.
    Permette di filtrare e scaricare i dati.
    """
    st.header("📋 Visualizzatore Dati Grezzi")

    if df is None or df.empty:
        st.warning("⚠️ Nessun dato da visualizzare")
        return

    # Filtri
    st.subheader("🔍 Filtri")

    col1, col2 = st.columns(2)

    with col1:
        # Ricerca testuale
        search_term = st.text_input("🔍 Cerca nei dati:")

    with col2:
        # Selezione colonne
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("📊 Seleziona colonne:", all_columns, default=all_columns[:10])

    # Applica filtri
    display_df = df.copy()

    if search_term:
        mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        display_df = df[mask]
        st.info(f"📊 Trovate {len(display_df):,} righe contenenti '{search_term}'")

    if selected_columns:
        display_df = display_df[selected_columns]

    # Mostra dati
    st.subheader(f"📊 Dati ({len(display_df):,} righe)")
    st.dataframe(display_df, use_container_width=True, height=500)

    # Download
    if not display_df.empty:
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Scarica CSV",
            data=csv,
            file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    """
    Funzione principale dell'app Streamlit.
    Gestisce la connessione a BigQuery, il recupero delle tabelle e la visualizzazione dei dati.
    """
    st.title("🎓 Student Analytics Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Configurazione")

    # Inizializzazione BigQuery
    with st.spinner("🔄 Connessione a BigQuery..."):
        client, status = init_bigquery_client()

    # Mostra status connessione
    if "✅" in status:
        st.sidebar.success(status)
    else:
        st.sidebar.error(status)
        st.error("❌ **Impossibile connettersi a BigQuery**")
        st.info("🔧 **Verifica:**")
        st.write("1. File credentials.json presente")
        st.write("2. Credenziali Google Cloud valide")
        st.write("3. Permessi di accesso al progetto e dataset")
        st.stop()

    # Recupera tabelle
    with st.spinner("📊 Recupero tabelle..."):
        tables, table_status = get_all_tables()

    if not tables:
        st.error(f"❌ **Errore nel recupero tabelle:** {table_status}")
        st.stop()

    st.sidebar.success(f"✅ Trovate {len(tables)} tabelle")

    # Selezione modalità
    mode = st.sidebar.radio(
        "📋 Modalità:",
        ["🏠 Panoramica Tabelle", "🔍 Analisi Dettagliata"]
    )

    if mode == "🏠 Panoramica Tabelle":
        render_tables_overview(tables)

    else:  # Analisi Dettagliata
        # Selezione tabella
        table_options = {f"{t['type']} {t['name']}": t for t in tables}
        selected_table_name = st.sidebar.selectbox("📊 Seleziona tabella:", list(table_options.keys()))
        selected_table = table_options[selected_table_name]

        # Limite righe
        row_limit = st.sidebar.slider("📏 Limite righe:", 100, 5000, 1000, 100)

        # Pulsante refresh
        if st.sidebar.button("🔄 Aggiorna Cache"):
            st.cache_data.clear()
            st.rerun()

        # Carica dati
        with st.spinner(f"⏳ Caricamento {selected_table['name']}..."):
            df, load_status = load_table_data(selected_table['id'], row_limit)

        if "✅" in load_status:
            st.success(load_status)
        else:
            st.error(load_status)

        # Tabs per analisi
        if df is not None and not df.empty:
            tab1, tab2 = st.tabs(["📈 Analisi", "📋 Dati Grezzi"])

            with tab1:
                render_data_analysis(df, selected_table)

            with tab2:
                render_raw_data_viewer(df)

    # Footer
    st.markdown("---")
    st.markdown(
        "🎓 **Student Analytics Dashboard** | "
        f"📊 Dataset: `{DATASET_ID}` | "
        f"🏗️ Progetto: `{PROJECT_ID}`"
    )

if __name__ == "__main__":
    main()
