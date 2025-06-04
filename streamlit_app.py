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

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurazione
PROJECT_ID = "laboratorio-ai-460517"
DATASET_ID = "dataset"
CACHE_TTL = 300

# Setup pagina Streamlit
st.set_page_config(
    page_title="🎓 Student Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_credentials():
    """Ottieni credenziali da Streamlit Secrets o file locale"""
    try:
        # Prova prima con Streamlit Secrets (per deploy)
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            logger.info("✅ Usando Streamlit Secrets")
            return dict(st.secrets["gcp_service_account"]), "Streamlit Secrets"
        
        # Fallback per ambiente locale
        elif os.path.exists('credentials.json'):
            logger.info("✅ Usando file credentials.json locale")
            with open('credentials.json', 'r') as f:
                return json.load(f), "File locale"
        
        else:
            return None, "Nessuna configurazione trovata"
            
    except Exception as e:
        logger.error(f"Errore nel recupero credenziali: {e}")
        return None, f"Errore: {str(e)}"

def check_credentials():
    """Verifica presenza e validità delle credenziali"""
    try:
        creds, source = get_credentials()
        
        if not creds:
            return False, "Credenziali mancanti"
        
        # Verifica campi obbligatori
        required_fields = ['type', 'project_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if field not in creds]
        
        if missing_fields:
            return False, f"Campi mancanti: {missing_fields}"
        
        return True, f"OK - {source}"
        
    except Exception as e:
        return False, str(e)

@st.cache_resource
def init_bigquery_client():
    """Inizializza client BigQuery con gestione errori migliorata"""
    try:
        # Verifica credenziali
        creds_valid, status = check_credentials()
        if not creds_valid:
            return None, f"❌ {status}"
        
        # Ottieni credenziali
        creds_dict, source = get_credentials()
        
        # Crea credenziali Google
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        # Crea client
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        
        # Test connessione semplice
        test_query = "SELECT 1 as test_connection"
        result = client.query(test_query).result()
        
        # Verifica risultato
        for row in result:
            if row.test_connection == 1:
                logger.info("✅ Connessione BigQuery stabilita")
                return client, f"✅ Connessione BigQuery OK ({source})"
        
        return None, "❌ Test connessione fallito"
        
    except Exception as e:
        error_msg = f"❌ Errore BigQuery: {str(e)}"
        logger.error(f"Errore dettagliato: {traceback.format_exc()}")
        return None, error_msg

@st.cache_data(ttl=CACHE_TTL)
def get_all_tables():
    """Recupera tutte le tabelle dal dataset"""
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
    """Classifica il tipo di tabella"""
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
    """Ottiene descrizione dettagliata della tabella"""
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
    """Carica dati da una tabella specifica"""
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
    """Renderizza panoramica delle tabelle"""
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
    """Renderizza analisi dettagliata dei dati"""
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
    """Renderizza informazioni dettagliate sulle colonne"""
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
    """Renderizza analisi delle distribuzioni"""
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
    """Renderizza analisi delle correlazioni"""
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
    """Renderizza grafici e visualizzazioni"""
    st.subheader("📈 Grafici e Visualizzazioni")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Istogrammi variabili numeriche
    if numeric_cols:
        st.write("**📊 Distribuzioni Numeriche:**")
        
        selected_numeric = st.selectbox("Seleziona variabile numerica:", numeric_cols)
        if selected_numeric:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x=selected_numeric, title=f'Istogramma - {selected_numeric}')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y=selected_numeric, title=f'Box Plot - {selected_numeric}')
                st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot se ci sono almeno 2 variabili numeriche
    if len(numeric_cols) >= 2:
        st.write("**🎯 Scatter Plot:**")
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Variabile X:", numeric_cols)
        with col2:
            y_var = st.selectbox("Variabile Y:", [col for col in numeric_cols if col != x_var])
        
        if x_var and y_var:
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
    """Renderizza visualizzatore dati grezzi"""
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
            file_name=f"export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_secrets_configuration():
    """Mostra guida per configurazione secrets"""
    st.error("❌ **Configurazione mancante per Streamlit Cloud**")
    
    st.info("🔧 **Configura i Secrets su Streamlit Cloud:**")
    
    st.write("**1. Vai su [share.streamlit.io](https://share.streamlit.io)**")
    st.write("**2. Trova la tua app e clicca sui tre puntini ⋮**")
    st.write("**3. Seleziona 'Settings' → 'Secrets'**")
    st.write("**4. Incolla questa configurazione:**")
    
    secrets_config = f"""[gcp_service_account]
type = "service_account"
project_id = "{PROJECT_ID}"
private_key_id = "your_private_key_id_here"
private_key = '''-----BEGIN PRIVATE KEY-----
your_private_key_here
-----END PRIVATE KEY-----'''
client_email = "your_service_account_email@{PROJECT_ID}.iam.gserviceaccount.com"
client_id = "your_client_id_here"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your_service_account_email%40{PROJECT_ID}.iam.gserviceaccount.com"
universe_domain = "googleapis.com"
"""
    
    st.code(secrets_config, language='toml')
    
    st.write("**5. Clicca 'Save' e riavvia l'app**")
    
    st.warning("⚠️ **Importante:** Usa le triple virgolette `'''` per la private_key come mostrato sopra!")

def main():
    """Funzione principale dell'app Streamlit"""
    try:
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
            
            # Mostra guida configurazione se le credenziali mancano
            if "Credenziali" in status or "mancanti" in status:
                show_secrets_configuration()
                return
        
        # Se non riusciamo a connetterci, mostra modalità offline
        if not client:
            st.warning("⚠️ **Modalità Offline** - Connessione BigQuery non disponibile")
            st.info("🔧 Controlla la configurazione delle credenziali nella sidebar")
            
            # Mostra dati di esempio
            st.header("📊 Dati di Esempio")
            sample_data = pd.DataFrame({
                'id': range(1, 101),
                'nome': [f'Studente_{i}' for i in range(1, 101)],
                'voto_medio': np.random.normal(7.5, 1.5, 100),
                'presenze': np.random.randint(80, 100, 100),
                'corso': np.random.choice(['Informatica', 'Matematica', 'Fisica'], 100)
            })
            
            st.dataframe(sample_data.head(20), use_container_width=True)
            return
        
        # Menu principale
        menu_options = ["📊 Panoramica Dataset", "📈 Analisi Dati", "📋 Visualizza Dati"]
        selected_menu = st.sidebar.selectbox("🔧 Seleziona Funzione:", menu_options)
        
        # Carica tabelle disponibili
        with st.spinner("📂 Caricamento tabelle..."):
            tables, tables_status = get_all_tables()
        
        if "❌" in tables_status:
            st.error(f"Errore nel caricamento tabelle: {tables_status}")
            return
        
        st.sidebar.info(tables_status)
        
        # Renderizza contenuto basato su selezione menu
        if selected_menu == "📊 Panoramica Dataset":
            render_tables_overview(tables)
        
        elif selected_menu == "📈 Analisi Dati":
            if not tables:
                st.warning("⚠️ Nessuna tabella disponibile per l'analisi")
                return
            
            # Selezione tabella
            table_names = [t['name'] for t in tables]
            selected_table = st.sidebar.selectbox("📋 Seleziona Tabella:", table_names)
            
            if selected_table:
                # Trova info tabella
                table_info = next((t for t in tables if t['name'] == selected_table), None)
                
                if table_info:
                    # Carica dati
                    with st.spinner(f"⏳ Caricamento dati da {selected_table}..."):
                        df, load_status = load_table_data(selected_table)
                    
                    st.sidebar.info(load_status)
                    
                    # Analisi dati
                    render_data_analysis(df, table_info)
        
        elif selected_menu == "📋 Visualizza Dati":
            if not tables:
                st.warning("⚠️ Nessuna tabella disponibile")
                return
            
            # Selezione tabella
            table_names = [t['name'] for t in tables]
            selected_table = st.sidebar.selectbox("📋 Seleziona Tabella:", table_names)
            
            if selected_table:
                # Limiti di caricamento
                limit = st.sidebar.number_input("📏 Limite righe:", min_value=100, max_value=10000, value=1000, step=100)
                
                # Carica dati
                with st.spinner(f"⏳ Caricamento {limit} righe da {selected_table}..."):
                    df, load_status = load_table_data(selected_table, limit)
                
                st.sidebar.info(load_status)
                
                # Visualizza dati
                render_raw_data_viewer(df)
        
        # Footer
        st.markdown("---")
        st.markdown("🎓 **Student Analytics Dashboard** - Powered by Streamlit & BigQuery")
        
    except Exception as e:
        st.error(f"❌ **Errore critico nell'applicazione:**")
        st.error(str(e))
        
        # Mostra traceback in expander per debug
        with st.expander("🔍 Dettagli errore (per debug)"):
            st.code(traceback.format_exc())
        
        st.info("💡 **Suggerimenti:**")
        st.write("- Verifica le credenziali BigQuery")
        st.write("- Controlla la connessione internet")
        st.write("- Riavvia l'applicazione")

if __name__ == "__main__":
    main()
