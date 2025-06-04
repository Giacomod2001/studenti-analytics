import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
from google.oauth2 import service_account
import time

# ========================================
# CONFIGURAZIONE E STILE
# ========================================
PROJECT_ID = "laboratorio-ai-460517"
DATASET_ID = "dataset"

st.set_page_config(
    page_title="🎓 Student Analytics Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato per un look moderno
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .analysis-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# CONNESSIONE BIGQUERY
# ========================================
@st.cache_resource
def init_bigquery():
    """Connessione BigQuery ottimizzata"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        client.query("SELECT 1").result()
        return client, True
    except Exception as e:
        st.error(f"🔌 Errore connessione: {str(e)}")
        return None, False

@st.cache_data(ttl=300)
def load_table_list():
    """Carica lista tabelle disponibili"""
    client, connected = init_bigquery()
    if not connected:
        return []
    
    try:
        tables = []
        for table in client.list_tables(f"{PROJECT_ID}.{DATASET_ID}"):
            tables.append({
                'id': table.table_id,
                'name': get_display_name(table.table_id),
                'type': get_table_type(table.table_id),
                'icon': get_table_icon(table.table_id)
            })
        return sorted(tables, key=lambda x: x['type'])
    except:
        return []

def get_display_name(table_id):
    """Nomi user-friendly per le tabelle"""
    names = {
        'studenti': 'Dati Studenti Base',
        'studenti_churn_pred': 'Previsioni Abbandono',
        'studenti_cluster': 'Gruppi Studenti',
        'studenti_soddisfazione_btr': 'Livelli Soddisfazione',
        'feature_importance_studenti': 'Fattori Importanti',
        'student_churn_rf': 'Modello Previsioni',
        'student_kmeans': 'Modello Clustering'
    }
    return names.get(table_id, table_id.replace('_', ' ').title())

def get_table_type(table_id):
    """Categoria della tabella"""
    if 'churn' in table_id.lower():
        return 'prediction'
    elif 'cluster' in table_id.lower() or 'kmeans' in table_id.lower():
        return 'clustering'
    elif 'soddisfazione' in table_id.lower():
        return 'satisfaction'
    elif table_id == 'studenti':
        return 'base'
    else:
        return 'analysis'

def get_table_icon(table_id):
    """Icone per le tabelle"""
    icons = {
        'prediction': '🔮',
        'clustering': '👥',
        'satisfaction': '😊',
        'base': '📊',
        'analysis': '⚙️'
    }
    return icons.get(get_table_type(table_id), '📋')

@st.cache_data(ttl=300)
def load_data(table_id, limit=1000):
    """Carica dati da tabella"""
    client, connected = init_bigquery()
    if not connected:
        return None
    
    try:
        query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_id}` LIMIT {limit}"
        return client.query(query).to_dataframe()
    except Exception as e:
        st.error(f"❌ Errore caricamento {table_id}: {str(e)}")
        return None

# ========================================
# COMPONENTI UI MODERNI
# ========================================
def show_welcome_screen():
    """Schermata di benvenuto moderna"""
    st.markdown('<h1 class="main-header">🎓 Student Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analisi Intelligente dei Dati Studenteschi</p>', unsafe_allow_html=True)
    
    # Hero section con informazioni
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3 style="margin:0; text-align:center;">🚀 Benvenuto nella Piattaforma di Analytics</h3>
            <p style="margin:0.5rem 0; text-align:center;">
                Esplora i dati dei tuoi studenti con strumenti di analisi avanzati ma semplici da usare
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cards delle funzionalità principali
    st.subheader("🎯 Cosa Puoi Fare")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="analysis-card">
            <h4>🔮 Previsioni Smart</h4>
            <p>Identifica studenti a rischio abbandono prima che accada</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="analysis-card">
            <h4>👥 Raggruppa Studenti</h4>
            <p>Scopri pattern nascosti e crea gruppi omogenei</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="analysis-card">
            <h4>😊 Misura Soddisfazione</h4>
            <p>Monitora il livello di gradimento dei corsi</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="analysis-card">
            <h4>📈 Analisi Avanzate</h4>
            <p>Grafici interattivi e insights personalizzati</p>
        </div>
        """, unsafe_allow_html=True)

def show_metrics_dashboard(tables):
    """Dashboard metriche principali stile moderno"""
    st.subheader("📊 Panoramica Sistema")
    
    # Metriche principali con design moderno
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin:0;">{len(tables)}</h2>
            <p style="margin:0;">Tabelle Dati</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        prediction_tables = len([t for t in tables if t['type'] == 'prediction'])
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin:0;">{prediction_tables}</h2>
            <p style="margin:0;">Modelli Predittivi</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        analysis_tables = len([t for t in tables if t['type'] in ['clustering', 'satisfaction']])
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin:0;">{analysis_tables}</h2>
            <p style="margin:0;">Analisi Avanzate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin:0;">✅</h2>
            <p style="margin:0;">Sistema Attivo</p>
        </div>
        """, unsafe_allow_html=True)

def show_table_explorer(tables):
    """Explorer delle tabelle con design moderno"""
    st.subheader("🗂️ Esplora i Tuoi Dati")
    
    # Raggrupa tabelle per tipo
    table_groups = {}
    for table in tables:
        if table['type'] not in table_groups:
            table_groups[table['type']] = []
        table_groups[table['type']].append(table)
    
    # Mostra tabelle raggruppate
    type_names = {
        'base': '📊 Dati Base',
        'prediction': '🔮 Previsioni',
        'clustering': '👥 Raggruppamenti',
        'satisfaction': '😊 Soddisfazione',
        'analysis': '⚙️ Analisi'
    }
    
    for table_type, group_tables in table_groups.items():
        with st.expander(f"{type_names.get(table_type, table_type)} ({len(group_tables)} tabelle)", expanded=True):
            cols = st.columns(min(len(group_tables), 3))
            
            for i, table in enumerate(group_tables):
                with cols[i % 3]:
                    if st.button(f"{table['icon']} {table['name']}", key=f"btn_{table['id']}", use_container_width=True):
                        st.session_state.selected_table = table['id']
                        st.session_state.analysis_mode = True
                        st.rerun()

def analyze_churn_predictions(df):
    """Analisi previsioni abbandono con design moderno"""
    st.markdown('<h2 style="color: #1f77b4;">🔮 Analisi Previsioni Abbandono</h2>', unsafe_allow_html=True)
    
    # Spiegazione semplice
    st.markdown("""
    <div class="info-box">
        <h4 style="margin:0;">💡 Come Funziona</h4>
        <p style="margin:0.5rem 0;">
            Il sistema analizza voti, presenze e comportamenti per prevedere quali studenti potrebbero abbandonare.
            Una probabilità alta significa che lo studente ha bisogno di supporto immediato!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Trova colonna probabilità
    prob_cols = [col for col in df.columns if 'prob' in col.lower() or 'pred' in col.lower()]
    if not prob_cols:
        st.warning("⚠️ Colonna probabilità non trovata")
        return
    
    prob_col = prob_cols[0]
    
    # Calcola statistiche rischio
    high_risk = (df[prob_col] > 0.7).sum()
    medium_risk = ((df[prob_col] > 0.3) & (df[prob_col] <= 0.7)).sum()
    low_risk = (df[prob_col] <= 0.3).sum()
    
    # Metriche di rischio
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="warning-card">
            <h2 style="margin:0;">{high_risk}</h2>
            <p style="margin:0;">🚨 Alto Rischio</p>
            <small>Probabilità > 70%</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); padding: 1rem; border-radius: 10px; color: #2d3436; text-align: center;">
            <h2 style="margin:0;">{medium_risk}</h2>
            <p style="margin:0;">⚠️ Medio Rischio</p>
            <small>Probabilità 30-70%</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="success-card">
            <h2 style="margin:0;">{low_risk}</h2>
            <p style="margin:0;">✅ Basso Rischio</p>
            <small>Probabilità < 30%</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Grafico distribuzione
    st.subheader("📊 Distribuzione del Rischio")
    
    df['Categoria_Rischio'] = pd.cut(df[prob_col], 
                                   bins=[0, 0.3, 0.7, 1.0], 
                                   labels=['🟢 Basso', '🟡 Medio', '🔴 Alto'])
    
    fig = px.histogram(df, x='Categoria_Rischio', 
                      title='Distribuzione Studenti per Livello di Rischio',
                      color='Categoria_Rischio',
                      color_discrete_map={'🟢 Basso': '#00b894', '🟡 Medio': '#fdcb6e', '🔴 Alto': '#e84393'})
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top studenti a rischio
    if high_risk > 0:
        st.subheader("🚨 Studenti da Monitorare")
        high_risk_students = df[df[prob_col] > 0.7].nlargest(10, prob_col)
        
        for idx, student in high_risk_students.iterrows():
            with st.expander(f"Studente ID: {student.get('student_id', idx)} - Rischio: {student[prob_col]:.1%}"):
                cols = st.columns(len([col for col in student.index if col != prob_col]))
                for i, (col, val) in enumerate(student.items()):
                    if col != prob_col:
                        with cols[i % len(cols)]:
                            st.metric(col.replace('_', ' ').title(), val)

def analyze_student_clusters(df):
    """Analisi clustering studenti ottimizzata"""
    st.markdown('<h2 style="color: #1f77b4;">👥 Analisi Gruppi Studenti</h2>', unsafe_allow_html=True)
    
    # Trova colonna cluster
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
    if not cluster_cols:
        st.warning("⚠️ Colonna cluster non trovata")
        return
    
    cluster_col = cluster_cols[0]
    unique_clusters = sorted(df[cluster_col].unique())
    
    # Statistiche rapide
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Totale Studenti", len(df))
    with col2:
        st.metric("🏷️ Numero Gruppi", len(unique_clusters))
    with col3:
        avg_size = len(df) // len(unique_clusters)
        st.metric("📊 Dimensione Media", avg_size)
    
    # Selector cluster
    st.subheader("🎯 Scegli Gruppi da Confrontare")
    selected_clusters = st.multiselect(
        "Seleziona uno o più gruppi:",
        options=unique_clusters,
        default=unique_clusters[:2] if len(unique_clusters) >= 2 else unique_clusters,
        format_func=lambda x: f"Gruppo {x}"
    )
    
    if not selected_clusters:
        st.warning("Seleziona almeno un gruppo!")
        return
    
    # Filtra dati
    filtered_df = df[df[cluster_col].isin(selected_clusters)]
    
    # Analisi per ogni cluster
    for cluster_id in selected_clusters:
        cluster_data = filtered_df[filtered_df[cluster_col] == cluster_id]
        
        st.markdown(f"### 🏷️ Gruppo {cluster_id}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Studenti", len(cluster_data))
        with col2:
            pct = (len(cluster_data) / len(df)) * 100
            st.metric("📊 % del Totale", f"{pct:.1f}%")
        
        # Caratteristiche principali
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            with col3:
                main_feature = numeric_cols[1]
                avg_val = cluster_data[main_feature].mean()
                st.metric(f"Avg {main_feature}", f"{avg_val:.2f}")
        
        st.markdown("---")
    
    # Grafico confronto
    if len(selected_clusters) > 1:
        st.subheader("📊 Confronto Cluster")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Variabile X:", numeric_cols, key="cluster_x")
            with col2:
                y_var = st.selectbox("Variabile Y:", [c for c in numeric_cols if c != x_var], key="cluster_y")
            
            fig = px.scatter(
                filtered_df, 
                x=x_var, 
                y=y_var,
                color=cluster_col,
                title=f'{x_var} vs {y_var} per Cluster',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def show_general_analysis(df, table_name):
    """Analisi generale ottimizzata"""
    st.markdown(f'<h2 style="color: #1f77b4;">📈 Analisi: {table_name}</h2>', unsafe_allow_html=True)
    
    # Statistiche base
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Righe", f"{len(df):,}")
    with col2:
        st.metric("📊 Colonne", len(df.columns))
    with col3:
        missing = df.isnull().sum().sum()
        total = len(df) * len(df.columns)
        st.metric("❓ Mancanti", f"{(missing/total)*100:.1f}%")
    with col4:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("🔢 Numeriche", numeric_cols)
    
    # Statistiche descrittive
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.subheader("📊 Statistiche Descrittive")
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
    
    # Grafici
    if len(numeric_cols) >= 2:
        st.subheader("📈 Analisi Grafica")
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Asse X:", numeric_cols)
        with col2:
            y_var = st.selectbox("Asse Y:", [c for c in numeric_cols if c != x_var])
        
        if x_var and y_var:
            fig = px.scatter(df, x=x_var, y=y_var, title=f'{x_var} vs {y_var}')
            st.plotly_chart(fig, use_container_width=True)
    
    # Dati raw
    st.subheader("🔍 Dati Raw")
    st.dataframe(df.head(100), use_container_width=True)

# ========================================
# APP PRINCIPALE
# ========================================
def main():
    # Inizializzazione sessione
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = False
    if 'selected_table' not in st.session_state:
        st.session_state.selected_table = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Controllo Sistema")
        
        # Status connessione
        client, connected = init_bigquery()
        if connected:
            st.success("🟢 Sistema Online")
        else:
            st.error("🔴 Connessione Fallita")
            st.stop()
        
        # Pulsante reset
        if st.button("🔄 Reset Vista", use_container_width=True):
            st.session_state.analysis_mode = False
            st.session_state.selected_table = None
            st.rerun()
        
        # Info sistema
        st.markdown("---")
        st.markdown("### 📊 Info Sistema")
        st.info(f"**Progetto:** {PROJECT_ID}\n**Dataset:** {DATASET_ID}")
    
    # Carica tabelle
    tables = load_table_list()
    
    if not tables:
        st.error("❌ Nessuna tabella trovata")
        return
    
    # Modalità analisi
    if st.session_state.analysis_mode and st.session_state.selected_table:
        # Pulsante indietro
        if st.button("← Torna alla Dashboard", key="back_btn"):
            st.session_state.analysis_mode = False
            st.session_state.selected_table = None
            st.rerun()
        
        # Carica e analizza dati
        with st.spinner("🔄 Caricamento dati..."):
            df = load_data(st.session_state.selected_table)
        
        if df is not None:
            table_name = get_display_name(st.session_state.selected_table)
            table_type = get_table_type(st.session_state.selected_table)
            
            # Analisi specifica per tipo
            if table_type == 'prediction':
                analyze_churn_predictions(df)
            elif table_type == 'clustering':
                analyze_student_clusters(df)
            else:
                show_general_analysis(df, table_name)
    else:
        # Homepage
        show_welcome_screen()
        show_metrics_dashboard(tables)
        show_table_explorer(tables)

if __name__ == "__main__":
    main()
