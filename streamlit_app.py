import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from google.cloud import bigquery
from google.oauth2 import service_account

# Configurazione
PROJECT_ID = "laboratorio-ai-460517"
DATASET_ID = "dataset"

# Configurazione della pagina Streamlit
st.set_page_config(
    page_title="Student Analytics",
    page_icon="📊",
    layout="wide"
)

# Funzione per ottenere il client BigQuery
@st.cache_resource
def get_bigquery_client():
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        return client
    except Exception as e:
        st.error(f"Errore nella connessione a BigQuery: {e}")
        return None

# Funzione per ottenere l'elenco delle tabelle
@st.cache_data(ttl=600)
def get_tables():
    client = get_bigquery_client()
    if not client:
        return []

    try:
        dataset = client.get_dataset(f"{PROJECT_ID}.{DATASET_ID}")
        tables = [table.table_id for table in client.list_tables(dataset)]
        return tables
    except Exception as e:
        st.error(f"Errore nel recupero delle tabelle: {e}")
        return []

# Funzione per caricare i dati da una tabella
@st.cache_data(ttl=300)
def load_table_data(table_name, limit=1000):
    client = get_bigquery_client()
    if not client:
        return None

    try:
        query = f"""
        SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`
        LIMIT {limit}
        """
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"Errore nel caricamento dei dati: {e}")
        return None

# Funzione per eseguire una query personalizzata
@st.cache_data(ttl=300)
def execute_custom_query(query):
    client = get_bigquery_client()
    if not client:
        return None

    try:
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"Errore nell'esecuzione della query: {e}")
        return None

# Funzione per mostrare le statistiche di base
def show_basic_stats(df):
    st.subheader("Statistiche di Base")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Numero di righe", len(df))
    with col2:
        st.metric("Numero di colonne", len(df.columns))
    with col3:
        missing = df.isnull().sum().sum()
        st.metric("Valori mancanti", missing)

# Funzione per mostrare l'anteprima dei dati
def show_data_preview(df):
    st.subheader("Anteprima dei Dati")
    st.dataframe(df.head(20))

# Funzione per mostrare le informazioni sulle colonne
def show_column_info(df):
    st.subheader("Informazioni sulle Colonne")
    info_data = []
    for col in df.columns:
        info_data.append({
            'Colonna': col,
            'Tipo': str(df[col].dtype),
            'Non-null': df[col].count(),
            'Null': df[col].isnull().sum(),
            'Unici': df[col].nunique()
        })
    info_df = pd.DataFrame(info_data)
    st.dataframe(info_df)

# Funzione per mostrare l'analisi delle variabili numeriche
def show_numeric_analysis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("Nessuna colonna numerica trovata")
        return

    st.subheader("Analisi delle Variabili Numeriche")
    st.write("**Statistiche Descrittive**")
    st.dataframe(df[numeric_cols].describe())

    if len(numeric_cols) > 0:
        st.write("**Distribuzione delle Variabili**")
        selected_col = st.selectbox("Seleziona colonna:", numeric_cols, key="numeric_dist")
        fig = px.histogram(df, x=selected_col, title=f"Distribuzione di {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

# Funzione per mostrare l'analisi delle variabili categoriche
def show_categorical_analysis(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols:
        st.warning("Nessuna colonna categorica trovata")
        return

    st.subheader("Analisi delle Variabili Categoriche")
    selected_col = st.selectbox("Seleziona colonna:", cat_cols, key="cat_analysis")
    if selected_col:
        value_counts = df[selected_col].value_counts().head(20)
        st.write(f"**Top 20 valori per {selected_col}**")
        st.dataframe(value_counts)
        fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Distribuzione di {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

# Funzione per mostrare l'analisi di correlazione
def show_correlation_analysis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Servono almeno 2 colonne numeriche per l'analisi di correlazione")
        return

    st.subheader("Analisi di Correlazione")
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Matrice di Correlazione")
    st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) >= 2:
        st.write("**Scatter Plot**")
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Variabile X:", numeric_cols, key="scatter_x")
        with col2:
            y_var = st.selectbox("Variabile Y:", [c for c in numeric_cols if c != x_var], key="scatter_y")
        if x_var and y_var:
            fig = px.scatter(df, x=x_var, y=y_var, title=f"{x_var} vs {y_var}")
            st.plotly_chart(fig, use_container_width=True)

# Funzione per analizzare i dati di churn
def analyze_churn_data(df):
    st.subheader("Analisi delle Predizioni di Abbandono")
    prob_cols = [col for col in df.columns if 'prob' in col.lower() or 'pred' in col.lower()]
    if prob_cols:
        prob_col = st.selectbox("Seleziona colonna probabilità:", prob_cols)
        if prob_col:
            fig = px.histogram(df, x=prob_col, title=f"Distribuzione di {prob_col}")
            st.plotly_chart(fig, use_container_width=True)
            df['Risk_Category'] = pd.cut(df[prob_col], bins=[0, 0.3, 0.7, 1.0], labels=['Basso', 'Medio', 'Alto'])
            risk_counts = df['Risk_Category'].value_counts()
            st.write("**Distribuzione del Rischio**")
            st.dataframe(risk_counts)
            high_risk = df[df[prob_col] > 0.7].nlargest(10, prob_col)
            if len(high_risk) > 0:
                st.write("**Top 10 Studenti a Rischio**")
                st.dataframe(high_risk)

# Funzione per analizzare i dati di clustering
def analyze_cluster_data(df):
    st.subheader("Analisi di Clustering")
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
    if cluster_cols:
        cluster_col = st.selectbox("Seleziona colonna cluster:", cluster_cols)
        if cluster_col:
            cluster_counts = df[cluster_col].value_counts()
            st.write("**Distribuzione dei Cluster**")
            st.dataframe(cluster_counts)
            fig = px.bar(x=cluster_counts.index, y=cluster_counts.values, title="Distribuzione dei Cluster")
            st.plotly_chart(fig, use_container_width=True)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                st.write("**Confronto dei Cluster**")
                x_var = st.selectbox("Variabile X:", numeric_cols, key="cluster_x")
                y_var = st.selectbox("Variabile Y:", [c for c in numeric_cols if c != x_var], key="cluster_y")
                if x_var and y_var:
                    fig = px.scatter(df, x=x_var, y=y_var, color=cluster_col, title=f"{x_var} vs {y_var} per Cluster")
                    st.plotly_chart(fig, use_container_width=True)

# Funzione principale
def main():
    st.title("📊 Student Analytics Platform")
    st.write("Piattaforma semplice per l'analisi dei dati studenteschi")

    with st.sidebar:
        st.header("Controlli")
        client = get_bigquery_client()
        if client:
            st.success("✅ Connesso a BigQuery")
        else:
            st.error("❌ Errore di connessione")
            st.stop()
        mode = st.radio("Modalità:", ["Esplora Tabelle", "Query Personalizzata"])

    if mode == "Esplora Tabelle":
        show_table_explorer()
    else:
        show_custom_query()

# Funzione per esplorare le tabelle
def show_table_explorer():
    st.header("Esplora Tabelle")
    tables = get_tables()
    if not tables:
        st.warning("Nessuna tabella trovata")
        return
    selected_table = st.selectbox("Seleziona tabella:", tables)
    if selected_table:
        with st.spinner("Caricamento dati..."):
            df = load_table_data(selected_table)
        if df is not None:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Panoramica", "Numeriche", "Categoriche", "Correlazioni", "Specializzata"])
            with tab1:
                show_basic_stats(df)
                show_column_info(df)
                show_data_preview(df)
            with tab2:
                show_numeric_analysis(df)
            with tab3:
                show_categorical_analysis(df)
            with tab4:
                show_correlation_analysis(df)
            with tab5:
                if 'churn' in selected_table.lower():
                    analyze_churn_data(df)
                elif 'cluster' in selected_table.lower():
                    analyze_cluster_data(df)
                else:
                    st.info("Nessuna analisi specializzata disponibile per questa tabella")

# Funzione per eseguire query personalizzate
def show_custom_query():
    st.header("Query Personalizzata")
    query = st.text_area("Inserisci query SQL:", height=200, placeholder=f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.your_table` LIMIT 100")
    if st.button("Esegui Query"):
        if query.strip():
            with st.spinner("Esecuzione query..."):
                df = execute_custom_query(query)
            if df is not None:
                st.success(f"Query eseguita con successo! {len(df)} righe recuperate.")
                st.subheader("Risultati")
                show_basic_stats(df)
                st.dataframe(df)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.subheader("Analisi Rapida")
                    st.dataframe(df[numeric_cols].describe())
        else:
            st.warning("Inserisci una query valida")

if __name__ == "__main__":
    main()
