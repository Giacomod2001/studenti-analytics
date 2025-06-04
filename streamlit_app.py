import streamlit as st
import pandas as pd
import plotly.express as px
from google.cloud import bigquery
from google.oauth2 import service_account

# Configurazione della pagina
st.set_page_config(
    page_title="Student Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Titolo della dashboard
st.title("Student Analytics Dashboard")

# Descrizione della dashboard
st.markdown("""
Questa dashboard è progettata per aiutare le università a comprendere la soddisfazione degli studenti,
i fattori che influenzano questa soddisfazione e il rischio di abbandono dei corsi di studio.
""")

# Funzione per ottenere il client BigQuery
@st.cache_resource
def get_bigquery_client():
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        return client
    except Exception as e:
        st.error(f"Errore nella connessione a BigQuery: {e}")
        return None

# Funzione per caricare i dati da BigQuery
@st.cache_data
def load_data(query):
    client = get_bigquery_client()
    if client:
        try:
            query_job = client.query(query)
            data = query_job.to_dataframe()
            return data
        except Exception as e:
            st.error(f"Errore nell'esecuzione della query: {e}")
            return None
    return None

# Query per ottenere i dati
query = """
SELECT
    student_id,
    soddisfazione,
    abbandono,
    partecipazione_eventi,
    ore_studio_settimanali,
    media_voti,
    corso_di_studi,
    anno_iscrizione
FROM `laboratorio-ai-460517.dataset.studenti`
LIMIT 1000
"""

# Caricamento dei dati
data = load_data(query)

if data is not None:
    # Sidebar per la selezione delle variabili
    st.sidebar.header("Filtri")

    # Selezione delle variabili numeriche
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_numeric_columns = st.sidebar.multiselect("Seleziona variabili numeriche", numeric_columns, default=numeric_columns)

    # Selezione delle variabili categoriche
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    selected_categorical_columns = st.sidebar.multiselect("Seleziona variabili categoriche", categorical_columns, default=categorical_columns)

    # Filtro per corso di studi
    selected_corso = st.sidebar.selectbox("Seleziona corso di studi", data['corso_di_studi'].unique())

    # Filtro per anno di iscrizione
    selected_anno = st.sidebar.selectbox("Seleziona anno di iscrizione", data['anno_iscrizione'].unique())

    # Applicazione dei filtri
    filtered_data = data[(data['corso_di_studi'] == selected_corso) & (data['anno_iscrizione'] == selected_anno)]

    # Sezione KPI
    st.header("Indicatori Chiave di Prestazione (KPI)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Soddisfazione Media")
        st.write(f"{filtered_data['soddisfazione'].mean():.2f}/10")
        st.markdown("**Descrizione**: Questo indicatore mostra la soddisfazione media degli studenti.")
    with col2:
        st.subheader("Tasso di Abbandono")
        st.write(f"{filtered_data['abbandono'].mean():.2%}")
        st.markdown("**Descrizione**: Questo indicatore mostra il tasso medio di abbandono.")
    with col3:
        st.subheader("Partecipazione agli Eventi")
        st.write(f"{filtered_data['partecipazione_eventi'].mean():.2f}/10")
        st.markdown("**Descrizione**: Questo indicatore mostra la partecipazione media agli eventi.")

    # Sezione Grafici
    st.header("Analisi Dettagliata")

    # Grafico della soddisfazione degli studenti
    st.subheader("Soddisfazione degli Studenti")
    fig_soddisfazione = px.histogram(filtered_data, x='soddisfazione', title="Distribuzione della Soddisfazione degli Studenti")
    st.plotly_chart(fig_soddisfazione)
    st.markdown("**Interpretazione**: Questo grafico mostra la distribuzione della soddisfazione degli studenti.")

    # Grafico del tasso di abbandono
    st.subheader("Tasso di Abbandono")
    fig_abbandono = px.histogram(filtered_data, x='abbandono', title="Distribuzione del Tasso di Abbandono")
    st.plotly_chart(fig_abbandono)
    st.markdown("**Interpretazione**: Questo grafico mostra la distribuzione del tasso di abbandono.")

    # Grafico della partecipazione agli eventi
    st.subheader("Partecipazione agli Eventi")
    fig_partecipazione = px.histogram(filtered_data, x='partecipazione_eventi', title="Distribuzione della Partecipazione agli Eventi")
    st.plotly_chart(fig_partecipazione)
    st.markdown("**Interpretazione**: Questo grafico mostra la distribuzione della partecipazione agli eventi.")

    # Sezione Analisi delle Correlazioni
    st.header("Analisi delle Correlazioni")

    # Matrice di correlazione
    st.subheader("Matrice di Correlazione")
    corr_matrix = filtered_data[selected_numeric_columns].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Matrice di Correlazione")
    st.plotly_chart(fig_corr)
    st.markdown("**Interpretazione**: Questa matrice di correlazione mostra come le diverse variabili sono correlate tra loro.")

    # Sezione Conclusioni
    st.header("Conclusioni")

    st.markdown("""
    ### Sintesi delle Scoperte

    1. **Soddisfazione degli Studenti**:
       - I corsi di studi con la soddisfazione più alta sono quelli che probabilmente offrono un buon equilibrio tra carico di lavoro e supporto agli studenti.
       - **Raccomandazione**: Identificare le pratiche dei corsi con alta soddisfazione e applicarle agli altri corsi.

    2. **Tasso di Abbandono**:
       - I corsi di studi con il tasso di abbandono più alto potrebbero avere problemi specifici che necessitano di essere indagati.
       - **Raccomandazione**: Condurre indagini qualitative per comprendere le ragioni dell'abbandono e intervenire di conseguenza.

    3. **Partecipazione agli Eventi**:
       - La partecipazione agli eventi è un indicatore di coinvolgimento degli studenti nella vita universitaria.
       - **Raccomandazione**: Promuovere eventi e attività che possano aumentare la partecipazione e il senso di comunità tra gli studenti.
    """)

    # Sezione Guide e Tutorial
    st.header("Guide e Tutorial")

    st.markdown("""
    ### Come Utilizzare la Dashboard

    1. **Filtri**: Utilizza la sidebar per selezionare le variabili e i filtri desiderati. Puoi selezionare variabili numeriche e categoriche, e filtrare per corso di studi e anno di iscrizione.

    2. **KPI**: Questa sezione mostra i principali indicatori chiave di prestazione. Ogni KPI è accompagnato da una descrizione che spiega cosa rappresenta e come interpretarlo.

    3. **Analisi Dettagliata**: Questa sezione mostra grafici dettagliati che ti permettono di esplorare i dati in modo più approfondito. Ogni grafico è accompagnato da un'interpretazione che ti aiuta a comprendere i dati visualizzati.

    4. **Analisi delle Correlazioni**: Questa sezione mostra una matrice di correlazione che ti aiuta a comprendere come le diverse variabili sono correlate tra loro.

    5. **Conclusioni**: Questa sezione fornisce una sintesi delle principali conclusioni tratte dai dati e raccomandazioni su come agire in base a queste scoperte.
    """)
else:
    st.error("Impossibile caricare i dati da BigQuery.")
