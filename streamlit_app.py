# ========================================
# STUDENT ANALYTICS PLATFORM - VERSIONE OTTIMIZZATA
# ========================================
"""
Piattaforma di analisi dati per studenti universitari.
Questo sistema permette di:
- Connettersi a BigQuery per recuperare dati
- Esplorare tabelle e dataset
- Eseguire analisi statistiche avanzate
- Creare visualizzazioni interattive
- Analizzare predizioni di abbandono e clustering

Autore: Student Analytics Team
Data: 2025
Versione: 3.0 - Ottimizzata
"""

# ========================================
# IMPORTAZIONE LIBRERIE
# ========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.cloud import bigquery
from google.oauth2 import service_account
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import scipy.stats as stats
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ========================================
# CONFIGURAZIONE GLOBALE
# ========================================
PROJECT_ID = "laboratorio-ai-460517"
DATASET_ID = "dataset"

st.set_page_config(
    page_title="Student Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# GESTIONE CONNESSIONE BIGQUERY (OTTIMIZZATA)
# ========================================
@st.cache_resource
def get_bigquery_client():
    """Inizializza client BigQuery con gestione errori migliorata."""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        # Test di connessione
        test_query = f"SELECT table_name FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.TABLES` LIMIT 1"
        client.query(test_query).result()
        return client
    except KeyError:
        st.error("❌ Credenziali GCP non trovate nei secrets")
        st.info("💡 Configura 'gcp_service_account' nei secrets di Streamlit")
        return None
    except Exception as e:
        st.error(f"❌ Errore connessione BigQuery: {str(e)}")
        return None

@st.cache_data(ttl=600)
def get_tables():
    """Recupera lista tabelle con informazioni aggiuntive dal dataset."""
    client = get_bigquery_client()
    if not client:
        return []
    try:
        query = f"""
        SELECT 
            table_name,
            row_count,
            size_bytes,
            creation_time,
            last_modified_time
        FROM `{PROJECT_ID}.{DATASET_ID}.__TABLES__`
        ORDER BY table_name
        """
        result = client.query(query).to_dataframe()
        return result.to_dict('records')
    except Exception:
        # Fallback semplice se __TABLES__ non è disponibile
        try:
            dataset_ref = client.get_dataset(f"{PROJECT_ID}.{DATASET_ID}")
            tables = [{'table_name': table.table_id} for table in client.list_tables(dataset_ref)]
            return tables
        except Exception as e:
            st.error(f"❌ Errore nel recupero delle tabelle: {str(e)}")
            return []

@st.cache_data(ttl=300)
def load_table_data_optimized(table_name, limit=1000, offset=0, columns=None, where_clause=None):
    """
    Versione ottimizzata del caricamento dati con filtri.
    - table_name: nome della tabella su BigQuery
    - limit, offset: paginazione
    - columns: lista colonne da selezionare o None per tutte
    - where_clause: stringa SQL per clausola WHERE (senza 'WHERE')
    """
    client = get_bigquery_client()
    if not client:
        return None
    try:
        select_clause = "*" if not columns else ", ".join(columns)
        where_sql = f"WHERE {where_clause}" if where_clause else ""
        query = f"""
        SELECT {select_clause}
        FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`
        {where_sql}
        ORDER BY 1
        LIMIT {limit}
        OFFSET {offset}
        """
        job_config = bigquery.QueryJobConfig()
        job_config.use_query_cache = True
        job_config.maximum_bytes_billed = 500 * 1024 * 1024  # Limite 500MB
        df = client.query(query, job_config=job_config).to_dataframe()
        return df
    except Exception as e:
        st.error(f"❌ Errore caricamento dati: {str(e)}")
        return None

# ========================================
# FUNZIONI DI ANALISI AVANZATA
# ========================================
def show_advanced_statistics(df):
    """Analisi statistiche avanzate con test di ipotesi."""
    st.subheader("📊 Statistiche Avanzate")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 1:
        st.warning("Nessuna colonna numerica per analisi avanzate")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Test Normalità", "Test Correlazione", "ANOVA", "Regressione"])

    # ------- Test di Normalità -------
    with tab1:
        st.write("**Test di Normalità (Shapiro-Wilk)**")
        selected_col = st.selectbox("Seleziona variabile:", numeric_cols, key="normality_test")
        if selected_col:
            data = df[selected_col].dropna()
            if len(data) < 4:
                st.info("Occorrono almeno 4 osservazioni per effettuare il test.")
            else:
                if len(data) > 5000:
                    data = data.sample(5000, random_state=42)
                    st.info("Campione ridotto a 5000 osservazioni per il test")
                stat, p_value = stats.shapiro(data)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Statistica W", f"{stat:.4f}")
                    st.metric("P-value", f"{p_value:.6f}")
                    alpha = 0.05
                    if p_value > alpha:
                        st.success("✅ I dati seguono una distribuzione normale")
                    else:
                        st.warning("⚠️ I dati NON seguono una distribuzione normale")
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    stats.probplot(data, dist="norm", plot=ax)
                    ax.set_title(f"Q-Q Plot - {selected_col}")
                    st.pyplot(fig)

    # ------- Test di Correlazione -------
    with tab2:
        st.write("**Test di Correlazione (Pearson)**")
        if len(numeric_cols) < 2:
            st.info("Servono almeno 2 variabili numeriche per questo test.")
        else:
            col1_select = st.selectbox("Prima variabile:", numeric_cols, key="corr_var1")
            col2_select = st.selectbox("Seconda variabile:", [c for c in numeric_cols if c != col1_select], key="corr_var2")
            if col1_select and col2_select:
                data1 = df[col1_select].dropna()
                data2 = df[col2_select].dropna()
                valid_idx = data1.index.intersection(data2.index)
                if len(valid_idx) < 3:
                    st.info("Occorrono almeno 3 coppie di valori non nulli.")
                else:
                    corr_coef, p_value = stats.pearsonr(data1[valid_idx], data2[valid_idx])
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Coefficiente r", f"{corr_coef:.4f}")
                        st.metric("P-value", f"{p_value:.6f}")
                        if p_value < 0.05:
                            st.success("✅ Correlazione statisticamente significativa")
                        else:
                            st.warning("⚠️ Correlazione NON significativa")
                    with c2:
                        fig = px.scatter(
                            x=data1[valid_idx],
                            y=data2[valid_idx],
                            title=f"{col1_select} vs {col2_select}",
                            trendline="ols"
                        )
                        fig.update_xaxes(title=col1_select)
                        fig.update_yaxes(title=col2_select)
                        st.plotly_chart(fig, use_container_width=True)

    # ------- Analisi della Varianza (ANOVA) -------
    with tab3:
        st.write("**ANOVA (Analisi della Varianza)**")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cat_cols or not numeric_cols:
            st.info("Servono almeno una variabile categorica e una numerica.")
        else:
            numeric_var = st.selectbox("Variabile numerica:", numeric_cols, key="anova_numeric")
            cat_var = st.selectbox("Variabile categorica:", cat_cols, key="anova_cat")
            if numeric_var and cat_var:
                groups = [df[df[cat_var] == grp][numeric_var].dropna() for grp in df[cat_var].dropna().unique()]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) < 2:
                    st.info("Occorrono almeno 2 gruppi non vuoti.")
                else:
                    try:
                        f_stat, p_val = stats.f_oneway(*groups)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("F-statistica", f"{f_stat:.4f}")
                            st.metric("P-value", f"{p_val:.6f}")
                            if p_val < 0.05:
                                st.success("✅ Differenze significative tra gruppi")
                            else:
                                st.warning("⚠️ Nessuna differenza significativa")
                        with col2:
                            fig = px.box(
                                df,
                                x=cat_var,
                                y=numeric_var,
                                title=f"Box plot di {numeric_var} per {cat_var}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Errore nell'ANOVA: {str(e)}")

    # ------- Regressione Lineare Semplice -------
    with tab4:
        st.write("**Regressione Lineare Semplice**")
        if len(numeric_cols) < 2:
            st.info("Servono almeno 2 variabili numeriche.")
        else:
            x_var = st.selectbox("Variabile indipendente (X):", numeric_cols, key="reg_x")
            y_var = st.selectbox("Variabile dipendente (Y):", [c for c in numeric_cols if c != x_var], key="reg_y")
            if x_var and y_var:
                reg_df = df[[x_var, y_var]].dropna()
                if len(reg_df) < 3:
                    st.info("Occorrono almeno 3 osservazioni non nulle.")
                else:
                    slope, intercept, r_val, p_val, std_err = stats.linregress(reg_df[x_var], reg_df[y_var])
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R²", f"{r_val**2:.4f}")
                        st.metric("Pendenza", f"{slope:.4f}")
                        st.metric("Intercetta", f"{intercept:.4f}")
                        st.metric("P-value", f"{p_val:.6f}")
                        st.write(f"**Equazione:** y = {slope:.3f}·x + {intercept:.3f}")
                    with col2:
                        fig = px.scatter(
                            reg_df,
                            x=x_var,
                            y=y_var,
                            title=f"Regressione: {y_var} ~ {x_var}",
                            trendline="ols"
                        )
                        st.plotly_chart(fig, use_container_width=True)

def show_clustering_analysis(df):
    """Analisi di clustering con K-means."""
    st.subheader("🔍 Analisi di Clustering")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Servono almeno 2 variabili numeriche per il clustering")
        return

    selected_vars = st.multiselect(
        "Seleziona variabili per clustering:",
        numeric_cols,
        default=numeric_cols[:min(4, len(numeric_cols))]
    )
    if len(selected_vars) < 2:
        st.info("Seleziona almeno 2 variabili")
        return

    cluster_df = df[selected_vars].dropna()
    if len(cluster_df) < 10:
        st.warning("Troppo pochi dati per clustering affidabile")
        return

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Parametri Clustering**")
        n_clusters = st.slider("Numero cluster:", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)
        cluster_df["Cluster"] = labels
        st.write("**Statistiche Cluster**")
        counts = pd.Series(labels).value_counts().sort_index()
        for idx, cnt in counts.items():
            st.write(f"Cluster {idx}: {cnt} punti ({cnt/len(labels)*100:.1f}%)")
        st.metric("Inerzia", f"{kmeans.inertia_:.2f}")

    with col2:
        if len(selected_vars) == 2:
            fig = px.scatter(
                cluster_df,
                x=selected_vars[0],
                y=selected_vars[1],
                color="Cluster",
                title="K-means Clustering (2D)",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            for i, c in enumerate(centroids):
                fig.add_scatter(
                    x=[c[0]],
                    y=[c[1]],
                    mode="markers",
                    marker=dict(size=12, symbol="x", color="black"),
                    name=f"Centroide {i}"
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            pca = PCA(n_components=2)
            pca_res = pca.fit_transform(scaled)
            pca_df = pd.DataFrame({
                "PC1": pca_res[:, 0],
                "PC2": pca_res[:, 1],
                "Cluster": labels
            })
            fig = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color="Cluster",
                title=f"K-means + PCA (varianza sp. {pca.explained_variance_ratio_.sum():.1%})",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"**Varianza spiegata**: PC1 = {pca.explained_variance_ratio_[0]:.1%}, "
                     f"PC2 = {pca.explained_variance_ratio_[1]:.1%}")

    st.write("**Metodo del Gomito**")
    with st.expander("Analizza numero ottimale di cluster"):
        max_k = min(10, len(cluster_df) // 2)
        k_range = list(range(1, max_k + 1))
        inertias = []
        for k in k_range:
            km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_temp.fit(scaled)
            inertias.append(km_temp.inertia_)
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=k_range,
            y=inertias,
            mode="lines+markers",
            name="Inerzia"
        ))
        fig_elbow.update_layout(
            title="Metodo del Gomito per K-means",
            xaxis_title="Numero di Cluster (k)",
            yaxis_title="Inerzia",
            height=400
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

def show_prediction_analysis(df):
    """Analisi predittiva con Random Forest."""
    st.subheader("🎯 Analisi Predittiva")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Servono almeno 2 variabili numeriche per analisi predittiva")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Configurazione Modello**")
        target_var = st.selectbox(
            "Variabile target (da predire):",
            numeric_cols + cat_cols,
            key="prediction_target"
        )
        if not target_var:
            return
        available_feats = [c for c in numeric_cols if c != target_var]
        selected_feats = st.multiselect(
            "Variabili predittive:",
            available_feats,
            default=available_feats[:min(5, len(available_feats))]
        )
        if len(selected_feats) < 1:
            st.info("Seleziona almeno una variabile predittiva")
            return
        test_size = st.slider("% dati per test:", 10, 40, 20) / 100
        n_estimators = st.slider("Numero alberi Random Forest:", 10, 200, 100)

    with col2:
        model_df = df[selected_feats + [target_var]].dropna()
        if len(model_df) < 50:
            st.warning("Troppo pochi dati per modello affidabile (min 50)")
            return
        X = model_df[selected_feats]
        y = model_df[target_var]
        is_classification = (y.dtype == 'object'
                             or y.dtype.name == 'category'
                             or y.dtype == 'bool'
                             or y.nunique() <= 10)
        try:
            if is_classification:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=42,
                    max_depth=10
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = model.score(X_test, y_test)
                st.success(f"**Accuratezza (Accuracy):** {acc:.3f}")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title="Matrice di Confusione",
                    labels=dict(x="Predetto", y="Reale")
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                with st.expander("Report di Classificazione"):
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(3))
            else:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import mean_squared_error, r2_score
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=42,
                    max_depth=10
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("R² Score", f"{r2:.3f}")
                with c2:
                    st.metric("RMSE", f"{rmse:.3f}")
                fig_scatter = px.scatter(
                    x=y_test,
                    y=y_pred,
                    title="Predizioni vs Valori Reali",
                    labels={'x': 'Valori Reali', 'y': 'Predizioni'}
                )
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig_scatter.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(dash="dash", color="red")
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.write("**Importanza delle Variabili**")
            feat_imp = pd.DataFrame({
                'Feature': selected_feats,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            fig_imp = px.bar(
                feat_imp,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Importanza delle Variabili"
            )
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

        except Exception as e:
            st.error(f"Errore nel modello predittivo: {str(e)}")

def show_correlation_analysis(df):
    """Analisi delle correlazioni tra variabili numeriche."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("⚠️ Servono almeno 2 variabili numeriche per l'analisi di correlazione")
        return

    st.subheader("🔗 Analisi Correlazioni")
    tab1, tab2, tab3 = st.tabs(["Matrice Correlazione", "Scatter Plot", "Correlazioni Forti"])

    # ------- Matrice di Correlazione -------
    with tab1:
        corr_method = st.selectbox(
            "Metodo di correlazione:",
            ["pearson", "spearman", "kendall"],
            help="Pearson: lineare, Spearman: monotona, Kendall: più robusto"
        )
        corr_matrix = df[numeric_cols].corr(method=corr_method)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title=f"Matrice di Correlazione ({corr_method.title()})",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    # ------- Scatter Plot Interattivo -------
    with tab2:
        st.write("**Scatter Plot Interattivo**")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_var = st.selectbox("Variabile X:", numeric_cols, key="scatter_x")
        with col2:
            y_var = st.selectbox("Variabile Y:", [c for c in numeric_cols if c != x_var], key="scatter_y")
        with col3:
            color_var = st.selectbox("Colora per (opzionale):", ["Nessuno"] + df.columns.tolist(), key="scatter_color")
        if x_var and y_var:
            color_col = None if color_var == "Nessuno" else color_var
            fig_scatter = px.scatter(
                df,
                x=x_var,
                y=y_var,
                color=color_col,
                title=f"{x_var} vs {y_var}",
                trendline="ols"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ------- Correlazioni Forti -------
    with tab3:
        st.write("**Correlazioni Forti (assoluto ≥ soglia)**")
        corr_matrix = df[numeric_cols].corr(method="pearson")
        threshold = st.slider("Soglia correlazione (valore assoluto):", 0.1, 0.9, 0.5, 0.1)
        strong_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if not pd.isna(corr_val) and abs(corr_val) >= threshold:
                    strength = "Forte" if abs(corr_val) > 0.7 else "Media"
                    strong_pairs.append({
                        "Var1": corr_matrix.columns[i],
                        "Var2": corr_matrix.columns[j],
                        "Corr.": corr_val,
                        "Forza": strength
                    })
        if strong_pairs:
            strong_df = pd.DataFrame(strong_pairs).sort_values("Corr.", key=abs, ascending=False)
            st.dataframe(strong_df, use_container_width=True)
        else:
            st.info("Nessuna coppia di variabili supera la soglia selezionata.")

# ========================================
# MAIN: INTERFACCIA PRINCIPALE
# ========================================
def main():
    st.title("📊 Student Analytics Platform")
    st.markdown("""
    Benvenuto nella piattaforma di analisi dati per studenti universitari.  
    Usa il menu a sinistra per:
    1. Selezionare la tabella dal dataset BigQuery  
    2. Esplorare i dati  
    3. Eseguire analisi statistiche, clustering e predizioni  
    """)

    # Sidebar: selezione tabella
    tables_info = get_tables()
    if not tables_info:
        st.error("Nessuna tabella disponibile nel dataset. Controlla la connessione.")
        return

    table_names = [tbl["table_name"] for tbl in tables_info]
    selected_table = st.sidebar.selectbox("Seleziona tabella:", table_names)
    limit = st.sidebar.number_input("Numero di righe da caricare:", min_value=100, max_value=10000, value=1000, step=100)
    offset = st.sidebar.number_input("Offset (per paginazione):", min_value=0, value=0, step=100)
    st.sidebar.markdown("---")

    # Caricamento dati
    with st.spinner("Caricamento dati in corso..."):
        df = load_table_data_optimized(table_name=selected_table, limit=limit, offset=offset)
    if df is None:
        st.error("Errore nel caricamento dei dati.")
        return
    if df.empty:
        st.warning("Tabella selezionata vuota o con i parametri scelti non ci sono righe.")
        return

    # Mostra informazioni sul dataset
    st.header(f"Dati: {selected_table}")
    st.markdown(f"- **Righe caricate:** {df.shape[0]:,}  \n- **Colonne totali:** {df.shape[1]:,}")
    with st.expander("Anteprima del dataset"):
        st.dataframe(df.head(20), use_container_width=True)

    # Sezione Analisi Descrittiva
    st.header("1. Analisi Descrittiva")
    st.subheader("Statistica di riepilogo")
    st.dataframe(df.describe(include='all').transpose(), use_container_width=True)

    # Sezione Filtri su colonne categoriche
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        with st.expander("Filtri avanzati (colmone categoriche)"):
            for col in categorical_cols:
                unique_vals = df[col].dropna().unique().tolist()
                if len(unique_vals) <= 20:
                    sel = st.multiselect(f"Seleziona valori '{col}':", options=unique_vals, default=unique_vals)
                    df = df[df[col].isin(sel)]
                else:
                    # Per colonne con molti valori, skip o filtro testuale
                    continue

    # Sezione Correlazione
    show_correlation_analysis(df)

    # Sezione Statistiche Avanzate
    show_advanced_statistics(df)

    # Sezione Clustering
    show_clustering_analysis(df)

    # Sezione Predittiva
    show_prediction_analysis(df)

    # Esporta dati filtrati
    st.header("📥 Esporta Dati")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Scarica CSV dei dati attuali",
        data=csv_data,
        file_name=f"{selected_table}_filtered.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
