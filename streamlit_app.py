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
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import scipy.stats as stats
import matplotlib.pyplot as plt
import io
import base64

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
        
        # Test connessione più robusto
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
    """Recupera lista tabelle con informazioni aggiuntive."""
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
        
    except Exception as e:
        # Fallback al metodo originale
        try:
            dataset_ref = client.get_dataset(f"{PROJECT_ID}.{DATASET_ID}")
            tables = [{'table_name': table.table_id} for table in client.list_tables(dataset_ref)]
            return tables
        except Exception:
            st.error(f"❌ Errore nel recupero delle tabelle: {str(e)}")
            return []

@st.cache_data(ttl=300)
def load_table_data_optimized(table_name, limit=1000, offset=0, columns=None, where_clause=None):
    """Versione ottimizzata del caricamento dati con filtri."""
    client = get_bigquery_client()
    if not client:
        return None
    
    try:
        # Costruisce SELECT dinamicamente
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
        
        # Configurazione job ottimizzata
        job_config = bigquery.QueryJobConfig()
        job_config.use_query_cache = True
        job_config.maximum_bytes_billed = 500 * 1024 * 1024  # 500MB limit
        
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
    
    with tab1:
        st.write("**Test di Normalità (Shapiro-Wilk)**")
        
        selected_col = st.selectbox("Seleziona variabile:", numeric_cols, key="normality_test")
        
        if selected_col and len(df[selected_col].dropna()) > 3:
            data = df[selected_col].dropna()
            
            # Limite campione per Shapiro-Wilk
            if len(data) > 5000:
                data = data.sample(5000)
                st.info("Campione ridotto a 5000 osservazioni per il test")
            
            stat, p_value = stats.shapiro(data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Statistica", f"{stat:.4f}")
                st.metric("P-value", f"{p_value:.6f}")
                
                alpha = 0.05
                if p_value > alpha:
                    st.success("✅ I dati seguono una distribuzione normale")
                else:
                    st.warning("⚠️ I dati NON seguono una distribuzione normale")
            
            with col2:
                # Q-Q plot
                fig, ax = plt.subplots(figsize=(8, 6))
                stats.probplot(data, dist="norm", plot=ax)
                ax.set_title(f"Q-Q Plot - {selected_col}")
                st.pyplot(fig)
    
    with tab2:
        st.write("**Test di Correlazione (Pearson)**")
        
        if len(numeric_cols) >= 2:
            col1_select = st.selectbox("Prima variabile:", numeric_cols, key="corr_var1")
            col2_select = st.selectbox("Seconda variabile:", 
                                     [col for col in numeric_cols if col != col1_select], 
                                     key="corr_var2")
            
            if col1_select and col2_select:
                data1 = df[col1_select].dropna()
                data2 = df[col2_select].dropna()
                
                # Prendi intersezione degli indici validi
                valid_idx = data1.index.intersection(data2.index)
                
                if len(valid_idx) > 2:
                    corr_coef, p_value = stats.pearsonr(data1[valid_idx], data2[valid_idx])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Coefficiente di correlazione", f"{corr_coef:.4f}")
                        st.metric("P-value", f"{p_value:.6f}")
                        
                        if p_value < 0.05:
                            st.success("✅ Correlazione statisticamente significativa")
                        else:
                            st.warning("⚠️ Correlazione NON significativa")
                    
                    with col2:
                        # Scatter plot
                        fig = px.scatter(
                            x=data1[valid_idx], 
                            y=data2[valid_idx],
                            title=f"Correlazione: {col1_select} vs {col2_select}",
                            trendline="ols"
                        )
                        fig.update_xaxes(title=col1_select)
                        fig.update_yaxes(title=col2_select)
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.write("**Analisi della Varianza (ANOVA)**")
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(cat_cols) > 0 and len(numeric_cols) > 0:
            numeric_var = st.selectbox("Variabile numerica:", numeric_cols, key="anova_numeric")
            cat_var = st.selectbox("Variabile categorica:", cat_cols, key="anova_cat")
            
            if numeric_var and cat_var:
                # Prepara dati per ANOVA
                groups = []
                group_names = []
                
                for group in df[cat_var].unique():
                    if pd.notna(group):
                        group_data = df[df[cat_var] == group][numeric_var].dropna()
                        if len(group_data) > 0:
                            groups.append(group_data)
                            group_names.append(str(group))
                
                if len(groups) >= 2:
                    try:
                        f_stat, p_value = stats.f_oneway(*groups)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("F-statistica", f"{f_stat:.4f}")
                            st.metric("P-value", f"{p_value:.6f}")
                            
                            if p_value < 0.05:
                                st.success("✅ Differenze significative tra gruppi")
                            else:
                                st.warning("⚠️ Nessuna differenza significativa")
                        
                        with col2:
                            # Box plot per gruppi
                            fig = px.box(
                                df, 
                                x=cat_var, 
                                y=numeric_var,
                                title=f"Distribuzione {numeric_var} per {cat_var}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Errore nell'ANOVA: {str(e)}")
        else:
            st.info("Servono variabili numeriche e categoriche per l'ANOVA")
    
    with tab4:
        st.write("**Regressione Lineare Semplice**")
        
        if len(numeric_cols) >= 2:
            x_var = st.selectbox("Variabile indipendente (X):", numeric_cols, key="reg_x")
            y_var = st.selectbox("Variabile dipendente (Y):", 
                               [col for col in numeric_cols if col != x_var], 
                               key="reg_y")
            
            if x_var and y_var:
                # Rimuovi valori mancanti
                reg_data = df[[x_var, y_var]].dropna()
                
                if len(reg_data) > 2:
                    X = reg_data[x_var].values.reshape(-1, 1)
                    y = reg_data[y_var].values
                    
                    # Calcola regressione
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        reg_data[x_var], reg_data[y_var]
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R²", f"{r_value**2:.4f}")
                        st.metric("Pendenza", f"{slope:.4f}")
                        st.metric("Intercetta", f"{intercept:.4f}")
                        st.metric("P-value", f"{p_value:.6f}")
                        
                        # Equazione
                        st.write(f"**Equazione:** y = {slope:.3f}x + {intercept:.3f}")
                    
                    with col2:
                        # Grafico regressione
                        fig = px.scatter(
                            reg_data, 
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
    
    # Selezione variabili
    selected_vars = st.multiselect(
        "Seleziona variabili per clustering:",
        numeric_cols,
        default=numeric_cols[:min(4, len(numeric_cols))]  # Max 4 di default
    )
    
    if len(selected_vars) < 2:
        st.info("Seleziona almeno 2 variabili")
        return
    
    # Prepara dati
    cluster_data = df[selected_vars].dropna()
    
    if len(cluster_data) < 10:
        st.warning("Troppo pochi dati per clustering affidabile")
        return
    
    # Normalizzazione
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Parametri clustering
        st.write("**Parametri:**")
        n_clusters = st.slider("Numero cluster:", 2, 10, 3)
        
        # Esegui clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Aggiungi labels al dataset
        cluster_data_with_labels = cluster_data.copy()
        cluster_data_with_labels['Cluster'] = cluster_labels
        
        # Statistiche cluster
        st.write("**Statistiche Cluster:**")
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        for i, count in enumerate(cluster_counts):
            st.write(f"Cluster {i}: {count} punti ({count/len(cluster_labels)*100:.1f}%)")
        
        # Inerzia (within-cluster sum of squares)
        st.metric("Inerzia", f"{kmeans.inertia_:.2f}")
    
    with col2:
        # Visualizzazione
        if len(selected_vars) == 2:
            # Scatter plot 2D
            fig = px.scatter(
                cluster_data_with_labels,
                x=selected_vars[0],
                y=selected_vars[1],
                color='Cluster',
                title="Clustering K-means",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            # Aggiungi centroidi
            centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
            
            for i, centroid in enumerate(centroids_original):
                fig.add_scatter(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    mode='markers',
                    marker=dict(size=15, symbol='x', color='black'),
                    name=f'Centroide {i}',
                    showlegend=True
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # PCA per visualizzazione multidimensionale
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            
            pca_df = pd.DataFrame({
                'PC1': pca_data[:, 0],
                'PC2': pca_data[:, 1],
                'Cluster': cluster_labels
            })
            
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                title=f"Clustering K-means (PCA - {pca.explained_variance_ratio_.sum():.1%} varianza)",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Varianza spiegata
            st.write(f"**Varianza spiegata PCA:** PC1: {pca.explained_variance_ratio_[0]:.1%}, "
                    f"PC2: {pca.explained_variance_ratio_[1]:.1%}")
    
    # Metodo del gomito per determinare numero ottimale di cluster
    st.write("**Metodo del Gomito:**")
    
    with st.expander("Analizza numero ottimale di cluster"):
        max_k = min(10, len(cluster_data) // 2)
        k_range = range(1, max_k + 1)
        inertias = []
        
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(scaled_data)
            inertias.append(kmeans_temp.inertia_)
        
        # Grafico del gomito
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inerzia'
        ))
        fig.update_layout(
            title="Metodo del Gomito per K-means",
            xaxis_title="Numero di Cluster (k)",
            yaxis_title="Inerzia",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

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
        st.write("**Configurazione Modello:**")
        
        # Selezione variabile target
        target_var = st.selectbox(
            "Variabile target (da predire):",
            numeric_cols + cat_cols,
            key="prediction_target"
        )
        
        if not target_var:
            return
        
        # Selezione features
        available_features = [col for col in numeric_cols if col != target_var]
        
        selected_features = st.multiselect(
            "Variabili predittive:",
            available_features,
            default=available_features[:min(5, len(available_features))]
        )
        
        if len(selected_features) < 1:
            st.info("Seleziona almeno una variabile predittiva")
            return
        
        # Parametri del modello
        test_size = st.slider("% dati per test:", 10, 40, 20) / 100
        n_estimators = st.slider("Numero alberi:", 10, 200, 100)
        
    with col2:
        # Prepara dati
        model_data = df[selected_features + [target_var]].dropna()
        
        if len(model_data) < 50:
            st.warning("Troppo pochi dati per modello affidabile (minimo 50)")
            return
        
        X = model_data[selected_features]
        y = model_data[target_var]
        
        # Determina tipo di problema
        is_classification = (
            y.dtype == 'object' or 
            y.dtype == 'category' or 
            y.dtype == 'bool' or
            y.nunique() <= 10
        )
        
        try:
            if is_classification:
                # Classificazione
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=42,
                    max_depth=10
                )
                model.fit(X_train, y_train)
                
                # Predizioni
                y_pred = model.predict(X_test)
                accuracy = model.score(X_test, y_test)
                
                st.success(f"**Accuratezza:** {accuracy:.3f}")
                
                # Matrice di confusione
                cm = confusion_matrix(y_test, y_pred)
                
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title="Matrice di Confusione",
                    labels=dict(x="Predetto", y="Reale")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Report classificazione
                with st.expander("Report Dettagliato"):
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(3))
                
            else:
                # Regressione
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
                
                # Predizioni
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("R² Score", f"{r2:.3f}")
                with col_b:
                    st.metric("RMSE", f"{rmse:.3f}")
                
                # Grafico predizioni vs reali
                fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    title="Predizioni vs Valori Reali",
                    labels={'x': 'Valori Reali', 'y': 'Predizioni'}
                )
                
                # Linea perfetta
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(dash="dash", color="red")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Importanza features
            st.write("**Importanza Variabili:**")
            
            feature_importance = pd.DataFrame({
                'Feature': selected_features,
                'Importanza': model.feature_importances_
            }).sort_values('Importanza', ascending=False)
            
            fig = px.bar(
                feature_importance,
                x='Importanza',
                y='Feature',
                orientation='h',
                title="Importanza delle Variabili"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Errore nel modello: {str(e)}")

# ========================================
# CONTINUA DALL'ANALISI CORRELAZIONI (COMPLETAMENTO)
# ========================================

def show_correlation_analysis(df):
    """Analisi delle correlazioni tra variabili numeriche - COMPLETATA."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Servono almeno 2 variabili numeriche per l'analisi di correlazione")
        return
    
    st.subheader("🔗 Analisi Correlazioni")
    
    tab1, tab2, tab3 = st.tabs(["Matrice Correlazione", "Scatter Plots", "Correlazioni Forti"])
    
    with tab1:
        corr_method = st.selectbox(
            "Metodo di correlazione:",
            ["pearson", "spearman", "kendall"],
            help="Pearson: lineare, Spearman: monotonica, Kendall: più robusto"
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
        
        # Correlazioni forti
        st.write("**Correlazioni più forti:**")
        strong_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if not pd.isna(corr_matrix.iloc[i, j]):
                    strong_corr.append({
                        'Variabile 1': corr_matrix.columns[i],
                        'Variabile 2': corr_matrix.columns[j],
                        'Correlazione': corr_matrix.iloc[i, j],
                        'Forza': 'Forte' if abs(corr_matrix.iloc[i, j]) > 0.7 else 
                                'Media' if abs(corr_matrix.iloc[i, j]) > 0.3 else 'Debole'
                    })
        
        if strong_corr:
            corr_df = pd.DataFrame(strong_corr)
            corr_df = corr_df.sort_values('Correlazione', key=abs, ascending=False)
            st.dataframe(corr_df.head(10), use_container_width=True)
    
    with tab2:
        st.write("**Scatter Plot Interattivo**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox("Variabile X:", numeric_cols, key="scatter_x")
        with col2:
            y_var = st.selectbox("Variabile Y:", 
                               [col for col in numeric_cols if col != x_var], 
                               key="scatter_y")
        with col3:
            color_var = st.selectbox("Colora per:", 
                                   ["Nessuno"] + df.columns.tolist(), 
                                   key="scatter_color")
        
        if x_var and y_var:
            color_col = None if color_var == "Nessuno" else color_var
            
            fig = px.scatter(
                df,
                x=x_var,
                y=y_var,
                color=color_col,
                title=f"Correlazione: {x_var} vs {y_var}",
                trendline="ols"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.write("**Analisi Correlazioni Forti**")
        
        threshold = st.slider("Soglia correlazione:", 0.1, 0.9, 0.5, 0.1)
        
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if not pd.isna(corr_val) and abs(corr_val) >= threshold:
                    strong_correlations.append({
                        'Var1': corr_matrix.columns[i],
                        'Var2': cor
