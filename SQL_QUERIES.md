# BigQuery ML Queries - Student Intelligence Hub

This document contains the SQL queries used in Google BigQuery to train ML models and generate predictions for the Student Intelligence Hub dashboard.

---

## 1. Churn Prediction (Random Forest)

### 1.1 Model Training

```sql
CREATE OR REPLACE MODEL `laboratorio-ai-460517.dataset.studenti_churn_rf`
OPTIONS(
  model_type = 'RANDOM_FOREST_CLASSIFIER',
  input_label_cols = ['churned'],
  num_parallel_tree = 50,
  max_tree_depth = 6,
  subsample = 0.7,
  min_tree_child_weight = 5,
  l2_reg = 0.1,
  early_stop = TRUE
) AS
SELECT
  CASE WHEN eta BETWEEN 18 AND 35 THEN eta ELSE NULL END as eta,
  CASE WHEN ore_studio_settimanali <= 50 THEN ore_studio_settimanali ELSE 50 END as ore_studio_settimanali,
  LEAST(numero_esami_superati, 30) as numero_esami_superati,
  CASE WHEN media_voti BETWEEN 18 AND 30 THEN media_voti ELSE NULL END as media_voti,
  partecipazione_eventi,
  lavoro_part_time,
  carriera_in_corso,
  sesso,
  CASE 
    WHEN regione_residenza IN ('Lombardia', 'Lazio', 'Campania', 'Sicilia', 'Veneto') 
    THEN regione_residenza 
    ELSE 'Altre' 
  END as regione_residenza,
  tipo_corso,
  gruppo_corso,
  churned
FROM `laboratorio-ai-460517.dataset.studenti`
WHERE churned IS NOT NULL
  AND anno_iscrizione BETWEEN 2021 AND 2023
  AND eta BETWEEN 18 AND 40
  AND media_voti >= 18
LIMIT 50000;
```

### 1.2 Predictions

```sql
CREATE OR REPLACE TABLE `laboratorio-ai-460517.dataset.studenti_churn_pred` AS
WITH raw_predictions AS (
  SELECT
    student_id,
    predicted_churned as churn_pred,
    predicted_churned_probs
  FROM ML.PREDICT(
    MODEL `laboratorio-ai-460517.dataset.studenti_churn_rf`,
    (SELECT student_id, eta, ore_studio_settimanali, numero_esami_superati, 
            media_voti, partecipazione_eventi, lavoro_part_time, 
            carriera_in_corso, sesso, regione_residenza, tipo_corso, gruppo_corso
     FROM `laboratorio-ai-460517.dataset.studenti`
     WHERE anno_iscrizione > 2020 AND eta BETWEEN 18 AND 40)
  )
)
SELECT 
  student_id,
  churn_pred,
  ROUND(raw_churn_prob * 100, 1) AS churn_percentage,
  CASE 
    WHEN raw_churn_prob >= 0.7 THEN 'Alto Rischio'
    WHEN raw_churn_prob >= 0.4 THEN 'Medio Rischio'
    ELSE 'Basso Rischio'
  END as categoria_rischio
FROM raw_predictions;
```

---

## 2. Student Clustering (K-Means)

### 2.1 Model Training

```sql
CREATE OR REPLACE MODEL `laboratorio-ai-460517.dataset.studenti_kmeans`
OPTIONS(
  model_type = 'kmeans',
  num_clusters = 4,
  standardize_features = TRUE,
  kmeans_init_method = 'KMEANS_PLUS_PLUS',
  max_iterations = 50
) AS
SELECT
  COALESCE(anno_iscrizione, 2022) as anno_iscrizione,
  eta,
  ore_studio_settimanali,
  COALESCE(numero_esami_superati, 0) as numero_esami_superati,
  media_voti,
  partecipazione_eventi as partecipazione_eventi_num,
  lavoro_part_time as lavoro_part_time_num,
  carriera_in_corso as carriera_in_corso_num
FROM `laboratorio-ai-460517.dataset.studenti`
WHERE eta IS NOT NULL
  AND ore_studio_settimanali IS NOT NULL
  AND anno_iscrizione BETWEEN 2020 AND 2024
LIMIT 100000;
```

### 2.2 Predictions

```sql
CREATE OR REPLACE TABLE `laboratorio-ai-460517.dataset.studenti_cluster` AS
SELECT
  student_id,
  centroid_id as cluster,
  ROUND(nearest_centroids_distance[OFFSET(0)].distance, 3) as distance_to_centroid,
  CASE
    WHEN distance_to_centroid <= 2.0 THEN 'Tipico'
    WHEN distance_to_centroid <= 4.0 THEN 'Moderato'
    ELSE 'Atipico'
  END as cluster_quality
FROM ML.PREDICT(
  MODEL `laboratorio-ai-460517.dataset.studenti_kmeans`,
  (SELECT * FROM `laboratorio-ai-460517.dataset.studenti` WHERE anno_iscrizione BETWEEN 2020 AND 2024)
);
```

---

## 3. Satisfaction Analysis (Boosted Tree Regressor)

### 3.1 Model Training

```sql
CREATE OR REPLACE MODEL `laboratorio-ai-460517.dataset.studenti_soddisfazione_btr`
OPTIONS(
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['soddisfazione'],
  max_iterations = 100,
  learn_rate = 0.1,
  min_tree_child_weight = 1,
  subsample = 0.8
) AS
SELECT
  CAST(anno_iscrizione AS FLOAT64) AS anno_iscrizione,
  CAST(eta AS FLOAT64) AS eta,
  CAST(ore_studio_settimanali AS FLOAT64) AS ore_studio_settimanali,
  CAST(numero_esami_superati AS FLOAT64) AS numero_esami_superati,
  CAST(media_voti AS FLOAT64) AS media_voti,
  CAST(partecipazione_eventi AS FLOAT64) AS partecipazione_eventi,
  CAST(lavoro_part_time AS FLOAT64) AS lavoro_part_time,
  CAST(carriera_in_corso AS FLOAT64) AS carriera_in_corso,
  sesso, regione_residenza, ateneo, tipo_corso, gruppo_corso, classe_corso,
  CAST(soddisfazione AS FLOAT64) AS soddisfazione
FROM `laboratorio-ai-460517.dataset.studenti`
WHERE soddisfazione IS NOT NULL AND soddisfazione BETWEEN 1 AND 10;
```

### 3.2 Predictions & Report

```sql
CREATE OR REPLACE TABLE `laboratorio-ai-460517.dataset.report_finale_soddisfazione_studenti` AS
SELECT
  student_id,
  ROUND(soddisfazione_reale, 1) AS soddisfazione_reale,
  ROUND(predicted_soddisfazione, 2) AS soddisfazione_predetta,
  CASE 
    WHEN predicted_soddisfazione >= 8.0 THEN 'Molto Soddisfatto'
    WHEN predicted_soddisfazione >= 6.5 THEN 'Soddisfatto'
    WHEN predicted_soddisfazione >= 5.0 THEN 'Neutrale'
    ELSE 'Insoddisfatto'
  END AS categoria_soddisfazione
FROM ML.PREDICT(MODEL `laboratorio-ai-460517.dataset.studenti_soddisfazione_btr`, ...);
```

---

## 4. Feature Importance

```sql
CREATE OR REPLACE TABLE `laboratorio-ai-460517.dataset.feature_importance_studenti` AS
SELECT 
  feature AS caratteristica,
  ROUND(importance_weight, 4) AS peso_importanza,
  ROUND(importance_weight / SUM(importance_weight) OVER() * 100, 2) AS percentuale_importanza,
  CASE 
    WHEN RANK() OVER (ORDER BY importance_weight DESC) <= 3 THEN 'Molto Importante'
    WHEN RANK() OVER (ORDER BY importance_weight DESC) <= 6 THEN 'Importante'
    ELSE 'Moderatamente Importante'
  END AS categoria_importanza
FROM ML.FEATURE_IMPORTANCE(MODEL `laboratorio-ai-460517.dataset.studenti_soddisfazione_btr`)
ORDER BY importance_weight DESC;
```

---

**Dataset**: `laboratorio-ai-460517.dataset`  
**Tables Created**: `studenti_churn_pred`, `studenti_cluster`, `report_finale_soddisfazione_studenti`, `feature_importance_studenti`
