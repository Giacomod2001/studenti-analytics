# Student Intelligence Hub 🎓

> **Advanced Predictive Analytics for Higher Education Retention**

A cloud-native platform engineered to transform raw academic data into actionable retention strategies. Powered by **Google BigQuery** and **Streamlit**, this system provides a real-time "Control Tower" for university administrators to monitor, predict, and prevent student dropout.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://progettodatamining.streamlit.app)

---

## 🏗️ Architecture: The "Control Tower" Concept

The application is structured around a workflow-first approach, moving from high-level strategy to granular intervention.

### 1. 📊 Control Tower (Executive Overview)

The central command center.

- **Real-Time KPIs**: Monitor total enrollment, active risk cases, and satisfaction pulse.
- **Split-Screen Intelligence**:
  - **Left Wing (Tactical)**: Immediate "Intervention Queue" showing top 5 critical risk students.
  - **Right Wing (Strategic)**: Machine Learning insights on behavioral clusters and satisfaction drivers.

### 2. 🚨 Intervention Console (Retention Management)

A dedicated workspace for academic counselors.

- **Risk Tiers**: Filter students by Critical (>80%), High (>60%), or Moderate risk.
- **Actionable Lists**: Direct export of at-risk cohorts for email campaigns or counseling scheduling.
- **Churn Prediction Model**: Uses a Random Forest classifier (50 trees, depth 6) to calculate precise dropout probabilities.

### 3. 🧩 Student 360° (Holistic Profiling)

Understanding the student beyond the grades.

- **Behavioral Clustering (K-Means)**: Segments students into 4 distinct archetypes (e.g., "Disengaged", "High Achievers").
- **Satisfaction Analysis (Boosted Tree)**: Predicts student satisfaction scores (0-10) based on academic load and performance.
- **Feature Importance**: Explains *why* the models are making certain predictions.

### 4. 💾 Raw Data Explorer

Direct access to the underlying BigQuery warehouse for auditing and custom analysis.

---

## 🛠️ Technical Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Frontend** | Streamlit | Python-based reactive web framework |
| **Data Warehouse** | Google BigQuery | Serverless, highly scalable cloud data warehouse |
| **ML Models** | BigQuery ML | In-database machine learning (Random Forest, K-Means, Boosted Tree) |
| **Styling** | CSS3 / Custom | "Professional Dark" theme with high-contrast UI components |

---

## 🚀 Installation & Setup

### Prerequisites

* Python 3.9+
- Google Cloud Platform Service Account (JSON Key)

### Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/Giacomod2001/studenti-analytics.git
    cd studenti-analytics
    ```

2. **Install Requirements**

    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Credentials**
    Create `.streamlit/secrets.toml`:

    ```toml
    [gcp_service_account]
    type = "service_account"
    project_id = "laboratorio-ai-460517"
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----..."
    client_email = "..."
    ```

4. **Run the Application**

    ```bash
    streamlit run streamlit_app.py
    ```

---

## 📄 License & Authors

**Authors:**
- Alessandro Geli
- Giacomo Dellacqua
- Paolo Junior Del Giudice
- Ruben Scoletta
- Luca Tallarico

**License:**
This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for details.

---
*Built for the Data Mining & Text Analytics Design Project - 2026*
