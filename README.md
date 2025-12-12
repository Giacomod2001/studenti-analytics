# Student Analytics Dashboard

> **Data-Driven Retention Strategies for Higher Education**

**Student Analytics Dashboard** is a comprehensive, cloud-native Machine Learning platform designed to predict university dropout risk and analyze student satisfaction. Built on **Google BigQuery** and **Streamlit**, it provides actionable insights into student behavior, enabling proactive retention interventions.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://progettodatamining.streamlit.app/#student-analytics)
**[Live Demo](https://progettodatamining.streamlit.app/#student-analytics)**

---

## Features

### Dropout Prediction (Churn)
- **Random Forest Model**: Predicts the probability of student dropout with high accuracy.
- **Risk Scoring**: Validates churn risk for every student, allowing targeted counseling.
- **Explainable AI**: Visualizes Feature Importance to understand *why* a student is at risk (e.g., low attendance, failing grades).

### Behavioral Segmentation
- **K-Means Clustering**: Unsupervised learning groups students into distinct personas (e.g., "High Achievers", "Struggling", "Disengaged").
- **Tailored Strategies**: Enables the university to design specific support programs for each cluster.

### Satisfaction Analysis
- **Boosted Tree Regression**: Analyzes survey data to correlate satisfaction with academic performance and event participation.
- **Automated Reporting**: Generates comprehensive PDF/HTML reports for administrative review.

### Cloud-Native Architecture
- **BigQuery Integration**: Direct, optimized connection to enterprise-grade data warehouse.
- **Real-Time Caching**: Implements intelligent caching strategies (`ttl`, `cache_data`, `cache_resource`) for sub-second dashboard performance.

---

## Technical Stack

The application is engineered using a robust modern stack:

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![BigQuery](https://img.shields.io/badge/BigQuery-669DF6?style=for-the-badge&logo=google-cloud&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- Google Cloud Service Account Credentials (JSON)

### Step-by-Step Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/student-analytics.git
   cd student-analytics
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets**
   Create a `.streamlit/secrets.toml` file with your BigQuery credentials:
   ```toml
   [gcp_service_account]
   type = "service_account"
   project_id = "your-project-id"
   private_key_id = "..."
   private_key = "..."
   client_email = "..."
   ```

4. **Launch the Application**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## Usage Manual

1. **Home Dashboard**:
   - View high-level KPIs: Total students, dataset size, last update time.
   - Access the **Data Catalogue** for an overview of all available tables.

2. **Data Inspection**:
   - Navigate to specific tables (e.g., *Dropout Prediction*) via the Sidebar.
   - Use **Advanced Filters** to search and segment data.
   - Visualize distributions using interactive Plotly charts.
   - Access the **Risk Monitor** for dropout prediction actionable insights.

3. **Export**:
   - Download processed datasets as CSV for offline analysis.

---

## License

This project is licensed under the Apache License 2.0 - see the `LICENSE` file for details.

---

## Acknowledgments

The authors would like to acknowledge the assistance of the AI tool Claude (Anthropic) for coding suggestions and debugging support during the development phase. 

All final implementations, testing, and documentation were carried out independently by the project team.
