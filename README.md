# Student Intelligence Hub

> **Advanced Predictive Analytics for Higher Education Retention**

A cloud-native platform engineered to transform raw academic data into actionable retention strategies. Powered by **Google BigQuery** and **Streamlit**, this system provides a real-time dashboard for university administrators to monitor, predict, and prevent student dropout.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://progettodatamining.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0-0077B5?style=flat)](README.md)

---

## Live Demo

**[Access Student Intelligence Hub](https://progettodatamining.streamlit.app)**

---

## Features

### 1. Dashboard (Executive Overview)
- **Real-Time KPIs**: Total enrollment, risk cases, satisfaction pulse
- **Priority Intervention Queue**: Students requiring immediate attention
- **Strategic Insights**: Satisfaction drivers and cluster trends

### 2. Intervention Console
- **Risk Tiers**: Critical (>75%), Monitor (35-75%), Safe (<35%)
- **Actionable Lists**: Export at-risk cohorts for outreach
- **Model Disclaimer**: Built-in warning about prediction polarization

### 3. Student 360° (Holistic Profiling)
- **Behavioral Clustering**: K-Means segmentation into 4 archetypes
- **Silent Burnout Detection**: High grades + low satisfaction alerts
- **Feature Importance**: Model explainability

### 4. Alex AI Assistant
- **Contextual Help**: Sidebar chatbot for guidance
- **Page-Aware Responses**: Answers adapt to current view

---

## Technical Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Frontend** | Streamlit | Python reactive web framework |
| **Backend** | Google BigQuery | Serverless cloud data warehouse |
| **ML Models** | BigQuery ML | Random Forest, K-Means, Boosted Tree |
| **Styling** | CSS3 | LinkedIn-inspired premium dark theme |
| **AI Assistant** | Alex | Rule-based contextual advisor |

---

## Project Structure

```text
studenti-analytics/
├── streamlit_app.py    # Main application (~570 lines)
├── ml_utils.py         # ML utilities & Alex AI Assistant
├── data_utils.py       # Optimized BigQuery data loading
├── constants.py        # Configuration and table metadata
├── styles_config.py    # LinkedIn-style CSS theme
├── requirements.txt    # Python dependencies
├── SQL_QUERIES.md      # BigQuery ML queries
├── LICENSE             # Apache License 2.0
└── README.md           # Documentation
```

---

## Installation

### Prerequisites
- Python 3.9+
- Google Cloud Service Account (JSON Key)

### Setup

```bash
# Clone
git clone https://github.com/Giacomod2001/studenti-analytics.git
cd studenti-analytics

# Install dependencies
pip install -r requirements.txt

# Configure credentials (.streamlit/secrets.toml)
# See README for format

# Run
streamlit run streamlit_app.py
```

---

## ML Models

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| **Churn Prediction** | Random Forest | Dropout probability scoring |
| **Clustering** | K-Means (K=4) | Behavioral segmentation |
| **Satisfaction** | Boosted Tree | Experience score prediction |

> **Note**: Model predictions may show polarized distributions. See [SQL_QUERIES.md](SQL_QUERIES.md) for implementation details.

---

## Authors (IULM University - A.Y. 2025-2026)

- Alessandro Geli
- Giacomo Dellacqua
- Paolo Junior Del Giudice
- Ruben Scoletta
- Luca Tallarico

---

## License

Apache License 2.0 - See [LICENSE](LICENSE)
