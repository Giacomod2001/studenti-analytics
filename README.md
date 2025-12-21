# Student Analytics Dashboard

> Data-Driven Retention Strategies for Higher Education

A comprehensive, cloud-native Machine Learning platform designed to predict university dropout risk and analyze student satisfaction. Built on **Google BigQuery** and **Streamlit**, it provides actionable insights into student behavior, enabling proactive retention interventions.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://progettodatamining.streamlit.app/#student-analytics)
**[Live Demo](https://progettodatamining.streamlit.app/#student-analytics)**

---

## Features

### Home Dashboard
- **Real-Time KPIs**: Total students, average dropout risk, high-risk count, behavioral segments
- **Risk Analysis**: Visual breakdown of students by risk category (Low/Medium/High)
- **Actionable Recommendations**: Immediate, short-term, and long-term intervention strategies
- **Quick Navigation**: Organized sidebar with ML tables and raw data separated

### Dropout Prediction
- **Random Forest Model**: Predicts the probability of student dropout with high accuracy
- **Risk Scoring**: Validates churn risk for every student, allowing targeted counseling
- **Risk Categories**: Students classified as Low (<30%), Medium (30-70%), or High (>70%) risk

### Feature Importance
- **Explainable AI**: Visualizes which factors most influence dropout risk
- **Top Drivers**: Identifies key variables like attendance, grades, participation
- **Actionable Insights**: Helps administrators focus on high-impact interventions

### Behavioral Segmentation
- **K-Means Clustering**: Groups students into distinct behavioral profiles
- **Persona Identification**: Segments like "High Achievers", "Struggling", "Disengaged"
- **Tailored Strategies**: Design specific support programs for each cluster

### Satisfaction Analysis
- **Boosted Tree Regression**: Correlates satisfaction with academic performance
- **Survey Data Analysis**: Processes Likert scale responses
- **Improvement Recommendations**: Suggests actions to enhance student experience

### Data Exploration
- **Interactive Tables**: Filter, search, and sort any dataset
- **Custom Visualizations**: Histogram, Box Plot, Scatter, Bar Chart, Heatmap
- **CSV Export**: Download any table for offline analysis

---

## Technical Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core programming language |
| Streamlit | Web application framework |
| Google BigQuery | Cloud data warehouse |
| Scikit-Learn | Machine learning models |
| Pandas | Data manipulation |
| Plotly | Interactive visualizations |

---

## Installation

### Prerequisites
- Python 3.8+
- Google Cloud Service Account Credentials

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Giacomod2001/studenti-analytics.git
   cd studenti-analytics
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   type = "service_account"
   project_id = "your-project-id"
   private_key_id = "..."
   private_key = "..."
   client_email = "..."
   ```

4. **Launch**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## Usage

### Home Dashboard
View key performance indicators at a glance:
- Total student count and dropout risk metrics
- Risk distribution with color-coded categories
- Recommended actions for intervention

### Data Inspection
1. Select any table from the sidebar
2. Use the **Explore Data** tab to filter and search
3. Use the **Statistics and Charts** tab for visualizations
4. Use the **Info and Origin** tab for data documentation

### Export
Click "Export CSV" on any table view to download data.

---

## Version History

### v2.0 (Current)
- Redesigned home dashboard with real KPIs and risk analysis
- Premium dark theme with transparent charts
- Improved sidebar with organized navigation and quick stats
- Removed charts from home for cleaner text-based insights
- Professional formal presentation throughout

### v1.0
- Initial release with core ML functionality
- Basic data exploration capabilities

---

## Architecture

```
studenti-analytics/
├── streamlit_app.py    # Main application entry point
├── app.py              # Alternative app version
├── data_utils.py       # BigQuery data loading functions
├── ml_utils.py         # Machine learning utilities
├── constants.py        # Table metadata and descriptions
├── styles_config.py    # CSS theme and chart styling
├── requirements.txt    # Python dependencies
└── README.md           # This documentation
```

---

## Privacy and Security

- **Cloud Processing**: Data stays in Google BigQuery
- **Secure Credentials**: Service account authentication
- **No External Sharing**: Data is not sent to third parties
- **Role-Based Access**: BigQuery permissions control data access

---

## License

This project is licensed under the Apache License 2.0 - see the `LICENSE` file for details.

---

## Acknowledgments

The authors would like to acknowledge the assistance of the AI tool Gemini 3 Pro High and the agentic system Antigravity for coding suggestions and debugging support during the development phase.

All final implementations, testing, and documentation were carried out independently by the project team.

---

## Authors

- Alessandro Geli
- Giacomo Dellacqua
- Paolo Junior Del Giudice
- Ruben Scoletta
- Luca Tallarico

---

**Powered by BigQuery and Streamlit**
