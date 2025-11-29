# Student Analytics Dashboard

## Abstract

This repository presents a functional Minimum Viable Product (MVP) of a cloud-native machine learning platform designed to predict university dropout risk and demonstrate data-driven retention strategies. The platform leverages Google BigQuery for scalable data warehousing and Streamlit for interactive data visualization.

## Overview

The Student Analytics Dashboard is an enterprise-grade analytics solution that integrates multiple machine learning models to provide actionable insights into student performance, behavior, and retention risk. The platform processes institutional data through a robust ETL pipeline and presents results via an intuitive web interface.

### Core Capabilities

The system implements three primary analytical frameworks:

1. **Dropout Risk Prediction**: Utilizes Random Forest classification to identify students at risk of withdrawal, enabling proactive intervention strategies.

2. **Behavioral Segmentation**: Employs K-means clustering (K=4) to categorize students into distinct behavioral profiles for targeted support programs.

3. **Satisfaction Forecasting**: Implements Boosted Tree regression (XGBoost) to model and predict student satisfaction metrics from survey data.

## Technical Architecture

### Technology Stack

- **Frontend**: Streamlit 1.x
- **Data Warehouse**: Google BigQuery
- **Visualization**: Plotly 5.x
- **Data Processing**: Pandas, NumPy
- **Authentication**: Google Cloud Service Account
- **ML Framework**: Scikit-learn, XGBoost (upstream models)

### System Requirements

- Python 3.8 or higher
- Google Cloud Platform account with BigQuery API enabled
- Service account credentials with appropriate IAM permissions
- Minimum 4GB RAM recommended for local execution

## Installation

### 1. Repository Setup

```bash
git clone https://github.com/YOUR_USERNAME/studenti-analytics.git
cd studenti-analytics
```

### 2. Dependency Installation

```bash
pip install -r requirements.txt
```

### 3. Authentication Configuration

Create a `.streamlit/secrets.toml` file in the project root with the following structure:

```toml
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
universe_domain = "googleapis.com"
```

**Security Notice**: Ensure `secrets.toml` is added to `.gitignore` to prevent credential exposure.

## Data Schema

The platform expects the following BigQuery tables within the configured dataset:

| Table Identifier | Description | Primary Use |
|------------------|-------------|-------------|
| `studenti` | Student demographic and academic performance records | Base analytics |
| `studenti_churn_pred` | Dropout probability predictions (0-1 scale) | Risk assessment |
| `studenti_cluster` | K-means cluster assignments | Segmentation analysis |
| `feature_importance_studenti` | Model feature importance rankings | Model interpretability |
| `studenti_soddisfazione_btr` | Satisfaction score predictions | Student experience metrics |
| `student_churn_rf` | Random Forest model metadata and performance metrics | Model monitoring |
| `student_kmeans` | K-means centroids and inertia values | Clustering validation |
| `report_finale_soddisfazione_studenti` | Aggregated satisfaction analysis | Executive reporting |

## Machine Learning Models

### Dropout Prediction Model

**Algorithm**: Random Forest Classifier  
**Objective**: Binary classification of student dropout risk  
**Input Features**: Academic performance, attendance patterns, engagement metrics  
**Output**: Probability score [0,1] indicating dropout likelihood  
**Performance Metrics**: Accuracy, Precision, Recall, F1-Score (stored in `student_churn_rf`)

### Student Clustering Model

**Algorithm**: K-means Clustering  
**Number of Clusters**: 4  
**Objective**: Behavioral segmentation for targeted interventions  
**Features**: Study hours, academic performance, absence frequency  
**Preprocessing**: Z-score normalization  
**Validation**: Elbow method, silhouette analysis

### Satisfaction Prediction Model

**Algorithm**: Gradient Boosted Trees (XGBoost)  
**Objective**: Regression for satisfaction score prediction  
**Input**: Survey responses (Likert scale 1-5), demographic data  
**Output**: Continuous satisfaction score  
**Performance Metrics**: R², RMSE, MAE

## Usage

### Application Launch

Execute the following command from the project root:

```bash
streamlit run app.py
```

The dashboard will be accessible at `http://localhost:8501`

### Navigation

- **Home Dashboard**: Executive summary with KPI metrics and data catalogue
- **Dataset Views**: Detailed exploration of individual tables with filtering and export capabilities
- **Specialized Visualizations**: Auto-generated charts tailored to each dataset type

## Features

### Data Management
- Three-tier data loading strategy with automatic fallback mechanisms
- Intelligent caching system (TTL: 10 minutes)
- Type optimization for memory efficiency
- Support for datasets exceeding 1M rows

### Visualization
- Specialized charts for ML model outputs (dropout distribution, feature importance, cluster analysis)
- Interactive Plotly-based visualizations with custom theming
- Correlation matrices and distribution analysis
- CSV export functionality for all datasets

### User Interface
- Responsive layout optimized for desktop environments
- Collapsible sidebar navigation
- Advanced filtering with multi-column search
- Premium design system with Inter font family

## Performance Optimization

The application implements several performance enhancements:

1. **Multi-tier Data Loading**: BQ Storage API → REST API → Manual iteration fallback
2. **Streamlit Caching**: `@st.cache_data` for query results, `@st.cache_resource` for client connections
3. **Type Optimization**: Automatic categorical encoding for low-cardinality columns
4. **Query Optimization**: Selective column loading and row limiting where applicable

## Security Considerations

- Service account credentials should be stored securely using environment-specific secret management
- Implement authentication middleware for production deployments
- Enable BigQuery audit logging for compliance requirements
- Use least-privilege IAM roles for service accounts
- Rotate service account keys regularly according to organizational policy

## Deployment

For production deployment, consider:

1. **Containerization**: Use Docker for consistent environment management
2. **Cloud Hosting**: Deploy to Google Cloud Run, AWS App Runner, or Azure Container Instances
3. **Authentication**: Implement OAuth 2.0 or SAML-based SSO
4. **Monitoring**: Integrate with application performance monitoring (APM) tools
5. **Scalability**: Configure horizontal scaling based on concurrent user load

## License

This project is distributed under the MIT License. See `LICENSE` file for complete terms.

## Contributing

Contributions are evaluated based on:
- Code quality and adherence to PEP 8 standards
- Comprehensive unit test coverage
- Documentation completeness
- Backward compatibility maintenance

Please submit contributions via pull request with detailed descriptions of changes.

## Acknowledgments

Built using open-source technologies: Streamlit, Plotly, Pandas, Google Cloud BigQuery API.

## Support

For technical issues or feature requests, please submit a GitHub issue with:
- Detailed problem description
- Steps to reproduce (if applicable)
- Environment details (Python version, OS)
- Relevant log outputs

---

**Version**: 2.0.0  
**Last Updated**: 2025-11-29  
**Maintainer**: Giacomo Dellacqua
