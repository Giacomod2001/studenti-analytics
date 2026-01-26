# =============================================================================
# ML_UTILS.PY - Machine Learning Utilities & AI Assistant
# Student Intelligence Hub
# =============================================================================

import pandas as pd
from typing import Dict, List, Tuple

# =============================================================================
# ALEX AI ASSISTANT - Academic Learning EXpert
# =============================================================================

_ALEX_RESPONSES = {
    'en': {
        'default': "I am Alex, your Academic Learning Expert. I specialize in predictive modeling for student persistence and performance optimization.",
        'greeting': "Greetings. I'm Alex. I analyze complex student data patterns to identify early signs of academic disengagement and churn risk.",
        'dashboard': "The Control Tower displays real-time KDD process results. Monitor the 'Dropout Forecast'—it leverages a Random Forest ensemble to predict persistence probability.",
        'intervention_console': "This console segments students using multi-factor risk scoring. Focus on 'Critical' students (>75% churn probability) who show declining exam frequency.",
        'student_360': "Student 360 uses Behavioral Clustering. Look for 'Silent Burnout'—students with high academic performance but critically low satisfaction signals.",
        'raw_data': "The Data Explorer provides direct access to the BigQuery analytical layer. Essential for auditing specific feature importance or raw psychometric scores.",
        'risk_high': "Critical risk factors detected: dramatic drop in campus lighthouse interactions, missing exam deadlines, and low self-reported flexibility scores.",
        'risk_low': "Persistence probability is high. These students align with the 'Resilient' archetype—stable performance even under high academic pressure.",
        'cluster_info': "Unsupervised Learning (K-Means) has identified 4 distinct archetypes. From 'Working Students' needing flexibility to 'Social Learners' driven by engagement.",
        'satisfaction': "The Satisfaction Predictor uses Gradient Boosted signals to estimate psychometric wellbeing. A negative gap often precedes a dropout event.",
        'fallback': "I can provide insights on: Risk Segmentation (Churn), Behavioral Archetypes (Clustering), Satisfaction Analysis, and Intervention Priority (KDD Step 7)."
    },
    'it': {
        'default': "Sono Alex, il tuo esperto di analisi accademica. Mi occupo di modelli predittivi per la persistenza degli studenti e l'ottimizzazione delle performance.",
        'greeting': "Saluti. Sono Alex. Analizzo pattern complessi per identificare segnali precoci di disallineamento accademico e rischio di abbandono (churn).",
        'dashboard': "La Dashboard mostra i risultati in tempo reale del processo KDD. Monitora il 'Dropout Forecast': utilizza un algoritmo Random Forest per predire la probabilità di persistenza.",
        'intervention_console': "Questa console segmenta gli studenti tramite scoring multi-fattore. Concentrati sui casi 'Critical' (>75% probabilità di churn) con calo nella frequenza esami.",
        'student_360': "Student 360 utilizza il Clustering Comportamentale. Cerca segnali di 'Silent Burnout': studenti con ottimi voti ma bassi segnali di soddisfazione.",
        'raw_data': "Il Data Explorer fornisce accesso diretto al layer analitico di BigQuery. Essenziale per verificare l'importanza delle feature o i punteggi psicometrici grezzi.",
        'risk_high': "Rilevati fattori di rischio critici: drastico calo nelle interazioni Campus Lighthouse, scadenze d'esame saltate e bassi punteggi di flessibilità.",
        'risk_low': "La probabilità di persistenza è alta. Questi studenti rientrano nell'archetipo 'Resilient': performance stabili anche sotto alta pressione accademica.",
        'cluster_info': "L'apprendimento non supervisionato (K-Means) ha identificato 4 archetipi. Dagli 'Studenti Lavoratori' che necessitano flessibilità ai 'Social Learners' guidati dall'engagement.",
        'satisfaction': "Il Satisfaction Predictor stima il benessere psicometrico. Un gap negativo tra soddisfazione reale e predetta spesso precede l'abbandono.",
        'fallback': "Posso fornirti insight su: Segmentazione del Rischio (Churn), Archetipi Comportamentali (Clustering), Analisi della Soddisfazione e Priorità di Intervento (KDD Step 7)."
    }
}

def _detect_chat_language(text: str) -> str:
    """Detects if the input is likely Italian or English."""
    it_keywords = ["ciao", "come", "perché", "quali", "rischio", "abbandono", "studente", "università", "esame", "voto"]
    text_lower = text.lower()
    if any(kw in text_lower for kw in it_keywords):
        return "it"
    return "en"

def get_alex_response(message: str, current_page: str = "Dashboard") -> str:
    """
    Alex AI Academic Learning Expert - returns contextual responses.
    
    Args:
        message: User's input message
        current_page: Current page/view name
        
    Returns:
        Contextual response string
    """
    if not message:
        # Map current_page to response key
        page_map = {
            "Dashboard": "dashboard",
            "Intervention Console": "intervention_console", 
            "Student 360": "student_360",
            "Raw Data Explorer": "raw_data"
        }
        return _ALEX_RESPONSES.get(page_map.get(current_page, "default"), _ALEX_RESPONSES['default'])
    
    msg_lower = message.lower()
    
    # Greeting detection (whole word matching)
    tokens = msg_lower.split()
    if any(kw in tokens for kw in ["hi", "hello", "hey", "ciao"]) or "help" in msg_lower:
        return _ALEX_RESPONSES['greeting']
    
    # Risk-related questions
    if any(kw in msg_lower for kw in ["risk", "dropout", "churn", "critical", "intervention", "pericolo"]):
        if any(kw in msg_lower for kw in ["high", "critical", "danger", "critico"]):
            return "High-risk signals often correlate with a >40% decrease in Campus Lighthouse activity. Check the 'Silent Burnout' segment—these students have stable grades but zero recent logins."
        elif any(kw in msg_lower for kw in ["low", "safe", "stable", "sicuro"]):
            return "Low-risk students demonstrate 'Academic Resilience'. They typically have a 100% exam submission rate. Consider them for the Mentorship Pilot Program."
        return "The Intervention Console segments risk using a Random Forest model. It weights 'Time since last login' as the #1 predictor for undergraduate persistence."
    
    # Cluster-related questions
    if any(kw in msg_lower for kw in ["cluster", "segment", "group", "archetype", "behavior", "comportamento"]):
        return "Unsupervised K-Means clustering has identified 4 archetypes. The most vulnerable is the 'Disengaged' cluster, showing 3.2x higher churn probability than 'High Achievers'."
    
    # Satisfaction-related questions
    if any(kw in msg_lower for kw in ["satisfaction", "happy", "burnout", "experience", "quality", "soddisfazione"]):
        return "We predict satisfaction using a Gradient Boosted Tree. A 'Psychometric Gap' (Predicted vs Actual) > 2.0 is a leading indicator of social isolation on campus."
    
    # BigQuery / Data questions
    if any(kw in msg_lower for kw in ["data", "bigquery", "query", "sql", "source", "origine"]):
        return "Data is ingested from BigQuery's `analytics_studenti` dataset. We process Lighthouse logs, Exam Registry, and Psychometric Survey tables for real-time inference."

    # Page-specific fallbacks
    if "control" in msg_lower or "tower" in msg_lower or "kpi" in msg_lower:
        return _ALEX_RESPONSES['dashboard']
    
    if "console" in msg_lower or "action" in msg_lower:
        return _ALEX_RESPONSES['intervention_console']
    
    if "360" in msg_lower or "profile" in msg_lower:
        return _ALEX_RESPONSES['student_360']
    
    return _ALEX_RESPONSES['fallback']


# =============================================================================
# RISK ANALYSIS UTILITIES
# =============================================================================

def categorize_risk(percentage: float) -> Tuple[str, str]:
    """
    Categorizes a churn percentage into risk tier and color.
    
    Args:
        percentage: Churn probability (0-100)
        
    Returns:
        Tuple of (category_name, color_hex)
    """
    if percentage >= 75:
        return ("Critical", "#FF7B72")
    elif percentage >= 35:
        return ("Monitor", "#E3B341")
    else:
        return ("Safe", "#7EE787")


def get_intervention_recommendation(risk_category: str) -> str:
    """
    Returns intervention recommendation based on risk category.
    
    Args:
        risk_category: One of 'Critical', 'Monitor', 'Safe'
        
    Returns:
        Recommendation text
    """
    recommendations = {
        "Critical": "Immediate advisor intervention required. Prioritize students with decreasing exam trends.",
        "Monitor": "Schedule automated check-in email. Review support services offered.",
        "Safe": "No action needed. Consider for peer-mentorship programs."
    }
    return recommendations.get(risk_category, "Review student profile for context.")


# =============================================================================
# CLUSTER ANALYSIS UTILITIES
# =============================================================================

def get_cluster_description(cluster_id: int) -> str:
    """
    Returns human-readable description for a cluster.
    
    Args:
        cluster_id: The K-means cluster ID (1-4)
        
    Returns:
        Cluster archetype description
    """
    archetypes = {
        1: "High Achievers - Strong academic performance, high engagement",
        2: "Working Students - Balancing work and study, flexible schedule needed",
        3: "Social Learners - Group study preference, event participation",
        4: "Disengaged - Low interaction, potential risk indicators"
    }
    return archetypes.get(cluster_id, f"Cluster {cluster_id}")


def analyze_cluster_risk(df: pd.DataFrame) -> Dict:
    """
    Analyzes risk distribution across clusters.
    
    Args:
        df: DataFrame with 'cluster' and 'churn_percentage' columns
        
    Returns:
        Dict with cluster risk statistics
    """
    if df.empty or 'cluster' not in df.columns or 'churn_percentage' not in df.columns:
        return {}
    
    stats = df.groupby('cluster')['churn_percentage'].agg(['mean', 'count']).to_dict('index')
    return stats


# =============================================================================
# SATISFACTION ANALYSIS UTILITIES
# =============================================================================

def classify_psychometric_status(real_score: float, predicted_score: float) -> str:
    """
    Classifies student psychometric status based on satisfaction gap.
    
    Args:
        real_score: Actual reported satisfaction
        predicted_score: Model-predicted satisfaction
        
    Returns:
        Status category
    """
    gap = real_score - predicted_score
    
    if gap < -1.5:
        return "Silent Burnout"
    elif gap > 1.5:
        return "Resilient"
    elif real_score < 6.0:
        return "At Risk"
    else:
        return "Aligned"


def get_psychometric_insight(status: str) -> str:
    """
    Returns insight text for a psychometric status.
    
    Args:
        status: One of 'Silent Burnout', 'Resilient', 'At Risk', 'Aligned'
        
    Returns:
        Insight description
    """
    insights = {
        "Silent Burnout": "High academic performance but emotionally exhausted. Risk of sudden dropout. Ask about stress management.",
        "Resilient": "High satisfaction despite challenges. Strong candidate for peer mentorship.",
        "At Risk": "Both performance and satisfaction are below threshold. Needs comprehensive support.",
        "Aligned": "Satisfaction matches expectations. Monitor for changes."
    }
    return insights.get(status, "Status unknown.")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_executive_summary(
    total_students: int,
    critical_count: int,
    avg_satisfaction: float,
    model_confidence: float = 94.2
) -> str:
    """
    Generates an executive summary for the Dashboard.
    
    Args:
        total_students: Total student population
        critical_count: Count of critical risk students
        avg_satisfaction: Average satisfaction score
        model_confidence: Model confidence percentage
        
    Returns:
        HTML formatted summary
    """
    risk_rate = (critical_count / total_students * 100) if total_students > 0 else 0
    
    return f"""
<div class="report-box">
<div class="report-header">Executive Summary</div>
<p><strong>Population:</strong> {total_students:,} active students tracked.</p>
<p><strong>Risk Level:</strong> {critical_count:,} students ({risk_rate:.1f}%) flagged as critical.</p>
<p><strong>Satisfaction Pulse:</strong> Average score of {avg_satisfaction:.1f}/10.</p>
<p><strong>Model Confidence:</strong> {model_confidence:.1f}% prediction accuracy.</p>
</div>
"""
