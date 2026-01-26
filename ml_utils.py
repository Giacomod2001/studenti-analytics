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
    },
    'es': {
        'default': "Soy Alex, tu experto en análisis académico. Me especializo en modelos predictivos para la persistencia estudiantil y optimización del rendimiento.",
        'greeting': "Saludos. Soy Alex. Analizo patrones complejos para identificar señales tempranas de desvinculación académica y riesgo de abandono (churn).",
        'dashboard': "El Panel de Control muestra resultados en tiempo real del proceso KDD. Monitorea el 'Dropout Forecast'—utiliza Random Forest para predecir la persistencia.",
        'intervention_console': "Esta consola segmenta a los estudiantes mediante puntuación multifactorial. Enfócate en casos 'Críticos' (>75% de riesgo) con baja frecuencia de exámenes.",
        'student_360': "Student 360 usa Modelado de Comportamiento. Busca el 'Silent Burnout'—estudiantes con notas altas pero baja satisfacción.",
        'raw_data': "Data Explorer da acceso a la capa analítica de BigQuery. Esencial para auditar la importancia de las variables o puntuaciones psicométricas.",
        'risk_high': "Riesgo crítico: caída drástica en interacciones en el campus, falta a exámenes y baja flexibilidad reportada.",
        'risk_low': "Probabilidad de persistencia alta. Estos estudiantes son del arquetipo 'Resilient'—estables bajo presión académica.",
        'cluster_info': "K-Means identificó 4 arquetipos: desde 'Estudiantes Trabajadores' hasta 'Social Learners' impulsados por el compromiso.",
        'satisfaction': "Satisfaction Predictor estima el bienestar. Una brecha negativa entre satisfacción real y predicha suele preceder al abandono.",
        'fallback': "Puedo darte insights sobre: Segmentación de Riesgo, Arquetipos, Análisis de Satisfacción y Prioridad de Intervención."
    },
    'fr': {
        'default': "Je suis Alex, votre expert en analyse académique. Je me spécialise dans la modélisation prédictive de la persévérance scolaire.",
        'greeting': "Salutations. Je suis Alex. J'analyse les schémas de données complexes pour identifier les signes précoces de décrochage (churn).",
        'dashboard': "Le Tableau de Bord affiche les résultats du processus KDD. Surveillez le 'Dropout Forecast'—il utilise Random Forest pour prédire la persévérance.",
        'intervention_console': "Cette console segmente les étudiants par score multifactoriel. Ciblez les cas 'Critiques' (>75% de risque) avec une baisse de fréquence d'examens.",
        'student_360': "Student 360 utilise le Clustering Comportamentale. Cherchez le 'Silent Burnout'—étudiants avec de bonnes notes mais une faible satisfaction.",
        'raw_data': "Data Explorer donne accès à la couche analytique BigQuery. Essentiel pour auditer l'importance des variables ou les scores psychométriques.",
        'risk_high': "Risque critique : chute des interactions sur le campus, examens manqués et faibles scores de flexibilité.",
        'risk_low': "Probabilité de persévérance élevée. Étudiants de type 'Résilient'—stables même sous pression académique.",
        'cluster_info': "K-Means a identifié 4 archétypes : des 'Étudiants Salariés' aux 'Apprenants Sociaux' motivés par l'engagement.",
        'satisfaction': "Satisfaction Predictor estime le bien-être. Un écart négatif entre satisfaction réelle et prédite précède souvent le décrochage.",
        'fallback': "Je peux vous éclairer sur : la Segmentation des Risques, les Archétypes, l'Analyse de Satisfaction et les Priorités d'Intervention."
    }
}

def _detect_chat_language(text: str) -> str:
    """Detects if the input is likely Italian, Spanish, French or English."""
    langs = {
        'it': ["ciao", "come", "perché", "quali", "rischio", "abbandono", "studente", "università", "esame", "voto"],
        'es': ["hola", "como", "porque", "riesgo", "abandono", "estudiante", "universidad", "examen", "nota"],
        'fr': ["salut", "comment", "pourquoi", "risque", "abandon", "étudiant", "université", "examen", "note"]
    }
    text_lower = text.lower()
    for lang, keywords in langs.items():
        if any(kw in text_lower for kw in keywords):
            return lang
    return "en"

def get_alex_response(message: str, current_page: str = "Dashboard", lang: str = "en") -> str:
    """
    Alex AI Academic Learning Expert - returns contextual responses in specified language.
    """
    responses = _ALEX_RESPONSES.get(lang, _ALEX_RESPONSES['en'])
    
    if not message:
        # Map current_page to response key
        page_map = {
            "Dashboard": "dashboard",
            "Intervention Console": "intervention_console", 
            "Student 360": "student_360",
            "Raw Data Explorer": "raw_data"
        }
        return responses.get(page_map.get(current_page, "default"), responses['default'])
    
    msg_lower = message.lower()
    
    # 1. Risk & Churn (Priority)
    if any(kw in msg_lower for kw in ["risk", "dropout", "churn", "critical", "intervention", "pericolo", "rischio", "abbandono", "riesgo", "abandon", "risque"]):
        if any(kw in msg_lower for kw in ["high", "critical", "danger", "critico", "pericoloso", "peligro", "dangereux"]):
            return responses['risk_high']
        elif any(kw in msg_lower for kw in ["low", "safe", "stable", "sicuro", "tranquillo", "seguro", "estable", "stable"]):
            return responses['risk_low']
        return responses['intervention_console']
    
    # 2. Clustering
    if any(kw in msg_lower for kw in ["cluster", "segment", "group", "archetype", "behavior", "comportamento", "gruppo", "archetipo", "comportamiento", "groupe", "archétype"]):
        return responses['cluster_info']
    
    # 3. Satisfaction & Burnout
    if any(kw in msg_lower for kw in ["satisfaction", "happy", "burnout", "experience", "quality", "soddisfazione", "felice", "esperienza", "satisfacción", "experiencia", "qualité"]):
        return responses['satisfaction']
    
    # 4. Data & BigQuery
    if any(kw in msg_lower for kw in ["data", "bigquery", "query", "sql", "source", "origine", "dati", "datos", "données"]):
        return responses['raw_data']

    # 5. Greetings (Fallback if no specific topic)
    tokens = msg_lower.split()
    greetings = ["hi", "hello", "hey", "ciao", "buongiorno", "hola", "salut", "bonjour"]
    if any(kw in tokens for kw in greetings) or any(kw in msg_lower for kw in ["help", "aiuto", "ayuda", "aide"]):
        return responses['greeting']
    
    # 6. Page-specific fallbacks

    # Page-specific fallbacks
    if any(kw in msg_lower for kw in ["dashboard", "control", "tower", "kpi"]): return responses['dashboard']
    if any(kw in msg_lower for kw in ["console", "action", "intervento", "intervención", "intervention"]): return responses['intervention_console']
    if any(kw in msg_lower for kw in ["360", "profile", "profilo", "perfil", "profil"]): return responses['student_360']
    
    return responses['fallback']


# =============================================================================
# RISK ANALYSIS UTILITIES
# =============================================================================

def categorize_risk(percentage: float) -> Tuple[str, str]:
    """
    Categorizes a churn percentage into risk tier and color.
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
