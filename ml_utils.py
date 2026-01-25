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
    'default': "I am Alex, your Academic Learning Expert. I can help you understand student analytics, risk predictions, and retention strategies.",
    'greeting': "Hello! I am Alex, your AI consultant for student retention. I analyze patterns in academic data to help you make data-driven decisions.",
    'control_tower': "The Control Tower shows your key performance indicators. Focus on the Dropout Forecast metric - this uses a Random Forest model to predict at-risk students.",
    'intervention_console': "The Intervention Console segments students by risk level. Critical (>75%) students need immediate attention - consider scheduling counseling sessions.",
    'student_360': "Student 360 provides holistic profiling. The behavioral clusters identify distinct student archetypes. Look for 'Silent Burnout' cases - high performers with low satisfaction.",
    'raw_data': "The Raw Data Explorer gives you direct access to BigQuery tables. Use this for custom analysis or data auditing.",
    'risk_high': "High-risk students typically show: declining exam performance, reduced campus engagement, or work-life conflicts. Early intervention is key.",
    'risk_low': "Low-risk students are stable, but consider engaging them as peer mentors. They can positively influence at-risk peers.",
    'cluster_info': "K-Means clustering groups students by behavioral patterns. Each cluster represents a distinct 'archetype' - from high achievers to disengaged learners.",
    'satisfaction': "The Boosted Tree model predicts satisfaction scores. A gap between predicted and actual satisfaction indicates potential issues.",
    'fallback': "I can help you with: risk analysis, student clustering, satisfaction metrics, and intervention strategies. What would you like to know?"
}

def get_alex_response(message: str, current_page: str = "Control Tower") -> str:
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
            "Control Tower": "control_tower",
            "Intervention Console": "intervention_console", 
            "Student 360": "student_360",
            "Raw Data Explorer": "raw_data"
        }
        return _ALEX_RESPONSES.get(page_map.get(current_page, "default"), _ALEX_RESPONSES['default'])
    
    msg_lower = message.lower()
    
    # Greeting detection
    if any(kw in msg_lower for kw in ["hi", "hello", "hey", "ciao", "help"]):
        return _ALEX_RESPONSES['greeting']
    
    # Risk-related questions
    if any(kw in msg_lower for kw in ["risk", "dropout", "churn", "critical", "intervention"]):
        if any(kw in msg_lower for kw in ["high", "critical", "danger"]):
            return _ALEX_RESPONSES['risk_high']
        elif any(kw in msg_lower for kw in ["low", "safe", "stable"]):
            return _ALEX_RESPONSES['risk_low']
        return _ALEX_RESPONSES['intervention_console']
    
    # Cluster-related questions
    if any(kw in msg_lower for kw in ["cluster", "segment", "group", "archetype", "behavior"]):
        return _ALEX_RESPONSES['cluster_info']
    
    # Satisfaction-related questions
    if any(kw in msg_lower for kw in ["satisfaction", "happy", "burnout", "experience", "quality"]):
        return _ALEX_RESPONSES['satisfaction']
    
    # Page-specific fallbacks
    if "control" in msg_lower or "tower" in msg_lower or "kpi" in msg_lower:
        return _ALEX_RESPONSES['control_tower']
    
    if "console" in msg_lower or "action" in msg_lower:
        return _ALEX_RESPONSES['intervention_console']
    
    if "360" in msg_lower or "profile" in msg_lower:
        return _ALEX_RESPONSES['student_360']
    
    if "data" in msg_lower or "table" in msg_lower or "raw" in msg_lower:
        return _ALEX_RESPONSES['raw_data']
    
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
    Generates an executive summary for the Control Tower.
    
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
