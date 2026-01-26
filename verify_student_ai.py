import sys
import os

# Add project directory to path
sys.path.append(os.path.abspath(os.curdir))

from ml_utils import get_alex_response, classify_psychometric_status, get_psychometric_insight

def test_alex_responses():
    print("--- Test 1: Alex Specificity ---")
    
    # Test high risk question
    resp_risk = get_alex_response("How to identify high risk students?")
    print(f"User: How to identify high risk students?")
    print(f"Alex: {resp_risk}")
    
    if "Silent Burnout" in resp_risk or ">40%" in resp_risk:
        print("[OK] SUCCESS: Alex provided specific academic/BQuery contextual advice.")
    else:
        print("[FAIL] FAILURE: Alex response was generic.")

    # Test BQuery question
    resp_bq = get_alex_response("Where does the data come from?")
    print(f"\nUser: Where does the data come from?")
    print(f"Alex: {resp_bq}\n")
    
    if "BigQuery" in resp_bq:
        print("[OK] SUCCESS: Alex correctly identified the data source.")
    else:
        print("[FAIL] FAILURE: Alex missed the BigQuery reference.")

def test_psychometric_logic():
    print("\n--- Test 2: Psychometric Status Logic ---")
    
    # Test Silent Burnout (high performance, low real satisfaction vs predicted)
    status = classify_psychometric_status(real_score=4.0, predicted_score=7.0)
    insight = get_psychometric_insight(status)
    
    print(f"Real Score: 4.0, Predicted: 7.0")
    print(f"Status: {status}")
    print(f"Insight: {insight}")
    
    if status == "Silent Burnout":
        print("[OK] SUCCESS: Silent Burnout correctly identified.")
    else:
        print(f"[FAIL] FAILURE: Expected Silent Burnout, got {status}")

if __name__ == "__main__":
    try:
        test_alex_responses()
        test_psychometric_logic()
        print("\n[VERIFICATION COMPLETE]")
    except Exception as e:
        print(f"\n[FAIL] ERROR during verification: {e}")
