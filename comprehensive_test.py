import sys
import os
import py_compile
from typing import List

# Setup path
sys.path.append(os.path.abspath(os.curdir))

def run_syntax_check(files: List[str]):
    print("--- 1) Syntax Verification ---")
    all_pass = True
    for f in files:
        try:
            py_compile.compile(f, doraise=True)
            print(f"[OK] {f}")
        except Exception as e:
            print(f"[FAIL] {f}: {e}")
            all_pass = False
    return all_pass

def test_polyglot_alex():
    print("\n--- 2) Alex AI Polyglot Verification ---")
    from ml_utils import get_alex_response, _detect_chat_language
    
    test_cases = [
        # ITALIANO
        ("Ciao Alex, come riduco il rischio di abbandono?", "it", ["abbandono", "churn", "modelli"]),
        # ENGLISH
        ("Hi Alex, how do I analyze the dropout forecast?", "en", ["dropout", "forecast", "proccess"]),
        # SPANISH
        ("Hola Alex, ¿cómo puedo ver los grupos de estudiantes?", "es", ["grupos", "arquetipos", "estudiantes"]),
        # FRENCH
        ("Bonjour Alex, quels sont les risques critiques?", "fr", ["risques", "critiques", "chute"])
    ]
    
    all_pass = True
    for msg, expected_lang, keywords in test_cases:
        actual_lang = _detect_chat_language(msg)
        response = get_alex_response(msg, lang=actual_lang)
        
        lang_match = actual_lang == expected_lang
        keyword_match = any(kw.lower() in response.lower() for kw in keywords)
        
        print(f"[{'OK' if lang_match else 'FAIL'}] Lang Detect: '{msg[:20]}...' -> Detected: {actual_lang} (Expected: {expected_lang})")
        print(f"[{'OK' if keyword_match else 'FAIL'}] Keywords check in response.")
        
        if not lang_match or not keyword_match:
            all_pass = False
            if not keyword_match:
                print(f"      Response: {response[:100]}...")
                
    return all_pass

def audit_streamlit_callbacks():
    print("\n--- 3) Streamlit Callback Audit ---")
    with open("streamlit_app.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check for st.rerun() inside common callback patterns
    if "st.rerun()" in content and "def process_alex_chat()" in content:
        # We already know we commented it out or removed it, but let's be sure
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "st.rerun()" in line and "process_alex_chat" in "".join(lines[max(0, i-10):i]):
                if line.strip().startswith("#"):
                    print("[OK] st.rerun() in process_alex_chat is commented out.")
                else:
                    print("[WARN] st.rerun() found uncommented in process_alex_chat logic.")
                    return False
    
    print("[OK] Callback audit passed.")
    return True

if __name__ == "__main__":
    files_to_check = ["constants.py", "data_utils.py", "ml_utils.py", "streamlit_app.py", "styles_config.py"]
    
    s_pk = run_syntax_check(files_to_check)
    l_pk = test_polyglot_alex()
    a_pk = audit_streamlit_callbacks()
    
    if s_pk and l_pk and a_pk:
        print("\n[SUCCESS] COMPREHENSIVE TEST PASSED!")
    else:
        print("\n[FAILURE] TEST FAILED - CHECK LOGS ABOVE")
        sys.exit(1)
