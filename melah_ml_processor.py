# core/melah_ml_processor.py
import logging
from typing import List, Tuple
import os
import shutil
import datetime
import joblib
import json

logger = logging.getLogger(__name__)

class MelahMLProcessor:
    """Mock MelahMLProcessor for testing"""
    
    def __init__(self, **kwargs):
        logger.info("🧠 Mock MelahMLProcessor initialized.")
        
    def get_context_relevance_scores(self, texts: List[str], query: str) -> List[Tuple[str, float]]:
        """Mock method to simulate relevance scoring"""
        return [(text, 0.5) for text in texts]
        
    def analyze_patterns(self, data: List[dict]) -> dict:
        """Mock method to simulate pattern analysis"""
        return {
            "patterns_found": 2,
            "confidence_score": 0.8
        }

    def predict_user_intent(self, user_input: str, conversation_history: list[dict] = None) -> dict:
        """
        (ML/NLP) ทำนายเจตนา (intent) ของผู้ใช้จาก input และประวัติการสนทนา
        อาจจะคืนค่าเป็น {"intent_label": "ask_question", "confidence": 0.9, "entities": {...}}
        (Placeholder)
        """
        print(f"🧠 ML: Predicting user intent for: '{user_input[:50]}...'")
        intent = "unknown_intent"
        confidence = 0.5
        user_input_lower = user_input.lower()

        if "คืออะไร" in user_input_lower or "?" in user_input_lower or "บอกหน่อย" in user_input_lower:
            intent = "question_asking"
            confidence = 0.8
        elif "สวัสดี" in user_input_lower or "ขอบคุณ" in user_input_lower or "เยี่ยม" in user_input_lower: #
            intent = "greeting_or_feedback"
            confidence = 0.75
        elif "ทำไม" in user_input_lower:
            intent = "seeking_explanation"
            confidence = 0.7
        
        return {"intent_label": intent, "confidence": confidence, "entities_placeholder": []}

    def update_weights(self, concept_data: dict): #
        """ (Placeholder) อัปเดต weights หรือ logic ภายในของ ML model بناءً على concept ใหม่ """
        concept_text = concept_data.get("concept_text", "unknown_concept")
        print(f"  🧠 ML: Placeholder - Would update internal model/weights with concept: '{concept_text[:50]}...'")
        # Logic for model retraining or online learning would go here in the future

    def retrain_model(self, new_data):
        self.model.fit(new_data['X'], new_data['y'])
        os.makedirs('models/staging', exist_ok=True)
        staging_path = 'models/staging/ml_model_candidate.pkl'
        joblib.dump(self.model, staging_path)
        print(f"[ML] Candidate model saved to {staging_path}. รอการอนุมัติ.")

    def approve_model(self):
        os.makedirs('models/backup', exist_ok=True)
        main_path = 'models/ml_model_latest.pkl'
        staging_path = 'models/staging/ml_model_candidate.pkl'
        if not os.path.exists(staging_path):
            print("[ML] No candidate model to approve.")
            return
        if os.path.exists(main_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"models/backup/ml_model_{timestamp}.pkl"
            shutil.copy2(main_path, backup_path)
            print(f"[ML] Backup saved to {backup_path}")
        shutil.move(staging_path, main_path)
        print(f"[ML] Model updated: {main_path}")

    def rollback_model(self, backup_path):
        main_path = 'models/ml_model_latest.pkl'
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, main_path)
            print(f"[ML] Model rolled back to {backup_path}")
        else:
            print(f"[ML] Backup not found: {backup_path}")

    def log_learning_event(self, event: dict):
        """บันทึก learning event/insight ลงไฟล์ staging"""
        os.makedirs('models', exist_ok=True)
        staging_path = 'models/ml_learning_staging.json'
        with open(staging_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
        print(f"[ML] Learning event logged to {staging_path}")

if __name__ == "__main__":
    ml_test = MelahMLProcessor()
    core_defs_for_scoring = [
        "Truth Core Principles: Be factual, no hallucination, acknowledge uncertainty.", #
        "Emotional Core Guide: Express warmth, empathy, and natural emotional responses.", #
        "Ethical Core Rules: Prioritize user safety, do no harm, respect freedom." #
    ]
    query = "What are your ethical rules regarding safety?"
    scored_defs = ml_test.get_context_relevance_scores(core_defs_for_scoring, query)
    print(f"\nScored Core Definitions for query '{query}':")
    for item, score in scored_defs:
        print(f"  Score: {score:.2f} - Item: {item}")

    intent1 = ml_test.predict_user_intent("สวัสดีจ้ะเมล่า สบายดีไหมเอ่ย?")
    print(f"Predicted Intent 1: {intent1}")
    intent2 = ml_test.predict_user_intent("หลักการ Truth Core ของคุณมีอะไรบ้าง?")
    print(f"Predicted Intent 2: {intent2}")