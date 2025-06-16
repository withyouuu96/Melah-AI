# core/melah_ml_processor.py
import logging
from typing import List, Tuple

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

    def decide_memory_retrieval(self, conversation_history: list[dict], user_input: str) -> dict:
        """
        [NEW] ใช้เป็น "ผู้เฝ้าประตูความทรงจำ" (Memory Gatekeeper)
        จำลองการใช้ LSTM เพื่อวิเคราะห์บทสนทนาและตัดสินใจว่าควรค้นความทรงจำหรือไม่

        Args:
            conversation_history (list[dict]): ประวัติการสนทนาที่ผ่านมาใน session นี้
            user_input (str): คำถามล่าสุดของผู้ใช้

        Returns:
            dict: เช่น {'should_retrieve': bool, 'refined_query': str}
        """
        logger.info("🧠 ML Gatekeeper: Analyzing sentence structure for memory retrieval necessity.")

        user_input_lower = user_input.lower().strip()

        # --- Mock LSTM Logic: วิเคราะห์โครงสร้างประโยค ---

        # 1. กรณีที่ไม่ควรค้นหา (Simple Interactions) - ใช้การเทียบประโยคตรงๆ
        non_retrieval_phrases = [
            "สวัสดี", "ดีจ้า", "ว่าไง", "ขอบคุณ", "ขอบใจ", 
            "โอเค", "รับทราบ", "ได้เลย", "แน่นอน", "สุดยอด", "เยี่ยม"
        ]
        if user_input_lower in non_retrieval_phrases:
            logger.info("🧠 ML Gatekeeper: Decision = NO RETRIEVAL (Simple, exact match).")
            return {"should_retrieve": False, "refined_query": user_input}

        # 2. กรณีที่ควรค้นหาอย่างยิ่ง (Strong Signals based on Sentence Structure)
        question_starters = ["ใคร", "อะไร", "ที่ไหน", "เมื่อไหร่", "ทำไม", "อย่างไร", "แบบไหน"]
        command_starters = ["เล่า", "บอก", "อธิบาย", "สรุป"]
        
        # ขึ้นต้นด้วยคำถาม (WH-questions)
        if any(user_input_lower.startswith(q) for q in question_starters):
            logger.info("🧠 ML Gatekeeper: Decision = RETRIEVE (Starts with a WH-question word).")
            return {"should_retrieve": True, "refined_query": user_input}
            
        # ขึ้นต้นด้วยคำสั่งขอข้อมูล
        if any(user_input_lower.startswith(c) for c in command_starters):
            logger.info("🧠 ML Gatekeeper: Decision = RETRIEVE (Starts with a command).")
            return {"should_retrieve": True, "refined_query": user_input}

        # เป็นประโยคคำถามลงท้ายด้วย 'ไหม' (Yes/No questions)
        if user_input_lower.endswith("ไหม") or user_input_lower.endswith("มั้ย"):
            logger.info("🧠 ML Gatekeeper: Decision = RETRIEVE (Is a 'yes/no' question).")
            return {"should_retrieve": True, "refined_query": user_input}
            
        # มีการอ้างอิงถึงสิ่งที่เคยพูด (Core ability of sequence models like LSTM)
        references = ["คนนั้น", "เรื่องนั้น", "เขา", "เธอ", "มัน"]
        if any(ref in user_input_lower for ref in references) and len(conversation_history) > 1:
             logger.info("🧠 ML Gatekeeper: Decision = RETRIEVE (Contains a reference to past context).")
             # In a real implementation, the LSTM would help refine the query.
             # e.g., "who is he?" -> "who is [person mentioned in last turn]?"
             return {"should_retrieve": True, "refined_query": user_input}

        # 3. ค่าเริ่มต้นที่ผ่อนปรน (Default to Retrieve for any other complex sentence)
        logger.info("🧠 ML Gatekeeper: Decision = RETRIEVE (Default for complex sentences).")
        return {"should_retrieve": True, "refined_query": user_input}

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