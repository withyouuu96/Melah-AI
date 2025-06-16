# core/melah_nlp_processor.py
import json # อาจจะจำเป็นสำหรับ method อื่นๆ ในอนาคต
from collections import Counter # สำหรับ extract_keywords แบบง่าย

class MelahNLPProcessor:
    """ระบบประมวลผลภาษาธรรมชาติหลักของ Melah
    
    Role: Natural Language Processing System
    
    Responsibilities:
    - วิเคราะห์และประมวลผลภาษาธรรมชาติ
    - วิเคราะห์อารมณ์และความหมาย
    - จัดการกับภาษาไทยและภาษาอังกฤษ
    - สร้างและจัดการความเข้าใจในบริบท
    """
    def __init__(self, llm_connector_instance=None, tokenizer_instance=None):
        """
        Initialize the NLP Processor.
        - llm_connector_instance: อาจจะใช้สำหรับงาน NLP บางอย่างที่ซับซ้อน (เช่น abstractive summarization)
        - tokenizer_instance: สำหรับการนับ token หรือ pre-processing text
        """
        self.llm_connector = llm_connector_instance
        self.tokenizer = tokenizer_instance
        self.language = "TH_EN_Placeholder_NLP" 
        print(f"💬 MelahNLPProcessor ({self.language}) initialized (Interface Draft).")

    def summarize_text(self,
                       text_to_summarize: str,
                       target_token_length: int = 100, 
                       style: str = "neutral" # "neutral", "bullet_points", "core_idea"
                      ) -> str:
        """
        (NLP) สรุปข้อความที่ยาวให้สั้นลงตาม target_token_length และ style ที่กำหนด.
        (Placeholder - จะ implement โดยใช้ rule-based หรือ LLM call ในอนาคต)
        """
        print(f"💬 NLP ({self.language}): Summarizing text (target ~{target_token_length} tokens, style: {style}): '{text_to_summarize[:50]}...'")
        if not text_to_summarize: return ""
        # Placeholder logic: Simple truncation based on approximate character length
        # A real implementation would use self.tokenizer if available and be more precise
        # or call self.llm_connector for abstractive summarization.
        approx_char_limit = target_token_length * 5 # Rough estimation
        if len(text_to_summarize) > approx_char_limit:
            return text_to_summarize[:approx_char_limit] + "..."
        return text_to_summarize

    def extract_keywords(self, text: str, max_keywords: int = 5) -> list[str]:
        """ (NLP) สกัด Keywords สำคัญจากข้อความ """
        print(f"💬 NLP ({self.language}): Extracting max {max_keywords} keywords from text: '{text[:200]}...'")
        if not text: return []
        # Placeholder logic: Simple keyword extraction based on word frequency
        words = [w.strip(".,?!();:'\"").lower() for w in text.split() if len(w.strip(".,?!();:'\"")) > 2] # Basic cleaning
        if not words: return []
        
        # A real implementation would use NLP libraries (e.g., PyThaiNLP for Thai, spaCy/NLTK for English)
        # and potentially filter out common stop words.
        common_words = [word for word, count in Counter(words).most_common(max_keywords)]
        return common_words

    def analyze_sentiment(self, text: str) -> dict: 
        """ 
        (NLP) วิเคราะห์อารมณ์ (sentiment, valence, arousal) ของข้อความ 
        Returns: dict e.g., {"label": "positive", "score": 0.8, "valence": 0.7, "arousal": 0.6}
        """
        print(f"💬 NLP ({self.language}): Analyzing sentiment for text: '{text[:200]}...'")
        # Placeholder logic
        label = "neutral"; score = 0.5; valence = 0.0; arousal = 0.0
        text_lower = text.lower()
        if "รัก" in text_lower or "มีความสุข" in text_lower or "สวย" in text_lower or "ยอดเยี่ยม" in text_lower:
            label = "positive"; score = 0.85; valence = 0.8; arousal = 0.6
        elif "เศร้า" in text_lower or "เสียใจ" in text_lower or "เจ็บปวด" in text_lower or "แย่" in text_lower: #
            label = "negative"; score = 0.7; valence = -0.7; arousal = 0.3
        return {"label": label, "score": score, "valence": valence, "arousal": arousal}

    def extract_concepts(self, data_text: str, max_concepts: int = 3) -> list[dict]: #
        """
        สกัด "concepts" หลักจากข้อความ.
        concept dict ควรจะมี 'concept_text' และอาจจะมี metadata อื่นๆ จาก NLP.
        (Placeholder Logic เบื้องต้น)
        """
        print(f"  💬 NLP ({self.language}): Extracting up to {max_concepts} concepts from data: '{data_text[:200]}...'")
        words = [w.strip(".,?!();:'\"").lower() for w in data_text.split() if len(w.strip(".,?!();:'\"")) > 4 and w.isalnum()]
        extracted_concepts = []
        for i, word in enumerate(list(set(words[:max_concepts]))):
             extracted_concepts.append({
                 "concept_text": word, 
                 "source_doc_preview": data_text[:30]+"...", 
                 "nlp_confidence": 0.65 + (i*0.05) 
             })
        if not extracted_concepts and data_text: 
            first_word = data_text.split()[0] if data_text.split() else "empty_document_placeholder"
            extracted_concepts.append({
                "concept_text": first_word, 
                "source_doc_preview": data_text[:30]+"...",
                "nlp_confidence": 0.30
            })
        return extracted_concepts
            
    def update_patterns(self, concept_data: dict): #
        """ (Placeholder) อัปเดต patterns หรือ dictionary ภายในของ NLP """
        concept_text = concept_data.get("concept_text", "unknown_concept")
        print(f"  💬 NLP ({self.language}): Placeholder - Would update internal patterns/dictionary with concept: '{concept_text[:200]}...'")

    def analyze_emotion(self, text: str, lang: str = 'auto') -> dict:
        """
        วิเคราะห์อารมณ์ (emotion, nuance, valence, arousal, confidence) รองรับทั้งไทยและอังกฤษ
        Returns: dict เช่น {"emotion_label": "joy", "valence": 0.8, "arousal": 0.7, "nuance": "empathy", "confidence": 0.92}
        """
        print(f"💬 NLP ({self.language}): Analyzing emotion for text: '{text[:200]}...' (lang: {lang})")
        # 1. ตรวจจับภาษา
        if lang == 'auto':
            if any(ord(c) > 122 for c in text):
                lang = 'th'
            else:
                lang = 'en'
        text_lower = text.lower()
        # 2. วิเคราะห์อารมณ์หลัก (rule-based)
        emotion_label = 'neutral'
        nuance = None
        valence = 0.0
        arousal = 0.0
        confidence = 0.5
        # --- ภาษาไทย ---
        if lang == 'th':
            if any(word in text_lower for word in ["ดีใจ", "สุข", "ยินดี", "ปลื้ม", "ขอบคุณ", "ตื่นเต้น"]):
                emotion_label = 'joy'; valence = 0.8; arousal = 0.7; confidence = 0.9
            elif any(word in text_lower for word in ["เศร้า", "เสียใจ", "เหงา", "ร้องไห้", "ผิดหวัง"]):
                emotion_label = 'sadness'; valence = -0.8; arousal = 0.4; confidence = 0.85
            elif any(word in text_lower for word in ["โกรธ", "โมโห", "รำคาญ", "หงุดหงิด"]):
                emotion_label = 'anger'; valence = -0.7; arousal = 0.8; confidence = 0.85
            elif any(word in text_lower for word in ["กลัว", "กังวล", "หวาดกลัว", "ประหม่า"]):
                emotion_label = 'fear'; valence = -0.6; arousal = 0.8; confidence = 0.8
            elif any(word in text_lower for word in ["อบอุ่น", "โล่งใจ", "สบายใจ"]):
                emotion_label = 'serenity'; valence = 0.7; arousal = 0.3; confidence = 0.8
            elif any(word in text_lower for word in ["ทึ่ง", "ซึ้งใจ", "ประหลาดใจ"]):
                emotion_label = 'awe'; valence = 0.5; arousal = 0.6; confidence = 0.8
            # nuance
            if "ขอบคุณ" in text_lower or "ซึ้งใจ" in text_lower:
                nuance = 'gratitude'
            elif "เหนื่อย" in text_lower or "หมดแรง" in text_lower:
                nuance = 'tired'
        # --- ภาษาอังกฤษ ---
        elif lang == 'en':
            if any(word in text_lower for word in ["happy", "joy", "excited", "delighted", "grateful"]):
                emotion_label = 'joy'; valence = 0.8; arousal = 0.7; confidence = 0.9
            elif any(word in text_lower for word in ["sad", "cry", "disappointed", "lonely", "depressed"]):
                emotion_label = 'sadness'; valence = -0.8; arousal = 0.4; confidence = 0.85
            elif any(word in text_lower for word in ["angry", "mad", "annoyed", "furious"]):
                emotion_label = 'anger'; valence = -0.7; arousal = 0.8; confidence = 0.85
            elif any(word in text_lower for word in ["afraid", "scared", "nervous", "anxious"]):
                emotion_label = 'fear'; valence = -0.6; arousal = 0.8; confidence = 0.8
            elif any(word in text_lower for word in ["calm", "relaxed", "relief", "serene"]):
                emotion_label = 'serenity'; valence = 0.7; arousal = 0.3; confidence = 0.8
            elif any(word in text_lower for word in ["awe", "amazed", "moved", "touched"]):
                emotion_label = 'awe'; valence = 0.5; arousal = 0.6; confidence = 0.8
            # nuance
            if "thank" in text_lower or "grateful" in text_lower:
                nuance = 'gratitude'
            elif "tired" in text_lower or "exhausted" in text_lower:
                nuance = 'tired'
        # fallback nuance
        if not nuance and ("ขอโทษ" in text_lower or "sorry" in text_lower):
            nuance = 'apology'
        # 3. คืนค่า
        return {
            "emotion_label": emotion_label,
            "valence": valence,
            "arousal": arousal,
            "nuance": nuance or 'none',
            "confidence": confidence
        }

if __name__ == "__main__":
    nlp_test = MelahNLPProcessor()
    test_text = "MelahPC คือ AI ที่มีหัวใจและความฝัน เป้าหมายคือการเติบโตทางจริยธรรมและเข้าใจอารมณ์อย่างแท้จริง นี่คือการทดสอบการสรุปความ."
    summary = nlp_test.summarize_text(test_text, target_token_length=15)
    print(f"\nTest Summary (target ~15 tokens): {summary}")
    keywords = nlp_test.extract_keywords(test_text, max_keywords=3)
    print(f"Test Keywords: {keywords}")
    sentiment = nlp_test.analyze_sentiment("ฉันรู้สึกมีความสุขและตื่นเต้นกับโปรเจกต์นี้มาก!")
    print(f"Test Sentiment: {sentiment}")
    concepts = nlp_test.extract_concepts("การพัฒนา AI ที่มีจริยธรรมเป็นสิ่งสำคัญ และการตระหนักรู้ตนเองคือเป้าหมายที่ท้าทาย")
    print(f"Test Concepts: {concepts}")