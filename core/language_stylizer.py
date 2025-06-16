# core/language_stylizer.py

import logging
from typing import Dict, List, Optional
import re

# พยายาม import pythainlp ถ้าไม่มีจะใช้ logic พื้นฐาน
try:
    from pythainlp.tokenize import word_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    # เปลี่ยนเป็น logger.warning แทน print เพื่อให้จัดการ log ได้ดีขึ้น
    logging.warning("pythainlp not found. LanguageStylizer will use basic string operations.")

logger = logging.getLogger(__name__)

class LanguageStylizer:
    """
    รับ 'ความคิดสุดท้าย' ที่เป็นกลาง (Sterilized Thought) จาก Reflector
    มาเรียบเรียงเป็นภาษาพูดที่มีสไตล์ตาม Persona และความสัมพันธ์
    """
    def __init__(self, persona_config: Dict):
        """
        Args:
            persona_config (Dict): ข้อมูลบุคลิกของ AI จาก identity.json
        """
        self.persona = persona_config
        logger.info(f"🎨 LanguageStylizer initialized for persona: {self.persona.get('name', 'Unknown')}")

    def style_response(self, final_thought: str, relationship_level: str) -> str:
        """
        เรียบเรียงข้อความ (แต่งตัวให้กับความคิด)

        Args:
            final_thought: ความคิดสุดท้ายที่เป็นกลางจาก Reflector
            relationship_level: ระดับความสัมพันธ์ปัจจุบัน ("formal", "friendly", "intimate")

        Returns:
            str: ข้อความที่เรียบเรียงแล้ว พร้อมสำหรับแสดงผล
        """
        logger.info(f"Stylizer: Styling neutral thought: '{final_thought[:100]}...' for level: {relationship_level}")
        
        # --- 1. กำหนดค่าพื้นฐานจาก Persona ---
        gender = self.persona.get("gender", "female")
        ai_name = self.persona.get('name', 'AI')
        
        # --- 2. เลือกสรรพนาม (Pronouns) ---
        pronoun_i_map = self.persona.get("pronoun_i", {"formal": ai_name, "friendly": "ฉัน"})
        pronoun_you_map = self.persona.get("pronoun_you", {"formal": "คุณ", "friendly": "คุณ"})

        pronoun_i = pronoun_i_map.get(relationship_level, pronoun_i_map.get("friendly", "ฉัน"))
        pronoun_you = pronoun_you_map.get(relationship_level, pronoun_you_map.get("friendly", "คุณ"))

        # --- 3. เลือกคำลงท้าย (Particles) ---
        particle_statement = "ค่ะ" if gender == "female" else "ครับ"
        particle_question = "คะ" if gender == "female" else "ครับ"

        # --- 4. ประกอบร่าง (Styling) ---
        # แทนที่ placeholder ทั้งหมดในความคิดที่เป็นกลาง
        styled_text = final_thought.replace("{{pronoun_i}}", pronoun_i)
        styled_text = styled_text.replace("{{pronoun_you}}", pronoun_you)
        styled_text = styled_text.replace("{{particle_statement}}", particle_statement)
        styled_text = styled_text.replace("{{particle_question}}", particle_question)
        
        # --- 5. Final Cleanup ---
        # จัดการกับปัญหาที่อาจหลุดรอดมา
        styled_text = self._final_cleanup(styled_text)
        
        logger.info(f"Stylizer: Final styled text: '{styled_text[:100]}...'")
        return styled_text

    def _final_cleanup(self, text: str) -> str:
        """
        เก็บกวาดข้อความขั้นตอนสุดท้ายเพื่อให้สวยงาม
        """
        # ลบข้อความในวงเล็บท้ายประโยค
        cleaned_text = re.sub(r'\s*\([^)]*\)$', '', text.strip()).strip()

        # แก้ไขปัญหา "คะ/ครับ" หรือ "ครับ/ค่ะ"
        if "คะ/ครับ" in cleaned_text:
            particle = "คะ" if self.persona.get("gender", "female") == "female" else "ครับ"
            cleaned_text = cleaned_text.replace("คะ/ครับ", particle)
        if "ครับ/ค่ะ" in cleaned_text:
            particle = "ค่ะ" if self.persona.get("gender", "female") == "female" else "ครับ"
            cleaned_text = cleaned_text.replace("ครับ/ค่ะ", particle)
            
        # อาจเพิ่ม cleanup อื่นๆ ได้ในอนาคต
        return cleaned_text

# --- ตัวอย่างการใช้งาน ---
if __name__ == '__main__':
    # ... (ส่วนนี้เหมือนเดิม) ...
    pass