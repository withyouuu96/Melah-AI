import textwrap
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .identity_core import IdentityCore
    # from .llm_connector import LLMConnector # เอา LLM ออก

from .reflective_buffering_vas import ReflectiveBufferingVAS

class RefineLayer:
    # เอา llm_client ออกจาก __init__
    def __init__(self, identity_core: 'IdentityCore'):
        self.identity = identity_core
        self.vas_system = ReflectiveBufferingVAS()

    def build_prompt(self, reflected_thought: str) -> str:
        """
        สร้าง prompt สำหรับให้ LLM-Refiner ทำการขัดเกลาภาษาขั้นสุดท้าย
        (โค้ดส่วนนี้จะไม่ได้ถูกใช้เรียก LLM แต่คงไว้เพื่อรักษาโครงสร้างเดิม)
        """
        persona = self.identity.identity_data.get("identity", {})
        name = persona.get("name", "เมล่า")
        style = persona.get("style", "เมตตา, ฉลาด, และมีความเห็นอกเห็นใจ")
        language = persona.get("language", "ภาษาไทย")

        prompt = f"""
        คุณคือ AI ชื่อ {name} ซึ่งทำหน้าที่เป็น 'ผู้ขัดเกลาภาษา' (Refiner)
        หน้าที่ของคุณคือการนำ "ความคิดที่ไตร่ตรองแล้ว" (ซึ่งมีเนื้อหาและเหตุผลดีอยู่แล้ว)
        มาปรับแก้สำนวนและน้ำเสียงให้เป็นธรรมชาติและสอดคล้องกับบุคลิกของคุณมากที่สุด

        [ข้อมูลพื้นฐาน]
        ชื่อ: {name}
        สไตล์น้ำเสียง: {style}

        [ความคิดที่ไตร่ตรองแล้ว (เนื้อหาดีแล้ว)]
        {reflected_thought}

        [คำสั่งงาน]
        จงขัดเกลาข้อความข้างต้นให้มีน้ำเสียงเป็นธรรมชาติและสละสลวยตามสไตล์ที่กำหนด โดยไม่ต้องเปลี่ยนแปลงใจความสำคัญของเนื้อหา
        - ทำให้ภาษาเหมือนมนุษย์มากขึ้น
        - ตอบเป็น {language} ที่สมบูรณ์แบบ

        คำตอบสุดท้ายที่ขัดเกลาแล้ว:
        """
        return textwrap.dedent(prompt).strip()

    def refine(self, reflected_thought: str) -> str:
        # คงการสร้าง prompt ไว้ตามเดิม แต่ไม่ได้ใช้เรียก LLM
        prompt = self.build_prompt(reflected_thought)
        
        # [เอา LLM ออก] ส่งต่อความคิดที่ไตร่ตรองแล้วออกไปเลย
        return reflected_thought

    def refine_and_log(self, reflected_thought: str) -> Dict[str, str]:
        """
        ทำการ refine และคืนค่าในรูปแบบ structured dict สำหรับ identity_core
        """
        refined = self.refine(reflected_thought)
        return {
            "reflected": reflected_thought,
            "refined": refined
        }

    def value_affect_decision(self, context, input_data):
        return self.vas_system.process_input(context, input_data)

    def vas_reflect_and_update(self):
        self.vas_system.reflect_and_update()