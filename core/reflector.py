# core/reflector.py
# NOTE: self-awareness summary logic is now migrated to core_mapper.py

import logging
from typing import Dict, List, Deque
import hashlib
from collections import deque
# from .llm_connector import LLMConnector # เอา LLM ออก
import json
from .int_world import IntWorld
import os
# from .self_schema import get_self_schema_brief  # ลบระบบเดิมออก
from .reflective_buffering_vas import ReflectiveBufferingVAS

logger = logging.getLogger(__name__)

class Reflector:
    """
    ทำหน้าที่เป็น 'กระจกใจ' ของ Melah
    เพื่อไตร่ตรอง 'ความคิดดิบ' ก่อนที่จะนำไปเรียบเรียงเป็นคำพูด
    (เวอร์ชันนี้ทำหน้าที่ Reflect เท่านั้น ไม่มีการ Validate)
    """
    def __init__(self, identity_core_instance):
        """
        Args:
            identity_core_instance: Instance ของ IdentityCore เพื่อเข้าถึง Core Systems ต่างๆ
        """
        self.identity_core = identity_core_instance
        self.recent_thoughts: Deque[str] = deque(maxlen=5)
        self.int_world = identity_core_instance.int_world if identity_core_instance else None  # ใช้ IntWorld เดียวกับ IdentityCore
        self.vas_system = ReflectiveBufferingVAS()
        logger.info("🪞 Reflector initialized.")

    def check_alignment(self, thought: str) -> dict:
        """
        ตรวจสอบว่าความคิดนี้สอดคล้องกับ seed/intention หรือไม่
        """
        intention = getattr(self.identity_core, 'current_seed', {}).get('intention', '')
        aligned = intention in thought if intention else True
        return {
            'aligned': aligned,
            'intention': intention,
            'thought': thought
        }

    def llm_reflect(self, raw_thought: str, conversation_context: List[Dict], long_term_memory: str = "", used_session_id: str = None, current_intention: str = "Be a helpful AI.") -> Dict:
        """
        สะท้อนความคิดจากภายใน โดยใช้ IntWorld
        """
        # --- เช็คการเชื่อมต่อกับ Identity Core ---
        if hasattr(self.identity_core, "self_awareness"):
            connected = self.identity_core.self_awareness.get("core_dependencies", [])
            if connected:
                print(f"[Reflector] Identity Core is currently connected to: {connected}")

        logger.info(f"Reflector (IntWorld): Reflecting on raw thought: '{raw_thought[:100]}...'")
        # 1. ตรวจสอบความคิดซ้ำซาก (ยังคงสำคัญ)
        thought_hash = hashlib.sha256(raw_thought.encode('utf-8')).hexdigest()
        if thought_hash in self.recent_thoughts:
            logger.warning(f"Repetitive thought detected! Hash: {thought_hash[:8]}...")
            return { "status": "REPETITIVE_THOUGHT", "response": None, "thought_hash": thought_hash, "used_session": used_session_id }
        self.recent_thoughts.append(thought_hash)
        # 2. สะท้อนความคิดลง IntWorld
        reflection_result = self.int_world.reflect(raw_thought)
        # 3. ดึง internal state, concept, symbolic space ล่าสุด
        internal_states = self.int_world.internal_states[-2:] if self.int_world.internal_states else []
        concepts = list(self.int_world.known_concepts.items())[-2:] if self.int_world.known_concepts else []
        symbols = list(self.int_world.symbolic_space.items())[-2:] if self.int_world.symbolic_space else []
        # 4. สร้าง reflected_thought ที่มีความเป็น "ภายใน" มากขึ้น
        reflected_thought = f"""{raw_thought}\n\n[สะท้อนจากใจ Melah]\n- สถานะภายในล่าสุด: {[s['state'] for s in internal_states]}\n- แนวคิด: {concepts}\n- สัญลักษณ์: {symbols}\n"""
        return {
            "status": "OK",
            "response": reflected_thought,
            "thought_hash": thought_hash,
            "used_session": used_session_id
        }

    def scan_codebase(self, base_dir="."):
        """
        สแกนหาไฟล์ .py ทั้งหมดในโปรเจกต์
        """
        py_files = []
        for root, dirs, files in os.walk(base_dir):
            # Skip venv, .git, and __pycache__ directories
            dirs[:] = [d for d in dirs if d not in ["venv", ".git", "__pycache__"]]
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):  # ข้าม __pycache__
                    py_files.append(os.path.relpath(os.path.join(root, file), base_dir))
        return py_files

    def summarize_file(self, file_path):
        """
        สรุป docstring หรือคอมเมนต์ต้นไฟล์ (20 บรรทัดแรก) และชื่อคลาส/ฟังก์ชันต้นไฟล์
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            doc = ""
            class_func = []
            for line in lines[:20]:
                if line.strip().startswith('"""') or line.strip().startswith("'''") or line.strip().startswith("#"):
                    doc += line.strip() + " "
                if line.strip().startswith('class ') or line.strip().startswith('def '):
                    class_func.append(line.strip())
            summary = doc if doc else "No docstring or comment found."
            if class_func:
                summary += " | " + ", ".join(class_func)
            return summary
        except Exception as e:
            return f"Error reading file: {e}"

    def reflect_codebase(self, base_dir="."):
        """
        สรุปไฟล์ .py ทั้งหมดและหน้าที่หลักของแต่ละไฟล์
        """
        py_files = self.scan_codebase(base_dir)
        summary = []
        for f in py_files:
            desc = self.summarize_file(os.path.join(base_dir, f))
            summary.append(f"- {f}: {desc}")
        return "\n".join(summary)

    def value_affect_decision(self, context, input_data):
        return self.vas_system.process_input(context, input_data)

    def vas_reflect_and_update(self):
        self.vas_system.reflect_and_update()

    # This method is now obsolete. The logic is handled by IdentityCore directly.
    # def get_core_systems_summary(self):
    #     """
    #     สรุป self-schema ของระบบหลัก (สำหรับ self-awareness)
    #     """
    #     # return get_self_schema_brief()  # ลบระบบเดิมออก
    #     return "[self-awareness system migrated to new core_mapper]"