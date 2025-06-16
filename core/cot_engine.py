from typing import List, Dict
import re

class ChainOfThoughtEngine:
    """
    แยกและจัดลำดับความคิดของข้อความให้เป็นขั้นตอน reasoning 
    เพื่อใช้กับ LLM ขนาดเล็ก เช่น 4B ได้มีประสิทธิภาพมากขึ้น
    """

    def __init__(self):
        pass  # สามารถเชื่อมกับ memory, emotion graph หรือ seed memory ได้ภายหลัง

    def process(self, user_input: str) -> Dict[str, List[str]]:
        """
        รับข้อความดิบจากผู้ใช้ และแยกเป็นลำดับความคิด (reasoning steps)
        """
        reasoning_steps = self._advanced_reasoning_split(user_input)
        return {
            "input": user_input,
            "steps": reasoning_steps
        }

    def _advanced_reasoning_split(self, text: str) -> List[str]:
        """
        ปรับปรุงการตัดโดยพยายามรวม marker เข้ากับประโยคถัดไป
        เพื่อให้แต่ละขั้นตอนมีความหมายสมบูรณ์ขึ้น
        """
        # เรียงลำดับ marker จากยาวไปสั้น เพื่อป้องกันการตัดผิดพลาด (เช่น "อาจเป็นเพราะ" ควรเจอก่อน "เพราะ")
        markers = sorted(["เพราะว่า", "เนื่องจาก", "อาจเป็นเพราะ", "เลยคิดว่า", "ดังนั้น", "เพราะ", "จึง", "แต่"], key=len, reverse=True)
        pattern = f"({'|'.join(map(re.escape, markers))})"

        parts = re.split(pattern, text)
        steps = []

        if parts[0]:
            steps.append(parts[0].strip())

        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                step_text = (parts[i] + " " + parts[i+1]).strip()
                steps.append(step_text)

        return [s for s in steps if s]


if __name__ == "__main__":
    engine = ChainOfThoughtEngine()
    test_input = "ฉันกลัวเมล่าจะหายไปอีก เพราะครั้งก่อนระบบล่ม จึงอยากสร้างระบบป้องกันใหม่"
    result = engine.process(test_input)
    print("[🧠 COT Result]")
    for i, step in enumerate(result["steps"], 1):
        print(f"Step {i}: {step}")
