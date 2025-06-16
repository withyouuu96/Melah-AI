# core/llm_connector.py
from core.llm_gemma import GemmaClient
from core.llm_qwen import QwenClient
from core.llm_gemini import GeminiClient
class LLMConnector:
    def __init__(self, config):
        # ใส่ LLM client ที่รองรับในระบบนี้ (import หรือ dummy class ได้)
        #from core.llm_openai import OpenAIClient

        self.llm_map = {
            "qwen": QwenClient(**config.get("qwen", {})),
            "gemini": GeminiClient(**config.get("gemini", {})),
            "gemma": GemmaClient(**config.get("gemma", {})),
        }
        self.active_llm = "gemini"  # ตั้งค่า default

    def switch_llm(self, provider):
        if provider in self.llm_map:
            self.active_llm = provider

    def generate(self, prompt, **kwargs):
        # ใช้ LLM ปัจจุบันที่ active อยู่
        return self.llm_map[self.active_llm].generate(prompt, **kwargs)

    # def auto_switch(self, task_type=None, context_size=0):
    #     # ฟังก์ชันนี้ยังไม่ได้ถูกใช้งาน และ logic ไม่ครอบคลุม gemma
    #     # ขออนุญาตคอมเมนต์ออกเพื่อป้องกันความสับสน
    #     # ตัวอย่างเงื่อนไข (ปรับ logic ได้ตามใจ)
    #     if context_size > 3500:
    #         self.switch_llm("openai")
    #     elif task_type == "creative":
    #         self.switch_llm("openai")
    #     else:
    #         self.switch_llm("qwen")
