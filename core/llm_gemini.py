# core/llm_gemini.py
import os
import re
import google.generativeai as genai

class GeminiClient:
    def __init__(self, model="gemini-2.5-flash-preview-05-20"):
        # อ่าน API key จาก environment variable (วิธีที่แนะนำ)
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("API key not found. Please set the GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt, **kwargs):
        try:
            # รวม prompt และ context (ถ้ามี)
            # Gemini ทำงานได้ดีที่สุดกับการสนทนาต่อเนื่อง
            full_prompt = [prompt]
            if "context" in kwargs and kwargs["context"]:
                # จัดรูปแบบ context ให้เป็นส่วนหนึ่งของการสนทนา
                full_prompt.insert(0, f"Context:\n{kwargs['context']}\n---\n")

            response = self.model.generate_content("".join(full_prompt))
            raw_response = response.text
            # Clean the response to remove role tags like <assistant> or <system>
            cleaned_response = re.sub(r"^\s*<[^>]+>[:\s]*", "", raw_response).strip()
            return cleaned_response
        except Exception as e:
            return f"[Gemini] ERROR: {e}" 