# core/llm_openai.py
class OpenAIClient:
    def __init__(self, **kwargs):
        pass
    def generate(self, prompt, **kwargs):
        return f"[OpenAI] ตอบ: {prompt} (context={kwargs.get('context', [])})"
