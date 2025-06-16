import requests

class GemmaClient:
    def __init__(self, host="localhost", port=8000):
        self.url = f"http://{host}:{port}/v1/chat/completions"

    def generate(self, prompt, **kwargs):
        payload = {
            "model": "gemma-3-4b-it",  # ใส่ชื่อ model ตรงนี้
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        # เพิ่ม context/summary ถ้าต้องการ
        if "context" in kwargs and kwargs["context"]:
            payload["messages"].append({"role": "system", "content": str(kwargs["context"])})
        if "summary" in kwargs and kwargs["summary"]:
            payload["messages"].append({"role": "system", "content": str(kwargs["summary"])})

        response = requests.post(self.url, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        return "[Gemma] ERROR: LLM response failed"
