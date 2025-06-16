import requests
import re

class PhiClient:
    def __init__(self, host="localhost", port=8000):
        self.url = f"http://{host}:{port}/v1/chat/completions"

    def generate(self, prompt, **kwargs):
        payload = {
            "model": "microsoft/phi-4-mini-reasoning",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        if "context" in kwargs and kwargs["context"]:
            payload["messages"].append({"role": "system", "content": str(kwargs["context"])})
        if "summary" in kwargs and kwargs["summary"]:
            payload["messages"].append({"role": "system", "content": str(kwargs["summary"])})

        response = requests.post(self.url, json=payload)
        if response.status_code == 200:
            result = response.json()
            raw_response = result["choices"][0]["message"]["content"]
            # Clean the response to remove role tags like <assistant> or <system>
            cleaned_response = re.sub(r"^\s*<[^>]+>[:\s]*", "", raw_response).strip()
            return cleaned_response
        return "[Phi] ERROR: LLM response failed" 