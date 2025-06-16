import sys
import os
import asyncio

# Add the project root to the Python path to allow imports from 'core'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.llm_gemini import GeminiClient

def test_gemini_connection():
    """
    Tests the connection to the Gemini API.
    """
    print("Initializing GeminiClient...")
    try:
        # The API key is hardcoded in your llm_gemini.py file,
        # so we don't need to pass any config here.
        client = GeminiClient()
        print(f"GeminiClient initialized successfully. Using model: {client.model.model_name}")

        prompt = "Hello, Gemini! Please introduce yourself in one sentence."
        print(f"\nSending prompt: '{prompt}'")
        
        response = client.generate(prompt)
        
        print("\n--- Gemini Response ---")
        print(response)
        print("-----------------------\n")

        if "[Gemini] ERROR" in response:
            print("Test failed. An error occurred.")
        else:
            print("Test successful!")

    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")
        print("Please double-check your API key in core/llm_gemini.py and your network connection.")

if __name__ == "__main__":
    test_gemini_connection() 