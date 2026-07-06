import os
import google.generativeai as genai
from typing import List, Dict, Optional

class GeminiLLM:

    SYSTEM_PROMPT = """ """

    def __init__(self, api_key: Optional[str] = None):
        # Allow API key via parameter or environment variable
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key must be provided via constructor or GEMINI_API_KEY env var")
        genai.configure(api_key=self.api_key)

        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(temperature=0.4, max_output_tokens=2048),
            system_instruction=self.SYSTEM_PROMPT      # system role
        )
        print("Gemini LLM ready")

    def invoke(self, query: str, context: str) -> str:
        # Bundle context + question and send via the model API
        user_message = f"Context:\n{context}\nQuestion: {query}"

        chat = self.model.start_chat(history=[])
        response = chat.send_message(user_message)
        return response.text

    def clear_chat(self):
        # No persistent chat state in this wrapper; call start_chat to reset session
        self.model.start_chat(history=[])
            

