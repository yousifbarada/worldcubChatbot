
from typing import List, Dict

from llm import GeminiLLM


class WorldCupChatbot:

    def __init__(self, retriever, llm: GeminiLLM):
        self.retriever = retriever
        self.llm = llm
        self.history: List[Dict[str, str]] = []  # {"role": "user"|"model", "content": ...}

    def chat(self, question: str) -> str:
        results = self.retriever.retrieve(question, top_k=3)
        context = "\n\n".join([doc['content'] for doc in results]) if results else "No context found."

        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        # Use the GeminiLLM interface instead of touching the model directly
        response_text = self.llm.invoke(question, context)

        # Save this turn
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "model", "content": response_text})

        return response_text

    def reset(self):
        self.history = []
        print("Gemini LLM ready for new conversation. History cleared.")
