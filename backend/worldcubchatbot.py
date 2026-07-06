from typing import List, Dict, Any

from backend.llm import GeminiLLM


class WorldCupChatbot:

    def __init__(self, retriever, llm: GeminiLLM):
        self.retriever = retriever
        self.llm = llm
        # Per-session history: session_id -> list of {"role": "user"|"model", "content": ...}
        self.histories: Dict[str, List[Dict[str, str]]] = {}

    def chat(self, session_id: str, question: str) -> Dict[str, Any]:
        history = self.histories.setdefault(session_id, [])

        results = self.retriever.retrieve(question, top_k=3)
        context = "\n\n".join([doc['content'] for doc in results]) if results else "No context found."

        response_text = self.llm.invoke(question, context, history=history)

        # Save this turn
        history.append({"role": "user", "content": question})
        history.append({"role": "model", "content": response_text})

        sources = [
            {
                "source_file": doc.get("metadata", {}).get("source_file", "unknown"),
                "similarity_score": doc.get("similarity_score"),
            }
            for doc in results
        ]

        return {"answer": response_text, "sources": sources}

    def reset(self, session_id: str):
        self.histories.pop(session_id, None)
        print(f"History cleared for session {session_id}.")