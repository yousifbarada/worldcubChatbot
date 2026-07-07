import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.llm import GeminiLLM
from backend.embeddedmanger import EmbeddingManager
from backend.vectorstore import VectorStore
from backend.RAG import RAGRetriever
from backend.worldcubchatbot import WorldCupChatbot


v_store = VectorStore(collection_name="pdf_documents", persist_directory="data/vector_store")
embedding_manager = EmbeddingManager()
rag_retriever = RAGRetriever(vector_store=v_store, embedding_manager=embedding_manager)

llm = GeminiLLM()
chatbot = WorldCupChatbot(retriever=rag_retriever, llm=llm)

app = FastAPI(title="SBS Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path("frontend")

print(f"Serving frontend from {FRONTEND_DIR}")
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


@app.get("/api/health")
def health():
    return {"status": "ok", "documents_in_store": v_store.count()}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())
    try:
        result = chatbot.chat(session_id, req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(answer=result["answer"], sources=result["sources"], session_id=session_id)


@app.post("/api/reset")
def reset(req: ResetRequest):
    chatbot.reset(req.session_id)
    return {"status": "reset"}


# --- Serve the frontend ---
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))