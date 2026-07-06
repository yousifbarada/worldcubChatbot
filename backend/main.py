import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.minipultetext import process_all_text_files, split_text_into_chunks
from backend.embeddedmanger import EmbeddingManager
from backend.llm import GeminiLLM
from backend.vectorstore import VectorStore
from backend.RAG import RAGRetriever
from backend.worldcubchatbot import WorldCupChatbot

# --- Build the RAG pipeline ---
text_directory = "backend/text_files"
v_store = VectorStore(collection_name="pdf_documents", persist_directory="data/vector_store")
embedding_manager = EmbeddingManager()
rag_retriever = RAGRetriever(vector_store=v_store, embedding_manager=embedding_manager)

# Only ingest+embed documents the first time (collection empty), otherwise we'd
# duplicate every chunk on every server restart.
if v_store.count() == 0:
    text_documents = process_all_text_files(text_directory)
    chunks = split_text_into_chunks(text_documents, chunk_size=600, chunk_overlap=150)
    if chunks:
        chunk_texts = [c.page_content for c in chunks]
        chunk_embeddings = embedding_manager.generate_embeddings(chunk_texts)
        v_store.add_documents(chunks, chunk_embeddings)
    else:
        print(f"No chunks found in '{text_directory}'. The bot will have no context until documents are added.")
else:
    print(f"Vector store already has {v_store.count()} documents, skipping ingestion.")

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