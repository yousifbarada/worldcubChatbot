import uuid

from fastapi.responses import FileResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import settings
from ultralytics import settings
from sentence_transformers.base import BaseModel

from minipultetext import *
from embeddedmanger import EmbeddingManager
from llm import GeminiLLM
from vectorstore import VectorStore
from RAG import RAGRetriever
from worldcubchatbot import WorldCupChatbot
from fastapi import FastAPI, HTTPException, HTTPException

text_directory = "text_files"
text_documents = process_all_text_files(text_directory)
chunks = split_text_into_chunks(text_documents, chunk_size=1200, chunk_overlap=120)
v_store = VectorStore(collection_name="pdf_documents", persist_directory="data/vector_store")
embedding_manager = EmbeddingManager()
rag_retriever = RAGRetriever(vector_store=v_store, embedding_manager=embedding_manager)
# Compute embeddings for chunks and add to vector store if collection is empty
from llm import GeminiLLM
llm = GeminiLLM()
chatbot = WorldCupChatbot(retriever=rag_retriever, llm=llm)


app = FastAPI(title="SBS Chatbot API")

app.add_middleware(
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


@app.on_event("startup")
def startup():
    settings.validate()


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
app.mount("/static",    StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    
@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
