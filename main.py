from minipultetext import *
from embeddedmanger import EmbeddingManager
from llm import GeminiLLM
from vectorstore import VectorStore
from RAG import RAGRetriever
from worldcubchatbot import WorldCupChatbot

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


