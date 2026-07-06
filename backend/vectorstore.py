import os
from langchain_core.documents import Document
import chromadb
import uuid
from typing import List, Dict, Any, Tuple
import numpy as np


class VectorStore:
    def __init__(self, collection_name="pdf_documents", persist_directory="../data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embeddings for RAG",
                    "hnsw:space": "cosine"  # explicit cosine similarity
                }
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def count(self) -> int:
        """Return number of documents currently stored. main.py's /api/health calls this."""
        if not self.collection:
            return 0
        return self.collection.count()

    def add_documents(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        print(f"Adding {len(documents)} documents to vector store...")
        ids, metadatas, documents_text, embeddings_list = [], [], [], []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            # Ensure embeddings are 1D lists of floats
            arr = np.asarray(embedding, dtype=float)
            # Normalize embeddings to unit length to match cosine space
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            embeddings_list.append(arr.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents")
            print(f"Total in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise