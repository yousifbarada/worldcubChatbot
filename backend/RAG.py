import numpy as np


class RAGRetriever:
    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 10, score_threshold: float = 0.0):
        print(f"Retrieving for: '{query}' | top_k={top_k} | threshold={score_threshold}")
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Normalize for cosine
        norm = np.linalg.norm(query_embedding)
        query_norm = query_embedding / norm if norm > 0 else query_embedding

        results = self.vector_store.collection.query(
            query_embeddings=[query_norm.tolist()],
            n_results=top_k  # ← fixed: was hardcoded 10
        )

        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # hnsw:space=cosine → distance = 1 - cosine_sim
                cosine_sim = 1.0 - distance
                if cosine_sim >= score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'similarity_score': cosine_sim,
                        'rank': i + 1
                    })

        print(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs

