
import numpy as np
from typing import List, Tuple
from kdra.core.schemas import PaperChunk
from kdra.core.retrieval.base import BaseRetriever
from kdra.core.reasoning.engine import OpenAIEngine
import os

class VectorRetriever(BaseRetriever):
    """
    Retriever that uses local Sentence-Transformers embeddings and cosine similarity.
    """
    
    def __init__(self, api_key: str = None):
        self.chunks: List[PaperChunk] = []
        self.embeddings: np.ndarray = None
        
        # Initialize Local Embedding Model
        try:
            from sentence_transformers import SentenceTransformer
            # Using BGE-small, a very fast and high-quality embedding model suitable for academic text
            self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            print("Loaded local embedding model: BAAI/bge-small-en-v1.5")
        except ImportError:
            print("Warning: sentence_transformers not installed. VectorRetriever will fail. Please run `pip install sentence-transformers`.")
            self.model = None

    def _get_embedding(self, text: str) -> List[float]:
        if not self.model:
            raise RuntimeError("Embedding model not loaded.")
        text = text.replace("\n", " ")
        # encode returns numpy array by default
        return self.model.encode(text, normalize_embeddings=True)

    def index(self, chunks: List[PaperChunk]) -> None:
        """
        Index chunks by computing their embeddings.
        """
        if not chunks:
            return
            
        self.chunks = chunks
        print(f"Indexing {len(chunks)} chunks with local model...")
        
        if not self.model:
            print("Cannot index: model not loaded.")
            return

        # Batch encode is much faster with sentence-transformers
        texts = [chunk.text.replace("\n", " ") for chunk in chunks]
        try:
            # Generate embedded vectors for all chunks at once
            self.embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        except Exception as e:
            print(f"Error embedding chunks: {e}")
            self.embeddings = np.zeros((len(chunks), 384)) # fallback to bge-small dim

        print("Indexing complete.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[PaperChunk, float]]:
        """
        Retrieve chunks using cosine similarity.
        """
        if not self.chunks or self.embeddings is None:
            return []
            
        try:
            query_emb = self._get_embedding(query)
            
            # Vectors are already normalized, so dot product == cosine similarity
            scores = np.dot(self.embeddings, query_emb)
            
            # Get top K indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append((self.chunks[idx], float(scores[idx])))
                
            return results
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
