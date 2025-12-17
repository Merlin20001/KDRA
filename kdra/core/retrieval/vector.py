
import numpy as np
from typing import List, Tuple
from kdra.core.schemas import PaperChunk
from kdra.core.retrieval.base import BaseRetriever
from kdra.core.reasoning.engine import OpenAIEngine
import os

class VectorRetriever(BaseRetriever):
    """
    Retriever that uses OpenAI embeddings and cosine similarity.
    """
    
    def __init__(self, api_key: str = None):
        self.chunks: List[PaperChunk] = []
        self.embeddings: np.ndarray = None
        
        # Initialize OpenAI client for embeddings
        try:
            from openai import OpenAI
            import yaml
            
            # Load config similar to Engine
            config = {}
            config_path = os.path.abspath("llm_config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}

            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or config.get("api_key")
            self.base_url = os.getenv("OPENAI_BASE_URL") or config.get("base_url")
            
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.model = "text-embedding-3-small" # Cost effective model
            
        except ImportError:
            print("Warning: OpenAI or PyYAML not installed. VectorRetriever will fail.")
            self.client = None

    def _get_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding

    def index(self, chunks: List[PaperChunk]) -> None:
        """
        Index chunks by computing their embeddings.
        """
        if not chunks:
            return
            
        self.chunks = chunks
        print(f"Indexing {len(chunks)} chunks...")
        
        embeddings_list = []
        # Batching could be better, but doing one by one for simplicity/safety first
        # In production, use batching!
        for i, chunk in enumerate(chunks):
            try:
                emb = self._get_embedding(chunk.text)
                embeddings_list.append(emb)
            except Exception as e:
                print(f"Error embedding chunk {i}: {e}")
                # Use zero vector as fallback to keep alignment
                embeddings_list.append([0.0] * 1536) 

        self.embeddings = np.array(embeddings_list)
        print("Indexing complete.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[PaperChunk, float]]:
        """
        Retrieve chunks using cosine similarity.
        """
        if not self.chunks or self.embeddings is None:
            return []
            
        try:
            query_emb = np.array(self._get_embedding(query))
            
            # Cosine similarity: (A . B) / (||A|| * ||B||)
            # Assuming OpenAI embeddings are normalized? Usually yes, but let's be safe.
            # Actually text-embedding-3-small are normalized.
            
            # Compute dot products
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
