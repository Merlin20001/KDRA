from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from kdra.core.schemas import PaperChunk

class BaseRetriever(ABC):
    """
    Abstract base class for retrieval systems.
    """
    
    @abstractmethod
    def index(self, chunks: List[PaperChunk]) -> None:
        """
        Index a list of paper chunks for retrieval.
        
        Args:
            chunks: List of PaperChunk objects to index.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[PaperChunk, float]]:
        """
        Retrieve relevant chunks for a given query.
        
        Args:
            query: The search query string.
            top_k: Number of top results to return.
            
        Returns:
            List of tuples containing (PaperChunk, similarity_score).
        """
        pass
