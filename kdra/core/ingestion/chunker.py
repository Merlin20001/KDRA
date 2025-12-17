import uuid
from typing import List, Optional
from kdra.core.schemas import PaperChunk, PaperMetadata

class BaseChunker:
    """
    Abstract base class for text chunking strategies.
    """
    def chunk(self, text: str, metadata: PaperMetadata) -> List[PaperChunk]:
        """
        Split text into semantic chunks.
        
        Args:
            text: The full text content of the paper.
            metadata: Metadata associated with the paper.
            
        Returns:
            List of PaperChunk objects.
        """
        raise NotImplementedError

class SimpleTextChunker(BaseChunker):
    """
    A simple chunker that splits text by paragraphs (double newlines)
    and enforces a maximum character limit per chunk.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum characters per chunk.
            overlap: Number of characters to overlap between chunks (not fully implemented in simple split).
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: PaperMetadata) -> List[PaperChunk]:
        """
        Splits text into chunks based on paragraphs.
        
        TODO: Implement more sophisticated sliding window with overlap.
        TODO: Implement section detection based on common headers (Introduction, Methods, etc.).
        """
        raw_paragraphs = text.split('\n\n')
        chunks: List[PaperChunk] = []
        
        current_chunk_text = ""
        
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph exceeds chunk size, save current chunk and start new
            if len(current_chunk_text) + len(para) > self.chunk_size:
                if current_chunk_text:
                    self._create_and_add_chunk(chunks, current_chunk_text, metadata)
                current_chunk_text = para
            else:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
        
        # Add the last chunk
        if current_chunk_text:
            self._create_and_add_chunk(chunks, current_chunk_text, metadata)
            
        return chunks

    def _create_and_add_chunk(self, chunks: List[PaperChunk], text: str, metadata: PaperMetadata):
        """Helper to create a PaperChunk and add it to the list."""
        chunk_id = str(uuid.uuid4())
        
        # TODO: Implement heuristic to detect section from text content
        section = None 
        
        chunk = PaperChunk(
            chunk_id=chunk_id,
            paper_id=metadata.paper_id,
            text=text,
            section=section,
            metadata=metadata
        )
        chunks.append(chunk)
