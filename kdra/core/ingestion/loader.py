from typing import List, Optional
import os
from kdra.core.schemas import PaperChunk, PaperMetadata
from kdra.core.ingestion.chunker import BaseChunker, SimpleTextChunker

class PaperIngestor:
    """
    Orchestrates the loading and chunking of papers.
    """
    
    def __init__(self, chunker: Optional[BaseChunker] = None):
        """
        Initialize the ingestor.
        
        Args:
            chunker: The chunking strategy to use. Defaults to SimpleTextChunker.
        """
        self.chunker = chunker or SimpleTextChunker()

    def ingest_text(self, text: str, metadata: PaperMetadata) -> List[PaperChunk]:
        """
        Ingest raw text directly.
        
        Args:
            text: The full text of the paper.
            metadata: Metadata for the paper.
            
        Returns:
            List of PaperChunk objects.
        """
        return self.chunker.chunk(text, metadata)

    def ingest_file(self, file_path: str, metadata: Optional[PaperMetadata] = None) -> List[PaperChunk]:
        """
        Ingest a paper from a file path.
        
        Args:
            file_path: Path to the file (PDF, TXT, etc.).
            metadata: Optional metadata. If not provided, minimal metadata is inferred.
            
        Returns:
            List of PaperChunk objects.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # TODO: Implement DOCX parsing logic
        
        text = ""
        if file_path.endswith('.txt') or file_path.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_path.endswith('.pdf'):
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except ImportError:
                raise ImportError("pypdf is required for PDF support. Install it with `pip install pypdf`.")
            except Exception as e:
                raise RuntimeError(f"Failed to read PDF file {file_path}: {e}")
        else:
            # Placeholder for binary formats
            raise NotImplementedError(f"File format not supported yet: {file_path}")
            
        if metadata is None:
            # Create minimal metadata if not provided
            # TODO: Extract metadata from file content or filename
            metadata = PaperMetadata(
                paper_id=os.path.basename(file_path),
                title=os.path.basename(file_path),
                year=None,
                venue=None,
                authors=[],
                url=f"file://{os.path.abspath(file_path)}"
            )
            
        return self.ingest_text(text, metadata)
