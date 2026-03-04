import uuid
import re
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
    A smart chunker that splits text by paragraphs, detects sections dynamically,
    and enforces a character limit per chunk with sliding window overlap.
    """
    
    def __init__(self, chunk_size: int = 1500, overlap: int = 200):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum characters per chunk.
            overlap: Target number of characters to overlap between adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Regex to detect common paper headers
        self.header_pattern = re.compile(
            r'^(?:(?:\d+\.)*\d*\s*)?(Abstract|Introduction|Background|Related Work|Methodology|Methods|Approach|Experiments|Experimental Setup|Results|Discussion|Conclusion|References|Acknowledgements)\s*$',
            re.IGNORECASE
        )

    def _detect_section(self, para: str) -> Optional[str]:
        """Attempt to extract a section title from a given text block."""
        lines = para.strip().split('\n')
        if not lines:
            return None
        
        first_line = lines[0].strip()
        # A header shouldn't be too long
        if len(first_line) < 100:
            # Match common explicit academic headers
            match = self.header_pattern.match(first_line)
            if match:
                return match.group(1).title()
            
            # Match heuristics: e.g. "3. Proposed Method"
            if re.match(r'^\d+(\.\d+)*\s+[A-Z][a-zA-Z\s]+$', first_line):
                # Strip out the numbering
                title = re.sub(r'^\d+(\.\d+)*\s+', '', first_line).title()
                if len(title.split()) < 8:
                    return title
                    
            # Match ALL CAPS standalone short titles
            if first_line.isupper() and len(first_line.split()) < 6:
                return first_line.title()
                
        return None

    def chunk(self, text: str, metadata: PaperMetadata) -> List[PaperChunk]:
        """
        Splits text into chunks using paragraph boundaries with overlap and section detection.
        """
        # Split purely by double line breaks (paragraphs)
        raw_paragraphs = re.split(r'\n\s*\n', text)
        chunks: List[PaperChunk] = []
        
        current_chunk_text = ""
        current_section = "Unknown (Start)"
        current_chunk_section = current_section
        
        def commit_chunk(text_to_commit: str, section_to_commit: str):
            text_to_commit = text_to_commit.strip()
            if not text_to_commit:
                return
            chunks.append(PaperChunk(
                chunk_id=str(uuid.uuid4()),
                paper_id=metadata.paper_id,
                text=text_to_commit,
                section=section_to_commit,
                metadata=metadata
            ))

        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Update section context if header detected
            detected_sec = self._detect_section(para)
            if detected_sec:
                current_section = detected_sec
                # If chunk is currently empty, attach to new section right away
                if not current_chunk_text:
                    current_chunk_section = current_section
                
            # If adding this paragraph exceeds chunk size, commit and start new chunk with overlap
            if len(current_chunk_text) + len(para) > self.chunk_size and current_chunk_text:
                commit_chunk(current_chunk_text, current_chunk_section)
                
                # Produce sliding window overlap from the tail of the committed chunk
                overlap_text = ""
                if self.overlap > 0 and len(current_chunk_text) > self.overlap:
                    overlap_tail = current_chunk_text[-self.overlap:]
                    # Try to align overlap to the start of a sentence
                    last_period_idx = overlap_tail.find('. ')
                    if last_period_idx != -1 and last_period_idx < len(overlap_tail) - 2:
                        overlap_text = overlap_tail[last_period_idx + 2:]
                    else:
                        overlap_text = overlap_tail
                elif self.overlap > 0:
                    overlap_text = current_chunk_text
                
                current_chunk_text = f"{overlap_text}\n\n{para}" if overlap_text else para
                current_chunk_section = current_section
            else:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
                    current_chunk_section = current_section
        
        # Add any trailing text
        if current_chunk_text:
            commit_chunk(current_chunk_text, current_chunk_section)
            
        return chunks
