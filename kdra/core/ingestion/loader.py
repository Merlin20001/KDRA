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

    def _parse_pdf(self, file_path: str) -> str:
        """
        Parses PDF files using a tiered fallback strategy for formatting and modality retention:
        Tier 1: Deep Learning Vision Models (MinerU / Marker)
        Tier 2: Advanced text+layout extractors (pymupdf4llm)
        Tier 3: Naive extraction (pypdf)
        """
        import logging
        text = ""

        # Tier 1 Attempt: MinerU (magic-pdf)
        try:
            logging.info(f"Attempting MinerU (magic-pdf) for Vision-based deep extraction: {file_path}")
            from magic_pdf.pipe.UNIPipe import UNIPipe
            from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
            import magic_pdf.model as model_config

            # Initialize model configuration for MinerU
            model_config.__use_inside_model__ = True
            pipe = UNIPipe(file_path, {"_ddXz": "ddXz"}, DiskReaderWriter(os.path.dirname(file_path)))
            pipe.pipe_classify()
            pipe.pipe_parse()
            md_content = pipe.pipe_mk_markdown()
            if md_content:
                logging.info("MinerU extraction successful.")
                return md_content
        except ImportError:
            pass
        except Exception as e:
            logging.warning(f"MinerU extraction failed: {e}")

        # Alternate Tier 1 Attempt: Marker
        try:
            logging.info(f"Attempting Marker for local Deep Learning extraction: {file_path}")
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
            
            converter = PdfConverter(artifact_dict=create_model_dict())
            rendered = converter(file_path)
            md_text, _, _ = text_from_rendered(rendered)
            
            if md_text:
                logging.info("Marker extraction successful.")
                return md_text
        except ImportError:
            pass
        except Exception as e:
            logging.warning(f"Marker extraction failed: {e}")

        # Tier 2 Attempt: pymupdf4llm (Lighter Markdown extractor)
        try:
            logging.info("Advanced vision extractors not available. Falling back to pymupdf4llm (Markdown preserving)...")
            import pymupdf4llm
            text = pymupdf4llm.to_markdown(file_path)
            logging.info("pymupdf4llm extraction successful.")
            return text
        except ImportError:
            logging.warning("pymupdf4llm not installed (`pip install pymupdf4llm`).")
        except Exception as e:
            logging.warning(f"pymupdf4llm extraction failed: {e}")

        # Tier 3 Fatal Fallback: Naive pypdf (Destroys tables/equations)
        try:
            logging.warning("Falling back to naive pypdf. Tables and formulas will likely be mangled.")
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except ImportError:
            raise ImportError("pypdf is required for PDF support. Install it with `pip install pypdf`.")
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF file {file_path}: {e}")

        return text

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
            text = self._parse_pdf(file_path)
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
