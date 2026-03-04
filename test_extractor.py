import sys, os
from kdra.core.schemas import PaperChunk, PaperMetadata
from kdra.core.reasoning.extractor import PaperExtractor
from kdra.core.reasoning.engine import MockReasoningEngine

print("Starting test...")
meta = PaperMetadata(paper_id="test1", title="Test Paper")
chunks = [
    PaperChunk(paper_id="test1", chunk_id="c1", text="Chunk A " * 500, metadata=meta), 
    PaperChunk(paper_id="test1", chunk_id="c2", text="Chunk B " * 500, metadata=meta), 
    PaperChunk(paper_id="test1", chunk_id="c3", text="Chunk C " * 500, metadata=meta), 
    PaperChunk(paper_id="test1", chunk_id="c4", text="Chunk D " * 500, metadata=meta), 
    PaperChunk(paper_id="test1", chunk_id="c5", text="Chunk E " * 500, metadata=meta), 
]

extractor = PaperExtractor(engine=MockReasoningEngine())
print("Calling extract...")
ext = extractor.extract(paper_id="test1", chunks=chunks)
print("Finished!")
print(ext.model_dump_json(indent=2))
