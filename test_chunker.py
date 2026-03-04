from kdra.core.ingestion.chunker import SimpleTextChunker
from kdra.core.schemas import PaperMetadata

text = """
1. Introduction
This is the introduction. Large language models (LLMs) are great.
They have transformed natural language processing. In this paper, we propose a new method.

2. Related Work
Many papers have studied this. 
We refer to Smith et al. and explicitly compare against their baseline.

Here is a big paragraph that does not have a header. """ + "Blah. " * 50 + """

3. Methodology
We use a transformer base.
"""

metadata = PaperMetadata(paper_id="test", title="Test Paper")
chunker = SimpleTextChunker(chunk_size=150, overlap=50) 
chunks = chunker.chunk(text, metadata)

for c in chunks:
    print("---")
    print(f"Section: {c.section}")
    print(f"Length: {len(c.text)}")
    print(f"Text: {c.text[:50]} ... {c.text[-50:]}")
