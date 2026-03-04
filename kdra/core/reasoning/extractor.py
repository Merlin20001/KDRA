import json
from typing import List, Dict, Any
from kdra.core.schemas import PaperChunk, PaperExtraction
from kdra.core.reasoning.engine import BaseReasoningEngine, MockReasoningEngine
from kdra.core.reasoning.prompts import (
    EXTRACTION_SYSTEM_PROMPT, 
    DRAFT_PROMPT_TEMPLATE, 
    VERIFICATION_PROMPT_TEMPLATE,
    REDUCE_PROMPT_TEMPLATE
)
from kdra.core.reasoning.ner import GlinerNER

class PaperExtractor:
    """
    Orchestrates the extraction of structured data from paper chunks using a multi-step reasoning process.
    
    Process:
    1. Aggregation: Combines chunks into a context window.
    2. Draft Extraction: Initial pass to identify key entities.
    3. Verification: Self-correction step to ensure evidence exists in text.
    4. Validation: Pydantic schema enforcement.
    """
    
    def __init__(self, engine: BaseReasoningEngine = None):
        """
        Initialize the extractor.
        
        Args:
            engine: The LLM engine to use. Defaults to MockReasoningEngine.
        """
        self.engine = engine or MockReasoningEngine()
        self.ner = GlinerNER()

    def extract(self, paper_id: str, chunks: List[PaperChunk], max_retries: int = 3) -> PaperExtraction:
        """
        Extract structured information from a list of chunks belonging to a single paper
        using a Map-Reduce strategy to handle long context.
        
        Args:
            paper_id: The ID of the paper.
            chunks: List of PaperChunk objects.
            max_retries: Maximum number of times to retry schema parsing and validation on failure.
            
        Returns:
            PaperExtraction object.
        """
        # Batch chunks to fit within context limits safely
        batches = self._batch_chunks(chunks, max_chars=10000)
        
        # Map Phase: Extract from each batch
        partial_extractions = []
        for i, batch_text in enumerate(batches):
            print(f"[{paper_id}] Extracting from chunk batch {i+1}/{len(batches)}...")
            partial_ext = self._extract_from_text(batch_text, max_retries)
            if partial_ext:
                partial_extractions.append(partial_ext)
                
        if not partial_extractions:
            print(f"Warning: No extractions produced for {paper_id}")
            # Return empty skeleton
            return PaperExtraction(paper_id=paper_id, methods=[], datasets=[], metrics=[], claims=[], limitations=[], concepts=[])
            
        if len(partial_extractions) == 1:
            # No need to reduce
            final_extraction = partial_extractions[0]
            final_extraction.paper_id = paper_id
            return final_extraction
            
        # Reduce Phase: Merge partial extractions
        print(f"[{paper_id}] Merging {len(partial_extractions)} partial extractions...")
        final_extraction = self._merge_extractions(partial_extractions, max_retries)
        final_extraction.paper_id = paper_id
        
        return final_extraction

    def _batch_chunks(self, chunks: List[PaperChunk], max_chars: int = 10000) -> List[str]:
        """Groups chunks into lists of strings roughly matching max_chars."""
        batches = []
        current_batch = []
        current_length = 0
        
        for chunk in chunks:
            chunk_len = len(chunk.text)
            if current_length + chunk_len > max_chars and current_batch:
                batches.append("\n\n".join(current_batch))
                current_batch = [chunk.text]
                current_length = chunk_len
            else:
                current_batch.append(chunk.text)
                current_length += chunk_len
                
        if current_batch:
            batches.append("\n\n".join(current_batch))
            
        return batches

    def _extract_from_text(self, text: str, max_retries: int) -> PaperExtraction:
        """Single extraction task for a specific text batch."""
        # 1. High-Recall GLiNER Extraction
        candidates = self.ner.extract_candidates(text)
        
        # 2. Draft Extraction
        draft_prompt = DRAFT_PROMPT_TEMPLATE.format(
            context=text,
            candidates=json.dumps(candidates, indent=2)
        )
        draft_response = self.engine.generate(draft_prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT)
        
        # 3. Verification & Validation (Self-Correction & Schema Guardrails)
        verify_prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            context=text,
            draft=draft_response
        )
        
        try:
            extraction = self.engine.generate_structured(
                prompt=verify_prompt,
                schema_class=PaperExtraction,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                max_retries=max_retries
            )
            return extraction
        except Exception as e:
            print(f"Failed to extract text segment after {max_retries} retries due to {e}")
            return None

    def _merge_extractions(self, extractions: List[PaperExtraction], max_retries: int) -> PaperExtraction:
        """Merge multiple partial extractions into a single comprehensive extraction using LLM."""
        # Serialize partial extractions for prompt
        partial_json_list = [ext.model_dump_json(indent=2) for ext in extractions]
        partial_extractions_str = "\n---\n".join(partial_json_list)
        
        reduce_prompt = REDUCE_PROMPT_TEMPLATE.format(
            partial_extractions=partial_extractions_str
        )
        
        try:
            final_extraction = self.engine.generate_structured(
                prompt=reduce_prompt,
                schema_class=PaperExtraction,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                max_retries=max_retries
            )
            return final_extraction
        except Exception as e:
            print(f"Failed to merge extractions after {max_retries} retries due to {e}")
            # Fallback: Just return the first one (naive failure handling)
            return extractions[0]

    def _clean_json_string(self, json_str: str) -> str:
        """Removes markdown formatting and extracts JSON object."""
        import re
        # Try to find JSON block within markdown code fences
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_str, re.DOTALL)
        if match:
            return match.group(1)
        
        # If no code fences, try to find the first '{' and last '}'
        start = json_str.find('{')
        end = json_str.rfind('}')
        if start != -1 and end != -1:
            return json_str[start:end+1]
            
        return json_str.strip()
