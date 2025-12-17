import json
from typing import List, Dict, Any
from kdra.core.schemas import PaperChunk, PaperExtraction
from kdra.core.reasoning.engine import BaseReasoningEngine, MockReasoningEngine
from kdra.core.reasoning.prompts import EXTRACTION_SYSTEM_PROMPT, DRAFT_PROMPT_TEMPLATE, VERIFICATION_PROMPT_TEMPLATE

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

    def extract(self, paper_id: str, chunks: List[PaperChunk]) -> PaperExtraction:
        """
        Extract structured information from a list of chunks belonging to a single paper.
        
        Args:
            paper_id: The ID of the paper.
            chunks: List of PaperChunk objects.
            
        Returns:
            PaperExtraction object.
        """
        # 1. Aggregation
        # TODO: Implement smarter context management (e.g., map-reduce for long papers)
        full_text = "\n\n".join([c.text for c in chunks])
        
        # 2. Draft Extraction
        draft_prompt = DRAFT_PROMPT_TEMPLATE.format(context=full_text[:10000]) # Truncate for safety in this demo
        draft_response = self.engine.generate(draft_prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT)
        
        # 3. Verification (Self-Correction)
        verify_prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            context=full_text[:10000],
            draft=draft_response
        )
        final_json_str = self.engine.generate(verify_prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT)
        
        # 4. Validation & Parsing
        try:
            # Clean up potential markdown code blocks from LLM
            cleaned_json = self._clean_json_string(final_json_str)
            data = json.loads(cleaned_json)
            
            # Ensure paper_id from arguments takes precedence and avoids conflict
            data["paper_id"] = paper_id
            
            # Enforce schema
            extraction = PaperExtraction(**data)
            return extraction
            
        except json.JSONDecodeError as e:
            # TODO: Implement retry logic or fallback parsing
            print(f"JSON Decode Error: {e}")
            print(f"Raw Output: {final_json_str}")
            raise e
        except Exception as e:
            print(f"Validation Error: {e}")
            raise e

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
