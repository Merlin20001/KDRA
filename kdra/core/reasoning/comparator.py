import json
from typing import List
from kdra.core.schemas import PaperExtraction, ComparativeInsight
from kdra.core.reasoning.engine import BaseReasoningEngine, MockReasoningEngine
from kdra.core.reasoning.prompts import COMPARISON_SYSTEM_PROMPT, COMPARISON_PROMPT_TEMPLATE

class ComparativeAnalyst:
    """
    Performs comparative reasoning over multiple structured paper extractions.
    """
    
    def __init__(self, engine: BaseReasoningEngine = None):
        """
        Initialize the analyst.
        
        Args:
            engine: The LLM engine to use. Defaults to MockReasoningEngine.
        """
        self.engine = engine or MockReasoningEngine()

    def compare(self, extractions: List[PaperExtraction], topic: str, max_retries: int = 3) -> List[ComparativeInsight]:
        """
        Generate comparative insights from a list of paper extractions.
        
        Args:
            extractions: List of PaperExtraction objects.
            topic: The research topic or question guiding the comparison.
            max_retries: Number of retries on JSON validation failure.
            
        Returns:
            List of ComparativeInsight objects.
        """
        if not extractions:
            return []

        # Serialize extractions to JSON for the prompt
        # We exclude evidence_spans to save context window, as we are reasoning over structured data
        data_for_prompt = [
            ext.model_dump(exclude={"evidence_spans"}) 
            for ext in extractions
        ]
        data_str = json.dumps(data_for_prompt, indent=2)
        
        prompt = COMPARISON_PROMPT_TEMPLATE.format(
            count=len(extractions),
            topic=topic,
            data=data_str
        )
        
        response_str = self.engine.generate(prompt, system_prompt=COMPARISON_SYSTEM_PROMPT)
        
        import pydantic
        
        for attempt in range(max_retries):
            try:
                cleaned_json = self._clean_json_string(response_str)
                data = json.loads(cleaned_json)
                
                # Handle wrapped response (e.g. {"insights": [...]})
                if isinstance(data, dict) and "insights" in data:
                    data = data["insights"]
                
                insights = [ComparativeInsight(**item) for item in data]
                return insights
                
            except (json.JSONDecodeError, pydantic.ValidationError) as e:
                print(f"Validation/Parse Error in ComparativeAnalyst on attempt {attempt + 1}: {e}")
                
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts.")
                    return []
                    
                print(f"Retrying LLM generation. Attempt {attempt + 2}/{max_retries}...")
                
                schema_json = json.dumps(ComparativeInsight.model_json_schema(), indent=2)
                retry_prompt = (
                    "The previously generated JSON failed parsing or validation. "
                    "Please carefully review the error and output strictly valid JSON matching the schema.\n\n"
                    f"Error Context:\n{e}\n\n"
                    f"Required JSON Schema for items:\n{schema_json}\n\n"
                    f"Original Faulty JSON:\n{response_str}"
                )
                
                response_str = self.engine.generate(retry_prompt, system_prompt=COMPARISON_SYSTEM_PROMPT)
            except Exception as e:
                print(f"Unexpected Error in ComparativeAnalyst: {e}")
                return []
                
        return []

    def _clean_json_string(self, json_str: str) -> str:
        """Removes markdown formatting like ```json ... ```"""
        json_str = json_str.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        return json_str.strip()
