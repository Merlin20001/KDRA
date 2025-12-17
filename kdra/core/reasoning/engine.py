from abc import ABC, abstractmethod
from typing import Optional
import os

class BaseReasoningEngine(ABC):
    """
    Abstract base class for LLM interactions.
    """
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, json_mode: bool = True) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system instruction.
            json_mode: Whether to enforce JSON output format.
            
        Returns:
            The generated text response.
        """
        pass

class MockReasoningEngine(BaseReasoningEngine):
    """
    A mock engine that returns dummy JSON for testing.
    """
    def generate(self, prompt: str, system_prompt: Optional[str] = None, json_mode: bool = True) -> str:
        if not json_mode:
            return "This is a mock answer to your question based on the provided context."

        # Return a valid JSON string matching PaperExtraction schema
        return """
        {
            "paper_id": "mock_id",
            "methods": ["MockMethod"],
            "datasets": ["MockDataset"],
            "metrics": [
                {"name": "Accuracy", "value": 0.95, "unit": "float", "dataset": "MockDataset"}
            ],
            "claims": ["This is a mock claim."],
            "limitations": ["Mock limitation."],
            "concepts": ["MockConcept1", "MockConcept2"],
            "evidence_spans": {}
        }
        """

class OpenAIEngine(BaseReasoningEngine):
    """
    Reasoning engine using OpenAI-compatible API.
    """
    def __init__(self, model: str = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
            import yaml
        except ImportError:
            raise ImportError("openai and pyyaml packages are required. Install with `pip install openai pyyaml`.")
            
        # Load config from file if exists
        config = {}
        config_path = os.path.abspath("llm_config.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}")

        # Priority: Constructor Args > Env Vars > Config File
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or config.get("api_key")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or config.get("base_url")
        self.model = model or os.getenv("OPENAI_MODEL_NAME") or config.get("model_name") or "gpt-4o"

        if not self.api_key or self.api_key == "sk-...":
             # Don't raise error yet, let the caller handle it if they try to use it without key
             pass

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None, json_mode: bool = True) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0, # Deterministic for extraction
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")
