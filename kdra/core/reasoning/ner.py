import logging

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False

class GlinerNER:
    """
    Local Zero-shot NER using GLiNER to boost entity recall before LLM processing.
    """
    def __init__(self, model_version="urchade/gliner_medium-v2.1"):
        # We target specific academic labels
        self.labels = ["Method", "Dataset", "Metric", "Academic Concept", "Algorithm"]
        
        if GLINER_AVAILABLE:
            logging.info(f"Loading GLiNER model: {model_version} (This may take a moment on first run)...")
            # Load the model, device mapping usually handles itself
            self.model = GLiNER.from_pretrained(model_version)
        else:
            self.model = None
            logging.warning("gliner package not installed. High-recall local NER will be skipped. Run `pip install gliner` to enable.")

    def extract_candidates(self, text: str) -> dict:
        """
        Extract entities from text. 
        Returns a dictionary of found candidates categorized by our labels.
        """
        candidates = {
            "Method": [],
            "Dataset": [],
            "Metric": [],
            "Concept": []
        }
        
        if not self.model or not text.strip():
            return candidates

        # Limit total text to process to avoid extreme runtime
        text_to_process = text[:15000] 
        
        # GLiNER has a hard context window token limit of 384 tokens (Subword tokenizer).
        # We split the text into much smaller chunks (~500 chars) to prevent truncation loss and warnings.
        chunk_size = 500
        overlap = 50
        chunks = []
        for i in range(0, len(text_to_process), chunk_size - overlap):
            chunks.append(text_to_process[i:i + chunk_size])
        
        try:
            for chunk in chunks:
                # Predict entities per chunk
                entities = self.model.predict_entities(chunk, self.labels)
                
                for ent in entities:
                    label = ent["label"]
                    # Only keep entities with reasonable confidence
                    if ent.get("score", 1.0) > 0.4:
                        val = ent["text"].strip()
                        
                        # Map back to our simplified keys
                        target_key = None
                        if label in ["Method", "Algorithm"]:
                            target_key = "Method"
                        elif label == "Dataset":
                            target_key = "Dataset"
                        elif label == "Metric":
                            target_key = "Metric"
                        elif label == "Academic Concept":
                            target_key = "Concept"
                            
                        if target_key and val not in candidates[target_key]:
                            # Basic length filtering for noise reduction
                            if 2 < len(val) < 80:
                                candidates[target_key].append(val)
                            
        except Exception as e:
            logging.error(f"GLiNER prediction failed: {e}")
            
        return candidates