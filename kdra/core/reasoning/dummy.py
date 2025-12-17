import re
from typing import List
from kdra.core.schemas import PaperExtraction, PaperChunk, MetricValue, ComparativeInsight
from kdra.core.reasoning.extractor import PaperExtractor
from kdra.core.reasoning.comparator import ComparativeAnalyst

class DummyExtractor(PaperExtractor):
    """
    Heuristic-based extractor that does not require an LLM.
    """
    
    def __init__(self):
        # No engine needed
        pass

    def extract(self, paper_id: str, chunks: List[PaperChunk]) -> PaperExtraction:
        """
        Extract information using regex and simple heuristics.
        """
        full_text = "\n".join([c.text for c in chunks])
        
        # Heuristic: Find capitalized words ending in 'Net', 'Former', 'GPT' as methods
        # This is very naive but deterministic.
        method_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:Net|Former|GPT|BERT|Model)\b'
        methods = list(set(re.findall(method_pattern, full_text)))
        if not methods:
            methods = ["ProposedMethod"]
            
        # Heuristic: Find datasets
        dataset_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:Set|Data|Corpus|Bank)\b'
        datasets = list(set(re.findall(dataset_pattern, full_text)))
        if not datasets:
            datasets = ["StandardDataset"]

        # Heuristic: Find metrics (e.g., "95.4% accuracy", "accuracy of 0.95")
        metrics = []
        # Pattern: number followed by % or word 'accuracy' nearby
        # Look for "accuracy: 0.95" or "95% accuracy"
        
        # 1. "95.5% accuracy"
        p1 = re.findall(r'(\d+(?:\.\d+)?)%\s+([a-zA-Z]+)', full_text)
        for val, name in p1:
            metrics.append(MetricValue(
                name=name, 
                value=float(val), 
                unit="%", 
                dataset=datasets[0] if datasets else "Unknown"
            ))
            
        # 2. "accuracy of 0.95"
        p2 = re.findall(r'([a-zA-Z]+)\s+of\s+(\d+(?:\.\d+)?)', full_text)
        for name, val in p2:
            metrics.append(MetricValue(
                name=name, 
                value=float(val), 
                unit="float", 
                dataset=datasets[0] if datasets else "Unknown"
            ))
            
        if not metrics:
            metrics.append(MetricValue(name="Accuracy", value=0.0, unit="float", dataset="Unknown"))

        return PaperExtraction(
            paper_id=paper_id,
            methods=methods,
            datasets=datasets,
            metrics=metrics,
            claims=["Extracted via dummy heuristics"],
            limitations=["Heuristic limitation: No semantic understanding"],
            evidence_spans={}
        )

class DummyComparator(ComparativeAnalyst):
    """
    Heuristic-based comparator that does not require an LLM.
    """
    
    def __init__(self):
        # No engine needed
        pass

    def compare(self, extractions: List[PaperExtraction], topic: str) -> List[ComparativeInsight]:
        """
        Generate insights by comparing extracted metrics.
        """
        insights = []
        
        if not extractions:
            return []
            
        # Insight 1: Performance Comparison
        # Find the paper with the highest metric value (assuming higher is better for now)
        best_paper = None
        max_val = -1.0
        
        for ext in extractions:
            for m in ext.metrics:
                # Normalize value if %
                val = m.value
                if m.unit == "%":
                    val = val / 100.0
                
                if isinstance(val, (int, float)) and val > max_val:
                    max_val = val
                    best_paper = ext.paper_id
        
        if best_paper:
            insights.append(ComparativeInsight(
                topic=topic,
                papers_involved=[e.paper_id for e in extractions],
                insight_text=f"Paper {best_paper} reports the highest performance metric ({max_val}).",
                category="Performance"
            ))
            
        # Insight 2: Method Popularity
        all_methods = []
        for ext in extractions:
            all_methods.extend(ext.methods)
            
        if all_methods:
            from collections import Counter
            counts = Counter(all_methods)
            most_common = counts.most_common(1)[0]
            insights.append(ComparativeInsight(
                topic=topic,
                papers_involved=[e.paper_id for e in extractions],
                insight_text=f"The most frequently mentioned method is {most_common[0]} (count: {most_common[1]}).",
                category="Trend"
            ))
            
        return insights
