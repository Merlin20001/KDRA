import json
from typing import List, Dict, Any
from kdra.core.schemas import KnowledgeGraph, PaperExtraction
from kdra.core.reasoning.engine import BaseReasoningEngine, MockReasoningEngine
from kdra.core.reasoning.prompts import QA_SYSTEM_PROMPT, QA_PROMPT_TEMPLATE

class ResearchAssistant:
    """
    Provides Q&A capabilities over the processed papers.
    """
    
    def __init__(self, engine: BaseReasoningEngine = None):
        self.engine = engine or MockReasoningEngine()

    def answer(self, question: str, kg: KnowledgeGraph, extractions: List[PaperExtraction], retrieved_chunks: List[str] = None) -> str:
        """
        Answer a user question based on the KG, extractions, and optionally retrieved text chunks.
        """
        context_str = self._build_context(kg, extractions, retrieved_chunks)
        
        # Truncate context if too long (naive approach)
        # In a real system, we would use RAG to select relevant parts
        if len(context_str) > 30000: # Increased limit slightly
            context_str = context_str[:30000] + "...(truncated)"
            
        prompt = QA_PROMPT_TEMPLATE.format(context=context_str, question=question)
        
        # Note: We don't enforce JSON here, we want natural language
        return self.engine.generate(prompt, system_prompt=QA_SYSTEM_PROMPT, json_mode=False)

    def _build_context(self, kg: KnowledgeGraph, extractions: List[PaperExtraction], retrieved_chunks: List[str] = None) -> str:
        """
        Convert structured data into a text representation for the LLM.
        """
        context = []
        
        if retrieved_chunks:
            context.append("=== Relevant Text Segments (RAG) ===")
            for i, chunk in enumerate(retrieved_chunks):
                context.append(f"Segment {i+1}: {chunk}")
            context.append("\n")
        
        context.append("=== Knowledge Graph Nodes ===")
        for node in kg.nodes:
            props = json.dumps(node.properties)
            context.append(f"Node ({node.type}): {node.id} | Props: {props}")
            
        context.append("\n=== Knowledge Graph Edges ===")
        for edge in kg.edges:
            context.append(f"{edge.source} --[{edge.relation}]--> {edge.target}")
            
        context.append("\n=== Detailed Extractions ===")
        for ext in extractions:
            context.append(f"Paper: {ext.paper_id}")
            context.append(f"  Methods: {', '.join(ext.methods)}")
            context.append(f"  Datasets: {', '.join(ext.datasets)}")
            context.append(f"  Claims: {json.dumps(ext.claims)}")
            context.append(f"  Limitations: {json.dumps(ext.limitations)}")
            
        return "\n".join(context)
