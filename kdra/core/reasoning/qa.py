import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from kdra.core.schemas import KnowledgeGraph, PaperExtraction
from kdra.core.reasoning.engine import BaseReasoningEngine, MockReasoningEngine, OpenAIEngine

# LangChain Imports for Agentic RAG
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.tools import StructuredTool

# Define Tool Input Schemas
class VectorSearchInput(BaseModel):
    query: str = Field(description="The natural language query to search for specific text chunks or details within the papers.")

class KGSearchInput(BaseModel):
    entity: str = Field(description="The core entity name (e.g. 'lora', 'glue dataset') to search for in the Knowledge Graph.")

class CompareNodesInput(BaseModel):
    node_a: str = Field(description="First entity node ID")
    node_b: str = Field(description="Second entity node ID")

class ResearchAssistant:
    """
    Agentic GraphRAG Research Assistant. 
    Uses Multi-hop Reasoning with Vector + Graph tools via LangChain.
    """
    
    def __init__(self, engine: BaseReasoningEngine = None, retriever: Any = None):
        self.engine = engine or MockReasoningEngine()
        self.retriever = retriever
        self.kg: Optional[KnowledgeGraph] = None
        self.extractions: List[PaperExtraction] = []
        
        # Initialize LangChain LLM if real engine is configured
        if isinstance(self.engine, OpenAIEngine):
            import os
            self.llm = ChatOpenAI(
                api_key=self.engine.api_key or "sk-mock",
                base_url=self.engine.base_url,
                model=self.engine.model or "gpt-4o",
                temperature=0.0
            )
        else:
            self.llm = None

    def answer(self, question: str, kg: KnowledgeGraph, extractions: List[PaperExtraction]) -> str:
        """
        Answers a user question iteratively using graph tools and vector tools.
        """
        if not self.llm:
            return self.engine.generate(f"Mock Agentic QA for: {question}", system_prompt="", json_mode=False)
            
        self.kg = kg
        self.extractions = extractions
        
        # 1. Prepare Tools
        tools = [
            StructuredTool.from_function(
                func=self._search_vector,
                name="search_vector",
                description="Search the raw text chunks of papers for specific details or mentions. Use this to find exact text, metrics, or detailed paragraphs.",
                args_schema=VectorSearchInput
            ),
            StructuredTool.from_function(
                func=self._search_kg,
                name="search_kg",
                description="Search the knowledge graph for an entity ID to find all its 1-hop connected neighbors (papers, datasets, methods, concepts).",
                args_schema=KGSearchInput
            ),
            StructuredTool.from_function(
                func=self._compare_nodes,
                name="compare_nodes",
                description="Compare two knowledge graph nodes to find their common connections (e.g., to see if two methods are used in the same paper or evaluated on the same dataset).",
                args_schema=CompareNodesInput
            )
        ]
        
        # 2. Prepare Agent System Prompt
        active_papers = ", ".join([f'"{ext.paper_id}"' for ext in extractions]) if extractions else "None"
        
        system_prompt = (
             "You are KDRA, an expert Agentic GraphRAG research assistant. "
             "You have access to a semantic Vector Database and a structured Knowledge Graph of academic papers.\n"
             f"IMPORTANT - THE USER IS CURRENTLY VIEWING THESE PAPERS: [{active_papers}].\n"
             "CRITICAL RULE: NEVER ask the user which papers they mean. If the user refers to 'these papers' or asks for commonalities, "
             "IMMEDIATELY use the tool `compare_nodes` or `search_kg` or `search_vector` ON THE EXACT PAPER IDs LISTED ABOVE. "
             "Do not wait for the user to provide titles or authors. Just use the IDs directly.\n\n"
             "Guidelines:\n"
             "1. MULTI-HOP: If asked about relationships, trends, or comparisons between papers, start with `search_kg` or `compare_nodes` to identify structural connections.\n"
             "2. DETAIL: Once you find relevant papers/entities from the graph, use `search_vector` to fetch actual evidence snippets.\n"
             "3. REASONING: Explain your logical chain clearly based on the tool outputs.\n"
             "4. If standard `search_kg` yields no results, try simpler terms or rely on `search_vector`.\n"
             "Format your final output beautifully using Markdown."
        )
        
        # 3. Create Agent Execution Loop using StateGraph (LangGraph style)
        agent = create_react_agent(self.llm, tools, prompt=system_prompt)
        
        # Force inject context directly into user's prompt so the LLM cannot ignore it
        enhanced_question = f"Question: {question}\n\n[SYSTEM DIRECTIVE: The exact papers to analyze are: {active_papers}. You MUST use your tools (compare_nodes/search_kg/search_vector) immediately on these IDs. DO NOT ASK the user to specify the papers.]"
        
        try:
            # invoke returns a dict with 'messages' list; the last AI message is the final output
            response = agent.invoke({"messages": [("user", enhanced_question)]})
            final_message = response["messages"][-1]
            return final_message.content
        except Exception as e:
            return f"Agent reasoning pipeline failed: {str(e)}"

    # --- Tool Implementations ---

    def _search_vector(self, query: str) -> str:
        """Vector Text Search Tool Block"""
        if not self.retriever:
            return "Vector retriever is not available. You rely strictly on KG structure."
            
        results = self.retriever.retrieve(query, top_k=4)
        if not results:
            return f"No relevant semantic text chunks found for '{query}'."
        
        outputs = []
        for i, (chunk, score) in enumerate(results):
            paper_id = getattr(chunk, "paper_id", "Unknown Paper")
            outputs.append(f"[Score {score:.2f} | Paper: {paper_id}]:\n{chunk.text}")
        return "\n\n".join(outputs)
        
    def _search_kg(self, entity: str) -> str:
        """Graph 1-Hop Search Tool Block"""
        if not self.kg:
            return "Knowledge Graph is empty or not loaded."
            
        entity_lower = entity.lower().strip()
        connections = []
        
        for edge in self.kg.edges:
            s_id = edge.source.lower()
            t_id = edge.target.lower()
            
            # Substring match to be robust to full IDs like "Method:some_method"
            if entity_lower in s_id or entity_lower in t_id:
                other_id = edge.target if entity_lower in s_id else edge.source
                rel = edge.relation
                connections.append(f"- has relation [{rel}] with -> {other_id}")
                
        if not connections:
            return f"No connections found for entity matching '{entity}' in the graph."
            
        return f"KG Connections for '{entity}':\n" + "\n".join(set(connections))
        
    def _compare_nodes(self, node_a: str, node_b: str) -> str:
        """Graph Intersection Search Tool Block"""
        if not self.kg: 
            return "KG not available."
        
        a_id = node_a.lower().strip()
        b_id = node_b.lower().strip()
        
        a_neighbors = set()
        b_neighbors = set()
        
        for edge in self.kg.edges:
            s_id = edge.source.lower()
            t_id = edge.target.lower()
            
            target = edge.target if a_id in s_id else (edge.source if a_id in t_id else None)
            if target: a_neighbors.add(target)
                
            target_b = edge.target if b_id in s_id else (edge.source if b_id in t_id else None)
            if target_b: b_neighbors.add(target_b)
                
        common = a_neighbors.intersection(b_neighbors)
        
        if not common:
            # Maybe search individually
            return (f"No direct common neighbors found between '{node_a}' and '{node_b}'.\n"
                    f"'{node_a}' is connected to: {list(a_neighbors)[:3]}...\n"
                    f"'{node_b}' is connected to: {list(b_neighbors)[:3]}...")
            
        return f"Common connections shared by both '{node_a}' and '{node_b}': {', '.join(common)}"
