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

# --- Agentic Planner Schema ---
class QA_SubTask(BaseModel):
    task_description: str = Field(description="The specific sub-question to answer")
    suggested_tool: str = Field(description="The recommended tool to use ('search_vector', 'search_kg', 'compare_nodes')")

class QueryPlan(BaseModel):
    intent: str = Field(description="The primary intent of the user. E.g., 'compare_methods', 'find_dataset_details', 'summarize_paper'")
    strategy: str = Field(description="The chosen execution strategy: 'graph_first' (if needing relationships), 'vector_first' (if needing text details)")
    extracted_entities: List[str] = Field(description="Key entities/concepts mentioned in the question")
    sub_tasks: List[QA_SubTask] = Field(description="Broken down sequential questions needed to arrive at the final answer")

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

    def answer(self, question: str, kg: KnowledgeGraph, extractions: List[PaperExtraction], return_contexts: bool = False) -> Any:
        """
        Answers a user question iteratively using graph tools and vector tools.
        If return_contexts is True, returns a tuple (answer_string, list_of_context_strings).
        """
        if not self.llm:
            mock_ans = self.engine.generate(f"Mock Agentic QA for: {question}", system_prompt="", json_mode=False)
            return (mock_ans, ["Mock Context"]) if return_contexts else mock_ans
            
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
             "STRICT GROUNDING & ANTI-HALLUCINATION RULES (CRITICAL):\n"
             "1. You must ONLY answer using the exact entities, methods, datasets, and facts returned by your tools.\n"
             "2. SEVERELY PROHIBITED: Do not invent, guess, or provide external encyclopedic explanations for technical terms. "
             "If the KG says a paper uses 'dataset:hrsc', just state 'it uses the hrsc dataset'. DO NOT expand it to 'hrsc dataset for high-resolution ship detection' unless a Vector Search explicitly returned that textual definition.\n"
             "3. If explanations or meanings of methods/datasets are needed, YOU MUST call `search_vector` to fetch the descriptive text. If `search_vector` returns no text, YOU MUST NOT make up an explanation.\n"
             "4. Do not invent task/domain relationships. Just report the KG structure verbatim unless backed by vectors.\n\n"
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
        plan_str = ""
        
        # 3. Dynamic Agentic Planning Phase
        if hasattr(self.engine, "instructor_client") and self.engine.instructor_client:
            try:
                planner_prompt = (
                    f"Analyze this user question and create a step-by-step tool execution plan.\n"
                    f"Active Papers (Use as IDs): {active_papers}\n"
                    f"Question: {question}"
                )
                plan_schema = self.engine.generate_structured(
                    prompt=planner_prompt,
                    schema_class=QueryPlan,
                    system_prompt=(
                        "You are an expert Query routing agent. "
                        "Decompose the user's complex questions into logical sequential sub-tasks. "
                        "Determine if Graph-first, Vector-first, or Hybrid is best."
                    ),
                    max_retries=2
                )
                plan_str = "\n[DYNAMIC ROUTING PLAN (Follow this strictly)]\n"
                plan_str += f"Intent: {plan_schema.intent}\n"
                plan_str += f"Strategy: {plan_schema.strategy}\nRecommended Steps:\n"
                for i, task in enumerate(plan_schema.sub_tasks):
                    plan_str += f"{i+1}. Use {task.suggested_tool}: {task.task_description}\n"
                print(plan_str) # Intentionally print to stdout for visibility!
            except Exception as e:
                print(f"Warning: Planner routing failed, falling back to direct React behavior. ({e})")
                pass

        enhanced_question = f"""Question: {question}

[CRITICAL SYSTEM DIRECTIVE: 
1. The exact papers to analyze are: {active_papers}. 
2. You MUST use your tools (compare_nodes/search_kg/search_vector) immediately on these IDs. 
3. EXTREMELY IMPORTANT: When answering, you are STRICTLY FORBIDDEN from explaining, defining, or elaborating on any dataset, method, or concept unless you retrieved that exact definition using `search_vector`. 
4. If a tool just returns "dataset:hrsc", your final answer MUST simply say "hrsc". DO NOT say "hrsc (a dataset for ship detection)".
5. Any hallucination of background knowledge will result in immediate failure.]{plan_str}"""
        
        try:
            # invoke returns a dict with 'messages' list; the last AI message is the final output
            response = agent.invoke({"messages": [("user", enhanced_question)]})
            
            # Extract retrieved contexts from tool messages for Evaluation
            tool_contexts = []
            for msg in response["messages"]:
                # If message is a ToolMessage (which holds tool outputs)
                if hasattr(msg, "content") and getattr(msg, "type", "") == "tool":
                    tool_contexts.append(f"Tool {getattr(msg, 'name', 'unknown')} responded: " + msg.content)
            
            final_message = response["messages"][-1]
            answer_text = final_message.content
            
            if return_contexts:
                return answer_text, tool_contexts
            return answer_text
            
        except Exception as e:
            error_msg = f"Agent reasoning pipeline failed: {str(e)}"
            return (error_msg, []) if return_contexts else error_msg

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
