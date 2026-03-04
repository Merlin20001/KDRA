import json
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class FaithfulnessScore(BaseModel):
    reasoning: str = Field(description="Step by step reasoning about whether the answer is supported by the contexts.")
    score: float = Field(description="A score between 0.0 and 1.0 indicating how faithful the answer is to the context. 1.0 is perfect.", ge=0.0, le=1.0)
    hallucinations: List[str] = Field(description="List of claims in the answer that are not supported by the context.")

class RelevanceScore(BaseModel):
    reasoning: str = Field(description="Step by step reasoning about whether the answer directly addresses the user's question.")
    score: float = Field(description="A score between 0.0 and 1.0 indicating how relevant the answer is. 1.0 is perfect.", ge=0.0, le=1.0)

class KDRAEvaluator:
    """
    Industrial LLM-as-a-Judge Evaluation Pipeline for GraphRAG.
    Calculates Ragas-style metrics: Faithfulness and Answer Relevance.
    """
    
    def __init__(self, llm_config: Dict[str, str] = None):
        config = llm_config or {}
        api_key = config.get("api_key", "sk-mock")
        
        if api_key == "sk-mock" or "sk-your-llm-token" in api_key:
            logger.warning("No valid API Key provided to Evaluator. It will not work properly.")
            
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=config.get("base_url"),
            model=config.get("model_name", "gpt-4o"),
            temperature=0.0
        )
        
        # Initialize structured outputs for absolute JSON stability
        self.faithfulness_chain = self.llm.with_structured_output(FaithfulnessScore)
        self.relevance_chain = self.llm.with_structured_output(RelevanceScore)

    def evaluate_faithfulness(self, question: str, answer: str, context: List[str]) -> Dict[str, Any]:
        """
        Measures if the generated answer is strictly derived from the retrieved context.
        Penalizes hallucinations where the model makes up facts not in the context.
        """
        contexts_str = "\n\n---\n\n".join(context)
        
        prompt = f"""You are an expert GraphRAG evaluator measuring 'Faithfulness'.
Given the following QUESTION, RETRIEVED CONTEXT, and GENERATED ANSWER, determine if the answer contains any information that cannot be directly inferred from the context.

QUESTION:
{question}

RETRIEVED CONTEXT (Graph Nodes, Edges, and Text Chunks):
{contexts_str}

GENERATED ANSWER:
{answer}

Instructions:
1. Carefully map every specific claim in the Answer to the Context.
2. If the Answer contains numbers, method names, or facts NOT in the Context, penalize the score.
3. If the Context is empty but the Answer provides facts, the score must be very low (hallucination).
4. Output a logical reasoning steps, a score [0.0 to 1.0], and a list of hallucinated claims.
"""
        try:
            result: FaithfulnessScore = self.faithfulness_chain.invoke(prompt)
            return result.model_dump()
        except Exception as e:
            logger.error(f"Faithfulness eval failed: {e}")
            return {"score": 0.0, "reasoning": str(e), "hallucinations": []}

    def evaluate_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Measures if the generated answer directly addresses the user's question, 
        regardless of whether it's truthful or not.
        """
        prompt = f"""You are an expert evaluator measuring 'Answer Relevance'.
Given a QUESTION and a GENERATED ANSWER, evaluate how directly and thoroughly the answer addresses the question.

QUESTION:
{question}

GENERATED ANSWER:
{answer}

Instructions:
1. Does the answer get straight to the point?
2. Does it ignore the question and talk about something else? (Penalize)
3. If the answer says "I don't know based on the context", but it's a direct response to the query, it is still logically relevant, but maybe partial.
4. Output reasoning and a score between 0.0 and 1.0.
"""
        try:
            result: RelevanceScore = self.relevance_chain.invoke(prompt)
            return result.model_dump()
        except Exception as e:
            logger.error(f"Relevance eval failed: {e}")
            return {"score": 0.0, "reasoning": str(e)}

    def evaluate_turn(self, question: str, answer: str, context: List[str]) -> Dict[str, Any]:
        """Runs all metrics for a single Q&A turn."""
        faith_res = self.evaluate_faithfulness(question, answer, context)
        rel_res = self.evaluate_relevance(question, answer)
        
        return {
            "faithfulness": faith_res,
            "relevance": rel_res,
            "summary_score": (faith_res["score"] + rel_res["score"]) / 2.0
        }
