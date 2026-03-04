import os
import json
import logging
from pyaml_env import parse_config
from kdra.pipeline.orchestrator import KDRAPipeline
from kdra.evaluation.evaluator import KDRAEvaluator
logging.basicConfig(level=logging.ERROR)

def run_quick_eval():
    print("=== Quick Eval ===")
    config_path = "llm_config.yaml"
    llm_config = parse_config(config_path) if os.path.exists(config_path) else {}
    
    kg_path = "./output/knowledge_graph.json"
    ext_path = "./output/extractions.json"
    with open(kg_path, "r", encoding="utf-8") as f: kg_data = json.load(f)
    with open(ext_path, "r", encoding="utf-8") as f: ext_data = json.load(f)
        
    from kdra.core.schemas import KnowledgeGraph, PaperExtraction
    kg = KnowledgeGraph(**kg_data)
    extractions = [PaperExtraction(**e) for e in ext_data]
    
    pipeline = KDRAPipeline(output_dir="./output", use_dummy=False, llm_config=llm_config)
    evaluator = KDRAEvaluator(llm_config=llm_config)
    
    q = "What datasets were used to evaluate the methods?"
    print(f"\nQuestion: {q}")
    answer, contexts = pipeline.assistant.answer(q, kg, extractions, return_contexts=True)
    
    print(f"\n[AGENT ANSWER]\n{answer}\n")
    print(f"[CONTEXTS RETRIEVED]\n{len(contexts)} contexts retrieved.")
    
    eval_metrics = evaluator.evaluate_turn(q, answer, contexts)
    print(f"\n[SCORE] Faithfulness: {eval_metrics.get('faithfulness', {}).get('score')}")
    print(f"[REASONING]: {eval_metrics.get('faithfulness', {}).get('reasoning')}")

if __name__ == "__main__":
    run_quick_eval()
