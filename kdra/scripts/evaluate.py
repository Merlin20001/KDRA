import sys
import os
import json
from pyaml_env import parse_config
from kdra.pipeline.orchestrator import KDRAPipeline
from kdra.evaluation.evaluator import KDRAEvaluator

def run_evaluation(topic: str = "Transformer Architecture", output_dir: str = "./output"):
    print("=== KDRA Evaluation Pipeline (Running on Existing Knowledge Graph) ===")
    
    # 1. Load config
    config_path = "llm_config.yaml"
    if not os.path.exists(config_path):
        print("Error: llm_config.yaml is missing.")
        sys.exit(1)
        
    try:
        llm_config = parse_config(config_path)
    except Exception as e:
        print(f"Make sure pyaml-env is installed: pip install pyaml-env. Error: {e}")
        llm_config = {}

    # 2. Check if output files exist
    kg_path = os.path.join(output_dir, "knowledge_graph.json")
    ext_path = os.path.join(output_dir, "extractions.json")
    
    if not os.path.exists(kg_path) or not os.path.exists(ext_path):
        print(f"Error: Could not find existing Knowledge Graph at {output_dir}")
        print("Please process some papers first before running evaluation.")
        sys.exit(1)
        
    print(f"\n1. Loading existing Knowledge Graph and Extractions from [{output_dir}]...")
    
    # Load knowledge base
    try:
        with open(kg_path, "r", encoding="utf-8") as f:
            kg_data = json.load(f)
        with open(ext_path, "r", encoding="utf-8") as f:
            extractions_data = json.load(f)
    except Exception as e:
        print(f"Failed to load output files: {e}.")
        sys.exit(1)
        
    from kdra.core.schemas import KnowledgeGraph, PaperExtraction
    kg = KnowledgeGraph(**kg_data)
    extractions = [PaperExtraction(**e) for e in extractions_data]
    
    # Initialize pipeline simply to get access to the initialized Agent
    pipeline = KDRAPipeline(output_dir=output_dir, use_dummy=False, llm_config=llm_config)
    
    # 3. Setup Evaluator
    evaluator = KDRAEvaluator(llm_config=llm_config)
    
    # 4. Generate Test Questions
    test_questions = [
        "What are the main methods proposed in these papers?",
        "What datasets were used to evaluate the methods?",
        "Are there any common limitations mentioned in these papers?"
    ]

    # 5. Run Questions and Evaluate
    print("\n2. Running QA & Evaluation metrics...")
    
    results = []
    total_faithfulness = 0.0
    total_relevance = 0.0
    
    for i, q in enumerate(test_questions):
        print(f"\nEvaluating Q{i+1}: {q}")
        
        # Call the Agent with return_contexts=True
        answer, contexts = pipeline.assistant.answer(q, kg, extractions, return_contexts=True)
        
        print(f"-> Tool Contexts Retrieved: {len(contexts)}")
        print(f"-> Agent Output length: {len(answer)}")
        
        # Evaluate 
        eval_metrics = evaluator.evaluate_turn(q, answer, contexts)
        
        faith_score = eval_metrics.get('faithfulness', {}).get('score', 0.0)
        rel_score = eval_metrics.get('relevance', {}).get('score', 0.0)
        
        print(f"   [Score] Faithfulness: {faith_score:.2f}")
        for hall in eval_metrics.get('faithfulness', {}).get('hallucinations', []):
            print(f"      - Hallucination: {hall}")
        print(f"   [Score] Relevance:    {rel_score:.2f}")
        
        total_faithfulness += faith_score
        total_relevance += rel_score
        results.append({
            "question": q,
            "answer": answer,
            "metrics": eval_metrics
        })

    # Summary
    print("\n--- EVALUATION SUMMARY ---")
    print(f"Average Faithfulness : {total_faithfulness/len(test_questions):.2f}")
    print(f"Average Relevance    : {total_relevance/len(test_questions):.2f}")
    
    with open(os.path.join(output_dir, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed report saved to {os.path.join(output_dir, 'evaluation_report.json')}")

if __name__ == "__main__":
    run_evaluation()
