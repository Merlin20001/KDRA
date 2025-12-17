import os
from typing import Dict, Any, List
from kdra.pipeline.orchestrator import KDRAPipeline

def run_topic(topic: str, top_k: int = 10, mode: str = "dummy") -> Dict[str, Any]:
    """
    Run the KDRA pipeline for a specific topic.
    
    Args:
        topic: The research topic.
        top_k: Maximum number of papers to process.
        mode: Execution mode ("dummy" or "real").
        
    Returns:
        Dictionary containing results.
    """
    # 1. Locate Data
    # Assume data is in ./data relative to current working directory
    data_dir = os.path.abspath("./data")
    if not os.path.exists(data_dir):
        # Try relative to package if not found in CWD
        # This is a fallback for when running from site-packages or similar
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(package_dir, "../data")
        
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory not found at {data_dir}. Using empty list.")
        files = []
    else:
        files = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith(('.txt', '.md', '.pdf'))
        ]
        print(f"Found {len(files)} files in {data_dir}:")
        for f in files:
            print(f" - {os.path.basename(f)}")
    
    # Limit to top_k
    files = files[:top_k]
    
    # 2. Initialize Pipeline
    use_dummy = (mode == "dummy")
    output_dir = "./output"
    
    pipeline = KDRAPipeline(output_dir=output_dir, use_dummy=use_dummy)
    
    # 3. Run
    results = pipeline.run_topic(topic, files)
    
    return results
