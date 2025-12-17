import argparse
import sys
import os
import json

# Ensure the package is in path if run as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from kdra.pipeline.run_topic import run_topic

def main():
    parser = argparse.ArgumentParser(description="KDRA CLI Demo")
    parser.add_argument("--topic", type=str, required=True, help="Research topic")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top papers to retrieve")
    parser.add_argument("--mode", type=str, choices=["dummy", "real"], default="dummy", help="Execution mode")
    
    args = parser.parse_args()
    
    print(f"Running KDRA Demo (Topic: '{args.topic}', Mode: {args.mode})...", file=sys.stderr)
    
    try:
        results = run_topic(topic=args.topic, top_k=args.top_k, mode=args.mode)
        print(json.dumps(results, indent=2))
        sys.exit(0)
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
