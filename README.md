# Knowledge-Driven Research Agent (KDRA)

A modular AI system for topic-driven academic paper analysis, structured information extraction, and comparative reasoning.

## Quickstart

### 1. Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
```

To install visualization dependencies:

```bash
pip install -e ".[viz]"
```

### 2. Run Demo (Offline Mode)

Run the CLI demo using the dummy heuristic backend (no LLM required):

```bash
python -m kdra.scripts.demo --topic "Transformer Architecture" --mode dummy
```

Or use the installed script alias:

```bash
kdra-demo --topic "Transformer Architecture" --mode dummy
```

### 3. Visualization

To launch the interactive dashboard:

```bash
streamlit run kdra/viz/app.py
```

## Project Structure

- `kdra/core`: Core logic modules (Ingestion, Retrieval, Reasoning, KG, Ontology).
- `kdra/pipeline`: End-to-end orchestration.
- `kdra/viz`: Streamlit visualization dashboard.
- `kdra/scripts`: CLI entry points.
