# Knowledge-Driven Research Agent (KDRA)

A powerful, modular "Agentic GraphRAG" AI pipeline dedicated to academic document analysis. It bridges semantic vector databases, multi-level logic reasoning, and a deterministically constructed knowledge graph to provide deep research insights, structured extractions, and multi-hop comparations across academic papers.

## Features

- **Multi-tiered Deep Ingestion**: Safeguards tables, formats, and equations through a fallback approach parsing via MinerU/Marker, PyMuPDF, and PyPDF.
- **Offline Embeddings**: Uses fast, local `sentence-transformers` (`BAAI/bge-small-en-v1.5`) to eliminate heavy API cost dependencies for RAG vector retrieval.
- **Agentic Multi-Hop GraphRAG**: Driven by LangGraph's ReAct Agent (`create_react_agent`). Capable of traversing nodes via `search_kg` or pulling textual evidence with `search_vector`.
- **Hybrid NLP Entity Extraction**: Uses `GLiNER` locally alongside structured LLM extraction for precise, schema-constrained Graph creation.
- **Interactive Visualization**: Employs Streamlit combined with `pyvis` for interactive, physics-based, session-filtered knowledge graphs that bypass historical node bloat.

## Quickstart

### 1. Installation

Clone the repository and install the core pipeline in editable mode:

```bash
pip install -e .
```

Install standard Agent & Visualization dependencies:

```bash
pip install -e ".[viz]"
pip install streamlit pyvis sentence-transformers gliner langgraph langchain-openai langchain-core
```

### 2. Configuration (For LLM Agent Layer)

Ensure you have a `llm_config.yaml` file in the root environment, looking like:
```yaml
api_key: "sk-your-llm-token"
base_url: "https://api.your-provider.com/v1"
model_name: "gpt-4o-mini" # Or any capable OpenAI-compatible tool-calling model
```

### 3. Execution

**Interactive Visualization (Recommended)**
Launch the Web UI dashboard for paper processing, graph viewing, and multi-hop Q&A:

```bash
streamlit run kdra/viz/app.py
```

**Run Demo (Offline Mock Mode)**
Experience the pipeline logically with dummy outputs:
```bash
python -m kdra.scripts.demo --topic "Transformer Architecture" --mode dummy
```

## Architecture Map

- `kdra/core/ingestion`: Multi-stage PDF fallback chunker.
- `kdra/core/reasoning`: Houses the Agentic Q&A LangGraph runner (`qa.py`), LLM extractor (`extractor.py`), and GLiNER extractor (`ner.py`).
- `kdra/core/retrieval`: Localized Sentence-Transformer Vector Retriever and ArXiv external fetching.
- `kdra/core/kg/ & schemas/`: Pydantic enforced Schema validations and persistent Knowledge graph merging.
- `kdra/pipeline`: Connects parsing -> extraction -> graph creation into single cohesive commands.
- `kdra/viz`: Interactive Frontend logic (`app.py`).
