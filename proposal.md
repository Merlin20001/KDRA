Knowledge-Driven Research Agent (KDRA)

Engineering-Level Project Specification for Code Generation

⸻

0. Project Objective (Non-Negotiable)

You are asked to generate a production-style research assistant system called Knowledge-Driven Research Agent (KDRA).

The system must support:
	•	Topic-driven academic paper analysis
	•	Structured information extraction (methods, datasets, metrics, limitations)
	•	Comparative reasoning across multiple papers
	•	Lightweight knowledge-graph construction
	•	Interactive visualization for research insights

This is NOT a chatbot.
This is a pipeline-oriented, modular AI system intended for research analysis and demonstration.

⸻

1. Functional Requirements (What the system MUST do)

1.1 Input
	•	A research topic string (e.g., "long-term memory for LLM agents")
	•	Optional constraints:
	•	publication year range
	•	venue filter
	•	maximum number of papers

1.2 Core Capabilities

The system MUST:
	1.	Retrieve semantically relevant academic papers using vector similarity search.
	2.	Segment papers into chunks with metadata (paper_id, section, year, venue).
	3.	Extract structured, schema-constrained information from each paper:
	•	Methods
	•	Datasets
	•	Metrics
	•	Claims
	•	Limitations
	4.	Perform comparative reasoning across multiple papers to identify:
	•	Methodological differences
	•	Strengths and weaknesses
	•	Research gaps and trends
	5.	Construct a knowledge graph linking:
	•	Papers
	•	Methods
	•	Datasets
	•	Metrics
	•	High-level concepts
	6.	Provide interactive visualization:
	•	Comparative tables
	•	Trend plots over time
	•	Graph visualization of the knowledge structure

⸻

2. Non-Functional Constraints (Critical)
	•	All extracted outputs MUST conform to strict JSON schemas
	•	Every extracted claim MUST be traceable to evidence text spans
	•	System must be modular and pipeline-oriented, not monolithic
	•	All components must be replaceable (LLM, vector DB, graph backend)
	•	The system must support incremental development and demo-ready execution

⸻

3. Architecture Overview (Copilot must follow this)

The system follows a linear-to-graph hybrid pipeline:
Topic
  ↓
Semantic Retrieval (Vector DB)
  ↓
Paper Chunk Grouping
  ↓
LLM-based Structured Extraction (JSON)
  ↓
Ontology / Concept Alignment
  ↓
Comparative Reasoning
  ↓
Knowledge Graph Construction
  ↓
Interactive Visualization

Each stage MUST be implemented as a separate module with clean interfaces.

⸻

4. Module Responsibilities (Strict Separation)

4.1 Ingestion Module

Responsible for:
	•	Loading papers (PDF / arXiv / URLs)
	•	Chunking text into semantically meaningful segments
	•	Attaching metadata (paper_id, title, year, venue, section)

Must NOT:
	•	Perform reasoning
	•	Perform comparisons

⸻

4.2 Retrieval Module

Responsible for:
	•	Embedding queries and chunks
	•	Vector similarity search
	•	Returning top-K relevant chunks with scores

Must return:
	•	paper_id
	•	chunk_text
	•	similarity_score
	•	metadata

⸻

4.3 Schema Definition Module (Critical)

All extracted content MUST follow predefined schemas.

Schemas MUST include:
	•	PaperExtraction
	•	MetricValue
	•	KnowledgeGraphNode
	•	KnowledgeGraphEdge
	•	ComparativeInsight

Schemas MUST:
	•	Allow missing values (Optional fields)
	•	Be deterministic
	•	Be serializable to JSON without post-processing

⸻

4.4 Reasoning & Extraction Module

Responsible for:
	•	Using LLMs to extract structured information
	•	Enforcing schema compliance
	•	Performing self-verification (missing fields, contradictions)

Extraction MUST:
	•	Use multi-step prompting
	•	Produce no free-form text outside schema
	•	Preserve evidence spans for traceability

⸻

4.5 Comparative Reasoning Module

Responsible for:
	•	Comparing extracted results across papers
	•	Producing:
	•	Comparative tables
	•	Cross-paper insights
	•	Identified research gaps

This module reasons over structured data, not raw text.

⸻

4.6 Ontology & Concept Alignment Module

Responsible for:
	•	Mapping extracted methods and concepts to:
	•	External ontologies (e.g., ACM CCS, OpenAlex)
	•	Or internal normalized concept IDs

Must:
	•	Resolve synonyms
	•	Track concept frequency and temporal trends

⸻

4.7 Knowledge Graph Module

Responsible for:
	•	Building a graph with:
	•	Nodes: Paper, Method, Dataset, Metric, Concept
	•	Edges: USES, EVALUATED_ON, REPORTS, RELATED_TO
	•	Supporting:
	•	JSON-based lightweight graph
	•	Optional Neo4j backend

Graph generation MUST be deterministic.

⸻

4.8 Visualization Module

Responsible for:
	•	Interactive UI (Streamlit-style)
	•	Components:
	•	Topic input panel
	•	Paper list
	•	Comparative table
	•	Trend plots
	•	Knowledge graph visualization

Visualization MUST consume pipeline outputs directly.

⸻

5. Project Structure (Copilot must generate this)

The codebase MUST follow this structure:

kdra/
├── core/
│   ├── schemas/
│   ├── ingestion/
│   ├── retrieval/
│   ├── reasoning/
│   ├── kg/
│   └── ontology/
├── pipeline/
├── viz/
├── scripts/
├── tests/
└── README.md

Each directory must:
	•	Contain an __init__.py
	•	Expose a minimal public API
	•	Avoid cross-layer imports (no circular dependencies)

⸻

6. Execution Modes

The system MUST support:
	1.	CLI mode
	•	Run full pipeline from terminal
	2.	Interactive demo mode
	•	Streamlit UI
	3.	Programmatic API
	•	Import pipeline as a Python module

⸻

7. Deliverables the System Must Enable

By design, the system must make it trivial to produce:
	•	A live interactive demo
	•	Screenshots and videos for presentation
	•	A structured technical report describing:
	•	Architecture
	•	Design decisions
	•	Example research insights

⸻

8. Design Philosophy (Copilot must respect this)
	•	Prefer clarity over cleverness
	•	Prefer explicit schemas over flexible text
	•	Prefer pipeline composition over agent chaos
	•	Prefer traceable reasoning over fluent language

This system is a research infrastructure, not a chat assistant.

⸻

9. Final Instruction to Code Generator

Generate the entire project following this specification exactly.
Do not simplify modules.
Do not merge responsibilities.
Do not remove schemas.
Assume the system will be evaluated on architectural soundness, not just functionality.

⸻
