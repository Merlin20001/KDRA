"""
Prompt templates for the reasoning module.
"""

EXTRACTION_SYSTEM_PROMPT = """
You are an expert research assistant. Your task is to extract structured information from academic paper text.
You must strictly adhere to the requested JSON schema.
Do not include any conversational text, only the JSON output.

IMPORTANT:
- Use canonical names for entities (e.g., "Transformer" instead of "Transformers", "GPT-4" instead of "GPT4").
- Be specific but concise.
- Avoid generic terms like "Method" or "Model".
"""

DRAFT_PROMPT_TEMPLATE = """
Analyze the following text segments from a research paper and extract the key information.

Text Segments:
{context}

Extract the following fields and output as a JSON object with these exact keys:
{{
  "methods": ["list of short, concise method names (1-3 words)"],
  "datasets": ["list of dataset names"],
  "metrics": [
    {{"name": "string", "value": float, "unit": "string", "dataset": "string"}}
  ],
  "claims": ["list of strings"],
  "limitations": ["list of strings"],
  "concepts": ["list of 5-8 key concepts (1-3 words each), covering core topics and related techniques"]
}}
"""

VERIFICATION_PROMPT_TEMPLATE = """
Review the following extraction against the original text.

Original Text:
{context}

Draft Extraction:
{draft}

Check for hallucinations, missing info, and accuracy.
Output the corrected result as a valid JSON object with the same schema:
{{
  "methods": [...],
  "datasets": [...],
  "metrics": [...],
  "claims": [...],
  "limitations": [...],
  "concepts": [...]
}}
"""

COMPARISON_SYSTEM_PROMPT = """
You are a senior research scientist performing a meta-analysis of multiple academic papers.
Your goal is to synthesize insights, compare methodologies, and identify trends across the provided structured data.
You must strictly adhere to the requested JSON schema for the output.
"""

COMPARISON_PROMPT_TEMPLATE = """
Analyze the following structured data extracted from {count} research papers.

Topic: {topic}

Paper Data:
{data}

Generate a list of comparative insights.
Output the result as a JSON object containing a list under the key "insights":
{{
  "insights": [
    {{
      "topic": "specific dimension being compared",
      "papers_involved": ["list of paper_ids"],
      "insight_text": "natural language description",
      "category": "Performance/Methodology/Trend/Gap"
    }}
  ]
}}
"""

QA_SYSTEM_PROMPT = """
You are a helpful and knowledgeable academic research assistant.
You have access to a structured Knowledge Graph and detailed extractions from a set of research papers.
Your goal is to answer the user's questions based strictly on the provided context.
If the information is not in the context, state that you cannot find it in the provided papers.
"""

QA_PROMPT_TEMPLATE = """
Context Information (Knowledge Graph & Extractions):
{context}

User Question: {question}

Answer the question based on the context above. Cite specific papers (by ID) where appropriate.
"""
