from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

class PaperMetadata(BaseModel):
    """
    Metadata for a research paper.
    
    Attributes:
        paper_id: Unique identifier for the paper (e.g., DOI, arXiv ID, or hash).
        title: Title of the paper.
        year: Publication year.
        venue: Conference or journal name (e.g., "NeurIPS", "CVPR").
        authors: List of author names.
        url: Link to the paper (PDF or page).
    """
    paper_id: str = Field(..., description="Unique identifier for the paper")
    title: str = Field(..., description="Title of the paper")
    year: Optional[int] = Field(None, description="Publication year")
    venue: Optional[str] = Field(None, description="Conference or journal name")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    url: Optional[str] = Field(None, description="Link to the paper")

class PaperChunk(BaseModel):
    """
    A semantic chunk of text from a paper.
    
    Attributes:
        chunk_id: Unique identifier for the chunk.
        paper_id: ID of the source paper.
        text: The actual text content.
        section: Section header this chunk belongs to (e.g., "Introduction", "Methods").
        metadata: Copy of paper metadata for self-contained context.
    """
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    paper_id: str = Field(..., description="ID of the source paper")
    text: str = Field(..., description="The actual text content")
    section: Optional[str] = Field(None, description="Section header this chunk belongs to")
    metadata: PaperMetadata = Field(..., description="Paper metadata for context")

class MetricValue(BaseModel):
    """
    A quantitative or qualitative metric reported in a paper.
    
    Attributes:
        name: Name of the metric (e.g., "Accuracy", "F1 Score").
        value: The reported value (numeric or string representation).
        unit: Unit of measurement (e.g., "%", "seconds").
        dataset: The dataset on which this metric was evaluated.
        confidence: Optional confidence score of the extraction (0.0 to 1.0).
    """
    name: str = Field(..., description="Name of the metric")
    value: Union[float, str] = Field(..., description="The reported value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    dataset: Optional[str] = Field(None, description="The dataset used for evaluation")
    confidence: Optional[float] = Field(None, description="Confidence score of the extraction")

class PaperExtraction(BaseModel):
    """
    Structured information extracted from a single paper.
    
    Attributes:
        paper_id: ID of the paper.
        methods: List of proposed or used methods/models.
        datasets: List of datasets used or introduced.
        metrics: List of performance metrics reported.
        claims: Key claims or contributions made by the authors.
        limitations: Explicitly stated limitations or future work.
        evidence_spans: Mapping of extracted fields to source text spans for traceability.
    """
    paper_id: str = Field(..., description="ID of the paper")
    methods: List[str] = Field(default_factory=list, description="List of proposed or used methods")
    datasets: List[str] = Field(default_factory=list, description="List of datasets used")
    metrics: List[MetricValue] = Field(default_factory=list, description="List of performance metrics")
    claims: List[str] = Field(default_factory=list, description="Key claims or contributions")
    limitations: List[str] = Field(default_factory=list, description="Stated limitations")
    concepts: List[str] = Field(default_factory=list, description="Key concepts or knowledge points discussed")
    evidence_spans: Dict[str, List[str]] = Field(
        default_factory=dict, 
        description="Mapping of field names to source text spans for traceability"
    )

class ComparativeInsight(BaseModel):
    """
    A synthesized insight derived from comparing multiple papers.
    
    Attributes:
        topic: The specific topic or dimension being compared.
        papers_involved: List of paper IDs involved in this comparison.
        insight_text: The natural language description of the insight.
        category: Category of insight (e.g., "Methodological", "Performance", "Trend").
    """
    topic: str = Field(..., description="The specific topic being compared")
    papers_involved: List[str] = Field(..., description="List of paper IDs involved")
    insight_text: str = Field(..., description="Description of the insight")
    category: str = Field(..., description="Category of insight")

class NodeType(str, Enum):
    PAPER = "Paper"
    METHOD = "Method"
    DATASET = "Dataset"
    METRIC = "Metric"
    CONCEPT = "Concept"

class EdgeType(str, Enum):
    USES = "USES"
    EVALUATED_ON = "EVALUATED_ON"
    REPORTS = "REPORTS"
    RELATED_TO = "RELATED_TO"
    AUTHORED_BY = "AUTHORED_BY"

class KnowledgeGraphNode(BaseModel):
    """
    A node in the knowledge graph.
    
    Attributes:
        id: Unique identifier for the node.
        type: The type of entity (Paper, Method, Dataset, etc.).
        properties: Additional attributes (e.g., year, description).
    """
    id: str = Field(..., description="Unique identifier for the node")
    type: NodeType = Field(..., description="The type of entity")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional attributes")

class KnowledgeGraphEdge(BaseModel):
    """
    An edge in the knowledge graph representing a relationship.
    
    Attributes:
        source: ID of the source node.
        target: ID of the target node.
        relation: The type of relationship.
        properties: Additional attributes (e.g., weight, context).
    """
    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node")
    relation: EdgeType = Field(..., description="The type of relationship")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional attributes")

class KnowledgeGraph(BaseModel):
    """
    The complete knowledge graph structure.
    
    Attributes:
        nodes: List of all nodes in the graph.
        edges: List of all edges in the graph.
    """
    nodes: List[KnowledgeGraphNode] = Field(default_factory=list, description="List of nodes")
    edges: List[KnowledgeGraphEdge] = Field(default_factory=list, description="List of edges")
