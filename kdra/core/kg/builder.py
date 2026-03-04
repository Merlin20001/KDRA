from typing import List, Set
from kdra.core.schemas import (
    PaperExtraction, 
    KnowledgeGraph, 
    KnowledgeGraphNode, 
    KnowledgeGraphEdge, 
    NodeType, 
    EdgeType
)
from kdra.core.ontology.normalizer import VectorOntologyNormalizer

class GraphBuilder:
    """
    Constructs knowledge graph components from structured paper extractions.
    Ensures deterministic node IDs and relationship mapping.
    """
    
    def __init__(self):
        self.normalizer = VectorOntologyNormalizer()
        
    def build_subgraph(self, extraction: PaperExtraction) -> KnowledgeGraph:
        """
        Converts a single paper extraction into a subgraph of nodes and edges.
        
        Args:
            extraction: The structured data extracted from a paper.
            
        Returns:
            A KnowledgeGraph object containing the nodes and edges for this paper.
        """
        nodes: List[KnowledgeGraphNode] = []
        edges: List[KnowledgeGraphEdge] = []
        
        # 1. Create Paper Node
        paper_props = {
            "claims": extraction.claims,
            "limitations": extraction.limitations
        }
        
        # Inject real metadata into the node properties if available
        if getattr(extraction, "metadata", None):
            meta = extraction.metadata
            paper_props["title"] = meta.title
            paper_props["authors"] = meta.authors
            paper_props["year"] = meta.year
            paper_props["venue"] = meta.venue
            paper_props["url"] = meta.url

        paper_node = KnowledgeGraphNode(
            id=extraction.paper_id,
            type=NodeType.PAPER,
            properties=paper_props
        )
        nodes.append(paper_node)
        
        # 2. Create Method Nodes & Edges
        for method_name in extraction.methods:
            method_id = self._normalize_id(NodeType.METHOD, method_name)
            nodes.append(KnowledgeGraphNode(
                id=method_id,
                type=NodeType.METHOD,
                properties={"name": method_name}
            ))
            edges.append(KnowledgeGraphEdge(
                source=extraction.paper_id,
                target=method_id,
                relation=EdgeType.USES
            ))
            
        # 3. Create Dataset Nodes & Edges
        for dataset_name in extraction.datasets:
            dataset_id = self._normalize_id(NodeType.DATASET, dataset_name)
            nodes.append(KnowledgeGraphNode(
                id=dataset_id,
                type=NodeType.DATASET,
                properties={"name": dataset_name}
            ))
            edges.append(KnowledgeGraphEdge(
                source=extraction.paper_id,
                target=dataset_id,
                relation=EdgeType.EVALUATED_ON
            ))
            
        # 4. Create Concept Nodes & Edges
        for concept_name in extraction.concepts:
            concept_id = self._normalize_id(NodeType.CONCEPT, concept_name)
            nodes.append(KnowledgeGraphNode(
                id=concept_id,
                type=NodeType.CONCEPT,
                properties={"name": concept_name}
            ))
            edges.append(KnowledgeGraphEdge(
                source=extraction.paper_id,
                target=concept_id,
                relation=EdgeType.RELATED_TO
            ))
            
        # 4.5 Create Co-occurrence Edges (Concept-Concept, Method-Concept)
        # This enriches the graph by showing how concepts and methods relate independent of the paper
        import itertools
        
        # Concept-Concept Co-occurrence (Restored but limited)
        # Link top concepts to show relationships (Limit to 6 to keep edges manageable ~15 edges)
        top_concepts = extraction.concepts[:6]
        for c1, c2 in itertools.combinations(top_concepts, 2):
            id1 = self._normalize_id(NodeType.CONCEPT, c1)
            id2 = self._normalize_id(NodeType.CONCEPT, c2)
            if id1 < id2:
                edges.append(KnowledgeGraphEdge(source=id1, target=id2, relation=EdgeType.RELATED_TO))
            else:
                edges.append(KnowledgeGraphEdge(source=id2, target=id1, relation=EdgeType.RELATED_TO))

        # Method-Concept Co-occurrence (Method implements/uses Concept)
        # Link methods to top concepts
        if len(extraction.methods) < 10:
            for method in extraction.methods:
                m_id = self._normalize_id(NodeType.METHOD, method)
                for concept in top_concepts:
                    c_id = self._normalize_id(NodeType.CONCEPT, concept)
                    edges.append(KnowledgeGraphEdge(source=m_id, target=c_id, relation=EdgeType.RELATED_TO))

        # 5. Create Metric Nodes & Edges
        # Metrics are unique per paper/dataset/method tuple usually, but here we simplify.
        # We treat the metric value as a property of the edge or a specific node?
        # Schema says MetricValue has name, value, unit.
        # Let's create a Metric Node for the *type* of metric (e.g. "Accuracy") 
        # and store the value in the edge or a specific instance node.
        # Given the schema simplicity, let's make the Metric Node specific to the measurement event 
        # OR link Paper -> MetricType with value in edge.
        # Let's go with: Paper -> REPORTS -> MetricNode (which contains the value)
        # This allows the graph to hold the specific results.
        
        for i, metric in enumerate(extraction.metrics):
            # Unique ID for this specific result
            metric_id = f"{extraction.paper_id}_metric_{i}"
            nodes.append(KnowledgeGraphNode(
                id=metric_id,
                type=NodeType.METRIC,
                properties={
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "dataset": metric.dataset
                }
            ))
            edges.append(KnowledgeGraphEdge(
                source=extraction.paper_id,
                target=metric_id,
                relation=EdgeType.REPORTS
            ))
            
            # If dataset is known, link Metric -> Dataset
            if metric.dataset:
                dataset_id = self._normalize_id(NodeType.DATASET, metric.dataset)
                edges.append(KnowledgeGraphEdge(
                    source=metric_id,
                    target=dataset_id,
                    relation=EdgeType.RELATED_TO
                ))

        return KnowledgeGraph(nodes=nodes, edges=edges)

    def _normalize_id(self, node_type: NodeType, name: str) -> str:
        """
        Creates a deterministic ID from a name, resolving semantic duplicates
        using Vector embeddings if available.
        e.g., "Method" + "Transformer" -> "method:transformer"
        """
        return self.normalizer.normalize_and_dedupe(node_type.value, name)

    def merge_graphs(self, subgraphs: List[KnowledgeGraph]) -> KnowledgeGraph:
        """
        Merges multiple subgraphs into a single Knowledge Graph, deduplicating nodes and edges.
        
        Args:
            subgraphs: A list of KnowledgeGraph objects.
            
        Returns:
            A single merged KnowledgeGraph.
        """
        merged_nodes = {}
        merged_edges = set()
        
        for graph in subgraphs:
            for node in graph.nodes:
                if node.id not in merged_nodes:
                    merged_nodes[node.id] = node
                else:
                    # Optional: Merge properties if needed. For now, first writer wins or we assume consistency.
                    # If we wanted to merge lists (like claims), we could do it here.
                    pass
            
            for edge in graph.edges:
                # Create a tuple representation for deduplication
                edge_tuple = (edge.source, edge.target, edge.relation)
                if edge_tuple not in merged_edges:
                    merged_edges.add(edge_tuple)
        
        # Convert edge tuples back to objects (or just keep the original objects if we tracked them)
        # Since we only stored tuples in the set, we need to reconstruct or store the object.
        # Better approach: Store the object in a list if the tuple hasn't been seen.
        
        final_edges = []
        seen_edges = set()
        
        for graph in subgraphs:
            for edge in graph.edges:
                edge_key = (edge.source, edge.target, edge.relation)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    final_edges.append(edge)
                    
        return KnowledgeGraph(
            nodes=list(merged_nodes.values()),
            edges=final_edges
        )
