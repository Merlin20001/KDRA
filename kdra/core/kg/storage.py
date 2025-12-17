import json
import os
from typing import Dict
from kdra.core.schemas import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge

class GraphStorage:
    """
    Handles persistence of the Knowledge Graph.
    """
    
    def save_graph(self, graph: KnowledgeGraph, file_path: str):
        """
        Save the graph to a JSON file.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(graph.model_dump_json(indent=2))

    def load(self, file_path: str) -> KnowledgeGraph:
        """
        Load the graph from a JSON file.
        """
        if not os.path.exists(file_path):
            return KnowledgeGraph(nodes=[], edges=[])
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return KnowledgeGraph(**data)

    def merge(self, graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
        """
        Merges multiple subgraphs into a single unified graph.
        Deduplicates nodes by ID and edges by source+target+relation.
        """
        merged_nodes: Dict[str, KnowledgeGraphNode] = {}
        merged_edges: Set[str] = set()
        final_edges: list[KnowledgeGraphEdge] = []
        
        for g in graphs:
            for node in g.nodes:
                if node.id not in merged_nodes:
                    merged_nodes[node.id] = node
                else:
                    # Optional: Merge properties if needed
                    pass
            
            for edge in g.edges:
                edge_key = f"{edge.source}|{edge.target}|{edge.relation}"
                if edge_key not in merged_edges:
                    merged_edges.add(edge_key)
                    final_edges.append(edge)
                    
        return KnowledgeGraph(
            nodes=list(merged_nodes.values()),
            edges=final_edges
        )
