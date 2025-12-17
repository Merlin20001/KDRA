import os
from typing import List, Dict, Optional, Any
from kdra.core.schemas import PaperMetadata, PaperExtraction, KnowledgeGraph, ComparativeInsight
from kdra.core.ingestion import PaperIngestor
from kdra.core.reasoning import (
    PaperExtractor, 
    ComparativeAnalyst, 
    ResearchAssistant,
    MockReasoningEngine,
    OpenAIEngine,
    DummyExtractor,
    DummyComparator
)
from kdra.core.kg import GraphBuilder, GraphStorage
from kdra.core.retrieval.simple import MockRetriever
from kdra.core.retrieval.vector import VectorRetriever

class KDRAPipeline:
    """
    The main orchestrator for the Knowledge-Driven Research Agent.
    Connects Ingestion -> Extraction -> KG Construction -> Comparative Reasoning.
    """
    
    def __init__(self, output_dir: str = "./output", use_dummy: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            output_dir: Directory to save results.
            use_dummy: If True, use heuristic-based dummy components instead of LLM.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.ingestor = PaperIngestor()
        self.use_dummy = use_dummy
        
        if use_dummy:
            print("Initializing KDRA with Dummy Components (Offline Mode)")
            self.extractor = DummyExtractor()
            self.comparator = DummyComparator()
            self.assistant = ResearchAssistant(engine=MockReasoningEngine())
            self.retriever = MockRetriever()
        else:
            print("Initializing KDRA with Real LLM Engine (OpenAI Compatible)")
            # Try to initialize engine (it will check config file internally)
            try:
                self.engine = OpenAIEngine()
                # Check if key is valid (simple check)
                if not self.engine.api_key or self.engine.api_key == "sk-...":
                     print("WARNING: No valid API Key found in llm_config.yaml or environment variables.")
                     print("Falling back to Mock Engine.")
                     self.engine = MockReasoningEngine()
                     self.retriever = MockRetriever()
                else:
                     self.retriever = VectorRetriever(api_key=self.engine.api_key)
            except Exception as e:
                print(f"Error initializing OpenAI Engine: {e}")
                self.engine = MockReasoningEngine()
                self.retriever = MockRetriever()
                
            self.extractor = PaperExtractor(engine=self.engine)
            self.comparator = ComparativeAnalyst(engine=self.engine)
            self.assistant = ResearchAssistant(engine=self.engine)
            
        self.graph_builder = GraphBuilder()
        self.graph_storage = GraphStorage()

    def process_papers(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest and extract information from papers to build the Knowledge Graph.
        Does NOT perform comparative analysis (no topic required).
        """
        print(f"Processing {len(file_paths)} papers...")
        
        all_extractions = []
        subgraphs = []
        all_chunks = []
        errors = []

        for path in file_paths:
            try:
                print(f"Processing: {path}")
                # Ingest
                metadata = PaperMetadata(
                    paper_id=os.path.basename(path),
                    title=os.path.basename(path),
                    url=f"file://{os.path.abspath(path)}"
                )
                chunks = self.ingestor.ingest_file(path, metadata)
                all_chunks.extend(chunks)
                
                # Extract
                extraction = self.extractor.extract(metadata.paper_id, chunks)
                all_extractions.append(extraction)
                
                # Build Subgraph
                subgraph = self.graph_builder.build_subgraph(extraction)
                subgraphs.append(subgraph)
                
            except Exception as e:
                error_msg = f"Error processing {os.path.basename(path)}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                continue

        if not all_extractions:
            print("No papers successfully processed.")
            return {"errors": errors}

        # 2. Indexing for RAG
        print("Indexing chunks for RAG...")
        self.retriever.index(all_chunks)

        # 3. Merge Graphs
        print("Building Knowledge Graph...")
        full_kg = self.graph_builder.merge_graphs(subgraphs)
        
        # 4. Save Results
        self.graph_storage.save_graph(full_kg, os.path.join(self.output_dir, "knowledge_graph.json"))
        
        # Save extractions
        import json
        with open(os.path.join(self.output_dir, "extractions.json"), 'w') as f:
            json.dump([e.model_dump() for e in all_extractions], f, indent=2)
            
        return {
            "paper_count": len(all_extractions),
            "knowledge_graph": full_kg.model_dump(),
            "extractions": [e.model_dump() for e in all_extractions]
        }

    def answer_question(self, question: str, kg_data: Dict, extractions_data: List[Dict]) -> str:
        """
        Answer a question using the provided KG and extractions context, plus RAG.
        """
        # Reconstruct objects from dicts
        kg = KnowledgeGraph(**kg_data)
        extractions = [PaperExtraction(**e) for e in extractions_data]
        
        # Retrieve relevant chunks
        retrieved_results = self.retriever.retrieve(question, top_k=5)
        retrieved_chunks = [r[0].text for r in retrieved_results]
        
        return self.assistant.answer(question, kg, extractions, retrieved_chunks)

    def run_topic(self, topic: str, file_paths: List[str]) -> Dict[str, Any]:
        """
        Run the full pipeline for a specific research topic.
        
        Args:
            topic: The research topic for comparative analysis.
            file_paths: List of paths to paper files (txt, md, etc.).
            
        Returns:
            Dictionary containing the results (graph, insights, extractions).
        """
        print(f"Starting KDRA Pipeline for topic: '{topic}'")
        
        all_extractions: List[PaperExtraction] = []
        subgraphs: List[KnowledgeGraph] = []
        
        # 1. Ingestion & Extraction Loop
        for path in file_paths:
            print(f"Processing: {path}")
            try:
                # Ingest
                # TODO: Extract real metadata from file
                metadata = PaperMetadata(
                    paper_id=os.path.basename(path),
                    title=os.path.basename(path),
                    url=f"file://{os.path.abspath(path)}"
                )
                chunks = self.ingestor.ingest_file(path, metadata)
                
                # Extract
                extraction = self.extractor.extract(metadata.paper_id, chunks)
                all_extractions.append(extraction)
                
                # Build Subgraph
                subgraph = self.graph_builder.build_subgraph(extraction)
                subgraphs.append(subgraph)
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        if not all_extractions:
            print("No papers successfully processed.")
            return {}

        # 2. KG Construction (Merge)
        print("Building Knowledge Graph...")
        unified_graph = self.graph_storage.merge(subgraphs)
        self.graph_storage.save_graph(unified_graph, os.path.join(self.output_dir, "knowledge_graph.json"))
        
        # 3. Comparative Reasoning
        print("Generating Comparative Insights...")
        insights = self.comparator.compare(all_extractions, topic)
        
        # 4. Save Results
        results = {
            "topic": topic,
            "paper_count": len(all_extractions),
            "knowledge_graph": unified_graph.model_dump(),
            "insights": [i.model_dump() for i in insights],
            "extractions": [e.model_dump() for e in all_extractions]
        }
        
        # Save artifacts
        import json
        with open(os.path.join(self.output_dir, "insights.json"), 'w') as f:
            json.dump(results["insights"], f, indent=2)
            
        with open(os.path.join(self.output_dir, "extractions.json"), 'w') as f:
            json.dump(results["extractions"], f, indent=2)
            
        print(f"Pipeline completed. Results saved to {self.output_dir}")
        return results
