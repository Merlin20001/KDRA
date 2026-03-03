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
    
    def __init__(self, output_dir: str = "./output", use_dummy: bool = True, llm_config: Optional[Dict[str, str]] = None):
        """
        Initialize the pipeline.
        
        Args:
            output_dir: Directory to save results.
            use_dummy: If True, use heuristic-based dummy components instead of LLM.
            llm_config: Dictionary containing 'api_key', 'base_url', and 'model_name'.
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
            self.retriever = MockRetriever()
            self.assistant = ResearchAssistant(engine=MockReasoningEngine(), retriever=self.retriever)
        else:
            print("Initializing KDRA with Real LLM Engine (OpenAI Compatible)")
            # Try to initialize engine (it will check config file internally)
            try:
                # Use provided config or empty dict
                config = llm_config or {}
                self.engine = OpenAIEngine(
                    api_key=config.get("api_key"),
                    base_url=config.get("base_url"),
                    model=config.get("model_name")
                )
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
            self.assistant = ResearchAssistant(engine=self.engine, retriever=self.retriever)
            
        self.graph_builder = GraphBuilder()
        self.graph_storage = GraphStorage()

    def process_papers(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest and extract information from papers to build the Knowledge Graph.
        Does NOT perform comparative analysis (no topic required).
        Supports incremental updates.
        """
        print(f"Processing {len(file_paths)} papers (Incremental Mode)...")
        import json
        
        existing_extractions = []
        existing_paper_ids = set()
        extractions_path = os.path.join(self.output_dir, "extractions.json")
        
        # Load existing extractions
        if os.path.exists(extractions_path):
            try:
                with open(extractions_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        ext = PaperExtraction(**item)
                        existing_extractions.append(ext)
                        existing_paper_ids.add(ext.paper_id)
                print(f"Loaded {len(existing_extractions)} existing extractions.")
            except Exception as e:
                print(f"Failed to load existing extractions: {e}")

        all_extractions = existing_extractions.copy()
        subgraphs = []
        all_chunks = []
        errors = []
        
        # Incremental: Load existing knowledge graph if any
        kg_path = os.path.join(self.output_dir, "knowledge_graph.json")
        if os.path.exists(kg_path):
            existing_graph = self.graph_storage.load(kg_path)
            subgraphs.append(existing_graph)
            # Sync normalizer memory to maintain deduplication consistency across increments
            self.graph_builder.normalizer.sync_from_graph(existing_graph)

        new_papers_processed = 0

        for path in file_paths:
            paper_id = os.path.basename(path)
            if paper_id in existing_paper_ids:
                print(f"Skipping already processed paper: {paper_id}")
                continue

            try:
                print(f"Processing new paper: {path}")
                # Ingest
                metadata = PaperMetadata(
                    paper_id=paper_id,
                    title=paper_id,
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
                new_papers_processed += 1
                
            except Exception as e:
                error_msg = f"Error processing {paper_id}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                continue

        if not all_extractions:
            print("No papers available in the project.")
            return {"errors": errors}

        if new_papers_processed > 0:
            # 2. Indexing for RAG (Only for new chunks)
            print("Indexing new chunks for RAG...")
            self.retriever.index(all_chunks)
            
            # 3. Merge Graphs
            print("Building/Updating Knowledge Graph...")
            full_kg = self.graph_storage.merge(subgraphs)
            
            # 4. Save Results
            self.graph_storage.save_graph(full_kg, kg_path)
            with open(extractions_path, 'w', encoding='utf-8') as f:
                json.dump([e.model_dump() for e in all_extractions], f, indent=2, ensure_ascii=False)
        else:
            print("No new papers to process. Returning existing graph and extractions.")
            # Retrieve from existing state
            full_kg = self.graph_storage.merge(subgraphs)
            
        return {
            "paper_count": len(all_extractions),
            "knowledge_graph": full_kg.model_dump(),
            "extractions": [e.model_dump() for e in all_extractions]
        }

    def answer_question(self, question: str, kg_data: Dict, extractions_data: List[Dict]) -> str:
        """
        Answer a question using the provided KG and extractions context, plus RAG.
        Now uses Agentic multi-hop retrieval.
        """
        # Reconstruct objects from dicts
        kg = KnowledgeGraph(**kg_data)
        extractions = [PaperExtraction(**e) for e in extractions_data]
        
        # Agent will internally use self.retriever and the KG to find answers
        return self.assistant.answer(question, kg, extractions)

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
        
        import json
        existing_extractions: List[PaperExtraction] = []
        existing_paper_ids = set()
        extractions_path = os.path.join(self.output_dir, "extractions.json")
        
        # Incremental: Load existing extractions
        if os.path.exists(extractions_path):
            print("Loading existing extractions for incremental updates...")
            try:
                with open(extractions_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        ext = PaperExtraction(**item)
                        existing_extractions.append(ext)
                        existing_paper_ids.add(ext.paper_id)
                print(f"Loaded {len(existing_extractions)} existing extractions.")
            except Exception as e:
                print(f"Failed to load existing extractions: {e}")
        
        all_extractions: List[PaperExtraction] = existing_extractions.copy()
        subgraphs: List[KnowledgeGraph] = []
        
        # Incremental: Load existing knowledge graph if any
        kg_path = os.path.join(self.output_dir, "knowledge_graph.json")
        if os.path.exists(kg_path):
            existing_graph = self.graph_storage.load(kg_path)
            subgraphs.append(existing_graph)
            # Sync normalizer memory to maintain deduplication consistency across increments
            self.graph_builder.normalizer.sync_from_graph(existing_graph)
            print(f"Loaded existing knowledge graph with {len(existing_graph.nodes)} nodes and {len(existing_graph.edges)} edges.")
        
        new_papers_processed = 0
        
        # 1. Ingestion & Extraction Loop (Only for new papers)
        for path in file_paths:
            paper_id = os.path.basename(path)
            if paper_id in existing_paper_ids:
                print(f"Skipping already processed paper: {paper_id}")
                continue
                
            print(f"Processing new paper: {path}")
            try:
                # Ingest
                # TODO: Extract real metadata from file
                metadata = PaperMetadata(
                    paper_id=paper_id,
                    title=paper_id,
                    url=f"file://{os.path.abspath(path)}"
                )
                chunks = self.ingestor.ingest_file(path, metadata)
                
                # Extract
                extraction = self.extractor.extract(metadata.paper_id, chunks)
                all_extractions.append(extraction)
                
                # Build Subgraph
                subgraph = self.graph_builder.build_subgraph(extraction)
                subgraphs.append(subgraph)
                new_papers_processed += 1
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        if not all_extractions:
            print("No papers successfully processed.")
            return {}

        if new_papers_processed > 0:
            print(f"Successfully processed {new_papers_processed} new papers.")
        else:
            print("No new papers to process. Using existing graph and extractions.")

        # 2. KG Construction (Incremental Merge)
        print("Building/Updating Knowledge Graph...")
        unified_graph = self.graph_storage.merge(subgraphs)
        self.graph_storage.save_graph(unified_graph, kg_path)
        
        # 3. Comparative Reasoning (Always re-run across all extractions)
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
        with open(os.path.join(self.output_dir, "insights.json"), 'w', encoding='utf-8') as f:
            json.dump(results["insights"], f, indent=2, ensure_ascii=False)
            
        with open(os.path.join(self.output_dir, "extractions.json"), 'w', encoding='utf-8') as f:
            json.dump(results["extractions"], f, indent=2, ensure_ascii=False)
            
        print(f"Pipeline completed. Results saved to {self.output_dir}")
        return results
