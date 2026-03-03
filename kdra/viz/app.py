import streamlit as st
import os
import json
import pandas as pd
import graphviz
import sys
import base64
from kdra.core.retrieval.external import ExternalRetriever

# Ensure we can import the kdra package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from kdra.pipeline import KDRAPipeline

def main():
    st.set_page_config(page_title="KDRA Research Assistant", layout="wide")
    st.title("KDRA: Knowledge-Driven Research Assistant")

    # --- Sidebar: Configuration ---
    st.sidebar.header("Data Source")
    data_dir = st.sidebar.text_input("Input Directory", value="./data")
    output_dir = st.sidebar.text_input("Output Directory", value="./output")
    
    # Mode Selection
    mode = st.sidebar.radio("Mode", ["Real (LLM)", "Dummy (Offline)"], index=0)
    use_dummy = (mode == "Dummy (Offline)")
    
    llm_config = {}
    if not use_dummy:
        st.sidebar.subheader("LLM Configuration")
        with st.sidebar.expander("Configure LLM", expanded=True):
            api_key = st.text_input("API Key", type="password", help="e.g., sk-...")
            base_url = st.text_input("Base URL", value="https://api.openai.com/v1", help="e.g., https://api.deepseek.com/v1")
            model_name = st.text_input("Model Name", value="gpt-4o", help="e.g., gpt-4o, deepseek-chat")
            
            if api_key:
                llm_config = {
                    "api_key": api_key,
                    "base_url": base_url,
                    "model_name": model_name
                }
            else:
                st.warning("Please enter an API Key to use Real Mode.")
    
    # File Upload
    uploaded_files = st.sidebar.file_uploader("Upload Papers", type=['pdf', 'txt', 'md'], accept_multiple_files=True)
    if uploaded_files:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(data_dir, uploaded_file.name)
            # Only save if it doesn't exist or if we want to overwrite. 
            # For simplicity, we overwrite to ensure latest version.
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.sidebar.success(f"Uploaded {len(uploaded_files)} files.")

    # File Selection
    selected_files = []
    if os.path.exists(data_dir):
        all_files = [f for f in os.listdir(data_dir) if f.endswith(('.txt', '.md', '.pdf'))]
        if all_files:
            selected_filenames = st.sidebar.multiselect(
                "Select files to process", 
                all_files, 
                default=all_files
            )
            selected_files = [os.path.join(data_dir, f) for f in selected_filenames]
        else:
            st.sidebar.warning("No compatible files found.")
    
    process_btn = st.sidebar.button("Process Selected Papers")

    # --- Pipeline Execution ---
    if process_btn:
        if not selected_files:
            st.error("Please select at least one file to process.")
        else:
            with st.spinner(f"Ingesting and Building Knowledge Graph from {len(selected_files)} papers..."):
                # Initialize pipeline
                pipeline = KDRAPipeline(output_dir=output_dir, use_dummy=use_dummy, llm_config=llm_config)
                # Run processing (No topic needed)
                full_results = pipeline.process_papers(selected_files)
                
                # --- FILTERING RESULTS FOR CURRENT SESSION ---
                # The user wants current session UI to only focus on the selected papers, 
                # hiding the massive background historical graph.
                selected_paper_ids = [os.path.basename(f) for f in selected_files]
                # In KDRA, the paper node IDs are directly the paper_ids (basenames), without a "Paper:" prefix.
                selected_paper_node_ids = set(selected_paper_ids)
                
                filtered_extractions = [e for e in full_results.get("extractions", []) if e.get("paper_id") in selected_paper_ids]
                
                all_nodes = full_results.get("knowledge_graph", {}).get("nodes", [])
                all_edges = full_results.get("knowledge_graph", {}).get("edges", [])
                
                filtered_edges = []
                # Always start with selected paper nodes in case they have no edges
                connected_node_ids = set()
                
                for edge in all_edges:
                    # Keep edges that are directly connected to our selected papers
                    if edge.get("source") in selected_paper_node_ids or edge.get("target") in selected_paper_node_ids:
                        filtered_edges.append(edge)
                        connected_node_ids.add(edge.get("source"))
                        connected_node_ids.add(edge.get("target"))
                
                # In case some papers had no connections, ensure they are still included
                for pid in selected_paper_node_ids:
                    connected_node_ids.add(pid)
                    
                filtered_nodes = [n for n in all_nodes if n.get("id") in connected_node_ids]
                
                results = {
                    "extractions": filtered_extractions,
                    "errors": full_results.get("errors", []),
                    "knowledge_graph": {
                        "nodes": filtered_nodes,
                        "edges": filtered_edges
                    }
                }
                # ---------------------------------------------
                
                if results.get("errors"):
                    for err in results["errors"]:
                        st.error(err)
                
                if not results.get("extractions"):
                    st.error("Processing failed for all papers. Check logs/errors above.")
                else:
                    # Store in session state
                    st.session_state['results'] = results
                    st.session_state['pipeline'] = pipeline # Store pipeline instance for QA
                    st.session_state['chat_history'] = [] # Reset chat
                    
                    # Stats
                    kg = results.get("knowledge_graph", {})
                    node_count = len(kg.get("nodes", []))
                    edge_count = len(kg.get("edges", []))
                    
                    st.success(f"Processing completed! Built graph with {node_count} nodes and {edge_count} relations. Go to the 'Knowledge Graph' tab to view.")

    # --- Load Results (from session only) ---
    # We strictly require the user to process papers in the current session to ensure context validity.
    results = st.session_state.get('results')
    
    # --- Main Interface ---
    tab_search, tab_chat, tab_kg, tab_data, tab_discover = st.tabs([
        "🔍 Paper Search", 
        "💬 Chat & Q&A", 
        "🕸️ Knowledge Graph", 
        "📄 Extracted Data",
        "💡 Discover & Recommend"
    ])

    with tab_search:
        render_search(data_dir)

    if results:
        # Tab 2: Chat Interface
        with tab_chat:
            st.subheader("Ask questions about your papers")
            
            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # --- Input Area (Fixed at Top) ---
            with st.form(key="qa_form", clear_on_submit=True):
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    user_input = st.text_input("Question", placeholder="Ex: What methods are proposed?", label_visibility="collapsed")
                with col2:
                    submit_btn = st.form_submit_button("Ask")
            
            if submit_btn and user_input:
                # 1. Append User Question
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # 2. Generate Answer
                with st.spinner("Analyzing & Generating Answer..."):
                    pipeline = st.session_state['pipeline']
                    response = pipeline.answer_question(
                        user_input, 
                        results["knowledge_graph"], 
                        results["extractions"]
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            st.divider()
            
            # --- Chat History (Newest First / Reverse Chronological) ---
            # This ensures the input box stays at the top and doesn't move down.
            # We display the newest interaction first, but within interaction, Q is above A.
            
            history = st.session_state.chat_history
            interactions = []
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    interactions.append([history[i], history[i+1]])
                else:
                    interactions.append([history[i]])
            
            for interaction in reversed(interactions):
                for message in interaction:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

        # Tab 3: Knowledge Graph
        with tab_kg:
            render_kg(results)

        # Tab 4: Data View
        with tab_data:
            render_data(results)

        # Tab 5: Discover & Recommend
        with tab_discover:
            render_discover(results)
            
    else:
        with tab_chat:
            st.info("👈 Please select your data directory, choose files, and click 'Process Selected Papers' to begin.")
        with tab_kg:
            st.info("Process papers to view Knowledge Graph.")
        with tab_data:
            st.info("Process papers to view Extracted Data.")
        with tab_discover:
            st.info("Process papers to unlock Discovery & Recommendations.")

def render_kg(results):
    import streamlit.components.v1 as components
    try:
        from pyvis.network import Network
    except ImportError:
        st.error("Pyvis is not installed. Please run `pip install pyvis`.")
        return

    st.subheader("Interactive Knowledge Graph Visualization")
    kg = results.get("knowledge_graph", {})
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])
    
    st.info(f"Generated Interactive Graph with {len(nodes)} nodes and {len(edges)} edges.")
    
    # --- Filter Controls ---
    st.markdown("**Filter Nodes:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    show_paper = col1.checkbox("Papers", value=True)
    show_concept = col2.checkbox("Concepts", value=True)
    show_method = col3.checkbox("Methods", value=True)
    show_dataset = col4.checkbox("Datasets", value=False)
    show_metric = col5.checkbox("Metrics", value=False)
    
    allowed_types = []
    if show_paper: allowed_types.append("Paper")
    if show_concept: allowed_types.append("Concept")
    if show_method: allowed_types.append("Method")
    if show_dataset: allowed_types.append("Dataset")
    if show_metric: allowed_types.append("Metric")

    if not nodes:
        st.warning("Knowledge graph is empty.")
        return

    # 1. Identify visible nodes
    visible_node_ids = set()
    found_types = set()
    for node in nodes:
        raw_type = str(node.get("type", ""))
        clean_type = raw_type.replace("NodeType.", "").title()
        found_types.add(clean_type)
        if clean_type in allowed_types:
            visible_node_ids.add(node["id"])
    
    if not visible_node_ids:
        st.warning(f"No nodes match the current filters. Available types: {found_types}")
        return

    # 2. Build Pyvis Network
    # Initialize PyVis with larger height for better local viewing
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    
    # Add physics for better layout and to avoid overlap
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)
    
    # Add Nodes
    for node in nodes:
        if node["id"] not in visible_node_ids:
            continue

        raw_type = str(node.get("type", ""))
        node_type = raw_type.replace("NodeType.", "").title()

        color = "#D3D3D3" # Lightgrey default
        if node_type == "Paper": color = "#ADD8E6"
        elif node_type == "Method": color = "#90EE90"
        elif node_type == "Dataset": color = "#FFB347"
        elif node_type == "Metric": color = "#FFB6C1"
        elif node_type == "Concept": color = "#FFFFE0"
        
        raw_label = node.get("properties", {}).get("name", node["id"].split(':')[-1])
        # truncate label
        display_label = raw_label[:20] + "..." if len(raw_label) > 20 else raw_label
        
        net.add_node(
            node["id"], 
            label=display_label, 
            title=f"{node_type}: {raw_label}", # Tooltip on hover
            color=color,
            shape="box" if node_type == "Paper" else "ellipse"
        )

    # Add Edges
    for edge in edges:
        if edge["source"] in visible_node_ids and edge["target"] in visible_node_ids:
            rel_label = str(edge.get("relation", "")).replace("RelationType.", "").replace("EdgeType.", "")
            if rel_label == "MENTIONS": rel_label = ""
            
            net.add_edge(
                edge["source"], 
                edge["target"], 
                title=rel_label, 
                label=rel_label if rel_label else None, 
                arrows={'to': {'enabled': True, 'scaleFactor': 0.5}}
            )

    # 3. Generate and Render HTML
    html_file = "kg_visualization.html"
    try:
        net.save_graph(html_file)
        
        # Read the generated HTML file to embed it
        with open(html_file, "r", encoding="utf-8") as f:
            source_code = f.read()
        
        # Modern browsers block opening `data:text/html` in new tabs for security reasons.
        # So we provide a proper download button instead.
        st.download_button(
            label="📺 Download Interactive HTML for Fullscreen Viewing",
            data=source_code,
            file_name="fullscreen_graph.html",
            mime="text/html",
            help="Download and double-click to open in any browser for an immersive, 100% full-screen view without UI constraints.",
            use_container_width=True
        )
        
        # Render in Streamlit with significantly increased height mask
        components.html(source_code, height=820, scrolling=True)
        
    except Exception as e:
        st.error(f"Failed to render interactive graph: {e}")

def render_data(results):
    st.subheader("Extracted Structured Data")
    extractions = results.get("extractions", [])
    
    if extractions:
        df_papers = pd.DataFrame([
            {
                "Paper ID": e["paper_id"],
                "Methods": ", ".join(e.get("methods", [])),
                "Datasets": ", ".join(e.get("datasets", [])),
                "Concepts": ", ".join(e.get("concepts", [])),
                "Claims": len(e.get("claims", []))
            }
            for e in extractions
        ])
        # use_container_width is deprecated in newer Streamlit versions
        try:
            st.dataframe(df_papers, use_container_width=True)
        except TypeError:
             # Fallback for very new versions if they removed it completely
             st.dataframe(df_papers)
        
        st.divider()
        selected_paper = st.selectbox("Select Paper for Details", [e["paper_id"] for e in extractions])
        if selected_paper:
            paper_data = next(p for p in extractions if p["paper_id"] == selected_paper)
            st.json(paper_data)

def render_search(data_dir):
    st.subheader("Search & Retrieve Papers")
    st.markdown("Search for papers on **arXiv** and download them directly to your workspace.")
    
    # Use a form for better alignment and enter-to-submit functionality
    with st.form(key="search_form"):
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            # label_visibility="collapsed" removes the label space, aligning it with the button
            query = st.text_input("Search Query", placeholder="e.g., 'Large Language Models for Knowledge Graphs'", label_visibility="collapsed")
        with col2:
            # use_container_width makes the button fill the column width
            search_btn = st.form_submit_button("🔍 Search", use_container_width=True)
        
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
        
    if search_btn and query:
        with st.spinner("Searching arXiv..."):
            retriever = ExternalRetriever()
            results = retriever.search_arxiv(query, max_results=5)
            st.session_state.search_results = results
            if not results:
                st.warning("No results found.")
                
    if st.session_state.search_results:
        st.divider()
        st.markdown(f"**Found {len(st.session_state.search_results)} results:**")
        for i, paper in enumerate(st.session_state.search_results):
            # Use a container for each result to make it look cleaner
            with st.container():
                col_text, col_action = st.columns([0.8, 0.2])
                with col_text:
                    st.markdown(f"### [{paper['title']}]({paper['pdf_url']})")
                    st.caption(f"**Authors:** {', '.join(paper['authors'])} | **Published:** {paper['published']}")
                    with st.expander("Show Abstract"):
                        st.write(paper['summary'])
                
                with col_action:
                    st.write("") # Spacer
                    st.write("") # Spacer
                    if st.button(f"⬇️ Download", key=f"dl_{i}", use_container_width=True):
                        try:
                            with st.spinner("Downloading..."):
                                retriever = ExternalRetriever()
                                # Download the PDF
                                path = retriever.download_pdf(paper['pdf_url'], data_dir)
                                st.success(f"Saved!")
                        except Exception as e:
                            st.error(f"Failed: {e}")
            st.divider()

def render_discover(results):
    st.subheader("💡 Discovery & Recommendation Engine")
    st.markdown("Based on the generated Knowledge Graph, discover related methodologies and datasets logically connected to your topics of interest.")
    
    kg = results.get("knowledge_graph", {})
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])
    
    if not nodes:
        st.warning("Knowledge graph is empty.")
        return
        
    # Get all concepts
    concepts = [n["properties"].get("name", n["id"]) for n in nodes if n["type"] == "Concept"]
    # Fallback if no Concepts
    if not concepts:
        concepts = [n["properties"].get("name", n["id"]) for n in nodes if n["type"] != "Paper"]
        
    if not concepts:
        st.info("Not enough conceptual data to provide recommendations.")
        return
        
    selected_concept = st.selectbox("Select a Research Concept or Topic:", sorted(list(set(concepts))))
    
    if selected_concept:
        st.divider()
        
        # 1. Find target Node ID(s) based on selection
        target_ids = [n["id"] for n in nodes if n["properties"].get("name") == selected_concept or n["id"] == selected_concept]
                
        # 2. Find papers related to this concept
        related_papers = set()
        for e in edges:
            if e["target"] in target_ids and "RELATED_TO" in e["relation"]:
                related_papers.add(e["source"])
            elif e["source"] in target_ids and "RELATED_TO" in e["relation"]:
                related_papers.add(e["target"])
                
        # Handle nodes connecting implicitly to paper
        for e in edges:
            if e["target"] in target_ids or e["source"] in target_ids:
                other = e["source"] if e["target"] in target_ids else e["target"]
                for n in nodes:
                    if n["id"] == other and n["type"] == "Paper":
                        related_papers.add(other)

        # 3. Aggregate methodologies and datasets
        recommended_methods = {}
        recommended_datasets = {}
        
        for e in edges:
            if e["source"] in related_papers:
                t_id = e["target"]
                t_node = next((n for n in nodes if n["id"] == t_id), None)
                if t_node:
                    name = t_node["properties"].get("name", t_id)
                    if t_node["type"] == "Method" or e["relation"] == "USES":
                        recommended_methods[name] = recommended_methods.get(name, 0) + 1
                    elif t_node["type"] == "Dataset" or e["relation"] == "EVALUATED_ON":
                        recommended_datasets[name] = recommended_datasets.get(name, 0) + 1

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🛠️ Recommended Methods")
            if recommended_methods:
                sorted_methods = sorted(recommended_methods.items(), key=lambda x: x[1], reverse=True)
                for method, count in sorted_methods:
                    st.success(f"**{method}** (Found in {count} related paper(s))")
            else:
                st.info("No related methods found in the graph for this concept.")
                
        with col2:
            st.markdown(f"### 🗄️ Recommended Datasets")
            if recommended_datasets:
                sorted_datasets = sorted(recommended_datasets.items(), key=lambda x: x[1], reverse=True)
                for dataset, count in sorted_datasets:
                    st.info(f"**{dataset}** (Found in {count} related paper(s))")
            else:
                st.info("No related datasets found in the graph for this concept.")
                
        st.divider()
        st.markdown("### 📄 Contextual Source Papers")
        if related_papers:
            for p in related_papers:
                st.markdown(f"- `{p}`")
        else:
            st.info("No direct papers found for this concept.")

if __name__ == "__main__":
    main()
