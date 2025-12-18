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
                pipeline = KDRAPipeline(output_dir=output_dir, use_dummy=use_dummy)
                # Run processing (No topic needed)
                results = pipeline.process_papers(selected_files)
                
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
    tab_search, tab_chat, tab_kg, tab_data = st.tabs(["🔍 Paper Search", "💬 Chat & Q&A", "🕸️ Knowledge Graph", "📄 Extracted Data"])
    
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
            
    else:
        with tab_chat:
            st.info("👈 Please select your data directory, choose files, and click 'Process Selected Papers' to begin.")
        with tab_kg:
            st.info("Process papers to view Knowledge Graph.")
        with tab_data:
            st.info("Process papers to view Extracted Data.")

def render_kg(results):
    import shutil
    if not shutil.which("dot"):
        st.error("Graphviz executable 'dot' not found. Please install Graphviz (e.g., `conda install graphviz` or `brew install graphviz`).")
        return

    st.subheader("Knowledge Graph Visualization")
    kg = results.get("knowledge_graph", {})
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])
    
    st.info(f"Debug Info: Found {len(nodes)} nodes and {len(edges)} edges to render.")
    
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

    if nodes:
        try:
            graph = graphviz.Digraph()
            graph.attr(rankdir='LR')
            # Global node attributes for cleaner look
            graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='10')
            
            # Helper to sanitize IDs for Graphviz (replace colons and dots to avoid syntax errors)
            def clean_id(id_str):
                # Replace : . - and space with _ to ensure valid DOT identifiers
                return id_str.replace(":", "__").replace(".", "_").replace("-", "_").replace(" ", "_")

            # 1. Identify visible nodes
            visible_node_ids = set()
            found_types = set()
            for node in nodes:
                # Handle NodeType enum or string
                raw_type = str(node["type"])
                # Normalize: remove NodeType. prefix, and ensure Title Case (Paper, Method, etc.)
                clean_type = raw_type.replace("NodeType.", "")
                # Handle uppercase cases (PAPER -> Paper) if necessary
                if clean_type.isupper():
                    clean_type = clean_type.title()
                
                found_types.add(clean_type)
                
                if clean_type in allowed_types:
                    visible_node_ids.add(node["id"])
            
            # Debug: Show found types if nothing is visible
            if not visible_node_ids and nodes:
                st.warning(f"No nodes visible. Found node types: {found_types}. Allowed: {allowed_types}")

            # 2. Add Nodes
            for node in nodes:
                if node["id"] not in visible_node_ids:
                    continue

                raw_type = str(node["type"])
                node_type = raw_type.replace("NodeType.", "")
                if node_type.isupper(): node_type = node_type.title()

                color = "lightgrey"
                if node_type == "Paper": color = "#ADD8E6"      # Light Blue
                elif node_type == "Method": color = "#90EE90"   # Light Green
                elif node_type == "Dataset": color = "#FFB347"  # Pastel Orange
                elif node_type == "Metric": color = "#FFB6C1"   # Light Pink
                elif node_type == "Concept": color = "#FFFFE0"  # Light Yellow
                
                safe_id = clean_id(node["id"])
                
                # Use name from properties if available, else ID suffix
                raw_label = node.get("properties", {}).get("name", node["id"].split(':')[-1])
                
                # Truncate label for cleaner graph
                if len(raw_label) > 20:
                    display_label = raw_label[:17] + "..."
                else:
                    display_label = raw_label
                
                # Tooltip shows full name and type
                tooltip = f"{node_type}: {raw_label}"
                
                graph.node(safe_id, label=display_label, fillcolor=color, tooltip=tooltip)
            
            # 3. Add Edges
            for edge in edges:
                if edge["source"] in visible_node_ids and edge["target"] in visible_node_ids:
                    # Clean relation label
                    rel_label = str(edge["relation"]).replace("RelationType.", "").replace("EdgeType.", "")
                    # Simplify relation labels
                    if rel_label == "MENTIONS": rel_label = "" # Too common, hide it
                    
                    graph.edge(clean_id(edge["source"]), clean_id(edge["target"]), label=rel_label, fontsize="8", color="gray50")
            
            # --- Rendering ---
            
            # 1. Generate SVG for high-res viewing
            try:
                svg_data = graph.pipe(format='svg')
                b64_svg = base64.b64encode(svg_data).decode('utf-8')
                
                # Link to open SVG in new tab
                href = f'<a href="data:image/svg+xml;base64,{b64_svg}" target="_blank" style="text-decoration:none;">' \
                       f'<button style="background-color:#FF4B4B; color:white; padding:8px 16px; border:none; border-radius:4px; cursor:pointer; margin-bottom:10px;">' \
                       f'🔍 Click to Open Zoomable Graph (New Tab)</button></a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Download button
                st.download_button(
                    label="⬇️ Download Graph (SVG)",
                    data=svg_data,
                    file_name="knowledge_graph.svg",
                    mime="image/svg+xml"
                )
                
            except Exception as e:
                st.warning(f"Could not generate SVG for download: {e}")

            # 2. Render Interactive Chart in App
            st.graphviz_chart(graph, use_container_width=True)
            
            # Debug: Show source
            with st.expander("View Graphviz DOT Source"):
                st.code(graph.source)
            
        except Exception as e:
            st.error(f"Error rendering graph: {e}")
            # Check if it's a path issue
            import os
            st.text(f"PATH: {os.environ.get('PATH')}")
    else:
        st.write("Empty Knowledge Graph.")

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

if __name__ == "__main__":
    main()
