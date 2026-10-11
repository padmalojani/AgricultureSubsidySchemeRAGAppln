import streamlit as st
import os
import config
from vector_store import load_vector_store, create_vector_store, add_documents_to_store, clear_vector_store, search_similar_documents
from ingestion import process_uploaded_pdf
from rag_chain import get_rag_chain
from prompts import QA_PROMPT
from utils import validate_pdf_file, format_sources
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="TN Agriculture Subsidy Assistant",
    page_icon="🌾",
    layout="wide"
)

# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🌾 Tamil Nadu Agriculture Subsidy Scheme Assistant")
st.markdown("Find official agriculture subsidies, eligibility, and application guidelines from verified government documents.")

# Sidebar for Document Management
with st.sidebar:
    st.header("📂 Administrator / Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload Scheme PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    chunk_size = st.slider("Chunk Size", 500, 2000, config.DEFAULT_CHUNK_SIZE, 100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 400, config.DEFAULT_CHUNK_OVERLAP, 25)
    
    if st.button("Process & Index Documents", type="primary"):
        if uploaded_files:
            all_chunks = []
            with st.spinner("Processing uploaded PDFs..."):
                for file in uploaded_files:
                    is_valid, msg = validate_pdf_file(file)
                    if not is_valid:
                        st.error(f"{file.name}: {msg}")
                        continue
                    try:
                        chunks = process_uploaded_pdf(file, chunk_size, chunk_overlap)
                        all_chunks.extend(chunks)
                        st.success(f"Processed: {file.name} ({len(chunks)} chunks)")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
            
            if all_chunks:
                with st.spinner("Updating vector database..."):
                    if st.session_state.vector_store is None:
                        st.session_state.vector_store = create_vector_store(all_chunks)
                    else:
                        st.session_state.vector_store = add_documents_to_store(st.session_state.vector_store, all_chunks)
                    st.success("Documents indexed and persisted successfully!")
        else:
            st.warning("Please upload at least one PDF file.")
            
    st.divider()
    
    # System stats
    indexed_status = "Loaded & Ready" if st.session_state.vector_store else "No Index Found"
    st.metric("Vector Index Status", indexed_status)
    
    if st.button("⚠️ Clear / Rebuild Index"):
        clear_vector_store()
        st.session_state.vector_store = None
        st.success("Vector store cleared successfully!")
        st.rerun()

# Main Chat Interface
st.subheader("💬 Ask a Question")

suggested_questions = [
    "What agriculture subsidy schemes are available in Tamil Nadu?",
    "Am I eligible for a subsidy for drip irrigation?",
    "What subsidy is available for farm machinery?",
    "What documents are required to apply?",
    "Is there any subsidy for horticulture?"
]

st.markdown("**Suggested Questions:**")
cols = st.columns(2)
selected_query = None
for i, q in enumerate(suggested_questions):
    if cols[i % 2].button(q, key=f"sug_{i}"):
        selected_query = q

# Display Chat History
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat Input
user_query = st.chat_input("Ask about Tamil Nadu agriculture subsidies...")
if selected_query:
    user_query = selected_query

if user_query:
    if not st.session_state.vector_store:
        st.warning("No scheme documents have been indexed yet. Please upload PDFs in the sidebar first.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching official documents and generating answer..."):
                retrieved_docs = search_similar_documents(st.session_state.vector_store, user_query, k=config.DEFAULT_RETRIEVAL_K)
                
                if not retrieved_docs:
                    response_text = "I could not find any relevant information in the available documents to answer your question."
                else:
                    chat_model, format_docs = get_rag_chain()
                    context = format_docs(retrieved_docs)
                    
                    prompt_value = QA_PROMPT.invoke({
                        "context": context,
                        "chat_history": st.session_state.chat_history[:-1],
                        "question": user_query
                    })
                    
                    response = chat_model.invoke(prompt_value)
                    sources_str = format_sources(retrieved_docs)
                    
                    response_text = f"{response.content}\n\n**Sources:**\n{sources_str}"
                
                st.markdown(response_text)
                st.session_state.chat_history.append(AIMessage(content=response_text))