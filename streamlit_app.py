import os
import tempfile
from typing import List, Optional
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from backend import LegalRewriter

load_dotenv()

# Page config
st.set_page_config(page_title="Legal Reformatting System", page_icon="⚖️", layout="wide")

# Initialize the rewriter (RAG-based, no PDF path needed)
@st.cache_resource
def get_rewriter():
    return LegalRewriter()

rewriter = get_rewriter()

# Helper functions for ingestion
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        text.append(page.get_text("text"))
    doc.close()
    return "\n".join(text)

def embed_texts(client: OpenAI, model: str, texts: List[str], progress_callback=None) -> List[List[float]]:
    embeddings = []
    total = len(texts)
    for idx, t in enumerate(texts):
        resp = client.embeddings.create(model=model, input=t)
        embeddings.append(resp.data[0].embedding)
        if progress_callback:
            progress_callback(idx + 1, total)
    return embeddings

def ensure_collection(qdrant: QdrantClient, name: str, vector_size: int = 1536, distance=qmodels.Distance.COSINE):
    try:
        qdrant.delete_collection(collection_name=name)
    except Exception:
        pass
    qdrant.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
    )

def upsert_chunks(qdrant: QdrantClient, name: str, chunks: List[str], vectors: List[List[float]], source_file: str, progress_callback=None):
    points = []
    total = len(chunks)
    batch_size = 100
    
    for idx, (vec, text) in enumerate(zip(vectors, chunks)):
        points.append(
            qmodels.PointStruct(
                id=idx,
                vector=vec,
                payload={"text": text, "chunk_id": idx, "source_file": source_file},
            )
        )
        
        if len(points) >= batch_size or idx == total - 1:
            qdrant.upsert(collection_name=name, points=points)
            points = []
            if progress_callback:
                progress_callback(idx + 1, total)

def get_collection_info(qdrant: QdrantClient, collection_name: str) -> Optional[dict]:
    try:
        info = qdrant.get_collection(collection_name=collection_name)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "optimizer_status": info.optimizer_status,
        }
    except Exception:
        return None

def get_qdrant_client():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url and qdrant_api_key:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return None

# Main UI
st.title("⚖️ Legal Text Reformatting System")
st.markdown("AI-powered legal text reformatting with RAG (Retrieval-Augmented Generation)")

# Create tabs
tab1, tab2, tab3 = st.tabs(["✍️ Reformat", "📤 Ingest Documents", "🗑️ Manage Vectors"])

# TAB 1: REFORMAT
with tab1:
    st.header("✍️ Reformat Legal Text")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_text = st.text_area(
            "Input Text",
            placeholder="Enter paragraph to reformat...",
            height=250,
            key="reformat_input"
        )
        
        # System prompt editor
        with st.expander("⚙️ Edit System Prompt (Optional)", expanded=False):
            default_ui_prompt = rewriter.get_ui_prompt()
            ui_prompt = st.text_area(
                "System Prompt",
                value=default_ui_prompt,
                height=200,
                help="Customize instructions for the AI. Legal vocabulary will be retrieved via RAG.",
                key="system_prompt"
            )
        
        if st.button("🚀 Reformat Text", type="primary", use_container_width=True):
            if not input_text.strip():
                st.warning("⚠️ Please enter some text to reformat.")
            else:
                # Update prompt if changed
                if ui_prompt and ui_prompt.strip() and ui_prompt != default_ui_prompt:
                    rewriter.set_ui_prompt(ui_prompt)
                
                st.markdown("### 📝 Reformatted Output")
                output_placeholder = st.empty()
                full_response = ""
                
                try:
                    for token in rewriter.rewrite_stream(input_text):
                        full_response += token
                        output_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Make sure you've ingested legal vocabulary in the 'Ingest Documents' tab")
    
    with col2:
        st.markdown("### ℹ️ How It Works")
        st.info("""
        **RAG-Powered Reformatting:**
        
        1. Your paragraph is analyzed
        2. Relevant legal terms are retrieved from Qdrant
        3. GPT-5 reformats using those terms
        4. Output in clean Markdown format
        
        **Features:**
        - Formal legal language
        - Preserves original meaning
        - Citation-ready formatting
        - Fast & cost-efficient
        """)
        
        # Show collection status
        st.markdown("### 📊 Vector Database")
        qdrant = get_qdrant_client()
        if qdrant:
            collection_name = os.getenv("QDRANT_COLLECTION", "legal_vocab")
            col_info = get_collection_info(qdrant, collection_name)
            if col_info:
                st.success(f"✅ Connected")
                st.metric("Vectors Available", col_info.get("points_count", 0))
            else:
                st.warning("⚠️ No vectors found. Ingest documents first!")
        else:
            st.error("❌ Qdrant not configured")

# TAB 2: INGEST DOCUMENTS
with tab2:
    st.header("📤 Ingest Legal Documents")
    st.markdown("Upload PDF files to add legal vocabulary to your vector database")
    
    # Config sidebar in expander
    with st.expander("⚙️ Configuration", expanded=False):
        collection_name = st.text_input(
            "Collection Name",
            value=os.getenv("QDRANT_COLLECTION", "legal_vocab"),
            help="Collection name in your Qdrant cluster"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large"],
            index=0
        )
        
        chunk_size = st.slider("Chunk Size", 400, 1200, 800, 50)
        overlap = st.slider("Chunk Overlap", 50, 200, 100, 10)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a legal vocabulary PDF",
            type=["pdf"],
            help="Upload PDF containing legal terminology and vocabulary"
        )
        
        if uploaded_file:
            file_size = len(uploaded_file.getvalue()) / 1024
            st.info(f"📁 **{uploaded_file.name}** ({file_size:.2f} KB)")
            
            if st.button("🚀 Start Ingestion", type="primary", use_container_width=True):
                qdrant = get_qdrant_client()
                if not qdrant:
                    st.error("❌ Qdrant not configured! Check your .env file")
                else:
                    progress_container = st.container()
                    metrics_container = st.container()
                    
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Extract
                        with progress_container:
                            with st.spinner("📖 Extracting text..."):
                                text = extract_text_from_pdf(tmp_path)
                                st.success(f"✅ Extracted {len(text):,} characters")
                        
                        # Chunk
                        with progress_container:
                            with st.spinner("✂️ Chunking..."):
                                chunks = chunk_text(text, chunk_size, overlap)
                                st.success(f"✅ Created {len(chunks):,} chunks")
                        
                        # Embed
                        with progress_container:
                            st.markdown("🧮 Generating embeddings...")
                            embed_progress = st.progress(0)
                            embed_status = st.empty()
                            
                            def embed_callback(current, total):
                                embed_progress.progress(current / total)
                                embed_status.text(f"Embedding {current}/{total}")
                            
                            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                            vectors = embed_texts(openai_client, embedding_model, chunks, embed_callback)
                            st.success(f"✅ Generated {len(vectors):,} embeddings")
                        
                        # Create collection
                        with progress_container:
                            with st.spinner("🗄️ Preparing collection..."):
                                vec_size = len(vectors[0]) if vectors else 1536
                                ensure_collection(qdrant, collection_name, vector_size=vec_size)
                                st.success(f"✅ Collection ready")
                        
                        # Upload
                        with progress_container:
                            st.markdown("💾 Uploading to Qdrant...")
                            upsert_progress = st.progress(0)
                            upsert_status = st.empty()
                            
                            def upsert_callback(current, total):
                                upsert_progress.progress(current / total)
                                upsert_status.text(f"Uploading {current}/{total}")
                            
                            upsert_chunks(qdrant, collection_name, chunks, vectors, uploaded_file.name, upsert_callback)
                            st.success(f"✅ Uploaded {len(chunks):,} vectors")
                        
                        # Metrics
                        with metrics_container:
                            st.markdown("---")
                            st.markdown("### 📊 Ingestion Complete")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Characters", f"{len(text):,}")
                            m2.metric("Chunks", f"{len(chunks):,}")
                            m3.metric("Vectors", f"{len(vectors):,}")
                            m4.metric("Dimensions", vec_size)
                            
                            st.balloons()
                        
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        if 'vectors' in locals():
                            st.info("💡 Embeddings were generated (already paid for)")
    
    with col2:
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Upload PDF** with legal vocabulary
        2. **Configure** chunk size (optional)
        3. **Click Ingest** to process
        4. **Wait** for completion (5-10 min for large files)
        
        ---
        
        ### 💰 Cost Estimate
        - Small PDF (50 pages): ~$0.10
        - Medium PDF (200 pages): ~$0.50
        - Large PDF (500+ pages): ~$1-2
        
        Embeddings are one-time cost!
        """)

# TAB 3: MANAGE VECTORS  
with tab3:
    st.header("🗑️ Manage Vector Database")
    st.markdown("Delete vectors to free up space or refresh your legal vocabulary")
    
    qdrant = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION", "legal_vocab")
    
    if not qdrant:
        st.error("❌ Qdrant not configured")
    else:
        col_info = get_collection_info(qdrant, collection_name)
        
        if col_info:
            st.success(f"✅ Connected to collection: `{collection_name}`")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Vectors", col_info.get("points_count", 0))
            col2.metric("Status", col_info.get("status", "unknown"))
            col3.metric("Indexed", col_info.get("vectors_count", 0))
            
            st.markdown("---")
            
            # Delete all vectors
            st.subheader("🗑️ Delete All Vectors")
            st.warning("⚠️ This will permanently delete ALL vectors in the collection. You'll need to re-ingest documents.")
            
            if st.button("💣 Delete Entire Collection", type="secondary"):
                confirm = st.checkbox("I understand this will delete all data", key="confirm_delete_all")
                if confirm:
                    try:
                        qdrant.delete_collection(collection_name=collection_name)
                        st.success(f"✅ Collection `{collection_name}` deleted!")
                        st.info("💡 The collection will be recreated automatically when you ingest new documents")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            st.markdown("---")
            
            # Delete by file
            st.subheader("📄 Delete Vectors by Source File")
            st.info("Delete all vectors from a specific ingested PDF file")
            
            # Get unique source files
            try:
                # Scroll through all points to get unique source files
                all_files = set()
                scroll_result = qdrant.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    with_payload=True
                )
                
                for point in scroll_result[0]:
                    payload = getattr(point, "payload", {})
                    source_file = payload.get("source_file")
                    if source_file:
                        all_files.add(source_file)
                
                if all_files:
                    file_to_delete = st.selectbox(
                        "Select file to delete",
                        sorted(list(all_files)),
                        help="All vectors from this file will be deleted"
                    )
                    
                    if st.button(f"🗑️ Delete vectors from '{file_to_delete}'", type="secondary"):
                        try:
                            qdrant.delete(
                                collection_name=collection_name,
                                points_selector=qmodels.FilterSelector(
                                    filter=qmodels.Filter(
                                        must=[
                                            qmodels.FieldCondition(
                                                key="source_file",
                                                match=qmodels.MatchValue(value=file_to_delete),
                                            )
                                        ]
                                    )
                                )
                            )
                            st.success(f"✅ Deleted all vectors from '{file_to_delete}'")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.info("ℹ️ No source file metadata found. Vectors were likely ingested without file tracking.")
                    st.caption("Use 'Delete All' to clear the collection and re-ingest with file tracking.")
            
            except Exception as e:
                st.warning(f"⚠️ Could not retrieve file list: {str(e)}")
        
        else:
            st.info(f"ℹ️ Collection `{collection_name}` doesn't exist yet. Ingest documents first!")