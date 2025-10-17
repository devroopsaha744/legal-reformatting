import os
import tempfile
from typing import List, Optional
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
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
    """Extract all text from a PDF."""
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        text.append(page.get_text("text"))
    doc.close()
    return "\n".join(text)


def embed_texts(client: OpenAI, model: str, texts: List[str], progress_callback=None) -> List[List[float]]:
    """Generate embeddings for a list of texts with progress tracking."""
    embeddings = []
    total = len(texts)
    for idx, t in enumerate(texts):
        resp = client.embeddings.create(model=model, input=t)
        embeddings.append(resp.data[0].embedding)
        if progress_callback:
            progress_callback(idx + 1, total)
    return embeddings


def ensure_collection(qdrant: QdrantClient, name: str, vector_size: int = 1536, distance=qmodels.Distance.COSINE):
    """Create collection if it doesn't exist, or recreate if exists."""
    try:
        qdrant.delete_collection(collection_name=name)
    except Exception:
        pass
    
    qdrant.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
    )


def upsert_chunks(qdrant: QdrantClient, name: str, chunks: List[str], vectors: List[List[float]], progress_callback=None):
    """Upsert chunks with vectors into Qdrant with progress tracking."""
    points = []
    total = len(chunks)
    batch_size = 100
    
    for idx, (vec, text) in enumerate(zip(vectors, chunks)):
        points.append(
            qmodels.PointStruct(
                id=idx,
                vector=vec,
                payload={"text": text, "chunk_id": idx},
            )
        )
        
        # Batch upsert every 100 points
        if len(points) >= batch_size or idx == total - 1:
            qdrant.upsert(collection_name=name, points=points)
            points = []
            if progress_callback:
                progress_callback(idx + 1, total)


def get_collection_info(qdrant: QdrantClient, collection_name: str) -> Optional[dict]:
    """Get collection statistics."""
    try:
        info = qdrant.get_collection(collection_name=collection_name)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "optimizer_status": info.optimizer_status,
        }
    except Exception as e:
        return None


# Streamlit UI
st.set_page_config(page_title="Legal Vocabulary Ingestion", page_icon="📄", layout="wide")

st.title("📄 Legal Vocabulary Ingestion System")
st.markdown("Upload a legal vocabulary PDF to ingest it into your **Qdrant cluster** as a searchable collection for RAG-powered legal reformatting.")

st.info("ℹ️ **Cluster vs Collection**: Your Qdrant URL points to a **cluster** (the server). This app creates a **collection** (database table) within that cluster to store your legal vocabulary vectors.")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Load config from environment
collection_name = st.sidebar.text_input(
    "Collection Name",
    value=os.getenv("QDRANT_COLLECTION", "legal_vocab"),
    help="Name of the collection to create WITHIN your Qdrant cluster"
)

qdrant_url = st.sidebar.text_input(
    "Qdrant Cluster URL",
    value=os.getenv("QDRANT_URL", ""),
    type="password",
    help="Your Qdrant Cloud cluster endpoint URL"
)

qdrant_api_key = st.sidebar.text_input(
    "Qdrant API Key",
    value=os.getenv("QDRANT_API_KEY", ""),
    type="password",
    help="Your Qdrant cluster API key"
)

embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
    index=0,
    help="OpenAI embedding model to use"
)

chunk_size = st.sidebar.slider("Chunk Size (characters)", 400, 1200, 800, 50)
overlap = st.sidebar.slider("Chunk Overlap (characters)", 50, 200, 100, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Cluster & Collection Info")

# Check if Qdrant is configured
if qdrant_url and qdrant_api_key:
    try:
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        col_info = get_collection_info(qdrant, collection_name)
        
        if col_info:
            st.sidebar.success(f"✅ Connected to cluster")
            st.sidebar.info(f"Collection: `{collection_name}`")
            vectors_count = col_info.get('vectors_count', 0)
            points_count = col_info.get('points_count', 0)
            status = col_info.get('status', 'unknown')
            st.sidebar.metric("Vectors", vectors_count)
            st.sidebar.metric("Points", points_count)
            st.sidebar.caption(f"Status: {status}")
        else:
            st.sidebar.warning(f"⚠️ Collection `{collection_name}` not found in cluster")
            st.sidebar.info("Will be created during ingestion")
    except Exception as e:
        st.sidebar.error(f"❌ Cannot connect to Qdrant cluster")
        st.sidebar.error(f"Error: {str(e)}")
        st.sidebar.warning("Check your QDRANT_URL and QDRANT_API_KEY")
else:
    st.sidebar.warning("⚠️ Qdrant cluster credentials not configured")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a legal vocabulary PDF file",
        type=["pdf"],
        help="Upload the PDF containing legal vocabulary and terminology"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        st.info(f"📁 **File:** {uploaded_file.name} | **Size:** {file_size:.2f} KB")
        
        # Ingest button
        if st.button("🚀 Start Ingestion", type="primary", use_container_width=True):
            if not qdrant_url or not qdrant_api_key:
                st.error("❌ Please configure Qdrant cluster credentials in the sidebar!")
            else:
                # Create progress containers
                status_container = st.container()
                progress_container = st.container()
                metrics_container = st.container()
                
                try:
                    with status_container:
                        st.markdown("### 🔄 Ingestion Progress")
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Step 1: Extract text
                    with progress_container:
                        with st.spinner("📖 Extracting text from PDF..."):
                            text = extract_text_from_pdf(tmp_path)
                            st.success(f"✅ Extracted {len(text):,} characters")
                    
                    # Step 2: Chunk text
                    with progress_container:
                        with st.spinner("✂️ Chunking text..."):
                            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                            st.success(f"✅ Created {len(chunks):,} chunks")
                    
                    # Step 3: Generate embeddings
                    with progress_container:
                        st.markdown("🧮 Generating embeddings...")
                        embed_progress = st.progress(0)
                        embed_status = st.empty()
                        
                        def embed_callback(current, total):
                            progress = current / total
                            embed_progress.progress(progress)
                            embed_status.text(f"Embedding chunk {current}/{total}")
                        
                        try:
                            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                            vectors = embed_texts(openai_client, embedding_model, chunks, embed_callback)
                            st.success(f"✅ Generated {len(vectors):,} embeddings")
                        except Exception as e:
                            st.error(f"❌ Error generating embeddings: {str(e)}")
                            st.error("Check your OPENAI_API_KEY and API quota")
                            raise
                    
                    # Step 4: Create/update collection
                    with progress_container:
                        with st.spinner("🗄️ Creating collection in Qdrant cluster..."):
                            try:
                                vec_size = len(vectors[0]) if vectors else 1536
                                qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                                ensure_collection(qdrant, collection_name, vector_size=vec_size)
                                st.success(f"✅ Collection `{collection_name}` ready in cluster")
                            except Exception as e:
                                st.error(f"❌ Error creating collection: {str(e)}")
                                st.warning("Your embeddings are safe. Check Qdrant credentials and try again.")
                                raise
                    
                    # Step 5: Upsert vectors
                    with progress_container:
                        st.markdown("💾 Uploading vectors to Qdrant cluster...")
                        upsert_progress = st.progress(0)
                        upsert_status = st.empty()
                        
                        def upsert_callback(current, total):
                            progress = current / total
                            upsert_progress.progress(progress)
                            upsert_status.text(f"Uploading chunk {current}/{total}")
                        
                        try:
                            upsert_chunks(qdrant, collection_name, chunks, vectors, upsert_callback)
                            st.success(f"✅ Uploaded {len(chunks):,} vectors to Qdrant")
                        except Exception as e:
                            st.error(f"❌ Error uploading to Qdrant: {str(e)}")
                            st.warning("Some vectors may have been uploaded. Check your Qdrant cluster.")
                            raise
                    
                    # Display final metrics
                    with metrics_container:
                        st.markdown("---")
                        st.markdown("### 📊 Ingestion Summary")
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric("Characters Extracted", f"{len(text):,}")
                        
                        with metric_col2:
                            st.metric("Chunks Created", f"{len(chunks):,}")
                        
                        with metric_col3:
                            st.metric("Vectors Generated", f"{len(vectors):,}")
                        
                        with metric_col4:
                            st.metric("Vector Dimensions", vec_size)
                        
                        # Get updated collection info
                        col_info = get_collection_info(qdrant, collection_name)
                        if col_info:
                            vectors_count = col_info.get('vectors_count', 0)
                            status = col_info.get('status', 'unknown')
                            st.success(f"""
                            ✅ **Ingestion Complete!**
                            
                            - Collection: `{collection_name}`
                            - Total Vectors: {vectors_count:,}
                            - Status: {status}
                            
                            Your legal vocabulary is now ready for RAG-powered reformatting!
                            """)
                        else:
                            st.success(f"""
                            ✅ **Ingestion Complete!**
                            
                            - Collection: `{collection_name}`
                            - Vectors uploaded successfully
                            
                            Your legal vocabulary is now ready for RAG-powered reformatting!
                            """)
                    
                    # Cleanup
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass  # Ignore cleanup errors
                    
                    # Auto-refresh sidebar
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ **Error during ingestion**")
                    st.error(f"**Error details:** {str(e)}")
                    
                    # Show what was completed
                    st.warning("### ⚠️ Progress so far:")
                    if 'text' in locals():
                        st.info(f"✅ Text extraction: {len(text):,} characters")
                    if 'chunks' in locals():
                        st.info(f"✅ Chunking: {len(chunks):,} chunks created")
                    if 'vectors' in locals():
                        st.info(f"✅ Embeddings: {len(vectors):,} vectors generated (SAVED)")
                        st.success("💡 Your embeddings are safe! The error happened after embedding.")
                    
                    st.markdown("---")
                    st.markdown("### 🔧 What to do:")
                    st.markdown("""
                    1. **Check the error message above**
                    2. **If embeddings were generated**, they're already paid for - don't regenerate!
                    3. **If Qdrant error**, verify your cluster URL and API key
                    4. **If OpenAI error**, check your API key and quota
                    5. **Try again** - if embeddings were done, they won't be regenerated
                    """)
                    
                    import traceback
                    with st.expander("🐛 Full error trace (for debugging)"):
                        st.code(traceback.format_exc())

with col2:
    st.header("ℹ️ How It Works")
    
    st.markdown("""
    ### 🏢 Qdrant Architecture
    
    ```
    Qdrant Cloud Cluster (Server)
    └── Collection: legal_vocab
        ├── Vector 1 + text chunk
        ├── Vector 2 + text chunk
        ├── Vector 3 + text chunk
        └── ... (N vectors)
    ```
    
    **Your Setup:**
    - **Cluster URL**: Your server endpoint
    - **Collection**: Database table within cluster
    - **Vectors**: Embedded text chunks stored in collection
    
    ---
    
    ### Ingestion Pipeline
    
    1. **📤 Upload PDF**
       - Select your legal vocabulary document
    
    2. **📖 Text Extraction**
       - Extracts all text from PDF pages
    
    3. **✂️ Chunking**
       - Splits text into overlapping chunks
       - Default: 800 chars with 100 char overlap
    
    4. **🧮 Embedding**
       - Converts chunks to vectors
       - Uses OpenAI embeddings
    
    5. **💾 Storage**
       - Stores vectors in Qdrant
       - Enables semantic search
    
    ### RAG Benefits
    
    ✅ **Semantic Search:** Find relevant legal terms  
    ✅ **Scalable:** Handle any PDF size  
    ✅ **Fast:** Retrieve only what's needed  
    ✅ **Cost-Efficient:** Reduce token usage  
    
    ---
    
    ### Next Steps
    
    After ingestion, use the main reformatting app to:
    - Input paragraphs to reformat
    - Get legal terminology suggestions
    - Receive formatted output
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Troubleshooting")
    
    with st.expander("Cannot connect to Qdrant cluster"):
        st.markdown("""
        - Verify `QDRANT_URL` is your **cluster endpoint** (not collection name)
        - Check `QDRANT_API_KEY` is valid for that cluster
        - Ensure Qdrant cluster is active in your cloud dashboard
        - Check network/firewall settings
        - Test in browser: paste your cluster URL
        
        **Remember**: 
        - Cluster = The server (URL)
        - Collection = Created within the cluster (by this app)
        """)
    
    with st.expander("OpenAI API errors"):
        st.markdown("""
        - Verify `OPENAI_API_KEY` in `.env`
        - Check API quota/billing
        - Ensure key has embedding permissions
        """)
    
    with st.expander("Large PDF processing"):
        st.markdown("""
        - Ingestion may take 5-10 minutes
        - Progress bars show real-time status
        - Embeddings are batched (100 per call)
        - Don't close the browser during ingestion
        """)
