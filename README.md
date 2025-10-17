# Legal Text Reformatting System with RAG

A powerful AI-powered legal text reformatting tool using Retrieval-Augmented Generation (RAG) with Qdrant vector database and GPT-5.

## 🎯 Features

- **RAG-Powered Reformatting**: Semantic search retrieves relevant legal vocabulary chunks
- **Separate Ingestion UI**: Easy PDF upload and automatic ingestion with real-time metrics
- **Streamlit Interface**: User-friendly web applications for both ingestion and reformatting
- **GPT-5 Integration**: Uses `gpt-5-2025-08-07` model snapshot
- **Markdown Output**: Clean, formatted legal text with headings, lists, and bold terms
- **Scalable**: Handles PDFs of any size through vector database

## 📁 Project Structure

```
legal-reformat/
├── .env                    # Environment configuration (API keys, Qdrant config)
├── requirements.txt        # Python dependencies
├── backend.py             # Core LegalRewriter class with RAG logic
├── ingest.py              # CLI ingestion script (optional)
├── ingest_app.py          # 📤 Streamlit UI for PDF ingestion
├── streamlit_app.py       # ✍️ Streamlit UI for text reformatting
└── legal word ver 2.pdf   # Legal vocabulary source document
```

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Create virtual environment (optional but recommended)
python -m venv venv
.\\venv\\Scripts\\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` file and add your API keys:

```env
# OpenAI API Key (required for embeddings and GPT-5)
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Cloud Configuration (required for RAG)
# QDRANT_URL = Your Qdrant CLUSTER endpoint (not collection!)
# Example: https://abc-123.eu-west-2-0.aws.cloud.qdrant.io
QDRANT_URL=https://your-cluster-id.region.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_cluster_api_key_here

# QDRANT_COLLECTION = Name of the collection to create WITHIN your cluster
# Collections are like database tables within the cluster
QDRANT_COLLECTION=legal_vocab

# OpenAI Embedding Model
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

**Key Distinction:**
- **Cluster** = Your Qdrant Cloud server (the URL endpoint)
- **Collection** = A database table within that cluster (stores your vectors)
- You have ONE cluster, but can have MULTIPLE collections within it

### 3. Ingest Legal Vocabulary

**Option A: Using Streamlit UI (Recommended)** 📤

```powershell
streamlit run ingest_app.py
```

Then:
1. Upload your legal vocabulary PDF
2. Configure chunk size and overlap (optional)
3. Click "Start Ingestion"
4. Watch real-time progress and metrics

**Option B: Using CLI** 💻

```powershell
python ingest.py "legal word ver 2.pdf" --collection legal_vocab
```

### 4. Run Reformatting App

```powershell
streamlit run streamlit_app.py
```

Then:
1. Enter or paste text to reformat
2. Edit system prompt if needed
3. Click "Reformat" to see streaming output

## 📱 Applications

### 🔹 Ingestion App (`ingest_app.py`)

**Purpose**: Upload and ingest legal vocabulary PDFs into Qdrant

**Features**:
- ✅ PDF file upload interface
- ✅ Real-time ingestion progress bars
- ✅ Metrics display (chunks, vectors, dimensions)
- ✅ Collection info sidebar
- ✅ Configurable chunking parameters
- ✅ Error handling and troubleshooting tips

**When to use**: 
- First-time setup
- Adding new legal vocabulary documents
- Updating existing vocabulary

### 🔹 Reformatting App (`streamlit_app.py`)

**Purpose**: Reformat paragraphs using legal vocabulary from Qdrant

**Features**:
- ✅ Single editable system prompt
- ✅ Real-time streaming output
- ✅ Markdown-formatted results
- ✅ Clean, professional legal language
- ✅ RAG-powered semantic retrieval

**When to use**:
- Daily legal text reformatting
- Converting informal text to legal language
- Enhancing legal document clarity

## 🏗️ Architecture

### RAG Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION (One-time)                      │
└─────────────────────────────────────────────────────────────┘

PDF → Extract Text → Chunk (800 chars) → Embed → Store in Qdrant


┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL (Per query)                     │
└─────────────────────────────────────────────────────────────┘

User Input → Embed → Search Qdrant → Top 6 Chunks → GPT-5 → Output
```

### Component Breakdown

1. **Frontend** (`ingest_app.py` + `streamlit_app.py`)
   - Streamlit-based UI
   - File upload and progress tracking
   - Real-time streaming display

2. **Backend** (`backend.py`)
   - `LegalRewriter` class
   - RAG logic (`_retrieve_context`)
   - OpenAI embeddings (`_embed`)
   - Qdrant integration (`_get_qdrant`)

3. **Vector Database** (Qdrant)
   - Stores legal vocabulary chunks
   - Enables semantic search
   - Scalable to unlimited documents

## ⚙️ Configuration

### Chunking Parameters

Adjust in `ingest_app.py` sidebar or CLI:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 800 | Characters per chunk |
| `overlap` | 100 | Overlapping characters between chunks |

**Recommendations**:
- **Smaller chunks (400-600)**: Better precision, more vectors
- **Larger chunks (1000-1200)**: More context, fewer vectors
- **Overlap (50-150)**: Prevents context loss at boundaries

### Embedding Models

Supported OpenAI models:

| Model | Dimensions | Cost | Use Case |
|-------|-----------|------|----------|
| `text-embedding-3-small` | 1536 | Low | Default, balanced |
| `text-embedding-3-large` | 3072 | Medium | Higher accuracy |
| `text-embedding-ada-002` | 1536 | Low | Legacy, compatible |

### Retrieval Parameters

In `backend.py`, adjust `_retrieve_context`:

```python
def _retrieve_context(self, query: str, top_k: int = 6) -> str:
```

- **`top_k`**: Number of chunks to retrieve (default: 6)
  - Lower (3-4): Faster, more focused
  - Higher (8-10): More comprehensive

## 📊 Metrics & Monitoring

### Ingestion Metrics

Displayed in `ingest_app.py`:

- **Characters Extracted**: Total text extracted from PDF
- **Chunks Created**: Number of text segments
- **Vectors Generated**: Number of embeddings created
- **Vector Dimensions**: Embedding size (1536 or 3072)

### Collection Info

Sidebar displays:
- **Vectors Count**: Total vectors in collection
- **Points Count**: Total data points stored
- **Status**: Collection health status

## 🔧 Troubleshooting

### Cannot Connect to Qdrant

```
❌ Error: "Qdrant cluster not configured"
```

**Solutions**:
1. Verify `QDRANT_URL` in `.env` points to your **cluster endpoint**
2. Check `QDRANT_API_KEY` is valid for that cluster
3. Test connection in browser: `https://your-cluster-url`
4. Ensure Qdrant cluster is active in your cloud dashboard

**Important**: 
- `QDRANT_URL` = Your Qdrant Cloud **cluster** endpoint (the server)
- `QDRANT_COLLECTION` = The **collection** name (created within the cluster)
- Collections are like database tables within your cluster

### OpenAI API Errors

```
❌ Error: "Invalid API key"
```

**Solutions**:
1. Check `OPENAI_API_KEY` in `.env`
2. Verify API quota and billing
3. Ensure key has embedding permissions
4. Test with: `openai api keys.list`

### Large PDF Processing

```
⚠️ Ingestion taking too long
```

**Expected Times**:
- **Small PDF (10-50 pages)**: 1-3 minutes
- **Medium PDF (50-200 pages)**: 5-10 minutes
- **Large PDF (200+ pages)**: 10-20 minutes

**Tips**:
- Don't close browser during ingestion
- Watch progress bars for status
- Embeddings are batched automatically
- First ingestion is one-time only

## 💡 Usage Tips

### Best Practices

1. **Ingest Once**: Run ingestion only when vocabulary changes
2. **Monitor Metrics**: Check collection info before reformatting
3. **Test Chunks**: Try different chunk sizes for your document type
4. **Adjust Top-K**: Increase if results lack terminology, decrease for speed

### System Prompt Editing

The default prompt enforces:
- ✅ Formal legal tone
- ✅ Markdown formatting
- ✅ Preservation of meaning
- ✅ No vocabulary disclosure

**Customize** via Streamlit UI to:
- Add domain-specific rules
- Change formatting preferences
- Adjust formality level

### Cost Optimization

**Ingestion Costs** (one-time):
- 200-page PDF ≈ 500 chunks ≈ $0.50 for embeddings

**Query Costs** (per reformat):
- RAG approach: ~5K tokens ≈ $0.00015/request
- Full PDF approach: ~500K tokens ≈ $15/request
- **Savings: 99.99%**

## 🔐 Security

- ✅ API keys stored in `.env` (gitignored)
- ✅ Qdrant credentials in password fields
- ✅ No API keys in code
- ✅ Environment variable isolation

**Recommendations**:
- Never commit `.env` to git
- Rotate API keys regularly
- Use separate keys for dev/prod
- Enable Qdrant authentication

## 📦 Dependencies

Key packages (see `requirements.txt`):

```
streamlit          # Web UI framework
openai             # GPT-5 and embeddings
qdrant-client      # Vector database
PyMuPDF (fitz)     # PDF text extraction
python-dotenv      # Environment management
```

## 🎓 Learn More

### RAG Concepts

- **Embeddings**: Vector representations of text capturing semantic meaning
- **Vector Database**: Stores and searches embeddings efficiently
- **Semantic Search**: Finds similar content by meaning, not keywords
- **Chunking**: Splits documents into manageable pieces

### Why RAG?

- ✅ Handles unlimited document sizes
- ✅ Retrieves only relevant context
- ✅ 99% cost reduction vs full-context
- ✅ Faster response times
- ✅ Scalable to multiple documents

## 🚧 Roadmap

Potential enhancements:

- [ ] Multi-document collections
- [ ] Collection management UI
- [ ] Chunk preview in reformatting UI
- [ ] Adjustable top-k slider
- [ ] Export/import collections
- [ ] Batch reformatting
- [ ] API endpoint mode

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Support

For issues or questions:
1. Check troubleshooting section above
2. Review `.env` configuration
3. Test Qdrant connectivity
4. Verify OpenAI API status

---

**Built with ❤️ using Streamlit, OpenAI GPT-5, and Qdrant**
