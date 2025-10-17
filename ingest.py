import os
import argparse
from typing import List
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv()


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
    return "\n".join(text)


def embed_texts(client: OpenAI, model: str, texts: List[str]) -> List[List[float]]:
    embeddings = []
    # Batch to avoid token limits; simple loop for clarity
    for t in texts:
        resp = client.embeddings.create(model=model, input=t)
        embeddings.append(resp.data[0].embedding)
    return embeddings


def ensure_collection(qdrant: QdrantClient, name: str, vector_size: int = 1536, distance=qmodels.Distance.COSINE):
    exists = False
    try:
        info = qdrant.get_collection(name)
        exists = info is not None
    except Exception:
        exists = False
    if not exists:
        qdrant.recreate_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
        )


def upsert_chunks(qdrant: QdrantClient, name: str, chunks: List[str], vectors: List[List[float]]):
    points = []
    for idx, (vec, text) in enumerate(zip(vectors, chunks)):
        points.append(
            qmodels.PointStruct(
                id=idx,
                vector=vec,
                payload={"text": text},
            )
        )
    qdrant.upsert(collection_name=name, points=points)


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into Qdrant as chunks with embeddings")
    parser.add_argument("pdf", help="Path to the legal vocabulary PDF")
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "legal_vocab"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    if not args.qdrant_url or not args.qdrant_api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be provided via args or environment")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in environment")

    text = extract_text_from_pdf(args.pdf)
    chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)

    client = OpenAI(api_key=openai_api_key)
    vectors = embed_texts(client, args.embedding_model, chunks)

    # Infer vector size from first vector
    vec_size = len(vectors[0]) if vectors else 1536

    qdrant = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    ensure_collection(qdrant, args.collection, vector_size=vec_size)
    upsert_chunks(qdrant, args.collection, chunks, vectors)

    print(f"Ingested {len(chunks)} chunks into collection '{args.collection}'.")


if __name__ == "__main__":
    main()
