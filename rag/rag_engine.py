"""
rag_engine.py
-------------
RAG using FAISS + sentence-transformers (all-MiniLM-L6-v2).
Replaces TF-IDF with real dense embeddings for better semantic retrieval.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DOCS_DIR   = os.path.join(os.path.dirname(__file__), "../docs")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"


class RAGEngine:
    def __init__(self):
        self.chunks    = []
        self.model     = None
        self.index     = None  # FAISS index

    def build_index(self) -> None:
        print("[RAG] Loading sentence transformer model...")
        self.model = SentenceTransformer(MODEL_NAME)

        raw_chunks = []
        for fname in os.listdir(DOCS_DIR):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(DOCS_DIR, fname)) as f:
                text = f.read()
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
            raw_chunks.extend(paragraphs)
            print(f"[RAG] Loaded {len(paragraphs)} chunks from {fname}")

        self.chunks = raw_chunks

        print("[RAG] Generating embeddings...")
        embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalize
        self.index.add(embeddings)

        print(f"[RAG] FAISS index built: {len(self.chunks)} chunks, dim={dim}")

    def save(self) -> None:
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "index" : faiss.serialize_index(self.index),
                "model_name": MODEL_NAME,
            }, f)
        print(f"[RAG] Saved → {INDEX_PATH}")

    def load(self) -> bool:
        if not os.path.exists(INDEX_PATH):
            return False
        print("[RAG] Loading FAISS index...")
        with open(INDEX_PATH, "rb") as f:
            d = pickle.load(f)
        self.chunks = d["chunks"]
        self.index  = faiss.deserialize_index(d["index"])
        self.model  = SentenceTransformer(d.get("model_name", MODEL_NAME))
        print(f"[RAG] Loaded ({len(self.chunks)} chunks)")
        return True

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        q_vec = self.model.encode([query])
        q_vec = np.array(q_vec).astype("float32")
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score > 0.1:
                results.append(self.chunks[idx])
        return results


_rag: RAGEngine | None = None


def get_rag() -> RAGEngine:
    global _rag
    if _rag is None:
        _rag = RAGEngine()
        if not _rag.load():
            _rag.build_index()
            _rag.save()
    return _rag


if __name__ == "__main__":
    rag = RAGEngine()
    rag.build_index()
    rag.save()
    results = rag.retrieve("why is electronics merchant category high risk?")
    print("\n--- Retrieved chunks ---")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r[:300]}")