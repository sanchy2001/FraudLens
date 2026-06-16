import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR   = os.path.join(os.path.dirname(__file__), "../docs")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "tfidf_index.pkl")


class RAGEngine:
    def __init__(self):
        self.chunks     = []
        self.vectorizer = None
        self.matrix     = None

    def build_index(self):
        raw_chunks = []
        for fname in os.listdir(DOCS_DIR):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(DOCS_DIR, fname)) as f:
                text = f.read()
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
            raw_chunks.extend(paragraphs)
            print(f"[RAG] Loaded {len(paragraphs)} chunks from {fname}")

        self.chunks     = raw_chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.matrix     = self.vectorizer.fit_transform(self.chunks)
        print(f"[RAG] Index built: {len(self.chunks)} chunks")

    def save(self):
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({
                "chunks"    : self.chunks,
                "vectorizer": self.vectorizer,
                "matrix"    : self.matrix
            }, f)
        print(f"[RAG] Saved → {INDEX_PATH}")

    def load(self):
        if not os.path.exists(INDEX_PATH):
            return False
        with open(INDEX_PATH, "rb") as f:
            d = pickle.load(f)
        self.chunks     = d["chunks"]
        self.vectorizer = d["vectorizer"]
        self.matrix     = d["matrix"]
        print(f"[RAG] Loaded ({len(self.chunks)} chunks)")
        return True

    def retrieve(self, query, top_k=4):
        q_vec  = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = scores.argsort()[-top_k:][::-1]
        return [self.chunks[i] for i in top_idx if scores[i] > 0.05]


_rag = None

def get_rag():
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