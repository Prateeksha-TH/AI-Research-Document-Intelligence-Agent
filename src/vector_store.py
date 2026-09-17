from pathlib import Path
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

from .config import CHROMA_DIR, EMBEDDING_MODEL


class VectorStore:
    def __init__(self, persist_dir: str | Path = CHROMA_DIR):
        self.persist_dir = str(persist_dir)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="research_documents",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    @staticmethod
    def _id(text: str, source: str, page: int, chunk_id: str) -> str:
        raw = f"{source}|{page}|{chunk_id}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def add_chunks(self, chunks):
        if not chunks:
            return 0

        documents = [c.text for c in chunks]
        embeddings = self.embedder.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        ids = [self._id(c.text, c.source, c.page, c.chunk_id) for c in chunks]
        metadatas = [
            {"source": c.source, "page": c.page, "chunk_id": c.chunk_id}
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for doc, meta, distance in zip(docs, metas, distances):
            hits.append({
                "text": doc,
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", "?"),
                "chunk_id": meta.get("chunk_id", ""),
                "distance": float(distance),
            })

        return hits

    def count(self):
        return self.collection.count()

    def reset(self):
        self.client.delete_collection("research_documents")
        self.collection = self.client.get_or_create_collection(
            name="research_documents",
            metadata={"hnsw:space": "cosine"},
        )
