import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        dim: int = 2048,
    ):
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        collection = collection or os.getenv("QDRANT_COLLECTION", "docs_nemotron_3")
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        else:
            collection_info = self.client.get_collection(self.collection)
            vectors_config = collection_info.config.params.vectors
            existing_dim = getattr(vectors_config, "size", None)
            if existing_dim is not None and existing_dim != dim:
                raise RuntimeError(
                    f"Qdrant collection '{self.collection}' uses {existing_dim}-dimensional "
                    f"vectors, but the configured embedding model uses {dim}. "
                    "Choose a new QDRANT_COLLECTION and re-ingest the PDFs."
                )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k=5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        ).points
        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
