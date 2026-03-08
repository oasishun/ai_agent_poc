"""RAG Pipeline: ChromaDB + OpenAI Embedding for semantic search."""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import config
from src.knowledge.loader import Document, KnowledgeType, load_all_knowledge

_COLLECTION_NAMES: dict[KnowledgeType, str] = {
    "structured": "logistics_structured",
    "unstructured": "logistics_unstructured",
    "tribal": "logistics_tribal",
}


class RAGPipeline:
    """Manages ChromaDB collections and provides similarity search."""

    def __init__(self):
        persist_dir = config.chroma_persist_dir
        self._embeddings = OpenAIEmbeddings(
            model=config.openai_embedding_model,
            openai_api_key=config.openai_api_key,
        )
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._stores: dict[KnowledgeType, Chroma] = {}
        self._init_stores()

    def _init_stores(self) -> None:
        for ktype, col_name in _COLLECTION_NAMES.items():
            self._stores[ktype] = Chroma(
                client=self._client,
                collection_name=col_name,
                embedding_function=self._embeddings,
            )

    def ingest_documents(self, docs: list[Document], knowledge_type: KnowledgeType) -> int:
        """Ingest a list of Documents into the appropriate ChromaDB collection."""
        store = self._stores[knowledge_type]
        texts = [d.content for d in docs]
        metadatas = [d.metadata for d in docs]
        if not texts:
            return 0
        store.add_texts(texts=texts, metadatas=metadatas)
        return len(texts)

    def ingest_all(self) -> dict[KnowledgeType, int]:
        """Load all knowledge files and ingest into ChromaDB."""
        all_docs = load_all_knowledge()
        counts: dict[KnowledgeType, int] = {}
        for ktype, docs in all_docs.items():
            n = self.ingest_documents(docs, ktype)
            counts[ktype] = n
            print(f"  Ingested {n} chunks into '{ktype}' collection")
        return counts

    def similarity_search(
        self,
        query: str,
        knowledge_type: KnowledgeType | None = None,
        k: int | None = None,
    ) -> list[dict]:
        """Search across collections and return top-k results with scores."""
        top_k = k or config.rag_top_k
        results: list[dict] = []

        search_types: list[KnowledgeType] = (
            [knowledge_type] if knowledge_type else list(_COLLECTION_NAMES.keys())
        )

        for ktype in search_types:
            store = self._stores[ktype]
            try:
                docs_and_scores = store.similarity_search_with_relevance_scores(query, k=top_k)
                for doc, score in docs_and_scores:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": round(score, 4),
                        "knowledge_type": ktype,
                    })
            except Exception:
                pass  # Collection may be empty

        # Sort by score descending and return top-k
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def reset_collection(self, knowledge_type: KnowledgeType) -> None:
        """Clear and recreate a collection."""
        col_name = _COLLECTION_NAMES[knowledge_type]
        try:
            self._client.delete_collection(col_name)
        except Exception:
            pass
        self._stores[knowledge_type] = Chroma(
            client=self._client,
            collection_name=col_name,
            embedding_function=self._embeddings,
        )


# Module-level singleton (lazy init)
_rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
