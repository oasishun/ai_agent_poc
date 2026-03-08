"""Knowledge management: store, RAG pipeline, and document loader.

Note: RAGPipeline and get_rag_pipeline use lazy imports to avoid
chromadb being loaded at module import time (compatibility with Python 3.14).
"""
from .store import add_knowledge, get_knowledge, list_all, search_by_keyword, update_knowledge, soft_delete
from .loader import load_all_knowledge, Document


def get_rag_pipeline():
    """Lazy import of RAGPipeline to avoid chromadb loading at module level."""
    from .rag import get_rag_pipeline as _get_rag
    return _get_rag()


__all__ = [
    "add_knowledge", "get_knowledge", "list_all", "search_by_keyword",
    "update_knowledge", "soft_delete", "get_rag_pipeline",
    "load_all_knowledge", "Document",
]
