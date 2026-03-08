"""Unit tests for Knowledge Store and RAG pipeline."""
import pytest
from pathlib import Path
import sys

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge.loader import load_file, chunk_documents, load_all_knowledge, Document
from src.knowledge.store import (
    add_knowledge, get_knowledge, list_all, search_by_keyword,
    update_knowledge, soft_delete,
)


class TestDocumentLoader:
    def test_load_json_file(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('[{"id": 1, "name": "test"}]', encoding="utf-8")
        docs = load_file(f, "structured")
        assert len(docs) == 1
        assert "id" in docs[0].content

    def test_load_md_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nSome content here.", encoding="utf-8")
        docs = load_file(f, "tribal")
        assert len(docs) == 1
        assert "Title" in docs[0].content
        assert docs[0].metadata["knowledge_type"] == "tribal"

    def test_chunk_long_document(self):
        long_content = "word " * 300  # ~1500 chars
        docs = [Document(content=long_content, metadata={"source": "test"})]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.content) <= 300  # chunks shouldn't be massively over limit

    def test_unsupported_format(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("a,b,c", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(f, "structured")

    def test_load_all_knowledge(self):
        """Integration test: load all real data files."""
        result = load_all_knowledge()
        assert "structured" in result
        assert "unstructured" in result
        assert "tribal" in result
        # Each category should have at least some documents
        for ktype, docs in result.items():
            assert len(docs) > 0, f"No documents loaded for {ktype}"


class TestKnowledgeStore:
    def test_add_and_get(self, tmp_path, monkeypatch):
        # Point store to temp file
        import src.knowledge.store as store_mod
        monkeypatch.setattr(store_mod, "_STORE_FILE", tmp_path / "store.json")

        item = add_knowledge("Test content", "tribal", source="test")
        assert item["id"].startswith("k-")
        assert item["content"] == "Test content"
        assert item["deprecated"] is False

        fetched = get_knowledge(item["id"])
        assert fetched is not None
        assert fetched["content"] == "Test content"

    def test_list_all(self, tmp_path, monkeypatch):
        import src.knowledge.store as store_mod
        monkeypatch.setattr(store_mod, "_STORE_FILE", tmp_path / "store.json")

        add_knowledge("Structured data", "structured")
        add_knowledge("Tribal tip", "tribal")
        add_knowledge("Unstructured doc", "unstructured")

        all_items = list_all()
        assert len(all_items) == 3

        structured = list_all("structured")
        assert len(structured) == 1

    def test_search_by_keyword(self, tmp_path, monkeypatch):
        import src.knowledge.store as store_mod
        monkeypatch.setattr(store_mod, "_STORE_FILE", tmp_path / "store.json")

        add_knowledge("HMM reefer booking tips", "tribal")
        add_knowledge("MAERSK schedule information", "structured")

        results = search_by_keyword("reefer")
        assert len(results) == 1
        assert "reefer" in results[0]["content"]

    def test_update_knowledge(self, tmp_path, monkeypatch):
        import src.knowledge.store as store_mod
        monkeypatch.setattr(store_mod, "_STORE_FILE", tmp_path / "store.json")

        item = add_knowledge("Original content", "tribal")
        updated = update_knowledge(item["id"], "Updated content")

        assert updated is not None
        assert updated["content"] == "Updated content"
        assert updated["version"] == 2
        assert len(updated["history"]) == 1

    def test_soft_delete(self, tmp_path, monkeypatch):
        import src.knowledge.store as store_mod
        monkeypatch.setattr(store_mod, "_STORE_FILE", tmp_path / "store.json")

        item = add_knowledge("To be deleted", "tribal")
        result = soft_delete(item["id"])
        assert result is True

        fetched = get_knowledge(item["id"])
        assert fetched is None  # soft deleted

        all_items = list_all()
        assert len(all_items) == 0
