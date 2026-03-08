"""Knowledge Store: CRUD operations over the knowledge base."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.config import config
from src.knowledge.loader import Document, KnowledgeType, load_file, chunk_documents

_STORE_FILE = config.data_dir / "knowledge_store.json"


def _load_store() -> list[dict]:
    if not _STORE_FILE.exists():
        return []
    return json.loads(_STORE_FILE.read_text(encoding="utf-8"))


def _save_store(items: list[dict]) -> None:
    _STORE_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def add_knowledge(
    content: str,
    knowledge_type: KnowledgeType,
    source: str = "manual",
    tags: list[str] | None = None,
) -> dict:
    """Add a new knowledge item."""
    item = {
        "id": f"k-{uuid.uuid4().hex[:8]}",
        "content": content,
        "knowledge_type": knowledge_type,
        "source": source,
        "tags": tags or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "deprecated": False,
        "version": 1,
        "history": [],
    }
    store = _load_store()
    store.append(item)
    _save_store(store)
    return item


def get_knowledge(item_id: str) -> dict | None:
    """Retrieve a knowledge item by ID."""
    store = _load_store()
    return next((i for i in store if i["id"] == item_id and not i.get("deprecated")), None)


def list_all(knowledge_type: KnowledgeType | None = None) -> list[dict]:
    """List all non-deprecated knowledge items, optionally filtered by type."""
    store = _load_store()
    return [
        i for i in store
        if not i.get("deprecated")
        and (knowledge_type is None or i["knowledge_type"] == knowledge_type)
    ]


def search_by_keyword(query: str, knowledge_type: KnowledgeType | None = None) -> list[dict]:
    """Simple keyword-based search over knowledge items."""
    query_lower = query.lower()
    results = []
    for item in list_all(knowledge_type):
        if query_lower in item["content"].lower():
            results.append(item)
    return results


def update_knowledge(item_id: str, content: str) -> dict | None:
    """Update an existing knowledge item (preserving history)."""
    store = _load_store()
    for item in store:
        if item["id"] == item_id and not item.get("deprecated"):
            item["history"].append({
                "version": item["version"],
                "content": item["content"],
                "updated_at": item["updated_at"],
            })
            item["content"] = content
            item["version"] += 1
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save_store(store)
            return item
    return None


def soft_delete(item_id: str) -> bool:
    """Soft-delete a knowledge item by marking it deprecated."""
    store = _load_store()
    for item in store:
        if item["id"] == item_id:
            item["deprecated"] = True
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save_store(store)
            return True
    return False
