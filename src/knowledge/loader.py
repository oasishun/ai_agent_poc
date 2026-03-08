"""JSON/MD file loader and chunker for knowledge ingestion."""
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config

KnowledgeType = Literal["structured", "unstructured", "tribal"]


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)


def _load_json_file(path: Path, knowledge_type: KnowledgeType) -> list[Document]:
    """Convert JSON file to Documents (one per top-level record)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            content = json.dumps(item, ensure_ascii=False, indent=2)
            docs.append(
                Document(
                    content=content,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "knowledge_type": knowledge_type,
                        "record_index": i,
                        "format": "json",
                    },
                )
            )
    else:
        docs.append(
            Document(
                content=json.dumps(data, ensure_ascii=False, indent=2),
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "knowledge_type": knowledge_type,
                    "format": "json",
                },
            )
        )
    return docs


def _load_md_file(path: Path, knowledge_type: KnowledgeType) -> list[Document]:
    """Load Markdown file as a single Document."""
    content = path.read_text(encoding="utf-8")
    return [
        Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "knowledge_type": knowledge_type,
                "format": "markdown",
            },
        )
    ]


def load_file(path: Path, knowledge_type: KnowledgeType) -> list[Document]:
    """Load a JSON or Markdown file into Documents."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_file(path, knowledge_type)
    elif suffix in (".md", ".markdown"):
        return _load_md_file(path, knowledge_type)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def chunk_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split long documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.rag_chunk_size,
        chunk_overlap=chunk_overlap or config.rag_chunk_overlap,
        length_function=len,
    )
    result: list[Document] = []
    for doc in docs:
        chunks = splitter.split_text(doc.content)
        for i, chunk in enumerate(chunks):
            meta = dict(doc.metadata)
            meta["chunk_index"] = i
            meta["total_chunks"] = len(chunks)
            result.append(Document(content=chunk, metadata=meta))
    return result


def load_all_knowledge() -> dict[KnowledgeType, list[Document]]:
    """Load and chunk all knowledge files from data directory."""
    mapping: dict[KnowledgeType, tuple[Path, KnowledgeType]] = {
        "structured": (config.structured_dir, "structured"),
        "unstructured": (config.unstructured_dir, "unstructured"),
        "tribal": (config.tribal_dir, "tribal"),
    }
    result: dict[KnowledgeType, list[Document]] = {}
    for key, (directory, ktype) in mapping.items():
        docs: list[Document] = []
        for path in sorted(directory.glob("*")):
            if path.suffix.lower() in (".json", ".md", ".markdown"):
                # Skip tracking_events for RAG (it's API data, not knowledge)
                if path.name == "tracking_events.json":
                    continue
                raw = load_file(path, ktype)
                docs.extend(chunk_documents(raw))
        result[key] = docs
    return result
