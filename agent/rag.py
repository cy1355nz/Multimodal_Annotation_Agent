"""
Lightweight retrieval for annotation examples and approved cases.

This deliberately avoids a vector database dependency so the demo can run
locally in interviews. The interface is small enough to swap in FAISS, Chroma,
or a managed vector store later.
"""
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List

from schemas.state_schema import RetrievedContext
from utils.path_tool import get_abs_path


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class KnowledgeDocument:
    """A retrievable piece of annotation knowledge."""

    source: str
    content: str


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


class SimpleRAGStore:
    """Keyword-overlap retriever over standalone RAG assets and approved cases."""

    def __init__(self, documents: Iterable[KnowledgeDocument]):
        self.documents = list(documents)

    @classmethod
    def from_project_defaults(cls) -> "SimpleRAGStore":
        documents: List[KnowledgeDocument] = []

        rag_dir = get_abs_path("data/RAG")
        if os.path.isdir(rag_dir):
            documents.extend(_load_rag_documents(rag_dir))

        memory_path = get_abs_path("data/memory/annotation_memory.jsonl")
        if os.path.exists(memory_path):
            documents.extend(_load_approved_memory_documents(memory_path))

        return cls(documents)

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedContext]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: List[RetrievedContext] = []
        for document in self.documents:
            doc_tokens = _tokenize(document.content)
            if not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens)
            score = overlap / max(len(query_tokens), 1)
            if score > 0:
                scored.append(
                    RetrievedContext(
                        source=document.source,
                        content=document.content[:2500],
                        score=round(score, 4),
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]


def _load_rag_documents(rag_dir: str) -> List[KnowledgeDocument]:
    documents: List[KnowledgeDocument] = []
    for root, _, filenames in os.walk(rag_dir):
        for filename in sorted(filenames):
            if not filename.lower().endswith((".txt", ".md", ".json")):
                continue
            path = os.path.join(root, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                rel_path = os.path.relpath(path, rag_dir)
                documents.append(KnowledgeDocument(source=f"rag:{rel_path}", content=content))
    return documents


def _load_approved_memory_documents(memory_path: str) -> List[KnowledgeDocument]:
    documents: List[KnowledgeDocument] = []
    with open(memory_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("approved") is not True or record.get("source") != "human_review":
                continue
            request_id = record.get("request_id", "unknown")
            description = record.get("description", "")
            annotation = json.dumps(record.get("annotation", {}), ensure_ascii=False)
            content = f"Approved annotation case\nDescription: {description}\nAnnotation JSON: {annotation}"
            documents.append(KnowledgeDocument(source=f"approved_memory:{request_id}", content=content))
    return documents
