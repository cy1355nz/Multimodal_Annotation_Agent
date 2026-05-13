"""
Hybrid retrieval for annotation examples and approved cases.

Static few-shot examples and approved historical annotations are retrieved with
BM25 + optional embedding similarity. Raw vehicle-side detections are handled by
the deterministic clip/timestamp lookup tool, not by this retriever.
"""
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from models.factory import embedding_model
from schemas.state_schema import RetrievedContext
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class KnowledgeDocument:
    """A retrievable piece of annotation knowledge."""

    source: str
    content: str
    retrieval_text: str
    metadata: dict = field(default_factory=dict)


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class SimpleRAGStore:
    """Hybrid BM25 + embedding retriever over examples and approved cases."""

    def __init__(
        self,
        documents: Iterable[KnowledgeDocument],
        embeddings=None,
        bm25_weight: float = 0.55,
        embedding_weight: float = 0.45,
    ):
        self.documents = list(documents)
        self.embeddings = embeddings
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self._tokenized_docs = [_tokenize(document.retrieval_text) for document in self.documents]
        self._doc_freq = self._build_doc_freq(self._tokenized_docs)
        self._avg_doc_len = (
            sum(len(tokens) for tokens in self._tokenized_docs) / len(self._tokenized_docs)
            if self._tokenized_docs
            else 0.0
        )
        self._document_vectors: Optional[List[List[float]]] = None

    @classmethod
    def from_project_defaults(cls) -> "SimpleRAGStore":
        documents: List[KnowledgeDocument] = []

        rag_dir = get_abs_path("data/RAG")
        if os.path.isdir(rag_dir):
            documents.extend(_load_rag_documents(rag_dir))

        memory_path = get_abs_path("data/memory/annotation_memory.jsonl")
        if os.path.exists(memory_path):
            documents.extend(_load_approved_memory_documents(memory_path))

        return cls(documents, embeddings=embedding_model)

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedContext]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        bm25_scores = [self._bm25_score(query_tokens, index) for index in range(len(self.documents))]
        embedding_scores = self._embedding_scores(query)
        normalized_bm25 = _min_max_normalize(bm25_scores)
        normalized_embedding = _min_max_normalize(embedding_scores)

        scored: List[RetrievedContext] = []
        for index, document in enumerate(self.documents):
            score = (
                self.bm25_weight * normalized_bm25[index]
                + self.embedding_weight * normalized_embedding[index]
            )
            if score <= 0:
                continue
            scored.append(
                RetrievedContext(
                    source=document.source,
                    content=document.content[:2500],
                    score=round(score, 4),
                    metadata={
                        **document.metadata,
                        "bm25_score": round(bm25_scores[index], 4),
                        "embedding_score": round(embedding_scores[index], 4),
                    },
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]

    def _bm25_score(self, query_tokens: List[str], doc_index: int) -> float:
        tokens = self._tokenized_docs[doc_index]
        if not tokens:
            return 0.0

        token_counts = Counter(tokens)
        doc_len = len(tokens)
        k1 = 1.5
        b = 0.75
        score = 0.0
        total_docs = len(self.documents)

        for token in query_tokens:
            freq = token_counts.get(token, 0)
            if freq == 0:
                continue
            doc_freq = self._doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / max(self._avg_doc_len, 1e-9))
            score += idf * (freq * (k1 + 1)) / denom
        return score

    def _embedding_scores(self, query: str) -> List[float]:
        if not self.documents or self.embeddings is None:
            return [0.0 for _ in self.documents]
        try:
            if self._document_vectors is None:
                self._document_vectors = self.embeddings.embed_documents(
                    [document.retrieval_text for document in self.documents]
                )
            query_vector = self.embeddings.embed_query(query)
            return [_cosine_similarity(query_vector, doc_vector) for doc_vector in self._document_vectors]
        except Exception as exc:
            logger.warning("[rag] embedding retrieval unavailable, falling back to BM25: %s", exc)
            return [0.0 for _ in self.documents]

    @staticmethod
    def _build_doc_freq(tokenized_docs: List[List[str]]) -> Counter:
        doc_freq = Counter()
        for tokens in tokenized_docs:
            doc_freq.update(set(tokens))
        return doc_freq


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
                documents.append(
                    KnowledgeDocument(
                        source=f"rag:{rel_path}",
                        content=content,
                        retrieval_text=_extract_input_text(content) or content,
                        metadata={"retrieval_type": "static_rag"},
                    )
                )
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
            documents.append(
                KnowledgeDocument(
                    source=f"approved_memory:{request_id}",
                    content=content,
                    retrieval_text=description,
                    metadata={"retrieval_type": "approved_memory"},
                )
            )
    return documents


def _extract_input_text(content: str) -> str:
    match = re.search(r"Input:\s*(.*?)(?:\n\nExpected annotation pattern:|\n\nGuidance:|\Z)", content, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(max_score, min_score):
        return [1.0 if score > 0 else 0.0 for score in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
