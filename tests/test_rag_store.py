import json

from agent.rag import KnowledgeDocument, SimpleRAGStore, _load_approved_memory_documents


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        text = text.lower()
        return [
            1.0 if "ambulance" in text or "emergency" in text else 0.0,
            1.0 if "traffic" in text or "red" in text else 0.0,
        ]


class FailingEmbeddings:
    def embed_documents(self, texts):
        raise RuntimeError("embedding service unavailable")


def test_project_rag_loads_data_rag_and_not_prompt_or_sample_dirs(monkeypatch, tmp_path):
    project_root = tmp_path
    rag_dir = project_root / "data" / "RAG"
    prompt_dir = project_root / "prompts"
    sample_dir = project_root / "sample_data" / "scene_descriptions"
    rag_dir.mkdir(parents=True)
    prompt_dir.mkdir()
    sample_dir.mkdir(parents=True)

    (rag_dir / "traffic_light.md").write_text("red traffic light stop urban road", encoding="utf-8")
    (prompt_dir / "few_shot_examples.txt").write_text("PROMPT SHOULD NOT LOAD", encoding="utf-8")
    (sample_dir / "scene.txt").write_text("SAMPLE SHOULD NOT LOAD", encoding="utf-8")

    monkeypatch.setattr("agent.rag.get_abs_path", lambda path: str(project_root / path))

    store = SimpleRAGStore.from_project_defaults()

    assert [doc.source for doc in store.documents] == ["rag:traffic_light.md"]
    assert "PROMPT SHOULD NOT LOAD" not in store.documents[0].content
    assert "SAMPLE SHOULD NOT LOAD" not in store.documents[0].content
    assert store.documents[0].retrieval_text == "red traffic light stop urban road"


def test_retrieve_returns_relevant_rag_case(monkeypatch, tmp_path):
    project_root = tmp_path
    rag_dir = project_root / "data" / "RAG"
    rag_dir.mkdir(parents=True)
    (rag_dir / "traffic_light.md").write_text(
        "Input:\nCloudy dusk at an urban intersection with a red traffic light. Ego should stop.\n\nGuidance:\nUse decelerate.",
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.rag.get_abs_path", lambda path: str(project_root / path))
    monkeypatch.setattr("agent.rag.embedding_model", None)
    store = SimpleRAGStore.from_project_defaults()

    hits = store.retrieve("red traffic light urban intersection stop", k=2)

    assert hits
    assert hits[0].source.startswith("rag:")
    assert "traffic" in hits[0].content.lower()


def test_hybrid_retrieval_uses_embedding_similarity():
    store = SimpleRAGStore(
        [
            KnowledgeDocument(source="doc:traffic", content="traffic light case", retrieval_text="red signal stop"),
            KnowledgeDocument(source="doc:emergency", content="ambulance yield case", retrieval_text="ambulance emergency yield"),
        ],
        embeddings=FakeEmbeddings(),
        bm25_weight=0.2,
        embedding_weight=0.8,
    )

    hits = store.retrieve("priority emergency vehicle", k=2)

    assert hits[0].source == "doc:emergency"
    assert hits[0].metadata["embedding_score"] > 0


def test_hybrid_retrieval_falls_back_to_bm25_when_embeddings_fail():
    store = SimpleRAGStore(
        [
            KnowledgeDocument(source="doc:traffic", content="traffic light case", retrieval_text="red traffic light stop"),
            KnowledgeDocument(source="doc:snow", content="snow case", retrieval_text="snowy highway"),
        ],
        embeddings=FailingEmbeddings(),
    )

    hits = store.retrieve("red traffic stop", k=1)

    assert hits[0].source == "doc:traffic"
    assert hits[0].metadata["embedding_score"] == 0.0


def test_empty_query_returns_no_results():
    store = SimpleRAGStore.from_project_defaults()

    assert store.retrieve("") == []


def test_memory_loader_only_accepts_human_approved_cases(tmp_path, valid_annotation_data):
    memory_path = tmp_path / "annotation_memory.jsonl"
    rows = [
        {"request_id": "approved", "approved": True, "source": "human_review", "description": "red light", "annotation": valid_annotation_data},
        {"request_id": "wrong_source", "approved": True, "source": "fake_test", "description": "green light", "annotation": valid_annotation_data},
        {"request_id": "not_approved", "approved": False, "source": "human_review", "description": "snow", "annotation": valid_annotation_data},
    ]
    memory_path.write_text(
        "\n".join([json.dumps(row) for row in rows] + ["not json"]),
        encoding="utf-8",
    )

    documents = _load_approved_memory_documents(str(memory_path))

    assert len(documents) == 1
    assert documents[0].source == "approved_memory:approved"
    assert "red light" in documents[0].content
