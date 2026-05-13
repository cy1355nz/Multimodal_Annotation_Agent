import json
import os

import pytest

import agent.annotation_agent as annotation_agent_module
from agent.annotation_agent import AnnotationAgent, PersistenceAgent
from agent.memory import AnnotationMemory
from agent.rag import SimpleRAGStore
from schemas.state_schema import AnnotationState


def _agent_with_fake_model(tmp_path, monkeypatch, queue_chat_model, responses):
    monkeypatch.setattr(annotation_agent_module, "chat_model", queue_chat_model(responses))
    monkeypatch.setattr(annotation_agent_module, "get_abs_path", lambda path: str(tmp_path / path))

    agent = AnnotationAgent()
    agent.retrieval_agent.store = SimpleRAGStore([])
    agent.memory_agent.memory = AnnotationMemory(
        memory_path=str(tmp_path / "memory" / "annotation_memory.jsonl")
    )
    return agent


def test_generate_for_review_does_not_save_or_write_memory(
    tmp_path,
    monkeypatch,
    queue_chat_model,
    valid_annotation_data,
):
    agent = _agent_with_fake_model(tmp_path, monkeypatch, queue_chat_model, [valid_annotation_data])

    state = agent.generate_for_review("Cloudy day with red traffic light.")

    assert state.review_status == "pending_review"
    assert state.result.traffic_lights[0].color == "red"
    assert state.output_path is None
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "memory" / "annotation_memory.jsonl").exists()
    assert [trace.agent for trace in state.trace] == [
        "RetrievalAgent",
        "PerceptionAgent",
        "AnnotationWriterAgent",
        "QualityAgent",
        "QualityAgent",
        "MemoryAgent",
    ]
    assert state.trace[3].message == "Calling tool `validate_json_output`."


def test_generate_for_review_retrieves_raw_detection_context(
    tmp_path,
    monkeypatch,
    queue_chat_model,
    valid_annotation_data,
):
    db_dir = tmp_path / "data" / "detection_db"
    db_dir.mkdir(parents=True)
    (db_dir / "detection_results.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "clip_id": "clip_1",
                        "timestamp": "100",
                        "detections": {"traffic_lights": [{"color": "red"}]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.tools.detection_tools.get_abs_path", lambda path: str(tmp_path / path))
    agent = _agent_with_fake_model(tmp_path, monkeypatch, queue_chat_model, [valid_annotation_data])

    state = agent.generate_for_review(
        "Cloudy day with red traffic light.",
        clip_id="clip_1",
        timestamp="100",
    )

    assert state.detection_context["found"] is True
    assert state.detection_context["detections"]["traffic_lights"][0]["color"] == "red"
    assert any("retrieve_detection_results" in trace.message for trace in state.trace)


def test_revise_with_feedback_records_feedback_and_revalidates(
    tmp_path,
    monkeypatch,
    queue_chat_model,
    valid_annotation_data,
):
    revised_data = json.loads(json.dumps(valid_annotation_data))
    revised_data["traffic_lights"][0]["color"] = "green"
    revised_data["ego_vehicle"]["longitudinal_action"] = "maintain_speed"
    agent = _agent_with_fake_model(
        tmp_path,
        monkeypatch,
        queue_chat_model,
        [valid_annotation_data, revised_data],
    )
    state = agent.generate_for_review("Traffic light ahead.")

    revised_state = agent.revise_with_feedback(
        state,
        "The traffic light is green, and ego should maintain speed.",
    )

    assert revised_state.review_status == "pending_review"
    assert revised_state.user_feedback == [
        "The traffic light is green, and ego should maintain speed."
    ]
    assert revised_state.result.traffic_lights[0].color == "green"
    assert revised_state.result.ego_vehicle.longitudinal_action == "maintain_speed"


def test_approve_and_save_persists_output_and_memory(
    tmp_path,
    monkeypatch,
    queue_chat_model,
    valid_annotation_data,
):
    agent = _agent_with_fake_model(tmp_path, monkeypatch, queue_chat_model, [valid_annotation_data])
    state = agent.generate_for_review("Cloudy day with red traffic light.")

    saved_state = agent.approve_and_save(state)

    assert saved_state.review_status == "saved"
    assert saved_state.output_path is not None
    assert os.path.exists(saved_state.output_path)

    memory_path = tmp_path / "memory" / "annotation_memory.jsonl"
    record = json.loads(memory_path.read_text(encoding="utf-8").strip())
    assert record["approved"] is True
    assert record["source"] == "human_review"
    assert record["annotation"]["traffic_lights"][0]["color"] == "red"


def test_persistence_agent_rejects_unapproved_state(tmp_path, monkeypatch, valid_annotation_data):
    monkeypatch.setattr(annotation_agent_module, "get_abs_path", lambda path: str(tmp_path / path))
    state = AnnotationState(description="red light", review_status="pending_review")
    state.draft_json = json.dumps(valid_annotation_data)
    from schemas.annotation_schema import AnnotationResult

    state.result = AnnotationResult.model_validate(valid_annotation_data)

    with pytest.raises(ValueError, match="human approval"):
        PersistenceAgent().run(state)
