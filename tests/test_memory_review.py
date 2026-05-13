import json

from agent.memory import AnnotationMemory
from schemas.annotation_schema import AnnotationResult
from schemas.state_schema import AnnotationState


def test_request_review_sets_pending_status(valid_annotation_data):
    state = AnnotationState(description="red light")
    state.result = AnnotationResult.model_validate(valid_annotation_data)
    memory = AnnotationMemory(memory_path="/tmp/not-used.jsonl")

    memory.request_review(state)

    assert state.review_status == "pending_review"
    assert state.trace[-1].agent == "MemoryAgent"


def test_record_feedback_appends_non_empty_feedback_only():
    state = AnnotationState(description="red light")
    memory = AnnotationMemory(memory_path="/tmp/not-used.jsonl")

    memory.record_feedback(state, "  ")
    memory.record_feedback(state, "Change light color to green.")

    assert state.user_feedback == ["Change light color to green."]
    assert state.review_status == "draft"


def test_mark_approved_sets_approved_status():
    state = AnnotationState(description="red light", review_status="pending_review")
    memory = AnnotationMemory(memory_path="/tmp/not-used.jsonl")

    memory.mark_approved(state)

    assert state.review_status == "approved"


def test_save_success_case_writes_human_approved_jsonl(tmp_path, valid_annotation_data):
    memory_path = tmp_path / "memory" / "annotation_memory.jsonl"
    memory = AnnotationMemory(memory_path=str(memory_path))
    annotation = AnnotationResult.model_validate(valid_annotation_data)

    memory.save_success_case(
        request_id="request-1",
        description="red traffic light",
        annotation=annotation,
        output_path="/tmp/output.json",
        feedback_count=2,
    )

    record = json.loads(memory_path.read_text(encoding="utf-8").strip())
    assert record["request_id"] == "request-1"
    assert record["approved"] is True
    assert record["source"] == "human_review"
    assert record["feedback_count"] == 2
    assert record["annotation"]["traffic_lights"][0]["color"] == "red"
