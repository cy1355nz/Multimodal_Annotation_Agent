"""
Conversation memory for human review and approved annotations.

RAG handles similarity search over examples and approved cases. This module
keeps the current review loop explicit: draft -> user feedback -> revised draft
-> approval -> durable memory.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict

from schemas.annotation_schema import AnnotationResult
from schemas.state_schema import AnnotationState
from utils.path_tool import get_abs_path


class AnnotationMemory:
    """Manage review state and append approved annotations to durable memory."""

    def __init__(self, memory_path: str | None = None):
        self.memory_path = memory_path or get_abs_path("data/memory/annotation_memory.jsonl")

    def request_review(self, state: AnnotationState) -> AnnotationState:
        """Mark the validated draft as waiting for human review."""

        state.review_status = "pending_review"
        state.add_trace("MemoryAgent", "Draft is ready for human review before persistence.")
        return state

    def record_feedback(self, state: AnnotationState, feedback: str) -> AnnotationState:
        """Store reviewer feedback in the in-session state."""

        clean_feedback = feedback.strip()
        if clean_feedback:
            state.user_feedback.append(clean_feedback)
            state.review_status = "draft"
            state.add_trace("MemoryAgent", "Recorded reviewer feedback for revision.")
        return state

    def mark_approved(self, state: AnnotationState) -> AnnotationState:
        """Mark a reviewed annotation as approved."""

        state.review_status = "approved"
        state.add_trace("MemoryAgent", "Reviewer approved the annotation.")
        return state

    def save_success_case(
        self,
        request_id: str,
        description: str,
        annotation: AnnotationResult,
        output_path: str | None,
        feedback_count: int = 0,
    ) -> None:
        """Append a successful, human-approved annotation to JSONL memory."""

        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        record: Dict[str, Any] = {
            "request_id": request_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "approved": True,
            "source": "human_review",
            "description": description,
            "annotation": annotation.model_dump(mode="json"),
            "output_path": output_path,
            "feedback_count": feedback_count,
        }
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
