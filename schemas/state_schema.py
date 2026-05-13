"""
Runtime state models for the annotation workflow.

The state object makes the agent pipeline inspectable: every specialist agent
reads from and writes to one shared structure instead of passing loose strings.
"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from schemas.annotation_schema import AnnotationResult
from utils.logger_handler import logger


class RetrievedContext(BaseModel):
    """One retrieved knowledge or memory item."""

    source: str
    content: str
    score: float = Field(ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    """A compact trace item for demos, logs, and debugging."""

    agent: str
    message: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class AnnotationState(BaseModel):
    """Shared state for one annotation request."""

    request_id: str = Field(default_factory=lambda: uuid4().hex)
    description: str
    image_paths: List[str] = Field(default_factory=list)
    clip_id: Optional[str] = None
    timestamp: Optional[str] = None
    retrieved_context: List[RetrievedContext] = Field(default_factory=list)
    detection_context: Optional[Dict[str, Any]] = None
    visual_observations: Optional[str] = None
    draft_json: Optional[str] = None
    validation_errors: List[str] = Field(default_factory=list)
    result: Optional[AnnotationResult] = None
    output_path: Optional[str] = None
    review_status: Literal["draft", "pending_review", "approved", "saved"] = "draft"
    user_feedback: List[str] = Field(default_factory=list)
    trace: List[AgentTrace] = Field(default_factory=list)

    def add_trace(self, agent: str, message: str) -> None:
        """Append a human-readable trace entry."""

        trace = AgentTrace(agent=agent, message=message)
        self.trace.append(trace)
        logger.info(
            "[agent trace] request_id=%s status=%s agent=%s message=%s",
            self.request_id,
            self.review_status,
            trace.agent,
            trace.message,
        )
