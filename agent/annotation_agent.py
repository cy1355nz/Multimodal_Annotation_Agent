"""
Multimodal Annotation Agent core implementation.

The workflow is intentionally split into specialist agents:
- RetrievalAgent: pulls relevant examples/guidelines from local RAG.
- MemoryAgent: manages human review feedback and stores approved annotations.
- PerceptionAgent: summarizes text and image evidence.
- AnnotationWriterAgent: generates schema-aligned JSON.
- QualityAgent: validates and repairs JSON.
- PersistenceAgent: saves final annotations.
"""
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.memory import AnnotationMemory
from agent.rag import SimpleRAGStore
from agent.tools.annotation_tools import validate_json_output
from agent.tools.detection_tools import retrieve_detection_results
from agent.tools.rag_tools import retrieve_annotation_context
from models.factory import chat_model
from schemas.annotation_schema import AnnotationResult
from schemas.state_schema import AnnotationState, RetrievedContext
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_system_prompts


def _invoke_tool(tool_obj, tool_input: Dict[str, Any], state: AnnotationState, agent_name: str) -> str:
    """Invoke a LangChain tool with trace/log visibility."""

    state.add_trace(agent_name, f"Calling tool `{tool_obj.name}`.")
    logger.info(
        "[tool call] request_id=%s agent=%s tool=%s args=%s",
        state.request_id,
        agent_name,
        tool_obj.name,
        tool_input,
    )
    result = tool_obj.invoke(tool_input)
    logger.info(
        "[tool result] request_id=%s agent=%s tool=%s result_preview=%s",
        state.request_id,
        agent_name,
        tool_obj.name,
        str(result)[:500],
    )
    return result


class RetrievalAgent:
    """Retrieve RAG context and exact frame-level detection results."""

    def __init__(self, store: Optional[SimpleRAGStore] = None):
        self.store = store

    def run(self, state: AnnotationState) -> AnnotationState:
        if self.store is not None:
            state.retrieved_context = self.store.retrieve(state.description, k=3)
        else:
            tool_result = _invoke_tool(
                retrieve_annotation_context,
                {"query": state.description, "k": 3},
                state,
                "RetrievalAgent",
            )
            state.retrieved_context = [
                RetrievedContext.model_validate(item)
                for item in json.loads(tool_result)
            ]
        state.add_trace("RetrievalAgent", f"Retrieved {len(state.retrieved_context)} local knowledge items.")
        if state.clip_id and state.timestamp:
            detection_result = _invoke_tool(
                retrieve_detection_results,
                {"clip_id": state.clip_id, "timestamp": state.timestamp},
                state,
                "RetrievalAgent",
            )
            state.detection_context = json.loads(detection_result)
            if state.detection_context.get("found"):
                state.add_trace(
                    "RetrievalAgent",
                    f"Retrieved raw detection results for clip_id={state.clip_id}, timestamp={state.timestamp}.",
                )
            else:
                state.add_trace(
                    "RetrievalAgent",
                    f"No raw detection results found for clip_id={state.clip_id}, timestamp={state.timestamp}.",
                )
        return state


class MemoryAgent:
    """Manage human review state and persist approved annotations."""

    def __init__(self, memory: AnnotationMemory):
        self.memory = memory

    def request_review(self, state: AnnotationState) -> AnnotationState:
        return self.memory.request_review(state)

    def record_feedback(self, state: AnnotationState, feedback: str) -> AnnotationState:
        return self.memory.record_feedback(state, feedback)

    def mark_approved(self, state: AnnotationState) -> AnnotationState:
        return self.memory.mark_approved(state)

    def save(self, state: AnnotationState) -> AnnotationState:
        if state.result:
            self.memory.save_success_case(
                state.request_id,
                state.description,
                state.result,
                state.output_path,
                feedback_count=len(state.user_feedback),
            )
            state.review_status = "saved"
            state.add_trace("MemoryAgent", "Saved approved annotation as a reusable RAG memory case.")
        return state


class PerceptionAgent:
    """Create a compact evidence summary from text and optional images."""

    system_prompt = (
        "You are a perception specialist for autonomous-driving annotation. "
        "Extract observable facts only. Mention uncertainty explicitly. "
        "Focus on weather, time, road type, lanes, traffic signs/lights, objects, "
        "relative positions, and ego-vehicle decision cues."
    )

    def run(self, state: AnnotationState) -> AnnotationState:
        content_parts: List[Dict[str, Any]] = [
            {"type": "text", "text": f"Scene description:\n{state.description}\n\nReturn concise visual/text observations."}
        ]
        for path in state.image_paths[:3]:
            content_parts.append({"type": "image", "image": path})

        if not state.image_paths:
            state.visual_observations = f"No images provided. Text evidence: {state.description}"
            state.add_trace("PerceptionAgent", "Used text-only evidence because no images were provided.")
            return state

        response = chat_model.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=content_parts),
            ]
        )
        state.visual_observations = _message_to_text(response.content)
        state.add_trace("PerceptionAgent", f"Generated multimodal scene observations: \n{response.content}\n")
        return state


class AnnotationWriterAgent:
    """Generate an AnnotationResult JSON draft."""

    def __init__(self):
        self.system_prompt = load_system_prompts()
        self.schema_json = json.dumps(AnnotationResult.model_json_schema(), ensure_ascii=False, indent=2)

    def run(self, state: AnnotationState) -> AnnotationState:
        prompt = self._build_prompt(state)
        response = chat_model.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=[{"type": "text", "text": prompt}]),
            ]
        )
        state.draft_json = _extract_json(_message_to_text(response.content))
        state.add_trace("AnnotationWriterAgent", "Generated first JSON draft.")
        return state

    def repair(self, state: AnnotationState) -> AnnotationState:
        prompt = (
            "Repair the following JSON so it strictly validates against the AnnotationResult schema. "
            "Return JSON only, with no markdown.\n\n"
            f"Validation errors:\n{chr(10).join(state.validation_errors)}\n\n"
            f"Schema:\n{self.schema_json}\n\n"
            f"Draft JSON:\n{state.draft_json}"
        )
        response = chat_model.invoke(
            [
                SystemMessage(content="You repair invalid JSON annotations. Return valid JSON only."),
                HumanMessage(content=prompt),
            ]
        )
        state.draft_json = _extract_json(_message_to_text(response.content))
        state.add_trace("AnnotationWriterAgent", "Repaired JSON draft after validation feedback.")
        return state

    def revise_with_feedback(self, state: AnnotationState) -> AnnotationState:
        latest_feedback = state.user_feedback[-1] if state.user_feedback else ""
        prompt = (
            "Revise the validated annotation JSON according to the human reviewer feedback. "
            "Keep all unchanged fields stable unless the feedback requires a change. "
            "Return JSON only, with no markdown.\n\n"
            f"Schema:\n{self.schema_json}\n\n"
            f"Input description:\n{state.description}\n\n"
            f"Perception observations:\n{state.visual_observations or 'None'}\n\n"
            f"Retrieved examples/guidelines:\n{_format_context(state.retrieved_context)}\n\n"
            f"Raw detection results:\n{_format_detection_context(state.detection_context)}\n\n"
            f"Current annotation JSON:\n{json.dumps(state.result.model_dump(mode='json') if state.result else json.loads(state.draft_json or '{}'), ensure_ascii=False, indent=2)}\n\n"
            f"Reviewer feedback:\n{latest_feedback}"
        )
        response = chat_model.invoke(
            [
                SystemMessage(content="You revise structured annotations from human feedback. Return valid JSON only."),
                HumanMessage(content=prompt),
            ]
        )
        state.draft_json = _extract_json(_message_to_text(response.content))
        state.result = None
        state.add_trace("AnnotationWriterAgent", "Revised JSON draft using reviewer feedback.")
        return state

    def _build_prompt(self, state: AnnotationState) -> str:
        return (
            "Generate one JSON object that validates against the AnnotationResult Pydantic schema. "
            "Return JSON only, without markdown fences or explanatory prose.\n\n"
            f"Schema:\n{self.schema_json}\n\n"
            f"Input description:\n{state.description}\n\n"
            f"Perception observations:\n{state.visual_observations or 'None'}\n\n"
            f"Raw detection results:\n{_format_detection_context(state.detection_context)}\n\n"
            f"Retrieved examples and approved cases:\n{_format_context(state.retrieved_context)}"
        )


class QualityAgent:
    """Validate schema compliance and trigger one repair attempt if needed."""

    def __init__(self, writer: AnnotationWriterAgent):
        self.writer = writer

    def run(self, state: AnnotationState) -> AnnotationState:
        state.result = None
        for attempt in range(2):
            try:
                if not state.draft_json:
                    raise ValueError("Empty JSON draft.")
                tool_result = _invoke_tool(
                    validate_json_output,
                    {"json_string": state.draft_json},
                    state,
                    "QualityAgent",
                )
                if tool_result != "valid":
                    raise ValueError(tool_result)
                data = json.loads(state.draft_json)
                state.result = AnnotationResult.model_validate(data)
                state.validation_errors = []
                state.add_trace("QualityAgent", f"Validation passed on attempt {attempt + 1}.")
                return state
            except Exception as exc:
                state.validation_errors.append(str(exc))
                state.add_trace("QualityAgent", f"Validation failed on attempt {attempt + 1}.")
                if attempt == 0:
                    state = self.writer.repair(state)

        raise ValueError("Failed to produce a valid AnnotationResult JSON.")


class PersistenceAgent:
    """Persist validated output to disk."""

    def run(self, state: AnnotationState) -> AnnotationState:
        if not state.result:
            raise ValueError("Cannot save annotation before validation passes.")
        if state.review_status != "approved":
            raise ValueError("Cannot save annotation before human approval.")

        output_dir = get_abs_path("data/output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"annotation_{timestamp}_{state.request_id[:8]}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(state.result.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        state.output_path = output_path
        state.add_trace("PersistenceAgent", f"Saved annotation to {output_path}.")
        return state


class AnnotationAgent:
    """Multi-agent annotation orchestrator."""

    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.memory_agent = MemoryAgent(AnnotationMemory())
        self.perception_agent = PerceptionAgent()
        self.writer_agent = AnnotationWriterAgent()
        self.quality_agent = QualityAgent(self.writer_agent)
        self.persistence_agent = PersistenceAgent()

    def generate_for_review(
        self,
        description: str,
        image_paths: Optional[List[str]] = None,
        clip_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> AnnotationState:
        state = AnnotationState(
            description=description,
            image_paths=self._normalize_image_paths(image_paths),
            clip_id=clip_id or None,
            timestamp=timestamp or None,
        )
        for stage in (
            self.retrieval_agent.run,
            self.perception_agent.run,
            self.writer_agent.run,
            self.quality_agent.run,
            self.memory_agent.request_review,
        ):
            state = stage(state)
        return state

    def revise_with_feedback(self, state: AnnotationState, feedback: str) -> AnnotationState:
        logger.info(
            "[review revision] request_id=%s status=%s event=feedback_received",
            state.request_id,
            state.review_status,
        )
        state = self.memory_agent.record_feedback(state, feedback)
        logger.info(
            "[review revision] request_id=%s status=%s agent=AnnotationWriterAgent event=start",
            state.request_id,
            state.review_status,
        )
        state = self.writer_agent.revise_with_feedback(state)
        logger.info(
            "[review revision] request_id=%s status=%s agent=QualityAgent event=start",
            state.request_id,
            state.review_status,
        )
        state = self.quality_agent.run(state)
        state = self.memory_agent.request_review(state)
        logger.info(
            "[review revision] request_id=%s status=%s event=completed",
            state.request_id,
            state.review_status,
        )
        return state

    def approve_and_save(self, state: AnnotationState) -> AnnotationState:
        state = self.memory_agent.mark_approved(state)
        state = self.persistence_agent.run(state)
        state = self.memory_agent.save(state)
        return state

    def execute(
        self,
        description: str,
        image_paths: Optional[List[str]] = None,
        auto_save: bool = False,
        clip_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> AnnotationState:
        state = self.generate_for_review(description, image_paths, clip_id=clip_id, timestamp=timestamp)
        if auto_save:
            state = self.approve_and_save(state)
        return state

    def execute_stream(
        self,
        description: str,
        image_paths: Optional[List[str]] = None,
        clip_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        state = AnnotationState(
            description=description,
            image_paths=self._normalize_image_paths(image_paths),
            clip_id=clip_id or None,
            timestamp=timestamp or None,
        )
        yield "### Agent trace\n"

        stages = [
            ("RetrievalAgent", self.retrieval_agent.run),
            ("PerceptionAgent", self.perception_agent.run),
            ("AnnotationWriterAgent", self.writer_agent.run),
            ("QualityAgent", self.quality_agent.run),
            ("MemoryAgent", self.memory_agent.request_review),
        ]

        for name, stage in stages:
            logger.info(
                "[agent stage] request_id=%s status=%s agent=%s event=start",
                state.request_id,
                state.review_status,
                name,
            )
            yield f"\n**{name}** running...\n"
            state = stage(state)
            yield f"{state.trace[-1].message}\n"

        self.last_state = state
        yield "\nDraft is ready for human review below.\n"
        yield "\nStatus: `pending_review`"

    def _normalize_image_paths(self, image_paths: Optional[Iterable[str]]) -> List[str]:
        if not image_paths:
            return []
        normalized = []
        for path in image_paths:
            if os.path.isabs(path):
                normalized.append(path)
            else:
                normalized.append(get_abs_path(path))
        return normalized[:3]


def _format_context(items: List[RetrievedContext]) -> str:
    if not items:
        return "None"
    return "\n\n".join(
        f"[{item.source} | score={item.score}]\n{item.content}"
        for item in items
    )


def _format_detection_context(context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return "None"
    return json.dumps(context, ensure_ascii=False, indent=2)


def _message_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


if __name__ == "__main__":
    agent = AnnotationAgent()
    description = (
        "You are driving on an urban road and approaching an intersection with traffic lights. "
        "The traffic light is currently red, so you need to slow down and come to a complete stop."
    )
    for chunk in agent.execute_stream(description):
        print(chunk, end="", flush=True)
