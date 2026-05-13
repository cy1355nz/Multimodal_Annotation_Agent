import json

import pytest

from agent.annotation_agent import QualityAgent
from schemas.state_schema import AnnotationState


class RepairingWriter:
    def __init__(self, repaired_json):
        self.repaired_json = repaired_json
        self.repair_calls = 0

    def repair(self, state):
        self.repair_calls += 1
        state.draft_json = json.dumps(self.repaired_json)
        state.add_trace("AnnotationWriterAgent", "Repaired in test.")
        return state


class NonRepairingWriter:
    def repair(self, state):
        state.add_trace("AnnotationWriterAgent", "Still invalid in test.")
        return state


def test_quality_agent_repairs_invalid_json_then_validates(valid_annotation_data):
    writer = RepairingWriter(valid_annotation_data)
    quality_agent = QualityAgent(writer)
    state = AnnotationState(description="red light", draft_json="{bad json")

    state = quality_agent.run(state)

    assert writer.repair_calls == 1
    assert state.result is not None
    assert state.result.driving_environment.road_type == "Urban_Road"
    assert state.validation_errors == []


def test_quality_agent_raises_when_repair_still_invalid():
    quality_agent = QualityAgent(NonRepairingWriter())
    state = AnnotationState(description="red light", draft_json="{bad json")

    with pytest.raises(ValueError, match="Failed to produce"):
        quality_agent.run(state)

    assert len(state.validation_errors) == 2
