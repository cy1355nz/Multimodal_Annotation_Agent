import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def valid_annotation_data():
    return {
        "driving_environment": {
            "weather": "cloudy",
            "time": "day",
            "visibility": "good",
            "road_type": "Urban_Road",
        },
        "traffic_lights": [
            {
                "light_id": "light_1",
                "bbox": [700, 20, 750, 80],
                "color": "red",
                "state": "solid",
                "applicable_lanes": ["Lane1"],
            }
        ],
        "ego_vehicle": {
            "lateral_action": "keep_lane",
            "longitudinal_action": "decelerate",
            "reason": "Stop for red traffic light",
        },
        "scene_description": "Cloudy day on urban road with red traffic light ahead.",
    }


class FakeMessage:
    def __init__(self, content):
        self.content = content


class QueueChatModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("No fake model responses left.")
        response = self.responses.pop(0)
        if not isinstance(response, str):
            response = json.dumps(response)
        return FakeMessage(response)


@pytest.fixture
def queue_chat_model():
    return QueueChatModel
