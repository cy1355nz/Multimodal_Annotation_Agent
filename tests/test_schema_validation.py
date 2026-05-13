import pytest
from pydantic import ValidationError

from schemas.annotation_schema import AnnotationResult


def test_valid_annotation_passes_schema_validation(valid_annotation_data):
    result = AnnotationResult.model_validate(valid_annotation_data)

    assert result.driving_environment.weather == "cloudy"
    assert result.traffic_lights[0].color == "red"
    assert result.ego_vehicle.longitudinal_action == "decelerate"


def test_invalid_enum_value_fails_schema_validation(valid_annotation_data):
    valid_annotation_data["driving_environment"]["weather"] = "clear"

    with pytest.raises(ValidationError):
        AnnotationResult.model_validate(valid_annotation_data)


def test_missing_required_environment_field_fails_validation(valid_annotation_data):
    del valid_annotation_data["driving_environment"]["road_type"]

    with pytest.raises(ValidationError):
        AnnotationResult.model_validate(valid_annotation_data)


def test_invalid_vehicle_type_fails_schema_validation(valid_annotation_data):
    valid_annotation_data["vehicles"] = [
        {
            "obj_id": "vehicle_1",
            "bbox": [100, 200, 300, 400],
            "type": "car",
        }
    ]

    with pytest.raises(ValidationError):
        AnnotationResult.model_validate(valid_annotation_data)
