"""
Annotation tools for multimodal annotation agent.
Provides image processing, text reading, and vehicle data query capabilities.
"""
import os
import json
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.file_handler import read_text_file as _read_txt
from schemas.annotation_schema import AnnotationResult


@tool(description="Read txt format annotation description file and return string content")
def read_text_file(file_path: str) -> str:
    """
    Read natural language description file written by annotators.

    Args:
        file_path: Absolute or relative path to txt file.

    Returns:
        String content of the file.
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"[read_text_file] File does not exist: {file_path}")
            return f"Error: File {file_path} does not exist"

        content = _read_txt(file_path)

        # # debug only
        # content = "Sunny day, ego vehicle driving on urban expressway with a white foam box in the right lane ahead"

        logger.info(f"[read_text_file] Successfully read file: {file_path}, length: {len(content)}")
        return content
    except Exception as e:
        logger.error(f"[read_text_file] Failed to read file: {str(e)}")
        return f"Error: Failed to read file - {str(e)}"


@tool(description="Analyze image and extract visual information, return descriptive text of the image")
def analyze_image(image_path: str) -> str:
    """
    Analyze single image and extract key visual information from the scene.

    Args:
        image_path: Absolute or relative path to image file.

    Returns:
        Visual description text of the image.
    """
    try:
        original_input = image_path
        if not os.path.isabs(image_path):
            logger.info(f"[analyze_image] Received relative path: {image_path}, attempting to resolve...")

            # Strategy 1: Check data/temp/
            temp_path = get_abs_path(os.path.join("data/temp", image_path))
            if os.path.exists(temp_path):
                image_path = temp_path
                logger.info(f"[analyze_image] Resolved to temp: {image_path}")

            # Strategy 2: Check data/samples/images/
            else:
                sample_path = get_abs_path(os.path.join("data/samples/images", image_path))
                if os.path.exists(sample_path):
                    image_path = sample_path
                    logger.info(f"[analyze_image] Resolved to samples: {image_path}")

                # Strategy 3: Assume it's relative to project root
                else:
                    image_path = get_abs_path(image_path)
                    logger.info(f"[analyze_image] Resolved to root: {image_path}")

        if not os.path.exists(image_path):
            logger.warning(
                f"[analyze_image] Image does not exist. Original input: {original_input}, Resolved: {image_path}")
            return f"Error: Image not found. Tried: {image_path}"

        # TODO: Integrate actual image analysis model here
        # Currently returns placeholder, should call VLM for image understanding in production
        logger.info(f"[analyze_image] Analyzing image: {image_path}")
        return (f"**Analyzing Image**: Image [{os.path.basename(image_path)}] loaded, contains driving scene visual "
                f"information")
    except Exception as e:
        logger.error(f"[analyze_image] Failed to analyze image: {str(e)}")
        return f"Error: Failed to analyze image - {str(e)}"


@tool(description="Query real-time vehicle detection results such as object detection, lane detection, etc.")
def query_vehicle_data(query_type: str, frame_id: Optional[str] = None):
    """
    Query real-time detection results from vehicle data storage.

    Args:
        query_type: Query type, e.g., object_detection, lane_detection, traffic_signal.
        frame_id: Optional specific frame ID.

    Returns:
        JSON string of query results.
    """
    try:
        # TODO: Connect to actual vehicle data storage (database or file system)
        # Currently returns mock data, should query real detection results in production

        mock_data = {
            "object_detection": {
                "objects": [
                    {"type": "vehicle", "confidence": 0.95, "bbox": [100, 200, 300, 400]},
                    {"type": "pedestrian", "confidence": 0.88, "bbox": [400, 150, 450, 350]}
                ]
            },
            "lane_detection": {
                "lanes": [
                    {"lane_id": "Lane1", "type": "solid_white", "confidence": 0.92},
                    {"lane_id": "Lane2", "type": "dashed_white", "confidence": 0.89}
                ]
            },
            "traffic_signal": {
                "signals": [
                    {"type": "traffic_light", "color": "red", "confidence": 0.96}
                ]
            }
        }

        result = mock_data.get(query_type, {})
        logger.info(f"[query_vehicle_data] Query type: {query_type}, result: {json.dumps(result, ensure_ascii=False)}")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[query_vehicle_data] Query failed: {str(e)}")
        return f"Error: Failed to query vehicle data - {str(e)}"


@tool(description="Validate generated JSON against AnnotationResult schema specification")
def validate_json_output(json_string: str) -> str:
    """
    Validate JSON string against AnnotationResult Pydantic schema.

    Args:
        json_string: JSON string to validate.

    Returns:
        Validation result: "valid" or error message.
    """
    try:
        data = json.loads(json_string)
        # Validate using Pydantic model
        AnnotationResult.model_validate(data)
        logger.info("[validate_json_output] JSON validation passed")
        return "valid"
    except json.JSONDecodeError as e:
        logger.error(f"[validate_json_output] JSON format error: {str(e)}")
        return f"JSON format error: {str(e)}"
    except Exception as e:
        logger.error(f"[validate_json_output] Schema validation failed: {str(e)}")
        return f"Schema validation failed: {str(e)}"


@tool(description="Save annotation result as JSON file")
def save_annotation_result(json_string: str) -> str:
    """
    Save annotation result to JSON file.

    Args:
        json_string: JSON string content.
        output_path: Output file path.

    Returns:
        Save result message.
    """
    try:
        # Validate JSON format
        data = json.loads(json_string)
        output_dir = get_abs_path("data/output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"annotation_{timestamp}.json")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[save_annotation_result] Successfully saved to: {output_path}")
        return f"Successfully saved annotation result to: {output_path}"
    except Exception as e:
        logger.error(f"[save_annotation_result] Failed to save: {str(e)}")
        return f"Error: Failed to save - {str(e)}"
