"""
LangChain tools for deterministic clip/timestamp detection lookup.
"""
import json
import os
from typing import Any, Dict

from langchain_core.tools import tool

from utils.logger_handler import logger
from utils.path_tool import get_abs_path


@tool(description="Retrieve raw onboard detection results by exact clip id and timestamp")
def retrieve_detection_results(clip_id: str, timestamp: str) -> str:
    """
    Retrieve raw detection results for a specific clip frame.

    Args:
        clip_id: Unique driving clip id.
        timestamp: Frame timestamp inside the clip.

    Returns:
        JSON string with found status and raw detection payload.
    """
    db_path = get_abs_path("data/detection_db/detection_results.json")
    if not os.path.exists(db_path):
        logger.warning("[tool] detection database not found: %s", db_path)
        return json.dumps(
            {
                "found": False,
                "clip_id": clip_id,
                "timestamp": timestamp,
                "error": f"Detection database not found: {db_path}",
            },
            ensure_ascii=False,
        )

    with open(db_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    for record in database.get("frames", []):
        if str(record.get("clip_id")) == str(clip_id) and str(record.get("timestamp")) == str(timestamp):
            payload: Dict[str, Any] = {"found": True, **record}
            logger.info("[tool] retrieve_detection_results found clip_id=%s timestamp=%s", clip_id, timestamp)
            return json.dumps(payload, ensure_ascii=False)

    logger.info("[tool] retrieve_detection_results found no match clip_id=%s timestamp=%s", clip_id, timestamp)
    return json.dumps(
        {
            "found": False,
            "clip_id": clip_id,
            "timestamp": timestamp,
            "error": "No detection result found for clip_id and timestamp.",
        },
        ensure_ascii=False,
    )
