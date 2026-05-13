import json

from agent.tools.detection_tools import retrieve_detection_results


def test_retrieve_detection_results_by_clip_id_and_timestamp(monkeypatch, tmp_path):
    db_dir = tmp_path / "data" / "detection_db"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "detection_results.json"
    db_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "clip_id": "clip_1",
                        "timestamp": "100",
                        "detections": {
                            "traffic_lights": [
                                {"light_id": "light_1", "color": "red", "bbox": [1, 2, 3, 4]}
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.tools.detection_tools.get_abs_path", lambda path: str(tmp_path / path))

    result = json.loads(retrieve_detection_results.invoke({"clip_id": "clip_1", "timestamp": "100"}))

    assert result["found"] is True
    assert result["detections"]["traffic_lights"][0]["color"] == "red"


def test_retrieve_detection_results_returns_not_found_for_missing_frame(monkeypatch, tmp_path):
    db_dir = tmp_path / "data" / "detection_db"
    db_dir.mkdir(parents=True)
    (db_dir / "detection_results.json").write_text(json.dumps({"frames": []}), encoding="utf-8")
    monkeypatch.setattr("agent.tools.detection_tools.get_abs_path", lambda path: str(tmp_path / path))

    result = json.loads(retrieve_detection_results.invoke({"clip_id": "missing", "timestamp": "999"}))

    assert result["found"] is False
    assert "No detection result" in result["error"]
