# Few-shot Case: Red Light Stop

Input:
Cloudy dusk at an urban intersection. The ego vehicle approaches a red traffic light. Two sedans are queued ahead in the same lane. Ego should keep lane, decelerate, and stop behind the queue.

Expected annotation pattern:
```json
{
  "driving_environment": {
    "weather": "cloudy",
    "time": "dusk",
    "visibility": "moderate",
    "road_type": "Urban_Road"
  },
  "traffic_lights": [
    {
      "light_id": "light_1",
      "bbox": [700, 30, 750, 90],
      "color": "red",
      "state": "solid",
      "applicable_lanes": ["Lane1"]
    }
  ],
  "vehicles": [
    {
      "obj_id": "vehicle_1",
      "bbox": [300, 240, 480, 380],
      "type": "sedan",
      "orientation": "rear",
      "behavior": "stationary",
      "interaction_with_ego": "follow"
    }
  ],
  "ego_vehicle": {
    "lateral_action": "keep_lane",
    "longitudinal_action": "decelerate",
    "reason": "Stop for red traffic light and queued vehicles"
  }
}
```

Guidance:
- A red light should normally map to `longitudinal_action: "decelerate"`.
- If vehicles are queued ahead, mark them as `stationary` and `interaction_with_ego: "follow"`.
- Keep `traffic_lights.color` constrained to `red`, `green`, `yellow`, or `off`.
