# Few-shot Case: Soft Obstacle Avoidance

Input:
Sunny day on an urban expressway. Ego vehicle is in the middle lane. A white foam box is ahead near the right edge of the ego lane. No vehicles are immediately adjacent. Ego should slow down and slightly move left to avoid the object.

Expected annotation pattern:
```json
{
  "driving_environment": {
    "weather": "sunny",
    "time": "day",
    "visibility": "good",
    "road_type": "Highway"
  },
  "static_objects": [
    {
      "obj_id": "obj_1",
      "bbox": [610, 315, 660, 355],
      "type": "plastic_bag",
      "compressible": "Yes",
      "position": "front_right"
    }
  ],
  "ego_vehicle": {
    "lateral_action": "nudge_left",
    "longitudinal_action": "decelerate",
    "reason": "Avoid a soft obstacle near the right side of the ego lane"
  }
}
```

Guidance:
- Use `nudge_left` or `nudge_right` for small avoidance while staying in the lane.
- Use `lane_change_left/right` only when the vehicle must fully enter another lane.
- A foam box or plastic bag should usually be `compressible: "Yes"` unless visual evidence indicates a rigid container.
