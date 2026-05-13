# Few-shot Case: Emergency Vehicle Yield

Input:
Snowy day on an urban road. An ambulance with flashing lights approaches from the rear-left lane. A police car is parked on the right shoulder. Ego should yield by slowing down and moving slightly right when safe.

Expected annotation pattern:
```json
{
  "driving_environment": {
    "weather": "snowy",
    "time": "day",
    "visibility": "good",
    "road_type": "Urban_Road"
  },
  "vehicles": [
    {
      "obj_id": "vehicle_1",
      "bbox": [120, 220, 360, 420],
      "type": "ambulance",
      "orientation": "rear",
      "behavior": "moving",
      "interaction_with_ego": "yield"
    },
    {
      "obj_id": "vehicle_2",
      "bbox": [1280, 370, 1540, 530],
      "type": "police_car",
      "orientation": "rear",
      "behavior": "parked",
      "interaction_with_ego": "ignore"
    }
  ],
  "ego_vehicle": {
    "lateral_action": "nudge_right",
    "longitudinal_action": "decelerate",
    "reason": "Yield to emergency vehicle approaching from rear-left"
  }
}
```

Guidance:
- Emergency vehicles should usually trigger `interaction_with_ego: "yield"`.
- If the ego vehicle only creates space and does not fully change lanes, use `nudge_right` or `nudge_left`.
- Parked police cars on shoulder usually have `interaction_with_ego: "ignore"` unless they block the ego lane.
