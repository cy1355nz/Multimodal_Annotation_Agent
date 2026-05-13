# Annotation Guidelines

Object ids:
- Lanes: `Lane1`, `Lane2`, `Lane3`, ...
- Traffic signs: `sign_1`, `sign_2`, ...
- Traffic lights: `light_1`, `light_2`, ...
- Vehicles: `vehicle_1`, `vehicle_2`, ...
- Static objects: `obj_1`, `obj_2`, ...

Ego action rules:
- Use `keep_lane` when no lateral avoidance or lane change is needed.
- Use `nudge_left` or `nudge_right` for slight lateral offsets within the current lane.
- Use `lane_change_left` or `lane_change_right` only for a full lane change.
- Use `decelerate` for red lights, stopped leading vehicles, poor visibility, construction zones, and uncertain obstacles.
- Use `maintain_speed` only when the scene is stable and no immediate hazard requires slowing down.

Uncertainty rules:
- If a field is optional and uncertain, prefer `null` or omission.
- Do not invent traffic signs, traffic lights, or vehicles that are not visible or described.
- Bbox values may be approximate in this demo, but must always be four numbers: `[x1, y1, x2, y2]`.

Schema alignment:
- Use only enum values defined in `schemas/annotation_schema.py`.
- Do not output `car`; choose a supported vehicle subtype such as `sedan`, `SUV`, `truck`, or `van`.
- Static road debris should map to one of the supported `StaticObject.type` values.
