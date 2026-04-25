"""
Multimodal annotation data schema definitions.
Uses Pydantic to define structured annotation output format for VLM.
Supports flexible extension and iteration.

通用街景要素识别DrivingEnvironment：Weather, Time, Visibility, Road_type
静态道路场景理解：Lane、Traffic Sign、Traffic Light
动态交通要素理解：Dynamic Object
场景空间关系理解：Lane2Sign, Lane2Light, Lane2Obj, Sign2Light
障碍物交互决策：Interaction Decision-Making
自车行为决策：Ego Vehicle Behavior Decision-Making
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class DrivingEnvironment(BaseModel):
    """Driving environment information."""
    weather: Literal["sunny", "rainy", "cloudy", "foggy", "snowy", "overcast"] = Field(
        ..., description="Weather condition"
    )
    time: Literal["day", "night", "dawn", "dusk"] = Field(
        ..., description="Time of day"
    )
    visibility: Literal["good", "moderate", "poor"] = Field(
        None, description="Visibility level"
    )
    road_type: Literal["Highway", "Urban_Road", "Rural_Road", "Tunnel", "Bridge", "Parking_Lot"] = Field(
        ..., description="Road type"
    )


class LaneInfo(BaseModel):
    """Lane information."""
    lane_id: str = Field(description="Lane identifier, e.g., Lane1, Lane2")
    bbox: List[float] = Field(description="Bounding box coordinates [x1, y1, x2, y2]")
    is_ego: bool = Field(description="Whether this is the ego vehicle's lane")
    lane_type: Literal["driving_lane", "shoulder", "emergency_lane", "bus_lane", "bike_lane", "turning_lane"] = Field(
        None, description="Lane type"
    )
    direction: Literal["same_direction", "opposite_direction"] = Field(
        None, description="Lane direction relative to ego vehicle"
    )


class TrafficSignInfo(BaseModel):
    """Traffic sign information."""
    sign_id: str = Field(description="Sign identifier, e.g., sign_1, sign_2")
    bbox: List[float] = Field(description="Bounding box coordinates [x1, y1, x2, y2]")
    type: Literal[
        "speed_limit", "stop", "yield", "no_entry", "no_parking",
        "warning", "direction", "pedestrian_crossing", "construction"
    ] = Field(..., description="Sign type")
    content: Optional[str] = Field(
        None, description="Sign content or value, e.g., '60' for speed limit, 'STOP' for stop sign"
    )
    applicable_lanes: Optional[List[str]] = Field(
        None, description="List of lane IDs this sign applies to"
    )


class TrafficLightInfo(BaseModel):
    """Traffic light information."""
    light_id: str = Field(description="Light identifier, e.g., light_1, light_2")
    bbox: List[float] = Field(description="Bounding box coordinates [x1, y1, x2, y2]")
    color: Literal["red", "green", "yellow", "off"] = Field(
        ..., description="Current color"
    )
    state: Literal["solid", "flashing", "arrow_left", "arrow_right", "arrow_straight", "arrow_u_turn"] = Field(
        None, description="Light state"
    )
    countdown: Optional[int] = Field(
        None, ge=0, le=999, description="Countdown timer in seconds, if available"
    )
    applicable_lanes: Optional[List[str]] = Field(
        None, description="List of lane IDs this light controls"
    )


class VehicleObject(BaseModel):
    """Vehicle object information."""
    obj_id: str = Field(..., description="Object identifier, e.g., vehicle_1, vehicle_2")
    bbox: List[float] = Field(description="Bounding box coordinates [x1, y1, x2, y2]")
    type: Literal[
        "sedan", "SUV", "truck", "bus", "motorcycle", "bicycle",
        "van", "taxi", "police_car", "ambulance", "fire_truck"
    ] = Field(..., description="Vehicle type")
    orientation: Literal[
        "front", "rear", "left", "right", "front_left", "front_right", "rear_left", "rear_right"] = Field(
        None, description="Orientation relative to ego vehicle"
    )
    behavior: Literal["stationary", "moving", "turning", "accelerating", "decelerating", "parked"] = Field(
        None, description="Behavior state"
    )
    interaction_with_ego: Literal["bypass", "yield", "ignore", "follow"] = Field(
        None, description="which action ego vehicle should perform"
    )


class StaticObject(BaseModel):
    """Static obstacle information."""
    obj_id: str = Field(description="Object identifier, e.g., obj_1, obj_2")
    bbox: List[float] = Field(description="Bounding box coordinates [x1, y1, x2, y2]")
    type: Literal[
        "plastic_bag", "cardboard_box", "cone", "barrier", "rock",
        "tree_branch", "tire", "debris", "animal"
    ] = Field(..., description="Object type")
    compressible: Literal["Yes", "No", "Unknown"] = Field(
        None, description="Compressibility"
    )
    position: Literal["front", "rear", "left", "right", "front_left", "front_right", "rear_left", "rear_right"] = Field(
        None, description="Relative position to ego vehicle"
    )



class EgoVehicle(BaseModel):
    """Ego vehicle behavior and action information."""
    lateral_action: Literal[
        "lane_change_left",
        "lane_change_right",
        "nudge_left",
        "nudge_right",
        "keep_lane"
    ] = Field(
        ..., description="Lateral action: lane change left/right, nudge left/right to bypass obstacles, or keep current lane"
    )
    longitudinal_action: Literal[
        "accelerate",
        "maintain_speed",
        "decelerate"
    ] = Field(
        ..., description="Longitudinal action: accelerate, maintain current speed, or decelerate"
    )
    reason: Optional[str] = Field(
        None, description="Reason for the action, e.g., 'avoid obstacle', 'follow traffic light', 'merge into traffic'"
    )


class AnnotationResult(BaseModel):
    """Complete annotation result."""
    driving_environment: DrivingEnvironment = Field(..., description="Driving environment information")
    lanes: Optional[List[LaneInfo]] = Field(None, description="List of lane information")
    traffic_signs: Optional[List[TrafficSignInfo]] = Field(None, description="List of traffic signs")
    traffic_lights: Optional[List[TrafficLightInfo]] = Field(None, description="List of traffic lights")
    vehicles: Optional[List[VehicleObject]] = Field(None, description="List of dynamic vehicle objects")
    static_objects: Optional[List[StaticObject]] = Field(None, description="List of static obstacles")
    ego_vehicle: Optional[EgoVehicle] = Field(None, description="Ego vehicle behavior and actions")
    scene_description: Optional[str] = Field(None, description="Natural language description of the scene")
    class Config:
        json_schema_extra = {
            "example": {
                "driving_environment": {
                    "weather": "sunny",
                    "time": "day",
                    "visibility": "good",
                    "road_type": "Highway"
                },
                "lanes": [
                    {"lane_id": "Lane1", "is_ego": True, "lane_type": "driving_lane", "bbox": [0, 400, 800, 600]},
                    {"lane_id": "Lane2", "is_ego": False, "lane_type": "driving_lane", "bbox": [800, 400, 1600, 600]}
                ],
                "traffic_signs": [
                    {
                        "sign_id": "sign_1",
                        "type": "speed_limit",
                        "content": "60",
                        "bbox": [100, 50, 150, 100],
                        "applicable_lanes": ["Lane1", "Lane2"]
                    }
                ],
                "traffic_lights": [
                    {
                        "light_id": "light_1",
                        "color": "red",
                        "state": "solid",
                        "bbox": [700, 20, 750, 80],
                        "countdown": 15,
                        "applicable_lanes": ["Lane1"]
                    }
                ],
                "vehicles": [
                    {
                        "obj_id": "vehicle_1",
                        "type": "sedan",
                        "bbox": [300, 200, 500, 350],
                        "orientation": "front",
                        "behavior": "moving",
                        "interaction_with_ego": "follow"
                    }
                ],
                "static_objects": [
                    {
                        "obj_id": "obj_1",
                        "type": "plastic_bag",
                        "compressible": "Yes",
                        "position": "front_right",
                        "bbox": [600, 300, 650, 340]
                    }
                ],
                "ego_vehicle": {
                    "lateral_action": "nudge_left",
                    "longitudinal_action": "decelerate",
                    "reason": "Avoid foam box in right lane ahead"
                },
                "scene_description": "Sunny day, ego vehicle driving on urban expressway with a white foam box in the right lane ahead. Ego vehicle should slow down and avoid"
            }
        }