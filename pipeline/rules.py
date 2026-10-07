import math
from typing import Dict, List, Any, Optional
from shapely.geometry import Point, LineString, Polygon
from core.logger import logger

class TrafficRuleEngine:
    def __init__(
        self, 
        stop_line: List[list] = None, 
        stop_line_direction: str = None,
        restricted_zones: Dict[str, list] = None, 
        no_parking_zones: Dict[str, list] = None
    ):
        # Initialize Geometries
        self.stop_line = LineString(stop_line) if stop_line and len(stop_line) == 2 else None
        self.stop_line_direction = stop_line_direction
        
        self.restricted_zones = {name: Polygon(pts) for name, pts in (restricted_zones or {}).items() if len(pts) >= 3}
        self.no_parking_zones = {name: Polygon(pts) for name, pts in (no_parking_zones or {}).items() if len(pts) >= 3}

        # Temporal state tracking (Now using Frame Counts instead of time.time)
        self.stationary_track_frames = {}
        
        # TESTING THRESHOLD: 30 frames (about ~1.3 seconds at 23 FPS)
        # Change this back to ~230 frames (10 seconds) for production!
        self.parking_frame_threshold = 30 

        # Your updated labels
        self.no_parking_labels = {"pn", "no_parking", "p11"} 

    def check_violations(
        self, 
        tracked_vehicles: List[Dict[str, Any]], 
        light_state: Optional[str], 
        timestamp: str,
        detected_signs: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        violations = []
        active_tracks = set()

        # 1. Determine if a No Parking sign is currently visible
        no_parking_sign_present = False
        if detected_signs:
            for sign in detected_signs:
                if sign.get("class_name") in self.no_parking_labels:
                    no_parking_sign_present = True
                    break

        # 2. Evaluate all vehicles
        for v in tracked_vehicles:
            track_id = v["track_id"]
            active_tracks.add(track_id)
            bottom_center = v.get("bottom_center")
            history = v.get("history", [])
            
            if not bottom_center:
                continue

            # RULE 1: Red Light Running
            if len(history) >= 2:
                lookback_frames = min(15, len(history))
                past_pt = history[-lookback_frames]
                
                if past_pt != bottom_center:
                    trajectory = LineString([past_pt, bottom_center])
                    rlr_violation = self._check_red_light_run(v, trajectory, light_state, timestamp)
                    if rlr_violation:
                        violations.append(rlr_violation)

            # RULE 2: Illegal Parking
            parking_violation = self._check_parking_violation(v, no_parking_sign_present, timestamp)
            if parking_violation:
                violations.append(parking_violation)

        self._cleanup_stale_timers(active_tracks)
        return violations

    def _check_red_light_run(
        self, vehicle: Dict[str, Any], trajectory: LineString, light_state: Optional[str], timestamp: str
    ) -> Optional[Dict[str, Any]]:
        if not self.stop_line or light_state != "RED":
            return None

        veh_direction = vehicle.get("direction", "UNKNOWN")
        if self.stop_line_direction and veh_direction != "UNKNOWN":
            if veh_direction != self.stop_line_direction:
                return None

        if trajectory.intersects(self.stop_line):
            logger.info(f"🚨 RED LIGHT RUN DETECTED: Track #{vehicle['track_id']}")
            return {
                "type": "RED_LIGHT_RUN",
                "violation_flag": True,
                "track_id": vehicle["track_id"],
                "class_name": vehicle["class_name"],
                "timestamp": timestamp
            }
        return None

    def _check_parking_violation(
        self, vehicle: Dict[str, Any], no_parking_sign_present: bool, timestamp: str
    ) -> Optional[Dict[str, Any]]:
        
        track_id = vehicle["track_id"]

        if not self.no_parking_zones or not no_parking_sign_present:
            self._reset_parking_timer(track_id)
            return None

        # Check if the car is moving (velocity > 1.0 pixels per frame)
        velocity = vehicle.get("velocity_px_frame", 10.0)
        if velocity > 1.0:
            self._reset_parking_timer(track_id)
            return None
            
        # ---------------------------------------------------------
        # CHANGED: Calculate exact center-center of the bounding box
        # ---------------------------------------------------------
        box = vehicle.get("bbox", {})
        if not box:
            return None
            
        center_x = (box["x1"] + box["x2"]) / 2.0
        center_y = (box["y1"] + box["y2"]) / 2.0
        center_center = (center_x, center_y)
            
        # Geometrically test if the center of the car is inside the drawn polygon
        pt = Point(center_center)
        in_no_parking_zone = any(poly.contains(pt) for poly in self.no_parking_zones.values())

        if in_no_parking_zone:
            # If it just stopped, initialize its frame counter
            if track_id not in self.stationary_track_frames:
                self.stationary_track_frames[track_id] = 1
                return None
                
            # Increment the frame counter
            self.stationary_track_frames[track_id] += 1
            
            # If it has been parked longer than our frame limit
            if self.stationary_track_frames[track_id] >= self.parking_frame_threshold:
                logger.info(f"🚨 ILLEGAL PARKING DETECTED: Track #{track_id} ({vehicle['class_name']})")
                
                # Reset counter to 0 so we only fire one violation
                self.stationary_track_frames[track_id] = 0
                
                return {
                    "type": "ILLEGAL_PARKING",
                    "violation_flag": True,
                    "track_id": track_id,
                    "class_name": vehicle["class_name"],
                    "timestamp": timestamp
                }
        else:
            self._reset_parking_timer(track_id)
            
        return None

    def _reset_parking_timer(self, track_id: int):
        if track_id in self.stationary_track_frames:
            del self.stationary_track_frames[track_id]

    def _cleanup_stale_timers(self, active_tracks: set):
        stale_tracks = set(self.stationary_track_frames.keys()) - active_tracks
        for stale_id in stale_tracks:
            del self.stationary_track_frames[stale_id]