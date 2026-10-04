import time
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

        # Temporal state tracking
        self.stationary_track_timers = {}
        self.parking_time_threshold_seconds = 10.0

    def check_violations(
        self, 
        tracked_vehicles: List[Dict[str, Any]], 
        light_state: Optional[str], 
        timestamp: str
    ) -> List[Dict[str, Any]]:
        """Evaluates traffic rules using purely spatial and temporal telemetry."""
        violations = []
        active_tracks = set()

        for v in tracked_vehicles:
            track_id = v["track_id"]
            active_tracks.add(track_id)
            history = v.get("history", [])
            bottom_center = v.get("bottom_center")
            
            # Require at least 2 points to form a trajectory line
            if not bottom_center or len(history) < 2:
                continue

            # Look back up to 15 frames to ensure the line segment is long enough
            lookback_frames = min(15, len(history))
            past_pt = history[-lookback_frames]
            
            # Guard against zero-length lines (stationary vehicles)
            if past_pt == bottom_center:
                continue

            trajectory = LineString([past_pt, bottom_center])

            # 1. Check Red Light Running
            rlr_violation = self._check_red_light_run(v, trajectory, light_state, timestamp)
            if rlr_violation:
                violations.append(rlr_violation)

        self._cleanup_stale_timers(active_tracks)
        return violations

    def _check_red_light_run(
        self, 
        vehicle: Dict[str, Any], 
        trajectory: LineString, 
        light_state: Optional[str],
        timestamp: str
    ) -> Optional[Dict[str, Any]]:
        
        if not self.stop_line or light_state != "RED":
            return None

        veh_direction = vehicle.get("direction", "UNKNOWN")

        # DIRECTION FILTER: Ignore explicit cross-traffic, but allow "UNKNOWN"
        if self.stop_line_direction and veh_direction != "UNKNOWN":
            if veh_direction != self.stop_line_direction:
                return None

        # Geometrically test if the trajectory line segment crosses the stop line
        if trajectory.intersects(self.stop_line):
            logger.info(f"🚨 RED LIGHT RUN DETECTED: Track #{vehicle['track_id']} ({vehicle['class_name']})")
            return {
                "type": "RED_LIGHT_RUN",
                "violation_flag": True,
                "track_id": vehicle["track_id"],
                "class_name": vehicle["class_name"],
                "timestamp": timestamp
            }
            
        return None

    def _cleanup_stale_timers(self, active_tracks: set):
        stale_tracks = set(self.stationary_track_timers.keys()) - active_tracks
        for stale_id in stale_tracks:
            del self.stationary_track_timers[stale_id]