import sys
import cv2
import yaml
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class VideoCalibrator:
    def __init__(self, video_path: str, output_yaml_name: str = "cam_00.yaml"):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            print(f"Error: File not found: {self.video_path}")
            sys.exit(1)

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"Error: Unable to open video: {self.video_path}")
            sys.exit(1)

        self.output_yaml_path = PROJECT_ROOT / "configs" / "cameras" / output_yaml_name
        self.output_yaml_path.parent.mkdir(parents=True, exist_ok=True)

        self.window_name = "Traffic Vision Calibration Tool"
        self.native_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.native_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.native_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30

        self.base_frame = None
        self.mode = "SEEK"  # "SEEK" -> "CALIBRATE"
        
        # Added the new zones to the stages pipeline
        self.stages = [
            "TRAFFIC_LIGHTS", 
            "STOP_LINE", 
            "RESTRICTED_ZONE", 
            "NO_PARKING_ZONE", 
            "SPEED_ZONE", 
            "DONE"
        ]
        self.stage_idx = 0
        self.current_points = []

        self.config = {
            "camera_id": self.output_yaml_path.stem,
            "stream_fps": self.native_fps,
            "traffic_light_rois": [],  
            "stop_line": [],
            "restricted_zones": {"zone_1": []},
            "stop_line_direction": "NORTH",  # Added default direction
            "no_parking_zones": {"zone_1": []},
            "speed_zones": {"zone_1": []}
        }

    def append_instructions(self, disp: np.ndarray) -> np.ndarray:
        bar = np.zeros((65, self.native_w, 3), dtype=np.uint8)
        
        if self.mode == "SEEK":
            text = "PREVIEW MODE: [SPACE] Pause/Resume | [D] Step 1 Frame | [C] Calibrate This Frame"
            color = (0, 255, 255)
        else:
            stage = self.stages[self.stage_idx]
            instructions = {
                "TRAFFIC_LIGHTS": f"STEP 1/5: Click 4 CORNERS of light (TL->TR->BR->BL) ({len(self.current_points)}/4). [SPACE] Next.",
                "STOP_LINE": f"STEP 2/5: Click 2 points for Stop Line ({len(self.current_points)}/2).",
                "RESTRICTED_ZONE": f"STEP 3/5: Click polygon vertices for RESTRICTED ZONE ({len(self.current_points)} pts). [SPACE] Next.",
                "NO_PARKING_ZONE": f"STEP 4/5: Click polygon vertices for NO PARKING ZONE ({len(self.current_points)} pts). [SPACE] Next.",
                "SPEED_ZONE": f"STEP 5/5: Click polygon vertices for SPEED ZONE ({len(self.current_points)} pts). [SPACE] Finish.",
                "DONE": "CALIBRATION FINISHED: Press [S] to Save YAML & Exit | [ESC] Cancel"
            }
            text = instructions[stage]
            color = (0, 255, 0)

        cv2.putText(bar, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return np.vstack((disp, bar))

    def _draw_polygon_zone(self, disp, zone_dict, current_stage_name, color, label):
        """Helper function to draw completed zones and in-progress polygons."""
        # Draw completed
        zone_pts = zone_dict.get("zone_1", [])
        if len(zone_pts) >= 3:
            pts = np.array(zone_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(disp, label, (zone_pts[0][0], max(20, zone_pts[0][1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw in-progress
        if self.mode == "CALIBRATE" and self.stages[self.stage_idx] == current_stage_name:
            for pt in self.current_points:
                cv2.circle(disp, pt, 4, color, -1)
            if len(self.current_points) > 1:
                pts = np.array(self.current_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(disp, [pts], isClosed=False, color=color, thickness=1)

    def render_calibration(self) -> np.ndarray:
        disp = self.base_frame.copy()

        # 1. Draw Traffic Light Polygons (Yellow)
        for idx, quad in enumerate(self.config["traffic_light_rois"]):
            pts = np.array(quad, np.int32).reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(disp, f"L{idx}", (quad[0][0], max(15, quad[0][1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if self.mode == "CALIBRATE" and self.stages[self.stage_idx] == "TRAFFIC_LIGHTS":
            for pt in self.current_points:
                cv2.circle(disp, pt, 4, (0, 255, 255), -1)
            if len(self.current_points) > 1:
                pts = np.array(self.current_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(disp, [pts], isClosed=False, color=(0, 255, 255), thickness=1)

        # 2. Draw Stop Line (Red)
        if len(self.config["stop_line"]) == 2:
            p1 = tuple(self.config["stop_line"][0])
            p2 = tuple(self.config["stop_line"][1])
            cv2.line(disp, p1, p2, (0, 0, 255), 3)
            cv2.putText(disp, "STOP LINE", (p1[0], max(20, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif self.mode == "CALIBRATE" and self.stages[self.stage_idx] == "STOP_LINE":
            if len(self.current_points) == 1:
                cv2.circle(disp, self.current_points[0], 4, (0, 0, 255), -1)

        # 3. Draw Restricted Zones (Magenta)
        self._draw_polygon_zone(disp, self.config["restricted_zones"], "RESTRICTED_ZONE", (255, 0, 255), "RESTRICTED")

        # 4. Draw No Parking Zones (Blue)
        self._draw_polygon_zone(disp, self.config["no_parking_zones"], "NO_PARKING_ZONE", (255, 0, 0), "NO PARKING")

        # 5. Draw Speed Zones (Cyan)
        self._draw_polygon_zone(disp, self.config["speed_zones"], "SPEED_ZONE", (255, 255, 0), "SPEED ZONE")

        return disp

    def on_mouse(self, event, x, y, flags, param):
        if self.mode != "CALIBRATE" or event != cv2.EVENT_LBUTTONDOWN or y >= self.native_h:
            return

        stage = self.stages[self.stage_idx]

        if stage == "TRAFFIC_LIGHTS":
            self.current_points.append([int(x), int(y)])
            if len(self.current_points) == 4:
                self.config["traffic_light_rois"].append(self.current_points)
                self.current_points = []

        elif stage == "STOP_LINE":
            self.current_points.append([int(x), int(y)])
            if len(self.current_points) == 2:
                self.config["stop_line"] = self.current_points
                self.current_points = []
                self.stage_idx += 1

        elif stage in ["RESTRICTED_ZONE", "NO_PARKING_ZONE", "SPEED_ZONE"]:
            self.current_points.append([int(x), int(y)])

        disp = self.render_calibration()
        cv2.imshow(self.window_name, self.append_instructions(disp))

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        is_paused = False

        while self.mode == "SEEK":
            if not is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.base_frame = frame

            view = self.append_instructions(self.base_frame.copy())
            cv2.imshow(self.window_name, view)

            key = cv2.waitKey(30 if not is_paused else 10) & 0xFF
            if key == 27:  
                self.cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):  
                is_paused = not is_paused
            elif key in (ord('d'), ord('D')):  
                ret, frame = self.cap.read()
                if ret:
                    self.base_frame = frame
                is_paused = True
            elif key in (ord('c'), ord('C')):  
                self.mode = "CALIBRATE"
                break

        disp = self.render_calibration()
        cv2.imshow(self.window_name, self.append_instructions(disp))

        while self.mode == "CALIBRATE":
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  
                break
            
            if key == ord(' '):
                stage = self.stages[self.stage_idx]
                
                if stage == "TRAFFIC_LIGHTS":
                    self.current_points = []
                    self.stage_idx += 1
                
                elif stage == "RESTRICTED_ZONE":
                    if len(self.current_points) >= 3:
                        self.config["restricted_zones"]["zone_1"] = self.current_points
                    else:
                        self.config["restricted_zones"] = {}
                    self.current_points = []
                    self.stage_idx += 1
                    
                elif stage == "NO_PARKING_ZONE":
                    if len(self.current_points) >= 3:
                        self.config["no_parking_zones"]["zone_1"] = self.current_points
                    else:
                        self.config["no_parking_zones"] = {}
                    self.current_points = []
                    self.stage_idx += 1
                    
                elif stage == "SPEED_ZONE":
                    if len(self.current_points) >= 3:
                        self.config["speed_zones"]["zone_1"] = self.current_points
                    else:
                        self.config["speed_zones"] = {}
                    self.current_points = []
                    self.stage_idx += 1

                disp = self.render_calibration()
                cv2.imshow(self.window_name, self.append_instructions(disp))

            if key in (ord('s'), ord('S')) and self.stages[self.stage_idx] == "DONE":
                self.save_yaml()
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def save_yaml(self):
        # Ensure empty dicts if no zones were drawn, matching the requested output format
        if not self.config["restricted_zones"].get("zone_1"):
            self.config["restricted_zones"] = {}
        if not self.config["no_parking_zones"].get("zone_1"):
            self.config["no_parking_zones"] = {}
        if not self.config["speed_zones"].get("zone_1"):
            self.config["speed_zones"] = {}

        with open(self.output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, sort_keys=False, default_flow_style=None)

        print("\n" + "="*50)
        print(f"CALIBRATION SAVED TO: {self.output_yaml_path}")
        print("="*50)
        with open(self.output_yaml_path, "r") as f:
            print(f.read())


if __name__ == "__main__":
    target_video = "raw_data/videos/WeixinVideos2026-10-06_214655_668.mp4"
    if len(sys.argv) > 1:
        target_video = sys.argv[1]
    
    calibrator = VideoCalibrator(video_path=target_video, output_yaml_name="cam_01.yaml")
    calibrator.run()