import os
import time
from typing import Literal
import cv2
import numpy as np

TrafficLightState = Literal["RED", "YELLOW", "GREEN", "UNKNOWN"]


class TrafficLightClassifier:
    """
    Adaptive spatial classifier for traffic lights using Dynamic Brightness.
    - Vertical (h > w):   Top (Red)    | Middle (Yellow) | Bottom (Green)
    - Horizontal (w > h): Left (Red)   | Center (Yellow) | Right (Green)
    """

    def __init__(self, min_brightness_threshold: int = 140, debug_export_dir: str = "test/output/debug_slots"):
        self.min_v = min_brightness_threshold
        self.debug_dir = debug_export_dir
        
        # Create debug directory if it doesn't exist
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

    def get_state(self, crop_bgr: np.ndarray, light_id: str = "unknown") -> TrafficLightState:
        if crop_bgr is None or crop_bgr.size == 0:
            return "UNKNOWN"

        h, w = crop_bgr.shape[:2]
        if h < 6 or w < 6:
            return "UNKNOWN"

        # ---------------------------------------------------------
        # 1. TRIM MARGINS to remove yellow backplates and sky noise
        # ---------------------------------------------------------
        is_vertical = h >= w
        
        if is_vertical and w > 10:
            # Vertical lights: trim 25% off left and right
            trim_x = int(w * 0.25)
            working_crop = crop_bgr[:, trim_x : w - trim_x]
        elif not is_vertical and h > 10:
            # Horizontal lights: trim 25% off top and bottom
            trim_y = int(h * 0.25)
            working_crop = crop_bgr[trim_y : h - trim_y, :]
        else:
            working_crop = crop_bgr

        wh, ww = working_crop.shape[:2]

        # 2. Slice into Top/Mid/Bot or Left/Center/Right
        if is_vertical:
            slot_size = wh // 3
            slot_area = slot_size * ww
            red_crop = working_crop[0:slot_size, :]
            yellow_crop = working_crop[slot_size:2 * slot_size, :]
            green_crop = working_crop[2 * slot_size:wh, :]
        else:
            slot_size = ww // 3
            slot_area = slot_size * wh
            red_crop = working_crop[:, 0:slot_size]
            yellow_crop = working_crop[:, slot_size:2 * slot_size]
            green_crop = working_crop[:, 2 * slot_size:ww]

        # ---------------------------------------------------------
        # 3. DYNAMIC BRIGHTNESS THRESHOLDING
        # ---------------------------------------------------------
        hsv_full = cv2.cvtColor(working_crop, cv2.COLOR_BGR2HSV)
        max_v = np.percentile(hsv_full[:, :, 2], 95)
        dynamic_min_v = max(80, int(max_v) - 80)

        # RED: Widen Hue down to 150 to catch magenta/burgundy, drop Saturation to 10
        red_hsv = cv2.cvtColor(red_crop, cv2.COLOR_BGR2HSV)
        r_mask1 = cv2.inRange(red_hsv, np.array([0, 10, dynamic_min_v]), np.array([15, 255, 255]))
        r_mask2 = cv2.inRange(red_hsv, np.array([150, 10, dynamic_min_v]), np.array([180, 255, 255]))
        red_mask = r_mask1 | r_mask2
        red_score = cv2.countNonZero(red_mask)

        # YELLOW: Raise Saturation floor to 60. True yellow traffic lights are highly 
        # saturated; this strict floor prevents warm sun reflections on black plastic 
        # from registering as yellow pixels.
        yellow_hsv = cv2.cvtColor(yellow_crop, cv2.COLOR_BGR2HSV)
        y_mask = cv2.inRange(yellow_hsv, np.array([15, 60, dynamic_min_v]), np.array([35, 255, 255]))
        yellow_score = cv2.countNonZero(y_mask)

        # GREEN: Standard bounds
        green_hsv = cv2.cvtColor(green_crop, cv2.COLOR_BGR2HSV)
        g_mask = cv2.inRange(green_hsv, np.array([40, 45, dynamic_min_v]), np.array([90, 255, 255]))
        green_score = cv2.countNonZero(g_mask)

        scores = {
            "RED": red_score,
            "YELLOW": yellow_score,
            "GREEN": green_score
        }

        best_color, max_score = max(scores.items(), key=lambda x: x[1])
        min_pixels = max(3, int(slot_area * 0.03)) 

        result_state = best_color if max_score >= min_pixels else "UNKNOWN"

        # ---------------------------------------------------------
        # DEBUG TASK 1: PRINT LOGS AND EXPORT FRAGMENTS
        # ---------------------------------------------------------
        print(f"\n[Classifier Debug] Size: {ww}x{wh} (Trimmed) | Peak Brightness: {max_v:.1f} -> Threshold: {dynamic_min_v}")
        print(f"  -> RED Slot px:    {red_score}")
        print(f"  -> YELLOW Slot px: {yellow_score}")
        print(f"  -> GREEN Slot px:  {green_score}")
        print(f"  -> VERDICT: {result_state} (Needed {min_pixels} px)")

        try:
            r_mask_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
            y_mask_bgr = cv2.cvtColor(y_mask, cv2.COLOR_GRAY2BGR)
            g_mask_bgr = cv2.cvtColor(g_mask, cv2.COLOR_GRAY2BGR)

            if is_vertical:
                original_stack = np.vstack([red_crop, yellow_crop, green_crop])
                mask_stack = np.vstack([r_mask_bgr, y_mask_bgr, g_mask_bgr])
                comparison = np.hstack([original_stack, mask_stack])
            else:
                original_stack = np.hstack([red_crop, yellow_crop, green_crop])
                mask_stack = np.hstack([r_mask_bgr, y_mask_bgr, g_mask_bgr])
                comparison = np.vstack([original_stack, mask_stack])

            comparison_zoomed = cv2.resize(comparison, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
            
            import time
            ts = int(time.time() * 1000) % 100000
            file_name = f"{self.debug_dir}/light_{ts}_{result_state}.jpg"
            cv2.imwrite(file_name, comparison_zoomed)
        except Exception as e:
            print(f"Failed to export debug image: {e}")

        return result_state