Here is a systematic 4-phase plan to debug and stabilize your pipeline based on the issues found in the telemetry output.

---

### Phase 1: Fix Visual Alignment & Traffic Light Classifier

The traffic light state is permanently `UNKNOWN`. We must verify whether the crop coordinates match reality or if the HSV threshold is rejecting the signal.

1. **Verify Crop Alignment:**
* In `pipeline/engine.py`, add a temporary debug save on frame 0:
```python
if frame_idx == 0:
    for idx, roi in enumerate(self.light_rois):
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(f"test/output/debug_light_crop_{idx}.jpg", crop)

```


* Inspect the exported images. If they show trees, asphalt, or sky instead of the light housing, re-run `tools/calibrate.py` using the exact resolution of `1de6cf...mp4`.


2. **Tune HSV Mask Sensitivity:**
* If the crop correctly contains the light, log raw pixel counts inside `TrafficLightClassifier.get_state`:
```python
print(f"[DEBUG HSV] Counts: {counts}")

```


* If the dominant color count is between 5 and 19 pixels (below the default `> 20` threshold), lower the threshold to `> 8` or adapt it relative to crop area:
```python
min_pixels = max(5, int(crop_bgr.shape[0] * crop_bgr.shape[1] * 0.02))

```





---

### Phase 2: Implement Direction & Velocity Calculation

The tracker produces history points, but velocity and direction calculations are missing or unassigned.

1. **Implement Trajectory Vector Math in `pipeline/tracker.py`:**
* Calculate displacement over a rolling window (e.g., last 5 frames) to suppress frame-to-frame pixel jitter:
```python
import math

def compute_motion_vectors(history, fps=30):
    if len(history) < 2:
        return 0.0, "STATIONARY"

    # Compare current position with position 3-5 frames ago
    p_curr = history[-1]
    p_prev = history[-min(5, len(history))]

    dx = p_curr[0] - p_prev[0]
    dy = p_curr[1] - p_prev[1]
    dist = math.hypot(dx, dy)

    velocity_px = round(dist / min(5, len(history)), 2)

    if dist < 3.0:
        return 0.0, "STATIONARY"

    # Map coordinate deltas to cardinal/screen directions
    if abs(dy) > abs(dx):
        direction = "SOUTH" if dy > 0 else "NORTH"
    else:
        direction = "EAST" if dx > 0 else "WEST"

    return velocity_px, direction

```




2. **Cap Trajectory Memory:**
* Change `history` from an unbounded list to `collections.deque(maxlen=30)` to eliminate the memory leak.



---

### Phase 3: Eliminate Duplicate Detections (Car vs. Truck)

The detector triggers simultaneous, overlapping bounding boxes for both `car` and `truck` on the same physical vehicle (`track_id: 3` and `track_id: 7`).

1. **Enable Class-Agnostic NMS:**
* In `pipeline/tracker.py`, when calling `self.model.track()`, configure class-agnostic Non-Maximum Suppression so higher-confidence boxes suppress lower-confidence overlapping boxes regardless of vehicle label:
```python
results = self.model.track(
    source=frame,
    persist=True,
    conf=self.conf_threshold,
    iou=0.5,             # Tighten IoU threshold
    agnostic_nms=True,   # Suppresses car/truck cross-duplicates
    classes=[2, 3, 5, 7] # COCO: car, motorcycle, bus, truck
)

```





---

### Phase 4: Stabilize Initial Sign Perception

The sign detector missed the sign on frames 0–2 and only caught it at frame 3.

1. **Add Startup Warm-Up:**
* Update the caching condition in `pipeline/engine.py` to check the first 5 frames continuously until a confident detection is cached, or allow retrying for the first 10 frames before reverting to the 5-second interval:
```python
# Keep scanning early frames if nothing is cached yet, then stick to the interval
should_scan_signs = (frame_idx < 10 and not self.cached_signs) or (frame_idx % self.sign_update_interval == 0)

if should_scan_signs:
    sign_candidates = self.sign_detector.detect(frame)
    if sign_candidates:
        self.cached_signs = self.sign_classifier.classify(sign_candidates)

```





---

### Recommended Execution Order

1. **Step 1:** Implement the crop debug dump in `engine.py` and inspect `debug_light_crop_0.jpg`.
2. **Step 2:** Add `agnostic_nms=True` and `classes=[2, 3, 5, 7]` in `tracker.py` to fix the duplicate tracks immediately.
3. **Step 3:** Implement vector math and add `deque(maxlen=30)` in `tracker.py`.
4. **Step 4:** Re-run `python test/test_service.py` for 30 frames and verify the new JSONL records.