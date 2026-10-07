from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.rules import TrafficRuleEngine


def resolve_path(input_path: str | Path, default_dir: Path) -> Path:
    """Resolve a path: check if it exists as-is, else check in default_dir."""
    p = Path(input_path)
    if p.exists():
        return p.resolve()
    fallback = default_dir / input_path
    if fallback.exists():
        return fallback.resolve()
    return p.resolve()


def test_rules_from_json(config_input: str | Path, json_input: str | Path):
    config_path = resolve_path(config_input, PROJECT_ROOT / "configs" / "cameras")
    json_path = resolve_path(json_input, PROJECT_ROOT / "test" / "output")

    if not config_path.exists():
        print(f"❌ Error: Config file not found at: {config_path}")
        return

    if not json_path.exists():
        print(f"❌ Error: Telemetry file not found at: {json_path}")
        return

    print(f"Using Config:    {config_path}")
    print(f"Using Telemetry: {json_path}\n" + "=" * 50)

    with open(config_path, "r", encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)

    # 1. Initialize engine with spatial zones for parking/restricted checks
    rule_engine = TrafficRuleEngine(
        stop_line=camera_config.get("stop_line"),
        stop_line_direction=camera_config.get("stop_line_direction"),
        restricted_zones=camera_config.get("restricted_zones"),
        no_parking_zones=camera_config.get("no_parking_zones")
    )

    # Track by (violation_type, car_id) so we don't spam the console, 
    # but still catch distinct violations for the same car
    reported_violations = set()
    frame_count = 0

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            frame_data = json.loads(line)
            frame_count += 1

            # 2. Extract vehicles, signs, and light state
            vehicles = frame_data.get("telemetry", {}).get("vehicles", [])
            signs = frame_data.get("telemetry", {}).get("signs", [])
            light_state = frame_data.get("traffic_light_state")
            timestamp = frame_data.get("timestamp")

            # 3. Evaluate rules (now passing detected_signs)
            violations = rule_engine.check_violations(
                tracked_vehicles=vehicles,
                light_state=light_state,
                timestamp=timestamp,
                detected_signs=signs
            )

            for v in violations:
                violation_type = v["type"]
                car_number = v["track_id"]
                violation_key = (violation_type, car_number)

                # Only print the payload the first time this specific violation occurs for this car
                if violation_key not in reported_violations:
                    reported_violations.add(violation_key)
                    print(f"🚨 VIOLATION: {violation_type}")
                    print(f"   Car Number: #{car_number} ({v['class_name']})")
                    print(f"   Time:       {v['timestamp']}")
                    print("-" * 50)

    print(f"\nFinished parsing {frame_count} frames.\n" + "=" * 50)

    # 4. Generate summary report of [Type + Car ID]
    if reported_violations:
        print("Summary of Violations:")
        # Sort alphabetically by violation type, then numerically by car ID
        sorted_violations = sorted(list(reported_violations), key=lambda x: (x[0], x[1]))
        for v_type, c_id in sorted_violations:
            print(f"  - {v_type} | Car #{c_id}")
    else:
        print("Violations Detected: None")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate traffic violations using camera YAML config and telemetry JSONL."
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path or filename of the camera YAML configuration (e.g. cam_01.yaml).",
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path or filename of the telemetry JSON/JSONL file (e.g. telemetry.jsonl).",
    )

    args = parser.parse_args()
    test_rules_from_json(args.yaml_file, args.json_file)