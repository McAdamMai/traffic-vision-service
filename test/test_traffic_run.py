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

    rule_engine = TrafficRuleEngine(
        stop_line=camera_config.get("stop_line"),
        stop_line_direction=camera_config.get("stop_line_direction"),
    )

    violating_cars = set()
    frame_count = 0

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            frame_data = json.loads(line)
            frame_count += 1

            vehicles = frame_data.get("telemetry", {}).get("vehicles", [])
            light_state = frame_data.get("traffic_light_state")
            timestamp = frame_data.get("timestamp")

            violations = rule_engine.check_violations(
                tracked_vehicles=vehicles,
                light_state=light_state,
                timestamp=timestamp,
            )

            for v in violations:
                car_number = v["track_id"]

                # Only print the payload the first time this specific car crosses
                if car_number not in violating_cars:
                    violating_cars.add(car_number)
                    print(f"🚨 VIOLATION: {v['type']}")
                    print(f"   Car Number: #{car_number} ({v['class_name']})")
                    print(f"   Time:       {v['timestamp']}")
                    print("-" * 50)

    print(f"\nFinished parsing {frame_count} frames.")

    if violating_cars:
        print(f"Violating Car Numbers: {sorted(list(violating_cars))}")
    else:
        print("Violating Car Numbers: None")


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