"""Offline controller handoff demo. Never actuates real hardware."""
import argparse
import csv
import json
from pathlib import Path
import cv2
from lane_interface import AutoDriveLaneProvider, observation_from_boundaries

def example_controller(obs):
    """Illustrative controller only. Replace with partner's steering/throttle logic."""
    if obs.status != "detected" or obs.error_normalized is None:
        return {"steering_normalized": 0.0, "throttle_normalized": 0.0, "mode": "STOP_UNCERTAIN"}
    steering = max(-1.0, min(1.0, 0.7 * obs.error_normalized))
    return {"steering_normalized": steering, "throttle_normalized": 0.2, "mode": "SIMULATED"}

def synthetic():
    w, h = 960, 540
    for center in (480, 510, 450, 530):
        left = [[center-120, 300], [center-120, 500]]
        right = [[center+120, 300], [center+120, 500]]
        yield observation_from_boundaries(
            {"left_lane_boundary": left, "right_lane_boundary": right},
            {"reason": "synthetic", "selected_channels": [1, 2]}, w, h)
    yield observation_from_boundaries({}, {"reason": "no_ego_pair"}, w, h)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--video")
    parser.add_argument("--repo", default="external/Ultra-Fast-Lane-Detection-v2")
    parser.add_argument("--weights", default="weights/culane_res18.pth")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--output", default="output/partner_observations.csv")
    args = parser.parse_args()
    if not args.synthetic and not args.video:
        parser.error("Specify --synthetic or --video")
    if args.synthetic:
        observations = synthetic()
    else:
        provider = AutoDriveLaneProvider(repo=args.repo, weights=args.weights)
        capture = cv2.VideoCapture(args.video)
        if not capture.isOpened():
            raise FileNotFoundError(args.video)
        def video_observations():
            try:
                for _ in range(args.max_frames):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    yield provider.process_frame(frame, allow_estimates=False)
            finally:
                capture.release()
        observations = video_observations()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = None
        for observation in observations:
            row = {**observation.as_dict(), **example_controller(observation)}
            row["selected_channels"] = json.dumps(row["selected_channels"])
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            print(observation.status, observation.error_normalized,
                  row["steering_normalized"], row["throttle_normalized"])
    print("Wrote", destination)

if __name__ == "__main__":
    main()
