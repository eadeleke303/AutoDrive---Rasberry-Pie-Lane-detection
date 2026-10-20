"""Stable, hardware-independent lane observations for a steering/throttle controller."""
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

@dataclass
class LaneObservation:
    status: str
    error_px: Optional[float]
    error_normalized: Optional[float]
    lane_center_x_px: Optional[float]
    lane_width_px: Optional[float]
    image_width: int
    image_height: int
    lookahead_y_px: int
    selected_channels: list
    reason: str

    def as_dict(self):
        return asdict(self)

def _interpolate_x(points, y):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        return None
    p = p[np.argsort(p[:, 1])]
    if y < p[0, 1] or y > p[-1, 1]:
        return None
    return float(np.interp(y, p[:, 1], p[:, 0]))

def observation_from_boundaries(boundaries, debug, width, height, lookahead_fraction=0.82):
    """Return errors only where both boundaries are actually observed."""
    if not 0 < lookahead_fraction < 1:
        raise ValueError("lookahead_fraction must be between 0 and 1")
    y = int(height * lookahead_fraction)
    left = _interpolate_x(boundaries.get("left_lane_boundary", []), y)
    right = _interpolate_x(boundaries.get("right_lane_boundary", []), y)
    reason = str(debug.get("reason", "unknown"))
    channels = list(debug.get("selected_channels") or [])
    if left is None or right is None or right <= left or not (0 <= left < right <= width):
        return LaneObservation("missing", None, None, None, None, width, height, y, channels, reason or "no_overlap_at_lookahead")
    center = (left + right) / 2
    error = center - width / 2
    status = "estimated" if debug.get("estimated", False) else "detected"
    return LaneObservation(status, error, error / (width / 2), center, right-left, width, height, y, channels, reason)

class AutoDriveLaneProvider:
    def __init__(self, repo="external/Ultra-Fast-Lane-Detection-v2", weights="weights/culane_res18.pth", device="auto"):
        from pretrained_ufld import PretrainedLaneDetector
        self.detector = PretrainedLaneDetector(repo=repo, weights=weights, device=device)

    def reset(self):
        self.detector.reset_tracking()

    def process_frame(self, frame, allow_estimates=False, lookahead_fraction=0.82):
        if frame is None or getattr(frame, "ndim", None) != 3:
            raise ValueError("Expected BGR image with shape (height,width,3)")
        height, width = frame.shape[:2]
        bounds = self.detector.predict(frame, allow_recovery=allow_estimates)
        return observation_from_boundaries(bounds, self.detector.last_debug, width, height, lookahead_fraction)
