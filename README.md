# AutoDrive — Lane Detection Partner Handoff

A Python interface connecting the AutoDrive **v16/v18-style pretrained UFLD-v2 lane detector** to a steering/throttle controller in **offline simulation**. This repository does not control a physical RC car.

## Quick start on Windows (Python 3.11)

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe demo_integration.py --synthetic
```

Synthetic mode does **not** require a checkpoint, video, or the upstream UFLD-v2 repository. Results are written to `output/partner_observations.csv`.

## Running on dashcam video

Copy the **CPU-compatible** official UFLD-v2 repository from the existing working AutoDrive installation to `external/Ultra-Fast-Lane-Detection-v2`, and copy its checkpoint to `weights/culane_res18.pth`. These are intentionally excluded from Git. The upstream repository may need the existing local `.cuda()` and NVIDIA DALI compatibility fixes.

```bat
.venv\Scripts\python.exe demo_integration.py --video "C:\path\to\dashcam.mp4" --max-frames 300
```

## Partner integration contract

```python
from lane_interface import AutoDriveLaneProvider

provider = AutoDriveLaneProvider()
observation = provider.process_frame(frame)

if observation.status == "detected":
    steering, throttle = partner_controller(observation.error_normalized)
else:
    steering, throttle = 0.0, 0.0  # simulation-only fail-safe
```

The partner supplies `partner_controller`. The observation includes `status`, `error_px`, `error_normalized`, `lane_center_x_px`, `lane_width_px`, `selected_channels`, and `reason`. Positive error means the detected lane center is **right** of image center. Sign, command units, steering limits, throttle units, and controller dynamics must be verified by the partner.

**Never treat a missing error (`None`) as zero error.** `estimated` results are also not confirmed detections. Reset the provider between different driving videos.

## Limitations and verification

- Synthetic interface tests run without model weights. The actual UFLD-v2 checkpoint and full video inference have **not** been tested in this repository.
- The source is adapted for a handoff from the v16/v18 baseline; verify outputs against the known working local project before interpreting results as equivalent.
- This is not validated for autonomous vehicle control, Raspberry Pi performance, or direct motor actuation.
- No CVAT ground truth is used during inference.
- Check upstream license requirements before redistributing models or third-party sources.
