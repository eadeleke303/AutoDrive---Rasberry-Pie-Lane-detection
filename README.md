# AutoDrive — Lane Detection Integration

This repository contains a Python interface for connecting AutoDrive's pretrained UFLD-v2 lane detection to a steering and throttle simulator.

## Setup

Use Python 3.11 and install dependencies with `python -m pip install -r requirements.txt`.

## Synthetic demo (no model weights required)

`python demo_integration.py --synthetic`

## Tests

`python -m unittest discover -s tests -v`

## Model integration

See `lane_interface.py` and `demo_integration.py`. The large pretrained model checkpoint and the upstream UFLD-v2 repository are not included. Keep the checkpoint and any physical vehicle actuation code out of version control. On missing/uncertain detections, use a safe stop rather than treating lane error as zero.

This interface is intended for software simulation and is not validated for autonomous physical driving.
