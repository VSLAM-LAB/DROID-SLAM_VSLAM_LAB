# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the **`droidslam-dev`** baseline for VSLAM-LAB: a source build of [DROID-SLAM](https://arxiv.org/abs/2108.10869) (Teed & Deng) wrapped so it can be driven by the VSLAM-LAB pipeline. It is a fork of `princeton-vl/DROID-SLAM` with three VSLAM-LAB entry-point scripts and a CUDA extension (`droid_backends`) layered on top of the original repo.

This directory is not meant to be run standalone in normal operation — it is built and executed through `pixi` tasks defined in the parent VSLAM-LAB repo's `pixi.toml` (feature `droidslam-dev`), and registered as `DROIDSLAM_baseline_dev` in `Baselines/baseline_files/baseline_droidslam.py`. It shares the underlying network/algorithm code with the `droidslam` (non-dev) baseline in the sibling `DROID-SLAM` directory, which instead installs a pre-built conda package from the `fontan` channel.

## Build & Common Commands

All commands are run from the **VSLAM-LAB repo root** via pixi, using the `droidslam-dev` environment/feature:

```bash
# Clone this baseline's source (if not already present)
pixi run -e droidslam-dev git-clone

# Build the CUDA extension + install the package (runs `python setup.py install` here)
pixi run -e droidslam-dev install

# Run VSLAM-LAB experiments through the standard pipeline (Module: droidslam-dev in exp yaml)
pixi run vslamlab configs/exp_vslamlab.yaml
pixi run demo droidslam-dev <dataset> <sequence> <mono|rgbd|stereo>

# Train the network (thin wrapper around train.py, from repo root of DROID-SLAM-DEV)
pixi run -e droidslam-dev train iters=1 steps=50000 lr=0.000025 datapath=/path/to/TartanAir ckpt=droid.pth
```

Building the extension requires CUDA (`system-requirements.cuda = "12.0"` in the feature) and reads Eigen headers from `$CONDA_PREFIX/include/eigen3` (see `setup.py`). Do not run `python setup.py install` outside the pixi `droidslam-dev` environment — it depends on that environment's torch/CUDA/Eigen versions.

## Architecture

### Two layers of scripts

- **VSLAM-LAB entry points** (`vslamlab_droidslam_mono.py`, `_rgbd.py`, `_stereo.py`): what pixi's `execute-mono` / `execute-rgbd` / `execute-stereo` tasks actually run. Each parses `--sequence_path`, `--calibration_yaml`, `--rgb_csv`, `--exp_folder`, `--exp_it`, reads algorithm settings from a settings YAML (`droid_slam/configs/vslamlab_droidslam-dev_settings.yaml`, falling back to the copy at repo root, `vslamlab_droidslam-dev_settings.yaml`), builds an `image_stream` generator from VSLAM-LAB's `rgb_exp.csv` + `calibration.yaml` format, feeds it into `Droid`, and writes `<exp_it>_KeyFrameTrajectory.csv` into `--exp_folder`. They diverge only in how they build the image stream:
  - `mono`: single camera (`cam_mono`), undistorts/resizes via `cv2.initUndistortRectifyMap`.
  - `rgbd`: like mono but also loads an aligned depth image per frame (`depth_name`/`depth_factor` from calibration), passed to `Droid.track`.
  - `stereo`: two cameras (`cam_stereo: [rgb_0, rgb_1]`), computes stereo rectification from each camera's `T_BS` extrinsic via `cv2.stereoRectify`, yields a stacked left/right image tensor.
  - After processing all frames, each script calls `droid.terminate(...)` (second pass over the image stream) to get final poses, then force-kills the process group via `os.killpg` — expected behavior, not a bug.

- **Upstream DROID-SLAM scripts** (`demo.py`, `train.py`, `evaluation_scripts/*`, `tools/*.sh`): unmodified/near-unmodified from the original repo, used for standalone demos, network training, and benchmark evaluation (TartanAir/EuRoC/TUM/ETH3D-SLAM) independent of the VSLAM-LAB harness. `train.py` supports multi-GPU DDP training on TartanAir via `data_readers/tartan.py` + `data_readers/factory.py`.

### Core algorithm package (`droid_slam/`)

- `droid.py` — `Droid` class: top-level orchestrator wiring together `DroidNet` (weights loaded from `droid.pth`), `DepthVideo` (shared frame/pose/depth buffer across processes), `MotionFilter` (keyframe gating), `DroidFrontend` (online local BA), `DroidBackend` (final global BA, run at increasing confidence levels 7 then 12 in `terminate()`), and `PoseTrajectoryFiller` (interpolates non-keyframe poses at the end).
- `droid_net.py` — the network (feature/context extractors + GRU update operator + correlation volume).
- `droid_async.py` — `DroidAsync`, an alternate `Droid` that runs frontend/backend as separate processes/devices for async multi-GPU inference (`--asynchronous`, `--frontend_device`, `--backend_device` in `demo.py`).
- `factor_graph.py`, `geom/` (`ba.py`, `projective_ops.py`, `chol.py`, `losses.py`, `graph_utils.py`) — the pose-graph / bundle-adjustment math, built on `lietorch` (SE3/SO3/Sim3).
- `modules/` (`extractor.py`, `gru.py`, `corr.py`, `clipping.py`) — network submodules used by `droid_net.py`.
- `data_readers/` — training-only dataset loaders (TartanAir), not used by the VSLAM-LAB inference path.
- `visualizer/` — optional Open3D-based live visualization (`--disable_vis` / `verbose` flag controls this), runs in a separate `Process`.

### CUDA extension (`src/`, built by `setup.py`)

`droid_backends` (`src/droid.cpp`, `src/droid_kernels.cu`, `src/correlation_kernels.cu`, `src/altcorr_kernel.cu`) — custom CUDA kernels for the correlation lookup and bundle-adjustment ops used by `geom/` and `modules/corr.py`. Built as a `torch.utils.cpp_extension.CUDAExtension`; installed status is checked by `is_installed()` in the VSLAM-LAB baseline class by looking for `build/lib.linux-x86_64-cpython-311/droid_backends.so`.

### Settings

`vslamlab_droidslam-dev_settings.yaml` (and its copy under `droid_slam/configs/`) holds all frontend/backend tuning parameters (`t0`, `stride`, `buffer`, `beta`, `filter_thresh`, `warmup`, `keyframe_thresh`, `frontend_thresh`, `frontend_window`, `frontend_radius`, `frontend_nms`, `backend_thresh`, `backend_radius`, `backend_nms`) plus the camera name(s) to use per mode (`cam_mono`, `cam_rgbd`, `cam_stereo`). These are read by the `vslamlab_droidslam_*.py` scripts, not by `demo.py`/`train.py` (which take equivalent values as CLI args instead).
