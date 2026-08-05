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

# Train the network (pixi task hardcodes: --steps 250000 --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt droid.pth)
pixi run -e droidslam-dev train
# To override, invoke train.py directly with its argparse flags (see `python train.py --help`), e.g.:
pixi run -e droidslam-dev python train.py --datasets tartan --datapath /path/to/TartanAir --gpus 4 --lr 0.00025
```

Building the extension requires CUDA (`system-requirements.cuda = "12.0"` in the feature) and reads Eigen headers from `$CONDA_PREFIX/include/eigen3` (see `setup.py`). Do not run `python setup.py install` outside the pixi `droidslam-dev` environment — it depends on that environment's torch/CUDA/Eigen versions.

## Architecture

### Two layers of scripts

- **VSLAM-LAB entry points** (`vslamlab_droidslam_mono.py`, `_rgbd.py`, `_stereo.py`): what pixi's `execute-mono` / `execute-rgbd` / `execute-stereo` tasks actually run. Each parses `--sequence_path`, `--calibration_yaml`, `--rgb_csv`, `--exp_folder`, `--exp_it`, reads algorithm settings from a settings YAML (`droid_slam/configs/vslamlab_droidslam-dev_settings.yaml`, falling back to the copy at repo root, `vslamlab_droidslam-dev_settings.yaml`), builds an `image_stream` generator from VSLAM-LAB's `rgb_exp.csv` + `calibration.yaml` format, feeds it into `Droid`, and writes `<exp_it>_KeyFrameTrajectory.csv` into `--exp_folder`. They diverge only in how they build the image stream:
  - `mono`: single camera (`cam_mono`), undistorts/resizes via `cv2.initUndistortRectifyMap`.
  - `rgbd`: like mono but also loads an aligned depth image per frame (`depth_name`/`depth_factor` from calibration), passed to `Droid.track`.
  - `stereo`: two cameras (`cam_stereo: [rgb_0, rgb_1]`), computes stereo rectification from each camera's `T_BS` extrinsic via `cv2.stereoRectify`, yields a stacked left/right image tensor.
  - After processing all frames, each script calls `droid.terminate(...)` (second pass over the image stream) to get final poses, then force-kills the process group via `os.killpg` — expected behavior, not a bug.

- **Upstream DROID-SLAM scripts** (`demo.py`, `train.py`, `evaluation_scripts/*`, `tools/*.sh`): unmodified/near-unmodified from the original repo, used for standalone demos, network training, and benchmark evaluation (TartanAir/EuRoC/TUM/ETH3D-SLAM) independent of the VSLAM-LAB harness. `train.py` supports multi-GPU DDP training (see `data_readers/` below for which datasets it can train on).

### Core algorithm package (`droid_slam/`)

- `droid.py` — `Droid` class: top-level orchestrator wiring together `DroidNet` (weights loaded from `droid.pth`), `DepthVideo` (shared frame/pose/depth buffer across processes), `MotionFilter` (keyframe gating), `DroidFrontend` (online local BA), `DroidBackend` (final global BA, run at increasing confidence levels 7 then 12 in `terminate()`), and `PoseTrajectoryFiller` (interpolates non-keyframe poses at the end).
- `droid_net.py` — the network (feature/context extractors + GRU update operator + correlation volume).
- `droid_async.py` — `DroidAsync`, an alternate `Droid` that runs frontend/backend as separate processes/devices for async multi-GPU inference (`--asynchronous`, `--frontend_device`, `--backend_device` in `demo.py`).
- `factor_graph.py`, `geom/` (`ba.py`, `projective_ops.py`, `chol.py`, `losses.py`, `graph_utils.py`) — the pose-graph / bundle-adjustment math, built on `lietorch` (SE3/SO3/Sim3).
- `modules/` (`extractor.py`, `gru.py`, `corr.py`, `clipping.py`) — network submodules used by `droid_net.py`.
- `data_readers/` — training-only dataset loaders, not used by the VSLAM-LAB inference path. `tartan.py` reads TartanAir; `vslamlab.py` (`VSLAMLAB` class, not in upstream DROID-SLAM) reads sequences directly from `VSLAM-LAB-Benchmark` for training on VSLAM-LAB data — selected via `--datasets vslamlab` in `train.py`, dispatched through the `dataset_map` in `factory.py`. `train.py` supports multi-GPU DDP training; each rank constructs its own dataset instance, so any lazy setup in `base.py`/`vslamlab.py` (cache dirs, etc.) must be rank-safe (see recent fixes for `os.mkdir` races and NaN/import bugs across ranks).
- `visualizer/` — optional Open3D-based live visualization (`--disable_vis` / `verbose` flag controls this), runs in a separate `Process`.

### CUDA extension (`src/`, built by `setup.py`)

`droid_backends` (`src/droid.cpp`, `src/droid_kernels.cu`, `src/correlation_kernels.cu`, `src/altcorr_kernel.cu`) — custom CUDA kernels for the correlation lookup and bundle-adjustment ops used by `geom/` and `modules/corr.py`. Built as a `torch.utils.cpp_extension.CUDAExtension`; installed status is checked by `is_installed()` in the VSLAM-LAB baseline class by looking for `build/lib.linux-x86_64-cpython-311/droid_backends.so`.

### Settings

`vslamlab_droidslam-dev_settings.yaml` (and its copy under `droid_slam/configs/`) holds all frontend/backend tuning parameters (`t0`, `stride`, `buffer`, `beta`, `filter_thresh`, `warmup`, `keyframe_thresh`, `frontend_thresh`, `frontend_window`, `frontend_radius`, `frontend_nms`, `backend_thresh`, `backend_radius`, `backend_nms`) plus the camera name(s) to use per mode (`cam_mono`, `cam_rgbd`, `cam_stereo`). These are read by the `vslamlab_droidslam_*.py` scripts, not by `demo.py`/`train.py` (which take equivalent values as CLI args instead).

### Fine-tuning gotchas (train.py): checklist if ATE regresses after fine-tuning

Symptom seen in practice: fine-tuning `droid.pth` on VSLAM-LAB/ETH3D sequences (`--datasets vslamlab`), then evaluating on those *same* sequences, made ATE worse rather than better. Lowering `--lr` alone did not fix it. Candidate causes, most to least likely, none yet confirmed as root cause:

- **Missing-depth pixels were labeled as confident "far" ground truth, not masked out — fixed.** Verified directly on `ETH/table_3` depth PNGs (`cv2.IMREAD_ANYDEPTH`): 39%-64% of pixels per sampled frame have `raw==0` ("no reading" — normal for a structured-light/ToF sensor on dark/reflective/out-of-range surfaces; saturation was 0% in the same sample, so this is entirely a missing-data issue, not a range/saturation one — for a desk/tabletop scene like `table_3`, "no reading" is far more likely a close reflective/dark object than distant background). `depth_read` (`vslamlab.py:128-141`) maps every such pixel to `DEPTH_FAR = 1e3` (disp ≈ 0.001). Since we genuinely don't know the true depth for these pixels, the only correct fix is exclusion, not a better guess at distance. Ground truth depth is consumed in exactly two places in the training path (`geodesic_loss` only uses poses; `residual_loss` comes from the network's own predicted residuals, not ground truth depth, so neither was affected):
  - `flow_loss` (`geom/losses.py:99`) previously masked only `disps[:,ii] > 0` — `DEPTH_FAR`'s disp is `>0` so it was never excluded. **Fixed**: changed to `disps[:,ii] > 0.01`, matching the disparity-space validity cutoff already used in `base.py`'s per-sample scale computation (`DEPTH_FAR`'s disp of 0.001 sits safely below it, ~100x margin).
  - `build_frame_graph`'s `read_disp` (`base.py:86-89`) fed `DEPTH_FAR`-tagged depth straight into `compute_distance_matrix_flow`; `projective_ops.py`'s `projective_transform` only excludes points *too close* (`MIN_DEPTH=0.2`), never too far, so these pixels contributed near-zero induced flow for every frame pair regardless of true camera motion, biasing the co-visibility graph toward marking distant frames as covisible. **Fixed**: `read_disp` now also excludes `depth >= DEPTH_FAR` from the mean/replacement, via `getattr(self.__class__, 'DEPTH_FAR', None)` so `tartan.py` (no such attribute) is unaffected.
  This pipeline was originally built/tuned against TartanAir's dense synthetic depth, which essentially never hits this failure mode — real ETH3D-style sensor data does, on well over a third of every frame.
- **Poses likely not time-synchronized with frames — ruled out for this project** (already using `pixi run synch-gt` / `Datasets/extra-files/synch_gt.py` before training). Leaving the mechanism documented in case a *specific* sequence/copy slips through: `vslamlab.py:_build_dataset` (lines 60-72) reads `images`/`depths` from `rgb.csv` and `poses` from `groundtruth.csv` and pairs them **positionally by row index** with no timestamp join of its own — it relies entirely on `rgb.csv`/`groundtruth.csv` already being 1:1 on disk. `synch_gt.py`'s `synch_pair()` (TUM-`associate.py`-style nearest-neighbor match) is what makes that true, rewriting both files in place and keeping `*_raw.csv` backups; a sequence with no `*_raw.csv` backup hasn't been through it. Worth a quick check that every sequence actually used for training has that backup present, since a mismatch would silently misalign supervision (and would likely surface as an index/shape error in `compute_distance_matrix_flow`, `rgbd_utils.py:106-131`, the first time that scene is built without a stale `droid_slam/data_readers/cache/vslamlab.pickle` masking it).
- **`--steps`/OneCycleLR budget not scaled down for fine-tuning.** `train.py:85-86` builds `OneCycleLR(optimizer, args.lr, args.steps, pct_start=0.01, ...)`. The pixi task hardcodes `--steps 250000` (`pixi.toml`, `feature.droidslam-dev.tasks.train`) — sized for pretraining on all of TartanAir. Reusing it to fine-tune on a handful of ETH sequences means thousands of epochs over a narrow dataset; a lower peak LR slows the drift but doesn't bound it. Try cutting `--steps` (and thus the OneCycleLR schedule) down to something proportionate to the fine-tuning set size, and confirm the checkpoint you evaluate is the true final one (post-anneal), not an intermediate `checkpoints/<name>_XXXXXX.pth` snapshot from mid-ramp.
- **Training loss is not a proxy for ATE.** `geodesic_loss`/`residual_loss`/`flow_loss` (`train.py:136-138`) are computed on fixed `n_frames=7` co-visible windows from `build_frame_graph`'s optical-flow graph (`droid_slam/data_readers/base.py:84-107`). ATE depends on the full online run: `MotionFilter` keyframe gating, `DroidFrontend`'s sliding-window local BA (`frontend_window`/`frontend_radius`/`keyframe_thresh` from the settings YAML — a different graph topology than training), `DroidBackend`'s global BA, then `PoseTrajectoryFiller`. A network can look fine (or even improve) on windowed training/validation loss and still regress on full-trajectory ATE. There is no way around this other than running actual inference + ATE on saved checkpoints — no loss curve (train or held-out) guarantees it.
- **No validation loop exists anywhere in the training code**, and this is not a regression — it never existed, including for TartanAir. `base.py:74`'s `"Reserving {} for validation"` only excludes a scene from `dataset_index` (`base.py:65-74`) so its frames are never sampled for a gradient step; nothing then loads those frames back in to compute a val loss. `logger.py`'s `Logger` only tracks training loss. Upstream's actual generalization check is `evaluation_scripts/test_tartanair.py`, a separate script run manually against saved checkpoints via `TartanAirTestStream`/`--gt_path`.
- **`is_test_scene` is dead code for the `vslamlab` dataset**, so "held out" scenes are never actually held out. `droid_slam/data_readers/vslamlab.py:40-43` reuses `tartan.py`'s `test_split` (loaded from `tartan_test.txt`, TartanAir scene names like `abandonedfactory/Easy/P011`) — those substrings never match VSLAM-LAB scene names like `ETH/table_3`, so `is_test_scene` always returns `False` and every scene passed via `--datapath` lands in the training set with no held-out subset. Only matters if you want a true generalization test (some ETH sequences never trained on); doesn't by itself explain a regression on sequences you intended to train on.
- **Anchor count may be tiny for short/low-motion sequences.** `base.py:71` only keeps an anchor frame if `len(graph[i][0]) > n_frames` (7 by default) co-visible frames exist. A short or mostly-static sequence (e.g. a fixed-camera table scene) can yield very few usable anchors; combined with a large `--steps`, that's effectively many repeated passes over a handful of frame triplets — a fast route to a narrow optimum that doesn't generalize back to the full-trajectory BA behavior used at inference. Check the logged `RGBDDataset ready: N scenes, M training samples` (`base.py:62-63`) for how small `M` actually is.
- **Sanity check first:** confirm the eval run is actually pointed at the new fine-tuned checkpoint and not still `droid.pth`.
