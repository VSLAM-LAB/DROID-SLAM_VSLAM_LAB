# Training log

Tracks fine-tuning runs for `droidslam-dev` (`train.py`). Background/gotchas live in `CLAUDE.md` (`### Fine-tuning gotchas` and `### Further training improvements`) — this file is just the experiment log: what was run, with what config, what happened, and what's still open. Update it every time a run starts, finishes, or gets evaluated — don't let it drift out of sync with `checkpoints/`.

Commands below are given as pixi task entries (`cmd`/`cwd`) — copy a row's `Command` cell straight into `pixi.toml` under `[feature.droidslam-dev.tasks]` (or run its `cmd` value directly from `Baselines/DROID-SLAM-DEV` in the `droidslam-dev` pixi env).

**Reminder on checkpoint naming:** `train.py` saves every 10000 steps (round numbers, e.g. `_080000.pth`) plus one unconditional save at the true end of the run (odd step count, e.g. `_092568.pth`, `train.py:189-192`). Only the odd-numbered final file has a fully-annealed `OneCycleLR` schedule — always evaluate that one, not the last round-numbered snapshot, unless they happen to coincide.

**Reminder on what "ATE" means below:** evaluated on ETH sequences the network was trained on (`is_test_scene` is currently dead code for the `vslamlab` dataset — no sequence is genuinely held out, see `CLAUDE.md`). Treat these numbers as "fits the training sequences well," not yet "generalizes to unseen ones."

## Phase 1 — lr/epochs sweep (from `droid.pth`)

Goal: find out whether the first confirmed win (`eth`, `lr=0.00002 epochs=8`) is near a local optimum or whether a nearby setting does better, per `CLAUDE.md`'s "Sweep `--lr` and `--epochs`" suggestion. Each row is an independent run starting from stock `droid.pth` — none of these chain off each other. `eth_lr2e-5_ep8` is the same config as the already-completed `eth` run — **don't re-run it**, just reuse `eth`'s result for that grid cell.

| Run name | lr | epochs | Command | Started | Status | Final ckpt file | ATE (ETH) | Comments |
|---|---|---|---|---|---|---|---|---|
| `eth_lr2e-5_ep8` | 0.00002 | 8 | `train = {cmd = 'python train.py --name eth_lr2e-5_ep8 --epochs 8 --lr 0.00002 --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt droid.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | 2026-08-08 | Done + evaluated | `eth_lr2e-5_ep8.pth` | mean 10.6 cm / median 0.52 cm | First confirmed net win; motivated this sweep. |
| `eth_lr1e-5_ep8` | 0.00001 | 8 | `train = {cmd = 'python train.py --name eth_lr1e-5_ep8 --epochs 8 --lr 0.00001 --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt droid.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | 2026-08-08 | Done + evaluated | `eth_lr1e-5_ep8_092568.pth` | mean 12.3 cm / median 0.51 cm | Clearly worst of the fine-tuned group — lr too low to finish adapting. |
| `eth_lr5e-5_ep8` | 0.00005 | 8 | `train = {cmd = 'python train.py --name eth_lr5e-5_ep8 --epochs 8 --lr 0.00005 --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt droid.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | 2026-08-08 | Done + evaluated | `eth_lr5e-5_ep8_092568.pth` | mean 10.1 cm / median 0.52 cm | Best mean, but the edge over lr2e-5 comes almost entirely from one sequence (`einstein_global_light_changes_1`: 76→10 cm); excluding it, it's not ahead. |
| `eth_lr2e-5_ep4` | 0.00002 | 4 | `train = {cmd = 'python train.py --name eth_lr2e-5_ep4 --epochs 4 --lr 0.00002 --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt droid.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | 2026-08-08 | Done + evaluated | `eth_lr2e-5_ep4_046284.pth` | mean 10.7 cm / median 0.50 cm | Statistically tied with ep8/ep12. |
| `eth_lr2e-5_ep12` | 0.00002 | 12 | `train = {cmd = 'python train.py --name eth_lr2e-5_ep12 --epochs 12 --lr 0.00002 --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt droid.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | 2026-08-08 | Done + evaluated | `eth_lr2e-5_ep12_138852.pth` | mean 10.3 cm / median 0.56 cm | No overfitting signal vs ep4/ep8; best on `einstein_global_light_changes_3` (26 cm vs ~56 cm others). |

**Estimated compute:** 4 new runs (skipping the `eth` duplicate) = 8+8+4+12 = 32 epoch-units, on top of the 8 already spent on `eth`.

**Winner:** effectively a four-way tie. ATE = mean/median RMSE over 55 ETH sequences × 3 runs each (from `VSLAM-LAB-Evaluation/exp_train_droid_*/ETH/*/vslamlab_evaluation/ate.csv`); un-fine-tuned `droid.pth` baseline: mean 18.6 cm / median 0.62 cm. Everything except `lr1e-5` lands at mean 10.1–10.7 cm, and average per-sequence run-to-run std is ~1 cm with only 3 runs, so that ordering is inside the noise. `lr1e-5` (12.3 cm) is the only config that's genuinely worse. Nominal pick for Phase 2 / future work: **`eth_lr2e-5_ep8`** (center of the tied region; the lr5e-5 "win" is a single-sequence effect, see its row).

**Phase 1 findings (2026-08-12 evaluation):**

- **Training set: all ETH sequences, run on the HPC** (the local `VSLAM-LAB-Benchmark/scenes.yaml` listing only `table_3` reflects the local machine, not what trained). So every ETH number above is fit-to-training-data, per the reminder at the top of this file — and, importantly, the tail sequences below failed *despite being in the training set*.
- **Scene-info cache was fresh for all Phase 1 runs**: the HPC's `droid_slam/data_readers/cache/vslamlab.pickle` was deleted before Phase 1 launched (confirmed 2026-08-13), so the co-visibility graphs were built with the post-`8a8fd3c` missing-depth fix and the full ETH scene list — no stale-cache asterisk on these numbers. Reminder for future runs: the cache is keyed only by the name `vslamlab` (`base.py:43`), so delete it again whenever `scenes.yaml`, `depth_read`, or `build_frame_graph` changes.
- **The lr/epochs axis is saturated.** All reasonable settings converge to the same behavior; further sweeping this grid has near-zero expected value.
- **The mean is dominated by a ~5-sequence failure tail** (top-5 sequences carry ~470–570 cm of the ~560–680 cm total summed ATE). The tail decomposes by failure mode, not by hyperparameter:
  - `ceiling_1` (~201 cm): byte-identical across *all* checkpoints including stock `droid.pth` — completely insensitive to training. This is an inference-side failure (degenerate/low-texture scene), not a network-weights problem.
  - `kidnap_1` (~133–147 cm): a relocalization test by construction; DROID-SLAM doesn't relocalize after a kidnap, so this measures a capability the system lacks, not fine-tune quality.
  - Illumination-change sequences (`einstein_flashlight` 120→3–7 cm, `einstein_global_light_changes_1/3`): the one family that *is* training-sensitive, but inconsistently — different configs fix different sequences, which looks like luck-of-the-draw rather than systematic robustness. Photometric augmentation is the targeted lever here.
  - `mannequin_face_1` (~26–39 cm): not improved by any config, slightly worse in some.
- **REPLICA (8 sequences, not in the ETH training data): 0.53 cm baseline → 0.44–0.53 cm fine-tuned.** Fine-tuning on ETH did not damage cross-dataset performance; `lr1e-5` was best there (0.44 cm).
- Median ETH ATE barely moved (0.62 → 0.50–0.56 cm): on the ~50 "easy" sequences the system was already near its floor; all the headroom is in the tail.

## Phase 2 — refine from the phase-1 winner

Only run this once Phase 1 is fully evaluated and a winner is picked. Single short low-lr continuation from the winning checkpoint, not another blind full-scale run — see `CLAUDE.md`'s discussion of why lower lr slows overfitting but doesn't prevent it.

| Run name | lr | epochs | Command | Started | Status | Final ckpt file | ATE (ETH) | Comments |
|---|---|---|---|---|---|---|---|---|
| `eth_phase2` | *(winner lr / 10, TBD)* | 3 | `train = {cmd = 'python train.py --name eth_phase2 --epochs 3 --lr <winner_lr / 10> --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt checkpoints/<winner_name>_<final_step_count>.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | — | Blocked on Phase 1 | — | — | Keep only if it beats the Phase 1 winner's ATE. |

## Phase 3 — regularization-off overfit run (from `droid.pth`)

Goal: maximize fit to the ETH training sequences by removing the stochastic/regularizing parts of training, since Phase 1 showed the lr/epochs axis is saturated. Changes vs the Phase 1 center (`eth_lr2e-5_ep8`): photometric augmentation off (`--no_aug_photo`: no color jitter / random grayscale), deterministic fixed-scale resize+center-crop (`--no_aug_crop`), and weight decay off (`--wd 0`). `fmin` stays at the default 8.0 so the run is a clean A/B against Phase 1 — the only delta is regularization; lowering `--fmin` (e.g. to 4, letting low-parallax pairs into training windows, `base.py:128` — widens per-window frame choice but does **not** change the anchor count `M`, which depends only on the graph-size-vs-`n_frames` threshold, `base.py:71`) is a separate single-variable follow-up if this run moves the needle. All changes are sample-time only, so **no `vslamlab.pickle` deletion is needed** (the cached graph is unaffected). Flags added 2026-08-13 (`train.py`, threaded through `dataset_factory` → `RGBDDataset` → `RGBDAugmentor`).

Judge on the `einstein_*` illumination family and the median, not the mean — `ceiling_1`/`kidnap_1`/`mannequin_face_1` are training-insensitive (see Phase 1 findings). Also re-check REPLICA to quantify what pure memorization costs cross-dataset (Phase 1 fine-tunes held it at 0.44–0.53 cm).

| Run name | lr | epochs | Command | Started | Status | Final ckpt file | ATE (ETH) | Comments |
|---|---|---|---|---|---|---|---|---|
| `eth_overfit_noaug` | 0.00002 | 8 | `train-overfit = {cmd = 'python train.py --name eth_overfit_noaug --epochs 8 --lr 0.00002 --wd 0 --no_aug_photo --no_aug_crop --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt droid.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | — | Not started | — | — | Clean A/B vs `eth_lr2e-5_ep8`: only aug/wd differ. If photometric aug removal hurts the `einstein_*` family specifically, re-run with photo aug back on (`--no_aug_crop --wd 0` only) to separate the two effects. |

## Open follow-ups (not yet scheduled)

These are further ideas from `CLAUDE.md`'s "Further training improvements" section, not yet turned into runs:

- Fix `is_test_scene` for the `vslamlab` dataset (`droid_slam/data_readers/vslamlab.py:40-43`) so a real held-out ETH split exists — would make every ATE number above trustworthy as a generalization signal, not just a training-fit signal.
- Add a validation loss loop once held-out scenes exist, to get a cheaper per-checkpoint signal than a full ATE eval.
- Check how many distinct scenes/anchors are actually feeding each run (`RGBDDataset ready: N scenes, M training samples` log line, `base.py:62-63`) — note it per row above once available, since a tiny `M` raises overfitting risk for the higher-epoch rows.
- Re-verify the missing-depth masking fix's `raw==0` fraction on ETH sequences beyond `table_3`.
