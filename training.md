# Training log

Tracks fine-tuning runs for `droidslam-dev` (`train.py`). Background/gotchas live in `CLAUDE.md` (`### Fine-tuning gotchas` and `### Further training improvements`) — this file is just the experiment log: what was run, with what config, what happened, and what's still open. Update it every time a run starts, finishes, or gets evaluated — don't let it drift out of sync with `checkpoints/`.

Commands below are given as pixi task entries (`cmd`/`cwd`) — copy a row's `Command` cell straight into `pixi.toml` under `[feature.droidslam-dev.tasks]` (or run its `cmd` value directly from `Baselines/DROID-SLAM-DEV` in the `droidslam-dev` pixi env).

**Reminder on checkpoint naming:** `train.py` saves every 10000 steps (round numbers, e.g. `_080000.pth`) plus one unconditional save at the true end of the run (odd step count, e.g. `_092568.pth`, `train.py:189-192`). Only the odd-numbered final file has a fully-annealed `OneCycleLR` schedule — always evaluate that one, not the last round-numbered snapshot, unless they happen to coincide.

**Reminder on what "ATE" means below:** evaluated on ETH sequences the network was trained on (`is_test_scene` is currently dead code for the `vslamlab` dataset — no sequence is genuinely held out, see `CLAUDE.md`). Treat these numbers as "fits the training sequences well," not yet "generalizes to unseen ones."

## Phase 1 — lr/epochs sweep (from `droid.pth`)

Goal: find out whether the first confirmed win (`eth`, `lr=0.00002 epochs=8`) is near a local optimum or whether a nearby setting does better, per `CLAUDE.md`'s "Sweep `--lr` and `--epochs`" suggestion. Each row is an independent run starting from stock `droid.pth` — none of these chain off each other. `eth_lr2e-5_ep8` is the same config as the already-completed `eth` run — **don't re-run it**, just reuse `eth`'s result for that grid cell.

| Run name | lr | epochs | Command | Started | Status | Final ckpt file | ATE (ETH) | Comments |
|---|---|---|---|---|---|---|---|---|
| `eth_lr2e-5_ep8` | 0.00002 | 8 | `train = {cmd = 'python train.py --name eth_lr2e-5_ep8 --epochs 8 --lr 0.00002 --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt droid.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | 2026-08-08 | Done | `eth_lr2e-5_ep8.pth` | Better than `droid.pth` baseline | First confirmed net win; motivated this sweep. |
| `eth_lr1e-5_ep8` | 0.00001 | 8 | `train = {cmd = 'python train.py --name eth_lr1e-5_ep8 --epochs 8 --lr 0.00001 --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt droid.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | 2026-08-08 | Started | — | — | Lower lr than `eth`, same epochs. |
| `eth_lr5e-5_ep8` | 0.00005 | 8 | `train = {cmd = 'python train.py --name eth_lr5e-5_ep8 --epochs 8 --lr 0.00005 --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt droid.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | — | Not started | — | — | Higher lr than `eth`, same epochs. |
| `eth_lr2e-5_ep4` | 0.00002 | 4 | `train = {cmd = 'python train.py --name eth_lr2e-5_ep4 --epochs 4 --lr 0.00002 --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt droid.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | — | Not started | — | — | Same lr as `eth`, shorter schedule. |
| `eth_lr2e-5_ep12` | 0.00002 | 12 | `train = {cmd = 'python train.py --name eth_lr2e-5_ep12 --epochs 12 --lr 0.00002 --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt droid.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | — | Not started | — | — | Same lr as `eth`, longer schedule — watch for overfitting per `CLAUDE.md`. |

**Estimated compute:** 4 new runs (skipping the `eth` duplicate) = 8+8+4+12 = 32 epoch-units, on top of the 8 already spent on `eth`.

**Winner (fill in once all 5 rows are evaluated):** _TBD_

## Phase 2 — refine from the phase-1 winner

Only run this once Phase 1 is fully evaluated and a winner is picked. Single short low-lr continuation from the winning checkpoint, not another blind full-scale run — see `CLAUDE.md`'s discussion of why lower lr slows overfitting but doesn't prevent it.

| Run name | lr | epochs | Command | Started | Status | Final ckpt file | ATE (ETH) | Comments |
|---|---|---|---|---|---|---|---|---|
| `eth_phase2` | *(winner lr / 10, TBD)* | 3 | `train = {cmd = 'python train.py --name eth_phase2 --epochs 3 --lr <winner_lr / 10> --datasets vslamlab --datapath ../../../VSLAM-LAB-Benchmark --ckpt checkpoints/<winner_name>_<final_step_count>.pth', cwd = 'Baselines/DROID-SLAM-DEV'}` | — | Blocked on Phase 1 | — | — | Keep only if it beats the Phase 1 winner's ATE. |

## Open follow-ups (not yet scheduled)

These are further ideas from `CLAUDE.md`'s "Further training improvements" section, not yet turned into runs:

- Fix `is_test_scene` for the `vslamlab` dataset (`droid_slam/data_readers/vslamlab.py:40-43`) so a real held-out ETH split exists — would make every ATE number above trustworthy as a generalization signal, not just a training-fit signal.
- Add a validation loss loop once held-out scenes exist, to get a cheaper per-checkpoint signal than a full ATE eval.
- Check how many distinct scenes/anchors are actually feeding each run (`RGBDDataset ready: N scenes, M training samples` log line, `base.py:62-63`) — note it per row above once available, since a tiny `M` raises overfitting risk for the higher-epoch rows.
- Re-verify the missing-depth masking fix's `raw==0` fraction on ETH sequences beyond `table_3`.
