# Training Reproduction Guide

This document explains how to run the 48-frame 9Mix training recipe from a fresh checkout.

Target command:

```bash
VIDEOMAE2_CKPT=/path/to/vit_g_hybrid_pt_1200e.pth \
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

Target configs:

```text
configs/repro/worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_eval64clip/model_effective.yaml
configs/repro/worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_eval64clip/train_effective.yaml
```

## 1. What This Run Does

The training script launches `torchrun` and trains a 48-frame D4RT model on the 9Mix dataset mixture.

Important defaults:

```text
clip_frames: 48
image_size: 256 x 256
train_batch_size: 2
val_batch_size: 2
total_steps: 30000
peak_lr: 4e-6
final_lr: 4e-7
save_every_steps: 1000
validate_every_steps: 2000
auto_eval_worldtrack_step.enabled: true
auto_eval_worldtrack_step.num_frames: 64
```

The run initializes model weights from a 32-frame OpenD4RT checkpoint and resizes the learned timestep embeddings to 48 frames.

## 2. Prerequisites

### Python environment

Use either conda or any existing Python environment.

Conda example:

```bash
conda env create -f environment.yml
conda activate d4rt
```

Pip example:

```bash
pip install -r requirements.txt
```

The scripts do not require conda. They only auto-activate a conda environment if you explicitly set:

```bash
CONDA_ENV=d4rt
```

### Required checkpoints

The script expects these files by default:

```text
checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt
checkpoints/VideoMAE2/weights/mae-g/vit_g_hybrid_pt_1200e.pth
```

You can override them:

```bash
INIT_CKPT=/path/to/opend4rt_32clip.ckpt
VIDEOMAE2_CKPT=/path/to/vit_g_hybrid_pt_1200e.pth
```

### Required datasets

Prepare the 9Mix dataset roots described in [dataset.md](dataset.md).

## 3. Training Script Behavior

The main entrypoint is:

```text
scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

The script does the following:

1. Moves to the repository root.
2. Optionally activates a conda environment if `CONDA_ENV` is set.
3. Resolves the config paths and required checkpoint paths.
4. Checks `WORLD_SIZE == EXPECTED_WORLD_SIZE`.
5. Injects config overrides through `--override`.
6. Launches:

```bash
torchrun ... train.py --tb_log --model-config ... --train-config ... --init-model ...
```

Inside `train.py`, the code:

1. Loads the model and train YAML configs.
2. Applies command-line overrides.
3. Writes effective configs to:

```text
<OUT_DIR>/config/model_effective.yaml
<OUT_DIR>/config/train_effective.yaml
```

4. Builds dataloaders for:

```text
train
val
reference val
per-dataset val loaders
```

5. Builds the model and loss.
6. Loads `--init-model` weights and resizes timestep embeddings if needed.
7. Creates the TensorBoard writer when `--tb_log` is enabled.
8. Starts the trainer loop.

## 4. Recommended Bring-Up Sequence

### Step 1: Verify files without launching training

```bash
DRY_RUN=1 \
VIDEOMAE2_CKPT=/path/to/vit_g_hybrid_pt_1200e.pth \
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

If this fails, the problem is usually one of:

- missing config file
- missing init checkpoint
- missing VideoMAE2 checkpoint
- wrong `EXPECTED_WORLD_SIZE`

### Step 2: Run a one-GPU smoke test

This is the fastest way to catch dataset layout and dataloader issues.

```bash
CUDA_VISIBLE_DEVICES=0 \
EXPECTED_WORLD_SIZE=1 \
NPROC_PER_NODE=1 \
TOTAL_STEPS=10 \
SAVE_EVERY_STEPS=10 \
STEP_SAVE_EVERY_STEPS=10 \
AUTO_EVAL_WORLDTRACK_ENABLED=false \
VIDEOMAE2_CKPT=/path/to/vit_g_hybrid_pt_1200e.pth \
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

What this checks:

- `torchrun` startup
- DDP/single-process config path
- all training dataset roots
- all validation dataset roots
- checkpoint loading
- first forward/backward passes
- checkpoint writing
- TensorBoard writing

### Step 3: Run the intended 8-GPU job

```bash
VIDEOMAE2_CKPT=/path/to/vit_g_hybrid_pt_1200e.pth \
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

If you want to choose devices explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VIDEOMAE2_CKPT=/path/to/vit_g_hybrid_pt_1200e.pth \
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

## 5. Important Environment Variables

### Core launch variables

```text
CUDA_VISIBLE_DEVICES
NPROC_PER_NODE
NUM_MACHINES
MACHINE_RANK
MASTER_ADDR
MASTER_PORT
EXPECTED_WORLD_SIZE
```

### Checkpoint and config overrides

```text
MODEL_CONFIG
TRAIN_CONFIG
INIT_CKPT
INIT_TIMESTEP_EMBED_RESIZE
VIDEOMAE2_CKPT
OUT_ROOT
OUT_DIR
EXP_NAME
EXP_OUTPUT
```

### Dataset root overrides

```text
POINTODYSSEY_ROOT
DYNAMIC_REPLICA_ROOT
KUBRIC_FULL_ROOT
KUBRIC_FULL_PROCESSED_ROOT
TARTANAIR_ROOT
VIRTUAL_KITTI2_ROOT
SCANNET_ROOT
BLENDERMVS_ROOTS
CO3D_ROOT
MVS_SYNTH_ROOT
```

### Training overrides

```text
TOTAL_STEPS
WARMUP_STEPS
PEAK_LR
FINAL_LR
TRAIN_BATCH_SIZE
VAL_BATCH_SIZE
TRAIN_NUM_WORKERS
VAL_NUM_WORKERS
SAVE_EVERY_STEPS
STEP_SAVE_EVERY_STEPS
VALIDATE_EVERY_STEPS
VALIDATE_MAX_SAMPLES_GLOBAL
AUTO_EVAL_WORLDTRACK_ENABLED
AUTO_EVAL_WORLDTRACK_NUM_FRAMES
```

### Optional distributed-network variables

Only set these if your cluster needs them:

```text
NCCL_SOCKET_IFNAME
GLOO_SOCKET_IFNAME
```

The scripts do not force a network interface by default.

## 6. Output Layout

By default the run writes to:

```text
output/exp_worldtrack_sota_0512/worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_eval64clip
```

Inside that directory you should expect:

```text
config/
  model_effective.yaml
  train_effective.yaml
checkpoints/
  last.ckpt
  step_0001000.ckpt
  ...
tensorboard/
visualizations/
metrics.jsonl
run_metadata.json
train.log
```

## 7. Validation and Auto Evaluation

This recipe enables two kinds of validation:

1. In-training validation on the configured validation mixtures.
2. Automatic WorldTrack checkpoint evaluation every saved step checkpoint.

The automatic WorldTrack evaluation is controlled by:

```yaml
checkpoint:
  auto_eval_worldtrack_step:
    enabled: true
    num_frames: 64
```

The evaluation helper script is:

```text
scripts/eval_worldtrack/run_batch_eval_worldtrack_step_ckpts.sh
```

If you only want training bring-up, disable it:

```bash
AUTO_EVAL_WORLDTRACK_ENABLED=false
```

This is strongly recommended for the first smoke run.

## 8. TensorBoard and Visualization

`train.py` enables TensorBoard when the script passes `--tb_log`.

Outputs:

```text
<OUT_DIR>/tensorboard
<OUT_DIR>/visualizations
```

The active config also enables intermediate train/val image visualization:

```yaml
logging:
  visualization:
    enabled: true
    train_every_steps: 1000
    on_validation: true
```

## 9. Common Failure Modes

### `Required file not found: checkpoints/...`

The required checkpoint path does not exist. Fix `INIT_CKPT` or `VIDEOMAE2_CKPT`.

### `Expected WORLD_SIZE=8, got 1`

You are doing a local smoke run with fewer GPUs. Set:

```bash
EXPECTED_WORLD_SIZE=1
NPROC_PER_NODE=1
```

### `No valid <dataset> scenes found`

The dataset root exists, but the on-disk structure does not match the loader expectations. Re-check [dataset.md](dataset.md).

### `ScanNet split file not found`

The ScanNet split text files are missing even if the scene data exists.

### `Failed to read EXR depth`

Usually MVS-Synth depth decoding support is broken in the current environment.

### Training hangs during distributed startup

Check:

```text
CUDA visibility
MASTER_ADDR / MASTER_PORT
WORLD_SIZE / NPROC_PER_NODE
cluster-specific NCCL network variables
```

### Immediate OOM

For bring-up, reduce parallelism:

```bash
TRAIN_BATCH_SIZE=1
VAL_BATCH_SIZE=1
TRAIN_NUM_WORKERS=1
VAL_NUM_WORKERS=1
AUTO_EVAL_WORLDTRACK_ENABLED=false
```

## 10. Minimal Repro Commands

### Dry-run

```bash
DRY_RUN=1 \
VIDEOMAE2_CKPT=/path/to/vit_g_hybrid_pt_1200e.pth \
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

### Single-GPU smoke run

```bash
CUDA_VISIBLE_DEVICES=0 \
EXPECTED_WORLD_SIZE=1 \
NPROC_PER_NODE=1 \
TOTAL_STEPS=10 \
SAVE_EVERY_STEPS=10 \
STEP_SAVE_EVERY_STEPS=10 \
AUTO_EVAL_WORLDTRACK_ENABLED=false \
VIDEOMAE2_CKPT=/path/to/vit_g_hybrid_pt_1200e.pth \
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

### Intended 8-GPU run

```bash
VIDEOMAE2_CKPT=/path/to/vit_g_hybrid_pt_1200e.pth \
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```
