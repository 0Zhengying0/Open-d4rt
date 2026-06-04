# Training Guide

This page keeps only the minimum information needed to prepare the environment,
prepare the data, prepare the checkpoints, and launch the 48-frame 9Mix
training recipe.

Main training script:

```text
scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

## 1. Environment

Use either conda:

```bash
conda env create -f environment.yml
conda activate d4rt
```

or pip:

```bash
pip install -r requirements.txt
```

If you want the training script to auto-activate conda, set:

```bash
CONDA_ENV=d4rt
```

## 2. Required Checkpoints

The training script expects these files by default:

```text
checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt
checkpoints/VideoMAE2/weights/mae-g/vit_g_hybrid_pt_1200e.pth
```

You can override them at launch time:

```bash
INIT_CKPT=/path/to/opend4rt_32clip.ckpt
VIDEOMAE2_CKPT=/path/to/vit_g_hybrid_pt_1200e.pth
```

## 3. Required Datasets

The reproduction config uses the following dataset roots by default:

| Dataset | Config Key | Default Root | Notes |
| --- | --- | --- | --- |
| PointOdyssey | `data.pointodyssey.root` | `data/pointodyssey/v2` | [pointodyssey.md](dataset/pointodyssey.md) |
| Dynamic Replica | `data.dynamic_replica.root` | `data/dynamic-replica/v2` | [dynamic_replica.md](dataset/dynamic_replica.md) |
| Kubric Full (processed) | `data.kubric_full.processed_root` | `data/kubric_full/kubric_full_process_v1` | [kubric_full.md](dataset/kubric_full.md) |
| TartanAir V2 | `data.tartanair.root` | `data/tartanair/v2` | [tartanair.md](dataset/tartanair.md) |
| Virtual KITTI 2 | `data.virtual_kitti2.root` | `data/vitual-kitti-2/v2` | [virtual_kitti2.md](dataset/virtual_kitti2.md) |
| ScanNet / ScanNet++ | `data.scannet.root` | `data/scannet/plus-v2/data` | [scannet.md](dataset/scannet.md) |
| BlenderMVS | `data.blendermvs.roots` | `data/blendermvs/...` | [blendermvs.md](dataset/blendermvs.md) |
| CO3D v2 | `data.co3d.root` | `data/co3d/v2` | [co3d.md](dataset/co3d.md) |
| MVS-Synth | `data.mvs_synth.root` | `data/mvs-synth/v1` | [mvs_synth.md](dataset/mvs_synth.md) |

Dataset index:
[docs/dataset/README.md](dataset/README.md)

If your local paths differ from the defaults, override them with environment
variables before launching the script, for example:

```bash
POINTODYSSEY_ROOT=/path/to/pointodyssey
DYNAMIC_REPLICA_ROOT=/path/to/dynamic-replica
KUBRIC_FULL_PROCESSED_ROOT=/path/to/kubric_processed
TARTANAIR_ROOT=/path/to/tartanair
VIRTUAL_KITTI2_ROOT=/path/to/vkitti2
SCANNET_ROOT=/path/to/scannet_plus
BLENDERMVS_ROOTS=/path/to/root_a,/path/to/root_b,/path/to/root_c
CO3D_ROOT=/path/to/co3d
MVS_SYNTH_ROOT=/path/to/mvs_synth
```

## 4. Preflight Check

Check that configs, checkpoints, and launch settings are valid without starting
training:

```bash
DRY_RUN=1 \
VIDEOMAE2_CKPT=/path/to/vit_g_hybrid_pt_1200e.pth \
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

## 5. One-GPU Smoke Test

Run a short smoke test before the full job:

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

## 6. Full Training Command

Run the intended 8-GPU job:

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

## 7. Effective Configs

The training run uses:

```text
configs/repro/worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_eval64clip/model_effective.yaml
configs/repro/worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_eval64clip/train_effective.yaml
```
