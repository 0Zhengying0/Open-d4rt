<div align="center">
  <h1>OpenD4RT</h1>
  <p><b>An unofficial PyTorch/GPU implementation of D4RT for 4D reconstruction and tracking</b></p>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="docs/D4RT_paper.pdf"><img src="https://img.shields.io/badge/report-D4RT%20PDF-lightgrey.svg" alt="D4RT paper"></a>
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/task-WorldTrack%203D%20Tracking-green.svg" alt="WorldTrack">
</div>

OpenD4RT is an unofficial open-source PyTorch/GPU implementation of D4RT,
developed to reproduce the model architecture, training recipe, evaluation
protocols, and implementation details described in the D4RT paper and
appendix. The current public repo includes the released Hugging Face
checkpoint, the model, WorldTrack evaluation, and Viser visualization
tools, with complete training and evaluation code planned for release.

<p align="center">
  <img src="docs/image.png" width="950" alt="D4RT overview">
</p>

## What is D4RT?

D4RT is a feedforward video model for reconstructing and tracking dynamic
scenes. It uses a unified transformer architecture to infer depth,
spatio-temporal correspondence, and camera parameters from a single video. Its
query interface probes the 3D position of a source pixel `(u, v, t_src)` at a
target timestep `t_tgt` in a selected camera coordinate frame `t_cam`, enabling
sparse tracking, all-pixel tracking, and 4D scene reconstruction through the
same model interface.

See [docs/D4RT_paper.pdf](docs/D4RT_paper.pdf) for the local paper PDF
included in this repository.

## Repository Layout

```text
OpenD4RT/
  checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/
    opend4rt.ckpt        # downloaded from Hugging Face
    model.yaml
  eval_track3d_in_worldtrack.py
  infer_track_3d.py
  run_eval_worldtrack.sh
  run_build_worldtrack_demo.sh
  src/
  vis/
  docs/image.png
```

## Checkpoint Zoo

| Name | Training data | Augmentation | Input length | Status | Checkpoint path | Hugging Face |
| --- | --- | --- | --- | --- | --- | --- |
| OpenD4RT_32CLIP_9Dataset_NoAUG | 9-dataset mixture | No color/crop augmentation | 32 frames | Released | `checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt` | [ckpt](https://huggingface.co/Lijiaxin0111/OpenD4RT/blob/main/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt) / [config](https://huggingface.co/Lijiaxin0111/OpenD4RT/blob/main/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml) |
| OpenD4RT-9Mix-NoCropAug-Clip48 | 9-dataset mixture | No crop augmentation | 48 frames | Coming | TBD |  |
| OpenD4RT-9Mix-Aug-Clip48 | 9-dataset mixture | With data augmentation | 48 frames | Coming | TBD |  |
| OpenD4RT-10Mix-SynthVerse-NoAug-Clip32 | 10-dataset mixture | No data augmentation | 32 frames | Coming | TBD |  |
| OpenD4RT-10Mix-SynthVerse-Aug-Clip48 | 10-dataset mixture | With data augmentation | 48 frames | Coming | TBD |  |

Tip: the 9-dataset mixture uses PointOdyssey, Dynamic Replica, Kubric Full,
TartanAir, Virtual KITTI 2, ScanNet, BlendedMVS, CO3D, and MVS-Synth. The
10-dataset mixture additionally includes SynthVerse.

## Checkpoint Download

Download the released checkpoint and model config from
[Lijiaxin0111/OpenD4RT](https://huggingface.co/Lijiaxin0111/OpenD4RT/tree/main/checkpoints)
into the default path used by the scripts:

```bash
pip install -U huggingface_hub

huggingface-cli download Lijiaxin0111/OpenD4RT \
  --repo-type model \
  --include "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt" \
  --include "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml" \
  --local-dir .
```

Expected local files:

```text
checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/
  opend4rt.ckpt
  model.yaml
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate d4rt
```

Or install into an existing Python environment:

```bash
pip install -r requirements.txt
```

The visualization package builder calls the `ffmpeg` command-line tool to
write MP4 assets for Viser. The conda environment includes `ffmpeg`; if you use
`pip install -r requirements.txt`, install `ffmpeg` separately if needed.

## WorldTrack Data

Download the WorldTrack release from:

```text
https://drive.google.com/drive/folders/1-JW88ru30irMYyFab_4YBQbGbd9tKpXV
```

Place the `.npz` files under:

```text
data/worldtrack_release/
  adt_mini/*.npz
  po_mini/*.npz
  pstudio_mini/*.npz
  ds_mini/*.npz
```

## Evaluation

Run a quick smoke test on one `adt_mini` sequence:

```bash
LIMIT_SEQS=1 SUBSETS=adt_mini NUM_FRAMES=24 OUTPUT_DIR=tmp/eval_smoke bash run_eval_worldtrack.sh
```

Run the full WorldTrack evaluation:

```bash
bash run_eval_worldtrack.sh
```

Equivalent explicit command:

```bash
EXP=checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG

python eval_track3d_in_worldtrack.py \
  --model-config "$EXP/model.yaml" \
  --ckpt-path "$EXP/opend4rt.ckpt" \
  --data-root data/worldtrack_release \
  --subsets adt_mini,po_mini,pstudio_mini,ds_mini \
  --num-frames 24 \
  --query-chunk-size 4096 \
  --output-dir tmp/eval_worldtrack \
  --device cuda \
  --save-per-sequence
```

Useful overrides:

```bash
QUERY_CHUNK_SIZE=1024 bash run_eval_worldtrack.sh
CUDA_VISIBLE_DEVICES=1 DEVICE=cuda bash run_eval_worldtrack.sh
SUBSETS=adt_mini LIMIT_SEQS=1 NUM_FRAMES=24 bash run_eval_worldtrack.sh
```

## Results

```

| Subset | APD global | EPE global | APD global dyn | EPE global dyn | Queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 0.7434 | 0.2477 | 0.8345 | 0.2251 | 22187 |
| `po_mini` | 0.6912 | 0.3186 | 0.7819 | 0.2544 | 53468 |
| `pstudio_mini` | 0.8253 | 0.1560 | 0.8259 | 0.1572 | 8720 |
| `ds_mini` | 0.7247 | 0.2980 | 0.7622 | 0.2647 | 52462 |

## Viser Demo Visualization

Build a Viser demo data package for one WorldTrack `.npz` case:

```bash
WORLDTRACK_NPZ=$(find data/worldtrack_release/adt_mini -name '*.npz' | head -n 1)
WORLDTRACK_NPZ="$WORLDTRACK_NPZ" OUTPUT_DIR=tmp/worldtrack_demo bash run_build_worldtrack_demo.sh
```

For a lighter/faster package, reduce the point and track counts:

```bash
WORLDTRACK_NPZ="$WORLDTRACK_NPZ" \
OUTPUT_DIR=tmp/worldtrack_demo_small \
POINT_GRID_COLS=32 POINT_GRID_ROWS=32 POINT_MAX_POINTS=1024 TRACK_MAX_POINTS=96 \
bash run_build_worldtrack_demo.sh
```

Start the interactive Viser viewer:

```bash
python vis/serve_demo_viser.py --root tmp/worldtrack_demo --port 8081
```

The generated demo package contains `assets/demo_data.json`,
`assets/input_video.mp4`, rendered diagnostic videos, and `manifest.json`.

## ToDo

- [x] Release the OpenD4RT model runtime for the 32-frame 9-dataset checkpoint.
- [x] Release WorldTrack evaluation scripts and archived metrics.
- [x] Release Viser-based qualitative visualization tools.
- [ ] Release complete training code.
- [ ] Release additional checkpoints listed in the Checkpoint Zoo.
- [ ] Release SynthVerse evaluation results.
- [ ] Release full evaluation code for the benchmarks reported in the D4RT
  paper and appendix.

## License

OpenD4RT is an unofficial implementation and is not affiliated with or endorsed
by the original D4RT authors. The code in this repository is released under the
Apache 2.0 license; see [LICENSE](LICENSE). The D4RT paper, project page,
datasets, third-party assets, and upstream dependencies remain under their
respective licenses and terms.

## Citation

If OpenD4RT is useful for your research, please cite the original D4RT paper:

```bibtex
@article{zhang2025d4rt,
  title={Efficiently Reconstructing Dynamic Scenes One D4RT at a Time},
  author={Zhang, Chuhan and Le Moing, Guillaume and Koppula, Skanda and Rocco, Ignacio and Momeni, Liliane and Xie, Junyu and Sun, Shuyang and Sukthankar, Rahul and Barral, Jo{\"e}lle K. and Hadsell, Raia and Ghahramani, Zoubin and Zisserman, Andrew and Zhang, Junlin and Sajjadi, Mehdi S. M.},
  journal={arXiv preprint},
  year={2025}
}
```

Official D4RT project page: <https://d4rt-paper.github.io/>.

## Acknowledgements

This project is built upon the D4RT paper and official project materials. We
thank the original D4RT authors for introducing the D4RT formulation, releasing
the project page, and documenting the paper and appendix details that this
implementation follows. We also acknowledge the contributors and resources
credited on the official D4RT website, including colleagues who supported
project advice, manuscript feedback, early development, code review,
visualization, baseline comparisons, and data generation. We also thank the
splat viewer authors for the WebGL renderer used by the official D4RT
visualization pipeline. Please refer to the official D4RT project page for the
full original acknowledgements.
