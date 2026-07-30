# Public WorldTrack Artifacts

This directory contains small OpenD4RT WorldTrack outputs that are safe to keep
in git. Full demo packages, datasets, checkpoint weights, local logs, and
environment files are intentionally excluded.

## Figures

| File | What it shows |
| --- | --- |
| `opend4rt_demo_overview.png` | Input frame, 2D GT/pred overlay, 3D GT tracks, 3D predicted tracks, checkpoint, metrics, and runtime. |
| `t_cam_query_semantics.png` | The same 5 high-motion query points visualized with `t_cam=t_tgt` and `t_cam=0`. |
| `frame_count_comparison.png` | 32-frame vs 64-frame APD/EPE and runtime comparison using the same 32-clip checkpoint. |

## JSON

| File | Contents |
| --- | --- |
| `worldtrack_evaluation_summary.json` | Full WorldTrack mini evaluation summary, stored as strict JSON. |
| `t_cam_query_semantics.json` | Selected point IDs, frame-0 UVs, motion scores, visible-frame counts, parameters, runtime, and trajectories. |
| `frame_count_comparison.json` | APD, EPE, dynamic APD/EPE, valid query count, and runtime for 32-frame and 64-frame `juggle_5`. |
| `worldtrack_case_analysis.json` | Small best/worst-case summary derived from the full mini evaluation. |

## Key Results

- `pstudio_mini/juggle_5.npz`, 32 frames: APD 0.9916, EPE 0.0533,
  runtime 65.1s.
- `pstudio_mini/juggle_5.npz`, 64 frames via anchor-clip inference: APD 0.9554,
  EPE 0.0693, runtime 148.3s.
- The 64-frame run uses the same `OpenD4RT_32CLIP_9Dataset_NoAUG` checkpoint.
  It is not a 64-frame model context.

## Regenerate

Generate demo packages into `tmp/`, then rebuild the public artifacts:

```bash
python scripts/build_public_artifacts.py \
  --summary-json artifacts/worldtrack_evaluation_summary.json \
  --demo-64-dir tmp/worldtrack_demo_64f \
  --demo-32-dir tmp/worldtrack_demo_32f \
  --device cuda \
  --precision fp16 \
  --max-gpu-memory-gib 20 \
  --num-frames 64 \
  --query-chunk-size 512
```
