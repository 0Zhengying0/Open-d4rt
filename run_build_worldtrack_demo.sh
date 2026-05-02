#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-d4rt}"
fi

if [[ -z "${WORLDTRACK_NPZ:-}" ]]; then
  echo "WORLDTRACK_NPZ is required. Example: WORLDTRACK_NPZ=data/worldtrack_release/adt_mini/example.npz bash $0" >&2
  exit 2
fi

EXP="${EXP:-checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG}"
OUTPUT_DIR="${OUTPUT_DIR:-tmp/worldtrack_demo}"
NUM_FRAMES="${NUM_FRAMES:-24}"
DEVICE="${DEVICE:-cuda}"
POINT_GRID_COLS="${POINT_GRID_COLS:-64}"
POINT_GRID_ROWS="${POINT_GRID_ROWS:-64}"
POINT_MAX_POINTS="${POINT_MAX_POINTS:-4096}"
TRACK_MAX_POINTS="${TRACK_MAX_POINTS:-256}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-1024}"
POINT_QUERY_CHUNK_SIZE="${POINT_QUERY_CHUNK_SIZE:-512}"
CAMERA_QUERY_CHUNK_SIZE="${CAMERA_QUERY_CHUNK_SIZE:-1024}"

python vis/build_like_demo_for_worldtrack.py \
  --config "$EXP/model.yaml" \
  --ckpt-path "$EXP/opend4rt.ckpt" \
  --worldtrack-npz "$WORLDTRACK_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --num-frames "$NUM_FRAMES" \
  --device "$DEVICE" \
  --point-grid-cols "$POINT_GRID_COLS" \
  --point-grid-rows "$POINT_GRID_ROWS" \
  --point-max-points "$POINT_MAX_POINTS" \
  --track-max-points "$TRACK_MAX_POINTS" \
  --query-chunk-size "$QUERY_CHUNK_SIZE" \
  --point-query-chunk-size "$POINT_QUERY_CHUNK_SIZE" \
  --camera-query-chunk-size "$CAMERA_QUERY_CHUNK_SIZE"
