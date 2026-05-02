#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-data/worldtrack_release}"
GIF_CASE_RANK="${GIF_CASE_RANK:-1}"
GIF_CASE_PRESET="${GIF_CASE_PRESET:-}"
GIF_CASES_TO_RUN="${GIF_CASES_TO_RUN:-}"

GIF_CASES=(
  "pstudio_mini/juggle_5.npz"
  "pstudio_mini/softball_25.npz"
  "pstudio_mini/tennis_5.npz"
  "pstudio_mini/football_16.npz"
  "pstudio_mini/basketball_3.npz"
)

RUN_CASES=()
if [[ -n "$GIF_CASES_TO_RUN" ]]; then
  while IFS= read -r case; do
    [[ -n "$case" ]] && RUN_CASES+=("$case")
  done < <(printf '%s\n' "$GIF_CASES_TO_RUN" | tr ',;' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
elif [[ "$GIF_CASE_PRESET" == "sports_three" ]]; then
  RUN_CASES=(
    "pstudio_mini/softball_25.npz"
    "pstudio_mini/football_16.npz"
    "pstudio_mini/basketball_3.npz"
  )
elif [[ -n "${WORLDTRACK_NPZ:-}" ]]; then
  RUN_CASES=("$WORLDTRACK_NPZ")
elif [[ -n "${DEMO_CASE:-}" ]]; then
  RUN_CASES=("$DEMO_CASE")
else
  if ! [[ "$GIF_CASE_RANK" =~ ^[0-9]+$ ]] || (( GIF_CASE_RANK < 1 || GIF_CASE_RANK > ${#GIF_CASES[@]} )); then
    echo "GIF_CASE_RANK must be between 1 and ${#GIF_CASES[@]}." >&2
    exit 2
  else
    RUN_CASES=("${GIF_CASES[$((GIF_CASE_RANK - 1))]}")
  fi
fi

EXP="${EXP:-checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG}"
OUTPUT_ROOT="${OUTPUT_ROOT:-tmp/vis_gif}"
NUM_FRAMES="${NUM_FRAMES:-32}"
DEVICE="${DEVICE:-cuda}"
GIF_MAX_TRACKS="${GIF_MAX_TRACKS:-64}"
MAX_GIF_FRAMES="${MAX_GIF_FRAMES:-32}"
GIF_FPS="${GIF_FPS:-10}"
GIF_OUTPUTS="${GIF_OUTPUTS:-gt_pred_3d,rgb_gt_pred_2d}"
TRAIL_LENGTH="${TRAIL_LENGTH:-32}"
HIDE_FRAME_LABELS="${HIDE_FRAME_LABELS:-1}"
CLEAN_OUTPUT_GIFS="${CLEAN_OUTPUT_GIFS:-1}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-1024}"
POINT_QUERY_CHUNK_SIZE="${POINT_QUERY_CHUNK_SIZE:-256}"
CAMERA_QUERY_CHUNK_SIZE="${CAMERA_QUERY_CHUNK_SIZE:-1024}"

LABEL_ARGS=()
if [[ "$HIDE_FRAME_LABELS" != "0" ]]; then
  LABEL_ARGS+=(--hide-frame-labels)
fi
CLEAN_ARGS=()
if [[ "$CLEAN_OUTPUT_GIFS" != "0" ]]; then
  CLEAN_ARGS+=(--clean-output-gifs)
fi

for CASE_SPEC in "${RUN_CASES[@]}"; do
  if [[ "$CASE_SPEC" = /* || -f "$CASE_SPEC" ]]; then
    CASE_NPZ="$CASE_SPEC"
  else
    CASE_NPZ="$DATA_ROOT/$CASE_SPEC"
  fi

  if [[ ! -f "$CASE_NPZ" ]]; then
    echo "WorldTrack case not found: $CASE_NPZ" >&2
    echo "Set DATA_ROOT, WORLDTRACK_NPZ, DEMO_CASE, GIF_CASES_TO_RUN, GIF_CASE_PRESET, or GIF_CASE_RANK." >&2
    echo "Recommended complex-motion cases:" >&2
    for case in "${GIF_CASES[@]}"; do
      echo "  $DATA_ROOT/$case" >&2
    done
    exit 2
  fi

  CASE_NAME="$(basename "$CASE_NPZ" .npz)"
  if [[ -n "${OUTPUT_DIR:-}" && "${#RUN_CASES[@]}" -eq 1 ]]; then
    CASE_OUTPUT_DIR="$OUTPUT_DIR"
  elif [[ -n "${OUTPUT_DIR:-}" ]]; then
    CASE_OUTPUT_DIR="$OUTPUT_DIR/$CASE_NAME"
  else
    CASE_OUTPUT_DIR="$OUTPUT_ROOT/$CASE_NAME"
  fi

  echo "Using WorldTrack GIF case: $CASE_NPZ"
  echo "Writing GIFs to: $CASE_OUTPUT_DIR"

  python vis/build_worldtrack_track_gifs.py \
    --config "$EXP/model.yaml" \
    --ckpt-path "$EXP/opend4rt.ckpt" \
    --worldtrack-npz "$CASE_NPZ" \
    --output-dir "$CASE_OUTPUT_DIR" \
    --num-frames "$NUM_FRAMES" \
    --device "$DEVICE" \
    --gif-max-tracks "$GIF_MAX_TRACKS" \
    --max-gif-frames "$MAX_GIF_FRAMES" \
    --outputs "$GIF_OUTPUTS" \
    --trail-length "$TRAIL_LENGTH" \
    --fps "$GIF_FPS" \
    --query-chunk-size "$QUERY_CHUNK_SIZE" \
    --point-query-chunk-size "$POINT_QUERY_CHUNK_SIZE" \
    --camera-query-chunk-size "$CAMERA_QUERY_CHUNK_SIZE" \
    "${CLEAN_ARGS[@]}" \
    "${LABEL_ARGS[@]}"
done
