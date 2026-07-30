#!/usr/bin/env python3
"""Build lightweight public artifacts from existing WorldTrack outputs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_track3d_in_worldtrack import PIXEL_TO_FIXED_METRIC_THRESH
from infer_track_3d import _infer_tracks, _resolve_device, _resize_video
from src.core import (
    build_inference_model_from_checkpoint,
    configure_cpu_thread_limits,
    load_yaml_config,
    resource_snapshot,
    seed_everything,
)
from src.model import build_model
from vis.build_like_demo import _compute_point_motion_scores
from vis.build_like_demo_for_worldtrack import _disable_encoder_pretrain, load_worldtrack_sequence


LOGGER = logging.getLogger("build_public_artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact public artifacts for D4RT WorldTrack demos.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--summary-json", default="artifacts/worldtrack_evaluation_summary.json")
    parser.add_argument("--demo-64-dir", default="tmp/worldtrack_demo_64f")
    parser.add_argument("--demo-32-dir", default="tmp/worldtrack_demo_32f")
    parser.add_argument("--worldtrack-npz", default="data/worldtrack_release/pstudio_mini/juggle_5.npz")
    parser.add_argument("--config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--precision", default="auto", choices=("auto", "fp32", "fp16"))
    parser.add_argument("--max-gpu-memory-gib", type=float, default=20.0)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--t-cam-query-count", type=int, default=5)
    parser.add_argument("--t-cam-min-visible-frames", type=int, default=16)
    parser.add_argument("--t-cam-min-spatial-distance-px", type=float, default=10.0)
    parser.add_argument("--min-dynamic-queries", type=int, default=20)
    parser.add_argument("--skip-t-cam", action="store_true")
    return parser.parse_args()


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _sanitize_json(value.tolist())
    if isinstance(value, np.generic):
        return _sanitize_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitize_json(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _public_sequence_label(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    marker = "worldtrack_release/"
    if marker in text:
        return text.split(marker, 1)[1]
    return text


def _sanitize_summary_for_public(summary: dict[str, Any]) -> dict[str, Any]:
    public = _sanitize_json(summary)

    def scrub_paths(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: scrub_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scrub_paths(item) for item in value]
        if isinstance(value, str):
            label = _public_sequence_label(value)
            if label is not None and label != value:
                return label
            checkpoint_marker = "checkpoints/"
            if checkpoint_marker in value:
                return value.split(checkpoint_marker, 1)[1]
        return value

    public = scrub_paths(public)
    inputs = public.get("inputs")
    if isinstance(inputs, dict):
        ckpt_path = str(inputs.pop("ckpt_path", "") or "")
        if ckpt_path:
            parts = Path(ckpt_path).parts
            if len(parts) >= 2:
                inputs["checkpoint"] = parts[-2]
        inputs.pop("data_root", None)
    return public


def _to_array(payload: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    return np.asarray(payload, dtype=dtype)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _extract_video_frame(video_path: Path, frame_index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = int(np.clip(frame_index, 0, max(total - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to decode frame {target} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _write_poster_for_video(video_path: Path, poster_path: Path) -> None:
    frame = _extract_video_frame(video_path, 0)
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(poster_path), frame[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def refresh_manifest_posters(demo_dir: Path) -> dict[str, Any]:
    manifest_path = demo_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    pairs = [
        ("video_copy", "video_poster"),
        ("depth_pred_video", "depth_pred_poster"),
        ("tracks_2d_overlay", "tracks_2d_overlay_poster"),
        ("tracks_3d", "tracks_3d_poster"),
    ]
    for video_key, poster_key in pairs:
        rel_video = manifest.get(video_key)
        if not rel_video:
            continue
        video_path = demo_dir / str(rel_video)
        poster_name = f"{Path(str(rel_video)).stem}_poster.jpg"
        poster_rel = str(Path(str(rel_video)).parent / poster_name)
        _write_poster_for_video(video_path, demo_dir / poster_rel)
        manifest[poster_key] = poster_rel
    _write_json(manifest_path, manifest)
    return manifest


def _apply_visibility(xyz_qt3: np.ndarray, visibility_qt: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz_qt3, dtype=np.float64).copy()
    vis = np.asarray(visibility_qt, dtype=bool)
    xyz[~vis] = np.nan
    return xyz


def _global_scale(gt_tq3: np.ndarray, pred_tq3: np.ndarray) -> float:
    gt = np.asarray(gt_tq3, dtype=np.float64).reshape(-1, 3)
    pred = np.asarray(pred_tq3, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(gt).all(axis=-1) & np.isfinite(pred).all(axis=-1)
    if not np.any(finite):
        return 1.0
    gt_norm = np.linalg.norm(gt[finite], axis=-1)
    pred_norm = np.linalg.norm(pred[finite], axis=-1)
    return float(np.median(np.maximum(gt_norm, 1e-12)) / max(float(np.median(np.maximum(pred_norm, 1e-12))), 1e-12))


def _apd_epe(gt_tq3: np.ndarray, pred_tq3: np.ndarray) -> tuple[float | None, float | None, dict[str, float]]:
    scale = _global_scale(gt_tq3, pred_tq3)
    pred_aligned = np.asarray(pred_tq3, dtype=np.float64) * scale
    dists = np.linalg.norm(pred_aligned - np.asarray(gt_tq3, dtype=np.float64), axis=-1)
    finite = np.isfinite(dists)
    if not np.any(finite):
        return None, None, {}
    fractions: dict[str, float] = {}
    for key, threshold in PIXEL_TO_FIXED_METRIC_THRESH.items():
        fractions[str(key)] = float(np.count_nonzero(finite & (dists <= float(threshold))) / max(int(np.count_nonzero(finite)), 1))
    return float(np.mean(list(fractions.values()))), float(np.mean(dists[finite])), fractions


def compute_demo_metrics(demo_dir: Path) -> dict[str, Any]:
    data = _load_json(demo_dir / "assets" / "demo_data.json")
    runtime = _load_json(demo_dir / "runtime_resources.json")
    gt_qt3 = _to_array(data["tracksGt"]["xyzRef0"], np.float64)
    pred_qt3 = _to_array(data["tracks"]["xyzRef0"], np.float64)
    gt_vis_qt = _to_array(data["tracksGt"]["visibility"], np.int32).astype(bool)
    pred_vis_qt = _to_array(data["tracks"]["visibility"], np.int32).astype(bool)
    shared_vis = gt_vis_qt & pred_vis_qt
    gt_tq3 = np.transpose(_apply_visibility(gt_qt3, shared_vis), (1, 0, 2))
    pred_tq3 = np.transpose(_apply_visibility(pred_qt3, shared_vis), (1, 0, 2))
    apd, epe, fractions = _apd_epe(gt_tq3, pred_tq3)

    gt_visible = _apply_visibility(gt_qt3, gt_vis_qt)
    gt_visible_tq3 = np.transpose(gt_visible, (1, 0, 2))
    motion = np.linalg.norm(gt_visible_tq3[1:] - gt_visible_tq3[:-1], axis=-1)
    dyn_score = np.nansum(motion, axis=0)
    dyn_mask = np.isfinite(dyn_score) & (dyn_score > 0.01)
    dyn_apd = dyn_epe = None
    dyn_fractions: dict[str, float] = {}
    if int(np.count_nonzero(dyn_mask)) > 0:
        dyn_apd, dyn_epe, dyn_fractions = _apd_epe(gt_tq3[:, dyn_mask], pred_tq3[:, dyn_mask])

    meta = data.get("meta", {})
    worldtrack = meta.get("worldtrack", {})
    valid_query_count = int(np.count_nonzero(np.any(np.isfinite(gt_tq3).all(axis=-1), axis=0)))
    return {
        "demo_dir": demo_dir.as_posix(),
        "sequence": _public_sequence_label(worldtrack.get("npz")),
        "num_frames": int(meta.get("numFrames", gt_qt3.shape[1])),
        "clip_frames": int(meta.get("clipFrames", 32)),
        "checkpoint": "OpenD4RT_32CLIP_9Dataset_NoAUG",
        "comparison_note": "Same 32-clip checkpoint; 64-frame sequences are handled by anchor-clip inference rather than a longer model context.",
        "valid_query_count": valid_query_count,
        "dynamic_query_count": int(np.count_nonzero(dyn_mask)),
        "apd_global": apd,
        "epe_global": epe,
        "fractions_global": fractions,
        "apd_global_dynamic": dyn_apd,
        "epe_global_dynamic": dyn_epe,
        "fractions_global_dynamic": dyn_fractions,
        "runtime": runtime,
    }


def build_frame_count_comparison(artifact_root: Path, demo_32_dir: Path, demo_64_dir: Path) -> dict[str, Any]:
    metrics_32 = compute_demo_metrics(demo_32_dir)
    metrics_64 = compute_demo_metrics(demo_64_dir)
    metrics_32.pop("demo_dir", None)
    metrics_64.pop("demo_dir", None)
    metrics_32["case"] = "32_frames"
    metrics_64["case"] = "64_frames_anchor_clip"
    payload = {
        "description": "Frame-count quality comparison on the same WorldTrack juggle_5 case.",
        "important_note": "Uses the same 32-clip checkpoint. The 32-frame case fits in one model clip; the 64-frame case is processed by anchor-clip inference, not by increasing model context to 64.",
        "cases": {
            "32_frames": metrics_32,
            "64_frames_anchor_clip": metrics_64,
        },
    }
    _write_json(artifact_root / "frame_count_comparison.json", payload)

    labels = ["32 frames", "64 frames\n(anchor-clip)"]
    apd = [metrics_32["apd_global"], metrics_64["apd_global"]]
    epe = [metrics_32["epe_global"], metrics_64["epe_global"]]
    dyn_apd = [metrics_32["apd_global_dynamic"], metrics_64["apd_global_dynamic"]]
    dyn_epe = [metrics_32["epe_global_dynamic"], metrics_64["epe_global_dynamic"]]
    runtime = [
        metrics_32["runtime"].get("elapsed_seconds"),
        metrics_64["runtime"].get("elapsed_seconds"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=180)
    x = np.arange(2)
    axes[0].bar(x - 0.16, [v or 0 for v in apd], width=0.32, label="APD")
    axes[0].bar(x + 0.16, [v or 0 for v in dyn_apd], width=0.32, label="dynamic APD")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("APD higher is better")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x - 0.16, [v or 0 for v in epe], width=0.32, label="EPE")
    axes[1].bar(x + 0.16, [v or 0 for v in dyn_epe], width=0.32, label="dynamic EPE")
    axes[1].set_title("EPE lower is better")
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(x, [v or 0 for v in runtime], width=0.45, color=["#2f7f6f", "#9b5d3b"])
    axes[2].set_title("Runtime seconds")
    axes[2].set_xticks(x, labels)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Same 32-clip checkpoint: 32 frames vs 64-frame anchor-clip inference", fontsize=11)
    fig.tight_layout()
    fig.savefig(artifact_root / "frame_count_comparison.png")
    plt.close(fig)
    return payload


def _sequence_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset, subset_payload in summary.get("subsets", {}).items():
        for seq in subset_payload.get("sequences", []):
            row = dict(seq)
            row["subset"] = subset
            rows.append(row)
    return rows


def analyze_summary(summary_path: Path, artifact_root: Path, min_dynamic_queries: int) -> dict[str, Any]:
    summary = _sanitize_summary_for_public(_load_json(summary_path))
    _write_json(summary_path, summary)
    rows = _sequence_rows(summary)
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        runtime = row.get("runtime", {}) if isinstance(row.get("runtime"), dict) else {}
        flat_rows.append(
            {
                "subset": row.get("subset"),
                "video_name": row.get("video_name"),
                "sequence_path": row.get("sequence_path"),
                "avg_pts_global": _finite_float(row.get("avg_pts_global")),
                "epe_global": _finite_float(row.get("epe_global")),
                "avg_pts_global_dyn": _finite_float(row.get("avg_pts_global_dyn")),
                "epe_global_dyn": _finite_float(row.get("epe_global_dyn")),
                "dyn_count": int(row.get("dyn_count", 0) or 0),
                "dyn_fraction": _finite_float(row.get("dyn_fraction")),
                "num_queries": int(row.get("num_queries", 0) or 0),
                "elapsed_seconds": _finite_float(runtime.get("elapsed_seconds")),
                "cuda_peak_reserved_gib": _finite_float(runtime.get("cuda_peak_reserved_gib")),
            }
        )

    overall = [row for row in flat_rows if row["avg_pts_global"] is not None and row["epe_global"] is not None]
    dynamic = [
        row for row in flat_rows
        if row["dyn_count"] >= int(min_dynamic_queries)
        and row["avg_pts_global_dyn"] is not None
        and row["epe_global_dyn"] is not None
    ]

    analysis = {
        "source_summary": summary_path.as_posix(),
        "filters": {"min_dynamic_queries": int(min_dynamic_queries)},
        "sequence_count": len(flat_rows),
        "overall": {
            "best_by_apd": max(overall, key=lambda row: row["avg_pts_global"]) if overall else None,
            "worst_by_apd": min(overall, key=lambda row: row["avg_pts_global"]) if overall else None,
            "best_by_epe": min(overall, key=lambda row: row["epe_global"]) if overall else None,
            "worst_by_epe": max(overall, key=lambda row: row["epe_global"]) if overall else None,
        },
        "dynamic": {
            "best_by_dynamic_apd": max(dynamic, key=lambda row: row["avg_pts_global_dyn"]) if dynamic else None,
            "worst_by_dynamic_apd": min(dynamic, key=lambda row: row["avg_pts_global_dyn"]) if dynamic else None,
            "best_by_dynamic_epe": min(dynamic, key=lambda row: row["epe_global_dyn"]) if dynamic else None,
            "worst_by_dynamic_epe": max(dynamic, key=lambda row: row["epe_global_dyn"]) if dynamic else None,
        },
    }
    _write_json(artifact_root / "worldtrack_case_analysis.json", analysis)

    return analysis


def _fmt_optional(value: Any) -> str:
    number = _finite_float(value)
    return "null" if number is None else f"{number:.4f}"


def _track_colors(n: int) -> np.ndarray:
    cmap = plt.get_cmap("tab10")
    return np.asarray([cmap(i % 10)[:3] for i in range(n)], dtype=np.float32)


def _set_equal_3d(ax: Any, arrays: list[np.ndarray]) -> None:
    pts = []
    for arr in arrays:
        flat = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
        valid = np.isfinite(flat).all(axis=-1)
        if np.any(valid):
            pts.append(flat[valid])
    if not pts:
        return
    all_pts = np.concatenate(pts, axis=0)
    lo = np.min(all_pts, axis=0)
    hi = np.max(all_pts, axis=0)
    center = (lo + hi) * 0.5
    radius = max(float(np.max(hi - lo)) * 0.55, 0.1)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def build_t_cam_semantics_demo(args: argparse.Namespace, artifact_root: Path) -> dict[str, Any]:
    configure_cpu_thread_limits()
    cfg = load_yaml_config(args.config)
    _disable_encoder_pretrain(cfg)
    seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)
    device = _resolve_device(args.device)
    model, runtime, _ = build_inference_model_from_checkpoint(
        lambda: build_model(cfg["model"]),
        checkpoint_path=args.ckpt_path,
        device=device,
        precision=args.precision,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
    )

    sample = load_worldtrack_sequence(Path(args.worldtrack_npz), num_frames=int(args.num_frames))
    video_rgb = np.asarray(sample["video_rgb"], dtype=np.uint8)
    image_size = cfg.get_path("model.input.image_size", [256, 256])
    video_model_rgb = _resize_video(video_rgb, image_hw=(int(image_size[0]), int(image_size[1])))
    visibility = np.asarray(sample["visibility"], dtype=bool)
    tracks_uv = np.asarray(sample["tracks_uv"], dtype=np.float32)
    tracks_cam = np.asarray(sample["tracks_xyz_cam"], dtype=np.float32)
    tracks_world = np.asarray(sample["tracks_xyz_world"], dtype=np.float32)

    frame0_vis = visibility[0]
    frame0_ids = np.flatnonzero(frame0_vis)
    uv0 = tracks_uv[0, frame0_ids]
    depth0 = tracks_cam[0, frame0_ids, 2]
    keep = np.isfinite(uv0).all(axis=-1) & np.isfinite(depth0) & (depth0 > 1e-6)
    candidate_ids = frame0_ids[keep]
    candidate_world_tq3 = tracks_world[:, candidate_ids]
    candidate_vis_tq = visibility[:, candidate_ids]
    motion_scores, visible_counts = _compute_point_motion_scores(
        xyz_ref0=candidate_world_tq3,
        visibility=candidate_vis_tq,
        confidence=np.ones_like(candidate_vis_tq, dtype=np.float32),
    )
    eligible = visible_counts >= int(args.t_cam_min_visible_frames)
    scores = np.where(eligible, motion_scores + 0.01 * visible_counts.astype(np.float32), -np.inf)
    ranked_local = np.argsort(scores)[::-1]
    ranked_local = ranked_local[np.isfinite(scores[ranked_local])]
    selected_list: list[int] = []
    selected_uv: list[np.ndarray] = []
    min_distance = max(0.0, float(args.t_cam_min_spatial_distance_px))
    requested_count = max(1, int(args.t_cam_query_count))
    for local_idx in ranked_local.tolist():
        uv = np.asarray(tracks_uv[0, candidate_ids[int(local_idx)]], dtype=np.float32)
        if not np.isfinite(uv).all():
            continue
        if selected_uv:
            dists = [float(np.linalg.norm(uv - prev_uv)) for prev_uv in selected_uv]
            if min(dists) < min_distance:
                continue
        selected_list.append(int(local_idx))
        selected_uv.append(uv)
        if len(selected_list) >= requested_count:
            break
    selected_local = np.asarray(selected_list, dtype=np.int64)
    if selected_local.size <= 0:
        raise RuntimeError("No eligible high-motion queries found for t_cam semantics demo.")
    if selected_local.size < requested_count:
        LOGGER.warning(
            "Only selected %d/%d spatially distinct t_cam queries with min distance %.1f px.",
            int(selected_local.size),
            int(requested_count),
            float(min_distance),
        )

    selected_point_ids = candidate_ids[selected_local]
    query_uv_px = tracks_uv[0, selected_point_ids].astype(np.float32)
    query_uv_norm = query_uv_px.copy()
    query_uv_norm[:, 0] /= float(max(video_rgb.shape[2] - 1, 1))
    query_uv_norm[:, 1] /= float(max(video_rgb.shape[1] - 1, 1))
    query_uv_norm = np.clip(query_uv_norm, 0.0, 1.0)

    started = time.perf_counter()
    pred = _infer_tracks(
        model=model,
        video_model_rgb=video_model_rgb,
        query_uv_norm=query_uv_norm.astype(np.float32),
        query_chunk_size=int(args.query_chunk_size),
    )
    elapsed = time.perf_counter() - started
    local_qt3 = np.asarray(pred["tracks_xyz_local"], dtype=np.float32)
    ref0_qt3 = np.asarray(pred["tracks_xyz_ref0"], dtype=np.float32)
    vis_qt = np.asarray(pred["tracks_visibility"], dtype=bool)

    colors = _track_colors(int(selected_point_ids.shape[0]))
    fig = plt.figure(figsize=(14, 5.6), dpi=180)
    ax_img = fig.add_subplot(1, 3, 1)
    ax_local = fig.add_subplot(1, 3, 2, projection="3d")
    ax_ref = fig.add_subplot(1, 3, 3, projection="3d")

    ax_img.imshow(video_rgb[0])
    ax_img.set_title("Frame-0 query points")
    ax_img.axis("off")
    for qi, uv in enumerate(query_uv_px):
        ax_img.scatter([uv[0]], [uv[1]], s=70, color=colors[qi], edgecolor="white", linewidth=1.0)
        ax_img.text(float(uv[0]) + 4, float(uv[1]) + 4, str(qi + 1), color="white", fontsize=8, weight="bold")

    for qi in range(int(selected_point_ids.shape[0])):
        valid = vis_qt[qi] & np.isfinite(local_qt3[qi]).all(axis=-1)
        if int(np.count_nonzero(valid)) >= 2:
            pts = local_qt3[qi, valid]
            ax_local.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[qi], linewidth=1.8)
        valid = vis_qt[qi] & np.isfinite(ref0_qt3[qi]).all(axis=-1)
        if int(np.count_nonzero(valid)) >= 2:
            pts = ref0_qt3[qi, valid]
            ax_ref.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[qi], linewidth=1.8)

    ax_local.set_title("t_cam = t_tgt\ncurrent camera coordinates")
    ax_ref.set_title("t_cam = 0\nfixed reference coordinates")
    for ax in (ax_local, ax_ref):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=24, azim=45)
        ax.grid(alpha=0.25)
    _set_equal_3d(ax_local, [local_qt3])
    _set_equal_3d(ax_ref, [ref0_qt3])
    fig.suptitle("D4RT query semantics: same (u, v, t_src, t_tgt), different t_cam", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    out_png = artifact_root / "t_cam_query_semantics.png"
    fig.savefig(out_png)
    plt.close(fig)

    selected = []
    for qi, point_id in enumerate(selected_point_ids.tolist()):
        selected.append(
            {
                "label": qi + 1,
                "worldtrack_point_id": int(point_id),
                "query_uv_px": [float(query_uv_px[qi, 0]), float(query_uv_px[qi, 1])],
                "motion_score": float(motion_scores[selected_local[qi]]),
                "visible_frames": int(visible_counts[selected_local[qi]]),
            }
        )
    payload = {
        "description": "Same selected query points run with t_cam=t_tgt and t_cam=0 to show D4RT coordinate-frame semantics.",
        "implementation_note": "_infer_tracks encodes each clip once and evaluates current-camera and reference-camera query heads against the shared encoded memory.",
        "checkpoint": "OpenD4RT_32CLIP_9Dataset_NoAUG",
        "sequence": _public_sequence_label(args.worldtrack_npz),
        "num_frames": int(args.num_frames),
        "query_chunk_size": int(args.query_chunk_size),
        "selection": {
            "requested_query_count": int(requested_count),
            "selected_query_count": int(selected_local.size),
            "min_visible_frames": int(args.t_cam_min_visible_frames),
            "min_spatial_distance_px": float(args.t_cam_min_spatial_distance_px),
            "ranking_score": "motion_score + 0.01 * visible_frame_count",
        },
        "clip_frames": int(pred["clip_frames"]),
        "runtime": {
            "elapsed_seconds": elapsed,
            "precision": runtime.precision,
            "device": str(runtime.device),
            **resource_snapshot(device),
        },
        "selected_queries": selected,
        "trajectories": {
            "t_cam_equals_t_tgt_xyz_current_camera": np.round(local_qt3, 5).tolist(),
            "t_cam_equals_0_xyz_reference": np.round(ref0_qt3, 5).tolist(),
            "visibility": vis_qt.astype(np.int32).tolist(),
        },
        "outputs": {"figure": out_png.as_posix()},
    }
    _write_json(artifact_root / "t_cam_query_semantics.json", payload)
    return payload


def _draw_2d_overlay_from_data(frame_rgb: np.ndarray, data: dict[str, Any], frame_idx: int, max_tracks: int = 80) -> np.ndarray:
    out = np.asarray(frame_rgb, dtype=np.uint8).copy()
    gt_uv = _to_array(data["tracksGt"]["uvPx"], np.float32)
    pred_uv = _to_array(data["tracks"]["uvPx"], np.float32)
    gt_vis = _to_array(data["tracksGt"]["visibility"], np.int32).astype(bool)
    pred_vis = _to_array(data["tracks"]["visibility"], np.int32).astype(bool)
    count = min(int(gt_uv.shape[0]), int(max_tracks))
    colors = (_track_colors(count) * 255).astype(np.uint8)
    for qi in range(count):
        color = tuple(int(v) for v in colors[qi].tolist())
        if bool(gt_vis[qi, frame_idx]) and np.isfinite(gt_uv[qi, frame_idx]).all():
            p = tuple(np.rint(gt_uv[qi, frame_idx]).astype(np.int32))
            cv2.circle(out, p, 3, color, -1, cv2.LINE_AA)
        if bool(pred_vis[qi, frame_idx]) and np.isfinite(pred_uv[qi, frame_idx]).all():
            p = tuple(np.rint(pred_uv[qi, frame_idx]).astype(np.int32))
            cv2.drawMarker(out, p, color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
    return out


def build_demo_overview(artifact_root: Path, demo_64_dir: Path, comparison: dict[str, Any]) -> None:
    data = _load_json(demo_64_dir / "assets" / "demo_data.json")
    manifest = _load_json(demo_64_dir / "manifest.json")
    meta = data.get("meta", {})
    num_frames = int(meta.get("numFrames", 64))
    frame_idx = min(num_frames - 1, max(0, num_frames // 2))
    input_frame = _extract_video_frame(demo_64_dir / manifest["video_copy"], frame_idx)
    try:
        overlay_frame = _extract_video_frame(demo_64_dir / manifest["tracks_2d_overlay"], frame_idx)
    except Exception:
        overlay_frame = _draw_2d_overlay_from_data(input_frame, data, frame_idx)

    gt = _to_array(data["tracksGt"]["xyzRef0"], np.float32)
    pred = _to_array(data["tracks"]["xyzRef0"], np.float32)
    gt_vis = _to_array(data["tracksGt"]["visibility"], np.int32).astype(bool)
    pred_vis = _to_array(data["tracks"]["visibility"], np.int32).astype(bool)
    motion_scores, _ = _compute_point_motion_scores(
        xyz_ref0=np.transpose(gt, (1, 0, 2)),
        visibility=np.transpose(gt_vis, (1, 0)),
        confidence=np.ones((gt.shape[1], gt.shape[0]), dtype=np.float32),
    )
    top = np.argsort(motion_scores)[::-1][: min(60, gt.shape[0])]
    colors = _track_colors(int(top.shape[0]))

    fig = plt.figure(figsize=(14, 8), dpi=180)
    ax_input = fig.add_subplot(2, 3, 1)
    ax_overlay = fig.add_subplot(2, 3, 2)
    ax_info = fig.add_subplot(2, 3, 3)
    ax_gt = fig.add_subplot(2, 3, 4, projection="3d")
    ax_pred = fig.add_subplot(2, 3, 5, projection="3d")
    ax_blank = fig.add_subplot(2, 3, 6)

    ax_input.imshow(input_frame)
    ax_input.set_title(f"Input frame {frame_idx}")
    ax_input.axis("off")
    ax_overlay.imshow(overlay_frame)
    ax_overlay.set_title("2D GT/pred overlay")
    ax_overlay.axis("off")
    ax_info.axis("off")
    case64 = comparison["cases"]["64_frames_anchor_clip"]
    info = [
        "OpenD4RT_32CLIP_9Dataset_NoAUG",
        _public_sequence_label(meta.get("worldtrack", {}).get("npz")) or "pstudio_mini/juggle_5.npz",
        "64 frames via anchor-clip inference",
        f"APD: {_fmt_optional(case64.get('apd_global'))}",
        f"EPE: {_fmt_optional(case64.get('epe_global'))}",
        f"Dynamic APD: {_fmt_optional(case64.get('apd_global_dynamic'))}",
        f"Dynamic EPE: {_fmt_optional(case64.get('epe_global_dynamic'))}",
        f"Runtime: {float(case64.get('runtime', {}).get('elapsed_seconds', 0.0)):.1f}s",
        f"Valid queries: {case64.get('valid_query_count')}",
    ]
    ax_info.text(0.02, 0.96, "\n".join(info), va="top", ha="left", fontsize=10)

    for local_i, qi in enumerate(top.tolist()):
        color = colors[local_i]
        valid_gt = gt_vis[qi] & np.isfinite(gt[qi]).all(axis=-1)
        valid_pred = pred_vis[qi] & np.isfinite(pred[qi]).all(axis=-1)
        if int(np.count_nonzero(valid_gt)) >= 2:
            pts = gt[qi, valid_gt]
            ax_gt.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=0.9, alpha=0.8)
        if int(np.count_nonzero(valid_pred)) >= 2:
            pts = pred[qi, valid_pred]
            ax_pred.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=0.9, alpha=0.8)

    ax_gt.set_title("3D GT tracks")
    ax_pred.set_title("3D predicted tracks")
    for ax in (ax_gt, ax_pred):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=24, azim=45)
        ax.grid(alpha=0.25)
    _set_equal_3d(ax_gt, [gt[top]])
    _set_equal_3d(ax_pred, [pred[top]])
    ax_blank.axis("off")
    ax_blank.text(
        0.02,
        0.95,
        "What this shows\n- The input video frame used for querying.\n- 2D GT/pred tracking overlay.\n- GT and predicted 3D track structure.\n\nAll numbers are computed on the selected public demo track set.",
        va="top",
        fontsize=10,
    )
    fig.suptitle("OpenD4RT WorldTrack demo overview", fontsize=13)
    fig.tight_layout()
    fig.savefig(artifact_root / "opend4rt_demo_overview.png")
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    demo_64_dir = Path(args.demo_64_dir)
    demo_32_dir = Path(args.demo_32_dir)

    refresh_manifest_posters(demo_64_dir)
    refresh_manifest_posters(demo_32_dir)
    analysis = analyze_summary(Path(args.summary_json), artifact_root, min_dynamic_queries=int(args.min_dynamic_queries))
    comparison = build_frame_count_comparison(artifact_root, demo_32_dir=demo_32_dir, demo_64_dir=demo_64_dir)
    build_demo_overview(artifact_root, demo_64_dir=demo_64_dir, comparison=comparison)
    t_cam = None
    if not bool(args.skip_t_cam):
        t_cam = build_t_cam_semantics_demo(args, artifact_root)
    print(
        json.dumps(
            _sanitize_json(
                {
                    "outputs": {
                        "demo_overview": str(artifact_root / "opend4rt_demo_overview.png"),
                        "frame_count_comparison": str(artifact_root / "frame_count_comparison.png"),
                        "t_cam_query_semantics": None if t_cam is None else str(artifact_root / "t_cam_query_semantics.png"),
                        "case_analysis": str(artifact_root / "worldtrack_case_analysis.json"),
                    },
                    "best_overall": analysis["overall"]["best_by_apd"],
                    "worst_dynamic": analysis["dynamic"]["worst_by_dynamic_epe"],
                }
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
