"""Resource-safe helpers shared by the inference entrypoints."""

from __future__ import annotations

import contextlib
import gc
import os
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import torch


Precision = Literal["fp32", "fp16"]


@dataclass(frozen=True)
class InferenceRuntime:
    """Resolved inference settings recorded with evaluation artifacts."""

    precision: Precision
    device: torch.device
    max_gpu_memory_gib: float | None


def _unwrap_checkpoint_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Extract common model-state containers without touching tensor contents."""
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module", "network", "net"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload
    return {}


def _restore_nonpersistent_meta_buffers(model: torch.nn.Module) -> None:
    """Restore small buffers intentionally omitted from a state dict.

    A meta-constructed module has no storage for non-persistent buffers, which
    by definition are absent from checkpoints.  Modules that own such buffers
    provide a deterministic reset hook.  Failing closed here is preferable to
    silently running inference with an uninitialized buffer.
    """
    for module in model.modules():
        reset = getattr(module, "reset_nonpersistent_buffers", None)
        if callable(reset):
            reset(device=torch.device("cpu"))
    remaining = [name for name, buffer in model.named_buffers() if buffer is not None and buffer.is_meta]
    if remaining:
        raise RuntimeError(
            "Checkpoint loading left meta buffers without deterministic initialization: "
            + ", ".join(remaining)
        )


def build_inference_model_from_checkpoint(
    model_factory: Callable[[], torch.nn.Module],
    *,
    checkpoint_path: str | Path,
    device: torch.device,
    precision: str,
    max_gpu_memory_gib: float | int | None = None,
) -> tuple[torch.nn.Module, InferenceRuntime, torch.nn.modules.module._IncompatibleKeys]:
    """Materialize an inference model without a CPU model/checkpoint peak.

    The training checkpoint includes substantially more than the model weights.
    ``mmap=True`` leaves those tensor storages file-backed until referenced.
    Constructing the module on ``meta`` and using ``assign=True`` then attaches
    only model tensors without allocating a second CPU copy.  ``model.to``
    subsequently converts and moves one parameter at a time to the constrained
    inference device.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with torch.device("meta"):
        model = model_factory()
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError(
            "This guarded inference path requires a mmap-compatible PyTorch checkpoint; "
            "an eager CPU checkpoint load is intentionally disabled."
        ) from exc
    state_dict = _unwrap_checkpoint_state_dict(payload)
    if not state_dict:
        del payload
        raise RuntimeError(f"No model weights found in checkpoint: {path}")
    try:
        load_result = model.load_state_dict(state_dict, strict=False, assign=True)
    finally:
        # With assign=True parameters retain their own references to only the
        # tensors they need; optimizer/trainer tensors can be unmapped now.
        del state_dict
        del payload
        gc.collect()
    _restore_nonpersistent_meta_buffers(model)
    runtime = prepare_model_for_inference(
        model,
        device=device,
        precision=precision,
        max_gpu_memory_gib=max_gpu_memory_gib,
    )
    return model, runtime, load_result


def resolve_precision(raw: str, device: torch.device) -> Precision:
    """Resolve a user-facing precision choice without silently using BF16."""
    value = str(raw).strip().lower()
    if value not in {"auto", "fp32", "fp16"}:
        raise ValueError(f"Unsupported precision {raw!r}; expected auto, fp32, or fp16.")
    if value == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    if value == "fp16" and device.type != "cuda":
        raise ValueError("--precision fp16 requires --device cuda.")
    return value  # type: ignore[return-value]


def configure_cpu_thread_limits(num_threads: int | None = None) -> int:
    """Cap thread pools used by Torch/OpenCV without overriding explicit user input."""
    value = int(num_threads if num_threads is not None else os.environ.get("D4RT_CPU_THREADS", "2"))
    value = max(1, value)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(value))
    torch.set_num_threads(value)
    try:
        torch.set_num_interop_threads(value)
    except RuntimeError:
        # PyTorch only permits this before its inter-op pool has been used.
        pass
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        pass
    return value


def configure_cuda_memory_budget(device: torch.device, max_gpu_memory_gib: float | int | None) -> float | None:
    """Apply a per-process PyTorch allocator cap and return the effective GiB limit."""
    if device.type != "cuda" or max_gpu_memory_gib is None or float(max_gpu_memory_gib) <= 0:
        return None
    # torch.cuda.set_per_process_memory_fraction requires an explicit index,
    # whereas CLI users naturally pass the valid shorthand ``cuda``.
    cuda_device = device if device.index is not None else torch.device("cuda", torch.cuda.current_device())
    total_bytes = int(torch.cuda.get_device_properties(cuda_device).total_memory)
    requested_bytes = int(float(max_gpu_memory_gib) * (1024**3))
    if requested_bytes <= 0:
        return None
    fraction = min(1.0, requested_bytes / float(total_bytes))
    torch.cuda.set_per_process_memory_fraction(fraction, device=cuda_device)
    return min(float(max_gpu_memory_gib), total_bytes / float(1024**3))


def prepare_model_for_inference(
    model: torch.nn.Module,
    *,
    device: torch.device,
    precision: str,
    max_gpu_memory_gib: float | int | None = None,
) -> InferenceRuntime:
    """Move a loaded model into its constrained, inference-only runtime state."""
    resolved = resolve_precision(precision, device)
    budget = configure_cuda_memory_budget(device, max_gpu_memory_gib)
    dtype = torch.float16 if resolved == "fp16" else torch.float32
    model.to(device=device, dtype=dtype)
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    return InferenceRuntime(precision=resolved, device=device, max_gpu_memory_gib=budget)


@contextlib.contextmanager
def inference_context(model: torch.nn.Module) -> Iterator[None]:
    """Run model calls without autograd and with autocast matching model weights."""
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32
    amp_enabled = device.type == "cuda" and dtype == torch.float16
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if amp_enabled
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        yield


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def resource_snapshot(device: torch.device) -> dict[str, float | int | None]:
    """Return portable RSS and CUDA allocator metrics for artifact metadata."""
    rss_bytes: int | None = None
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    rss_bytes = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if rss_bytes is None:
        # Linux reports KiB; macOS reports bytes.
        rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rss_bytes = rss_raw * 1024 if os.name != "darwin" else rss_raw
    result: dict[str, float | int | None] = {"cpu_rss_gib": rss_bytes / float(1024**3)}
    if device.type != "cuda":
        result.update(
            {
                "cuda_allocated_gib": None,
                "cuda_reserved_gib": None,
                "cuda_peak_allocated_gib": None,
                "cuda_peak_reserved_gib": None,
            }
        )
        return result
    scale = float(1024**3)
    result.update(
        {
            "cuda_allocated_gib": torch.cuda.memory_allocated(device) / scale,
            "cuda_reserved_gib": torch.cuda.memory_reserved(device) / scale,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / scale,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / scale,
        }
    )
    return result
