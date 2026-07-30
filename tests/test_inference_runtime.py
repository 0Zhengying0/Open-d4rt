"""Unit coverage for the resource-safe inference helpers."""

from __future__ import annotations

import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch

from src.core.inference_runtime import (
    configure_cuda_memory_budget,
    build_inference_model_from_checkpoint,
    inference_context,
    prepare_model_for_inference,
    resolve_precision,
    resource_snapshot,
)


class InferenceRuntimeTest(unittest.TestCase):
    def test_precision_resolution_on_cpu(self) -> None:
        cpu = torch.device("cpu")
        self.assertEqual(resolve_precision("auto", cpu), "fp32")
        self.assertEqual(resolve_precision("fp32", cpu), "fp32")
        with self.assertRaises(ValueError):
            resolve_precision("fp16", cpu)

    def test_cuda_budget_uses_requested_fraction(self) -> None:
        total = 24 * 1024**3
        with mock.patch(
            "src.core.inference_runtime.torch.cuda.get_device_properties",
            return_value=types.SimpleNamespace(total_memory=total),
        ), mock.patch("src.core.inference_runtime.torch.cuda.set_per_process_memory_fraction") as set_fraction:
            effective = configure_cuda_memory_budget(torch.device("cuda:0"), 20)
        self.assertEqual(effective, 20.0)
        set_fraction.assert_called_once_with(20 / 24, device=torch.device("cuda:0"))

    def test_cuda_budget_resolves_unspecified_cuda_device(self) -> None:
        total = 24 * 1024**3
        with mock.patch("src.core.inference_runtime.torch.cuda.current_device", return_value=0), mock.patch(
            "src.core.inference_runtime.torch.cuda.get_device_properties",
            return_value=types.SimpleNamespace(total_memory=total),
        ), mock.patch("src.core.inference_runtime.torch.cuda.set_per_process_memory_fraction") as set_fraction:
            configure_cuda_memory_budget(torch.device("cuda"), 20)
        set_fraction.assert_called_once_with(20 / 24, device=torch.device("cuda:0"))

    def test_cpu_context_and_snapshot(self) -> None:
        model = torch.nn.Linear(2, 2)
        runtime = prepare_model_for_inference(model, device=torch.device("cpu"), precision="auto")
        with inference_context(model):
            output = model(torch.ones((1, 2)))
        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertEqual(runtime.precision, "fp32")
        self.assertIsNone(resource_snapshot(torch.device("cpu"))["cuda_peak_reserved_gib"])

    def test_mmap_meta_checkpoint_load_avoids_eager_cpu_model(self) -> None:
        source = torch.nn.Linear(3, 2)
        expected = {key: value.detach().clone() for key, value in source.state_dict().items()}
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny.ckpt"
            torch.save({"model": expected, "optimizer": {"unused": torch.ones(64)}}, checkpoint)
            model, runtime, result = build_inference_model_from_checkpoint(
                lambda: torch.nn.Linear(3, 2),
                checkpoint_path=checkpoint,
                device=torch.device("cpu"),
                precision="fp32",
            )
        self.assertFalse(result.missing_keys)
        self.assertFalse(result.unexpected_keys)
        self.assertEqual(runtime.precision, "fp32")
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]))


if __name__ == "__main__":
    unittest.main()
