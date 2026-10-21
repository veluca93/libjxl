"""Pytest configuration for jxl_wasserstein tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = (
      f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_deterministic_ops=true".strip()
  )

import numpy as np
from PIL import Image
import pytest


@pytest.fixture(scope="session")
def kodim01_path() -> Path:
  path = Path(
      "/usr/local/google/home/veluca/libjxl/image-corpus/qoi_benchmark_suite/photo_kodak/kodim01.png"
  )
  assert path.exists(), f"Missing Kodak test image: {path}"
  return path


@pytest.fixture(scope="session")
def kodim01_rgb_u8(kodim01_path: Path) -> np.ndarray:
  with Image.open(kodim01_path) as im:
    arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
  return np.ascontiguousarray(arr)
