"""VGG-16 AvgPool feature extractor for Wasserstein Distortion."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any
import urllib.request

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jxl_wasserstein.wasserstein import lowpass

jax.config.update("jax_default_matmul_precision", "float32")

Array = jax.Array

VGG16_SAFETENSORS_URL = (
    "https://huggingface.co/timm/vgg16.tv_in1k/resolve/main/model.safetensors"
)
_CACHE_FILENAME = "vgg16_tv_in1k_conv_v1.npz"

# (pytorch_features_idx, flax_block_idx, flax_conv_idx)
_PT_TO_FLAX_MAP: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (2, 0, 1),
    (5, 1, 0),
    (7, 1, 1),
    (10, 2, 0),
    (12, 2, 1),
    (14, 2, 2),
    (17, 3, 0),
    (19, 3, 1),
    (21, 3, 2),
    (24, 4, 0),
    (26, 4, 1),
    (28, 4, 2),
)

_VGG_BLOCKS: tuple[tuple[int, int], ...] = (
    (64, 2),
    (128, 2),
    (256, 3),
    (512, 3),
    (512, 3),
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_PARAMS_MEM_CACHE: dict[Path, dict[str, Any]] = {}


class VggBlock(nn.Module):
  """One convolutional block within VGG-16."""

  num_features: int
  num_layers: int

  @nn.compact
  def __call__(self, x: Array) -> Array:
    for _ in range(self.num_layers):
      x = nn.Conv(
          features=self.num_features,
          kernel_size=(3, 3),
          padding="same",
      )(x)
      x = nn.relu(x)
    return x


class VggNet(nn.Module):
  """5-block VGG-16 with 2x2 VALID AvgPool returning post-ReLU activations."""

  @nn.compact
  def __call__(self, x: Array) -> list[Array]:
    if x.shape[-2] < 16 or x.shape[-3] < 16:
      raise ValueError(
          f"VggNet requires spatial dims >= 16, got shape {x.shape}."
      )
    outputs: list[Array] = []
    for num_features, num_layers in _VGG_BLOCKS:
      x = VggBlock(num_features=num_features, num_layers=num_layers)(x)
      outputs.append(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
    return outputs


def _read_exact_prefix(url: str, num_bytes: int, timeout: int) -> bytes:
  """Reads the first `num_bytes` from `url` via Range request with stream fallback."""
  req = urllib.request.Request(url, headers={"Range": f"bytes=0-{num_bytes - 1}"})
  chunks: list[bytes] = []
  remaining = num_bytes
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    while remaining > 0:
      chunk = resp.read(min(remaining, 1024 * 1024))
      if not chunk:
        break
      chunks.append(chunk)
      remaining -= len(chunk)
  blob = b"".join(chunks)
  if len(blob) < num_bytes:
    raise IOError(
        f"Incomplete VGG-16 weight read from {url}: got {len(blob)} bytes, "
        f"expected {num_bytes}."
    )
  return blob


def _download_vgg16_conv_npz(
    target_path: Path,
    url: str = VGG16_SAFETENSORS_URL,
) -> None:
  """Downloads only the conv1-conv5 prefix from HF safetensors and saves atomically."""
  target_path.parent.mkdir(parents=True, exist_ok=True)

  # Step 1: Read first 16 KB to parse 8-byte header size and JSON header.
  head_bytes = _read_exact_prefix(url, 16384, timeout=30)
  header_len = struct.unpack("<Q", head_bytes[:8])[0]
  if 8 + header_len > len(head_bytes):
    head_bytes = _read_exact_prefix(url, 8 + header_len, timeout=30)
  header = json.loads(head_bytes[8 : 8 + header_len])

  # Step 2: Compute exact byte range covering all 26 `features.*` tensors.
  max_data_end = max(
      info["data_offsets"][1]
      for name, info in header.items()
      if name != "__metadata__" and name.startswith("features.")
  )
  total_needed = 8 + header_len + max_data_end

  # Step 3: Fetch only `total_needed` bytes (~58.86 MB instead of 553 MB).
  blob = _read_exact_prefix(url, total_needed, timeout=120)

  # Step 4: Extract and transpose kernels (O, I, H, W) -> (H, W, I, O).
  arrays: dict[str, np.ndarray] = {}
  data_start = 8 + header_len
  for pt_idx, blk_idx, conv_idx in _PT_TO_FLAX_MAP:
    w_info = header[f"features.{pt_idx}.weight"]
    b_info = header[f"features.{pt_idx}.bias"]
    ws, we = w_info["data_offsets"]
    bs, be = b_info["data_offsets"]
    w_pt = np.frombuffer(
        blob[data_start + ws : data_start + we], dtype="<f4"
    ).reshape(w_info["shape"])
    b_pt = np.frombuffer(
        blob[data_start + bs : data_start + be], dtype="<f4"
    ).reshape(b_info["shape"])
    arrays[f"VggBlock_{blk_idx}.Conv_{conv_idx}.kernel"] = np.ascontiguousarray(
        w_pt.transpose(2, 3, 1, 0)
    )
    arrays[f"VggBlock_{blk_idx}.Conv_{conv_idx}.bias"] = np.ascontiguousarray(b_pt)

  # Step 5: Write atomically via temp file in same directory + os.replace.
  fd, tmp_name = tempfile.mkstemp(
      dir=str(target_path.parent), prefix=".tmp_vgg16_", suffix=".npz"
  )
  try:
    with os.fdopen(fd, "wb") as f:
      np.savez(f, **arrays)
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp_name, target_path)
  except BaseException:
    if os.path.exists(tmp_name):
      os.unlink(tmp_name)
    raise


def load_vgg16_params(cache_dir: str | Path | None = None) -> dict[str, Any]:
  """Loads pretrained ImageNet VGG-16 conv1-conv5 parameters as a Flax PyTree."""
  resolved_dir = (
      Path(cache_dir).expanduser().resolve()
      if cache_dir is not None
      else (Path.home() / ".cache" / "jxl_wasserstein").resolve()
  )
  cache_path = resolved_dir / _CACHE_FILENAME

  if cache_path in _PARAMS_MEM_CACHE:
    return _PARAMS_MEM_CACHE[cache_path]

  if not cache_path.exists():
    _download_vgg16_conv_npz(cache_path)

  params_blocks: dict[str, dict[str, dict[str, Array]]] = {}
  with np.load(cache_path, allow_pickle=False) as data:
    for _, blk_idx, conv_idx in _PT_TO_FLAX_MAP:
      blk_key = f"VggBlock_{blk_idx}"
      conv_key = f"Conv_{conv_idx}"
      params_blocks.setdefault(blk_key, {})[conv_key] = {
          "kernel": jnp.asarray(
              data[f"{blk_key}.{conv_key}.kernel"], dtype=jnp.float32
          ),
          "bias": jnp.asarray(
              data[f"{blk_key}.{conv_key}.bias"], dtype=jnp.float32
          ),
      }

  params = {"params": params_blocks}
  _PARAMS_MEM_CACHE[cache_path] = params
  return params


def ensure_chw_rgb(image_rgb_01: Array) -> Array:
  """Converts (H, W, 3) to (3, H, W) if needed, preserving (3, H, W)."""
  img = jnp.asarray(image_rgb_01)
  if img.ndim != 3:
    raise ValueError(f"Expected 3D image array, got shape {img.shape}.")
  if img.shape[0] == 3:
    return img
  if img.shape[-1] == 3:
    return jnp.transpose(img, (2, 0, 1))
  raise ValueError(
      f"Expected channel dim of size 3 at axis 0 or 2, got shape {img.shape}."
  )


def select_num_scales(h: int, w: int, max_scales: int = 3) -> int:
  """Selects the largest VGG pyramid scale count keeping spatial dimensions >= 16."""
  if h < 16 or w < 16:
    raise ValueError(f"Image dimensions must be >= 16x16, got {(h, w)}.")
  sh, sw = h, w
  scales = 1
  for _ in range(1, max_scales):
    sh = (sh + 1) // 2
    sw = (sw + 1) // 2
    if sh < 16 or sw < 16:
      break
    scales += 1
  return scales


def multiscale_vgg_features(
    params: Any,
    image_rgb_01: Array,
    num_scales: int = 3,
) -> Sequence[Array]:
  """Extracts 1 + 5 * num_scales multi-scale VGG-16 feature maps in (C, H, W) layout."""
  if num_scales < 1:
    raise ValueError(f"`num_scales` must be at least 1, got {num_scales}.")
  image = ensure_chw_rgb(jnp.asarray(image_rgb_01, dtype=jnp.float32))
  mean = jnp.array(IMAGENET_MEAN, dtype=image.dtype)[:, None, None]
  std = jnp.array(IMAGENET_STD, dtype=image.dtype)[:, None, None]

  # Normalize once at full scale (matches public CoDeX and balle-lab/wasserstein-distortion).
  image = (image - mean) / std
  features: list[Array] = [image]

  vgg_net = VggNet()
  for scale_idx in range(num_scales):
    if scale_idx > 0:
      image = lowpass(image, stride=2)
    nhwc_in = jnp.transpose(image, (1, 2, 0))[None, ...]
    block_outs = vgg_net.apply(params, nhwc_in)
    features.extend(
        jnp.transpose(jnp.squeeze(out, axis=0), (2, 0, 1))
        for out in block_outs
    )
  return features
