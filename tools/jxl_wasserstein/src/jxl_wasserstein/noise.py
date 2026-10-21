"""Deterministic JPEG XL Xorshift128Plus PRNG and differentiable NoiseParams synthesis in JAX."""

from __future__ import annotations

from collections.abc import Sequence
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

NOISE_PRECISION: float = 1024.0
NOISE_LUT_MAX: float = 1023.0 / 1024.0
NUM_NOISE_POINTS: int = 8
NOISE_NORM_CONST: float = 0.22
RG_CORR: float = 0.9921875  # 127.0 / 128.0
RGN_CORR: float = 0.0078125  # 1.0 / 128.0
GROUP_DIM: int = 256


def _splitmix64_u64(z: np.ndarray) -> np.ndarray:
  """Exact uint64 SplitMix64 step from lib/jxl/xorshift128plus-inl.h:86-90."""
  with np.errstate(over="ignore"):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


class Xorshift128Plus:
  """Exact port of jxl::Xorshift128Plus (lib/jxl/xorshift128plus-inl.h)."""

  N: int = 8

  def __init__(
      self,
      seed1: int,
      seed2: int | None = None,
      seed3: int | None = None,
      seed4: int | None = None,
  ) -> None:
    self.s0 = np.zeros(self.N, dtype=np.uint64)
    self.s1 = np.zeros(self.N, dtype=np.uint64)
    golden = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
      if seed2 is None and seed3 is None and seed4 is None:
        s = np.uint64(int(seed1) & 0xFFFFFFFFFFFFFFFF)
        self.s0[0] = _splitmix64_u64(s + golden)
        self.s1[0] = _splitmix64_u64(self.s0[0])
        for i in range(1, self.N):
          self.s0[i] = _splitmix64_u64(self.s1[i - 1])
          self.s1[i] = _splitmix64_u64(self.s0[i])
      else:
        u1 = np.uint64(int(seed1) & 0xFFFFFFFF)
        u2 = np.uint64(int(0 if seed2 is None else seed2) & 0xFFFFFFFF)
        u3 = np.uint64(int(0 if seed3 is None else seed3) & 0xFFFFFFFF)
        u4 = np.uint64(int(0 if seed4 is None else seed4) & 0xFFFFFFFF)
        self.s0[0] = _splitmix64_u64(((u1 << np.uint64(32)) + u2) + golden)
        self.s1[0] = _splitmix64_u64(((u3 << np.uint64(32)) + u4) + golden)
        for i in range(1, self.N):
          self.s0[i] = _splitmix64_u64(self.s0[i - 1])
          self.s1[i] = _splitmix64_u64(self.s1[i - 1])

  def fill_u64(self) -> np.ndarray:
    """Generates N=8 random uint64 values and advances state."""
    with np.errstate(over="ignore"):
      s1 = self.s0.copy()
      s0 = self.s1.copy()
      bits = s1 + s0
      self.s0 = s0
      s1 = s1 ^ (s1 << np.uint64(23))
      s1 = s1 ^ s0 ^ (s1 >> np.uint64(18)) ^ (s0 >> np.uint64(5))
      self.s1 = s1
      return bits

  Fill = fill_u64

  def fill_f32_batch(self, subtract_one: bool = False) -> np.ndarray:
    """Generates 16 float32 values via BitsToFloat ((u32 >> 9) | 0x3F800000)."""
    batch64 = self.fill_u64()
    batch32 = batch64.view(np.uint32)
    u32_bits = (batch32 >> np.uint32(9)) | np.uint32(0x3F800000)
    floats = u32_bits.view(np.float32).copy()
    if subtract_one:
      return floats - np.float32(1.0)
    return floats


@functools.lru_cache(maxsize=32)
def generate_raw_noise_planes_np(
    height: int,
    width: int,
    visible_frame_index: int = 1,
    nonvisible_frame_index: int = 0,
    group_dim: int = GROUP_DIM,
) -> np.ndarray:
  """Generates the 3 raw random float32 noise planes (3, H, W) matching PrepareNoiseInput."""
  h = int(height)
  w = int(width)
  gdim = int(group_dim)
  if h <= 0 or w <= 0:
    raise ValueError(f"Invalid noise plane dimensions ({h}, {w})")
  if gdim <= 0:
    raise ValueError(f"Invalid group_dim {gdim}")

  planes = np.zeros((3, h, w), dtype=np.float32)
  num_gx = math.ceil(w / gdim)
  num_gy = math.ceil(h / gdim)

  for gy in range(num_gy):
    y0 = gy * gdim
    ysize = min(gdim, h - y0)
    for gx in range(num_gx):
      x0 = gx * gdim
      xsize = min(gdim, w - x0)
      rng = Xorshift128Plus(
          int(visible_frame_index),
          int(nonvisible_frame_index),
          x0,
          y0,
      )
      batches_per_row = math.ceil(xsize / 16)
      for c in range(3):
        for y in range(ysize):
          if batches_per_row == 1:
            row_vals = rng.fill_f32_batch()[:xsize]
          else:
            row_chunks = [rng.fill_f32_batch() for _ in range(batches_per_row)]
            row_vals = np.concatenate(row_chunks, axis=0)[:xsize]
          planes[c, y0 + y, x0 : x0 + xsize] = row_vals

  planes.setflags(write=False)
  return planes


def convolve_noise_planes(raw_noise: jax.Array | np.ndarray) -> jax.Array:
  """Applies ConvolveNoiseStage 5x5 Laplacian high-pass filter with symmetric border=2.

  Matches lib/jxl/render_pipeline/stage_noise.cc:258-298:
    pixels = others * 0.16f + p00 * (-3.84f)
  where `others` is the sum of the 24 non-center neighbors in the 5x5 window
  under Mirror (symmetric) boundary padding of radius 2.
  """
  arr = jnp.asarray(raw_noise, dtype=jnp.float32)
  if arr.ndim != 3 or arr.shape[0] != 3:
    raise ValueError(f"Expected raw_noise of shape (3, H, W), got {arr.shape}")
  h, w = arr.shape[1], arr.shape[2]
  padded = jnp.pad(arr, ((0, 0), (2, 2), (2, 2)), mode="symmetric")
  p00 = padded[:, 2 : 2 + h, 2 : 2 + w]
  others = jnp.zeros_like(p00)
  for dy in range(-2, 3):
    for dx in range(-2, 3):
      if dy == 0 and dx == 0:
        continue
      others = others + padded[:, 2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w]
  return others * jnp.float32(0.16) + p00 * jnp.float32(-3.84)


@functools.lru_cache(maxsize=32)
def _cached_convolved_noise_np(
    height: int,
    width: int,
    visible_frame_index: int = 1,
    nonvisible_frame_index: int = 0,
    group_dim: int = GROUP_DIM,
) -> np.ndarray:
  raw = generate_raw_noise_planes_np(
      height=int(height),
      width=int(width),
      visible_frame_index=int(visible_frame_index),
      nonvisible_frame_index=int(nonvisible_frame_index),
      group_dim=int(group_dim),
  )
  h, w = raw.shape[1], raw.shape[2]
  padded = np.pad(raw, ((0, 0), (2, 2), (2, 2)), mode="symmetric")
  p00 = padded[:, 2 : 2 + h, 2 : 2 + w]
  others = np.zeros_like(p00, dtype=np.float32)
  for dy in range(-2, 3):
    for dx in range(-2, 3):
      if dy == 0 and dx == 0:
        continue
      others = others + padded[:, 2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w]
  conv = (others * np.float32(0.16) + p00 * np.float32(-3.84)).astype(np.float32)
  conv.setflags(write=False)
  return conv


def get_convolved_noise_planes(
    height: int,
    width: int,
    visible_frame_index: int = 1,
    nonvisible_frame_index: int = 0,
    group_dim: int = GROUP_DIM,
) -> jax.Array:
  """Returns the deterministic pre-convolved (3, H, W) noise planes as a JAX array."""
  conv_np = _cached_convolved_noise_np(
      int(height),
      int(width),
      int(visible_frame_index),
      int(nonvisible_frame_index),
      int(group_dim),
  )
  return jnp.asarray(conv_np, dtype=jnp.float32)


def quantize_noise_lut_ste(noise_lut: jax.Array | Sequence[float]) -> jax.Array:
  """Applies 10-bit (1/1024) JXL NoiseParams LUT quantization with Straight-Through Estimator."""
  lut = jnp.asarray(noise_lut, dtype=jnp.float32)
  clamped = jnp.clip(lut, jnp.float32(0.0), jnp.float32(NOISE_LUT_MAX))
  quantized = (
      jnp.round(clamped * jnp.float32(NOISE_PRECISION))
      / jnp.float32(NOISE_PRECISION)
  )
  return clamped + jax.lax.stop_gradient(quantized - clamped)


def eval_noise_strength_lut(
    vx: jax.Array, noise_lut: jax.Array | Sequence[float]
) -> jax.Array:
  """Piecewise-linear 8-float LUT evaluation matching StrengthEvalLut + NoiseStrength.

  Args:
    vx: Intensity input `0.5 * (Y - X)` or `0.5 * (Y + X)`.
    noise_lut: 1D array of 8 float32 LUT values.

  Returns:
    Clamped noise strength in `[0.0, 1.0]`, differentiable wrt both `vx` and `noise_lut`.
  """
  lut = jnp.asarray(noise_lut, dtype=jnp.float32)
  if lut.shape != (NUM_NOISE_POINTS,):
    raise ValueError(f"Expected noise_lut of shape (8,), got {lut.shape}")
  scaled_vx = jnp.clip(
      vx * jnp.float32(NUM_NOISE_POINTS - 2),
      jnp.float32(0.0),
      jnp.float32(NUM_NOISE_POINTS - 1),
  )
  floor_int = jnp.clip(
      jnp.floor(jax.lax.stop_gradient(scaled_vx)).astype(jnp.int32),
      0,
      NUM_NOISE_POINTS - 2,
  )
  frac_x = scaled_vx - floor_int.astype(jnp.float32)
  low = lut[floor_int]
  hi = lut[floor_int + 1]
  interpolated = low + frac_x * (hi - low)
  return jnp.clip(interpolated, jnp.float32(0.0), jnp.float32(1.0))


def add_jxl_noise_xyb(
    xyb: jax.Array,
    convolved_noise: jax.Array,
    noise_lut: jax.Array | Sequence[float],
    ytox: float | jax.Array = 0.0,
    ytob: float | jax.Array = 1.0,
    ste_quantize_lut: bool = True,
    quantize_lut: bool | None = None,
) -> jax.Array:
  """Adds correlated intensity-dependent JXL noise to an XYB image (`AddNoiseStage`).

  Supports both `(3, H, W)` (CHW) and `(H, W, 3)` (HWC) XYB layouts and returns
  the same layout.
  """
  arr = jnp.asarray(xyb, dtype=jnp.float32)
  if arr.ndim != 3:
    raise ValueError(f"Expected 3D XYB tensor, got shape {arr.shape}")

  is_chw = arr.shape[0] == 3 and arr.shape[2] != 3
  if not is_chw and arr.shape[2] == 3:
    arr_chw = jnp.moveaxis(arr, -1, 0)
  elif arr.shape[0] == 3:
    arr_chw = arr
    is_chw = True
  else:
    raise ValueError(f"Expected XYB shape (3, H, W) or (H, W, 3), got {arr.shape}")

  lut = jnp.asarray(noise_lut, dtype=jnp.float32)
  if lut.shape != (NUM_NOISE_POINTS,):
    raise ValueError(f"Expected noise_lut of shape (8,), got {lut.shape}")
  do_quantize = bool(quantize_lut) if quantize_lut is not None else bool(ste_quantize_lut)
  if do_quantize:
    lut = quantize_noise_lut_ste(lut)

  conv = jnp.asarray(convolved_noise, dtype=jnp.float32)
  if conv.shape != arr_chw.shape:
    raise ValueError(
        f"convolved_noise shape {conv.shape} does not match XYB (3, H, W) shape {arr_chw.shape}"
    )

  vx = arr_chw[0]
  vy = arr_chw[1]
  vb = arr_chw[2]

  in_g = (vy - vx) * jnp.float32(0.5)
  in_r = (vy + vx) * jnp.float32(0.5)

  strength_g = eval_noise_strength_lut(in_g, lut)
  strength_r = eval_noise_strength_lut(in_r, lut)

  rnd_r = conv[0] * jnp.float32(NOISE_NORM_CONST)
  rnd_g = conv[1] * jnp.float32(NOISE_NORM_CONST)
  rnd_c = conv[2] * jnp.float32(NOISE_NORM_CONST)

  red_noise = strength_r * (
      jnp.float32(RGN_CORR) * rnd_r + jnp.float32(RG_CORR) * rnd_c
  )
  green_noise = strength_g * (
      jnp.float32(RGN_CORR) * rnd_g + jnp.float32(RG_CORR) * rnd_c
  )

  rg_noise = red_noise + green_noise
  ytox_f = jnp.asarray(ytox, dtype=jnp.float32)
  ytob_f = jnp.asarray(ytob, dtype=jnp.float32)

  out_x = vx + ytox_f * rg_noise + (red_noise - green_noise)
  out_y = vy + rg_noise
  out_b = vb + ytob_f * rg_noise

  out_chw = jnp.stack([out_x, out_y, out_b], axis=0)
  return out_chw if is_chw else jnp.moveaxis(out_chw, 0, -1)


def synthesize_jxl_noise(
    xyb: jax.Array,
    noise_lut: jax.Array | Sequence[float],
    ytox: float | jax.Array = 0.0,
    ytob: float | jax.Array = 1.0,
    visible_frame_index: int = 1,
    nonvisible_frame_index: int = 0,
    group_dim: int = GROUP_DIM,
    ste_quantize_lut: bool = True,
    quantize_lut: bool | None = None,
) -> jax.Array:
  """End-to-end deterministic JXL noise synthesis in XYB space.

  Note on frame indices (`lib/jxl/dec_frame.cc:160-168`):
    `FrameDecoder::InitFrame` increments `visible_frame_index` to `1` and resets
    `nonvisible_frame_index` to `0` on the first visible `kRegularFrame` (both
    in single-frame mode and on Frame 1 of 2-frame `kReferenceOnly` + `kRegularFrame`
    patch/alpha modes). Thus `visible_frame_index=1, nonvisible_frame_index=0`
    is the exact default for all visible output frames.
  """
  arr = jnp.asarray(xyb, dtype=jnp.float32)
  if arr.ndim != 3:
    raise ValueError(f"Expected 3D XYB tensor, got shape {arr.shape}")
  if arr.shape[0] == 3 and arr.shape[2] != 3:
    h, w = int(arr.shape[1]), int(arr.shape[2])
  elif arr.shape[2] == 3:
    h, w = int(arr.shape[0]), int(arr.shape[1])
  else:
    raise ValueError(f"Expected XYB shape (3, H, W) or (H, W, 3), got {arr.shape}")

  conv = get_convolved_noise_planes(
      height=h,
      width=w,
      visible_frame_index=int(visible_frame_index),
      nonvisible_frame_index=int(nonvisible_frame_index),
      group_dim=int(group_dim),
  )
  return add_jxl_noise_xyb(
      xyb=arr,
      convolved_noise=conv,
      noise_lut=noise_lut,
      ytox=ytox,
      ytob=ytob,
      ste_quantize_lut=ste_quantize_lut,
      quantize_lut=quantize_lut,
  )
