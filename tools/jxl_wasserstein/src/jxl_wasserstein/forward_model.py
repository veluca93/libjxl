"""Differentiable JPEG XL VarDCT, multi-resolution upsampling, XYB patch, and alpha compositing forward model in JAX."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jxl_wasserstein.cpp_bridge import (
    ACS_COVERED_BLOCKS_X,
    ACS_COVERED_BLOCKS_Y,
    PATCH_BLEND_ADD,
    PATCH_BLEND_REPLACE,
    JxlVarDCTState,
    PatchEntry,
)
from jxl_wasserstein.noise import synthesize_jxl_noise

synthesize_jxl_noise_xyb = synthesize_jxl_noise

PATCH_BLEND_NONE: int = 0
PATCH_BLEND_MUL: int = 3
PATCH_BLEND_ALPHA_ABOVE: int = 4
PATCH_BLEND_ALPHA_BELOW: int = 5
PATCH_BLEND_ALPHA_WEIGHTED_ADD_ABOVE: int = 6
PATCH_BLEND_ALPHA_WEIGHTED_ADD_BELOW: int = 7

ACS_NAMES: tuple[str, ...] = (
    "DCT",
    "IDENTITY",
    "DCT2X2",
    "DCT4X4",
    "DCT16X16",
    "DCT32X32",
    "DCT16X8",
    "DCT8X16",
    "DCT32X8",
    "DCT8X32",
    "DCT32X16",
    "DCT16X32",
    "DCT4X8",
    "DCT8X4",
    "AFV0",
    "AFV1",
    "AFV2",
    "AFV3",
    "DCT64X64",
    "DCT64X32",
    "DCT32X64",
    "DCT128X128",
    "DCT128X64",
    "DCT64X128",
    "DCT256X256",
    "DCT256X128",
    "DCT128X256",
)

K_GLOBAL_SCALE_DENOM: float = 65536.0
K_QUANT_FIELD_TARGET: float = 0.7850993928208519
K_COLOR_TILE_DIM_IN_BLOCKS: int = 8

K_DC_QUANT: tuple[float, float, float] = (1.0 / 4096.0, 1.0 / 512.0, 1.0 / 256.0)

K_DEFAULT_QUANT_BIAS: tuple[float, float, float, float] = (
    1.0 - 0.05465007330715401,
    1.0 - 0.07005449891748593,
    1.0 - 0.049935103337343655,
    0.145,
)

K_DEFAULT_COLOR_FACTOR: int = 84
K_BASE_CORRELATION_X: float = 0.0
K_BASE_CORRELATION_B: float = 1.0

K_DEFAULT_GAB_WEIGHTS: tuple[float, ...] = (
    0.115169525,
    0.061248592,
    0.115169525,
    0.061248592,
    0.115169525,
    0.061248592,
)
K_DEFAULT_EPF_QUANT_MUL: float = 0.46
K_DEFAULT_EPF_PASS0_SIGMA_SCALE: float = 0.9
K_DEFAULT_EPF_PASS2_SIGMA_SCALE: float = 6.5
K_DEFAULT_EPF_BORDER_SAD_MUL: float = 2.0 / 3.0
K_DEFAULT_EPF_CHANNEL_SCALE: tuple[float, float, float] = (40.0, 5.0, 3.5)
K_DEFAULT_EPF_SHARP_LUT: tuple[float, ...] = tuple(i / 7.0 for i in range(8))

K_INV_SIGMA_NUM: float = -1.1715728752538099024
K_MIN_SIGMA: float = -3.90524291751269967465540850526868

K_OPSIN_ABSORBANCE_MATRIX: tuple[tuple[float, float, float], ...] = (
    (0.30, 1.0 - 0.078 - 0.30, 0.078),
    (0.23, 1.0 - 0.078 - 0.23, 0.078),
    (
        0.24342268924547819,
        0.20476744424496821,
        1.0 - 0.24342268924547819 - 0.20476744424496821,
    ),
)
K_OPSIN_ABSORBANCE_BIAS: tuple[float, float, float] = (
    0.0037930732552754493,
    0.0037930732552754493,
    0.0037930732552754493,
)
K_DEFAULT_INVERSE_OPSIN_MATRIX: tuple[tuple[float, float, float], ...] = (
    (11.031566901960783, -9.866943921568629, -0.16462299647058826),
    (-3.254147380392157, 4.418770392156863, -0.16462299647058826),
    (-3.6588512862745097, 2.7129230470588235, 1.9459282392156863),
)
K_NEG_OPSIN_ABSORBANCE_BIAS_RGB: tuple[float, float, float, float] = (
    -0.0037930732552754493,
    -0.0037930732552754493,
    -0.0037930732552754493,
    1.0,
)


# -----------------------------------------------------------------------------
# 1. Color Space Conversions (sRGB <-> Linear RGB <-> XYB)
# -----------------------------------------------------------------------------


def srgb_to_linear(srgb: jax.Array) -> jax.Array:
  """Converts nonlinear sRGB in [0, 1] to linear RGB in [0, 1] (IEC 61966-2-1)."""
  arr = jnp.asarray(srgb, dtype=jnp.float32)
  clamped = jnp.clip(arr, 0.0, 1.0)
  safe_hi = jnp.maximum((clamped + 0.055) / 1.055, 1e-12)
  return jnp.where(
      clamped <= 0.04045,
      clamped / 12.92,
      jnp.power(safe_hi, 2.4),
  )


def linear_to_srgb(linear_rgb: jax.Array, clamp: bool = True) -> jax.Array:
  """Converts linear RGB to nonlinear sRGB (IEC 61966-2-1 / libjxl TF_SRGB)."""
  arr = jnp.asarray(linear_rgb, dtype=jnp.float32)
  sign = jnp.where(arr < 0.0, -1.0, 1.0)
  abs_lin = jnp.abs(arr)
  safe_lin = jnp.maximum(abs_lin, 1e-12)
  encoded = jnp.where(
      abs_lin <= 0.0031308,
      12.92 * abs_lin,
      1.055 * jnp.power(safe_lin, 1.0 / 2.4) - 0.055,
  )
  out = sign * encoded
  if clamp:
    out_clamped = jnp.clip(out, 0.0, 1.0)
    return out + jax.lax.stop_gradient(out_clamped - out)
  return out


def srgb_to_xyb(srgb: jax.Array) -> jax.Array:
  """Exact pure-JAX SRGBToXYB (lib/jxl/enc_xyb.cc:49-105).

  Supports both (3, H, W) (CHW) and (H, W, 3) (HWC) layouts and returns the same layout.
  """
  arr = jnp.asarray(srgb, dtype=jnp.float32)
  if arr.ndim != 3:
    raise ValueError(f"Expected 3D sRGB tensor, got shape {arr.shape}")
  is_chw = arr.shape[0] == 3 and arr.shape[2] != 3
  if not is_chw and arr.shape[2] == 3:
    arr_chw = jnp.moveaxis(arr, -1, 0)
  elif arr.shape[0] == 3:
    arr_chw = arr
    is_chw = True
  else:
    raise ValueError(f"Expected sRGB shape (3, H, W) or (H, W, 3), got {arr.shape}")

  linear_rgb = srgb_to_linear(arr_chw)
  absorb_mat = jnp.array(K_OPSIN_ABSORBANCE_MATRIX, dtype=jnp.float32)
  bias = jnp.array(K_OPSIN_ABSORBANCE_BIAS, dtype=jnp.float32)[:, None, None]
  neg_bias_cbrt = jnp.cbrt(-bias)

  mixed = jnp.einsum("ij,jhw->ihw", absorb_mat, linear_rgb) + bias
  mixed = jnp.maximum(mixed, 0.0)
  cbrt_mixed = jnp.cbrt(mixed) + neg_bias_cbrt

  val_x = 0.5 * (cbrt_mixed[0] - cbrt_mixed[1])
  val_y = 0.5 * (cbrt_mixed[0] + cbrt_mixed[1])
  val_b = cbrt_mixed[2]
  xyb_chw = jnp.stack([val_x, val_y, val_b], axis=0)
  return xyb_chw if is_chw else jnp.moveaxis(xyb_chw, 0, -1)


def opsin_to_srgb(
    xyb: jax.Array,
    opsin_inv_matrix: jax.Array | None = None,
    opsin_biases: Sequence[float] = K_NEG_OPSIN_ABSORBANCE_BIAS_RGB,
    clamp: bool = True,
) -> jax.Array:
  """Exact OpsinToLinear + LinearToSRGB (dec_xyb-inl.h:47-84).

  Supports both (3, H, W) (CHW) and (H, W, 3) (HWC) inputs and returns the same layout.
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

  inv_mat = (
      jnp.array(K_DEFAULT_INVERSE_OPSIN_MATRIX, dtype=jnp.float32)
      if opsin_inv_matrix is None
      else jnp.asarray(opsin_inv_matrix, dtype=jnp.float32)
  )
  neg_bias_rgb = jnp.array(opsin_biases[:3], dtype=jnp.float32)[:, None, None]
  neg_bias_cbrt = jnp.cbrt(neg_bias_rgb)

  gamma_x = arr_chw[1] + arr_chw[0] - neg_bias_cbrt[0]
  gamma_y = arr_chw[1] - arr_chw[0] - neg_bias_cbrt[1]
  gamma_b = arr_chw[2] - neg_bias_cbrt[2]

  mixed_r = gamma_x * gamma_x * gamma_x + neg_bias_rgb[0]
  mixed_g = gamma_y * gamma_y * gamma_y + neg_bias_rgb[1]
  mixed_b = gamma_b * gamma_b * gamma_b + neg_bias_rgb[2]
  mixed = jnp.stack([mixed_r, mixed_g, mixed_b], axis=0)

  linear_rgb = jnp.einsum("ij,jhw->ihw", inv_mat, mixed)
  srgb_chw = linear_to_srgb(linear_rgb, clamp=clamp)
  return srgb_chw if is_chw else jnp.moveaxis(srgb_chw, 0, -1)


xyb_to_srgb = opsin_to_srgb


# -----------------------------------------------------------------------------
# 2. DCT / IDCT, Resample Scales, and AcStrategy Transforms
# -----------------------------------------------------------------------------


def dct_resample_scales(from_dim: int, to_dim: int) -> np.ndarray:
  """Exact DCTResampleScales<FROM, TO> from lib/jxl/dct_scales.h:34-232."""
  if from_dim == to_dim:
    return np.ones(from_dim, dtype=np.float64)
  if from_dim < to_dim:
    return 1.0 / dct_resample_scales(to_dim, from_dim)
  k = to_dim
  n = from_dim
  if n != 8 * k:
    raise ValueError(f"Expected from_dim = 8 * to_dim, got {from_dim} vs {to_dim}")
  i = np.arange(k, dtype=np.float64)
  return (
      np.cos(i * np.pi / (2.0 * n))
      * np.cos(i * np.pi / float(n))
      * np.cos(i * np.pi / (0.5 * n))
  )


def make_dct1d_matrix(n: int) -> np.ndarray:
  """libjxl DCT1D<N> forward matrix: (D_N)_{k, i} = (c_k / N) * cos((i + 0.5) * k * pi / N)."""
  k = np.arange(n, dtype=np.float64)[:, None]
  i = np.arange(n, dtype=np.float64)[None, :]
  c = np.where(k == 0, 1.0, np.sqrt(2.0))
  return (c / n) * np.cos((i + 0.5) * k * np.pi / n)


def make_idct1d_matrix(n: int) -> np.ndarray:
  """libjxl IDCT1D<N> inverse matrix: (D_N^{-1})_{i, k} = c_k * cos((i + 0.5) * k * pi / N)."""
  i = np.arange(n, dtype=np.float64)[:, None]
  k = np.arange(n, dtype=np.float64)[None, :]
  c = np.where(k == 0, 1.0, np.sqrt(2.0))
  return c * np.cos((i + 0.5) * k * np.pi / n)


def compute_scaled_dct_2d(pixels: np.ndarray) -> np.ndarray:
  """Exact ComputeScaledDCT<ROWS, COLS> from lib/jxl/dct-inl.h:353-373."""
  rows, cols = pixels.shape
  d_r = make_dct1d_matrix(rows)
  d_c = make_dct1d_matrix(cols)
  dct_rc = d_r @ pixels @ d_c.T
  return dct_rc if rows < cols else dct_rc.T


def compute_scaled_idct_2d(coeffs: np.ndarray, rows: int, cols: int) -> np.ndarray:
  """Exact ComputeScaledIDCT<ROWS, COLS> from lib/jxl/dct-inl.h:376-396."""
  d_r_inv = make_idct1d_matrix(rows)
  d_c_inv = make_idct1d_matrix(cols)
  if rows < cols:
    return d_r_inv @ coeffs @ d_c_inv.T
  return d_r_inv @ coeffs.T @ d_c_inv.T


K_4X4_AFV_BASIS = np.array(
    [
        [
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        ],
        [
            0.876902929799142, 0.2206518106944235, -0.10140050393753763, -0.1014005039375375,
            0.2206518106944236, -0.10140050393753777, -0.10140050393753772, -0.10140050393753763,
            -0.10140050393753758, -0.10140050393753769, -0.1014005039375375, -0.10140050393753768,
            -0.10140050393753768, -0.10140050393753759, -0.10140050393753763, -0.10140050393753741,
        ],
        [
            0.0, 0.0, 0.40670075830260755, 0.44444816619734445,
            0.0, 0.0, 0.19574399372042936, 0.2929100136981264,
            -0.40670075830260716, -0.19574399372042872, 0.0, 0.11379074460448091,
            -0.44444816619734384, -0.29291001369812636, -0.1137907446044814, 0.0,
        ],
        [
            0.0, 0.0, -0.21255748058288748, 0.3085497062849767,
            0.0, 0.4706702258572536, -0.1621205195722993, 0.0,
            -0.21255748058287047, -0.16212051957228327, -0.47067022585725277, -0.1464291867126764,
            0.3085497062849487, 0.0, -0.14642918671266536, 0.4251149611657548,
        ],
        [
            0.0, -0.7071067811865474, 0.0, 0.0,
            0.7071067811865476, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ],
        [
            -0.4105377591765233, 0.6235485373547691, -0.06435071657946274, -0.06435071657946266,
            0.6235485373547694, -0.06435071657946284, -0.0643507165794628, -0.06435071657946274,
            -0.06435071657946272, -0.06435071657946279, -0.06435071657946266, -0.06435071657946277,
            -0.06435071657946277, -0.06435071657946273, -0.06435071657946274, -0.0643507165794626,
        ],
        [
            0.0, 0.0, -0.4517556589999482, 0.15854503551840063,
            0.0, -0.04038515160822202, 0.0074182263792423875, 0.39351034269210167,
            -0.45175565899994635, 0.007418226379244351, 0.1107416575309343, 0.08298163094882051,
            0.15854503551839705, 0.3935103426921022, 0.0829816309488214, -0.45175565899994796,
        ],
        [
            0.0, 0.0, -0.304684750724869, 0.5112616136591823,
            0.0, 0.0, -0.290480129728998, -0.06578701549142804,
            0.304684750724884, 0.2904801297290076, 0.0, -0.23889773523344604,
            -0.5112616136592012, 0.06578701549142545, 0.23889773523345467, 0.0,
        ],
        [
            0.0, 0.0, 0.3017929516615495, 0.25792362796341184,
            0.0, 0.16272340142866204, 0.09520022653475037, 0.0,
            0.3017929516615503, 0.09520022653475055, -0.16272340142866173, -0.35312385449816297,
            0.25792362796341295, 0.0, -0.3531238544981624, -0.6035859033230976,
        ],
        [
            0.0, 0.0, 0.40824829046386274, 0.0,
            0.0, 0.0, 0.0, -0.4082482904638628,
            -0.4082482904638635, 0.0, 0.0, -0.40824829046386296,
            0.0, 0.4082482904638634, 0.408248290463863, 0.0,
        ],
        [
            0.0, 0.0, 0.1747866975480809, 0.0812611176717539,
            0.0, 0.0, -0.3675398009862027, -0.307882213957909,
            -0.17478669754808135, 0.3675398009862011, 0.0, 0.4826689115059883,
            -0.08126111767175039, 0.30788221395790305, -0.48266891150598584, 0.0,
        ],
        [
            0.0, 0.0, -0.21105601049335784, 0.18567180916109802,
            0.0, 0.0, 0.49215859013738733, -0.38525013709251915,
            0.21105601049335806, -0.49215859013738905, 0.0, 0.17419412659916217,
            -0.18567180916109904, 0.3852501370925211, -0.1741941265991621, 0.0,
        ],
        [
            0.0, 0.0, -0.14266084808807264, -0.3416446842253372,
            0.0, 0.7367497537172237, 0.24627107722075148, -0.08574019035519306,
            -0.14266084808807344, 0.24627107722075137, 0.14883399227113567, -0.04768680350229251,
            -0.3416446842253373, -0.08574019035519267, -0.047686803502292804, -0.14266084808807242,
        ],
        [
            0.0, 0.0, -0.13813540350758585, 0.3302282550303788,
            0.0, 0.08755115000587084, -0.07946706605909573, -0.4613374887461511,
            -0.13813540350758294, -0.07946706605910261, 0.49724647109535086, 0.12538059448563663,
            0.3302282550303805, -0.4613374887461554, 0.12538059448564315, -0.13813540350758452,
        ],
        [
            0.0, 0.0, -0.17437602599651067, 0.0702790691196284,
            0.0, -0.2921026642334881, 0.3623817333531167, 0.0,
            -0.1743760259965108, 0.36238173335311646, 0.29210266423348785, -0.4326608024727445,
            0.07027906911962818, 0.0, -0.4326608024727457, 0.34875205199302267,
        ],
        [
            0.0, 0.0, 0.11354987314994337, -0.07417504595810355,
            0.0, 0.19402893032594343, -0.435190496523228, 0.21918684838857466,
            0.11354987314994257, -0.4351904965232251, 0.5550443808910661, -0.25468277124066463,
            -0.07417504595810233, 0.2191868483885728, -0.25468277124066413, 0.1135498731499429,
        ],
    ],
    dtype=np.float64,
)


def _idct2_top_block(block: np.ndarray, s: int) -> np.ndarray:
  """Exact IDCT2TopBlock<S> from lib/jxl/dec_transforms-inl.h:66-93."""
  out = block.copy()
  num_2x2 = s // 2
  temp = np.zeros((s, s), dtype=np.float64)
  for y in range(num_2x2):
    for x in range(num_2x2):
      c00 = block[y, x]
      c01 = block[y, num_2x2 + x]
      c10 = block[y + num_2x2, x]
      c11 = block[y + num_2x2, num_2x2 + x]
      temp[2 * y, 2 * x] = c00 + c01 + c10 + c11
      temp[2 * y, 2 * x + 1] = c00 + c01 - c10 - c11
      temp[2 * y + 1, 2 * x] = c00 - c01 + c10 - c11
      temp[2 * y + 1, 2 * x + 1] = c00 - c01 - c10 + c11
  out[:s, :s] = temp
  return out


def _afv_transform_to_pixels(coeffs_8x8: np.ndarray, afv_kind: int) -> np.ndarray:
  """Exact AFVTransformToPixels<afv_kind> from lib/jxl/dec_transforms-inl.h:399-454."""
  c = coeffs_8x8.reshape(64)
  afv_x = afv_kind & 1
  afv_y = afv_kind // 2
  block00 = c[0]
  block01 = c[1]
  block10 = c[8]
  dcs0 = (block00 + block10 + block01) * 4.0
  dcs1 = block00 + block10 - block01
  dcs2 = block00 - block10

  pixels = np.zeros((8, 8), dtype=np.float64)

  coeff_ee = coeffs_8x8[0::2, 0::2].copy()
  coeff_ee[0, 0] = dcs0
  block_ee = (K_4X4_AFV_BASIS.T @ coeff_ee.reshape(16)).reshape(4, 4)
  for iy in range(4):
    for ix in range(4):
      src_y = (3 - iy) if afv_y == 1 else iy
      src_x = (3 - ix) if afv_x == 1 else ix
      pixels[afv_y * 4 + iy, afv_x * 4 + ix] = block_ee[src_y, src_x]

  block_eo = coeffs_8x8[0::2, 1::2].copy()
  block_eo[0, 0] = dcs1
  px_eo = compute_scaled_idct_2d(block_eo, 4, 4)
  x_off = 0 if afv_x == 1 else 4
  pixels[afv_y * 4 : (afv_y + 1) * 4, x_off : x_off + 4] = px_eo

  block_oy = coeffs_8x8[1::2, :].copy()
  block_oy[0, 0] = dcs2
  px_oy = compute_scaled_idct_2d(block_oy, 4, 8)
  y_off = 0 if afv_y == 1 else 4
  pixels[y_off : y_off + 4, :] = px_oy

  return pixels


def build_1x1_strategy_basis_matrices() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
  """Builds exact 64x64 inverse (coeffs->pixels) and forward (pixels->coeffs) basis matrices."""

  def apply_to_pixels(s: int, coeff_8x8: np.ndarray) -> np.ndarray:
    c = coeff_8x8.reshape(64)
    if s == 0:
      return compute_scaled_idct_2d(coeff_8x8, 8, 8)
    if s == 1:
      b00, b01, b10, b11 = c[0], c[1], c[8], c[9]
      dcs = [
          b00 + b01 + b10 + b11,
          b00 + b01 - b10 - b11,
          b00 - b01 + b10 - b11,
          b00 - b01 - b10 + b11,
      ]
      pixels = np.zeros((8, 8), dtype=np.float64)
      for y in range(2):
        for x in range(2):
          block_dc = dcs[y * 2 + x]
          residual_sum = 0.0
          for iy in range(4):
            for ix in range(4):
              if ix == 0 and iy == 0:
                continue
              residual_sum += coeff_8x8[y + iy * 2, x + ix * 2]
          p11 = block_dc - residual_sum * (1.0 / 16.0)
          pixels[4 * y + 1, 4 * x + 1] = p11
          for iy in range(4):
            for ix in range(4):
              if ix == 1 and iy == 1:
                continue
              pixels[4 * y + iy, 4 * x + ix] = coeff_8x8[y + iy * 2, x + ix * 2] + p11
          pixels[4 * y, 4 * x] = coeff_8x8[y + 2, x + 2] + p11
      return pixels
    if s == 2:
      b = _idct2_top_block(coeff_8x8, 2)
      b = _idct2_top_block(b, 4)
      b = _idct2_top_block(b, 8)
      return b
    if s == 3:
      b00, b01, b10, b11 = c[0], c[1], c[8], c[9]
      dcs = [
          b00 + b01 + b10 + b11,
          b00 + b01 - b10 - b11,
          b00 - b01 + b10 - b11,
          b00 - b01 - b10 + b11,
      ]
      pixels = np.zeros((8, 8), dtype=np.float64)
      for y in range(2):
        for x in range(2):
          sub = coeff_8x8[y::2, x::2].copy()
          sub[0, 0] = dcs[y * 2 + x]
          pixels[4 * y : 4 * (y + 1), 4 * x : 4 * (x + 1)] = compute_scaled_idct_2d(sub, 4, 4)
      return pixels
    if s == 12:
      b0, b1 = c[0], c[8]
      dcs = [b0 + b1, b0 - b1]
      pixels = np.zeros((8, 8), dtype=np.float64)
      for y in range(2):
        sub = coeff_8x8[y::2, :].copy()
        sub[0, 0] = dcs[y]
        pixels[4 * y : 4 * (y + 1), :] = compute_scaled_idct_2d(sub, 4, 8)
      return pixels
    if s == 13:
      b0, b1 = c[0], c[8]
      dcs = [b0 + b1, b0 - b1]
      pixels = np.zeros((8, 8), dtype=np.float64)
      for x in range(2):
        sub = coeff_8x8[x::2, :].copy()
        sub[0, 0] = dcs[x]
        pixels[:, 4 * x : 4 * (x + 1)] = compute_scaled_idct_2d(sub, 8, 4)
      return pixels
    if s in (14, 15, 16, 17):
      return _afv_transform_to_pixels(coeff_8x8, s - 14)
    raise ValueError(f"Unsupported 1x1 strategy {s}")

  inv_basis: dict[int, np.ndarray] = {}
  fwd_basis: dict[int, np.ndarray] = {}
  for s in (0, 1, 2, 3, 12, 13, 14, 15, 16, 17):
    mat = np.zeros((64, 64), dtype=np.float64)
    for i in range(64):
      e = np.zeros((8, 8), dtype=np.float64)
      e.flat[i] = 1.0
      mat[:, i] = apply_to_pixels(s, e).reshape(64)
    inv_basis[s] = mat.astype(np.float32)
    fwd_basis[s] = np.linalg.inv(mat).astype(np.float32)
  return inv_basis, fwd_basis


BASIS_1X1_INV_NP, BASIS_1X1_FWD_NP = build_1x1_strategy_basis_matrices()
MAX_1X1_ID: int = 18
BASIS_1X1_INV_TABLE_NP = np.zeros((MAX_1X1_ID, 64, 64), dtype=np.float32)
BASIS_1X1_FWD_TABLE_NP = np.zeros((MAX_1X1_ID, 64, 64), dtype=np.float32)
for _s in BASIS_1X1_INV_NP:
  BASIS_1X1_INV_TABLE_NP[_s] = BASIS_1X1_INV_NP[_s]
  BASIS_1X1_FWD_TABLE_NP[_s] = BASIS_1X1_FWD_NP[_s]

BASIS_1X1_INV_TABLE: jax.Array = jnp.asarray(BASIS_1X1_INV_TABLE_NP)
BASIS_1X1_FWD_TABLE: jax.Array = jnp.asarray(BASIS_1X1_FWD_TABLE_NP)


def transform_to_pixels_for_strategy(strategy_id: int, coeffs: jax.Array) -> jax.Array:
  """Transforms canonical coefficient block(s) to spatial pixels for any AcStrategy."""
  s = int(strategy_id)
  arr = jnp.asarray(coeffs, dtype=jnp.float32)
  cx = ACS_COVERED_BLOCKS_X[s]
  cy = ACS_COVERED_BLOCKS_Y[s]
  if cx == 1 and cy == 1:
    inv_mat = jnp.asarray(BASIS_1X1_INV_TABLE_NP[s], dtype=jnp.float32)
    orig_shape = arr.shape[:-2]
    flat = arr.reshape(-1, 64)
    px = jnp.einsum("ij,nj->ni", inv_mat, flat)
    return px.reshape(orig_shape + (8, 8))
  rows, cols = 8 * cy, 8 * cx
  idct_r = jnp.asarray(make_idct1d_matrix(rows), dtype=jnp.float32)
  idct_c = jnp.asarray(make_idct1d_matrix(cols), dtype=jnp.float32)
  c_in = jnp.swapaxes(arr, -2, -1) if cy >= cx else arr
  return jnp.einsum("ij,...jk,lk->...il", idct_r, c_in, idct_c)


def transform_from_pixels_for_strategy(strategy_id: int, pixels: jax.Array) -> jax.Array:
  """Transforms spatial pixel block(s) `(..., 8*cy, 8*cx)` to canonical coefficients `(..., 8*min, 8*max)`."""
  s = int(strategy_id)
  arr = jnp.asarray(pixels, dtype=jnp.float32)
  cx = ACS_COVERED_BLOCKS_X[s]
  cy = ACS_COVERED_BLOCKS_Y[s]
  if cx == 1 and cy == 1:
    fwd_mat = jnp.asarray(BASIS_1X1_FWD_TABLE_NP[s], dtype=jnp.float32)
    orig_shape = arr.shape[:-2]
    flat = arr.reshape(-1, 64)
    c_out = jnp.einsum("ij,nj->ni", fwd_mat, flat)
    return c_out.reshape(orig_shape + (8, 8))
  rows, cols = 8 * cy, 8 * cx
  dct_r = jnp.asarray(make_dct1d_matrix(rows), dtype=jnp.float32)
  dct_c = jnp.asarray(make_dct1d_matrix(cols), dtype=jnp.float32)
  c_rc = jnp.einsum("ij,...jk,lk->...il", dct_r, arr, dct_c)
  return jnp.swapaxes(c_rc, -2, -1) if cy >= cx else c_rc


def lowest_frequencies_from_dc_for_strategy(strategy_id: int, dc_block: jax.Array) -> jax.Array:
  """Computes canonical LLF coefficients `(..., min(cy,cx), max(cy,cx))` from DC `(..., cy, cx)`."""
  s = int(strategy_id)
  dc = jnp.asarray(dc_block, dtype=jnp.float32)
  cx = ACS_COVERED_BLOCKS_X[s]
  cy = ACS_COVERED_BLOCKS_Y[s]
  if cx == 1 and cy == 1:
    return dc
  d_cy = jnp.asarray(make_dct1d_matrix(cy), dtype=jnp.float32)
  d_cx = jnp.asarray(make_dct1d_matrix(cx), dtype=jnp.float32)
  rs_y = jnp.asarray(dct_resample_scales(cy, 8 * cy), dtype=jnp.float32)
  rs_x = jnp.asarray(dct_resample_scales(cx, 8 * cx), dtype=jnp.float32)
  llf = jnp.einsum("ij,...jk,lk->...il", d_cy, dc, d_cx)
  llf = llf * (rs_y[:, None] * rs_x[None, :])
  return jnp.swapaxes(llf, -2, -1) if cy >= cx else llf


def dc_from_lowest_frequencies_for_strategy(strategy_id: int, coeffs: jax.Array) -> jax.Array:
  """Extracts DC block `(..., cy, cx)` from canonical coefficients `(..., 8*min, 8*max)`."""
  s = int(strategy_id)
  c_arr = jnp.asarray(coeffs, dtype=jnp.float32)
  cx = ACS_COVERED_BLOCKS_X[s]
  cy = ACS_COVERED_BLOCKS_Y[s]
  if cx == 1 and cy == 1:
    return c_arr[..., :1, :1]
  cy_l, cx_l = min(cy, cx), max(cy, cx)
  llf = c_arr[..., :cy_l, :cx_l]
  llf_rc = jnp.swapaxes(llf, -2, -1) if cy >= cx else llf
  rs_y = jnp.asarray(dct_resample_scales(8 * cy, cy), dtype=jnp.float32)
  rs_x = jnp.asarray(dct_resample_scales(8 * cx, cx), dtype=jnp.float32)
  llf_scaled = llf_rc * (rs_y[:, None] * rs_x[None, :])
  idct_cy = jnp.asarray(make_idct1d_matrix(cy), dtype=jnp.float32)
  idct_cx = jnp.asarray(make_idct1d_matrix(cx), dtype=jnp.float32)
  return jnp.einsum("ij,...jk,lk->...il", idct_cy, llf_scaled, idct_cx)


K_AC_STRATEGY_TO_QUANT_TABLE_MAP: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9,
    10, 10, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16,
)
K_QUANT_TABLE_REQUIRED_SIZE_X: tuple[int, ...] = (
    1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 8, 4, 16, 8, 32, 16,
)
K_QUANT_TABLE_REQUIRED_SIZE_Y: tuple[int, ...] = (
    1, 1, 1, 1, 2, 4, 2, 4, 4, 1, 1, 8, 8, 16, 16, 32, 32,
)
K_QUANT_TABLE_TO_CANONICAL_STRATEGY: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 18, 19, 21, 22, 24, 25,
)

_LIBRARY_DCT_BANDS: dict[int, tuple[tuple[float, ...], ...]] = {
    0: (
        (3150.0, 0.0, -0.4, -0.4, -0.4, -2.0),
        (560.0, 0.0, -0.3, -0.3, -0.3, -0.3),
        (512.0, -2.0, -1.0, 0.0, -1.0, -2.0),
    ),
    3: (
        (2200.0, 0.0, 0.0, 0.0),
        (392.0, 0.0, 0.0, 0.0),
        (112.0, -0.25, -0.25, -0.5),
    ),
    4: (
        (
            8996.8725711814115328,
            -1.3000777393353804,
            -0.49424529824571225,
            -0.439093774457103443,
            -0.6350101832695744,
            -0.90177264050827612,
            -1.6162099239887414,
        ),
        (
            3191.48366296844234752,
            -0.67424582104194355,
            -0.80745813428471001,
            -0.44925837484843441,
            -0.35865440981033403,
            -0.31322389111877305,
            -0.37615025315725483,
        ),
        (
            1157.50408145487200256,
            -2.0531423165804414,
            -1.4,
            -0.50687130033378396,
            -0.42708730624733904,
            -1.4856834539296244,
            -4.9209142884401604,
        ),
    ),
    5: (
        (
            15718.40830982518931456,
            -1.025,
            -0.98,
            -0.9012,
            -0.4,
            -0.48819395464,
            -0.421064,
            -0.27,
        ),
        (
            7305.7636810695983104,
            -0.8041958212306401,
            -0.7633036457487539,
            -0.55660379990111464,
            -0.49785304658857626,
            -0.43699592683512467,
            -0.40180866526242109,
            -0.27321683125358037,
        ),
        (
            3803.53173721215041536,
            -3.060733579805728,
            -2.0413270132490346,
            -2.0235650159727417,
            -0.5495389509954993,
            -0.4,
            -0.4,
            -0.3,
        ),
    ),
    6: (
        (7240.7734393502, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5),
        (1448.15468787004, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2),
        (506.854140754517, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6),
    ),
    7: (
        (
            16283.2494710648897,
            -1.7812845336559429,
            -1.6309059012653515,
            -1.0382179034313539,
            -0.85,
            -0.7,
            -0.9,
            -1.2360638576849587,
        ),
        (
            5089.15750884921511936,
            -0.320049391452786891,
            -0.35362849922161446,
            -0.30340000000000003,
            -0.61,
            -0.5,
            -0.5,
            -0.6,
        ),
        (
            3397.77603275308720128,
            -0.321327362693153371,
            -0.34507619223117997,
            -0.70340000000000003,
            -0.9,
            -1.0,
            -1.0,
            -1.1754605576265209,
        ),
    ),
    8: (
        (
            13844.97076442300573,
            -0.97113799999999995,
            -0.658,
            -0.42026,
            -0.22712,
            -0.2206,
            -0.226,
            -0.6,
        ),
        (
            4798.964084220744293,
            -0.61125308982767057,
            -0.83770786552491361,
            -0.79014862079498627,
            -0.2692727459704829,
            -0.38272769465388551,
            -0.22924222653091453,
            -0.20719098826199578,
        ),
        (1807.236946760964614, -1.2, -1.2, -0.7, -0.7, -0.7, -0.4, -0.5),
    ),
    9: (
        (
            2198.050556016380522,
            -0.96269623020744692,
            -0.76194253026666783,
            -0.6551140670773547,
        ),
        (
            764.3655248643528689,
            -0.92630200888366945,
            -0.9675229603596517,
            -0.27845290869168118,
        ),
        (
            527.107573587542228,
            -1.4594385811273854,
            -1.450082094097871593,
            -1.5843722511996204,
        ),
    ),
    11: (
        (
            0.9 * 26629.073922049845,
            -1.025,
            -0.78,
            -0.65012,
            -0.19041574084286472,
            -0.20819395464,
            -0.421064,
            -0.32733845535848671,
        ),
        (
            0.9 * 9311.3238710010046,
            -0.3041958212306401,
            -0.3633036457487539,
            -0.35660379990111464,
            -0.3443074455424403,
            -0.33699592683512467,
            -0.30180866526242109,
            -0.27321683125358037,
        ),
        (0.9 * 4992.2486445538634, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5),
    ),
    12: (
        (
            0.65 * 23629.073922049845,
            -1.025,
            -0.78,
            -0.65012,
            -0.19041574084286472,
            -0.20819395464,
            -0.421064,
            -0.32733845535848671,
        ),
        (
            0.65 * 8611.3238710010046,
            -0.3041958212306401,
            -0.3633036457487539,
            -0.35660379990111464,
            -0.3443074455424403,
            -0.33699592683512467,
            -0.30180866526242109,
            -0.27321683125358037,
        ),
        (0.65 * 4492.2486445538634, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5),
    ),
    13: (
        (
            1.8 * 26629.073922049845,
            -1.025,
            -0.78,
            -0.65012,
            -0.19041574084286472,
            -0.20819395464,
            -0.421064,
            -0.32733845535848671,
        ),
        (
            1.8 * 9311.3238710010046,
            -0.3041958212306401,
            -0.3633036457487539,
            -0.35660379990111464,
            -0.3443074455424403,
            -0.33699592683512467,
            -0.30180866526242109,
            -0.27321683125358037,
        ),
        (1.8 * 4992.2486445538634, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5),
    ),
    14: (
        (
            1.3 * 23629.073922049845,
            -1.025,
            -0.78,
            -0.65012,
            -0.19041574084286472,
            -0.20819395464,
            -0.421064,
            -0.32733845535848671,
        ),
        (
            1.3 * 8611.3238710010046,
            -0.3041958212306401,
            -0.3633036457487539,
            -0.35660379990111464,
            -0.3443074455424403,
            -0.33699592683512467,
            -0.30180866526242109,
            -0.27321683125358037,
        ),
        (1.3 * 4492.2486445538634, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5),
    ),
    15: (
        (
            3.6 * 26629.073922049845,
            -1.025,
            -0.78,
            -0.65012,
            -0.19041574084286472,
            -0.20819395464,
            -0.421064,
            -0.32733845535848671,
        ),
        (
            3.6 * 9311.3238710010046,
            -0.3041958212306401,
            -0.3633036457487539,
            -0.35660379990111464,
            -0.3443074455424403,
            -0.33699592683512467,
            -0.30180866526242109,
            -0.27321683125358037,
        ),
        (3.6 * 4992.2486445538634, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5),
    ),
    16: (
        (
            2.6 * 23629.073922049845,
            -1.025,
            -0.78,
            -0.65012,
            -0.19041574084286472,
            -0.20819395464,
            -0.421064,
            -0.32733845535848671,
        ),
        (
            2.6 * 8611.3238710010046,
            -0.3041958212306401,
            -0.3633036457487539,
            -0.35660379990111464,
            -0.3443074455424403,
            -0.33699592683512467,
            -0.30180866526242109,
            -0.27321683125358037,
        ),
        (2.6 * 4492.2486445538634, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5),
    ),
}


def _fast_powf_np(base: np.ndarray, exponent: np.ndarray) -> np.ndarray:
  """Exact port of jxl::FastPowf (lib/jxl/base/fast_math-inl.h:48-92)."""
  b = np.asarray(base, dtype=np.float32)
  e = np.asarray(exponent, dtype=np.float32)
  x_bits = b.view(np.int32)
  exp_bits = (x_bits - np.int32(0x3F2AAAAB)).astype(np.int32)
  exp_shifted = np.right_shift(exp_bits, np.int32(23)).astype(np.int32)
  mant_bits = (x_bits - np.left_shift(exp_shifted, np.int32(23))).astype(
      np.int32
  )
  mant = mant_bits.view(np.float32) - np.float32(1.0)
  p0 = np.float32(-1.8503833400518310e-06)
  p1 = np.float32(1.4287160470083755e00)
  p2 = np.float32(7.4245873327820566e-01)
  q0 = np.float32(9.9032814277590719e-01)
  q1 = np.float32(1.0096718572241148e00)
  q2 = np.float32(1.7409343003366853e-01)
  yp = np.float32(p2 * mant + p1) * mant + p0
  yq = np.float32(q2 * mant + q1) * mant + q0
  fast_log2 = np.float32(yp / yq) + exp_shifted.astype(np.float32)

  prod = np.float32(fast_log2 * e)
  floorx = np.floor(prod).astype(np.float32)
  exp2_bits = np.left_shift(
      floorx.astype(np.int32) + np.int32(127), np.int32(23)
  ).astype(np.int32)
  exp2 = exp2_bits.view(np.float32)
  frac = np.float32(prod - floorx)
  num = np.float32(frac + np.float32(1.01749063e01))
  num = np.float32(num * frac + np.float32(4.88687798e01))
  num = np.float32(num * frac + np.float32(9.85506591e01))
  num = np.float32(num * exp2)
  den = np.float32(
      frac * np.float32(2.10242958e-01) + np.float32(-2.22328856e-02)
  )
  den = np.float32(den * frac + np.float32(-1.94414990e01))
  den = np.float32(den * frac + np.float32(9.85506633e01))
  return np.float32(num / den)


def _quant_mult_f32(v: float) -> np.float32:
  vf = np.float32(v)
  if vf > np.float32(0.0):
    return np.float32(np.float32(1.0) + vf)
  return np.float32(np.float32(1.0) / np.float32(np.float32(1.0) - vf))


def _get_quant_weights_np(
    rows: int, cols: int, distance_bands: Sequence[Sequence[float]]
) -> np.ndarray:
  """Exact port of jxl::GetQuantWeights (lib/jxl/quant_weights.cc:130-160)."""
  num_bands = len(distance_bands[0])
  out = np.empty((3, rows, cols), dtype=np.float32)
  k_sqrt2 = np.float32(1.4142135623730951)
  scale = np.float32(
      np.float32(num_bands - 1) / np.float32(float(k_sqrt2) + 1e-6)
  )
  rcpcol = np.float32(scale / np.float32(cols - 1))
  rcprow = np.float32(scale / np.float32(rows - 1))
  ys = np.arange(rows, dtype=np.float32)[:, None]
  xs = np.arange(cols, dtype=np.float32)[None, :]
  dy = np.float32(ys * rcprow)
  dy2 = np.float32(dy * dy)
  dx = np.float32(xs * rcpcol)
  scaled_dist = np.sqrt(np.float32(dx * dx + dy2)).astype(np.float32)
  idx = scaled_dist.astype(np.int32)
  frac = np.float32(scaled_dist - idx.astype(np.float32))

  for c in range(3):
    bands = np.zeros(num_bands + 1, dtype=np.float32)
    bands[0] = np.float32(distance_bands[c][0])
    for i in range(1, num_bands):
      bands[i] = np.float32(bands[i - 1] * _quant_mult_f32(distance_bands[c][i]))
    bands[num_bands] = bands[num_bands - 1]
    if num_bands == 1:
      out[c] = bands[0]
    else:
      a = bands[idx]
      b = bands[idx + 1]
      out[c] = np.float32(a * _fast_powf_np(np.float32(b / a), frac))
  return out


@functools.lru_cache(maxsize=1)
def get_default_dequant_matrices() -> dict[int, np.ndarray]:
  """Computes and caches the 27 universal JPEG XL default DequantMatrices in pure Python/NumPy.

  Exact port of `DequantMatrices::LibraryInit()` and `ComputeQuantTable` from
  `lib/jxl/quant_weights.cc:48-1110`. Never invokes `dump_init_state` or any
  external binary.
  """
  qt_tables: dict[int, np.ndarray] = {}

  # Modes using kQuantModeDCT directly
  for qt_idx in (0, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16):
    wrows = 8 * K_QUANT_TABLE_REQUIRED_SIZE_X[qt_idx]
    wcols = 8 * K_QUANT_TABLE_REQUIRED_SIZE_Y[qt_idx]
    weights = _get_quant_weights_np(wrows, wcols, _LIBRARY_DCT_BANDS[qt_idx])
    qt_tables[qt_idx] = (np.float32(1.0) / weights).astype(np.float32)

  # qt_idx = 1: IDENTITY (kQuantModeID)
  id_weights_cfg = (
      (280.0, 3160.0, 3160.0),
      (60.0, 864.0, 864.0),
      (18.0, 200.0, 200.0),
  )
  w_id = np.empty((3, 8, 8), dtype=np.float32)
  for c in range(3):
    w_id[c, :, :] = np.float32(id_weights_cfg[c][0])
    w_id[c, 0, 1] = np.float32(id_weights_cfg[c][1])
    w_id[c, 1, 0] = np.float32(id_weights_cfg[c][1])
    w_id[c, 1, 1] = np.float32(id_weights_cfg[c][2])
  qt_tables[1] = (np.float32(1.0) / w_id).astype(np.float32)

  # qt_idx = 2: DCT2X2 (kQuantModeDCT2)
  dct2_cfg = (
      (3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0),
      (960.0, 640.0, 320.0, 180.0, 140.0, 120.0),
      (640.0, 320.0, 128.0, 64.0, 32.0, 16.0),
  )
  w_dct2 = np.empty((3, 8, 8), dtype=np.float32)
  for c in range(3):
    w_dct2[c, 0, 0] = np.float32(0xBAD)
    w_dct2[c, 0, 1] = np.float32(dct2_cfg[c][0])
    w_dct2[c, 1, 0] = np.float32(dct2_cfg[c][0])
    w_dct2[c, 1, 1] = np.float32(dct2_cfg[c][1])
    w_dct2[c, 0:2, 2:4] = np.float32(dct2_cfg[c][2])
    w_dct2[c, 2:4, 0:2] = np.float32(dct2_cfg[c][2])
    w_dct2[c, 2:4, 2:4] = np.float32(dct2_cfg[c][3])
    w_dct2[c, 0:4, 4:8] = np.float32(dct2_cfg[c][4])
    w_dct2[c, 4:8, 0:4] = np.float32(dct2_cfg[c][4])
    w_dct2[c, 4:8, 4:8] = np.float32(dct2_cfg[c][5])
  qt_tables[2] = (np.float32(1.0) / w_dct2).astype(np.float32)

  # qt_idx = 3: DCT4X4 (kQuantModeDCT4)
  w4x4 = _get_quant_weights_np(4, 4, _LIBRARY_DCT_BANDS[3])
  w_dct4 = np.repeat(np.repeat(w4x4, 2, axis=1), 2, axis=2)
  qt_tables[3] = (np.float32(1.0) / w_dct4).astype(np.float32)

  # qt_idx = 9: DCT4X8 (kQuantModeDCT4X8)
  w4x8 = _get_quant_weights_np(4, 8, _LIBRARY_DCT_BANDS[9])
  w_dct4x8 = np.repeat(w4x8, 2, axis=1)
  qt_tables[9] = (np.float32(1.0) / w_dct4x8).astype(np.float32)

  # qt_idx = 10: AFV0 (kQuantModeAFV)
  afv_cfg = (
      (3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0),
      (1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0),
      (384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25),
  )
  k_freqs = np.array(
      [
          float(0xBAD),
          float(0xBAD),
          0.8517778890324296,
          5.37778436506804,
          float(0xBAD),
          float(0xBAD),
          4.734747904497923,
          5.449245381693219,
          1.6598270267479331,
          4.0,
          7.275749096817861,
          10.423227632456525,
          2.662932286148962,
          7.630657783650829,
          8.962388608184032,
          12.97166202570235,
      ],
      dtype=np.float32,
  )
  lo = np.float32(0.8517778890324296)
  hi = np.float32(np.float32(12.97166202570235) - lo + np.float32(1e-6))
  w_afv = np.empty((3, 8, 8), dtype=np.float32)
  for c in range(3):
    bands = np.zeros(4, dtype=np.float32)
    bands[0] = np.float32(afv_cfg[c][5])
    for i in range(1, 4):
      bands[i] = np.float32(bands[i - 1] * _quant_mult_f32(afv_cfg[c][i + 5]))
    w_afv[c, 0, 0] = np.float32(1.0)
    w_afv[c, 1, 0] = np.float32(afv_cfg[c][0])
    w_afv[c, 0, 1] = np.float32(afv_cfg[c][1])
    w_afv[c, 2, 0] = np.float32(afv_cfg[c][2])
    w_afv[c, 0, 2] = np.float32(afv_cfg[c][3])
    w_afv[c, 2, 2] = np.float32(afv_cfg[c][4])
    for y in range(4):
      for x in range(4):
        if x < 2 and y < 2:
          continue
        pos = np.float32(k_freqs[y * 4 + x] - lo)
        scaled_pos = np.float32(pos * np.float32(3.0) / hi)
        idx = int(scaled_pos)
        frac = np.float32(scaled_pos - np.float32(idx))
        a = bands[idx]
        b = bands[idx + 1]
        val = np.float32(a * _fast_powf_np(np.float32(b / a), frac))
        w_afv[c, 2 * y, 2 * x] = val
    for y in range(4):
      for x in range(8):
        if x == 0 and y == 0:
          continue
        w_afv[c, 2 * y + 1, x] = w4x8[c, y, x]
    for y in range(4):
      for x in range(4):
        if x == 0 and y == 0:
          continue
        w_afv[c, 2 * y, 2 * x + 1] = w4x4[c, y, x]
  qt_tables[10] = (np.float32(1.0) / w_afv).astype(np.float32)

  return {
      s: qt_tables[K_AC_STRATEGY_TO_QUANT_TABLE_MAP[s]].copy()
      for s in range(len(ACS_COVERED_BLOCKS_X))
  }


@functools.lru_cache(maxsize=1)
def get_default_inv_dequant_matrices() -> dict[int, np.ndarray]:
  """Returns 1.0 / default_dequant_matrices for each of the 27 AcStrategy types."""
  dq = get_default_dequant_matrices()
  return {k: (1.0 / np.maximum(v, 1e-12)).astype(np.float32) for k, v in dq.items()}


def _f16_round_jax(x: jax.Array) -> jax.Array:
  """Rounds float32 array to IEEE-754 float16 precision and casts back to float32."""
  return jnp.asarray(x, dtype=jnp.float16).astype(jnp.float32)


def _project_single_table_jax(
    mat: jax.Array,
    default_mat: jax.Array,
    mode: str = "dct",
) -> jax.Array:
  """Projects one `(3, H_q, W_q)` dequant matrix to libjxl's quantized QuantEncoding space."""
  m_in = jnp.clip(jnp.asarray(mat, dtype=jnp.float32), 1e-7, 10.0)
  m_def = jnp.asarray(default_mat, dtype=jnp.float32)
  orig_shape = m_in.shape
  if m_in.ndim == 2 and m_in.shape == (3, 64) and m_def.shape == (3, 8, 8):
    m_3d = m_in.reshape(3, 8, 8)
  elif m_in.shape != m_def.shape and m_in.size == 3 * 64:
    m_3d = m_in.reshape(3, 8, 8)
    m_def = jax.image.resize(m_def, (3, 8, 8), method="linear")
  else:
    m_3d = m_in

  is_default = jnp.max(jnp.abs(m_3d - m_def) / jnp.maximum(m_def, 1e-7)) < 1e-6

  if mode.lower() == "raw":
    max_val = jnp.max(m_3d)
    qtable_den = jnp.clip(_f16_round_jax(max_val / 255.0), 1e-7, 10.0)
    qtable_int = jnp.clip(jnp.round(m_3d / qtable_den), 1.0, 255.0)
    m_proj = jnp.clip(qtable_den * qtable_int, 1e-7, 10.0)
  else:
    # Parametric DCT distance-band projection (kQuantModeDCT / F16Coder)
    _, rows, cols = m_3d.shape
    num_bands = 6 if (rows == 8 and cols == 8) else (7 if max(rows, cols) == 16 else 8)
    k_sqrt2 = jnp.float32(1.4142135623730951)
    scale = jnp.float32(num_bands - 1) / (k_sqrt2 + 1e-6)
    ys = jnp.arange(rows, dtype=jnp.float32)[:, None] * (scale / float(max(rows - 1, 1)))
    xs = jnp.arange(cols, dtype=jnp.float32)[None, :] * (scale / float(max(cols - 1, 1)))
    scaled_dist = jnp.sqrt(xs * xs + ys * ys)
    idx = jnp.clip(jnp.floor(scaled_dist).astype(jnp.int32), 0, num_bands - 2)
    frac = scaled_dist - idx.astype(jnp.float32)

    basis_list = []
    for b in range(num_bands):
      w_b = jnp.where(idx == b, 1.0 - frac, 0.0) + jnp.where(
          idx + 1 == b, frac, 0.0
      )
      basis_list.append(w_b.reshape(-1))
    a_mat = jnp.stack(basis_list, axis=-1)  # (N_px, num_bands)
    ata = a_mat.T @ a_mat + 1e-6 * jnp.eye(num_bands, dtype=jnp.float32)

    # Fit log(weights) = -log(m_3d / m_def) relative to m_def so default tables project to exact m_def
    log_rel_w = -jnp.log(m_3d / jnp.maximum(m_def, 1e-7)).reshape(3, -1)
    rhs = log_rel_w @ a_mat  # (3, num_bands)
    log_bands_rel = jnp.linalg.solve(ata, rhs.T).T  # (3, num_bands)
    bands_rel = jnp.exp(jnp.clip(log_bands_rel, -12.0, 12.0))

    # Quantize band[0] and step multipliers via F16Coder
    b0_q = jnp.clip(_f16_round_jax(bands_rel[:, 0]), 1e-4, 1e4)
    bands_q = [b0_q]
    for i in range(1, num_bands):
      ratio = bands_rel[:, i] / jnp.maximum(bands_q[-1], 1e-6)
      v_i = jnp.where(ratio >= 1.0, ratio - 1.0, 1.0 - 1.0 / jnp.maximum(ratio, 1e-6))
      v_q = _f16_round_jax(v_i)
      mult_q = jnp.where(v_q > 0.0, 1.0 + v_q, 1.0 / jnp.maximum(1.0 - v_q, 1e-6))
      bands_q.append(jnp.clip(bands_q[-1] * mult_q, 1e-4, 1e4))
    bands_q_arr = jnp.stack(bands_q, axis=-1)  # (3, num_bands)

    a_band = jnp.take(bands_q_arr, idx.reshape(-1), axis=-1).reshape(3, rows, cols)
    b_band = jnp.take(bands_q_arr, (idx + 1).reshape(-1), axis=-1).reshape(3, rows, cols)
    rel_w_proj = a_band * jnp.power(
        jnp.maximum(b_band / jnp.maximum(a_band, 1e-7), 1e-7), frac[None, :, :]
    )
    m_proj = jnp.clip(m_def / jnp.maximum(rel_w_proj, 1e-7), 1e-7, 10.0)

  out_3d = jnp.where(is_default, m_def, m_proj)
  return out_3d.reshape(orig_shape)


def project_custom_dequant_matrices(
    dequant_matrices: Any,
    dc_quant: jax.Array | np.ndarray | Sequence[float] | None = None,
    mode: str = "dct",
) -> Any:
  """Projects continuous custom `dequant_matrices` (and optional `dc_quant` `(3,)`) to libjxl's quantized encoding space."""
  default_dq = get_default_dequant_matrices()
  if isinstance(dequant_matrices, Mapping):
    proj_dict: dict[int, jax.Array] = {}
    for s, mat in dequant_matrices.items():
      s_int = int(s)
      def_mat = jnp.asarray(default_dq[s_int], dtype=jnp.float32)
      proj_dict[s_int] = _project_single_table_jax(
          jnp.asarray(mat, dtype=jnp.float32), def_mat, mode=mode
      )
    proj_dq: Any = proj_dict
  else:
    arr = jnp.asarray(dequant_matrices, dtype=jnp.float32)
    orig_shape = arr.shape
    if arr.ndim >= 3 and arr.shape[1] == 3 and (
        arr.shape[2:] == (64,) or arr.shape[2:] == (8, 8)
    ):
      n_slots = arr.shape[0]
      proj_list = []
      for s in range(n_slots):
        s_idx = min(s, len(ACS_COVERED_BLOCKS_X) - 1)
        def_s = jnp.asarray(default_dq[s_idx], dtype=jnp.float32)
        if def_s.shape != (3, 8, 8):
          def_s = jax.image.resize(def_s, (3, 8, 8), method="linear")
        proj_s = _project_single_table_jax(
            arr[s].reshape(3, 8, 8), def_s, mode=mode
        )
        proj_list.append(proj_s.reshape(orig_shape[1:]))
      proj_dq = jnp.stack(proj_list, axis=0)
    else:
      proj_dq = jnp.clip(_f16_round_jax(jnp.clip(arr, 1e-6, 10.0)), 1e-6, 10.0)

  def_dc = jnp.array(K_DC_QUANT, dtype=jnp.float32)
  if dc_quant is None:
    return proj_dq, def_dc

  dc_arr = jnp.asarray(dc_quant, dtype=jnp.float32).reshape(3)
  is_def_dc = jnp.max(jnp.abs(dc_arr - def_dc) / def_dc) < 1e-6
  dc_clipped = jnp.clip(dc_arr, 1e-6, 1.0)
  dc_proj = jnp.where(
      is_def_dc,
      def_dc,
      jnp.clip(_f16_round_jax(dc_clipped * 128.0) / 128.0, 1e-6, 1.0),
  )
  return proj_dq, dc_proj


def _ste_exact(x: jax.Array, proj: jax.Array) -> jax.Array:
  """Exact forward STE (`proj + (x - stop_gradient(x))`) avoiding float32 cancellation when `|x| >> |proj|`."""
  return jax.lax.stop_gradient(proj) + (x - jax.lax.stop_gradient(x))


def ste_dequant_matrices(
    dequant_matrices: Any,
    dc_quant: jax.Array | np.ndarray | Sequence[float] | None = None,
    mode: str = "dct",
    hard: bool = False,
) -> Any:
  """Applies Straight-Through Estimator (STE) projection to custom `dequant_matrices` and optional `dc_quant`."""
  if dc_quant is None:
    proj_dq, _ = project_custom_dequant_matrices(
        dequant_matrices, dc_quant=None, mode=mode
    )
    if hard:
      return proj_dq
    if isinstance(dequant_matrices, Mapping):
      return {
          int(s): _ste_exact(
              jnp.asarray(v, dtype=jnp.float32), proj_dq[int(s)]
          )
          for s, v in dequant_matrices.items()
      }
    arr = jnp.asarray(dequant_matrices, dtype=jnp.float32)
    return _ste_exact(arr, proj_dq)

  proj_dq, proj_dc = project_custom_dequant_matrices(
      dequant_matrices, dc_quant=dc_quant, mode=mode
  )
  dc_arr = jnp.asarray(dc_quant, dtype=jnp.float32).reshape(3)
  if hard:
    return proj_dq, proj_dc
  ste_dc = _ste_exact(dc_arr, proj_dc)
  if isinstance(dequant_matrices, Mapping):
    ste_dq: Any = {
        int(s): _ste_exact(
            jnp.asarray(v, dtype=jnp.float32), proj_dq[int(s)]
        )
        for s, v in dequant_matrices.items()
    }
  else:
    arr = jnp.asarray(dequant_matrices, dtype=jnp.float32)
    ste_dq = _ste_exact(arr, proj_dq)
  return ste_dq, ste_dc


def _resolve_strategy_dequant_matrix_jax(
    s: int,
    custom_dq: Any,
    fallback_mat: jax.Array,
) -> jax.Array:
  """Resolves a differentiable `(3, 8*cy_l, 8*cx_l)` dequant matrix for strategy `s`."""
  if custom_dq is None:
    return fallback_mat

  cx = ACS_COVERED_BLOCKS_X[s]
  cy = ACS_COVERED_BLOCKS_Y[s]
  cy_l, cx_l = min(cy, cx), max(cy, cx)
  target_shape = (3, 8 * cy_l, 8 * cx_l)
  def_mat = jnp.asarray(get_default_dequant_matrices()[s], dtype=jnp.float32)

  raw_entry = None
  if isinstance(custom_dq, Mapping):
    if s in custom_dq:
      raw_entry = custom_dq[s]
    else:
      canon_s = K_QUANT_TABLE_TO_CANONICAL_STRATEGY[
          K_AC_STRATEGY_TO_QUANT_TABLE_MAP[s]
      ]
      if canon_s in custom_dq:
        raw_entry = custom_dq[canon_s]
  else:
    arr = jnp.asarray(custom_dq, dtype=jnp.float32)
    if arr.ndim == 1 and arr.shape == (3,):
      return def_mat * arr[:, None, None]
    if arr.ndim == 2 and arr.shape[1] == 3 and s < arr.shape[0]:
      return def_mat * arr[s, :, None, None]
    if arr.ndim >= 3 and s < arr.shape[0]:
      raw_entry = arr[s]
    elif arr.ndim >= 3:
      canon_s = K_QUANT_TABLE_TO_CANONICAL_STRATEGY[
          K_AC_STRATEGY_TO_QUANT_TABLE_MAP[s]
      ]
      if canon_s < arr.shape[0]:
        raw_entry = arr[canon_s]

  if raw_entry is None:
    return fallback_mat

  entry_arr = jnp.asarray(raw_entry, dtype=jnp.float32)
  if entry_arr.shape == target_shape:
    return entry_arr
  if entry_arr.size == 3 * (8 * cy_l) * (8 * cx_l):
    return entry_arr.reshape(target_shape)
  if entry_arr.size == 3 * 64:
    mat_8x8 = entry_arr.reshape(3, 8, 8)
    if target_shape == (3, 8, 8):
      return mat_8x8
    def_8x8 = jax.image.resize(def_mat, (3, 8, 8), method="linear")
    ratio_8x8 = mat_8x8 / jnp.maximum(def_8x8, 1e-12)
    ratio_full = jax.image.resize(ratio_8x8, target_shape, method="linear")
    return def_mat * ratio_full
  if entry_arr.shape == (3,):
    return def_mat * entry_arr[:, None, None]
  raise ValueError(
      f"Unsupported custom dequant matrix shape {entry_arr.shape} for strategy {s} "
      f"(expected {target_shape} or (3, 64) or (3, 8, 8))"
  )


# -----------------------------------------------------------------------------
# 3. Quantization Bias, DC Smoothing, Gaborish, and EPF Stages
# -----------------------------------------------------------------------------


def adjust_quant_bias_hard(q_int: jax.Array) -> jax.Array:
  """Exact libjxl AdjustQuantBias on (3, ...) integer/rounded AC quantized values."""
  q = jnp.asarray(q_int, dtype=jnp.float32)
  biases = jnp.array(K_DEFAULT_QUANT_BIAS[:3], dtype=jnp.float32).reshape(
      (3,) + (1,) * (q.ndim - 1)
  )
  k_bias_num = jnp.float32(K_DEFAULT_QUANT_BIAS[3])
  abs_q = jnp.abs(q)
  safe_q = jnp.where(q == 0.0, 1.0, q)
  return jnp.where(
      q == 0.0,
      0.0,
      jnp.where(
          abs_q == 1.0,
          jnp.sign(q) * biases,
          q - k_bias_num / safe_q,
      ),
  )


def adjust_quant_bias_ste(
    q: jax.Array, hard: bool = False, ste_strength: float = 0.25
) -> jax.Array:
  """Applies rounding + AdjustQuantBias with Straight-Through Estimator (STE) when hard=False.

  Uses partial STE (`ste_strength = 0.25`) on the fractional rounding residual
  `round(q) - q` and full STE on the integer quantization bias shift
  `adjust_quant_bias_hard(round(q)) - round(q)`, so that:
  - At every integer `q`, the forward value equals `adjust_quant_bias_hard(q)`
    exactly.
  - Inside each quantization bin `(k - 0.5, k + 0.5)`, the forward slope is
    `1 - ste_strength = 0.75 > 0`, providing stable interior stationary points
    at high step counts (1000..50000) without boundary limit-cycle chatter.
  - The backward derivative is identically `1.0` everywhere.
  """
  q_f = jnp.asarray(q, dtype=jnp.float32)
  q_round = jnp.round(q_f)
  q_hat_hard = adjust_quant_bias_hard(q_round)
  if hard:
    return q_hat_hard
  return (
      q_f
      + jnp.float32(ste_strength) * jax.lax.stop_gradient(q_round - q_f)
      + jax.lax.stop_gradient(q_hat_hard - q_round)
  )


dequant_ac_coeff = adjust_quant_bias_ste


def adaptive_dc_smoothing(dc: jax.Array, mul_dc: jax.Array | Sequence[float]) -> jax.Array:
  """Exact AdaptiveDCSmoothing (lib/jxl/compressed_dc.cc:49-198) on (3, H8, W8) DC grid."""
  dc_arr = jnp.asarray(dc, dtype=jnp.float32)
  mul_arr = jnp.asarray(mul_dc, dtype=jnp.float32).reshape(3, 1, 1)
  _, h8, w8 = dc_arr.shape
  if h8 <= 2 or w8 <= 2:
    return dc_arr

  w1 = jnp.float32(0.20345139757231578)
  w2 = jnp.float32(0.0334829185968739)
  w0 = jnp.float32(1.0 - 4.0 * (0.20345139757231578 + 0.0334829185968739))

  top = dc_arr[:, :-2, :]
  center = dc_arr[:, 1:-1, :]
  bottom = dc_arr[:, 2:, :]

  sm_top = w2 * (top[:, :, :-2] + top[:, :, 2:]) + w1 * top[:, :, 1:-1]
  sm_center = w1 * (center[:, :, :-2] + center[:, :, 2:]) + w0 * center[:, :, 1:-1]
  sm_bottom = w2 * (bottom[:, :, :-2] + bottom[:, :, 2:]) + w1 * bottom[:, :, 1:-1]
  sm = sm_top + sm_center + sm_bottom

  dc_int = dc_arr[:, 1:-1, 1:-1]
  gap_c = jnp.abs(dc_int - sm) / mul_arr
  gap = jnp.maximum(0.5, jnp.max(gap_c, axis=0, keepdims=True))
  factor = jnp.maximum(0.0, 3.0 - 4.0 * gap)
  dc_int_out = dc_int + factor * (sm - dc_int)
  return dc_arr.at[:, 1:-1, 1:-1].set(dc_int_out)


def dequant_and_smooth_dc(
    dc_quant: jax.Array,
    global_scale: int | float | jax.Array,
    quant_dc: int | float | jax.Array,
    base_correlation_x: float = K_BASE_CORRELATION_X,
    base_correlation_b: float = K_BASE_CORRELATION_B,
    ytox_dc: int = 0,
    ytob_dc: int = 0,
    color_factor: int = K_DEFAULT_COLOR_FACTOR,
    hard: bool = False,
    do_adaptive_dc_smoothing: bool = True,
    dc_quant_weights: jax.Array | Sequence[float] | None = None,
) -> jax.Array:
  """Exact DequantDC + AdaptiveDCSmoothing (compressed_dc.cc:49-231)."""
  dc_q_f = jnp.asarray(dc_quant, dtype=jnp.float32)
  q_dc = (
      jnp.round(dc_q_f)
      if hard
      else (dc_q_f + jax.lax.stop_gradient(jnp.round(dc_q_f) - dc_q_f))
  )
  gs_f = jnp.asarray(global_scale, dtype=jnp.float32)
  qdc_f = jnp.asarray(quant_dc, dtype=jnp.float32)
  inv_global_scale = K_GLOBAL_SCALE_DENOM / gs_f
  inv_quant_dc = inv_global_scale / qdc_f
  k_dc = (
      jnp.asarray(dc_quant_weights, dtype=jnp.float32).reshape(3)
      if dc_quant_weights is not None
      else jnp.array(K_DC_QUANT, dtype=jnp.float32)
  )
  mul_dc = inv_quant_dc * k_dc

  cfl_x = float(base_correlation_x) + jnp.asarray(ytox_dc, dtype=jnp.float32) / float(color_factor)
  cfl_b = float(base_correlation_b) + jnp.asarray(ytob_dc, dtype=jnp.float32) / float(color_factor)

  dc_y = q_dc[1] * mul_dc[1]
  dc_x = q_dc[0] * mul_dc[0] + cfl_x * dc_y
  dc_b = q_dc[2] * mul_dc[2] + cfl_b * dc_y
  dc = jnp.stack([dc_x, dc_y, dc_b], axis=0)

  if not do_adaptive_dc_smoothing:
    return dc
  return adaptive_dc_smoothing(dc, mul_dc)


def gaborish_3x3(
    xyb: jax.Array,
    gab_weights: Sequence[float] = K_DEFAULT_GAB_WEIGHTS,
) -> jax.Array:
  """Exact 3x3 Gaborish convolution with Mirror/symmetric padding (stage_gaborish.cc:33-99)."""
  arr = jnp.asarray(xyb, dtype=jnp.float32)
  padded = jnp.pad(arr, ((0, 0), (1, 1), (1, 1)), mode="symmetric")
  outs = []
  for c in range(3):
    w1 = float(gab_weights[2 * c])
    w2 = float(gab_weights[2 * c + 1])
    norm = 1.0 / (1.0 + 4.0 * (w1 + w2))
    wc = norm
    wt = w1 * norm
    wd = w2 * norm
    p = padded[c]
    center = p[1:-1, 1:-1]
    axial = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    diag = p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
    outs.append(wc * center + wt * axial + wd * diag)
  return jnp.stack(outs, axis=0)


def gaborish_inverse_3x3(
    xyb: jax.Array,
    gab_weights: Sequence[float] = K_DEFAULT_GAB_WEIGHTS,
) -> jax.Array:
  """Exact 3x3 inverse Gaborish sharpening filter (lib/jxl/enc_gaborish.cc:28-79)."""
  arr = jnp.asarray(xyb, dtype=jnp.float32)
  padded = jnp.pad(arr, ((0, 0), (1, 1), (1, 1)), mode="symmetric")
  outs = []
  for c in range(3):
    w1 = float(gab_weights[2 * c])
    w2 = float(gab_weights[2 * c + 1])
    mul = 1.0 / (1.0 + 4.0 * (w1 + w2))
    p = padded[c]
    center = p[1:-1, 1:-1]
    axial = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    diag = p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
    wc = 1.0 + 4.0 * (w1 + w2) + 4.0 * (w1 * w1 + w2 * w2) * mul
    wt = -w1
    wd = -w2
    norm = 1.0 / (wc + 4.0 * (wt + wd))
    outs.append((wc * center + wt * axial + wd * diag) * norm)
  return jnp.stack(outs, axis=0)


def compute_epf_inv_sigma_grid(
    raw_quant_tl_grid: jax.Array,
    epf_sharpness: jax.Array,
    global_scale: int | float | jax.Array,
    epf_quant_mul: float = K_DEFAULT_EPF_QUANT_MUL,
    epf_sharp_lut: Sequence[float] = K_DEFAULT_EPF_SHARP_LUT,
) -> jax.Array:
  """Exact ComputeSigma (lib/jxl/epf.cc:39-132)."""
  quant_scale = jnp.asarray(global_scale, dtype=jnp.float32) / K_GLOBAL_SCALE_DENOM
  sigma_quant = float(epf_quant_mul) / (
      quant_scale * jnp.asarray(raw_quant_tl_grid, dtype=jnp.float32) * K_INV_SIGMA_NUM
  )
  lut = jnp.array(epf_sharp_lut, dtype=jnp.float32)
  sharp_val = lut[jnp.asarray(epf_sharpness, dtype=jnp.int32)]
  sigma = jnp.minimum(-1e-4, sigma_quant * sharp_val)
  return 1.0 / sigma


compute_epf_inv_sigma = compute_epf_inv_sigma_grid


def _epf_pixel_inv_sigma(
    inv_sigma_grid: jax.Array,
    h: int,
    w: int,
    sigma_scale: float,
    border_sad_mul: float,
) -> tuple[jax.Array, jax.Array]:
  """Computes per-pixel effective inv_sigma (stage_epf.cc:98-131, 237-271, 428-462)."""
  y_idx = jnp.arange(h)
  x_idx = jnp.arange(w)
  ymod = (y_idx % 8)[:, None]
  xmod = (x_idx % 8)[None, :]

  h8, w8 = inv_sigma_grid.shape
  row_sigma_px = jnp.broadcast_to(
      inv_sigma_grid[:, None, :, None], (h8, 8, w8, 8)
  ).reshape(h8 * 8, w8 * 8)[:h, :w]
  sm = float(sigma_scale) * 1.65
  bsm = sm * float(border_sad_mul)
  is_border = (ymod == 0) | (ymod == 7) | (xmod == 0) | (xmod == 7)
  vsm = jnp.where(is_border, jnp.float32(bsm), jnp.float32(sm))
  return row_sigma_px * vsm, row_sigma_px


@functools.partial(jax.checkpoint, prevent_cse=False, static_argnums=(2, 3, 4))
def _epf_stage0_ckpt(
    arr: jax.Array,
    inv_sigma_grid: jax.Array,
    pass0_sigma_scale: float,
    border_sad_mul: float,
    channel_scale: tuple[float, ...],
) -> jax.Array:
  _, h, w = arr.shape
  inv_sigma_eff, row_sigma_px = _epf_pixel_inv_sigma(
      inv_sigma_grid, h, w, pass0_sigma_scale, border_sad_mul
  )
  ch_scale = jnp.array(channel_scale, dtype=jnp.float32)[:, None, None]

  pad = jnp.pad(arr, ((0, 0), (3, 3), (3, 3)), mode="symmetric")

  def shift(dy: int, dx: int) -> jax.Array:
    return pad[:, 3 + dy : 3 + dy + h, 3 + dx : 3 + dx + w]

  plus_offsets = ((0, 0), (-1, 0), (0, -1), (1, 0), (0, 1))
  neighbor_offsets = (
      (-2, 0), (-1, -1), (-1, 0), (-1, 1),
      (0, -2), (0, -1), (0, 1), (0, 2),
      (1, -1), (1, 0), (1, 1), (2, 0),
  )

  w_sum = jnp.ones((1, h, w), dtype=jnp.float32)
  x_sum = arr.copy()

  for dy, dx in neighbor_offsets:
    sad = jnp.zeros((h, w), dtype=jnp.float32)
    for pdy, pdx in plus_offsets:
      diff = jnp.abs(shift(pdy, pdx) - shift(dy + pdy, dx + pdx))
      sad = sad + jnp.sum(diff * ch_scale, axis=0)
    weight = jnp.maximum(0.0, 1.0 + sad * inv_sigma_eff)[None, :, :]
    w_sum = w_sum + weight
    x_sum = x_sum + weight * shift(dy, dx)

  filtered = x_sum / w_sum
  return jnp.where((row_sigma_px >= K_MIN_SIGMA)[None, :, :], filtered, arr)


def epf_stage0(
    xyb: jax.Array,
    inv_sigma_grid: jax.Array,
    pass0_sigma_scale: float = K_DEFAULT_EPF_PASS0_SIGMA_SCALE,
    border_sad_mul: float = K_DEFAULT_EPF_BORDER_SAD_MUL,
    channel_scale: Sequence[float] = K_DEFAULT_EPF_CHANNEL_SCALE,
) -> jax.Array:
  """Exact EPF0Stage (lib/jxl/render_pipeline/stage_epf.cc:54-193)."""
  return _epf_stage0_ckpt(
      jnp.asarray(xyb, dtype=jnp.float32),
      jnp.asarray(inv_sigma_grid, dtype=jnp.float32),
      float(pass0_sigma_scale),
      float(border_sad_mul),
      tuple(float(x) for x in channel_scale),
  )


@functools.partial(jax.checkpoint, prevent_cse=False, static_argnums=(2, 3))
def _epf_stage1_ckpt(
    arr: jax.Array,
    inv_sigma_grid: jax.Array,
    border_sad_mul: float,
    channel_scale: tuple[float, ...],
) -> jax.Array:
  _, h, w = arr.shape
  inv_sigma_eff, row_sigma_px = _epf_pixel_inv_sigma(
      inv_sigma_grid, h, w, 1.0, border_sad_mul
  )
  ch_scale = jnp.array(channel_scale, dtype=jnp.float32)[:, None, None]

  pad = jnp.pad(arr, ((0, 0), (2, 2), (2, 2)), mode="symmetric")

  def shift(dy: int, dx: int) -> jax.Array:
    return pad[:, 2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w]

  plus_offsets = ((0, 0), (-1, 0), (0, -1), (1, 0), (0, 1))
  neighbor_offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))

  w_sum = jnp.ones((1, h, w), dtype=jnp.float32)
  x_sum = arr.copy()

  for dy, dx in neighbor_offsets:
    sad = jnp.zeros((h, w), dtype=jnp.float32)
    for pdy, pdx in plus_offsets:
      diff = jnp.abs(shift(pdy, pdx) - shift(dy + pdy, dx + pdx))
      sad = sad + jnp.sum(diff * ch_scale, axis=0)
    weight = jnp.maximum(0.0, 1.0 + sad * inv_sigma_eff)[None, :, :]
    w_sum = w_sum + weight
    x_sum = x_sum + weight * shift(dy, dx)

  filtered = x_sum / w_sum
  return jnp.where((row_sigma_px >= K_MIN_SIGMA)[None, :, :], filtered, arr)


def epf_stage1(
    xyb: jax.Array,
    inv_sigma_grid: jax.Array,
    border_sad_mul: float = K_DEFAULT_EPF_BORDER_SAD_MUL,
    channel_scale: Sequence[float] = K_DEFAULT_EPF_CHANNEL_SCALE,
) -> jax.Array:
  """Exact EPF1Stage (lib/jxl/render_pipeline/stage_epf.cc:197-379)."""
  return _epf_stage1_ckpt(
      jnp.asarray(xyb, dtype=jnp.float32),
      jnp.asarray(inv_sigma_grid, dtype=jnp.float32),
      float(border_sad_mul),
      tuple(float(x) for x in channel_scale),
  )


@functools.partial(jax.checkpoint, prevent_cse=False, static_argnums=(2, 3, 4))
def _epf_stage2_ckpt(
    arr: jax.Array,
    inv_sigma_grid: jax.Array,
    pass2_sigma_scale: float,
    border_sad_mul: float,
    channel_scale: tuple[float, ...],
) -> jax.Array:
  _, h, w = arr.shape
  inv_sigma_eff, row_sigma_px = _epf_pixel_inv_sigma(
      inv_sigma_grid, h, w, pass2_sigma_scale, border_sad_mul
  )
  ch_scale = jnp.array(channel_scale, dtype=jnp.float32)[:, None, None]

  pad = jnp.pad(arr, ((0, 0), (1, 1), (1, 1)), mode="symmetric")

  def shift(dy: int, dx: int) -> jax.Array:
    return pad[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]

  neighbor_offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))

  w_sum = jnp.ones((1, h, w), dtype=jnp.float32)
  x_sum = arr.copy()

  for dy, dx in neighbor_offsets:
    sad = jnp.sum(jnp.abs(arr - shift(dy, dx)) * ch_scale, axis=0)
    weight = jnp.maximum(0.0, 1.0 + sad * inv_sigma_eff)[None, :, :]
    w_sum = w_sum + weight
    x_sum = x_sum + weight * shift(dy, dx)

  filtered = x_sum / w_sum
  return jnp.where((row_sigma_px >= K_MIN_SIGMA)[None, :, :], filtered, arr)


def epf_stage2(
    xyb: jax.Array,
    inv_sigma_grid: jax.Array,
    pass2_sigma_scale: float = K_DEFAULT_EPF_PASS2_SIGMA_SCALE,
    border_sad_mul: float = K_DEFAULT_EPF_BORDER_SAD_MUL,
    channel_scale: Sequence[float] = K_DEFAULT_EPF_CHANNEL_SCALE,
) -> jax.Array:
  """Exact EPF2Stage (lib/jxl/render_pipeline/stage_epf.cc:383-506)."""
  return _epf_stage2_ckpt(
      jnp.asarray(xyb, dtype=jnp.float32),
      jnp.asarray(inv_sigma_grid, dtype=jnp.float32),
      float(pass2_sigma_scale),
      float(border_sad_mul),
      tuple(float(x) for x in channel_scale),
  )


# -----------------------------------------------------------------------------
# 4. Static VarDCT Topology & Single-Frame Reconstruction
# -----------------------------------------------------------------------------


class StaticVarDCTTopology:
  """Precomputed static topology from JxlVarDCTState for fast JIT compilation."""

  def __init__(
      self,
      state: JxlVarDCTState,
      force_strategies: Sequence[int] | None = None,
  ) -> None:
    self.state = state
    self.height = int(state.height)
    self.width = int(state.width)
    self.h8 = int(state.h8)
    self.w8 = int(state.w8)
    self.ht = int(state.ht)
    self.wt = int(state.wt)
    self.global_scale = int(state.global_scale)
    self.quant_dc = int(state.quant_dc)
    self.base_correlation_x = float(state.base_correlation_x)
    self.base_correlation_b = float(state.base_correlation_b)
    self.ytox_dc = int(state.ytox_dc)
    self.ytob_dc = int(state.ytob_dc)
    self.color_factor = int(state.color_factor)
    self.x_qm_scale = int(state.x_qm_scale)
    self.b_qm_scale = int(state.b_qm_scale)
    self.epf_iters = int(state.epf_iters)
    self.epf_quant_mul = float(state.epf_quant_mul)
    self.epf_pass0_sigma_scale = float(state.epf_pass0_sigma_scale)
    self.epf_pass2_sigma_scale = float(state.epf_pass2_sigma_scale)
    self.epf_border_sad_mul = float(state.epf_border_sad_mul)
    self.epf_sharp_lut = tuple(float(x) for x in state.epf_sharp_lut)
    self.epf_channel_scale = tuple(float(x) for x in state.epf_channel_scale)
    self.gab = bool(state.gab)
    self.gab_weights = tuple(float(x) for x in state.gab_weights)
    self.opsin_inv_matrix = jnp.array(K_DEFAULT_INVERSE_OPSIN_MATRIX, dtype=jnp.float32)
    self.opsin_biases = K_NEG_OPSIN_ABSORBANCE_BIAS_RGB
    self.epf_sharpness = jnp.array(state.epf_sharpness, dtype=jnp.int32)

    default_dq = get_default_dequant_matrices()
    dq_dict = state.dequant_matrices if state.dequant_matrices else default_dq

    acs_np = np.asarray(state.ac_strategy, dtype=np.int32)
    tl_by_grid = np.broadcast_to(
        np.arange(self.h8, dtype=np.int32)[:, None], (self.h8, self.w8)
    ).copy()
    tl_bx_grid = np.broadcast_to(
        np.arange(self.w8, dtype=np.int32)[None, :], (self.h8, self.w8)
    ).copy()
    is_1x1_np = (acs_np <= 3) | ((acs_np >= 12) & (acs_np <= 17))
    mask_1x1_np = is_1x1_np.astype(np.float32)
    first_block_mask_np = mask_1x1_np.copy()

    blocks_multi: dict[int, list[tuple[int, int]]] = {}
    if np.any(~is_1x1_np):
      covered_m = np.zeros((self.h8, self.w8), dtype=bool)
      m_bys, m_bxs = np.nonzero(~is_1x1_np)
      for by, bx in zip(m_bys.tolist(), m_bxs.tolist()):
        if covered_m[by, bx]:
          continue
        s = int(acs_np[by, bx])
        cx = ACS_COVERED_BLOCKS_X[s]
        cy = ACS_COVERED_BLOCKS_Y[s]
        covered_m[by : by + cy, bx : bx + cx] = True
        tl_by_grid[by : by + cy, bx : bx + cx] = by
        tl_bx_grid[by : by + cy, bx : bx + cx] = bx
        first_block_mask_np[by, bx] = 1.0
        blocks_multi.setdefault(s, []).append((by, bx))

    self.tl_by_grid_np = tl_by_grid
    self.tl_bx_grid_np = tl_bx_grid
    self.tl_by_grid = jnp.array(tl_by_grid, dtype=jnp.int32)
    self.tl_bx_grid = jnp.array(tl_bx_grid, dtype=jnp.int32)
    self.mask_1x1 = jnp.array(mask_1x1_np, dtype=jnp.float32)
    self.first_block_mask = jnp.array(first_block_mask_np, dtype=jnp.float32)

    by_1x1, bx_1x1 = np.nonzero(is_1x1_np)
    by_1x1 = by_1x1.astype(np.int32)
    bx_1x1 = bx_1x1.astype(np.int32)
    s_1x1 = acs_np[by_1x1, bx_1x1]
    self.has_1x1 = bool(by_1x1.size > 0)
    self._dq_dict = dq_dict
    self._default_dq = default_dq
    self._dq_1x1_lazy: jax.Array | None = None
    self._basis_1x1_lazy: jax.Array | None = None
    self._fwd_basis_1x1_lazy: jax.Array | None = None
    self.groups_1x1: list[dict[str, Any]] = []
    if self.has_1x1:
      self._s_1x1_np = s_1x1
      self.by_1x1 = jnp.array(by_1x1)
      self.bx_1x1 = jnp.array(bx_1x1)
      self.s_1x1 = jnp.array(s_1x1)
      used_1x1_s = set(int(x) for x in np.unique(s_1x1))
    else:
      used_1x1_s = set()

    if force_strategies is not None:
      for s_f in sorted(set(int(x) for x in force_strategies)):
        if ACS_COVERED_BLOCKS_X[s_f] == 1 and ACS_COVERED_BLOCKS_Y[s_f] == 1:
          used_1x1_s.add(s_f)
        else:
          blocks_multi.setdefault(s_f, [])
      self.has_1x1 = len(used_1x1_s) > 0

    if self.has_1x1:
      for s_v in sorted(used_1x1_s):
        mask_2d_np = (acs_np == int(s_v)) & is_1x1_np
        by_s, bx_s = np.nonzero(mask_2d_np)
        by_s = by_s.astype(np.int32)
        bx_s = bx_s.astype(np.int32)
        self.groups_1x1.append({
            "s": int(s_v),
            "by": jnp.array(by_s, dtype=jnp.int32),
            "bx": jnp.array(bx_s, dtype=jnp.int32),
            "mask_2d": jnp.array(mask_2d_np, dtype=jnp.bool_),
            "dq_mat": jnp.array(
                dq_dict.get(int(s_v), default_dq[int(s_v)]), dtype=jnp.float32
            ),
            "basis": jnp.array(
                BASIS_1X1_INV_TABLE_NP[int(s_v)], dtype=jnp.float32
            ),
            "fwd_basis": jnp.array(
                BASIS_1X1_FWD_TABLE_NP[int(s_v)], dtype=jnp.float32
            ),
        })

    self.multi_groups: list[dict[str, Any]] = []
    for s, coords in sorted(blocks_multi.items()):
      name = ACS_NAMES[s]
      cx = ACS_COVERED_BLOCKS_X[s]
      cy = ACS_COVERED_BLOCKS_Y[s]
      nh_cell = self.h8 // cy
      nw_cell = self.w8 // cx
      if len(coords) == 0 and (nh_cell <= 0 or nw_cell <= 0):
        continue
      cy_l, cx_l = min(cy, cx), max(cy, cx)
      rows, cols = 8 * cy, 8 * cx
      by_np = np.array([c[0] for c in coords], dtype=np.int32)
      bx_np = np.array([c[1] for c in coords], dtype=np.int32)
      by_arr = jnp.array(by_np, dtype=jnp.int32)
      bx_arr = jnp.array(bx_np, dtype=jnp.int32)
      dq_mat = jnp.array(dq_dict.get(int(s), default_dq[int(s)]), dtype=jnp.float32)

      grid_aligned = bool(
          nh_cell > 0
          and nw_cell > 0
          and (
              by_np.size == 0
              or (
                  np.all((by_np % cy) == 0)
                  and np.all((bx_np % cx) == 0)
                  and np.all((by_np // cy) < nh_cell)
                  and np.all((bx_np // cx) < nw_cell)
              )
          )
      )
      if grid_aligned:
        cell_mask_np = np.zeros((nh_cell, nw_cell), dtype=bool)
        if by_np.size > 0:
          cell_mask_np[by_np // cy, bx_np // cx] = True
        blk_mask_np = np.repeat(np.repeat(cell_mask_np, cy, axis=0), cx, axis=1)
        cell_mask_j = jnp.array(cell_mask_np, dtype=jnp.bool_)
        cell_mask_f = jnp.array(cell_mask_np, dtype=jnp.float32)
        blk_mask_j = jnp.array(blk_mask_np, dtype=jnp.bool_)
      else:
        cell_mask_j = None
        cell_mask_f = None
        blk_mask_j = None

      self.multi_groups.append({
          "s": s,
          "name": name,
          "cy": cy,
          "cx": cx,
          "cy_l": cy_l,
          "cx_l": cx_l,
          "by": by_arr,
          "bx": bx_arr,
          "grid_aligned": grid_aligned,
          "nh_cell": nh_cell,
          "nw_cell": nw_cell,
          "h_lim": nh_cell * cy,
          "w_lim": nw_cell * cx,
          "cell_mask": cell_mask_j,
          "cell_mask_f": cell_mask_f,
          "blk_mask": blk_mask_j,
          "dq_mat": dq_mat,
          "d_cy": jnp.array(make_dct1d_matrix(cy), dtype=jnp.float32),
          "d_cx": jnp.array(make_dct1d_matrix(cx), dtype=jnp.float32),
          "idct_cy": jnp.array(make_idct1d_matrix(cy), dtype=jnp.float32),
          "idct_cx": jnp.array(make_idct1d_matrix(cx), dtype=jnp.float32),
          "rs_to_llf_y": jnp.array(dct_resample_scales(cy, 8 * cy), dtype=jnp.float32),
          "rs_to_llf_x": jnp.array(dct_resample_scales(cx, 8 * cx), dtype=jnp.float32),
          "rs_from_llf_y": jnp.array(dct_resample_scales(8 * cy, cy), dtype=jnp.float32),
          "rs_from_llf_x": jnp.array(dct_resample_scales(8 * cx, cx), dtype=jnp.float32),
          "dct_r": jnp.array(make_dct1d_matrix(rows), dtype=jnp.float32),
          "dct_c": jnp.array(make_dct1d_matrix(cols), dtype=jnp.float32),
          "idct_r": jnp.array(make_idct1d_matrix(rows), dtype=jnp.float32),
          "idct_c": jnp.array(make_idct1d_matrix(cols), dtype=jnp.float32),
      })
    self.all_grid_aligned = all(
        bool(g.get("grid_aligned", False)) for g in self.multi_groups
    )

  def broadcast_top_left_grid(
      self, grid_2d: jax.Array, acs_grid: jax.Array | None = None
  ) -> jax.Array:
    """Broadcasts top-left anchor values of each multi-block transform across its covered `(cy, cx)` blocks without 2D gather when grid-aligned."""
    if len(self.multi_groups) == 0:
      return grid_2d
    if not self.all_grid_aligned:
      return grid_2d[self.tl_by_grid, self.tl_bx_grid]
    out = grid_2d
    for grp in self.multi_groups:
      cy, cx = grp["cy"], grp["cx"]
      nh_cell, nw_cell = grp["nh_cell"], grp["nw_cell"]
      h_lim, w_lim = grp["h_lim"], grp["w_lim"]
      tl_vals = grid_2d[:h_lim:cy, :w_lim:cx]
      tl_bcast = jnp.broadcast_to(
          tl_vals[:, None, :, None], (nh_cell, cy, nw_cell, cx)
      ).reshape(h_lim, w_lim)
      blk_mask = (
          (acs_grid[:h_lim, :w_lim] == grp["s"])
          if acs_grid is not None
          else grp["blk_mask"]
      )
      sub = jnp.where(blk_mask, tl_bcast, out[:h_lim, :w_lim])
      if h_lim == self.h8 and w_lim == self.w8:
        out = sub
      else:
        out = out.at[:h_lim, :w_lim].set(sub)
    return out

  @property
  def dq_1x1(self) -> jax.Array:
    if self._dq_1x1_lazy is None:
      dq_np = np.stack(
          [
              self._dq_dict.get(int(s), self._default_dq[int(s)])
              for s in self._s_1x1_np
          ],
          axis=0,
      )
      self._dq_1x1_lazy = jnp.array(dq_np, dtype=jnp.float32)
    return self._dq_1x1_lazy

  @property
  def basis_1x1(self) -> jax.Array:
    if self._basis_1x1_lazy is None:
      self._basis_1x1_lazy = jnp.array(
          BASIS_1X1_INV_TABLE_NP[self._s_1x1_np], dtype=jnp.float32
      )
    return self._basis_1x1_lazy

  @property
  def fwd_basis_1x1(self) -> jax.Array:
    if self._fwd_basis_1x1_lazy is None:
      self._fwd_basis_1x1_lazy = jnp.array(
          BASIS_1X1_FWD_TABLE_NP[self._s_1x1_np], dtype=jnp.float32
      )
    return self._fwd_basis_1x1_lazy


def reconstruct_single_frame_xyb(
    state_or_topo: JxlVarDCTState | StaticVarDCTTopology,
    quant_ac: jax.Array | None = None,
    dc_quant: jax.Array | None = None,
    raw_quant_field: jax.Array | None = None,
    ytox_map: jax.Array | None = None,
    ytob_map: jax.Array | None = None,
    params_override: Mapping[str, Any] | None = None,
    hard: bool = False,
    skip_adaptive_dc_smoothing: bool = False,
    include_noise: bool = False,
    noise_lut: jax.Array | Sequence[float] | None = None,
    visible_frame_index: int = 1,
    nonvisible_frame_index: int = 0,
    ste_quantize_noise_lut: bool = True,
    dequant_matrices: Any = None,
    dc_quant_weights: jax.Array | Sequence[float] | None = None,
    epf_iters_override: int | None = None,
    gab_override: bool | None = None,
    enable_noise: bool | None = None,
    acs_grid: jax.Array | None = None,
) -> jax.Array:
  """Reconstructs a single VarDCT frame into XYB `(3, H, W)` after Gaborish, EPF, and optional Noise."""
  if isinstance(state_or_topo, StaticVarDCTTopology):
    topo = state_or_topo
    state = getattr(topo, "state", None)
  else:
    state = state_or_topo
    topo = StaticVarDCTTopology(state)

  ov = params_override or {}
  q_ac_in = (
      quant_ac
      if quant_ac is not None
      else ov.get("quant_ac", state.quant_ac if state is not None else None)
  )

  # Disambiguate 3D quantized DC coefficient grid (3, H8, W8) vs 1D DC quant step weights (3,)
  dc_grid_in: Any = None
  dc_weights_in: Any = dc_quant_weights
  if dc_quant is not None:
    dc_arg_arr = jnp.asarray(dc_quant, dtype=jnp.float32)
    if dc_arg_arr.ndim == 1 and dc_arg_arr.size == 3:
      if dc_weights_in is None:
        dc_weights_in = dc_arg_arr
    else:
      dc_grid_in = dc_arg_arr

  if "dc_quant" in ov and ov["dc_quant"] is not None:
    ov_dc_arr = jnp.asarray(ov["dc_quant"], dtype=jnp.float32)
    if ov_dc_arr.ndim == 1 and ov_dc_arr.size == 3:
      if dc_weights_in is None:
        dc_weights_in = ov_dc_arr
    elif dc_grid_in is None:
      dc_grid_in = ov_dc_arr

  if dc_grid_in is None:
    dc_grid_in = ov.get(
        "dc_quant_grid", state.dc_quant if state is not None else None
    )
  if dc_weights_in is None:
    dc_weights_in = ov.get(
        "dc_quant_weights",
        getattr(state, "dc_quant_weights", None) if state is not None else None,
    )

  raw_q_in = (
      raw_quant_field
      if raw_quant_field is not None
      else ov.get("raw_quant_field", state.raw_quant_field if state is not None else None)
  )
  ytox_in = (
      ytox_map
      if ytox_map is not None
      else ov.get("ytox_map", state.ytox_map if state is not None else None)
  )
  ytob_in = (
      ytob_map
      if ytob_map is not None
      else ov.get("ytob_map", state.ytob_map if state is not None else None)
  )

  if any(x is None for x in (q_ac_in, dc_grid_in, raw_q_in, ytox_in, ytob_in)):
    raise ValueError(
        "When passing StaticVarDCTTopology without JxlVarDCTState, all 5 tensors "
        "(quant_ac, dc_quant, raw_quant_field, ytox_map, ytob_map) must be provided."
    )

  q_ac_arr = jnp.asarray(q_ac_in, dtype=jnp.float32)
  dc_q_arr = jnp.asarray(dc_grid_in, dtype=jnp.float32)
  raw_q_arr = jnp.asarray(raw_q_in, dtype=jnp.float32)
  ytox_arr = jnp.asarray(ytox_in, dtype=jnp.float32)
  ytob_arr = jnp.asarray(ytob_in, dtype=jnp.float32)

  gs_val = jnp.asarray(ov.get("global_scale", topo.global_scale), dtype=jnp.float32)
  qdc_val = jnp.asarray(ov.get("quant_dc", topo.quant_dc), dtype=jnp.float32)
  x_qm_val = jnp.asarray(ov.get("x_qm_scale", topo.x_qm_scale), dtype=jnp.float32)
  b_qm_val = jnp.asarray(ov.get("b_qm_scale", topo.b_qm_scale), dtype=jnp.float32)
  custom_dq = (
      dequant_matrices
      if dequant_matrices is not None
      else ov.get("dequant_matrices", None)
  )

  dc_smoothed = dequant_and_smooth_dc(
      dc_quant=dc_q_arr,
      global_scale=gs_val,
      quant_dc=qdc_val,
      base_correlation_x=topo.base_correlation_x,
      base_correlation_b=topo.base_correlation_b,
      ytox_dc=ov.get("ytox_dc", topo.ytox_dc),
      ytob_dc=ov.get("ytob_dc", topo.ytob_dc),
      color_factor=topo.color_factor,
      hard=hard,
      do_adaptive_dc_smoothing=not skip_adaptive_dc_smoothing,
      dc_quant_weights=dc_weights_in,
  )

  q_hat_all = adjust_quant_bias_ste(q_ac_arr, hard=hard)
  q_hat_blocks = jnp.transpose(
      q_hat_all.reshape(3, topo.h8, 8, topo.w8, 8), (1, 3, 0, 2, 4)
  )

  raw_q_f = (
      jnp.round(raw_q_arr)
      if hard
      else (raw_q_arr + jax.lax.stop_gradient(jnp.round(raw_q_arr) - raw_q_arr))
  )
  raw_q_f = jnp.maximum(raw_q_f, 1.0)
  scaled_dequant_grid = (K_GLOBAL_SCALE_DENOM / gs_val) / raw_q_f

  x_dm_mul = jnp.power(jnp.float32(1.0 / 1.25), x_qm_val - 2.0)
  b_dm_mul = jnp.power(jnp.float32(1.0 / 1.25), b_qm_val - 2.0)

  ytox_f = (
      jnp.round(ytox_arr)
      if hard
      else (ytox_arr + jax.lax.stop_gradient(jnp.round(ytox_arr) - ytox_arr))
  )
  ytob_f = (
      jnp.round(ytob_arr)
      if hard
      else (ytob_arr + jax.lax.stop_gradient(jnp.round(ytob_arr) - ytob_arr))
  )
  x_cc_tile = float(topo.base_correlation_x) + ytox_f / float(topo.color_factor)
  b_cc_tile = float(topo.base_correlation_b) + ytob_f / float(topo.color_factor)
  x_cc_grid = jnp.broadcast_to(
      x_cc_tile[:, None, :, None], (topo.ht, 8, topo.wt, 8)
  ).reshape(topo.ht * 8, topo.wt * 8)[: topo.h8, : topo.w8]
  b_cc_grid = jnp.broadcast_to(
      b_cc_tile[:, None, :, None], (topo.ht, 8, topo.wt, 8)
  ).reshape(topo.ht * 8, topo.wt * 8)[: topo.h8, : topo.w8]

  xyb_blocks = jnp.zeros((topo.h8, topo.w8, 3, 8, 8), dtype=jnp.float32)

  if topo.has_1x1:
    q_b_all = q_hat_blocks.at[:, :, :, 0, 0].set(0.0)
    sd_all = scaled_dequant_grid[:, :, None, None]
    x_cc_all = x_cc_grid[:, :, None, None]
    b_cc_all = b_cc_grid[:, :, None, None]
    dc_vals_all = jnp.transpose(dc_smoothed, (1, 2, 0))
    for idx_1x1, grp1 in enumerate(topo.groups_1x1):
      s_id = grp1["s"]
      dq_mat = (
          _resolve_strategy_dequant_matrix_jax(s_id, custom_dq, grp1["dq_mat"])
          if custom_dq is not None
          else grp1["dq_mat"]
      )
      wy = dq_mat[1][None, None]
      wx = dq_mat[0][None, None]
      wb = dq_mat[2][None, None]

      coeff_y = q_b_all[:, :, 1] * wy * sd_all
      coeff_x = q_b_all[:, :, 0] * wx * (sd_all * x_dm_mul) + x_cc_all * coeff_y
      coeff_b = q_b_all[:, :, 2] * wb * (sd_all * b_dm_mul) + b_cc_all * coeff_y
      coeffs = jnp.stack([coeff_x, coeff_y, coeff_b], axis=2)
      coeffs = coeffs.at[:, :, :, 0, 0].set(dc_vals_all)

      coeffs_flat = coeffs.reshape(topo.h8, topo.w8, 3, 64)
      pixels_flat = jnp.einsum("ij,hwcj->hwci", grp1["basis"], coeffs_flat)
      pixels_1x1 = pixels_flat.reshape(topo.h8, topo.w8, 3, 8, 8)
      if idx_1x1 == 0:
        xyb_blocks = pixels_1x1
      else:
        mask_2d = (acs_grid == s_id) if acs_grid is not None else grp1["mask_2d"]
        xyb_blocks = jnp.where(
            mask_2d[:, :, None, None, None], pixels_1x1, xyb_blocks
        )

  for grp in topo.multi_groups:
    cy = grp["cy"]
    cx = grp["cx"]
    cy_l = grp["cy_l"]
    cx_l = grp["cx_l"]
    dq_mat = (
        _resolve_strategy_dequant_matrix_jax(grp["s"], custom_dq, grp["dq_mat"])
        if custom_dq is not None
        else grp["dq_mat"]
    )
    if grp.get("grid_aligned", False):
      nh_cell = grp["nh_cell"]
      nw_cell = grp["nw_cell"]
      h_lim = grp["h_lim"]
      w_lim = grp["w_lim"]
      sub_q = q_hat_blocks[:h_lim, :w_lim].reshape(
          nh_cell, cy, nw_cell, cx, 3, 8, 8
      )
      sub_q = jnp.transpose(sub_q, (0, 2, 4, 1, 5, 3, 6)).reshape(
          nh_cell, nw_cell, 3, 8 * cy_l, 8 * cx_l
      )
      sub_q = sub_q.at[:, :, :, :cy_l, :cx_l].set(0.0)

      sd = scaled_dequant_grid[:h_lim:cy, :w_lim:cx][:, :, None, None]
      x_cc = x_cc_grid[:h_lim:cy, :w_lim:cx][:, :, None, None]
      b_cc = b_cc_grid[:h_lim:cy, :w_lim:cx][:, :, None, None]

      coeff_y = sub_q[:, :, 1] * dq_mat[1][None, None] * sd
      coeff_x = (
          sub_q[:, :, 0] * dq_mat[0][None, None] * (sd * x_dm_mul)
          + x_cc * coeff_y
      )
      coeff_b = (
          sub_q[:, :, 2] * dq_mat[2][None, None] * (sd * b_dm_mul)
          + b_cc * coeff_y
      )
      coeffs = jnp.stack([coeff_x, coeff_y, coeff_b], axis=2)

      dc_sub = jnp.transpose(
          dc_smoothed[:, :h_lim, :w_lim].reshape(3, nh_cell, cy, nw_cell, cx),
          (1, 3, 0, 2, 4),
      )
      llf = jnp.einsum("ij,hwcjk,lk->hwcil", grp["d_cy"], dc_sub, grp["d_cx"])
      llf = llf * (
          grp["rs_to_llf_y"][None, None, None, :, None]
          * grp["rs_to_llf_x"][None, None, None, None, :]
      )
      if cy >= cx:
        llf = jnp.swapaxes(llf, -2, -1)
      coeffs = coeffs.at[:, :, :, :cy_l, :cx_l].set(llf)

      coeffs_for_idct = jnp.swapaxes(coeffs, -2, -1) if cy >= cx else coeffs
      pixels = jnp.einsum(
          "ij,hwcjk,lk->hwcil", grp["idct_r"], coeffs_for_idct, grp["idct_c"]
      )
      pixels_blk = jnp.transpose(
          pixels.reshape(nh_cell, nw_cell, 3, cy, 8, cx, 8),
          (0, 3, 1, 5, 2, 4, 6),
      ).reshape(h_lim, w_lim, 3, 8, 8)
      blk_mask = (
          (acs_grid[:h_lim, :w_lim] == grp["s"])
          if acs_grid is not None
          else grp["blk_mask"]
      )
      sub_xyb = jnp.where(
          blk_mask[:, :, None, None, None],
          pixels_blk,
          xyb_blocks[:h_lim, :w_lim],
      )
      if h_lim == topo.h8 and w_lim == topo.w8:
        xyb_blocks = sub_xyb
      else:
        xyb_blocks = xyb_blocks.at[:h_lim, :w_lim].set(sub_xyb)
    else:
      by = grp["by"]
      bx = grp["bx"]
      m = by.shape[0]
      dy = jnp.arange(cy)[None, :, None]
      dx = jnp.arange(cx)[None, None, :]
      sub_q = q_hat_blocks[by[:, None, None] + dy, bx[:, None, None] + dx]
      sub_q = jnp.transpose(sub_q, (0, 3, 1, 4, 2, 5)).reshape(
          m, 3, 8 * cy_l, 8 * cx_l
      )
      sub_q = sub_q.at[:, :, :cy_l, :cx_l].set(0.0)

      sd = scaled_dequant_grid[by, bx][:, None, None]
      ty = by // 8
      tx = bx // 8
      x_cc = x_cc_tile[ty, tx][:, None, None]
      b_cc = b_cc_tile[ty, tx][:, None, None]

      coeff_y = sub_q[:, 1] * dq_mat[1][None] * sd
      coeff_x = sub_q[:, 0] * dq_mat[0][None] * (sd * x_dm_mul) + x_cc * coeff_y
      coeff_b = sub_q[:, 2] * dq_mat[2][None] * (sd * b_dm_mul) + b_cc * coeff_y
      coeffs = jnp.stack([coeff_x, coeff_y, coeff_b], axis=1)

      dc_sub = jnp.transpose(
          dc_smoothed[:, by[:, None, None] + dy, bx[:, None, None] + dx],
          (1, 0, 2, 3),
      )
      llf = jnp.einsum("ij,mcjk,lk->mcil", grp["d_cy"], dc_sub, grp["d_cx"])
      llf = llf * (
          grp["rs_to_llf_y"][None, None, :, None]
          * grp["rs_to_llf_x"][None, None, None, :]
      )
      if cy >= cx:
        llf = jnp.swapaxes(llf, -2, -1)
      coeffs = coeffs.at[:, :, :cy_l, :cx_l].set(llf)

      coeffs_for_idct = jnp.swapaxes(coeffs, -2, -1) if cy >= cx else coeffs
      pixels = jnp.einsum(
          "ij,mcjk,lk->mcil", grp["idct_r"], coeffs_for_idct, grp["idct_c"]
      )
      pixels_blk = jnp.transpose(
          pixels.reshape(m, 3, cy, 8, cx, 8), (0, 2, 4, 1, 3, 5)
      )
      xyb_blocks = xyb_blocks.at[
          by[:, None, None] + dy, bx[:, None, None] + dx
      ].set(pixels_blk)

  xyb = jnp.transpose(xyb_blocks, (2, 0, 3, 1, 4)).reshape(3, topo.h8 * 8, topo.w8 * 8)
  # Crop to unpadded frame size BEFORE Gaborish and EPF so mirror padding reflects at (H, W)
  xyb = xyb[:, : topo.height, : topo.width]

  use_gab = (
      bool(gab_override)
      if gab_override is not None
      else bool(ov.get("gab", topo.gab))
  )
  if use_gab:
    xyb = gaborish_3x3(xyb, topo.gab_weights)

  epf_iters = (
      int(epf_iters_override)
      if epf_iters_override is not None
      else int(ov.get("epf_iters", topo.epf_iters))
  )
  if epf_iters > 0:
    epf_sharp_val = jnp.asarray(
        ov.get("epf_sharpness", topo.epf_sharpness), dtype=jnp.int32
    )
    raw_q_tl_grid = topo.broadcast_top_left_grid(raw_q_f, acs_grid=acs_grid)
    inv_sigma_grid = compute_epf_inv_sigma_grid(
        raw_quant_tl_grid=raw_q_tl_grid,
        epf_sharpness=epf_sharp_val,
        global_scale=gs_val,
        epf_quant_mul=topo.epf_quant_mul,
        epf_sharp_lut=topo.epf_sharp_lut,
    )
    if epf_iters >= 3:
      xyb = epf_stage0(
          xyb=xyb,
          inv_sigma_grid=inv_sigma_grid,
          pass0_sigma_scale=topo.epf_pass0_sigma_scale,
          border_sad_mul=topo.epf_border_sad_mul,
          channel_scale=topo.epf_channel_scale,
      )
    if epf_iters >= 1:
      xyb = epf_stage1(
          xyb=xyb,
          inv_sigma_grid=inv_sigma_grid,
          border_sad_mul=topo.epf_border_sad_mul,
          channel_scale=topo.epf_channel_scale,
      )
    if epf_iters >= 2:
      xyb = epf_stage2(
          xyb=xyb,
          inv_sigma_grid=inv_sigma_grid,
          pass2_sigma_scale=topo.epf_pass2_sigma_scale,
          border_sad_mul=topo.epf_border_sad_mul,
          channel_scale=topo.epf_channel_scale,
      )

  lut_to_use = noise_lut if noise_lut is not None else ov.get("noise_lut", None)
  do_noise = (
      bool(enable_noise)
      if enable_noise is not None
      else bool(ov.get("enable_noise", include_noise))
  )
  if do_noise and lut_to_use is not None:
    xyb = synthesize_jxl_noise(
        xyb=xyb,
        noise_lut=lut_to_use,
        ytox=topo.base_correlation_x,
        ytob=topo.base_correlation_b,
        visible_frame_index=visible_frame_index,
        nonvisible_frame_index=nonvisible_frame_index,
        ste_quantize_lut=ste_quantize_noise_lut,
    )

  return xyb


def reconstruct_single_frame(
    state_or_topo: JxlVarDCTState | StaticVarDCTTopology,
    params_override: Mapping[str, Any] | None = None,
    include_noise: bool = False,
    noise_lut: jax.Array | Sequence[float] | None = None,
    quant_ac: jax.Array | None = None,
    dc_quant: jax.Array | None = None,
    raw_quant_field: jax.Array | None = None,
    ytox_map: jax.Array | None = None,
    ytob_map: jax.Array | None = None,
    hard: bool = False,
    skip_adaptive_dc_smoothing: bool = False,
    visible_frame_index: int = 1,
    nonvisible_frame_index: int = 0,
    ste_quantize_noise_lut: bool = True,
    output_layout: str = "HWC",
    dequant_matrices: Any = None,
    dc_quant_weights: jax.Array | Sequence[float] | None = None,
    epf_iters_override: int | None = None,
    gab_override: bool | None = None,
    enable_noise: bool | None = None,
    acs_grid: jax.Array | None = None,
) -> jax.Array:
  """Full differentiable single-frame VarDCT reconstruction in JAX -> nonlinear sRGB in [0, 1].

  Returns `(H, W, 3)` when `output_layout="HWC"` (default) or `(3, H, W)` when `"CHW"`.
  """
  topo = (
      state_or_topo
      if isinstance(state_or_topo, StaticVarDCTTopology)
      else StaticVarDCTTopology(state_or_topo)
  )
  xyb = reconstruct_single_frame_xyb(
      state_or_topo=state_or_topo,
      quant_ac=quant_ac,
      dc_quant=dc_quant,
      raw_quant_field=raw_quant_field,
      ytox_map=ytox_map,
      ytob_map=ytob_map,
      params_override=params_override,
      hard=hard,
      skip_adaptive_dc_smoothing=skip_adaptive_dc_smoothing,
      include_noise=include_noise,
      noise_lut=noise_lut,
      visible_frame_index=visible_frame_index,
      nonvisible_frame_index=nonvisible_frame_index,
      ste_quantize_noise_lut=ste_quantize_noise_lut,
      dequant_matrices=dequant_matrices,
      dc_quant_weights=dc_quant_weights,
      epf_iters_override=epf_iters_override,
      gab_override=gab_override,
      enable_noise=enable_noise,
      acs_grid=acs_grid,
  )
  srgb_chw = opsin_to_srgb(xyb, topo.opsin_inv_matrix, topo.opsin_biases, clamp=True)
  if output_layout.upper() == "CHW":
    return srgb_chw
  return jnp.moveaxis(srgb_chw, 0, -1)


# -----------------------------------------------------------------------------
# 5. Pure-JAX Forward Analysis from Raw sRGB [0, 1] (Zero libjxl Choices)
# -----------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _get_compiled_dct8_init(
    h: int, w: int, gab: bool, x_qm_scale: int = 3, b_qm_scale: int = 2
):
  """Returns JIT-compiled DCT8 (dc_quant, quant_ac) initializer for `init_state_from_rgb_jax`."""
  h8 = (h + 7) // 8
  w8 = (w + 7) // 8
  ht = (h8 + 7) // 8
  wt = (w8 + 7) // 8
  default_dq0 = jnp.asarray(get_default_dequant_matrices()[0], dtype=jnp.float32)
  fwd_basis0 = jnp.asarray(BASIS_1X1_FWD_TABLE_NP[0], dtype=jnp.float32)
  wy = default_dq0[1][None, None]
  wx = default_dq0[0][None, None]
  wb = default_dq0[2][None, None]
  x_qm_mul = jnp.float32(1.25 ** (float(x_qm_scale) - 2.0))
  b_qm_mul = jnp.float32(1.25 ** (float(b_qm_scale) - 2.0))
  k_dc = jnp.asarray(K_DC_QUANT, dtype=jnp.float32)

  @jax.jit
  def _fn(arr_hwc, gs, qdc, rq, yx, yb, ytox_dc_f, ytob_dc_f):
    srgb_chw = jnp.moveaxis(arr_hwc, -1, 0)
    xyb = srgb_to_xyb(srgb_chw)
    if gab:
      xyb = gaborish_inverse_3x3(xyb, K_DEFAULT_GAB_WEIGHTS)
    pad_h = h8 * 8 - h
    pad_w = w8 * 8 - w
    if pad_h > 0 or pad_w > 0:
      xyb = jnp.pad(xyb, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    px_flat = jnp.transpose(xyb.reshape(3, h8, 8, w8, 8), (1, 3, 0, 2, 4)).reshape(
        h8, w8, 3, 64
    )
    coeffs = jnp.einsum("ij,hwcj->hwci", fwd_basis0, px_flat).reshape(
        h8, w8, 3, 8, 8
    )
    dc_continuous = jnp.transpose(coeffs[:, :, :, 0, 0], (2, 0, 1))
    qs_all = ((gs / K_GLOBAL_SCALE_DENOM) * jnp.maximum(rq, 1.0))[:, :, None, None]
    x_cc_tile = float(K_BASE_CORRELATION_X) + yx / float(K_DEFAULT_COLOR_FACTOR)
    b_cc_tile = float(K_BASE_CORRELATION_B) + yb / float(K_DEFAULT_COLOR_FACTOR)
    x_cc_all = jnp.broadcast_to(
        x_cc_tile[:, None, :, None], (ht, 8, wt, 8)
    ).reshape(ht * 8, wt * 8)[:h8, :w8, None, None]
    b_cc_all = jnp.broadcast_to(
        b_cc_tile[:, None, :, None], (ht, 8, wt, 8)
    ).reshape(ht * 8, wt * 8)[:h8, :w8, None, None]
    cy_val = coeffs[:, :, 1]
    cx_val = coeffs[:, :, 0] - x_cc_all * cy_val
    cb_val = coeffs[:, :, 2] - b_cc_all * cy_val
    qy = jnp.round(cy_val * (qs_all / wy))
    qx = jnp.round(cx_val * (qs_all * x_qm_mul / wx))
    qb = jnp.round(cb_val * (qs_all * b_qm_mul / wb))
    q_blk = jnp.stack([qx, qy, qb], axis=2).at[:, :, :, 0, 0].set(0.0)
    quant_ac = jnp.transpose(q_blk, (2, 0, 3, 1, 4)).reshape(
        3, h8 * 8, w8 * 8
    ).astype(jnp.int32)
    inv_gs = K_GLOBAL_SCALE_DENOM / gs
    mul_dc = (inv_gs / qdc) * k_dc
    cfl_dc_x = float(K_BASE_CORRELATION_X) + ytox_dc_f / float(K_DEFAULT_COLOR_FACTOR)
    cfl_dc_b = float(K_BASE_CORRELATION_B) + ytob_dc_f / float(K_DEFAULT_COLOR_FACTOR)
    q_dc_y = jnp.round(dc_continuous[1] / mul_dc[1])
    q_dc_x = jnp.round(
        (dc_continuous[0] - q_dc_y * (mul_dc[1] * cfl_dc_x))
        / mul_dc[0]
    )
    q_dc_b = jnp.round(
        (dc_continuous[2] - q_dc_y * (mul_dc[1] * cfl_dc_b))
        / mul_dc[2]
    )
    dc_quant = jnp.stack([q_dc_x, q_dc_y, q_dc_b], axis=0).astype(jnp.int32)
    return dc_quant, quant_ac

  return _fn


def init_state_from_rgb_jax(
    srgb_hwc: np.ndarray | jax.Array,
    distance: float = 1.0,
    ac_strategy: np.ndarray | None = None,
    raw_quant_field: np.ndarray | None = None,
    epf_sharpness: np.ndarray | None = None,
    ytox_map: np.ndarray | None = None,
    ytob_map: np.ndarray | None = None,
    epf_iters: int = 1,
    gab: bool = True,
    *,
    ac_anchors: np.ndarray | None = None,
    ytox_dc: int = 0,
    ytob_dc: int = 0,
) -> JxlVarDCTState:
  """Constructs a complete JxlVarDCTState from raw RGB [0, 1] 100% in JAX/Python.

  Never invokes cjxl or --dump-init on the target image; uses analytical
  distance scaling calibrated to libjxl's Quantizer::ComputeGlobalScaleAndQuant /
  InitialQuantDC curve and pure-JAX SRGBToXYB + inverse Gaborish + forward DCT.
  """
  arr = np.asarray(srgb_hwc, dtype=np.float32)
  if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
    arr = np.moveaxis(arr, 0, -1)
  if arr.max() > 1.5:
    arr = arr / 255.0
  h, w, _ = arr.shape
  h8 = (h + 7) // 8
  w8 = (w + 7) // 8
  ht = (h8 + 7) // 8
  wt = (w8 + 7) // 8

  d = max(float(distance), 0.05)
  # Smooth analytical calibration matching libjxl's InitialQuantDC(d) and
  # Quantizer::ComputeGlobalScaleAndQuant (kQuantFieldTarget = 5..6, zero power-law explosion).
  bt_dc = max(0.5 * d, min(d, 0.3 * ((d / 0.3) ** 0.83)))
  qdc_float = min(1.095924047623553 / bt_dc, 50.0)
  q_med = 0.390 / d
  global_scale = int(np.clip(round(K_GLOBAL_SCALE_DENOM * q_med / 5.0), 1, 32768))
  scaled_qdc = max(1, int(qdc_float * 4096.0 * 1.6))
  if global_scale > scaled_qdc:
    global_scale = scaled_qdc
  inv_global_scale = K_GLOBAL_SCALE_DENOM / float(global_scale)
  quant_dc = int(np.clip(round(qdc_float * inv_global_scale), 4, 65535))
  gab_iqf_mul = (1.0 / 0.62) if not gab else 1.0
  default_raw_q = int(
      np.clip(round((0.47 / d) * gab_iqf_mul * inv_global_scale), 1, 256)
  )

  x_qm_scale = 3 if d <= 2.5 else 2
  b_qm_scale = 2
  color_factor = K_DEFAULT_COLOR_FACTOR
  base_correlation_x = K_BASE_CORRELATION_X
  base_correlation_b = K_BASE_CORRELATION_B
  ytox_dc = int(np.clip(int(ytox_dc), -128, 127))
  ytob_dc = int(np.clip(int(ytob_dc), -128, 127))

  if ac_strategy is None:
    ac_strategy = np.zeros((h8, w8), dtype=np.uint8)
  if raw_quant_field is None:
    raw_quant_field = np.full((h8, w8), default_raw_q, dtype=np.int32)
  if epf_sharpness is None:
    epf_sharpness = np.full((h8, w8), 2 if epf_iters > 0 else 0, dtype=np.uint8)
  if ytox_map is None:
    ytox_map = np.zeros((ht, wt), dtype=np.int8)
  if ytob_map is None:
    ytob_map = np.zeros((ht, wt), dtype=np.int8)

  default_dq = get_default_dequant_matrices()
  used_acs = 0
  for s in np.unique(ac_strategy):
    used_acs |= 1 << int(s)

  state = JxlVarDCTState(
      height=h,
      width=w,
      h8=h8,
      w8=w8,
      ht=ht,
      wt=wt,
      global_scale=global_scale,
      quant_dc=quant_dc,
      x_qm_scale=x_qm_scale,
      b_qm_scale=b_qm_scale,
      color_factor=color_factor,
      base_correlation_x=base_correlation_x,
      base_correlation_b=base_correlation_b,
      ytox_dc=ytox_dc,
      ytob_dc=ytob_dc,
      epf_iters=epf_iters,
      epf_quant_mul=K_DEFAULT_EPF_QUANT_MUL,
      epf_pass0_sigma_scale=K_DEFAULT_EPF_PASS0_SIGMA_SCALE,
      epf_pass2_sigma_scale=K_DEFAULT_EPF_PASS2_SIGMA_SCALE,
      epf_border_sad_mul=K_DEFAULT_EPF_BORDER_SAD_MUL,
      epf_channel_scale=K_DEFAULT_EPF_CHANNEL_SCALE,
      epf_sharp_lut=K_DEFAULT_EPF_SHARP_LUT,
      gab=gab,
      gab_weights=K_DEFAULT_GAB_WEIGHTS,
      used_acs=used_acs,
      ac_strategy=np.asarray(ac_strategy, dtype=np.uint8),
      raw_quant_field=np.asarray(raw_quant_field, dtype=np.int32),
      epf_sharpness=np.asarray(epf_sharpness, dtype=np.uint8),
      ytox_map=np.asarray(ytox_map, dtype=np.int8),
      ytob_map=np.asarray(ytob_map, dtype=np.int8),
      dc_quant=np.zeros((3, h8, w8), dtype=np.int32),
      quant_ac=np.zeros((3, h8 * 8, w8 * 8), dtype=np.int32),
      dequant_matrices={k: v.copy() for k, v in default_dq.items()},
  )
  if ac_anchors is not None:
    state.ac_strategy_Anchor = np.asarray(ac_anchors, dtype=bool)
  if float(arr.max()) == 0.0 and float(arr.min()) == 0.0:
    return state
  if used_acs == 1:
    dc_j, ac_j = _get_compiled_dct8_init(
        h, w, bool(gab), int(x_qm_scale), int(b_qm_scale)
    )(
        jnp.asarray(arr, dtype=jnp.float32),
        jnp.float32(global_scale),
        jnp.float32(quant_dc),
        jnp.asarray(state.raw_quant_field, dtype=jnp.float32),
        jnp.asarray(state.ytox_map, dtype=jnp.float32),
        jnp.asarray(state.ytob_map, dtype=jnp.float32),
        jnp.float32(ytox_dc),
        jnp.float32(ytob_dc),
    )
    state.dc_quant = np.asarray(dc_j, dtype=np.int32)
    state.quant_ac = np.asarray(ac_j, dtype=np.int32)
    return state

  topo = StaticVarDCTTopology(state)
  srgb_chw = jnp.array(np.transpose(arr, (2, 0, 1)), dtype=jnp.float32)
  xyb = srgb_to_xyb(srgb_chw)
  if gab:
    xyb = gaborish_inverse_3x3(xyb, state.gab_weights)

  pad_h = h8 * 8 - h
  pad_w = w8 * 8 - w
  if pad_h > 0 or pad_w > 0:
    xyb = jnp.pad(xyb, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

  xyb_blocks = jnp.transpose(xyb.reshape(3, h8, 8, w8, 8), (1, 3, 0, 2, 4))

  raw_q_f = jnp.maximum(jnp.array(state.raw_quant_field, dtype=jnp.float32), 1.0)
  qac_scale_grid = (float(global_scale) / K_GLOBAL_SCALE_DENOM) * raw_q_f
  x_qm_mul = float(1.25 ** (x_qm_scale - 2.0))
  b_qm_mul = float(1.25 ** (b_qm_scale - 2.0))

  x_cc_tile = (
      float(base_correlation_x)
      + jnp.array(state.ytox_map, dtype=jnp.float32) / float(color_factor)
  )
  b_cc_tile = (
      float(base_correlation_b)
      + jnp.array(state.ytob_map, dtype=jnp.float32) / float(color_factor)
  )
  x_cc_grid = jnp.broadcast_to(
      x_cc_tile[:, None, :, None], (ht, 8, wt, 8)
  ).reshape(ht * 8, wt * 8)[:h8, :w8]
  b_cc_grid = jnp.broadcast_to(
      b_cc_tile[:, None, :, None], (ht, 8, wt, 8)
  ).reshape(ht * 8, wt * 8)[:h8, :w8]

  dc_continuous = jnp.zeros((3, h8, w8), dtype=jnp.float32)
  q_ac_blocks = jnp.zeros((h8, w8, 3, 8, 8), dtype=jnp.float32)

  if topo.has_1x1:
    px_flat = xyb_blocks.reshape(h8, w8, 3, 64)
    qs_all = qac_scale_grid[:, :, None, None]
    x_cc_all = x_cc_grid[:, :, None, None]
    b_cc_all = b_cc_grid[:, :, None, None]
    for idx_1x1, grp1 in enumerate(topo.groups_1x1):
      coeffs = jnp.einsum("ij,hwcj->hwci", grp1["fwd_basis"], px_flat).reshape(
          h8, w8, 3, 8, 8
      )
      dc_s = jnp.transpose(coeffs[:, :, :, 0, 0], (2, 0, 1))
      dq_mat = grp1["dq_mat"]
      wy = dq_mat[1][None, None]
      wx = dq_mat[0][None, None]
      wb = dq_mat[2][None, None]

      cy_val = coeffs[:, :, 1]
      cx_val = coeffs[:, :, 0] - x_cc_all * cy_val
      cb_val = coeffs[:, :, 2] - b_cc_all * cy_val

      qy = jnp.round(cy_val * (qs_all / wy))
      qx = jnp.round(cx_val * (qs_all * x_qm_mul / wx))
      qb = jnp.round(cb_val * (qs_all * b_qm_mul / wb))
      q_blk = jnp.stack([qx, qy, qb], axis=2).at[:, :, :, 0, 0].set(0.0)
      if idx_1x1 == 0:
        dc_continuous = dc_s
        q_ac_blocks = q_blk
      else:
        m2d = grp1["mask_2d"]
        dc_continuous = jnp.where(m2d[None, :, :], dc_s, dc_continuous)
        q_ac_blocks = jnp.where(m2d[:, :, None, None, None], q_blk, q_ac_blocks)

  for grp in topo.multi_groups:
    cy = grp["cy"]
    cx = grp["cx"]
    cy_l = grp["cy_l"]
    cx_l = grp["cx_l"]
    by = grp["by"]
    bx = grp["bx"]
    m = by.shape[0]
    dy = jnp.arange(cy)[None, :, None]
    dx = jnp.arange(cx)[None, None, :]

    px_sub = xyb_blocks[by[:, None, None] + dy, bx[:, None, None] + dx]
    px_full = jnp.transpose(px_sub, (0, 3, 1, 4, 2, 5)).reshape(m, 3, 8 * cy, 8 * cx)
    coeffs_rc = jnp.einsum("ij,mcjk,lk->mcil", grp["dct_r"], px_full, grp["dct_c"])
    coeffs = jnp.swapaxes(coeffs_rc, -2, -1) if cy >= cx else coeffs_rc

    llf = coeffs[:, :, :cy_l, :cx_l]
    llf_rc = jnp.swapaxes(llf, -2, -1) if cy >= cx else llf
    llf_scaled = llf_rc * (
        grp["rs_from_llf_y"][None, None, :, None] * grp["rs_from_llf_x"][None, None, None, :]
    )
    dc_sub = jnp.einsum("ij,mcjk,lk->mcil", grp["idct_cy"], llf_scaled, grp["idct_cx"])
    dc_continuous = dc_continuous.at[:, by[:, None, None] + dy, bx[:, None, None] + dx].set(
        jnp.transpose(dc_sub, (1, 0, 2, 3))
    )

    qs = qac_scale_grid[by, bx][:, None, None]
    ty = by // 8
    tx = bx // 8
    x_cc = x_cc_tile[ty, tx][:, None, None]
    b_cc = b_cc_tile[ty, tx][:, None, None]

    dq_mat = grp["dq_mat"]
    cy_val = coeffs[:, 1]
    cx_val = coeffs[:, 0] - x_cc * cy_val
    cb_val = coeffs[:, 2] - b_cc * cy_val

    qy = jnp.round(cy_val * (qs / dq_mat[1][None]))
    qx = jnp.round(cx_val * (qs * x_qm_mul / dq_mat[0][None]))
    qb = jnp.round(cb_val * (qs * b_qm_mul / dq_mat[2][None]))
    q_canon = jnp.stack([qx, qy, qb], axis=1).at[:, :, :cy_l, :cx_l].set(0.0)
    q_sub = jnp.transpose(q_canon.reshape(m, 3, cy, 8, cx, 8), (0, 2, 4, 1, 3, 5))
    q_ac_blocks = q_ac_blocks.at[by[:, None, None] + dy, bx[:, None, None] + dx].set(q_sub)

  mul_dc = [inv_global_scale / float(quant_dc) * k for k in K_DC_QUANT]
  cfl_dc_x = float(base_correlation_x) + float(ytox_dc) / float(color_factor)
  cfl_dc_b = float(base_correlation_b) + float(ytob_dc) / float(color_factor)

  q_dc_y = jnp.round(dc_continuous[1] / mul_dc[1])
  q_dc_x = jnp.round((dc_continuous[0] - q_dc_y * (mul_dc[1] * cfl_dc_x)) / mul_dc[0])
  q_dc_b = jnp.round((dc_continuous[2] - q_dc_y * (mul_dc[1] * cfl_dc_b)) / mul_dc[2])
  state.dc_quant = np.asarray(jnp.stack([q_dc_x, q_dc_y, q_dc_b], axis=0), dtype=np.int32)
  state.quant_ac = np.asarray(
      jnp.transpose(q_ac_blocks, (2, 0, 3, 1, 4)).reshape(3, h8 * 8, w8 * 8),
      dtype=np.int32,
  )
  return state


# -----------------------------------------------------------------------------
# 6. Multi-Resolution 5x5 Upsampling, XYB Patch Compositing & Alpha Blending
# -----------------------------------------------------------------------------

_WEIGHTS_2: tuple[float, ...] = (
    -0.01716200, -0.03452303, -0.04022174, -0.02921014, -0.00624645,
    0.14111091,  0.28896755,  0.00278718,  -0.01610267, 0.56661550,
    0.03777607,  -0.01986694, -0.03144731, -0.01185068, -0.00213539,
)

_WEIGHTS_4: tuple[float, ...] = (
    -0.02419067, -0.03491987, -0.03693351, -0.03094285, -0.00529785,
    -0.01663432, -0.03556863, -0.03888905, -0.03516850, -0.00989469,
    0.23651958,  0.33392945,  -0.01073543, -0.01313181, -0.03556694,
    0.13048175,  0.40103025,  0.03951150,  -0.02077584, 0.46914198,
    -0.00209270, -0.01484589, -0.04064806, 0.18942530,  0.56279892,
    0.06674400,  -0.02335494, -0.03551682, -0.00754830, -0.02267919,
    -0.02363578, 0.00315804,  -0.03399098, -0.01359519, -0.00091653,
    -0.00335467, -0.01163294, -0.01610294, -0.00974088, -0.00191622,
    -0.01095446, -0.03198464, -0.04455121, -0.02799790, -0.00645912,
    0.06390599,  0.22963888,  0.00630981,  -0.01897349, 0.67537268,
    0.08483369,  -0.02534994, -0.02205197, -0.01667999, -0.00384443,
)

_WEIGHTS_8: tuple[float, ...] = (
    -0.02928613, -0.03706353, -0.03783812, -0.03324558, -0.00447632,
    -0.02519406, -0.03752601, -0.03901508, -0.03663285, -0.00646649,
    -0.02066407, -0.03838633, -0.04002101, -0.03900035, -0.00901973,
    -0.01626393, -0.03954148, -0.04046620, -0.03979621, -0.01224485,
    0.29895328,  0.35757708,  -0.02447552, -0.01081748, -0.04314594,
    0.23903219,  0.41119301,  -0.00573046, -0.01450239, -0.04246845,
    0.17567618,  0.45220643,  0.02287757,  -0.01936783, -0.03583255,
    0.11572472,  0.47416733,  0.06284440,  -0.02685066, 0.42720050,
    -0.02248939, -0.01155273, -0.04562755, 0.28689496,  0.49093869,
    -0.00007891, -0.01545926, -0.04562659, 0.21238920,  0.53980934,
    0.03369474,  -0.02070211, -0.03866988, 0.14229550,  0.56593398,
    0.08045181,  -0.02888298, -0.03680918, -0.00542229, -0.02920477,
    -0.02788574, -0.02118180, -0.03942402, -0.00775547, -0.02433614,
    -0.03193943, -0.02030828, -0.04044014, -0.01074016, -0.01930822,
    -0.03620399, -0.01974125, -0.03919545, -0.01456093, -0.00045072,
    -0.00360110, -0.01020207, -0.01231907, -0.00638988, -0.00071592,
    -0.00279122, -0.00957115, -0.01288327, -0.00730937, -0.00107783,
    -0.00210156, -0.00890705, -0.01317668, -0.00813895, -0.00153491,
    -0.02128481, -0.04173044, -0.04831487, -0.03293190, -0.00525260,
    -0.01720322, -0.04052736, -0.05045706, -0.03607317, -0.00738030,
    -0.01341764, -0.03965629, -0.05151616, -0.03814886, -0.01005819,
    0.18968273,  0.33063684,  -0.01300105, -0.01372950, -0.04017465,
    0.13727832,  0.36402234,  0.01027890,  -0.01832107, -0.03365072,
    0.08734506,  0.38194295,  0.04338228,  -0.02525993, 0.56408126,
    0.00458352,  -0.01648227, -0.04887868, 0.24585519,  0.62026135,
    0.04314807,  -0.02213737, -0.04158014, 0.16637289,  0.65027023,
    0.09621636,  -0.03101388, -0.04082742, -0.00904519, -0.02790922,
    -0.02117818, 0.00798662,  -0.03995711, -0.01243427, -0.02231705,
    -0.02946266, 0.00992055,  -0.03600283, -0.01684920, -0.00111684,
    -0.00411204, -0.01297130, -0.01723725, -0.01022545, -0.00165306,
    -0.00313110, -0.01218016, -0.01763266, -0.01125620, -0.00231663,
    -0.01374149, -0.03797620, -0.05142937, -0.03117307, -0.00581914,
    -0.01064003, -0.03608089, -0.05272168, -0.03375670, -0.00795586,
    0.09628104,  0.27129991,  -0.00353779, -0.01734151, -0.03153981,
    0.05686230,  0.28500998,  0.02230594,  -0.02374955, 0.68214326,
    0.05018048,  -0.02320852, -0.04383616, 0.18459474,  0.71517975,
    0.10805613,  -0.03263677, -0.03637639, -0.01394373, -0.02511203,
    -0.01728636, 0.05407331,  -0.02867568, -0.01893131, -0.00240854,
    -0.00446511, -0.01636187, -0.02377053, -0.01522848, -0.00333334,
    -0.00819975, -0.02964169, -0.04499287, -0.02745350, -0.00612408,
    0.02727416,  0.19446600,  0.00159832,  -0.02232473, 0.74982506,
    0.11452620,  -0.03348048, -0.01605681, -0.02070339, -0.00458223,
)


def build_upsampling_kernel_jxl(upsampling: int) -> jax.Array:
  """Builds the exact (N, N, 5, 5) symmetric upsampling kernel from libjxl."""
  if upsampling == 2:
    weights = _WEIGHTS_2
  elif upsampling == 4:
    weights = _WEIGHTS_4
  elif upsampling == 8:
    weights = _WEIGHTS_8
  else:
    raise ValueError(f"Unsupported upsampling factor: {upsampling}")

  n = upsampling
  h = n // 2
  kernel = np.zeros((n, n, 5, 5), dtype=np.float32)
  for ky in range(h):
    for kx in range(h):
      for py in range(5):
        for px in range(5):
          j = 5 * ky + py
          i = 5 * kx + px
          my = min(i, j)
          mx = max(i, j)
          idx = 5 * h * my - my * (my - 1) // 2 + mx - my
          w = weights[idx]
          kernel[ky, kx, py, px] = w
          kernel[ky, n - 1 - kx, py, 4 - px] = w
          kernel[n - 1 - ky, kx, 4 - py, px] = w
          kernel[n - 1 - ky, n - 1 - kx, 4 - py, 4 - px] = w
  return jnp.asarray(kernel, dtype=jnp.float32)


_UPSAMPLING_KERNELS: dict[int, jax.Array] = {
    2: build_upsampling_kernel_jxl(2),
    4: build_upsampling_kernel_jxl(4),
    8: build_upsampling_kernel_jxl(8),
}


def mirror_index_1d(idx: jax.Array, size: int) -> jax.Array:
  """Exact port of jxl::Mirror(x, size) in lib/jxl/image_ops.h:184-196."""
  for _ in range(4):
    idx = jnp.where(idx < 0, -idx - 1, idx)
    idx = jnp.where(idx >= size, 2 * size - 1 - idx, idx)
  return idx


def pad_mirror_2d(plane: jax.Array, pad: int) -> jax.Array:
  """Applies exact libjxl Mirror boundary padding of width `pad` on last 2 dims (H, W)."""
  if pad == 0:
    return plane
  h, w = plane.shape[-2], plane.shape[-1]
  if h >= pad and w >= pad:
    pad_spec = [(0, 0)] * (plane.ndim - 2) + [(pad, pad), (pad, pad)]
    return jnp.pad(plane, pad_spec, mode="symmetric")
  iy = mirror_index_1d(jnp.arange(-pad, h + pad, dtype=jnp.int32), h)
  ix = mirror_index_1d(jnp.arange(-pad, w + pad, dtype=jnp.int32), w)
  return plane[..., iy[:, None], ix[None, :]]


def upsample_jxl(
    plane: jax.Array,
    upsampling: int,
    out_hw: tuple[int, int] | None = None,
) -> jax.Array:
  """Differentiable JXL 5x5 kernel upsampling with 5x5 neighborhood [min, max] clamping.

  Supports `(C, H_in, W_in)`, `(H_in, W_in, C)`, and `(H_in, W_in)` inputs.
  """
  arr = jnp.asarray(plane, dtype=jnp.float32)
  u = int(upsampling)
  if u == 1:
    if out_hw is not None:
      if arr.ndim == 3 and arr.shape[0] != 3 and arr.shape[2] == 3:
        return arr[: out_hw[0], : out_hw[1], :]
      return arr[..., : out_hw[0], : out_hw[1]]
    return arr
  if u not in (2, 4, 8):
    raise ValueError(f"upsampling must be in {{1, 2, 4, 8}}, got {upsampling}")

  squeeze_c = False
  is_hwc = False
  if arr.ndim == 2:
    arr_chw = arr[None, :, :]
    squeeze_c = True
  elif arr.ndim == 3:
    if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
      arr_chw = arr
    elif arr.shape[2] in (1, 3) and arr.shape[0] not in (1, 3):
      arr_chw = jnp.moveaxis(arr, -1, 0)
      is_hwc = True
    else:
      arr_chw = arr
  else:
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

  c, h_in, w_in = arr_chw.shape
  kernel = _UPSAMPLING_KERNELS[u]

  padded = pad_mirror_2d(arr_chw, 2)
  lhs = padded.reshape(c, 1, h_in + 4, w_in + 4)
  rhs = kernel.reshape(u * u, 1, 5, 5)
  raw = jax.lax.conv_general_dilated(
      lhs=lhs,
      rhs=rhs,
      window_strides=(1, 1),
      padding="VALID",
      dimension_numbers=("NCHW", "OIHW", "NCHW"),
      precision=jax.lax.Precision.HIGHEST,
  ).reshape(c, u, u, h_in, w_in)

  # Separable 5x5 min/max avoids XLA's slow 2D select_and_scatter in the backward pass
  h_min = jnp.minimum(
      jnp.minimum(
          jnp.minimum(padded[:, :, :-4], padded[:, :, 1:-3]),
          padded[:, :, 2:-2],
      ),
      jnp.minimum(padded[:, :, 3:-1], padded[:, :, 4:]),
  )
  h_max = jnp.maximum(
      jnp.maximum(
          jnp.maximum(padded[:, :, :-4], padded[:, :, 1:-3]),
          padded[:, :, 2:-2],
      ),
      jnp.maximum(padded[:, :, 3:-1], padded[:, :, 4:]),
  )
  mins = jnp.minimum(
      jnp.minimum(
          jnp.minimum(h_min[:, :-4, :], h_min[:, 1:-3, :]),
          h_min[:, 2:-2, :],
      ),
      jnp.minimum(h_min[:, 3:-1, :], h_min[:, 4:, :]),
  )
  maxs = jnp.maximum(
      jnp.maximum(
          jnp.maximum(h_max[:, :-4, :], h_max[:, 1:-3, :]),
          h_max[:, 2:-2, :],
      ),
      jnp.maximum(h_max[:, 3:-1, :], h_max[:, 4:, :]),
  )

  clamped = jnp.clip(raw, mins[:, None, None, :, :], maxs[:, None, None, :, :])
  out = clamped.transpose(0, 3, 1, 4, 2).reshape(c, h_in * u, w_in * u)

  if out_hw is not None:
    out = out[:, : out_hw[0], : out_hw[1]]
  if squeeze_c:
    return out[0]
  if is_hwc:
    return jnp.moveaxis(out, 0, -1)
  return out


def _blend_xyb_pair(
    cur_xyb: jax.Array,
    ref_xyb: jax.Array,
    blend_mode: int,
    alpha_3d: jax.Array | None = None,
) -> jax.Array:
  """Blends `cur_xyb` (Frame 1) and `ref_xyb` (Frame 0 patch) in XYB space per PatchBlendMode."""
  mode = int(blend_mode)
  if mode == PATCH_BLEND_NONE:
    return cur_xyb
  if mode == PATCH_BLEND_REPLACE:
    return ref_xyb
  if mode == PATCH_BLEND_ADD:
    return cur_xyb + ref_xyb
  if mode == PATCH_BLEND_MUL:
    return cur_xyb * ref_xyb
  if mode == PATCH_BLEND_ALPHA_ABOVE:
    if alpha_3d is None:
      return ref_xyb
    return (1.0 - alpha_3d) * cur_xyb + alpha_3d * ref_xyb
  if mode == PATCH_BLEND_ALPHA_BELOW:
    if alpha_3d is None:
      return cur_xyb
    return alpha_3d * cur_xyb + (1.0 - alpha_3d) * ref_xyb
  if mode == PATCH_BLEND_ALPHA_WEIGHTED_ADD_ABOVE:
    if alpha_3d is None:
      return cur_xyb + ref_xyb
    return cur_xyb + alpha_3d * ref_xyb
  if mode == PATCH_BLEND_ALPHA_WEIGHTED_ADD_BELOW:
    if alpha_3d is None:
      return ref_xyb
    return ref_xyb + alpha_3d * cur_xyb
  raise ValueError(f"Unsupported patch blend_mode: {blend_mode}")


def apply_xyb_patches(
    ref_xyb_ups: jax.Array,
    main_xyb: jax.Array,
    patches: Sequence[PatchEntry],
    alpha_ups: jax.Array | None = None,
) -> jax.Array:
  """Applies PatchDictionaryStage in XYB domain (lib/jxl/dec_patch_dictionary.cc:290-361)."""
  ref_arr = jnp.asarray(ref_xyb_ups, dtype=jnp.float32)
  main_arr = jnp.asarray(main_xyb, dtype=jnp.float32)
  is_hwc = main_arr.ndim == 3 and main_arr.shape[2] == 3 and main_arr.shape[0] != 3
  if is_hwc:
    ref_chw = jnp.moveaxis(ref_arr, -1, 0)
    comp = jnp.moveaxis(main_arr, -1, 0)
  else:
    ref_chw = ref_arr
    comp = main_arr

  if (
      len(patches) > 4
      and alpha_ups is None
      and all(
          p.ref == 0
          and int(p.x0) == int(p.x)
          and int(p.y0) == int(p.y)
          and int(p.blend_mode) in (PATCH_BLEND_REPLACE, PATCH_BLEND_ADD)
          for p in patches
      )
  ):
    _, h, w = comp.shape
    w_ref = np.zeros((1, h, w), dtype=np.float32)
    w_main = np.ones((1, h, w), dtype=np.float32)
    for p in patches:
      x0 = max(0, min(int(p.x), w))
      y0 = max(0, min(int(p.y), h))
      x1 = max(x0, min(int(p.x) + int(p.xsize), w))
      y1 = max(y0, min(int(p.y) + int(p.ysize), h))
      if int(p.blend_mode) == PATCH_BLEND_REPLACE:
        w_ref[:, y0:y1, x0:x1] = 1.0
        w_main[:, y0:y1, x0:x1] = 0.0
      elif int(p.blend_mode) == PATCH_BLEND_ADD:
        w_ref[:, y0:y1, x0:x1] = 1.0
        w_main[:, y0:y1, x0:x1] = 1.0
    comp = jnp.asarray(w_main) * comp + jnp.asarray(w_ref) * ref_chw[:, :h, :w]
    return jnp.moveaxis(comp, 0, -1) if is_hwc else comp

  alpha_2d: jax.Array | None = None
  if alpha_ups is not None:
    a_arr = jnp.asarray(alpha_ups, dtype=jnp.float32)
    if a_arr.ndim == 3 and a_arr.shape[0] == 1:
      alpha_2d = a_arr[0]
    elif a_arr.ndim == 3 and a_arr.shape[2] == 1:
      alpha_2d = a_arr[:, :, 0]
    else:
      alpha_2d = a_arr

  for p in patches:
    if p.ref != 0:
      raise ValueError(f"Expected patch ref == 0, got {p.ref}")
    ref_slice = jax.lax.dynamic_slice(
        ref_chw,
        (0, int(p.y0), int(p.x0)),
        (3, int(p.ysize), int(p.xsize)),
    )
    if p.blend_mode == PATCH_BLEND_REPLACE:
      comp = jax.lax.dynamic_update_slice(comp, ref_slice, (0, int(p.y), int(p.x)))
    else:
      cur_slice = jax.lax.dynamic_slice(
          comp,
          (0, int(p.y), int(p.x)),
          (3, int(p.ysize), int(p.xsize)),
      )
      a_slice_3d: jax.Array | None = None
      if alpha_2d is not None:
        a_slice = jax.lax.dynamic_slice(
            alpha_2d,
            (int(p.y), int(p.x)),
            (int(p.ysize), int(p.xsize)),
        )
        a_slice_3d = a_slice[None, :, :]
      blended_slice = _blend_xyb_pair(
          cur_slice, ref_slice, int(p.blend_mode), alpha_3d=a_slice_3d
      )
      comp = jax.lax.dynamic_update_slice(
          comp, blended_slice, (0, int(p.y), int(p.x))
      )
  return jnp.moveaxis(comp, 0, -1) if is_hwc else comp


def reconstruct_patch_composite(
    ref_state: JxlVarDCTState | StaticVarDCTTopology,
    main_state: JxlVarDCTState | StaticVarDCTTopology,
    upsampling: int,
    patches: Sequence[PatchEntry],
    include_noise: bool = False,
    noise_lut: jax.Array | Sequence[float] | None = None,
    ref_params_override: Mapping[str, Any] | None = None,
    main_params_override: Mapping[str, Any] | None = None,
    ref_quant_ac: jax.Array | None = None,
    ref_dc_quant: jax.Array | None = None,
    ref_raw_quant_field: jax.Array | None = None,
    ref_ytox_map: jax.Array | None = None,
    ref_ytob_map: jax.Array | None = None,
    main_quant_ac: jax.Array | None = None,
    main_dc_quant: jax.Array | None = None,
    main_raw_quant_field: jax.Array | None = None,
    main_ytox_map: jax.Array | None = None,
    main_ytob_map: jax.Array | None = None,
    hard: bool = False,
    visible_frame_index: int = 1,
    nonvisible_frame_index: int = 0,
    ste_quantize_noise_lut: bool = True,
    output_layout: str = "HWC",
    ref_dequant_matrices: Any = None,
    main_dequant_matrices: Any = None,
    ref_dc_quant_weights: jax.Array | Sequence[float] | None = None,
    main_dc_quant_weights: jax.Array | Sequence[float] | None = None,
    enable_noise: bool | None = None,
) -> jax.Array:
  """Differentiable 2-frame ReferenceOnly + XYB PatchDictionary forward reconstruction.

  Stage order matching lib/jxl/dec_cache.cc:193-270:
    1. Frame 0 (`ref_state`): VarDCT IDCT -> crop to `(H_ref, W_ref)` -> Gaborish -> EPF -> `upsample_jxl(U)` in XYB -> crop to `(H, W)`
    2. Frame 1 (`main_state`): VarDCT IDCT -> crop to `(H, W)` -> Gaborish -> EPF in XYB
    3. `PatchDictionaryStage`: `apply_xyb_patches(ref_xyb_ups, main_xyb, patches)` in XYB
    4. Optional `Noise`: `synthesize_jxl_noise` in XYB (with `visible_frame_index=1, nonvisible_frame_index=0`)
    5. `XYBStage` + `FromLinearStage`: `opsin_to_srgb` -> nonlinear sRGB in `[0, 1]`.
  """
  main_topo = (
      main_state
      if isinstance(main_state, StaticVarDCTTopology)
      else StaticVarDCTTopology(main_state)
  )
  h_full, w_full = main_topo.height, main_topo.width

  ref_xyb_low = reconstruct_single_frame_xyb(
      state_or_topo=ref_state,
      quant_ac=ref_quant_ac,
      dc_quant=ref_dc_quant,
      raw_quant_field=ref_raw_quant_field,
      ytox_map=ref_ytox_map,
      ytob_map=ref_ytob_map,
      params_override=ref_params_override,
      hard=hard,
      include_noise=False,
      dequant_matrices=ref_dequant_matrices,
      dc_quant_weights=ref_dc_quant_weights,
  )
  ref_xyb_ups = upsample_jxl(ref_xyb_low, int(upsampling), out_hw=(h_full, w_full))

  main_xyb = reconstruct_single_frame_xyb(
      state_or_topo=main_state,
      quant_ac=main_quant_ac,
      dc_quant=main_dc_quant,
      raw_quant_field=main_raw_quant_field,
      ytox_map=main_ytox_map,
      ytob_map=main_ytob_map,
      params_override=main_params_override,
      hard=hard,
      include_noise=False,
      dequant_matrices=main_dequant_matrices,
      dc_quant_weights=main_dc_quant_weights,
  )

  comp_xyb = apply_xyb_patches(ref_xyb_ups, main_xyb, patches)

  lut_to_use = (
      noise_lut
      if noise_lut is not None
      else (main_params_override.get("noise_lut") if main_params_override else None)
  )
  do_noise = (
      bool(enable_noise)
      if enable_noise is not None
      else bool(
          main_params_override.get("enable_noise", include_noise)
          if main_params_override
          else include_noise
      )
  )
  if do_noise and lut_to_use is not None:
    comp_xyb = synthesize_jxl_noise_xyb(
        xyb=comp_xyb,
        noise_lut=lut_to_use,
        ytox=main_topo.base_correlation_x,
        ytob=main_topo.base_correlation_b,
        visible_frame_index=visible_frame_index,
        nonvisible_frame_index=nonvisible_frame_index,
        ste_quantize_lut=ste_quantize_noise_lut,
    )

  srgb_chw = opsin_to_srgb(
      comp_xyb, main_topo.opsin_inv_matrix, main_topo.opsin_biases, clamp=True
  )
  if output_layout.upper() == "CHW":
    return srgb_chw
  return jnp.moveaxis(srgb_chw, 0, -1)


def ste_quantize_alpha_01(alpha_01: jax.Array, hard: bool = False) -> jax.Array:
  """Clamps alpha to [0, 1] and applies 8-bit uint8 rounding with Straight-Through Estimator."""
  arr = jnp.asarray(alpha_01, dtype=jnp.float32)
  clipped = jnp.clip(arr, 0.0, 1.0)
  quantized = jnp.round(clipped * 255.0) / 255.0
  if hard:
    return quantized
  return clipped + jax.lax.stop_gradient(quantized - clipped)


def _resolve_alpha_composite_args(
    upsampling: int | jax.Array | np.ndarray,
    dim_shift: int | jax.Array | np.ndarray,
    alpha_01: jax.Array | np.ndarray | int | None,
    alpha_dim_shift: int | None,
) -> tuple[int, int, jax.Array]:
  """Resolves `(upsampling, dim_shift, alpha_01)`, `(upsampling, alpha_01, alpha_dim_shift=...)`, and `(alpha_01, upsampling, alpha_dim_shift)` call conventions."""
  ups_arr = jnp.asarray(upsampling)
  ds_arr = jnp.asarray(dim_shift)
  if ups_arr.ndim >= 2:
    actual_alpha = jnp.asarray(upsampling, dtype=jnp.float32)
    actual_upsampling = int(dim_shift) if int(dim_shift) > 0 else 2
    if alpha_dim_shift is not None:
      actual_dim_shift = int(alpha_dim_shift)
    elif alpha_01 is not None and jnp.asarray(alpha_01).ndim == 0:
      actual_dim_shift = int(alpha_01)
    else:
      actual_dim_shift = 0
  elif ds_arr.ndim >= 2:
    actual_upsampling = int(upsampling)
    actual_alpha = jnp.asarray(dim_shift, dtype=jnp.float32)
    if alpha_dim_shift is not None:
      actual_dim_shift = int(alpha_dim_shift)
    elif alpha_01 is not None and jnp.asarray(alpha_01).ndim == 0:
      actual_dim_shift = int(alpha_01)
    else:
      actual_dim_shift = 0
  else:
    if alpha_01 is None:
      raise ValueError("alpha_01 must be provided to reconstruct_alpha_composite")
    actual_alpha = jnp.asarray(alpha_01, dtype=jnp.float32)
    actual_upsampling = int(upsampling)
    actual_dim_shift = (
        int(alpha_dim_shift) if alpha_dim_shift is not None else int(dim_shift)
    )
  return actual_upsampling, actual_dim_shift, actual_alpha


def reconstruct_alpha_composite_xyb(
    ref_state: JxlVarDCTState | StaticVarDCTTopology,
    main_state: JxlVarDCTState | StaticVarDCTTopology,
    upsampling: int | jax.Array | np.ndarray = 2,
    dim_shift: int = 0,
    alpha_01: jax.Array | np.ndarray | int | None = None,
    include_noise: bool = False,
    noise_lut: jax.Array | Sequence[float] | None = None,
    ref_params_override: Mapping[str, Any] | None = None,
    main_params_override: Mapping[str, Any] | None = None,
    ref_quant_ac: jax.Array | None = None,
    ref_dc_quant: jax.Array | None = None,
    ref_raw_quant_field: jax.Array | None = None,
    ref_ytox_map: jax.Array | None = None,
    ref_ytob_map: jax.Array | None = None,
    main_quant_ac: jax.Array | None = None,
    main_dc_quant: jax.Array | None = None,
    main_raw_quant_field: jax.Array | None = None,
    main_ytox_map: jax.Array | None = None,
    main_ytob_map: jax.Array | None = None,
    hard: bool = False,
    quantize_alpha: bool = True,
    visible_frame_index: int = 1,
    nonvisible_frame_index: int = 0,
    ste_quantize_noise_lut: bool = True,
    alpha_dim_shift: int | None = None,
    alpha_blend_mode: int = PATCH_BLEND_ALPHA_BELOW,
    patches: Sequence[PatchEntry] | None = None,
    ref_dequant_matrices: Any = None,
    main_dequant_matrices: Any = None,
    ref_dc_quant_weights: jax.Array | Sequence[float] | None = None,
    main_dc_quant_weights: jax.Array | Sequence[float] | None = None,
    enable_noise: bool | None = None,
) -> jax.Array:
  """Differentiable 2-frame ReferenceOnly + ExtraChannel::kAlpha XYB PatchDictionary reconstruction (`(3, H, W)` XYB).

  Stage order matching `lib/jxl/dec_cache.cc:193-270` and `lib/jxl/dec_patch_dictionary.cc:290-361`:
    1. Frame 0 (`ref_state`, `save_before_color_transform=True`):
       VarDCT IDCT -> crop to `(H_ref, W_ref)` -> Gaborish -> EPF -> `upsample_jxl(U)` in XYB -> crop to `(H, W)` -> `ref_xyb_ups`
    2. Frame 1 (`main_state`, `save_before_color_transform=True` during `PatchDictionaryStage`):
       VarDCT IDCT -> crop to `(H, W)` -> Gaborish -> EPF in XYB (`include_noise=False`) -> `main_xyb`
    3. ExtraChannel 0 (`alpha_01`, `save_before_color_transform=True`):
       `ste_quantize_alpha_01` -> `upsample_jxl(1 << alpha_dim_shift, out_hw=(H, W))` -> `alpha_ups`
    4. `PatchDictionaryStage` in XYB space (`alpha_blend_mode`, default `5 = kBlendBelow`):
       `comp_xyb = alpha_ups * main_xyb + (1 - alpha_ups) * ref_xyb_ups`
    5. Optional `Noise` in XYB (`synthesize_jxl_noise_xyb`) applied to `comp_xyb` AFTER XYB patch blending.
  """
  actual_upsampling, actual_dim_shift, alpha_arr = _resolve_alpha_composite_args(
      upsampling, dim_shift, alpha_01, alpha_dim_shift
  )
  main_topo = (
      main_state
      if isinstance(main_state, StaticVarDCTTopology)
      else StaticVarDCTTopology(main_state)
  )
  h_full, w_full = main_topo.height, main_topo.width

  ref_xyb_low = reconstruct_single_frame_xyb(
      state_or_topo=ref_state,
      quant_ac=ref_quant_ac,
      dc_quant=ref_dc_quant,
      raw_quant_field=ref_raw_quant_field,
      ytox_map=ref_ytox_map,
      ytob_map=ref_ytob_map,
      params_override=ref_params_override,
      hard=hard,
      include_noise=False,
      dequant_matrices=ref_dequant_matrices,
      dc_quant_weights=ref_dc_quant_weights,
  )
  ref_xyb_ups = upsample_jxl(
      ref_xyb_low, int(actual_upsampling), out_hw=(h_full, w_full)
  )

  main_xyb = reconstruct_single_frame_xyb(
      state_or_topo=main_state,
      quant_ac=main_quant_ac,
      dc_quant=main_dc_quant,
      raw_quant_field=main_raw_quant_field,
      ytox_map=main_ytox_map,
      ytob_map=main_ytob_map,
      params_override=main_params_override,
      hard=hard,
      include_noise=False,
      dequant_matrices=main_dequant_matrices,
      dc_quant_weights=main_dc_quant_weights,
  )

  if alpha_arr.ndim == 3 and alpha_arr.shape[0] == 1:
    alpha_arr = alpha_arr[0]
  elif alpha_arr.ndim == 3 and alpha_arr.shape[2] == 1:
    alpha_arr = alpha_arr[:, :, 0]
  if quantize_alpha:
    alpha_q = ste_quantize_alpha_01(alpha_arr, hard=hard)
  else:
    alpha_q = jnp.clip(alpha_arr, 0.0, 1.0)

  ec_ups = 1 << int(actual_dim_shift)
  alpha_ups = upsample_jxl(alpha_q, ec_ups, out_hw=(h_full, w_full))
  alpha_ups_3 = alpha_ups[None, :, :]

  if patches is not None and len(patches) > 0:
    comp_xyb = apply_xyb_patches(
        ref_xyb_ups, main_xyb, patches, alpha_ups=alpha_ups
    )
  else:
    comp_xyb = _blend_xyb_pair(
        main_xyb, ref_xyb_ups, int(alpha_blend_mode), alpha_3d=alpha_ups_3
    )

  lut_to_use = (
      noise_lut
      if noise_lut is not None
      else (main_params_override.get("noise_lut") if main_params_override else None)
  )
  do_noise = (
      bool(enable_noise)
      if enable_noise is not None
      else bool(
          main_params_override.get("enable_noise", include_noise)
          if main_params_override
          else include_noise
      )
  )
  if do_noise and lut_to_use is not None:
    comp_xyb = synthesize_jxl_noise_xyb(
        xyb=comp_xyb,
        noise_lut=lut_to_use,
        ytox=main_topo.base_correlation_x,
        ytob=main_topo.base_correlation_b,
        visible_frame_index=visible_frame_index,
        nonvisible_frame_index=nonvisible_frame_index,
        ste_quantize_lut=ste_quantize_noise_lut,
    )

  return comp_xyb


def reconstruct_alpha_composite(
    ref_state: JxlVarDCTState | StaticVarDCTTopology,
    main_state: JxlVarDCTState | StaticVarDCTTopology,
    upsampling: int | jax.Array | np.ndarray = 2,
    dim_shift: int = 0,
    alpha_01: jax.Array | np.ndarray | int | None = None,
    include_noise: bool = False,
    noise_lut: jax.Array | Sequence[float] | None = None,
    ref_params_override: Mapping[str, Any] | None = None,
    main_params_override: Mapping[str, Any] | None = None,
    ref_quant_ac: jax.Array | None = None,
    ref_dc_quant: jax.Array | None = None,
    ref_raw_quant_field: jax.Array | None = None,
    ref_ytox_map: jax.Array | None = None,
    ref_ytob_map: jax.Array | None = None,
    main_quant_ac: jax.Array | None = None,
    main_dc_quant: jax.Array | None = None,
    main_raw_quant_field: jax.Array | None = None,
    main_ytox_map: jax.Array | None = None,
    main_ytob_map: jax.Array | None = None,
    hard: bool = False,
    quantize_alpha: bool = True,
    visible_frame_index: int = 1,
    nonvisible_frame_index: int = 0,
    ste_quantize_noise_lut: bool = True,
    output_layout: str = "HWC",
    alpha_dim_shift: int | None = None,
    alpha_blend_mode: int = PATCH_BLEND_ALPHA_BELOW,
    patches: Sequence[PatchEntry] | None = None,
    ref_dequant_matrices: Any = None,
    main_dequant_matrices: Any = None,
    ref_dc_quant_weights: jax.Array | Sequence[float] | None = None,
    main_dc_quant_weights: jax.Array | Sequence[float] | None = None,
    enable_noise: bool | None = None,
) -> jax.Array:
  """Differentiable 2-frame ReferenceOnly + ExtraChannel::kAlpha XYB PatchDictionary forward reconstruction.

  Stage order matching `lib/jxl/dec_cache.cc:193-270` and `lib/jxl/dec_patch_dictionary.cc:290-361`:
    1. Frame 0 (`ref_state`, `save_before_color_transform=True`):
       VarDCT IDCT -> crop to `(H_ref, W_ref)` -> Gaborish -> EPF -> `upsample_jxl(U)` in XYB -> crop to `(H, W)` -> `ref_xyb_ups`
    2. Frame 1 (`main_state`):
       VarDCT IDCT -> crop to `(H, W)` -> Gaborish -> EPF in XYB (`include_noise=False`) -> `main_xyb`
    3. ExtraChannel 0 (`alpha_01`, `save_before_color_transform=True`):
       `ste_quantize_alpha_01` -> `upsample_jxl(1 << alpha_dim_shift, out_hw=(H, W))` -> `alpha_ups`
    4. `PatchDictionaryStage` in XYB space (`alpha_blend_mode`, default `5 = kBlendBelow`):
       `comp_xyb = alpha_ups * main_xyb + (1 - alpha_ups) * ref_xyb_ups`
    5. Optional `Noise` in XYB (`synthesize_jxl_noise_xyb`) applied to `comp_xyb` AFTER XYB patch blending and BEFORE `opsin_to_srgb`.
    6. `XYBStage` + `FromLinearStage`: `opsin_to_srgb(comp_xyb, clamp=True)` -> nonlinear sRGB in `[0, 1]`.
  """
  main_topo = (
      main_state
      if isinstance(main_state, StaticVarDCTTopology)
      else StaticVarDCTTopology(main_state)
  )
  comp_xyb = reconstruct_alpha_composite_xyb(
      ref_state=ref_state,
      main_state=main_state,
      upsampling=upsampling,
      dim_shift=dim_shift,
      alpha_01=alpha_01,
      include_noise=include_noise,
      noise_lut=noise_lut,
      ref_params_override=ref_params_override,
      main_params_override=main_params_override,
      ref_quant_ac=ref_quant_ac,
      ref_dc_quant=ref_dc_quant,
      ref_raw_quant_field=ref_raw_quant_field,
      ref_ytox_map=ref_ytox_map,
      ref_ytob_map=ref_ytob_map,
      main_quant_ac=main_quant_ac,
      main_dc_quant=main_dc_quant,
      main_raw_quant_field=main_raw_quant_field,
      main_ytox_map=main_ytox_map,
      main_ytob_map=main_ytob_map,
      hard=hard,
      quantize_alpha=quantize_alpha,
      visible_frame_index=visible_frame_index,
      nonvisible_frame_index=nonvisible_frame_index,
      ste_quantize_noise_lut=ste_quantize_noise_lut,
      alpha_dim_shift=alpha_dim_shift,
      alpha_blend_mode=alpha_blend_mode,
      patches=patches,
      ref_dequant_matrices=ref_dequant_matrices,
      main_dequant_matrices=main_dequant_matrices,
      ref_dc_quant_weights=ref_dc_quant_weights,
      main_dc_quant_weights=main_dc_quant_weights,
      enable_noise=enable_noise,
  )
  srgb_chw = opsin_to_srgb(
      comp_xyb, main_topo.opsin_inv_matrix, main_topo.opsin_biases, clamp=True
  )
  if output_layout.upper() == "CHW":
    return srgb_chw
  return jnp.moveaxis(srgb_chw, 0, -1)

