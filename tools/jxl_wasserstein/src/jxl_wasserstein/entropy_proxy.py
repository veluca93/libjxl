"""Differentiable JPEG XL token/context entropy proxy in JAX.

Synthesizes:
1. VarDCT AC token & context entropy model (`soft_ac_rate_bits`, `ste_round`)
   with anchored log2-magnitude coordinate `ell(|q|) = log2(|q| + 0.5)`,
   reverse cumulative `log_sigmoid` EOB survival probabilities, per-(channel,
   frequency-band) Shannon entropy + extra bits, neighbor-predicted `nzeros`
   token rate, and canonical LLF exclusion across all 27 `AcStrategy` types.
2. VarDCT DC (`soft_dc_rate_bits`), AC metadata (`soft_ac_meta_rate_bits` /
   `soft_ac_metadata_bits`), and frame/codestream header (`soft_header_rate_bits`)
   models using 2D `ClampedGradient` prediction (`soft_clamped_gradient_2d`)
   and the bi-scale logistic interval kernel over the 44 signed intervals of
   `PackSigned` + `HybridUint(4, 2, 0)` with KL self-entropy debiasing.
3. Multi-resolution `PatchDictionary` (`soft_patch_dictionary_bits`,
   `soft_patch_composite_rate_proxy`), Modular `ExtraChannel::kAlpha`
   (`soft_alpha_channel_bits`, `soft_alpha_composite_rate_proxy`), `NoiseParams`
   80-bit header (`soft_noise_header_bits`), and unified single-frame rate
   proxy (`soft_jxl_rate_proxy`, `RateBreakdown`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jxl_wasserstein.cpp_bridge import (
    ACS_COVERED_BLOCKS_X,
    ACS_COVERED_BLOCKS_Y,
    PATCH_BLEND_REPLACE,
    JxlVarDCTState,
    PatchEntry,
)
from jxl_wasserstein.forward_model import (
    K_DC_QUANT,
    StaticVarDCTTopology,
    get_default_dequant_matrices,
)


class RateBreakdown(NamedTuple):
  """Component-wise breakdown of the differentiable JXL rate proxy in bits."""

  total_bits: jax.Array
  bpp: jax.Array
  ac_bits: jax.Array
  dc_bits: jax.Array
  ac_meta_bits: jax.Array
  header_bits: jax.Array
  noise_bits: jax.Array
  patch_bits: jax.Array
  alpha_bits: jax.Array


# -----------------------------------------------------------------------------
# 1. Exact JXL AC Context & Scan-Order Constants (lib/jxl/ac_context.h &
#    lib/jxl/ac_strategy.cc)
# -----------------------------------------------------------------------------

# lib/jxl/ac_context.h:62-70 — maps normalized scan index k_norm in [0..63]
# to 26 frequency contexts (index 0 is LLF sentinel, mapped to 0).
K_COEFF_FREQ_CONTEXT: tuple[int, ...] = (
    0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 10, 11, 11, 12,
    12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20,
    20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 23,
    23, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25,
)
NUM_FREQ_CONTEXTS: int = 26

# 8-band grouping of the 26 frequency contexts matching libjxl's post-clustering
# ANS histogram granularity across small and large frames.
K_FREQ_CTX_TO_BAND8: tuple[int, ...] = (
    0, 0, 1, 1, 2, 2, 2, 3, 3, 3,
    4, 4, 4, 5, 5, 5, 5, 6, 6, 6,
    6, 6, 7, 7, 7, 7,
)
NUM_FREQ_BANDS8: int = 8

# Number of non-zero AC magnitude categories s in {1 .. MAX_AC_CATEGORY}
# corresponding to integer intervals [2^(s-1), 2^s - 1] with half-integer
# boundaries [2^(s-1) - 0.5, 2^s - 0.5).
MAX_AC_CATEGORY: int = 11  # covers |q| up to 2047


def compute_natural_coeff_order(cx_in: int, cy_in: int) -> np.ndarray:
  """Exact port of AcStrategy::ComputeNaturalCoeffOrder (lib/jxl/ac_strategy.cc:29-83).

  Returns a 1D int32 array `order` of length `64 * cx_in * cy_in` mapping
  scan index `k` to flat canonical coefficient index `y * (8 * cx_layout) + x`
  in the `(8 * cy_layout, 8 * cx_layout)` canonical coefficient block where
  `cy_layout = min(cy_in, cx_in)` and `cx_layout = max(cy_in, cx_in)`.

  The first `B = cx_in * cy_in` entries `order[:B]` are the canonical LLF
  coefficients (`0 <= y < cy_layout, 0 <= x < cx_layout`); the remaining
  `63 * B` entries `order[B:]` are the AC coefficients in natural zigzag order.
  """
  cy = min(cy_in, cx_in)
  cx = max(cy_in, cx_in)
  b_dim = 8
  xs = cx // cy
  xsm = xs - 1
  xss = (xs - 1).bit_length()
  cur = cx * cy
  out = np.zeros(cx * cy * 64, dtype=np.int32)

  for i in range(cx * b_dim):
    for j in range(i + 1):
      x = j
      y = i - j
      if i % 2:
        x, y = y, x
      if (y & xsm) != 0:
        continue
      y >>= xss
      if x < cx and y < cy:
        val = y * cx + x
      else:
        val = cur
        cur += 1
      out[val] = y * cx * b_dim + x

  for ip in range(cx * b_dim - 1, 0, -1):
    i = ip - 1
    for j in range(i + 1):
      x = cx * b_dim - 1 - (i - j)
      y = cx * b_dim - 1 - j
      if i % 2:
        x, y = y, x
      if (y & xsm) != 0:
        continue
      y >>= xss
      val = cur
      cur += 1
      out[val] = y * cx * b_dim + x

  return out


_NATURAL_AC_ORDER_CACHE: dict[
    tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
] = {}


def get_canonical_ac_scan_tables(
    cy_l: int, cx_l: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns `(ac_flat_indices, freq_ctx_26, freq_band_8)` for canonical `(cy_l, cx_l)`."""
  key = (int(cy_l), int(cx_l))
  if key not in _NATURAL_AC_ORDER_CACHE:
    order = compute_natural_coeff_order(cx_l, cy_l)
    b = cy_l * cx_l
    log2_b = (b - 1).bit_length()
    ac_order = np.asarray(order[b:], dtype=np.int32)  # shape (63 * B,)
    k_indices = np.arange(b, 64 * b, dtype=np.int32)
    k_norm = k_indices >> log2_b  # in [1..63]
    freq_ctx_np = np.array(
        [K_COEFF_FREQ_CONTEXT[int(kn)] for kn in k_norm], dtype=np.int32
    )
    freq_band8_np = np.array(
        [K_FREQ_CTX_TO_BAND8[int(fc)] for fc in freq_ctx_np], dtype=np.int32
    )
    _NATURAL_AC_ORDER_CACHE[key] = (
        ac_order,
        freq_ctx_np,
        freq_band8_np,
    )
  ac_order, freq_ctx_np, freq_band8_np = _NATURAL_AC_ORDER_CACHE[key]
  with jax.ensure_compile_time_eval():
    return (
        jnp.asarray(ac_order, dtype=jnp.int32),
        jnp.asarray(freq_ctx_np, dtype=jnp.int32),
        jnp.asarray(freq_band8_np, dtype=jnp.int32),
    )


class _LightweightAcStrategyTopology:
  """Cached block grouping parsed from a raw `ac_strategy` `(h8, w8)` array."""

  def __init__(self, ac_strategy: np.ndarray) -> None:
    acs_np = np.asarray(ac_strategy, dtype=np.int32)
    self.h8, self.w8 = acs_np.shape
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
        cx = ACS_COVERED_BLOCKS_X[s] if 0 <= s < len(ACS_COVERED_BLOCKS_X) else 1
        cy = ACS_COVERED_BLOCKS_Y[s] if 0 <= s < len(ACS_COVERED_BLOCKS_Y) else 1
        covered_m[by : by + cy, bx : bx + cx] = True
        first_block_mask_np[by, bx] = 1.0
        blocks_multi.setdefault(s, []).append((by, bx))

    self.mask_1x1 = mask_1x1_np
    self.first_block_mask = first_block_mask_np
    by_1x1, bx_1x1 = np.nonzero(is_1x1_np)
    self.has_1x1 = bool(by_1x1.size > 0)
    self.by_1x1 = by_1x1.astype(np.int32)
    self.bx_1x1 = bx_1x1.astype(np.int32)

    self.multi_groups: list[dict[str, Any]] = []
    for s, coords in sorted(blocks_multi.items()):
      cx = ACS_COVERED_BLOCKS_X[s]
      cy = ACS_COVERED_BLOCKS_Y[s]
      cy_l, cx_l = min(cy, cx), max(cy, cx)
      by_np = np.array([c[0] for c in coords], dtype=np.int32)
      bx_np = np.array([c[1] for c in coords], dtype=np.int32)
      nh_cell = self.h8 // cy
      nw_cell = self.w8 // cx
      grid_aligned = bool(
          nh_cell > 0
          and nw_cell > 0
          and np.all((by_np % cy) == 0)
          and np.all((bx_np % cx) == 0)
          and np.all((by_np // cy) < nh_cell)
          and np.all((bx_np // cx) < nw_cell)
      )
      if grid_aligned:
        cell_mask_np = np.zeros((nh_cell, nw_cell), dtype=bool)
        cell_mask_np[by_np // cy, bx_np // cx] = True
        blk_mask_np = np.repeat(np.repeat(cell_mask_np, cy, axis=0), cx, axis=1)
        cell_mask_f = cell_mask_np.astype(np.float32)
      else:
        cell_mask_np = None
        blk_mask_np = None
        cell_mask_f = None
      self.multi_groups.append({
          "s": s,
          "cy": cy,
          "cx": cx,
          "cy_l": cy_l,
          "cx_l": cx_l,
          "by": by_np,
          "bx": bx_np,
          "grid_aligned": grid_aligned,
          "nh_cell": nh_cell,
          "nw_cell": nw_cell,
          "h_lim": nh_cell * cy,
          "w_lim": nw_cell * cx,
          "cell_mask": cell_mask_np,
          "cell_mask_f": cell_mask_f,
          "blk_mask": blk_mask_np,
      })
    self.all_grid_aligned = all(
        bool(g.get("grid_aligned", False)) for g in self.multi_groups
    )


_ACS_TOPO_CACHE: dict[tuple[tuple[int, int], bytes], _LightweightAcStrategyTopology] = {}


def _resolve_topology(
    topology: StaticVarDCTTopology | JxlVarDCTState | None,
    ac_strategy: np.ndarray | jax.Array | None,
) -> StaticVarDCTTopology | _LightweightAcStrategyTopology | None:
  """Resolves block-strategy topology from `topology`, `JxlVarDCTState`, or `ac_strategy`."""
  if isinstance(topology, StaticVarDCTTopology):
    return topology
  if isinstance(topology, JxlVarDCTState):
    with jax.ensure_compile_time_eval():
      return StaticVarDCTTopology(topology)
  if topology is not None and hasattr(topology, "multi_groups"):
    return topology  # type: ignore[return-value]
  if ac_strategy is not None and not isinstance(ac_strategy, jax.core.Tracer):
    acs_np = np.ascontiguousarray(np.asarray(ac_strategy, dtype=np.uint8))
    if not np.any(acs_np != 0):
      return None
    key = (acs_np.shape, acs_np.tobytes())
    if key not in _ACS_TOPO_CACHE:
      if len(_ACS_TOPO_CACHE) >= 32:
        _ACS_TOPO_CACHE.clear()
      _ACS_TOPO_CACHE[key] = _LightweightAcStrategyTopology(acs_np)
    return _ACS_TOPO_CACHE[key]
  return None


def get_first_block_and_llf_masks(
    ac_strategy: np.ndarray | jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Computes first-block mask `(bh, bw)` and LLF mask `(bh, bw, 8, 8)` from `ac_strategy`."""
  acs_np = np.asarray(ac_strategy, dtype=np.int32)
  bh, bw = acs_np.shape
  covered = np.zeros((bh, bw), dtype=bool)
  first_mask = np.zeros((bh, bw), dtype=bool)
  llf_mask = np.zeros((bh, bw, 8, 8), dtype=bool)

  for by in range(bh):
    for bx in range(bw):
      if covered[by, bx]:
        continue
      s = int(acs_np[by, bx])
      cx = ACS_COVERED_BLOCKS_X[s] if 0 <= s < len(ACS_COVERED_BLOCKS_X) else 1
      cy = ACS_COVERED_BLOCKS_Y[s] if 0 <= s < len(ACS_COVERED_BLOCKS_Y) else 1
      cy_l, cx_l = min(cy, cx), max(cy, cx)
      covered[by : by + cy, bx : bx + cx] = True
      first_mask[by, bx] = True
      canon_llf = np.zeros((8 * cy_l, 8 * cx_l), dtype=bool)
      canon_llf[:cy_l, :cx_l] = True
      blk_llf = canon_llf.reshape(cy, 8, cx, 8).transpose(0, 2, 1, 3)
      llf_mask[by : by + cy, bx : bx + cx] = blk_llf

  with jax.ensure_compile_time_eval():
    return jnp.asarray(first_mask), jnp.asarray(llf_mask)


get_first_block_And_llf_masks = get_first_block_and_llf_masks


def _infer_expected_hw8(
    *,
    state: JxlVarDCTState | None = None,
    topology: Any | None = None,
    ac_strategy: Any | None = None,
    raw_quant_field: Any | None = None,
) -> tuple[int, int] | None:
  """Infers expected `(h8, w8)` 8x8 block dimensions from state, topology, or metadata."""
  if (
      topology is not None
      and hasattr(topology, "h8")
      and hasattr(topology, "w8")
  ):
    return int(topology.h8), int(topology.w8)
  if state is not None and hasattr(state, "height") and hasattr(state, "width"):
    return (int(state.height) + 7) // 8, (int(state.width) + 7) // 8
  if (
      ac_strategy is not None
      and hasattr(ac_strategy, "shape")
      and len(ac_strategy.shape) == 2
  ):
    return int(ac_strategy.shape[0]), int(ac_strategy.shape[1])
  if (
      raw_quant_field is not None
      and hasattr(raw_quant_field, "shape")
      and len(raw_quant_field.shape) == 2
  ):
    return int(raw_quant_field.shape[0]), int(raw_quant_field.shape[1])
  return None


def normalize_quant_ac_shape(
    quant_ac: jax.Array,
    expected_hw8: tuple[int, int] | None = None,
) -> jax.Array:
  """Normalizes `quant_ac` from `(3, H, W)`, `(3, bh, bw, 8, 8)`, or `(bh, bw, 3, 8, 8)` to `(3, bh, bw, 8, 8)`."""
  q = jnp.asarray(quant_ac, dtype=jnp.float32)
  if q.ndim == 3:
    c, h, w = q.shape
    bh, bw = h // 8, w // 8
    return jnp.transpose(q.reshape(c, bh, 8, bw, 8), (0, 1, 3, 2, 4))
  if q.ndim == 5:
    if expected_hw8 is not None:
      eh8, ew8 = int(expected_hw8[0]), int(expected_hw8[1])
      is_chw = q.shape[:3] == (3, eh8, ew8)
      is_hwc = q.shape[:3] == (eh8, ew8, 3)
      if is_chw and not is_hwc:
        return q
      if is_hwc and not is_chw:
        return jnp.transpose(q, (2, 0, 1, 3, 4))
    if q.shape[0] == 3 and q.shape[2] != 3:
      return q
    if q.shape[2] == 3 and q.shape[0] != 3:
      return jnp.transpose(q, (2, 0, 1, 3, 4))
    return q
  raise ValueError(f"Expected quant_ac to have 3 or 5 dims, got shape {q.shape}")


def _normalize_quant_ac_blocks_hwc(
    quant_ac: jax.Array,
    expected_hw8: tuple[int, int] | None = None,
) -> tuple[jax.Array, int, int]:
  """Normalizes `quant_ac` into `(h8, w8, 3, 8, 8)` block-major order."""
  q_f = jnp.asarray(quant_ac, dtype=jnp.float32)
  if q_f.ndim == 3:
    c_dim, h_px, w_px = q_f.shape
    h8, w8 = h_px // 8, w_px // 8
    return (
        jnp.transpose(q_f.reshape(c_dim, h8, 8, w8, 8), (1, 3, 0, 2, 4)),
        h8,
        w8,
    )
  if q_f.ndim == 5:
    q_5d = normalize_quant_ac_shape(q_f, expected_hw8=expected_hw8)
    _, h8, w8, _, _ = q_5d.shape
    return jnp.transpose(q_5d, (1, 2, 0, 3, 4)), h8, w8
  raise ValueError(
      f"Expected quant_ac to have 3 or 5 dims, got shape {q_f.shape}"
  )


# -----------------------------------------------------------------------------
# 2. Sign-Preserving STE & Differentiable VarDCT AC Rate Model (`soft_ac_rate_bits`)
# -----------------------------------------------------------------------------


def ste_round(q: jax.Array) -> jax.Array:
  """Sign-preserving Straight-Through Estimator for `round(q)`.

  Preserves the IEEE-754 negative zero sign bit (`-0.0`) when `-0.5 < q < 0` so
  that `jnp.where(jnp.signbit(x), -x, x)` has exact radial gradient `sign(q)`
  even at sub-deadzone coefficients that round to zero.
  """
  q_f = jnp.asarray(q, dtype=jnp.float32)
  abs_q = jnp.where(jnp.signbit(q_f), -q_f, q_f)
  abs_ste = abs_q + jax.lax.stop_gradient(jnp.abs(jnp.round(q_f)) - abs_q)
  sgn = jnp.where(jnp.signbit(q_f), -1.0, 1.0)
  return jax.lax.stop_gradient(sgn) * abs_ste


def compute_continuous_quant_ac_from_dct(
    scaled_dct_blocks: jax.Array,
    raw_quant_field: jax.Array,
    global_scale: int,
    dequant_1x1: jax.Array,
    x_qm_scale: int = 2,
    b_qm_scale: int = 2,
    use_ste_round: bool = False,
) -> jax.Array:
  """Computes continuous (or STE-rounded) `quant_ac` from forward DCT & `raw_quant_field`."""
  qac_scale = (float(global_scale) / 65536.0) * jnp.maximum(
      jnp.asarray(raw_quant_field, dtype=jnp.float32), 1.0
  )
  x_qm_mul = float(1.25 ** (x_qm_scale - 2.0))
  b_qm_mul = float(1.25 ** (b_qm_scale - 2.0))
  ch_mul = jnp.array(
      [x_qm_mul, 1.0, b_qm_mul], dtype=scaled_dct_blocks.dtype
  )[None, None, :, None, None]
  inv_dq = 1.0 / jnp.asarray(dequant_1x1, dtype=scaled_dct_blocks.dtype)[
      None, None, :, :, :
  ]
  q_cont = (
      scaled_dct_blocks * qac_scale[:, :, None, None, None] * ch_mul * inv_dq
  )
  q_cont = q_cont.at[:, :, :, 0, 0].set(0.0)
  if use_ste_round:
    q_cont = ste_round(q_cont)
  return q_cont


def soft_zero_and_active(
    c: jax.Array,
    tau: float = 0.25,
) -> tuple[jax.Array, jax.Array]:
  """Anchored smooth indicator of zero vs active (non-zero) quantized values.

  Uses log2-magnitude coordinate `ell(|c|) = log2(|c| + 0.5)` without
  `jnp.maximum(0.0, ...)` tie-killing so that at `c == 0`:
    - `p_nz == 0.0` and `p_zero == 1.0` exactly, and
    - `dp_nz / d|c| > 0` strictly at `|c| == 0` and for all `|c| > 0`.
  """
  c_f = jnp.asarray(c, dtype=jnp.float32)
  abs_c = jnp.where(jnp.signbit(c_f), -c_f, c_f)
  temp = jnp.maximum(jnp.asarray(tau, dtype=jnp.float32), 1e-4)
  ell = jnp.log2(abs_c + 0.5)
  raw_sig = jax.nn.sigmoid(ell / temp)
  sig_at_zero = jax.nn.sigmoid(-1.0 / temp)
  p_nz = (raw_sig - sig_at_zero) / (1.0 - sig_at_zero)
  log_p_zero = jax.nn.log_sigmoid(-ell / temp) - jax.nn.log_sigmoid(1.0 / temp)
  p_zero = jnp.exp(log_p_zero)
  return p_zero, p_nz


def soft_category_probs(
    c: jax.Array,
    max_cat: int = MAX_AC_CATEGORY,
    tau: float = 0.25,
) -> jax.Array:
  """Anchored soft probability distribution over categories `{0, 1, ..., max_cat}`."""
  c_f = jnp.asarray(c, dtype=jnp.float32)
  abs_c = jnp.where(jnp.signbit(c_f), -c_f, c_f)
  temp = jnp.maximum(jnp.asarray(tau, dtype=jnp.float32), 1e-4)
  ell = jnp.log2(abs_c + 0.5)
  s_idx = jnp.arange(1, max_cat + 1, dtype=c_f.dtype)
  thresholds = s_idx - 1.0
  raw_sig = jax.nn.sigmoid((ell[..., None] - thresholds) / temp)
  sig_at_zero = jax.nn.sigmoid(-s_idx / temp)
  cdf_ge = (raw_sig - sig_at_zero) / (1.0 - sig_at_zero)
  cdf_next = jnp.concatenate(
      [cdf_ge[..., 1:], jnp.zeros_like(cdf_ge[..., :1])], axis=-1
  )
  cat_weights = cdf_ge - cdf_next
  p_zero = 1.0 - cdf_ge[..., :1]
  return jnp.concatenate([p_zero, cat_weights], axis=-1)


def _soft_category_weights(
    abs_q: jax.Array,
    temp: float = 0.25,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Computes anchored soft zero probability, non-zero probability, and category weights.

  Uses log2-magnitude coordinate `ell(|q|) = log2(|q| + 0.5)`, which maps:
    - `|q| = 0` -> `ell = -1`
    - `|q| = 2^(s-1) - 0.5` -> `ell = s - 1` (exact half-integer dyadic boundary
      for every category `s in 1..MAX_AC_CATEGORY`).

  Crucially avoids `jnp.maximum(0.0, raw_sig - sig_at_zero)` so the derivative
  at `|q| == 0` is never killed by JAX's tie-breaking rule on `lax.max`.
  """
  ell = jnp.log2(abs_q + 0.5)
  s_idx = jnp.arange(1, MAX_AC_CATEGORY + 1, dtype=abs_q.dtype)
  thresholds = s_idx - 1.0

  raw_sig = jax.nn.sigmoid((ell[..., None] - thresholds) / temp)
  sig_at_zero = jax.nn.sigmoid(-s_idx / temp)
  norm_denom = 1.0 - sig_at_zero

  cdf_ge = (raw_sig - sig_at_zero) / norm_denom
  cdf_next = jnp.concatenate(
      [cdf_ge[..., 1:], jnp.zeros_like(cdf_ge[..., :1])], axis=-1
  )
  cat_weights = cdf_ge - cdf_next
  nz_prob = cdf_ge[..., 0]
  log_zero_prob = jax.nn.log_sigmoid(-ell / temp) - jax.nn.log_sigmoid(
      jnp.asarray(1.0 / temp, dtype=abs_q.dtype)
  )
  zero_prob = jnp.exp(log_zero_prob)
  return zero_prob, nz_prob, cat_weights, cdf_ge, log_zero_prob


def _extract_ac_tokens_from_blocks(
    quant_ac: jax.Array,
    topology: StaticVarDCTTopology | _LightweightAcStrategyTopology | None = None,
    expected_hw8: tuple[int, int] | None = None,
) -> tuple[
    jax.Array,
    list[Any],
    tuple[jax.Array, jax.Array] | None,
    tuple[int, int],
]:
  """Extracts non-LLF AC coefficients in natural scan order per block & channel."""
  eff_hw8 = expected_hw8 or _infer_expected_hw8(topology=topology)
  q_ac_blocks, h8, w8 = _normalize_quant_ac_blocks_hwc(
      quant_ac, expected_hw8=eff_hw8
  )

  ac_idx_1x1, _, _ = get_canonical_ac_scan_tables(1, 1)

  if topology is None or len(topology.multi_groups) == 0:
    flat_all = q_ac_blocks.reshape(h8 * w8, 3, 64)
    ac_1x1 = flat_all[:, :, ac_idx_1x1]  # (H8*W8, 3, 63) — index 0 (LLF) excluded!
    return ac_1x1, [], None, (h8, w8)

  if topology.has_1x1:
    with jax.ensure_compile_time_eval():
      by_1x1 = jnp.asarray(topology.by_1x1, dtype=jnp.int32)
      bx_1x1 = jnp.asarray(topology.bx_1x1, dtype=jnp.int32)
    flat_1x1 = q_ac_blocks[by_1x1, bx_1x1].reshape(-1, 3, 64)
    ac_1x1 = flat_1x1[:, :, ac_idx_1x1]
    coords_1x1 = (by_1x1, bx_1x1)
  else:
    ac_1x1 = jnp.zeros((0, 3, 63), dtype=q_ac_blocks.dtype)
    coords_1x1 = (
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.int32),
    )

  multi_ac_list = []
  for grp in topology.multi_groups:
    cy = grp["cy"]
    cx = grp["cx"]
    cy_l = grp["cy_l"]
    cx_l = grp["cx_l"]
    with jax.ensure_compile_time_eval():
      by = jnp.asarray(grp["by"], dtype=jnp.int32)
      bx = jnp.asarray(grp["bx"], dtype=jnp.int32)
    m = by.shape[0]
    b = cy * cx
    dy = jnp.arange(cy)[None, :, None]
    dx = jnp.arange(cx)[None, None, :]
    sub_q = q_ac_blocks[by[:, None, None] + dy, bx[:, None, None] + dx]
    q_canon = jnp.transpose(sub_q, (0, 3, 1, 4, 2, 5)).reshape(m, 3, 64 * b)
    ac_idx_m, _, freq_band8_m = get_canonical_ac_scan_tables(cy_l, cx_l)
    ac_coeffs_m = q_canon[:, :, ac_idx_m]  # (M, 3, 63*B) — all B LLF entries excluded!
    multi_ac_list.append((ac_coeffs_m, freq_band8_m, b, by, bx, cy, cx))

  return ac_1x1, multi_ac_list, coords_1x1, (h8, w8)


@jax.jit
def _compute_scan_eob_and_categories(
    ac_coeffs: jax.Array,
    temp: float = 0.25,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Computes interior-zero weights, category weights, cumulative weights, and soft nzeros."""
  abs_q = jnp.where(jnp.signbit(ac_coeffs), -ac_coeffs, ac_coeffs)
  zero_prob, nz_prob, cat_weights, cdf_ge, log_z = _soft_category_weights(
      abs_q, temp=temp
  )

  rev_cumsum = jnp.cumsum(log_z[..., ::-1], axis=-1)[..., ::-1]
  log_suffix_zero = jnp.concatenate(
      [rev_cumsum[..., 1:], jnp.zeros_like(rev_cumsum[..., :1])], axis=-1
  )
  suffix_all_zero_prob = jnp.exp(log_suffix_zero)

  interior_zero_w = zero_prob * (1.0 - suffix_all_zero_prob)
  block_nzeros = jnp.sum(nz_prob, axis=-1)
  return interior_zero_w, cat_weights, cdf_ge, block_nzeros


@functools.partial(jax.checkpoint, prevent_cse=False)
def _compute_scan_band_summaries(
    ac_coeffs: jax.Array,
    one_hot_b: jax.Array,
    temp: float | jax.Array = 0.25,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Computes band-contracted `(z_band, cat_band, cdf_band, block_nzeros)` without materializing `(M, 3, K, S)`."""
  abs_q = jnp.where(jnp.signbit(ac_coeffs), -ac_coeffs, ac_coeffs)
  temp_f = jnp.asarray(temp, dtype=abs_q.dtype)
  ell = jnp.log2(abs_q + 0.5)
  log_z = jax.nn.log_sigmoid(-ell / temp_f) - jax.nn.log_sigmoid(
      jnp.asarray(1.0, dtype=abs_q.dtype) / temp_f
  )
  zero_prob = jnp.exp(log_z)
  rev_cumsum = jnp.cumsum(log_z[..., ::-1], axis=-1)[..., ::-1]
  log_suffix_zero = jnp.concatenate(
      [rev_cumsum[..., 1:], jnp.zeros_like(rev_cumsum[..., :1])], axis=-1
  )
  interior_zero_w = zero_prob * (1.0 - jnp.exp(log_suffix_zero))
  z_band = jnp.einsum("mck,kb->mcb", interior_zero_w, one_hot_b)

  cdf_bands: list[jax.Array] = []
  block_nzeros: jax.Array | None = None
  for s in range(1, MAX_AC_CATEGORY + 1):
    sig_at_zero_s = jax.nn.sigmoid(
        jnp.asarray(-float(s), dtype=abs_q.dtype) / temp_f
    )
    cdf_s = (
        jax.nn.sigmoid((ell - float(s - 1)) / temp_f) - sig_at_zero_s
    ) / (1.0 - sig_at_zero_s)
    if s == 1:
      block_nzeros = jnp.sum(cdf_s, axis=-1)
    cdf_bands.append(jnp.einsum("mck,kb->mcb", cdf_s, one_hot_b))
  assert block_nzeros is not None
  cdf_band = jnp.stack(cdf_bands, axis=-1)
  cat_band = cdf_band - jnp.concatenate(
      [cdf_band[..., 1:], jnp.zeros_like(cdf_band[..., :1])], axis=-1
  )
  return z_band, cat_band, cdf_band, block_nzeros


@jax.jit
def _compute_nzeros_rate_bits(
    nzeros_norm_grid: jax.Array,
    first_block_mask: jax.Array | None = None,
) -> jax.Array:
  """Models the per-block `Token(nzero_ctx, nzeros)` bit cost in libjxl."""
  _, h8, w8 = nzeros_norm_grid.shape
  nz_sg = jax.lax.stop_gradient(nzeros_norm_grid)

  top = jnp.pad(nz_sg[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
  left = jnp.pad(nz_sg[:, :, :-1], ((0, 0), (0, 0), (1, 0)))

  by_idx = jnp.arange(h8)[:, None]
  bx_idx = jnp.arange(w8)[None, :]
  pred = jnp.where(
      (by_idx == 0) & (bx_idx == 0),
      32.0,
      jnp.where(
          by_idx == 0,
          left,
          jnp.where(bx_idx == 0, top, 0.5 * (top + left)),
      ),
  )

  active_blk = nzeros_norm_grid / (nzeros_norm_grid + 1.0)
  excess_over_pred = jax.nn.softplus(nzeros_norm_grid - pred)
  blk_bits = (
      1.5 * active_blk
      + jnp.log2(1.0 + nzeros_norm_grid)
      + 0.35 * jnp.log2(1.0 + excess_over_pred)
  )
  if first_block_mask is not None:
    blk_bits = blk_bits * first_block_mask[None, :, :]
  return jnp.sum(blk_bits)


NUM_CLUSTER_STRAT_GROUPS: int = 3
NUM_CLUSTER_QF_BUCKETS: int = 2
NUM_CLUSTER_NZ_BUCKETS: int = 2
NUM_BLOCK_CLUSTERS: int = (
    NUM_CLUSTER_STRAT_GROUPS * NUM_CLUSTER_QF_BUCKETS * NUM_CLUSTER_NZ_BUCKETS
)


def _compute_multi_cluster_ac_token_bits(
    cluster_groups: list[
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ]
    ],
    eps: float = 1e-3,
) -> jax.Array:
  """Computes multi-cluster plug-in histogram AC entropy + ANS cluster header bits.

  Partitions blocks across `12 = 3 * 2 * 2` `(strategy_group, qf_bucket,
  nzeros_activity_bucket)` context clusters matching `FindBestBlockEntropyModel`
  and `NonZeroContext` in `libjxl`.
  """
  if not cluster_groups:
    return jnp.asarray(0.0, dtype=jnp.float32)

  rqf_all = jnp.concatenate([g[-3] for g in cluster_groups], axis=0)
  nz_all = jnp.concatenate([g[-2] for g in cluster_groups], axis=0)
  val_all = jnp.concatenate([g[-1] for g in cluster_groups], axis=0)

  val_sum = jnp.maximum(jnp.sum(val_all), 1.0)
  qf_mean = jnp.sum(rqf_all * val_all) / val_sum
  nz_mean = jnp.sum(nz_all * val_all) / val_sum

  rqf_valid = jnp.where(val_all > 0.5, rqf_all, qf_mean)
  nz_valid = jnp.where(val_all > 0.5, nz_all, nz_mean)

  qf_var = jnp.sum(((rqf_valid - qf_mean) ** 2) * val_all) / val_sum
  nz_var = jnp.sum(((nz_valid - nz_mean) ** 2) * val_all) / val_sum
  qf_std = jnp.sqrt(qf_var + 1e-6)
  nz_std = jnp.sqrt(nz_var + 1e-6)

  has_qf_var = (jnp.max(rqf_valid) - jnp.min(rqf_valid)) > 0.5
  has_nz_var = (jnp.max(nz_valid) - jnp.min(nz_valid)) > 1.0

  c_hist_zero = jnp.zeros((NUM_BLOCK_CLUSTERS, 3, NUM_FREQ_BANDS8), dtype=jnp.float32)
  c_hist_cats = jnp.zeros(
      (NUM_BLOCK_CLUSTERS, 3, NUM_FREQ_BANDS8, MAX_AC_CATEGORY),
      dtype=jnp.float32,
  )
  c_hist_cdfs = jnp.zeros(
      (NUM_BLOCK_CLUSTERS, 3, NUM_FREQ_BANDS8, MAX_AC_CATEGORY),
      dtype=jnp.float32,
  )

  for grp_item in cluster_groups:
    if len(grp_item) == 8:
      int_z, cat_w, cdf, one_hot_b, strat_w, rqf_m, nz_m, valid_m = grp_item
      z_band = jnp.einsum("mck,kb->mcb", int_z, one_hot_b)
      cat_band = jnp.einsum("mcks,kb->mcbs", cat_w, one_hot_b)
      cdf_band = jnp.einsum("mcks,kb->mcbs", cdf, one_hot_b)
    else:
      z_band, cat_band, cdf_band, strat_w, rqf_m, nz_m, valid_m = grp_item
    w_qf_hi = jnp.where(
        has_qf_var,
        jax.nn.sigmoid((rqf_m - qf_mean) / jnp.maximum(0.5 * qf_std, 0.5)),
        0.0,
    )
    w_qf = jnp.stack([1.0 - w_qf_hi, w_qf_hi], axis=-1)
    w_nz_hi = jnp.where(
        has_nz_var,
        jax.nn.sigmoid((nz_m - nz_mean) / jnp.maximum(0.5 * nz_std, 0.75)),
        0.0,
    )
    w_nz = jnp.stack([1.0 - w_nz_hi, w_nz_hi], axis=-1)
    w_clust = jax.lax.stop_gradient(
        (
            strat_w[:, :, None, None]
            * w_qf[:, None, :, None]
            * w_nz[:, None, None, :]
        ).reshape(-1, NUM_BLOCK_CLUSTERS)
        * valid_m[:, None]
    )
    c_hist_zero = c_hist_zero + jnp.einsum("mg,mcb->gcb", w_clust, z_band)
    c_hist_cats = c_hist_cats + jnp.einsum("mg,mcbs->gcbs", w_clust, cat_band)
    c_hist_cdfs = c_hist_cdfs + jnp.einsum("mg,mcbs->gcbs", w_clust, cdf_band)

  eps_c = float(eps) / float(NUM_BLOCK_CLUSTERS)
  s_idx = jnp.arange(1, MAX_AC_CATEGORY + 1, dtype=jnp.float32)[
      None, None, None, :
  ]
  prior_cats = eps_c * jnp.power(0.5, s_idx - 1.0)
  counts_zero = c_hist_zero[..., None] + eps_c
  counts_cats = c_hist_cats + prior_cats
  counts_all = jnp.concatenate([counts_zero, counts_cats], axis=-1)
  totals = jnp.sum(counts_all, axis=-1, keepdims=True)

  ac_token_entropy_bits = -jnp.sum(counts_all * jnp.log2(counts_all / totals))
  ac_extra_bits = jnp.sum(c_hist_cats * s_idx)
  forward_ac_bits = ac_token_entropy_bits + ac_extra_bits

  counts_prev = jnp.concatenate([counts_zero, counts_cats[..., :-1]], axis=-1)
  raw_step_cost = 1.0 + jnp.log2(
      jax.lax.stop_gradient(counts_prev / counts_cats)
  )
  min_step_deficit = jnp.maximum(0.0, 0.25 - raw_step_cost)
  monotonic_correction = jnp.sum(c_hist_cdfs * min_step_deficit)
  ac_bits = forward_ac_bits + (
      monotonic_correction - jax.lax.stop_gradient(monotonic_correction)
  )

  # ANS context map + per-active-cluster histogram table header overhead.
  tot_gc = jnp.sum(c_hist_zero, axis=-1) + jnp.sum(c_hist_cats, axis=(-2, -1))
  active_gc = tot_gc / (tot_gc + 24.0)
  extra_active_clusters = (
      jax.nn.softplus((jnp.sum(active_gc) - 3.0) * 4.0) / 4.0
  )
  cluster_header_bits = 24.0 * extra_active_clusters
  return ac_bits + cluster_header_bits


def soft_ac_rate_bits(
    quant_ac: jax.Array,
    topology: StaticVarDCTTopology | JxlVarDCTState | None = None,
    ac_strategy: np.ndarray | jax.Array | None = None,
    tau: float = 0.25,
    *,
    temp: float | None = None,
    raw_quant_field: jax.Array | None = None,
    llf_mask: jax.Array | None = None,
    eps: float = 1e-3,
    multi_cluster: bool = False,
    cluster_blend: float | jax.Array | None = None,
) -> jax.Array:
  """Differentiable VarDCT AC token & context bit-rate proxy for JPEG XL.

  Models `lib/jxl/enc_entropy_coder.cc:TokenizeCoefficients`:
  1. Canonical multi-block `AcStrategy` layout and strict exclusion of all
     `B = cy * cx` canonical LLF positions `[:, :, :cy_l, :cx_l]`.
  2. Per-block `nzeros` token + extra bits conditioned on `PredictFromTopAndLeft`
     spatial neighbor context.
  3. Early EOB loop termination (`while (nzeros != 0)`): trailing zeros after
     the last non-zero coefficient in the natural zigzag scan order cost 0 bits,
     while interior zeros (`z_k * (1 - S_k)`) emit symbol `0`.
  4. Per-(channel, frequency-band) Shannon entropy over interior zeros and
     dyadic magnitude categories `s in 1..11`, plus `s` sign/extra bits per
     category `s`, with optional multi-cluster block context partitioning
     `(strategy_group, qf_bucket, nzeros_activity_bucket)` + ANS cluster table
     header overhead when `multi_cluster=True` or `cluster_blend` is set.
  """
  del llf_mask
  eff_temp = float(temp) if temp is not None else float(tau)
  use_clusters = bool(multi_cluster) or (cluster_blend is not None)
  blend_w = (
      jnp.clip(jnp.asarray(cluster_blend, dtype=jnp.float32), 0.0, 1.0)
      if cluster_blend is not None
      else jnp.asarray(1.0 if multi_cluster else 0.0, dtype=jnp.float32)
  )
  resolved_topo = _resolve_topology(topology, ac_strategy)
  expected_hw8 = _infer_expected_hw8(
      topology=resolved_topo if resolved_topo is not None else topology,
      ac_strategy=ac_strategy,
      raw_quant_field=raw_quant_field,
  )

  if (
      resolved_topo is not None
      and len(resolved_topo.multi_groups) > 0
      and getattr(resolved_topo, "all_grid_aligned", False)
  ):
    q_ac_blocks, h8, w8 = _normalize_quant_ac_blocks_hwc(
        quant_ac, expected_hw8=expected_hw8
    )

    ac_idx_1x1, _, freq_band8_1x1 = get_canonical_ac_scan_tables(1, 1)
    one_hot_1x1 = jax.nn.one_hot(
        freq_band8_1x1, NUM_FREQ_BANDS8, dtype=jnp.float32
    )
    hist_zero = jnp.zeros((3, NUM_FREQ_BANDS8), dtype=jnp.float32)
    hist_cats = jnp.zeros(
        (3, NUM_FREQ_BANDS8, MAX_AC_CATEGORY), dtype=jnp.float32
    )
    hist_cdfs = jnp.zeros(
        (3, NUM_FREQ_BANDS8, MAX_AC_CATEGORY), dtype=jnp.float32
    )
    nzeros_norm_grid = jnp.zeros((3, h8, w8), dtype=jnp.float32)
    acs_traced = (
        jnp.asarray(ac_strategy, dtype=jnp.int32)
        if isinstance(ac_strategy, jax.core.Tracer)
        else None
    )
    if acs_traced is not None:
      mask_1x1 = (
          (acs_traced <= 3) | ((acs_traced >= 12) & (acs_traced <= 17))
      ).astype(jnp.float32)
      first_block_mask = mask_1x1
    else:
      with jax.ensure_compile_time_eval():
        mask_1x1 = jnp.asarray(resolved_topo.mask_1x1, dtype=jnp.float32)
        first_block_mask = jnp.asarray(
            resolved_topo.first_block_mask, dtype=jnp.float32
        )

    cluster_groups: list[Any] = []
    if use_clusters:
      if ac_strategy is not None:
        acs_grid_int = jnp.asarray(ac_strategy, dtype=jnp.int32)
      elif hasattr(resolved_topo, "ac_strategy"):
        with jax.ensure_compile_time_eval():
          acs_grid_int = jnp.asarray(
              getattr(resolved_topo, "ac_strategy"), dtype=jnp.int32
          )
      elif isinstance(topology, JxlVarDCTState):
        with jax.ensure_compile_time_eval():
          acs_grid_int = jnp.asarray(topology.ac_strategy, dtype=jnp.int32)
      else:
        acs_grid_int = jnp.zeros((h8, w8), dtype=jnp.int32)
      if raw_quant_field is not None:
        rqf_grid = jax.lax.stop_gradient(
            jnp.asarray(raw_quant_field, dtype=jnp.float32)
        )
      elif isinstance(topology, JxlVarDCTState):
        with jax.ensure_compile_time_eval():
          rqf_grid = jnp.asarray(topology.raw_quant_field, dtype=jnp.float32)
      else:
        rqf_grid = jnp.full((h8, w8), 16.0, dtype=jnp.float32)

    if resolved_topo.has_1x1:
      q_1x1 = (q_ac_blocks * mask_1x1[:, :, None, None, None]).reshape(
          h8 * w8, 3, 64
      )[:, :, ac_idx_1x1]
      zb_1x1, cb_1x1, cdfb_1x1, blk_nz_1x1 = (
          _compute_scan_band_summaries(q_1x1, one_hot_1x1, temp=eff_temp)
      )
      hist_zero = hist_zero + jnp.sum(zb_1x1, axis=0)
      hist_cats = hist_cats + jnp.sum(cb_1x1, axis=0)
      hist_cdfs = hist_cdfs + jnp.sum(cdfb_1x1, axis=0)
      nzeros_norm_grid = jnp.transpose(blk_nz_1x1, (1, 0)).reshape(3, h8, w8)
      if use_clusters:
        acs_1x1_flat = acs_grid_int.reshape(h8 * w8)
        is_s0 = (acs_1x1_flat == 0).astype(jnp.float32)
        strat_w_1x1 = jnp.stack(
            [is_s0, 1.0 - is_s0, jnp.zeros_like(is_s0)], axis=-1
        )
        cluster_groups.append((
            zb_1x1,
            cb_1x1,
            cdfb_1x1,
            strat_w_1x1,
            rqf_grid.reshape(h8 * w8),
            jax.lax.stop_gradient(jnp.mean(blk_nz_1x1, axis=-1)),
            mask_1x1.reshape(h8 * w8),
        ))

    for grp in resolved_topo.multi_groups:
      cy_m, cx_m = grp["cy"], grp["cx"]
      cy_l, cx_l = grp["cy_l"], grp["cx_l"]
      b_m = cy_m * cx_m
      nh_cell, nw_cell = grp["nh_cell"], grp["nw_cell"]
      h_lim, w_lim = grp["h_lim"], grp["w_lim"]
      if acs_traced is not None:
        cell_mask_f = (
            acs_traced[:h_lim:cy_m, :w_lim:cx_m] == int(grp["s"])
        ).astype(jnp.float32)
        first_block_mask = first_block_mask.at[:h_lim:cy_m, :w_lim:cx_m].add(
            cell_mask_f
        )
      else:
        with jax.ensure_compile_time_eval():
          cell_mask_f = jnp.asarray(grp["cell_mask_f"], dtype=jnp.float32)
      sub_q = (
          q_ac_blocks[:h_lim, :w_lim].reshape(
              nh_cell, cy_m, nw_cell, cx_m, 3, 8, 8
          )
          * cell_mask_f[:, None, :, None, None, None, None]
      )
      q_canon = jnp.transpose(sub_q, (0, 2, 4, 1, 5, 3, 6)).reshape(
          nh_cell * nw_cell, 3, 64 * b_m
      )
      ac_idx_m, _, freq_band8_m = get_canonical_ac_scan_tables(cy_l, cx_l)
      ac_coeffs_m = q_canon[:, :, ac_idx_m]
      one_hot_m = jax.nn.one_hot(
          freq_band8_m, NUM_FREQ_BANDS8, dtype=jnp.float32
      )
      zb_m, cb_m, cdfb_m, blk_nz_m = _compute_scan_band_summaries(
          ac_coeffs_m, one_hot_m, temp=eff_temp
      )
      hist_zero = hist_zero + jnp.sum(zb_m, axis=0)
      hist_cats = hist_cats + jnp.sum(cb_m, axis=0)
      hist_cdfs = hist_cdfs + jnp.sum(cdfb_m, axis=0)
      nz_cell = jnp.transpose(blk_nz_m / float(b_m), (1, 0)).reshape(
          3, nh_cell, 1, nw_cell, 1
      )
      nz_bcast = jnp.broadcast_to(
          nz_cell, (3, nh_cell, cy_m, nw_cell, cx_m)
      ).reshape(3, h_lim, w_lim)
      if h_lim == h8 and w_lim == w8:
        nzeros_norm_grid = nzeros_norm_grid + nz_bcast
      else:
        nzeros_norm_grid = nzeros_norm_grid.at[:, :h_lim, :w_lim].add(nz_bcast)
      if use_clusters:
        m_cells = nh_cell * nw_cell
        rqf_sub = rqf_grid[:h_lim, :w_lim].reshape(nh_cell, cy_m, nw_cell, cx_m)
        rqf_cell = jnp.mean(rqf_sub, axis=(1, 3)).reshape(m_cells)
        strat_w_m = jnp.broadcast_to(
            jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32), (m_cells, 3)
        )
        cluster_groups.append((
            zb_m,
            cb_m,
            cdfb_m,
            strat_w_m,
            rqf_cell,
            jax.lax.stop_gradient(jnp.mean(blk_nz_m, axis=-1) / float(b_m)),
            cell_mask_f.reshape(m_cells),
        ))

    s_idx = jnp.arange(1, MAX_AC_CATEGORY + 1, dtype=jnp.float32)[None, None, :]
    prior_cats = eps * jnp.power(0.5, s_idx - 1.0)
    counts_zero = hist_zero[..., None] + eps
    counts_cats = hist_cats + prior_cats
    counts_all = jnp.concatenate([counts_zero, counts_cats], axis=-1)
    totals = jnp.sum(counts_all, axis=-1, keepdims=True)

    ac_token_entropy_bits = -jnp.sum(counts_all * jnp.log2(counts_all / totals))
    ac_extra_bits = jnp.sum(hist_cats * s_idx)
    forward_ac_bits = ac_token_entropy_bits + ac_extra_bits

    counts_prev = jnp.concatenate([counts_zero, counts_cats[..., :-1]], axis=-1)
    raw_step_cost = 1.0 + jnp.log2(
        jax.lax.stop_gradient(counts_prev / counts_cats)
    )
    min_step_deficit = jnp.maximum(0.0, 0.25 - raw_step_cost)
    monotonic_correction = jnp.sum(hist_cdfs * min_step_deficit)
    ac_bits = forward_ac_bits + (
        monotonic_correction - jax.lax.stop_gradient(monotonic_correction)
    )
    if use_clusters:
      clustered_ac_bits = _compute_multi_cluster_ac_token_bits(
          cluster_groups, eps=eps
      )
      ac_bits = (1.0 - blend_w) * ac_bits + blend_w * clustered_ac_bits
    nzeros_bits = _compute_nzeros_rate_bits(
        nzeros_norm_grid,
        first_block_mask=first_block_mask,
    )
    return ac_bits + nzeros_bits

  ac_1x1, multi_ac_list, coords_1x1, (h8, w8) = _extract_ac_tokens_from_blocks(
      quant_ac, topology=resolved_topo, expected_hw8=expected_hw8
  )

  _, _, freq_band8_1x1 = get_canonical_ac_scan_tables(1, 1)
  one_hot_1x1 = jax.nn.one_hot(
      freq_band8_1x1, NUM_FREQ_BANDS8, dtype=jnp.float32
  )

  hist_zero = jnp.zeros((3, NUM_FREQ_BANDS8), dtype=jnp.float32)
  hist_cats = jnp.zeros(
      (3, NUM_FREQ_BANDS8, MAX_AC_CATEGORY), dtype=jnp.float32
  )
  hist_cdfs = jnp.zeros(
      (3, NUM_FREQ_BANDS8, MAX_AC_CATEGORY), dtype=jnp.float32
  )

  nzeros_norm_grid = jnp.zeros((3, h8, w8), dtype=jnp.float32)
  first_block_mask = jnp.ones((h8, w8), dtype=jnp.float32)
  cluster_groups_gen: list[Any] = []
  if use_clusters:
    if ac_strategy is not None:
      acs_grid_gen = jnp.asarray(ac_strategy, dtype=jnp.int32)
    elif resolved_topo is not None and hasattr(resolved_topo, "ac_strategy"):
      with jax.ensure_compile_time_eval():
        acs_grid_gen = jnp.asarray(
            getattr(resolved_topo, "ac_strategy"), dtype=jnp.int32
        )
    elif isinstance(topology, JxlVarDCTState):
      with jax.ensure_compile_time_eval():
        acs_grid_gen = jnp.asarray(topology.ac_strategy, dtype=jnp.int32)
    else:
      acs_grid_gen = jnp.zeros((h8, w8), dtype=jnp.int32)
    if raw_quant_field is not None:
      rqf_grid_gen = jax.lax.stop_gradient(
          jnp.asarray(raw_quant_field, dtype=jnp.float32)
      )
    elif isinstance(topology, JxlVarDCTState):
      with jax.ensure_compile_time_eval():
        rqf_grid_gen = jnp.asarray(topology.raw_quant_field, dtype=jnp.float32)
    else:
      rqf_grid_gen = jnp.full((h8, w8), 16.0, dtype=jnp.float32)

  if ac_1x1.shape[0] > 0:
    zb_1x1, cb_1x1, cdfb_1x1, blk_nz_1x1 = _compute_scan_band_summaries(
        ac_1x1, one_hot_1x1, temp=eff_temp
    )
    hist_zero = hist_zero + jnp.sum(zb_1x1, axis=0)
    hist_cats = hist_cats + jnp.sum(cb_1x1, axis=0)
    hist_cdfs = hist_cdfs + jnp.sum(cdfb_1x1, axis=0)

    if coords_1x1 is None:
      nzeros_norm_grid = jnp.transpose(blk_nz_1x1, (1, 0)).reshape(3, h8, w8)
      if use_clusters:
        acs_1x1_f = acs_grid_gen.reshape(h8 * w8)
        rqf_1x1_f = rqf_grid_gen.reshape(h8 * w8)
    else:
      by_1x1, bx_1x1 = coords_1x1
      nzeros_norm_grid = nzeros_norm_grid.at[:, by_1x1, bx_1x1].set(
          jnp.transpose(blk_nz_1x1, (1, 0))
      )
      if use_clusters:
        acs_1x1_f = acs_grid_gen[by_1x1, bx_1x1]
        rqf_1x1_f = rqf_grid_gen[by_1x1, bx_1x1]
    if use_clusters:
      is_s0_g = (acs_1x1_f == 0).astype(jnp.float32)
      strat_w_1x1_g = jnp.stack(
          [is_s0_g, 1.0 - is_s0_g, jnp.zeros_like(is_s0_g)], axis=-1
      )
      cluster_groups_gen.append((
          zb_1x1,
          cb_1x1,
          cdfb_1x1,
          strat_w_1x1_g,
          rqf_1x1_f,
          jax.lax.stop_gradient(jnp.mean(blk_nz_1x1, axis=-1)),
          jnp.ones((ac_1x1.shape[0],), dtype=jnp.float32),
      ))

  if len(multi_ac_list) > 0:
    first_block_mask = jnp.zeros((h8, w8), dtype=jnp.float32)
    if coords_1x1 is not None and ac_1x1.shape[0] > 0:
      first_block_mask = first_block_mask.at[coords_1x1[0], coords_1x1[1]].set(
          1.0
      )

    for (
        ac_coeffs_m,
        freq_band8_m,
        b_m,
        by_m,
        bx_m,
        cy_m,
        cx_m,
    ) in multi_ac_list:
      one_hot_m = jax.nn.one_hot(
          freq_band8_m, NUM_FREQ_BANDS8, dtype=jnp.float32
      )
      zb_m, cb_m, cdfb_m, blk_nz_m = _compute_scan_band_summaries(
          ac_coeffs_m, one_hot_m, temp=eff_temp
      )
      hist_zero = hist_zero + jnp.sum(zb_m, axis=0)
      hist_cats = hist_cats + jnp.sum(cb_m, axis=0)
      hist_cdfs = hist_cdfs + jnp.sum(cdfb_m, axis=0)

      nz_norm_m = jnp.transpose(blk_nz_m / float(b_m), (1, 0))
      dy = jnp.arange(cy_m)[None, :, None]
      dx = jnp.arange(cx_m)[None, None, :]
      nzeros_norm_grid = nzeros_norm_grid.at[
          :, by_m[:, None, None] + dy, bx_m[:, None, None] + dx
      ].set(nz_norm_m[:, :, None, None])
      first_block_mask = first_block_mask.at[by_m, bx_m].set(1.0)
      if use_clusters:
        m_cnt = ac_coeffs_m.shape[0]
        rqf_sub_m = rqf_grid_gen[
            by_m[:, None, None] + dy, bx_m[:, None, None] + dx
        ]
        rqf_m_mean = jnp.mean(rqf_sub_m, axis=(1, 2))
        strat_w_m_g = jnp.broadcast_to(
            jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32), (m_cnt, 3)
        )
        cluster_groups_gen.append((
            zb_m,
            cb_m,
            cdfb_m,
            strat_w_m_g,
            rqf_m_mean,
            jax.lax.stop_gradient(jnp.mean(blk_nz_m, axis=-1) / float(b_m)),
            jnp.ones((m_cnt,), dtype=jnp.float32),
        ))

  s_idx = jnp.arange(1, MAX_AC_CATEGORY + 1, dtype=jnp.float32)[None, None, :]
  prior_cats = eps * jnp.power(0.5, s_idx - 1.0)
  counts_zero = hist_zero[..., None] + eps
  counts_cats = hist_cats + prior_cats
  counts_all = jnp.concatenate([counts_zero, counts_cats], axis=-1)
  totals = jnp.sum(counts_all, axis=-1, keepdims=True)

  ac_token_entropy_bits = -jnp.sum(counts_all * jnp.log2(counts_all / totals))
  ac_extra_bits = jnp.sum(hist_cats * s_idx)
  forward_ac_bits = ac_token_entropy_bits + ac_extra_bits

  counts_prev = jnp.concatenate([counts_zero, counts_cats[..., :-1]], axis=-1)
  raw_step_cost = 1.0 + jnp.log2(
      jax.lax.stop_gradient(counts_prev / counts_cats)
  )
  min_step_deficit = jnp.maximum(0.0, 0.25 - raw_step_cost)
  monotonic_correction = jnp.sum(hist_cdfs * min_step_deficit)
  ac_bits = forward_ac_bits + (
      monotonic_correction - jax.lax.stop_gradient(monotonic_correction)
  )
  if use_clusters:
    clustered_ac_bits = _compute_multi_cluster_ac_token_bits(
        cluster_groups_gen, eps=eps
    )
    ac_bits = (1.0 - blend_w) * ac_bits + blend_w * clustered_ac_bits

  nzeros_bits = _compute_nzeros_rate_bits(
      nzeros_norm_grid,
      first_block_mask=first_block_mask if len(multi_ac_list) > 0 else None,
  )

  return ac_bits + nzeros_bits


# -----------------------------------------------------------------------------
# 3. Differentiable 2D ClampedGradient & Soft HybridUint(4, 2, 0) Model
# -----------------------------------------------------------------------------


def _precompute_hybrid_uint_420_signed_intervals(
    max_token: int = 44,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Precomputes signed residual intervals `[a_m, b_m]` mapped to `HybridUint(4, 2, 0)` tokens."""
  a_list: list[float] = []
  b_list: list[float] = []
  tok_list: list[int] = []
  nbits_list: list[float] = []

  for r_val in range(-8, 8):
    u_val = 2 * r_val if r_val >= 0 else (-2 * r_val - 1)
    a_list.append(float(r_val) - 0.5)
    b_list.append(float(r_val) + 0.5)
    tok_list.append(int(u_val))
    nbits_list.append(0.0)

  for t in range(16, max_token):
    k = t - 16
    e = (k >> 2) + 2
    m = k & 3
    u_lo = (4 + m) * (1 << e)
    u_hi = (4 + m + 1) * (1 << e) - 1

    r_pos_lo = u_lo // 2
    r_pos_hi = (u_hi - 1) // 2
    a_list.append(float(r_pos_lo) - 0.5)
    b_list.append(float(r_pos_hi) + 0.5)
    tok_list.append(t)
    nbits_list.append(float(e))

    r_neg_hi = -((u_lo + 2) // 2)
    r_neg_lo = -((u_hi + 1) // 2)
    a_list.append(float(r_neg_lo) - 0.5)
    b_list.append(float(r_neg_hi) + 0.5)
    tok_list.append(t)
    nbits_list.append(float(e))

  a_arr = np.array(a_list, dtype=np.float32)
  b_arr = np.array(b_list, dtype=np.float32)
  tok_arr = np.array(tok_list, dtype=np.int32)
  nbits_arr = np.array(nbits_list, dtype=np.float32)

  idx_min = int(np.argmin(a_arr))
  idx_max = int(np.argmax(b_arr))
  a_arr[idx_min] = -1e6
  b_arr[idx_max] = 1e6

  return a_arr, b_arr, tok_arr, nbits_arr


_NUM_TOKENS: int = 44
_INTERVAL_A_NP, _INTERVAL_B_NP, _INTERVAL_TOK_NP, _INTERVAL_NBITS_NP = (
    _precompute_hybrid_uint_420_signed_intervals(max_token=_NUM_TOKENS)
)
_INTERVAL_TO_TOK_MAT_NP = np.zeros(
    (len(_INTERVAL_TOK_NP), _NUM_TOKENS), dtype=np.float32
)
for _i, _t in enumerate(_INTERVAL_TOK_NP):
  _INTERVAL_TO_TOK_MAT_NP[_i, _t] = 1.0
_TOK_NBITS_NP = np.array(
    [0.0 if t < 16 else float(((t - 16) >> 2) + 2) for t in range(_NUM_TOKENS)],
    dtype=np.float32,
)
_ACS_INV_AREA_NP = np.array(
    [
        1.0 / float(ACS_COVERED_BLOCKS_X[s] * ACS_COVERED_BLOCKS_Y[s])
        for s in range(len(ACS_COVERED_BLOCKS_X))
    ],
    dtype=np.float32,
)
_ACS_INV_CY_NP = np.array(
    [
        1.0 / float(ACS_COVERED_BLOCKS_Y[s])
        for s in range(len(ACS_COVERED_BLOCKS_Y))
    ],
    dtype=np.float32,
)


@jax.jit
def soft_clamped_gradient_2d(plane: jax.Array, tau: float = 0.1) -> jax.Array:
  """Differentiable 2D `ClampedGradient` predictor matching `context_predict.h:385-402, 525-527`.

  Supports 2D `(H, W)` and arbitrary leading batch dimensions `(..., H, W)`.
  """
  plane_f = jnp.asarray(plane, dtype=jnp.float32)
  h, w = plane_f.shape[-2], plane_f.shape[-1]
  lead_shape = plane_f.shape[:-2]

  zeros_1x1 = jnp.zeros(lead_shape + (1, 1), dtype=jnp.float32)
  if w > 1:
    left_row0 = jnp.concatenate([zeros_1x1, plane_f[..., :1, :-1]], axis=-1)
  else:
    left_row0 = zeros_1x1

  if h > 1:
    left_col0 = plane_f[..., :-1, :1]
    if w > 1:
      left_rest = jnp.concatenate([left_col0, plane_f[..., 1:, :-1]], axis=-1)
      nw_rest = jnp.concatenate([left_col0, plane_f[..., :-1, :-1]], axis=-1)
    else:
      left_rest = left_col0
      nw_rest = left_col0
    left = jnp.concatenate([left_row0, left_rest], axis=-2)
    top = jnp.concatenate([left_row0, plane_f[..., :-1, :]], axis=-2)
    topleft = jnp.concatenate([left_row0, nw_rest], axis=-2)
  else:
    left = left_row0
    top = left_row0
    topleft = left_row0

  eps = jnp.maximum(jnp.float32(tau) * 0.05, 1e-4)
  diff_tl = top - left
  abs_tl = jnp.sqrt(diff_tl * diff_tl + eps * eps) - eps
  lo = 0.5 * (top + left - abs_tl)
  hi = 0.5 * (top + left + abs_tl)

  grad = top + left - topleft
  pred = (
      lo
      + eps * jax.nn.softplus((grad - lo) / eps)
      - eps * jax.nn.softplus((grad - hi) / eps)
  )
  return pred


@jax.jit
def soft_hybrid_uint_420_token_probs(
    residuals: jax.Array,
    tau: float = 0.1,
) -> tuple[jax.Array, jax.Array]:
  """Maps continuous signed residuals `(...)` to soft `HybridUint(4, 2, 0)` token probabilities `(..., 44)` and expected extra bits `(...)`."""
  r_flat = jnp.asarray(residuals, dtype=jnp.float32).reshape(-1, 1)
  a = jnp.array(_INTERVAL_A_NP, dtype=jnp.float32)[None, :]
  b = jnp.array(_INTERVAL_B_NP, dtype=jnp.float32)[None, :]
  to_tok = jnp.array(_INTERVAL_TO_TOK_MAT_NP, dtype=jnp.float32)
  tok_nbits = jnp.array(_TOK_NBITS_NP, dtype=jnp.float32)

  tau_sharp = jnp.maximum(jnp.float32(tau), 0.08)
  tau_wide = jnp.maximum(jnp.float32(tau) * 3.5, 0.35)
  alpha = 0.08

  w_sharp = jax.nn.sigmoid((r_flat - a) / tau_sharp) - jax.nn.sigmoid(
      (r_flat - b) / tau_sharp
  )
  w_wide = jax.nn.sigmoid((r_flat - a) / tau_wide) - jax.nn.sigmoid(
      (r_flat - b) / tau_wide
  )
  w_interval = (1.0 - alpha) * w_sharp + alpha * w_wide

  tok_probs_flat = w_interval @ to_tok
  tok_probs_flat = tok_probs_flat / jnp.maximum(
      jnp.sum(tok_probs_flat, axis=-1, keepdims=True), 1e-12
  )

  ebits_from_tok = tok_probs_flat @ tok_nbits
  abs_r = jnp.sqrt(r_flat[:, 0] * r_flat[:, 0] + tau_sharp * tau_sharp)
  tail_gate = jax.nn.sigmoid((abs_r - 7.5) / tau_wide)
  cont_ebits = jnp.log2(jnp.maximum(abs_r, 8.0)) - 1.0
  ebits_flat = (1.0 - 0.15 * tail_gate) * ebits_from_tok + (
      0.15 * tail_gate
  ) * cont_ebits

  out_shape = residuals.shape + (_NUM_TOKENS,)
  return tok_probs_flat.reshape(out_shape), ebits_flat.reshape(residuals.shape)


@jax.jit
def soft_stream_entropy_and_header_bits(
    tok_probs: jax.Array,
    ebits: jax.Array,
    weights: jax.Array | None = None,
    hist_overhead_per_active_sym: float = 6.2,
    base_hist_bits: float = 6.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes KL-debiased differentiable Shannon token bits, extra bits, and ANS histogram header overhead.

  Uses `cross_H - jax.lax.stop_gradient(self_H)` so constant streams have `0.0`
  phantom kernel self-entropy while preserving exact cross-entropy gradients.
  """
  probs_2d = tok_probs.reshape(-1, _NUM_TOKENS)
  eb_1d = ebits.reshape(-1)
  if weights is None:
    w = jnp.ones((probs_2d.shape[0],), dtype=jnp.float32)
  else:
    w = weights.reshape(-1).astype(jnp.float32)

  tok_idx = jnp.arange(_NUM_TOKENS, dtype=jnp.float32)
  prior_counts = 1e-3 * jnp.exp(-0.12 * tok_idx)
  counts = jnp.sum(probs_2d * w[:, None], axis=0)
  smoothed_counts = counts + prior_counts
  p_bar = smoothed_counts / jnp.maximum(jnp.sum(smoothed_counts), 1e-12)

  cross_h = -jnp.sum(counts * jnp.log2(jnp.maximum(p_bar, 1e-12)))
  self_h = -jnp.sum(
      w[:, None] * probs_2d * jnp.log2(jnp.maximum(probs_2d, 1e-12))
  )
  shannon_bits = cross_h - jax.lax.stop_gradient(self_h)
  extra_bits = jnp.sum(w * eb_1d)

  n_eff = jnp.maximum(jnp.sum(w), 1.0)
  excess_counts = jnp.maximum(0.0, counts - 0.012 * n_eff)
  active_syms = jnp.sum(1.0 - jnp.exp(-excess_counts / 1.5))
  non_singleton_gate = jax.nn.sigmoid((active_syms - 1.25) * 4.0)
  hist_bits = (
      base_hist_bits
      + non_singleton_gate
      * hist_overhead_per_active_sym
      * jnp.maximum(active_syms - 1.0, 0.0)
  )
  return shannon_bits, extra_bits, hist_bits


# -----------------------------------------------------------------------------
# 4. Differentiable VarDCT DC, AC Metadata & Header Bit Estimators
# -----------------------------------------------------------------------------


def soft_dc_rate_bits(
    dc_quant: jax.Array,
    ytox_dc: int | float | jax.Array = 0,
    ytob_dc: int | float | jax.Array = 0,
    ac_strategy: jax.Array | np.ndarray | None = None,
    tau: float = 0.1,
    color_factor: int = 84,
    dc_quant_is_raw: bool = False,
    base_correlation_x: float = 0.0,
    base_correlation_b: float = 1.0,
    return_breakdown: bool = False,
) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
  """Differentiable VarDCT DC rate estimator (`dc_bits = Dc + ModularDcGroup + ModularGlobal + ModularTree`)."""
  del ac_strategy
  dc_f = jnp.asarray(dc_quant, dtype=jnp.float32)
  _, h8, w8 = dc_f.shape
  ytox_dc_f = jnp.asarray(ytox_dc, dtype=jnp.float32)
  ytob_dc_f = jnp.asarray(ytob_dc, dtype=jnp.float32)

  q_y = dc_f[1]
  if dc_quant_is_raw:
    cfl_x = float(base_correlation_x) + ytox_dc_f / float(color_factor)
    cfl_b = float(base_correlation_b) + ytob_dc_f / float(color_factor)
    q_x = dc_f[0] - 8.0 * cfl_x * q_y
    q_b = dc_f[2] - 0.5 * cfl_b * q_y
  else:
    q_x = dc_f[0] - 8.0 * (ytox_dc_f / float(color_factor)) * q_y
    q_b = dc_f[2] - 0.5 * (ytob_dc_f / float(color_factor)) * q_y

  num_dc_groups_y = (h8 + 255) // 256
  num_dc_groups_x = (w8 + 255) // 256
  num_dc_groups = max(1, num_dc_groups_y * num_dc_groups_x)

  res_y_list = []
  res_x_list = []
  res_b_list = []
  for gy in range(num_dc_groups_y):
    y0, y1 = gy * 256, min((gy + 1) * 256, h8)
    for gx in range(num_dc_groups_x):
      x0, x1 = gx * 256, min((gx + 1) * 256, w8)
      sub_y = q_y[y0:y1, x0:x1]
      sub_x = q_x[y0:y1, x0:x1]
      sub_b = q_b[y0:y1, x0:x1]
      res_y_list.append(
          (sub_y - soft_clamped_gradient_2d(sub_y, tau=tau)).reshape(-1)
      )
      res_x_list.append(
          (sub_x - soft_clamped_gradient_2d(sub_x, tau=tau)).reshape(-1)
      )
      res_b_list.append(
          (sub_b - soft_clamped_gradient_2d(sub_b, tau=tau)).reshape(-1)
      )

  res_y = jnp.concatenate(res_y_list, axis=0)
  res_x = jnp.concatenate(res_x_list, axis=0)
  res_b = jnp.concatenate(res_b_list, axis=0)

  tp_y, eb_y = soft_hybrid_uint_420_token_probs(res_y, tau=tau)
  tp_x, eb_x = soft_hybrid_uint_420_token_probs(res_x, tau=tau)
  tp_b, eb_b = soft_hybrid_uint_420_token_probs(res_b, tau=tau)

  sh_y, ex_y, hist_y = soft_stream_entropy_and_header_bits(
      tp_y, eb_y, hist_overhead_per_active_sym=7.5, base_hist_bits=24.0
  )
  sh_x, ex_x, hist_x = soft_stream_entropy_and_header_bits(
      tp_x, eb_x, hist_overhead_per_active_sym=7.5, base_hist_bits=24.0
  )
  sh_b, ex_b, hist_b = soft_stream_entropy_and_header_bits(
      tp_b, eb_b, hist_overhead_per_active_sym=7.5, base_hist_bits=24.0
  )

  cc_nondefault = jnp.tanh(
      (ytox_dc_f * ytox_dc_f + ytob_dc_f * ytob_dc_f)
      / jnp.maximum(tau * tau, 1e-4)
  )
  dc_fixed_header = 2.0 + 50.0 * cc_nondefault + 6.0 * float(num_dc_groups)

  # Smooth residual tie-breaker guarantees every DC coefficient has a non-zero
  # gradient even at isolated constant patches.
  smooth_dc_tie = 1e-4 * (
      jnp.sum(jnp.sqrt(res_y * res_y + 1e-4))
      + jnp.sum(jnp.sqrt(res_x * res_x + 1e-4))
      + jnp.sum(jnp.sqrt(res_b * res_b + 1e-4))
  )

  dc_layer_bits = (
      dc_fixed_header
      + 1.10 * (sh_y + sh_x + sh_b)
      + (ex_y + ex_x + ex_b)
      + smooth_dc_tie
  )
  n_blocks = float(h8 * w8)
  size_factor = n_blocks / (n_blocks + 512.0)
  modular_tree_bits = jnp.float32(140.0 + 774.0 * size_factor)
  modular_global_bits = (40.0 + 180.0 * size_factor) + 1.45 * (
      hist_y + hist_x + hist_b
  )

  total_dc_bits = dc_layer_bits + modular_tree_bits + modular_global_bits
  if return_breakdown:
    return total_dc_bits, {
        "dc_layer_bits": dc_layer_bits,
        "modular_tree_bits": modular_tree_bits,
        "modular_global_bits": modular_global_bits,
        "dc_bits": total_dc_bits,
    }
  return total_dc_bits


def soft_ac_meta_rate_bits(
    raw_quant_field: jax.Array,
    ytox_map: jax.Array,
    ytob_map: jax.Array,
    ac_strategy: jax.Array | np.ndarray | None = None,
    epf_sharpness: jax.Array | None = None,
    tau: float = 0.1,
    global_scale: int | float | jax.Array = 8192,
    include_global_ac_headers: bool = True,
    return_breakdown: bool = False,
) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
  """Differentiable AC Metadata rate estimator (`ac_meta_bits = ControlFields + Order + Ac`).

  Matches `ModularFrameEncoder::AddACMetadata` (`lib/jxl/enc_modular.cc:1740-1806`),
  including vectorized `IsFirstBlock()` weighting for multi-block `AcStrategy`
  transforms and bi-scale logistic `HybridUint(4, 2, 0)` token entropy with KL
  self-entropy debiasing.
  """
  qf_f = jnp.asarray(raw_quant_field, dtype=jnp.float32)
  ytox_f = jnp.asarray(ytox_map, dtype=jnp.float32)
  ytob_f = jnp.asarray(ytob_map, dtype=jnp.float32)
  h8, w8 = qf_f.shape
  n_blocks = h8 * w8

  res_ytox = (ytox_f - soft_clamped_gradient_2d(ytox_f, tau=tau)).reshape(-1)
  res_ytob = (ytob_f - soft_clamped_gradient_2d(ytob_f, tau=tau)).reshape(-1)
  tp_ytox, eb_ytox = soft_hybrid_uint_420_token_probs(res_ytox, tau=tau)
  tp_ytob, eb_ytob = soft_hybrid_uint_420_token_probs(res_ytob, tau=tau)
  sh_ytox, ex_ytox, _ = soft_stream_entropy_and_header_bits(tp_ytox, eb_ytox)
  sh_ytob, ex_ytob, _ = soft_stream_entropy_and_header_bits(tp_ytob, eb_ytob)

  if ac_strategy is not None:
    acs_f = jnp.asarray(ac_strategy, dtype=jnp.float32).reshape(-1)
    s_idx = jnp.clip(
        jnp.round(jax.lax.stop_gradient(acs_f)).astype(jnp.int32),
        0,
        len(_ACS_INV_AREA_NP) - 1,
    )
    w_first = jnp.array(_ACS_INV_AREA_NP, dtype=jnp.float32)[s_idx]
    inv_cy = jnp.array(_ACS_INV_CY_NP, dtype=jnp.float32)[s_idx]
  else:
    acs_f = jnp.zeros((n_blocks,), dtype=jnp.float32)
    w_first = jnp.ones((n_blocks,), dtype=jnp.float32)
    inv_cy = jnp.ones((n_blocks,), dtype=jnp.float32)

  qf_m1_flat = (qf_f - 1.0).reshape(-1)
  qf_pred = jnp.concatenate([acs_f[:1], qf_m1_flat[:-1]], axis=0)
  res_qf = qf_m1_flat - qf_pred
  nz_jump = jax.nn.sigmoid((jnp.abs(res_qf) - 0.25) / jnp.maximum(tau, 0.08))
  w_qf = nz_jump * inv_cy + (1.0 - nz_jump) * w_first

  tp_qf, eb_qf = soft_hybrid_uint_420_token_probs(res_qf, tau=tau)
  sh_qf, ex_qf, _ = soft_stream_entropy_and_header_bits(
      tp_qf, eb_qf, weights=w_qf
  )

  tp_acs, eb_acs = soft_hybrid_uint_420_token_probs(acs_f, tau=tau)
  sh_acs, ex_acs, _ = soft_stream_entropy_and_header_bits(
      tp_acs, eb_acs, weights=w_first
  )

  if epf_sharpness is not None:
    epf_2d = jnp.asarray(epf_sharpness, dtype=jnp.float32)
  else:
    epf_2d = jnp.full((h8, w8), 2.0, dtype=jnp.float32)
  epf_flat = epf_2d.reshape(-1)
  tp_epf_raw, eb_epf_raw = soft_hybrid_uint_420_token_probs(epf_flat, tau=tau)
  sh_epf_raw, ex_epf_raw, _ = soft_stream_entropy_and_header_bits(
      tp_epf_raw, eb_epf_raw
  )
  epf_const_bits = 16.0 * jax.nn.sigmoid((jnp.mean(epf_flat) - 0.2) * 5.0)
  b_epf = epf_const_bits + sh_epf_raw + ex_epf_raw

  num_dc_groups = max(1, ((h8 + 255) // 256) * ((w8 + 255) // 256))
  count_bits = float(max(1, math.ceil(math.log2(max(2, n_blocks))))) * float(
      num_dc_groups
  )
  stream_hdr_bits = 4.0 * float(num_dc_groups)
  qf_init_jump = jnp.abs(res_qf[0])
  qf_x0_bits = jax.nn.sigmoid((qf_init_jump - 0.5) * 4.0) * (
      jnp.log2(jnp.maximum(jnp.sum(w_first), 2.0)) + 2.5
  )

  smooth_meta_tie = 1e-4 * (
      jnp.sum(jnp.sqrt(res_qf * res_qf + 1e-4))
      + jnp.sum(jnp.sqrt(res_ytox * res_ytox + 1e-4))
      + jnp.sum(jnp.sqrt(res_ytob * res_ytob + 1e-4))
      + 0.1 * jnp.sum(jnp.sqrt(jnp.square(qf_f - 0.25) + 1e-4))
      + 0.1 * jnp.sum(jnp.sqrt(jnp.square(ytox_f - 0.25) + 1e-4))
      + 0.1 * jnp.sum(jnp.sqrt(jnp.square(ytob_f - 0.25) + 1e-4))
  )

  footprint_token_factor = 0.35 + 0.65 * jnp.mean(w_first)
  control_fields_bits = footprint_token_factor * (
      count_bits
      + stream_hdr_bits
      + qf_x0_bits
      + 1.04
      * (
          (sh_ytox + ex_ytox)
          + (sh_ytob + ex_ytob)
          + (sh_qf + ex_qf)
          + (sh_acs + ex_acs)
          + b_epf
      )
      + smooth_meta_tie
  )

  gs_f = jnp.asarray(global_scale, dtype=jnp.float32)
  mean_qf = jnp.mean(jnp.maximum(qf_f, 1.0))
  eff_ac_scale = (gs_f / 65536.0) * mean_qf
  acs_active = sh_acs / jnp.maximum(jnp.sum(w_first), 1.0)
  size_factor = float(n_blocks) / (float(n_blocks) + 1024.0)
  order_bits = (80.0 + 240.0 * size_factor) + 480.0 * acs_active
  ac_histo_bits = (1800.0 + 10000.0 * size_factor) + (
      450.0 + 1800.0 * size_factor
  ) * jnp.log2(jnp.maximum(eff_ac_scale, 0.03))

  total_ac_meta_bits = control_fields_bits + (
      order_bits + ac_histo_bits if include_global_ac_headers else 0.0
  )
  if return_breakdown:
    return total_ac_meta_bits, {
        "control_fields_bits": control_fields_bits,
        "ytox_bits": sh_ytox + ex_ytox,
        "ytob_bits": sh_ytob + ex_ytob,
        "qf_bits": sh_qf + ex_qf + qf_x0_bits,
        "acs_bits": sh_acs + ex_acs,
        "epf_bits": b_epf,
        "order_bits": order_bits,
        "ac_histo_bits": ac_histo_bits,
        "ac_meta_bits": total_ac_meta_bits,
    }
  return total_ac_meta_bits


soft_ac_metadata_bits = soft_ac_meta_rate_bits


def soft_noise_header_bits(
    enable_noise: bool = False,
    noise_lut: Sequence[float] | jax.Array | None = None,
) -> jax.Array:
  """Differentiable JXL `NoiseParams` LUT header cost in bits (`80.0` bits when active)."""
  active = bool(enable_noise) or (noise_lut is not None)
  if not active:
    return jnp.asarray(0.0, dtype=jnp.float32)
  base_bits = jnp.asarray(80.0, dtype=jnp.float32)
  if noise_lut is not None:
    lut_arr = jnp.asarray(noise_lut, dtype=jnp.float32)
    base_bits = base_bits + 1e-3 * jnp.sum(
        jnp.sqrt(jnp.square(lut_arr) + 1e-8)
    )
  return base_bits


def soft_patch_dictionary_bits(
    patches: Sequence[PatchEntry | Any] | None = None,
    upsampling: int = 2,
) -> jax.Array:
  """Models the serialized bit cost of `PatchDictionary` (`LayerType::Dictionary`)."""
  del upsampling
  if not patches:
    return jnp.asarray(0.0, dtype=jnp.float32)

  ref_groups: dict[
      tuple[int, int, int, int, int], list[tuple[int, int, int]]
  ] = {}
  for p in patches:
    if isinstance(p, dict):
      ref_val = int(p.get("ref", p.get("ref_idx", 0)))
      key = (
          ref_val,
          int(p["x0"]),
          int(p["y0"]),
          int(p["xsize"]),
          int(p["ysize"]),
      )
      val = (
          int(p["x"]),
          int(p["y"]),
          int(p.get("blend_mode", PATCH_BLEND_REPLACE)),
      )
    else:
      ref_val = int(getattr(p, "ref", getattr(p, "ref_idx", 0)))
      key = (ref_val, int(p.x0), int(p.y0), int(p.xsize), int(p.ysize))
      val = (
          int(p.x),
          int(p.y),
          int(getattr(p, "blend_mode", PATCH_BLEND_REPLACE)),
      )
    ref_groups.setdefault(key, []).append(val)

  num_ref = len(ref_groups)
  num_total = len(patches)

  bits = 74.0 + math.log2(num_ref + 1.0)
  for (_, x0, y0, xsize, ysize), pos_list in ref_groups.items():
    count = len(pos_list)
    bits += (
        2.5
        + math.log2(abs(x0) + 2.0)
        + math.log2(abs(y0) + 2.0)
        + math.log2(max(xsize - 1, 0) + 2.0)
        + math.log2(max(ysize - 1, 0) + 2.0)
        + math.log2(max(count - 1, 0) + 1.5)
    )
    prev_x, prev_y = 0, 0
    for idx, (tx, ty, blend_mode) in enumerate(pos_list):
      if idx == 0:
        bits += 2.0 + math.log2(abs(tx) + 2.0) + math.log2(abs(ty) + 2.0)
      else:
        dx = tx - prev_x
        dy = ty - prev_y
        bits += (
            2.0
            + math.log2(2.0 * abs(dx) + 2.0)
            + math.log2(2.0 * abs(dy) + 2.0)
        )
      bits += 1.2 + (0.5 if blend_mode > 1 else 0.0)
      prev_x, prev_y = tx, ty

  bits += (
      2.2
      * num_total
      * math.log2(max(num_total, 2))
      / max(math.log2(num_total + 2.0), 1.0)
  )
  return jnp.asarray(bits, dtype=jnp.float32)


def _single_dequant_table_rate_bits(
    mat: jax.Array,
    def_mat: jax.Array,
    weight: float | jax.Array = 1.0,
) -> jax.Array:
  """Computes smooth table encoding overhead bits for one custom dequant table."""
  m_pos = jnp.maximum(jnp.asarray(mat, dtype=jnp.float32), 1e-8)
  d_pos = jnp.maximum(jnp.asarray(def_mat, dtype=jnp.float32), 1e-8)
  if m_pos.shape != d_pos.shape:
    if m_pos.size == d_pos.size:
      d_pos = d_pos.reshape(m_pos.shape)
    else:
      d_3d = (
          d_pos.reshape(3, 8, 8)
          if d_pos.ndim == 2 and d_pos.size == 192
          else d_pos
      )
      target_3d = (
          (3, 8, 8)
          if m_pos.ndim == 2 and m_pos.size == 192
          else m_pos.shape
      )
      d_pos = jax.image.resize(d_3d, target_3d, method="linear").reshape(
          m_pos.shape
      )
  log_ratio = jnp.log2(m_pos / d_pos)
  mean_sq = jnp.mean(jnp.square(log_ratio))
  nondefault_gate = 1.0 - jnp.exp(-mean_sq / 1e-6)
  lr_flat = log_ratio.reshape(m_pos.shape[0], -1)
  if lr_flat.shape[-1] > 1:
    diff_freq = lr_flat[:, 1:] - lr_flat[:, :-1]
    smooth_cost = jnp.mean(jnp.square(diff_freq))
  else:
    smooth_cost = mean_sq
  forward_bits = jnp.asarray(weight, dtype=jnp.float32) * (
      96.0 * nondefault_gate + 24.0 * mean_sq + 8.0 * smooth_cost
  )
  # Zero-forward-value tangent ensures non-zero gradient even at exact default
  tangent_probe = 1e-2 * jnp.sum(log_ratio)
  return forward_bits + (tangent_probe - jax.lax.stop_gradient(tangent_probe))


def soft_dequant_matrices_rate_bits(
    dequant_matrices: dict[int, jax.Array] | jax.Array | None = None,
    dc_quant: jax.Array | Sequence[float] | None = None,
    ac_strategy: jax.Array | np.ndarray | None = None,
) -> jax.Array:
  """Differentiable rate-cost proxy for non-default `DequantMatrices` and `dc_quant`."""
  total_bits = jnp.asarray(0.0, dtype=jnp.float32)

  if dc_quant is not None:
    dc_arr = jnp.asarray(dc_quant, dtype=jnp.float32)
    if dc_arr.ndim == 1 and dc_arr.shape[0] == 3:
      def_dc = jnp.asarray(K_DC_QUANT, dtype=jnp.float32)
      dc_pos = jnp.maximum(dc_arr, 1e-8)
      log_ratio_dc = jnp.log2(dc_pos / def_dc)
      mean_sq_dc = jnp.mean(jnp.square(log_ratio_dc))
      dc_gate = 1.0 - jnp.exp(-mean_sq_dc / 1e-6)
      dc_fwd = 48.0 * dc_gate + 16.0 * jnp.sum(jnp.square(log_ratio_dc))
      dc_tan = 1e-2 * jnp.sum(
          log_ratio_dc * jnp.array([1.0, 1.5, 1.2], dtype=jnp.float32)
      )
      total_bits = total_bits + dc_fwd + (dc_tan - jax.lax.stop_gradient(dc_tan))

  if dequant_matrices is None:
    return total_bits

  default_dq = get_default_dequant_matrices()
  active_strategies: set[int] | None = None
  is_traced_acs = ac_strategy is not None and isinstance(
      ac_strategy, jax.core.Tracer
  )
  if ac_strategy is not None and not is_traced_acs:
    acs_np = np.asarray(ac_strategy, dtype=np.int32)
    active_strategies = set(int(x) for x in np.unique(acs_np))

  if isinstance(dequant_matrices, Mapping):
    for s, mat in sorted(dequant_matrices.items()):
      s_int = int(s)
      def_mat = jnp.asarray(default_dq.get(s_int, default_dq[0]), dtype=jnp.float32)
      if is_traced_acs:
        w_s: float | jax.Array = jnp.where(
            (s_int == 0) | jnp.any(ac_strategy == s_int), 1.0, 0.0
        )
      else:
        w_s = (
            1.0
            if (active_strategies is None or s_int in active_strategies)
            else 0.15
        )
      total_bits = total_bits + _single_dequant_table_rate_bits(
          jnp.asarray(mat, dtype=jnp.float32), def_mat, weight=w_s
      )
    return total_bits

  dq_arr = jnp.asarray(dequant_matrices, dtype=jnp.float32)
  if dq_arr.ndim >= 3 and dq_arr.shape[1] == 3 and (
      dq_arr.shape[2:] == (64,) or dq_arr.shape[2:] == (8, 8)
  ):
    n_slots = dq_arr.shape[0]
    for s in range(n_slots):
      s_idx = min(s, len(ACS_COVERED_BLOCKS_X) - 1)
      def_s = jnp.asarray(default_dq[s_idx], dtype=jnp.float32)
      w_s = (
          1.0
          if (active_strategies is None or s in active_strategies)
          else 0.15
      )
      total_bits = total_bits + _single_dequant_table_rate_bits(
          dq_arr[s], def_s, weight=w_s
      )
    return total_bits

  if dq_arr.ndim >= 2 and dq_arr.shape[0] == 3 and dq_arr.size == 3 * 64:
    def_0 = jnp.asarray(default_dq[0], dtype=jnp.float32)
    return total_bits + _single_dequant_table_rate_bits(dq_arr, def_0, weight=1.0)

  # Fallback for scale-factor arrays (e.g. (18, 3) or (3,))
  dq_pos = jnp.maximum(dq_arr, 1e-8)
  log_scale = jnp.log2(dq_pos)
  mean_sq_scale = jnp.mean(jnp.square(log_scale))
  scale_gate = 1.0 - jnp.exp(-mean_sq_scale / 1e-6)
  scale_fwd = 96.0 * scale_gate + 24.0 * jnp.sum(jnp.square(log_scale))
  scale_tan = 1e-2 * jnp.sum(log_scale)
  return total_bits + scale_fwd + (scale_tan - jax.lax.stop_gradient(scale_tan))


def soft_header_rate_bits(
    height: int = 256,
    width: int = 256,
    global_scale: int | float | jax.Array = 8192,
    quant_dc: int | float | jax.Array = 16,
    epf_iters: int = 1,
    gab: int | bool = 1,
    has_noise: bool = False,
    num_patches: int = 0,
    upsampling: int = 1,
    is_reference_only: bool = False,
    *,
    bh: int | None = None,
    bw: int | None = None,
    enable_noise: bool = False,
    noise_lut: Sequence[float] | jax.Array | None = None,
    patches: Sequence[PatchEntry | Any] | None = None,
    dequant_matrices: dict[int, jax.Array] | jax.Array | None = None,
    dc_quant: jax.Array | Sequence[float] | None = None,
    ac_strategy: jax.Array | np.ndarray | None = None,
) -> jax.Array:
  """Differentiable proxy for JXL frame & codestream headers (`Header + Toc + Quant + Dictionary + Splines + Noise`)."""
  h8 = int(bh) if bh is not None else (int(height) + 7) // 8
  w8 = int(bw) if bw is not None else (int(width) + 7) // 8
  num_groups = ((h8 + 31) // 32) * ((w8 + 31) // 32)
  num_dc_groups = ((h8 + 255) // 256) * ((w8 + 255) // 256)
  num_toc_entries = (
      1
      if (num_groups == 1 and num_dc_groups == 1)
      else (2 + num_dc_groups + num_groups)
  )

  codestream_hdr_bits = 0.0 if is_reference_only else 29.0
  frame_hdr_bits = 24.0
  if int(epf_iters) > 0:
    frame_hdr_bits += 12.0
  if int(gab) > 0:
    frame_hdr_bits += 2.0
  if int(upsampling) > 1:
    frame_hdr_bits += 4.0

  gs_f = jnp.asarray(global_scale, dtype=jnp.float32)
  qdc_f = jnp.asarray(quant_dc, dtype=jnp.float32)

  toc_base = 14.0 + 14.22 * float(num_toc_entries)
  toc_large_group_extra = float(
      num_groups + num_dc_groups
  ) * 8.0 * jax.nn.sigmoid((gs_f - 2600.0) / 450.0) + float(
      num_dc_groups
  ) * 8.0 * jax.nn.sigmoid((gs_f - 14000.0) / 2000.0)
  toc_bits = toc_base + toc_large_group_extra

  gs_bits = (
      13.0
      + 1.0 * jax.nn.sigmoid((gs_f - 4096.5) / 150.0)
      + 4.0 * jax.nn.sigmoid((gs_f - 8192.5) / 300.0)
  )
  qdc_is_16 = jnp.exp(-((qdc_f - 16.0) ** 2) / 0.25)
  qdc_bits = 2.0 * qdc_is_16 + (1.0 - qdc_is_16) * (
      7.0
      + 3.0 * jax.nn.sigmoid((qdc_f - 32.5) / 2.0)
      + 8.0 * jax.nn.sigmoid((qdc_f - 256.5) / 10.0)
  )
  quant_bits = 2.0 + gs_bits + qdc_bits

  noise_active = bool(has_noise) or bool(enable_noise) or (noise_lut is not None)
  noise_bits = soft_noise_header_bits(
      enable_noise=noise_active, noise_lut=noise_lut
  )
  if patches is not None:
    dict_bits = soft_patch_dictionary_bits(patches=patches)
  elif num_patches > 0:
    dict_bits = jnp.float32(48.0 + 38.0 * float(num_patches))
  else:
    dict_bits = jnp.float32(0.0)

  dequant_table_bits = soft_dequant_matrices_rate_bits(
      dequant_matrices=dequant_matrices,
      dc_quant=dc_quant,
      ac_strategy=ac_strategy,
  )

  return (
      jnp.float32(codestream_hdr_bits + frame_hdr_bits)
      + toc_bits
      + quant_bits
      + noise_bits
      + dict_bits
      + dequant_table_bits
  )


# -----------------------------------------------------------------------------
# 5. Differentiable Modular `ExtraChannel::kAlpha` Rate Proxy
# -----------------------------------------------------------------------------


def _box_downsample_2d(plane: jax.Array, factor: int) -> jax.Array:
  """Differentiable `factor x factor` box average pooling on a 2D `(H, W)` plane."""
  if factor <= 1:
    return plane
  h, w = plane.shape
  pad_h = (factor - (h % factor)) % factor
  pad_w = (factor - (w % factor)) % factor
  if pad_h > 0 or pad_w > 0:
    plane = jnp.pad(plane, ((0, pad_h), (0, pad_w)), mode="edge")
  h_ds = plane.shape[0] // factor
  w_ds = plane.shape[1] // factor
  return jnp.mean(plane.reshape(h_ds, factor, w_ds, factor), axis=(1, 3))


def _alpha_soft_zero_and_active(
    c: jax.Array,
    tau: float = 0.15,
) -> tuple[jax.Array, jax.Array]:
  """Smooth indicator of zero vs active residuals for Modular alpha channel."""
  c_f = jnp.asarray(c, dtype=jnp.float32)
  tau_safe = jnp.maximum(jnp.asarray(tau, dtype=jnp.float32), 1e-4)
  eps_sq = 1e-4
  abs_c = jnp.sqrt(jnp.square(c_f) + eps_sq)
  sig_raw = jax.nn.sigmoid((abs_c - 0.5) / tau_safe)
  sig_0 = jax.nn.sigmoid((jnp.sqrt(eps_sq) - 0.5) / tau_safe)
  p_nz = jnp.clip((sig_raw - sig_0) / (1.0 - sig_0), 0.0, 1.0)
  p_zero = 1.0 - p_nz
  return p_zero, p_nz


def _alpha_soft_category_probs(
    c: jax.Array,
    max_cat: int = 8,
    tau: float = 0.15,
) -> jax.Array:
  """Soft probability distribution over magnitude categories `{0, 1, ..., max_cat}` for Modular alpha."""
  c_f = jnp.asarray(c, dtype=jnp.float32)
  p_zero, p_nz = _alpha_soft_zero_and_active(c_f, tau=tau)
  abs_c = jnp.sqrt(jnp.square(c_f) + 1e-4)
  cont_cat = jnp.clip(
      jnp.log2(jnp.maximum(abs_c, 0.5)) + 0.5,
      1.0,
      float(max_cat),
  )
  cats = jnp.arange(1, max_cat + 1, dtype=jnp.float32)
  diff = cont_cat[..., None] - cats
  sigma = 0.55
  weights = jnp.exp(-0.5 * jnp.square(diff / sigma))
  weights = weights / jnp.maximum(
      jnp.sum(weights, axis=-1, keepdims=True), 1e-12
  )
  p_cats = p_nz[..., None] * weights
  return jnp.concatenate([p_zero[..., None], p_cats], axis=-1)


def soft_bounded_code_lengths(
    counts: jax.Array,
    total: jax.Array,
    max_bits: float = 15.0,
    eps: float = 1e-3,
) -> jax.Array:
  """Differentiable ANS/Huffman code length estimator bounded by `max_bits`."""
  num_bins = counts.shape[-1]
  probs = (counts + eps) / jnp.maximum(total + num_bins * eps, 1e-6)
  min_prob = float(2.0 ** (-max_bits))
  probs_clipped = jnp.clip(probs, min_prob, 1.0)
  return -jnp.log2(probs_clipped)


@jax.jit
def _soft_alpha_channel_bits_core(
    arr: jax.Array,
    eff_tau: float,
    max_val: float,
) -> jax.Array:
  """JIT-compiled core for 8-bit Modular `ExtraChannel::kAlpha` bit estimation."""
  a_scaled = jnp.clip(arr, -0.05, 1.05) * max_val

  pred = soft_clamped_gradient_2d(a_scaled, tau=eff_tau)
  raw_res = a_scaled - pred

  # Modular WP adaptation absorbs consistent neighbor ramp bias
  res_pad = jnp.pad(raw_res, ((1, 0), (1, 0)), mode="edge")
  res_bias = 0.5 * (res_pad[:-1, 1:] + res_pad[1:, :-1])
  consistency = jnp.exp(-0.5 * jnp.square((raw_res - res_bias) / 0.50))
  res = raw_res - 0.95 * consistency * res_bias

  # Local activity contexts matching Modular MA tree context splitting
  padded = jnp.pad(a_scaled, ((1, 0), (1, 0)), mode="edge")
  n = padded[:-1, 1:]
  west = padded[1:, :-1]
  nw = padded[:-1, :-1]
  activity = jnp.sqrt(jnp.square(n - nw) + 1e-4) + jnp.sqrt(
      jnp.square(west - nw) + 1e-4
  )
  act_log = jnp.clip(jnp.log2(activity + 1.0), 0.0, 6.0) / 2.0
  ctx_centers = jnp.arange(4, dtype=jnp.float32)
  ctx_weights = jnp.exp(
      -0.5 * jnp.square((act_log[..., None] - ctx_centers) / 0.65)
  )
  ctx_weights = ctx_weights / jnp.maximum(
      jnp.sum(ctx_weights, axis=-1, keepdims=True), 1e-12
  )

  _, p_nz = _alpha_soft_zero_and_active(res, tau=eff_tau)
  cat_probs = _alpha_soft_category_probs(res, max_cat=8, tau=eff_tau)

  ctx_counts = jnp.einsum("hwc,hwm->cm", ctx_weights, cat_probs)
  ctx_totals = jnp.sum(ctx_counts, axis=-1, keepdims=True)
  ctx_codelens = soft_bounded_code_lengths(ctx_counts, ctx_totals)
  hist_entropy_bits = jnp.sum(ctx_counts * ctx_codelens)

  p_pos = jax.nn.sigmoid(res / jnp.maximum(eff_tau, 1e-3))
  pos_count = jnp.sum(p_nz * p_pos)
  neg_count = jnp.sum(p_nz * (1.0 - p_pos))
  tot_nz = jnp.maximum(pos_count + neg_count, 1e-4)
  prob_pos = jnp.clip(pos_count / tot_nz, 1e-4, 1.0 - 1e-4)
  sign_entropy_per_nz = -prob_pos * jnp.log2(prob_pos) - (
      1.0 - prob_pos
  ) * jnp.log2(1.0 - prob_pos)

  abs_res = jnp.sqrt(jnp.square(res) + 1e-4)
  extra_mag_bits = jnp.sum(
      p_nz
      * (
          sign_entropy_per_nz
          + 0.90 * jnp.log2(jnp.maximum(abs_res - 0.5, 1.0))
      )
  )

  smooth_tie = 1e-3 * jnp.sum(abs_res) + 1e-4 * jnp.sum(
      jnp.sqrt(jnp.square(a_scaled - 0.5 * max_val) + 1.0)
  )

  p_bin_active = 1.0 - jnp.exp(-0.5 * ctx_counts)
  header_and_tree_bits = 145.0 + 4.5 * jnp.sum(p_bin_active)

  return hist_entropy_bits + extra_mag_bits + header_and_tree_bits + smooth_tie


def soft_alpha_channel_bits(
    alpha_01: jax.Array,
    dim_shift: int = 0,
    tau: float = 0.15,
    bits: int = 8,
    *,
    full_hw: tuple[int, int] | None = None,
) -> jax.Array:
  """Differentiable JXL Modular `ExtraChannel::kAlpha` rate proxy in bits.

  Supports both already-downsampled `alpha_01` of shape
  `(ceil(H >> dim_shift), ceil(W >> dim_shift))` and full-resolution `(H, W)`
  `alpha_01` (when `full_hw=(H, W)` is provided or when `dim_shift > 0` is
  explicitly passed on a full-resolution array). Uses 2D `ClampedGradient`
  residuals on `alpha_01 * (2^bits - 1)` with Modular WP ramp-bias adaptation,
  4-context activity-weighted bounded code-length entropy, sign entropy, and
  Modular tree/header overhead.
  """
  arr = jnp.asarray(alpha_01, dtype=jnp.float32)
  if arr.ndim == 3 and arr.shape[0] == 1:
    arr = arr[0]
  elif arr.ndim == 3 and arr.shape[2] == 1:
    arr = arr[:, :, 0]
  if arr.ndim != 2:
    raise ValueError(f"Expected 2D alpha_01 array, got shape {arr.shape}")

  shift = int(dim_shift)
  if shift > 0:
    factor = 1 << shift
    if full_hw is not None:
      exp_h = (int(full_hw[0]) + factor - 1) // factor
      exp_w = (int(full_hw[1]) + factor - 1) // factor
      if arr.shape == tuple(full_hw) and arr.shape != (exp_h, exp_w):
        arr = _box_downsample_2d(arr, factor)
    else:
      arr = _box_downsample_2d(arr, factor)

  eff_tau = max(float(tau), 0.15)
  max_val = float((1 << int(bits)) - 1)
  return _soft_alpha_channel_bits_core(arr, eff_tau, max_val)


# -----------------------------------------------------------------------------
# 6. Unified Single-Frame & Multi-Resolution Composite JXL Rate Proxies
# -----------------------------------------------------------------------------


def _extract_state_quant_ac(state: Any) -> jax.Array:
  if hasattr(state, "quant_ac"):
    return jnp.asarray(state.quant_ac, dtype=jnp.float32)
  return jnp.asarray(state.ac_quant, dtype=jnp.float32)


def soft_jxl_rate_proxy(
    state: JxlVarDCTState | None = None,
    quant_ac: jax.Array | None = None,
    dc_quant: jax.Array | None = None,
    raw_quant_field: jax.Array | None = None,
    ytox_map: jax.Array | None = None,
    ytob_map: jax.Array | None = None,
    ytox_dc: int | float | jax.Array | None = None,
    ytob_dc: int | float | jax.Array | None = None,
    enable_noise: bool = False,
    noise_lut: Sequence[float] | jax.Array | None = None,
    patch_bits: float | jax.Array = 0.0,
    alpha_bits: float | jax.Array = 0.0,
    tau: float = 0.25,
    return_breakdown: bool = False,
    *,
    quant_ac_override: jax.Array | None = None,
    ac_strategy: jax.Array | np.ndarray | None = None,
    epf_sharpness: jax.Array | None = None,
    global_scale: int | float | jax.Array | None = None,
    quant_dc: int | float | jax.Array | None = None,
    topology: StaticVarDCTTopology | None = None,
    patches: Sequence[PatchEntry | Any] | None = None,
    alpha_01: jax.Array | None = None,
    dim_shift: int = 0,
    upsampling: int = 1,
    is_reference_only: bool = False,
    dequant_matrices: dict[int, jax.Array] | jax.Array | None = None,
    dc_quant_weights: jax.Array | Sequence[float] | None = None,
    multi_cluster: bool = False,
    cluster_blend: float | jax.Array | None = None,
    params_override: Mapping[str, Any] | None = None,
) -> jax.Array | RateBreakdown:
  """Unified differentiable JXL bit-rate proxy in bits."""
  p = dict(params_override) if params_override is not None else {}
  if quant_ac is None and quant_ac_override is not None:
    quant_ac = quant_ac_override
  if quant_ac is None and "quant_ac" in p:
    quant_ac = p["quant_ac"]
  if quant_ac is None:
    if state is None:
      raise ValueError("Either `state` or `quant_ac` must be provided.")
    quant_ac = _extract_state_quant_ac(state)
  expected_hw8 = _infer_expected_hw8(
      state=state,
      topology=topology,
      ac_strategy=(
          ac_strategy if ac_strategy is not None else p.get("ac_strategy")
      ),
      raw_quant_field=(
          raw_quant_field
          if raw_quant_field is not None
          else p.get("raw_quant_field")
      ),
  )
  ac_5d = normalize_quant_ac_shape(quant_ac, expected_hw8=expected_hw8)
  _, bh, bw, _, _ = ac_5d.shape
  ac_3hw = jnp.transpose(ac_5d, (0, 1, 3, 2, 4)).reshape(3, bh * 8, bw * 8)

  height = int(state.height) if state is not None else bh * 8
  width = int(state.width) if state is not None else bw * 8
  if global_scale is None:
    global_scale = p.get(
        "global_scale", int(state.global_scale) if state is not None else 8192
    )
  if quant_dc is None:
    quant_dc = p.get(
        "quant_dc", int(state.quant_dc) if state is not None else 16
    )
  epf_iters = int(
      p.get("epf_iters", state.epf_iters if state is not None else 1)
  )
  gab = int(bool(p.get("gab", state.gab if state is not None else 1)))
  color_factor = int(
      p.get("color_factor", state.color_factor if state is not None else 84)
  )

  if dequant_matrices is None and "dequant_matrices" in p:
    dequant_matrices = p["dequant_matrices"]
  if dc_quant_weights is None and "dc_quant_weights" in p:
    dc_quant_weights = p["dc_quant_weights"]

  if dc_quant is None and "dc_quant" in p:
    dc_quant = p["dc_quant"]
  if dc_quant is not None:
    dc_q_arr = jnp.asarray(dc_quant, dtype=jnp.float32)
    if dc_q_arr.ndim == 1 and dc_q_arr.shape[0] == 3:
      if dc_quant_weights is None:
        dc_quant_weights = dc_q_arr
      dc_quant = (
          jnp.asarray(state.dc_quant, dtype=jnp.float32)
          if state is not None
          else jnp.zeros((3, bh, bw), dtype=jnp.float32)
      )
    else:
      dc_quant = dc_q_arr
  else:
    dc_quant = (
        jnp.asarray(state.dc_quant, dtype=jnp.float32)
        if state is not None
        else jnp.zeros((3, bh, bw), dtype=jnp.float32)
    )

  if raw_quant_field is None and "raw_quant_field" in p:
    raw_quant_field = p["raw_quant_field"]
  if raw_quant_field is None:
    raw_quant_field = (
        jnp.asarray(state.raw_quant_field, dtype=jnp.float32)
        if state is not None
        else jnp.full((bh, bw), 16.0, dtype=jnp.float32)
    )
  else:
    raw_quant_field = jnp.asarray(raw_quant_field, dtype=jnp.float32)

  th, tw = (bh + 7) // 8, (bw + 7) // 8
  if ytox_map is None and "ytox_map" in p:
    ytox_map = p["ytox_map"]
  if ytox_map is None:
    ytox_map = (
        jnp.asarray(state.ytox_map, dtype=jnp.float32)
        if state is not None
        else jnp.zeros((th, tw), dtype=jnp.float32)
    )
  else:
    ytox_map = jnp.asarray(ytox_map, dtype=jnp.float32)

  if ytob_map is None and "ytob_map" in p:
    ytob_map = p["ytob_map"]
  if ytob_map is None:
    ytob_map = (
        jnp.asarray(state.ytob_map, dtype=jnp.float32)
        if state is not None
        else jnp.zeros((th, tw), dtype=jnp.float32)
    )
  else:
    ytob_map = jnp.asarray(ytob_map, dtype=jnp.float32)

  if ytox_dc is None and "ytox_dc" in p:
    ytox_dc = p["ytox_dc"]
  if ytox_dc is None:
    ytox_dc_val: int | float | jax.Array = (
        int(state.ytox_dc) if state is not None else 0
    )
  else:
    ytox_dc_val = ytox_dc

  if ytob_dc is None and "ytob_dc" in p:
    ytob_dc = p["ytob_dc"]
  if ytob_dc is None:
    ytob_dc_val: int | float | jax.Array = (
        int(state.ytob_dc) if state is not None else 0
    )
  else:
    ytob_dc_val = ytob_dc

  if ac_strategy is None and "ac_strategy" in p:
    ac_strategy = p["ac_strategy"]
  acs_source = (
      ac_strategy
      if ac_strategy is not None
      else (state.ac_strategy if state is not None else None)
  )
  if epf_sharpness is None and "epf_sharpness" in p:
    epf_sharpness = p["epf_sharpness"]
  if epf_sharpness is None and state is not None:
    epf_sharpness = jnp.asarray(state.epf_sharpness, dtype=jnp.float32)

  tau_ac = float(tau)
  tau_mod = min(float(tau), 0.12)

  eff_multi_cluster = bool(p.get("multi_cluster", multi_cluster))
  eff_cluster_blend = p.get("cluster_blend", cluster_blend)
  ac_bits_val = soft_ac_rate_bits(
      ac_3hw,
      topology=topology,
      ac_strategy=acs_source,
      tau=tau_ac,
      raw_quant_field=raw_quant_field,
      multi_cluster=eff_multi_cluster,
      cluster_blend=eff_cluster_blend,
  )
  dc_bits_val = soft_dc_rate_bits(
      dc_quant,
      ytox_dc=ytox_dc_val,
      ytob_dc=ytob_dc_val,
      ac_strategy=acs_source,
      tau=tau_mod,
      color_factor=color_factor,
  )
  ac_meta_bits_val = soft_ac_meta_rate_bits(
      raw_quant_field,
      ytox_map,
      ytob_map,
      ac_strategy=acs_source,
      epf_sharpness=epf_sharpness,
      tau=tau_mod,
      global_scale=global_scale,
      include_global_ac_headers=False,
  )

  noise_bits_val = soft_noise_header_bits(
      enable_noise=enable_noise, noise_lut=noise_lut
  )
  if patches is not None:
    patch_bits_val = soft_patch_dictionary_bits(
        patches=patches, upsampling=upsampling
    )
  else:
    patch_bits_val = jnp.asarray(patch_bits, dtype=jnp.float32)

  base_header_bits = soft_header_rate_bits(
      height=height,
      width=width,
      global_scale=global_scale,
      quant_dc=quant_dc,
      epf_iters=epf_iters,
      gab=gab,
      has_noise=False,
      num_patches=0,
      upsampling=upsampling,
      is_reference_only=is_reference_only,
      dequant_matrices=dequant_matrices,
      dc_quant=dc_quant_weights,
      ac_strategy=acs_source,
  )
  header_bits_val = base_header_bits + noise_bits_val + patch_bits_val

  if alpha_01 is not None:
    alpha_bits_val = soft_alpha_channel_bits(
        alpha_01,
        dim_shift=dim_shift,
        tau=tau_mod,
        full_hw=(height, width),
    )
  else:
    alpha_bits_val = jnp.asarray(alpha_bits, dtype=jnp.float32)

  total_bits_val = (
      ac_bits_val
      + dc_bits_val
      + ac_meta_bits_val
      + header_bits_val
      + alpha_bits_val
  )
  num_pixels = float(max(height * width, 1))
  bpp_val = total_bits_val / num_pixels

  if return_breakdown:
    return RateBreakdown(
        total_bits=total_bits_val,
        bpp=bpp_val,
        ac_bits=ac_bits_val,
        dc_bits=dc_bits_val,
        ac_meta_bits=ac_meta_bits_val,
        header_bits=header_bits_val,
        noise_bits=noise_bits_val,
        patch_bits=patch_bits_val,
        alpha_bits=alpha_bits_val,
    )
  return total_bits_val


def compute_spatial_rate_map_bits(
    state: JxlVarDCTState | None = None,
    params_override: Mapping[str, Any] | None = None,
    temperature: float = 0.25,
    topology: StaticVarDCTTopology | None = None,
    **kwargs: Any,
) -> dict[str, jax.Array]:
  """Computes spatial `(H8, W8)` per-block AC, DC, and metadata bit-rate proxy maps.

  Explicitly distributes multi-block `AcStrategy` transform AC token/nzeros bits
  and single-token `(ac_strategy, raw_quant_field)` metadata overhead across
  each transform's covered `(cy, cx)` block footprint (`w_first = 1.0 / (cy * cx)`).

  Returns:
    Dictionary with keys:
      - `"block_bits"`: `(H8, W8)` per-block total rate map (`ac_block_bits + dc_block_bits + meta_block_bits`)
      - `"ac_block_bits"`: `(H8, W8)` per-block AC rate bits
      - `"dc_block_bits"`: `(H8, W8)` per-block DC prediction residual rate bits
      - `"meta_block_bits"`: `(H8, W8)` per-block metadata rate bits
      - `"header_bits"`: scalar header bits
      - `"total_bits"`: scalar total bits (`jnp.sum(block_bits) + header_bits + alpha_bits`)
  """
  p = dict(params_override) if params_override is not None else {}
  p.update(kwargs)

  eff_tau = float(p.get("tau", temperature))
  tau_mod = min(eff_tau, 0.12)

  quant_ac = p.get("quant_ac", p.get("quant_ac_override", None))
  if quant_ac is None:
    if state is None:
      raise ValueError("Either `state` or `quant_ac` must be provided.")
    quant_ac = _extract_state_quant_ac(state)
  expected_hw8 = _infer_expected_hw8(
      state=state,
      topology=topology,
      ac_strategy=p.get("ac_strategy"),
      raw_quant_field=p.get("raw_quant_field"),
  )
  ac_5d = normalize_quant_ac_shape(quant_ac, expected_hw8=expected_hw8)
  _, h8, w8, _, _ = ac_5d.shape
  ac_3hw = jnp.transpose(ac_5d, (0, 1, 3, 2, 4)).reshape(3, h8 * 8, w8 * 8)

  dc_quant_in = p.get("dc_quant", None)
  dc_quant_weights = p.get("dc_quant_weights", None)
  if dc_quant_in is not None:
    dc_q_arr = jnp.asarray(dc_quant_in, dtype=jnp.float32)
    if dc_q_arr.ndim == 1 and dc_q_arr.shape[0] == 3:
      if dc_quant_weights is None:
        dc_quant_weights = dc_q_arr
      dc_quant_grid = (
          jnp.asarray(state.dc_quant, dtype=jnp.float32)
          if state is not None
          else jnp.zeros((3, h8, w8), dtype=jnp.float32)
      )
    else:
      dc_quant_grid = dc_q_arr
  else:
    dc_quant_grid = (
        jnp.asarray(state.dc_quant, dtype=jnp.float32)
        if state is not None
        else jnp.zeros((3, h8, w8), dtype=jnp.float32)
    )

  raw_quant_field = p.get("raw_quant_field", None)
  if raw_quant_field is None:
    raw_quant_field = (
        jnp.asarray(state.raw_quant_field, dtype=jnp.float32)
        if state is not None
        else jnp.full((h8, w8), 16.0, dtype=jnp.float32)
    )
  else:
    raw_quant_field = jnp.asarray(raw_quant_field, dtype=jnp.float32)

  th, tw = (h8 + 7) // 8, (w8 + 7) // 8
  ytox_map = p.get("ytox_map", None)
  if ytox_map is None:
    ytox_map = (
        jnp.asarray(state.ytox_map, dtype=jnp.float32)
        if state is not None
        else jnp.zeros((th, tw), dtype=jnp.float32)
    )
  else:
    ytox_map = jnp.asarray(ytox_map, dtype=jnp.float32)

  ytob_map = p.get("ytob_map", None)
  if ytob_map is None:
    ytob_map = (
        jnp.asarray(state.ytob_map, dtype=jnp.float32)
        if state is not None
        else jnp.zeros((th, tw), dtype=jnp.float32)
    )
  else:
    ytob_map = jnp.asarray(ytob_map, dtype=jnp.float32)

  ytox_dc_val = p.get("ytox_dc", int(state.ytox_dc) if state is not None else 0)
  ytob_dc_val = p.get("ytob_dc", int(state.ytob_dc) if state is not None else 0)
  color_factor = int(
      p.get("color_factor", state.color_factor if state is not None else 84)
  )
  acs_source = p.get(
      "ac_strategy", state.ac_strategy if state is not None else None
  )
  epf_sharpness = p.get(
      "epf_sharpness",
      jnp.asarray(state.epf_sharpness, dtype=jnp.float32)
      if state is not None
      else None,
  )

  bd = soft_jxl_rate_proxy(
      state=state,
      quant_ac=ac_3hw,
      dc_quant=dc_quant_grid,
      raw_quant_field=raw_quant_field,
      ytox_map=ytox_map,
      ytob_map=ytob_map,
      ytox_dc=ytox_dc_val,
      ytob_dc=ytob_dc_val,
      enable_noise=bool(p.get("enable_noise", False)),
      noise_lut=p.get("noise_lut", None),
      patch_bits=p.get("patch_bits", 0.0),
      alpha_bits=p.get("alpha_bits", 0.0),
      tau=eff_tau,
      return_breakdown=True,
      ac_strategy=acs_source,
      epf_sharpness=epf_sharpness,
      topology=topology,
      patches=p.get("patches", None),
      alpha_01=p.get("alpha_01", None),
      dim_shift=int(p.get("dim_shift", 0)),
      upsampling=int(p.get("upsampling", 1)),
      is_reference_only=bool(p.get("is_reference_only", False)),
      dequant_matrices=p.get("dequant_matrices", None),
      dc_quant_weights=dc_quant_weights,
  )
  assert isinstance(bd, RateBreakdown)

  # 1. Per-block AC rate map (distributed across (cy, cx) covered blocks)
  resolved_topo = _resolve_topology(topology, acs_source)
  _, _, freq_band8_1x1 = get_canonical_ac_scan_tables(1, 1)
  one_hot_1x1 = jax.nn.one_hot(
      freq_band8_1x1, NUM_FREQ_BANDS8, dtype=jnp.float32
  )

  hist_zero = jnp.zeros((3, NUM_FREQ_BANDS8), dtype=jnp.float32)
  hist_cats = jnp.zeros(
      (3, NUM_FREQ_BANDS8, MAX_AC_CATEGORY), dtype=jnp.float32
  )
  nzeros_norm_grid = jnp.zeros((3, h8, w8), dtype=jnp.float32)

  use_grid_aligned = (
      resolved_topo is not None
      and len(resolved_topo.multi_groups) > 0
      and getattr(resolved_topo, "all_grid_aligned", False)
  )

  bz_1x1 = bc_1x1 = bcdf_1x1 = blk_nz_1x1 = None
  coords_1x1 = None
  multi_scan_stats = []
  multi_scan_grid_stats = []

  if use_grid_aligned:
    assert resolved_topo is not None
    acs_traced = (
        jnp.asarray(acs_source, dtype=jnp.int32)
        if isinstance(acs_source, jax.core.Tracer)
        else None
    )
    q_ac_blocks = jnp.transpose(jnp.asarray(ac_5d, dtype=jnp.float32), (1, 2, 0, 3, 4))
    if resolved_topo.has_1x1:
      if acs_traced is not None:
        mask_1x1 = (
            (acs_traced <= 3) | ((acs_traced >= 12) & (acs_traced <= 17))
        ).astype(jnp.float32)
      else:
        with jax.ensure_compile_time_eval():
          mask_1x1 = jnp.asarray(resolved_topo.mask_1x1, dtype=jnp.float32)
      q_1x1 = (q_ac_blocks * mask_1x1[:, :, None, None, None]).reshape(
          h8 * w8, 3, 64
      )
      ac_idx_1x1, _, _ = get_canonical_ac_scan_tables(1, 1)
      ac_1x1 = q_1x1[:, :, ac_idx_1x1]
      bz_1x1, bc_1x1, bcdf_1x1, blk_nz_1x1 = _compute_scan_band_summaries(
          ac_1x1, one_hot_1x1, temp=eff_tau
      )
      hist_zero = hist_zero + jnp.sum(bz_1x1, axis=0)
      hist_cats = hist_cats + jnp.sum(bc_1x1, axis=0)
      nzeros_norm_grid = jnp.transpose(blk_nz_1x1, (1, 0)).reshape(3, h8, w8)
    else:
      ac_1x1 = jnp.zeros((0, 3, 63), dtype=jnp.float32)

    for grp in resolved_topo.multi_groups:
      cy_m, cx_m = grp["cy"], grp["cx"]
      cy_l, cx_l = grp["cy_l"], grp["cx_l"]
      b_m = cy_m * cx_m
      nh_cell, nw_cell = grp["nh_cell"], grp["nw_cell"]
      h_lim, w_lim = grp["h_lim"], grp["w_lim"]
      if acs_traced is not None:
        cell_mask_f = (
            acs_traced[:h_lim:cy_m, :w_lim:cx_m] == int(grp["s"])
        ).astype(jnp.float32)
      else:
        with jax.ensure_compile_time_eval():
          cell_mask_f = jnp.asarray(grp["cell_mask_f"], dtype=jnp.float32)
      sub_q = (
          q_ac_blocks[:h_lim, :w_lim].reshape(
              nh_cell, cy_m, nw_cell, cx_m, 3, 8, 8
          )
          * cell_mask_f[:, None, :, None, None, None, None]
      )
      q_canon = jnp.transpose(sub_q, (0, 2, 4, 1, 5, 3, 6)).reshape(
          nh_cell * nw_cell, 3, 64 * b_m
      )
      ac_idx_m, _, freq_band8_m = get_canonical_ac_scan_tables(cy_l, cx_l)
      ac_coeffs_m = q_canon[:, :, ac_idx_m]
      one_hot_m = jax.nn.one_hot(
          freq_band8_m, NUM_FREQ_BANDS8, dtype=jnp.float32
      )
      bz_m, bc_m, bcdf_m, blk_nz_m = _compute_scan_band_summaries(
          ac_coeffs_m, one_hot_m, temp=eff_tau
      )
      hist_zero = hist_zero + jnp.sum(bz_m, axis=0)
      hist_cats = hist_cats + jnp.sum(bc_m, axis=0)
      nz_cell = jnp.transpose(blk_nz_m / float(b_m), (1, 0)).reshape(
          3, nh_cell, 1, nw_cell, 1
      )
      nz_bcast = jnp.broadcast_to(
          nz_cell, (3, nh_cell, cy_m, nw_cell, cx_m)
      ).reshape(3, h_lim, w_lim)
      if h_lim == h8 and w_lim == w8:
        nzeros_norm_grid = nzeros_norm_grid + nz_bcast
      else:
        nzeros_norm_grid = nzeros_norm_grid.at[:, :h_lim, :w_lim].add(nz_bcast)
      multi_scan_grid_stats.append(
          (bz_m, bc_m, bcdf_m, one_hot_m, b_m, grp, cell_mask_f)
      )
  else:
    ac_1x1, multi_ac_list, coords_1x1, _ = _extract_ac_tokens_from_blocks(
        ac_3hw, topology=resolved_topo, expected_hw8=(h8, w8)
    )
    if ac_1x1.shape[0] > 0:
      bz_1x1, bc_1x1, bcdf_1x1, blk_nz_1x1 = _compute_scan_band_summaries(
          ac_1x1, one_hot_1x1, temp=eff_tau
      )
      hist_zero = hist_zero + jnp.sum(bz_1x1, axis=0)
      hist_cats = hist_cats + jnp.sum(bc_1x1, axis=0)
      if coords_1x1 is None:
        nzeros_norm_grid = jnp.transpose(blk_nz_1x1, (1, 0)).reshape(3, h8, w8)
      else:
        nzeros_norm_grid = nzeros_norm_grid.at[:, coords_1x1[0], coords_1x1[1]].set(
            jnp.transpose(blk_nz_1x1, (1, 0))
        )

    for (
        ac_coeffs_m,
        freq_band8_m,
        b_m,
        by_m,
        bx_m,
        cy_m,
        cx_m,
    ) in multi_ac_list:
      one_hot_m = jax.nn.one_hot(freq_band8_m, NUM_FREQ_BANDS8, dtype=jnp.float32)
      bz_m, bc_m, bcdf_m, blk_nz_m = _compute_scan_band_summaries(
          ac_coeffs_m, one_hot_m, temp=eff_tau
      )
      hist_zero = hist_zero + jnp.sum(bz_m, axis=0)
      hist_cats = hist_cats + jnp.sum(bc_m, axis=0)

      nz_norm_m = jnp.transpose(blk_nz_m / float(b_m), (1, 0))
      dy = jnp.arange(cy_m)[None, :, None]
      dx = jnp.arange(cx_m)[None, None, :]
      nzeros_norm_grid = nzeros_norm_grid.at[
          :, by_m[:, None, None] + dy, bx_m[:, None, None] + dx
      ].set(nz_norm_m[:, :, None, None])
      multi_scan_stats.append(
          (bz_m, bc_m, bcdf_m, one_hot_m, b_m, by_m, bx_m, cy_m, cx_m)
      )

  eps = 1e-3
  s_idx = jnp.arange(1, MAX_AC_CATEGORY + 1, dtype=jnp.float32)[None, None, :]
  prior_cats = eps * jnp.power(0.5, s_idx - 1.0)
  counts_zero = hist_zero[..., None] + eps
  counts_cats = hist_cats + prior_cats
  counts_all = jnp.concatenate([counts_zero, counts_cats], axis=-1)
  totals = jnp.sum(counts_all, axis=-1, keepdims=True)

  codelen_zero = -jnp.log2(counts_zero[..., 0] / totals[..., 0])
  codelen_cats = -jnp.log2(counts_cats / totals) + s_idx
  counts_prev = jnp.concatenate([counts_zero, counts_cats[..., :-1]], axis=-1)
  raw_step_cost = 1.0 + jnp.log2(
      jax.lax.stop_gradient(counts_prev / counts_cats)
  )
  min_step_deficit = jnp.maximum(0.0, 0.25 - raw_step_cost)

  ac_tok_map = jnp.zeros((h8, w8), dtype=jnp.float32)
  if ac_1x1.shape[0] > 0 and bz_1x1 is not None and bc_1x1 is not None and bcdf_1x1 is not None:
    blk_fwd_1x1 = jnp.sum(bz_1x1 * codelen_zero[None, :, :], axis=(1, 2)) + jnp.sum(
        bc_1x1 * codelen_cats[None, :, :, :], axis=(1, 2, 3)
    )
    blk_mono_1x1 = jnp.sum(
        bcdf_1x1 * min_step_deficit[None, :, :, :], axis=(1, 2, 3)
    )
    blk_ac_1x1 = blk_fwd_1x1 + (
        blk_mono_1x1 - jax.lax.stop_gradient(blk_mono_1x1)
    )
    if coords_1x1 is None:
      ac_tok_map = blk_ac_1x1.reshape(h8, w8)
    else:
      ac_tok_map = ac_tok_map.at[coords_1x1[0], coords_1x1[1]].set(blk_ac_1x1)

  if use_grid_aligned:
    for (
        bz_m,
        bc_m,
        bcdf_m,
        _,
        b_m,
        grp,
        cell_mask_f,
    ) in multi_scan_grid_stats:
      cy_m, cx_m = grp["cy"], grp["cx"]
      nh_cell, nw_cell = grp["nh_cell"], grp["nw_cell"]
      h_lim, w_lim = grp["h_lim"], grp["w_lim"]
      blk_fwd_m = jnp.sum(bz_m * codelen_zero[None, :, :], axis=(1, 2)) + jnp.sum(
          bc_m * codelen_cats[None, :, :, :], axis=(1, 2, 3)
      )
      blk_mono_m = jnp.sum(
          bcdf_m * min_step_deficit[None, :, :, :], axis=(1, 2, 3)
      )
      blk_ac_m_cell = (
          (blk_fwd_m + (blk_mono_m - jax.lax.stop_gradient(blk_mono_m)))
          / float(b_m)
      ).reshape(nh_cell, nw_cell) * cell_mask_f
      tok_bcast = jnp.broadcast_to(
          blk_ac_m_cell[:, None, :, None], (nh_cell, cy_m, nw_cell, cx_m)
      ).reshape(h_lim, w_lim)
      if h_lim == h8 and w_lim == w8:
        ac_tok_map = ac_tok_map + tok_bcast
      else:
        ac_tok_map = ac_tok_map.at[:h_lim, :w_lim].add(tok_bcast)
  else:
    for (
        bz_m,
        bc_m,
        bcdf_m,
        _,
        b_m,
        by_m,
        bx_m,
        cy_m,
        cx_m,
    ) in multi_scan_stats:
      blk_fwd_m = jnp.sum(bz_m * codelen_zero[None, :, :], axis=(1, 2)) + jnp.sum(
          bc_m * codelen_cats[None, :, :, :], axis=(1, 2, 3)
      )
      blk_mono_m = jnp.sum(
          bcdf_m * min_step_deficit[None, :, :, :], axis=(1, 2, 3)
      )
      blk_ac_m_cell = (
          blk_fwd_m + (blk_mono_m - jax.lax.stop_gradient(blk_mono_m))
      ) / float(b_m)
      dy = jnp.arange(cy_m)[None, :, None]
      dx = jnp.arange(cx_m)[None, None, :]
      ac_tok_map = ac_tok_map.at[
          by_m[:, None, None] + dy, bx_m[:, None, None] + dx
      ].set(blk_ac_m_cell[:, None, None])

  # Per-block nzeros bit map (distributed over covered blocks for multi-block transforms)
  nz_sg = jax.lax.stop_gradient(nzeros_norm_grid)
  top_nz = jnp.pad(nz_sg[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
  left_nz = jnp.pad(nz_sg[:, :, :-1], ((0, 0), (0, 0), (1, 0)))
  by_idx = jnp.arange(h8)[:, None]
  bx_idx = jnp.arange(w8)[None, :]
  pred_nz = jnp.where(
      (by_idx == 0) & (bx_idx == 0),
      32.0,
      jnp.where(
          by_idx == 0,
          left_nz,
          jnp.where(bx_idx == 0, top_nz, 0.5 * (top_nz + left_nz)),
      ),
  )
  active_blk = nzeros_norm_grid / (nzeros_norm_grid + 1.0)
  excess_over_pred = jax.nn.softplus(nzeros_norm_grid - pred_nz)
  nz_blk_2d = jnp.sum(
      1.5 * active_blk
      + jnp.log2(1.0 + nzeros_norm_grid)
      + 0.35 * jnp.log2(1.0 + excess_over_pred),
      axis=0,
  )
  if use_grid_aligned:
    for _, _, _, _, b_m, grp, cell_mask_f in multi_scan_grid_stats:
      cy_m, cx_m = grp["cy"], grp["cx"]
      nh_cell, nw_cell = grp["nh_cell"], grp["nw_cell"]
      h_lim, w_lim = grp["h_lim"], grp["w_lim"]
      sub_nz = nz_blk_2d[:h_lim, :w_lim].reshape(nh_cell, cy_m, nw_cell, cx_m)
      tl_nz = (sub_nz[:, 0, :, 0] / float(b_m))[:, None, :, None]
      sub_nz_new = jnp.where(cell_mask_f[:, None, :, None] > 0.5, tl_nz, sub_nz)
      nz_blk_2d = nz_blk_2d.at[:h_lim, :w_lim].set(sub_nz_new.reshape(h_lim, w_lim))
  else:
    for _, _, _, _, b_m, by_m, bx_m, cy_m, cx_m in multi_scan_stats:
      top_left_nz = nz_blk_2d[by_m, bx_m] / float(b_m)
      dy = jnp.arange(cy_m)[None, :, None]
      dx = jnp.arange(cx_m)[None, None, :]
      nz_blk_2d = nz_blk_2d.at[
          by_m[:, None, None] + dy, bx_m[:, None, None] + dx
      ].set(top_left_nz[:, None, None])

  def _match_sum_exact_f32(arr_2d: jax.Array, target_sum: jax.Array) -> jax.Array:
    out = arr_2d
    for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
      out = out.at[0, 0].add(
          alpha * jax.lax.stop_gradient(target_sum - jnp.sum(out))
      )
    return out

  ac_raw_map = ac_tok_map + nz_blk_2d
  w_ac = jax.lax.stop_gradient(
      (ac_raw_map + 1e-6) / jnp.sum(ac_raw_map + 1e-6)
  )
  ac_block_bits = _match_sum_exact_f32(
      ac_raw_map + w_ac * (bd.ac_bits - jnp.sum(ac_raw_map)), bd.ac_bits
  )

  # 2. Per-block DC rate map
  ytox_dc_f = jnp.asarray(ytox_dc_val, dtype=jnp.float32)
  ytob_dc_f = jnp.asarray(ytob_dc_val, dtype=jnp.float32)
  q_y = dc_quant_grid[1]
  q_x = dc_quant_grid[0] - 8.0 * (ytox_dc_f / float(color_factor)) * q_y
  q_b = dc_quant_grid[2] - 0.5 * (ytob_dc_f / float(color_factor)) * q_y

  res_y_2d = jnp.zeros_like(q_y)
  res_x_2d = jnp.zeros_like(q_x)
  res_b_2d = jnp.zeros_like(q_b)
  num_dc_groups_y = (h8 + 255) // 256
  num_dc_groups_x = (w8 + 255) // 256
  for gy in range(num_dc_groups_y):
    y0, y1 = gy * 256, min((gy + 1) * 256, h8)
    for gx in range(num_dc_groups_x):
      x0, x1 = gx * 256, min((gx + 1) * 256, w8)
      sub_y = q_y[y0:y1, x0:x1]
      sub_x = q_x[y0:y1, x0:x1]
      sub_b = q_b[y0:y1, x0:x1]
      res_y_2d = res_y_2d.at[y0:y1, x0:x1].set(
          sub_y - soft_clamped_gradient_2d(sub_y, tau=tau_mod)
      )
      res_x_2d = res_x_2d.at[y0:y1, x0:x1].set(
          sub_x - soft_clamped_gradient_2d(sub_x, tau=tau_mod)
      )
      res_b_2d = res_b_2d.at[y0:y1, x0:x1].set(
          sub_b - soft_clamped_gradient_2d(sub_b, tau=tau_mod)
      )

  def _per_block_stream_bits(res_2d: jax.Array) -> jax.Array:
    tp_2d, eb_2d = soft_hybrid_uint_420_token_probs(res_2d, tau=tau_mod)
    probs_flat = tp_2d.reshape(-1, _NUM_TOKENS)
    tok_idx = jnp.arange(_NUM_TOKENS, dtype=jnp.float32)
    prior_counts = 1e-3 * jnp.exp(-0.12 * tok_idx)
    counts = jnp.sum(probs_flat, axis=0)
    smoothed_counts = counts + prior_counts
    p_bar = smoothed_counts / jnp.maximum(jnp.sum(smoothed_counts), 1e-12)
    cross_blk = -jnp.sum(
        tp_2d * jnp.log2(jnp.maximum(p_bar[None, None, :], 1e-12)), axis=-1
    )
    self_blk = -jnp.sum(tp_2d * jnp.log2(jnp.maximum(tp_2d, 1e-12)), axis=-1)
    sh_blk = cross_blk - jax.lax.stop_gradient(self_blk)
    tie_blk = 1e-4 * jnp.sqrt(res_2d * res_2d + 1e-4)
    return 1.10 * sh_blk + eb_2d + tie_blk

  dc_local_map = (
      _per_block_stream_bits(res_y_2d)
      + _per_block_stream_bits(res_x_2d)
      + _per_block_stream_bits(res_b_2d)
  )
  dc_block_bits = _match_sum_exact_f32(
      dc_local_map + (bd.dc_bits - jnp.sum(dc_local_map)) / float(h8 * w8),
      bd.dc_bits,
  )

  # 3. Per-block metadata rate map (weighted by footprint inverse area w_first = 1 / (cy * cx))
  if acs_source is not None:
    acs_f_2d = jnp.asarray(acs_source, dtype=jnp.float32)
    s_idx_2d = jnp.clip(
        jnp.round(jax.lax.stop_gradient(acs_f_2d)).astype(jnp.int32),
        0,
        len(_ACS_INV_AREA_NP) - 1,
    )
    w_first_2d = jnp.array(_ACS_INV_AREA_NP, dtype=jnp.float32)[s_idx_2d]
    inv_cy_2d = jnp.array(_ACS_INV_CY_NP, dtype=jnp.float32)[s_idx_2d]
    acs_flat = acs_f_2d.reshape(-1)
  else:
    w_first_2d = jnp.ones((h8, w8), dtype=jnp.float32)
    inv_cy_2d = jnp.ones((h8, w8), dtype=jnp.float32)
    acs_flat = jnp.zeros((h8 * w8,), dtype=jnp.float32)

  qf_m1_flat = (raw_quant_field - 1.0).reshape(-1)
  qf_pred = jnp.concatenate([acs_flat[:1], qf_m1_flat[:-1]], axis=0)
  res_qf_2d = (qf_m1_flat - qf_pred).reshape(h8, w8)
  nz_jump_2d = jax.nn.sigmoid(
      (jnp.abs(res_qf_2d) - 0.25) / jnp.maximum(tau_mod, 0.08)
  )
  w_qf_2d = nz_jump_2d * inv_cy_2d + (1.0 - nz_jump_2d) * w_first_2d
  _, eb_qf_2d = soft_hybrid_uint_420_token_probs(res_qf_2d, tau=tau_mod)

  qf_act_2d = w_qf_2d * (
      eb_qf_2d + 0.1 * jnp.log2(1.0 + jnp.abs(res_qf_2d))
  )
  if use_grid_aligned:
    for _, _, _, _, b_m, grp, cell_mask_f in multi_scan_grid_stats:
      cy_m, cx_m = grp["cy"], grp["cx"]
      nh_cell, nw_cell = grp["nh_cell"], grp["nw_cell"]
      h_lim, w_lim = grp["h_lim"], grp["w_lim"]
      sub_qf_act = qf_act_2d[:h_lim, :w_lim].reshape(nh_cell, cy_m, nw_cell, cx_m)
      mean_qf_act = jnp.mean(sub_qf_act, axis=(1, 3), keepdims=True)
      sub_qf_new = jnp.where(
          cell_mask_f[:, None, :, None] > 0.5, mean_qf_act, sub_qf_act
      )
      qf_act_2d = qf_act_2d.at[:h_lim, :w_lim].set(
          sub_qf_new.reshape(h_lim, w_lim)
      )
  else:
    for _, _, _, _, b_m, by_m, bx_m, cy_m, cx_m in multi_scan_stats:
      dy = jnp.arange(cy_m)[None, :, None]
      dx = jnp.arange(cx_m)[None, None, :]
      sub_qf_act = qf_act_2d[by_m[:, None, None] + dy, bx_m[:, None, None] + dx]
      mean_qf_act = jnp.mean(sub_qf_act, axis=(1, 2), keepdims=True)
      qf_act_2d = qf_act_2d.at[
          by_m[:, None, None] + dy, bx_m[:, None, None] + dx
      ].set(mean_qf_act)

  meta_raw_map = w_first_2d + 0.20 * qf_act_2d
  meta_block_bits = _match_sum_exact_f32(
      bd.ac_meta_bits * (meta_raw_map / jnp.maximum(jnp.sum(meta_raw_map), 1e-8)),
      bd.ac_meta_bits,
  )

  target_block_sum = bd.total_bits - bd.header_bits - bd.alpha_bits
  block_bits = _match_sum_exact_f32(
      ac_block_bits + dc_block_bits + meta_block_bits,
      target_block_sum,
  )
  return {
      "block_bits": block_bits,
      "ac_block_bits": ac_block_bits,
      "dc_block_bits": dc_block_bits,
      "meta_block_bits": meta_block_bits,
      "header_bits": bd.header_bits,
      "alpha_bits": bd.alpha_bits,
      "ac_bits": bd.ac_bits,
      "dc_bits": bd.dc_bits,
      "ac_meta_bits": bd.ac_meta_bits,
      "total_bits": bd.total_bits,
      "total_bpp": bd.bpp,
      "bpp": bd.bpp,
  }


def soft_patch_composite_rate_proxy(
    ref_state: JxlVarDCTState,
    main_state: JxlVarDCTState,
    patches: Sequence[PatchEntry | Any],
    upsampling: int = 2,
    *,
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
    enable_noise: bool = False,
    noise_lut: Sequence[float] | jax.Array | None = None,
    tau: float = 0.25,
    return_breakdown: bool = False,
    ref_dequant_matrices: dict[int, jax.Array] | jax.Array | None = None,
    main_dequant_matrices: dict[int, jax.Array] | jax.Array | None = None,
    multi_cluster: bool = False,
    cluster_blend: float | jax.Array | None = None,
    ref_cluster_blend: float | jax.Array | None = None,
    main_cluster_blend: float | jax.Array | None = None,
) -> jax.Array | RateBreakdown:
  """Differentiable total bit-rate proxy for 2-frame XYB `PatchDictionary` compositing."""
  eff_ref_cb = ref_cluster_blend if ref_cluster_blend is not None else cluster_blend
  eff_main_cb = (
      main_cluster_blend if main_cluster_blend is not None else cluster_blend
  )
  ref_bd = soft_jxl_rate_proxy(
      ref_state,
      quant_ac=ref_quant_ac,
      dc_quant=ref_dc_quant,
      raw_quant_field=ref_raw_quant_field,
      ytox_map=ref_ytox_map,
      ytob_map=ref_ytob_map,
      enable_noise=False,
      tau=tau,
      upsampling=upsampling,
      is_reference_only=True,
      dequant_matrices=ref_dequant_matrices,
      multi_cluster=multi_cluster,
      cluster_blend=eff_ref_cb,
      return_breakdown=True,
  )
  main_bd = soft_jxl_rate_proxy(
      main_state,
      quant_ac=main_quant_ac,
      dc_quant=main_dc_quant,
      raw_quant_field=main_raw_quant_field,
      ytox_map=main_ytox_map,
      ytob_map=main_ytob_map,
      enable_noise=enable_noise,
      noise_lut=noise_lut,
      patches=patches,
      tau=tau,
      upsampling=1,
      is_reference_only=False,
      dequant_matrices=main_dequant_matrices,
      multi_cluster=multi_cluster,
      cluster_blend=eff_main_cb,
      return_breakdown=True,
  )
  assert isinstance(ref_bd, RateBreakdown)
  assert isinstance(main_bd, RateBreakdown)
  total = ref_bd.total_bits + main_bd.total_bits
  num_pixels = float(max(main_state.height * main_state.width, 1))
  if return_breakdown:
    return RateBreakdown(
        total_bits=total,
        bpp=total / num_pixels,
        ac_bits=ref_bd.ac_bits + main_bd.ac_bits,
        dc_bits=ref_bd.dc_bits + main_bd.dc_bits,
        ac_meta_bits=ref_bd.ac_meta_bits + main_bd.ac_meta_bits,
        header_bits=ref_bd.header_bits + main_bd.header_bits,
        noise_bits=main_bd.noise_bits,
        patch_bits=main_bd.patch_bits,
        alpha_bits=jnp.asarray(0.0, dtype=jnp.float32),
    )
  return total


def soft_alpha_composite_rate_proxy(
    ref_state: JxlVarDCTState,
    main_state: JxlVarDCTState,
    alpha_01: jax.Array,
    upsampling: int = 2,
    dim_shift: int = 0,
    *,
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
    enable_noise: bool = False,
    noise_lut: Sequence[float] | jax.Array | None = None,
    tau: float = 0.25,
    return_breakdown: bool = False,
    ref_dequant_matrices: dict[int, jax.Array] | jax.Array | None = None,
    main_dequant_matrices: dict[int, jax.Array] | jax.Array | None = None,
    multi_cluster: bool = False,
    cluster_blend: float | jax.Array | None = None,
    ref_cluster_blend: float | jax.Array | None = None,
    main_cluster_blend: float | jax.Array | None = None,
) -> jax.Array | RateBreakdown:
  """Differentiable total bit-rate proxy for 2-frame Modular `ExtraChannel::kAlpha` blending."""
  eff_ref_cb = ref_cluster_blend if ref_cluster_blend is not None else cluster_blend
  eff_main_cb = (
      main_cluster_blend if main_cluster_blend is not None else cluster_blend
  )
  ref_bd = soft_jxl_rate_proxy(
      ref_state,
      quant_ac=ref_quant_ac,
      dc_quant=ref_dc_quant,
      raw_quant_field=ref_raw_quant_field,
      ytox_map=ref_ytox_map,
      ytob_map=ref_ytob_map,
      enable_noise=False,
      tau=tau,
      upsampling=upsampling,
      is_reference_only=True,
      dequant_matrices=ref_dequant_matrices,
      multi_cluster=multi_cluster,
      cluster_blend=eff_ref_cb,
      return_breakdown=True,
  )
  main_bd = soft_jxl_rate_proxy(
      main_state,
      quant_ac=main_quant_ac,
      dc_quant=main_dc_quant,
      raw_quant_field=main_raw_quant_field,
      ytox_map=main_ytox_map,
      ytob_map=main_ytob_map,
      enable_noise=enable_noise,
      noise_lut=noise_lut,
      alpha_01=alpha_01,
      dim_shift=dim_shift,
      tau=tau,
      upsampling=1,
      is_reference_only=False,
      dequant_matrices=main_dequant_matrices,
      multi_cluster=multi_cluster,
      cluster_blend=eff_main_cb,
      return_breakdown=True,
  )
  assert isinstance(ref_bd, RateBreakdown)
  assert isinstance(main_bd, RateBreakdown)
  total = ref_bd.total_bits + main_bd.total_bits
  num_pixels = float(max(main_state.height * main_state.width, 1))
  if return_breakdown:
    return RateBreakdown(
        total_bits=total,
        bpp=total / num_pixels,
        ac_bits=ref_bd.ac_bits + main_bd.ac_bits,
        dc_bits=ref_bd.dc_bits + main_bd.dc_bits,
        ac_meta_bits=ref_bd.ac_meta_bits + main_bd.ac_meta_bits,
        header_bits=ref_bd.header_bits + main_bd.header_bits,
        noise_bits=main_bd.noise_bits,
        patch_bits=jnp.asarray(0.0, dtype=jnp.float32),
        alpha_bits=main_bd.alpha_bits,
    )
  return total


__all__ = [
    "K_COEFF_FREQ_CONTEXT",
    "K_FREQ_CTX_TO_BAND8",
    "MAX_AC_CATEGORY",
    "NUM_FREQ_BANDS8",
    "NUM_FREQ_CONTEXTS",
    "RateBreakdown",
    "compute_continuous_quant_ac_from_dct",
    "compute_natural_coeff_order",
    "compute_spatial_rate_map_bits",
    "get_canonical_ac_scan_tables",
    "get_first_block_And_llf_masks",
    "get_first_block_and_llf_masks",
    "normalize_quant_ac_shape",
    "soft_ac_meta_rate_bits",
    "soft_ac_metadata_bits",
    "soft_ac_rate_bits",
    "soft_alpha_channel_bits",
    "soft_alpha_composite_rate_proxy",
    "soft_bounded_code_lengths",
    "soft_category_probs",
    "soft_clamped_gradient_2d",
    "soft_dc_rate_bits",
    "soft_dequant_matrices_rate_bits",
    "soft_header_rate_bits",
    "soft_hybrid_uint_420_token_probs",
    "soft_jxl_rate_proxy",
    "soft_noise_header_bits",
    "soft_patch_composite_rate_proxy",
    "soft_patch_dictionary_bits",
    "soft_stream_entropy_and_header_bits",
    "soft_zero_and_active",
    "ste_round",
]


