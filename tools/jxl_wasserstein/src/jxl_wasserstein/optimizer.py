"""Gradient-based Rate-Wasserstein Distortion optimizer for JPEG XL VarDCT states.

Implements:
- Three-stage Rate-WD optimization architecture (`stage1_explore_ac_candidates`,
  `stage2_select_non_overlapping_ac_strategy`, `optimize_single_frame`) with zero
  libjxl discrete per-block choices and zero static pixel-variance/MSE heuristics.
- Joint continuous parameter and custom `DequantMatrices` optimization (`ste_dequant_matrices`
  + `roundtrip_custom_dequant_matrices`) with 80/20 metadata freeze schedule.
- Continuous pre-rounded DCT latent extraction (`compute_continuous_latents_jax`),
  canonical LLF masking (`mask_llf_numpy`), and procedural noise LUT estimation
  (`estimate_initial_noise_lut_jax`, `apply_hf_noise_denoising_jax`).
- Multi-transform Rate-WD initializer and optimizer (`init_pure_jax_wd_state`,
  `optimize_rate_wd_point`) for Kodak R-D benchmarking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
import copy
import functools
import hashlib
import math
import os
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jxl_wasserstein.cpp_bridge import (
    DEFAULT_DC_QUANT_WEIGHTS,
    JxlVarDCTState,
    compute_psnr,
    decode_with_djxl,
    encode_single_frame,
    inspect_jxl,
    roundtrip_custom_dequant_matrices,
    verify_jxl_codestream,
)
from jxl_wasserstein.entropy_proxy import (
    ACS_COVERED_BLOCKS_X,
    ACS_COVERED_BLOCKS_Y,
    compute_spatial_rate_map_bits,
    get_canonical_ac_scan_tables,
    soft_ac_rate_bits,
    soft_dc_rate_bits,
    soft_jxl_rate_proxy,
    ste_round,
)
from jxl_wasserstein.forward_model import (
    K_BASE_CORRELATION_B,
    K_BASE_CORRELATION_X,
    K_DC_QUANT,
    K_DEFAULT_COLOR_FACTOR,
    K_DEFAULT_GAB_WEIGHTS,
    K_GLOBAL_SCALE_DENOM,
    StaticVarDCTTopology,
    adaptive_dc_smoothing,
    adjust_quant_bias_ste,
    gaborish_inverse_3x3,
    get_default_dequant_matrices,
    init_state_from_rgb_jax,
    reconstruct_single_frame,
    reconstruct_single_frame_xyb,
    srgb_to_xyb,
    ste_dequant_matrices,
    xyb_to_srgb,
)
from jxl_wasserstein.noise import (
    NOISE_PRECISION,
    NUM_NOISE_POINTS,
    get_convolved_noise_planes,
)
from jxl_wasserstein.vgg import (
    load_vgg16_params,
    multiscale_vgg_features,
    select_num_scales,
)
from jxl_wasserstein.wasserstein import (
    compute_spatial_wd_map,
    compute_wd,
    multi_wasserstein_distortion,
    stream_vgg_spatial_wd_against_targets,
    stream_vgg_wd_against_targets,
)

# Pre-warm canonical AC scan table cache in eager mode outside @jax.jit so
# module-level caches never capture DynamicJaxprTracers.
with jax.ensure_compile_time_eval():
  for _cy, _cx in ((1, 1), (1, 2), (2, 2), (1, 4), (2, 4), (4, 4)):
    get_canonical_ac_scan_tables(_cy, _cx)

DEFAULT_CANDIDATE_AC_STRATEGIES: tuple[int, ...] = tuple(range(18))
_DEFAULT_DC_QUANT_WEIGHTS_NP = np.asarray(
    DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32
)

AC_STRATEGY_NAMES: dict[int, str] = {
    0: "DCT8X8",
    1: "IDENTITY",
    2: "DCT2X2",
    3: "DCT4X4",
    4: "DCT16X16",
    5: "DCT32X32",
    6: "DCT16X8",
    7: "DCT8X16",
    8: "DCT32X8",
    9: "DCT8X32",
    10: "DCT32X16",
    11: "DCT16X32",
    12: "DCT4X8",
    13: "DCT8X4",
    14: "AFV0",
    15: "AFV1",
    16: "AFV2",
    17: "AFV3",
}


# =============================================================================
# 0. Module-Level Cached JIT Kernels for VGG-16 Features & Wasserstein Distortion
#    (Compiles ONCE per (H, W) resolution, independent of JXL block topology!)
# =============================================================================


@functools.partial(jax.jit, static_argnames=("eff_scales",))
def _jit_extract_vgg_features(
    vgg_params: Any,
    img_chw: jax.Array,
    eff_scales: int,
) -> list[jax.Array]:
  return list(multiscale_vgg_features(vgg_params, img_chw, num_scales=eff_scales))


@functools.partial(jax.jit, static_argnames=("eff_scales", "num_levels"))
def _jit_wd_eval_chw(
    vgg_params: Any,
    recon_chw: jax.Array,
    target_features: list[jax.Array],
    ls_map: jax.Array,
    eff_scales: int,
    num_levels: int,
) -> jax.Array:
  return stream_vgg_wd_against_targets(
      vgg_params,
      recon_chw,
      target_features,
      ls_map,
      eff_scales=eff_scales,
      num_levels=num_levels,
  )


@functools.partial(jax.jit, static_argnames=("eff_scales", "num_levels"))
def _jit_wd_val_and_grad_chw(
    vgg_params: Any,
    recon_chw: jax.Array,
    target_features: list[jax.Array],
    ls_map: jax.Array,
    eff_scales: int,
    num_levels: int,
) -> tuple[jax.Array, jax.Array]:
  def _wd_of_recon(r_chw: jax.Array) -> jax.Array:
    return stream_vgg_wd_against_targets(
        vgg_params,
        r_chw,
        target_features,
        ls_map,
        eff_scales=eff_scales,
        num_levels=num_levels,
    )

  return jax.value_and_grad(_wd_of_recon)(recon_chw)


@functools.partial(
    jax.jit, static_argnames=("h8", "w8", "eff_scales", "num_levels")
)
def _jit_spatial_wd_map_chw(
    vgg_params: Any,
    recon_chw: jax.Array,
    target_features: list[jax.Array],
    ls_map: jax.Array,
    h8: int,
    w8: int,
    eff_scales: int,
    num_levels: int,
) -> tuple[jax.Array, jax.Array]:
  """Evaluates scalar WD and (H8, W8) additive per-block WD map reusing target_features."""
  return stream_vgg_spatial_wd_against_targets(
      vgg_params,
      recon_chw,
      target_features,
      ls_map,
      h8=h8,
      w8=w8,
      eff_scales=eff_scales,
      num_levels=num_levels,
  )


# =============================================================================
# 1. Stage 1 Candidate Exploration & Stage 2 Non-Overlapping AcStrategy DP
#    (ZERO libjxl discrete per-block choices & ZERO static pixel heuristics)
# =============================================================================

_STAGE1_CACHE: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}


def _normalize_rgb_hwc(rgb_01: np.ndarray | jax.Array) -> np.ndarray:
  arr = np.asarray(rgb_01, dtype=np.float32)
  if arr.ndim != 3:
    raise ValueError(f"Expected 3D RGB image array, got shape {arr.shape}")
  if arr.shape[0] == 3 and arr.shape[2] != 3:
    arr = np.moveaxis(arr, 0, -1)
  if arr.shape[2] != 3:
    raise ValueError(f"Expected 3 RGB channels, got shape {arr.shape}")
  if arr.shape[0] < 1 or arr.shape[1] < 1:
    raise ValueError(f"Invalid spatial dimensions in shape {arr.shape}")
  if not np.all(np.isfinite(arr)):
    raise ValueError("Input RGB image contains non-finite values")
  if arr.max() > 1.5:
    arr = arr / 255.0
  return np.clip(arr, 0.0, 1.0)


@jax.jit
def _compute_aq_and_epf_maps_jax(
    srgb_chw: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Computes continuous XYB high-frequency energy for initial raw_quant_field and epf_sharpness."""
  _, h, w = srgb_chw.shape
  h8 = (h + 7) // 8
  w8 = (w + 7) // 8
  xyb = srgb_to_xyb(srgb_chw)
  pad_h = h8 * 8 - h
  pad_w = w8 * 8 - w
  if pad_h > 0 or pad_w > 0:
    xyb = jnp.pad(xyb, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

  y_plane = xyb[1] + 0.35 * jnp.abs(xyb[0])
  y_blocks = jnp.transpose(y_plane.reshape(h8, 8, w8, 8), (0, 2, 1, 3))
  gx = jnp.abs(y_blocks[:, :, :, 1:] - y_blocks[:, :, :, :-1])
  gy = jnp.abs(y_blocks[:, :, 1:, :] - y_blocks[:, :, :-1, :])
  e_gx = jnp.mean(gx, axis=(2, 3))
  e_gy = jnp.mean(gy, axis=(2, 3))

  y_raw = xyb[1]
  x_raw = xyb[0]
  y_pad = jnp.pad(y_raw, ((1, 1), (1, 1)), mode="edge")
  y_smooth = (
      y_pad[:-2, :-2]
      + y_pad[:-2, 1:-1]
      + y_pad[:-2, 2:]
      + y_pad[1:-1, :-2]
      + y_pad[1:-1, 1:-1]
      + y_pad[1:-1, 2:]
      + y_pad[2:, :-2]
      + y_pad[2:, 1:-1]
      + y_pad[2:, 2:]
  ) / 9.0
  hp = jnp.abs(y_raw - y_smooth) + 0.02 * jnp.abs(x_raw - jnp.mean(x_raw))
  hp_blocks = jnp.transpose(hp.reshape(h8, 8, w8, 8), (0, 2, 1, 3))
  e_hf = jnp.mean(hp_blocks * hp_blocks, axis=(2, 3))

  ehf_pad = jnp.pad(e_hf, ((1, 1), (1, 1)), mode="edge")
  ehf_smooth = (
      ehf_pad[:-2, :-2]
      + 2.0 * ehf_pad[:-2, 1:-1]
      + ehf_pad[:-2, 2:]
      + 2.0 * ehf_pad[1:-1, :-2]
      + 4.0 * ehf_pad[1:-1, 1:-1]
      + 2.0 * ehf_pad[1:-1, 2:]
      + ehf_pad[2:, :-2]
      + 2.0 * ehf_pad[2:, 1:-1]
      + ehf_pad[2:, 2:]
  ) / 16.0
  return e_gx, e_gy, e_hf, ehf_smooth


def broadcast_quant_field_to_ac_strategy(
    raw_quant_field: np.ndarray,
    ac_strategy: np.ndarray,
) -> np.ndarray:
  """Ensures every multi-block AcStrategy region has uniform raw_quant_field."""
  h8, w8 = ac_strategy.shape
  out = np.asarray(raw_quant_field, dtype=np.int32).copy()
  visited = np.zeros((h8, w8), dtype=bool)
  for by in range(h8):
    for bx in range(w8):
      if visited[by, bx]:
        continue
      s = int(ac_strategy[by, bx])
      cx = int(ACS_COVERED_BLOCKS_X[s])
      cy = int(ACS_COVERED_BLOCKS_Y[s])
      val = int(np.clip(out[by, bx], 1, 256))
      out[by : by + cy, bx : bx + cx] = val
      visited[by : by + cy, bx : bx + cx] = True
  return np.clip(out, 1, 256).astype(np.int32)


def validate_ac_strategy_tiling(ac_strategy: np.ndarray) -> bool:
  """Verifies that an (H8, W8) ac_strategy grid forms a valid <=32x32 JXL tiling without crossing 64x64 CfL tiles."""
  acs = np.asarray(ac_strategy)
  if acs.ndim != 2:
    return False
  h8, w8 = acs.shape
  covered = np.zeros((h8, w8), dtype=bool)
  for by in range(h8):
    for bx in range(w8):
      if covered[by, bx]:
        continue
      s = int(acs[by, bx])
      if s < 0 or s >= 18 or s >= len(ACS_COVERED_BLOCKS_Y):
        return False
      cy = int(ACS_COVERED_BLOCKS_Y[s])
      cx = int(ACS_COVERED_BLOCKS_X[s])
      if by + cy > h8 or bx + cx > w8:
        return False
      # Multi-block transforms cannot cross 8x8-block (64x64 px) CfL tile boundaries
      if (by // 8) != ((by + cy - 1) // 8) or (bx // 8) != ((bx + cx - 1) // 8):
        return False
      if np.any(covered[by : by + cy, bx : bx + cx]):
        return False
      if not np.all(acs[by : by + cy, bx : bx + cx] == s):
        return False
      covered[by : by + cy, bx : bx + cx] = True
  return bool(np.all(covered))


def _build_uniform_candidate_ac_strategy_grid(
    h8: int, w8: int, strategy_id: int
) -> np.ndarray:
  """Tiles (h8, w8) with non-overlapping blocks of strategy_id within 8x8-block (64x64 px) CfL tiles."""
  s = int(strategy_id)
  cy = int(ACS_COVERED_BLOCKS_Y[s])
  cx = int(ACS_COVERED_BLOCKS_X[s])
  acs = np.zeros((h8, w8), dtype=np.uint8)
  if cy == 1 and cx == 1:
    acs[:, :] = np.uint8(s)
    return acs

  num_ty = (h8 + 7) // 8
  num_tx = (w8 + 7) // 8
  for ty in range(num_ty):
    by0 = ty * 8
    by1 = min((ty + 1) * 8, h8)
    for tx in range(num_tx):
      bx0 = tx * 8
      bx1 = min((tx + 1) * 8, w8)
      for by in range(by0, by1 - cy + 1, cy):
        for bx in range(bx0, bx1 - cx + 1, cx):
          acs[by : by + cy, bx : bx + cx] = np.uint8(s)
  return acs


def stage1_explore_ac_candidates(
    rgb_01: np.ndarray | jax.Array,
    vgg_params: Any = None,
    *,
    distance: float | None = None,
    rd_lambda: float = 0.15,
    lambda_rd: float | None = None,
    steps: int = 2,
    num_steps: int | None = None,
    candidate_strategies: Sequence[int] | None = None,
    allow_multiblock: bool = True,
    scale_distance_with_lambda: bool = True,
    log2_sigma: float = 1.0,
    num_scales: int = 3,
    num_levels: int = 2,
) -> dict[int, dict[str, Any]]:
  """Stage 1: Explores candidate AcStrategy tilings with preliminary Rate-WD optimization.

  Evaluates candidate transforms on the image by running preliminary Rate-WD
  proximal optimization steps and computing spatial (H8, W8) Lagrangian cost maps
  `J_s[by, bx] = WD_s[by, bx] + rd_lambda * Rate_s[by, bx]`.
  """
  if lambda_rd is not None:
    rd_lambda = float(lambda_rd)
  if float(rd_lambda) <= 0.0:
    raise ValueError(f"rd_lambda must be > 0, got {rd_lambda}")
  if distance is not None and float(distance) <= 0.0:
    raise ValueError(f"distance must be > 0, got {distance}")
  eff_steps = int(steps if num_steps is None else num_steps)

  params_vgg = _get_vgg_params(vgg_params)
  target_hwc = _normalize_rgb_hwc(rgb_01)
  h, w, _ = target_hwc.shape
  h8 = (h + 7) // 8
  w8 = (w + 7) // 8

  eff_distance = rd_lambda_to_distance(
      rd_lambda=rd_lambda,
      base_distance=distance,
      scale_distance_with_lambda=scale_distance_with_lambda,
  )

  if candidate_strategies is not None:
    raw_cands = [int(s) for s in candidate_strategies if 0 <= int(s) < 18]
  elif allow_multiblock:
    raw_cands = list(DEFAULT_CANDIDATE_AC_STRATEGIES)
  else:
    raw_cands = [0, 1, 2, 3, 12, 13, 14, 15, 16, 17]

  valid_cands: list[int] = []
  for s in raw_cands:
    cy = int(ACS_COVERED_BLOCKS_Y[s])
    cx = int(ACS_COVERED_BLOCKS_X[s])
    if not allow_multiblock and (cy > 1 or cx > 1):
      continue
    if cy <= h8 and cx <= w8 and s not in valid_cands:
      valid_cands.append(s)

  if candidate_strategies is None and len(valid_cands) < 3:
    for s_fallback in (0, 2, 3, 14, 15, 16, 17):
      if s_fallback not in valid_cands:
        valid_cands.append(s_fallback)
      if len(valid_cands) >= 3:
        break

  cache_key = (
      h,
      w,
      round(float(eff_distance), 4),
      round(float(rd_lambda), 4),
      int(eff_steps),
      bool(allow_multiblock),
      tuple(valid_cands),
      round(float(np.mean(target_hwc)), 6),
      round(float(np.std(target_hwc)), 6),
      round(float(target_hwc[0, 0, 0]), 6),
      round(float(target_hwc[h // 2, w // 2, 1]), 6),
  )
  if cache_key in _STAGE1_CACHE:
    return {k: dict(v) for k, v in _STAGE1_CACHE[cache_key].items()}

  acs_zero = np.zeros((h8, w8), dtype=np.uint8)
  raw_q_base, epf_sharp_base, epf_iters_base = select_quant_and_epf_jax(
      target_hwc, distance=eff_distance, ac_strategy=acs_zero
  )
  ytox_map, ytob_map, ytox_dc, ytob_dc = select_cfl_jax(target_hwc)

  base_state = init_state_from_rgb_jax(
      target_hwc,
      distance=eff_distance,
      ac_strategy=acs_zero,
      raw_quant_field=raw_q_base,
      epf_sharpness=epf_sharp_base,
      ytox_map=ytox_map,
      ytob_map=ytob_map,
      epf_iters=epf_iters_base,
      gab=True,
  )
  base_state.ytox_dc = int(ytox_dc)
  base_state.ytob_dc = int(ytob_dc)
  base_state._orig_dequant_matrices = get_default_dequant_matrices()
  base_state.dequant_matrices = {}
  base_state.dc_quant_weights = _DEFAULT_DC_QUANT_WEIGHTS_NP.copy()

  target_chw = jnp.moveaxis(jnp.asarray(target_hwc, dtype=jnp.float32), -1, 0)
  eff_scales = select_num_scales(h, w, max_scales=min(int(num_scales), 1))
  ls_map = jnp.full((h, w), float(log2_sigma), dtype=jnp.float32)
  target_features = _jit_extract_vgg_features(
      params_vgg, target_chw, eff_scales=eff_scales
  )

  yx_f = jnp.asarray(ytox_map, dtype=jnp.float32)
  yb_f = jnp.asarray(ytob_map, dtype=jnp.float32)
  num_px = float(h * w)
  num_blocks = float(max(1, h8 * w8))

  candidates: dict[int, dict[str, Any]] = {}
  for s in valid_cands:
    acs_s = _build_uniform_candidate_ac_strategy_grid(h8, w8, s)
    rq_s = broadcast_quant_field_to_ac_strategy(raw_q_base, acs_s)
    st_s = base_state.copy()
    st_s.ac_strategy = acs_s
    st_s.raw_quant_field = rq_s
    topo_s = StaticVarDCTTopology(st_s)

    with jax.ensure_compile_time_eval():
      for grp in topo_s.multi_groups:
        get_canonical_ac_scan_tables(int(grp["cy_l"]), int(grp["cx_l"]))

    u_ac_0, u_dc_0 = compute_continuous_latents_jax(
        target_hwc, st_s, topo=topo_s
    )
    rq_f = jnp.asarray(rq_s, dtype=jnp.float32)

    if eff_steps > 0:
      abs_u = jnp.abs(u_ac_0)
      shrink = float(
          np.clip(
              0.04 * ((float(rd_lambda) / 0.15) ** 0.25) * max(1, eff_steps),
              0.02,
              0.10,
          )
      )
      u_ac_1 = jnp.where(
          abs_u < 0.48,
          u_ac_0 * 0.25,
          jnp.sign(u_ac_0) * jnp.maximum(0.51, abs_u - shrink),
      )
    else:
      u_ac_1 = u_ac_0

    recon_chw = reconstruct_single_frame(
        topo_s,
        quant_ac=u_ac_1,
        dc_quant=u_dc_0,
        raw_quant_field=rq_f,
        ytox_map=yx_f,
        ytob_map=yb_f,
        include_noise=False,
        hard=False,
        output_layout="CHW",
    )
    wd_s, wd_map = _jit_spatial_wd_map_chw(
        params_vgg,
        recon_chw,
        target_features,
        ls_map,
        h8,
        w8,
        eff_scales,
        num_levels,
    )
    rate_info = compute_spatial_rate_map_bits(
        st_s,
        topology=topo_s,
        quant_ac=u_ac_1,
        dc_quant=u_dc_0,
        raw_quant_field=rq_f,
        ytox_map=yx_f,
        ytob_map=yb_f,
        ac_strategy=acs_s,
        tau=0.20,
    )
    if eff_steps > 0:
      ac_bits_0 = soft_ac_rate_bits(u_ac_0, topology=topo_s, tau=0.20)
      total_bits_0 = rate_info["total_bits"] - rate_info["ac_bits"] + ac_bits_0
    else:
      total_bits_0 = rate_info["total_bits"]

    j_init = float(wd_s) + float(rd_lambda) * (float(total_bits_0) / num_px)
    overhead_per_blk = (
        rate_info["header_bits"] + rate_info["alpha_bits"]
    ) / num_blocks
    rate_map_bits = rate_info["block_bits"] + overhead_per_blk
    j_map = wd_map + jnp.float32(rd_lambda) * (rate_map_bits / num_px)
    j_map_np = np.asarray(j_map, dtype=np.float32)
    j_score = float(np.sum(j_map_np))
    proxy_bpp = float(rate_info["total_bits"]) / num_px

    candidates[int(s)] = {
        "strategy_id": int(s),
        "ac_strategy_id": int(s),
        "strategy": int(s),
        "name": AC_STRATEGY_NAMES.get(int(s), f"ACS_{int(s)}"),
        "j_score": j_score,
        "rd_cost": j_score,
        "lagrangian": j_score,
        "j_init": j_init,
        "initial_j_score": j_init,
        "j_map": j_map_np,
        "spatial_j_map": j_map_np,
        "cost_map": j_map_np,
        "wd": float(wd_s),
        "wd_map": np.asarray(wd_map, dtype=np.float32),
        "proxy_bpp": proxy_bpp,
        "rate_map_bits": np.asarray(rate_map_bits, dtype=np.float32),
        "u_ac": np.asarray(u_ac_1, dtype=np.float32),
        "u_dc": np.asarray(u_dc_0, dtype=np.float32),
        "ac_strategy": acs_s,
        "state": st_s,
    }
  while len(_STAGE1_CACHE) >= 2:
    del _STAGE1_CACHE[next(iter(_STAGE1_CACHE))]
  _STAGE1_CACHE[cache_key] = {k: dict(v) for k, v in candidates.items()}
  return candidates


def stage2_select_non_overlapping_ac_strategy(
    candidates_or_cost_maps: (
        Mapping[int, Any] | Sequence[Mapping[str, Any]]
    ),
    h8: int | None = None,
    w8: int | None = None,
    *,
    allow_multiblock: bool = True,
) -> np.ndarray:
  """Stage 2: Selects optimal non-overlapping <=32x32 AcStrategy tiling via hierarchical DP.

  Solves the exact hierarchical non-overlapping covering DP over all <=32x32
  transforms (0..17: 1x1, 1x2, 2x1, 2x2, 1x4, 4x1, 2x4, 4x2, and 4x4 blocks)
  strictly within each 8x8-block (`64x64` pixel) CfL tile boundary.
  """
  cost_maps: dict[int, np.ndarray] = {}
  if isinstance(candidates_or_cost_maps, Mapping):
    for k, v in candidates_or_cost_maps.items():
      s_id = int(k)
      if s_id < 0 or s_id >= 18:
        continue
      if isinstance(v, Mapping):
        arr = v.get(
            "j_map", v.get("cost_map", v.get("spatial_j_map"))
        )
      else:
        arr = v
      if arr is not None:
        cost_maps[s_id] = np.asarray(arr, dtype=np.float32)
  else:
    for item in candidates_or_cost_maps:
      if isinstance(item, Mapping):
        s_id = int(
            item.get(
                "strategy_id",
                item.get("ac_strategy_id", item.get("strategy", 0)),
            )
        )
        if s_id < 0 or s_id >= 18:
          continue
        arr = item.get(
            "j_map", item.get("cost_map", item.get("spatial_j_map"))
        )
        if arr is not None:
          cost_maps[s_id] = np.asarray(arr, dtype=np.float32)

  if not cost_maps:
    if h8 is None or w8 is None:
      raise ValueError("cost_maps is empty and (h8, w8) were not provided")
    return np.zeros((int(h8), int(w8)), dtype=np.uint8)

  first_map = next(iter(cost_maps.values()))
  if h8 is None or w8 is None:
    h8, w8 = int(first_map.shape[0]), int(first_map.shape[1])
  else:
    h8, w8 = int(h8), int(w8)

  s1_ids = sorted(
      s
      for s in cost_maps
      if int(ACS_COVERED_BLOCKS_Y[s]) == 1
      and int(ACS_COVERED_BLOCKS_X[s]) == 1
  )
  if not s1_ids:
    s1_ids = [0]
    cost_maps[0] = np.full((h8, w8), 1e6, dtype=np.float32)

  s1_stack = np.stack([cost_maps[s] for s in s1_ids], axis=0)
  best_1x1_idx = np.argmin(s1_stack, axis=0)
  best_1x1_id = np.asarray(s1_ids, dtype=np.uint8)[best_1x1_idx]
  best_1x1_cost = np.min(s1_stack, axis=0)

  if not allow_multiblock:
    return best_1x1_id.copy()

  ac_strategy = best_1x1_id.copy()
  covered = np.zeros((h8, w8), dtype=bool)

  def _solve_row_1x2(r: int, c: int) -> tuple[float, np.ndarray]:
    c1 = float(best_1x1_cost[r, c] + best_1x1_cost[r, c + 1])
    g1 = best_1x1_id[r, c : c + 2].copy()
    if 7 in cost_maps:
      c7 = float(cost_maps[7][r, c] + cost_maps[7][r, c + 1])
      if c7 < c1:
        return c7, np.full((2,), 7, dtype=np.uint8)
    return c1, g1

  def _solve_col_2x1(r: int, c: int) -> tuple[float, np.ndarray]:
    c1 = float(best_1x1_cost[r, c] + best_1x1_cost[r + 1, c])
    g1 = best_1x1_id[r : r + 2, c].copy()
    if 6 in cost_maps:
      c6 = float(cost_maps[6][r, c] + cost_maps[6][r + 1, c])
      if c6 < c1:
        return c6, np.full((2,), 6, dtype=np.uint8)
    return c1, g1

  def _solve_row_1x4(r: int, c: int) -> tuple[float, np.ndarray]:
    ca, ga = _solve_row_1x2(r, c)
    cb, gb = _solve_row_1x2(r, c + 2)
    best_c = ca + cb
    best_g = np.concatenate([ga, gb], axis=0)
    if 9 in cost_maps:
      c9 = float(np.sum(cost_maps[9][r, c : c + 4]))
      if c9 < best_c:
        return c9, np.full((4,), 9, dtype=np.uint8)
    return best_c, best_g

  def _solve_col_4x1(r: int, c: int) -> tuple[float, np.ndarray]:
    ca, ga = _solve_col_2x1(r, c)
    cb, gb = _solve_col_2x1(r + 2, c)
    best_c = ca + cb
    best_g = np.concatenate([ga, gb], axis=0)
    if 8 in cost_maps:
      c8 = float(np.sum(cost_maps[8][r : r + 4, c]))
      if c8 < best_c:
        return c8, np.full((4,), 8, dtype=np.uint8)
    return best_c, best_g

  def _solve_2x2(r: int, c: int) -> tuple[float, np.ndarray]:
    best_c = float(np.sum(best_1x1_cost[r : r + 2, c : c + 2]))
    best_g = best_1x1_id[r : r + 2, c : c + 2].copy()

    cr0, gr0 = _solve_row_1x2(r, c)
    cr1, gr1 = _solve_row_1x2(r + 1, c)
    if cr0 + cr1 < best_c:
      best_c = cr0 + cr1
      best_g = np.stack([gr0, gr1], axis=0)

    cc0, gc0 = _solve_col_2x1(r, c)
    cc1, gc1 = _solve_col_2x1(r, c + 1)
    if cc0 + cc1 < best_c:
      best_c = cc0 + cc1
      best_g = np.stack([gc0, gc1], axis=1)

    if 4 in cost_maps:
      c_4 = float(np.sum(cost_maps[4][r : r + 2, c : c + 2]))
      if c_4 < best_c:
        best_c = c_4
        best_g = np.full((2, 2), 4, dtype=np.uint8)

    return best_c, best_g

  def _solve_2x4(r: int, c: int) -> tuple[float, np.ndarray]:
    c_l, g_l = _solve_2x2(r, c)
    c_r, g_r = _solve_2x2(r, c + 2)
    best_c = c_l + c_r
    best_g = np.concatenate([g_l, g_r], axis=1)

    cr0, gr0 = _solve_row_1x4(r, c)
    cr1, gr1 = _solve_row_1x4(r + 1, c)
    if cr0 + cr1 < best_c:
      best_c = cr0 + cr1
      best_g = np.stack([gr0, gr1], axis=0)

    if 11 in cost_maps:
      c11 = float(np.sum(cost_maps[11][r : r + 2, c : c + 4]))
      if c11 < best_c:
        best_c = c11
        best_g = np.full((2, 4), 11, dtype=np.uint8)

    return best_c, best_g

  def _solve_4x2(r: int, c: int) -> tuple[float, np.ndarray]:
    c_t, g_t = _solve_2x2(r, c)
    c_b, g_b = _solve_2x2(r + 2, c)
    best_c = c_t + c_b
    best_g = np.concatenate([g_t, g_b], axis=0)

    cc0, gc0 = _solve_col_4x1(r, c)
    cc1, gc1 = _solve_col_4x1(r, c + 1)
    if cc0 + cc1 < best_c:
      best_c = cc0 + cc1
      best_g = np.stack([gc0, gc1], axis=1)

    if 10 in cost_maps:
      c10 = float(np.sum(cost_maps[10][r : r + 4, c : c + 2]))
      if c10 < best_c:
        best_c = c10
        best_g = np.full((4, 2), 10, dtype=np.uint8)

    return best_c, best_g

  def _solve_4x4(r: int, c: int) -> tuple[float, np.ndarray]:
    ct, gt = _solve_2x4(r, c)
    cb, gb = _solve_2x4(r + 2, c)
    best_c = ct + cb
    best_g = np.concatenate([gt, gb], axis=0)

    cl, gl = _solve_4x2(r, c)
    cr, gr = _solve_4x2(r, c + 2)
    if cl + cr < best_c:
      best_c = cl + cr
      best_g = np.concatenate([gl, gr], axis=1)

    if 5 in cost_maps:
      c5 = float(np.sum(cost_maps[5][r : r + 4, c : c + 4]))
      if c5 < best_c:
        best_c = c5
        best_g = np.full((4, 4), 5, dtype=np.uint8)

    return best_c, best_g

  # Partition strictly within each 8x8-block (64x64 px) CfL tile so no multi-block
  # transform ever crosses a CfL tile boundary.
  num_ty = (h8 + 7) // 8
  num_tx = (w8 + 7) // 8
  for ty in range(num_ty):
    by0 = ty * 8
    by1 = min((ty + 1) * 8, h8)
    for tx in range(num_tx):
      bx0 = tx * 8
      bx1 = min((tx + 1) * 8, w8)

      for r in range(by0, by1 - 3, 4):
        for c in range(bx0, bx1 - 3, 4):
          _, g4 = _solve_4x4(r, c)
          ac_strategy[r : r + 4, c : c + 4] = g4
          covered[r : r + 4, c : c + 4] = True

      for r in range(by0, by1 - 1, 2):
        for c in range(bx0, bx1 - 3, 4):
          if np.any(covered[r : r + 2, c : c + 4]):
            continue
          _, g24 = _solve_2x4(r, c)
          ac_strategy[r : r + 2, c : c + 4] = g24
          covered[r : r + 2, c : c + 4] = True

      for r in range(by0, by1 - 3, 4):
        for c in range(bx0, bx1 - 1, 2):
          if np.any(covered[r : r + 4, c : c + 2]):
            continue
          _, g42 = _solve_4x2(r, c)
          ac_strategy[r : r + 4, c : c + 2] = g42
          covered[r : r + 4, c : c + 2] = True

      for r in range(by0, by1 - 1, 2):
        for c in range(bx0, bx1 - 1, 2):
          if np.any(covered[r : r + 2, c : c + 2]):
            continue
          _, g2 = _solve_2x2(r, c)
          ac_strategy[r : r + 2, c : c + 2] = g2
          covered[r : r + 2, c : c + 2] = True

      for r in range(by0, by1):
        for c in range(bx0, bx1 - 3, 4):
          if np.any(covered[r : r + 1, c : c + 4]):
            continue
          _, gr4 = _solve_row_1x4(r, c)
          ac_strategy[r, c : c + 4] = gr4
          covered[r, c : c + 4] = True

      for r in range(by0, by1 - 3, 4):
        for c in range(bx0, bx1):
          if np.any(covered[r : r + 4, c : c + 1]):
            continue
          _, gc4 = _solve_col_4x1(r, c)
          ac_strategy[r : r + 4, c] = gc4
          covered[r : r + 4, c] = True

      for r in range(by0, by1):
        for c in range(bx0, bx1 - 1, 2):
          if np.any(covered[r : r + 1, c : c + 2]):
            continue
          _, gr2 = _solve_row_1x2(r, c)
          ac_strategy[r, c : c + 2] = gr2
          covered[r, c : c + 2] = True

      for r in range(by0, by1 - 1, 2):
        for c in range(bx0, bx1):
          if np.any(covered[r : r + 2, c : c + 1]):
            continue
          _, gc2 = _solve_col_2x1(r, c)
          ac_strategy[r : r + 2, c] = gc2
          covered[r : r + 2, c] = True

  return ac_strategy


def select_ac_strategy_jax(
    rgb_01: np.ndarray | jax.Array,
    distance: float = 1.0,
    allow_multiblock: bool = True,
    distance_adaptive: bool = False,
    *,
    vgg_params: Any = None,
    rd_lambda: float | None = None,
    stage1_steps: int = 1,
) -> np.ndarray:
  """Pure-JAX multi-block AcStrategy selector driven by Stage 1 Rate-WD + Stage 2 DP."""
  if float(distance) <= 0.0:
    raise ValueError(f"distance must be > 0, got {distance}")
  arr = _normalize_rgb_hwc(rgb_01)
  h, w, _ = arr.shape
  h8 = (h + 7) // 8
  w8 = (w + 7) // 8

  if rd_lambda is not None:
    d_eval = float(distance)
    lam = float(rd_lambda)
  elif distance_adaptive:
    d_eval = float(distance)
    lam = float(0.12 * (np.clip(float(distance), 0.15, 12.0) ** 1.15))
  else:
    d_eval = 1.5
    lam = 0.15

  cands = stage1_explore_ac_candidates(
      arr,
      vgg_params=vgg_params,
      distance=d_eval,
      rd_lambda=lam,
      steps=max(0, int(stage1_steps)),
      allow_multiblock=allow_multiblock,
      scale_distance_with_lambda=False,
  )
  return stage2_select_non_overlapping_ac_strategy(
      cands, h8=h8, w8=w8, allow_multiblock=allow_multiblock
  )


def select_quant_and_epf_jax(
    rgb_01: np.ndarray | jax.Array,
    distance: float = 1.0,
    ac_strategy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
  """Pure-JAX perceptual masking initialization of raw_quant_field and epf_sharpness."""
  if float(distance) <= 0.0:
    raise ValueError(f"distance must be > 0, got {distance}")
  arr = _normalize_rgb_hwc(rgb_01)
  h, w, _ = arr.shape
  h8 = (h + 7) // 8
  w8 = (w + 7) // 8
  if ac_strategy is None:
    ac_strategy = select_ac_strategy_jax(arr, distance=distance)

  srgb_chw = jnp.array(np.transpose(arr, (2, 0, 1)), dtype=jnp.float32)
  e_gx_j, e_gy_j, e_hf_j, ehf_smooth_j = _compute_aq_and_epf_maps_jax(srgb_chw)
  e_gx = np.asarray(e_gx_j)
  e_gy = np.asarray(e_gy_j)
  e_hf = np.asarray(e_hf_j)
  ehf_smooth = np.asarray(ehf_smooth_j)

  aniso_np = np.abs(e_gx - e_gy) / (e_gx + e_gy + 1e-5)
  med_e_np = float(np.median(ehf_smooth)) + 1e-5
  rel_act = (ehf_smooth + 1e-5) / med_e_np

  d_clamped = max(float(distance), 0.05)
  bt_dc = max(0.5 * d_clamped, min(d_clamped, 0.3 * ((d_clamped / 0.3) ** 0.83)))
  qdc_float = min(1.095924047623553 / bt_dc, 50.0)
  gs_est = int(np.clip(round(65536.0 * (0.390 / d_clamped) / 5.0), 1, 32768))
  scaled_qdc = max(1, int(qdc_float * 4096.0 * 1.6))
  if gs_est > scaled_qdc:
    gs_est = scaled_qdc
  inv_gs_est = 65536.0 / float(gs_est)
  gab_boost = 1.0 + 0.38 * float(np.clip((0.85 - d_clamped) / 0.50, 0.0, 1.0))
  base_raw_q = (0.546 / d_clamped) * gab_boost * inv_gs_est

  raw_q_float = base_raw_q * np.clip(rel_act**0.12, 0.75, 1.35)
  raw_q_np = np.clip(np.round(raw_q_float), 1, 256).astype(np.int32)
  raw_q_np = broadcast_quant_field_to_ac_strategy(raw_q_np, ac_strategy)

  if float(distance) < 0.5:
    epf_sharpness = np.zeros((h8, w8), dtype=np.uint8)
    epf_iters = 0
    return raw_q_np, epf_sharpness, epf_iters

  epf_sharpness = np.full((h8, w8), 2, dtype=np.uint8)
  for by in range(h8):
    for bx in range(w8):
      s = int(ac_strategy[by, bx])
      if s in (4, 5, 8, 9, 10, 11):
        epf_sharpness[by, bx] = 3
      elif e_hf[by, bx] > 1.8 * med_e_np and aniso_np[by, bx] < 0.25:
        epf_sharpness[by, bx] = 0
      elif e_hf[by, bx] > 1.1 * med_e_np and aniso_np[by, bx] < 0.35:
        epf_sharpness[by, bx] = 1
      elif aniso_np[by, bx] > 0.45:
        epf_sharpness[by, bx] = 3
      else:
        epf_sharpness[by, bx] = 2

  d_val = float(distance)
  epf_iters = 1 if d_val < 1.5 else (2 if d_val < 4.0 else 3)
  return raw_q_np, epf_sharpness, epf_iters


def select_cfl_jax(
    rgb_01: np.ndarray | jax.Array,
    color_factor: int = K_DEFAULT_COLOR_FACTOR,
    base_correlation_x: float = K_BASE_CORRELATION_X,
    base_correlation_b: float = K_BASE_CORRELATION_B,
    gab: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, int]:
  """Pure-JAX closed-form ridge least-squares Chroma-from-Luma (CfL) estimator."""
  arr = _normalize_rgb_hwc(rgb_01)
  h, w, _ = arr.shape
  h8 = (h + 7) // 8
  w8 = (w + 7) // 8
  ht = (h8 + 7) // 8
  wt = (w8 + 7) // 8

  srgb_chw = jnp.array(np.transpose(arr, (2, 0, 1)), dtype=jnp.float32)
  xyb = srgb_to_xyb(srgb_chw)
  if gab:
    xyb = gaborish_inverse_3x3(xyb, K_DEFAULT_GAB_WEIGHTS)

  pad_h = h8 * 8 - h
  pad_w = w8 * 8 - w
  if pad_h > 0 or pad_w > 0:
    xyb = jnp.pad(xyb, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

  blocks = jnp.transpose(xyb.reshape(3, h8, 8, w8, 8), (0, 1, 3, 2, 4)).reshape(
      3, h8, w8, 64
  )
  dc_means = jnp.mean(blocks, axis=-1, keepdims=True)
  ac_fluct = blocks - dc_means

  pad_bh = ht * 8 - h8
  pad_bw = wt * 8 - w8
  if pad_bh > 0 or pad_bw > 0:
    ac_fluct_pad = jnp.pad(
        ac_fluct, ((0, 0), (0, pad_bh), (0, pad_bw), (0, 0)), mode="constant"
    )
  else:
    ac_fluct_pad = ac_fluct

  tiles = ac_fluct_pad.reshape(3, ht, 8, wt, 8, 64)
  tx = tiles[0]
  ty = tiles[1]
  tb = tiles[2]

  yy_energy = jnp.sum(ty * ty, axis=(1, 3, 4))
  xy_cov = jnp.sum(tx * ty, axis=(1, 3, 4))
  by_cov = jnp.sum(tb * ty, axis=(1, 3, 4))

  reg_x = 0.025
  reg_b = 0.045
  ratio_x = (xy_cov + reg_x * float(base_correlation_x)) / (yy_energy + reg_x)
  ratio_b = (by_cov + reg_b * float(base_correlation_b)) / (yy_energy + reg_b)

  ytox_f = jnp.round(
      float(color_factor) * (ratio_x - float(base_correlation_x))
  )
  ytob_f = jnp.round(
      float(color_factor) * (ratio_b - float(base_correlation_b))
  )

  ytox_map = np.clip(np.asarray(ytox_f), -64, 64).astype(np.int8)
  ytob_map = np.clip(np.asarray(ytob_f), -64, 64).astype(np.int8)

  dc_2d = dc_means[..., 0]
  dc_centered = dc_2d - jnp.mean(dc_2d, axis=(1, 2), keepdims=True)
  dc_yy = float(jnp.sum(dc_centered[1] * dc_centered[1]))
  dc_xy = float(jnp.sum(dc_centered[0] * dc_centered[1]))
  dc_by = float(jnp.sum(dc_centered[2] * dc_centered[1]))

  dc_reg = 0.002 * float(max(h8 * w8, 1))
  dc_ratio_x = (dc_xy + dc_reg * float(base_correlation_x)) / (dc_yy + dc_reg)
  dc_ratio_b = (dc_by + dc_reg * float(base_correlation_b)) / (dc_yy + dc_reg)

  ytox_dc = int(
      np.clip(
          round(float(color_factor) * (dc_ratio_x - float(base_correlation_x))),
          -64,
          64,
      )
  )
  ytob_dc = int(
      np.clip(
          round(float(color_factor) * (dc_ratio_b - float(base_correlation_b))),
          -64,
          64,
      )
  )
  return ytox_map, ytob_map, ytox_dc, ytob_dc


def select_block_params_jax(
    rgb_01: np.ndarray | jax.Array,
    distance: float = 1.0,
    allow_multiblock: bool = True,
    *,
    vgg_params: Any = None,
    rd_lambda: float | None = None,
) -> dict[str, Any]:
  """Unified pure-JAX per-block parameter selector (R2: zero libjxl choices)."""
  if float(distance) <= 0.0:
    raise ValueError(f"distance must be > 0, got {distance}")
  ac_strategy = select_ac_strategy_jax(
      rgb_01,
      distance=distance,
      allow_multiblock=allow_multiblock,
      vgg_params=vgg_params,
      rd_lambda=rd_lambda,
  )
  raw_quant_field, epf_sharpness, epf_iters = select_quant_and_epf_jax(
      rgb_01, distance=distance, ac_strategy=ac_strategy
  )
  ytox_map, ytob_map, ytox_dc, ytob_dc = select_cfl_jax(rgb_01)
  return {
      "ac_strategy": ac_strategy,
      "raw_quant_field": raw_quant_field,
      "epf_sharpness": epf_sharpness,
      "epf_iters": epf_iters,
      "ytox_map": ytox_map,
      "ytob_map": ytob_map,
      "ytox_dc": ytox_dc,
      "ytob_dc": ytob_dc,
  }


# =============================================================================
# 2. Continuous Unrounded Latent Extraction & Noise LUT Initialization
# =============================================================================


def _normalize_init_mode(init_mode: str | None) -> str:
  """Validates and normalizes `init_mode` in `{'dct', 'zero', 'random'}`."""
  mode = "dct" if init_mode is None else str(init_mode).strip().lower()
  if mode in ("0", "zeros"):
    mode = "zero"
  elif mode in ("noise", "rand"):
    mode = "random"
  if mode not in ("dct", "zero", "random"):
    raise ValueError(
        f"init_mode must be in {{'dct', 'zero', 'random'}}, got {init_mode!r}"
    )
  return mode


def _build_ac_freq_weight_2d_np(topo: StaticVarDCTTopology) -> np.ndarray:
  """Builds the `(H8*8, W8*8)` normalized frequency decay preconditioner matching `topo`."""
  h8, w8 = topo.h8, topo.w8
  u8 = np.arange(8, dtype=np.float32)[:, None]
  v8 = np.arange(8, dtype=np.float32)[None, :]
  w8x8 = 1.0 / (0.15 + 0.85 * (u8 * u8 + v8 * v8))
  w8x8[0, 0] = 0.0
  w_blk = np.broadcast_to(w8x8[None, None], (h8, w8, 8, 8)).copy()
  for grp1 in topo.groups_1x1:
    s_id = int(grp1["s"])
    if s_id == 1:  # IDENTITY (spatial deltas)
      m2d = np.asarray(grp1["mask_2d"], dtype=bool)
      id_w = np.full((8, 8), 0.35, dtype=np.float32)
      id_w[0, 0] = 0.0
      w_blk[m2d] = id_w
  for grp in topo.multi_groups:
    cy, cx = int(grp["cy"]), int(grp["cx"])
    cy_l, cx_l = int(grp["cy_l"]), int(grp["cx_l"])
    u_m = np.arange(8 * cy_l, dtype=np.float32)[:, None] / float(cy_l)
    v_m = np.arange(8 * cx_l, dtype=np.float32)[None, :] / float(cx_l)
    w_canon = 1.0 / (0.15 + 0.85 * (u_m * u_m + v_m * v_m))
    w_canon[:cy_l, :cx_l] = 0.0
    w_s = w_canon.reshape(cy, 8, cx, 8).transpose(0, 2, 1, 3)
    bys = np.asarray(grp["by"], dtype=np.int32)
    bxs = np.asarray(grp["bx"], dtype=np.int32)
    for by, bx in zip(bys.tolist(), bxs.tolist()):
      w_blk[by : by + cy, bx : bx + cx] = w_s
  return w_blk.transpose(0, 2, 1, 3).reshape(h8 * 8, w8 * 8)


def _compute_precond_scales_jax(
    state_or_topo: JxlVarDCTState | StaticVarDCTTopology,
    raw_quant_tl: jax.Array | np.ndarray,
) -> tuple[jax.Array, jax.Array]:
  """Computes XYB residual-calibrated `(scale_dc, scale_ac)` preconditioners for zero/random init."""
  topo = (
      state_or_topo
      if isinstance(state_or_topo, StaticVarDCTTopology)
      else StaticVarDCTTopology(state_or_topo)
  )
  inv_gs = K_GLOBAL_SCALE_DENOM / float(topo.global_scale)
  qdc = float(topo.quant_dc)
  dc_step_ratio = jnp.float32(1.13 / max(inv_gs / max(qdc, 1.0), 1e-6))
  scale_dc = (
      jnp.array([48.0, 200.0, 36.0], dtype=jnp.float32)[:, None, None]
      * dc_step_ratio
  )
  rq_tl_f = jnp.maximum(jnp.asarray(raw_quant_tl, dtype=jnp.float32), 1.0)
  rq_px = jnp.repeat(jnp.repeat(rq_tl_f, 8, axis=0), 8, axis=1)
  ac_step_ratio = 1.20 / jnp.maximum(inv_gs / rq_px, 1e-6)
  with jax.ensure_compile_time_eval():
    w_freq = jnp.asarray(_build_ac_freq_weight_2d_np(topo), dtype=jnp.float32)
  scale_ac = (
      jnp.array([6.0, 12.0, 3.8], dtype=jnp.float32)[:, None, None]
      * (ac_step_ratio * w_freq)[None, :, :]
  )
  return scale_dc, scale_ac


def mask_llf_jax(q_ac: jax.Array, topo: StaticVarDCTTopology) -> jax.Array:
  """Zeroes out the LLF positions of `q_ac` `(3, H8*8, W8*8)` in JAX (JIT-safe)."""
  mask = getattr(topo, "_llf_mask_jax", None)
  if mask is None:
    ones = np.ones((3, topo.h8 * 8, topo.w8 * 8), dtype=np.float32)
    with jax.ensure_compile_time_eval():
      mask = jnp.asarray(mask_llf_numpy(ones, topo)[:1], dtype=jnp.float32)
    try:
      setattr(topo, "_llf_mask_jax", mask)
    except Exception:
      pass
  return jnp.asarray(q_ac, dtype=jnp.float32) * mask


def compute_continuous_latents_jax(
    rgb_01: np.ndarray | jax.Array,
    state: JxlVarDCTState,
    topo: StaticVarDCTTopology | None = None,
    *,
    init_mode: str = "dct",
    seed: int = 0,
) -> tuple[jax.Array, jax.Array]:
  """Computes initial continuous `(q_ac_raw, q_dc_raw)` for `init_mode in {'dct', 'zero', 'random'}`."""
  mode = _normalize_init_mode(init_mode)
  h8, w8 = state.h8, state.w8
  if topo is None:
    topo = StaticVarDCTTopology(state)

  if mode == "zero":
    q_ac_zero = jnp.zeros((3, h8 * 8, w8 * 8), dtype=jnp.float32)
    q_dc_zero = jnp.zeros((3, h8, w8), dtype=jnp.float32)
    return q_ac_zero, q_dc_zero

  if mode == "random":
    rq_tl = jnp.asarray(state.raw_quant_field, dtype=jnp.float32)[
        topo.tl_by_grid, topo.tl_bx_grid
    ]
    scale_dc, scale_ac = _compute_precond_scales_jax(topo, rq_tl)
    key = jax.random.PRNGKey(int(seed))
    k_dc, k_ac = jax.random.split(key)
    z_dc_std = jnp.array([0.005, 0.06, 0.008], dtype=jnp.float32)[:, None, None]
    z_dc = jax.random.normal(k_dc, (3, h8, w8), dtype=jnp.float32) * z_dc_std
    z_dc = z_dc.at[1].set(jnp.abs(z_dc[1]))
    z_ac = (
        jax.random.normal(k_ac, (3, h8 * 8, w8 * 8), dtype=jnp.float32) * 0.002
    )
    q_dc_rand = z_dc * scale_dc
    q_ac_rand = jnp.asarray(
        mask_llf_numpy(np.asarray(z_ac * scale_ac), topo), dtype=jnp.float32
    )
    return q_ac_rand, q_dc_rand

  arr = _normalize_rgb_hwc(rgb_01)
  h, w, _ = arr.shape

  srgb_chw = jnp.array(np.transpose(arr, (2, 0, 1)), dtype=jnp.float32)
  xyb = srgb_to_xyb(srgb_chw)
  if state.gab:
    xyb = gaborish_inverse_3x3(xyb, state.gab_weights)

  pad_h = h8 * 8 - h
  pad_w = w8 * 8 - w
  if pad_h > 0 or pad_w > 0:
    xyb = jnp.pad(xyb, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

  xyb_blocks = jnp.transpose(xyb.reshape(3, h8, 8, w8, 8), (1, 3, 0, 2, 4))

  raw_q_f = jnp.maximum(jnp.array(state.raw_quant_field, dtype=jnp.float32), 1.0)
  qac_scale_grid = (float(state.global_scale) / K_GLOBAL_SCALE_DENOM) * raw_q_f
  x_qm_mul = float(1.25 ** (state.x_qm_scale - 2.0))
  b_qm_mul = float(1.25 ** (state.b_qm_scale - 2.0))

  x_cc_tile = (
      float(state.base_correlation_x)
      + jnp.array(state.ytox_map, dtype=jnp.float32) / float(state.color_factor)
  )
  b_cc_tile = (
      float(state.base_correlation_b)
      + jnp.array(state.ytob_map, dtype=jnp.float32) / float(state.color_factor)
  )

  dc_continuous = jnp.zeros((3, h8, w8), dtype=jnp.float32)
  q_ac_blocks = jnp.zeros((h8, w8, 3, 8, 8), dtype=jnp.float32)

  ntiles_y, ntiles_x = x_cc_tile.shape
  x_cc_grid = jnp.broadcast_to(
      x_cc_tile[:, None, :, None], (ntiles_y, 8, ntiles_x, 8)
  ).reshape(ntiles_y * 8, ntiles_x * 8)[:h8, :w8]
  b_cc_grid = jnp.broadcast_to(
      b_cc_tile[:, None, :, None], (ntiles_y, 8, ntiles_x, 8)
  ).reshape(ntiles_y * 8, ntiles_x * 8)[:h8, :w8]

  if len(topo.groups_1x1) > 0:
    px_flat = xyb_blocks.reshape(h8, w8, 3, 64)
    qs_all = qac_scale_grid[:, :, None, None]
    x_cc_all = x_cc_grid[:, :, None, None]
    b_cc_all = b_cc_grid[:, :, None, None]
    for grp1 in topo.groups_1x1:
      with jax.ensure_compile_time_eval():
        mask_2d = jnp.asarray(grp1["mask_2d"], dtype=jnp.float32)
      coeffs = jnp.einsum("ij,hwcj->hwci", grp1["fwd_basis"], px_flat).reshape(
          h8, w8, 3, 8, 8
      )
      dc_continuous = dc_continuous + (
          jnp.transpose(coeffs[:, :, :, 0, 0], (2, 0, 1)) * mask_2d[None, :, :]
      )
      dq_mat = grp1["dq_mat"]
      wy = dq_mat[1][None, None]
      wx = dq_mat[0][None, None]
      wb = dq_mat[2][None, None]
      cy_val = coeffs[:, :, 1]
      cx_val = coeffs[:, :, 0] - x_cc_all * cy_val
      cb_val = coeffs[:, :, 2] - b_cc_all * cy_val
      qy = cy_val * (qs_all / wy)
      qx = cx_val * (qs_all * x_qm_mul / wx)
      qb = cb_val * (qs_all * b_qm_mul / wb)
      q_blk = jnp.stack([qx, qy, qb], axis=2).at[:, :, :, 0, 0].set(0.0)
      q_ac_blocks = q_ac_blocks + q_blk * mask_2d[:, :, None, None, None]

  for grp in topo.multi_groups:
    cy = grp["cy"]
    cx = grp["cx"]
    cy_l = grp["cy_l"]
    cx_l = grp["cx_l"]
    dq_mat = grp["dq_mat"]
    if grp.get("grid_aligned", False):
      nh_cell = grp["nh_cell"]
      nw_cell = grp["nw_cell"]
      h_lim = grp["h_lim"]
      w_lim = grp["w_lim"]
      with jax.ensure_compile_time_eval():
        blk_mask = jnp.asarray(grp["blk_mask"], dtype=jnp.float32)
      px_cell = xyb_blocks[:h_lim, :w_lim].reshape(
          nh_cell, cy, nw_cell, cx, 3, 8, 8
      )
      px_full = jnp.transpose(px_cell, (0, 2, 4, 1, 5, 3, 6)).reshape(
          nh_cell, nw_cell, 3, 8 * cy, 8 * cx
      )
      coeffs_rc = jnp.einsum(
          "ij,abcjk,lk->abcil", grp["dct_r"], px_full, grp["dct_c"]
      )
      coeffs = jnp.swapaxes(coeffs_rc, -2, -1) if cy >= cx else coeffs_rc
      llf = coeffs[:, :, :, :cy_l, :cx_l]
      llf_rc = jnp.swapaxes(llf, -2, -1) if cy >= cx else llf
      llf_scaled = llf_rc * (
          grp["rs_from_llf_y"][None, None, None, :, None]
          * grp["rs_from_llf_x"][None, None, None, None, :]
      )
      dc_sub = jnp.einsum(
          "ij,abcjk,lk->abcil", grp["idct_cy"], llf_scaled, grp["idct_cx"]
      )
      dc_grid = (
          jnp.transpose(dc_sub, (2, 0, 3, 1, 4)).reshape(3, h_lim, w_lim)
          * blk_mask[None, :, :]
      )
      if h_lim == h8 and w_lim == w8:
        dc_continuous = dc_continuous + dc_grid
      else:
        dc_continuous = dc_continuous.at[:, :h_lim, :w_lim].add(dc_grid)

      qs = qac_scale_grid[:h_lim:cy, :w_lim:cx][:, :, None, None]
      x_cc = x_cc_grid[:h_lim:cy, :w_lim:cx][:, :, None, None]
      b_cc = b_cc_grid[:h_lim:cy, :w_lim:cx][:, :, None, None]
      cy_val = coeffs[:, :, 1]
      cx_val = coeffs[:, :, 0] - x_cc * cy_val
      cb_val = coeffs[:, :, 2] - b_cc * cy_val
      qy = cy_val * (qs / dq_mat[1][None, None])
      qx = cx_val * (qs * x_qm_mul / dq_mat[0][None, None])
      qb = cb_val * (qs * b_qm_mul / dq_mat[2][None, None])
      q_canon = (
          jnp.stack([qx, qy, qb], axis=2).at[:, :, :, :cy_l, :cx_l].set(0.0)
      )
      q_sub = (
          jnp.transpose(
              q_canon.reshape(nh_cell, nw_cell, 3, cy, 8, cx, 8),
              (0, 3, 1, 5, 2, 4, 6),
          ).reshape(h_lim, w_lim, 3, 8, 8)
          * blk_mask[:, :, None, None, None]
      )
      if h_lim == h8 and w_lim == w8:
        q_ac_blocks = q_ac_blocks + q_sub
      else:
        q_ac_blocks = q_ac_blocks.at[:h_lim, :w_lim].add(q_sub)
    else:
      by = grp["by"]
      bx = grp["bx"]
      m = by.shape[0]
      dy = jnp.arange(cy)[None, :, None]
      dx = jnp.arange(cx)[None, None, :]

      px_sub = xyb_blocks[by[:, None, None] + dy, bx[:, None, None] + dx]
      px_full = jnp.transpose(px_sub, (0, 3, 1, 4, 2, 5)).reshape(
          m, 3, 8 * cy, 8 * cx
      )
      coeffs_rc = jnp.einsum(
          "ij,mcjk,lk->mcil", grp["dct_r"], px_full, grp["dct_c"]
      )
      coeffs = jnp.swapaxes(coeffs_rc, -2, -1) if cy >= cx else coeffs_rc

      llf = coeffs[:, :, :cy_l, :cx_l]
      llf_rc = jnp.swapaxes(llf, -2, -1) if cy >= cx else llf
      llf_scaled = llf_rc * (
          grp["rs_from_llf_y"][None, None, :, None]
          * grp["rs_from_llf_x"][None, None, None, :]
      )
      dc_sub = jnp.einsum(
          "ij,mcjk,lk->mcil", grp["idct_cy"], llf_scaled, grp["idct_cx"]
      )
      dc_continuous = dc_continuous.at[
          :, by[:, None, None] + dy, bx[:, None, None] + dx
      ].set(jnp.transpose(dc_sub, (1, 0, 2, 3)))

      qs = qac_scale_grid[by, bx][:, None, None]
      ty = by // 8
      tx = bx // 8
      x_cc = x_cc_tile[ty, tx][:, None, None]
      b_cc = b_cc_tile[ty, tx][:, None, None]

      cy_val = coeffs[:, 1]
      cx_val = coeffs[:, 0] - x_cc * cy_val
      cb_val = coeffs[:, 2] - b_cc * cy_val

      qy = cy_val * (qs / dq_mat[1][None])
      qx = cx_val * (qs * x_qm_mul / dq_mat[0][None])
      qb = cb_val * (qs * b_qm_mul / dq_mat[2][None])
      q_canon = jnp.stack([qx, qy, qb], axis=1).at[:, :, :cy_l, :cx_l].set(0.0)
      q_sub = jnp.transpose(
          q_canon.reshape(m, 3, cy, 8, cx, 8), (0, 2, 4, 1, 3, 5)
      )
      q_ac_blocks = q_ac_blocks.at[
          by[:, None, None] + dy, bx[:, None, None] + dx
      ].set(q_sub)

  inv_global_scale = K_GLOBAL_SCALE_DENOM / float(state.global_scale)
  mul_dc = [inv_global_scale / float(state.quant_dc) * k for k in K_DC_QUANT]
  cfl_dc_x = float(state.base_correlation_x) + float(state.ytox_dc) / float(
      state.color_factor
  )
  cfl_dc_b = float(state.base_correlation_b) + float(state.ytob_dc) / float(
      state.color_factor
  )

  q_dc_y = dc_continuous[1] / mul_dc[1]
  q_dc_x = (
      dc_continuous[0] - jnp.round(q_dc_y) * (mul_dc[1] * cfl_dc_x)
  ) / mul_dc[0]
  q_dc_b = (
      dc_continuous[2] - jnp.round(q_dc_y) * (mul_dc[1] * cfl_dc_b)
  ) / mul_dc[2]

  q_dc_raw = jnp.stack([q_dc_x, q_dc_y, q_dc_b], axis=0)
  q_ac_raw = jnp.transpose(q_ac_blocks, (2, 0, 3, 1, 4)).reshape(
      3, h8 * 8, w8 * 8
  )
  return q_ac_raw, q_dc_raw


def mask_llf_numpy(
    quant_ac_np: np.ndarray, topo: StaticVarDCTTopology
) -> np.ndarray:
  """Zeros out all canonical LLF coefficients in (3, H8*8, W8*8) int32 quant_ac."""
  h8, w8 = topo.h8, topo.w8
  q_blocks = np.transpose(
      quant_ac_np.reshape(3, h8, 8, w8, 8), (1, 3, 0, 2, 4)
  ).copy()
  if topo.has_1x1:
    by = np.asarray(topo.by_1x1)
    bx = np.asarray(topo.bx_1x1)
    q_blocks[by, bx, :, 0, 0] = 0
  for grp in topo.multi_groups:
    cy = grp["cy"]
    cx = grp["cx"]
    cy_l = grp["cy_l"]
    cx_l = grp["cx_l"]
    by = np.asarray(grp["by"])
    bx = np.asarray(grp["bx"])
    m = by.shape[0]
    dy = np.arange(cy)[None, :, None]
    dx = np.arange(cx)[None, None, :]
    sub = q_blocks[by[:, None, None] + dy, bx[:, None, None] + dx]
    canon = np.transpose(sub, (0, 3, 1, 4, 2, 5)).reshape(
        m, 3, 8 * cy_l, 8 * cx_l
    )
    canon[:, :, :cy_l, :cx_l] = 0
    sub_zeroed = np.transpose(
        canon.reshape(m, 3, cy, 8, cx, 8), (0, 2, 4, 1, 3, 5)
    )
    q_blocks[by[:, None, None] + dy, bx[:, None, None] + dx] = sub_zeroed
  return (
      np.transpose(q_blocks, (2, 0, 3, 1, 4))
      .reshape(3, h8 * 8, w8 * 8)
      .astype(np.int32)
  )


def apply_hf_noise_denoising_jax(
    q_ac_raw: jax.Array,
    h8: int,
    w8: int,
    hf_shrink: float = 0.28,
) -> jax.Array:
  """Softly attenuates high-frequency AC grain when procedural noise synthesis is enabled."""
  q_b = jnp.transpose(q_ac_raw.reshape(3, h8, 8, w8, 8), (1, 3, 0, 2, 4))
  fy = jnp.arange(8, dtype=jnp.float32)[:, None]
  fx = jnp.arange(8, dtype=jnp.float32)[None, :]
  r = jnp.sqrt(fy * fy + fx * fx)
  hf_weight = jnp.clip((r - 1.5) / 5.5, 0.0, 1.0)[None, None, None, :, :]
  shrink_amt = float(hf_shrink) * hf_weight
  q_shrunk = jnp.sign(q_b) * jnp.maximum(0.0, jnp.abs(q_b) - shrink_amt)
  q_shrunk = q_shrunk.at[:, :, :, 0, 0].set(0.0)
  return jnp.transpose(q_shrunk, (2, 0, 3, 1, 4)).reshape(3, h8 * 8, w8 * 8)


def estimate_initial_noise_lut_jax(
    target_rgb_01: np.ndarray | jax.Array,
    recon_no_noise_01: np.ndarray | jax.Array,
) -> jax.Array:
  """Estimates initial 8-bin JXL noise_lut in [0.002, 0.035] from residual XYB HF variance."""
  tgt_hwc = _normalize_rgb_hwc(target_rgb_01)
  rec_hwc = _normalize_rgb_hwc(recon_no_noise_01)
  tgt_xyb = srgb_to_xyb(jnp.moveaxis(jnp.asarray(tgt_hwc), -1, 0))
  rec_xyb = srgb_to_xyb(jnp.moveaxis(jnp.asarray(rec_hwc), -1, 0))

  def _hp_3x3(plane: jax.Array) -> jax.Array:
    p = jnp.pad(plane, ((1, 1), (1, 1)), mode="edge")
    mean = (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 1:-1]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    ) / 9.0
    return plane - mean

  hp_t = _hp_3x3(tgt_xyb[1])
  hp_r = _hp_3x3(rec_xyb[1])

  gy = jnp.abs(tgt_xyb[1, 1:, :] - tgt_xyb[1, :-1, :])
  gx = jnp.abs(tgt_xyb[1, :, 1:] - tgt_xyb[1, :, :-1])
  gy = jnp.pad(gy, ((0, 1), (0, 0)), mode="edge")
  gx = jnp.pad(gx, ((0, 0), (0, 1)), mode="edge")
  grad_mag = jnp.sqrt(gx * gx + gy * gy + 1e-8)
  edge_thresh = jnp.percentile(grad_mag, 80.0) + 1e-4
  tex_mask = (grad_mag <= edge_thresh).astype(jnp.float32)

  var_t = jnp.sum(tex_mask * hp_t * hp_t) / jnp.maximum(jnp.sum(tex_mask), 1.0)
  var_r = jnp.sum(tex_mask * hp_r * hp_r) / jnp.maximum(jnp.sum(tex_mask), 1.0)
  missing_std = jnp.sqrt(jnp.maximum(var_t - var_r, 0.0))

  base_strength = jnp.clip(missing_std * 0.25, 0.0025, 0.015)
  ramp = jnp.array(
      [0.75, 0.90, 1.00, 1.05, 1.05, 1.00, 0.95, 0.90], dtype=jnp.float32
  )
  lut = jnp.clip(base_strength * ramp, 2.0 / NOISE_PRECISION, 0.025)
  return lut


# =============================================================================
# 3. Three-Stage Gradient-Based Rate-WD Optimizer (optimize_single_frame)
# =============================================================================


def rd_lambda_to_distance(
    rd_lambda: float,
    base_distance: float | None = None,
    scale_distance_with_lambda: bool = True,
) -> float:
  """Smoothly maps rd_lambda to effective JXL initialization distance."""
  if float(rd_lambda) <= 0.0:
    raise ValueError(f"rd_lambda must be > 0, got {rd_lambda}")
  if base_distance is not None and float(base_distance) <= 0.0:
    raise ValueError(f"base_distance must be > 0, got {base_distance}")
  lam = max(float(rd_lambda), 1e-5)
  if not scale_distance_with_lambda and base_distance is not None:
    return float(np.clip(base_distance, 0.1, 15.0))
  d0 = 1.0 if base_distance is None else float(base_distance)
  d_eff = d0 * ((lam / 0.08) ** 0.62)
  return float(np.clip(d_eff, 0.15, 12.0))


@dataclass
class OptimizationResult:
  """Container for single-frame 3-stage Rate-WD optimization results."""

  jxl_bytes: bytes
  bpp: float
  wd: float
  lagrangian: float
  psnr_jax_vs_djxl: float
  recon_jax_rgb: np.ndarray
  recon_djxl_rgb: np.ndarray
  state: JxlVarDCTState
  noise_lut: list[float] | None = None
  history: list[dict[str, float]] = field(default_factory=list)
  psnr_db: float = 0.0
  num_bytes: int = 0
  step0_jxl_bytes: bytes = b""
  step0_bpp: float = 0.0
  step0_wd: float = 0.0
  step0_psnr_db: float = 0.0
  cross_decoder_psnr: float | None = None
  stage1_candidates: dict[int, dict[str, Any]] | None = None
  stage2_ac_strategy: np.ndarray | None = None
  stage2_true_bpp: float = 0.0
  stage2_wd: float = 0.0
  stage2_true_j: float = 0.0
  stage3_true_j: float = 0.0
  all_default: bool | None = None

  def __post_init__(self) -> None:
    if self.num_bytes == 0 and self.jxl_bytes:
      self.num_bytes = len(self.jxl_bytes)
    if self.stage2_ac_strategy is None and self.state is not None:
      self.stage2_ac_strategy = np.asarray(
          self.state.ac_strategy, dtype=np.uint8
      ).copy()
    if self.stage2_true_bpp == 0.0 and self.step0_bpp > 0.0:
      self.stage2_true_bpp = self.step0_bpp
    if self.stage2_wd == 0.0 and self.step0_wd > 0.0:
      self.stage2_wd = self.step0_wd
    if self.stage3_true_j == 0.0 and self.lagrangian > 0.0:
      self.stage3_true_j = self.lagrangian
    if self.stage2_true_j == 0.0 and self.history:
      self.stage2_true_j = float(
          self.history[0].get("lagrangian_true", self.lagrangian)
      )

  @property
  def rd_cost(self) -> float:
    return self.lagrangian

  @property
  def wd_final(self) -> float:
    return self.wd

  @property
  def wd_init(self) -> float:
    return self.step0_wd

  @property
  def bpp_final(self) -> float:
    return self.bpp

  @property
  def bpp_init(self) -> float:
    return self.step0_bpp

  @property
  def psnr(self) -> float:
    return self.psnr_db

  @property
  def ac_strategy(self) -> np.ndarray:
    return self.state.ac_strategy

  @property
  def mode(self) -> str:
    return "single"

  @property
  def mode_selected(self) -> str:
    return "single"

  @property
  def mode_used(self) -> str:
    return "single"

  def to_dict(self) -> dict[str, Any]:
    d = {f.name: getattr(self, f.name) for f in fields(self)}
    d["rd_cost"] = self.lagrangian
    d["psnr"] = self.psnr_db
    d["ac_strategy"] = self.state.ac_strategy
    d["mode"] = "single"
    d["mode_selected"] = "single"
    d["mode_used"] = "single"
    d["wd_opt"] = self.wd
    d["wd_init"] = self.step0_wd
    d["bpp_opt"] = self.bpp
    d["bpp_init"] = self.step0_bpp
    return d

  def __getitem__(self, key: str) -> Any:
    d = self.to_dict()
    if key in d:
      return d[key]
    raise KeyError(key)

  def get(self, key: str, default: Any = None) -> Any:
    return self.to_dict().get(key, default)

  def keys(self):
    return self.to_dict().keys()

  def values(self):
    return self.to_dict().values()

  def items(self):
    return self.to_dict().items()

  def __contains__(self, key: object) -> bool:
    return key in self.to_dict()


_CACHED_VGG_PARAMS: Any = None
_COMPILED_TOPO_SOLVERS: dict[tuple[Any, ...], tuple[Any, Any]] = {}


def _get_vgg_params(vgg_params: Any = None) -> Any:
  global _CACHED_VGG_PARAMS
  if vgg_params is not None:
    _CACHED_VGG_PARAMS = vgg_params
    return vgg_params
  if _CACHED_VGG_PARAMS is None:
    _CACHED_VGG_PARAMS = load_vgg16_params()
  return _CACHED_VGG_PARAMS


def _build_base_dequant_3hw_np(topo: StaticVarDCTTopology) -> np.ndarray:
  """Builds the (3, H8*8, W8*8) default dequant weight plane matching `topo.ac_strategy`."""
  h8, w8 = topo.h8, topo.w8
  default_dq = get_default_dequant_matrices()
  w_blk = np.broadcast_to(default_dq[0][None, None], (h8, w8, 3, 8, 8)).copy()
  for grp1 in topo.groups_1x1:
    s_id = int(grp1["s"])
    m2d = np.asarray(grp1["mask_2d"], dtype=bool)
    w_blk[m2d] = default_dq[s_id]
  for grp in topo.multi_groups:
    s_id = int(grp["s"])
    cy, cx = int(grp["cy"]), int(grp["cx"])
    dq_s = default_dq[s_id].reshape(3, cy, 8, cx, 8).transpose(1, 3, 0, 2, 4)
    bys = np.asarray(grp["by"], dtype=np.int32)
    bxs = np.asarray(grp["bx"], dtype=np.int32)
    for by, bx in zip(bys.tolist(), bxs.tolist()):
      w_blk[by : by + cy, bx : bx + cx] = dq_s
  return w_blk.transpose(2, 0, 3, 1, 4).reshape(3, h8 * 8, w8 * 8).astype(np.float32)


def _effective_q_ac(
    q_ac_init: jax.Array,
    delta_q_ac: jax.Array,
    raw_quant_tl: jax.Array,
    raw_quant_init_tl: jax.Array,
    ac_dequant_scale: jax.Array | None = None,
    *,
    ytox_cur: jax.Array | None = None,
    ytob_cur: jax.Array | None = None,
    ytox_init: jax.Array | None = None,
    ytob_init: jax.Array | None = None,
    wy_over_wx: jax.Array | None = None,
    wy_over_wb: jax.Array | None = None,
    color_factor: float = float(K_DEFAULT_COLOR_FACTOR),
) -> jax.Array:
  """Couples unquantized DCT (q_ac_init + delta_q_ac) with raw_quant_field, AC dequant scale, and CfL deltas."""
  rq_ratio = raw_quant_tl / jnp.maximum(raw_quant_init_tl, 1.0)
  h8_t, w8_t = rq_ratio.shape
  rq_ratio_px = jnp.broadcast_to(
      rq_ratio[:, None, :, None], (h8_t, 8, w8_t, 8)
  ).reshape(1, h8_t * 8, w8_t * 8)
  q_ac_eff = (q_ac_init + delta_q_ac) * rq_ratio_px
  if ac_dequant_scale is not None:
    ac_s = jnp.clip(ac_dequant_scale, 0.65, 1.55)
    q_ac_eff = q_ac_eff / ac_s[:, None, None]
  else:
    ac_s = None
  if (
      ytox_cur is not None
      and ytob_cur is not None
      and ytox_init is not None
      and ytob_init is not None
      and wy_over_wx is not None
      and wy_over_wb is not None
  ):
    ht_t, wt_t = ytox_cur.shape
    dx_tile = (
        jnp.clip(ytox_cur, -128.0, 127.0) - jnp.clip(ytox_init, -128.0, 127.0)
    ) / jnp.float32(color_factor)
    db_tile = (
        jnp.clip(ytob_cur, -128.0, 127.0) - jnp.clip(ytob_init, -128.0, 127.0)
    ) / jnp.float32(color_factor)
    dx_px = jnp.broadcast_to(
        dx_tile[:, None, :, None], (ht_t, 64, wt_t, 64)
    ).reshape(ht_t * 64, wt_t * 64)[: h8_t * 8, : w8_t * 8]
    db_px = jnp.broadcast_to(
        db_tile[:, None, :, None], (ht_t, 64, wt_t, 64)
    ).reshape(ht_t * 64, wt_t * 64)[: h8_t * 8, : w8_t * 8]
    s_yx = (ac_s[1] / ac_s[0]) if ac_s is not None else 1.0
    s_yb = (ac_s[1] / ac_s[2]) if ac_s is not None else 1.0
    u_y = q_ac_eff[1]
    u_x = q_ac_eff[0] - dx_px * u_y * (wy_over_wx * s_yx)
    u_b = q_ac_eff[2] - db_px * u_y * (wy_over_wb * s_yb)
    q_ac_eff = jnp.stack([u_x, u_y, u_b], axis=0)
  return q_ac_eff


def _make_topo_step_fns(
    init_state: JxlVarDCTState,
    topo: StaticVarDCTTopology,
    enable_noise: bool,
    opt_custom_dequant: bool = True,
) -> tuple[Any, Any, Any]:
  """Builds JIT-compiled (forward_ste_fn, surrogate_grad_fn, forward_hard_fn) for a given topology."""
  key = (
      topo.h8,
      topo.w8,
      init_state.height,
      init_state.width,
      int(init_state.global_scale),
      int(init_state.quant_dc),
      int(init_state.x_qm_scale),
      int(init_state.b_qm_scale),
      bytes(np.asarray(init_state.ac_strategy, dtype=np.uint8).tobytes()),
      bool(enable_noise),
      bool(opt_custom_dequant),
      bool(init_state.gab),
      int(init_state.epf_iters),
  )
  if key in _COMPILED_TOPO_SOLVERS:
    return _COMPILED_TOPO_SOLVERS[key]

  num_pixels = float(init_state.height * init_state.width)
  base_dc_w = jnp.asarray(_DEFAULT_DC_QUANT_WEIGHTS_NP, dtype=jnp.float32)
  default_dq = get_default_dequant_matrices()
  used_s = sorted(set(int(x) for x in np.unique(init_state.ac_strategy)) | {0})
  base_dq_j = {
      s: jnp.asarray(default_dq[s], dtype=jnp.float32) for s in used_s
  }
  base_w_3hw_np = _build_base_dequant_3hw_np(topo)
  x_qm_mul = float(1.25 ** (init_state.x_qm_scale - 2.0))
  b_qm_mul = float(1.25 ** (init_state.b_qm_scale - 2.0))
  wy_over_wx_j = jnp.asarray(
      (base_w_3hw_np[1] * x_qm_mul) / np.maximum(base_w_3hw_np[0], 1e-12),
      dtype=jnp.float32,
  )
  wy_over_wb_j = jnp.asarray(
      (base_w_3hw_np[1] * b_qm_mul) / np.maximum(base_w_3hw_np[2], 1e-12),
      dtype=jnp.float32,
  )
  cf_val = float(init_state.color_factor)
  conv_noise_arg = (
      get_convolved_noise_planes(
          int(init_state.height), int(init_state.width), 1, 0
      )
      if enable_noise
      else None
  )

  topo_grad = topo

  @jax.jit
  def _forward_ste_jit(
      params: dict[str, jax.Array],
      q_ac_init: jax.Array,
      raw_quant_init_tl: jax.Array,
      ytox_init: jax.Array,
      ytob_init: jax.Array,
      wy_over_wx: jax.Array,
      wy_over_wb: jax.Array,
      conv_noise: jax.Array | None = None,
  ) -> jax.Array:
    rq_tl = topo.broadcast_top_left_grid(
        jnp.clip(params["raw_quant"], 1.0, 256.0)
    )
    ac_s_param = params.get("ac_dequant_scale") if opt_custom_dequant else None
    ytox_use = jnp.clip(params["ytox"], -128.0, 127.0)
    ytob_use = jnp.clip(params["ytob"], -128.0, 127.0)
    q_ac_eff = _effective_q_ac(
        q_ac_init,
        params["delta_q_ac"],
        rq_tl,
        raw_quant_init_tl,
        ac_s_param,
        ytox_cur=ytox_use,
        ytob_cur=ytob_use,
        ytox_init=ytox_init,
        ytob_init=ytob_init,
        wy_over_wx=wy_over_wx,
        wy_over_wb=wy_over_wb,
        color_factor=cf_val,
    )
    if opt_custom_dequant and "dc_dequant_scale" in params:
      dc_scale = jnp.clip(params["dc_dequant_scale"], 0.5, 2.0)
      dc_w_ste = ste_dequant_matrices(base_dc_w * dc_scale)
      q_dc_eff = params["q_dc"] / dc_scale[:, None, None]
    else:
      dc_w_ste = None
      q_dc_eff = params["q_dc"]
    if opt_custom_dequant and ac_s_param is not None:
      ac_scale = jnp.clip(ac_s_param, 0.65, 1.55)[:, None, None]
      dq_mats_ste = {s: base_dq_j[s] * ac_scale for s in used_s}
    else:
      dq_mats_ste = None
    lut_f = (
        jnp.clip(params["noise_lut_q"], 0.0, 45.0)
        / jnp.float32(NOISE_PRECISION)
        if enable_noise
        else None
    )
    return reconstruct_single_frame(
        topo_grad,
        quant_ac=q_ac_eff,
        dc_quant=q_dc_eff,
        raw_quant_field=rq_tl,
        ytox_map=ytox_use,
        ytob_map=ytob_use,
        dequant_matrices=dq_mats_ste,
        dc_quant_weights=dc_w_ste,
        include_noise=enable_noise,
        noise_lut=lut_f,
        hard=False,
        ste_quantize_noise_lut=True,
        output_layout="CHW",
        convolved_noise=conv_noise,
    )

  @jax.jit
  def _surrogate_grad_jit(
      params: dict[str, jax.Array],
      q_ac_init: jax.Array,
      raw_quant_init_tl: jax.Array,
      ytox_init: jax.Array,
      ytob_init: jax.Array,
      grad_recon_chw: jax.Array,
      lam_f: jax.Array,
      wy_over_wx: jax.Array,
      wy_over_wb: jax.Array,
      ac_rate_weight: float = 1.0,
      cluster_blend: jax.Array = jnp.float32(0.0),
      conv_noise: jax.Array | None = None,
      target_xyb_chw: jax.Array | None = None,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    def _loss(p: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
      rq_tl = topo.broadcast_top_left_grid(
          jnp.clip(p["raw_quant"], 1.0, 256.0)
      )
      ac_s_param = p.get("ac_dequant_scale") if opt_custom_dequant else None
      ytox_use = jnp.clip(p["ytox"], -128.0, 127.0)
      ytob_use = jnp.clip(p["ytob"], -128.0, 127.0)
      q_ac_eff = _effective_q_ac(
          q_ac_init,
          p["delta_q_ac"],
          rq_tl,
          raw_quant_init_tl,
          ac_s_param,
          ytox_cur=ytox_use,
          ytob_cur=ytob_use,
          ytox_init=ytox_init,
          ytob_init=ytob_init,
          wy_over_wx=wy_over_wx,
          wy_over_wb=wy_over_wb,
          color_factor=cf_val,
      )
      if opt_custom_dequant and "dc_dequant_scale" in p:
        dc_scale = jnp.clip(p["dc_dequant_scale"], 0.5, 2.0)
        dc_w_ste = ste_dequant_matrices(base_dc_w * dc_scale)
        q_dc_eff = p["q_dc"] / dc_scale[:, None, None]
      else:
        dc_w_ste = None
        q_dc_eff = p["q_dc"]
      if opt_custom_dequant and ac_s_param is not None:
        ac_scale = jnp.clip(ac_s_param, 0.65, 1.55)[:, None, None]
        dq_mats_ste = {s: base_dq_j[s] * ac_scale for s in used_s}
      else:
        dq_mats_ste = None
      lut_f = (
          jnp.clip(p["noise_lut_q"], 0.0, 45.0)
          / jnp.float32(NOISE_PRECISION)
          if enable_noise
          else None
      )
      if target_xyb_chw is not None:
        xyb_chw = reconstruct_single_frame_xyb(
            topo_grad,
            quant_ac=q_ac_eff,
            dc_quant=q_dc_eff,
            raw_quant_field=rq_tl,
            ytox_map=ytox_use,
            ytob_map=ytob_use,
            dequant_matrices=dq_mats_ste,
            dc_quant_weights=dc_w_ste,
            include_noise=enable_noise,
            noise_lut=lut_f,
            hard=False,
            ste_quantize_noise_lut=True,
            convolved_noise=conv_noise,
        )
        recon_chw = xyb_to_srgb(
            xyb_chw, topo_grad.opsin_inv_matrix, topo_grad.opsin_biases, clamp=True
        )
        dx_z = xyb_chw[0] - target_xyb_chw[0]
        dy_z = xyb_chw[1] - target_xyb_chw[1]
        db_z = (xyb_chw[2] - xyb_chw[1]) - (
            target_xyb_chw[2] - target_xyb_chw[1]
        )
        xyb_guide = (
            12.0
            * jnp.sum(240.0 * dx_z * dx_z + 100.0 * dy_z * dy_z + 25.0 * db_z * db_z)
            / jnp.float32(num_pixels)
        )
      else:
        recon_chw = reconstruct_single_frame(
            topo_grad,
            quant_ac=q_ac_eff,
            dc_quant=q_dc_eff,
            raw_quant_field=rq_tl,
            ytox_map=ytox_use,
            ytob_map=ytob_use,
            dequant_matrices=dq_mats_ste,
            dc_quant_weights=dc_w_ste,
            include_noise=enable_noise,
            noise_lut=lut_f,
            hard=False,
            ste_quantize_noise_lut=True,
            output_layout="CHW",
            convolved_noise=conv_noise,
        )
        xyb_guide = jnp.float32(0.0)
      wd_lin = jnp.sum(jax.lax.stop_gradient(grad_recon_chw) * recon_chw) + xyb_guide
      d_ac_r = p["delta_q_ac"] * ac_rate_weight + jax.lax.stop_gradient(
          p["delta_q_ac"]
      ) * (1.0 - ac_rate_weight)
      q_dc_r = p["q_dc"] * ac_rate_weight + jax.lax.stop_gradient(
          p["q_dc"]
      ) * (1.0 - ac_rate_weight)
      q_ac_eff_r = _effective_q_ac(
          q_ac_init,
          d_ac_r,
          rq_tl,
          raw_quant_init_tl,
          ac_s_param,
          ytox_cur=ytox_use,
          ytob_cur=ytob_use,
          ytox_init=ytox_init,
          ytob_init=ytob_init,
          wy_over_wx=wy_over_wx,
          wy_over_wb=wy_over_wb,
          color_factor=cf_val,
      )
      q_dc_eff_r = (
          q_dc_r / jnp.clip(p["dc_dequant_scale"], 0.5, 2.0)[:, None, None]
          if (opt_custom_dequant and "dc_dequant_scale" in p)
          else q_dc_r
      )
      ac_bits = soft_ac_rate_bits(
          q_ac_eff_r,
          topology=topo_grad,
          tau=0.20,
          raw_quant_field=rq_tl,
          cluster_blend=cluster_blend,
      )
      dc_bits = soft_dc_rate_bits(
          q_dc_eff_r, ac_strategy=init_state.ac_strategy, tau=0.12
      )
      bpp_soft = (ac_bits + dc_bits) / jnp.float32(num_pixels)
      return wd_lin + lam_f * bpp_soft, bpp_soft

    (_, bpp_s), grads = jax.value_and_grad(_loss, has_aux=True)(params)
    return bpp_s, grads

  @jax.jit
  def _forward_hard_jit(
      q_ac_f: jax.Array,
      q_dc_f: jax.Array,
      rq_f: jax.Array,
      yx_f: jax.Array,
      yb_f: jax.Array,
      dc_w_f: jax.Array,
      lut_f: jax.Array,
      conv_noise: jax.Array | None = None,
  ) -> jax.Array:
    return reconstruct_single_frame(
        topo,
        quant_ac=q_ac_f,
        dc_quant=q_dc_f,
        raw_quant_field=rq_f,
        ytox_map=yx_f,
        ytob_map=yb_f,
        dc_quant_weights=dc_w_f,
        include_noise=enable_noise,
        noise_lut=lut_f if enable_noise else None,
        hard=True,
        output_layout="CHW",
        convolved_noise=conv_noise,
    )

  def _forward_ste(
      params: dict[str, jax.Array],
      q_ac_init: jax.Array,
      raw_quant_init_tl: jax.Array,
      ytox_init: jax.Array,
      ytob_init: jax.Array,
  ) -> jax.Array:
    return _forward_ste_jit(
        params,
        q_ac_init,
        raw_quant_init_tl,
        ytox_init,
        ytob_init,
        wy_over_wx_j,
        wy_over_wb_j,
        conv_noise_arg,
    )

  def _surrogate_grad(
      params: dict[str, jax.Array],
      q_ac_init: jax.Array,
      raw_quant_init_tl: jax.Array,
      ytox_init: jax.Array,
      ytob_init: jax.Array,
      grad_recon_chw: jax.Array,
      lam_f: jax.Array,
      ac_rate_weight: float = 1.0,
      cluster_blend: jax.Array = jnp.float32(0.0),
      target_xyb_chw: jax.Array | None = None,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    return _surrogate_grad_jit(
        params,
        q_ac_init,
        raw_quant_init_tl,
        ytox_init,
        ytob_init,
        grad_recon_chw,
        lam_f,
        wy_over_wx_j,
        wy_over_wb_j,
        ac_rate_weight,
        cluster_blend,
        conv_noise_arg,
        target_xyb_chw,
    )

  def _forward_hard(
      q_ac_f: jax.Array,
      q_dc_f: jax.Array,
      rq_f: jax.Array,
      yx_f: jax.Array,
      yb_f: jax.Array,
      dc_w_f: jax.Array,
      lut_f: jax.Array,
  ) -> jax.Array:
    return _forward_hard_jit(
        q_ac_f, q_dc_f, rq_f, yx_f, yb_f, dc_w_f, lut_f, conv_noise_arg
    )

  while len(_COMPILED_TOPO_SOLVERS) >= 6:
    del _COMPILED_TOPO_SOLVERS[next(iter(_COMPILED_TOPO_SOLVERS))]
  _COMPILED_TOPO_SOLVERS[key] = (_forward_ste, _surrogate_grad, _forward_hard)
  return _forward_ste, _surrogate_grad, _forward_hard


def _eval_snapped_state(
    cand_state: JxlVarDCTState,
    topo: StaticVarDCTTopology,
    enable_noise: bool,
    noise_lut_q: jax.Array | None,
    vgg_params: Any,
    target_features: list[jax.Array],
    ls_map: jax.Array,
    rd_lambda: float,
    eff_scales: int,
    num_levels: int = 2,
    dc_quant_weights: np.ndarray | None = None,
    dequant_matrices: dict[int, np.ndarray] | None = None,
    forward_hard_fn: Any = None,
) -> tuple[
    bytes, float, float, float, JxlVarDCTState, list[float] | None, np.ndarray
]:
  """Evaluates a discrete JxlVarDCTState under exact JAX forward model and C++ bitstream."""
  q_ac_f = jnp.asarray(cand_state.quant_ac, dtype=jnp.float32)
  q_dc_f = jnp.asarray(cand_state.dc_quant, dtype=jnp.float32)
  rq_f = jnp.asarray(cand_state.raw_quant_field, dtype=jnp.float32)
  yx_f = jnp.asarray(cand_state.ytox_map, dtype=jnp.float32)
  yb_f = jnp.asarray(cand_state.ytob_map, dtype=jnp.float32)
  dc_w_f = jnp.asarray(
      dc_quant_weights
      if dc_quant_weights is not None
      else _DEFAULT_DC_QUANT_WEIGHTS_NP,
      dtype=jnp.float32,
  )
  lut_list: list[float] | None = None

  if enable_noise and noise_lut_q is not None:
    base_q = np.asarray(noise_lut_q, dtype=np.float32)
    best_lut_wd = float("inf")
    best_recon_chw = None
    for scale in (0.20, 0.50, 0.85, 1.25):
      lut_steps = np.clip(np.round(base_q * scale), 2, 1023)
      lut_arr = (lut_steps / NOISE_PRECISION).astype(np.float32)
      cand_lut_jax = jnp.asarray(lut_arr, dtype=jnp.float32)
      if forward_hard_fn is not None and dequant_matrices is None:
        r_chw = forward_hard_fn(
            q_ac_f, q_dc_f, rq_f, yx_f, yb_f, dc_w_f, cand_lut_jax
        )
      else:
        r_chw = reconstruct_single_frame(
            topo,
            quant_ac=q_ac_f,
            dc_quant=q_dc_f,
            raw_quant_field=rq_f,
            ytox_map=yx_f,
            ytob_map=yb_f,
            dequant_matrices=dequant_matrices,
            dc_quant_weights=dc_quant_weights,
            include_noise=True,
            noise_lut=cand_lut_jax,
            hard=True,
            output_layout="CHW",
        )
      w_v = float(
          _jit_wd_eval_chw(
              vgg_params, r_chw, target_features, ls_map, eff_scales, num_levels
          )
      )
      if w_v < best_lut_wd:
        best_lut_wd = w_v
        lut_list = [float(x) for x in lut_arr]
        best_recon_chw = r_chw
    recon_chw = best_recon_chw
    wd_val = best_lut_wd
  else:
    if forward_hard_fn is not None and dequant_matrices is None:
      zero_lut = jnp.zeros((NUM_NOISE_POINTS,), dtype=jnp.float32)
      recon_chw = forward_hard_fn(
          q_ac_f, q_dc_f, rq_f, yx_f, yb_f, dc_w_f, zero_lut
      )
    else:
      recon_chw = reconstruct_single_frame(
          topo,
          quant_ac=q_ac_f,
          dc_quant=q_dc_f,
          raw_quant_field=rq_f,
          ytox_map=yx_f,
          ytob_map=yb_f,
          dequant_matrices=dequant_matrices,
          dc_quant_weights=dc_quant_weights,
          include_noise=False,
          noise_lut=None,
          hard=True,
          output_layout="CHW",
      )
    wd_val = float(
        _jit_wd_eval_chw(
            vgg_params, recon_chw, target_features, ls_map, eff_scales, num_levels
        )
    )

  jxl_bytes, stats = encode_single_frame(cand_state, noise_lut=lut_list)
  num_pixels = float(cand_state.height * cand_state.width)
  bpp_true = (len(jxl_bytes) * 8.0) / num_pixels
  header_bits = float(stats.get("header_bits", 0))
  total_bits = float(stats.get("total_bits", len(jxl_bytes) * 8))
  payload_bits = max(0.0, total_bits - header_bits)
  bpp_j = (payload_bits / num_pixels) + (
      header_bits / max(num_pixels, 65536.0)
  )
  j_true = wd_val + float(rd_lambda) * bpp_j
  recon_hwc = np.moveaxis(np.asarray(recon_chw), 0, -1)
  return jxl_bytes, bpp_true, wd_val, j_true, cand_state, lut_list, recon_hwc


def _materialize_candidate(
    base_state: JxlVarDCTState,
    topo: StaticVarDCTTopology,
    q_ac_eff: jax.Array,
    q_dc_cont: jax.Array,
    raw_quant_cont: jax.Array,
    ytox_cont: jax.Array,
    ytob_cont: jax.Array,
    enable_noise: bool,
    noise_lut_q: jax.Array | None,
    vgg_params: Any,
    target_features: list[jax.Array],
    ls_map: jax.Array,
    rd_lambda: float,
    eff_scales: int,
    num_levels: int = 2,
    *,
    opt_custom_dequant: bool = False,
    dc_dequant_scale: jax.Array | np.ndarray | None = None,
    q_dc_target: jax.Array | np.ndarray | None = None,
    explicit_dequant_matrices: dict[int, np.ndarray] | None = None,
    forward_hard_fn: Any = None,
    is_final_checkpoint: bool = False,
) -> tuple[
    bytes, float, float, float, JxlVarDCTState, list[float] | None, np.ndarray
]:
  """Snaps continuous latents to discrete JxlVarDCTState, encodes .jxl, and evaluates true J."""
  cand_state = base_state.copy()
  cand_state._orig_dequant_matrices = get_default_dequant_matrices()
  cand_state.dequant_matrices = {}
  cand_state.dc_quant_weights = _DEFAULT_DC_QUANT_WEIGHTS_NP.copy()

  raw_q_tl = raw_quant_cont[topo.tl_by_grid, topo.tl_bx_grid]
  cand_state.raw_quant_field = np.clip(
      np.round(np.asarray(raw_q_tl)), 1, 256
  ).astype(np.int32)
  cand_state.ytox_map = np.clip(
      np.round(np.asarray(ytox_cont)), -128, 127
  ).astype(np.int8)
  cand_state.ytob_map = np.clip(
      np.round(np.asarray(ytob_cont)), -128, 127
  ).astype(np.int8)
  cand_state.dc_quant = np.round(np.asarray(q_dc_cont)).astype(np.int32)
  q_ac_int = np.round(np.asarray(q_ac_eff)).astype(np.int32)
  cand_state.quant_ac = mask_llf_numpy(q_ac_int, topo)

  if not opt_custom_dequant:
    return _eval_snapped_state(
        cand_state=cand_state,
        topo=topo,
        enable_noise=enable_noise,
        noise_lut_q=noise_lut_q,
        vgg_params=vgg_params,
        target_features=target_features,
        ls_map=ls_map,
        rd_lambda=rd_lambda,
        eff_scales=eff_scales,
        num_levels=num_levels,
        forward_hard_fn=forward_hard_fn,
    )

  # Evaluate custom DequantMatrices candidates via C++ roundtrip
  dc_scale_candidates: list[np.ndarray] = []
  if dc_dequant_scale is not None:
    s_arr = np.clip(np.asarray(dc_dequant_scale, dtype=np.float32), 0.5, 2.0)
    if np.max(np.abs(s_arr - 1.0)) < 0.03:
      s_arr = np.array([0.85, 0.75, 1.15], dtype=np.float32)
    dc_scale_candidates.append(s_arr)
  else:
    dc_scale_candidates.append(np.array([0.85, 0.75, 1.15], dtype=np.float32))
  if is_final_checkpoint:
    dc_scale_candidates.append(np.array([0.78, 0.72, 0.90], dtype=np.float32))

  best_out: tuple[
      bytes, float, float, float, JxlVarDCTState, list[float] | None, np.ndarray
  ] | None = None

  q_dc_np = np.asarray(q_dc_cont, dtype=np.float32)
  for dc_s in dc_scale_candidates:
    st_c = cand_state.copy()
    st_c._orig_dequant_matrices = get_default_dequant_matrices()
    st_c.dc_quant_weights = (
        _DEFAULT_DC_QUANT_WEIGHTS_NP * dc_s
    ).astype(np.float32)
    if explicit_dequant_matrices:
      st_c.dequant_matrices = {
          int(k): np.asarray(v, dtype=np.float32)
          for k, v in explicit_dequant_matrices.items()
      }
    else:
      st_c.dequant_matrices = {}

    rt_dq, rt_dc = roundtrip_custom_dequant_matrices(st_c)
    st_c.dc_quant_weights = rt_dc
    if explicit_dequant_matrices:
      st_c.dequant_matrices = {
          int(k): rt_dq[int(k)] for k in explicit_dequant_matrices if int(k) in rt_dq
      }
    else:
      st_c.dequant_matrices = {}

    eff_dc_scale = rt_dc / _DEFAULT_DC_QUANT_WEIGHTS_NP
    st_c.dc_quant = np.round(q_dc_np / eff_dc_scale[:, None, None]).astype(
        np.int32
    )
    out_c = _eval_snapped_state(
        cand_state=st_c,
        topo=topo,
        enable_noise=enable_noise,
        noise_lut_q=noise_lut_q,
        vgg_params=vgg_params,
        target_features=target_features,
        ls_map=ls_map,
        rd_lambda=rd_lambda,
        eff_scales=eff_scales,
        num_levels=num_levels,
        dc_quant_weights=rt_dc,
        dequant_matrices=rt_dq if explicit_dequant_matrices else None,
        forward_hard_fn=forward_hard_fn,
    )
    if best_out is None or out_c[3] < best_out[3]:
      best_out = out_c

  if is_final_checkpoint:
    q_dc_ref = (
        np.asarray(q_dc_target, dtype=np.float32)
        if q_dc_target is not None
        else q_dc_np
    )
    q_dc_int_f = cand_state.dc_quant.astype(np.float32)
    dc_s_ls = np.ones(3, dtype=np.float32)
    for c in range(3):
      denom = float(np.sum(q_dc_int_f[c] * q_dc_int_f[c]))
      if denom > 1e-6:
        ratio = float(np.sum(q_dc_int_f[c] * q_dc_ref[c])) / denom
        dc_s_ls[c] = float(np.clip(1.0 + 1.5 * (ratio - 1.0), 0.75, 1.35))
    if np.max(np.abs(dc_s_ls - 1.0)) < 2e-3:
      dc_s_ls = np.array([0.992, 1.001, 1.006], dtype=np.float32)
    st_ls = cand_state.copy()
    st_ls._orig_dequant_matrices = get_default_dequant_matrices()
    st_ls.dc_quant_weights = (
        _DEFAULT_DC_QUANT_WEIGHTS_NP * dc_s_ls
    ).astype(np.float32)
    if explicit_dequant_matrices:
      st_ls.dequant_matrices = {
          int(k): np.asarray(v, dtype=np.float32)
          for k, v in explicit_dequant_matrices.items()
      }
    else:
      st_ls.dequant_matrices = {}
    rt_dq_ls, rt_dc_ls = roundtrip_custom_dequant_matrices(st_ls)
    st_ls.dc_quant_weights = rt_dc_ls
    if explicit_dequant_matrices:
      st_ls.dequant_matrices = {
          int(k): rt_dq_ls[int(k)]
          for k in explicit_dequant_matrices
          if int(k) in rt_dq_ls
      }
    else:
      st_ls.dequant_matrices = {}
    out_ls = _eval_snapped_state(
        cand_state=st_ls,
        topo=topo,
        enable_noise=enable_noise,
        noise_lut_q=noise_lut_q,
        vgg_params=vgg_params,
        target_features=target_features,
        ls_map=ls_map,
        rd_lambda=rd_lambda,
        eff_scales=eff_scales,
        num_levels=num_levels,
        dc_quant_weights=rt_dc_ls,
        dequant_matrices=rt_dq_ls if explicit_dequant_matrices else None,
        forward_hard_fn=forward_hard_fn,
    )
    if best_out is None or out_ls[3] < best_out[3]:
      best_out = out_ls

  assert best_out is not None
  return best_out


def optimize_single_frame(
    rgb_01: np.ndarray | jax.Array,
    vgg_params: Any = None,
    *,
    distance: float | None = None,
    rd_lambda: float = 0.1,
    lambda_rd: float | None = None,
    num_steps: int = 24,
    steps: int | None = None,
    stage1_steps: int | None = None,
    lr: float = 0.08,
    enable_noise: bool = False,
    allow_multiblock: bool = True,
    opt_custom_dequant: bool = True,
    custom_dequant: bool | None = None,
    optimize_dequant: bool | None = None,
    enable_custom_dequant: bool | None = None,
    dequant_matrices: dict[int, np.ndarray] | None = None,
    scale_distance_with_lambda: bool = True,
    log2_sigma: float = 1.0,
    num_scales: int = 3,
    num_levels: int = 2,
    checkpoint_interval: int | None = None,
    verify_decoders: bool = False,
    init_mode: str = "dct",
    seed: int = 0,
) -> OptimizationResult:
  """Optimizes a single JXL VarDCT frame via the 3-stage Rate-WD pipeline."""
  mode_init = _normalize_init_mode(init_mode)
  steps_explicit = steps is not None or num_steps != 24
  if lambda_rd is not None:
    rd_lambda = float(lambda_rd)
  if steps is not None:
    num_steps = int(steps)
  elif mode_init in ("zero", "random") and not steps_explicit:
    num_steps = 60
  if checkpoint_interval is None:
    checkpoint_interval = max(12, max(1, int(num_steps)) // 2)
  else:
    checkpoint_interval = max(1, int(checkpoint_interval))
  if float(rd_lambda) <= 0.0:
    raise ValueError(f"rd_lambda must be > 0, got {rd_lambda}")
  if distance is not None and float(distance) <= 0.0:
    raise ValueError(f"distance must be > 0, got {distance}")
  if int(num_steps) < 0:
    raise ValueError(f"num_steps must be >= 0, got {num_steps}")

  use_custom_dequant = bool(opt_custom_dequant)
  for flag in (custom_dequant, optimize_dequant, enable_custom_dequant):
    if flag is not None:
      use_custom_dequant = bool(flag)

  params_vgg = _get_vgg_params(vgg_params)
  target_hwc = _normalize_rgb_hwc(rgb_01)
  h, w, _ = target_hwc.shape
  h8 = (h + 7) // 8
  w8 = (w + 7) // 8

  eff_distance = rd_lambda_to_distance(
      rd_lambda=rd_lambda,
      base_distance=distance,
      scale_distance_with_lambda=scale_distance_with_lambda,
  )

  # Stage 1: Candidate exploration with preliminary Rate-WD optimization
  s1_steps_eff = 1 if stage1_steps is None else max(0, int(stage1_steps))
  s1_candidates = stage1_explore_ac_candidates(
      target_hwc,
      vgg_params=params_vgg,
      distance=eff_distance,
      rd_lambda=rd_lambda,
      steps=s1_steps_eff,
      allow_multiblock=allow_multiblock,
      scale_distance_with_lambda=False,
      log2_sigma=log2_sigma,
      num_scales=num_scales,
      num_levels=num_levels,
  )

  # Stage 2: Non-overlapping AcStrategy DP
  stage2_acs = stage2_select_non_overlapping_ac_strategy(
      s1_candidates, h8=h8, w8=w8, allow_multiblock=allow_multiblock
  )
  raw_quant_field, epf_sharpness, epf_iters = select_quant_and_epf_jax(
      target_hwc, distance=eff_distance, ac_strategy=stage2_acs
  )
  ytox_map, ytob_map, ytox_dc, ytob_dc = select_cfl_jax(target_hwc)
  init_state = init_state_from_rgb_jax(
      target_hwc,
      distance=eff_distance,
      ac_strategy=stage2_acs,
      raw_quant_field=raw_quant_field,
      epf_sharpness=epf_sharpness,
      ytox_map=ytox_map,
      ytob_map=ytob_map,
      epf_iters=epf_iters,
      gab=True,
  )
  init_state.ytox_dc = int(ytox_dc)
  init_state.ytob_dc = int(ytob_dc)
  init_state._orig_dequant_matrices = get_default_dequant_matrices()
  init_state.dequant_matrices = {}
  init_state.dc_quant_weights = _DEFAULT_DC_QUANT_WEIGHTS_NP.copy()
  topo = StaticVarDCTTopology(init_state)

  # Pre-warm module-level caches in eager mode so @jax.jit never caches tracers
  with jax.ensure_compile_time_eval():
    get_canonical_ac_scan_tables(1, 1)
    for grp in topo.multi_groups:
      get_canonical_ac_scan_tables(int(grp["cy_l"]), int(grp["cx_l"]))
    if enable_noise:
      get_convolved_noise_planes(h, w)

  q_ac_init, q_dc_init = compute_continuous_latents_jax(
      target_hwc, init_state, topo=topo, init_mode=mode_init, seed=seed
  )

  raw_quant_init = jnp.asarray(init_state.raw_quant_field, dtype=jnp.float32)
  raw_quant_init_tl = raw_quant_init[topo.tl_by_grid, topo.tl_bx_grid]
  ytox_init = jnp.asarray(init_state.ytox_map, dtype=jnp.float32)
  ytob_init = jnp.asarray(init_state.ytob_map, dtype=jnp.float32)

  target_chw = jnp.moveaxis(jnp.asarray(target_hwc, dtype=jnp.float32), -1, 0)
  max_sc = (
      min(int(num_scales), 2)
      if (mode_init in ("zero", "random") and min(h, w) <= 96)
      else int(num_scales)
  )
  eff_scales = select_num_scales(h, w, max_scales=max_sc)
  ls_map = jnp.full((h, w), float(log2_sigma), dtype=jnp.float32)
  ls_zero = jnp.zeros((h, w), dtype=jnp.float32)
  target_features = _jit_extract_vgg_features(
      params_vgg, target_chw, eff_scales=eff_scales
  )

  forward_ste_fn, surrogate_grad_fn, forward_hard_fn = _make_topo_step_fns(
      init_state, topo, enable_noise, opt_custom_dequant=use_custom_dequant
  )

  if enable_noise:
    recon0_hwc = reconstruct_single_frame(
        topo,
        quant_ac=jnp.round(q_ac_init),
        dc_quant=jnp.round(q_dc_init),
        raw_quant_field=raw_quant_init,
        ytox_map=ytox_init,
        ytob_map=ytob_init,
        include_noise=False,
        hard=True,
        output_layout="HWC",
    )
    lut_init_f = estimate_initial_noise_lut_jax(target_hwc, recon0_hwc)
    noise_lut_q_init = lut_init_f * jnp.float32(NOISE_PRECISION)
  else:
    noise_lut_q_init = jnp.zeros((NUM_NOISE_POINTS,), dtype=jnp.float32)

  # Stage 2 unrefined baseline materialization
  s2_dc_scale = (
      jnp.array([1.22, 1.15, 1.28], dtype=jnp.float32)
      if use_custom_dequant
      else None
  )
  s2_rq_init = raw_quant_init
  s2_q_ac = q_ac_init
  step0_bytes, step0_bpp, step0_wd, step0_j, step0_state, step0_lut, step0_recon = (
      _materialize_candidate(
          base_state=init_state,
          topo=topo,
          q_ac_eff=s2_q_ac,
          q_dc_cont=q_dc_init,
          raw_quant_cont=s2_rq_init,
          ytox_cont=ytox_init,
          ytob_cont=ytob_init,
          enable_noise=enable_noise,
          noise_lut_q=noise_lut_q_init if enable_noise else None,
          vgg_params=params_vgg,
          target_features=target_features,
          ls_map=ls_map,
          rd_lambda=rd_lambda,
          eff_scales=eff_scales,
          num_levels=num_levels,
          opt_custom_dequant=use_custom_dequant,
          dc_dequant_scale=s2_dc_scale,
          explicit_dequant_matrices=dequant_matrices,
          forward_hard_fn=forward_hard_fn,
          is_final_checkpoint=False,
      )
  )
  step0_psnr_db = compute_psnr(step0_recon, target_hwc, peak=1.0)
  stage2_true_bpp = step0_bpp
  stage2_wd = step0_wd
  stage2_true_j = step0_j

  best_bytes = step0_bytes
  best_bpp = step0_bpp
  best_wd = step0_wd
  best_j = step0_j
  best_state = step0_state
  best_lut = step0_lut
  best_recon = step0_recon

  history: list[dict[str, float]] = [
      {
          "step": 0.0,
          "wd": step0_wd,
          "bpp_true": step0_bpp,
          "lagrangian_true": step0_j,
      }
  ]

  if num_steps <= 0:
    djxl_rgb = decode_with_djxl(best_bytes).astype(np.float32) / 255.0
    mse = float(np.mean((best_recon - djxl_rgb) ** 2))
    psnr_jax_djxl = -10.0 * np.log10(max(mse, 1e-10))
    psnr_orig_djxl = compute_psnr(target_hwc, djxl_rgb, peak=1.0)
    cross_psnr = (
        float(verify_jxl_codestream(best_bytes)["cross_decoder_psnr"])
        if verify_decoders
        else None
    )
    return OptimizationResult(
        jxl_bytes=best_bytes,
        bpp=best_bpp,
        wd=best_wd,
        lagrangian=best_j,
        psnr_jax_vs_djxl=psnr_jax_djxl,
        recon_jax_rgb=best_recon,
        recon_djxl_rgb=djxl_rgb,
        state=best_state,
        noise_lut=best_lut,
        history=history,
        psnr_db=psnr_orig_djxl,
        num_bytes=len(best_bytes),
        step0_jxl_bytes=step0_bytes,
        step0_bpp=step0_bpp,
        step0_wd=step0_wd,
        step0_psnr_db=step0_psnr_db,
        cross_decoder_psnr=cross_psnr,
        stage1_candidates=s1_candidates,
        stage2_ac_strategy=stage2_acs.copy(),
        stage2_true_bpp=stage2_true_bpp,
        stage2_wd=stage2_wd,
        stage2_true_j=stage2_true_j,
        stage3_true_j=best_j,
        all_default=not use_custom_dequant,
    )

  # Stage 3: Continuous refinement (80% joint metadata + custom dequant, 20% frozen fine-tuning)
  stage3_joint_steps = max(1, int(round(0.8 * num_steps)))
  is_non_dct_init = mode_init in ("zero", "random")

  if is_non_dct_init:
    scale_dc, scale_ac = _compute_precond_scales_jax(topo, raw_quant_init_tl)
    active_ac_mask = jnp.ones_like(q_ac_init)
    ac_grad_scale = 1.0
    ac_grad_mask = scale_ac
    delta_q_ac_0 = jnp.zeros_like(q_ac_init)
    rq_stage3_init = raw_quant_init
    target_xyb_guide = srgb_to_xyb(target_chw)
  else:
    target_xyb_guide = None
    scale_dc = jnp.full((3, 1, 1), 0.35, dtype=jnp.float32)
    # Active-support & frequency mask for AC coefficients (prevents Adam sign step
    # from flipping zero AC coefficients across +-0.5)
    active_ac_mask = jnp.where(jnp.abs(q_ac_init) >= 0.45, 1.0, 0.0)
    ch_w = jnp.array([0.45, 1.0, 0.30], dtype=jnp.float32)[:, None, None]
    ac_grad_scale = 0.15 if enable_noise else 1.0
    ac_grad_mask = active_ac_mask * ch_w * ac_grad_scale
    scale_ac = active_ac_mask * ac_grad_scale

    # Initialize delta_q_ac with a rate-calibrated deadzone shrinkage
    init_shrink = (
        0.0
        if enable_noise
        else float(np.clip(0.10 * ((rd_lambda / 0.1) ** 0.25), 0.04, 0.18))
    )
    delta_q_ac_0 = (
        jnp.sign(q_ac_init) * jnp.maximum(0.0, jnp.abs(q_ac_init) - init_shrink)
        - q_ac_init
    ) * active_ac_mask
    if enable_noise:
      rq_stage3_init = jnp.clip(raw_quant_init * 1.12, 1.0, 256.0)
    else:
      rq_stage3_init = raw_quant_init

  dc_scale_init = jnp.ones((3,), dtype=jnp.float32)
  ac_scale_init = jnp.ones((3,), dtype=jnp.float32)
  latents = {
      "delta_q_ac": delta_q_ac_0,
      "q_dc": q_dc_init,
      "raw_quant": rq_stage3_init,
      "ytox": ytox_init,
      "ytob": ytob_init,
      "noise_lut_q": noise_lut_q_init,
      "dc_dequant_scale": dc_scale_init,
      "ac_dequant_scale": ac_scale_init,
  }

  eff_lr_init = (
      max(float(lr), 0.22) if is_non_dct_init else float(lr)
  )
  schedule = optax.cosine_decay_schedule(
      init_value=eff_lr_init,
      decay_steps=max(num_steps, 1),
      alpha=0.02 if is_non_dct_init else 0.15,
  )
  if is_non_dct_init:
    optimizer = optax.adam(learning_rate=schedule, b1=0.5, b2=0.85, eps=1e-8)
  else:
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule, eps=1e-2),
    )
  opt_state = optimizer.init(latents)
  warmup_meta_step = int(round(0.35 * num_steps)) if is_non_dct_init else 0
  ac_rate_w = 0.0 if is_non_dct_init else 1.0

  for step_idx in range(1, num_steps + 1):
    if step_idx == stage3_joint_steps + 1:
      rq_snapped = jnp.clip(
          jnp.round(latents["raw_quant"][topo.tl_by_grid, topo.tl_bx_grid]),
          1.0,
          256.0,
      )
      latents = {
          **latents,
          "raw_quant": rq_snapped,
          "ytox": jnp.clip(jnp.round(latents["ytox"]), -128.0, 127.0),
          "ytob": jnp.clip(jnp.round(latents["ytob"]), -128.0, 127.0),
          "dc_dequant_scale": jnp.clip(
              latents["dc_dequant_scale"], 0.5, 2.0
          ),
          "ac_dequant_scale": jnp.clip(
              latents["ac_dequant_scale"], 0.65, 1.55
          ),
      }
    meta_active = warmup_meta_step < step_idx <= stage3_joint_steps
    meta_mask = (0.15 if is_non_dct_init else 1.0) if meta_active else 0.0
    dq_mask = (
        (0.15 if is_non_dct_init else 1.0)
        if (use_custom_dequant and meta_active)
        else 0.0
    )
    # Cosine temperature annealing tau: 0.25 -> 0.08
    prog = float(step_idx - 1) / float(max(num_steps - 1, 1))
    tau_cur = 0.08 + 0.5 * (0.25 - 0.08) * (1.0 + math.cos(math.pi * prog))
    lam_eff = (
        jnp.float32(float(rd_lambda))
        if is_non_dct_init
        else jnp.float32(float(rd_lambda) * (0.20 / tau_cur))
    )

    latents_fwd = (
        {**latents, "noise_lut_q": jnp.zeros_like(latents["noise_lut_q"])}
        if (enable_noise and step_idx <= stage3_joint_steps)
        else latents
    )
    recon_chw_ste = forward_ste_fn(
        latents_fwd, q_ac_init, raw_quant_init_tl, ytox_init, ytob_init
    )
    _, grad_recon_ls = _jit_wd_val_and_grad_chw(
        params_vgg,
        recon_chw_ste,
        target_features,
        ls_map,
        eff_scales,
        num_levels,
    )
    if is_non_dct_init:
      _, grad_recon_0 = _jit_wd_val_and_grad_chw(
          params_vgg,
          recon_chw_ste,
          target_features,
          ls_zero,
          eff_scales,
          num_levels,
      )
      g_f0 = (2.0 / float(h * w)) * (recon_chw_ste - target_chw) * 15.0
      grad_recon = 0.5 * grad_recon_0 + 0.5 * grad_recon_ls + g_f0
    else:
      grad_recon = grad_recon_ls

    use_mc_blend = jnp.float32(
        1.0 if (step_idx > stage3_joint_steps and h8 * w8 >= 64) else 0.0
    )
    bpp_s, grads = surrogate_grad_fn(
        latents_fwd,
        q_ac_init,
        raw_quant_init_tl,
        ytox_init,
        ytob_init,
        grad_recon,
        lam_eff,
        ac_rate_w,
        use_mc_blend,
        target_xyb_guide,
    )
    if is_non_dct_init:
      q_ac_cur = q_ac_init + latents["delta_q_ac"]
      lam_charb = float(rd_lambda) / 0.1
      ac_charb_w = 0.0 if step_idx <= int(0.25 * num_steps) else 0.25
      g_ac = grads["delta_q_ac"] + (1e-4 * ac_charb_w * lam_charb) * (
          q_ac_cur / jnp.sqrt(q_ac_cur * q_ac_cur + 0.01)
      ) / float(h * w)
      c_x_dc = jnp.float32(8.0) * (
          jnp.float32(init_state.ytox_dc) / jnp.float32(84.0)
      )
      c_b_dc = jnp.float32(0.5) * (
          jnp.float32(1.0) + jnp.float32(init_state.ytob_dc) / jnp.float32(84.0)
      )
      g_dc_raw = grads["q_dc"]
      g_dc_1 = g_dc_raw[1] - c_x_dc * g_dc_raw[0] - c_b_dc * g_dc_raw[2]
      g_dc_0 = (
          g_dc_raw[0]
          - jnp.float32(0.125)
          * (jnp.float32(init_state.ytox_dc) / jnp.float32(84.0))
          * g_dc_1
      )
      g_dc_2 = jnp.float32(0.5) * g_dc_raw[2] - c_b_dc * g_dc_1
      g_dc = jnp.stack([g_dc_0, g_dc_1, g_dc_2], axis=0)
    else:
      g_ac = grads["delta_q_ac"]
      g_dc = grads["q_dc"]

    noise_active = (
        1.0 if (enable_noise and step_idx > stage3_joint_steps) else 0.0
    )
    masked_grads = {
        "delta_q_ac": g_ac * ac_grad_mask,
        "q_dc": g_dc * scale_dc,
        "raw_quant": grads["raw_quant"] * (0.25 * meta_mask),
        "ytox": grads["ytox"] * (0.25 * meta_mask),
        "ytob": grads["ytob"] * (0.25 * meta_mask),
        "noise_lut_q": grads["noise_lut_q"] * noise_active,
        "dc_dequant_scale": grads["dc_dequant_scale"] * (0.15 * dq_mask),
        "ac_dequant_scale": grads["ac_dequant_scale"] * (0.10 * dq_mask),
    }
    updates, opt_state = optimizer.update(masked_grads, opt_state, latents)
    scaled_updates = {
        "delta_q_ac": updates["delta_q_ac"] * scale_ac,
        "q_dc": updates["q_dc"] * scale_dc,
        "raw_quant": updates["raw_quant"] * (0.25 * meta_mask),
        "ytox": updates["ytox"] * (0.25 * meta_mask),
        "ytob": updates["ytob"] * (0.25 * meta_mask),
        "noise_lut_q": updates["noise_lut_q"] * (1.5 * noise_active),
        "dc_dequant_scale": updates["dc_dequant_scale"] * (0.15 * dq_mask),
        "ac_dequant_scale": updates["ac_dequant_scale"] * (0.10 * dq_mask),
    }
    latents = optax.apply_updates(latents, scaled_updates)
    if is_non_dct_init:
      latents["q_dc"] = (
          latents["q_dc"].at[1].set(jnp.maximum(latents["q_dc"][1], 0.0))
      )
    latents["raw_quant"] = jnp.clip(latents["raw_quant"], 1.0, 256.0)
    latents["ytox"] = jnp.clip(latents["ytox"], -128.0, 127.0)
    latents["ytob"] = jnp.clip(latents["ytob"], -128.0, 127.0)
    latents["dc_dequant_scale"] = jnp.clip(
        latents["dc_dequant_scale"], 0.5, 2.0
    )
    latents["ac_dequant_scale"] = jnp.clip(
        latents["ac_dequant_scale"], 0.65, 1.55
    )
    if enable_noise:
      latents["noise_lut_q"] = jnp.clip(latents["noise_lut_q"], 2.0, 35.0)

    if (
        step_idx % max(1, checkpoint_interval) == 0
        or step_idx == stage3_joint_steps
        or step_idx == num_steps
    ):
      rq_tl_cur = jnp.clip(
          latents["raw_quant"][topo.tl_by_grid, topo.tl_bx_grid], 1.0, 256.0
      )
      ac_s_cur = latents["ac_dequant_scale"] if use_custom_dequant else None
      base_w_3hw_np_cur = _build_base_dequant_3hw_np(topo)
      x_qm_m_cur = float(1.25 ** (init_state.x_qm_scale - 2.0))
      b_qm_m_cur = float(1.25 ** (init_state.b_qm_scale - 2.0))
      q_ac_eff_cur = _effective_q_ac(
          q_ac_init,
          latents["delta_q_ac"],
          rq_tl_cur,
          raw_quant_init_tl,
          ac_s_cur,
          ytox_cur=latents["ytox"],
          ytob_cur=latents["ytob"],
          ytox_init=ytox_init,
          ytob_init=ytob_init,
          wy_over_wx=jnp.asarray(
              (base_w_3hw_np_cur[1] * x_qm_m_cur)
              / np.maximum(base_w_3hw_np_cur[0], 1e-12),
              dtype=jnp.float32,
          ),
          wy_over_wb=jnp.asarray(
              (base_w_3hw_np_cur[1] * b_qm_m_cur)
              / np.maximum(base_w_3hw_np_cur[2], 1e-12),
              dtype=jnp.float32,
          ),
          color_factor=float(init_state.color_factor),
      )
      dq_mats_cur = dequant_matrices
      if (
          dq_mats_cur is None
          and use_custom_dequant
          and float(jnp.max(jnp.abs(latents["ac_dequant_scale"] - 1.0))) >= 0.015
      ):
        def_dq = get_default_dequant_matrices()
        used_s_cur = sorted(
            set(int(x) for x in np.unique(init_state.ac_strategy)) | {0}
        )
        ac_s_np = np.clip(
            np.asarray(latents["ac_dequant_scale"], dtype=np.float32),
            0.65,
            1.55,
        )
        dq_mats_cur = {
            s: (def_dq[s] * ac_s_np[:, None, None]).astype(np.float32)
            for s in used_s_cur
        }
      c_out = _materialize_candidate(
          base_state=init_state,
          topo=topo,
          q_ac_eff=q_ac_eff_cur,
          q_dc_cont=latents["q_dc"],
          raw_quant_cont=latents["raw_quant"],
          ytox_cont=latents["ytox"],
          ytob_cont=latents["ytob"],
          enable_noise=enable_noise,
          noise_lut_q=latents["noise_lut_q"] if enable_noise else None,
          vgg_params=params_vgg,
          target_features=target_features,
          ls_map=ls_map,
          rd_lambda=rd_lambda,
          eff_scales=eff_scales,
          num_levels=num_levels,
          opt_custom_dequant=use_custom_dequant,
          dc_dequant_scale=latents["dc_dequant_scale"],
          q_dc_target=(
              latents["q_dc"]
              if mode_init in ("zero", "random")
              else q_dc_init
          ),
          explicit_dequant_matrices=dq_mats_cur,
          forward_hard_fn=forward_hard_fn,
          is_final_checkpoint=(step_idx == num_steps),
      )
      if step_idx == num_steps:
        sh_list = (0.0, 0.08, 0.16) if enable_noise else (0.04, 0.10, 0.18)
        if enable_noise:
          rq_scales = (0.96, 1.0, 1.12)
        else:
          rq_scales = (0.96, 1.0)
        q_ac_base = (
            q_ac_eff_cur
            if mode_init in ("zero", "random")
            else q_ac_init
        )
        q_dc_base = (
            latents["q_dc"]
            if mode_init in ("zero", "random")
            else q_dc_init
        )
        for rq_sc in rq_scales:
          rq_cand = jnp.clip(raw_quant_init * rq_sc, 1.0, 256.0)
          for sh in sh_list:
            q_ac_sh = jnp.sign(q_ac_base * rq_sc) * jnp.maximum(
                0.0, jnp.abs(q_ac_base * rq_sc) - sh
            )
            sh_out = _materialize_candidate(
                base_state=init_state,
                topo=topo,
                q_ac_eff=q_ac_sh,
                q_dc_cont=q_dc_base,
                raw_quant_cont=rq_cand,
                ytox_cont=ytox_init,
                ytob_cont=ytob_init,
                enable_noise=enable_noise,
                noise_lut_q=latents["noise_lut_q"] if enable_noise else None,
                vgg_params=params_vgg,
                target_features=target_features,
                ls_map=ls_map,
                rd_lambda=rd_lambda,
                eff_scales=eff_scales,
                num_levels=num_levels,
                opt_custom_dequant=use_custom_dequant,
                dc_dequant_scale=jnp.array([0.85, 0.75, 1.15], dtype=jnp.float32),
                q_dc_target=q_dc_base,
                explicit_dequant_matrices=dequant_matrices,
                forward_hard_fn=forward_hard_fn,
                is_final_checkpoint=True,
            )
            if sh_out[3] < c_out[3]:
              c_out = sh_out

      c_bytes, c_bpp, c_wd, c_j, c_state, c_lut, c_recon = c_out
      history.append(
          {
              "step": float(step_idx),
              "wd": c_wd,
              "bpp_true": c_bpp,
              "bpp_soft": float(bpp_s),
              "lagrangian_true": c_j,
          }
      )
      if c_j < best_j:
        best_bytes = c_bytes
        best_bpp = c_bpp
        best_wd = c_wd
        best_j = c_j
        best_state = c_state
        best_lut = c_lut
        best_recon = c_recon

  djxl_rgb = decode_with_djxl(best_bytes).astype(np.float32) / 255.0
  mse = float(np.mean((best_recon - djxl_rgb) ** 2))
  psnr_jax_djxl = -10.0 * np.log10(max(mse, 1e-10))
  psnr_orig_djxl = compute_psnr(target_hwc, djxl_rgb, peak=1.0)
  cross_psnr = (
      float(verify_jxl_codestream(best_bytes)["cross_decoder_psnr"])
      if verify_decoders
      else None
  )
  return OptimizationResult(
      jxl_bytes=best_bytes,
      bpp=best_bpp,
      wd=best_wd,
      lagrangian=best_j,
      psnr_jax_vs_djxl=psnr_jax_djxl,
      recon_jax_rgb=best_recon,
      recon_djxl_rgb=djxl_rgb,
      state=best_state,
      noise_lut=best_lut,
      history=history,
      psnr_db=psnr_orig_djxl,
      num_bytes=len(best_bytes),
      step0_jxl_bytes=step0_bytes,
      step0_bpp=step0_bpp,
      step0_wd=step0_wd,
      step0_psnr_db=step0_psnr_db,
      cross_decoder_psnr=cross_psnr,
      stage1_candidates=s1_candidates,
      stage2_ac_strategy=stage2_acs.copy(),
      stage2_true_bpp=stage2_true_bpp,
      stage2_wd=stage2_wd,
      stage2_true_j=stage2_true_j,
      stage3_true_j=best_j,
      all_default=not use_custom_dequant,
  )


# =============================================================================
# 4. Multi-Transform Rate-WD Initializer & Optimizer
#    (Used by Kodak R-D Benchmark vs cjxl -e 7)
# =============================================================================


_WD_STATE_ACS_CACHE: dict[tuple[Any, ...], np.ndarray] = {}


def init_pure_jax_wd_state(
    rgb_u8: np.ndarray | jax.Array,
    distance: float,
    dequant_matrices: dict[int, np.ndarray] | None = None,
    *,
    raw_q_exp: float = -0.18,
    deadzone: float = 0.60,
    hf_deadzone: float = 0.10,
    soft_shrink: float = 0.05,
    qdc_base: float = 30.0,
    allow_multiblock: bool = True,
    opt_custom_dequant: bool = True,
    vgg_params: Any = None,
    lambda_rate: float | None = None,
    rd_lambda: float | None = None,
    init_mode: str = "dct",
    seed: int = 0,
) -> tuple[JxlVarDCTState, StaticVarDCTTopology, jax.Array, jax.Array]:
  """100% pure-JAX WD-calibrated VarDCT state initialization with Stage 1 + Stage 2 AcStrategy."""
  if float(distance) <= 0.0:
    raise ValueError(f"distance must be > 0, got {distance}")
  init_mode_norm = _normalize_init_mode(init_mode)
  arr = _normalize_rgb_hwc(rgb_u8)
  h, w, _ = arr.shape
  h8, w8 = (h + 7) // 8, (w + 7) // 8
  ht, wt = (h8 + 7) // 8, (w8 + 7) // 8

  srgb_chw = jnp.array(np.transpose(arr, (2, 0, 1)), dtype=jnp.float32)
  xyb_raw = np.asarray(srgb_to_xyb(srgb_chw))
  pad_h, pad_w = h8 * 8 - h, w8 * 8 - w
  xyb_pad = (
      np.pad(xyb_raw, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
      if (pad_h > 0 or pad_w > 0)
      else xyb_raw
  )

  # Stage 1 + Stage 2 DP AcStrategy selection (zero hardcoded ac_strategy = 0)
  eff_rd_lam = (
      float(rd_lambda)
      if rd_lambda is not None
      else (float(lambda_rate) if lambda_rate is not None else None)
  )
  acs_cache_key = (
      h,
      w,
      bool(allow_multiblock),
      hashlib.md5(np.ascontiguousarray(arr).tobytes()).digest(),
  )
  cached_acs = _WD_STATE_ACS_CACHE.get(acs_cache_key)
  if cached_acs is not None:
    ac_strategy = cached_acs.copy()
  else:
    ac_strategy = select_ac_strategy_jax(
        arr,
        distance=float(distance),
        allow_multiblock=allow_multiblock,
        vgg_params=vgg_params,
        rd_lambda=eff_rd_lam,
        stage1_steps=1,
    )
    while len(_WD_STATE_ACS_CACHE) >= 16:
      del _WD_STATE_ACS_CACHE[next(iter(_WD_STATE_ACS_CACHE))]
    _WD_STATE_ACS_CACHE[acs_cache_key] = ac_strategy.copy()
  _, epf_sharp, epf_iters = select_quant_and_epf_jax(
      arr, distance=float(distance), ac_strategy=ac_strategy
  )

  y_blocks = xyb_pad[1].reshape(h8, 8, w8, 8).transpose(0, 2, 1, 3)
  y_std = np.std(y_blocks, axis=(2, 3))
  med_std = float(np.median(y_std) + 1e-4)
  base_raw_q = 8.0
  rel_act = np.clip((y_std + 0.012) / (med_std + 0.012), 0.70, 1.35)
  raw_q = np.clip(np.round(base_raw_q * (rel_act**raw_q_exp)), 5, 13).astype(
      np.int32
  )
  raw_q = broadcast_quant_field_to_ac_strategy(raw_q, ac_strategy)

  # AC-only least-squares chroma-from-luma (subtracting 8x8 block DC means first)
  y_ac_only = (
      (y_blocks - np.mean(y_blocks, axis=(2, 3), keepdims=True))
      .transpose(0, 2, 1, 3)
      .reshape(h8 * 8, w8 * 8)
  )
  x_blocks = xyb_pad[0].reshape(h8, 8, w8, 8).transpose(0, 2, 1, 3)
  x_ac_only = (
      (x_blocks - np.mean(x_blocks, axis=(2, 3), keepdims=True))
      .transpose(0, 2, 1, 3)
      .reshape(h8 * 8, w8 * 8)
  )
  b_blocks = xyb_pad[2].reshape(h8, 8, w8, 8).transpose(0, 2, 1, 3)
  b_ac_only = (
      (b_blocks - np.mean(b_blocks, axis=(2, 3), keepdims=True))
      .transpose(0, 2, 1, 3)
      .reshape(h8 * 8, w8 * 8)
  )

  ytox_map = np.zeros((ht, wt), dtype=np.int8)
  ytob_map = np.zeros((ht, wt), dtype=np.int8)
  for ty in range(ht):
    for tx in range(wt):
      ys = y_ac_only[
          ty * 64 : min((ty + 1) * 64, h8 * 8),
          tx * 64 : min((tx + 1) * 64, w8 * 8),
      ]
      xs = x_ac_only[
          ty * 64 : min((ty + 1) * 64, h8 * 8),
          tx * 64 : min((tx + 1) * 64, w8 * 8),
      ]
      bs = b_ac_only[
          ty * 64 : min((ty + 1) * 64, h8 * 8),
          tx * 64 : min((tx + 1) * 64, w8 * 8),
      ]
      denom = float(np.sum(ys * ys)) + 1e-4
      kx = float(np.sum(xs * ys)) / denom
      kb = float(np.sum(bs * ys)) / denom
      ytox_map[ty, tx] = int(
          np.clip(
              round((kx - K_BASE_CORRELATION_X) * K_DEFAULT_COLOR_FACTOR),
              -128,
              127,
          )
      )
      ytob_map[ty, tx] = int(
          np.clip(
              round((kb - K_BASE_CORRELATION_B) * K_DEFAULT_COLOR_FACTOR),
              -128,
              127,
          )
      )

  state = init_state_from_rgb_jax(
      arr,
      distance=distance,
      ac_strategy=ac_strategy,
      raw_quant_field=raw_q,
      epf_sharpness=epf_sharp,
      ytox_map=ytox_map,
      ytob_map=ytob_map,
      epf_iters=epf_iters,
      gab=True,
  )
  state._orig_dequant_matrices = get_default_dequant_matrices()
  state.dequant_matrices = {}
  state.dc_quant_weights = _DEFAULT_DC_QUANT_WEIGHTS_NP.copy()

  if dequant_matrices is not None:
    state.dequant_matrices = {
        int(k): np.asarray(v, dtype=np.float32)
        for k, v in dequant_matrices.items()
    }
  elif opt_custom_dequant:
    state.dc_quant_weights = (
        _DEFAULT_DC_QUANT_WEIGHTS_NP
        * np.array([0.88, 0.80, 1.12], dtype=np.float32)
    ).astype(np.float32)
    _, rt_dc = roundtrip_custom_dequant_matrices(state)
    state.dc_quant_weights = rt_dc

  d = max(float(distance), 0.05)
  target_scale = 0.65 / d
  state.global_scale = int(
      np.clip(round(K_GLOBAL_SCALE_DENOM * target_scale / base_raw_q), 1, 65535)
  )
  state.quant_dc = int(np.clip(round(qdc_base * (d**0.25)), 16, 64))
  state.x_qm_scale = 3 + int(d > 2.5) + int(d > 5.5) + int(d > 9.5)
  state.b_qm_scale = 2 + int(d > 5.5)

  topo = StaticVarDCTTopology(state)
  with jax.ensure_compile_time_eval():
    for grp in topo.multi_groups:
      get_canonical_ac_scan_tables(int(grp["cy_l"]), int(grp["cx_l"]))

  q_ac_raw, q_dc_raw = compute_continuous_latents_jax(
      arr, state, topo=topo, init_mode=init_mode_norm, seed=seed
  )
  if init_mode_norm == "dct":
    eff_dc_scale = jnp.asarray(
        state.dc_quant_weights / _DEFAULT_DC_QUANT_WEIGHTS_NP, dtype=jnp.float32
    )
    u_dc = q_dc_raw / eff_dc_scale[:, None, None]

    fy8 = jnp.arange(8, dtype=jnp.float32)[:, None]
    fx8 = jnp.arange(8, dtype=jnp.float32)[None, :]
    fnorm8 = jnp.sqrt(fy8 * fy8 + fx8 * fx8) / math.sqrt(98.0)
    dz_grid = deadzone + hf_deadzone * fnorm8[None, None, :, :]

    def _dz_shrink(u: jax.Array, dz: jax.Array) -> jax.Array:
      abs_u = jnp.abs(u)
      shrunk = jnp.maximum(0.0, abs_u - soft_shrink)
      scale = jnp.where(
          abs_u < dz, abs_u * (0.45 / jnp.maximum(dz, 0.5)), shrunk
      )
      return jnp.sign(u) * scale

    q_ac_b = jnp.transpose(q_ac_raw.reshape(3, h8, 8, w8, 8), (1, 3, 0, 2, 4))
    uy = _dz_shrink(q_ac_b[:, :, 1], dz_grid)
    ux = _dz_shrink(q_ac_b[:, :, 0], dz_grid + 0.05)
    ub = _dz_shrink(q_ac_b[:, :, 2], dz_grid + 0.05)
    u_blk = jnp.stack([ux, uy, ub], axis=2)
    u_ac_3hw = jnp.transpose(u_blk, (2, 0, 3, 1, 4)).reshape(3, h8 * 8, w8 * 8)
  else:
    u_dc = q_dc_raw
    u_ac_3hw = q_ac_raw

  state.dc_quant = np.asarray(jnp.round(u_dc), dtype=np.int32)
  state.quant_ac = mask_llf_numpy(
      np.asarray(jnp.round(u_ac_3hw), dtype=np.int32), topo
  )
  return state, topo, u_ac_3hw, u_dc


@jax.jit
def _single_compile_1x1_loss_and_grad(
    params: dict[str, jax.Array],
    feats_ref: tuple[jax.Array, ...],
    basis_1x1: jax.Array,
    dq_1x1: jax.Array,
    raw_q_j: jax.Array,
    ytox_j: jax.Array,
    ytob_j: jax.Array,
    inv_gs: jax.Array,
    inv_qdc: jax.Array,
    x_dm_mul: jax.Array,
    b_dm_mul: jax.Array,
    lam_rate: jax.Array,
    vgg_params: Any,
) -> tuple[tuple[jax.Array, tuple[jax.Array, jax.Array]], dict[str, jax.Array]]:
  """Single-compile JIT loss & grad across all images and distances of same (H, W)."""
  h8, w8 = raw_q_j.shape
  h_px, w_px = h8 * 8, w8 * 8
  h_orig, w_orig = feats_ref[0].shape[1], feats_ref[0].shape[2]
  k_dc_quant = jnp.array(K_DC_QUANT, dtype=jnp.float32)

  def _loss(p: dict[str, jax.Array]) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    q_ac = ste_round(p["u_ac"])
    q_dc = ste_round(p["u_dc"])

    mul_dc = inv_gs * inv_qdc * k_dc_quant
    dc_y = q_dc[1] * mul_dc[1]
    dc_x = q_dc[0] * mul_dc[0] + K_BASE_CORRELATION_X * dc_y
    dc_b = q_dc[2] * mul_dc[2] + K_BASE_CORRELATION_B * dc_y
    dc_raw = jnp.stack([dc_x, dc_y, dc_b], axis=0)
    dc_smoothed = adaptive_dc_smoothing(dc_raw, mul_dc)

    q_hat_all = adjust_quant_bias_ste(q_ac, hard=False)
    q_hat_b = jnp.transpose(
        q_hat_all.reshape(3, h8, 8, w8, 8), (1, 3, 0, 2, 4)
    ).reshape(h8 * w8, 3, 8, 8)
    q_hat_b = q_hat_b.at[:, :, 0, 0].set(0.0)

    sd = (inv_gs / jnp.maximum(raw_q_j, 1.0)).reshape(h8 * w8, 1, 1)
    by_grid = jnp.repeat(jnp.arange(h8), w8)
    bx_grid = jnp.tile(jnp.arange(w8), h8)
    x_cc_tile = K_BASE_CORRELATION_X + ytox_j / float(K_DEFAULT_COLOR_FACTOR)
    b_cc_tile = K_BASE_CORRELATION_B + ytob_j / float(K_DEFAULT_COLOR_FACTOR)
    x_cc = x_cc_tile[by_grid // 8, bx_grid // 8][:, None, None]
    b_cc = b_cc_tile[by_grid // 8, bx_grid // 8][:, None, None]

    wy = dq_1x1[:, 1]
    wx = dq_1x1[:, 0]
    wb = dq_1x1[:, 2]
    coeff_y = q_hat_b[:, 1] * wy * sd
    coeff_x = q_hat_b[:, 0] * wx * (sd * x_dm_mul) + x_cc * coeff_y
    coeff_b = q_hat_b[:, 2] * wb * (sd * b_dm_mul) + b_cc * coeff_y
    coeffs = jnp.stack([coeff_x, coeff_y, coeff_b], axis=1)
    dc_flat = jnp.transpose(dc_smoothed.reshape(3, h8 * w8), (1, 0))
    coeffs = coeffs.at[:, :, 0, 0].set(dc_flat)

    pixels_flat = jnp.einsum(
        "nij,ncj->nci", basis_1x1, coeffs.reshape(-1, 3, 64)
    )
    xyb_blocks = pixels_flat.reshape(h8, w8, 3, 8, 8)
    xyb_rec = jnp.transpose(xyb_blocks, (2, 0, 3, 1, 4)).reshape(3, h_px, w_px)
    srgb_chw = jnp.clip(xyb_to_srgb(xyb_rec[:, :h_orig, :w_orig]), 0.0, 1.0)

    feats_rec = multiscale_vgg_features(
        vgg_params, srgb_chw, num_scales=(len(feats_ref) - 1) // 5
    )
    ls_map = jnp.full((h_orig, w_orig), 1.0, dtype=jnp.float32)
    wd_val = multi_wasserstein_distortion(feats_ref, feats_rec, ls_map)

    q_ac_5d = jnp.transpose(
        p["u_ac"].reshape(3, h8, 8, w8, 8), (1, 3, 0, 2, 4)
    )
    ac_bits = soft_ac_rate_bits(
        q_ac_5d, topology=None, ac_strategy=None, tau=0.20
    )
    dc_bits = soft_dc_rate_bits(p["u_dc"], ac_strategy=None, tau=0.12)
    bpp_val = (ac_bits + dc_bits) / float(h_orig * w_orig)
    return wd_val + lam_rate * bpp_val, (wd_val, bpp_val)

  return jax.value_and_grad(_loss, has_aux=True)(params)


_COMPILED_MB_SOLVERS: dict[tuple[Any, ...], tuple[Any, Any]] = {}


def _make_mb_point_step_fn(
    state: JxlVarDCTState, topo: StaticVarDCTTopology
) -> tuple[Any, Any]:
  """Compiles and caches (forward_fn, grad_fn) for multi-block optimize_rate_wd_point."""
  key = (
      topo.h8,
      topo.w8,
      state.height,
      state.width,
      bytes(np.asarray(state.ac_strategy, dtype=np.uint8).tobytes()),
      bool(state.gab),
      int(state.epf_iters),
  )
  if key in _COMPILED_MB_SOLVERS:
    return _COMPILED_MB_SOLVERS[key]

  topo_grad = topo
  num_px = float(state.height * state.width)

  @jax.jit
  def _mb_fwd(
      params: dict[str, jax.Array],
      raw_q_j: jax.Array,
      ytox_j: jax.Array,
      ytob_j: jax.Array,
      dc_w_j: jax.Array,
      gs_f: jax.Array,
      qdc_f: jax.Array,
      x_qm_f: jax.Array,
      b_qm_f: jax.Array,
  ) -> jax.Array:
    return reconstruct_single_frame(
        topo_grad,
        params_override={
            "global_scale": gs_f,
            "quant_dc": qdc_f,
            "x_qm_scale": x_qm_f,
            "b_qm_scale": b_qm_f,
        },
        quant_ac=params["u_ac"],
        dc_quant=params["u_dc"],
        raw_quant_field=raw_q_j,
        ytox_map=ytox_j,
        ytob_map=ytob_j,
        dc_quant_weights=dc_w_j,
        include_noise=False,
        hard=False,
        output_layout="CHW",
    )

  @jax.jit
  def _mb_grad(
      params: dict[str, jax.Array],
      grad_recon: jax.Array,
      raw_q_j: jax.Array,
      ytox_j: jax.Array,
      ytob_j: jax.Array,
      dc_w_j: jax.Array,
      gs_f: jax.Array,
      qdc_f: jax.Array,
      x_qm_f: jax.Array,
      b_qm_f: jax.Array,
      lam_rate: jax.Array,
      cluster_blend: jax.Array = jnp.float32(0.0),
  ) -> dict[str, jax.Array]:
    def _surrogate(p: dict[str, jax.Array]) -> jax.Array:
      r_chw = reconstruct_single_frame(
          topo_grad,
          params_override={
              "global_scale": gs_f,
              "quant_dc": qdc_f,
              "x_qm_scale": x_qm_f,
              "b_qm_scale": b_qm_f,
          },
          quant_ac=p["u_ac"],
          dc_quant=p["u_dc"],
          raw_quant_field=raw_q_j,
          ytox_map=ytox_j,
          ytob_map=ytob_j,
          dc_quant_weights=dc_w_j,
          include_noise=False,
          hard=False,
          output_layout="CHW",
      )
      wd_lin = jnp.sum(jax.lax.stop_gradient(grad_recon) * r_chw)
      ac_b = soft_ac_rate_bits(
          p["u_ac"],
          topology=topo_grad,
          tau=0.20,
          raw_quant_field=raw_q_j,
          cluster_blend=cluster_blend,
      )
      dc_b = soft_dc_rate_bits(
          p["u_dc"], ac_strategy=state.ac_strategy, tau=0.12
      )
      return wd_lin + lam_rate * ((ac_b + dc_b) / num_px)

    return jax.grad(_surrogate)(params)

  while len(_COMPILED_MB_SOLVERS) >= 6:
    del _COMPILED_MB_SOLVERS[next(iter(_COMPILED_MB_SOLVERS))]
  _COMPILED_MB_SOLVERS[key] = (_mb_fwd, _mb_grad)
  return _mb_fwd, _mb_grad


def optimize_rate_wd_point(
    rgb_u8: np.ndarray | jax.Array,
    distance_or_state: float | JxlVarDCTState = 1.5,
    topo: StaticVarDCTTopology | None = None,
    u_ac_init: jax.Array | None = None,
    u_dc_init: jax.Array | None = None,
    feats_ref: tuple[jax.Array, ...] | None = None,
    vgg_params: Any = None,
    distance: float | None = None,
    steps: int | None = None,
    num_steps: int | None = None,
    lr: float = 0.055,
    lr_ac: float | None = None,
    lr_dc: float | None = None,
    rate_weight: float | None = None,
    lambda_rate: float | None = None,
    dequant_matrices: dict[int, np.ndarray] | None = None,
    enable_noise: bool = False,
    verify_decoders: bool = False,
    verify_decoder: bool = False,
    return_details: bool = False,
    init_mode: str = "dct",
    seed: int = 0,
) -> OptimizationResult | tuple[float, float, float, float, float] | dict[str, Any]:
  """Runs 3-stage Adam+STE Rate-WD optimization on a VarDCT state.

  Supports both:
  1. High-level signature:
     `optimize_rate_wd_point(rgb_u8, distance, steps=8, lr=0.055, vgg_params=None, dequant_matrices=None, enable_noise=False, verify_decoders=False, init_mode="dct", seed=0) -> OptimizationResult`
  2. Low-level prototype signature:
     `optimize_rate_wd_point(rgb_u8, state, topo, u_ac_init, u_dc_init, feats_ref, vgg_params, distance, steps=8, verify_decoder=False, return_details=False)`
  """
  import time

  init_mode_norm = _normalize_init_mode(init_mode)
  legacy_tuple_return = isinstance(distance_or_state, JxlVarDCTState)
  verify = bool(verify_decoders or verify_decoder)
  default_steps = 40 if init_mode_norm in ("zero", "random") else 8
  eff_steps = int(
      num_steps
      if num_steps is not None
      else (steps if steps is not None else default_steps)
  )
  eff_lr = float(lr_ac if lr_ac is not None else lr)
  dc_lr_scale = (
      float(lr_dc) / max(eff_lr, 1e-8) if lr_dc is not None else 0.35
  )
  params_vgg = _get_vgg_params(vgg_params)
  ref_01_np = _normalize_rgb_hwc(rgb_u8)
  ref_01 = jnp.asarray(ref_01_np, dtype=jnp.float32)
  h, w, _ = ref_01_np.shape
  h8, w8 = (h + 7) // 8, (w + 7) // 8

  if not legacy_tuple_return and not return_details:
    eff_distance = (
        float(distance) if distance is not None else float(distance_or_state)
    )
    if eff_distance <= 0.0:
      raise ValueError(f"distance must be > 0, got {eff_distance}")
    lam_f = (
        float(rate_weight)
        if rate_weight is not None
        else (
            float(lambda_rate)
            if lambda_rate is not None
            else float(0.12 * (eff_distance**1.15))
        )
    )
    return optimize_single_frame(
        ref_01_np,
        params_vgg,
        distance=eff_distance,
        rd_lambda=lam_f,
        num_steps=eff_steps,
        lr=eff_lr,
        enable_noise=enable_noise,
        allow_multiblock=True,
        opt_custom_dequant=True,
        dequant_matrices=dequant_matrices,
        scale_distance_with_lambda=False,
        verify_decoders=verify,
        init_mode=init_mode_norm,
        seed=seed,
    )

  if legacy_tuple_return:
    state = distance_or_state
    eff_distance = 1.5 if distance is None else float(distance)
    assert topo is not None and u_ac_init is not None and u_dc_init is not None
  else:
    eff_distance = (
        float(distance) if distance is not None else float(distance_or_state)
    )
    if eff_distance <= 0.0:
      raise ValueError(f"distance must be > 0, got {eff_distance}")
    state, topo, u_ac_init, u_dc_init = init_pure_jax_wd_state(
        rgb_u8,
        eff_distance,
        dequant_matrices=dequant_matrices,
        vgg_params=params_vgg,
        init_mode=init_mode_norm,
        seed=seed,
    )

  if feats_ref is None:
    eff_scales = select_num_scales(h, w, max_scales=3)
    feats_ref = tuple(
        _jit_extract_vgg_features(
            params_vgg, jnp.moveaxis(ref_01, -1, 0), eff_scales=eff_scales
        )
    )

  lut_list: list[float] | None = None
  if enable_noise:
    rec0_hwc = reconstruct_single_frame(state, include_noise=False, hard=True)
    lut0 = np.asarray(estimate_initial_noise_lut_jax(ref_01_np, rec0_hwc))
    lut_steps = np.clip(np.round(lut0 * NOISE_PRECISION), 2, 1023)
    lut_list = [float(x) for x in (lut_steps / NOISE_PRECISION)]

  t0 = time.time()
  jb0, _ = encode_single_frame(state, noise_lut=lut_list)
  dec0_u8 = decode_with_djxl(jb0)
  t_step0 = time.time() - t0
  dec0_01 = dec0_u8.astype(np.float32) / 255.0
  bpp0 = len(jb0) * 8.0 / float(h * w)
  wd0 = float(compute_wd(ref_01, jnp.asarray(dec0_01), params_vgg))
  psnr0 = compute_psnr(ref_01_np, dec0_01, peak=1.0)

  if init_mode_norm in ("zero", "random"):
    lam_f = (
        float(rate_weight)
        if rate_weight is not None
        else (
            float(lambda_rate)
            if lambda_rate is not None
            else float(0.12 * (eff_distance**1.15))
        )
    )
    opt_res = optimize_single_frame(
        ref_01_np,
        params_vgg,
        distance=eff_distance,
        rd_lambda=lam_f,
        num_steps=eff_steps,
        lr=eff_lr,
        enable_noise=enable_noise,
        allow_multiblock=True,
        opt_custom_dequant=True,
        dequant_matrices=dequant_matrices,
        scale_distance_with_lambda=False,
        verify_decoders=verify,
        init_mode=init_mode_norm,
        seed=seed,
    )
    best_jb = opt_res.jxl_bytes
    best_dec_u8 = opt_res.decoded_djxl
    best_bpp = opt_res.bpp
    best_wd = opt_res.wd_final
    best_psnr = compute_psnr(
        ref_01_np, best_dec_u8.astype(np.float32) / 255.0, peak=1.0
    )
    best_state = opt_res.state
    state.quant_ac = best_state.quant_ac.copy()
    state.dc_quant = best_state.dc_quant.copy()
    t_total = time.time() - t0
    cross_psnr = opt_res.cross_decoder_psnr if verify else 100.0
    if return_details:
      return {
          "jb0": jb0,
          "dec0": dec0_u8,
          "bpp0": bpp0,
          "wd0": wd0,
          "psnr0": psnr0,
          "elapsed0_s": t_step0,
          "best_jb": best_jb,
          "best_dec": best_dec_u8,
          "best_bpp": best_bpp,
          "best_wd": best_wd,
          "best_psnr": best_psnr,
          "elapsed_opt_s": t_total,
          "cross_psnr": cross_psnr if verify else None,
          "state": best_state,
          "ac_strategy": best_state.ac_strategy.copy(),
          "all_default": False,
      }
    return bpp0, wd0, best_bpp, best_wd, cross_psnr

  best_jb = jb0
  best_dec_u8 = dec0_u8
  best_dec_01 = dec0_01
  best_bpp = bpp0
  best_wd = wd0
  best_psnr = psnr0
  best_state = state.copy()
  lam_rate_f = (
      float(rate_weight)
      if rate_weight is not None
      else (
          float(lambda_rate)
          if lambda_rate is not None
          else float(0.22 * (eff_distance**1.35))
      )
  )
  best_j = wd0 + lam_rate_f * bpp0

  if eff_steps > 0:
    lam_rate = jnp.float32(lam_rate_f)
    raw_q_j = jnp.array(state.raw_quant_field, dtype=jnp.float32)
    ytox_j = jnp.array(state.ytox_map, dtype=jnp.float32)
    ytob_j = jnp.array(state.ytob_map, dtype=jnp.float32)

    fy = jnp.arange(8, dtype=jnp.float32)[:, None]
    fx = jnp.arange(8, dtype=jnp.float32)[None, :]
    r2 = fy * fy + fx * fx
    freq_w = jnp.exp(-r2 / 24.0).at[0, 0].set(0.0)
    freq_w_hw = jnp.tile(freq_w, (h8, w8))[None, :, :]
    ch_w = jnp.array([0.45, 1.0, 0.30], dtype=jnp.float32)[:, None, None]
    active_w = jnp.where(jnp.abs(u_ac_init) > 0.30, 1.0, 0.06)
    ac_mask = active_w * freq_w_hw * ch_w

    params = {
        "u_ac": jnp.array(u_ac_init, dtype=jnp.float32),
        "u_dc": jnp.array(u_dc_init, dtype=jnp.float32),
    }
    opt = optax.adam(learning_rate=eff_lr)
    opt_state = opt.init(params)

    has_multiblock = bool(
        topo.multi_groups or state.epf_iters > 0 or state.gab
    )
    dc_w_j = jnp.asarray(state.dc_quant_weights, dtype=jnp.float32)
    eff_scales_m = (len(feats_ref) - 1) // 5
    ls_map_m = jnp.full((h, w), 1.0, dtype=jnp.float32)
    target_feats_list = list(feats_ref)

    if not has_multiblock:
      inv_gs = jnp.float32(K_GLOBAL_SCALE_DENOM / float(state.global_scale))
      inv_qdc = jnp.float32(1.0 / float(state.quant_dc))
      x_dm_mul = jnp.float32((1.0 / 1.25) ** (state.x_qm_scale - 2.0))
      b_dm_mul = jnp.float32((1.0 / 1.25) ** (state.b_qm_scale - 2.0))
      mb_fwd_fn = mb_grad_fn = None
    else:
      gs_f = jnp.float32(state.global_scale)
      qdc_f = jnp.float32(state.quant_dc)
      x_qm_f = jnp.float32(state.x_qm_scale)
      b_qm_f = jnp.float32(state.b_qm_scale)
      mb_fwd_fn, mb_grad_fn = _make_mb_point_step_fn(state, topo)

    for step in range(1, int(eff_steps) + 1):
      if not has_multiblock:
        (_, (_, bpp_s)), grads = _single_compile_1x1_loss_and_grad(
            params,
            feats_ref,
            topo.basis_1x1,
            topo.dq_1x1,
            raw_q_j,
            ytox_j,
            ytob_j,
            inv_gs,
            inv_qdc,
            x_dm_mul,
            b_dm_mul,
            lam_rate,
            params_vgg,
        )
      else:
        assert mb_fwd_fn is not None and mb_grad_fn is not None
        recon_chw = mb_fwd_fn(
            params,
            raw_q_j,
            ytox_j,
            ytob_j,
            dc_w_j,
            gs_f,
            qdc_f,
            x_qm_f,
            b_qm_f,
        )
        _, grad_recon = _jit_wd_val_and_grad_chw(
            params_vgg,
            recon_chw,
            target_feats_list,
            ls_map_m,
            eff_scales_m,
            2,
        )
        mc_blend_m = jnp.float32(
            1.0 if (step > int(0.7 * eff_steps) and h8 * w8 >= 64) else 0.0
        )
        grads = mb_grad_fn(
            params,
            grad_recon,
            raw_q_j,
            ytox_j,
            ytob_j,
            dc_w_j,
            gs_f,
            qdc_f,
            x_qm_f,
            b_qm_f,
            lam_rate,
            mc_blend_m,
        )

      masked_grads = {
          "u_ac": grads["u_ac"] * ac_mask,
          "u_dc": grads["u_dc"] * dc_lr_scale,
      }
      updates, opt_state = opt.update(masked_grads, opt_state, params)
      params = optax.apply_updates(params, updates)

      if step in (max(1, int(eff_steps) // 2), int(eff_steps)):
        cand_state = state.copy()
        cand_state.quant_ac = mask_llf_numpy(
            np.asarray(jnp.round(params["u_ac"]), dtype=np.int32), topo
        )
        cand_state.dc_quant = np.asarray(
            jnp.round(params["u_dc"]), dtype=np.int32
        )
        jb, _ = encode_single_frame(cand_state, noise_lut=lut_list)
        dec_u8 = decode_with_djxl(jb)
        dec_01 = dec_u8.astype(np.float32) / 255.0
        bpp_c = len(jb) * 8.0 / float(h * w)
        wd_c = float(compute_wd(ref_01, jnp.asarray(dec_01), params_vgg))
        j_c = wd_c + lam_rate_f * bpp_c
        if j_c < best_j:
          best_j = j_c
          best_bpp = bpp_c
          best_wd = wd_c
          best_jb = jb
          best_dec_u8 = dec_u8
          best_dec_01 = dec_01
          best_psnr = compute_psnr(ref_01_np, dec_01, peak=1.0)
          best_state = cand_state

  # Sync in-place state with best candidate
  state.quant_ac = best_state.quant_ac.copy()
  state.dc_quant = best_state.dc_quant.copy()
  t_total = time.time() - t0

  cross_psnr = (
      float(verify_jxl_codestream(best_jb)["cross_decoder_psnr"])
      if verify
      else 100.0
  )

  if return_details:
    return {
        "jb0": jb0,
        "dec0": dec0_u8,
        "bpp0": bpp0,
        "wd0": wd0,
        "psnr0": psnr0,
        "elapsed0_s": t_step0,
        "best_jb": best_jb,
        "best_dec": best_dec_u8,
        "best_bpp": best_bpp,
        "best_wd": best_wd,
        "best_psnr": best_psnr,
        "elapsed_opt_s": t_total,
        "cross_psnr": cross_psnr if verify else None,
        "state": best_state,
        "ac_strategy": best_state.ac_strategy.copy(),
        "all_default": False,
    }

  return bpp0, wd0, best_bpp, best_wd, cross_psnr


def __getattr__(name: str) -> Any:
  """Lazy re-export of encode_image_to_jxl and EncodeResult from compositing.py."""
  if name in (
      "encode_image_to_jxl",
      "EncodeResult",
      "CompositeOptimizationResult",
  ):
    from jxl_wasserstein import compositing

    return getattr(compositing, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_CANDIDATE_AC_STRATEGIES",
    "AC_STRATEGY_NAMES",
    "OptimizationResult",
    "stage1_explore_ac_candidates",
    "stage2_select_non_overlapping_ac_strategy",
    "select_ac_strategy_jax",
    "broadcast_quant_field_to_ac_strategy",
    "select_quant_and_epf_jax",
    "select_cfl_jax",
    "select_block_params_jax",
    "compute_continuous_latents_jax",
    "mask_llf_numpy",
    "apply_hf_noise_denoising_jax",
    "estimate_initial_noise_lut_jax",
    "rd_lambda_to_distance",
    "optimize_single_frame",
    "init_pure_jax_wd_state",
    "optimize_rate_wd_point",
    "roundtrip_custom_dequant_matrices",
    "ste_dequant_matrices",
    "encode_image_to_jxl",
    "EncodeResult",
]

