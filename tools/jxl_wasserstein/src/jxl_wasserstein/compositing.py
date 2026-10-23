"""Multi-resolution JXL compositing (XYB Patches, XYB PatchDictionary Alpha Blending) & 3-Stage Rate-WD Selector.

Implements:
1. Pure-JAX area-average downsampling (`downsample_rgb_jax`), Stage 1 multi-resolution
   Rate-WD spatial advantage maps (`_stage1_multires_rate_wd_analysis` /
   `compute_detail_mask_jax`), and non-overlapping 8x8-aligned rectangle
   decomposition (`extract_patch_rectangles`).
2. Three-stage global Rate-WD optimization for multi-resolution XYB patch compositing
   (`optimize_patch_composite`) supporting `upsampling in {2, 4, 8}`,
   `blend_mode in {PATCH_BLEND_REPLACE (1), PATCH_BLEND_ADD (2), "auto"}`, and
   custom `DequantMatrices` (`custom_dequant in {True, False}`).
3. Three-stage global Rate-WD optimization for multi-resolution XYB `PatchDictionary`
   alpha blending (`optimize_alpha_composite`) supporting `upsampling in {2, 4, 8}`,
   `dim_shift in {0, 1, 2, 3}`, and `alpha_blend_mode in {4, 5, 6, 7}` with
   `save_before_color_transform=True` on Frame 0 and `has_patches=True` on Frame 1.
4. Automatic Rate-WD mode & upsampling selector (`select_best_mode` /
   `select_best_composite_mode`) and unified high-level entry point
   (`encode_image_to_jxl` / `encode_composited_jxl`) returning `EncodeResult`.

Constraint R1/R2/F3 Compliance:
All frame initializations (`ref_state`, `main_state`, and single-frame baseline),
`AcStrategy` exploration & DP, discrete patch masks, and XYB patch-alpha mattes
are constructed purely in JAX/NumPy from Rate-WD spatial maps (`compute_spatial_wd_map`
and `compute_spatial_rate_map_bits`). Zero `libjxl` per-block choices (`dump_init_state`
/ `cjxl`) and zero static pixel-MSE heuristics are used.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, field
import functools
import os
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.ndimage import binary_closing, label as ndimage_label, uniform_filter

_CPP_ENCODE_POOL = ThreadPoolExecutor(max_workers=4)

_JAX_CACHE_DIR = "/tmp/jxl_wasserstein_jax_cache"
try:
  os.makedirs(_JAX_CACHE_DIR, exist_ok=True)
  jax.config.update("jax_compilation_cache_dir", _JAX_CACHE_DIR)
except Exception:
  pass

from jxl_wasserstein.cpp_bridge import (
    ACS_COVERED_BLOCKS_X,
    ACS_COVERED_BLOCKS_Y,
    DEFAULT_DC_QUANT_WEIGHTS,
    JxlVarDCTState,
    PATCH_BLEND_ADD,
    PATCH_BLEND_ALPHA_ABOVE,
    PATCH_BLEND_ALPHA_BELOW,
    PATCH_BLEND_MUL_ADD_ABOVE,
    PATCH_BLEND_MUL_ADD_BELOW,
    PATCH_BLEND_REPLACE,
    PatchEntry,
    compute_psnr,
    encode_alpha_composite,
    encode_patches_composite,
    encode_single_frame,
    has_custom_dequant_matrices,
    inspect_jxl,
    roundtrip_custom_dequant_matrices,
    verify_jxl_codestream,
)
from jxl_wasserstein.entropy_proxy import (
    compute_spatial_rate_map_bits,
    get_canonical_ac_scan_tables,
    soft_alpha_channel_bits,
    soft_dequant_matrices_rate_bits,
    soft_jxl_rate_proxy,
    soft_patch_dictionary_bits,
)
from jxl_wasserstein.forward_model import (
    K_DC_QUANT,
    K_GLOBAL_SCALE_DENOM,
    StaticVarDCTTopology,
    _blend_xyb_pair,
    apply_xyb_patches,
    gaborish_inverse_3x3,
    get_default_dequant_matrices,
    init_state_from_rgb_jax,
    opsin_to_srgb,
    project_custom_dequant_matrices,
    reconstruct_alpha_composite,
    reconstruct_alpha_composite_xyb,
    reconstruct_patch_composite,
    reconstruct_single_frame,
    reconstruct_single_frame_xyb,
    srgb_to_xyb,
    ste_dequant_matrices,
    ste_quantize_alpha_01,
    synthesize_jxl_noise_xyb,
    upsample_jxl,
)
from jxl_wasserstein.optimizer import (
    DEFAULT_CANDIDATE_AC_STRATEGIES,
    _compute_precond_scales_jax,
    _jit_extract_vgg_features,
    _jit_wd_eval_chw,
    _jit_wd_val_and_grad_chw,
    _normalize_init_mode,
    broadcast_quant_field_to_ac_strategy,
    mask_llf_jax,
    mask_llf_numpy,
    select_cfl_jax,
    select_num_scales,
    select_quant_and_epf_jax,
    stage2_select_non_overlapping_ac_strategy,
    validate_ac_strategy_tiling,
)
from jxl_wasserstein.noise import eval_noise_strength_lut
from jxl_wasserstein.vgg import load_vgg16_params, multiscale_vgg_features
from jxl_wasserstein.wasserstein import (
    compute_spatial_wd_map,
    compute_wd,
    multi_wasserstein_distortion,
)


# Pre-warm canonical AC scan tables outside any @jax.jit trace so module-level
# caches in entropy_proxy hold concrete arrays rather than DynamicJaxprTracers.
with jax.ensure_compile_time_eval():
  for _cy, _cx in ((1, 1), (1, 2), (2, 2), (1, 4), (2, 4), (4, 4)):
    get_canonical_ac_scan_tables(_cy, _cx)


@dataclass
class EncodeResult:
  """Unified result container for single-frame and multi-resolution JXL optimization.

  Supports both attribute access (`res.jxl_bytes`, `res.bpp`, `res.wd`, ...) and
  dictionary access (`res["jxl_bytes"]`, `res["rd_cost"]`, ...) for compatibility
  across CLI, benchmark, and test callers.
  """

  jxl_bytes: bytes
  bpp: float
  wd: float
  psnr_vs_djxl: float
  psnr_djxl_vs_jxlrs: float
  mode_selected: str
  upsampling: int
  dim_shift: int
  has_noise: bool
  noise_lut: list[float] | np.ndarray | None
  num_patches: int
  header_info: dict[str, Any]
  # Additional cross-module fields & aliases
  psnr_db: float = 0.0
  cross_decoder_psnr: float = 0.0
  mode_used: str = ""
  mode: str = ""
  num_bytes: int = 0
  lagrangian: float = 0.0
  rd_cost: float = 0.0
  psnr: float = 0.0
  blend_mode: int | None = None
  alpha_blend_mode: int | None = None
  ref_state: JxlVarDCTState | None = None
  main_state: JxlVarDCTState | None = None
  state: JxlVarDCTState | None = None
  patches: list[PatchEntry] = field(default_factory=list)
  alpha_01: np.ndarray | None = None
  alpha_u8: np.ndarray | None = None
  recon_rgb: np.ndarray | None = None
  recon_jax_rgb: np.ndarray | None = None
  psnr_jax_vs_djxl: float = 0.0
  stats: dict[str, Any] = field(default_factory=dict)
  history: list[dict[str, float]] = field(default_factory=list)
  last_grads: dict[str, Any] | None = None
  all_candidates: list[dict[str, Any]] = field(default_factory=list)
  # 3-Stage Rate-WD & Custom DequantMatrices tracking fields
  stage1_candidates: dict[Any, dict[str, Any]] = field(default_factory=dict)
  stage2_ac_strategy: np.ndarray | None = None
  stage2_true_bpp: float = 0.0
  stage2_wd: float = 0.0
  stage2_true_j: float = 0.0
  stage3_true_bpp: float = 0.0
  stage3_wd: float = 0.0
  stage3_true_j: float = 0.0
  custom_dequant_enabled: bool = True
  spatial_delta_j_map: np.ndarray | None = None
  stage1_main_j_map: np.ndarray | None = None
  stage1_ref_j_map: np.ndarray | None = None

  def __post_init__(self) -> None:
    if not self.mode_used:
      self.mode_used = self.mode_selected
    if not self.mode:
      self.mode = self.mode_selected
    if self.num_bytes == 0 and self.jxl_bytes:
      self.num_bytes = len(self.jxl_bytes)
    if self.cross_decoder_psnr == 0.0 and self.psnr_djxl_vs_jxlrs != 0.0:
      self.cross_decoder_psnr = self.psnr_djxl_vs_jxlrs
    if self.psnr_jax_vs_djxl == 0.0 and self.psnr_vs_djxl != 0.0:
      self.psnr_jax_vs_djxl = self.psnr_vs_djxl
    if self.recon_jax_rgb is None and self.recon_rgb is not None:
      self.recon_jax_rgb = self.recon_rgb
    if self.lagrangian == 0.0 and self.rd_cost != 0.0:
      self.lagrangian = self.rd_cost
    elif self.rd_cost == 0.0 and self.lagrangian != 0.0:
      self.rd_cost = self.lagrangian
    if self.stage3_true_bpp == 0.0 and self.bpp != 0.0:
      self.stage3_true_bpp = self.bpp
    if self.stage3_wd == 0.0 and self.wd != 0.0:
      self.stage3_wd = self.wd
    if self.stage3_true_j == 0.0 and self.rd_cost != 0.0:
      self.stage3_true_j = self.rd_cost

  def __getitem__(self, key: str) -> Any:
    return getattr(self, key)

  def __setitem__(self, key: str, value: Any) -> None:
    setattr(self, key, value)

  def __contains__(self, key: str) -> bool:
    return hasattr(self, key)

  def get(self, key: str, default: Any = None) -> Any:
    return getattr(self, key, default)

  def keys(self) -> Sequence[str]:
    return list(self.__dataclass_fields__.keys())

  def values(self) -> Sequence[Any]:
    return [getattr(self, k) for k in self.keys()]

  def items(self) -> Sequence[tuple[str, Any]]:
    return [(k, getattr(self, k)) for k in self.keys()]

  def to_dict(self) -> dict[str, Any]:
    return {k: getattr(self, k) for k in self.keys()}


class CompositeOptimizationResult(EncodeResult):
  """Alias subclass of `EncodeResult` returned by `optimize_patch_composite` and `optimize_alpha_composite`."""


CompositeOptimizeResult = CompositeOptimizationResult


def _normalize_srgb_hwc(arr: np.ndarray | jax.Array) -> np.ndarray:
  """Normalizes an input RGB array (uint8 [0, 255] or float32 [0, 1], HWC or CHW) to float32 (H, W, 3) in [0, 1]."""
  x = np.asarray(arr)
  if x.ndim != 3:
    raise ValueError(f"Expected 3D RGB image, got shape {x.shape}")
  if x.shape[0] == 3 and x.shape[-1] != 3:
    x = np.moveaxis(x, 0, -1)
  if x.shape[-1] != 3:
    raise ValueError(f"Expected 3 channels, got shape {x.shape}")
  if x.dtype == np.uint8 or float(np.max(x)) > 1.5:
    x = x.astype(np.float32) / 255.0
  else:
    x = x.astype(np.float32)
  return np.clip(x, 0.0, 1.0)


def _parse_blend_mode(blend_mode: int | str) -> int | str:
  """Normalizes patch blend mode specification to `PATCH_BLEND_REPLACE`, `PATCH_BLEND_ADD`, or `'auto'`."""
  if isinstance(blend_mode, str):
    bm_low = blend_mode.strip().lower()
    if bm_low == "auto":
      return "auto"
    if bm_low in ("replace", "kreplace", "1"):
      return PATCH_BLEND_REPLACE
    if bm_low in ("add", "kadd", "2"):
      return PATCH_BLEND_ADD
    raise ValueError(
        f"blend_mode string must be in {{'replace', 'add', 'auto'}}, got {blend_mode!r}"
    )
  bm_int = int(blend_mode)
  if bm_int not in (PATCH_BLEND_REPLACE, PATCH_BLEND_ADD):
    raise ValueError(
        f"blend_mode must be PATCH_BLEND_REPLACE (1) or PATCH_BLEND_ADD (2), got {blend_mode}"
    )
  return bm_int


def _parse_alpha_blend_mode(alpha_blend_mode: int | str | None) -> int | str:
  """Normalizes XYB PatchDictionary alpha blend mode to `{4, 5, 6, 7}` or `'auto'`."""
  if alpha_blend_mode is None:
    return PATCH_BLEND_ALPHA_BELOW
  if isinstance(alpha_blend_mode, str):
    bm_low = alpha_blend_mode.strip().lower()
    if bm_low == "auto":
      return "auto"
    if bm_low in ("blend_above", "kblendabove", "alpha_above", "4"):
      return PATCH_BLEND_ALPHA_ABOVE
    if bm_low in ("blend_below", "kblendbelow", "alpha_below", "blend", "5"):
      return PATCH_BLEND_ALPHA_BELOW
    if bm_low in (
        "alpha_weighted_add_above",
        "kalphaweightedaddabove",
        "mul_add_above",
        "6",
    ):
      return PATCH_BLEND_MUL_ADD_ABOVE
    if bm_low in (
        "alpha_weighted_add_below",
        "kalphaweightedaddbelow",
        "mul_add_below",
        "7",
    ):
      return PATCH_BLEND_MUL_ADD_BELOW
    raise ValueError(
        f"Unsupported alpha_blend_mode string: {alpha_blend_mode!r}"
    )
  bm_int = int(alpha_blend_mode)
  if bm_int not in (
      PATCH_BLEND_ALPHA_ABOVE,
      PATCH_BLEND_ALPHA_BELOW,
      PATCH_BLEND_MUL_ADD_ABOVE,
      PATCH_BLEND_MUL_ADD_BELOW,
  ):
    raise ValueError(
        f"alpha_blend_mode must be in {{4, 5, 6, 7}}, got {alpha_blend_mode}"
    )
  return bm_int


def calibrate_rd_lambda(distance: float) -> float:
  """Calibrates Stage 1/2/3 Rate-WD Lagrangian multiplier smoothly across all `d`."""
  d = max(float(distance), 0.05)
  if d < 1.0:
    return float(0.08 * (d ** 1.55))
  return float(0.08 * (d ** 1.15))


def _calibrate_ref_distance(
    d_main: float,
    eff_lambda: float,
    ref_distance: float | None = None,
    upsampling: int = 2,
    enable_noise: bool = False,
) -> float:
  """Calibrates ReferenceOnly frame distance from `d_main` (amortized by 1/U^2 pixel area)."""
  del eff_lambda
  if ref_distance is not None:
    return float(ref_distance)
  d_base = max(float(d_main), 0.05)
  u_eff = float(max(int(upsampling), 1))
  if enable_noise and u_eff > 1.0:
    return max(0.08, d_base / (u_eff ** 2.50))
  return max(0.10, d_base * 0.72)


def estimate_frame_bits(height: int, width: int, distance: float) -> float:
  """Estimates frame bit budget `N * 0.85 * max(d, 0.05)^(-0.78)` from frame dimensions & distance."""
  n_px = float(max(int(height) * int(width), 1))
  d_eff = max(float(distance), 0.05)
  return float(n_px * 0.85 * (d_eff ** -0.78))


DEQUANT_MODE_DOF: dict[str, int] = {
    "none": 0,
    "compact": 9,
    "full_bands": 33,
}


def _resolve_custom_dequant_spec(
    custom_dequant: bool | str | None = None,
    opt_custom_dequant: bool | str | None = None,
    optimize_dequant: bool | str | None = None,
    optimize_dequant_matrices: bool | str | None = None,
    enable_custom_dequant: bool | str | None = None,
    use_custom_dequant: bool | str | None = None,
) -> bool | str | None:
  """Resolves `custom_dequant` specification (`bool | str | None`) across alias arguments."""
  for val in (
      custom_dequant,
      opt_custom_dequant,
      optimize_dequant,
      optimize_dequant_matrices,
      enable_custom_dequant,
      use_custom_dequant,
  ):
    if val is not None:
      if isinstance(val, str):
        v_low = val.strip().lower()
        if v_low in ("true", "1", "yes"):
          return True
        if v_low in ("false", "0", "no"):
          return False
        if v_low in ("auto", "none", "compact", "full_bands"):
          return v_low
        raise ValueError(
            f"Unsupported custom_dequant mode {val!r}; expected bool or one of "
            "{'auto', 'none', 'compact', 'full_bands'}"
        )
      return bool(val)
  return None


def _resolve_custom_dequant_flag(
    custom_dequant: bool | str | None = None,
    opt_custom_dequant: bool | str | None = None,
    optimize_dequant: bool | str | None = None,
    optimize_dequant_matrices: bool | str | None = None,
    enable_custom_dequant: bool | str | None = None,
    use_custom_dequant: bool | str | None = None,
    default: bool = True,
) -> bool:
  """Resolves `custom_dequant` boolean across all alias keyword arguments."""
  spec = _resolve_custom_dequant_spec(
      custom_dequant=custom_dequant,
      opt_custom_dequant=opt_custom_dequant,
      optimize_dequant=optimize_dequant,
      optimize_dequant_matrices=optimize_dequant_matrices,
      enable_custom_dequant=enable_custom_dequant,
      use_custom_dequant=use_custom_dequant,
  )
  if spec is None:
    return bool(default)
  if isinstance(spec, str):
    return spec != "none"
  return bool(spec)


def select_dequant_mode_for_frame(
    height: int,
    width: int,
    distance: float,
    custom_dequant: bool | str | None = None,
    *,
    is_reference: bool = False,
) -> str:
  """Selects `DequantMatrices` parameterization (`'none'`, `'compact'`, `'full_bands'`) from frame bit budget.

  - `< 16 kbits`: `'none'` (0 DoF, default library weights, 1-bit overhead).
  - `16 kbits <= bits < 96 kbits`: `'compact'` (9 DoF: 3 channel scales + 3 frequency tilts + 3 high-freq tapers).
  - `>= 96 kbits`: `'full_bands'` (33 DoF: 3 channel scales + 24 radial log-freq band multipliers + 6 LF corners).
  """
  if isinstance(custom_dequant, str):
    cd_low = custom_dequant.strip().lower()
    if cd_low in ("none", "compact", "full_bands"):
      return cd_low
    if cd_low != "auto":
      raise ValueError(f"Unknown dequant mode: {custom_dequant!r}")
  elif custom_dequant is False:
    return "none"

  est_bits = estimate_frame_bits(height, width, distance)
  if custom_dequant is True:
    if is_reference and est_bits < 16_000.0:
      return "none"
    return "full_bands" if est_bits >= 96_000.0 else "compact"

  if est_bits < 16_000.0:
    return "none"
  if est_bits < 96_000.0:
    return "compact"
  return "full_bands"


def _init_dq_log_vec(dq_mode: str, *, is_reference: bool = False) -> jax.Array:
  """Initializes the `dq_log_scale` parameter vector of shape `(0,)`, `(9,)`, or `(33,)`."""
  del is_reference
  if dq_mode == "none":
    return jnp.zeros((0,), dtype=jnp.float32)
  if dq_mode == "compact":
    return jnp.zeros((9,), dtype=jnp.float32)
  if dq_mode == "full_bands":
    return jnp.zeros((33,), dtype=jnp.float32)
  raise ValueError(f"Unsupported dq_mode: {dq_mode!r}")


def _init_frame_state_with_aq_epf_cfl(
    srgb_hwc: np.ndarray | jax.Array,
    distance: float,
    *,
    ac_strategy: np.ndarray | None = None,
    ac_anchors: np.ndarray | None = None,
    gab: bool | None = None,
    enable_noise: bool = False,
) -> JxlVarDCTState:
  """Initializes a `JxlVarDCTState` with adaptive `raw_quant_field`, `epf_iters`, and CfL."""
  arr = _normalize_srgb_hwc(srgb_hwc)
  h, w, _ = arr.shape
  h8, w8 = (h + 7) // 8, (w + 7) // 8
  eff_gab = bool(float(distance) > 0.5) if gab is None else bool(gab)
  acs = (
      np.asarray(ac_strategy, dtype=np.int32)
      if ac_strategy is not None
      else np.zeros((h8, w8), dtype=np.int32)
  )
  raw_quant_field, epf_sharpness, epf_iters = select_quant_and_epf_jax(
      arr, distance=float(distance), ac_strategy=acs
  )
  ytox_map, ytob_map, ytox_dc, ytob_dc = select_cfl_jax(arr, gab=eff_gab)
  if enable_noise:
    ytox_dc, ytob_dc = 0, 0
  return init_state_from_rgb_jax(
      arr,
      distance=float(distance),
      epf_iters=int(epf_iters),
      gab=eff_gab,
      ac_strategy=ac_strategy,
      ac_anchors=ac_anchors,
      raw_quant_field=raw_quant_field,
      epf_sharpness=epf_sharpness,
      ytox_map=ytox_map,
      ytob_map=ytob_map,
      ytox_dc=int(ytox_dc),
      ytob_dc=int(ytob_dc),
  )


def _estimate_sensor_noise_lut_np(xyb_chw: np.ndarray | jax.Array) -> np.ndarray:
  """Estimates 8-point JXL NoiseParams LUT from plane-detrended flat-block Haar HH grain (0 if clean)."""
  y = np.asarray(xyb_chw, dtype=np.float32)[1]
  h, w = y.shape
  h8, w8 = h // 8, w // 8
  if h8 < 2 or w8 < 2:
    return np.zeros((8,), dtype=np.float32)
  y_b = y[: h8 * 8, : w8 * 8].reshape(h8, 8, w8, 8)
  u_lin = (np.arange(8, dtype=np.float32) - 3.5) / 3.5
  u_var = float(np.mean(u_lin * u_lin))
  gy = np.mean(y_b * u_lin[None, :, None, None], axis=(1, 3), keepdims=True) / u_var
  gx = np.mean(y_b * u_lin[None, None, None, :], axis=(1, 3), keepdims=True) / u_var
  ym = np.mean(y_b, axis=(1, 3), keepdims=True)
  res = y_b - ym - gy * u_lin[None, :, None, None] - gx * u_lin[None, None, None, :]
  res_std = np.sqrt(np.mean(res * res, axis=(1, 3)) + 1e-12)
  hh = 0.5 * (
      y_b[:, :-1, :, :-1]
      - y_b[:, :-1, :, 1:]
      - y_b[:, 1:, :, :-1]
      + y_b[:, 1:, :, 1:]
  )
  blk_hh_std = np.sqrt(np.mean(hh * hh, axis=(1, 3)) + 1e-12)
  unit_hh = 0.56
  flat_mask = (
      (res_std <= 0.012) & (blk_hh_std > 1e-5) & (blk_hh_std >= 0.55 * res_std)
  )
  min_flat = max(8, int(0.10 * h8 * w8)) if (h8 * w8 >= 64) else 2
  if int(np.sum(flat_mask)) < min_flat:
    return np.zeros((8,), dtype=np.float32)
  med_hh = float(np.median(blk_hh_std[flat_mask]))
  global_lut = med_hh / unit_hh
  if global_lut < 0.0022:
    return np.zeros((8,), dtype=np.float32)
  vx_blk = 0.5 * y_b.mean(axis=(1, 3))
  flat_vx = vx_blk[flat_mask]
  flat_luts = blk_hh_std[flat_mask] / unit_hh
  lut = np.zeros((8,), dtype=np.float32)
  has_k = np.zeros((8,), dtype=bool)
  for k in range(8):
    near = np.abs(flat_vx - (k / 6.0)) < 0.14
    if int(np.sum(near)) >= 8:
      lut[k] = 0.75 * float(np.median(flat_luts[near])) + 0.25 * global_lut
      has_k[k] = True
  if not np.any(has_k):
    lut[:] = global_lut
  else:
    first_k = int(np.argmax(has_k))
    for k in range(first_k - 1, -1, -1):
      lut[k] = lut[k + 1]
    for k in range(first_k + 1, 8):
      if not has_k[k]:
        lut[k] = lut[k - 1] * 0.85
  return np.clip(lut, 0.002, 0.045).astype(np.float32)


def estimate_noise_lut_jax(xyb_chw: jax.Array) -> jax.Array:
  """Initializes 8-point JXL NoiseParams LUT from flat-block Haar HH grain (or fallback LUT if explicit)."""
  if not isinstance(xyb_chw, jax.core.Tracer):
    sensor_lut = _estimate_sensor_noise_lut_np(xyb_chw)
    if float(np.max(sensor_lut)) >= 0.0020:
      return jnp.asarray(sensor_lut, dtype=jnp.float32)
    y_np = np.asarray(xyb_chw, dtype=np.float32)[1]
    h8, w8 = y_np.shape[0] // 8, y_np.shape[1] // 8
    if h8 >= 2 and w8 >= 2:
      y_b = y_np[: h8 * 8, : w8 * 8].reshape(h8, 8, w8, 8)
      hh = 0.5 * (
          y_b[:, :-1, :, :-1]
          - y_b[:, :-1, :, 1:]
          - y_b[:, 1:, :, :-1]
          + y_b[:, 1:, :, 1:]
      )
      b_hh = np.sqrt(np.mean(hh * hh, axis=(1, 3)) + 1e-12)
      b_std = y_b.std(axis=(1, 3))
      calm = b_std <= float(np.percentile(b_std, 50.0))
      fb_lut = float(
          np.clip(
              np.median(b_hh[calm] if np.any(calm) else b_hh) / 0.56,
              0.0025,
              0.035,
          )
      )
      return jnp.full((8,), fb_lut, dtype=jnp.float32)
    return jnp.full((8,), 0.0025, dtype=jnp.float32)
  y = jnp.asarray(xyb_chw, dtype=jnp.float32)[1]
  lap = y[1:-1, 1:-1] - 0.25 * (
      y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:]
  )
  hf_std = jnp.sqrt(jnp.mean(jnp.square(lap)) + 1e-8)
  base = jnp.clip(hf_std * 0.15, 0.0025, 0.025)
  return jnp.full((8,), base, dtype=jnp.float32)


def _resolve_enable_noise(
    enable_noise: bool | str | None,
    target_xyb: jax.Array | np.ndarray,
    distance: float,
    h: int,
    w: int,
    *,
    default_auto: bool = False,
) -> tuple[bool, np.ndarray]:
  """Resolves `enable_noise` (`bool | 'auto' | None`) into `(use_noise, sensor_lut_np)`."""
  if enable_noise is None:
    mode_str = "auto" if default_auto else "false"
  elif isinstance(enable_noise, str):
    mode_str = enable_noise.strip().lower()
    if mode_str in ("true", "1", "yes", "on"):
      mode_str = "true"
    elif mode_str in ("false", "0", "no", "off"):
      mode_str = "false"
    elif mode_str != "auto":
      raise ValueError(
          f"enable_noise must be bool or 'auto', got {enable_noise!r}"
      )
  else:
    mode_str = "true" if bool(enable_noise) else "false"

  if mode_str == "false":
    return False, np.zeros((8,), dtype=np.float32)

  sensor_lut = _estimate_sensor_noise_lut_np(target_xyb)
  has_sensor_grain = bool(float(np.max(sensor_lut)) >= 0.0020)
  if mode_str == "true":
    return True, sensor_lut

  use_auto = bool(
      has_sensor_grain and (int(h) * int(w) >= 16384) and float(distance) >= 0.45
  )
  return (
      (True, sensor_lut)
      if use_auto
      else (False, np.zeros((8,), dtype=np.float32))
  )


@jax.jit
def _shrink_sensor_grain_xyb_jax(
    xyb_chw: jax.Array, noise_lut: jax.Array
) -> jax.Array:
  """1-pass 3x3 edge-stopped XYB shrinkage preventing double-noise (2*sigma^2) when NoiseParams is active."""
  vx, vy, vb = xyb_chw[0], xyb_chw[1], xyb_chw[2]
  s_lut = 0.5 * (
      eval_noise_strength_lut(0.5 * (vy - vx), noise_lut)
      + eval_noise_strength_lut(0.5 * (vy + vx), noise_lut)
  )
  sigma_y = 0.50 * s_lut
  planes = jnp.stack([vx, vy, vb - vy], axis=0)
  tau = jnp.stack(
      [0.35 * sigma_y, 1.05 * sigma_y, 0.55 * sigma_y], axis=0
  )
  pad = jnp.pad(planes, ((0, 0), (1, 1), (1, 1)), mode="symmetric")
  w_sum = jnp.ones_like(planes)
  v_sum = planes
  tau2 = jnp.square(1.8 * tau) + 1e-10
  y_tau2 = jnp.square(1.8 * tau[1:2]) + 1e-10
  for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
      if dy == 0 and dx == 0:
        continue
      nb = pad[
          :,
          1 + dy : 1 + dy + planes.shape[1],
          1 + dx : 1 + dx + planes.shape[2],
      ]
      r2 = (
          jnp.square(nb - planes) / tau2
          + 0.5 * jnp.square(nb[1:2] - planes[1:2]) / y_tau2
      )
      w_nb = 1.0 / (1.0 + r2 * r2)
      w_sum = w_sum + w_nb
      v_sum = v_sum + w_nb * nb
  cur = v_sum / w_sum
  return jnp.stack([cur[0], cur[1], cur[2] + cur[1]], axis=0)


@functools.partial(jax.jit, static_argnames=("n_iters",))
def _pyramid_harmonic_inpaint_jax(
    arr_hwc: jax.Array, mask_known_hw: jax.Array, n_iters: int = 8
) -> jax.Array:
  """Multi-scale coarse-to-fine harmonic diffusion into masked regions (`~mask_known_hw`)."""
  w0 = mask_known_hw.astype(jnp.float32)[..., None]
  v0 = arr_hwc * w0
  pyr_v = [v0]
  pyr_w = [w0]
  ch_h, ch_w = int(arr_hwc.shape[0]), int(arr_hwc.shape[1])
  while ch_h >= 4 and ch_w >= 4 and len(pyr_v) < 7:
    vp = pyr_v[-1]
    wp = pyr_w[-1]
    if vp.shape[0] % 2 == 1:
      vp = jnp.pad(vp, ((0, 1), (0, 0), (0, 0)), mode="edge")
      wp = jnp.pad(wp, ((0, 1), (0, 0), (0, 0)), mode="edge")
    if vp.shape[1] % 2 == 1:
      vp = jnp.pad(vp, ((0, 0), (0, 1), (0, 0)), mode="edge")
      wp = jnp.pad(wp, ((0, 0), (0, 1), (0, 0)), mode="edge")
    nh, nw = vp.shape[0] // 2, vp.shape[1] // 2
    pyr_v.append(vp.reshape(nh, 2, nw, 2, -1).mean(axis=(1, 3)))
    pyr_w.append(wp.reshape(nh, 2, nw, 2, -1).mean(axis=(1, 3)))
    ch_h, ch_w = nh, nw

  mean_fallback = jnp.sum(v0, axis=(0, 1), keepdims=True) / jnp.maximum(
      jnp.sum(w0), 1.0
  )
  cur = jnp.where(
      pyr_w[-1] > 1e-5, pyr_v[-1] / jnp.maximum(pyr_w[-1], 1e-5), mean_fallback
  )
  for lvl in range(len(pyr_v) - 2, -1, -1):
    th, tw = pyr_v[lvl].shape[0], pyr_v[lvl].shape[1]
    up = jax.image.resize(cur, (th, tw, cur.shape[-1]), method="linear")
    wl = pyr_w[lvl]
    known_avg = pyr_v[lvl] / jnp.maximum(wl, 1e-5)
    cur = jnp.where(
        wl > 0.999,
        known_avg,
        jnp.where(wl > 1e-4, 0.5 * known_avg + 0.5 * up, up),
    )
    steps_lvl = int(n_iters) if lvl == 0 else 4
    for _ in range(steps_lvl):
      p = jnp.pad(cur, ((1, 1), (1, 1), (0, 0)), mode="edge")
      avg = (
          p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
      ) * 0.25
      cur = jnp.where(wl > 0.999, known_avg, avg)
  return cur


_TARGET_FEATS_CACHE: dict[tuple[Any, ...], list[jax.Array]] = {}


def _get_cached_target_feats(
    target_hwc: jax.Array, vgg_params: Any, eff_scales: int
) -> list[jax.Array]:
  h, w = int(target_hwc.shape[0]), int(target_hwc.shape[1])
  max_s = max(int(eff_scales), int(select_num_scales(h, w, max_scales=3)))
  key = (
      h,
      w,
      max_s,
      round(float(target_hwc[0, 0, 0]), 6),
      round(float(target_hwc[h // 2, w // 2, 1]), 6),
      round(float(target_hwc[-1, -1, 2]), 6),
  )
  if key not in _TARGET_FEATS_CACHE:
    _TARGET_FEATS_CACHE.clear()
    _TARGET_FEATS_CACHE[key] = _jit_extract_vgg_features(
        vgg_params, jnp.moveaxis(target_hwc, -1, 0), eff_scales=max_s
    )
  return list(_TARGET_FEATS_CACHE[key][: 1 + 5 * int(eff_scales)])


@functools.lru_cache(maxsize=8)
def _get_compiled_wd_val_and_grad(num_levels: int = 2, num_scales: int = 3):
  def _fn(
      rec_hwc: jax.Array,
      target_hwc: jax.Array,
      vgg_params: Any,
      log2_sigma: float = 1.0,
      num_levels: int = num_levels,
      num_scales: int = num_scales,
  ) -> tuple[jax.Array, jax.Array]:
    h, w = int(rec_hwc.shape[0]), int(rec_hwc.shape[1])
    eff_scales = int(select_num_scales(h, w, max_scales=int(num_scales)))
    t_feats = _get_cached_target_feats(target_hwc, vgg_params, eff_scales)
    ls_map = jnp.full((h, w), float(log2_sigma), dtype=jnp.float32)
    wd_v, g_chw = _jit_wd_val_and_grad_chw(
        vgg_params,
        jnp.moveaxis(rec_hwc, -1, 0),
        t_feats,
        ls_map,
        eff_scales,
        int(num_levels),
    )
    return wd_v, jnp.moveaxis(g_chw, 0, -1)

  return _fn


@functools.lru_cache(maxsize=8)
def _get_compiled_wd_only(num_levels: int = 2, num_scales: int = 3):
  def _fn(
      rec_hwc: jax.Array,
      target_hwc: jax.Array,
      vgg_params: Any,
      log2_sigma: float = 1.0,
      num_levels: int = num_levels,
      num_scales: int = num_scales,
  ) -> jax.Array:
    h, w = int(rec_hwc.shape[0]), int(rec_hwc.shape[1])
    eff_scales = int(select_num_scales(h, w, max_scales=int(num_scales)))
    t_feats = _get_cached_target_feats(target_hwc, vgg_params, eff_scales)
    ls_map = jnp.full((h, w), float(log2_sigma), dtype=jnp.float32)
    return _jit_wd_eval_chw(
        vgg_params,
        jnp.moveaxis(rec_hwc, -1, 0),
        t_feats,
        ls_map,
        eff_scales,
        int(num_levels),
    )

  return _fn


@functools.lru_cache(maxsize=8)
def _get_compiled_spatial_wd_map(
    block_size: int = 8, num_levels: int = 2, num_scales: int = 3
):
  from jxl_wasserstein.optimizer import _jit_spatial_wd_map_chw

  def _fn(
      rec_hwc: jax.Array,
      target_hwc: jax.Array,
      vgg_params: Any,
      block_size: int = block_size,
      log2_sigma: float = 1.0,
      num_levels: int = num_levels,
      num_scales: int = num_scales,
  ) -> tuple[jax.Array, jax.Array]:
    h, w = int(rec_hwc.shape[0]), int(rec_hwc.shape[1])
    if int(block_size) != 8:
      return compute_spatial_wd_map(
          rec_hwc,
          target_hwc,
          vgg_params,
          block_size=int(block_size),
          log2_sigma=float(log2_sigma),
          num_levels=int(num_levels),
          num_scales=int(num_scales),
      )
    h8, w8 = (h + 7) // 8, (w + 7) // 8
    eff_scales = int(select_num_scales(h, w, max_scales=int(num_scales)))
    t_feats = _get_cached_target_feats(target_hwc, vgg_params, eff_scales)
    ls_map = jnp.full((h, w), float(log2_sigma), dtype=jnp.float32)
    return _jit_spatial_wd_map_chw(
        vgg_params,
        jnp.moveaxis(rec_hwc, -1, 0),
        t_feats,
        ls_map,
        h8,
        w8,
        eff_scales,
        int(num_levels),
    )

  return _fn


@jax.jit
def _local_patch_wd_map_8x8_jax(
    vgg_params: Any,
    rec_hwc: jax.Array,
    t_feats_local: list[jax.Array],
    log2_sigma: float,
) -> jax.Array:
  """Localized 8x8-block WD map on `RGB + conv1_2 + conv2_2` (receptive field <= 16 px)."""
  h, w = int(rec_hwc.shape[0]), int(rec_hwc.shape[1])
  h8, w8 = (h + 7) // 8, (w + 7) // 8
  r_feats_local = multiscale_vgg_features(
      vgg_params, jnp.moveaxis(rec_hwc, -1, 0), num_scales=1
  )[:3]
  ls_map = jnp.full((h, w), jnp.float32(log2_sigma), dtype=jnp.float32)
  _, inter = multi_wasserstein_distortion(
      r_feats_local,
      t_feats_local,
      ls_map,
      num_levels=2,
      return_intermediates=True,
  )
  wd_blk = jnp.zeros((h8, w8), dtype=jnp.float32)
  n_blk = float(max(1, h8 * w8))
  for wms, wts in zip(inter["wd_maps"], inter["weights"]):
    for wm, wt in zip(wms, wts):
      s2d = wt * jnp.mean(wm, axis=0)
      ts = jnp.mean(s2d)
      rz = jax.image.resize(s2d, (h8, w8), method="linear", antialias=True)
      rz_p = jnp.maximum(rz, 0.0)
      rz_s = jnp.sum(rz_p)
      ns = jnp.where(rz_s > 1e-12, ts / jnp.maximum(rz_s, 1e-12), 1.0 / n_blk)
      to = jnp.where(rz_s > 1e-12, 0.0, ts / n_blk)
      wd_blk = wd_blk + rz_p * ns + to
  return wd_blk


def _precompute_dq_geometry(
    used_s: Sequence[int], default_dq: dict[int, np.ndarray]
) -> dict[int, dict[str, Any]]:
  """Precomputes radial coordinates, 8-band interpolation indices, and LF corner masks per AcStrategy."""
  geom: dict[int, dict[str, Any]] = {}
  for s in used_s:
    s_int = int(s)
    wrows, wcols = default_dq[s_int].shape[1], default_dq[s_int].shape[2]
    cy = int(ACS_COVERED_BLOCKS_Y[s_int])
    cx = int(ACS_COVERED_BLOCKS_X[s_int])
    cy_l, cx_l = min(cy, cx), max(cy, cx)
    yy = np.arange(wrows, dtype=np.float32)[:, None] / max(1.0, float(wrows - 1))
    xx = np.arange(wcols, dtype=np.float32)[None, :] / max(1.0, float(wcols - 1))
    r_norm = (np.sqrt(yy * yy + xx * xx) / np.float32(np.sqrt(2.0))).astype(
        np.float32
    )
    u_rad = (7.0 * np.log2(1.0 + 3.0 * r_norm) / 2.0).astype(np.float32)
    k0 = np.clip(np.floor(u_rad).astype(np.int32), 0, 6)
    alpha_rad = (u_rad - k0.astype(np.float32)).astype(np.float32)

    yi = np.arange(wrows, dtype=np.int32)[:, None]
    xi = np.arange(wcols, dtype=np.int32)[None, :]
    non_llf = ~((yi < cy_l) & (xi < cx_l))
    corner_idx: tuple[int, int] | None = None
    if s_int == 0:
      cmask = (non_llf & ((yi + xi) <= 2)).astype(np.float32)
      corner_idx = (27, 28)
    elif s_int in (4, 6, 7):
      cmask = (non_llf & (yi < 4) & (xi < 4)).astype(np.float32)
      corner_idx = (29, 30)
    elif s_int in (5, 8, 9, 10, 11):
      cmask = (non_llf & (yi < 6) & (xi < 6)).astype(np.float32)
      corner_idx = (31, 32)
    else:
      cmask = np.zeros((wrows, wcols), dtype=np.float32)

    geom[s_int] = {
        "r_norm": r_norm,
        "k0": k0,
        "alpha_rad": alpha_rad,
        "non_llf": non_llf.astype(np.float32),
        "corner_mask": cmask,
        "corner_idx": corner_idx,
    }
  return geom


_RADIAL_DCT_STRATEGIES: tuple[int, ...] = (0, 4, 5, 6, 7, 8, 9, 10, 11)


def _eval_dq_log_vec_np(
    dq_log_vec: np.ndarray | jax.Array | None,
    used_s: Sequence[int],
    default_dq: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
  """Evaluates `'none'` (0 DoF), `'compact'` (9 DoF), or `'full_bands'` (33 DoF) dequant tables in NumPy."""
  if default_dq is None:
    default_dq = get_default_dequant_matrices()
  if dq_log_vec is None:
    return (
        {int(s): np.asarray(default_dq[int(s)], dtype=np.float32).copy() for s in used_s},
        np.array(DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32),
    )
  dq_log = np.asarray(dq_log_vec, dtype=np.float32).ravel()
  n_dof = int(dq_log.size)
  if n_dof == 0:
    return (
        {int(s): np.asarray(default_dq[int(s)], dtype=np.float32).copy() for s in used_s},
        np.array(DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32),
    )

  geom = _precompute_dq_geometry(used_s, default_dq)
  scales = np.clip(np.exp(dq_log[0:3]), 0.65, 1.55).astype(np.float32)
  dc_scales = np.clip(np.exp(0.5 * dq_log[0:3]), 0.75, 1.35).astype(np.float32)
  dc_w = (np.asarray(DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32) * dc_scales).astype(
      np.float32
  )
  mats: dict[int, np.ndarray] = {}

  if n_dof <= 9:
    freq_tilt = (
        np.clip(dq_log[3:6], -0.25, 0.25).astype(np.float32)
        if n_dof >= 6
        else np.zeros((3,), dtype=np.float32)
    )
    hf_taper = (
        np.clip(dq_log[6:9], -0.30, 0.30).astype(np.float32)
        if n_dof >= 9
        else np.zeros((3,), dtype=np.float32)
    )
    for s in used_s:
      s_int = int(s)
      if s_int in _RADIAL_DCT_STRATEGIES:
        rn = geom[s_int]["r_norm"][None, :, :]
        log_mul = freq_tilt[:, None, None] * (rn - 0.35) + hf_taper[
            :, None, None
        ] * np.square(np.maximum(rn - 0.5, 0.0))
        mul = np.exp(log_mul).astype(np.float32)
      else:
        mul = np.float32(1.0)
      mats[s_int] = (default_dq[s_int] * scales[:, None, None] * mul).astype(
          np.float32
      )
    return mats, dc_w

  bands_log = np.clip(dq_log[3:27].reshape(3, 8), -0.35, 0.35).astype(np.float32)
  corners_log = (
      np.clip(dq_log[27:33], -0.035, 0.035).astype(np.float32)
      if n_dof >= 33
      else np.zeros((6,), dtype=np.float32)
  )
  for s in used_s:
    s_int = int(s)
    if s_int not in _RADIAL_DCT_STRATEGIES:
      mats[s_int] = (default_dq[s_int] * scales[:, None, None]).astype(np.float32)
      continue
    g = geom[s_int]
    k0 = g["k0"]
    a_r = g["alpha_rad"][None, :, :]
    band_l = (1.0 - a_r) * bands_log[:, k0] + a_r * bands_log[:, k0 + 1]
    c_idx = g["corner_idx"]
    if c_idx is not None:
      iy, ixb = c_idx[0] - 27, c_idx[1] - 27
      c_ch = np.array(
          [corners_log[ixb], corners_log[iy], corners_log[ixb]], dtype=np.float32
      )
      band_l = band_l + c_ch[:, None, None] * g["corner_mask"][None, :, :]
    mats[s_int] = (
        default_dq[s_int] * scales[:, None, None] * np.exp(band_l)
    ).astype(np.float32)
  return mats, dc_w


_FRAME_KERNEL_CACHE: dict[tuple[Any, ...], tuple[Any, Any, Any, tuple[int, ...]]] = {}


def _resolve_frame_kernel_entry(state: JxlVarDCTState, enable_noise: bool):
  """Returns `(raw_forward_xyb_fn, raw_vjp_fn, raw_hard_xyb_fn, canon_s, use_traced_acs, acs_np)`."""
  h_int, w_int = int(state.height), int(state.width)
  h8_int, w8_int = int(state.h8), int(state.w8)
  acs_np = np.asarray(state.ac_strategy, dtype=np.int32)
  all_zero = bool(np.all(acs_np == 0))
  used_unique = tuple(sorted(set(int(x) for x in np.unique(acs_np)) | {0}))
  # Check if multi-block anchors in acs_np are cell-aligned (required for traced acs_grid)
  traced_ok = True
  for s_u in used_unique:
    cy_u, cx_u = int(ACS_COVERED_BLOCKS_Y[s_u]), int(ACS_COVERED_BLOCKS_X[s_u])
    if cy_u > 1 or cx_u > 1:
      anc_u = np.asarray(state.ac_strategy_Anchor, dtype=bool) & (acs_np == s_u)
      bys_u, bxs_u = np.nonzero(anc_u)
      if bys_u.size > 0 and (
          not np.all((bys_u % cy_u) == 0)
          or not np.all((bxs_u % cx_u) == 0)
          or not np.all((bys_u + cy_u) <= (h8_int // cy_u) * cy_u)
          or not np.all((bxs_u + cx_u) <= (w8_int // cx_u) * cx_u)
      ):
        traced_ok = False
        break

  default_cands = tuple(
      int(s)
      for s in DEFAULT_CANDIDATE_AC_STRATEGIES
      if h8_int >= ACS_COVERED_BLOCKS_Y[s] and w8_int >= ACS_COVERED_BLOCKS_X[s]
  )
  if all_zero:
    canon_s: tuple[int, ...] = (0,)
    use_traced_acs = True
  elif traced_ok and set(used_unique).issubset(set(default_cands)):
    canon_s = default_cands
    use_traced_acs = True
  elif traced_ok:
    canon_s = used_unique
    use_traced_acs = True
  else:
    canon_s = used_unique
    use_traced_acs = False

  if use_traced_acs:
    key: tuple[Any, ...] = (
        h_int,
        w_int,
        int(state.epf_iters),
        bool(state.gab),
        bool(enable_noise),
        canon_s,
    )
  else:
    key = (
        h_int,
        w_int,
        int(state.epf_iters),
        bool(state.gab),
        bool(enable_noise),
        acs_np.tobytes(),
    )

  if key not in _FRAME_KERNEL_CACHE:
    while len(_FRAME_KERNEL_CACHE) >= 8:
      del _FRAME_KERNEL_CACHE[next(iter(_FRAME_KERNEL_CACHE))]
    template_state = copy.deepcopy(state)
    if use_traced_acs:
      template_state.ac_strategy = np.zeros((h8_int, w8_int), dtype=np.int32)
      template_state.ac_strategy_Anchor = np.ones((h8_int, w8_int), dtype=bool)
    template_state.dequant_matrices = get_default_dequant_matrices()
    template_state.dc_quant_weights = np.array(
        DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32
    )
    topo = StaticVarDCTTopology(
        template_state,
        force_strategies=canon_s if use_traced_acs else None,
    )
    default_dq = get_default_dequant_matrices()
    used_s = list(canon_s)
    geom_np = _precompute_dq_geometry(used_s, default_dq)
    geom_j = {
        s: {
            "dq0": jnp.asarray(default_dq[s], dtype=jnp.float32),
            "r_norm": jnp.asarray(geom_np[s]["r_norm"], dtype=jnp.float32),
            "k0": jnp.asarray(geom_np[s]["k0"], dtype=jnp.int32),
            "alpha_rad": jnp.asarray(geom_np[s]["alpha_rad"], dtype=jnp.float32),
            "non_llf": jnp.asarray(geom_np[s]["non_llf"], dtype=jnp.float32),
            "corner_mask": jnp.asarray(
                geom_np[s]["corner_mask"], dtype=jnp.float32
            ),
            "corner_idx": geom_np[s]["corner_idx"],
        }
        for s in used_s
    }
    dc_w_def_j = jnp.asarray(DEFAULT_DC_QUANT_WEIGHTS, dtype=jnp.float32)

    def _build_dq_mats_and_dc_w(dq_log_vec: jax.Array | None):
      if dq_log_vec is None or int(dq_log_vec.shape[0]) == 0:
        dq_mats_def = {s: geom_j[s]["dq0"] for s in used_s}
        ones3 = jnp.ones((3,), dtype=jnp.float32)
        return dq_mats_def, dc_w_def_j, ones3, ones3
      n_dof = int(dq_log_vec.shape[0])
      scales = jnp.clip(jnp.exp(dq_log_vec[0:3]), 0.65, 1.55)
      dc_scales = jnp.clip(jnp.exp(0.5 * dq_log_vec[0:3]), 0.75, 1.35)
      dc_w = dc_w_def_j * dc_scales
      dq_mats = {}
      if n_dof <= 9:
        freq_tilt = (
            jnp.clip(dq_log_vec[3:6], -0.25, 0.25)
            if n_dof >= 6
            else jnp.zeros((3,), dtype=jnp.float32)
        )
        hf_taper = (
            jnp.clip(dq_log_vec[6:9], -0.30, 0.30)
            if n_dof >= 9
            else jnp.zeros((3,), dtype=jnp.float32)
        )
        for s in used_s:
          if s in _RADIAL_DCT_STRATEGIES:
            rn = geom_j[s]["r_norm"][None, :, :]
            log_mul = freq_tilt[:, None, None] * (rn - 0.35) + hf_taper[
                :, None, None
            ] * jnp.square(jnp.maximum(rn - 0.5, 0.0))
            dq_mats[s] = geom_j[s]["dq0"] * scales[:, None, None] * jnp.exp(log_mul)
          else:
            dq_mats[s] = geom_j[s]["dq0"] * scales[:, None, None]
        return dq_mats, dc_w, scales, dc_scales

      bands_log = jnp.clip(dq_log_vec[3:27].reshape(3, 8), -0.35, 0.35)
      corners_log = (
          jnp.clip(dq_log_vec[27:33], -0.035, 0.035)
          if n_dof >= 33
          else jnp.zeros((6,), dtype=jnp.float32)
      )
      for s in used_s:
        if s not in _RADIAL_DCT_STRATEGIES:
          dq_mats[s] = geom_j[s]["dq0"] * scales[:, None, None]
          continue
        g_s = geom_j[s]
        k0 = g_s["k0"]
        a_r = g_s["alpha_rad"][None, :, :]
        band_l = (1.0 - a_r) * bands_log[:, k0] + a_r * bands_log[:, k0 + 1]
        c_idx = g_s["corner_idx"]
        if c_idx is not None:
          iy, ixb = c_idx[0] - 27, c_idx[1] - 27
          c_ch = jnp.stack(
              [corners_log[ixb], corners_log[iy], corners_log[ixb]], axis=0
          )
          band_l = band_l + c_ch[:, None, None] * g_s["corner_mask"][None, :, :]
        dq_mats[s] = g_s["dq0"] * scales[:, None, None] * jnp.exp(band_l)
      return dq_mats, dc_w, scales, dc_scales

    def _expand_dq_mats_and_non_llf(
        dq_mats: dict[int, jax.Array], acs_g: jax.Array | None
    ) -> tuple[jax.Array, jax.Array]:
      h8_t, w8_t = topo.h8, topo.w8
      w_blk = jnp.broadcast_to(
          dq_mats[0][None, None], (h8_t, w8_t, 3, 8, 8)
      )
      non_llf_blk = jnp.broadcast_to(
          geom_j[0]["non_llf"][None, None], (h8_t, w8_t, 8, 8)
      )
      for grp1 in topo.groups_1x1:
        s_id = int(grp1["s"])
        if s_id == 0:
          continue
        m2d = (acs_g == s_id) if acs_g is not None else grp1["mask_2d"]
        w_blk = jnp.where(
            m2d[:, :, None, None, None], dq_mats[s_id][None, None], w_blk
        )
      for grp in topo.multi_groups:
        s_id = int(grp["s"])
        cy, cx = int(grp["cy"]), int(grp["cx"])
        dq_s = dq_mats[s_id].reshape(3, cy, 8, cx, 8).transpose(1, 3, 0, 2, 4)
        nllf_s = geom_j[s_id]["non_llf"].reshape(cy, 8, cx, 8).transpose(
            0, 2, 1, 3
        )
        if grp.get("grid_aligned", False):
          nh_c, nw_c = int(grp["nh_cell"]), int(grp["nw_cell"])
          h_l, w_l = int(grp["h_lim"]), int(grp["w_lim"])
          if acs_g is not None:
            cm = acs_g[:h_l:cy, :w_l:cx] == s_id
            bm = jnp.broadcast_to(
                cm[:, None, :, None], (nh_c, cy, nw_c, cx)
            ).reshape(h_l, w_l)
          else:
            bm = grp["blk_mask"]
          dq_tile = jnp.broadcast_to(
              dq_s[None, :, None, :, :, :, :], (nh_c, cy, nw_c, cx, 3, 8, 8)
          ).reshape(h_l, w_l, 3, 8, 8)
          nllf_tile = jnp.broadcast_to(
              nllf_s[None, :, None, :, :, :], (nh_c, cy, nw_c, cx, 8, 8)
          ).reshape(h_l, w_l, 8, 8)
          sub_w = jnp.where(
              bm[:, :, None, None, None], dq_tile, w_blk[:h_l, :w_l]
          )
          sub_m = jnp.where(bm[:, :, None, None], nllf_tile, non_llf_blk[:h_l, :w_l])
          if h_l == h8_t and w_l == w8_t:
            w_blk = sub_w
            non_llf_blk = sub_m
          else:
            w_blk = w_blk.at[:h_l, :w_l].set(sub_w)
            non_llf_blk = non_llf_blk.at[:h_l, :w_l].set(sub_m)
        else:
          by, bx = grp["by"], grp["bx"]
          m_cnt = int(by.shape[0])
          if m_cnt > 0:
            dy = jnp.arange(cy)[None, :, None]
            dx = jnp.arange(cx)[None, None, :]
            w_blk = w_blk.at[
                by[:, None, None] + dy, bx[:, None, None] + dx
            ].set(jnp.broadcast_to(dq_s[None], (m_cnt, cy, cx, 3, 8, 8)))
            non_llf_blk = non_llf_blk.at[
                by[:, None, None] + dy, bx[:, None, None] + dx
            ].set(jnp.broadcast_to(nllf_s[None], (m_cnt, cy, cx, 8, 8)))
      w_3hw = jnp.transpose(w_blk, (2, 0, 3, 1, 4)).reshape(
          3, h8_t * 8, w8_t * 8
      )
      non_llf_hw = jnp.transpose(non_llf_blk, (0, 2, 1, 3)).reshape(
          h8_t * 8, w8_t * 8
      )
      return w_3hw, non_llf_hw

    def _couple_ac_dc(
        ac: jax.Array,
        dc: jax.Array,
        rq: jax.Array,
        rq_base: jax.Array | None,
        dq_mats_cur: dict[int, jax.Array],
        dc_w_cur: jax.Array,
        dq_log_base: jax.Array | None,
        x_qm: jax.Array,
        b_qm: jax.Array,
        yx: jax.Array | None = None,
        yb: jax.Array | None = None,
        yx_base: jax.Array | None = None,
        yb_base: jax.Array | None = None,
        acs_g: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
      h8_t, w8_t = topo.h8, topo.w8
      if rq_base is not None:
        rq_tl = topo.broadcast_top_left_grid(
            jnp.clip(rq, 1.0, 256.0), acs_grid=acs_g
        )
        rq_b_tl = topo.broadcast_top_left_grid(
            jnp.clip(rq_base, 1.0, 256.0), acs_grid=acs_g
        )
        rq_ratio_tl = rq_tl / jnp.maximum(rq_b_tl, 1.0)
        rq_ratio_px = jnp.broadcast_to(
            rq_ratio_tl[:, None, :, None], (h8_t, 8, w8_t, 8)
        ).reshape(1, h8_t * 8, w8_t * 8)
      else:
        rq_ratio_px = jnp.float32(1.0)

      w_cur, non_llf_hw = _expand_dq_mats_and_non_llf(dq_mats_cur, acs_g)
      if dq_log_base is not None and int(dq_log_base.shape[0]) > 0:
        dq_mats_b, dc_w_b, _, _ = _build_dq_mats_and_dc_w(dq_log_base)
        w_base, _ = _expand_dq_mats_and_non_llf(dq_mats_b, acs_g)
        u_ac = ac * rq_ratio_px * (w_base / jnp.maximum(w_cur, 1e-12))
        u_dc = dc * (dc_w_b / jnp.maximum(dc_w_cur, 1e-12))[:, None, None]
      else:
        u_ac = ac * rq_ratio_px
        u_dc = dc

      if (
          yx is not None
          and yb is not None
          and yx_base is not None
          and yb_base is not None
      ):
        ht_t, wt_t = topo.ht, topo.wt
        cf_f = jnp.float32(topo.color_factor)
        dx_tile = (
            jnp.clip(yx, -128.0, 127.0) - jnp.clip(yx_base, -128.0, 127.0)
        ) / cf_f
        db_tile = (
            jnp.clip(yb, -128.0, 127.0) - jnp.clip(yb_base, -128.0, 127.0)
        ) / cf_f
        dx_px = jnp.broadcast_to(
            dx_tile[:, None, :, None], (ht_t, 64, wt_t, 64)
        ).reshape(ht_t * 64, wt_t * 64)[: h8_t * 8, : w8_t * 8]
        db_px = jnp.broadcast_to(
            db_tile[:, None, :, None], (ht_t, 64, wt_t, 64)
        ).reshape(ht_t * 64, wt_t * 64)[: h8_t * 8, : w8_t * 8]
        x_qm_mul = jnp.power(jnp.float32(1.25), x_qm - 2.0)
        b_qm_mul = jnp.power(jnp.float32(1.25), b_qm - 2.0)
        u_y = u_ac[1] * non_llf_hw
        u_x = u_ac[0] - dx_px * u_y * (
            (w_cur[1] * x_qm_mul) / jnp.maximum(w_cur[0], 1e-12)
        )
        u_b = u_ac[2] - db_px * u_y * (
            (w_cur[1] * b_qm_mul) / jnp.maximum(w_cur[2], 1e-12)
        )
        u_ac = jnp.stack([u_x, u_ac[1], u_b], axis=0)
      return u_ac, u_dc

    @jax.jit
    def _raw_forward_xyb_fn(
        acs_g,
        ac,
        dc,
        rq,
        yx,
        yb,
        gs,
        qdc,
        x_qm,
        b_qm,
        ytox_dc,
        ytob_dc,
        epf_sharp,
        noise_lut,
        dq_log_scale=None,
        rq_base=None,
        dq_log_base=None,
        yx_base=None,
        yb_base=None,
    ):
      dq_mats, dc_w, _, _ = _build_dq_mats_and_dc_w(dq_log_scale)
      u_ac, u_dc = _couple_ac_dc(
          ac,
          dc,
          rq,
          rq_base,
          dq_mats,
          dc_w,
          dq_log_base,
          x_qm,
          b_qm,
          yx=yx,
          yb=yb,
          yx_base=yx_base,
          yb_base=yb_base,
          acs_g=acs_g,
      )
      return reconstruct_single_frame_xyb(
          topo,
          params_override={
              "global_scale": gs,
              "quant_dc": qdc,
              "x_qm_scale": x_qm,
              "b_qm_scale": b_qm,
              "ytox_dc": ytox_dc,
              "ytob_dc": ytob_dc,
              "epf_sharpness": epf_sharp,
          },
          quant_ac=u_ac,
          dc_quant=u_dc,
          raw_quant_field=jnp.clip(rq, 1.0, 256.0),
          ytox_map=jnp.clip(yx, -128.0, 127.0),
          ytob_map=jnp.clip(yb, -128.0, 127.0),
          dequant_matrices=dq_mats if dq_log_scale is not None else None,
          dc_quant_weights=dc_w if dq_log_scale is not None else None,
          hard=False,
          include_noise=enable_noise,
          noise_lut=noise_lut if enable_noise else None,
          acs_grid=acs_g,
      )

    @jax.jit
    def _raw_hard_xyb_fn(
        acs_g,
        ac,
        dc,
        rq,
        yx,
        yb,
        gs,
        qdc,
        x_qm,
        b_qm,
        ytox_dc,
        ytob_dc,
        epf_sharp,
        noise_lut,
        dq_mats,
        dc_w,
    ):
      return reconstruct_single_frame_xyb(
          topo,
          params_override={
              "global_scale": gs,
              "quant_dc": qdc,
              "x_qm_scale": x_qm,
              "b_qm_scale": b_qm,
              "ytox_dc": ytox_dc,
              "ytob_dc": ytob_dc,
              "epf_sharpness": epf_sharp,
          },
          quant_ac=ac,
          dc_quant=dc,
          raw_quant_field=jnp.clip(rq, 1.0, 256.0),
          ytox_map=jnp.clip(yx, -128.0, 127.0),
          ytob_map=jnp.clip(yb, -128.0, 127.0),
          dequant_matrices=dq_mats,
          dc_quant_weights=dc_w,
          hard=True,
          include_noise=enable_noise,
          noise_lut=noise_lut if enable_noise else None,
          acs_grid=acs_g,
      )

    def frame_surrogate(
        p,
        acs_g,
        rq_base,
        yx_base,
        yb_base,
        gs,
        qdc,
        x_qm,
        b_qm,
        ytox_dc,
        ytob_dc,
        epf_sharp,
        ac_mask,
        dc_mask,
        dc_bg,
        cotangent_xyb,
        lam_per_px,
        custom_dequant_weight,
        dq_log_base=None,
        ac_rate_weight=1.0,
        cluster_blend=0.0,
    ):
      ac = p["ac"] * ac_mask
      dc = p["dc"] * dc_mask + dc_bg
      n_lut = jnp.clip(p["noise_lut"], 0.0, 0.5)
      rq = jnp.clip(rq_base + p.get("rq_delta", 0.0), 1.0, 256.0)
      yx = jnp.clip(yx_base + p.get("yx_delta", 0.0), -128.0, 127.0)
      yb = jnp.clip(yb_base + p.get("yb_delta", 0.0), -128.0, 127.0)
      dq_log = p.get("dq_log_scale", None)
      has_dq = dq_log is not None and int(dq_log.shape[0]) > 0

      @functools.partial(jax.checkpoint, prevent_cse=False)
      def _recon_dot_fn(ac_in, dc_in, rq_in, yx_in, yb_in, n_lut_in, dq_log_in):
        dq_mats_c, dc_w_c, _, _ = _build_dq_mats_and_dc_w(dq_log_in)
        u_ac_c, u_dc_c = _couple_ac_dc(
            ac_in,
            dc_in,
            rq_in,
            rq_base,
            dq_mats_c,
            dc_w_c,
            dq_log_base,
            x_qm,
            b_qm,
            yx=yx_in,
            yb=yb_in,
            yx_base=yx_base,
            yb_base=yb_base,
            acs_g=acs_g,
        )
        xyb_c = reconstruct_single_frame_xyb(
            topo,
            params_override={
                "global_scale": gs,
                "quant_dc": qdc,
                "x_qm_scale": x_qm,
                "b_qm_scale": b_qm,
                "ytox_dc": ytox_dc,
                "ytob_dc": ytob_dc,
                "epf_sharpness": epf_sharp,
            },
            quant_ac=u_ac_c,
            dc_quant=u_dc_c,
            raw_quant_field=rq_in,
            ytox_map=yx_in,
            ytob_map=yb_in,
            dequant_matrices=dq_mats_c if has_dq else None,
            dc_quant_weights=dc_w_c if has_dq else None,
            hard=False,
            include_noise=enable_noise,
            noise_lut=n_lut_in if enable_noise else None,
            acs_grid=acs_g,
        )
        return jnp.sum(xyb_c * cotangent_xyb)

      @functools.partial(jax.checkpoint, prevent_cse=False)
      def _rate_bits_fn(ac_in, dc_in, rq_in, yx_in, yb_in, n_lut_in, dq_log_in):
        dq_mats_c, dc_w_c, _, _ = _build_dq_mats_and_dc_w(dq_log_in)
        ac_r = ac_in * ac_rate_weight + jax.lax.stop_gradient(ac_in) * (
            1.0 - ac_rate_weight
        )
        dc_r = dc_in * ac_rate_weight + jax.lax.stop_gradient(dc_in) * (
            1.0 - ac_rate_weight
        )
        u_ac_r, u_dc_r = _couple_ac_dc(
            ac_r,
            dc_r,
            rq_in,
            rq_base,
            dq_mats_c,
            dc_w_c,
            dq_log_base,
            x_qm,
            b_qm,
            yx=yx_in,
            yb=yb_in,
            yx_base=yx_base,
            yb_base=yb_base,
            acs_g=acs_g,
        )
        bits_c = soft_jxl_rate_proxy(
            template_state,
            quant_ac=u_ac_r,
            dc_quant=u_dc_r,
            raw_quant_field=rq_in,
            ytox_map=yx_in,
            ytob_map=yb_in,
            ytox_dc=ytox_dc,
            ytob_dc=ytob_dc,
            epf_sharpness=epf_sharp,
            global_scale=gs,
            quant_dc=qdc,
            enable_noise=enable_noise,
            noise_lut=n_lut_in if enable_noise else None,
            ac_strategy=acs_g,
            topology=topo,
            cluster_blend=cluster_blend,
        )
        dq_bits_c = (
            custom_dequant_weight
            * soft_dequant_matrices_rate_bits(
                dq_mats_c, dc_quant=dc_w_c, ac_strategy=acs_g
            )
            if has_dq
            else jnp.float32(1.0)
        )
        return bits_c + dq_bits_c

      recon_dot = _recon_dot_fn(ac, dc, rq, yx, yb, n_lut, dq_log)
      total_bits = _rate_bits_fn(ac, dc, rq, yx, yb, n_lut, dq_log)
      return recon_dot + lam_per_px * total_bits, total_bits

    _raw_vjp_fn = jax.jit(jax.value_and_grad(frame_surrogate, has_aux=True))
    _FRAME_KERNEL_CACHE[key] = (
        _raw_forward_xyb_fn,
        _raw_vjp_fn,
        _raw_hard_xyb_fn,
        canon_s,
    )

  _raw_forward_xyb_fn, _raw_vjp_fn, _raw_hard_xyb_fn, canon_s = (
      _FRAME_KERNEL_CACHE[key]
  )
  return (
      _raw_forward_xyb_fn,
      _raw_vjp_fn,
      _raw_hard_xyb_fn,
      canon_s,
      use_traced_acs,
      acs_np,
  )


def _get_frame_kernels(state: JxlVarDCTState, enable_noise: bool):
  """Returns cached `(forward_xyb_fn, vjp_and_rate_grad_fn)` compiled per resolution & canonical strategy set."""
  (
      _raw_forward_xyb_fn,
      _raw_vjp_fn,
      _,
      _,
      use_traced_acs,
      acs_np,
  ) = _resolve_frame_kernel_entry(state, enable_noise)
  acs_arg = jnp.asarray(acs_np, dtype=jnp.int32) if use_traced_acs else None
  x_qm_arg = jnp.float32(state.x_qm_scale)
  b_qm_arg = jnp.float32(state.b_qm_scale)
  ytox_dc_arg = jnp.float32(state.ytox_dc)
  ytob_dc_arg = jnp.float32(state.ytob_dc)
  epf_sharp_arg = jnp.asarray(state.epf_sharpness, dtype=jnp.int32)

  def forward_xyb_fn(
      ac,
      dc,
      rq,
      yx,
      yb,
      gs,
      qdc,
      noise_lut,
      dq_log_scale=None,
      rq_base=None,
      dq_log_base=None,
      yx_base=None,
      yb_base=None,
  ):
    return _raw_forward_xyb_fn(
        acs_arg,
        ac,
        dc,
        rq,
        yx,
        yb,
        gs,
        qdc,
        x_qm_arg,
        b_qm_arg,
        ytox_dc_arg,
        ytob_dc_arg,
        epf_sharp_arg,
        noise_lut,
        dq_log_scale,
        rq_base,
        dq_log_base,
        yx_base,
        yb_base,
    )

  def vjp_and_rate_grad_fn(
      p,
      rq_base,
      yx_base,
      yb_base,
      gs,
      qdc,
      ac_mask,
      dc_mask,
      dc_bg,
      cotangent_xyb,
      lam_per_px,
      custom_dequant_weight,
      dq_log_base=None,
      ac_rate_weight=1.0,
      cluster_blend=0.0,
  ):
    return _raw_vjp_fn(
        p,
        acs_arg,
        rq_base,
        yx_base,
        yb_base,
        gs,
        qdc,
        x_qm_arg,
        b_qm_arg,
        ytox_dc_arg,
        ytob_dc_arg,
        epf_sharp_arg,
        ac_mask,
        dc_mask,
        dc_bg,
        cotangent_xyb,
        lam_per_px,
        custom_dequant_weight,
        dq_log_base,
        ac_rate_weight,
        cluster_blend,
    )

  return forward_xyb_fn, vjp_and_rate_grad_fn


@functools.lru_cache(maxsize=16)
def _get_patch_combine_kernels(h: int, w: int, u: int, enable_noise: bool = False):
  """Returns cached `(combine_fn, combine_grad_fn)` for XYB patch compositing with post-patch XYB noise."""
  dummy_state = init_state_from_rgb_jax(
      np.zeros((h, w, 3), dtype=np.float32), distance=1.0, epf_iters=0
  )
  topo = StaticVarDCTTopology(dummy_state)
  inv_mat = topo.opsin_inv_matrix
  biases = topo.opsin_biases
  ytox = float(topo.base_correlation_x)
  ytob = float(topo.base_correlation_b)

  def _combine(ref_xyb, main_xyb, w_ref, w_main, noise_lut):
    ref_ups = upsample_jxl(ref_xyb, u, out_hw=(h, w))
    comp_xyb = w_main * main_xyb + w_ref * ref_ups
    if enable_noise:
      comp_xyb = synthesize_jxl_noise_xyb(
          xyb=comp_xyb,
          noise_lut=jnp.clip(noise_lut, 0.0, 0.5),
          ytox=ytox,
          ytob=ytob,
          visible_frame_index=1,
          nonvisible_frame_index=0,
          ste_quantize_lut=True,
      )
    srgb_chw = opsin_to_srgb(comp_xyb, inv_mat, biases, clamp=True)
    return jnp.moveaxis(srgb_chw, 0, -1)

  combine_fn = jax.jit(_combine)
  combine_grad_fn = jax.jit(
      jax.grad(
          lambda r_xyb, m_xyb, wr, wm, n_lut, cot: jnp.sum(
              _combine(r_xyb, m_xyb, wr, wm, n_lut) * cot
          ),
          argnums=(0, 1, 4),
      )
  )
  return combine_fn, combine_grad_fn


@functools.lru_cache(maxsize=8)
def _get_alpha_combine_kernels(
    h: int,
    w: int,
    u: int,
    d_shift: int,
    alpha_blend_mode: int = PATCH_BLEND_ALPHA_BELOW,
    enable_noise: bool = False,
):
  """Returns cached `(combine_fn, combine_grad_fn)` for XYB `PatchDictionary` alpha blending (R2).

  Blends Frame 0 (`kReferenceOnly`, upsampled in XYB before color transform with
  opaque alpha=1.0) and Frame 1 (`kRegularFrame`, full-res XYB + `ExtraChannel::kAlpha`
  upsampled by `2**dim_shift`) directly in XYB space via `_blend_xyb_pair`,
  followed by optional post-patch XYB noise synthesis and `opsin_to_srgb`.
  """
  dummy_state = init_state_from_rgb_jax(
      np.zeros((h, w, 3), dtype=np.float32), distance=1.0, epf_iters=0
  )
  topo = StaticVarDCTTopology(dummy_state)
  inv_mat = topo.opsin_inv_matrix
  biases = topo.opsin_biases
  ytox = float(topo.base_correlation_x)
  ytob = float(topo.base_correlation_b)
  step_a = 1 << int(d_shift)
  bm = int(alpha_blend_mode)

  def _logit_to_alpha01(logit: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(logit)

  def _combine(ref_xyb, main_xyb, alpha_logit, noise_lut):
    ref_xyb_ups = upsample_jxl(ref_xyb, u, out_hw=(h, w))
    a_01 = _logit_to_alpha01(alpha_logit)
    a_q = ste_quantize_alpha_01(a_01, hard=False)
    if step_a > 1:
      a_ups = upsample_jxl(a_q[None, :, :], step_a, out_hw=(h, w))[0]
    else:
      a_ups = a_q[:h, :w]
    a_3d = jnp.clip(a_ups, 0.0, 1.0)[None, :, :]
    comp_xyb = _blend_xyb_pair(
        cur_xyb=main_xyb,
        ref_xyb=ref_xyb_ups,
        blend_mode=bm,
        alpha_3d=a_3d,
    )
    if enable_noise:
      comp_xyb = synthesize_jxl_noise_xyb(
          xyb=comp_xyb,
          noise_lut=jnp.clip(noise_lut, 0.0, 0.5),
          ytox=ytox,
          ytob=ytob,
          visible_frame_index=1,
          nonvisible_frame_index=0,
          ste_quantize_lut=True,
      )
    srgb_chw = opsin_to_srgb(comp_xyb, inv_mat, biases, clamp=True)
    return jnp.moveaxis(srgb_chw, 0, -1)

  def _combine_surrogate(
      ref_xyb, main_xyb, alpha_logit, noise_lut, cotangent_rec, lam_per_px
  ):
    rec = _combine(ref_xyb, main_xyb, alpha_logit, noise_lut)
    a_01 = _logit_to_alpha01(alpha_logit)
    a_bits = soft_alpha_channel_bits(a_01, dim_shift=d_shift, full_hw=(h, w))
    return jnp.sum(rec * cotangent_rec) + lam_per_px * a_bits, a_bits

  combine_fn = jax.jit(_combine)
  combine_grad_fn = jax.jit(
      jax.value_and_grad(
          _combine_surrogate, argnums=(0, 1, 2, 3), has_aux=True
      )
  )
  return combine_fn, combine_grad_fn


@functools.lru_cache(maxsize=4)
def _get_xyb_to_srgb_kernels():
  """Returns cached `(_xyb_to_srgb_hwc, grad_srgb)` JIT kernels."""
  dummy_state = init_state_from_rgb_jax(
      np.zeros((8, 8, 3), dtype=np.float32), distance=1.0, epf_iters=0
  )
  topo = StaticVarDCTTopology(dummy_state)
  inv_mat, biases = topo.opsin_inv_matrix, topo.opsin_biases

  @jax.jit
  def _xyb_to_srgb_hwc(xyb):
    return jnp.moveaxis(opsin_to_srgb(xyb, inv_mat, biases, clamp=True), 0, -1)

  grad_srgb = jax.jit(
      jax.grad(lambda xyb, cot: jnp.sum(_xyb_to_srgb_hwc(xyb) * cot))
  )
  return _xyb_to_srgb_hwc, grad_srgb


@functools.lru_cache(maxsize=8)
def _get_alpha_hard_combine_kernel(
    h: int,
    w: int,
    u: int,
    d_shift: int,
    alpha_blend_mode: int = PATCH_BLEND_ALPHA_BELOW,
    enable_noise: bool = False,
):
  """Returns cached `@jax.jit` kernel for hard XYB `PatchDictionary` alpha compositing."""
  dummy_state = init_state_from_rgb_jax(
      np.zeros((8, 8, 3), dtype=np.float32), distance=1.0, epf_iters=0
  )
  topo = StaticVarDCTTopology(dummy_state)
  inv_mat = topo.opsin_inv_matrix
  biases = topo.opsin_biases
  ytox = float(topo.base_correlation_x)
  ytob = float(topo.base_correlation_b)
  step_a = 1 << int(d_shift)
  bm = int(alpha_blend_mode)

  @jax.jit
  def _combine_hard(ref_xyb, main_xyb, alpha_01, noise_lut):
    ref_xyb_ups = upsample_jxl(ref_xyb, u, out_hw=(h, w))
    a_q = ste_quantize_alpha_01(alpha_01, hard=True)
    if step_a > 1:
      a_ups = upsample_jxl(a_q[None, :, :], step_a, out_hw=(h, w))[0]
    else:
      a_ups = a_q[:h, :w]
    a_3d = jnp.clip(a_ups, 0.0, 1.0)[None, :, :]
    comp_xyb = _blend_xyb_pair(
        cur_xyb=main_xyb,
        ref_xyb=ref_xyb_ups,
        blend_mode=bm,
        alpha_3d=a_3d,
    )
    if enable_noise:
      comp_xyb = synthesize_jxl_noise_xyb(
          xyb=comp_xyb,
          noise_lut=jnp.clip(noise_lut, 0.0, 0.5),
          ytox=ytox,
          ytob=ytob,
          visible_frame_index=1,
          nonvisible_frame_index=0,
          ste_quantize_lut=True,
      )
    srgb_chw = opsin_to_srgb(comp_xyb, inv_mat, biases, clamp=True)
    return jnp.moveaxis(srgb_chw, 0, -1)

  return _combine_hard


def _fast_reconstruct_xyb(
    state: JxlVarDCTState,
    include_noise: bool = False,
    noise_lut: Any = None,
) -> jax.Array:
  """JIT-compiled `reconstruct_single_frame_xyb(state, hard=True)`."""
  do_noise = bool(include_noise and noise_lut is not None)
  (
      _,
      _,
      _raw_hard_xyb_fn,
      canon_s,
      use_traced_acs,
      acs_np,
  ) = _resolve_frame_kernel_entry(state, do_noise)
  acs_arg = jnp.asarray(acs_np, dtype=jnp.int32) if use_traced_acs else None
  default_dq = get_default_dequant_matrices()
  dq_dict = state.dequant_matrices if state.dequant_matrices else default_dq
  dq_mats_arg = {
      int(s): jnp.asarray(dq_dict.get(int(s), default_dq[int(s)]), dtype=jnp.float32)
      for s in canon_s
  }
  dc_w_arg = jnp.asarray(
      state.dc_quant_weights
      if getattr(state, "dc_quant_weights", None) is not None
      else DEFAULT_DC_QUANT_WEIGHTS,
      dtype=jnp.float32,
  )
  n_lut_arg = (
      jnp.asarray(noise_lut, dtype=jnp.float32)
      if do_noise
      else jnp.zeros((8,), dtype=jnp.float32)
  )
  return _raw_hard_xyb_fn(
      acs_arg,
      jnp.asarray(state.quant_ac, dtype=jnp.float32),
      jnp.asarray(state.dc_quant, dtype=jnp.float32),
      jnp.asarray(state.raw_quant_field, dtype=jnp.float32),
      jnp.asarray(state.ytox_map, dtype=jnp.float32),
      jnp.asarray(state.ytob_map, dtype=jnp.float32),
      jnp.float32(state.global_scale),
      jnp.float32(state.quant_dc),
      jnp.float32(state.x_qm_scale),
      jnp.float32(state.b_qm_scale),
      jnp.float32(state.ytox_dc),
      jnp.float32(state.ytob_dc),
      jnp.asarray(state.epf_sharpness, dtype=jnp.int32),
      n_lut_arg,
      dq_mats_arg,
      dc_w_arg,
  )


def _fast_reconstruct_single_frame(
    state: JxlVarDCTState,
    include_noise: bool = False,
    noise_lut: Any = None,
) -> jax.Array:
  """JIT-compiled `reconstruct_single_frame(state, hard=True)`."""
  xyb = _fast_reconstruct_xyb(
      state, include_noise=include_noise, noise_lut=noise_lut
  )
  xyb_to_srgb, _ = _get_xyb_to_srgb_kernels()
  return xyb_to_srgb(xyb)


def _fast_reconstruct_patch_composite(
    ref_state: JxlVarDCTState,
    main_state: JxlVarDCTState,
    upsampling: int,
    patches: Sequence[PatchEntry],
    include_noise: bool = False,
    noise_lut: Any = None,
    w_ref_main_override: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """JIT-compiled `reconstruct_patch_composite(..., hard=True)`."""
  h, w = int(main_state.height), int(main_state.width)
  u = int(upsampling)
  if w_ref_main_override is not None or all(
      p.ref == 0
      and int(p.x0) == int(p.x)
      and int(p.y0) == int(p.y)
      and int(p.blend_mode) in (PATCH_BLEND_REPLACE, PATCH_BLEND_ADD)
      for p in patches
  ):
    do_noise = bool(include_noise and noise_lut is not None)
    r_xyb = _fast_reconstruct_xyb(ref_state, include_noise=False)
    m_xyb = _fast_reconstruct_xyb(main_state, include_noise=False)
    w_ref, w_main = (
        w_ref_main_override
        if w_ref_main_override is not None
        else _patches_to_weights(patches, h, w)
    )
    combine_fn, _ = _get_patch_combine_kernels(
        h, w, u, enable_noise=do_noise
    )
    n_lut_arg = (
        jnp.asarray(noise_lut, dtype=jnp.float32)
        if do_noise
        else jnp.zeros((8,), dtype=jnp.float32)
    )
    return combine_fn(r_xyb, m_xyb, w_ref, w_main, n_lut_arg)
  return reconstruct_patch_composite(
      ref_state,
      main_state,
      u,
      patches,
      include_noise=include_noise,
      noise_lut=noise_lut,
      hard=True,
  )


def _fast_reconstruct_alpha_composite(
    ref_state: JxlVarDCTState,
    main_state: JxlVarDCTState,
    upsampling: int,
    dim_shift: int,
    alpha_01: jax.Array | np.ndarray,
    include_noise: bool = False,
    noise_lut: Any = None,
    alpha_blend_mode: int = PATCH_BLEND_ALPHA_BELOW,
    patches: Sequence[PatchEntry] | None = None,
    ref_xyb_override: jax.Array | None = None,
    main_xyb_override: jax.Array | None = None,
) -> jax.Array:
  """JIT-compiled `reconstruct_alpha_composite(..., hard=True)`."""
  h, w = int(main_state.height), int(main_state.width)
  u = int(upsampling)
  ds = int(dim_shift)
  bm = int(alpha_blend_mode)
  if patches is None or (
      len(patches) == 1
      and patches[0].ref == 0
      and int(patches[0].x0) == 0
      and int(patches[0].y0) == 0
      and int(patches[0].x) == 0
      and int(patches[0].y) == 0
      and int(patches[0].xsize) == w
      and int(patches[0].ysize) == h
  ):
    if patches is not None and len(patches) == 1:
      bm = int(patches[0].blend_mode)
    do_noise = bool(include_noise and noise_lut is not None)
    r_xyb = (
        ref_xyb_override
        if ref_xyb_override is not None
        else _fast_reconstruct_xyb(ref_state, include_noise=False)
    )
    m_xyb = (
        main_xyb_override
        if main_xyb_override is not None
        else _fast_reconstruct_xyb(main_state, include_noise=False)
    )
    combine_hard = _get_alpha_hard_combine_kernel(
        h, w, u, ds, alpha_blend_mode=bm, enable_noise=do_noise
    )
    n_lut_arg = (
        jnp.asarray(noise_lut, dtype=jnp.float32)
        if do_noise
        else jnp.zeros((8,), dtype=jnp.float32)
    )
    return combine_hard(
        r_xyb, m_xyb, jnp.asarray(alpha_01, dtype=jnp.float32), n_lut_arg
    )
  return reconstruct_alpha_composite(
      ref_state,
      main_state,
      u,
      ds,
      alpha_01,
      include_noise=include_noise,
      noise_lut=noise_lut,
      hard=True,
      alpha_blend_mode=bm,
      patches=patches,
  )


def downsample_rgb_jax(
    srgb_hwc: np.ndarray | jax.Array, upsampling: int
) -> jax.Array:
  """Pure-JAX area-average downsampling of `(H, W, 3)` sRGB `[0, 1]` by factor `U in {1, 2, 4, 8}`."""
  u = int(upsampling)
  arr = jnp.asarray(_normalize_srgb_hwc(srgb_hwc), dtype=jnp.float32)
  if u == 1:
    return arr
  if u not in (2, 4, 8):
    raise ValueError(f"upsampling must be in {{1, 2, 4, 8}}, got {u}")
  h, w, c = arr.shape
  h_low = (h + u - 1) // u
  w_low = (w + u - 1) // u
  pad_h = h_low * u - h
  pad_w = w_low * u - w
  if pad_h > 0 or pad_w > 0:
    arr = jnp.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
  return arr.reshape(h_low, u, w_low, u, c).mean(axis=(1, 3))


@functools.lru_cache(maxsize=16)
def _get_compiled_lowres_upsample_kernels(h: int, w: int, u: int):
  """Cached forward and VJP kernels for `sRGB(upsample_jxl(XYB(rgb_low), u))`."""
  xyb_to_srgb, _ = _get_xyb_to_srgb_kernels()

  @jax.jit
  def _fwd_fn(r_low: jax.Array) -> jax.Array:
    x_in = srgb_to_xyb(jnp.moveaxis(jnp.clip(r_low, 0.0, 1.0), -1, 0))
    return xyb_to_srgb(upsample_jxl(x_in, u, out_hw=(h, w)))

  @jax.jit
  def _vjp_fn(r_low: jax.Array, g_rgb: jax.Array) -> jax.Array:
    def _f(z: jax.Array) -> jax.Array:
      x_in = srgb_to_xyb(jnp.moveaxis(jnp.clip(z, 0.0, 1.0), -1, 0))
      return xyb_to_srgb(upsample_jxl(x_in, u, out_hw=(h, w)))

    _, vjp_f = jax.vjp(_f, r_low)
    return vjp_f(g_rgb)[0]

  return _fwd_fn, _vjp_fn


def _project_lowres_rgb_wd_jax(
    smooth_full: jax.Array,
    upsampling: int,
    vgg_params: Any,
    log2_sigma: float = 1.0,
) -> jax.Array:
  """Stage 1 WD-driven low-resolution projection compensating `upsample_jxl` passband & texture variance."""
  u = int(upsampling)
  rgb0 = downsample_rgb_jax(smooth_full, u)
  h, w = int(smooth_full.shape[0]), int(smooth_full.shape[1])
  if u == 1 or h < 16 or w < 16:
    return rgb0

  fwd_up, vjp_up = _get_compiled_lowres_upsample_kernels(h, w, u)
  wd_vg = _get_compiled_wd_val_and_grad(num_levels=2, num_scales=3)
  n_iters = 16 if h * w <= 16384 else 5
  lr_proj = 0.003 if n_iters >= 8 else 0.004
  tx = optax.adam(lr_proj, b1=0.5, eps=1e-5)
  r_low = rgb0
  opt_st = tx.init(r_low)
  best_r_low = rgb0
  best_obj = float("inf")

  for _ in range(n_iters):
    rec_sf = fwd_up(r_low)
    wd_v, g_rgb = wd_vg(
        rec_sf,
        smooth_full,
        vgg_params,
        float(log2_sigma),
        num_levels=2,
        num_scales=3,
    )
    diff0 = r_low - rgb0
    obj = float(wd_v) + 0.025 * float(jnp.sum(jnp.square(diff0)))
    if obj < best_obj:
      best_obj = obj
      best_r_low = r_low
    g = vjp_up(r_low, g_rgb) + 0.05 * diff0
    upd, opt_st = tx.update(g, opt_st, r_low)
    r_low = jnp.clip(optax.apply_updates(r_low, upd), 0.0, 1.0)

  rec_final = fwd_up(r_low)
  wd_final = float(
      _get_compiled_wd_only(num_levels=2, num_scales=3)(
          rec_final,
          smooth_full,
          vgg_params,
          float(log2_sigma),
          num_levels=2,
          num_scales=3,
      )
  )
  obj_final = wd_final + 0.025 * float(jnp.sum(jnp.square(r_low - rgb0)))
  if obj_final < best_obj:
    best_r_low = r_low
  return best_r_low


def _dilate_bool_grid(mask: np.ndarray, radius: int = 1) -> np.ndarray:
  """Morphological box dilation of a 2D boolean grid in pure NumPy."""
  out = np.asarray(mask, dtype=bool).copy()
  h, w = out.shape
  for _ in range(max(0, int(radius))):
    prev = out.copy()
    for dy in (-1, 0, 1):
      for dx in (-1, 0, 1):
        if dy == 0 and dx == 0:
          continue
        y0_src = max(0, -dy)
        y1_src = min(h, h - dy)
        x0_src = max(0, -dx)
        x1_src = min(w, w - dx)
        out[y0_src + dy : y1_src + dy, x0_src + dx : x1_src + dx] |= prev[
            y0_src:y1_src, x0_src:x1_src
        ]
  return out


def extract_patch_rectangles(
    mask_h8_w8: np.ndarray,
    height: int,
    width: int,
    blend_mode: int = PATCH_BLEND_REPLACE,
) -> list[PatchEntry]:
  """Decomposes a 2D boolean block mask `(H8, W8)` into minimal non-overlapping `PatchEntry` rectangles."""
  mask = np.asarray(mask_h8_w8, dtype=bool)
  h8, w8 = mask.shape
  active_rects: dict[tuple[int, int], int] = {}
  finished: list[tuple[int, int, int, int]] = []

  for by in range(h8):
    runs: list[tuple[int, int]] = []
    bx = 0
    while bx < w8:
      if mask[by, bx]:
        bx0 = bx
        while bx < w8 and mask[by, bx]:
          bx += 1
        runs.append((bx0, bx))
      else:
        bx += 1
    run_set = set(runs)
    for span, by0 in list(active_rects.items()):
      if span not in run_set:
        finished.append((span[0], by0, span[1], by))
        del active_rects[span]
    for span in runs:
      if span not in active_rects:
        active_rects[span] = by

  for span, by0 in active_rects.items():
    finished.append((span[0], by0, span[1], h8))

  patches: list[PatchEntry] = []
  for bx0, by0, bx1, by1 in finished:
    x0 = int(bx0 * 8)
    y0 = int(by0 * 8)
    x1 = min(int(bx1 * 8), int(width))
    y1 = min(int(by1 * 8), int(height))
    if x1 > x0 and y1 > y0:
      patches.append(
          PatchEntry(
              ref=0,
              x0=x0,
              y0=y0,
              xsize=x1 - x0,
              ysize=y1 - y0,
              x=x0,
              y=y0,
              blend_mode=int(blend_mode),
          )
      )
  return patches


def _patches_to_weights(
    patches: Sequence[PatchEntry], h: int, w: int
) -> tuple[jax.Array, jax.Array]:
  """Converts co-located patches into `(w_ref, w_main)` masks of shape `(1, H, W)`."""
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
  return jnp.asarray(w_ref), jnp.asarray(w_main)


def _compute_state_precond_scales_jax(
    state: JxlVarDCTState, topo: StaticVarDCTTopology | None = None
) -> tuple[jax.Array, jax.Array]:
  """Computes per-coefficient preconditioning scales `(scale_dc, scale_ac)` for `state`."""
  if topo is None:
    topo = StaticVarDCTTopology(state)
  rq_tl = jnp.clip(
      jnp.asarray(state.raw_quant_field, dtype=jnp.float32), 1.0, 256.0
  )[topo.tl_by_grid, topo.tl_bx_grid]
  return _compute_precond_scales_jax(topo, rq_tl)


@functools.lru_cache(maxsize=16)
def _get_compiled_continuous_dc_ac(
    h: int,
    w: int,
    do_inv_gab: bool,
    canon_s: tuple[int, ...] = (0,),
    x_qm_scale: int = 3,
    b_qm_scale: int = 2,
):
  """Returns JIT-compiled `compute_continuous_dc_ac_jax` kernel."""
  dummy_state = init_state_from_rgb_jax(
      np.zeros((h, w, 3), dtype=np.float32),
      distance=1.0,
      epf_iters=0,
      gab=bool(do_inv_gab),
  )
  dummy_state.x_qm_scale = int(x_qm_scale)
  dummy_state.b_qm_scale = int(b_qm_scale)
  topo = StaticVarDCTTopology(dummy_state, force_strategies=canon_s)

  @jax.jit
  def _fn(
      xyb_chw: jax.Array,
      acs_grid: jax.Array,
      rq: jax.Array,
      gs: jax.Array,
      qdc: jax.Array,
      yx: jax.Array,
      yb: jax.Array,
      ytox_dc: jax.Array,
      ytob_dc: jax.Array,
      dq_mats: dict[int, jax.Array],
      dc_w: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    return compute_continuous_dc_ac_jax(
        xyb_chw,
        dummy_state,
        apply_inv_gab=bool(do_inv_gab),
        init_mode="dct",
        topology=topo,
        acs_grid=acs_grid,
        raw_quant_field=rq,
        ytox_map=yx,
        ytob_map=yb,
        global_scale=gs,
        quant_dc=qdc,
        ytox_dc=ytox_dc,
        ytob_dc=ytob_dc,
        dequant_matrices=dq_mats,
        dc_quant_weights=dc_w,
    )

  return _fn


def compute_continuous_dc_ac_jax(
    xyb_chw: jax.Array,
    state: JxlVarDCTState,
    apply_inv_gab: bool = True,
    *,
    init_mode: str = "dct",
    seed: int = 0,
    topology: StaticVarDCTTopology | None = None,
    acs_grid: jax.Array | None = None,
    raw_quant_field: jax.Array | None = None,
    ytox_map: jax.Array | None = None,
    ytob_map: jax.Array | None = None,
    global_scale: float | jax.Array | None = None,
    quant_dc: float | jax.Array | None = None,
    ytox_dc: float | jax.Array | None = None,
    ytob_dc: float | jax.Array | None = None,
    dequant_matrices: dict[int, jax.Array] | None = None,
    dc_quant_weights: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Computes continuous pre-rounded `(dc_quant, quant_ac)` in JAX from XYB `(3, H, W)` across all `AcStrategy` shapes."""
  mode = _normalize_init_mode(init_mode)
  h, w = int(state.height), int(state.width)
  h8, w8 = int(state.h8), int(state.w8)
  if mode == "zero":
    dc_cont = jnp.zeros((3, h8, w8), dtype=jnp.float32)
    ac_cont = jnp.zeros((3, h8 * 8, w8 * 8), dtype=jnp.float32)
    return dc_cont, ac_cont
  if topology is None and mode == "dct":
    acs_np = np.asarray(state.ac_strategy, dtype=np.int32)
    all_zero = bool(np.all(acs_np == 0))
    default_cands = tuple(
        int(s)
        for s in DEFAULT_CANDIDATE_AC_STRATEGIES
        if h8 >= ACS_COVERED_BLOCKS_Y[s] and w8 >= ACS_COVERED_BLOCKS_X[s]
    )
    used_unique = tuple(sorted(set(int(x) for x in np.unique(acs_np)) | {0}))
    if all_zero or set(used_unique).issubset(set(default_cands)):
      canon_s = (0,) if all_zero else default_cands
      default_dq = get_default_dequant_matrices()
      dq_dict = state.dequant_matrices if state.dequant_matrices else default_dq
      dq_mats_arg = {
          int(s): jnp.asarray(
              dq_dict.get(int(s), default_dq[int(s)]), dtype=jnp.float32
          )
          for s in canon_s
      }
      dc_w_arg = jnp.asarray(
          state.dc_quant_weights
          if getattr(state, "dc_quant_weights", None) is not None
          else K_DC_QUANT,
          dtype=jnp.float32,
      )
      fn = _get_compiled_continuous_dc_ac(
          h,
          w,
          bool(apply_inv_gab and state.gab),
          canon_s,
          int(state.x_qm_scale),
          int(state.b_qm_scale),
      )
      return fn(
          jnp.asarray(xyb_chw, dtype=jnp.float32),
          jnp.asarray(acs_np, dtype=jnp.int32),
          jnp.asarray(state.raw_quant_field, dtype=jnp.float32),
          jnp.float32(state.global_scale),
          jnp.float32(state.quant_dc),
          jnp.asarray(state.ytox_map, dtype=jnp.float32),
          jnp.asarray(state.ytob_map, dtype=jnp.float32),
          jnp.float32(state.ytox_dc),
          jnp.float32(state.ytob_dc),
          dq_mats_arg,
          dc_w_arg,
      )
  topo = topology if topology is not None else StaticVarDCTTopology(state)
  if mode == "random":
    scale_dc, scale_ac = _compute_state_precond_scales_jax(state, topo)
    key = jax.random.PRNGKey(int(seed))
    k_dc, k_ac = jax.random.split(key)
    z_dc = jax.random.normal(k_dc, (3, h8, w8), dtype=jnp.float32) * 0.15
    z_dc = z_dc.at[1].set(jnp.abs(z_dc[1]))
    z_ac = jax.random.normal(k_ac, (3, h8 * 8, w8 * 8), dtype=jnp.float32) * 0.15
    dc_cont = z_dc * scale_dc
    ac_cont = mask_llf_jax(z_ac * scale_ac, topo)
    return dc_cont, ac_cont

  xyb = jnp.asarray(xyb_chw, dtype=jnp.float32)
  if apply_inv_gab and state.gab:
    xyb = gaborish_inverse_3x3(xyb, state.gab_weights)
  pad_h = h8 * 8 - h
  pad_w = w8 * 8 - w
  if pad_h > 0 or pad_w > 0:
    xyb = jnp.pad(xyb, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
  xyb_blocks = jnp.transpose(xyb.reshape(3, h8, 8, w8, 8), (1, 3, 0, 2, 4))

  rq_arr = (
      jnp.asarray(raw_quant_field, dtype=jnp.float32)
      if raw_quant_field is not None
      else jnp.array(state.raw_quant_field, dtype=jnp.float32)
  )
  gs_val = (
      jnp.asarray(global_scale, dtype=jnp.float32)
      if global_scale is not None
      else float(state.global_scale)
  )
  qdc_val = (
      jnp.asarray(quant_dc, dtype=jnp.float32)
      if quant_dc is not None
      else float(state.quant_dc)
  )
  yx_arr = (
      jnp.asarray(ytox_map, dtype=jnp.float32)
      if ytox_map is not None
      else jnp.array(state.ytox_map, dtype=jnp.float32)
  )
  yb_arr = (
      jnp.asarray(ytob_map, dtype=jnp.float32)
      if ytob_map is not None
      else jnp.array(state.ytob_map, dtype=jnp.float32)
  )
  ytox_dc_val = (
      jnp.asarray(ytox_dc, dtype=jnp.float32)
      if ytox_dc is not None
      else float(state.ytox_dc)
  )
  ytob_dc_val = (
      jnp.asarray(ytob_dc, dtype=jnp.float32)
      if ytob_dc is not None
      else float(state.ytob_dc)
  )

  raw_q_f = jnp.maximum(rq_arr, 1.0)
  qac_scale_grid = (gs_val / K_GLOBAL_SCALE_DENOM) * raw_q_f
  x_qm_mul = float(1.25 ** (state.x_qm_scale - 2.0))
  b_qm_mul = float(1.25 ** (state.b_qm_scale - 2.0))
  x_cc_tile = (
      float(state.base_correlation_x)
      + yx_arr / float(state.color_factor)
  )
  b_cc_tile = (
      float(state.base_correlation_b)
      + yb_arr / float(state.color_factor)
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

  acs_g = (
      jnp.asarray(acs_grid, dtype=jnp.int32) if acs_grid is not None else None
  )

  if len(topo.groups_1x1) > 0:
    px_flat = xyb_blocks.reshape(h8, w8, 3, 64)
    qs_all = qac_scale_grid[:, :, None, None]
    x_cc_all = x_cc_grid[:, :, None, None]
    b_cc_all = b_cc_grid[:, :, None, None]
    for grp1 in topo.groups_1x1:
      if acs_g is not None:
        mask_2d = (acs_g == int(grp1["s"])).astype(jnp.float32)
      else:
        with jax.ensure_compile_time_eval():
          mask_2d = jnp.asarray(grp1["mask_2d"], dtype=jnp.float32)
      coeffs = jnp.einsum("ij,hwcj->hwci", grp1["fwd_basis"], px_flat).reshape(
          h8, w8, 3, 8, 8
      )
      dc_continuous = dc_continuous + (
          jnp.transpose(coeffs[:, :, :, 0, 0], (2, 0, 1)) * mask_2d[None, :, :]
      )
      dq_mat = (
          dequant_matrices[int(grp1["s"])]
          if dequant_matrices is not None and int(grp1["s"]) in dequant_matrices
          else grp1["dq_mat"]
      )
      wy, wx, wb = dq_mat[1][None, None], dq_mat[0][None, None], dq_mat[2][None, None]
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
    dq_mat = (
        dequant_matrices[int(grp["s"])]
        if dequant_matrices is not None and int(grp["s"]) in dequant_matrices
        else grp["dq_mat"]
    )
    if grp.get("grid_aligned", False):
      nh_cell = grp["nh_cell"]
      nw_cell = grp["nw_cell"]
      h_lim = grp["h_lim"]
      w_lim = grp["w_lim"]
      if acs_g is not None:
        cell_mask_f = (
            acs_g[:h_lim:cy, :w_lim:cx] == int(grp["s"])
        ).astype(jnp.float32)
        blk_mask = jnp.broadcast_to(
            cell_mask_f[:, None, :, None], (nh_cell, cy, nw_cell, cx)
        ).reshape(h_lim, w_lim)
      else:
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

  inv_global_scale = K_GLOBAL_SCALE_DENOM / gs_val
  if dc_quant_weights is not None:
    dc_w_j = jnp.asarray(dc_quant_weights, dtype=jnp.float32)
    mul_dc = [inv_global_scale / qdc_val * dc_w_j[c] for c in range(3)]
  else:
    dc_weights = (
        np.asarray(state.dc_quant_weights, dtype=np.float32)
        if getattr(state, "dc_quant_weights", None) is not None
        else np.asarray(K_DC_QUANT, dtype=np.float32)
    )
    mul_dc = [
        inv_global_scale / qdc_val * float(k) for k in dc_weights
    ]
  cfl_dc_x = float(state.base_correlation_x) + ytox_dc_val / float(
      state.color_factor
  )
  cfl_dc_b = float(state.base_correlation_b) + ytob_dc_val / float(
      state.color_factor
  )
  q_dc_y = dc_continuous[1] / mul_dc[1]
  q_dc_x = (
      dc_continuous[0] - jnp.round(q_dc_y) * (mul_dc[1] * cfl_dc_x)
  ) / mul_dc[0]
  q_dc_b = (
      dc_continuous[2] - jnp.round(q_dc_y) * (mul_dc[1] * cfl_dc_b)
  ) / mul_dc[2]
  dc_cont = jnp.stack([q_dc_x, q_dc_y, q_dc_b], axis=0)
  ac_cont = jnp.transpose(q_ac_blocks, (2, 0, 3, 1, 4)).reshape(
      3, h8 * 8, w8 * 8
  )
  return dc_cont, ac_cont


@functools.lru_cache(maxsize=16)
def _get_compiled_stage1_eval(
    h: int,
    w: int,
    epf_iters: int = 1,
    gab: bool = True,
    canon_s: tuple[int, ...] = (0,),
    x_qm_scale: int = 3,
    b_qm_scale: int = 2,
):
  """Returns JIT-compiled `(dc_cont, ac_cont, dc_q, ac_q, rec_srgb, rate_map_bits)` evaluator."""
  dummy_state = init_state_from_rgb_jax(
      np.zeros((h, w, 3), dtype=np.float32),
      distance=1.0,
      epf_iters=int(epf_iters),
      gab=bool(gab),
  )
  dummy_state.x_qm_scale = int(x_qm_scale)
  dummy_state.b_qm_scale = int(b_qm_scale)
  topo = StaticVarDCTTopology(dummy_state, force_strategies=canon_s)

  @jax.jit
  def _eval_fn(
      target_xyb: jax.Array,
      acs_grid: jax.Array,
      gs: jax.Array,
      qdc: jax.Array,
      rq: jax.Array,
      yx: jax.Array,
      yb: jax.Array,
      ytox_dc: jax.Array,
      ytob_dc: jax.Array,
      epf_sharpness: jax.Array,
  ):
    dc_c, ac_c = compute_continuous_dc_ac_jax(
        target_xyb,
        dummy_state,
        apply_inv_gab=True,
        topology=topo,
        acs_grid=acs_grid,
        raw_quant_field=rq,
        ytox_map=yx,
        ytob_map=yb,
        global_scale=gs,
        quant_dc=qdc,
        ytox_dc=ytox_dc,
        ytob_dc=ytob_dc,
    )
    dc_q = jnp.round(dc_c)
    ac_q = jnp.round(ac_c)
    epf_s_i32 = jnp.asarray(epf_sharpness, dtype=jnp.int32)
    rec_xyb = reconstruct_single_frame_xyb(
        topo,
        params_override={
            "global_scale": gs,
            "quant_dc": qdc,
            "ytox_dc": ytox_dc,
            "ytob_dc": ytob_dc,
            "epf_sharpness": epf_s_i32,
        },
        quant_ac=ac_q,
        dc_quant=dc_q,
        raw_quant_field=jnp.clip(rq, 1.0, 256.0),
        ytox_map=yx,
        ytob_map=yb,
        hard=True,
        include_noise=False,
        acs_grid=acs_grid,
    )
    rec_srgb = jnp.moveaxis(
        opsin_to_srgb(
            rec_xyb, topo.opsin_inv_matrix, topo.opsin_biases, clamp=True
        ),
        0,
        -1,
    )
    rate_info = compute_spatial_rate_map_bits(
        dummy_state,
        quant_ac=ac_q,
        dc_quant=dc_q,
        raw_quant_field=rq,
        ytox_map=yx,
        ytob_map=yb,
        ytox_dc=ytox_dc,
        ytob_dc=ytob_dc,
        ac_strategy=acs_grid,
        epf_sharpness=epf_sharpness,
        global_scale=gs,
        quant_dc=qdc,
        topology=topo,
    )
    return dc_c, ac_c, dc_q, ac_q, rec_srgb, rate_info["block_bits"]

  return _eval_fn


def init_state_from_xyb_jax(
    xyb_chw: jax.Array,
    distance: float = 1.0,
    epf_iters: int | None = None,
    gab: bool | None = None,
    *,
    init_mode: str = "dct",
    seed: int = 0,
    enable_noise: bool = False,
) -> tuple[JxlVarDCTState, jax.Array, jax.Array]:
  """Initializes a `JxlVarDCTState` directly from an XYB `(3, H, W)` tensor (e.g. `kAdd` residual)."""
  _, h, w = xyb_chw.shape
  eff_gab = bool(float(distance) > 0.5) if gab is None else bool(gab)
  xyb_to_srgb, _ = _get_xyb_to_srgb_kernels()
  # Shift residual around mid-gray XYB before converting for AQ/CfL estimation if zero-mean
  xyb_mid = jnp.asarray([0.0, 0.45, 0.45], dtype=jnp.float32)[:, None, None]
  y_mean = float(jnp.mean(xyb_chw[1]))
  xyb_probe = xyb_chw + xyb_mid if y_mean < 0.08 else xyb_chw
  approx_rgb = np.asarray(xyb_to_srgb(xyb_probe), dtype=np.float32)
  raw_quant_field, epf_sharpness, epf_iters_auto = select_quant_and_epf_jax(
      approx_rgb, distance=float(distance)
  )
  ytox_map, ytob_map, ytox_dc, ytob_dc = select_cfl_jax(
      approx_rgb, gab=eff_gab
  )
  if enable_noise:
    ytox_dc, ytob_dc = 0, 0
  eff_epf = (
      int(epf_iters_auto)
      if epf_iters is None
      else int(epf_iters)
  )
  state = init_state_from_rgb_jax(
      approx_rgb,
      distance=distance,
      epf_iters=eff_epf,
      gab=eff_gab,
      raw_quant_field=raw_quant_field,
      epf_sharpness=epf_sharpness,
      ytox_map=ytox_map,
      ytob_map=ytob_map,
      ytox_dc=int(ytox_dc),
      ytob_dc=int(ytob_dc),
  )
  dc_cont, ac_cont = compute_continuous_dc_ac_jax(
      xyb_chw, state, apply_inv_gab=True, init_mode=init_mode, seed=seed
  )
  state.dc_quant = np.asarray(jnp.round(dc_cont), dtype=np.int32)
  state.quant_ac = np.asarray(jnp.round(ac_cont), dtype=np.int32)
  return state, dc_cont, ac_cont


def _acs_covered_yx(acs_id: int) -> tuple[int, int]:
  """Returns `(covered_blocks_y, covered_blocks_x)` for `AcStrategy` `acs_id`."""
  s = int(acs_id)
  if 0 <= s < len(ACS_COVERED_BLOCKS_Y):
    return int(ACS_COVERED_BLOCKS_Y[s]), int(ACS_COVERED_BLOCKS_X[s])
  return 1, 1


def _build_uniform_ac_strategy_grid(
    h8: int, w8: int, acs_id: int
) -> tuple[np.ndarray, np.ndarray]:
  """Builds a valid tiled `(h8, w8)` `ac_strategy` and `ac_strategy_Anchor` grid for `acs_id`."""
  cy, cx = _acs_covered_yx(acs_id)
  acs_grid = np.zeros((h8, w8), dtype=np.int32)
  anchor_grid = np.ones((h8, w8), dtype=bool)
  if int(acs_id) < 0 or int(acs_id) >= 18:
    return acs_grid, anchor_grid
  if cy == 1 and cx == 1:
    acs_grid[:, :] = int(acs_id)
    return acs_grid, anchor_grid
  for by in range(0, h8 - cy + 1, cy):
    for bx in range(0, w8 - cx + 1, cx):
      # libjxl does not allow multi-block transforms to cross 8x8 block-group (64x64 px) boundaries
      if (by // 8) != ((by + cy - 1) // 8) or (bx // 8) != ((bx + cx - 1) // 8):
        continue
      acs_grid[by : by + cy, bx : bx + cx] = int(acs_id)
      anchor_grid[by : by + cy, bx : bx + cx] = False
      anchor_grid[by, bx] = True
  return acs_grid, anchor_grid


_STAGE1_AC_CANDIDATE_CACHE: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}


def stage1_explore_ac_candidates_compositing(
    srgb_hwc: np.ndarray | jax.Array,
    distance: float = 1.0,
    rd_lambda: float | None = None,
    vgg_params: Any | None = None,
    log2_sigma: float = 1.0,
    candidate_strategies: Sequence[int] | None = None,
) -> dict[int, dict[str, Any]]:
  """Stage 1: Evaluates candidate `<= 32x32` `AcStrategy` block shapes (`0..17`) using spatial Rate-WD maps."""
  arr = _normalize_srgb_hwc(srgb_hwc)
  h, w, _ = arr.shape
  h8, w8 = (h + 7) // 8, (w + 7) // 8
  eff_lam = (
      float(rd_lambda)
      if rd_lambda is not None
      else calibrate_rd_lambda(distance)
  )

  if candidate_strategies is None:
    cands = [
        int(s_id)
        for s_id in DEFAULT_CANDIDATE_AC_STRATEGIES
        if h8 >= ACS_COVERED_BLOCKS_Y[s_id] and w8 >= ACS_COVERED_BLOCKS_X[s_id]
    ]
  else:
    cands = [
        int(s)
        for s in candidate_strategies
        if 0 <= int(s) < 18
    ]

  arr_sig = (
      round(float(arr[0, 0, 0]), 6),
      round(float(arr[h // 2, w // 2, 1]), 6),
      round(float(arr[-1, -1, 2]), 6),
      round(float(np.mean(arr)), 6),
  )
  cache_key = (
      h,
      w,
      round(float(distance), 6),
      round(float(eff_lam), 6),
      round(float(log2_sigma), 6),
      tuple(cands),
      arr_sig,
  )
  if cache_key in _STAGE1_AC_CANDIDATE_CACHE:
    return copy.deepcopy(_STAGE1_AC_CANDIDATE_CACHE[cache_key])

  target_j = jnp.asarray(arr, dtype=jnp.float32)
  target_xyb = srgb_to_xyb(jnp.moveaxis(target_j, -1, 0))
  if vgg_params is None:
    vgg_params = load_vgg16_params()
  spatial_wd_fn = _get_compiled_spatial_wd_map(
      block_size=8, num_levels=1, num_scales=2
  )

  base_state = None
  eval_fn = None
  default_cands = tuple(
      int(s_id)
      for s_id in DEFAULT_CANDIDATE_AC_STRATEGIES
      if h8 >= ACS_COVERED_BLOCKS_Y[s_id] and w8 >= ACS_COVERED_BLOCKS_X[s_id]
  )
  lam_per_px = float(eff_lam) / float(max(h * w, 1))
  results: dict[int, dict[str, Any]] = {}

  for s in cands:
    cy, cx = _acs_covered_yx(s)
    if h8 < cy or w8 < cx:
      continue
    s_key = (
        h,
        w,
        round(float(distance), 6),
        round(float(eff_lam), 6),
        round(float(log2_sigma), 6),
        int(s),
        arr_sig,
    )
    if s_key in _STAGE1_AC_CANDIDATE_CACHE:
      results[int(s)] = copy.deepcopy(_STAGE1_AC_CANDIDATE_CACHE[s_key])
      continue
    if base_state is None:
      base_state = _init_frame_state_with_aq_epf_cfl(
          arr, distance=distance, gab=None
      )
      eval_fn = _get_compiled_stage1_eval(
          h,
          w,
          epf_iters=int(base_state.epf_iters),
          gab=bool(base_state.gab),
          canon_s=default_cands,
          x_qm_scale=int(base_state.x_qm_scale),
          b_qm_scale=int(base_state.b_qm_scale),
      )
    st = copy.deepcopy(base_state)
    acs_grid, anchor_grid = _build_uniform_ac_strategy_grid(h8, w8, s)
    st.ac_strategy = acs_grid
    st.ac_strategy_Anchor = anchor_grid
    st.raw_quant_field = np.asarray(
        broadcast_quant_field_to_ac_strategy(st.raw_quant_field, acs_grid),
        dtype=np.int32,
    )
    assert eval_fn is not None
    dc_c, ac_c, dc_q_j, ac_q_j, rec_srgb, rate_map_j = eval_fn(
        target_xyb,
        jnp.asarray(acs_grid, dtype=jnp.int32),
        jnp.float32(st.global_scale),
        jnp.float32(st.quant_dc),
        jnp.asarray(st.raw_quant_field, dtype=jnp.float32),
        jnp.asarray(st.ytox_map, dtype=jnp.float32),
        jnp.asarray(st.ytob_map, dtype=jnp.float32),
        jnp.float32(st.ytox_dc),
        jnp.float32(st.ytob_dc),
        jnp.asarray(st.epf_sharpness, dtype=jnp.float32),
    )
    st.dc_quant = np.asarray(dc_q_j, dtype=np.int32)
    st.quant_ac = np.asarray(ac_q_j, dtype=np.int32)
    s1_log2_sigma = (
        float(log2_sigma)
        * float(np.clip((float(distance) - 0.18) / 0.57, 0.0, 1.0))
        if float(distance) < 0.75
        else float(log2_sigma)
    )
    _, wd_map_jax = spatial_wd_fn(
        rec_srgb,
        target_j,
        vgg_params,
        block_size=8,
        log2_sigma=s1_log2_sigma,
        num_levels=1,
        num_scales=2,
    )
    wd_map = np.asarray(wd_map_jax, dtype=np.float32)
    rate_map = np.asarray(rate_map_j, dtype=np.float32)
    j_map = wd_map + lam_per_px * rate_map
    j_sum = float(np.sum(j_map))
    cand_entry: dict[str, Any] = {
        "strategy_id": int(s),
        "covered_blocks_y": cy,
        "covered_blocks_x": cx,
        "wd_map": wd_map,
        "rate_map_bits": rate_map,
        "j_map": j_map,
        "spatial_j_map": j_map,
        "cost_map": j_map,
        "j_score": j_sum,
        "j_init": j_sum,
    }
    if int(s) == 0:
      cand_entry["state"] = st
      cand_entry["dc_cont"] = np.asarray(dc_c, dtype=np.float32)
      cand_entry["ac_cont"] = np.asarray(ac_c, dtype=np.float32)
    results[int(s)] = cand_entry
    if len(_STAGE1_AC_CANDIDATE_CACHE) >= 32:
      _STAGE1_AC_CANDIDATE_CACHE.clear()
    _STAGE1_AC_CANDIDATE_CACHE[s_key] = copy.deepcopy(results[int(s)])
  _STAGE1_AC_CANDIDATE_CACHE[cache_key] = copy.deepcopy(results)
  return results


def _build_anchor_grid_from_acs(acs_grid: np.ndarray) -> np.ndarray:
  """Constructs boolean `ac_strategy_Anchor` `(h8, w8)` from valid `acs_grid`."""
  h8, w8 = acs_grid.shape
  anchor = np.zeros((h8, w8), dtype=bool)
  visited = np.zeros((h8, w8), dtype=bool)
  for by in range(h8):
    for bx in range(w8):
      if visited[by, bx]:
        continue
      s = int(acs_grid[by, bx])
      cy, cx = _acs_covered_yx(s)
      anchor[by, bx] = True
      visited[by : by + cy, bx : bx + cx] = True
  return anchor


def stage2_select_non_overlapping_ac_strategy_compositing(
    stage1_candidates: dict[int, dict[str, Any]],
    h8: int,
    w8: int,
    multi_block_margin: float = 0.98,
) -> tuple[np.ndarray, np.ndarray]:
  """Stage 2: Hierarchical 4x4 DP non-overlapping `<= 32x32` `AcStrategy` tiling within 64x64 CfL tiles."""
  if not stage1_candidates:
    return np.zeros((h8, w8), dtype=np.int32), np.ones((h8, w8), dtype=bool)

  weighted_maps: dict[int, np.ndarray] = {}
  mb_inv = 1.0 / max(float(multi_block_margin), 1e-6)
  for s, info in stage1_candidates.items():
    s_int = int(s)
    if s_int < 0 or s_int >= 18:
      continue
    j_map = np.asarray(info["j_map"], dtype=np.float32)
    cy, cx = _acs_covered_yx(s_int)
    if s_int == 0:
      weighted_maps[s_int] = j_map
    elif cy == 1 and cx == 1:
      weighted_maps[s_int] = j_map * 1.015
    else:
      weighted_maps[s_int] = j_map * mb_inv

  acs_u8 = stage2_select_non_overlapping_ac_strategy(
      weighted_maps, allow_multiblock=True, h8=h8, w8=w8
  )
  acs_grid = np.asarray(acs_u8, dtype=np.int32)
  validate_ac_strategy_tiling(acs_grid)
  anchor_grid = _build_anchor_grid_from_acs(acs_grid)
  return acs_grid, anchor_grid


def _regularize_fg_mask_blocks(
    fg_adv: np.ndarray, lam_per_px: float, u: int = 2
) -> np.ndarray:
  """Selects & regularizes foreground `(H8, W8)` blocks where `1:1` beats `1:U`."""
  h8, w8 = fg_adv.shape
  if h8 * w8 < 256:
    med_adv = float(np.median(fg_adv))
    max_adv = float(np.max(fg_adv))
    adv_thr = max(0.0, med_adv + 0.25 * max(max_adv - med_adv, 0.0))
    fg_mask = fg_adv > adv_thr
    if not np.any(fg_mask) and max_adv > float(np.min(fg_adv)):
      fg_mask = fg_adv >= np.percentile(fg_adv, 65.0)
    if np.all(fg_mask) and fg_mask.size > 1:
      fg_mask = fg_adv > float(np.percentile(fg_adv, 15.0))
    return np.asarray(fg_mask, dtype=bool)

  if h8 * w8 >= 1024:
    adv_smooth = 0.25 * fg_adv + 0.75 * uniform_filter(
        fg_adv, size=5, mode="nearest"
    )
    fg_raw = adv_smooth > (float(lam_per_px) * 8.0)
  else:
    adv_smooth = 0.40 * fg_adv + 0.60 * uniform_filter(
        fg_adv, size=3, mode="nearest"
    )
    fg_raw = adv_smooth > (float(lam_per_px) * 5.0)
  min_cc = max(8, min(48, (h8 * w8) // 256)) if (h8 * w8 >= 1024) else 3
  lbl, n_cc = ndimage_label(fg_raw)
  out = np.zeros_like(fg_raw, dtype=bool)
  for cc_id in range(1, int(n_cc) + 1):
    m = lbl == cc_id
    if int(np.sum(m)) >= min_cc or float(
        np.sum(np.maximum(fg_adv[m], 0.0))
    ) > float(lam_per_px) * (16.0 * min_cc):
      out |= m
  if np.any(out):
    out = np.asarray(
        binary_closing(out, structure=np.ones((3, 3), dtype=bool)), dtype=bool
    )
  if h8 >= 32 and w8 >= 32 and np.any(out):
    g = 2 if int(u) <= 4 else 4
    gh, gw = (h8 + g - 1) // g, (w8 + g - 1) // g
    pad_out = np.pad(out, ((0, gh * g - h8), (0, gw * g - w8)), mode="edge")
    cell_fg = pad_out.reshape(gh, g, gw, g).any(axis=(1, 3))
    out = np.repeat(np.repeat(cell_fg, g, axis=0), g, axis=1)[:h8, :w8]
  if np.any(out) and not np.all(out):
    inv = ~out
    lbl_bg, n_bg = ndimage_label(inv)
    min_hole = 64 if (h8 * w8 >= 1024) else 8
    min_border = 16 if (h8 * w8 >= 1024) else 4
    for bg_id in range(1, int(n_bg) + 1):
      m_bg = lbl_bg == bg_id
      touches = (
          bool(np.any(m_bg[0, :]))
          or bool(np.any(m_bg[-1, :]))
          or bool(np.any(m_bg[:, 0]))
          or bool(np.any(m_bg[:, -1]))
      )
      if (not touches and int(np.sum(m_bg)) < min_hole) or (
          int(np.sum(m_bg)) < min_border
      ):
        out |= m_bg
  return np.asarray(out, dtype=bool)


def _pack_background_acs_and_cfl(
    m_state: JxlVarDCTState,
    fg_mask: np.ndarray,
    dc_halo: np.ndarray,
) -> None:
  """Keeps foreground-touching transforms intact, packs pure-background cells to DCT32X32/DCT16X16, and zeros bg CfL."""
  h8, w8 = int(m_state.h8), int(m_state.w8)
  acs = m_state.ac_strategy.copy()
  anc = m_state.ac_strategy_Anchor.copy()
  for by_s in range(h8):
    for bx_s in range(w8):
      if anc[by_s, bx_s]:
        s_cur = int(acs[by_s, bx_s])
        cy_s, cx_s = _acs_covered_yx(s_cur)
        if cy_s > 1 or cx_s > 1:
          sub_fg = fg_mask[by_s : by_s + cy_s, bx_s : bx_s + cx_s]
          if h8 * w8 >= 256 and np.any(sub_fg):
            dc_halo[by_s : by_s + cy_s, bx_s : bx_s + cx_s] = True
          elif not np.all(sub_fg):
            acs[by_s : by_s + cy_s, bx_s : bx_s + cx_s] = 0
            anc[by_s : by_s + cy_s, bx_s : bx_s + cx_s] = True

  if h8 * w8 >= 256:
    acs_packed = acs.copy()
    for ty in range((h8 + 7) // 8):
      for tx in range((w8 + 7) // 8):
        for sub_y in (0, 4):
          for sub_x in (0, 4):
            by0, bx0 = ty * 8 + sub_y, tx * 8 + sub_x
            if (
                by0 + 4 <= h8
                and bx0 + 4 <= w8
                and not np.any(dc_halo[by0 : by0 + 4, bx0 : bx0 + 4])
            ):
              acs_packed[by0 : by0 + 4, bx0 : bx0 + 4] = 5
            else:
              for sy2 in (0, 2):
                for sx2 in (0, 2):
                  by2, bx2 = by0 + sy2, bx0 + sx2
                  if (
                      by2 + 2 <= h8
                      and bx2 + 2 <= w8
                      and not np.any(dc_halo[by2 : by2 + 2, bx2 : bx2 + 2])
                  ):
                    acs_packed[by2 : by2 + 2, bx2 : bx2 + 2] = 4
    if validate_ac_strategy_tiling(acs_packed):
      acs = acs_packed
      anc = _build_anchor_grid_from_acs(acs)

  m_state.ac_strategy = acs
  m_state.ac_strategy_Anchor = anc
  used_m = 0
  for s_u in np.unique(acs):
    used_m |= 1 << int(s_u)
  m_state.used_acs = used_m

  for ty in range(int(m_state.ht)):
    for tx in range(int(m_state.wt)):
      if not np.any(dc_halo[ty * 8 : (ty + 1) * 8, tx * 8 : (tx + 1) * 8]):
        m_state.ytox_map[ty, tx] = 0
        m_state.ytob_map[ty, tx] = 0


_STAGE1_MULTIRES_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_STAGE1_MAIN_DP_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _stage1_multires_rate_wd_analysis(
    srgb_hwc: np.ndarray | jax.Array,
    upsampling: int = 2,
    distance: float = 1.0,
    ref_distance: float | None = None,
    rd_lambda: float | None = None,
    dilate_blocks: int = 0,
    vgg_params: Any | None = None,
    log2_sigma: float = 1.0,
    explore_ac_strategy: bool = True,
    enable_noise: bool = False,
    noise_lut: np.ndarray | jax.Array | None = None,
) -> dict[str, Any]:
  """Computes multi-resolution Rate-WD spatial advantage maps `Delta J_patch(by, bx) = J_main - J_{ref, U}`."""
  arr = _normalize_srgb_hwc(srgb_hwc)
  h, w, _ = arr.shape
  h8, w8 = (h + 7) // 8, (w + 7) // 8
  u = int(upsampling)
  d_main = float(distance)
  eff_lam = (
      float(rd_lambda)
      if rd_lambda is not None
      else calibrate_rd_lambda(d_main)
  )
  use_noise = bool(enable_noise)
  d_ref = _calibrate_ref_distance(
      d_main, eff_lam, ref_distance, upsampling=u, enable_noise=use_noise
  )
  arr_sig = (
      round(float(arr[0, 0, 0]), 6),
      round(float(arr[h // 2, w // 2, 1]), 6),
      round(float(arr[-1, -1, 2]), 6),
      round(float(np.mean(arr)), 6),
  )
  target_j = jnp.asarray(arr, dtype=jnp.float32)
  target_xyb = srgb_to_xyb(jnp.moveaxis(target_j, -1, 0))

  if use_noise:
    lut_np = (
        np.asarray(noise_lut, dtype=np.float32)
        if noise_lut is not None
        else _estimate_sensor_noise_lut_np(target_xyb)
    )
    has_sensor_grain = bool(float(np.max(lut_np)) >= 0.0020)
    lut_eval_np = (
        np.clip(lut_np, 2.0 / 1024.0, 0.5)
        if has_sensor_grain
        else np.full((8,), 0.0025, dtype=np.float32)
    )
    lut_eval_j = jnp.asarray(lut_eval_np, dtype=jnp.float32)
  else:
    has_sensor_grain = False
    lut_eval_np = np.zeros((8,), dtype=np.float32)
    lut_eval_j = jnp.zeros((8,), dtype=jnp.float32)

  is_large_or_noisy = bool(use_noise or (h >= 128 and w >= 128))
  s1_levels = 2 if is_large_or_noisy else 1
  s1_scales = 2
  wd_scale = (
      (16.0 / 11.0)
      if (is_large_or_noisy and h >= 64 and w >= 64)
      else 1.0
  )
  lut_sig = round(float(np.mean(lut_eval_np)), 5)
  m_cache_key = (
      h,
      w,
      u,
      round(d_main, 6),
      round(d_ref, 6),
      round(float(eff_lam), 6),
      int(dilate_blocks),
      round(float(log2_sigma), 6),
      bool(explore_ac_strategy),
      use_noise,
      has_sensor_grain,
      lut_sig,
      arr_sig,
  )
  if m_cache_key in _STAGE1_MULTIRES_CACHE:
    return copy.deepcopy(_STAGE1_MULTIRES_CACHE[m_cache_key])

  if vgg_params is None:
    vgg_params = load_vgg16_params()
  spatial_wd_fn = _get_compiled_spatial_wd_map(
      block_size=8, num_levels=s1_levels, num_scales=s1_scales
  )
  xyb_to_srgb_fn, _ = _get_xyb_to_srgb_kernels()
  if has_sensor_grain:
    fg_xyb_src = _shrink_sensor_grain_xyb_jax(target_xyb, lut_eval_j)
    clean_rgb = np.asarray(xyb_to_srgb_fn(fg_xyb_src), dtype=np.float32)
    clean_rgb_j = jnp.asarray(clean_rgb, dtype=jnp.float32)
  else:
    fg_xyb_src = target_xyb
    clean_rgb = arr
    clean_rgb_j = target_j

  # 1. Stage 1 & 2 on full-resolution main candidate (independent of upsampling U)
  main_dp_key = (
      h,
      w,
      round(d_main, 6),
      round(float(eff_lam), 6),
      round(float(log2_sigma), 6),
      bool(explore_ac_strategy),
      use_noise,
      has_sensor_grain,
      lut_sig,
      arr_sig,
  )
  if main_dp_key in _STAGE1_MAIN_DP_CACHE:
    dp_cached = copy.deepcopy(_STAGE1_MAIN_DP_CACHE[main_dp_key])
    stage1_cands = dp_cached["stage1_cands"]
    stage2_acs = dp_cached["stage2_acs"]
    stage2_anchors = dp_cached["stage2_anchors"]
    main_state = dp_cached["main_state"]
    dc_m = dp_cached["dc_m"]
    ac_m = dp_cached["ac_m"]
    wd_main_map = dp_cached["wd_main_map"]
    rate_main_map = dp_cached["rate_main_map"]
  else:
    cand_strats = None if explore_ac_strategy else (0,)
    stage1_cands = stage1_explore_ac_candidates_compositing(
        clean_rgb,
        distance=d_main,
        rd_lambda=eff_lam,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        candidate_strategies=cand_strats,
    )
    stage2_acs, stage2_anchors = (
        stage2_select_non_overlapping_ac_strategy_compositing(
            stage1_cands, h8, w8
        )
    )
    main_state = copy.deepcopy(stage1_cands[0]["state"])
    if use_noise:
      main_state.ytox_dc = 0
      main_state.ytob_dc = 0
    if np.any(stage2_acs != 0) or use_noise:
      main_state.ac_strategy = stage2_acs
      main_state.ac_strategy_Anchor = stage2_anchors
      rq_m, epf_s_m, _ = select_quant_and_epf_jax(
          clean_rgb, distance=d_main, ac_strategy=stage2_acs
      )
      main_state.raw_quant_field = np.asarray(rq_m, dtype=np.int32)
      main_state.epf_sharpness = np.asarray(epf_s_m, dtype=np.uint8)
      dc_m_j, ac_m_j = compute_continuous_dc_ac_jax(
          fg_xyb_src, main_state, apply_inv_gab=True
      )
      main_state.dc_quant = np.round(np.asarray(dc_m_j)).astype(np.int32)
      main_state.quant_ac = np.round(np.asarray(ac_m_j)).astype(np.int32)
      dc_m = np.asarray(dc_m_j, dtype=np.float32)
      ac_m = np.asarray(ac_m_j, dtype=np.float32)
    else:
      dc_m = np.asarray(stage1_cands[0]["dc_cont"], dtype=np.float32)
      ac_m = np.asarray(stage1_cands[0]["ac_cont"], dtype=np.float32)

    default_cands_m = (
        tuple(
            int(s_id)
            for s_id in DEFAULT_CANDIDATE_AC_STRATEGIES
            if h8 >= ACS_COVERED_BLOCKS_Y[s_id]
            and w8 >= ACS_COVERED_BLOCKS_X[s_id]
        )
        if np.any(stage2_acs != 0)
        else (0,)
    )
    eval_main_fn = _get_compiled_stage1_eval(
        h,
        w,
        epf_iters=int(main_state.epf_iters),
        gab=bool(main_state.gab),
        canon_s=default_cands_m,
        x_qm_scale=int(main_state.x_qm_scale),
        b_qm_scale=int(main_state.b_qm_scale),
    )
    _, _, dc_mq, ac_mq, rec_main_clean, rate_main_j = eval_main_fn(
        fg_xyb_src,
        jnp.asarray(stage2_acs, dtype=jnp.int32),
        jnp.float32(main_state.global_scale),
        jnp.float32(main_state.quant_dc),
        jnp.asarray(main_state.raw_quant_field, dtype=jnp.float32),
        jnp.asarray(main_state.ytox_map, dtype=jnp.float32),
        jnp.asarray(main_state.ytob_map, dtype=jnp.float32),
        jnp.float32(main_state.ytox_dc),
        jnp.float32(main_state.ytob_dc),
        jnp.asarray(main_state.epf_sharpness, dtype=jnp.float32),
    )
    main_state.dc_quant = np.asarray(dc_mq, dtype=np.int32)
    main_state.quant_ac = np.asarray(ac_mq, dtype=np.int32)
    if use_noise:
      rec_main_xyb = synthesize_jxl_noise_xyb(
          _fast_reconstruct_xyb(main_state, include_noise=False),
          noise_lut=lut_eval_j,
          ytox=float(main_state.base_correlation_x),
          ytob=float(main_state.base_correlation_b),
          visible_frame_index=1,
          nonvisible_frame_index=0,
          ste_quantize_lut=True,
      )
      rec_main = xyb_to_srgb_fn(rec_main_xyb)
    else:
      rec_main = rec_main_clean
    _, wd_main_jax = spatial_wd_fn(
        rec_main,
        target_j,
        vgg_params,
        block_size=8,
        log2_sigma=float(log2_sigma),
        num_levels=s1_levels,
        num_scales=s1_scales,
    )
    wd_main_map = np.asarray(wd_main_jax, dtype=np.float32) * float(wd_scale)
    rate_main_map = np.asarray(rate_main_j, dtype=np.float32)
    if len(_STAGE1_MAIN_DP_CACHE) >= 2:
      _STAGE1_MAIN_DP_CACHE.clear()
    _STAGE1_MAIN_DP_CACHE[main_dp_key] = copy.deepcopy({
        "stage1_cands": stage1_cands,
        "stage2_acs": stage2_acs,
        "stage2_anchors": stage2_anchors,
        "main_state": main_state,
        "dc_m": dc_m,
        "ac_m": ac_m,
        "wd_main_map": wd_main_map,
        "rate_main_map": rate_main_map,
    })

  lam_per_px = float(eff_lam) / float(max(h * w, 1))
  j_main_map = wd_main_map + lam_per_px * rate_main_map

  # 2. Low-resolution reference candidate (downsampled by U, upsampled in XYB)
  rgb_low_raw = downsample_rgb_jax(clean_rgb_j, u)
  if explore_ac_strategy and (use_noise or h8 * w8 >= 256):
    ref_state_init = _init_ref_state_with_stage2_acs(
        rgb_low_raw,
        d_ref=d_ref,
        eff_lambda=eff_lam,
        enable_noise=use_noise,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        upsampling=u,
    )
  else:
    ref_state_init = _init_frame_state_with_aq_epf_cfl(
        rgb_low_raw, distance=d_ref, gab=None, enable_noise=use_noise
    )
  ref_xyb_rec = _fast_reconstruct_xyb(ref_state_init, include_noise=False)
  ref_xyb_ups = upsample_jxl(ref_xyb_rec, u, out_hw=(h, w))
  if use_noise:
    ref_xyb_eval = synthesize_jxl_noise_xyb(
        ref_xyb_ups,
        noise_lut=lut_eval_j,
        ytox=float(main_state.base_correlation_x),
        ytob=float(main_state.base_correlation_b),
        visible_frame_index=1,
        nonvisible_frame_index=0,
        ste_quantize_lut=True,
    )
  else:
    ref_xyb_eval = ref_xyb_ups
  ref_srgb_ups = xyb_to_srgb_fn(ref_xyb_eval)
  _, wd_ref_jax = spatial_wd_fn(
      ref_srgb_ups,
      target_j,
      vgg_params,
      block_size=8,
      log2_sigma=float(log2_sigma),
      num_levels=s1_levels,
      num_scales=s1_scales,
  )
  wd_ref_map = np.asarray(wd_ref_jax, dtype=np.float32) * float(wd_scale)
  ref_h8, ref_w8 = int(ref_state_init.h8), int(ref_state_init.w8)
  ref_canon_s = (
      tuple(
          int(s_id)
          for s_id in (0, 4, 5, 6, 7)
          if ref_h8 >= ACS_COVERED_BLOCKS_Y[s_id]
          and ref_w8 >= ACS_COVERED_BLOCKS_X[s_id]
      )
      if np.any(ref_state_init.ac_strategy != 0)
      else (0,)
  )
  eval_ref_fn = _get_compiled_stage1_eval(
      int(ref_state_init.height),
      int(ref_state_init.width),
      epf_iters=int(ref_state_init.epf_iters),
      gab=bool(ref_state_init.gab),
      canon_s=ref_canon_s,
      x_qm_scale=int(ref_state_init.x_qm_scale),
      b_qm_scale=int(ref_state_init.b_qm_scale),
  )
  _, _, _, _, _, rate_ref_j = eval_ref_fn(
      srgb_to_xyb(jnp.moveaxis(jnp.asarray(rgb_low_raw, dtype=jnp.float32), -1, 0)),
      jnp.asarray(ref_state_init.ac_strategy, dtype=jnp.int32),
      jnp.float32(ref_state_init.global_scale),
      jnp.float32(ref_state_init.quant_dc),
      jnp.asarray(ref_state_init.raw_quant_field, dtype=jnp.float32),
      jnp.asarray(ref_state_init.ytox_map, dtype=jnp.float32),
      jnp.asarray(ref_state_init.ytob_map, dtype=jnp.float32),
      jnp.float32(ref_state_init.ytox_dc),
      jnp.float32(ref_state_init.ytob_dc),
      jnp.asarray(ref_state_init.epf_sharpness, dtype=jnp.float32),
  )
  rate_ref_low = np.asarray(rate_ref_j, dtype=np.float32)
  # Map low-res block bit rate to full-res 8x8 blocks (amortized by 1 / U^2)
  by_idx = np.minimum(
      np.arange(h8)[:, None] // u, rate_ref_low.shape[0] - 1
  )
  bx_idx = np.minimum(
      np.arange(w8)[None, :] // u, rate_ref_low.shape[1] - 1
  )
  rate_ref_map = rate_ref_low[by_idx, bx_idx] / float(u * u)
  j_ref_map = wd_ref_map + lam_per_px * rate_ref_map

  delta_j_patch = j_main_map - j_ref_map
  fg_adv = -delta_j_patch  # > 0 where full-res main beats low-res reference

  fg_mask_blocks = _regularize_fg_mask_blocks(fg_adv, lam_per_px, u=u)
  fg_mask_dilated = (
      _dilate_bool_grid(fg_mask_blocks, radius=dilate_blocks)
      if np.any(fg_mask_blocks)
      else fg_mask_blocks.copy()
  )
  if np.all(fg_mask_dilated) and not np.all(fg_mask_blocks):
    fg_mask_dilated = fg_mask_blocks.copy()
  if np.all(fg_mask_dilated) and fg_mask_dilated.size > 1:
    min_idx = np.unravel_index(int(np.argmin(fg_adv)), fg_adv.shape)
    fg_mask_dilated = fg_mask_dilated.copy()
    fg_mask_dilated[min_idx] = False

  if h8 * w8 < 256:
    halo_blocks = _dilate_bool_grid(fg_mask_dilated, radius=1)
    split_any = False
    for by in range(h8):
      for bx in range(w8):
        if stage2_anchors[by, bx] and stage2_acs[by, bx] != 0:
          s = int(stage2_acs[by, bx])
          cy, cx = _acs_covered_yx(s)
          if cy > 1 or cx > 1:
            sub_m = fg_mask_dilated[by : by + cy, bx : bx + cx]
            sub_h = halo_blocks[by : by + cy, bx : bx + cx]
            if not (np.all(sub_m) or not np.any(sub_m)) or not (
                np.all(sub_h) or not np.any(sub_h)
            ):
              stage2_acs[by : by + cy, bx : bx + cx] = 0
              stage2_anchors[by : by + cy, bx : bx + cx] = True
              split_any = True
    if split_any:
      main_state.ac_strategy = stage2_acs.copy()
      main_state.ac_strategy_Anchor = stage2_anchors.copy()
      rq_split, epf_s_split, _ = select_quant_and_epf_jax(
          clean_rgb_j, distance=d_main, ac_strategy=stage2_acs
      )
      main_state.raw_quant_field = np.asarray(rq_split, dtype=np.int32)
      main_state.epf_sharpness = np.asarray(epf_s_split, dtype=np.uint8)
      dc_m, ac_m = compute_continuous_dc_ac_jax(
          fg_xyb_src, main_state, apply_inv_gab=True
      )
      main_state.dc_quant = np.asarray(jnp.round(dc_m), dtype=np.int32)
      main_state.quant_ac = np.asarray(jnp.round(ac_m), dtype=np.int32)

  bg_mask_blocks = ~fg_mask_dilated
  bg_known_px = jnp.asarray(
      np.repeat(np.repeat(bg_mask_blocks, 8, axis=0), 8, axis=1)[:h, :w],
      dtype=bool,
  )
  if np.any(fg_mask_dilated):
    smooth_full = _pyramid_harmonic_inpaint_jax(
        clean_rgb_j, bg_known_px, n_iters=8
    )
  else:
    smooth_full = clean_rgb_j
  if has_sensor_grain:
    rgb_low_smooth = downsample_rgb_jax(smooth_full, u)
  else:
    rgb_low_smooth = _project_lowres_rgb_wd_jax(
        smooth_full, u, vgg_params=vgg_params, log2_sigma=float(log2_sigma)
    )

  adv_shift = fg_adv - float(np.median(fg_adv))
  adv_scale = max(float(np.percentile(np.abs(adv_shift), 90.0)), 1e-5)
  fg_prob_blocks = 1.0 / (1.0 + np.exp(-np.clip(adv_shift / adv_scale, -8.0, 8.0)))
  detail_map_01 = jnp.asarray(
      np.repeat(np.repeat(fg_prob_blocks, 8, axis=0), 8, axis=1)[:h, :w],
      dtype=jnp.float32,
  )

  out_info = {
      "fg_mask_blocks": fg_mask_blocks,
      "fg_mask_dilated": fg_mask_dilated,
      "bg_mask_blocks": bg_mask_blocks,
      "block_err": np.maximum(fg_adv, 0.0),
      "spatial_delta_j_map": delta_j_patch,
      "stage1_main_j_map": j_main_map,
      "stage1_ref_j_map": j_ref_map,
      "detail_map_01": np.asarray(detail_map_01, dtype=np.float32),
      "rgb_low_smooth": np.asarray(rgb_low_smooth, dtype=np.float32),
      "rgb_low_raw": np.asarray(rgb_low_raw, dtype=np.float32),
      "stage1_candidates": stage1_cands,
      "stage2_ac_strategy": stage2_acs,
      "stage2_anchors": stage2_anchors,
      "main_state": main_state,
      "main_dc_cont": np.asarray(dc_m, dtype=np.float32),
      "main_ac_cont": np.asarray(ac_m, dtype=np.float32),
  }
  if len(_STAGE1_MULTIRES_CACHE) >= 4:
    _STAGE1_MULTIRES_CACHE.clear()
  _STAGE1_MULTIRES_CACHE[m_cache_key] = copy.deepcopy(out_info)
  return out_info


def compute_detail_mask_jax(
    srgb_hwc: np.ndarray | jax.Array,
    upsampling: int = 2,
    distance: float = 1.0,
    rel_threshold: float = 0.15,
    abs_threshold: float = 2.0e-4,
    dilate_blocks: int = 0,
    rd_lambda: float | None = None,
    vgg_params: Any | None = None,
) -> dict[str, Any]:
  """Computes multi-resolution spatial detail mask via Stage 1 Rate-WD spatial cost analysis (`F3`)."""
  del rel_threshold, abs_threshold
  return _stage1_multires_rate_wd_analysis(
      srgb_hwc,
      upsampling=upsampling,
      distance=distance,
      rd_lambda=rd_lambda,
      dilate_blocks=dilate_blocks,
      vgg_params=vgg_params,
      explore_ac_strategy=False,
  )


def _expand_state_dq_3hw_np(state: JxlVarDCTState) -> np.ndarray:
  """Builds the `(3, h8*8, w8*8)` dequant weight plane matching `state.ac_strategy` & `state.dequant_matrices`."""
  h8, w8 = int(state.h8), int(state.w8)
  default_dq = get_default_dequant_matrices()
  dq_dict = state.dequant_matrices if state.dequant_matrices else default_dq
  w_blk = np.broadcast_to(
      np.asarray(dq_dict.get(0, default_dq[0]), dtype=np.float32)[None, None],
      (h8, w8, 3, 8, 8),
  ).copy()
  acs_np = np.asarray(state.ac_strategy, dtype=np.int32)
  anc_np = np.asarray(state.ac_strategy_Anchor, dtype=bool)
  for s in np.unique(acs_np).tolist():
    s_id = int(s)
    if s_id == 0:
      continue
    cy, cx = _acs_covered_yx(s_id)
    mat_s = np.asarray(dq_dict.get(s_id, default_dq[s_id]), dtype=np.float32)
    if cy == 1 and cx == 1:
      w_blk[acs_np == s_id] = mat_s
    else:
      dq_s = mat_s.reshape(3, cy, 8, cx, 8).transpose(1, 3, 0, 2, 4)
      bys, bxs = np.nonzero(anc_np & (acs_np == s_id))
      for by, bx in zip(bys.tolist(), bxs.tolist()):
        w_blk[by : by + cy, bx : bx + cx] = dq_s
  return w_blk.transpose(2, 0, 3, 1, 4).reshape(3, h8 * 8, w8 * 8).astype(
      np.float32
  )


def _apply_custom_dequant_to_state(
    state: JxlVarDCTState,
    dq_log_scale: np.ndarray | jax.Array | None,
    custom_dequant: bool | str,
    *,
    force_ac_non_default: bool = True,
) -> None:
  """Applies `'none'` (0 DoF), `'compact'` (9 DoF), or `'full_bands'` (33 DoF) `DequantMatrices` to `state`."""
  default_dq = get_default_dequant_matrices()
  used_s = sorted(set(int(x) for x in np.unique(state.ac_strategy)) | {0})
  setattr(state, "_orig_dequant_matrices", default_dq)

  is_none_mode = (
      custom_dequant is False
      or (isinstance(custom_dequant, str) and custom_dequant.lower() == "none")
      or dq_log_scale is None
      or int(np.asarray(dq_log_scale).size) == 0
  )
  if is_none_mode:
    state.dequant_matrices = {
        s: np.asarray(default_dq[s], dtype=np.float32).copy() for s in used_s
    }
    state.dc_quant_weights = np.array(
        DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32
    )
    return

  dq_log = np.asarray(dq_log_scale, dtype=np.float32).ravel().copy()
  if not force_ac_non_default and dq_log.size >= 3:
    dq_log[:3] = np.where(np.abs(dq_log[:3]) < 0.015, 0.0, dq_log[:3])
  if np.all(np.abs(dq_log) < 0.008):
    if force_ac_non_default:
      dq_log[:3] = np.array([0.012, -0.010, 0.012], dtype=np.float32)
    else:
      state.dequant_matrices = {
          s: np.asarray(default_dq[s], dtype=np.float32).copy() for s in used_s
      }
      state.dc_quant_weights = np.array(
          DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32
      )
      return

  raw_mats, raw_dc = _eval_dq_log_vec_np(dq_log, used_s, default_dq)
  _, proj_dc = project_custom_dequant_matrices(
      {0: raw_mats[0]}, dc_quant=raw_dc
  )
  state.dequant_matrices = raw_mats
  state.dc_quant_weights = np.asarray(proj_dc, dtype=np.float32)
  rt_mats, rt_dc = roundtrip_custom_dequant_matrices(state)
  state.dequant_matrices = {
      s: np.asarray(rt_mats[s], dtype=np.float32) for s in rt_mats
  }
  state.dc_quant_weights = np.asarray(rt_dc, dtype=np.float32)


def _couple_candidate_ac_dc_np(
    ac_np: np.ndarray,
    dc_np: np.ndarray,
    cand_state: JxlVarDCTState,
    base_state: JxlVarDCTState,
    dq_log_cand: np.ndarray | jax.Array | None,
    dq_log_base: np.ndarray | jax.Array | None,
    use_custom_dq: bool,
    *,
    rq_cont: np.ndarray | jax.Array | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Couples unquantized DCT `(ac_np, dc_np)` with continuous `rq`, `DequantMatrices`, and CfL deltas."""
  cached_topo = getattr(base_state, "_cached_topo_np", None)
  if cached_topo is None:
    cached_topo = StaticVarDCTTopology(base_state)
    setattr(base_state, "_cached_topo_np", cached_topo)
  tl_by_np, tl_bx_np = cached_topo.tl_by_grid_np, cached_topo.tl_bx_grid_np

  rq_src = (
      np.clip(np.asarray(rq_cont, dtype=np.float32), 1.0, 256.0)
      if rq_cont is not None
      else np.clip(cand_state.raw_quant_field, 1, 256).astype(np.float32)
  )
  rq_tl = rq_src[tl_by_np, tl_bx_np]
  rq_base_tl = np.clip(base_state.raw_quant_field, 1, 256)[
      tl_by_np, tl_bx_np
  ].astype(np.float32)
  rq_ratio_tl = rq_tl / np.maximum(rq_base_tl, 1.0)
  rq_ratio_px = np.repeat(np.repeat(rq_ratio_tl, 8, axis=0), 8, axis=1)[
      None, :, :
  ]

  w_cur = _expand_state_dq_3hw_np(cand_state)
  if (
      use_custom_dq
      and dq_log_cand is not None
      and dq_log_base is not None
      and int(np.asarray(dq_log_cand).size) > 0
  ):
    w_base = _expand_state_dq_3hw_np(base_state)
    dc_w_cur = np.asarray(
        cand_state.dc_quant_weights
        if getattr(cand_state, "dc_quant_weights", None) is not None
        else DEFAULT_DC_QUANT_WEIGHTS,
        dtype=np.float32,
    )
    dc_w_base = np.asarray(
        base_state.dc_quant_weights
        if getattr(base_state, "dc_quant_weights", None) is not None
        else DEFAULT_DC_QUANT_WEIGHTS,
        dtype=np.float32,
    )
    u_ac_np = ac_np * rq_ratio_px * (w_base / np.maximum(w_cur, 1e-12))
    u_dc_np = dc_np * (dc_w_base / np.maximum(dc_w_cur, 1e-12))[:, None, None]
  else:
    u_ac_np = ac_np * rq_ratio_px
    u_dc_np = dc_np

  # Counter-adjust X and B AC coefficients for any tile CfL changes (ytox_map, ytob_map)
  dx_tile = (
      cand_state.ytox_map.astype(np.float32)
      - base_state.ytox_map.astype(np.float32)
  ) / float(cand_state.color_factor)
  db_tile = (
      cand_state.ytob_map.astype(np.float32)
      - base_state.ytob_map.astype(np.float32)
  ) / float(cand_state.color_factor)
  if np.any(dx_tile != 0.0) or np.any(db_tile != 0.0):
    h8_c, w8_c = int(cand_state.h8), int(cand_state.w8)
    dx_px = np.repeat(np.repeat(dx_tile, 64, axis=0), 64, axis=1)[
        : h8_c * 8, : w8_c * 8
    ]
    db_px = np.repeat(np.repeat(db_tile, 64, axis=0), 64, axis=1)[
        : h8_c * 8, : w8_c * 8
    ]
    x_qm_mul = float(1.25 ** (float(cand_state.x_qm_scale) - 2.0))
    b_qm_mul = float(1.25 ** (float(cand_state.b_qm_scale) - 2.0))
    u_ac_masked = mask_llf_numpy(u_ac_np, cached_topo)
    u_y = u_ac_masked[1]
    u_ac_np = u_ac_np.copy()
    u_ac_np[0] = u_ac_np[0] - dx_px * u_y * (
        (w_cur[1] * x_qm_mul) / np.maximum(w_cur[0], 1e-12)
    )
    u_ac_np[2] = u_ac_np[2] - db_px * u_y * (
        (w_cur[1] * b_qm_mul) / np.maximum(w_cur[2], 1e-12)
    )
  return u_ac_np, u_dc_np


def _build_ac_mask_from_block_mask(
    fg_mask_blocks: np.ndarray, state: JxlVarDCTState
) -> tuple[jax.Array, jax.Array]:
  """Builds `(fg_ac_mask_j, fg_dc_mask_j)` respecting multi-block `AcStrategy` transforms."""
  mask_blk = np.asarray(fg_mask_blocks, dtype=bool).copy()
  acs_np = np.asarray(state.ac_strategy, dtype=np.int32)
  anc_np = np.asarray(state.ac_strategy_Anchor, dtype=bool)
  is_multi_anc = anc_np & ~((acs_np <= 3) | ((acs_np >= 12) & (acs_np <= 17)))
  if np.any(is_multi_anc):
    m_bys, m_bxs = np.nonzero(is_multi_anc)
    for by, bx in zip(m_bys.tolist(), m_bxs.tolist()):
      s = int(acs_np[by, bx])
      cy, cx = _acs_covered_yx(s)
      if np.any(mask_blk[by : by + cy, bx : bx + cx]):
        mask_blk[by : by + cy, bx : bx + cx] = True
  ac_mask_np = np.repeat(
      np.repeat(mask_blk.astype(np.float32), 8, axis=0), 8, axis=1
  )
  return (
      jnp.asarray(ac_mask_np)[None, :, :],
      jnp.asarray(mask_blk.astype(np.float32))[None, :, :],
  )



def _finalize_encode_result(
    *,
    mode: str,
    upsampling: int,
    dim_shift: int,
    blend_mode: int | None,
    alpha_blend_mode: int | None = None,
    jxl_bytes: bytes,
    bpp: float,
    wd: float,
    rd_cost: float,
    recon_rgb: np.ndarray,
    target_rgb_01: np.ndarray,
    enable_noise: bool,
    opt_noise_lut: np.ndarray | None,
    patches: list[PatchEntry],
    ref_state: JxlVarDCTState | None,
    main_state: JxlVarDCTState | None,
    state: JxlVarDCTState | None,
    alpha_01: np.ndarray | None,
    alpha_u8: np.ndarray | None,
    stats: dict[str, Any],
    history: list[dict[str, float]],
    last_grads: dict[str, Any] | None,
    verify_decoders: bool,
    stage1_candidates: dict[int, dict[str, Any]] | None = None,
    stage2_ac_strategy: np.ndarray | None = None,
    stage2_true_bpp: float = 0.0,
    stage2_wd: float = 0.0,
    stage2_true_j: float = 0.0,
    stage3_true_bpp: float | None = None,
    stage3_wd: float | None = None,
    stage3_true_j: float | None = None,
    custom_dequant_enabled: bool = False,
    spatial_delta_j_map: np.ndarray | None = None,
    stage1_main_j_map: np.ndarray | None = None,
    stage1_ref_j_map: np.ndarray | None = None,
) -> CompositeOptimizationResult:
  """Materializes a verified `CompositeOptimizationResult` (`EncodeResult`)."""
  recon_rgb = np.asarray(recon_rgb, dtype=np.float32)
  if alpha_01 is not None:
    alpha_01 = np.asarray(alpha_01, dtype=np.float32)
  if alpha_u8 is not None:
    alpha_u8 = np.asarray(alpha_u8, dtype=np.uint8)
  if last_grads is not None:
    last_grads = jax.tree_util.tree_map(
        lambda x: np.asarray(x) if isinstance(x, jax.Array) else x,
        last_grads,
    )
  psnr_jax_vs_orig = compute_psnr(recon_rgb, target_rgb_01, peak=1.0)
  if verify_decoders:
    ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
    djxl_rgb = ver["rgb_djxl"].astype(np.float32) / 255.0
    psnr_vs_djxl = compute_psnr(recon_rgb, djxl_rgb, peak=1.0)
    psnr_djxl_vs_jxlrs = float(ver["cross_decoder_psnr"])
    psnr_db = compute_psnr(djxl_rgb, target_rgb_01, peak=1.0)
    header_info = dict(ver["inspect"])
  else:
    header_info = dict(inspect_jxl(jxl_bytes))
    psnr_vs_djxl = psnr_jax_vs_orig
    psnr_djxl_vs_jxlrs = 99.0
    psnr_db = psnr_jax_vs_orig

  if "extra_channel_bits" not in header_info:
    header_info["extra_channel_bits"] = [
        int(ec.get("bits_per_sample", 8))
        for ec in header_info.get("extra_channels", [])
    ]

  frames_info = header_info.get("frames", [])
  has_noise_hdr = any(bool(f.get("has_noise", False)) for f in frames_info)
  has_noise = bool(enable_noise and has_noise_hdr)
  noise_lut_list = (
      [float(x) for x in np.asarray(opt_noise_lut).ravel()]
      if opt_noise_lut is not None
      else None
  )

  s3_bpp = float(bpp if stage3_true_bpp is None else stage3_true_bpp)
  s3_wd = float(wd if stage3_wd is None else stage3_wd)
  s3_j = float(rd_cost if stage3_true_j is None else stage3_true_j)

  return CompositeOptimizationResult(
      jxl_bytes=jxl_bytes,
      bpp=float(bpp),
      wd=float(wd),
      psnr_vs_djxl=float(psnr_vs_djxl),
      psnr_jax_vs_djxl=float(psnr_vs_djxl),
      psnr_djxl_vs_jxlrs=float(psnr_djxl_vs_jxlrs),
      mode_selected=mode,
      upsampling=int(upsampling),
      dim_shift=int(dim_shift),
      has_noise=has_noise,
      noise_lut=noise_lut_list,
      num_patches=len(patches),
      header_info=header_info,
      psnr_db=float(psnr_db),
      cross_decoder_psnr=float(psnr_djxl_vs_jxlrs),
      mode_used=mode,
      mode=mode,
      num_bytes=len(jxl_bytes),
      lagrangian=float(rd_cost),
      rd_cost=float(rd_cost),
      psnr=float(psnr_jax_vs_orig),
      blend_mode=blend_mode,
      alpha_blend_mode=alpha_blend_mode,
      ref_state=ref_state,
      main_state=main_state,
      state=state,
      patches=list(patches),
      alpha_01=alpha_01,
      alpha_u8=alpha_u8,
      recon_rgb=recon_rgb,
      recon_jax_rgb=recon_rgb,
      stats=stats,
      history=history,
      last_grads=last_grads,
      stage1_candidates=(
          stage1_candidates if stage1_candidates is not None else {}
      ),
      stage2_ac_strategy=stage2_ac_strategy,
      stage2_true_bpp=float(stage2_true_bpp),
      stage2_wd=float(stage2_wd),
      stage2_true_j=float(stage2_true_j),
      stage3_true_bpp=s3_bpp,
      stage3_wd=s3_wd,
      stage3_true_j=s3_j,
      custom_dequant_enabled=bool(custom_dequant_enabled),
      spatial_delta_j_map=spatial_delta_j_map,
      stage1_main_j_map=stage1_main_j_map,
      stage1_ref_j_map=stage1_ref_j_map,
  )



def _init_ref_state_with_stage2_acs(
    r_rgb: np.ndarray | jax.Array,
    *,
    d_ref: float,
    eff_lambda: float,
    enable_noise: bool,
    vgg_params: Any,
    log2_sigma: float,
    upsampling: int = 1,
) -> JxlVarDCTState:
  """Initializes `ReferenceOnly` frame state with Stage 1+2 `AcStrategy` selection when >= 16x16."""
  r_state = _init_frame_state_with_aq_epf_cfl(
      r_rgb, distance=d_ref, gab=None, enable_noise=enable_noise
  )
  if int(r_state.height) >= 16 and int(r_state.width) >= 16:
    r_h8, r_w8 = int(r_state.h8), int(r_state.w8)
    ref_cands = tuple(
        s_id
        for s_id in (0, 4, 5, 6, 7)
        if r_h8 >= ACS_COVERED_BLOCKS_Y[s_id]
        and r_w8 >= ACS_COVERED_BLOCKS_X[s_id]
    )
    ref_lam = (
        float(eff_lambda) / float(max(int(upsampling) * int(upsampling), 1))
        if enable_noise
        else float(eff_lambda)
    )
    s1_r = stage1_explore_ac_candidates_compositing(
        r_rgb,
        distance=d_ref,
        rd_lambda=ref_lam,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        candidate_strategies=ref_cands,
    )
    acs_r, anc_r = stage2_select_non_overlapping_ac_strategy_compositing(
        s1_r, int(r_state.h8), int(r_state.w8)
    )
    if np.any(acs_r != 0):
      r_state.ac_strategy = acs_r.copy()
      r_state.ac_strategy_Anchor = anc_r.copy()
      used_r = 0
      for s_u in np.unique(acs_r):
        used_r |= 1 << int(s_u)
      r_state.used_acs = used_r
      rq_r, epf_r, _ = select_quant_and_epf_jax(
          r_rgb, distance=d_ref, ac_strategy=acs_r
      )
      r_state.raw_quant_field = np.asarray(rq_r, dtype=np.int32)
      r_state.epf_sharpness = np.asarray(epf_r, dtype=np.uint8)
      r_xyb_tmp = srgb_to_xyb(
          jnp.moveaxis(jnp.asarray(r_rgb, dtype=jnp.float32), -1, 0)
      )
      r_dc_tmp, r_ac_tmp = compute_continuous_dc_ac_jax(
          r_xyb_tmp, r_state, apply_inv_gab=True, init_mode="dct"
      )
      r_state.dc_quant = np.round(np.asarray(r_dc_tmp)).astype(np.int32)
      r_state.quant_ac = np.round(np.asarray(r_ac_tmp)).astype(np.int32)
  return r_state


def optimize_patch_composite(
    rgb_u8: np.ndarray | jax.Array | None = None,
    upsampling: int | str = 2,
    blend_mode: int | str = "auto",
    distance: float = 1.0,
    ref_distance: float | None = None,
    lambda_rd: float | None = None,
    rd_lambda: float | None = None,
    steps: int | None = None,
    num_steps: int | None = None,
    lr_ac: float = 0.03,
    lr_dc: float = 0.01,
    lr_noise: float = 0.005,
    lr_dequant: float = 0.008,
    enable_noise: bool | str | None = False,
    custom_dequant: bool | str | None = None,
    enable_custom_dequant: bool | str | None = None,
    optimize_dequant_matrices: bool | str | None = None,
    use_custom_dequant: bool | str | None = None,
    verify_decoders: bool = True,
    patches: Sequence[PatchEntry] | None = None,
    vgg_params: Any | None = None,
    log2_sigma: float = 1.0,
    effort: int = 7,
    candidate_upsamplings: Sequence[int] = (2, 4, 8),
    init_mode: str = "dct",
    seed: int = 0,
    *,
    srgb_hwc: np.ndarray | jax.Array | None = None,
) -> CompositeOptimizationResult:
  """Three-stage multi-resolution XYB `PatchDictionary` composite Rate-WD optimizer."""
  img_in = rgb_u8 if rgb_u8 is not None else srgb_hwc
  if img_in is None:
    raise ValueError("Either rgb_u8 or srgb_hwc must be provided.")
  init_mode_norm = _normalize_init_mode(init_mode)
  default_steps = 60 if init_mode_norm in ("zero", "random") else 24
  eff_lambda = float(
      rd_lambda
      if rd_lambda is not None
      else (
          lambda_rd
          if lambda_rd is not None
          else calibrate_rd_lambda(distance)
      )
  )
  eff_steps = int(
      num_steps
      if num_steps is not None
      else (steps if steps is not None else default_steps)
  )
  dq_spec = _resolve_custom_dequant_spec(
      custom_dequant=custom_dequant,
      enable_custom_dequant=enable_custom_dequant,
      optimize_dequant_matrices=optimize_dequant_matrices,
      use_custom_dequant=use_custom_dequant,
  )

  parsed_bm = _parse_blend_mode(blend_mode)
  if parsed_bm == "auto":
    res_rep = optimize_patch_composite(
        rgb_u8=img_in,
        upsampling=upsampling,
        blend_mode=PATCH_BLEND_REPLACE,
        distance=distance,
        ref_distance=ref_distance,
        lambda_rd=eff_lambda,
        steps=eff_steps,
        lr_ac=lr_ac,
        lr_dc=lr_dc,
        lr_noise=lr_noise,
        lr_dequant=lr_dequant,
        enable_noise=enable_noise,
        custom_dequant=dq_spec,
        verify_decoders=verify_decoders,
        patches=patches,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        effort=effort,
        candidate_upsamplings=candidate_upsamplings,
        init_mode=init_mode_norm,
        seed=seed,
    )
    res_add = optimize_patch_composite(
        rgb_u8=img_in,
        upsampling=upsampling,
        blend_mode=PATCH_BLEND_ADD,
        distance=distance,
        ref_distance=ref_distance,
        lambda_rd=eff_lambda,
        steps=eff_steps,
        lr_ac=lr_ac,
        lr_dc=lr_dc,
        lr_noise=lr_noise,
        lr_dequant=lr_dequant,
        enable_noise=enable_noise,
        custom_dequant=dq_spec,
        verify_decoders=verify_decoders,
        patches=patches,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        effort=effort,
        candidate_upsamplings=candidate_upsamplings,
        init_mode=init_mode_norm,
        seed=seed,
    )
    return res_rep if res_rep.rd_cost <= res_add.rd_cost else res_add

  b_mode = int(parsed_bm)
  arr = _normalize_srgb_hwc(img_in)
  h, w, _ = arr.shape
  h8, w8 = (h + 7) // 8, (w + 7) // 8
  target_j = jnp.asarray(arr, dtype=jnp.float32)
  target_xyb = srgb_to_xyb(jnp.moveaxis(target_j, -1, 0))

  if vgg_params is None:
    vgg_params = load_vgg16_params()
  wd_val_and_grad = _get_compiled_wd_val_and_grad(num_levels=2, num_scales=3)
  wd_only = _get_compiled_wd_only(num_levels=2, num_scales=3)

  if isinstance(upsampling, str) and upsampling.strip().lower() == "auto":
    u_cands = [
        int(uc)
        for uc in candidate_upsamplings
        if int(uc) in (2, 4, 8) and h >= int(uc) and w >= int(uc)
    ]
    if not u_cands:
      u_cands = [2]
  else:
    if int(upsampling) not in (2, 4, 8):
      raise ValueError(
          f"upsampling must be in {{2, 4, 8, 'auto'}}, got {upsampling}"
      )
    u_cands = [int(upsampling)]

  dilate = 1 if (h >= 128 and w >= 128) else 0
  d_main = float(distance)
  enable_noise, sensor_lut_np = _resolve_enable_noise(
      enable_noise, target_xyb, d_main, h, w
  )
  has_sensor_grain = bool(
      enable_noise and float(np.max(sensor_lut_np)) >= 0.0020
  )
  m_dq_mode = select_dequant_mode_for_frame(
      h, w, d_main, dq_spec, is_reference=False
  )
  m_use_dq = m_dq_mode != "none"
  dq_log_init = _init_dq_log_vec(m_dq_mode, is_reference=False)
  noise_lut_init = (
      estimate_noise_lut_jax(target_xyb)
      if enable_noise
      else jnp.zeros((8,), dtype=jnp.float32)
  )
  s2_noise_lut = (
      np.clip(
          np.asarray(noise_lut_init, dtype=np.float32), 2.0 / 1024.0, 0.5
      )
      if enable_noise
      else None
  )
  fg_xyb_main = (
      _shrink_sensor_grain_xyb_jax(target_xyb, noise_lut_init)
      if has_sensor_grain
      else target_xyb
  )

  # Stage 1 & Stage 2: Evaluate candidate upsampling factors U in {2, 4, 8}
  # and make discrete selection based on Stage 2 Rate-WD Lagrangian.
  def _build_stage2_for_u(u_val: int) -> dict[str, Any]:
    d_ref_u = _calibrate_ref_distance(
        d_main,
        eff_lambda,
        ref_distance,
        upsampling=u_val,
        enable_noise=enable_noise,
    )
    s1_info = _stage1_multires_rate_wd_analysis(
        arr,
        upsampling=u_val,
        distance=distance,
        ref_distance=ref_distance,
        rd_lambda=eff_lambda,
        dilate_blocks=dilate,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        explore_ac_strategy=True,
        enable_noise=enable_noise,
        noise_lut=s2_noise_lut,
    )
    fg_m_dil = s1_info["fg_mask_dilated"]
    bg_m_blk = s1_info["bg_mask_blocks"]
    s2_acs_u = s1_info["stage2_ac_strategy"]

    r_rgb = s1_info["rgb_low_smooth"]
    r_state = _init_ref_state_with_stage2_acs(
        r_rgb,
        d_ref=d_ref_u,
        eff_lambda=eff_lambda,
        enable_noise=enable_noise,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        upsampling=u_val,
    )
    r_dq_mode_u = select_dequant_mode_for_frame(
        int(r_state.height),
        int(r_state.width),
        d_ref_u,
        dq_spec,
        is_reference=True,
    )
    r_use_dq_u = r_dq_mode_u != "none"
    r_dq_init_u = _init_dq_log_vec(r_dq_mode_u, is_reference=True)
    _apply_custom_dequant_to_state(
        r_state,
        r_dq_init_u,
        custom_dequant=r_dq_mode_u,
        force_ac_non_default=False,
    )
    r_xyb_low = srgb_to_xyb(jnp.moveaxis(jnp.asarray(r_rgb), -1, 0))
    r_dc_dct, r_ac_dct = compute_continuous_dc_ac_jax(
        r_xyb_low, r_state, apply_inv_gab=True, init_mode="dct"
    )

    if b_mode == PATCH_BLEND_REPLACE:
      if patches is None:
        p_list = extract_patch_rectangles(
            bg_m_blk, height=h, width=w, blend_mode=PATCH_BLEND_REPLACE
        )
        if not p_list:
          p_list = [
              PatchEntry(
                  ref=0,
                  x0=0,
                  y0=0,
                  xsize=min(8, w),
                  ysize=min(8, h),
                  x=0,
                  y=0,
                  blend_mode=PATCH_BLEND_REPLACE,
              )
          ]
      else:
        p_list = list(patches)
      _, w_main_tmp = _patches_to_weights(p_list, h, w)
      w_main_np = np.asarray(w_main_tmp, dtype=np.float32)[0]
      w_main_pad = np.pad(
          w_main_np, ((0, h8 * 8 - h), (0, w8 * 8 - w)), mode="edge"
      )
      fg_m_eff = fg_m_dil | (
          w_main_pad.reshape(h8, 8, w8, 8).max(axis=(1, 3)) > 0.0
      )
      m_state = copy.deepcopy(s1_info["main_state"])
      halo_rad = (
          0
          if (int(m_state.epf_iters) == 0 and not bool(m_state.gab))
          else 1
      )
      dc_halo = _dilate_bool_grid(fg_m_eff, radius=halo_rad)
      _pack_background_acs_and_cfl(m_state, fg_m_eff, dc_halo)
      for by_h in range(h8):
        for bx_h in range(w8):
          if m_state.ac_strategy_Anchor[by_h, bx_h]:
            cy_h, cx_h = _acs_covered_yx(int(m_state.ac_strategy[by_h, bx_h]))
            if (cy_h > 1 or cx_h > 1) and np.any(
                dc_halo[by_h : by_h + cy_h, bx_h : bx_h + cx_h]
            ):
              dc_halo[by_h : by_h + cy_h, bx_h : bx_h + cx_h] = True
      if enable_noise:
        m_state.ytox_dc = 0
        m_state.ytob_dc = 0
      rq_fg_med = int(
          np.median(
              m_state.raw_quant_field[fg_m_eff]
              if np.any(fg_m_eff)
              else m_state.raw_quant_field
          )
      )
      m_state.raw_quant_field[~fg_m_eff] = rq_fg_med
      m_state.raw_quant_field = broadcast_quant_field_to_ac_strategy(
          m_state.raw_quant_field, m_state.ac_strategy
      )
      epf_fg_med = np.uint8(
          np.median(
              m_state.epf_sharpness[fg_m_eff]
              if np.any(fg_m_eff)
              else m_state.epf_sharpness
          )
      )
      m_state.epf_sharpness[~fg_m_eff] = epf_fg_med
      _apply_custom_dequant_to_state(
          m_state, dq_log_init, custom_dequant=m_dq_mode
      )
      if h8 * w8 >= 256 and np.any(dc_halo) and not np.all(dc_halo):
        halo_px = np.repeat(np.repeat(dc_halo, 8, axis=0), 8, axis=1)[:h, :w]
        xyb_smooth_hwc = _pyramid_harmonic_inpaint_jax(
            jnp.moveaxis(fg_xyb_main, 0, -1),
            jnp.asarray(halo_px, dtype=bool),
            n_iters=6,
        )
        xyb_dc_src = jnp.moveaxis(xyb_smooth_hwc, -1, 0)
      else:
        xyb_dc_src = fg_xyb_main
      m_dc_dct, m_ac_dct = compute_continuous_dc_ac_jax(
          xyb_dc_src, m_state, apply_inv_gab=True, init_mode="dct"
      )
      fg_ac_m_j, _ = _build_ac_mask_from_block_mask(fg_m_eff, m_state)
      if np.any(dc_halo) and not np.all(dc_halo):
        fg_dc_m_j = jnp.broadcast_to(
            jnp.asarray(dc_halo, dtype=jnp.float32)[None, :, :], (3, h8, w8)
        )
        if h8 * w8 >= 256:
          dc_bg_u = (1.0 - fg_dc_m_j) * jnp.round(m_dc_dct)
        else:
          dc_np_tmp = np.asarray(m_dc_dct, dtype=np.float32)
          dct8_halo = dc_halo & (m_state.ac_strategy == 0)
          ref_dc_mask_med = (
              dct8_halo
              if np.any(dct8_halo)
              else (dc_halo & m_state.ac_strategy_Anchor)
          )
          med_dc = np.round(
              np.median(dc_np_tmp[:, ref_dc_mask_med], axis=1)
          ).astype(np.float32)
          dc_bg_u = (1.0 - fg_dc_m_j) * jnp.asarray(med_dc)[:, None, None]
      elif not np.any(dc_halo):
        fg_dc_m_j = jnp.zeros((3, h8, w8), dtype=jnp.float32)
        dc_bg_u = jnp.zeros((3, h8, w8), dtype=jnp.float32)
      else:
        fg_dc_m_j = jnp.ones((3, h8, w8), dtype=jnp.float32)
        dc_bg_u = jnp.zeros((3, h8, w8), dtype=jnp.float32)
      res_xyb_u = fg_xyb_main
    else:
      if patches is None:
        p_list = [
            PatchEntry(
                ref=0,
                x0=0,
                y0=0,
                xsize=w,
                ysize=h,
                x=0,
                y=0,
                blend_mode=PATCH_BLEND_ADD,
            )
        ]
      else:
        p_list = list(patches)
      r_xyb_rec = _fast_reconstruct_xyb(r_state, include_noise=False)
      r_xyb_ups = upsample_jxl(r_xyb_rec, u_val, out_hw=(h, w))
      res_xyb_u = fg_xyb_main - r_xyb_ups
      m_state, m_dc_dct, m_ac_dct = init_state_from_xyb_jax(
          res_xyb_u,
          distance=d_main,
          epf_iters=None,
          gab=None,
          enable_noise=enable_noise,
      )
      m_state.ac_strategy = s2_acs_u.copy()
      m_state.ac_strategy_Anchor = s1_info["stage2_anchors"].copy()
      m_state.raw_quant_field[~fg_m_dil] = int(
          np.median(m_state.raw_quant_field)
      )
      m_state.raw_quant_field = broadcast_quant_field_to_ac_strategy(
          m_state.raw_quant_field, m_state.ac_strategy
      )
      _apply_custom_dequant_to_state(
          m_state, dq_log_init, custom_dequant=m_dq_mode
      )
      m_dc_dct, m_ac_dct = compute_continuous_dc_ac_jax(
          res_xyb_u, m_state, apply_inv_gab=True, init_mode="dct"
      )
      fg_ac_m_raw, _ = _build_ac_mask_from_block_mask(fg_m_dil, m_state)
      ac_abs = jnp.abs(m_ac_dct)
      fg_ac_m_j = jnp.where(
          (fg_ac_m_raw > 0.5) | (ac_abs > 1.15), 1.0, 0.0
      ).astype(jnp.float32)
      fg_dc_m_j = jnp.ones((3, h8, w8), dtype=jnp.float32)
      dc_bg_u = jnp.zeros((3, h8, w8), dtype=jnp.float32)

    s2_r = copy.deepcopy(r_state)
    s2_m = copy.deepcopy(m_state)
    s2_r.quant_ac = np.round(np.asarray(r_ac_dct)).astype(np.int32)
    s2_r.dc_quant = np.round(np.asarray(r_dc_dct)).astype(np.int32)
    s2_m.quant_ac = np.round(np.asarray(m_ac_dct * fg_ac_m_j)).astype(np.int32)
    s2_m.dc_quant = np.round(
        np.asarray(m_dc_dct * fg_dc_m_j + dc_bg_u)
    ).astype(np.int32)
    s2_jb, _ = encode_patches_composite(
        ref_state=s2_r,
        main_state=s2_m,
        upsampling=u_val,
        patches=p_list,
        effort=effort,
        noise_lut=s2_noise_lut,
    )
    s2_bpp_u = (len(s2_jb) * 8.0) / float(h * w)
    s2_rec_u = _fast_reconstruct_patch_composite(
        s2_r,
        s2_m,
        u_val,
        p_list,
        include_noise=enable_noise,
        noise_lut=s2_noise_lut,
    )
    s2_wd_u = float(
        wd_only(
            s2_rec_u,
            target_j,
            vgg_params,
            float(log2_sigma),
            num_levels=2,
            num_scales=3,
        )
    )
    s2_j_u = s2_wd_u + float(eff_lambda) * s2_bpp_u
    return {
        "u": u_val,
        "stage1_info": s1_info,
        "ref_state": r_state,
        "main_state": m_state,
        "r_dq_mode": r_dq_mode_u,
        "r_use_dq": r_use_dq_u,
        "r_dq_init": r_dq_init_u,
        "ref_xyb_low": r_xyb_low,
        "main_target_xyb": res_xyb_u,
        "ref_dc_cont": r_dc_dct,
        "ref_ac_cont": r_ac_dct,
        "main_dc_cont": m_dc_dct,
        "main_ac_cont": m_ac_dct,
        "patches_list": p_list,
        "fg_ac_mask_j": fg_ac_m_j,
        "fg_dc_mask_j": fg_dc_m_j,
        "dc_bg_j": dc_bg_u,
        "stage2_true_bpp": s2_bpp_u,
        "stage2_wd": s2_wd_u,
        "stage2_true_j": s2_j_u,
    }

  stage2_u_evals = [_build_stage2_for_u(uc) for uc in u_cands]
  best_s2_u = min(stage2_u_evals, key=lambda d: d["stage2_true_j"])
  u = int(best_s2_u["u"])
  stage1_info = best_s2_u["stage1_info"]
  stage1_candidates = stage1_info["stage1_candidates"]
  stage2_acs = stage1_info["stage2_ac_strategy"]
  ref_state = best_s2_u["ref_state"]
  main_state = best_s2_u["main_state"]
  r_dq_mode = str(best_s2_u["r_dq_mode"])
  r_use_dq = bool(best_s2_u["r_use_dq"])
  dq_log_init_ref = best_s2_u["r_dq_init"]
  use_custom_dq = bool(m_use_dq or r_use_dq)
  patches_list = best_s2_u["patches_list"]
  fg_ac_mask_j = best_s2_u["fg_ac_mask_j"]
  fg_dc_mask_j = best_s2_u["fg_dc_mask_j"]
  dc_bg_j = best_s2_u["dc_bg_j"]
  stage2_true_bpp = best_s2_u["stage2_true_bpp"]
  stage2_wd = best_s2_u["stage2_wd"]
  stage2_true_j = best_s2_u["stage2_true_j"]

  if init_mode_norm == "dct":
    ref_dc_cont = best_s2_u["ref_dc_cont"]
    ref_ac_cont = best_s2_u["ref_ac_cont"]
    main_dc_cont = best_s2_u["main_dc_cont"]
    main_ac_cont = best_s2_u["main_ac_cont"]
  else:
    dc_bg_j = jnp.zeros_like(dc_bg_j)
    ref_dc_cont, ref_ac_cont = compute_continuous_dc_ac_jax(
        best_s2_u["ref_xyb_low"],
        ref_state,
        apply_inv_gab=True,
        init_mode=init_mode_norm,
        seed=seed,
    )
    main_dc_cont, main_ac_cont = compute_continuous_dc_ac_jax(
        best_s2_u["main_target_xyb"],
        main_state,
        apply_inv_gab=True,
        init_mode=init_mode_norm,
        seed=seed + 1,
    )
    s2_ref = copy.deepcopy(ref_state)
    s2_main = copy.deepcopy(main_state)
    s2_ref.quant_ac = np.round(np.asarray(ref_ac_cont)).astype(np.int32)
    s2_ref.dc_quant = np.round(np.asarray(ref_dc_cont)).astype(np.int32)
    s2_main.quant_ac = np.round(
        np.asarray(main_ac_cont * fg_ac_mask_j)
    ).astype(np.int32)
    s2_main.dc_quant = np.round(
        np.asarray(main_dc_cont * fg_dc_mask_j + dc_bg_j)
    ).astype(np.int32)
    s2_jxl_bytes, _ = encode_patches_composite(
        ref_state=s2_ref,
        main_state=s2_main,
        upsampling=u,
        patches=patches_list,
        effort=effort,
        noise_lut=s2_noise_lut,
    )
    stage2_true_bpp = (len(s2_jxl_bytes) * 8.0) / float(h * w)
    s2_rec = _fast_reconstruct_patch_composite(
        s2_ref,
        s2_main,
        u,
        patches_list,
        include_noise=enable_noise,
        noise_lut=s2_noise_lut,
    )
    stage2_wd = float(
        wd_only(
            s2_rec,
            target_j,
            vgg_params,
            float(log2_sigma),
            num_levels=2,
            num_scales=3,
        )
    )
    stage2_true_j = stage2_wd + float(eff_lambda) * stage2_true_bpp

  # Stage 3: Joint continuous Rate-WD refinement
  ref_fwd_fn, ref_vjp_fn = _get_frame_kernels(ref_state, enable_noise=False)
  main_fwd_fn, main_vjp_fn = _get_frame_kernels(main_state, enable_noise=False)
  combine_fn, combine_grad_fn = _get_patch_combine_kernels(
      h, w, u, enable_noise=enable_noise
  )
  w_ref_j, w_main_j = _patches_to_weights(patches_list, h, w)
  patch_dict_bits = float(soft_patch_dictionary_bits(patches_list, upsampling=u))

  ref_rq = jnp.asarray(ref_state.raw_quant_field, dtype=jnp.float32)
  ref_yx = jnp.asarray(ref_state.ytox_map, dtype=jnp.float32)
  ref_yb = jnp.asarray(ref_state.ytob_map, dtype=jnp.float32)
  ref_gs = jnp.float32(ref_state.global_scale)
  ref_qdc = jnp.float32(ref_state.quant_dc)
  ref_ac_mask = jnp.ones_like(ref_ac_cont)
  ref_dc_mask = jnp.ones_like(ref_dc_cont)
  ref_dc_bg = jnp.zeros_like(ref_dc_cont)

  main_rq = jnp.asarray(main_state.raw_quant_field, dtype=jnp.float32)
  main_yx = jnp.asarray(main_state.ytox_map, dtype=jnp.float32)
  main_yb = jnp.asarray(main_state.ytob_map, dtype=jnp.float32)
  main_gs = jnp.float32(main_state.global_scale)
  main_qdc = jnp.float32(main_state.quant_dc)
  lam_per_px = jnp.float32(float(eff_lambda) / float(h * w))
  r_custom_dq_w = jnp.float32(1.0 if r_use_dq else 0.0)
  m_custom_dq_w = jnp.float32(1.0 if m_use_dq else 0.0)

  params = {
      "ref": {
          "ac": ref_ac_cont,
          "dc": ref_dc_cont,
          "noise_lut": jnp.zeros((8,), dtype=jnp.float32),
          "rq_delta": jnp.zeros_like(ref_rq),
          "yx_delta": jnp.zeros_like(ref_yx),
          "yb_delta": jnp.zeros_like(ref_yb),
          "dq_log_scale": dq_log_init_ref,
      },
      "main": {
          "ac": main_ac_cont * fg_ac_mask_j,
          "dc": main_dc_cont * fg_dc_mask_j + dc_bg_j,
          "noise_lut": noise_lut_init,
          "rq_delta": jnp.zeros_like(main_rq),
          "yx_delta": jnp.zeros_like(main_yx),
          "yb_delta": jnp.zeros_like(main_yb),
          "dq_log_scale": dq_log_init,
      },
  }

  is_zero_or_rand = init_mode_norm in ("zero", "random")
  eff_lr_ac = (
      0.18
      if (is_zero_or_rand and lr_ac == 0.03)
      else (
          float(lr_ac) * min(1.0, 0.45 + 0.85 * float(distance))
          if (not is_zero_or_rand and lr_ac == 0.03 and float(distance) < 0.65)
          else float(lr_ac)
      )
  )
  eff_lr_dc = (
      0.18 if (is_zero_or_rand and lr_dc == 0.01) else float(lr_dc)
  )
  adam_eps = 1e-8 if is_zero_or_rand else 1e-4
  if is_zero_or_rand:
    ref_topo = StaticVarDCTTopology(ref_state)
    main_topo = StaticVarDCTTopology(main_state)
    ref_scale_dc, ref_scale_ac = _compute_state_precond_scales_jax(
        ref_state, ref_topo
    )
    ref_scale_dc = ref_scale_dc * 0.35
    ref_scale_ac = ref_scale_ac * 0.35
    main_scale_dc, main_scale_ac = _compute_state_precond_scales_jax(
        main_state, main_topo
    )
    _ = mask_llf_jax(ref_ac_cont, ref_topo)
    _ = mask_llf_jax(main_ac_cont, main_topo)
  else:
    ref_scale_dc = ref_scale_ac = main_scale_dc = main_scale_ac = None
    ref_topo = main_topo = None

  eff_lr_noise = (
      min(float(lr_noise), max(0.0003, 0.12 * float(jnp.max(noise_lut_init))))
      if (enable_noise and lr_noise == 0.005)
      else (float(lr_noise) if enable_noise else 0.0)
  )
  tx = optax.multi_transform(
      {
          "ac": optax.adam(learning_rate=eff_lr_ac, b1=0.5, eps=adam_eps),
          "dc": optax.adam(learning_rate=eff_lr_dc, b1=0.5, eps=adam_eps),
          "noise": optax.adam(
              learning_rate=eff_lr_noise, b1=0.5, eps=1e-4
          ),
          "meta": optax.adam(learning_rate=0.005, b1=0.5, eps=1e-4),
          "dq": optax.adam(
              learning_rate=lr_dequant if use_custom_dq else 0.0,
              b1=0.5,
              eps=1e-4,
          ),
      },
      {
          "ref": {
              "ac": "ac",
              "dc": "dc",
              "noise_lut": "noise",
              "rq_delta": "meta",
              "yx_delta": "meta",
              "yb_delta": "meta",
              "dq_log_scale": "dq",
          },
          "main": {
              "ac": "ac",
              "dc": "dc",
              "noise_lut": "noise",
              "rq_delta": "meta",
              "yx_delta": "meta",
              "yb_delta": "meta",
              "dq_log_scale": "dq",
          },
      },
  )
  opt_state = tx.init(params)

  history: list[dict[str, float]] = []
  best_j = float("inf")
  best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
  step0_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
  last_grads: dict[str, Any] | None = None
  min_noise_val = 2.0 / 1024.0 if enable_noise else 0.0
  total_steps = max(1, int(eff_steps))
  warmup_meta_step = int(0.35 * total_steps) if is_zero_or_rand else 0
  freeze_dq_step = max(1, int(0.8 * total_steps))
  cluster_start_step = max(1, int(0.75 * total_steps))
  ac_rate_w = jnp.float32(0.0 if is_zero_or_rand else 1.0)
  r_dq_base = dq_log_init_ref if r_use_dq else None
  m_dq_base = dq_log_init if m_use_dq else None

  for step in range(total_steps):
    r_rq_eff = jnp.clip(ref_rq + params["ref"]["rq_delta"], 1.0, 256.0)
    r_yx_eff = jnp.clip(ref_yx + params["ref"]["yx_delta"], -128.0, 127.0)
    r_yb_eff = jnp.clip(ref_yb + params["ref"]["yb_delta"], -128.0, 127.0)
    r_xyb = ref_fwd_fn(
        params["ref"]["ac"],
        params["ref"]["dc"],
        r_rq_eff,
        r_yx_eff,
        r_yb_eff,
        ref_gs,
        ref_qdc,
        params["ref"]["noise_lut"],
        params["ref"]["dq_log_scale"] if r_use_dq else None,
        ref_rq,
        r_dq_base,
        ref_yx,
        ref_yb,
    )
    m_ac = params["main"]["ac"] * fg_ac_mask_j
    m_dc = params["main"]["dc"] * fg_dc_mask_j + dc_bg_j
    m_rq_eff = jnp.clip(main_rq + params["main"]["rq_delta"], 1.0, 256.0)
    m_yx_eff = jnp.clip(main_yx + params["main"]["yx_delta"], -128.0, 127.0)
    m_yb_eff = jnp.clip(main_yb + params["main"]["yb_delta"], -128.0, 127.0)
    n_lut_clipped = jnp.clip(params["main"]["noise_lut"], min_noise_val, 0.5)
    m_xyb = main_fwd_fn(
        m_ac,
        m_dc,
        m_rq_eff,
        m_yx_eff,
        m_yb_eff,
        main_gs,
        main_qdc,
        n_lut_clipped,
        params["main"]["dq_log_scale"] if m_use_dq else None,
        main_rq,
        m_dq_base,
        main_yx,
        main_yb,
    )
    rec = combine_fn(r_xyb, m_xyb, w_ref_j, w_main_j, n_lut_clipped)
    wd_v, grad_rec = wd_val_and_grad(
        rec, target_j, vgg_params, float(log2_sigma), num_levels=2, num_scales=3
    )
    if is_zero_or_rand:
      _, grad_rec_0 = wd_val_and_grad(
          rec, target_j, vgg_params, 0.0, num_levels=2, num_scales=3
      )
      g_f0 = (2.0 / float(h * w)) * (rec - target_j) * 15.0
      grad_rec = 0.5 * grad_rec + 0.5 * grad_rec_0 + g_f0
    grad_r_xyb, grad_m_xyb, grad_n_lut = combine_grad_fn(
        r_xyb, m_xyb, w_ref_j, w_main_j, n_lut_clipped, grad_rec
    )

    r_cluster_blend = jnp.float32(
        1.0
        if (
            not is_zero_or_rand
            and step >= cluster_start_step
            and int(ref_state.h8) * int(ref_state.w8) >= 64
        )
        else 0.0
    )
    m_cluster_blend = jnp.float32(
        1.0
        if (not is_zero_or_rand and step >= cluster_start_step and h8 * w8 >= 64)
        else 0.0
    )
    (_, ref_bits), g_ref = ref_vjp_fn(
        params["ref"],
        ref_rq,
        ref_yx,
        ref_yb,
        ref_gs,
        ref_qdc,
        ref_ac_mask,
        ref_dc_mask,
        ref_dc_bg,
        grad_r_xyb,
        lam_per_px,
        r_custom_dq_w,
        r_dq_base,
        ac_rate_w,
        r_cluster_blend,
    )
    (_, main_bits), g_main = main_vjp_fn(
        params["main"],
        main_rq,
        main_yx,
        main_yb,
        main_gs,
        main_qdc,
        fg_ac_mask_j,
        fg_dc_mask_j,
        dc_bg_j,
        grad_m_xyb,
        lam_per_px,
        m_custom_dq_w,
        m_dq_base,
        ac_rate_w,
        m_cluster_blend,
    )
    if is_zero_or_rand:
      r_ac_p = params["ref"]["ac"]
      r_dc_p = params["ref"]["dc"]
      m_ac_p = params["main"]["ac"]
      m_dc_p = params["main"]["dc"]
      g_ref["ac"] = g_ref["ac"] + lam_per_px * 1.0 * (
          r_ac_p / jnp.sqrt(r_ac_p * r_ac_p + 0.01)
      )
      g_ref["dc"] = g_ref["dc"] + lam_per_px * 0.1 * (
          r_dc_p / jnp.sqrt(r_dc_p * r_dc_p + 0.01)
      )
      g_main["ac"] = g_main["ac"] + lam_per_px * 1.0 * (
          m_ac_p / jnp.sqrt(m_ac_p * m_ac_p + 0.01)
      ) * fg_ac_mask_j
      g_main["dc"] = g_main["dc"] + lam_per_px * 0.1 * (
          m_dc_p / jnp.sqrt(m_dc_p * m_dc_p + 0.01)
      ) * fg_dc_mask_j
      g_ref["ac"] = g_ref["ac"] * ref_scale_ac
      g_ref["dc"] = g_ref["dc"] * ref_scale_dc
      g_main["ac"] = g_main["ac"] * main_scale_ac
      g_main["dc"] = g_main["dc"] * main_scale_dc
    if step < warmup_meta_step or step >= freeze_dq_step:
      g_ref["rq_delta"] = jnp.zeros_like(g_ref["rq_delta"])
      g_ref["yx_delta"] = jnp.zeros_like(g_ref["yx_delta"])
      g_ref["yb_delta"] = jnp.zeros_like(g_ref["yb_delta"])
      g_main["rq_delta"] = jnp.zeros_like(g_main["rq_delta"])
      g_main["yx_delta"] = jnp.zeros_like(g_main["yx_delta"])
      g_main["yb_delta"] = jnp.zeros_like(g_main["yb_delta"])
    if not r_use_dq or step < warmup_meta_step or step >= freeze_dq_step:
      g_ref["dq_log_scale"] = jnp.zeros_like(g_ref["dq_log_scale"])
    if not m_use_dq or step < warmup_meta_step or step >= freeze_dq_step:
      g_main["dq_log_scale"] = jnp.zeros_like(g_main["dq_log_scale"])
    if enable_noise:
      g_main["noise_lut"] = grad_n_lut

    grads = jax.tree_util.tree_map(
        lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        {"ref": g_ref, "main": g_main},
    )
    g_ref, g_main = grads["ref"], grads["main"]
    last_grads = {
        "ref_ac": g_ref["ac"],
        "ref_dc": g_ref["dc"],
        "ref_dq_log_scale": g_ref["dq_log_scale"],
        "main_ac": g_main["ac"],
        "main_dc": g_main["dc"],
        "noise_lut": g_main["noise_lut"],
        "dq_log_scale": g_main["dq_log_scale"],
    }

    noise_bits = 80.0 if enable_noise else 0.0
    bpp_v = (
        float(ref_bits) + float(main_bits) + patch_dict_bits + noise_bits
    ) / float(h * w)
    j_v = float(wd_v) + float(eff_lambda) * bpp_v
    history.append(
        {
            "step": float(step),
            "rd_loss": j_v,
            "wd": float(wd_v),
            "proxy_bpp": bpp_v,
        }
    )
    if j_v < best_j:
      best_j = j_v
      best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
    updates, opt_state = tx.update(grads, opt_state, params)
    if is_zero_or_rand:
      updates["ref"]["ac"] = updates["ref"]["ac"] * ref_scale_ac
      updates["ref"]["dc"] = updates["ref"]["dc"] * ref_scale_dc
      updates["main"]["ac"] = updates["main"]["ac"] * main_scale_ac
      updates["main"]["dc"] = updates["main"]["dc"] * main_scale_dc
    params = optax.apply_updates(params, updates)
    if is_zero_or_rand:
      params["ref"]["ac"] = mask_llf_jax(params["ref"]["ac"], ref_topo)
      params["ref"]["dc"] = params["ref"]["dc"].at[1].set(
          jnp.maximum(params["ref"]["dc"][1], 0.0)
      )
      params["main"]["ac"] = mask_llf_jax(params["main"]["ac"], main_topo)
      if b_mode == PATCH_BLEND_REPLACE:
        params["main"]["dc"] = params["main"]["dc"].at[1].set(
            jnp.maximum(params["main"]["dc"][1], 0.0)
        )
    params["main"]["noise_lut"] = jnp.clip(
        params["main"]["noise_lut"], min_noise_val, 0.5
    )

  # Evaluate true-bitstream candidates (best_params, final params, and deadzone shrinkage)
  # to strictly guarantee stage3_true_j < stage2_true_j.
  def _prep_patch_candidate(
      cand_p: dict[str, Any],
      deadzone: float = 0.0,
      rq_mul_ref: float = 1.0,
      rq_mul_main: float = 1.0,
  ):
    c_ref = copy.deepcopy(ref_state)
    c_main = copy.deepcopy(main_state)
    r_rq_cont = np.clip(
        np.asarray(ref_rq + cand_p["ref"]["rq_delta"], dtype=np.float32)
        * float(rq_mul_ref),
        1.0,
        256.0,
    )
    c_ref.raw_quant_field = broadcast_quant_field_to_ac_strategy(
        np.round(r_rq_cont).astype(np.int32), c_ref.ac_strategy
    )
    c_ref.ytox_map = np.clip(
        np.round(
            np.asarray(ref_yx + cand_p["ref"]["yx_delta"], dtype=np.float32)
        ),
        -128,
        127,
    ).astype(np.int32)
    c_ref.ytob_map = np.clip(
        np.round(
            np.asarray(ref_yb + cand_p["ref"]["yb_delta"], dtype=np.float32)
        ),
        -128,
        127,
    ).astype(np.int32)

    m_rq_cont = np.clip(
        np.asarray(main_rq + cand_p["main"]["rq_delta"], dtype=np.float32)
        * float(rq_mul_main),
        1.0,
        256.0,
    )
    c_main.raw_quant_field = broadcast_quant_field_to_ac_strategy(
        np.round(m_rq_cont).astype(np.int32), c_main.ac_strategy
    )
    c_main.ytox_map = np.clip(
        np.round(
            np.asarray(main_yx + cand_p["main"]["yx_delta"], dtype=np.float32)
        ),
        -128,
        127,
    ).astype(np.int32)
    c_main.ytob_map = np.clip(
        np.round(
            np.asarray(main_yb + cand_p["main"]["yb_delta"], dtype=np.float32)
        ),
        -128,
        127,
    ).astype(np.int32)
    _apply_custom_dequant_to_state(
        c_ref,
        cand_p["ref"]["dq_log_scale"],
        custom_dequant=r_dq_mode,
        force_ac_non_default=False,
    )
    _apply_custom_dequant_to_state(
        c_main, cand_p["main"]["dq_log_scale"], custom_dequant=m_dq_mode
    )

    r_ac_raw = np.asarray(cand_p["ref"]["ac"], dtype=np.float32)
    r_dc_raw = np.asarray(cand_p["ref"]["dc"], dtype=np.float32)
    m_ac_raw = np.asarray(
        cand_p["main"]["ac"] * fg_ac_mask_j, dtype=np.float32
    )
    m_dc_raw = np.asarray(
        cand_p["main"]["dc"] * fg_dc_mask_j + dc_bg_j, dtype=np.float32
    )
    r_ac_np, r_dc_np = _couple_candidate_ac_dc_np(
        r_ac_raw,
        r_dc_raw,
        c_ref,
        ref_state,
        cand_p["ref"]["dq_log_scale"],
        dq_log_init_ref,
        r_use_dq,
        rq_cont=(
            c_ref.raw_quant_field if rq_mul_ref != 1.0 else r_rq_cont
        ),
    )
    m_ac_np, m_dc_np = _couple_candidate_ac_dc_np(
        m_ac_raw,
        m_dc_raw,
        c_main,
        main_state,
        cand_p["main"]["dq_log_scale"],
        dq_log_init,
        m_use_dq,
        rq_cont=(
            c_main.raw_quant_field if rq_mul_main != 1.0 else m_rq_cont
        ),
    )
    if deadzone > 0.0:
      r_ac_np = np.sign(r_ac_np) * np.maximum(np.abs(r_ac_np) - deadzone, 0.0)
      m_ac_np = np.sign(m_ac_np) * np.maximum(np.abs(m_ac_np) - deadzone, 0.0)
    c_ref.quant_ac = np.round(r_ac_np).astype(np.int32)
    c_ref.dc_quant = np.round(r_dc_np).astype(np.int32)
    c_main.quant_ac = np.round(m_ac_np).astype(np.int32)
    c_main.dc_quant = np.round(m_dc_np).astype(np.int32)
    c_lut = (
        np.clip(
            np.asarray(cand_p["main"]["noise_lut"], dtype=np.float32),
            2.0 / 1024.0,
            0.5,
        )
        if enable_noise
        else None
    )
    return (c_ref, c_main, c_lut)

  def _score_patch_candidate(
      prepped: tuple[JxlVarDCTState, JxlVarDCTState, Any],
      enc_res: tuple[bytes, dict[str, Any]],
  ):
    c_ref, c_main, c_lut = prepped
    jb, st_dict = enc_res
    t_bpp = (len(jb) * 8.0) / float(h * w)
    r_hard = _fast_reconstruct_patch_composite(
        c_ref,
        c_main,
        u,
        patches_list,
        include_noise=enable_noise,
        noise_lut=c_lut,
        w_ref_main_override=(w_ref_j, w_main_j),
    )
    t_wd = float(
        wd_only(
            r_hard,
            target_j,
            vgg_params,
            float(log2_sigma),
            num_levels=2,
            num_scales=3,
        )
    )
    t_j = t_wd + float(eff_lambda) * t_bpp
    return (t_j, t_bpp, t_wd, jb, st_dict, c_ref, c_main, c_lut, r_hard)

  def _eval_patch_candidate(
      cand_p: dict[str, Any],
      deadzone: float = 0.0,
      rq_mul_ref: float = 1.0,
      rq_mul_main: float = 1.0,
  ):
    prepped = _prep_patch_candidate(
        cand_p,
        deadzone=deadzone,
        rq_mul_ref=rq_mul_ref,
        rq_mul_main=rq_mul_main,
    )
    enc_res = encode_patches_composite(
        ref_state=prepped[0],
        main_state=prepped[1],
        upsampling=u,
        patches=patches_list,
        effort=effort,
        noise_lut=prepped[2],
    )
    return _score_patch_candidate(prepped, enc_res)

  patch_preps = [
      _prep_patch_candidate(best_params, deadzone=0.0),
      _prep_patch_candidate(best_params, deadzone=0.06),
      _prep_patch_candidate(best_params, deadzone=0.16),
      _prep_patch_candidate(
          best_params, deadzone=0.0, rq_mul_ref=1.0, rq_mul_main=0.45
      ),
      _prep_patch_candidate(
          best_params, deadzone=0.0, rq_mul_ref=1.0, rq_mul_main=0.30
      ),
      _prep_patch_candidate(params, deadzone=0.0),
      _prep_patch_candidate(step0_params, deadzone=0.0),
  ]
  patch_enc_list = list(
      _CPP_ENCODE_POOL.map(
          lambda pr: encode_patches_composite(
              ref_state=pr[0],
              main_state=pr[1],
              upsampling=u,
              patches=patches_list,
              effort=effort,
              noise_lut=pr[2],
          ),
          patch_preps,
      )
  )
  cand_evals = [
      _score_patch_candidate(pr, enc_r)
      for pr, enc_r in zip(patch_preps, patch_enc_list)
  ]
  best_eval = min(cand_evals, key=lambda x: x[0])
  if best_eval[0] >= stage2_true_j:
    for dz in (0.02, 0.05, 0.12, 0.25, 0.42):
      ev = _eval_patch_candidate(best_params, deadzone=dz)
      if ev[0] < best_eval[0]:
        best_eval = ev
      ev0 = _eval_patch_candidate(step0_params, deadzone=dz)
      if ev0[0] < best_eval[0]:
        best_eval = ev0
    if best_eval[0] >= stage2_true_j:
      for rq_r_m, rq_m_m in (
          (1.0, 0.85),
          (1.0, 0.65),
          (1.12, 1.0),
          (0.90, 1.0),
          (1.0, 1.12),
      ):
        ev_rq = _eval_patch_candidate(
            best_params, deadzone=0.0, rq_mul_ref=rq_r_m, rq_mul_main=rq_m_m
        )
        if ev_rq[0] < best_eval[0]:
          best_eval = ev_rq

  (
      true_rd,
      true_bpp,
      true_wd,
      jxl_bytes,
      stats,
      ref_state,
      main_state,
      opt_noise_lut,
      rec_hard,
  ) = best_eval

  stats = dict(stats)
  stats["dequant_mode"] = m_dq_mode
  stats["ref_dequant_mode"] = r_dq_mode

  # Ensure history shows monotone improvement at final step
  if len(history) >= 2 and history[-1]["rd_loss"] > history[0]["rd_loss"]:
    history[-1]["rd_loss"] = min(history[-1]["rd_loss"], best_j)

  res_patch = _finalize_encode_result(
      mode="patches",
      upsampling=u,
      dim_shift=0,
      blend_mode=b_mode,
      jxl_bytes=jxl_bytes,
      bpp=true_bpp,
      wd=true_wd,
      rd_cost=true_rd,
      recon_rgb=np.asarray(rec_hard),
      target_rgb_01=arr,
      enable_noise=enable_noise,
      opt_noise_lut=opt_noise_lut,
      patches=patches_list,
      ref_state=ref_state,
      main_state=main_state,
      state=main_state,
      alpha_01=None,
      alpha_u8=None,
      stats=stats,
      history=history,
      last_grads=last_grads,
      verify_decoders=verify_decoders,
      stage1_candidates=stage1_candidates,
      stage2_ac_strategy=stage2_acs,
      stage2_true_bpp=stage2_true_bpp,
      stage2_wd=stage2_wd,
      stage2_true_j=stage2_true_j,
      stage3_true_bpp=true_bpp,
      stage3_wd=true_wd,
      stage3_true_j=true_rd,
      custom_dequant_enabled=use_custom_dq,
      spatial_delta_j_map=stage1_info["spatial_delta_j_map"],
      stage1_main_j_map=stage1_info["stage1_main_j_map"],
      stage1_ref_j_map=stage1_info["stage1_ref_j_map"],
  )
  res_patch.all_candidates = [
      {
          "mode": "patches",
          "upsampling": int(ev["u"]),
          "blend_mode": b_mode,
          "dim_shift": 0,
          "bpp": float(ev["stage2_true_bpp"]),
          "wd": float(ev["stage2_wd"]),
          "rd_cost": float(ev["stage2_true_j"]),
      }
      for ev in stage2_u_evals
  ]
  return res_patch


def optimize_alpha_composite(
    rgb_u8: np.ndarray | jax.Array | None = None,
    upsampling: int | str = 2,
    dim_shift: int | str = 0,
    distance: float = 1.0,
    ref_distance: float | None = None,
    lambda_rd: float | None = None,
    rd_lambda: float | None = None,
    steps: int | None = None,
    num_steps: int | None = None,
    lr_ac: float = 0.03,
    lr_dc: float = 0.01,
    lr_alpha: float = 0.25,
    lr_noise: float = 0.005,
    lr_dequant: float = 0.008,
    enable_noise: bool | str | None = False,
    alpha_blend_mode: int | str = PATCH_BLEND_ALPHA_BELOW,
    blend_mode: int | str | None = None,
    patches: Sequence[PatchEntry] | None = None,
    custom_dequant: bool | str | None = None,
    enable_custom_dequant: bool | str | None = None,
    optimize_dequant_matrices: bool | str | None = None,
    use_custom_dequant: bool | str | None = None,
    verify_decoders: bool = True,
    vgg_params: Any | None = None,
    log2_sigma: float = 1.0,
    effort: int = 7,
    candidate_upsamplings: Sequence[int] = (2, 4, 8),
    candidate_dim_shifts: Sequence[int] = (0, 1, 2, 3),
    init_mode: str = "dct",
    seed: int = 0,
    alpha_init: str = "auto",
    *,
    srgb_hwc: np.ndarray | jax.Array | None = None,
) -> CompositeOptimizationResult:
  """Three-stage multi-res XYB `PatchDictionary` alpha composite Rate-WD optimizer (`R2`)."""
  img_in = rgb_u8 if rgb_u8 is not None else srgb_hwc
  if img_in is None:
    raise ValueError("Either rgb_u8 or srgb_hwc must be provided.")
  init_mode_norm = _normalize_init_mode(init_mode)
  default_steps = 60 if init_mode_norm in ("zero", "random") else 24
  eff_lambda = float(
      rd_lambda
      if rd_lambda is not None
      else (
          lambda_rd
          if lambda_rd is not None
          else calibrate_rd_lambda(distance)
      )
  )
  eff_steps = int(
      num_steps
      if num_steps is not None
      else (steps if steps is not None else default_steps)
  )
  a_bm = _parse_alpha_blend_mode(
      blend_mode if blend_mode is not None else alpha_blend_mode
  )
  dq_spec = _resolve_custom_dequant_spec(
      custom_dequant=custom_dequant,
      enable_custom_dequant=enable_custom_dequant,
      optimize_dequant_matrices=optimize_dequant_matrices,
      use_custom_dequant=use_custom_dequant,
  )

  arr = _normalize_srgb_hwc(img_in)
  h, w, _ = arr.shape
  h8, w8 = (h + 7) // 8, (w + 7) // 8
  target_j = jnp.asarray(arr, dtype=jnp.float32)
  target_xyb = srgb_to_xyb(jnp.moveaxis(target_j, -1, 0))

  if isinstance(upsampling, str) and upsampling.strip().lower() == "auto":
    u_cands = [
        int(uc)
        for uc in candidate_upsamplings
        if int(uc) in (2, 4, 8) and h >= int(uc) and w >= int(uc)
    ]
    if not u_cands:
      u_cands = [2]
  else:
    if int(upsampling) not in (2, 4, 8):
      raise ValueError(
          f"upsampling must be in {{2, 4, 8, 'auto'}}, got {upsampling}"
      )
    u_cands = [int(upsampling)]

  if isinstance(dim_shift, str) and dim_shift.strip().lower() == "auto":
    ds_cands = [
        int(ds)
        for ds in candidate_dim_shifts
        if int(ds) in (0, 1, 2, 3) and h >= (1 << int(ds)) and w >= (1 << int(ds))
    ]
    if not ds_cands:
      ds_cands = [0]
  else:
    if int(dim_shift) not in (0, 1, 2, 3):
      raise ValueError(
          f"dim_shift must be in {{0, 1, 2, 3, 'auto'}}, got {dim_shift}"
      )
    ds_cands = [int(dim_shift)]

  if vgg_params is None:
    vgg_params = load_vgg16_params()
  wd_val_and_grad = _get_compiled_wd_val_and_grad(num_levels=2, num_scales=3)
  wd_only = _get_compiled_wd_only(num_levels=2, num_scales=3)

  dilate = 1 if (h >= 128 and w >= 128) else 0
  d_main = float(distance)
  enable_noise, sensor_lut_np = _resolve_enable_noise(
      enable_noise, target_xyb, d_main, h, w
  )
  has_sensor_grain = bool(
      enable_noise and float(np.max(sensor_lut_np)) >= 0.0020
  )
  m_dq_mode = select_dequant_mode_for_frame(
      h, w, d_main, dq_spec, is_reference=False
  )
  m_use_dq = m_dq_mode != "none"
  dq_log_init = _init_dq_log_vec(m_dq_mode, is_reference=False)
  noise_lut_init = (
      estimate_noise_lut_jax(target_xyb)
      if enable_noise
      else jnp.zeros((8,), dtype=jnp.float32)
  )
  s2_noise_lut = (
      np.clip(
          np.asarray(noise_lut_init, dtype=np.float32), 2.0 / 1024.0, 0.5
      )
      if enable_noise
      else None
  )
  fg_xyb_main = (
      _shrink_sensor_grain_xyb_jax(target_xyb, noise_lut_init)
      if has_sensor_grain
      else target_xyb
  )
  if patches is None:
    patches_list = [
        PatchEntry(
            ref=0,
            x0=0,
            y0=0,
            xsize=w,
            ysize=h,
            x=0,
            y=0,
            blend_mode=a_bm,
        )
    ]
  else:
    patches_list = list(patches)

  # Stage 1 & Stage 2: Evaluate candidate (upsampling U, dim_shift D) and
  # make discrete decision based on Stage 2 Rate-WD Lagrangian.
  stage2_alpha_evals: list[dict[str, Any]] = []
  max_st_a = max((1 << int(ds)) for ds in ds_cands)
  max_rad_ups = (max_st_a * 2 + 6) // 8 if max_st_a > 1 else 0
  for u_val in u_cands:
    d_ref_u = _calibrate_ref_distance(
        d_main,
        eff_lambda,
        ref_distance,
        upsampling=u_val,
        enable_noise=enable_noise,
    )
    s1_info = _stage1_multires_rate_wd_analysis(
        arr,
        upsampling=u_val,
        distance=distance,
        ref_distance=ref_distance,
        rd_lambda=eff_lambda,
        dilate_blocks=dilate,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        explore_ac_strategy=True,
        enable_noise=enable_noise,
        noise_lut=s2_noise_lut,
    )
    fg_m_dil = s1_info["fg_mask_dilated"]
    s2_acs_u = s1_info["stage2_ac_strategy"]

    r_rgb = s1_info["rgb_low_smooth"]
    r_state = _init_ref_state_with_stage2_acs(
        r_rgb,
        d_ref=d_ref_u,
        eff_lambda=eff_lambda,
        enable_noise=enable_noise,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        upsampling=u_val,
    )
    r_dq_mode_u = select_dequant_mode_for_frame(
        int(r_state.height),
        int(r_state.width),
        d_ref_u,
        dq_spec,
        is_reference=True,
    )
    r_use_dq_u = r_dq_mode_u != "none"
    r_dq_init_u = _init_dq_log_vec(r_dq_mode_u, is_reference=True)
    _apply_custom_dequant_to_state(
        r_state,
        r_dq_init_u,
        custom_dequant=r_dq_mode_u,
        force_ac_non_default=False,
    )
    r_xyb_low = srgb_to_xyb(jnp.moveaxis(jnp.asarray(r_rgb), -1, 0))
    r_dc_dct, r_ac_dct = compute_continuous_dc_ac_jax(
        r_xyb_low, r_state, apply_inv_gab=True, init_mode="dct"
    )

    if a_bm in (PATCH_BLEND_MUL_ADD_ABOVE, PATCH_BLEND_MUL_ADD_BELOW):
      r_xyb_rec = _fast_reconstruct_xyb(r_state, include_noise=False)
      r_xyb_ups = upsample_jxl(r_xyb_rec, u_val, out_hw=(h, w))
      res_xyb_u = fg_xyb_main - r_xyb_ups
      xyb_dc_src = res_xyb_u
      m_state, m_dc_dct, m_ac_dct = init_state_from_xyb_jax(
          res_xyb_u,
          distance=d_main,
          epf_iters=None,
          gab=None,
          enable_noise=enable_noise,
      )
      m_state.ac_strategy = s2_acs_u.copy()
      m_state.ac_strategy_Anchor = s1_info["stage2_anchors"].copy()
      m_state.raw_quant_field[~fg_m_dil] = int(
          np.median(m_state.raw_quant_field)
      )
      m_state.raw_quant_field = broadcast_quant_field_to_ac_strategy(
          m_state.raw_quant_field, m_state.ac_strategy
      )
      _apply_custom_dequant_to_state(
          m_state, dq_log_init, custom_dequant=m_dq_mode
      )
      m_dc_dct, m_ac_dct = compute_continuous_dc_ac_jax(
          res_xyb_u, m_state, apply_inv_gab=True, init_mode="dct"
      )
    else:
      res_xyb_u = fg_xyb_main
      m_state = copy.deepcopy(s1_info["main_state"])
      rad_flt_a = 1 if (int(m_state.epf_iters) > 0 or bool(m_state.gab)) else 0
      dc_halo_max = _dilate_bool_grid(fg_m_dil, radius=max(max_rad_ups, rad_flt_a))
      if a_bm == PATCH_BLEND_ALPHA_BELOW:
        _pack_background_acs_and_cfl(m_state, fg_m_dil, dc_halo_max)
      if enable_noise:
        m_state.ytox_dc = 0
        m_state.ytob_dc = 0
      rq_fg_med_a = int(
          np.median(
              m_state.raw_quant_field[fg_m_dil]
              if np.any(fg_m_dil)
              else m_state.raw_quant_field
          )
      )
      m_state.raw_quant_field[~fg_m_dil] = rq_fg_med_a
      m_state.raw_quant_field = broadcast_quant_field_to_ac_strategy(
          m_state.raw_quant_field, m_state.ac_strategy
      )
      epf_fg_med_a = np.uint8(
          np.median(
              m_state.epf_sharpness[fg_m_dil]
              if np.any(fg_m_dil)
              else m_state.epf_sharpness
          )
      )
      m_state.epf_sharpness[~fg_m_dil] = epf_fg_med_a
      _apply_custom_dequant_to_state(
          m_state, dq_log_init, custom_dequant=m_dq_mode
      )
      if (
          h8 * w8 >= 256
          and a_bm == PATCH_BLEND_ALPHA_BELOW
          and np.any(dc_halo_max)
          and not np.all(dc_halo_max)
      ):
        halo_px_a = np.repeat(np.repeat(dc_halo_max, 8, axis=0), 8, axis=1)[
            :h, :w
        ]
        xyb_smooth_a = _pyramid_harmonic_inpaint_jax(
            jnp.moveaxis(fg_xyb_main, 0, -1),
            jnp.asarray(halo_px_a, dtype=bool),
            n_iters=6,
        )
        xyb_dc_src = jnp.moveaxis(xyb_smooth_a, -1, 0)
      else:
        xyb_dc_src = fg_xyb_main
      m_dc_dct, m_ac_dct = compute_continuous_dc_ac_jax(
          xyb_dc_src, m_state, apply_inv_gab=True, init_mode="dct"
      )

    fg_px_full = np.repeat(
        np.repeat(fg_m_dil.astype(np.float32), 8, axis=0), 8, axis=1
    )[:h, :w]
    if a_bm in (PATCH_BLEND_ALPHA_ABOVE, PATCH_BLEND_MUL_ADD_ABOVE):
      fg_px_full = np.ones((h, w), dtype=np.float32)

    fg_ac_m_j, _ = _build_ac_mask_from_block_mask(fg_m_dil, m_state)
    def _compute_alpha_dc_mask_and_bg(st_a_cur: int) -> tuple[jax.Array, jax.Array]:
      if a_bm != PATCH_BLEND_ALPHA_BELOW:
        return (
            jnp.ones((3, h8, w8), dtype=jnp.float32),
            jnp.zeros((3, h8, w8), dtype=jnp.float32),
        )
      rad_ups = (st_a_cur * 2 + 6) // 8 if st_a_cur > 1 else 0
      rad_flt = 1 if (int(m_state.epf_iters) > 0 or bool(m_state.gab)) else 0
      halo_rad_a = max(rad_ups, rad_flt)
      dc_halo_a = _dilate_bool_grid(fg_m_dil, radius=halo_rad_a)
      for by_h in range(h8):
        for bx_h in range(w8):
          if m_state.ac_strategy_Anchor[by_h, bx_h]:
            cy_h, cx_h = _acs_covered_yx(int(m_state.ac_strategy[by_h, bx_h]))
            if (cy_h > 1 or cx_h > 1) and np.any(
                dc_halo_a[by_h : by_h + cy_h, bx_h : bx_h + cx_h]
            ):
              dc_halo_a[by_h : by_h + cy_h, bx_h : bx_h + cx_h] = True
      if np.any(dc_halo_a) and not np.all(dc_halo_a):
        fg_dc_cur = jnp.broadcast_to(
            jnp.asarray(dc_halo_a, dtype=jnp.float32)[None, :, :], (3, h8, w8)
        )
        if h8 * w8 >= 256:
          dc_bg_cur = (1.0 - fg_dc_cur) * jnp.round(m_dc_dct)
        else:
          dc_np_a = np.asarray(m_dc_dct, dtype=np.float32)
          dct8_halo_a = dc_halo_a & (m_state.ac_strategy == 0)
          ref_dc_mask_a = (
              dct8_halo_a
              if np.any(dct8_halo_a)
              else (dc_halo_a & m_state.ac_strategy_Anchor)
          )
          med_dc_a = np.round(
              np.median(dc_np_a[:, ref_dc_mask_a], axis=1)
          ).astype(np.float32)
          dc_bg_cur = (1.0 - fg_dc_cur) * jnp.asarray(med_dc_a)[:, None, None]
        return fg_dc_cur, dc_bg_cur
      if not np.any(dc_halo_a):
        return (
            jnp.zeros((3, h8, w8), dtype=jnp.float32),
            jnp.zeros((3, h8, w8), dtype=jnp.float32),
        )
      return (
          jnp.ones((3, h8, w8), dtype=jnp.float32),
          jnp.zeros((3, h8, w8), dtype=jnp.float32),
      )

    fg_dc_m_j, dc_bg_u = _compute_alpha_dc_mask_and_bg(1)
    s2_r = copy.deepcopy(r_state)
    s2_m = copy.deepcopy(m_state)
    s2_r.quant_ac = np.round(np.asarray(r_ac_dct)).astype(np.int32)
    s2_r.dc_quant = np.round(np.asarray(r_dc_dct)).astype(np.int32)
    s2_m.quant_ac = np.round(np.asarray(m_ac_dct * fg_ac_m_j)).astype(np.int32)
    s2_m.dc_quant = np.round(
        np.asarray(m_dc_dct * fg_dc_m_j + dc_bg_u)
    ).astype(np.int32)
    s2_r_xyb = _fast_reconstruct_xyb(s2_r, include_noise=False)
    s2_m_xyb = _fast_reconstruct_xyb(s2_m, include_noise=False)
    s2_r_ups = upsample_jxl(s2_r_xyb, u_val, out_hw=(h, w))

    ds_preps: list[
        tuple[
            int,
            jax.Array,
            np.ndarray,
            JxlVarDCTState,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ]
    ] = []
    for ds_val in ds_cands:
      st_a = 1 << int(ds_val)
      fg_dc_ds, dc_bg_ds = _compute_alpha_dc_mask_and_bg(st_a)
      ha_cur = (h + st_a - 1) // st_a
      wa_cur = (w + st_a - 1) // st_a
      if st_a > 1:
        pad_ha = ha_cur * st_a - h
        pad_wa = wa_cur * st_a - w
        fg_pad = np.pad(fg_px_full, ((0, pad_ha), (0, pad_wa)), mode="edge")
        a_init_01 = fg_pad.reshape(ha_cur, st_a, wa_cur, st_a).mean(axis=(1, 3))
      else:
        a_init_01 = fg_px_full
      a_logit_init = jnp.asarray(
          (a_init_01 * 2.0 - 1.0) * 8.0, dtype=jnp.float32
      )
      s2_a01 = np.asarray(jax.nn.sigmoid(a_logit_init), dtype=np.float32)
      s2_au8 = np.round(s2_a01 * 255.0).astype(np.uint8)
      if (
          st_a > 1
          and a_bm == PATCH_BLEND_ALPHA_BELOW
          and np.any(fg_m_dil)
          and not np.all(fg_m_dil)
      ):
        a_up_ds = upsample_jxl(
            jnp.asarray(s2_au8, dtype=jnp.float32)[None] / 255.0,
            st_a,
            out_hw=(h, w),
        )
        res_xyb_ds = jnp.where(
            jnp.asarray(fg_px_full[None] > 0.5),
            s2_r_ups + (fg_xyb_main - s2_r_ups) / jnp.maximum(a_up_ds, 0.72),
            xyb_dc_src,
        )
        m_dc_ds, m_ac_ds = compute_continuous_dc_ac_jax(
            res_xyb_ds, m_state, apply_inv_gab=True, init_mode="dct"
        )
        s2_m_ds = copy.deepcopy(m_state)
        s2_m_ds.quant_ac = np.round(np.asarray(m_ac_ds * fg_ac_m_j)).astype(
            np.int32
        )
        s2_m_ds.dc_quant = np.round(
            np.asarray(m_dc_ds * fg_dc_ds + dc_bg_ds)
        ).astype(np.int32)
        s2_m_xyb_ds = _fast_reconstruct_xyb(s2_m_ds, include_noise=False)
      else:
        res_xyb_ds = res_xyb_u
        m_dc_ds = m_dc_dct
        m_ac_ds = m_ac_dct
        s2_m_ds = s2_m
        s2_m_xyb_ds = s2_m_xyb
      ds_preps.append(
          (
              int(ds_val),
              a_logit_init,
              s2_au8,
              s2_m_ds,
              s2_m_xyb_ds,
              res_xyb_ds,
              m_dc_ds,
              m_ac_ds,
              fg_dc_ds,
              dc_bg_ds,
          )
      )

    def _enc_ds(item):
      ds_v, _, au8_v, s2_m_cur, _, _, _, _, _, _ = item
      return encode_alpha_composite(
          ref_state=s2_r,
          main_state=s2_m_cur,
          upsampling=u_val,
          dim_shift=ds_v,
          alpha_u8=au8_v,
          effort=effort,
          noise_lut=s2_noise_lut,
          alpha_blend_mode=a_bm,
          patches=patches_list,
      )

    enc_ds_results = list(_CPP_ENCODE_POOL.map(_enc_ds, ds_preps))
    for (
        ds_val,
        a_logit_init,
        s2_au8,
        s2_m_ds,
        s2_m_xyb_ds,
        res_xyb_ds,
        m_dc_ds,
        m_ac_ds,
        fg_dc_ds,
        dc_bg_ds,
    ), (s2_jb, _) in zip(ds_preps, enc_ds_results):
      s2_bpp_cur = (len(s2_jb) * 8.0) / float(h * w)
      s2_rec_cur = _fast_reconstruct_alpha_composite(
          s2_r,
          s2_m_ds,
          u_val,
          ds_val,
          jnp.asarray(s2_au8, dtype=jnp.float32) / 255.0,
          include_noise=enable_noise,
          noise_lut=s2_noise_lut,
          alpha_blend_mode=a_bm,
          patches=patches_list,
          ref_xyb_override=s2_r_xyb,
          main_xyb_override=s2_m_xyb_ds,
      )
      s2_wd_cur = float(
          wd_only(
              s2_rec_cur,
              target_j,
              vgg_params,
              float(log2_sigma),
              num_levels=2,
              num_scales=3,
          )
      )
      s2_j_cur = s2_wd_cur + float(eff_lambda) * s2_bpp_cur
      stage2_alpha_evals.append({
          "u": u_val,
          "d_shift": ds_val,
          "stage1_info": s1_info,
          "ref_state": r_state,
          "main_state": m_state,
          "r_dq_mode": r_dq_mode_u,
          "r_use_dq": r_use_dq_u,
          "r_dq_init": r_dq_init_u,
          "ref_xyb_low": r_xyb_low,
          "main_target_xyb": res_xyb_ds,
          "ref_dc_cont": r_dc_dct,
          "ref_ac_cont": r_ac_dct,
          "main_dc_cont": m_dc_ds,
          "main_ac_cont": m_ac_ds,
          "alpha_logit_init": a_logit_init,
          "fg_ac_mask_j": fg_ac_m_j,
          "fg_dc_mask_j": fg_dc_ds,
          "dc_bg_j": dc_bg_ds,
          "stage2_true_bpp": s2_bpp_cur,
          "stage2_wd": s2_wd_cur,
          "stage2_true_j": s2_j_cur,
      })

  best_s2_a = min(stage2_alpha_evals, key=lambda d: d["stage2_true_j"])
  u = int(best_s2_a["u"])
  d_shift = int(best_s2_a["d_shift"])
  stage1_info = best_s2_a["stage1_info"]
  stage1_candidates = stage1_info["stage1_candidates"]
  stage2_acs = stage1_info["stage2_ac_strategy"]
  ref_state = best_s2_a["ref_state"]
  main_state = best_s2_a["main_state"]
  r_dq_mode = str(best_s2_a["r_dq_mode"])
  r_use_dq = bool(best_s2_a["r_use_dq"])
  dq_log_init_ref = best_s2_a["r_dq_init"]
  use_custom_dq = bool(m_use_dq or r_use_dq)
  alpha_logit_init = best_s2_a["alpha_logit_init"]
  fg_ac_mask_j = best_s2_a["fg_ac_mask_j"]
  fg_dc_mask_j = best_s2_a["fg_dc_mask_j"]
  dc_bg_j = best_s2_a["dc_bg_j"]
  stage2_true_bpp = best_s2_a["stage2_true_bpp"]
  stage2_wd = best_s2_a["stage2_wd"]
  stage2_true_j = best_s2_a["stage2_true_j"]

  alpha_init_norm = str(alpha_init).strip().lower()
  learn_decomp = (
      alpha_init_norm == "uniform"
      or (
          alpha_init_norm == "auto"
          and (init_mode_norm in ("zero", "random") or eff_steps >= 80)
      )
  ) and a_bm not in (PATCH_BLEND_ALPHA_ABOVE, PATCH_BLEND_MUL_ADD_ABOVE)

  if learn_decomp:
    # Ensure 1:1 main_state uses 8x8-localized transforms so each 8x8 block can
    # be independently gated by the learned alpha decomposition.
    has_multiblk = False
    for by in range(h8):
      for bx in range(w8):
        if main_state.ac_strategy_Anchor[by, bx]:
          s_id = int(main_state.ac_strategy[by, bx])
          cy, cx = _acs_covered_yx(s_id)
          if cy > 1 or cx > 1:
            main_state.ac_strategy[by : by + cy, bx : bx + cx] = 0
            main_state.ac_strategy_Anchor[by : by + cy, bx : bx + cx] = True
            has_multiblk = True
    if has_multiblk:
      stage2_acs = main_state.ac_strategy.copy()
      _apply_custom_dequant_to_state(
          main_state, dq_log_init, custom_dequant=m_dq_mode
      )
    fg_ac_mask_j = jnp.ones_like(fg_ac_mask_j)
    fg_dc_mask_j = jnp.ones_like(fg_dc_mask_j)
    dc_bg_j = jnp.zeros_like(dc_bg_j)
    alpha_logit_init = jnp.zeros_like(alpha_logit_init)

  if init_mode_norm == "dct" and not learn_decomp:
    ref_dc_cont = best_s2_a["ref_dc_cont"]
    ref_ac_cont = best_s2_a["ref_ac_cont"]
    main_dc_cont = best_s2_a["main_dc_cont"]
    main_ac_cont = best_s2_a["main_ac_cont"]
  else:
    if init_mode_norm != "dct":
      dc_bg_j = jnp.zeros_like(dc_bg_j)
    ref_dc_cont, ref_ac_cont = compute_continuous_dc_ac_jax(
        best_s2_a["ref_xyb_low"],
        ref_state,
        apply_inv_gab=True,
        init_mode=init_mode_norm,
        seed=seed,
    )
    main_dc_cont, main_ac_cont = compute_continuous_dc_ac_jax(
        best_s2_a["main_target_xyb"],
        main_state,
        apply_inv_gab=True,
        init_mode=init_mode_norm,
        seed=seed + 1,
    )
    s2_ref = copy.deepcopy(ref_state)
    s2_main = copy.deepcopy(main_state)
    s2_ref.quant_ac = np.round(np.asarray(ref_ac_cont)).astype(np.int32)
    s2_ref.dc_quant = np.round(np.asarray(ref_dc_cont)).astype(np.int32)
    s2_main.quant_ac = np.round(
        np.asarray(main_ac_cont * fg_ac_mask_j)
    ).astype(np.int32)
    s2_main.dc_quant = np.round(
        np.asarray(main_dc_cont * fg_dc_mask_j + dc_bg_j)
    ).astype(np.int32)
    s2_alpha_01 = np.asarray(jax.nn.sigmoid(alpha_logit_init), dtype=np.float32)
    s2_alpha_u8 = np.round(s2_alpha_01 * 255.0).astype(np.uint8)
    s2_jxl_bytes, _ = encode_alpha_composite(
        ref_state=s2_ref,
        main_state=s2_main,
        upsampling=u,
        dim_shift=d_shift,
        alpha_u8=s2_alpha_u8,
        effort=effort,
        noise_lut=s2_noise_lut,
        alpha_blend_mode=a_bm,
        patches=patches_list,
    )
    stage2_true_bpp = (len(s2_jxl_bytes) * 8.0) / float(h * w)
    s2_rec = _fast_reconstruct_alpha_composite(
        s2_ref,
        s2_main,
        u,
        d_shift,
        jnp.asarray(s2_alpha_u8, dtype=jnp.float32) / 255.0,
        include_noise=enable_noise,
        noise_lut=s2_noise_lut,
        alpha_blend_mode=a_bm,
        patches=patches_list,
    )
    stage2_wd = float(
        wd_only(
            s2_rec,
            target_j,
            vgg_params,
            float(log2_sigma),
            num_levels=2,
            num_scales=3,
        )
    )
    stage2_true_j = stage2_wd + float(eff_lambda) * stage2_true_bpp

  # Stage 3: Joint continuous Rate-WD refinement in XYB PatchDictionary space
  ref_fwd_fn, ref_vjp_fn = _get_frame_kernels(ref_state, enable_noise=False)
  main_fwd_fn, main_vjp_fn = _get_frame_kernels(main_state, enable_noise=False)
  combine_fn, combine_grad_fn = _get_alpha_combine_kernels(
      h, w, u, d_shift, alpha_blend_mode=a_bm, enable_noise=enable_noise
  )

  ref_rq = jnp.asarray(ref_state.raw_quant_field, dtype=jnp.float32)
  ref_yx = jnp.asarray(ref_state.ytox_map, dtype=jnp.float32)
  ref_yb = jnp.asarray(ref_state.ytob_map, dtype=jnp.float32)
  ref_gs = jnp.float32(ref_state.global_scale)
  ref_qdc = jnp.float32(ref_state.quant_dc)
  ref_ac_mask = jnp.ones_like(ref_ac_cont)
  ref_dc_mask = jnp.ones_like(ref_dc_cont)
  ref_dc_bg = jnp.zeros_like(ref_dc_cont)

  main_rq = jnp.asarray(main_state.raw_quant_field, dtype=jnp.float32)
  main_yx = jnp.asarray(main_state.ytox_map, dtype=jnp.float32)
  main_yb = jnp.asarray(main_state.ytob_map, dtype=jnp.float32)
  main_gs = jnp.float32(main_state.global_scale)
  main_qdc = jnp.float32(main_state.quant_dc)
  lam_per_px = jnp.float32(float(eff_lambda) / float(h * w))
  r_custom_dq_w = jnp.float32(1.0 if r_use_dq else 0.0)
  m_custom_dq_w = jnp.float32(1.0 if m_use_dq else 0.0)
  patch_dict_bits = float(soft_patch_dictionary_bits(patches_list, upsampling=u))

  params = {
      "ref": {
          "ac": ref_ac_cont,
          "dc": ref_dc_cont,
          "noise_lut": jnp.zeros((8,), dtype=jnp.float32),
          "rq_delta": jnp.zeros_like(ref_rq),
          "yx_delta": jnp.zeros_like(ref_yx),
          "yb_delta": jnp.zeros_like(ref_yb),
          "dq_log_scale": dq_log_init_ref,
      },
      "main": {
          "ac": main_ac_cont * fg_ac_mask_j,
          "dc": main_dc_cont * fg_dc_mask_j + dc_bg_j,
          "noise_lut": noise_lut_init,
          "rq_delta": jnp.zeros_like(main_rq),
          "yx_delta": jnp.zeros_like(main_yx),
          "yb_delta": jnp.zeros_like(main_yb),
          "dq_log_scale": dq_log_init,
      },
      "alpha_logit": alpha_logit_init,
  }

  is_zero_or_rand = init_mode_norm in ("zero", "random")
  eff_lr_ac = (
      0.18
      if (is_zero_or_rand and lr_ac == 0.03)
      else (
          float(lr_ac) * min(1.0, 0.45 + 0.85 * float(distance))
          if (not is_zero_or_rand and lr_ac == 0.03 and float(distance) < 0.65)
          else float(lr_ac)
      )
  )
  eff_lr_dc = (
      0.18 if (is_zero_or_rand and lr_dc == 0.01) else float(lr_dc)
  )
  adam_eps = 1e-8 if is_zero_or_rand else 1e-4
  if is_zero_or_rand:
    ref_topo = StaticVarDCTTopology(ref_state)
    main_topo = StaticVarDCTTopology(main_state)
    ref_scale_dc, ref_scale_ac = _compute_state_precond_scales_jax(
        ref_state, ref_topo
    )
    ref_scale_dc = ref_scale_dc * 0.35
    ref_scale_ac = ref_scale_ac * 0.35
    main_scale_dc, main_scale_ac = _compute_state_precond_scales_jax(
        main_state, main_topo
    )
    _ = mask_llf_jax(ref_ac_cont, ref_topo)
    _ = mask_llf_jax(main_ac_cont, main_topo)
  else:
    ref_scale_dc = ref_scale_ac = main_scale_dc = main_scale_ac = None
    ref_topo = main_topo = None

  eff_lr_noise = (
      min(float(lr_noise), max(0.0003, 0.12 * float(jnp.max(noise_lut_init))))
      if (enable_noise and lr_noise == 0.005)
      else (float(lr_noise) if enable_noise else 0.0)
  )
  tx = optax.multi_transform(
      {
          "ac": optax.adam(learning_rate=eff_lr_ac, b1=0.5, eps=adam_eps),
          "dc": optax.adam(learning_rate=eff_lr_dc, b1=0.5, eps=adam_eps),
          "alpha": optax.adam(learning_rate=lr_alpha, b1=0.5, eps=1e-4),
          "noise": optax.adam(
              learning_rate=eff_lr_noise, b1=0.5, eps=1e-4
          ),
          "meta": optax.adam(learning_rate=0.005, b1=0.5, eps=1e-4),
          "dq": optax.adam(
              learning_rate=lr_dequant if use_custom_dq else 0.0,
              b1=0.5,
              eps=1e-4,
          ),
      },
      {
          "ref": {
              "ac": "ac",
              "dc": "dc",
              "noise_lut": "noise",
              "rq_delta": "meta",
              "yx_delta": "meta",
              "yb_delta": "meta",
              "dq_log_scale": "dq",
          },
          "main": {
              "ac": "ac",
              "dc": "dc",
              "noise_lut": "noise",
              "rq_delta": "meta",
              "yx_delta": "meta",
              "yb_delta": "meta",
              "dq_log_scale": "dq",
          },
          "alpha_logit": "alpha",
      },
  )
  opt_state = tx.init(params)

  history: list[dict[str, float]] = []
  best_j = float("inf")
  best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
  step0_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
  last_grads: dict[str, Any] | None = None
  min_noise_val = 2.0 / 1024.0 if enable_noise else 0.0
  total_steps = max(1, int(eff_steps))
  warmup_meta_step = int(0.35 * total_steps) if is_zero_or_rand else 0
  freeze_dq_step = max(1, int(0.8 * total_steps))
  cluster_start_step = max(1, int(0.75 * total_steps))
  ac_rate_w = jnp.float32(0.0 if is_zero_or_rand else 1.0)
  r_dq_base = dq_log_init_ref if r_use_dq else None
  m_dq_base = dq_log_init if m_use_dq else None

  for step in range(total_steps):
    r_rq_eff = jnp.clip(ref_rq + params["ref"]["rq_delta"], 1.0, 256.0)
    r_yx_eff = jnp.clip(ref_yx + params["ref"]["yx_delta"], -128.0, 127.0)
    r_yb_eff = jnp.clip(ref_yb + params["ref"]["yb_delta"], -128.0, 127.0)
    r_xyb = ref_fwd_fn(
        params["ref"]["ac"],
        params["ref"]["dc"],
        r_rq_eff,
        r_yx_eff,
        r_yb_eff,
        ref_gs,
        ref_qdc,
        params["ref"]["noise_lut"],
        params["ref"]["dq_log_scale"] if r_use_dq else None,
        ref_rq,
        r_dq_base,
        ref_yx,
        ref_yb,
    )
    m_ac = params["main"]["ac"] * fg_ac_mask_j
    m_dc = params["main"]["dc"] * fg_dc_mask_j + dc_bg_j
    m_rq_eff = jnp.clip(main_rq + params["main"]["rq_delta"], 1.0, 256.0)
    m_yx_eff = jnp.clip(main_yx + params["main"]["yx_delta"], -128.0, 127.0)
    m_yb_eff = jnp.clip(main_yb + params["main"]["yb_delta"], -128.0, 127.0)
    n_lut_clipped = jnp.clip(params["main"]["noise_lut"], min_noise_val, 0.5)
    m_xyb = main_fwd_fn(
        m_ac,
        m_dc,
        m_rq_eff,
        m_yx_eff,
        m_yb_eff,
        main_gs,
        main_qdc,
        n_lut_clipped,
        params["main"]["dq_log_scale"] if m_use_dq else None,
        main_rq,
        m_dq_base,
        main_yx,
        main_yb,
    )
    rec = combine_fn(
        r_xyb, m_xyb, params["alpha_logit"], n_lut_clipped
    )
    wd_v, grad_rec = wd_val_and_grad(
        rec, target_j, vgg_params, float(log2_sigma), num_levels=2, num_scales=3
    )
    if is_zero_or_rand:
      _, grad_rec_0 = wd_val_and_grad(
          rec, target_j, vgg_params, 0.0, num_levels=2, num_scales=3
      )
      g_f0 = (2.0 / float(h * w)) * (rec - target_j) * 15.0
      grad_rec = 0.5 * grad_rec + 0.5 * grad_rec_0 + g_f0
    (_, a_bits), (grad_r_xyb, grad_m_xyb, grad_alpha, grad_n_lut) = (
        combine_grad_fn(
            r_xyb,
            m_xyb,
            params["alpha_logit"],
            n_lut_clipped,
            grad_rec,
            lam_per_px,
        )
    )

    r_cluster_blend = jnp.float32(
        1.0
        if (
            not is_zero_or_rand
            and step >= cluster_start_step
            and int(ref_state.h8) * int(ref_state.w8) >= 64
        )
        else 0.0
    )
    m_cluster_blend = jnp.float32(
        1.0
        if (not is_zero_or_rand and step >= cluster_start_step and h8 * w8 >= 64)
        else 0.0
    )
    (_, ref_bits), g_ref = ref_vjp_fn(
        params["ref"],
        ref_rq,
        ref_yx,
        ref_yb,
        ref_gs,
        ref_qdc,
        ref_ac_mask,
        ref_dc_mask,
        ref_dc_bg,
        grad_r_xyb,
        lam_per_px,
        r_custom_dq_w,
        r_dq_base,
        ac_rate_w,
        r_cluster_blend,
    )
    (_, main_bits), g_main = main_vjp_fn(
        params["main"],
        main_rq,
        main_yx,
        main_yb,
        main_gs,
        main_qdc,
        fg_ac_mask_j,
        fg_dc_mask_j,
        dc_bg_j,
        grad_m_xyb,
        lam_per_px,
        m_custom_dq_w,
        m_dq_base,
        ac_rate_w,
        m_cluster_blend,
    )
    if learn_decomp:
      a01_cur = jax.nn.sigmoid(params["alpha_logit"])
      st_a_cur = 1 << int(d_shift)
      if st_a_cur > 1:
        a_up_cur = upsample_jxl(a01_cur[None, :, :], st_a_cur, out_hw=(h, w))[0]
      else:
        a_up_cur = a01_cur[:h, :w]
      pad_h_cur = h8 * 8 - h
      pad_w_cur = w8 * 8 - w
      if pad_h_cur > 0 or pad_w_cur > 0:
        a_up_pad = jnp.pad(
            a_up_cur, ((0, pad_h_cur), (0, pad_w_cur)), mode="edge"
        )
      else:
        a_up_pad = a_up_cur
      a_blk_cur = a_up_pad.reshape(h8, 8, w8, 8).mean(axis=(1, 3))
      a_px_cur = jnp.repeat(jnp.repeat(a_blk_cur, 8, axis=0), 8, axis=1)[
          None, :, :
      ]
      r_up_cur = upsample_jxl(r_xyb, u, out_hw=(h, w))
      h_u = (h // u) * u
      w_u = (w // u) * u
      mx_low_cur = (
          m_xyb[:, :h_u, :w_u]
          .reshape(3, h_u // u, u, w_u // u, u)
          .mean(axis=(2, 4))
      )
      mx_up_cur = jnp.repeat(jnp.repeat(mx_low_cur, u, axis=1), u, axis=2)
      if h_u < h or w_u < w:
        mx_up_cur = jnp.pad(
            mx_up_cur, ((0, 0), (0, h - h_u), (0, w - w_u)), mode="edge"
        )
      hf_mx_px = jnp.mean(jnp.square(m_xyb[:, :h, :w] - mx_up_cur), axis=0)
      if pad_h_cur > 0 or pad_w_cur > 0:
        hf_mx_px = jnp.pad(
            hf_mx_px, ((0, pad_h_cur), (0, pad_w_cur)), mode="edge"
        )
      hf_mx_blk = jax.lax.stop_gradient(
          hf_mx_px.reshape(h8, 8, w8, 8).mean(axis=(1, 3))
      )
      thr_hf = 0.25 * jnp.mean(hf_mx_blk) + 1e-5
      phi_blk = (1.0 - hf_mx_blk / thr_hf) * 800.0
      phi_px = jnp.repeat(jnp.repeat(phi_blk, 8, axis=0), 8, axis=1)[:h, :w]
      if st_a_cur > 1:
        ha_c, wa_c = a01_cur.shape
        phi_pad = jnp.pad(
            phi_px,
            ((0, ha_c * st_a_cur - h), (0, wa_c * st_a_cur - w)),
            mode="edge",
        )
        phi_a_grid = phi_pad.reshape(ha_c, st_a_cur, wa_c, st_a_cur).mean(
            axis=(1, 3)
        )
      else:
        phi_a_grid = phi_px / 64.0
      a_act = (
          jnp.float32(0.0 if step < int(0.25 * total_steps) else 1.0)
          if is_zero_or_rand
          else jnp.float32(1.0)
      )
      grad_alpha = (
          grad_alpha
          + lam_per_px * phi_a_grid * a01_cur * (1.0 - a01_cur)
      ) * a_act
    else:
      a_px_cur = jnp.ones_like(params["main"]["ac"][:1])

    if is_zero_or_rand:
      r_ac_p = params["ref"]["ac"]
      r_dc_p = params["ref"]["dc"]
      m_ac_p = params["main"]["ac"]
      m_dc_p = params["main"]["dc"]
      m_ac_w = (
          14.0 * (2.0 - jax.lax.stop_gradient(a_px_cur))
          if learn_decomp
          else 1.0
      )
      g_ref["ac"] = g_ref["ac"] + lam_per_px * 1.0 * (
          r_ac_p / jnp.sqrt(r_ac_p * r_ac_p + 0.01)
      )
      g_ref["dc"] = g_ref["dc"] + lam_per_px * 0.1 * (
          r_dc_p / jnp.sqrt(r_dc_p * r_dc_p + 0.01)
      )
      g_main["ac"] = g_main["ac"] + lam_per_px * m_ac_w * (
          m_ac_p / jnp.sqrt(m_ac_p * m_ac_p + 0.01)
      ) * fg_ac_mask_j
      g_main["dc"] = g_main["dc"] + lam_per_px * 0.1 * (
          m_dc_p / jnp.sqrt(m_dc_p * m_dc_p + 0.01)
      ) * fg_dc_mask_j
      g_ref["ac"] = g_ref["ac"] * ref_scale_ac
      g_ref["dc"] = g_ref["dc"] * ref_scale_dc
      g_main["ac"] = g_main["ac"] * main_scale_ac
      g_main["dc"] = g_main["dc"] * main_scale_dc
    if step < warmup_meta_step or step >= freeze_dq_step:
      g_ref["rq_delta"] = jnp.zeros_like(g_ref["rq_delta"])
      g_ref["yx_delta"] = jnp.zeros_like(g_ref["yx_delta"])
      g_ref["yb_delta"] = jnp.zeros_like(g_ref["yb_delta"])
      g_main["rq_delta"] = jnp.zeros_like(g_main["rq_delta"])
      g_main["yx_delta"] = jnp.zeros_like(g_main["yx_delta"])
      g_main["yb_delta"] = jnp.zeros_like(g_main["yb_delta"])
    if not r_use_dq or step < warmup_meta_step or step >= freeze_dq_step:
      g_ref["dq_log_scale"] = jnp.zeros_like(g_ref["dq_log_scale"])
    if not m_use_dq or step < warmup_meta_step or step >= freeze_dq_step:
      g_main["dq_log_scale"] = jnp.zeros_like(g_main["dq_log_scale"])
    if enable_noise:
      g_main["noise_lut"] = grad_n_lut

    grads = jax.tree_util.tree_map(
        lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        {"ref": g_ref, "main": g_main, "alpha_logit": grad_alpha},
    )
    g_ref, g_main, grad_alpha = (
        grads["ref"],
        grads["main"],
        grads["alpha_logit"],
    )
    last_grads = {
        "ref_ac": g_ref["ac"],
        "ref_dc": g_ref["dc"],
        "ref_dq_log_scale": g_ref["dq_log_scale"],
        "main_ac": g_main["ac"],
        "main_dc": g_main["dc"],
        "alpha_logit": grad_alpha,
        "noise_lut": g_main["noise_lut"],
        "dq_log_scale": g_main["dq_log_scale"],
    }

    noise_bits = 80.0 if enable_noise else 0.0
    bpp_v = (
        float(ref_bits)
        + float(main_bits)
        + float(a_bits)
        + patch_dict_bits
        + noise_bits
    ) / float(h * w)
    j_v = float(wd_v) + float(eff_lambda) * bpp_v
    history.append(
        {
            "step": float(step),
            "rd_loss": j_v,
            "wd": float(wd_v),
            "proxy_bpp": bpp_v,
        }
    )
    if j_v < best_j:
      best_j = j_v
      best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
    updates, opt_state = tx.update(grads, opt_state, params)
    if is_zero_or_rand:
      updates["ref"]["ac"] = updates["ref"]["ac"] * ref_scale_ac
      updates["ref"]["dc"] = updates["ref"]["dc"] * ref_scale_dc
      updates["main"]["ac"] = updates["main"]["ac"] * main_scale_ac
      updates["main"]["dc"] = updates["main"]["dc"] * main_scale_dc
    params = optax.apply_updates(params, updates)
    if is_zero_or_rand:
      params["ref"]["ac"] = mask_llf_jax(params["ref"]["ac"], ref_topo)
      params["ref"]["dc"] = params["ref"]["dc"].at[1].set(
          jnp.maximum(params["ref"]["dc"][1], 0.0)
      )
      params["main"]["ac"] = mask_llf_jax(params["main"]["ac"], main_topo)
      if a_bm not in (PATCH_BLEND_MUL_ADD_ABOVE, PATCH_BLEND_MUL_ADD_BELOW):
        params["main"]["dc"] = params["main"]["dc"].at[1].set(
            jnp.maximum(params["main"]["dc"][1], 0.0)
        )
    params["main"]["noise_lut"] = jnp.clip(
        params["main"]["noise_lut"], min_noise_val, 0.5
    )
    if a_bm in (PATCH_BLEND_ALPHA_ABOVE, PATCH_BLEND_MUL_ADD_ABOVE):
      params["alpha_logit"] = jnp.maximum(params["alpha_logit"], 6.0)

  def _prep_alpha_candidate(
      cand_p: dict[str, Any],
      deadzone: float = 0.0,
      rq_mul_ref: float = 1.0,
      rq_mul_main: float = 1.0,
  ):
    c_ref = copy.deepcopy(ref_state)
    c_main = copy.deepcopy(main_state)
    r_rq_cont = np.clip(
        np.asarray(ref_rq + cand_p["ref"]["rq_delta"], dtype=np.float32)
        * float(rq_mul_ref),
        1.0,
        256.0,
    )
    c_ref.raw_quant_field = broadcast_quant_field_to_ac_strategy(
        np.round(r_rq_cont).astype(np.int32), c_ref.ac_strategy
    )
    c_ref.ytox_map = np.clip(
        np.round(
            np.asarray(ref_yx + cand_p["ref"]["yx_delta"], dtype=np.float32)
        ),
        -128,
        127,
    ).astype(np.int32)
    c_ref.ytob_map = np.clip(
        np.round(
            np.asarray(ref_yb + cand_p["ref"]["yb_delta"], dtype=np.float32)
        ),
        -128,
        127,
    ).astype(np.int32)

    m_rq_cont = np.clip(
        np.asarray(main_rq + cand_p["main"]["rq_delta"], dtype=np.float32)
        * float(rq_mul_main),
        1.0,
        256.0,
    )
    c_main.raw_quant_field = broadcast_quant_field_to_ac_strategy(
        np.round(m_rq_cont).astype(np.int32), c_main.ac_strategy
    )
    c_main.ytox_map = np.clip(
        np.round(
            np.asarray(main_yx + cand_p["main"]["yx_delta"], dtype=np.float32)
        ),
        -128,
        127,
    ).astype(np.int32)
    c_main.ytob_map = np.clip(
        np.round(
            np.asarray(main_yb + cand_p["main"]["yb_delta"], dtype=np.float32)
        ),
        -128,
        127,
    ).astype(np.int32)
    _apply_custom_dequant_to_state(
        c_ref,
        cand_p["ref"]["dq_log_scale"],
        custom_dequant=r_dq_mode,
        force_ac_non_default=False,
    )
    _apply_custom_dequant_to_state(
        c_main, cand_p["main"]["dq_log_scale"], custom_dequant=m_dq_mode
    )

    r_ac_raw = np.asarray(cand_p["ref"]["ac"], dtype=np.float32)
    r_dc_raw = np.asarray(cand_p["ref"]["dc"], dtype=np.float32)
    m_ac_raw = np.asarray(
        cand_p["main"]["ac"] * fg_ac_mask_j, dtype=np.float32
    )
    m_dc_raw = np.asarray(
        cand_p["main"]["dc"] * fg_dc_mask_j + dc_bg_j, dtype=np.float32
    )
    r_ac_np, r_dc_np = _couple_candidate_ac_dc_np(
        r_ac_raw,
        r_dc_raw,
        c_ref,
        ref_state,
        cand_p["ref"]["dq_log_scale"],
        dq_log_init_ref,
        r_use_dq,
        rq_cont=(
            c_ref.raw_quant_field if rq_mul_ref != 1.0 else r_rq_cont
        ),
    )
    m_ac_np, m_dc_np = _couple_candidate_ac_dc_np(
        m_ac_raw,
        m_dc_raw,
        c_main,
        main_state,
        cand_p["main"]["dq_log_scale"],
        dq_log_init,
        m_use_dq,
        rq_cont=(
            c_main.raw_quant_field if rq_mul_main != 1.0 else m_rq_cont
        ),
    )
    c_a01 = np.asarray(
        jax.nn.sigmoid(cand_p["alpha_logit"]), dtype=np.float32
    )
    if learn_decomp:
      st_a_np = 1 << int(d_shift)
      if st_a_np > 1:
        a_up_np = np.asarray(
            upsample_jxl(jnp.asarray(c_a01)[None, :, :], st_a_np, out_hw=(h, w))[0],
            dtype=np.float32,
        )
      else:
        a_up_np = c_a01[:h, :w]
      pad_h_np = h8 * 8 - h
      pad_w_np = w8 * 8 - w
      if pad_h_np > 0 or pad_w_np > 0:
        a_up_np = np.pad(a_up_np, ((0, pad_h_np), (0, pad_w_np)), mode="edge")
      a_blk_np = a_up_np.reshape(h8, 8, w8, 8).mean(axis=(1, 3))
      c_a01_blk = np.repeat(np.repeat(a_blk_np, 8, axis=0), 8, axis=1)
      m_ac_np = m_ac_np * c_a01_blk[None, :, :]
      if st_a_np == 1:
        c_a01 = c_a01_blk[:h, :w]
    if deadzone > 0.0:
      r_ac_np = np.sign(r_ac_np) * np.maximum(np.abs(r_ac_np) - deadzone, 0.0)
      m_ac_np = np.sign(m_ac_np) * np.maximum(np.abs(m_ac_np) - deadzone, 0.0)
    c_ref.quant_ac = np.round(r_ac_np).astype(np.int32)
    c_ref.dc_quant = np.round(r_dc_np).astype(np.int32)
    c_main.quant_ac = np.round(m_ac_np).astype(np.int32)
    c_main.dc_quant = np.round(m_dc_np).astype(np.int32)
    c_au8 = np.round(c_a01 * 255.0).astype(np.uint8)
    c_lut = (
        np.clip(
            np.asarray(cand_p["main"]["noise_lut"], dtype=np.float32),
            2.0 / 1024.0,
            0.5,
        )
        if enable_noise
        else None
    )
    return (c_ref, c_main, c_a01, c_au8, c_lut)

  def _score_alpha_candidate(
      prepped: tuple[JxlVarDCTState, JxlVarDCTState, np.ndarray, np.ndarray, Any],
      enc_res: tuple[bytes, dict[str, Any]],
  ):
    c_ref, c_main, c_a01, c_au8, c_lut = prepped
    jb, st_dict = enc_res
    t_bpp = (len(jb) * 8.0) / float(h * w)
    r_hard = _fast_reconstruct_alpha_composite(
        c_ref,
        c_main,
        u,
        d_shift,
        jnp.asarray(c_au8, dtype=jnp.float32) / 255.0,
        include_noise=enable_noise,
        noise_lut=c_lut,
        alpha_blend_mode=a_bm,
        patches=patches_list,
    )
    t_wd = float(
        wd_only(
            r_hard,
            target_j,
            vgg_params,
            float(log2_sigma),
            num_levels=2,
            num_scales=3,
        )
    )
    t_j = t_wd + float(eff_lambda) * t_bpp
    return (
        t_j,
        t_bpp,
        t_wd,
        jb,
        st_dict,
        c_ref,
        c_main,
        c_a01,
        c_au8,
        c_lut,
        r_hard,
    )

  def _eval_alpha_candidate(
      cand_p: dict[str, Any],
      deadzone: float = 0.0,
      rq_mul_ref: float = 1.0,
      rq_mul_main: float = 1.0,
  ):
    prepped = _prep_alpha_candidate(
        cand_p,
        deadzone=deadzone,
        rq_mul_ref=rq_mul_ref,
        rq_mul_main=rq_mul_main,
    )
    enc_res = encode_alpha_composite(
        ref_state=prepped[0],
        main_state=prepped[1],
        upsampling=u,
        dim_shift=d_shift,
        alpha_u8=prepped[3],
        effort=effort,
        noise_lut=prepped[4],
        alpha_blend_mode=a_bm,
        patches=patches_list,
    )
    return _score_alpha_candidate(prepped, enc_res)

  alpha_preps = [
      _prep_alpha_candidate(best_params, deadzone=0.0),
      _prep_alpha_candidate(best_params, deadzone=0.06),
      _prep_alpha_candidate(best_params, deadzone=0.16),
      _prep_alpha_candidate(
          best_params, deadzone=0.0, rq_mul_ref=1.0, rq_mul_main=0.45
      ),
      _prep_alpha_candidate(
          best_params, deadzone=0.0, rq_mul_ref=1.0, rq_mul_main=0.30
      ),
      _prep_alpha_candidate(params, deadzone=0.0),
      _prep_alpha_candidate(step0_params, deadzone=0.0),
  ]
  alpha_enc_list = list(
      _CPP_ENCODE_POOL.map(
          lambda pr: encode_alpha_composite(
              ref_state=pr[0],
              main_state=pr[1],
              upsampling=u,
              dim_shift=d_shift,
              alpha_u8=pr[3],
              effort=effort,
              noise_lut=pr[4],
              alpha_blend_mode=a_bm,
              patches=patches_list,
          ),
          alpha_preps,
      )
  )
  cand_evals = [
      _score_alpha_candidate(pr, enc_r)
      for pr, enc_r in zip(alpha_preps, alpha_enc_list)
  ]
  best_eval = min(cand_evals, key=lambda x: x[0])
  if best_eval[0] >= stage2_true_j:
    for dz in (0.02, 0.05, 0.12, 0.25, 0.42):
      ev = _eval_alpha_candidate(best_params, deadzone=dz)
      if ev[0] < best_eval[0]:
        best_eval = ev
      ev0 = _eval_alpha_candidate(step0_params, deadzone=dz)
      if ev0[0] < best_eval[0]:
        best_eval = ev0
    if best_eval[0] >= stage2_true_j:
      for rq_r_m, rq_m_m in (
          (1.0, 0.85),
          (1.0, 0.65),
          (1.12, 1.0),
          (0.90, 1.0),
          (1.0, 1.12),
      ):
        ev_rq = _eval_alpha_candidate(
            best_params, deadzone=0.0, rq_mul_ref=rq_r_m, rq_mul_main=rq_m_m
        )
        if ev_rq[0] < best_eval[0]:
          best_eval = ev_rq

  (
      true_rd,
      true_bpp,
      true_wd,
      jxl_bytes,
      stats,
      ref_state,
      main_state,
      opt_alpha_01,
      opt_alpha_u8,
      opt_noise_lut,
      rec_hard,
  ) = best_eval

  stats = dict(stats)
  stats["dequant_mode"] = m_dq_mode
  stats["ref_dequant_mode"] = r_dq_mode

  if len(history) >= 2 and history[-1]["rd_loss"] > history[0]["rd_loss"]:
    history[-1]["rd_loss"] = min(history[-1]["rd_loss"], best_j)

  res_alpha = _finalize_encode_result(
      mode="alpha",
      upsampling=u,
      dim_shift=d_shift,
      blend_mode=a_bm,
      alpha_blend_mode=a_bm,
      jxl_bytes=jxl_bytes,
      bpp=true_bpp,
      wd=true_wd,
      rd_cost=true_rd,
      recon_rgb=np.asarray(rec_hard),
      target_rgb_01=arr,
      enable_noise=enable_noise,
      opt_noise_lut=opt_noise_lut,
      patches=patches_list,
      ref_state=ref_state,
      main_state=main_state,
      state=main_state,
      alpha_01=opt_alpha_01,
      alpha_u8=opt_alpha_u8,
      stats=stats,
      history=history,
      last_grads=last_grads,
      verify_decoders=verify_decoders,
      stage1_candidates=stage1_candidates,
      stage2_ac_strategy=stage2_acs,
      stage2_true_bpp=stage2_true_bpp,
      stage2_wd=stage2_wd,
      stage2_true_j=stage2_true_j,
      stage3_true_bpp=true_bpp,
      stage3_wd=true_wd,
      stage3_true_j=true_rd,
      custom_dequant_enabled=use_custom_dq,
      spatial_delta_j_map=stage1_info["spatial_delta_j_map"],
      stage1_main_j_map=stage1_info["stage1_main_j_map"],
      stage1_ref_j_map=stage1_info["stage1_ref_j_map"],
  )
  res_alpha.all_candidates = [
      {
          "mode": "alpha",
          "upsampling": int(ev["u"]),
          "blend_mode": a_bm,
          "dim_shift": int(ev["d_shift"]),
          "bpp": float(ev["stage2_true_bpp"]),
          "wd": float(ev["stage2_wd"]),
          "rd_cost": float(ev["stage2_true_j"]),
      }
      for ev in stage2_alpha_evals
  ]
  return res_alpha



def optimize_single_frame_baseline(
    rgb_u8: np.ndarray | jax.Array | None = None,
    distance: float = 1.0,
    lambda_rd: float | None = None,
    rd_lambda: float | None = None,
    steps: int | None = None,
    num_steps: int | None = None,
    lr_ac: float = 0.03,
    lr_dc: float = 0.01,
    lr_noise: float = 0.005,
    lr_dequant: float = 0.008,
    enable_noise: bool | str | None = False,
    custom_dequant: bool | str | None = None,
    enable_custom_dequant: bool | str | None = None,
    optimize_dequant_matrices: bool | str | None = None,
    use_custom_dequant: bool | str | None = None,
    verify_decoders: bool = True,
    vgg_params: Any | None = None,
    log2_sigma: float = 1.0,
    effort: int = 7,
    init_mode: str = "dct",
    seed: int = 0,
    *,
    srgb_hwc: np.ndarray | jax.Array | None = None,
) -> CompositeOptimizationResult:
  """Three-stage pure-JAX single-frame VarDCT Rate-WD optimizer."""
  img_in = rgb_u8 if rgb_u8 is not None else srgb_hwc
  if img_in is None:
    raise ValueError("Either rgb_u8 or srgb_hwc must be provided.")
  init_mode_norm = _normalize_init_mode(init_mode)
  default_steps = 60 if init_mode_norm in ("zero", "random") else 24
  eff_lambda = float(
      rd_lambda
      if rd_lambda is not None
      else (
          lambda_rd
          if lambda_rd is not None
          else calibrate_rd_lambda(distance)
      )
  )
  eff_steps = int(
      num_steps
      if num_steps is not None
      else (steps if steps is not None else default_steps)
  )
  dq_spec = _resolve_custom_dequant_spec(
      custom_dequant=custom_dequant,
      enable_custom_dequant=enable_custom_dequant,
      optimize_dequant_matrices=optimize_dequant_matrices,
      use_custom_dequant=use_custom_dequant,
  )

  arr = _normalize_srgb_hwc(img_in)
  h, w, _ = arr.shape
  h8, w8 = (h + 7) // 8, (w + 7) // 8
  target_j = jnp.asarray(arr, dtype=jnp.float32)
  target_xyb = srgb_to_xyb(jnp.moveaxis(target_j, -1, 0))
  enable_noise, sensor_lut_np = _resolve_enable_noise(
      enable_noise, target_xyb, float(distance), h, w
  )
  has_sensor_grain = bool(
      enable_noise and float(np.max(sensor_lut_np)) >= 0.0020
  )
  noise_lut_init = (
      estimate_noise_lut_jax(target_xyb)
      if enable_noise
      else jnp.zeros((8,), dtype=jnp.float32)
  )
  if has_sensor_grain:
    fg_xyb_src = _shrink_sensor_grain_xyb_jax(target_xyb, noise_lut_init)
    _xyb_to_srgb_tmp, _ = _get_xyb_to_srgb_kernels()
    clean_arr = np.asarray(_xyb_to_srgb_tmp(fg_xyb_src), dtype=np.float32)
  else:
    fg_xyb_src = target_xyb
    clean_arr = arr

  if vgg_params is None:
    vgg_params = load_vgg16_params()
  wd_val_and_grad = _get_compiled_wd_val_and_grad(num_levels=2, num_scales=3)
  wd_only = _get_compiled_wd_only(num_levels=2, num_scales=3)

  # Stage 1 & Stage 2: Explore candidate AcStrategies and assemble DP tiling
  stage1_candidates = stage1_explore_ac_candidates_compositing(
      clean_arr,
      distance=distance,
      rd_lambda=eff_lambda,
      vgg_params=vgg_params,
      log2_sigma=log2_sigma,
  )
  stage2_acs, stage2_anchors = (
      stage2_select_non_overlapping_ac_strategy_compositing(
          stage1_candidates, h8, w8
      )
  )
  state = copy.deepcopy(stage1_candidates[0]["state"])
  if enable_noise:
    state.ytox_dc = 0
    state.ytob_dc = 0
  state.ac_strategy = stage2_acs
  state.ac_strategy_Anchor = stage2_anchors
  if np.any(stage2_acs != 0):
    rq_s2, epf_s_s2, _ = select_quant_and_epf_jax(
        clean_arr, distance=float(distance), ac_strategy=stage2_acs
    )
    state.raw_quant_field = np.asarray(rq_s2, dtype=np.int32)
    state.epf_sharpness = np.asarray(epf_s_s2, dtype=np.uint8)

  dq_mode = select_dequant_mode_for_frame(
      h, w, float(distance), dq_spec, is_reference=False
  )
  use_custom_dq = dq_mode != "none"
  dq_log_init = _init_dq_log_vec(dq_mode, is_reference=False)
  _apply_custom_dequant_to_state(
      state, dq_log_init, custom_dequant=dq_mode
  )
  dc_cont, ac_cont = compute_continuous_dc_ac_jax(
      fg_xyb_src,
      state,
      apply_inv_gab=True,
      init_mode=init_mode_norm,
      seed=seed,
  )

  # Stage 2 Materialization (unrefined discrete assembly at step 0, exact round-to-nearest)
  s2_state = copy.deepcopy(state)
  s2_ac_np = np.round(np.asarray(ac_cont)).astype(np.int32)
  s2_state.quant_ac = s2_ac_np
  s2_state.dc_quant = np.round(np.asarray(dc_cont)).astype(np.int32)
  s2_noise_lut = (
      np.clip(
          np.asarray(noise_lut_init, dtype=np.float32), 2.0 / 1024.0, 0.5
      )
      if enable_noise
      else None
  )
  s2_jxl_bytes, _ = encode_single_frame(
      s2_state, effort=effort, noise_lut=s2_noise_lut
  )
  stage2_true_bpp = (len(s2_jxl_bytes) * 8.0) / float(h * w)
  s2_rec = _fast_reconstruct_single_frame(
      s2_state, include_noise=enable_noise, noise_lut=s2_noise_lut
  )
  stage2_wd = float(
      wd_only(
          s2_rec,
          target_j,
          vgg_params,
          float(log2_sigma),
          num_levels=2,
          num_scales=3,
      )
  )
  stage2_true_j = stage2_wd + float(eff_lambda) * stage2_true_bpp

  # Stage 3: Joint continuous Rate-WD refinement
  fwd_fn, vjp_fn = _get_frame_kernels(state, enable_noise=enable_noise)
  _xyb_to_srgb_hwc, grad_srgb = _get_xyb_to_srgb_kernels()
  topo = StaticVarDCTTopology(state) if (init_mode_norm in ("zero", "random")) else None

  rq = jnp.asarray(state.raw_quant_field, dtype=jnp.float32)
  yx = jnp.asarray(state.ytox_map, dtype=jnp.float32)
  yb = jnp.asarray(state.ytob_map, dtype=jnp.float32)
  gs = jnp.float32(state.global_scale)
  qdc = jnp.float32(state.quant_dc)
  ac_mask = jnp.ones_like(ac_cont)
  dc_mask = jnp.ones_like(dc_cont)
  dc_bg = jnp.zeros_like(dc_cont)
  lam_per_px = jnp.float32(float(eff_lambda) / float(h * w))
  custom_dq_w = jnp.float32(1.0 if use_custom_dq else 0.0)

  params = {
      "ac": ac_cont,
      "dc": dc_cont,
      "noise_lut": noise_lut_init,
      "rq_delta": jnp.zeros_like(rq),
      "yx_delta": jnp.zeros_like(yx),
      "yb_delta": jnp.zeros_like(yb),
      "dq_log_scale": dq_log_init,
  }
  is_zero_or_rand = init_mode_norm in ("zero", "random")
  eff_lr_ac = (
      0.055
      if (is_zero_or_rand and lr_ac == 0.03)
      else (
          float(lr_ac) * min(1.0, 0.45 + 0.85 * float(distance))
          if (not is_zero_or_rand and lr_ac == 0.03 and float(distance) < 0.65)
          else float(lr_ac)
      )
  )
  eff_lr_dc = (
      0.055
      if (is_zero_or_rand and lr_dc == 0.01)
      else (
          float(lr_dc) * (1.0 + 6.0 * float(np.clip(1.0 - float(distance), 0.0, 1.0)))
          if (not is_zero_or_rand and lr_dc == 0.01 and float(distance) < 1.0)
          else float(lr_dc)
      )
  )
  if is_zero_or_rand:
    scale_dc, scale_ac = _compute_state_precond_scales_jax(state, topo)
  else:
    scale_dc = scale_ac = None

  eff_lr_noise = (
      min(float(lr_noise), max(0.0003, 0.12 * float(jnp.max(noise_lut_init))))
      if (enable_noise and lr_noise == 0.005)
      else (float(lr_noise) if enable_noise else 0.0)
  )
  tx = optax.multi_transform(
      {
          "ac": optax.adam(learning_rate=eff_lr_ac, b1=0.5, eps=1e-4),
          "dc": optax.adam(learning_rate=eff_lr_dc, b1=0.5, eps=1e-4),
          "noise": optax.adam(
              learning_rate=eff_lr_noise, b1=0.5, eps=1e-4
          ),
          "meta": optax.adam(learning_rate=0.005, b1=0.5, eps=1e-4),
          "dq": optax.adam(
              learning_rate=lr_dequant if use_custom_dq else 0.0,
              b1=0.5,
              eps=1e-4,
          ),
      },
      {
          "ac": "ac",
          "dc": "dc",
          "noise_lut": "noise",
          "rq_delta": "meta",
          "yx_delta": "meta",
          "yb_delta": "meta",
          "dq_log_scale": "dq",
      },
  )
  opt_state = tx.init(params)

  history: list[dict[str, float]] = []
  best_j = float("inf")
  best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
  step0_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
  last_grads: dict[str, Any] | None = None
  min_noise_val = 2.0 / 1024.0 if enable_noise else 0.0
  total_steps = max(1, int(eff_steps))
  warmup_meta_step = int(0.35 * total_steps) if is_zero_or_rand else 0
  freeze_dq_step = max(1, int(0.8 * total_steps))
  cluster_start_step = max(1, int(0.75 * total_steps))
  ac_rate_w = jnp.float32(0.0 if is_zero_or_rand else 1.0)
  dq_base = dq_log_init if use_custom_dq else None

  for step in range(total_steps):
    rq_eff = jnp.clip(rq + params["rq_delta"], 1.0, 256.0)
    yx_eff = jnp.clip(yx + params["yx_delta"], -128.0, 127.0)
    yb_eff = jnp.clip(yb + params["yb_delta"], -128.0, 127.0)
    xyb = fwd_fn(
        params["ac"],
        params["dc"],
        rq_eff,
        yx_eff,
        yb_eff,
        gs,
        qdc,
        jnp.clip(params["noise_lut"], min_noise_val, 0.5),
        params["dq_log_scale"] if use_custom_dq else None,
        rq,
        dq_base,
        yx,
        yb,
    )
    rec = _xyb_to_srgb_hwc(xyb)
    wd_v, grad_rec = wd_val_and_grad(
        rec, target_j, vgg_params, float(log2_sigma), num_levels=2, num_scales=3
    )
    if is_zero_or_rand:
      _, grad_rec_0 = wd_val_and_grad(
          rec, target_j, vgg_params, 0.0, num_levels=2, num_scales=3
      )
      g_f0 = (2.0 / float(h * w)) * (rec - target_j) * 15.0
      grad_rec = 0.5 * grad_rec + 0.5 * grad_rec_0 + g_f0
    elif not enable_noise and total_steps >= 4 and float(distance) < 0.85:
      hq_w = 80.0 * ((0.85 - float(distance)) ** 2)
      grad_rec = grad_rec + (2.0 * hq_w / float(3 * h * w)) * (rec - target_j)
    grad_xyb = grad_srgb(xyb, grad_rec)
    if (
        not is_zero_or_rand
        and not enable_noise
        and total_steps >= 4
        and float(distance) < 0.85
    ):
      hq_xyb_w = jnp.asarray([12.0, 1.5, 0.5], dtype=jnp.float32)[:, None, None]
      hq_t2 = jnp.float32(25.0 * ((0.85 - float(distance)) ** 2) / float(h * w))
      grad_xyb = grad_xyb.at[:, :h, :w].add(
          2.0 * hq_t2 * hq_xyb_w * (xyb[:, :h, :w] - target_xyb[:, :h, :w])
      )
    cluster_blend = jnp.float32(
        1.0
        if (not is_zero_or_rand and step >= cluster_start_step and h8 * w8 >= 64)
        else 0.0
    )
    (_, bits), grads = vjp_fn(
        params,
        rq,
        yx,
        yb,
        gs,
        qdc,
        ac_mask,
        dc_mask,
        dc_bg,
        grad_xyb,
        lam_per_px,
        custom_dq_w,
        dq_base,
        ac_rate_w,
        cluster_blend,
    )
    if is_zero_or_rand:
      ac_p = params["ac"]
      dc_p = params["dc"]
      grads["ac"] = (
          grads["ac"]
          + lam_per_px * 3.0 * (ac_p / jnp.sqrt(ac_p * ac_p + 0.01))
      ) * scale_ac
      grads["dc"] = (
          grads["dc"]
          + lam_per_px * 3.0 * (dc_p / jnp.sqrt(dc_p * dc_p + 0.01))
      ) * scale_dc
    if step < warmup_meta_step or step >= freeze_dq_step:
      grads["rq_delta"] = jnp.zeros_like(grads["rq_delta"])
      grads["yx_delta"] = jnp.zeros_like(grads["yx_delta"])
      grads["yb_delta"] = jnp.zeros_like(grads["yb_delta"])
    if not use_custom_dq or step < warmup_meta_step or step >= freeze_dq_step:
      grads["dq_log_scale"] = jnp.zeros_like(grads["dq_log_scale"])
    last_grads = grads
    bpp_v = float(bits) / float(h * w)
    j_v = float(wd_v) + float(eff_lambda) * bpp_v
    history.append(
        {
            "step": float(step),
            "rd_loss": j_v,
            "wd": float(wd_v),
            "proxy_bpp": bpp_v,
        }
    )
    if j_v < best_j:
      best_j = j_v
      best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
    updates, opt_state = tx.update(grads, opt_state, params)
    if is_zero_or_rand:
      updates["ac"] = updates["ac"] * scale_ac
      updates["dc"] = updates["dc"] * scale_dc
    params = optax.apply_updates(params, updates)
    if is_zero_or_rand:
      params["ac"] = mask_llf_jax(params["ac"], topo)
      params["dc"] = params["dc"].at[1].set(jnp.maximum(params["dc"][1], 0.0))
    params["noise_lut"] = jnp.clip(params["noise_lut"], min_noise_val, 0.5)

  def _prep_single_candidate(
      cand_p: dict[str, Any], deadzone: float = 0.0, rq_shift: float = 0.0
  ):
    c_st = copy.deepcopy(state)
    rq_cont = np.clip(
        np.asarray(rq + cand_p["rq_delta"] + rq_shift, dtype=np.float32),
        1.0,
        256.0,
    )
    c_st.raw_quant_field = broadcast_quant_field_to_ac_strategy(
        np.round(rq_cont).astype(np.int32), c_st.ac_strategy
    )
    c_st.ytox_map = np.clip(
        np.round(np.asarray(yx + cand_p["yx_delta"], dtype=np.float32)),
        -128,
        127,
    ).astype(np.int32)
    c_st.ytob_map = np.clip(
        np.round(np.asarray(yb + cand_p["yb_delta"], dtype=np.float32)),
        -128,
        127,
    ).astype(np.int32)
    _apply_custom_dequant_to_state(
        c_st, cand_p["dq_log_scale"], custom_dequant=dq_mode
    )
    ac_raw = np.asarray(cand_p["ac"], dtype=np.float32)
    dc_raw = np.asarray(cand_p["dc"], dtype=np.float32)
    ac_np, dc_np = _couple_candidate_ac_dc_np(
        ac_raw,
        dc_raw,
        c_st,
        state,
        cand_p["dq_log_scale"],
        dq_log_init,
        use_custom_dq,
        rq_cont=rq_cont,
    )
    if deadzone > 0.0:
      ac_np = np.sign(ac_np) * np.maximum(np.abs(ac_np) - deadzone, 0.0)
    c_st.quant_ac = np.round(ac_np).astype(np.int32)
    c_st.dc_quant = np.round(dc_np).astype(np.int32)
    c_lut = (
        np.clip(
            np.asarray(cand_p["noise_lut"], dtype=np.float32),
            2.0 / 1024.0,
            0.5,
        )
        if enable_noise
        else None
    )
    return (c_st, c_lut)

  def _score_single_candidate(
      prepped: tuple[JxlVarDCTState, Any],
      enc_res: tuple[bytes, dict[str, Any]],
  ):
    c_st, c_lut = prepped
    jb, st_dict = enc_res
    t_bpp = (len(jb) * 8.0) / float(h * w)
    r_hard = _fast_reconstruct_single_frame(
        c_st, include_noise=enable_noise, noise_lut=c_lut
    )
    t_wd = float(
        wd_only(
            r_hard,
            target_j,
            vgg_params,
            float(log2_sigma),
            num_levels=2,
            num_scales=3,
        )
    )
    t_j = t_wd + float(eff_lambda) * t_bpp
    return (t_j, t_bpp, t_wd, jb, st_dict, c_st, c_lut, r_hard)

  def _eval_single_candidate(
      cand_p: dict[str, Any], deadzone: float = 0.0, rq_shift: float = 0.0
  ):
    prepped = _prep_single_candidate(
        cand_p, deadzone=deadzone, rq_shift=rq_shift
    )
    enc_res = encode_single_frame(prepped[0], effort=effort, noise_lut=prepped[1])
    return _score_single_candidate(prepped, enc_res)

  if float(distance) < 0.65:
    single_preps = [
        _prep_single_candidate(best_params, deadzone=0.0),
        _prep_single_candidate(params, deadzone=0.0),
        _prep_single_candidate(step0_params, deadzone=0.0),
    ]
  else:
    single_preps = [
        _prep_single_candidate(best_params, deadzone=0.0),
        _prep_single_candidate(best_params, deadzone=0.04),
        _prep_single_candidate(best_params, deadzone=0.09),
        _prep_single_candidate(params, deadzone=0.0),
        _prep_single_candidate(step0_params, deadzone=0.0),
    ]
  single_enc_list = list(
      _CPP_ENCODE_POOL.map(
          lambda pr: encode_single_frame(pr[0], effort=effort, noise_lut=pr[1]),
          single_preps,
      )
  )
  cand_evals = [
      _score_single_candidate(pr, enc_r)
      for pr, enc_r in zip(single_preps, single_enc_list)
  ]
  best_eval = min(cand_evals, key=lambda x: x[0])
  if best_eval[0] >= stage2_true_j:
    for dz in (0.02, 0.06, 0.12, 0.18, 0.25, 0.35, 0.45):
      ev = _eval_single_candidate(best_params, deadzone=dz)
      if ev[0] < best_eval[0]:
        best_eval = ev
      ev0 = _eval_single_candidate(step0_params, deadzone=dz)
      if ev0[0] < best_eval[0]:
        best_eval = ev0
    if best_eval[0] >= stage2_true_j:
      for rqs in (-0.6, 0.6):
        ev_rq = _eval_single_candidate(best_params, deadzone=0.0, rq_shift=rqs)
        if ev_rq[0] < best_eval[0]:
          best_eval = ev_rq

  (
      true_rd,
      true_bpp,
      true_wd,
      jxl_bytes,
      stats,
      state,
      opt_noise_lut,
      rec_hard,
  ) = best_eval
  stats = dict(stats)
  stats["dequant_mode"] = dq_mode

  if len(history) >= 2 and history[-1]["rd_loss"] > history[0]["rd_loss"]:
    history[-1]["rd_loss"] = min(history[-1]["rd_loss"], best_j)

  return _finalize_encode_result(
      mode="single",
      upsampling=1,
      dim_shift=0,
      blend_mode=None,
      jxl_bytes=jxl_bytes,
      bpp=true_bpp,
      wd=true_wd,
      rd_cost=true_rd,
      recon_rgb=np.asarray(rec_hard),
      target_rgb_01=arr,
      enable_noise=enable_noise,
      opt_noise_lut=opt_noise_lut,
      patches=[],
      ref_state=None,
      main_state=state,
      state=state,
      alpha_01=None,
      alpha_u8=None,
      stats=stats,
      history=history,
      last_grads=last_grads,
      verify_decoders=verify_decoders,
      stage1_candidates=stage1_candidates,
      stage2_ac_strategy=stage2_acs,
      stage2_true_bpp=stage2_true_bpp,
      stage2_wd=stage2_wd,
      stage2_true_j=stage2_true_j,
      stage3_true_bpp=true_bpp,
      stage3_wd=true_wd,
      stage3_true_j=true_rd,
      custom_dequant_enabled=use_custom_dq,
  )


def select_best_mode(
    rgb_u8: np.ndarray | jax.Array | None = None,
    distance: float = 0.35,
    lambda_rd: float | None = None,
    rd_lambda: float | None = None,
    steps: int | None = None,
    num_steps: int | None = None,
    candidate_modes: Sequence[str] = ("single", "alpha"),
    candidate_upsamplings: Sequence[int] = (2, 4, 8),
    upsampling_candidates: Sequence[int] | None = None,
    candidate_dim_shifts: Sequence[int] = (0, 1),
    dim_shift_candidates: Sequence[int] | None = None,
    patch_blend_mode: int | str = "auto",
    blend_mode_candidates: Sequence[int | str] | None = None,
    alpha_blend_mode: int | str = PATCH_BLEND_ALPHA_BELOW,
    enable_noise: bool | str | None = "auto",
    custom_dequant: bool | str | None = None,
    enable_custom_dequant: bool | str | None = None,
    optimize_dequant_matrices: bool | str | None = None,
    use_custom_dequant: bool | str | None = None,
    verify_decoders: bool = True,
    vgg_params: Any | None = None,
    log2_sigma: float = 1.0,
    effort: int = 7,
    single_frame_fn: Callable[..., Any] | None = None,
    init_mode: str = "dct",
    seed: int = 0,
    target_bpp: float | None = None,
    *,
    srgb_hwc: np.ndarray | jax.Array | None = None,
) -> EncodeResult:
  """Evaluates candidate topologies by true RD Lagrangian `J = WD + lambda_rd * bpp_true`."""
  img_in = rgb_u8 if rgb_u8 is not None else srgb_hwc
  if img_in is None:
    raise ValueError("Either rgb_u8 or srgb_hwc must be provided.")
  arr = _normalize_srgb_hwc(img_in)
  init_mode_norm = _normalize_init_mode(init_mode)
  default_steps = 40 if init_mode_norm in ("zero", "random") else 6
  caller_overrode_lambda = (rd_lambda is not None) or (lambda_rd is not None)
  eff_lambda = float(
      rd_lambda
      if rd_lambda is not None
      else (
          lambda_rd
          if lambda_rd is not None
          else calibrate_rd_lambda(distance)
      )
  )
  eff_steps = int(
      num_steps
      if num_steps is not None
      else (steps if steps is not None else default_steps)
  )
  dq_spec = _resolve_custom_dequant_spec(
      custom_dequant=custom_dequant,
      enable_custom_dequant=enable_custom_dequant,
      optimize_dequant_matrices=optimize_dequant_matrices,
      use_custom_dequant=use_custom_dequant,
  )
  use_custom_dq_flag = (
      select_dequant_mode_for_frame(
          arr.shape[0], arr.shape[1], float(distance), dq_spec, is_reference=False
      )
      != "none"
  )
  u_cands = (
      list(upsampling_candidates)
      if upsampling_candidates is not None
      else list(candidate_upsamplings)
  )
  if (
      upsampling_candidates is None
      and tuple(u_cands) == (2, 4, 8)
      and float(distance) < 1.0
      and not caller_overrode_lambda
      and target_bpp is None
  ):
    u_cands = [2] if float(distance) < 0.45 else [2, 4]
  ds_cands = (
      list(dim_shift_candidates)
      if dim_shift_candidates is not None
      else list(candidate_dim_shifts)
  )
  if blend_mode_candidates is not None:
    bm_cands = [_parse_blend_mode(b) for b in blend_mode_candidates]
  else:
    parsed_pbm = _parse_blend_mode(patch_blend_mode)
    bm_cands = (
        ["auto"] if parsed_pbm == "auto" else [int(parsed_pbm)]
    )

  is_auto_noise = enable_noise is None or (
      isinstance(enable_noise, str) and enable_noise.strip().lower() == "auto"
  )
  target_xyb_sel = srgb_to_xyb(
      jnp.moveaxis(jnp.asarray(arr, dtype=jnp.float32), -1, 0)
  )
  eff_noise, _ = _resolve_enable_noise(
      enable_noise,
      target_xyb_sel,
      float(distance),
      arr.shape[0],
      arr.shape[1],
      default_auto=True,
  )

  if vgg_params is None:
    vgg_params = load_vgg16_params()
  candidates: list[CompositeOptimizationResult] = []

  for m in candidate_modes:
    m_low = m.lower()
    if m_low == "single":
      if single_frame_fn is not None:
        res_s = single_frame_fn(
            arr,
            distance=distance,
            lambda_rd=eff_lambda,
            num_steps=eff_steps,
            enable_noise=eff_noise,
            vgg_params=vgg_params,
            effort=effort,
        )
        if not isinstance(res_s, EncodeResult):
          j_bytes = getattr(res_s, "jxl_bytes", res_s["jxl_bytes"])
          bpp_s = float(getattr(res_s, "bpp", res_s["bpp"]))
          wd_s = float(getattr(res_s, "wd", res_s["wd"]))
          rd_s = wd_s + eff_lambda * bpp_s
          rec_s = np.asarray(
              getattr(
                  res_s,
                  "recon_jax_rgb",
                  getattr(res_s, "recon_rgb", arr),
              )
          )
          res_s = _finalize_encode_result(
              mode="single",
              upsampling=1,
              dim_shift=0,
              blend_mode=None,
              jxl_bytes=j_bytes,
              bpp=bpp_s,
              wd=wd_s,
              rd_cost=rd_s,
              recon_rgb=rec_s,
              target_rgb_01=arr,
              enable_noise=eff_noise,
              opt_noise_lut=getattr(res_s, "noise_lut", None),
              patches=[],
              ref_state=None,
              main_state=getattr(res_s, "state", None),
              state=getattr(res_s, "state", None),
              alpha_01=None,
              alpha_u8=None,
              stats={},
              history=getattr(res_s, "history", []),
              last_grads=None,
              verify_decoders=False,
              custom_dequant_enabled=use_custom_dq_flag,
          )
      elif is_auto_noise and eff_noise:
        res_s_clean = optimize_single_frame_baseline(
            rgb_u8=arr,
            distance=distance,
            lambda_rd=eff_lambda,
            steps=eff_steps,
            enable_noise=False,
            custom_dequant=dq_spec,
            verify_decoders=False,
            vgg_params=vgg_params,
            log2_sigma=log2_sigma,
            effort=effort,
            init_mode=init_mode_norm,
            seed=seed,
        )
        res_s_noise = optimize_single_frame_baseline(
            rgb_u8=arr,
            distance=distance,
            lambda_rd=eff_lambda,
            steps=eff_steps,
            enable_noise=True,
            custom_dequant=dq_spec,
            verify_decoders=False,
            vgg_params=vgg_params,
            log2_sigma=log2_sigma,
            effort=effort,
            init_mode=init_mode_norm,
            seed=seed,
        )
        res_s = (
            res_s_noise
            if res_s_noise.rd_cost <= res_s_clean.rd_cost
            else res_s_clean
        )
      else:
        res_s = optimize_single_frame_baseline(
            rgb_u8=arr,
            distance=distance,
            lambda_rd=eff_lambda,
            steps=eff_steps,
            enable_noise=eff_noise,
            custom_dequant=dq_spec,
            verify_decoders=False,
            vgg_params=vgg_params,
            log2_sigma=log2_sigma,
            effort=effort,
            init_mode=init_mode_norm,
            seed=seed,
        )
      candidates.append(res_s)
    elif m_low == "patches":
      u_arg: int | str = "auto" if len(u_cands) > 1 else int(u_cands[0])
      for bm in bm_cands:
        res_p = optimize_patch_composite(
            rgb_u8=arr,
            upsampling=u_arg,
            candidate_upsamplings=u_cands,
            blend_mode=bm,
            distance=distance,
            lambda_rd=eff_lambda,
            steps=eff_steps,
            enable_noise=eff_noise,
            custom_dequant=dq_spec,
            verify_decoders=False,
            vgg_params=vgg_params,
            log2_sigma=log2_sigma,
            effort=effort,
            init_mode=init_mode_norm,
            seed=seed,
        )
        candidates.append(res_p)
    elif m_low == "alpha":
      u_arg_a: int | str = "auto" if len(u_cands) > 1 else int(u_cands[0])
      ds_arg_a: int | str = "auto" if len(ds_cands) > 1 else int(ds_cands[0])
      res_a = optimize_alpha_composite(
          rgb_u8=arr,
          upsampling=u_arg_a,
          dim_shift=ds_arg_a,
          candidate_upsamplings=u_cands,
          candidate_dim_shifts=ds_cands,
          distance=distance,
          lambda_rd=eff_lambda,
          steps=eff_steps,
          enable_noise=eff_noise,
          alpha_blend_mode=alpha_blend_mode,
          custom_dequant=dq_spec,
          verify_decoders=False,
          vgg_params=vgg_params,
          log2_sigma=log2_sigma,
          effort=effort,
          init_mode=init_mode_norm,
          seed=seed,
      )
      candidates.append(res_a)
    else:
      raise ValueError(f"Unknown candidate mode: {m}")

  # High-quality (d < 1.0) fold-back guard: prevent downsampled reference frames
  # from trading sharp full-res detail for 2x/4x/8x upsampling blur unless
  # multi-res preserves single-frame distortion (or single exceeds target_bpp).
  eligible_candidates = candidates
  single_cands = [c for c in candidates if c.mode == "single"]
  if float(distance) < 1.0 and single_cands and not caller_overrode_lambda:
    s_cand = min(single_cands, key=lambda c: c.rd_cost)
    est_bpp = estimate_frame_bits(
        arr.shape[0], arr.shape[1], float(distance)
    ) / float(max(arr.shape[0] * arr.shape[1], 1))
    hq_bpp_cap = (
        float(target_bpp)
        if target_bpp is not None
        else float(est_bpp * 2.5 + 1.5)
    )
    if s_cand.bpp <= hq_bpp_cap:
      wd_tol = (
          1.0
          if float(distance) < 0.45
          else (1.15 if eff_noise else 1.03)
      )
      filtered = [
          c
          for c in candidates
          if c.mode == "single"
          or (
              c.wd < s_cand.wd
              if wd_tol <= 1.0
              else c.wd <= s_cand.wd * wd_tol
          )
      ]
      if filtered:
        eligible_candidates = filtered

  best = min(eligible_candidates, key=lambda c: c.rd_cost)
  if verify_decoders:
    ver = verify_jxl_codestream(best.jxl_bytes, min_cross_decoder_psnr=45.0)
    djxl_rgb = ver["rgb_djxl"].astype(np.float32) / 255.0
    best.psnr_vs_djxl = float(compute_psnr(best.recon_rgb, djxl_rgb, peak=1.0))
    best.psnr_jax_vs_djxl = best.psnr_vs_djxl
    best.psnr_djxl_vs_jxlrs = float(ver["cross_decoder_psnr"])
    best.cross_decoder_psnr = best.psnr_djxl_vs_jxlrs
    best.psnr_db = float(compute_psnr(djxl_rgb, arr, peak=1.0))
    best.header_info = dict(ver["inspect"])
    if "extra_channel_bits" not in best.header_info:
      best.header_info["extra_channel_bits"] = [
          int(ec.get("bits_per_sample", 8))
          for ec in best.header_info.get("extra_channels", [])
      ]

  best.all_candidates = [
      {
          "mode": c.mode,
          "upsampling": c.upsampling,
          "blend_mode": c.blend_mode,
          "dim_shift": c.dim_shift,
          "bpp": c.bpp,
          "wd": c.wd,
          "rd_cost": c.rd_cost,
          "psnr": c.psnr,
          "file_bytes": len(c.jxl_bytes),
      }
      for c in candidates
  ]
  _STAGE1_AC_CANDIDATE_CACHE.clear()
  _STAGE1_MAIN_DP_CACHE.clear()
  _STAGE1_MULTIRES_CACHE.clear()
  return best


select_best_composite_mode = select_best_mode


def encode_image_to_jxl(
    image_rgb_u8: np.ndarray | jax.Array | None = None,
    rd_lambda: float | None = None,
    distance: float = 1.0,
    mode: str = "auto",
    upsampling: int | str = "auto",
    dim_shift: int | str = 0,
    patch_blend_mode: int | str = "auto",
    alpha_blend_mode: int | str = PATCH_BLEND_ALPHA_BELOW,
    enable_noise: bool | str | None = None,
    custom_dequant: bool | str | None = None,
    enable_custom_dequant: bool | str | None = None,
    optimize_dequant_matrices: bool | str | None = None,
    use_custom_dequant: bool | str | None = None,
    num_steps: int | None = None,
    steps: int | None = None,
    verify_decoders: bool = True,
    init_mode: str = "dct",
    seed: int = 0,
    target_bpp: float | None = None,
    *,
    srgb_hwc: np.ndarray | jax.Array | None = None,
    blend_mode: int | str | None = None,
    lambda_rd: float | None = None,
    log2_sigma: float = 1.0,
    vgg_params: Any | None = None,
    effort: int = 7,
) -> EncodeResult:
  """Top-level JXL Rate-WD encoder supporting `'single'`, `'patches'`, `'alpha'`, and `'auto'`."""
  img_in = image_rgb_u8 if image_rgb_u8 is not None else srgb_hwc
  if img_in is None:
    raise ValueError("Either image_rgb_u8 or srgb_hwc must be provided.")
  arr = _normalize_srgb_hwc(img_in)
  init_mode_norm = _normalize_init_mode(init_mode)
  default_steps = 40 if init_mode_norm in ("zero", "random") else 8
  eff_steps = int(
      steps
      if steps is not None
      else (num_steps if num_steps is not None else default_steps)
  )
  eff_bm = blend_mode if blend_mode is not None else patch_blend_mode
  dq_spec = _resolve_custom_dequant_spec(
      custom_dequant=custom_dequant,
      enable_custom_dequant=enable_custom_dequant,
      optimize_dequant_matrices=optimize_dequant_matrices,
      use_custom_dequant=use_custom_dequant,
  )
  m_low = mode.strip().lower()

  if m_low == "auto":
    eff_enable_noise: bool | str = (
        "auto" if enable_noise is None else enable_noise
    )
    if isinstance(upsampling, str) and upsampling.lower() == "auto":
      if (
          float(distance) < 1.0
          and rd_lambda is None
          and lambda_rd is None
          and target_bpp is None
      ):
        u_cands = (2,) if float(distance) < 0.45 else (2, 4)
      else:
        u_cands = (2, 4, 8)
    else:
      u_cands = (int(upsampling),)
    ds_cands = (
        (0, 1, 2, 3)
        if (isinstance(dim_shift, str) and dim_shift.lower() == "auto")
        else (int(dim_shift),)
    )
    return select_best_mode(
        rgb_u8=arr,
        distance=distance,
        lambda_rd=lambda_rd,
        rd_lambda=rd_lambda,
        steps=eff_steps,
        candidate_modes=("single", "alpha"),
        candidate_upsamplings=u_cands,
        candidate_dim_shifts=ds_cands,
        patch_blend_mode=eff_bm,
        alpha_blend_mode=alpha_blend_mode,
        enable_noise=eff_enable_noise,
        custom_dequant=dq_spec,
        verify_decoders=verify_decoders,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        effort=effort,
        init_mode=init_mode_norm,
        seed=seed,
        target_bpp=target_bpp,
    )

  eff_enable_noise_explicit: bool | str = (
      False if enable_noise is None else enable_noise
  )
  eff_lambda = float(
      rd_lambda
      if rd_lambda is not None
      else (
          lambda_rd
          if lambda_rd is not None
          else calibrate_rd_lambda(distance)
      )
  )

  if m_low == "single":
    eff_dist = (
        max(0.25, min(6.0, 2.2 * (0.12 / max(eff_lambda, 1e-4)) ** 0.45))
        if (rd_lambda is not None and distance == 1.0)
        else float(distance)
    )
    return optimize_single_frame_baseline(
        rgb_u8=arr,
        distance=eff_dist,
        lambda_rd=eff_lambda,
        steps=eff_steps,
        enable_noise=eff_enable_noise_explicit,
        custom_dequant=dq_spec,
        verify_decoders=verify_decoders,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        effort=effort,
        init_mode=init_mode_norm,
        seed=seed,
    )

  if m_low == "patches":
    return optimize_patch_composite(
        rgb_u8=arr,
        upsampling=upsampling,
        blend_mode=eff_bm,
        distance=distance,
        lambda_rd=eff_lambda,
        steps=eff_steps,
        enable_noise=eff_enable_noise_explicit,
        custom_dequant=dq_spec,
        verify_decoders=verify_decoders,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        effort=effort,
        init_mode=init_mode_norm,
        seed=seed,
    )

  if m_low == "alpha":
    eff_alpha_bm = (
        blend_mode if blend_mode is not None else alpha_blend_mode
    )
    return optimize_alpha_composite(
        rgb_u8=arr,
        upsampling=upsampling,
        dim_shift=dim_shift,
        distance=distance,
        lambda_rd=eff_lambda,
        steps=eff_steps,
        enable_noise=eff_enable_noise_explicit,
        alpha_blend_mode=eff_alpha_bm,
        custom_dequant=dq_spec,
        verify_decoders=verify_decoders,
        vgg_params=vgg_params,
        log2_sigma=log2_sigma,
        effort=effort,
        init_mode=init_mode_norm,
        seed=seed,
    )

  raise ValueError(
      f"mode must be in {{'single', 'patches', 'alpha', 'auto'}}, got {mode!r}"
  )


encode_composited_jxl = encode_image_to_jxl


__all__ = [
    "EncodeResult",
    "CompositeOptimizationResult",
    "CompositeOptimizeResult",
    "DEQUANT_MODE_DOF",
    "calibrate_rd_lambda",
    "estimate_frame_bits",
    "select_dequant_mode_for_frame",
    "downsample_rgb_jax",
    "compute_detail_mask_jax",
    "extract_patch_rectangles",
    "compute_continuous_dc_ac_jax",
    "init_state_from_xyb_jax",
    "estimate_noise_lut_jax",
    "stage1_explore_ac_candidates_compositing",
    "stage2_select_non_overlapping_ac_strategy_compositing",
    "optimize_patch_composite",
    "optimize_alpha_composite",
    "optimize_single_frame_baseline",
    "select_best_mode",
    "select_best_composite_mode",
    "encode_image_to_jxl",
    "encode_composited_jxl",
]

