"""Binary state serialization and subprocess bridge to jxl_wasserstein_helper, djxl, jxlinfo, and jxl_cli."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
HELPER_BIN = REPO_ROOT / "build" / "tools" / "jxl_wasserstein_helper"
CJXL_BIN = REPO_ROOT / "build" / "tools" / "cjxl"
DJXL_BIN = REPO_ROOT / "build" / "tools" / "djxl"
JXLINFO_BIN = REPO_ROOT / "build" / "tools" / "jxlinfo"
_JXL_CLI_DEFAULT = Path("/usr/local/google/home/veluca/jxl-rs/target/release/jxl_cli")
JXL_CLI_BIN = (
    _JXL_CLI_DEFAULT
    if _JXL_CLI_DEFAULT.exists()
    else (Path.home() / "jxl-rs" / "target" / "release" / "jxl_cli")
)

STATE_MAGIC = b"JXLWST01"
PATCHES_MAGIC = b"JXLWPT01"
ALPHA_MAGIC = b"JXLWAL01"

_STATE_HEADER_FMT = "<8s7I5i2f3i4f3f8fI6f2I6Q15Q"
_STATE_HEADER_SIZE = struct.calcsize(_STATE_HEADER_FMT)
assert _STATE_HEADER_SIZE == 340

_PATCHES_HEADER_FMT = "<8s2I"
_PATCHES_HEADER_SIZE = struct.calcsize(_PATCHES_HEADER_FMT)
assert _PATCHES_HEADER_SIZE == 16

_PATCH_ENTRY_FMT = "<8I"
_PATCH_ENTRY_SIZE = struct.calcsize(_PATCH_ENTRY_FMT)
assert _PATCH_ENTRY_SIZE == 32

_ALPHA_HEADER_FMT = "<8s4I"
_ALPHA_HEADER_SIZE = struct.calcsize(_ALPHA_HEADER_FMT)
assert _ALPHA_HEADER_SIZE == 24

PATCH_BLEND_ADD = 2
PATCH_BLEND_ALPHA_ABOVE = 4
PATCH_BLEND_ALPHA_BELOW = 5
PATCH_BLEND_MUL_ADD_ABOVE = 6
PATCH_BLEND_MUL_ADD_BELOW = 7

VALID_ALPHA_BLEND_MODES: tuple[int, ...] = (
    PATCH_BLEND_ALPHA_ABOVE,
    PATCH_BLEND_ALPHA_BELOW,
    PATCH_BLEND_MUL_ADD_ABOVE,
    PATCH_BLEND_MUL_ADD_BELOW,
)
VALID_ALPHA_BLEND_MODE_NAMES: dict[str, int] = {
    "kblendabove": PATCH_BLEND_ALPHA_ABOVE,
    "blend_above": PATCH_BLEND_ALPHA_ABOVE,
    "alpha_above": PATCH_BLEND_ALPHA_ABOVE,
    "4": PATCH_BLEND_ALPHA_ABOVE,
    "kblendbelow": PATCH_BLEND_ALPHA_BELOW,
    "blend_below": PATCH_BLEND_ALPHA_BELOW,
    "alpha_below": PATCH_BLEND_ALPHA_BELOW,
    "5": PATCH_BLEND_ALPHA_BELOW,
    "kalphaweightedaddabove": PATCH_BLEND_MUL_ADD_ABOVE,
    "alpha_weighted_add_above": PATCH_BLEND_MUL_ADD_ABOVE,
    "mul_add_above": PATCH_BLEND_MUL_ADD_ABOVE,
    "6": PATCH_BLEND_MUL_ADD_ABOVE,
    "kalphaweightedaddbelow": PATCH_BLEND_MUL_ADD_BELOW,
    "alpha_weighted_add_below": PATCH_BLEND_MUL_ADD_BELOW,
    "mul_add_below": PATCH_BLEND_MUL_ADD_BELOW,
    "alphamuladd": PATCH_BLEND_MUL_ADD_BELOW,
    "7": PATCH_BLEND_MUL_ADD_BELOW,
}

DEFAULT_DC_QUANT_WEIGHTS: tuple[float, float, float] = (
    1.0 / 4096.0,
    1.0 / 512.0,
    1.0 / 256.0,
)

_DCT8_LIB_BANDS: tuple[tuple[float, ...], ...] = (
    (3150.0, 0.0, -0.4, -0.4, -0.4, -2.0),
    (560.0, 0.0, -0.3, -0.3, -0.3, -0.3),
    (512.0, -2.0, -1.0, 0.0, -1.0, -2.0),
)

LAYER_NAMES: tuple[str, ...] = (
    "Header",
    "Toc",
    "Dictionary",
    "Splines",
    "Noise",
    "Quant",
    "ModularTree",
    "ModularGlobal",
    "Dc",
    "ModularDcGroup",
    "ControlFields",
    "Order",
    "Ac",
    "AcTokens",
    "ModularAcGroup",
)

ACS_COVERED_BLOCKS_X: tuple[int, ...] = (
    1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2, 4, 1, 1, 1, 1, 1, 1,
    8, 4, 8, 16, 8, 16, 32, 16, 32,
)
ACS_COVERED_BLOCKS_Y: tuple[int, ...] = (
    1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4, 2, 1, 1, 1, 1, 1, 1,
    8, 8, 4, 16, 16, 8, 32, 32, 16,
)


def _quant_mult(v: float) -> float:
  return 1.0 + v if v > 0.0 else 1.0 / (1.0 - v)


def _inv_quant_mult(ratio: float) -> float:
  r = max(1e-6, float(ratio))
  return (r - 1.0) if r >= 1.0 else (1.0 - 1.0 / r)


def _f16_roundtrip_float(val: float | np.ndarray) -> np.ndarray:
  arr32 = np.asarray(val, dtype=np.float32)
  bits32 = arr32.view(np.uint32)
  biased_exp32 = (bits32 >> np.uint32(23)) & np.uint32(0xFF)
  trunc_bits = np.where(
      biased_exp32 < np.uint32(113),
      bits32 & np.uint32(0x80000000),
      bits32 & np.uint32(0xFFFFE000),
  ).astype(np.uint32)
  return trunc_bits.view(np.float32)


def _eval_dct_quant_weights(
    wrows: int,
    wcols: int,
    distance_bands: Sequence[Sequence[float]],
) -> np.ndarray:
  out = np.zeros((3, wrows, wcols), dtype=np.float32)
  num_bands = len(distance_bands[0])
  scale = float(num_bands - 1) / (math.sqrt(2.0) + 1e-6)
  rcpcol = scale / float(wcols - 1)
  rcprow = scale / float(wrows - 1)
  yy, xx = np.meshgrid(
      np.arange(wrows, dtype=np.float32) * rcprow,
      np.arange(wcols, dtype=np.float32) * rcpcol,
      indexing="ij",
  )
  dist = np.sqrt(xx * xx + yy * yy)
  idx = np.clip(np.floor(dist).astype(np.int32), 0, max(0, num_bands - 2))
  frac = dist - idx.astype(np.float32)
  for c in range(3):
    bands = np.zeros(num_bands, dtype=np.float32)
    bands[0] = float(distance_bands[c][0])
    for k in range(1, num_bands):
      bands[k] = bands[k - 1] * _quant_mult(float(distance_bands[c][k]))
    if num_bands == 1:
      w = np.full((wrows, wcols), bands[0], dtype=np.float32)
    else:
      a = bands[idx]
      b = bands[idx + 1]
      w = a * np.power(np.maximum(1e-12, b / np.maximum(1e-12, a)), frac)
    out[c] = 1.0 / np.maximum(1e-12, w)
  return out


_DEFAULT_DCT8_DEQUANT: np.ndarray = _eval_dct_quant_weights(
    8, 8, _DCT8_LIB_BANDS
)


@dataclass(frozen=True)
class PatchEntry:
  """Single full-frame XYB patch add/alpha-blend instruction from a ReferenceOnly frame."""

  ref: int
  x0: int
  y0: int
  xsize: int
  ysize: int
  x: int
  y: int
  blend_mode: int = PATCH_BLEND_ADD


@dataclass
class JxlVarDCTState:
  """Complete VarDCT frame state exchanged with jxl_wasserstein_helper."""

  height: int
  width: int
  h8: int
  w8: int
  ht: int
  wt: int
  global_scale: int
  quant_dc: int
  x_qm_scale: int
  b_qm_scale: int
  color_factor: int
  base_correlation_x: float
  base_correlation_b: float
  ytox_dc: int
  ytob_dc: int
  epf_iters: int
  epf_quant_mul: float
  epf_pass0_sigma_scale: float
  epf_pass2_sigma_scale: float
  epf_border_sad_mul: float
  epf_channel_scale: tuple[float, float, float]
  epf_sharp_lut: tuple[float, ...]
  gab: bool
  gab_weights: tuple[float, ...]
  used_acs: int
  ac_strategy: np.ndarray
  raw_quant_field: np.ndarray
  epf_sharpness: np.ndarray
  ytox_map: np.ndarray
  ytob_map: np.ndarray
  dc_quant: np.ndarray
  quant_ac: np.ndarray
  dequant_matrices: dict[int, np.ndarray] = field(default_factory=dict)
  dc_quant_weights: np.ndarray = field(
      default_factory=lambda: np.array(
          DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32
      )
  )
  bit_stats: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    dc_arr = np.asarray(self.dc_quant)
    if dc_arr.ndim == 3:
      object.__setattr__(self, "_dc_quant_grid", dc_arr.astype(np.int32).copy())
    else:
      object.__setattr__(
          self,
          "_dc_quant_grid",
          np.zeros((3, self.h8, self.w8), dtype=np.int32),
      )
    object.__setattr__(
        self,
        "_orig_dequant_matrices",
        {int(k): np.asarray(v, dtype=np.float32).copy() for k, v in self.dequant_matrices.items()},
    )
    object.__setattr__(
        self,
        "dc_quant_weights",
        np.asarray(self.dc_quant_weights, dtype=np.float32).reshape(3).copy(),
    )
    acs_arr = np.asarray(self.ac_strategy, dtype=np.int32)
    if np.all(acs_arr == 0):
      anc = np.ones((self.h8, self.w8), dtype=bool)
    else:
      anc = np.zeros((self.h8, self.w8), dtype=bool)
      vis = np.zeros((self.h8, self.w8), dtype=bool)
      for by in range(self.h8):
        for bx in range(self.w8):
          if vis[by, bx]:
            continue
          s_id = int(acs_arr[by, bx])
          cy = ACS_COVERED_BLOCKS_Y[s_id] if 0 <= s_id < len(ACS_COVERED_BLOCKS_Y) else 1
          cx = ACS_COVERED_BLOCKS_X[s_id] if 0 <= s_id < len(ACS_COVERED_BLOCKS_X) else 1
          anc[by, bx] = True
          vis[by : by + cy, bx : bx + cx] = True
    object.__setattr__(self, "ac_strategy_Anchor", anc)

  def copy(self) -> JxlVarDCTState:
    copied = JxlVarDCTState(
        height=self.height,
        width=self.width,
        h8=self.h8,
        w8=self.w8,
        ht=self.ht,
        wt=self.wt,
        global_scale=self.global_scale,
        quant_dc=self.quant_dc,
        x_qm_scale=self.x_qm_scale,
        b_qm_scale=self.b_qm_scale,
        color_factor=self.color_factor,
        base_correlation_x=self.base_correlation_x,
        base_correlation_b=self.base_correlation_b,
        ytox_dc=self.ytox_dc,
        ytob_dc=self.ytob_dc,
        epf_iters=self.epf_iters,
        epf_quant_mul=self.epf_quant_mul,
        epf_pass0_sigma_scale=self.epf_pass0_sigma_scale,
        epf_pass2_sigma_scale=self.epf_pass2_sigma_scale,
        epf_border_sad_mul=self.epf_border_sad_mul,
        epf_channel_scale=tuple(self.epf_channel_scale),  # type: ignore[arg-type]
        epf_sharp_lut=tuple(self.epf_sharp_lut),
        gab=bool(self.gab),
        gab_weights=tuple(self.gab_weights),
        used_acs=self.used_acs,
        ac_strategy=self.ac_strategy.copy(),
        raw_quant_field=self.raw_quant_field.copy(),
        epf_sharpness=self.epf_sharpness.copy(),
        ytox_map=self.ytox_map.copy(),
        ytob_map=self.ytob_map.copy(),
        dc_quant=np.asarray(self.dc_quant).copy(),
        quant_ac=self.quant_ac.copy(),
        dequant_matrices={k: v.copy() for k, v in self.dequant_matrices.items()},
        dc_quant_weights=np.asarray(
            self.dc_quant_weights, dtype=np.float32
        ).copy(),
        bit_stats=dict(self.bit_stats),
    )
    orig_deq = getattr(self, "_orig_dequant_matrices", None)
    if orig_deq is not None:
      object.__setattr__(
          copied,
          "_orig_dequant_matrices",
          {int(k): np.asarray(v, dtype=np.float32).copy() for k, v in orig_deq.items()},
      )
    dc_grid = getattr(self, "_dc_quant_grid", None)
    if dc_grid is not None and np.asarray(self.dc_quant).ndim != 3:
      object.__setattr__(copied, "_dc_quant_grid", np.asarray(dc_grid).copy())
    anc = getattr(self, "ac_strategy_Anchor", None)
    if anc is not None:
      object.__setattr__(
          copied, "ac_strategy_Anchor", np.asarray(anc, dtype=bool).copy()
      )
    return copied


JxlVarDctState = JxlVarDCTState


def _get_dc_quant_grid(state: JxlVarDCTState) -> np.ndarray:
  dc_arr = np.asarray(state.dc_quant)
  if dc_arr.ndim == 3 and dc_arr.shape == (3, state.h8, state.w8):
    return dc_arr
  grid = getattr(state, "_dc_quant_grid", None)
  if grid is not None:
    return np.asarray(grid, dtype=np.int32)
  return np.zeros((3, state.h8, state.w8), dtype=np.int32)


def _get_dc_quant_weights(state: JxlVarDCTState) -> np.ndarray:
  dc_arr = np.asarray(state.dc_quant)
  if dc_arr.ndim == 1 and dc_arr.size == 3:
    return np.asarray(dc_arr, dtype=np.float32).reshape(3)
  custom_dc = getattr(state, "custom_dc_quant", None)
  if custom_dc is not None:
    return np.asarray(custom_dc, dtype=np.float32).reshape(3)
  if -1 in state.dequant_matrices:
    return np.asarray(state.dequant_matrices[-1], dtype=np.float32).reshape(3)
  return np.asarray(state.dc_quant_weights, dtype=np.float32).reshape(3)


def _has_custom_dc_quant(state: JxlVarDCTState) -> bool:
  dc_w = _get_dc_quant_weights(state)
  def_w = np.asarray(DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32)
  return bool(np.max(np.abs(dc_w - def_w)) > 1e-7)


def _get_reference_dequant_matrix(
    state: JxlVarDCTState, s: int
) -> np.ndarray | None:
  orig_map = getattr(state, "_orig_dequant_matrices", None)
  if orig_map and s in orig_map:
    return np.asarray(orig_map[s], dtype=np.float32)
  if s == 0:
    return _DEFAULT_DCT8_DEQUANT
  return None


def _has_custom_ac_dequant(state: JxlVarDCTState) -> bool:
  if not state.dequant_matrices:
    return False
  for s, mat in state.dequant_matrices.items():
    if s < 0:
      continue
    arr = np.asarray(mat, dtype=np.float32)
    ref = _get_reference_dequant_matrix(state, int(s))
    if ref is None or ref.shape != arr.shape:
      return True
    cx = ACS_COVERED_BLOCKS_X[s]
    cy = ACS_COVERED_BLOCKS_Y[s]
    ys = min(cy, cx)
    xs = max(cy, cx)
    mask = np.ones(arr.shape[1:], dtype=bool)
    mask[:ys, :xs] = False
    denom = np.maximum(1e-12, np.abs(ref[:, mask]))
    rel_diff = np.max(np.abs(arr[:, mask] - ref[:, mask]) / denom)
    if float(rel_diff) > 1e-5:
      return True
  return False


def has_custom_dequant_matrices(state: JxlVarDCTState) -> bool:
  """Returns True if state has non-default AC dequant_matrices or DC quant weights."""
  return _has_custom_ac_dequant(state) or _has_custom_dc_quant(state)


def _fit_dct_quant_params_np(
    wrows: int, wcols: int, xs: int, ys: int, custom_mat: np.ndarray
) -> list[list[float]]:
  num_bands = 8
  scale = float(num_bands - 1) / (math.sqrt(2.0) + 1e-6)
  rcpcol = scale / float(wcols - 1)
  rcprow = scale / float(wrows - 1)
  out_bands: list[list[float]] = []
  for c in range(3):
    ata = np.zeros((num_bands, num_bands), dtype=np.float64)
    atb = np.zeros(num_bands, dtype=np.float64)
    log_ws: list[float] = []
    for y in range(wrows):
      for x in range(wcols):
        if y < ys and x < xs:
          continue
        m_val = max(1e-12, float(custom_mat[c, y, x]))
        log_w = -math.log(m_val)
        log_ws.append(log_w)
        dist = math.sqrt((x * rcpcol) ** 2 + (y * rcprow) ** 2)
        idx = min(max(0, int(dist)), num_bands - 2)
        frac = dist - float(idx)
        w0 = 1.0 - frac
        w1 = frac
        ata[idx, idx] += w0 * w0
        ata[idx, idx + 1] += w0 * w1
        ata[idx + 1, idx] += w1 * w0
        ata[idx + 1, idx + 1] += w1 * w1
        atb[idx] += w0 * log_w
        atb[idx + 1] += w1 * log_w
    mean_log_w = float(np.mean(log_ws)) if log_ws else 6.0
    for k in range(num_bands):
      ata[k, k] += 1e-6
      atb[k] += 1e-6 * mean_log_w
      if k + 1 < num_bands:
        ata[k, k] += 1e-4
        ata[k + 1, k + 1] += 1e-4
        ata[k, k + 1] -= 1e-4
        ata[k + 1, k] -= 1e-4
    z = np.linalg.solve(ata, atb)
    bands = np.exp(np.clip(z, -10.0, 20.0))
    db = [float(max(1e-4, bands[0]))]
    for k in range(1, num_bands):
      db.append(_inv_quant_mult(float(bands[k] / max(1e-6, bands[k - 1]))))
    out_bands.append(db)
  return out_bands


def roundtrip_custom_dequant_matrices(
    state: JxlVarDCTState,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
  """Roundtrips custom AC dequant_matrices and DC quant weights through JXL F16 / QuantEncoding."""
  dc_w = _get_dc_quant_weights(state)
  if _has_custom_dc_quant(state):
    rt_dc = _f16_roundtrip_float(dc_w * np.float32(128.0)) * np.float32(
        1.0 / 128.0
    )
  else:
    rt_dc = np.asarray(DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32).copy()

  rt_dequant: dict[int, np.ndarray] = {}
  for s, mat in state.dequant_matrices.items():
    if s < 0:
      continue
    arr = np.asarray(mat, dtype=np.float32).copy()
    ref = _get_reference_dequant_matrix(state, int(s))
    cx = ACS_COVERED_BLOCKS_X[s]
    cy = ACS_COVERED_BLOCKS_Y[s]
    ys = min(cy, cx)
    xs = max(cy, cx)
    wrows = 8 * ys
    wcols = 8 * xs
    mask = np.ones((wrows, wcols), dtype=bool)
    mask[:ys, :xs] = False

    if ref is not None and ref.shape == arr.shape:
      denom = np.maximum(1e-12, np.abs(ref[:, mask]))
      max_diff = float(np.max(np.abs(arr[:, mask] - ref[:, mask]) / denom))
      if max_diff <= 1e-5:
        rt_dequant[s] = ref.copy()
        continue

      alphas = np.array(
          [
              float(np.mean(arr[c, mask] / np.maximum(1e-12, ref[c, mask])))
              for c in range(3)
          ],
          dtype=np.float32,
      )
      pred_scaled = ref * alphas[:, None, None]
      scaled_err = float(
          np.max(
              np.abs(arr[:, mask] - pred_scaled[:, mask])
              / np.maximum(1e-12, np.abs(arr[:, mask]))
          )
      )
      if scaled_err <= 2e-3:
        out_mat = ref.copy()
        for c in range(3):
          # Base weight w0 is divided by alpha[c] and roundtripped through F16(w0 / 64) * 64
          w0_lib = float(1.0 / max(1e-12, float(ref[c, 0, xs if xs < wcols else 1])))
          w0_scaled = w0_lib / max(1e-4, float(alphas[c]))
          w0_rt = float(_f16_roundtrip_float(w0_scaled / 64.0) * 64.0)
          eff_alpha = w0_lib / max(1e-12, w0_rt)
          out_mat[c] = ref[c] * np.float32(eff_alpha)
        rt_dequant[s] = out_mat
        continue

    db = _fit_dct_quant_params_np(wrows, wcols, xs, ys, arr)
    for c in range(3):
      db[c][0] = float(_f16_roundtrip_float(db[c][0] / 64.0) * 64.0)
      for k in range(1, len(db[c])):
        db[c][k] = float(_f16_roundtrip_float(db[c][k]))
    pred_dct = _eval_dct_quant_weights(wrows, wcols, db)
    dct_err = float(
        np.max(
            np.abs(arr[:, mask] - pred_dct[:, mask])
            / np.maximum(1e-12, np.abs(arr[:, mask]))
        )
    )
    if dct_err <= 0.05:
      if ref is not None and ref.shape == arr.shape:
        pred_dct[:, :ys, :xs] = ref[:, :ys, :xs]
      rt_dequant[s] = pred_dct
    else:
      qtable_den = np.float32(1.0 / 16384.0)
      qtable = np.clip(
          np.rint(np.maximum(1e-7, arr) / qtable_den), 1.0, 32767.0
      )
      rt_dequant[s] = (qtable * qtable_den).astype(np.float32)

  return rt_dequant, rt_dc


def load_state_bin(path: str | Path) -> JxlVarDCTState:
  """Loads a <state.bin> file written by jxl_wasserstein_helper or save_state_bin."""
  blob = Path(path).read_bytes()
  if len(blob) < _STATE_HEADER_SIZE:
    raise ValueError(f"State file too short ({len(blob)} bytes): {path}")

  unpacked = struct.unpack_from(_STATE_HEADER_FMT, blob, 0)
  magic = unpacked[0]
  if magic != STATE_MAGIC:
    raise ValueError(f"Invalid state magic {magic!r}, expected {STATE_MAGIC!r}")

  version = unpacked[1]
  if version != 1:
    raise ValueError(f"Unsupported state version {version}")

  height, width, h8, w8, ht, wt = unpacked[2:8]
  global_scale, quant_dc, x_qm_scale, b_qm_scale, color_factor = unpacked[8:13]
  base_correlation_x, base_correlation_b = unpacked[13:15]
  ytox_dc, ytob_dc, epf_iters = unpacked[15:18]
  (
      epf_quant_mul,
      epf_pass0_sigma_scale,
      epf_pass2_sigma_scale,
      epf_border_sad_mul,
  ) = unpacked[18:22]
  epf_channel_scale = tuple(float(x) for x in unpacked[22:25])
  epf_sharp_lut = tuple(float(x) for x in unpacked[25:33])
  gab = bool(unpacked[33])
  gab_weights = tuple(float(x) for x in unpacked[34:40])
  used_acs = int(unpacked[40])
  has_dequant_tables = int(unpacked[41])
  total_bits, header_bits, dc_bits, ac_meta_bits, ac_token_bits, file_bytes = (
      int(x) for x in unpacked[42:48]
  )
  layer_bits_raw = unpacked[48:63]
  bit_stats = {
      "total_bits": total_bits,
      "header_bits": header_bits,
      "dc_bits": dc_bits,
      "ac_meta_bits": ac_meta_bits,
      "ac_token_bits": ac_token_bits,
      "file_bytes": file_bytes,
      "layer_bits": {
          name: int(val) for name, val in zip(LAYER_NAMES, layer_bits_raw)
      },
  }

  offset = _STATE_HEADER_SIZE

  def _read_arr(dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    nonlocal offset
    count = math.prod(shape)
    arr = (
        np.frombuffer(blob, dtype=dtype, count=count, offset=offset)
        .reshape(shape)
        .copy()
    )
    offset += count * np.dtype(dtype).itemsize
    return arr

  ac_strategy = _read_arr("<u1", (h8, w8))
  raw_quant_field = _read_arr("<i4", (h8, w8))
  epf_sharpness = _read_arr("<u1", (h8, w8))
  ytox_map = _read_arr("<i1", (ht, wt))
  ytob_map = _read_arr("<i1", (ht, wt))
  dc_quant = _read_arr("<i4", (3, h8, w8))
  quant_ac = _read_arr("<i4", (3, h8 * 8, w8 * 8))

  dequant_matrices: dict[int, np.ndarray] = {}
  if (has_dequant_tables & 1) != 0:
    for s in range(len(ACS_COVERED_BLOCKS_X)):
      if (used_acs >> s) & 1:
        cx = ACS_COVERED_BLOCKS_X[s]
        cy = ACS_COVERED_BLOCKS_Y[s]
        y_layout = 8 * min(cy, cx)
        x_layout = 8 * max(cy, cx)
        if offset + 3 * y_layout * x_layout * 4 <= len(blob):
          dequant_matrices[s] = _read_arr("<f4", (3, y_layout, x_layout))

  dc_quant_weights = np.array(DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32)
  if (has_dequant_tables & 2) != 0 and offset + 12 <= len(blob):
    dc_quant_weights = _read_arr("<f4", (3,))

  return JxlVarDCTState(
      height=height,
      width=width,
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
      epf_quant_mul=epf_quant_mul,
      epf_pass0_sigma_scale=epf_pass0_sigma_scale,
      epf_pass2_sigma_scale=epf_pass2_sigma_scale,
      epf_border_sad_mul=epf_border_sad_mul,
      epf_channel_scale=epf_channel_scale,  # type: ignore[arg-type]
      epf_sharp_lut=epf_sharp_lut,
      gab=gab,
      gab_weights=gab_weights,
      used_acs=used_acs,
      ac_strategy=ac_strategy,
      raw_quant_field=raw_quant_field,
      epf_sharpness=epf_sharpness,
      ytox_map=ytox_map,
      ytob_map=ytob_map,
      dc_quant=dc_quant,
      quant_ac=quant_ac,
      dequant_matrices=dequant_matrices,
      dc_quant_weights=dc_quant_weights,
      bit_stats=bit_stats,
  )


def save_state_bin(
    state: JxlVarDCTState,
    path: str | Path,
    include_dequant: bool | None = None,
) -> Path:
  """Writes a JxlVarDCTState to a binary <state.bin> file."""
  out_path = Path(path)
  ac_dequant = {k: v for k, v in state.dequant_matrices.items() if int(k) >= 0}
  has_custom_dc = _has_custom_dc_quant(state)

  if include_dequant is None:
    write_ac = bool(ac_dequant)
    write_dc = has_custom_dc
  elif include_dequant:
    write_ac = bool(ac_dequant)
    write_dc = has_custom_dc
  else:
    write_ac = False
    write_dc = False

  has_dequant_flags = (1 if write_ac else 0) | (2 if write_dc else 0)
  used_acs = state.used_acs
  if write_ac:
    used_acs = 0
    for s in ac_dequant:
      used_acs |= 1 << int(s)

  layer_dict = state.bit_stats.get("layer_bits", {})
  layer_vals = [int(layer_dict.get(name, 0)) for name in LAYER_NAMES]

  header_bytes = struct.pack(
      _STATE_HEADER_FMT,
      STATE_MAGIC,
      1,
      int(state.height),
      int(state.width),
      int(state.h8),
      int(state.w8),
      int(state.ht),
      int(state.wt),
      int(state.global_scale),
      int(state.quant_dc),
      int(state.x_qm_scale),
      int(state.b_qm_scale),
      int(state.color_factor),
      float(state.base_correlation_x),
      float(state.base_correlation_b),
      int(state.ytox_dc),
      int(state.ytob_dc),
      int(state.epf_iters),
      float(state.epf_quant_mul),
      float(state.epf_pass0_sigma_scale),
      float(state.epf_pass2_sigma_scale),
      float(state.epf_border_sad_mul),
      *(float(x) for x in state.epf_channel_scale),
      *(float(x) for x in state.epf_sharp_lut),
      1 if state.gab else 0,
      *(float(x) for x in state.gab_weights),
      int(used_acs),
      int(has_dequant_flags),
      int(state.bit_stats.get("total_bits", 0)),
      int(state.bit_stats.get("header_bits", 0)),
      int(state.bit_stats.get("dc_bits", 0)),
      int(state.bit_stats.get("ac_meta_bits", 0)),
      int(state.bit_stats.get("ac_token_bits", 0)),
      int(state.bit_stats.get("file_bytes", 0)),
      *layer_vals,
  )

  dc_grid = _get_dc_quant_grid(state)
  chunks: list[bytes] = [
      header_bytes,
      np.ascontiguousarray(state.ac_strategy, dtype="<u1").tobytes(),
      np.ascontiguousarray(state.raw_quant_field, dtype="<i4").tobytes(),
      np.ascontiguousarray(state.epf_sharpness, dtype="<u1").tobytes(),
      np.ascontiguousarray(state.ytox_map, dtype="<i1").tobytes(),
      np.ascontiguousarray(state.ytob_map, dtype="<i1").tobytes(),
      np.ascontiguousarray(dc_grid, dtype="<i4").tobytes(),
      np.ascontiguousarray(state.quant_ac, dtype="<i4").tobytes(),
  ]
  if write_ac:
    for s in range(len(ACS_COVERED_BLOCKS_X)):
      if (used_acs >> s) & 1:
        chunks.append(
            np.ascontiguousarray(ac_dequant[s], dtype="<f4").tobytes()
        )
  if write_dc:
    dc_w = _get_dc_quant_weights(state)
    chunks.append(np.ascontiguousarray(dc_w, dtype="<f4").tobytes())

  with open(out_path, "wb") as f:
    for chunk in chunks:
      f.write(chunk)
  return out_path


def save_patches_bin(patches: Sequence[PatchEntry], path: str | Path) -> Path:
  """Writes a sequence of PatchEntry objects to <patches.bin>."""
  out_path = Path(path)
  chunks = [struct.pack(_PATCHES_HEADER_FMT, PATCHES_MAGIC, 1, len(patches))]
  for p in patches:
    chunks.append(
        struct.pack(
            _PATCH_ENTRY_FMT,
            int(p.ref),
            int(p.x0),
            int(p.y0),
            int(p.xsize),
            int(p.ysize),
            int(p.x),
            int(p.y),
            int(p.blend_mode),
        )
    )
  with open(out_path, "wb") as f:
    for chunk in chunks:
      f.write(chunk)
  return out_path


def save_alpha_bin(
    alpha_u8: np.ndarray, dim_shift: int, path: str | Path
) -> Path:
  """Writes an 8-bit alpha mask (H_alpha, W_alpha) to <alpha.bin>."""
  arr = np.ascontiguousarray(alpha_u8, dtype=np.uint8)
  if arr.ndim != 2:
    raise ValueError(f"Expected 2D alpha_u8 array, got shape {arr.shape}")
  h_a, w_a = arr.shape
  header = struct.pack(
      _ALPHA_HEADER_FMT, ALPHA_MAGIC, 1, int(h_a), int(w_a), int(dim_shift)
  )
  out_path = Path(path)
  out_path.write_bytes(header + arr.tobytes())
  return out_path


def _run_cmd(cmd: Sequence[str | Path]) -> subprocess.CompletedProcess[str]:
  str_cmd = [str(c) for c in cmd]
  res = subprocess.run(str_cmd, capture_output=True, text=True, check=False)
  if res.returncode != 0:
    raise RuntimeError(
        f"Command failed (exit {res.returncode}): {' '.join(str_cmd)}\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
  return res


def _format_noise_lut(noise_lut: Sequence[float]) -> str:
  if len(noise_lut) != 8:
    raise ValueError(f"Expected 8 noise_lut floats, got {len(noise_lut)}")
  return ",".join(f"{float(v):.8g}" for v in noise_lut)


_SHM_DIR: str | None = "/dev/shm" if Path("/dev/shm").is_dir() else None


def dump_init_state(
    input_image: str | Path | np.ndarray,
    distance: float = 1.0,
    effort: int = 7,
    output_state: str | Path | None = None,
    output_jxl: str | Path | None = None,
) -> JxlVarDCTState:
  """Runs `jxl_wasserstein_helper --dump-init` on an image path or RGB array."""
  with tempfile.TemporaryDirectory(prefix="jxlw_dump_", dir=_SHM_DIR) as tmpdir:
    tmp = Path(tmpdir)
    if isinstance(input_image, np.ndarray):
      arr = np.asarray(input_image)
      if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB array (H, W, 3), got {arr.shape}")
      if np.issubdtype(arr.dtype, np.floating):
        arr_u8 = np.clip(np.round(arr * 255.0), 0.0, 255.0).astype(np.uint8)
      else:
        arr_u8 = np.asarray(arr, dtype=np.uint8)
      in_path = tmp / "input.png"
      Image.fromarray(arr_u8, mode="RGB").save(in_path)
    else:
      in_path = Path(input_image)

    state_path = (
        Path(output_state) if output_state is not None else (tmp / "state.bin")
    )
    cmd: list[str | Path] = [
        HELPER_BIN,
        "--dump-init",
        "--input",
        in_path,
        "--distance",
        f"{float(distance):.6g}",
        "--effort",
        str(int(effort)),
        "--output-state",
        state_path,
    ]
    if output_jxl is not None:
      cmd.extend(["--output-jxl", Path(output_jxl)])

    _run_cmd(cmd)
    return load_state_bin(state_path)


def encode_single_frame(
    state: JxlVarDCTState | str | Path,
    output_jxl: str | Path | None = None,
    effort: int = 7,
    noise_lut: Sequence[float] | None = None,
) -> tuple[bytes, dict[str, Any]]:
  """Serializes a single VarDCT frame state into a valid .jxl bitstream."""
  with tempfile.TemporaryDirectory(prefix="jxlw_single_", dir=_SHM_DIR) as tmpdir:
    tmp = Path(tmpdir)
    if isinstance(state, JxlVarDCTState):
      state_path = save_state_bin(
          state,
          tmp / "state.bin",
          include_dequant=has_custom_dequant_matrices(state),
      )
    else:
      state_path = Path(state)

    jxl_path = Path(output_jxl) if output_jxl is not None else (tmp / "out.jxl")
    json_path = tmp / "stats.json"
    cmd: list[str | Path] = [
        HELPER_BIN,
        "--encode-single",
        "--state",
        state_path,
        "--output-jxl",
        jxl_path,
        "--effort",
        str(int(effort)),
        "--json-out",
        json_path,
    ]
    if noise_lut is not None:
      cmd.extend(["--noise-lut", _format_noise_lut(noise_lut)])

    _run_cmd(cmd)
    jxl_bytes = jxl_path.read_bytes()
    stats = json.loads(json_path.read_text())
    return jxl_bytes, stats


def _validate_single_full_frame_patch(
    patches: Sequence[PatchEntry],
    expected_hw: tuple[int, int] | None,
    allowed_blend_modes: Sequence[int],
) -> None:
  """Validates that `patches` contains strictly 1 full-frame `(0, 0, W, H)` patch with an allowed blend mode."""
  if len(patches) != 1:
    raise ValueError(
        f"Expected exactly 1 full-frame patch, got {len(patches)} patches"
    )
  p = patches[0]
  if int(p.ref) != 0 or int(p.x0) != 0 or int(p.y0) != 0 or int(p.x) != 0 or int(p.y) != 0:
    raise ValueError(
        f"Patch must be co-located at origin (0, 0) referencing frame 0, got {p!r}"
    )
  if expected_hw is not None:
    exp_h, exp_w = int(expected_hw[0]), int(expected_hw[1])
    if int(p.xsize) != exp_w or int(p.ysize) != exp_h:
      raise ValueError(
          f"Patch must span full frame ({exp_w}x{exp_h}), got ({p.xsize}x{p.ysize})"
      )
  if int(p.blend_mode) not in allowed_blend_modes:
    raise ValueError(
        f"Unsupported patch blend_mode {p.blend_mode} (allowed: {tuple(allowed_blend_modes)})"
    )


def encode_patches_composite(
    ref_state: JxlVarDCTState | str | Path,
    main_state: JxlVarDCTState | str | Path,
    upsampling: int,
    patches: Sequence[PatchEntry] | None = None,
    output_jxl: str | Path | None = None,
    effort: int = 7,
    noise_lut: Sequence[float] | None = None,
) -> tuple[bytes, dict[str, Any]]:
  """Serializes a 2-frame ReferenceOnly + single full-frame kAdd PatchDictionary composite .jxl bitstream."""
  expected_hw = (
      (int(main_state.height), int(main_state.width))
      if isinstance(main_state, JxlVarDCTState)
      else None
  )
  if patches is not None:
    _validate_single_full_frame_patch(patches, expected_hw, (PATCH_BLEND_ADD,))
  elif expected_hw is not None:
    patches = [
        PatchEntry(
            ref=0,
            x0=0,
            y0=0,
            xsize=expected_hw[1],
            ysize=expected_hw[0],
            x=0,
            y=0,
            blend_mode=PATCH_BLEND_ADD,
        )
    ]

  with tempfile.TemporaryDirectory(prefix="jxlw_patches_", dir=_SHM_DIR) as tmpdir:
    tmp = Path(tmpdir)
    if isinstance(ref_state, JxlVarDCTState):
      ref_path = save_state_bin(
          ref_state,
          tmp / "ref.bin",
          include_dequant=has_custom_dequant_matrices(ref_state),
      )
    else:
      ref_path = Path(ref_state)

    if isinstance(main_state, JxlVarDCTState):
      main_path = save_state_bin(
          main_state,
          tmp / "main.bin",
          include_dequant=has_custom_dequant_matrices(main_state),
      )
    else:
      main_path = Path(main_state)

    jxl_path = Path(output_jxl) if output_jxl is not None else (tmp / "out.jxl")
    json_path = tmp / "stats.json"
    cmd: list[str | Path] = [
        HELPER_BIN,
        "--encode-patches",
        "--ref-state",
        ref_path,
        "--main-state",
        main_path,
        "--upsampling",
        str(int(upsampling)),
        "--output-jxl",
        jxl_path,
        "--effort",
        str(int(effort)),
        "--json-out",
        json_path,
    ]
    if patches is not None:
      patches_path = save_patches_bin(patches, tmp / "patches.bin")
      cmd.extend(["--patches", patches_path])
    if noise_lut is not None:
      cmd.extend(["--noise-lut", _format_noise_lut(noise_lut)])

    _run_cmd(cmd)
    jxl_bytes = jxl_path.read_bytes()
    stats = json.loads(json_path.read_text())
    return jxl_bytes, stats


def encode_alpha_composite(
    ref_state: JxlVarDCTState | str | Path,
    main_state: JxlVarDCTState | str | Path,
    upsampling: int,
    dim_shift: int,
    alpha_u8: np.ndarray,
    output_jxl: str | Path | None = None,
    effort: int = 7,
    noise_lut: Sequence[float] | None = None,
    alpha_blend_mode: int | str = PATCH_BLEND_ALPHA_BELOW,
    patches: Sequence[PatchEntry] | None = None,
    alpha_bits: int = 8,
    alpha_clamp: bool = False,
) -> tuple[bytes, dict[str, Any]]:
  """Serializes a 2-frame ReferenceOnly + ExtraChannel::kAlpha single-patch XYB composite .jxl bitstream."""
  if isinstance(alpha_blend_mode, str):
    key = alpha_blend_mode.strip().lower()
    if key not in VALID_ALPHA_BLEND_MODE_NAMES:
      raise ValueError(
          f"Unsupported alpha_blend_mode {alpha_blend_mode!r}; expected one of {VALID_ALPHA_BLEND_MODES}"
      )
    bm_int = VALID_ALPHA_BLEND_MODE_NAMES[key]
  else:
    bm_int = int(alpha_blend_mode)
    if bm_int not in VALID_ALPHA_BLEND_MODES:
      raise ValueError(
          f"Unsupported alpha_blend_mode {alpha_blend_mode}; expected one of {VALID_ALPHA_BLEND_MODES}"
      )

  expected_hw = (
      (int(main_state.height), int(main_state.width))
      if isinstance(main_state, JxlVarDCTState)
      else None
  )
  if patches is not None:
    _validate_single_full_frame_patch(
        patches, expected_hw, VALID_ALPHA_BLEND_MODES
    )
  with tempfile.TemporaryDirectory(prefix="jxlw_alpha_", dir=_SHM_DIR) as tmpdir:
    tmp = Path(tmpdir)
    if isinstance(ref_state, JxlVarDCTState):
      ref_path = save_state_bin(
          ref_state,
          tmp / "ref.bin",
          include_dequant=has_custom_dequant_matrices(ref_state),
      )
    else:
      ref_path = Path(ref_state)

    if isinstance(main_state, JxlVarDCTState):
      main_path = save_state_bin(
          main_state,
          tmp / "main.bin",
          include_dequant=has_custom_dequant_matrices(main_state),
      )
    else:
      main_path = Path(main_state)

    alpha_path = save_alpha_bin(alpha_u8, dim_shift, tmp / "alpha.bin")
    jxl_path = Path(output_jxl) if output_jxl is not None else (tmp / "out.jxl")
    json_path = tmp / "stats.json"
    cmd: list[str | Path] = [
        HELPER_BIN,
        "--encode-alpha",
        "--ref-state",
        ref_path,
        "--main-state",
        main_path,
        "--upsampling",
        str(int(upsampling)),
        "--dim-shift",
        str(int(dim_shift)),
        "--alpha",
        alpha_path,
        "--alpha-blend-mode",
        str(alpha_blend_mode),
        "--output-jxl",
        jxl_path,
        "--effort",
        str(int(effort)),
        "--json-out",
        json_path,
    ]
    if patches is not None:
      patches_path = save_patches_bin(patches, tmp / "patches.bin")
      cmd.extend(["--patches", patches_path])
    if int(alpha_bits) != 8:
      cmd.extend(["--alpha-bits", str(int(alpha_bits))])
    if alpha_clamp:
      cmd.extend(["--alpha-clamp", "1"])
    if noise_lut is not None:
      cmd.extend(["--noise-lut", _format_noise_lut(noise_lut)])

    _run_cmd(cmd)
    jxl_bytes = jxl_path.read_bytes()
    stats = json.loads(json_path.read_text())
    return jxl_bytes, stats


def inspect_jxl(jxl_input: bytes | str | Path) -> dict[str, Any]:
  """Runs `jxl_wasserstein_helper --inspect` and returns the parsed JSON dictionary."""
  with tempfile.TemporaryDirectory(prefix="jxlw_inspect_") as tmpdir:
    tmp = Path(tmpdir)
    if isinstance(jxl_input, (bytes, bytearray)):
      in_path = tmp / "in.jxl"
      in_path.write_bytes(jxl_input)
    else:
      in_path = Path(jxl_input)
    res = _run_cmd([HELPER_BIN, "--inspect", "--input", in_path])
    return json.loads(res.stdout)


def _load_decoded_png_rgb(
    png_path: Path, check_opaque_alpha: bool = True
) -> np.ndarray:
  with Image.open(png_path) as im:
    arr = np.asarray(im)
  if arr.ndim == 2:
    arr = np.stack([arr, arr, arr], axis=-1)
  elif arr.ndim == 3 and arr.shape[2] == 4:
    if check_opaque_alpha and not np.all(arr[..., 3] == 255):
      min_a = int(np.min(arr[..., 3]))
      raise AssertionError(
          f"Decoded RGBA PNG has non-opaque alpha channel (min={min_a} != 255)"
      )
    arr = arr[..., :3]
  if arr.ndim != 3 or arr.shape[2] != 3:
    raise ValueError(f"Unexpected decoded image shape {arr.shape}")
  return np.ascontiguousarray(arr, dtype=np.uint8)


def decode_with_djxl(
    jxl_input: bytes | str | Path,
    check_opaque_alpha: bool = True,
) -> np.ndarray:
  """Decodes a .jxl codestream with libjxl's `djxl` into a uint8 (H, W, 3) RGB array."""
  with tempfile.TemporaryDirectory(prefix="jxlw_djxl_") as tmpdir:
    tmp = Path(tmpdir)
    if isinstance(jxl_input, (bytes, bytearray)):
      in_path = tmp / "in.jxl"
      in_path.write_bytes(jxl_input)
    else:
      in_path = Path(jxl_input)
    out_png = tmp / "out.png"
    _run_cmd([DJXL_BIN, in_path, out_png])
    return _load_decoded_png_rgb(out_png, check_opaque_alpha=check_opaque_alpha)


def decode_with_jxl_rs(
    jxl_input: bytes | str | Path,
    check_opaque_alpha: bool = True,
) -> np.ndarray:
  """Decodes a .jxl codestream with jxl-rs (`jxl_cli`) into a uint8 (H, W, 3) RGB array."""
  with tempfile.TemporaryDirectory(prefix="jxlw_jxlrs_") as tmpdir:
    tmp = Path(tmpdir)
    if isinstance(jxl_input, (bytes, bytearray)):
      in_path = tmp / "in.jxl"
      in_path.write_bytes(jxl_input)
    else:
      in_path = Path(jxl_input)
    out_png = tmp / "out.png"
    _run_cmd([JXL_CLI_BIN, in_path, out_png])
    return _load_decoded_png_rgb(out_png, check_opaque_alpha=check_opaque_alpha)


def verify_with_jxlinfo(jxl_input: bytes | str | Path) -> str:
  """Verifies a .jxl codestream with `jxlinfo -v` and returns its stdout."""
  with tempfile.TemporaryDirectory(prefix="jxlw_info_") as tmpdir:
    tmp = Path(tmpdir)
    if isinstance(jxl_input, (bytes, bytearray)):
      in_path = tmp / "in.jxl"
      in_path.write_bytes(jxl_input)
    else:
      in_path = Path(jxl_input)
    res = _run_cmd([JXLINFO_BIN, "-v", in_path])
    return res.stdout


def compute_psnr(
    img_a: np.ndarray,
    img_b: np.ndarray,
    peak: float | None = None,
) -> float:
  """Computes RGB PSNR in dB between two images (uint8 [0, 255] or float [0, 1])."""
  a = np.asarray(img_a, dtype=np.float64)
  b = np.asarray(img_b, dtype=np.float64)
  if a.shape != b.shape:
    raise ValueError(f"PSNR shape mismatch: {a.shape} vs {b.shape}")
  if peak is None:
    if np.issubdtype(np.asarray(img_a).dtype, np.integer) or np.max(a) > 1.5:
      peak = 255.0
    else:
      peak = 1.0
  mse = float(np.mean(np.square(a - b)))
  if mse <= 0.0:
    return 100.0
  return float(10.0 * math.log10((peak * peak) / mse))


def verify_jxl_codestream(
    jxl_input: bytes | str | Path,
    min_cross_decoder_psnr: float = 45.0,
) -> dict[str, Any]:
  """Verifies a .jxl codestream with jxlinfo, --inspect, djxl, and jxl-rs (jxl_cli)."""
  jxlinfo_out = verify_with_jxlinfo(jxl_input)
  inspect_data = inspect_jxl(jxl_input)
  rgb_djxl = decode_with_djxl(jxl_input)
  rgb_jxl_rs = decode_with_jxl_rs(jxl_input)
  cross_psnr = compute_psnr(rgb_djxl, rgb_jxl_rs, peak=255.0)
  if cross_psnr < min_cross_decoder_psnr:
    raise AssertionError(
        f"Cross-decoder PSNR(djxl, jxl_cli) = {cross_psnr:.2f} dB < "
        f"{min_cross_decoder_psnr:.2f} dB"
    )
  return {
      "jxlinfo": jxlinfo_out,
      "inspect": inspect_data,
      "rgb_djxl": rgb_djxl,
      "rgb_jxl_rs": rgb_jxl_rs,
      "cross_decoder_psnr": cross_psnr,
  }
