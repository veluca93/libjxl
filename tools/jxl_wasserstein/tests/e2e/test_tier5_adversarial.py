"""Tier 5 E2E Test Suite: White-Box Adversarial Hardening (23 tests across M1-M4).

Combines:
- Part 1 (8 tests: `test_adv_1_...` through `test_adv_8_...`):
  White-box adversarial coverage for M1-M3 modules (`vgg.py`, `wasserstein.py`,
  `cpp_bridge.py`, `tools/jxl_wasserstein_helper.cc`, `noise.py`,
  `forward_model.py`, `entropy_proxy.py`).
- Part 2 (15 tests: `test_adv_m4_01_...` through `test_adv_m4_15_...`):
  White-box adversarial coverage for M4 modules (`optimizer.py`,
  `compositing.py`, `cli.py`, `benchmark.py`, `__init__.py`, `pyproject.toml`).
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import struct
import sys
import tempfile
import tomllib
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", "/tmp/jxl_wasserstein_jax_cache"
)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")

import jax
import jax.numpy as jnp

Path("/tmp/jxl_wasserstein_jax_cache").mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jxl_wasserstein_jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

import numpy as np
from PIL import Image
import pytest

REPO_ROOT = Path("/usr/local/google/home/veluca/libjxl")
PKG_ROOT = REPO_ROOT / "tools" / "jxl_wasserstein"
SRC_DIR = PKG_ROOT / "src" / "jxl_wasserstein"
E2E_DIR = PKG_ROOT / "tests" / "e2e"

if str(E2E_DIR) not in sys.path:
  sys.path.insert(0, str(E2E_DIR))

import conftest as _e2e_conftest  # noqa: E402
import jxl_wasserstein  # noqa: E402
from jxl_wasserstein import benchmark, cli, compositing, cpp_bridge, optimizer  # noqa: E402
from jxl_wasserstein.cpp_bridge import (  # noqa: E402
    HELPER_BIN,
    PATCH_BLEND_ADD,
    PATCH_BLEND_REPLACE,
    PatchEntry,
    _format_noise_lut,
    _run_cmd,
    decode_with_djxl,
    decode_with_jxl_rs,
    dump_init_state,
    encode_single_frame,
    load_state_bin,
    save_alpha_bin,
    save_state_bin,
)
from jxl_wasserstein.entropy_proxy import (  # noqa: E402
    soft_ac_meta_rate_bits,
    soft_clamped_gradient_2d,
    soft_dc_rate_bits,
    soft_jxl_rate_proxy,
    soft_patch_dictionary_bits,
)
from jxl_wasserstein.forward_model import (  # noqa: E402
    StaticVarDCTTopology,
    apply_xyb_patches,
    init_state_from_rgb_jax,
    reconstruct_alpha_composite,
    reconstruct_single_frame,
    reconstruct_single_frame_xyb,
)
from jxl_wasserstein.noise import (  # noqa: E402
    add_jxl_noise_xyb,
    convolve_noise_planes,
    generate_raw_noise_planes_np,
)
from jxl_wasserstein.vgg import (  # noqa: E402
    VggNet,
    ensure_chw_rgb,
    load_vgg16_params,
    multiscale_vgg_features,
    select_num_scales,
)
from jxl_wasserstein.wasserstein import (  # noqa: E402
    compute_wd,
    multi_wasserstein_distortion,
)


@pytest.fixture(scope="module")
def vgg_params() -> Any:
  return load_vgg16_params()


# =============================================================================
# Part 1: M1-M3 White-Box Adversarial Tests (test_adv_1 .. test_adv_8)
# =============================================================================


# ---------------------------------------------------------------------------
# Gap 1: vgg.py & wasserstein.py — Asymmetric tiny dimension scale clamping
# & input validation branches
# ---------------------------------------------------------------------------
def test_adv_1_vgg_wasserstein_asymmetric_scales_and_validation(
    vgg_params: Any,
) -> None:
  """White-box audit of vgg.py and wasserstein.py scale clamping and guards.

  Exercises:
  - Asymmetric dimensions (16x64, 64x16, 30x64, 31x64, 64x31, 63x64) where
    min(H, W) triggers scale reduction while max(H, W) >= 64.
  - compute_wd with explicit num_scales > select_num_scales(H, W) auto-clamping,
    num_levels=0 (raw MSE on VGG features without pooling), and 2D log2_sigma.
  - Validation branches in ensure_chw_rgb, select_num_scales, VggNet.__call__,
    multiscale_vgg_features, multi_wasserstein_distortion, and compute_wd.
  """
  assert select_num_scales(16, 64, 4) == 1
  assert select_num_scales(64, 16, 4) == 1
  assert select_num_scales(30, 64, 4) == 1
  assert select_num_scales(31, 64, 4) == 2
  assert select_num_scales(64, 31, 4) == 2
  assert select_num_scales(63, 64, 4) == 3

  rng = np.random.default_rng(101)
  img_16x64_a = jnp.asarray(
      rng.uniform(0.0, 1.0, size=(16, 64, 3)).astype(np.float32)
  )
  img_16x64_b = jnp.clip(img_16x64_a + 0.05, 0.0, 1.0)
  sigma_2d = jnp.full((16, 64), 1.5, dtype=jnp.float32)
  wd_16x64 = float(
      compute_wd(
          img_16x64_a,
          img_16x64_b,
          vgg_params,
          log2_sigma=sigma_2d,
          num_levels=0,
          num_scales=4,
      )
  )
  assert np.isfinite(wd_16x64) and wd_16x64 > 0.0

  img_31x64_a = jnp.asarray(
      rng.uniform(0.0, 1.0, size=(31, 64, 3)).astype(np.float32)
  )
  img_31x64_b = jnp.clip(img_31x64_a + 0.05, 0.0, 1.0)
  wd_31x64 = float(
      compute_wd(img_31x64_a, img_31x64_b, vgg_params, num_scales=4)
  )
  assert np.isfinite(wd_31x64) and wd_31x64 > 0.0

  with pytest.raises(ValueError, match="Expected 3D image"):
    ensure_chw_rgb(jnp.zeros((16, 16), dtype=jnp.float32))
  with pytest.raises(ValueError, match="Expected channel dim of size 3"):
    ensure_chw_rgb(jnp.zeros((4, 16, 16), dtype=jnp.float32))
  with pytest.raises(ValueError, match="must be >= 16x16"):
    select_num_scales(15, 32)
  with pytest.raises(ValueError, match="requires spatial dims >= 16"):
    VggNet().apply(vgg_params, jnp.zeros((1, 15, 32, 3), dtype=jnp.float32))
  with pytest.raises(ValueError, match="must be at least 1"):
    multiscale_vgg_features(
        vgg_params, jnp.zeros((3, 16, 16), dtype=jnp.float32), num_scales=0
    )
  with pytest.raises(ValueError, match="Expected image_a_01 shape"):
    compute_wd(
        jnp.zeros((3, 16, 16), dtype=jnp.float32),
        jnp.zeros((3, 16, 16), dtype=jnp.float32),
        vgg_params,
    )
  with pytest.raises(ValueError, match="must have same shape"):
    compute_wd(
        jnp.zeros((16, 16, 3), dtype=jnp.float32),
        jnp.zeros((16, 24, 3), dtype=jnp.float32),
        vgg_params,
    )
  with pytest.raises(ValueError, match="num_levels must be >= 0"):
    compute_wd(
        jnp.zeros((16, 16, 3), dtype=jnp.float32),
        jnp.zeros((16, 16, 3), dtype=jnp.float32),
        vgg_params,
        num_levels=-1,
    )
  with pytest.raises(ValueError, match="num_scales must be >= 1"):
    compute_wd(
        jnp.zeros((16, 16, 3), dtype=jnp.float32),
        jnp.zeros((16, 16, 3), dtype=jnp.float32),
        vgg_params,
        num_scales=0,
    )
  with pytest.raises(ValueError, match="2D log2_sigma shape"):
    compute_wd(
        jnp.zeros((16, 16, 3), dtype=jnp.float32),
        jnp.zeros((16, 16, 3), dtype=jnp.float32),
        vgg_params,
        log2_sigma=jnp.zeros((8, 8), dtype=jnp.float32),
    )
  with pytest.raises(ValueError, match="log2_sigma must be scalar or 2D"):
    compute_wd(
        jnp.zeros((16, 16, 3), dtype=jnp.float32),
        jnp.zeros((16, 16, 3), dtype=jnp.float32),
        vgg_params,
        log2_sigma=jnp.zeros((1, 16, 16), dtype=jnp.float32),
    )
  with pytest.raises(ValueError, match="non-empty"):
    multi_wasserstein_distortion([], [], jnp.zeros((16, 16)))
  with pytest.raises(ValueError, match="must have same length"):
    multi_wasserstein_distortion(
        [jnp.zeros((4, 8, 8))],
        [jnp.zeros((4, 8, 8)), jnp.zeros((4, 8, 8))],
        jnp.zeros((16, 16)),
    )
  with pytest.raises(ValueError, match="Expected 2D"):
    multi_wasserstein_distortion(
        [jnp.zeros((4, 8, 8))],
        [jnp.zeros((4, 8, 8))],
        jnp.zeros((16,)),
    )
  with pytest.raises(ValueError, match="identical shapes"):
    multi_wasserstein_distortion(
        [jnp.zeros((4, 8, 8))],
        [jnp.zeros((4, 8, 16))],
        jnp.zeros((16, 16)),
    )


# ---------------------------------------------------------------------------
# Gap 2: cpp_bridge.py & jxl_wasserstein_helper.cc — Binary format & helper
# validation error branches
# ---------------------------------------------------------------------------
def test_adv_2_cpp_bridge_binary_state_and_helper_error_branches() -> None:
  """White-box audit of binary state serialization and C++ helper error paths.

  Exercises:
  - load_state_bin with unsupported version != 1 (cpp_bridge.py:179-181).
  - ReadStateFile in jxl_wasserstein_helper.cc:781-784 with valid 340-byte
    JXLWST01 header but truncated payload via --encode-single.
  - _format_noise_lut length != 8 check (cpp_bridge.py:412-414).
  - save_alpha_bin non-2D array check (cpp_bridge.py:390-391).
  - dump_init_state invalid array shape check (cpp_bridge.py:430-431).
  """
  rgb = np.zeros((16, 16, 3), dtype=np.float32)
  state = init_state_from_rgb_jax(rgb, distance=1.0)

  with tempfile.TemporaryDirectory(prefix="jxl_wst_adv2_") as tmpdir:
    state_path = os.path.join(tmpdir, "state.bin")
    save_state_bin(state, state_path)
    raw = bytearray(Path(state_path).read_bytes())

    bad_ver_raw = bytearray(raw)
    struct.pack_into("<I", bad_ver_raw, 8, 99)
    bad_ver_path = os.path.join(tmpdir, "bad_version.bin")
    Path(bad_ver_path).write_bytes(bytes(bad_ver_raw))
    with pytest.raises(ValueError, match="Unsupported state version 99"):
      load_state_bin(bad_ver_path)

    trunc_payload_path = os.path.join(tmpdir, "trunc_payload.bin")
    Path(trunc_payload_path).write_bytes(bytes(raw[:400]))
    with pytest.raises(RuntimeError, match="Truncated state payload"):
      _run_cmd([
          HELPER_BIN,
          "--encode-single",
          "--state",
          trunc_payload_path,
          "--output-jxl",
          os.path.join(tmpdir, "out.jxl"),
      ])

    with pytest.raises(ValueError, match="Expected 8 noise_lut floats"):
      _format_noise_lut([0.0] * 7)

    with pytest.raises(ValueError, match="Expected 2D alpha_u8 array"):
      save_alpha_bin(
          np.zeros((16, 16, 1), dtype=np.uint8),
          0,
          os.path.join(tmpdir, "alpha.bin"),
      )

    with pytest.raises(ValueError, match="Expected RGB array"):
      dump_init_state(np.zeros((16, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="Expected RGB array"):
      dump_init_state(np.zeros((16, 16, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# Gap 3: cpp_bridge.py & forward_model.py — Custom Gaborish weights, custom
# EPF sigma/channel weights, and epf_iters=3 bit-exactness vs djxl & jxl-rs
# ---------------------------------------------------------------------------
def test_adv_3_custom_gaborish_epf_iters3_and_custom_epf_weights() -> None:
  """White-box audit of custom LoopFilter serialization + 3-stage EPF in JAX.

  Exercises:
  - ConfigureFrameHeaderLoopFilterFromState in jxl_wasserstein_helper.cc:824-860
    setting gab_custom=true, epf_sigma_custom=true, epf_weight_custom=true, and
    epf_iters=3.
  - reconstruct_single_frame in forward_model.py:1242-1264 running all 3 EPF
    passes (epf_stage0 -> epf_stage1 -> epf_stage2) with custom Gaborish and
    non-uniform epf_sharpness fields.
  - Cross-verification against both libjxl (djxl) and jxl-rs decoders.
  """
  rng = np.random.default_rng(303)
  rgb = rng.uniform(0.05, 0.95, size=(32, 32, 3)).astype(np.float32)
  state = init_state_from_rgb_jax(rgb, distance=1.5, epf_iters=3, gab=True)

  state.gab = True
  state.gab_weights = (0.09, 0.05, 0.12, 0.06, 0.10, 0.04)
  state.epf_iters = 3
  state.epf_quant_mul = 0.72
  state.epf_pass0_sigma_scale = 1.45
  state.epf_pass2_sigma_scale = 5.80
  state.epf_border_sad_mul = 0.68
  state.epf_channel_scale = (35.0, 4.5, 3.2)
  state.epf_sharpness = (
      np.arange(16, dtype=np.uint8).reshape(4, 4) % 8
  ).astype(np.uint8)

  jxl_bytes, stats = encode_single_frame(state)
  assert stats["total_bits"] > 0

  dec_djxl = decode_with_djxl(jxl_bytes).astype(np.float32) / 255.0
  dec_jxl_rs = decode_with_jxl_rs(jxl_bytes).astype(np.float32) / 255.0
  jax_rec = np.asarray(reconstruct_single_frame(state, hard=True))

  max_err_djxl = float(np.max(np.abs(jax_rec - dec_djxl)))
  max_err_rs = float(np.max(np.abs(jax_rec - dec_jxl_rs)))
  assert max_err_djxl < 0.015, f"JAX vs djxl mismatch: {max_err_djxl}"
  assert max_err_rs < 0.015, f"JAX vs jxl-rs mismatch: {max_err_rs}"


# ---------------------------------------------------------------------------
# Gap 4: forward_model.py — Full-frame mixed rectangular/AFV AcStrategy
# (DCT4X8, DCT8X4, AFV0..3, DCT16X8, DCT8X16) + non-zero DC CfL (ytox_dc,
# ytob_dc) + non-default x_qm_scale / b_qm_scale
# ---------------------------------------------------------------------------
def test_adv_4_full_frame_mixed_sub8x8_afv_rect_dct_and_dc_cfl() -> None:
  """White-box audit of full-frame mixed AcStrategy + DC CfL + QM scales.

  Exercises:
  - Full-frame forward_DCT + quantize + dequant + IDCT with DCT4X8 (12),
    DCT8X4 (13), AFV0..3 (14..17), DCT4X4 (2), DCT2X2 (3), DCT16X8 (6),
    DCT8X16 (7), DCT16X16 (4), IDENTITY (1) in a single 32x32 frame.
  - Non-zero ytox_dc and ytob_dc in dequant_and_smooth_dc (forward_model.py:729)
    and jxl_wasserstein_helper.cc:934-943.
  - Non-default x_qm_scale=3 and b_qm_scale=1.
  """
  rng = np.random.default_rng(404)
  rgb = rng.uniform(0.1, 0.9, size=(32, 32, 3)).astype(np.float32)

  ac_strategy = np.array(
      [
          [4, 4, 6, 6],
          [4, 4, 12, 13],
          [7, 14, 16, 17],
          [7, 15, 2, 1],
      ],
      dtype=np.uint8,
  )

  state = init_state_from_rgb_jax(
      rgb,
      distance=1.2,
      ac_strategy=ac_strategy,
      epf_iters=1,
  )
  state.ytox_dc = 18
  state.ytob_dc = -25
  state.x_qm_scale = 3
  state.b_qm_scale = 1

  jxl_bytes, _ = encode_single_frame(state)
  dec_djxl = decode_with_djxl(jxl_bytes).astype(np.float32) / 255.0
  dec_jxl_rs = decode_with_jxl_rs(jxl_bytes).astype(np.float32) / 255.0
  jax_rec = np.asarray(reconstruct_single_frame(state, hard=True))

  assert float(np.max(np.abs(jax_rec - dec_djxl))) < 0.015
  assert float(np.max(np.abs(jax_rec - dec_jxl_rs))) < 0.015


# ---------------------------------------------------------------------------
# Gap 5: forward_model.py — StaticVarDCTTopology validation, apply_xyb_patches
# invalid ref/blend_mode guards, and reconstruct_alpha_composite 3D shapes
# ---------------------------------------------------------------------------
def test_adv_5_forward_model_topology_patch_and_alpha_edge_cases() -> None:
  """White-box audit of StaticVarDCTTopology, patch validation, and alpha shapes.

  Exercises:
  - reconstruct_single_frame_xyb raising ValueError when StaticVarDCTTopology
    is passed without all 5 state tensors (forward_model.py:1098-1102).
  - apply_xyb_patches raising ValueError when ref != 0 or blend_mode not in {1,2}
     and HWC vs CHW layout branch (forward_model.py:1760-1789).
  - reconstruct_alpha_composite with (1, H, W) and (H, W, 1) 3D alpha tensors
    and quantize_alpha=False vs True (forward_model.py:1986-1993).
  """
  rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
  state = init_state_from_rgb_jax(rgb, distance=1.0)
  topo = StaticVarDCTTopology(state)
  xyb_with_state = reconstruct_single_frame_xyb(topo)
  assert xyb_with_state.shape == (3, 16, 16)

  topo_detached = StaticVarDCTTopology(state)
  topo_detached.state = None
  with pytest.raises(ValueError, match="all 5 tensors"):
    reconstruct_single_frame_xyb(topo_detached)

  canvas_chw = jnp.zeros((3, 16, 16), dtype=jnp.float32)
  ref0_chw = jnp.ones((3, 16, 16), dtype=jnp.float32)
  bad_ref_patch = PatchEntry(
      ref=1,
      x0=0,
      y0=0,
      xsize=8,
      ysize=8,
      x=0,
      y=0,
      blend_mode=PATCH_BLEND_REPLACE,
  )
  with pytest.raises(ValueError, match="Expected patch ref == 0"):
    apply_xyb_patches(ref0_chw, canvas_chw, [bad_ref_patch])

  bad_blend_patch = PatchEntry(
      ref=0, x0=0, y0=0, xsize=8, ysize=8, x=0, y=0, blend_mode=99
  )
  with pytest.raises(ValueError, match="Unsupported patch blend_mode: 99"):
    apply_xyb_patches(ref0_chw, canvas_chw, [bad_blend_patch])

  valid_add_patch = PatchEntry(
      ref=0,
      x0=0,
      y0=0,
      xsize=8,
      ysize=8,
      x=4,
      y=4,
      blend_mode=PATCH_BLEND_ADD,
  )
  patched_hwc = apply_xyb_patches(
      jnp.ones((16, 16, 3), dtype=jnp.float32),
      jnp.zeros((16, 16, 3), dtype=jnp.float32),
      [valid_add_patch],
  )
  assert patched_hwc.shape == (16, 16, 3)
  assert float(patched_hwc[6, 6, 0]) == 1.0

  a_2d = jnp.full((16, 16), 0.503, dtype=jnp.float32)
  a_1hw = a_2d[None, :, :]
  a_hw1 = a_2d[:, :, None]

  comp_2d = reconstruct_alpha_composite(
      state, state, upsampling=1, dim_shift=0, alpha_01=a_2d, quantize_alpha=False
  )
  comp_1hw = reconstruct_alpha_composite(
      state, state, upsampling=1, dim_shift=0, alpha_01=a_1hw, quantize_alpha=False
  )
  comp_hw1 = reconstruct_alpha_composite(
      state, state, upsampling=1, dim_shift=0, alpha_01=a_hw1, quantize_alpha=False
  )
  assert jnp.allclose(comp_2d, comp_1hw, atol=1e-7)
  assert jnp.allclose(comp_2d, comp_hw1, atol=1e-7)


# ---------------------------------------------------------------------------
# Gap 6: noise.py — Steep positive/negative out-of-range LUT slopes with
# quantize_lut and non-default ytox/ytob correlation + RNG grid edge cases
# ---------------------------------------------------------------------------
def test_adv_6_noise_steep_lut_slopes_ytox_and_rng_validation() -> None:
  """White-box audit of noise synthesis with steep/negative LUT slopes and ytox.

  Exercises:
  - add_jxl_noise_xyb (noise.py:249-319) with steep alternating LUT values
    where adjacent LUT differences are both negative and > 1.0, testing the
    max_strength = max(k, 0.0) and clamp(strength, 0.0, max_strength) branch.
  - Non-zero ytox correlation (out_x = vx + ytox * rg_noise + (red - green))
    and HWC vs CHW layout preservation (noise.py:267-274, 319).
  - quantize_lut=False vs quantize_lut=True STE gradient flow.
  - generate_raw_noise_planes_np validation guards and single-batch row path
    (batches_per_row == 1 when group_dim=16).
  """
  with pytest.raises(ValueError, match="Invalid noise plane dimensions"):
    generate_raw_noise_planes_np(0, 16)
  with pytest.raises(ValueError, match="Invalid group_dim"):
    generate_raw_noise_planes_np(16, 16, group_dim=0)

  planes_g16 = generate_raw_noise_planes_np(16, 16, group_dim=16)
  assert planes_g16.shape == (3, 16, 16) and np.all(np.isfinite(planes_g16))

  rng = np.random.default_rng(606)
  xyb = jnp.asarray(
      rng.uniform(-0.05, 0.5, size=(3, 16, 16)).astype(np.float32)
  )
  conv_planes = convolve_noise_planes(
      jnp.asarray(generate_raw_noise_planes_np(16, 16))
  )
  steep_lut = jnp.array(
      [-0.25, 1.40, 0.02, 0.85, -0.10, 0.50, 0.999, 0.10], dtype=jnp.float32
  )

  noisy_q = add_jxl_noise_xyb(
      xyb,
      conv_planes,
      steep_lut,
      ytox=0.35,
      ytob=0.75,
      quantize_lut=True,
  )
  noisy_noq = add_jxl_noise_xyb(
      xyb,
      conv_planes,
      steep_lut,
      ytox=0.35,
      ytob=0.75,
      quantize_lut=False,
  )
  assert jnp.all(jnp.isfinite(noisy_q))
  assert jnp.all(jnp.isfinite(noisy_noq))

  xyb_hwc = jnp.moveaxis(xyb, 0, -1)
  noisy_hwc = add_jxl_noise_xyb(
      xyb_hwc, conv_planes, steep_lut, ytox=0.35, ytob=0.75, quantize_lut=True
  )
  assert noisy_hwc.shape == (16, 16, 3)
  assert jnp.allclose(jnp.moveaxis(noisy_q, 0, -1), noisy_hwc, atol=1e-6)

  with pytest.raises(ValueError, match="Expected 3D XYB tensor"):
    add_jxl_noise_xyb(jnp.zeros((16, 16)), conv_planes, steep_lut)
  with pytest.raises(ValueError, match="Expected XYB shape"):
    add_jxl_noise_xyb(jnp.zeros((4, 16, 16)), conv_planes, steep_lut)
  with pytest.raises(ValueError, match="Expected noise_lut of shape"):
    add_jxl_noise_xyb(xyb, conv_planes, jnp.zeros((7,)))
  with pytest.raises(ValueError, match="convolved_noise shape"):
    add_jxl_noise_xyb(xyb, jnp.zeros((3, 8, 8)), steep_lut)

  grad_lut = jax.grad(
      lambda lut: jnp.sum(
          add_jxl_noise_xyb(
              xyb, conv_planes, lut, ytox=0.35, quantize_lut=True
          )
          ** 2
      )
  )(steep_lut)
  assert jnp.all(jnp.isfinite(grad_lut))
  assert float(jnp.linalg.norm(grad_lut)) > 0.0


# ---------------------------------------------------------------------------
# Gap 7: entropy_proxy.py — soft_jxl_rate_proxy with state=None +
# quant_ac_override, extreme tau under @jax.jit, and degenerate 1D/thin planes
# ---------------------------------------------------------------------------
def test_adv_7_entropy_proxy_state_none_override_and_degenerate_planes() -> None:
  """White-box audit of soft_jxl_rate_proxy(state=None) and degenerate planes.

  Exercises:
  - soft_jxl_rate_proxy(state=None, quant_ac_override=...) fallback defaults
    (entropy_proxy.py:1564-1616) and ValueError when both state and
    quant_ac_override are None.
  - Extreme tau values (1e-6 and 10.0) under @jax.jit with gradient check.
  - soft_clamped_gradient_2d on degenerate (1, 1), (1, 8), and (8, 1) planes
    (entropy_proxy.py:790-810).
  """
  with pytest.raises(
      ValueError, match="Either `state` or `quant_ac` must be provided"
  ):
    soft_jxl_rate_proxy(None)

  rng = np.random.default_rng(707)
  q_ac = jnp.asarray(
      rng.normal(0.0, 2.5, size=(3, 16, 16)).astype(np.float32)
  )

  for tau_val in [1e-6, 10.0]:
    @jax.jit
    def rate_and_grad(ac: jnp.ndarray, _tau: float = tau_val):
      def loss_fn(x):
        res = soft_jxl_rate_proxy(
            None, quant_ac_override=x, tau=_tau, return_breakdown=True
        )
        return res.total_bits, res

      (val, bd), g = jax.value_and_grad(loss_fn, has_aux=True)(ac)
      return val, bd, g

    total_bits, bd, g = rate_and_grad(q_ac)
    assert jnp.isfinite(total_bits) and float(total_bits) > 0.0
    assert jnp.all(jnp.isfinite(g))
    assert float(bd.patch_bits) == 0.0
    assert float(bd.alpha_bits) == 0.0

  for shape in [(1, 1), (1, 8), (8, 1)]:
    plane = jnp.arange(shape[0] * shape[1], dtype=jnp.float32).reshape(shape)
    res_plane = soft_clamped_gradient_2d(plane, tau=0.25)
    assert res_plane.shape == shape
    assert jnp.all(jnp.isfinite(res_plane))


# ---------------------------------------------------------------------------
# Gap 8: entropy_proxy.py — soft_patch_dictionary_bits with dict patches &
# multi-target delta coding (idx > 0), soft_dc_rate_bits(dc_quant_is_raw=True),
# and soft_ac_meta_rate_bits(epf_sharpness=None, return_breakdown=True)
# ---------------------------------------------------------------------------
def test_adv_8_entropy_proxy_dict_patches_delta_coding_and_breakdowns() -> None:
  """White-box audit of dict patch entries, multi-target delta coding, and breakdowns.

  Exercises:
  - soft_patch_dictionary_bits with dict-typed patch entries (entropy_proxy.py:1195)
    and multiple target positions sharing a single reference rectangle (idx > 0
    delta coding dx = tx - prev_x, dy = ty - prev_y at entropy_proxy.py:1241-1244).
  - soft_dc_rate_bits with dc_quant_is_raw=True and return_breakdown=True
    (entropy_proxy.py:938-942, 1017-1023).
  - soft_ac_meta_rate_bits with epf_sharpness=None and return_breakdown=True
    (entropy_proxy.py:1091, 1147-1158).
  """
  dict_patches = [
      {
          "ref": 0,
          "x0": 0,
          "y0": 0,
          "xsize": 8,
          "ysize": 8,
          "x": 8,
          "y": 0,
          "blend_mode": PATCH_BLEND_REPLACE,
      },
      {
          "ref": 0,
          "x0": 0,
          "y0": 0,
          "xsize": 8,
          "ysize": 8,
          "x": 16,
          "y": 8,
          "blend_mode": PATCH_BLEND_ADD,
      },
      {
          "ref": 0,
          "x0": 0,
          "y0": 0,
          "xsize": 8,
          "ysize": 8,
          "x": 24,
          "y": 16,
          "blend_mode": PATCH_BLEND_REPLACE,
      },
      {
          "ref": 0,
          "x0": 8,
          "y0": 8,
          "xsize": 8,
          "ysize": 8,
          "x": 0,
          "y": 16,
      },
  ]
  bits_dict = soft_patch_dictionary_bits(dict_patches)
  assert jnp.isfinite(bits_dict) and float(bits_dict) > 74.0

  obj_patches = [
      PatchEntry(
          ref=0,
          x0=0,
          y0=0,
          xsize=8,
          ysize=8,
          x=8,
          y=0,
          blend_mode=PATCH_BLEND_REPLACE,
      ),
      PatchEntry(
          ref=0,
          x0=0,
          y0=0,
          xsize=8,
          ysize=8,
          x=16,
          y=8,
          blend_mode=PATCH_BLEND_ADD,
      ),
      PatchEntry(
          ref=0,
          x0=0,
          y0=0,
          xsize=8,
          ysize=8,
          x=24,
          y=16,
          blend_mode=PATCH_BLEND_REPLACE,
      ),
      PatchEntry(
          ref=0,
          x0=8,
          y0=8,
          xsize=8,
          ysize=8,
          x=0,
          y=16,
          blend_mode=PATCH_BLEND_REPLACE,
      ),
  ]
  bits_obj = soft_patch_dictionary_bits(obj_patches)
  assert jnp.allclose(bits_dict, bits_obj, atol=1e-6)

  dc_quant_arr = jnp.ones((3, 4, 4), dtype=jnp.float32) * 5.0
  dc_total_raw, dc_bd_raw = soft_dc_rate_bits(
      dc_quant_arr,
      ytox_dc=12,
      ytob_dc=-10,
      dc_quant_is_raw=True,
      base_correlation_x=0.05,
      base_correlation_b=1.0,
      return_breakdown=True,
  )
  dc_total_unraw, dc_bd_unraw = soft_dc_rate_bits(
      dc_quant_arr,
      ytox_dc=12,
      ytob_dc=-10,
      dc_quant_is_raw=False,
      base_correlation_x=0.05,
      base_correlation_b=1.0,
      return_breakdown=True,
  )
  assert jnp.isfinite(dc_total_raw) and float(dc_total_raw) > 0.0
  assert jnp.isfinite(dc_total_unraw) and float(dc_total_unraw) > 0.0
  assert "dc_layer_bits" in dc_bd_raw and "modular_tree_bits" in dc_bd_raw
  assert not jnp.allclose(dc_total_raw, dc_total_unraw)

  ac_meta_total, ac_meta_bd = soft_ac_meta_rate_bits(
      raw_quant_field=jnp.ones((4, 4), dtype=jnp.float32) * 4.0,
      ytox_map=jnp.zeros((1, 1), dtype=jnp.float32),
      ytob_map=jnp.zeros((1, 1), dtype=jnp.float32),
      ac_strategy=jnp.zeros((4, 4), dtype=jnp.int32),
      epf_sharpness=None,
      return_breakdown=True,
  )
  assert isinstance(ac_meta_bd, dict)
  assert float(ac_meta_bd["epf_bits"]) > 0.0
  assert jnp.isfinite(ac_meta_total)


# =============================================================================
# Part 2: M4 White-Box Adversarial Tests (test_adv_m4_01 .. test_adv_m4_10)
# =============================================================================


# ---------------------------------------------------------------------------
# 1. Strict R2 AST & Source Audit Across M4 Encoder Modules
# ---------------------------------------------------------------------------
def test_adv_m4_01_strict_r2_ast_and_source_audit() -> None:
  """White-box AST audit verifying 0 dump_init_state imports/calls across M4 modules."""
  m4_files = ("optimizer.py", "compositing.py", "cli.py", "benchmark.py")
  for fname in m4_files:
    fpath = SRC_DIR / fname
    source = fpath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(fpath))

    for node in ast.walk(tree):
      if isinstance(node, ast.ImportFrom):
        for alias in node.names:
          assert alias.name != "dump_init_state", (
              f"R2 AST violation: {fname} imports dump_init_state"
          )
          if fname != "benchmark.py":
            assert alias.name != "CJXL_BIN", (
                f"R2 AST violation: {fname} imports CJXL_BIN"
            )
      elif isinstance(node, ast.Name):
        assert node.id != "dump_init_state", (
            f"R2 AST violation: {fname} references symbol dump_init_state"
        )
        if fname != "benchmark.py":
          assert node.id != "CJXL_BIN", (
              f"R2 AST violation: {fname} references symbol CJXL_BIN"
          )
      elif isinstance(node, ast.Attribute):
        assert node.attr != "dump_init_state", (
            f"R2 AST violation: {fname} accesses attribute .dump_init_state"
        )
      elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        assert "--dump-init" not in node.value, (
            f"R2 AST violation: {fname} contains '--dump-init' string literal"
        )

    if fname == "benchmark.py":
      for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name != "encode_cjxl_baseline"
        ):
          fn_names = {
              sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)
          }
          assert "CJXL_BIN" not in fn_names, (
              f"R2 violation: benchmark.py function {node.name} references CJXL_BIN"
          )


# ---------------------------------------------------------------------------
# 2. Pure-JAX Block Strategy & Quant/EPF Selector on Adversarial Region Types
# ---------------------------------------------------------------------------
def test_adv_m4_02_select_block_params_extreme_regions_and_low_distance(
    vgg_params: Any,
) -> None:
  """Exercises Stage 1+2 Rate-WD DP in select_ac_strategy_jax, select_quant_and_epf_jax, and HF denoising."""
  h, w = 64, 64
  img = np.full((h, w, 3), 0.5, dtype=np.float32)

  # Top-left 32x32: completely flat -> selects multi-block DCT
  img[:32, :32] = 0.35

  # Top-right (by=0, bx=4..5): gentle vertical variation with zero horizontal gradient
  for y in range(8):
    img[y, 32:48] = 0.35 + (0.04 if (y >= 4) else -0.04)

  # Bottom-left (by=4, bx=0): moderate 4x4 quadrant step -> DCT4X4 (3) or sub-8x8
  img[32:36, 0:4] = 0.42
  img[32:36, 4:8] = 0.58
  img[36:40, 0:4] = 0.58
  img[36:40, 4:8] = 0.42

  # Bottom-left (by=4, bx=1): 2x2 sub-block step -> sub-8x8 transform
  for sy in range(4):
    for sx in range(4):
      img[32 + sy * 2 : 32 + (sy + 1) * 2, 8 + sx * 2 : 8 + (sx + 1) * 2] = (
          0.10 if ((sy + sx) % 2 == 0) else 0.90
      )

  # Bottom-left (by=4, bx=2): single-corner diagonal edge -> AFV0..3 (14..17) or sub-8x8
  for dy in range(4):
    for dx in range(4):
      img[32 + dy, 16 + dx] = 0.95 if (dy > dx) else 0.05

  acs = optimizer.select_ac_strategy_jax(
      img,
      distance=1.5,
      allow_multiblock=True,
      distance_adaptive=True,
      vgg_params=vgg_params,
  )
  assert acs.shape == (8, 8)
  optimizer.validate_ac_strategy_tiling(acs)
  assert int(acs[0, 0]) in (4, 5, 6, 7), (
      f"Expected multi-block DCT (4..7) on flat region, got {acs[0, 0]}"
  )
  assert len(np.unique(acs)) >= 3, (
      f"Expected diverse Rate-WD DP strategies on heterogeneous regions, got {np.unique(acs)}"
  )
  assert int(acs[4, 2]) in (1, 2, 3, 14, 15, 16, 17), (
      f"Expected sub-8x8 or AFV strategy on corner diagonal edge, got {acs[4, 2]}"
  )

  # Low distance < 0.2 branch in select_quant_and_epf_jax -> epf_iters == 0 and epf_sharpness == 0
  raw_q_low, epf_low, epf_iters_low = optimizer.select_quant_and_epf_jax(
      img, distance=0.1, ac_strategy=acs
  )
  assert epf_iters_low == 0
  assert np.all(epf_low == 0)
  assert raw_q_low.shape == (8, 8)

  # Heterogeneous raw_quant_field broadcast across multi-block strategies
  rng = np.random.default_rng(7)
  hetero_q = rng.integers(1, 256, size=(8, 8), dtype=np.int32)
  bcast_q = optimizer.broadcast_quant_field_to_ac_strategy(hetero_q, acs)
  cx00 = cpp_bridge.ACS_COVERED_BLOCKS_X[int(acs[0, 0])]
  cy00 = cpp_bridge.ACS_COVERED_BLOCKS_Y[int(acs[0, 0])]
  assert np.all(bcast_q[:cy00, :cx00] == bcast_q[0, 0])

  # Untested helper apply_hf_noise_denoising_jax
  q_ac_dummy = jnp.ones((3, 64, 64), dtype=jnp.float32)
  q_denoised = np.asarray(
      optimizer.apply_hf_noise_denoising_jax(
          q_ac_dummy, h8=8, w8=8, hf_shrink=0.3
      )
  )
  assert q_denoised.shape == (3, 64, 64)
  assert np.all(q_denoised[:, ::8, ::8] == 0.0)
  assert float(np.mean(np.abs(q_denoised))) < 1.0


# ---------------------------------------------------------------------------
# 3. Single-Frame Multi-Block LLF Masking & Stage-2 Metadata Freezing
# ---------------------------------------------------------------------------
def test_adv_m4_03_single_frame_multiblock_llf_masking_and_stage2_freeze(
    vgg_params: Any,
) -> None:
  """Tests optimize_single_frame across Stage-1 -> Stage-2 freeze boundary with multi-block + noise."""
  img = _e2e_conftest.make_mixed_test_image(64, 64)
  res = optimizer.optimize_single_frame(
      img,
      vgg_params,
      distance=2.0,
      rd_lambda=0.20,
      scale_distance_with_lambda=False,
      num_steps=3,
      lr=0.06,
      enable_noise=True,
      allow_multiblock=True,
      num_scales=1,
      num_levels=1,
      checkpoint_interval=2,
      verify_decoders=True,
  )
  assert res.cross_decoder_psnr is not None and res.cross_decoder_psnr >= 45.0
  assert res.noise_lut is not None and max(res.noise_lut) > 0.001
  topo = optimizer.StaticVarDCTTopology(res.state)
  masked = optimizer.mask_llf_numpy(res.state.quant_ac, topo)
  np.testing.assert_array_equal(res.state.quant_ac, masked)


# ---------------------------------------------------------------------------
# 4. Patch Extraction Edge Cases (0-Patch Fallback, Non-Multiple-of-8 Clipping, blend_mode="auto")
# ---------------------------------------------------------------------------
def test_adv_m4_04_patch_extraction_zero_fg_single_block_and_auto_blend(
    vgg_params: Any,
) -> None:
  """Tests 0-patch fallback, non-8-multiple rectangle clipping, and blend_mode='auto'."""
  empty_rects = compositing.extract_patch_rectangles(
      np.zeros((4, 4), dtype=bool), height=32, width=32
  )
  assert empty_rects == []

  mask_l = np.array(
      [
          [True, True, False, True],
          [True, False, False, True],
          [True, True, True, True],
      ],
      dtype=bool,
  )
  rects_l = compositing.extract_patch_rectangles(mask_l, height=19, width=29)
  assert len(rects_l) >= 3
  cov = np.zeros((19, 29), dtype=int)
  for p in rects_l:
    assert 0 <= p.x < 29 and 0 <= p.y < 19
    assert p.x + p.xsize <= 29 and p.y + p.ysize <= 19
    cov[p.y : p.y + p.ysize, p.x : p.x + p.xsize] += 1
  assert np.all(cov <= 1), (
      "Overlapping patches produced by extract_patch_rectangles!"
  )

  img64 = _e2e_conftest.make_composite_smooth_sharp_image(64, 64)
  res_auto_bm = compositing.optimize_patch_composite(
      img64,
      upsampling=2,
      blend_mode="auto",
      distance=0.45,
      lambda_rd=0.05,
      num_steps=1,
      verify_decoders=True,
      vgg_params=vgg_params,
  )
  assert res_auto_bm.blend_mode in (
      cpp_bridge.PATCH_BLEND_REPLACE,
      cpp_bridge.PATCH_BLEND_ADD,
  )
  assert res_auto_bm.psnr_djxl_vs_jxlrs >= 45.0


# ---------------------------------------------------------------------------
# 5. Alpha Composite with upsampling=2, dim_shift=3 (max(U, 1<<D) == 8) & Noise
# ---------------------------------------------------------------------------
def test_adv_m4_05_alpha_composite_u2_dimshift3_ec_upsampling_and_noise(
    vgg_params: Any,
) -> None:
  """Tests optimize_alpha_composite when 1<<dim_shift (8) > upsampling (2) in XYB PatchDictionary mode."""
  img64 = _e2e_conftest.make_composite_smooth_sharp_image(64, 64)
  res = compositing.optimize_alpha_composite(
      img64,
      upsampling=2,
      dim_shift=3,
      distance=0.45,
      lambda_rd=0.05,
      num_steps=1,
      enable_noise=True,
      verify_decoders=True,
      vgg_params=vgg_params,
  )
  assert res.mode_selected == "alpha"
  assert res.upsampling == 2
  assert res.dim_shift == 3
  assert res.has_noise is True
  assert res.alpha_u8 is not None and res.alpha_u8.shape == (8, 8)
  assert res.header_info["extra_channels"][0]["dim_shift"] == 3
  assert res.header_info["frames"][0]["upsampling"] == 2
  assert res.header_info["frames"][0]["ec_upsampling"] == [1]
  assert res.header_info["frames"][0]["effective_ec_upsampling"] == [8]
  assert res.header_info["frames"][0]["save_before_color_transform"] is True
  assert res.header_info["frames"][1]["has_patches"] is True
  assert res.header_info["frames"][1]["patch_blend_modes"] == [5]
  assert res.header_info["frames"][1]["patch_ec_blend_modes"] == [1]
  assert res.psnr_djxl_vs_jxlrs >= 45.0
  assert float(jnp.linalg.norm(res.last_grads["alpha_logit"])) > 0.0


# ---------------------------------------------------------------------------
# 6. CLI Error Paths (FileNotFoundError with -o, Invalid --distances, --json-out Only + Crop)
# ---------------------------------------------------------------------------
def test_adv_m4_06_cli_error_handling_missing_file_and_invalid_flags(
    tmp_path: Path,
) -> None:
  """Tests CLI error handling when input file is missing (with -o set) or flags are invalid."""
  with pytest.raises(FileNotFoundError, match="Input image not found"):
    cli.main([
        "--input",
        str(tmp_path / "missing_input.png"),
        "--output",
        str(tmp_path / "out.jxl"),
    ])

  with pytest.raises(SystemExit):
    cli.main(["--benchmark-kodak", "--distances", "not_a_float"])

  with pytest.raises(SystemExit):
    cli.main(["-i", "a.png", "-o", "b.jxl", "--upsampling", "3"])
  with pytest.raises(SystemExit):
    cli.main(
        ["-i", "a.png", "-o", "b.jxl", "--patch-blend-mode", "invalid_mode"]
    )

  in_png = tmp_path / "valid.png"
  json_out = tmp_path / "only_json.json"
  img_u8 = np.full((64, 64, 3), 128, dtype=np.uint8)
  Image.fromarray(img_u8, mode="RGB").save(in_png)
  rc = cli.main([
      "--input",
      str(in_png),
      "--json-out",
      str(json_out),
      "--crop-size",
      "64",
      "--steps",
      "0",
  ])
  assert rc == 0
  assert json_out.is_file()


# ---------------------------------------------------------------------------
# 7. Package __getattr__ Lazy Exports & Result Container Mapping Protocol
# ---------------------------------------------------------------------------
def test_adv_m4_07_init_lazy_exports_and_result_mapping_protocol(
    vgg_params: Any,
) -> None:
  """Tests __getattr__ error handling and dict/mapping methods on OptimizationResult and EncodeResult."""
  with pytest.raises(AttributeError):
    _ = getattr(jxl_wasserstein, "non_existent_symbol_xyz")
  with pytest.raises(AttributeError):
    _ = getattr(optimizer, "non_existent_symbol_xyz")

  img16 = np.full((16, 16, 3), 0.5, dtype=np.float32)
  opt_res = optimizer.optimize_single_frame(
      img16,
      vgg_params,
      rd_lambda=0.1,
      num_steps=0,
      num_scales=1,
      num_levels=1,
      allow_multiblock=False,
  )
  assert "jxl_bytes" in opt_res
  assert opt_res["mode"] == "single"
  assert opt_res.get("non_existent_key", 42) == 42
  assert len(list(opt_res.keys())) > 5
  assert len(list(opt_res.values())) == len(list(opt_res.items()))
  with pytest.raises(KeyError):
    _ = opt_res["non_existent_key"]

  enc_res = compositing.encode_image_to_jxl(
      img16,
      mode="single",
      num_steps=0,
      verify_decoders=False,
      vgg_params=vgg_params,
  )
  enc_res["custom_tag"] = "verified"
  assert enc_res["custom_tag"] == "verified"
  assert "jxl_bytes" in enc_res
  assert isinstance(enc_res.to_dict(), dict)


# ---------------------------------------------------------------------------
# 8. pyproject.toml Console Script Entry Point (jxl-wasserstein-benchmark = benchmark:main)
# ---------------------------------------------------------------------------
def test_adv_m4_08_pyproject_benchmark_console_script_entrypoint() -> None:
  """Verifies that every [project.scripts] entry point in pyproject.toml resolves to a callable."""
  pyproject_data = tomllib.loads(
      (PKG_ROOT / "pyproject.toml").read_text(encoding="utf-8")
  )
  scripts = pyproject_data.get("project", {}).get("scripts", {})
  assert "jxl-wasserstein" in scripts
  assert "jxl-wasserstein-benchmark" in scripts

  for script_name, target in scripts.items():
    mod_name, func_name = target.split(":", 1)
    mod = __import__(mod_name, fromlist=[func_name])
    assert hasattr(mod, func_name), (
        f"Console script {script_name!r} targets '{mod_name}:{func_name}', "
        f"but {mod_name} has no attribute {func_name!r}!"
    )
    fn = getattr(mod, func_name)
    assert callable(fn)

  with pytest.raises(SystemExit) as exc_info:
    benchmark.main(["--help"])
  assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# 9. benchmark.py Parameter Aliases (num_steps, verify_decoders, verify_all_points, bd_rate)
# ---------------------------------------------------------------------------
def test_adv_m4_09_benchmark_parameter_aliases_and_verify_all_points(
    vgg_params: Any,
) -> None:
  """Verifies benchmark.py supports num_steps, verify_decoders, verify_all_points, and bd_rate alias."""
  assert hasattr(benchmark, "bd_rate"), (
      "benchmark.py should export `bd_rate` as an alias for `compute_bd_rate`"
  )
  img_u8 = np.full((64, 64, 3), 120, dtype=np.uint8)
  img_u8[:32, :32] = 200
  res = benchmark.evaluate_image_rd_curve(
      img_u8,
      image_name="synthetic_64",
      distances=(1.4, 2.4, 4.0, 6.8),
      num_steps=0,
      verify_decoders=True,
      verify_all_points=True,
      vgg_params=vgg_params,
  )
  assert len(res.opt_points) == 4
  for pt in res.opt_points:
    assert pt.cross_decoder_psnr is not None and pt.cross_decoder_psnr >= 45.0


# ---------------------------------------------------------------------------
# 10. compute_bd_rate with Non-Monotonic Raw Points Requiring Pareto Envelope
# ---------------------------------------------------------------------------
def test_adv_m4_10_bd_rate_non_monotonic_pareto_envelope() -> None:
  """Verifies compute_bd_rate enforces np.minimum.accumulate envelope on non-monotonic raw R-D points."""
  dist_ref = np.array([0.4, 0.8, 1.6, 3.2], dtype=np.float64)
  rate_ref = np.array([2.4, 1.5, 0.95, 0.55], dtype=np.float64)

  rate_test_clean = 0.90 * rate_ref
  bdr_clean = benchmark.compute_bd_rate(
      rate_ref, dist_ref, rate_test_clean, dist_ref
  )
  assert bdr_clean == pytest.approx(-10.0, abs=1e-3)

  rate_test_spiked = np.array(
      [0.90 * 2.4, 0.90 * 1.5, 3.50, 0.90 * 0.55], dtype=np.float64
  )
  rate_test_pareto = np.minimum.accumulate(rate_test_spiked)
  bdr_spiked = benchmark.compute_bd_rate(
      rate_ref, dist_ref, rate_test_spiked, dist_ref
  )
  bdr_expected_pareto = benchmark.compute_bd_rate(
      rate_ref, dist_ref, rate_test_pareto, dist_ref
  )
  assert bdr_spiked == pytest.approx(bdr_expected_pareto, abs=1e-4), (
      f"compute_bd_rate failed to apply np.minimum.accumulate Pareto envelope on "
      f"non-monotonic rate points: got {bdr_spiked:.4f}% vs Pareto {bdr_expected_pareto:.4f}%"
  )


# ---------------------------------------------------------------------------
# 11. Multi-Group (>256x256) & Non-Multiple-of-32 AcStrategy 2D DP Tiling
# ---------------------------------------------------------------------------
def test_adv_m4_11_multigroup_and_non_mult32_ac_strategy_2d_dp_tiling() -> None:
  """White-box stress test of stage2_select_non_overlapping_ac_strategy and validate_ac_strategy_tiling.

  Probes:
  - Asymmetric non-multiple-of-32 block grids and multi-group (>32x32 blocks = >256x256 px)
    grids: (33, 35), (65, 33), (19, 37), (36, 68).
  - Adversarial cost maps strongly favoring DCT32X32 (4x4), DCT16X16 (2x2), DCT16X8 (2x1),
    and DCT8X16 (1x2) right on internal 32-block group boundaries (by=30..31, bx=30..31)
    and at odd right/bottom frame boundaries.
  - Negative white-box checks that validate_ac_strategy_tiling rejects group-boundary
    crossings, overlapping footprints, out-of-bounds footprints, and invalid strategy IDs.
  - Full C++ --encode-single + djxl + jxl_cli + JAX verification on a >256x256 (264x272,
    H8=33, W8=34) multi-group heterogeneous AcStrategy state.
  """
  rng = np.random.default_rng(20261020)
  for h8, w8 in ((33, 35), (65, 33), (19, 37), (36, 68)):
    best_1x1_id = rng.choice(
        np.array([0, 1, 2, 3, 14, 15, 16, 17], dtype=np.uint8), size=(h8, w8)
    )
    best_1x1_cost = rng.uniform(5.0, 12.0, size=(h8, w8)).astype(np.float32)

    # Make multi-block transforms strongly attractive everywhere, especially
    # at by=30..31 and bx=30..31 (straddling the 32-block group boundary!)
    cost_16x8 = rng.uniform(2.0, 8.0, size=(h8, w8)).astype(np.float32)
    cost_8x16 = rng.uniform(2.0, 8.0, size=(h8, w8)).astype(np.float32)
    cost_16x16 = rng.uniform(3.0, 14.0, size=(h8, w8)).astype(np.float32)
    cost_32x32 = rng.uniform(6.0, 24.0, size=(h8, w8)).astype(np.float32)

    for g_edge in (30, 31, 62, 63):
      if g_edge < h8:
        cost_16x8[g_edge, :] = -100.0
        cost_16x16[g_edge, :] = -200.0
        cost_32x32[g_edge, :] = -500.0
      if g_edge < w8:
        cost_8x16[:, g_edge] = -100.0
        cost_16x16[:, g_edge] = -200.0
        cost_32x32[:, g_edge] = -500.0

    cost_maps: dict[int, np.ndarray] = {}
    for sid_1x1 in (0, 1, 2, 3, 14, 15, 16, 17):
      cost_maps[sid_1x1] = np.where(
          best_1x1_id == sid_1x1, best_1x1_cost, best_1x1_cost + 5.0
      ).astype(np.float32)
    cost_maps[4] = cost_16x16
    cost_maps[5] = cost_32x32
    cost_maps[6] = cost_16x8
    cost_maps[7] = cost_8x16

    acs = optimizer.stage2_select_non_overlapping_ac_strategy(
        cost_maps, h8=h8, w8=w8, allow_multiblock=True
    )
    assert acs.shape == (h8, w8)
    assert optimizer.validate_ac_strategy_tiling(acs) is True
    assert bool(np.any(np.isin(acs, [4, 5, 6, 7])))

    # Also verify sequence-of-dicts input format to stage2_select_non_overlapping_ac_strategy
    cand_seq = [
        {"strategy_id": s_id, "cost_map": cmap}
        for s_id, cmap in cost_maps.items()
    ]
    acs_seq = optimizer.stage2_select_non_overlapping_ac_strategy(
        cand_seq, h8=h8, w8=w8, allow_multiblock=True
    )
    np.testing.assert_array_equal(acs, acs_seq)

    # Also test allow_multiblock=False fallback (must select only 1x1 strategies)
    acs_1x1_only = optimizer.stage2_select_non_overlapping_ac_strategy(
        cost_maps, h8=h8, w8=w8, allow_multiblock=False
    )
    assert acs_1x1_only.shape == (h8, w8)
    assert optimizer.validate_ac_strategy_tiling(acs_1x1_only) is True
    assert not bool(np.any(np.isin(acs_1x1_only, [4, 5, 6, 7])))

  # Negative white-box tests on validate_ac_strategy_tiling (returns False on invalid tilings)
  # 1) DCT16X16 (id=4, 2x2) at by=31 crossing the 32-block group boundary
  bad_group_y = np.zeros((34, 34), dtype=np.uint8)
  bad_group_y[31:33, 0:2] = 4
  assert optimizer.validate_ac_strategy_tiling(bad_group_y) is False

  # 2) DCT32X32 (id=5, 4x4) at bx=30 crossing the 32-block group boundary
  bad_group_x = np.zeros((34, 34), dtype=np.uint8)
  bad_group_x[0:4, 30:34] = 5
  assert optimizer.validate_ac_strategy_tiling(bad_group_x) is False

  # 3) Overlapping multi-block transforms within a group
  bad_overlap = np.zeros((8, 8), dtype=np.uint8)
  bad_overlap[0:2, 0:2] = 4  # DCT16X16
  bad_overlap[1:2, 1:3] = 7  # DCT8X16 overlapping at (1, 1)
  assert optimizer.validate_ac_strategy_tiling(bad_overlap) is False

  # 4) Transform overflowing frame boundary
  bad_oob = np.zeros((5, 5), dtype=np.uint8)
  bad_oob[4, 4] = 4  # DCT16X16 at bottom-right corner
  assert optimizer.validate_ac_strategy_tiling(bad_oob) is False

  # 5) Invalid strategy ID >= 27 and non-2D array
  bad_id = np.zeros((4, 4), dtype=np.uint8)
  bad_id[0, 0] = 27
  assert optimizer.validate_ac_strategy_tiling(bad_id) is False
  assert optimizer.validate_ac_strategy_tiling(np.zeros((4, 4, 1), dtype=np.uint8)) is False

  # End-to-end encode/decode on a >256x256 (264x272 -> H8=33, W8=34, 4 groups) image
  rgb_mg = np.zeros((264, 272, 3), dtype=np.float32)
  yy, xx = np.meshgrid(
      np.linspace(0.0, 1.0, 264, dtype=np.float32),
      np.linspace(0.0, 1.0, 272, dtype=np.float32),
      indexing="ij",
  )
  rgb_mg[..., 0] = 0.25 + 0.5 * yy
  rgb_mg[..., 1] = 0.30 + 0.4 * xx
  rgb_mg[..., 2] = 0.20 + 0.3 * np.sin(6.0 * yy) * np.cos(6.0 * xx)

  acs_mg = np.zeros((33, 34), dtype=np.uint8)
  acs_mg[0:4, 0:4] = 5  # DCT32X32 in group (0, 0)
  acs_mg[28:32, 28:32] = 5  # DCT32X32 right up against group boundary (31, 31)
  acs_mg[32, 32:34] = 7  # DCT8X16 in group (1, 1)
  acs_mg[4:6, 0:2] = 4  # DCT16X16
  acs_mg[6:8, 0] = 6  # DCT16X8
  acs_mg[0, 4] = 1  # IDENTITY
  acs_mg[0, 5] = 2  # DCT2X2
  acs_mg[0, 6] = 3  # DCT4X4
  acs_mg[0, 7] = 14  # AFV0
  assert optimizer.validate_ac_strategy_tiling(acs_mg) is True

  state_mg = init_state_from_rgb_jax(
      rgb_mg, distance=1.0, ac_strategy=acs_mg, epf_iters=1, gab=True
  )
  jxl_mg, _ = encode_single_frame(state_mg)
  ver_mg = cpp_bridge.verify_jxl_codestream(jxl_mg, min_cross_decoder_psnr=45.0)
  assert ver_mg["inspect"]["width"] == 272
  assert ver_mg["inspect"]["height"] == 264
  jax_mg = np.asarray(reconstruct_single_frame(state_mg, hard=True))
  psnr_jax_mg = cpp_bridge.compute_psnr(
      jax_mg * 255.0, ver_mg["rgb_djxl"].astype(np.float32), peak=255.0
  )
  assert psnr_jax_mg >= 40.0, (
      f"Multi-group (>256x256) JAX vs djxl PSNR = {psnr_jax_mg:.2f} dB < 40 dB"
  )


# ---------------------------------------------------------------------------
# 12. Extreme Alpha Mattes Across U in {2,4,8}, dim_shift in {0..3}, BlendModes {4..7}
# ---------------------------------------------------------------------------
def test_adv_m4_12_xyb_patch_alpha_extreme_mattes_and_blend_matrix() -> None:
  """White-box stress test of XYB PatchDictionary alpha blending with extreme mattes.

  Probes:
  - Adversarial alpha mattes: 1px alternating checkerboard, 1px sub-pixel thin strokes,
    and sharp vertical/horizontal/diagonal step edges.
  - Upsampling factors U in {2, 4, 8}, dim_shift in {0, 1, 2, 3}, and
    PatchBlendMode in {4 (kBlendAbove), 5 (kBlendBelow), 6 (kAlphaWeightedAddAbove),
    7 (kAlphaWeightedAddBelow)}, plus odd non-multiple-of-8 frame dimensions (37x29).
  - Verifies save_before_color_transform == True, has_patches == True, opaque decoded
    RGBA output (alpha == 255), djxl vs jxl_cli PSNR >= 45 dB, and JAX vs djxl PSNR >= 40 dB.
  """
  h, w = 32, 32
  yy, xx = np.meshgrid(
      np.arange(h, dtype=np.float32),
      np.arange(w, dtype=np.float32),
      indexing="ij",
  )
  bg_rgb = np.stack(
      [
          0.25 + 0.35 * (yy / h),
          0.30 + 0.30 * (xx / w),
          0.40 + 0.20 * np.cos(yy * 0.2),
      ],
      axis=-1,
  ).astype(np.float32)
  fg_rgb = np.stack(
      [
          0.55 + 0.25 * np.sin(xx * 0.3),
          0.35 + 0.35 * np.cos(yy * 0.3),
          0.25 + 0.40 * ((yy + xx) / (h + w)),
      ],
      axis=-1,
  ).astype(np.float32)

  # Construct 3 adversarial full-res alpha mattes in uint8 [0, 255]:
  # (a) 1px high-frequency checkerboard
  alpha_checker = (
      (((np.arange(h)[:, None] + np.arange(w)[None, :]) % 2) * 215 + 20)
      .astype(np.uint8)
  )
  # (b) 1px thin diagonal & crosshair strokes
  alpha_strokes = np.zeros((h, w), dtype=np.uint8)
  alpha_strokes[np.arange(h), np.arange(w)] = 255
  alpha_strokes[np.arange(h), (w - 1) - np.arange(w)] = 240
  alpha_strokes[h // 2, :] = 255
  alpha_strokes[:, w // 3] = 200
  # (c) Sharp step edge with sub-pixel transition band
  alpha_step = np.where(xx < 15, 0, np.where(xx == 15, 128, 255)).astype(
      np.uint8
  )

  # Test all U in {2, 4, 8}, dim_shift in {0, 1, 2, 3}, and PatchBlendMode in {4, 5, 6, 7}
  test_configs = [
      # (U, dim_shift, blend_mode, matte_name, full_alpha_u8)
      (2, 0, 5, "checkerboard_below_u2_d0", alpha_checker),
      (2, 0, 4, "checkerboard_above_u2_d0", alpha_checker),
      (2, 1, 5, "strokes_below_u2_d1", alpha_strokes),
      (2, 1, 6, "step_add_above_u2_d1", alpha_step),
      (2, 2, 7, "checker_add_below_u2_d2", alpha_checker),
      (4, 0, 5, "strokes_below_u4_d0", alpha_strokes),
      (4, 1, 7, "step_add_below_u4_d1", alpha_step),
      (4, 2, 5, "checker_below_u4_d2", alpha_checker),
      (4, 3, 5, "strokes_below_u4_d3", alpha_strokes),
      (8, 0, 5, "step_below_u8_d0", alpha_step),
      (8, 2, 5, "checker_below_u8_d2", alpha_checker),
      (8, 3, 7, "strokes_add_below_u8_d3", alpha_strokes),
  ]

  main_state = init_state_from_rgb_jax(fg_rgb, distance=0.8, epf_iters=1, gab=True)
  grad_checked_modes: set[int] = set()

  for u, dshift, bmode, tag, full_a_u8 in test_configs:
    h_ref = (h + u - 1) // u
    w_ref = (w + u - 1) // u
    ref_rgb = np.asarray(
        jax.image.resize(jnp.asarray(bg_rgb), (h_ref, w_ref, 3), method="linear")
    )
    ref_state = init_state_from_rgb_jax(ref_rgb, distance=0.8, epf_iters=1, gab=True)

    step = 1 << dshift
    a_u8 = full_a_u8[::step, ::step].copy()
    a_01 = a_u8.astype(np.float32) / 255.0

    # For additive XYB blend modes (6, 7), scale down main_state AC/DC slightly so
    # summed XYB stays well within dynamic range
    cur_main = main_state.copy()
    cur_ref = ref_state.copy()
    if bmode in (6, 7):
      cur_main = init_state_from_rgb_jax(
          fg_rgb * 0.35 + 0.15, distance=0.8, epf_iters=0, gab=False
      )
      cur_ref = init_state_from_rgb_jax(
          ref_rgb * 0.35 + 0.15, distance=0.8, epf_iters=0, gab=False
      )

    jxl_bytes, _ = cpp_bridge.encode_alpha_composite(
        cur_ref,
        cur_main,
        upsampling=u,
        dim_shift=dshift,
        alpha_u8=a_u8,
        alpha_blend_mode=bmode,
    )
    ver = cpp_bridge.verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
    ins = ver["inspect"]
    assert ins["frames"][0]["save_before_color_transform"] is True, tag
    assert ins["frames"][0]["effective_ec_upsampling"] == [max(u, 1 << dshift)], tag
    assert ins["frames"][1]["has_patches"] is True, tag
    assert ins["frames"][1]["patch_blend_modes"] == [bmode], tag
    assert ins["frames"][1]["patch_ec_blend_modes"] == [1], tag

    # In libjxl dec_patch_dictionary.cc:355 + blending.cc:125-169:
    # - kBlendBelow (5) and kAlphaWeightedAddBelow (7) treat Frame 1 (bg) as `top`
    #   and read Frame 1's extra channel 0 (`a_01`).
    # - kBlendAbove (4) and kAlphaWeightedAddAbove (6) treat Frame 0 (fg) as `top`
    #   and read Frame 0's extra channel 0 (which is filled with opaque 1.0f in
    #   jxl_wasserstein_helper.cc:2285 so EC kReplace restores opaque output alpha).
    eff_alpha_for_djxl = (
        jnp.asarray(a_01) if bmode in (5, 7) else jnp.ones_like(jnp.asarray(a_01))
    )
    jax_rgb = np.asarray(
        reconstruct_alpha_composite(
            cur_ref,
            cur_main,
            upsampling=u,
            dim_shift=dshift,
            alpha_01=eff_alpha_for_djxl,
            alpha_blend_mode=bmode,
            hard=True,
            quantize_alpha=True,
        )
    )
    psnr_jax_djxl = cpp_bridge.compute_psnr(
        jax_rgb * 255.0, ver["rgb_djxl"].astype(np.float32), peak=255.0
    )
    assert psnr_jax_djxl >= 40.0, (
        f"Adversarial alpha {tag} (U={u}, D={dshift}, mode={bmode}): "
        f"JAX vs djxl PSNR = {psnr_jax_djxl:.2f} dB < 40.0 dB"
    )

    # Also verify JAX differentiability w.r.t. non-constant alpha_01 across all modes {4, 5, 6, 7}
    if bmode not in grad_checked_modes:
      grad_checked_modes.add(bmode)
      g_alpha = jax.grad(
          lambda a: jnp.mean(
              jnp.square(
                  reconstruct_alpha_composite(
                      cur_ref,
                      cur_main,
                      upsampling=u,
                      dim_shift=dshift,
                      alpha_01=a,
                      alpha_blend_mode=bmode,
                      hard=False,
                      quantize_alpha=True,
                  )
              )
          )
      )(jnp.asarray(a_01))
      assert bool(jnp.all(jnp.isfinite(g_alpha))) and float(jnp.linalg.norm(g_alpha)) > 0.0
  assert grad_checked_modes == {4, 5, 6, 7}

  # Also test odd non-multiple-of-8 frame dimensions (37x29) with U=4, dim_shift=1
  h_odd, w_odd = 37, 29
  fg_odd = np.full((h_odd, w_odd, 3), 0.65, dtype=np.float32)
  bg_odd = np.full(((h_odd + 3) // 4, (w_odd + 3) // 4, 3), 0.25, dtype=np.float32)
  ref_odd = init_state_from_rgb_jax(bg_odd, distance=1.0)
  main_odd = init_state_from_rgb_jax(fg_odd, distance=1.0)
  h_a_odd = (h_odd + 1) // 2
  w_a_odd = (w_odd + 1) // 2
  a_odd_u8 = (
      (((np.arange(h_a_odd)[:, None] + np.arange(w_a_odd)[None, :]) % 2) * 255)
      .astype(np.uint8)
  )
  jxl_odd, _ = cpp_bridge.encode_alpha_composite(
      ref_odd,
      main_odd,
      upsampling=4,
      dim_shift=1,
      alpha_u8=a_odd_u8,
      alpha_blend_mode=5,
  )
  ver_odd = cpp_bridge.verify_jxl_codestream(jxl_odd, min_cross_decoder_psnr=45.0)
  jax_odd = np.asarray(
      reconstruct_alpha_composite(
          ref_odd,
          main_odd,
          upsampling=4,
          dim_shift=1,
          alpha_01=jnp.asarray(a_odd_u8.astype(np.float32) / 255.0),
          alpha_blend_mode=5,
          hard=True,
      )
  )
  psnr_odd = cpp_bridge.compute_psnr(
      jax_odd * 255.0, ver_odd["rgb_djxl"].astype(np.float32), peak=255.0
  )
  assert psnr_odd >= 40.0


# ---------------------------------------------------------------------------
# 13. Custom DequantMatrices F16 Mantissa Truncation & Extreme Weights Roundtrip
# ---------------------------------------------------------------------------
def test_adv_m4_13_custom_dequant_f16_mantissa_truncation_and_extreme_weights() -> None:
  """White-box stress test of custom DequantMatrices F16 truncation, kQuantModeDCT, and kQuantModeRAW.

  Probes:
  - F16 mantissa truncation on non-dyadic floats and subnormals in _f16_roundtrip_float,
    project_custom_dequant_matrices, ste_dequant_matrices, and roundtrip_custom_dequant_matrices.
  - Extreme per-channel scaling weights (0.25x, 4.0x, channel-asymmetric (0.35, 1.85, 3.25))
    across DCT8 (0), IDENTITY (1), DCT2X2 (2), DCT4X4 (3), DCT16X16 (4), DCT32X32 (5), AFV0 (14).
  - Smooth radial frequency modulation triggering FitDctQuantParams (kQuantModeDCT) AND
    high-variance checkerboard frequency modulation (>5% radial error) triggering kQuantModeRAW
    (qtable_den = 2^-16) in jxl_wasserstein_helper --encode-single.
  - Extreme custom dc_quant_weights with all_default == False, djxl vs jxl_cli PSNR >= 45 dB,
    and JAX vs djxl PSNR >= 40 dB.
  """
  # 1) F16 mantissa truncation property checks
  vals = np.array([1e-8, 1.0 / 4096.0, 0.1234567, 1.0001, 3150.123], dtype=np.float32)
  rt_vals = cpp_bridge._f16_roundtrip_float(vals)
  assert rt_vals.dtype == np.float32
  # Subnormal < 2^-14 (biased_exp32 < 113) flushes to 0 in _f16_roundtrip_float
  assert float(rt_vals[0]) == 0.0
  # Idempotence: roundtripping a roundtripped value is exact
  np.testing.assert_array_equal(rt_vals, cpp_bridge._f16_roundtrip_float(rt_vals))

  # 2) STE gradient preservation under both "dct" and "raw" projection modes
  def_dq = jxl_wasserstein.forward_model.get_default_dequant_matrices()
  dct8_def = jnp.asarray(def_dq[0], dtype=jnp.float32)
  dc_def = jnp.array(cpp_bridge.DEFAULT_DC_QUANT_WEIGHTS, dtype=jnp.float32)

  for proj_mode in ("dct", "raw"):
    def _ste_loss(scale_vec: jax.Array) -> jax.Array:
      custom_map = {0: dct8_def * scale_vec[:, None, None]}
      custom_dc = dc_def * scale_vec
      ste_dq, ste_dc = jxl_wasserstein.forward_model.ste_dequant_matrices(
          custom_map, dc_quant=custom_dc, mode=proj_mode, hard=False
      )
      return jnp.sum(ste_dq[0] ** 2) + jnp.sum(ste_dc**2)

    g_scale = jax.grad(_ste_loss)(jnp.array([0.25, 1.75, 4.0], dtype=jnp.float32))
    assert jnp.all(jnp.isfinite(g_scale)) and float(jnp.linalg.norm(g_scale)) > 0.0

  # 3) Extreme per-channel scaled custom tables across multi-strategy state
  img = _e2e_conftest.make_mixed_test_image(64, 64)
  acs = np.zeros((8, 8), dtype=np.uint8)
  acs[0:4, 0:4] = 5  # DCT32X32
  acs[0:2, 4:6] = 4  # DCT16X16
  acs[4, 0] = 1  # IDENTITY
  acs[4, 1] = 2  # DCT2X2
  acs[4, 2] = 3  # DCT4X4
  acs[4, 3] = 14  # AFV0
  assert optimizer.validate_ac_strategy_tiling(acs) is True

  state_scaled = init_state_from_rgb_jax(
      img, distance=1.2, ac_strategy=acs, epf_iters=1, gab=True
  )
  ch_scales = np.array([0.35, 1.85, 3.25], dtype=np.float32)
  for s_id in (0, 1, 2, 3, 4, 5, 14):
    state_scaled.dequant_matrices[s_id] = (
        def_dq[s_id] * ch_scales[:, None, None]
    ).astype(np.float32)
  state_scaled.dc_quant_weights = (
      np.array(cpp_bridge.DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32)
      * np.array([0.5, 1.5, 2.5], dtype=np.float32)
  )
  assert cpp_bridge.has_custom_dequant_matrices(state_scaled) is True

  rt_dq_scaled, rt_dc_scaled = cpp_bridge.roundtrip_custom_dequant_matrices(
      state_scaled
  )
  state_scaled.dequant_matrices.update(rt_dq_scaled)
  state_scaled.dc_quant_weights = rt_dc_scaled

  jxl_scaled, _ = encode_single_frame(state_scaled)
  ver_scaled = cpp_bridge.verify_jxl_codestream(
      jxl_scaled, min_cross_decoder_psnr=45.0
  )
  ins_scaled = ver_scaled["inspect"]
  assert ins_scaled["all_default"] is False
  assert ins_scaled["dequant_all_default"] is False
  assert ins_scaled["dc_quant_all_default"] is False
  np.testing.assert_allclose(
      np.asarray(ins_scaled["dc_quant_weights"], dtype=np.float32),
      rt_dc_scaled,
      rtol=1e-4,
      atol=1e-6,
  )
  jax_scaled = np.asarray(
      reconstruct_single_frame(
          state_scaled,
          dequant_matrices=rt_dq_scaled,
          dc_quant_weights=rt_dc_scaled,
          hard=True,
      )
  )
  psnr_scaled = cpp_bridge.compute_psnr(
      jax_scaled * 255.0, ver_scaled["rgb_djxl"].astype(np.float32), peak=255.0
  )
  assert psnr_scaled >= 40.0, (
      f"Extreme scaled custom DequantMatrices JAX vs djxl PSNR = {psnr_scaled:.2f} dB < 40 dB"
  )

  # 4) Non-scalar radial slope (FitDctQuantParams -> kQuantModeDCT) AND
  #    high-variance checkerboard table (>5% fit error -> kQuantModeRAW)
  state_dct_fit = init_state_from_rgb_jax(img, distance=1.2, epf_iters=1, gab=True)
  yy8, xx8 = np.meshgrid(
      np.linspace(0.0, 1.0, 8, dtype=np.float32),
      np.linspace(0.0, 1.0, 8, dtype=np.float32),
      indexing="ij",
  )
  radial_mod = np.exp(0.35 * np.sqrt(yy8 * yy8 + xx8 * xx8)).astype(np.float32)
  state_dct_fit.dequant_matrices[0] = (def_dq[0] * radial_mod[None, :, :]).astype(
      np.float32
  )
  rt_dq_fit, rt_dc_fit = cpp_bridge.roundtrip_custom_dequant_matrices(state_dct_fit)
  state_dct_fit.dequant_matrices.update(rt_dq_fit)
  jxl_fit, _ = encode_single_frame(state_dct_fit)
  ver_fit = cpp_bridge.verify_jxl_codestream(jxl_fit, min_cross_decoder_psnr=45.0)
  assert ver_fit["inspect"]["all_default"] is False
  jax_fit = np.asarray(
      reconstruct_single_frame(
          state_dct_fit, dequant_matrices=rt_dq_fit, dc_quant_weights=rt_dc_fit, hard=True
      )
  )
  assert (
      cpp_bridge.compute_psnr(
          jax_fit * 255.0, ver_fit["rgb_djxl"].astype(np.float32), peak=255.0
      )
      >= 40.0
  )

  # High-variance oscillatory multiplier (>5% radial error -> kQuantModeRAW branch)
  state_raw_q = init_state_from_rgb_jax(img, distance=1.2, epf_iters=1, gab=True)
  checker_8x8 = np.where(
      (np.arange(8)[:, None] + np.arange(8)[None, :]) % 2 == 0, 0.65, 1.55
  ).astype(np.float32)
  checker_8x8[0, 0] = 1.0
  state_raw_q.dequant_matrices[0] = (def_dq[0] * checker_8x8[None, :, :]).astype(
      np.float32
  )
  rt_dq_raw, rt_dc_raw = cpp_bridge.roundtrip_custom_dequant_matrices(state_raw_q)
  state_raw_q.dequant_matrices.update(rt_dq_raw)
  jxl_raw, _ = encode_single_frame(state_raw_q)
  ver_raw = cpp_bridge.verify_jxl_codestream(jxl_raw, min_cross_decoder_psnr=45.0)
  assert ver_raw["inspect"]["all_default"] is False
  jax_raw = np.asarray(
      reconstruct_single_frame(
          state_raw_q, dequant_matrices=rt_dq_raw, dc_quant_weights=rt_dc_raw, hard=True
      )
  )
  assert (
      cpp_bridge.compute_psnr(
          jax_raw * 255.0, ver_raw["rgb_djxl"].astype(np.float32), peak=255.0
      )
      >= 40.0
  )


# ---------------------------------------------------------------------------
# 14. Spatial WD Map & Spatial Rate Map Additivity Across All 12 Transform Types
# ---------------------------------------------------------------------------
def test_adv_m4_14_spatial_wd_and_rate_map_additivity_heterogeneous_transforms(
    vgg_params: Any,
) -> None:
  """White-box invariant test of spatial WD map and spatial rate map additivity.

  Probes:
  - Exact additivity `sum(wd_block_map) == total_wd` of `compute_spatial_wd_map`
    against `compute_wd` across square (64x64), non-multiple-of-8 (37x53),
    asymmetric (40x72) geometries, custom `num_scales`/`num_levels`, and 2D `log2_sigma`.
  - Exact additivity of `compute_spatial_rate_map_bits` (`sum(ac_block_bits) == ac_bits`,
    `sum(dc_block_bits) == dc_bits`, `sum(meta_block_bits) == ac_meta_bits`,
    `sum(block_bits) + header_bits + alpha_bits == total_bits`) on a single 64x64 grid
    simultaneously tiling all 12 required transforms:
    `DCT8 (0)`, `IDENTITY (1)`, `DCT2X2 (2)`, `DCT4X4 (3)`, `DCT16X16 (4)`,
    `DCT32X32 (5)`, `DCT16X8 (6)`, `DCT8X16 (7)`, `AFV0..3 (14..17)`.
  """
  rng = np.random.default_rng(1414)
  for (h, w), n_sc, n_lv, l2s in (
      ((64, 64), 1, 1, 3.5),
      ((37, 53), 1, 1, 2.5),
      ((40, 72), 2, 3, jnp.full((40, 72), 3.0, dtype=jnp.float32)),
  ):
    img_a = jnp.asarray(rng.uniform(0.05, 0.95, size=(h, w, 3)).astype(np.float32))
    img_b = jnp.clip(
        img_a + jnp.asarray(rng.normal(0.0, 0.08, size=(h, w, 3)).astype(np.float32)),
        0.0,
        1.0,
    )
    wd_scalar = float(
        jxl_wasserstein.wasserstein.compute_wd(
            img_a,
            img_b,
            vgg_params,
            log2_sigma=l2s,
            num_scales=n_sc,
            num_levels=n_lv,
        )
    )
    total_wd, wd_map = jxl_wasserstein.wasserstein.compute_spatial_wd_map(
        img_a,
        img_b,
        vgg_params,
        block_size=8,
        log2_sigma=l2s,
        num_scales=n_sc,
        num_levels=n_lv,
    )
    assert wd_map.shape == ((h + 7) // 8, (w + 7) // 8)
    assert bool(jnp.all(wd_map >= -1e-7))
    assert float(total_wd) == pytest.approx(wd_scalar, rel=1e-5, abs=1e-5)
    assert float(jnp.sum(wd_map)) == pytest.approx(wd_scalar, rel=1e-4, abs=1e-4)

    if (h, w) == (37, 53):
      # Zero-identity and differentiability check on non-multiple-of-8 spatial WD map
      _, wd_zero = jxl_wasserstein.wasserstein.compute_spatial_wd_map(
          img_a, img_a, vgg_params, block_size=8, num_scales=1, num_levels=1
      )
      assert float(jnp.max(jnp.abs(wd_zero))) < 1e-5
      g_wd = jax.grad(
          lambda x: jnp.sum(
              jxl_wasserstein.wasserstein.compute_spatial_wd_map(
                  img_a, x, vgg_params, block_size=8, num_scales=1, num_levels=1
              )[1]
          )
      )(img_b)
      assert bool(jnp.all(jnp.isfinite(g_wd))) and float(jnp.linalg.norm(g_wd)) > 0.0

  # Construct a single 64x64 (8x8 blocks) grid containing all 12 required transforms:
  # DCT8(0), IDENTITY(1), DCT2X2(2), DCT4X4(3), DCT16X16(4), DCT32X32(5),
  # DCT16X8(6), DCT8X16(7), AFV0(14), AFV1(15), AFV2(16), AFV3(17)
  acs_12 = np.zeros((8, 8), dtype=np.uint8)
  acs_12[0:4, 0:4] = 5  # DCT32X32 (4x4)
  acs_12[0:2, 4:6] = 4  # DCT16X16 (2x2)
  acs_12[0:2, 6] = 6  # DCT16X8 (2x1)
  acs_12[2, 4:6] = 7  # DCT8X16 (1x2)
  acs_12[4, 0] = 0  # DCT8
  acs_12[4, 1] = 1  # IDENTITY
  acs_12[4, 2] = 2  # DCT2X2
  acs_12[4, 3] = 3  # DCT4X4
  acs_12[4, 4] = 14  # AFV0
  acs_12[4, 5] = 15  # AFV1
  acs_12[4, 6] = 16  # AFV2
  acs_12[4, 7] = 17  # AFV3
  assert optimizer.validate_ac_strategy_tiling(acs_12) is True
  required_12 = {0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17}
  assert required_12.issubset(set(int(v) for v in np.unique(acs_12)))

  img64 = _e2e_conftest.make_mixed_test_image(64, 64)
  state_12 = init_state_from_rgb_jax(
      img64, distance=1.0, ac_strategy=acs_12, epf_iters=1, gab=True
  )
  topo_12 = StaticVarDCTTopology(state_12)
  alpha_64 = jnp.asarray(
      rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float32)
  )

  smap = jxl_wasserstein.entropy_proxy.compute_spatial_rate_map_bits(
      state_12,
      topology=topo_12,
      enable_noise=True,
      noise_lut=jnp.linspace(0.01, 0.08, 8, dtype=jnp.float32),
      alpha_01=alpha_64,
      dim_shift=1,
  )
  bd = soft_jxl_rate_proxy(
      state_12,
      topology=topo_12,
      enable_noise=True,
      noise_lut=jnp.linspace(0.01, 0.08, 8, dtype=jnp.float32),
      alpha_01=alpha_64,
      dim_shift=1,
      return_breakdown=True,
  )
  assert smap["block_bits"].shape == (8, 8)
  assert float(jnp.sum(smap["ac_block_bits"])) == pytest.approx(
      float(bd.ac_bits), rel=1e-4, abs=1e-2
  )
  assert float(jnp.sum(smap["dc_block_bits"])) == pytest.approx(
      float(bd.dc_bits), rel=1e-4, abs=1e-2
  )
  assert float(jnp.sum(smap["meta_block_bits"])) == pytest.approx(
      float(bd.ac_meta_bits), rel=1e-4, abs=1e-2
  )
  assert float(
      jnp.sum(smap["block_bits"]) + bd.header_bits + bd.alpha_bits
  ) == pytest.approx(float(bd.total_bits), rel=1e-4, abs=1e-2)

  # Multi-block footprint distribution check: cells [0:4, 0:4] (DCT32X32) other than
  # top-left (0, 0) (which receives the tiny f32 sum-matching residual) have identical
  # distributed AC rate
  ac_32x32_cells = np.asarray(smap["ac_block_bits"][:4, :4]).reshape(-1)[1:]
  np.testing.assert_allclose(ac_32x32_cells, ac_32x32_cells[0], rtol=1e-5, atol=1e-5)

  # Differentiability of compute_spatial_rate_map_bits w.r.t. quant_ac, dc_quant, raw_quant_field
  q_ac_f = jnp.asarray(state_12.quant_ac, dtype=jnp.float32)
  dc_q_f = jnp.asarray(state_12.dc_quant, dtype=jnp.float32)
  raw_q_f = jnp.asarray(state_12.raw_quant_field, dtype=jnp.float32)
  g_qac, g_dc, g_qf = jax.grad(
      lambda qa, dc, qf: jnp.sum(
          jxl_wasserstein.entropy_proxy.compute_spatial_rate_map_bits(
              state_12,
              quant_ac=qa,
              dc_quant=dc,
              raw_quant_field=qf,
              topology=topo_12,
          )["block_bits"]
      ),
      argnums=(0, 1, 2),
  )(q_ac_f, dc_q_f, raw_q_f)
  assert bool(jnp.all(jnp.isfinite(g_qac))) and float(jnp.linalg.norm(g_qac)) > 0.0
  assert bool(jnp.all(jnp.isfinite(g_dc))) and float(jnp.linalg.norm(g_dc)) > 0.0
  assert bool(jnp.all(jnp.isfinite(g_qf))) and float(jnp.linalg.norm(g_qf)) > 0.0


# ---------------------------------------------------------------------------
# 15. Strict Stage 3 Improvement (stage3_true_j < stage2_true_j) & Zero-Heuristics Monkeypatch
# ---------------------------------------------------------------------------
def test_adv_m4_15_strict_stage3_improvement_and_zero_heuristics_monkeypatch(
    vgg_params: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verifies strict Stage 3 true bitstream improvement and zero heuristics/warm-starts.

  Probes:
  - Monkeypatches `cpp_bridge.dump_init_state` and `subprocess.run` to fail immediately
    if `cjxl` or `--dump-init` is ever invoked during optimization.
  - Monkeypatches static pixel-variance/MSE heuristic symbols (`_compute_block_activity_metrics_jax`,
    `compute_detail_mask_jax`) if present to raise AssertionError if ever invoked.
  - Verifies strict `stage3_true_j < stage2_true_j` across `optimize_single_frame`
    (with custom dequant matrices), `optimize_patch_composite`, and `optimize_alpha_composite`.
  """
  if hasattr(optimizer, "_compute_block_activity_metrics_jax"):
    monkeypatch.setattr(
        optimizer,
        "_compute_block_activity_metrics_jax",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("Static pixel variance heuristic must not be called")
        ),
    )
  if hasattr(compositing, "compute_detail_mask_jax"):
    monkeypatch.setattr(
        compositing,
        "compute_detail_mask_jax",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("Static pixel MSE heuristic must not be called")
        ),
    )

  with _e2e_conftest.forbid_libjxl_per_block_choices(monkeypatch):
    img_single = _e2e_conftest.make_mixed_test_image(64, 64)[:32, :32]
    res_s = optimizer.optimize_single_frame(
        img_single,
        vgg_params,
        distance=1.5,
        rd_lambda=0.15,
        num_steps=3,
        lr=0.06,
        custom_dequant=True,
        verify_decoders=True,
    )
    assert res_s.stage3_true_j < res_s.stage2_true_j, (
        f"Single-frame Stage 3 true J ({res_s.stage3_true_j:.6f}) did not strictly "
        f"improve over Stage 2 ({res_s.stage2_true_j:.6f})"
    )
    ver_s = cpp_bridge.verify_jxl_codestream(
        res_s.jxl_bytes, min_cross_decoder_psnr=45.0
    )
    assert ver_s["inspect"]["all_default"] is False
    assert res_s.cross_decoder_psnr is not None and res_s.cross_decoder_psnr >= 45.0

    img_comp = _e2e_conftest.make_composite_smooth_sharp_image(64, 64)
    res_p = compositing.optimize_patch_composite(
        img_comp,
        upsampling=2,
        distance=0.45,
        lambda_rd=0.06,
        num_steps=2,
        verify_decoders=True,
        vgg_params=vgg_params,
    )
    assert res_p.stage3_true_j < res_p.stage2_true_j, (
        f"Patch composite Stage 3 true J ({res_p.stage3_true_j:.6f}) did not strictly "
        f"improve over Stage 2 ({res_p.stage2_true_j:.6f})"
    )
    assert res_p.psnr_djxl_vs_jxlrs >= 45.0
    assert res_p.psnr_jax_vs_djxl >= 40.0

    res_a = compositing.optimize_alpha_composite(
        img_comp,
        upsampling=2,
        dim_shift=1,
        distance=0.45,
        lambda_rd=0.06,
        num_steps=2,
        verify_decoders=True,
        vgg_params=vgg_params,
    )
    assert res_a.stage3_true_j < res_a.stage2_true_j, (
        f"Alpha composite Stage 3 true J ({res_a.stage3_true_j:.6f}) did not strictly "
        f"improve over Stage 2 ({res_a.stage2_true_j:.6f})"
    )
    assert res_a.header_info["frames"][0]["save_before_color_transform"] is True
    assert res_a.header_info["frames"][1]["has_patches"] is True
    assert res_a.psnr_djxl_vs_jxlrs >= 45.0
    assert res_a.psnr_jax_vs_djxl >= 40.0

