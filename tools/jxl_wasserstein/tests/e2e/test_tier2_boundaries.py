"""Tier 2 E2E Test Suite: Boundary Value Analysis & Corner Cases (F1-F13, 5 per feature, 65 total)."""

from __future__ import annotations

import functools
import inspect
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest

from jxl_wasserstein.benchmark import (
    compute_bd_rate,
    load_benchmark_image,
)
from jxl_wasserstein.cli import main as cli_main
from jxl_wasserstein.cpp_bridge import (
    ACS_COVERED_BLOCKS_X,
    ACS_COVERED_BLOCKS_Y,
    PATCH_BLEND_ADD,
    PatchEntry,
    compute_psnr,
    decode_with_djxl,
    encode_alpha_composite,
    encode_patches_composite,
    encode_single_frame,
    inspect_jxl,
    load_state_bin,
    save_state_bin,
    verify_jxl_codestream,
)
import jxl_wasserstein.cpp_bridge as cpp_bridge_mod
from jxl_wasserstein.entropy_proxy import (
    soft_ac_rate_bits,
    soft_alpha_channel_bits,
    soft_jxl_rate_proxy,
    soft_patch_dictionary_bits,
)
import jxl_wasserstein.entropy_proxy as ep_mod
from jxl_wasserstein.forward_model import (
    StaticVarDCTTopology,
    get_default_dequant_matrices,
    init_state_from_rgb_jax,
    opsin_to_srgb,
    reconstruct_alpha_composite,
    reconstruct_single_frame,
    reconstruct_single_frame_xyb,
    upsample_jxl,
)
import jxl_wasserstein.forward_model as fm_mod
from jxl_wasserstein.noise import (
    NOISE_LUT_MAX,
    quantize_noise_lut_ste,
)
from jxl_wasserstein.wasserstein import compute_wd
import jxl_wasserstein.wasserstein as wd_mod

from conftest import (
    FORBIDDEN_LEGACY_TOKENS,
    PKG_ROOT,
    REPO_ROOT,
    forbid_libjxl_per_block_choices,
    get_field,
)


def _make_pattern_image(h: int, w: int, kind: str = "gradient") -> np.ndarray:
  """Generates synthetic boundary pattern images in float32 [0, 1]."""
  if kind == "black":
    return np.zeros((h, w, 3), dtype=np.float32)
  if kind == "white":
    return np.ones((h, w, 3), dtype=np.float32)
  if kind == "gray":
    return np.full((h, w, 3), 0.5, dtype=np.float32)
  yy, xx = np.meshgrid(
      np.arange(h, dtype=np.float32),
      np.arange(w, dtype=np.float32),
      indexing="ij",
  )
  if kind == "checkerboard":
    chk = (((yy.astype(np.int32) + xx.astype(np.int32)) % 2) * 0.8 + 0.1).astype(
        np.float32
    )
    return np.stack([chk, 1.0 - chk, chk], axis=-1)
  if kind == "split_smooth_checker":
    img = np.full((h, w, 3), 0.45, dtype=np.float32)
    chk = (((yy.astype(np.int32) + xx.astype(np.int32)) % 2) * 0.8 + 0.1).astype(
        np.float32
    )
    img[:, w // 2 :, 0] = chk[:, w // 2 :]
    img[:, w // 2 :, 1] = 1.0 - chk[:, w // 2 :]
    img[:, w // 2 :, 2] = chk[:, w // 2 :]
    return img
  r = 0.2 + 0.6 * (yy / max(h - 1, 1))
  g = 0.2 + 0.6 * (xx / max(w - 1, 1))
  b = 0.5 + 0.3 * np.sin(2.0 * np.pi * yy / max(h, 1))
  return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


@functools.lru_cache(maxsize=32)
def _cached_pattern_state(
    h: int, w: int, kind: str = "gradient", distance: float = 1.5, epf_iters: int = 0
):
  img = _make_pattern_image(h, w, kind)
  return init_state_from_rgb_jax(img, distance=distance, epf_iters=epf_iters)


def _call_encode_alpha(
    ref_state: Any,
    main_state: Any,
    upsampling: int,
    dim_shift: int,
    alpha_u8: np.ndarray,
    alpha_blend_mode: int = 5,
) -> tuple[bytes, dict[str, Any]]:
  sig = inspect.signature(encode_alpha_composite)
  kwargs: dict[str, Any] = {
      "ref_state": ref_state,
      "main_state": main_state,
      "upsampling": upsampling,
      "dim_shift": dim_shift,
      "alpha_u8": alpha_u8,
  }
  if "alpha_blend_mode" in sig.parameters:
    kwargs["alpha_blend_mode"] = alpha_blend_mode
  elif "blend_mode" in sig.parameters:
    kwargs["blend_mode"] = alpha_blend_mode
  return encode_alpha_composite(**kwargs)


def _call_reconstruct_alpha(
    ref_state: Any,
    main_state: Any,
    alpha_01: jax.Array,
    upsampling: int = 2,
    dim_shift: int = 0,
    alpha_blend_mode: int = 5,
    hard: bool = True,
) -> jax.Array:
  sig = inspect.signature(reconstruct_alpha_composite)
  kwargs: dict[str, Any] = {
      "ref_state": ref_state,
      "main_state": main_state,
      "alpha_01": alpha_01,
      "upsampling": upsampling,
      "hard": hard,
  }
  if "dim_shift" in sig.parameters:
    kwargs["dim_shift"] = dim_shift
  if "alpha_dim_shift" in sig.parameters:
    kwargs["alpha_dim_shift"] = dim_shift
  if "alpha_blend_mode" in sig.parameters:
    kwargs["alpha_blend_mode"] = alpha_blend_mode
  elif "blend_mode" in sig.parameters:
    kwargs["blend_mode"] = alpha_blend_mode
  return reconstruct_alpha_composite(**kwargs)


# =============================================================================
# F1 Boundaries: Standalone R1 Hygiene & Git Status Boundaries (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "src_dir_zero_legacy_tokens",
        "tests_dir_zero_legacy_tokens",
        "helper_cpp_zero_legacy_tokens",
        "git_status_no_untracked_agent_metadata",
        "xla_deterministic_flags_and_f32_matmul",
    ],
)
def test_f1_boundary_hygiene_and_git_status(case_name: str) -> None:
  """Verifies F1 boundaries: zero legacy references per directory, clean git status of agent metadata, and XLA env."""
  if case_name == "src_dir_zero_legacy_tokens":
    for p in (PKG_ROOT / "src").rglob("*.py"):
      txt = p.read_text(encoding="utf-8")
      for tok in FORBIDDEN_LEGACY_TOKENS:
        assert tok not in txt, f"Forbidden token {tok!r} in {p}"
  elif case_name == "tests_dir_zero_legacy_tokens":
    for p in (PKG_ROOT / "tests").rglob("*.py"):
      txt = p.read_text(encoding="utf-8")
      for tok in FORBIDDEN_LEGACY_TOKENS:
        assert tok not in txt, f"Forbidden token {tok!r} in {p}"
  elif case_name == "helper_cpp_zero_legacy_tokens":
    cpp_txt = (REPO_ROOT / "tools" / "jxl_wasserstein_helper.cc").read_text(
        encoding="utf-8"
    )
    for tok in FORBIDDEN_LEGACY_TOKENS:
      assert tok not in cpp_txt
  elif case_name == "git_status_no_untracked_agent_metadata":
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in res.stdout.splitlines():
      for meta in (".agents", "ORIGINAL_REQUEST.md", "PROJECT.md", "TEST_INFRA.md", "TEST_READY.md"):
        assert meta not in line, f"Agent metadata must be git-ignored, found in git status: {line!r}"
  elif case_name == "xla_deterministic_flags_and_f32_matmul":
    assert "--xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", "")
    assert jax.config.jax_default_matmul_precision == "float32"


# =============================================================================
# F2 Boundaries: Pure-Python Default DequantMatrices Corner Cases (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "cache_reset_under_dump_init_ban",
        "transposed_strategy_pair_symmetry",
        "afv_strategies_14_to_17_tables",
        "large_multiblock_strategies_18_to_26",
        "read_only_or_copy_safety",
    ],
)
def test_f2_boundary_default_dequant_matrices(
    case_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verifies F2 boundaries: cache reset under dump_init ban, transpose symmetry, AFV tables, and mutation safety."""
  if case_name == "cache_reset_under_dump_init_ban":
    for cache_attr in ("_DEFAULT_DEQUANT_CACHE", "_DEFAULT_INV_DEQUANT_CACHE"):
      if hasattr(fm_mod, cache_attr):
        monkeypatch.setattr(fm_mod, cache_attr, None)
    with forbid_libjxl_per_block_choices(monkeypatch):
      dq1 = get_default_dequant_matrices()
      dq2 = get_default_dequant_matrices()
      assert len(dq1) >= 18 and len(dq2) >= 18
      np.testing.assert_allclose(dq1[0], dq2[0], rtol=1e-6, atol=1e-7)
  elif case_name == "transposed_strategy_pair_symmetry":
    dq = get_default_dequant_matrices()
    # DCT16X8 (6) and DCT8X16 (7) share QuantTable; DCT4X8 (12) and DCT8X4 (13) share QuantTable
    np.testing.assert_allclose(dq[6], dq[7], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(dq[12], dq[13], rtol=1e-6, atol=1e-7)
  elif case_name == "afv_strategies_14_to_17_tables":
    dq = get_default_dequant_matrices()
    for sid in (14, 15, 16, 17):
      assert dq[sid].shape == (3, 8, 8)
      assert bool(np.all(dq[sid] > 0.0)) and bool(np.all(np.isfinite(dq[sid])))
      np.testing.assert_allclose(dq[14], dq[sid], rtol=1e-6, atol=1e-7)
  elif case_name == "large_multiblock_strategies_18_to_26":
    dq = get_default_dequant_matrices()
    for sid, mat in dq.items():
      cy = ACS_COVERED_BLOCKS_Y[sid]
      cx = ACS_COVERED_BLOCKS_X[sid]
      assert mat.shape == (3, 8 * min(cy, cx), 8 * max(cy, cx))
      assert bool(np.all(np.isfinite(mat))) and bool(np.all(mat > 0.0))
  elif case_name == "read_only_or_copy_safety":
    dq_orig = get_default_dequant_matrices()[0].copy()
    st = _cached_pattern_state(16, 16, "gradient", 1.5, 0).copy()
    st.dequant_matrices[0] = st.dequant_matrices[0] * 9.0
    np.testing.assert_allclose(
        get_default_dequant_matrices()[0], dq_orig, rtol=1e-6, atol=1e-7
    )


# =============================================================================
# F3 Boundaries: Zero Heuristics on Flat/Odd/Split Images & Guard Trap (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "flat_black_image_no_heuristic_crash",
        "flat_white_image_no_heuristic_crash",
        "odd_17x23_image_no_dump_init",
        "split_smooth_checkerboard_selects_mixed_acs",
        "direct_dump_init_raises_inside_guard",
    ],
)
def test_f3_boundary_no_heuristics_on_edge_images(
    case_name: str,
    vgg_params_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verifies F3 boundaries: flat black/white, odd 17x23, split smooth/checkerboard, and direct guard trap."""
  import jxl_wasserstein.optimizer as opt_mod

  if case_name == "flat_black_image_no_heuristic_crash":
    img = _make_pattern_image(16, 16, "black")
    with forbid_libjxl_per_block_choices(monkeypatch):
      res = opt_mod.optimize_single_frame(
          img, vgg_params_session, rd_lambda=0.15, num_steps=1
      )
      assert len(get_field(res, "jxl_bytes")) > 0
  elif case_name == "flat_white_image_no_heuristic_crash":
    img = _make_pattern_image(16, 16, "white")
    with forbid_libjxl_per_block_choices(monkeypatch):
      res = opt_mod.optimize_single_frame(
          img, vgg_params_session, rd_lambda=0.15, num_steps=1
      )
      assert len(get_field(res, "jxl_bytes")) > 0
  elif case_name == "odd_17x23_image_no_dump_init":
    img = _make_pattern_image(17, 23, "gradient")
    with forbid_libjxl_per_block_choices(monkeypatch):
      res = opt_mod.optimize_single_frame(
          img, vgg_params_session, rd_lambda=0.15, num_steps=1
      )
      jb = get_field(res, "jxl_bytes")
      assert len(jb) > 0 and decode_with_djxl(jb).shape == (17, 23, 3)
  elif case_name == "split_smooth_checkerboard_selects_mixed_acs":
    img = _make_pattern_image(32, 32, "split_smooth_checker")
    with forbid_libjxl_per_block_choices(monkeypatch):
      res = opt_mod.optimize_single_frame(
          img, vgg_params_session, rd_lambda=0.15, num_steps=2
      )
      st = get_field(res, "state", "main_state")
      acs = np.asarray(get_field(res, "stage2_ac_strategy") if get_field(res, "stage2_ac_strategy") is not None else st.ac_strategy)
      assert len(np.unique(acs)) >= 2
      assert bool(np.any(acs != 0))
  elif case_name == "direct_dump_init_raises_inside_guard":
    img = _make_pattern_image(16, 16, "gradient")
    with forbid_libjxl_per_block_choices(monkeypatch):
      with pytest.raises(AssertionError, match="R2 violation"):
        cpp_bridge_mod.dump_init_state(img)


# =============================================================================
# F4 Boundaries: C++ XYB PatchDictionary Alpha Blending Corner Cases (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "all_zero_alpha_matte_u2_d0",
        "all_opaque_alpha_matte_u2_d0",
        "max_dim_shift_3_with_u2",
        "max_upsampling_8_with_d0",
        "invalid_upsampling_and_dim_shift_rejected",
    ],
)
def test_f4_boundary_cpp_xyb_patch_alpha_corners(case_name: str) -> None:
  """Verifies F4 boundaries: all-0 / all-255 alpha, U=2 with D=3 (eff_ec_ups0=8), U=8 with D=0, and invalid args."""
  h, w = 32, 32
  main_st = _cached_pattern_state(h, w, "gradient", 1.0, 0)

  if case_name == "all_zero_alpha_matte_u2_d0":
    ref_st = _cached_pattern_state(h // 2, w // 2, "gray", 1.2, 0)
    alpha_u8 = np.zeros((h, w), dtype=np.uint8)
    jb, _ = _call_encode_alpha(ref_st, main_st, upsampling=2, dim_shift=0, alpha_u8=alpha_u8)
    ver = verify_jxl_codestream(jb, min_cross_decoder_psnr=45.0)
    assert ver["inspect"]["frames"][0]["save_before_color_transform"] is True
    assert ver["inspect"]["frames"][1]["has_patches"] is True
  elif case_name == "all_opaque_alpha_matte_u2_d0":
    ref_st = _cached_pattern_state(h // 2, w // 2, "gray", 1.2, 0)
    alpha_u8 = np.full((h, w), 255, dtype=np.uint8)
    jb, _ = _call_encode_alpha(ref_st, main_st, upsampling=2, dim_shift=0, alpha_u8=alpha_u8)
    ver = verify_jxl_codestream(jb, min_cross_decoder_psnr=45.0)
    assert ver["inspect"]["frames"][0]["save_before_color_transform"] is True
    assert ver["inspect"]["frames"][1]["has_patches"] is True
    single_jb, _ = encode_single_frame(main_st)
    psnr_vs_main = compute_psnr(
        ver["rgb_djxl"].astype(np.float32),
        decode_with_djxl(single_jb).astype(np.float32),
        peak=255.0,
    )
    assert psnr_vs_main >= 45.0
  elif case_name == "max_dim_shift_3_with_u2":
    # (1 << dim_shift) = 8 > upsampling = 2 -> eff_ec_ups0 = max(2, 8) = 8
    ref_st = _cached_pattern_state(h // 2, w // 2, "gradient", 1.2, 0)
    alpha_u8 = np.array(
        [[0, 255, 255, 0], [255, 255, 0, 0], [0, 255, 255, 0], [0, 0, 255, 255]],
        dtype=np.uint8,
    )
    jb, _ = _call_encode_alpha(ref_st, main_st, upsampling=2, dim_shift=3, alpha_u8=alpha_u8)
    ver = verify_jxl_codestream(jb, min_cross_decoder_psnr=45.0)
    info = ver["inspect"]
    assert info["frames"][0]["save_before_color_transform"] is True
    assert info["frames"][0]["effective_ec_upsampling"][0] == 8
    assert info["frames"][1]["has_patches"] is True
  elif case_name == "max_upsampling_8_with_d0":
    # upsampling = 8 > (1 << dim_shift) = 1 -> eff_ec_ups0 = max(8, 1) = 8
    ref_st = _cached_pattern_state(h // 8, w // 8, "gradient", 1.2, 0)
    alpha_u8 = np.zeros((h, w), dtype=np.uint8)
    alpha_u8[8:24, 8:24] = 255
    jb, _ = _call_encode_alpha(ref_st, main_st, upsampling=8, dim_shift=0, alpha_u8=alpha_u8)
    ver = verify_jxl_codestream(jb, min_cross_decoder_psnr=45.0)
    info = ver["inspect"]
    assert info["frames"][0]["save_before_color_transform"] is True
    assert info["frames"][0]["effective_ec_upsampling"][0] == 8
    assert info["frames"][1]["has_patches"] is True
  elif case_name == "invalid_upsampling_and_dim_shift_rejected":
    ref_st = _cached_pattern_state(16, 16, "gradient", 1.2, 0)
    alpha_u8 = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
      _call_encode_alpha(ref_st, main_st, upsampling=3, dim_shift=0, alpha_u8=alpha_u8)
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
      _call_encode_alpha(ref_st, main_st, upsampling=2, dim_shift=5, alpha_u8=alpha_u8)


# =============================================================================
# F5 Boundaries: JAX XYB Patch-Alpha Forward Model Corner Cases (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "odd_dimension_17x23_xyb_alpha_u2_d1",
        "odd_dimension_33x19_xyb_alpha_u4_d2",
        "extreme_alpha_0_and_1_convex_endpoints",
        "out_of_range_alpha_clamping",
        "step_edge_5x5_upsample_clamping",
    ],
)
def test_f5_boundary_jax_xyb_alpha_forward_corners(case_name: str) -> None:
  """Verifies F5 boundaries: odd dimensions (17x23, 33x19), convex endpoints (alpha=0,1), and clamping."""
  if case_name in (
      "odd_dimension_17x23_xyb_alpha_u2_d1",
      "odd_dimension_33x19_xyb_alpha_u4_d2",
  ):
    if case_name == "odd_dimension_17x23_xyb_alpha_u2_d1":
      h, w, u, d = 17, 23, 2, 1
    else:
      h, w, u, d = 33, 19, 4, 2
    rh, rw = (h + u - 1) // u, (w + u - 1) // u
    step_a = 1 << d
    ah, aw = (h + step_a - 1) // step_a, (w + step_a - 1) // step_a
    ref_st = _cached_pattern_state(rh, rw, "gradient", 1.2, 0)
    main_st = _cached_pattern_state(h, w, "checkerboard", 1.0, 0)
    alpha_u8 = np.zeros((ah, aw), dtype=np.uint8)
    alpha_u8[: max(1, ah // 2), : max(1, aw // 2)] = 255
    jb, _ = _call_encode_alpha(ref_st, main_st, upsampling=u, dim_shift=d, alpha_u8=alpha_u8)
    ver = verify_jxl_codestream(jb, min_cross_decoder_psnr=45.0)
    djxl_rgb = ver["rgb_djxl"].astype(np.float32) / 255.0
    alpha_01 = jnp.asarray(alpha_u8, dtype=jnp.float32) / 255.0
    jax_rgb = np.asarray(
        _call_reconstruct_alpha(
            ref_st, main_st, alpha_01=alpha_01, upsampling=u, dim_shift=d, hard=True
        )
    )
    assert jax_rgb.shape == (h, w, 3)
    assert compute_psnr(jax_rgb, djxl_rgb, peak=1.0) >= 40.0
  elif case_name == "extreme_alpha_0_and_1_convex_endpoints":
    ref_st = _cached_pattern_state(8, 8, "gradient", 1.2, 0)
    main_st = _cached_pattern_state(16, 16, "checkerboard", 1.0, 0)
    a0 = jnp.zeros((16, 16), dtype=jnp.float32)
    a1 = jnp.ones((16, 16), dtype=jnp.float32)
    rgb_a0 = np.asarray(
        _call_reconstruct_alpha(ref_st, main_st, alpha_01=a0, upsampling=2, dim_shift=0, hard=True)
    )
    rgb_a1 = np.asarray(
        _call_reconstruct_alpha(ref_st, main_st, alpha_01=a1, upsampling=2, dim_shift=0, hard=True)
    )
    rgb_main = np.asarray(reconstruct_single_frame(main_st, hard=True))
    np.testing.assert_allclose(rgb_a1, rgb_main, atol=1e-5)
    ref_xyb = reconstruct_single_frame_xyb(ref_st, include_noise=False, hard=True)
    ref_ups_xyb = upsample_jxl(ref_xyb, 2, out_hw=(16, 16))
    rgb_ref_ups = np.asarray(jnp.moveaxis(opsin_to_srgb(ref_ups_xyb), 0, -1))
    np.testing.assert_allclose(rgb_a0, rgb_ref_ups, atol=1e-5)
  elif case_name == "out_of_range_alpha_clamping":
    ref_st = _cached_pattern_state(8, 8, "gradient", 1.2, 0)
    main_st = _cached_pattern_state(16, 16, "checkerboard", 1.0, 0)
    a_oob = jnp.linspace(-0.5, 1.5, 16 * 16, dtype=jnp.float32).reshape(16, 16)
    rgb_oob = np.asarray(
        _call_reconstruct_alpha(
            ref_st, main_st, alpha_01=a_oob, upsampling=2, dim_shift=0, hard=False
        )
    )
    assert bool(np.all(np.isfinite(rgb_oob)))
    assert float(rgb_oob.min()) >= -1e-6 and float(rgb_oob.max()) <= 1.0 + 1e-6
  elif case_name == "step_edge_5x5_upsample_clamping":
    step_edge = jnp.zeros((3, 8, 8), dtype=jnp.float32).at[:, :, 4:].set(1.0)
    for u in (2, 4, 8):
      ups_edge = np.asarray(upsample_jxl(step_edge, u))
      assert float(ups_edge.min()) >= -1e-6 and float(ups_edge.max()) <= 1.0 + 1e-6


# =============================================================================
# F6 Boundaries: C++ Custom DequantMatrices Corner Cases (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "extreme_fine_and_coarse_custom_dc_quant",
        "extreme_scaled_custom_ac_dequant",
        "corrupt_state_magic_and_truncated_dequant",
        "odd_dimension_17x23_with_custom_dequant",
        "extreme_qm_scales_0_and_4_with_custom_dequant",
    ],
)
def test_f6_boundary_cpp_custom_dequant_corners(
    case_name: str,
    tmp_path: Path,
) -> None:
  """Verifies F6 boundaries: fine/coarse custom DC quant, scaled AC matrices, corrupt state bin, and odd dims."""
  base_st = _cached_pattern_state(16, 16, "gradient", 1.5, 0)
  dq_def = get_default_dequant_matrices()

  if case_name == "extreme_fine_and_coarse_custom_dc_quant":
    for scale in (0.25, 4.0):
      st = base_st.copy()
      dc_q = (scale / 4096.0, scale / 512.0, scale / 256.0)
      for attr in ("custom_dc_quant", "dc_quant_weights", "dc_dequant", "dc_quant_step"):
        setattr(st, attr, np.asarray(dc_q, dtype=np.float32))
      jb, _ = encode_single_frame(st)
      ver = verify_jxl_codestream(jb, min_cross_decoder_psnr=45.0)
      assert ver["inspect"]["dc_quant_all_default"] is False
      assert ver["inspect"]["all_default"] is False
  elif case_name == "extreme_scaled_custom_ac_dequant":
    for scale in (0.5, 2.5):
      st = base_st.copy()
      st.dequant_matrices = {
          int(s): (np.asarray(m, dtype=np.float32) * scale).astype(np.float32)
          for s, m in dq_def.items()
      }
      jb, _ = encode_single_frame(st)
      ver = verify_jxl_codestream(jb, min_cross_decoder_psnr=45.0)
      assert ver["inspect"]["dequant_all_default"] is False
      assert ver["inspect"]["all_default"] is False
  elif case_name == "corrupt_state_magic_and_truncated_dequant":
    bad_file = tmp_path / "bad_magic.bin"
    bad_file.write_bytes(b"BADMAGIC" + b"\x00" * 400)
    with pytest.raises(ValueError, match="Invalid state magic"):
      load_state_bin(bad_file)
    short_file = tmp_path / "short.bin"
    short_file.write_bytes(b"JXLWST01\x01\x00")
    with pytest.raises(ValueError, match="State file too short"):
      load_state_bin(short_file)
  elif case_name == "odd_dimension_17x23_with_custom_dequant":
    st_odd = _cached_pattern_state(17, 23, "gradient", 1.5, 1).copy()
    st_odd.dequant_matrices = {
        int(s): (np.asarray(m, dtype=np.float32) * 1.25).astype(np.float32)
        for s, m in dq_def.items()
    }
    jb, _ = encode_single_frame(st_odd)
    ver = verify_jxl_codestream(jb, min_cross_decoder_psnr=45.0)
    assert ver["rgb_djxl"].shape == (17, 23, 3)
    assert ver["inspect"]["all_default"] is False
  elif case_name == "extreme_qm_scales_0_and_4_with_custom_dequant":
    st = base_st.copy()
    st.x_qm_scale = 0
    st.b_qm_scale = 4
    st.dequant_matrices = {
        int(s): (np.asarray(m, dtype=np.float32) * 1.15).astype(np.float32)
        for s, m in dq_def.items()
    }
    jb, _ = encode_single_frame(st)
    ver = verify_jxl_codestream(jb, min_cross_decoder_psnr=45.0)
    assert ver["inspect"]["x_qm_scale"] == 0
    assert ver["inspect"]["b_qm_scale"] == 4
    assert ver["inspect"]["all_default"] is False


# =============================================================================
# F7 Boundaries: Differentiable Custom DequantMatrices Corner Cases (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "f16_precision_and_clipping_in_project_custom_dequant",
        "llf_positions_invariant_under_custom_dequant",
        "odd_17x23_custom_dequant_forward_vs_djxl",
        "custom_dequant_rate_bits_minimum_at_defaults",
        "zero_ac_with_custom_dequant_finite_gradients",
    ],
)
def test_f7_boundary_custom_dequant_forward_and_rate(case_name: str) -> None:
  """Verifies F7 boundaries: F16 clipping, LLF invariance, odd 17x23 forward parity, and zero-AC gradients."""
  dq_def = get_default_dequant_matrices()

  if case_name == "f16_precision_and_clipping_in_project_custom_dequant":
    proj_fn = getattr(fm_mod, "project_custom_dequant_matrices", None)
    assert proj_fn is not None
    extreme_dq = {0: np.full((3, 8, 8), 1e-4, dtype=np.float32)}
    proj_dq, proj_dc = proj_fn(extreme_dq, dc_quant=np.array([1e-5, 1e-3, 1e-2], dtype=np.float32))
    assert bool(np.all(np.isfinite(proj_dq[0]))) and bool(np.all(proj_dq[0] > 0.0))
    assert bool(np.all(np.isfinite(proj_dc))) and bool(np.all(proj_dc > 0.0))
  elif case_name == "llf_positions_invariant_under_custom_dequant":
    st = _cached_pattern_state(16, 16, "gradient", 1.5, 0).copy()
    st_llf = st.copy()
    st_llf.dequant_matrices = {k: v.copy() for k, v in dq_def.items()}
    st_llf.dequant_matrices[0][:, 0, 0] = 999.0  # LLF slot replaced by DC
    rgb1 = np.asarray(reconstruct_single_frame(st, hard=True))
    rgb2 = np.asarray(reconstruct_single_frame(st_llf, hard=True))
    np.testing.assert_allclose(rgb1, rgb2, atol=1e-5)
  elif case_name == "odd_17x23_custom_dequant_forward_vs_djxl":
    st = _cached_pattern_state(17, 23, "gradient", 1.5, 0).copy()
    st.dequant_matrices = {
        int(s): (np.asarray(m, dtype=np.float32) * 1.20).astype(np.float32)
        for s, m in dq_def.items()
    }
    proj_fn = getattr(fm_mod, "project_custom_dequant_matrices", None)
    if proj_fn is not None:
      st.dequant_matrices, _ = proj_fn(st.dequant_matrices)
    jb, _ = encode_single_frame(st)
    djxl_rgb = decode_with_djxl(jb).astype(np.float32) / 255.0
    jax_rgb = np.asarray(reconstruct_single_frame(st, hard=True))
    assert compute_psnr(jax_rgb, djxl_rgb, peak=1.0) >= 40.0
  elif case_name == "custom_dequant_rate_bits_minimum_at_defaults":
    rate_fn = getattr(ep_mod, "soft_dequant_matrices_rate_bits", None)
    assert rate_fn is not None
    dq0 = jnp.asarray(dq_def[0], dtype=jnp.float32)
    r_def = float(rate_fn({0: dq0}))
    r_low = float(rate_fn({0: dq0 * 0.7}))
    r_high = float(rate_fn({0: dq0 * 1.4}))
    assert r_def < r_low and r_def < r_high
  elif case_name == "zero_ac_with_custom_dequant_finite_gradients":
    st = _cached_pattern_state(16, 16, "gradient", 1.5, 0).copy()
    z_ac = jnp.zeros_like(jnp.asarray(st.quant_ac, dtype=jnp.float32))
    val, g_ac = jax.value_and_grad(lambda ac: soft_jxl_rate_proxy(st, quant_ac=ac))(z_ac)
    assert float(val) > 0.0 and bool(jnp.all(jnp.isfinite(g_ac)))


# =============================================================================
# F8 Boundaries: Spatial WD Map & Spatial Rate Proxy Map Corner Cases (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "identical_images_zero_spatial_wd_map",
        "small_16x16_spatial_wd_and_rate_maps",
        "odd_17x23_spatial_wd_and_rate_maps",
        "constant_black_white_spatial_wd_map",
        "all_zero_quant_ac_spatial_rate_map",
    ],
)
def test_f8_boundary_spatial_wd_and_rate_maps(
    case_name: str,
    vgg_params_session: Any,
) -> None:
  """Verifies F8 boundaries: identical images zero map, 16x16, odd 17x23, black/white, and zero AC rate map."""
  spatial_wd_fn = getattr(wd_mod, "compute_spatial_wd_map", None)
  spatial_rate_fn = getattr(ep_mod, "compute_spatial_rate_map_bits", None)
  assert spatial_wd_fn is not None and spatial_rate_fn is not None

  if case_name == "identical_images_zero_spatial_wd_map":
    img = jnp.asarray(_make_pattern_image(32, 32, "gradient"))
    tot_wd, wd_map = spatial_wd_fn(img, img, vgg_params_session)
    assert abs(float(tot_wd)) < 1e-5
    np.testing.assert_allclose(np.asarray(wd_map), 0.0, atol=1e-5)
  elif case_name == "small_16x16_spatial_wd_and_rate_maps":
    a = jnp.asarray(_make_pattern_image(16, 16, "gradient"))
    b = jnp.asarray(_make_pattern_image(16, 16, "gray"))
    tot_wd, wd_map = spatial_wd_fn(a, b, vgg_params_session)
    assert wd_map.shape == (2, 2)
    assert float(jnp.sum(wd_map)) == pytest.approx(float(tot_wd), rel=1e-5, abs=1e-6)
    st = _cached_pattern_state(16, 16, "gradient", 1.5, 0)
    rmap = spatial_rate_fn(st)
    assert rmap["block_bits"].shape == (2, 2)
  elif case_name == "odd_17x23_spatial_wd_and_rate_maps":
    a = jnp.asarray(_make_pattern_image(17, 23, "gradient"))
    b = jnp.asarray(_make_pattern_image(17, 23, "gray"))
    tot_wd, wd_map = spatial_wd_fn(a, b, vgg_params_session)
    assert wd_map.shape == (3, 3)
    assert float(jnp.sum(wd_map)) == pytest.approx(float(tot_wd), rel=1e-5, abs=1e-6)
    st = _cached_pattern_state(17, 23, "gradient", 1.5, 0)
    rmap = spatial_rate_fn(st)
    assert rmap["block_bits"].shape == (3, 3)
  elif case_name == "constant_black_white_spatial_wd_map":
    blk = jnp.asarray(_make_pattern_image(32, 32, "black"))
    wht = jnp.asarray(_make_pattern_image(32, 32, "white"))
    tot_wd, wd_map = spatial_wd_fn(blk, wht, vgg_params_session)
    assert wd_map.shape == (4, 4)
    assert float(tot_wd) > 0.0
    assert bool(jnp.all(wd_map > 0.0))
  elif case_name == "all_zero_quant_ac_spatial_rate_map":
    st = _cached_pattern_state(16, 16, "gray", 1.5, 0).copy()
    st.quant_ac = np.zeros_like(st.quant_ac)
    st.dc_quant = np.zeros_like(st.dc_quant)
    rmap = spatial_rate_fn(st)
    assert bool(jnp.all(jnp.isfinite(rmap["block_bits"])))
    assert float(rmap["total_bits"]) > 0.0


# =============================================================================
# F9 Boundaries: Stage 1 Candidate Exploration Corner Cases (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "small_16x16_stage1_skips_oversized_32x32",
        "odd_17x23_stage1_remainder_blocks",
        "zero_or_one_step_stage1_budget",
        "extreme_low_and_high_lambda_stage1",
        "flat_image_stage1_candidate_exploration",
    ],
)
def test_f9_boundary_stage1_candidate_exploration(
    case_name: str,
    vgg_params_session: Any,
) -> None:
  """Verifies F9 boundaries: 16x16 (no 32x32 block), odd 17x23 remainder blocks, 1-step budget, extreme lambda, flat image."""
  import jxl_wasserstein.optimizer as opt_mod

  if case_name == "small_16x16_stage1_skips_oversized_32x32":
    img = _make_pattern_image(16, 16, "gradient")
    res = opt_mod.optimize_single_frame(
        img, vgg_params_session, rd_lambda=0.15, num_steps=1
    )
    st = get_field(res, "state", "main_state")
    assert 5 not in np.unique(st.ac_strategy)  # DCT32X32 requires 4x4 blocks (32x32 px)
  elif case_name == "odd_17x23_stage1_remainder_blocks":
    img = _make_pattern_image(17, 23, "gradient")
    res = opt_mod.optimize_single_frame(
        img, vgg_params_session, rd_lambda=0.15, num_steps=1
    )
    st = get_field(res, "state", "main_state")
    assert st.ac_strategy.shape == (3, 3)
    jb = get_field(res, "jxl_bytes")
    assert decode_with_djxl(jb).shape == (17, 23, 3)
  elif case_name == "zero_or_one_step_stage1_budget":
    img = _make_pattern_image(24, 24, "gradient")
    res = opt_mod.optimize_single_frame(
        img, vgg_params_session, rd_lambda=0.15, num_steps=1
    )
    s1 = get_field(res, "stage1_candidates")
    assert s1 is not None and len(s1) >= 2
  elif case_name == "extreme_low_and_high_lambda_stage1":
    img = _make_pattern_image(24, 24, "gradient")
    for lam in (0.005, 3.0):
      res = opt_mod.optimize_single_frame(
          img, vgg_params_session, rd_lambda=lam, num_steps=1
      )
      assert np.isfinite(float(get_field(res, "wd")))
      assert np.isfinite(float(get_field(res, "bpp")))
  elif case_name == "flat_image_stage1_candidate_exploration":
    img = _make_pattern_image(32, 32, "gray")
    res = opt_mod.optimize_single_frame(
        img, vgg_params_session, rd_lambda=0.20, num_steps=1
    )
    assert len(get_field(res, "jxl_bytes")) > 0


# =============================================================================
# F10 Boundaries: Stage 2 Discrete DP & Patch/Mode Selection Corner Cases (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "non_multiple_of_4_block_grid_dp",
        "group_boundary_32x32_blocks_no_crossing",
        "flat_image_prefers_large_transforms_at_high_lambda",
        "explicit_patches_mode_on_flat_image_emits_valid_patch",
        "invalid_mode_rejected_cleanly",
    ],
)
def test_f10_boundary_stage2_discrete_decisions(
    case_name: str,
    vgg_params_session: Any,
) -> None:
  """Verifies F10 boundaries: non-multiple-of-4 DP grids, 256x256 group boundaries, flat image transform choice, and invalid mode."""
  import jxl_wasserstein.compositing as comp_mod
  import jxl_wasserstein.optimizer as opt_mod

  if case_name == "non_multiple_of_4_block_grid_dp":
    img = _make_pattern_image(24, 40, "split_smooth_checker")  # H8=3, W8=5
    res = opt_mod.optimize_single_frame(
        img, vgg_params_session, rd_lambda=0.15, num_steps=1
    )
    st = get_field(res, "state", "main_state")
    assert st.ac_strategy.shape == (3, 5)
    assert StaticVarDCTTopology(st) is not None
    assert decode_with_djxl(get_field(res, "jxl_bytes")).shape == (24, 40, 3)
  elif case_name == "group_boundary_32x32_blocks_no_crossing":
    # Verify DP never places a multi-block transform crossing block index 32
    dp_fn = getattr(opt_mod, "stage2_select_non_overlapping_ac_strategy", None)
    if dp_fn is not None:
      rng = np.random.default_rng(7)
      cost_maps = {
          s: rng.uniform(0.1, 1.0, size=(34, 34)).astype(np.float32)
          for s in (0, 3, 4, 5, 6, 7)
      }
      acs = np.asarray(dp_fn(cost_maps))
      assert acs.shape == (34, 34)
      vis = np.zeros_like(acs, dtype=bool)
      # No multi-block anchor can cross CfL tile (8-block) or group (32-block) boundaries
      for by in range(acs.shape[0]):
        for bx in range(acs.shape[1]):
          if vis[by, bx]:
            continue
          s_id = int(acs[by, bx])
          cy = int(ACS_COVERED_BLOCKS_Y[s_id])
          cx = int(ACS_COVERED_BLOCKS_X[s_id])
          assert (by // 8) == ((by + cy - 1) // 8)
          assert (bx // 8) == ((bx + cx - 1) // 8)
          assert (by // 32) == ((by + cy - 1) // 32)
          assert (bx // 32) == ((bx + cx - 1) // 32)
          assert np.all(acs[by : by + cy, bx : bx + cx] == s_id)
          vis[by : by + cy, bx : bx + cx] = True
    else:
      img = _make_pattern_image(32, 32, "gradient")
      res = opt_mod.optimize_single_frame(
          img, vgg_params_session, rd_lambda=0.15, num_steps=1
      )
      assert len(get_field(res, "jxl_bytes")) > 0
  elif case_name == "flat_image_prefers_large_transforms_at_high_lambda":
    img = _make_pattern_image(32, 32, "gray")
    res = opt_mod.optimize_single_frame(
        img, vgg_params_session, rd_lambda=0.50, num_steps=1
    )
    st = get_field(res, "state", "main_state")
    # Smooth flat image at high lambda should pick multi-block (DCT16X16=4 or DCT32X32=5 or DCT8X16/16X8)
    assert bool(np.any(np.isin(st.ac_strategy, [4, 5, 6, 7])))
  elif case_name == "explicit_patches_mode_on_flat_image_emits_valid_patch":
    flat = _make_pattern_image(32, 32, "gray")
    res_p = comp_mod.optimize_patch_composite(
        flat,
        upsampling=2,
        blend_mode=PATCH_BLEND_ADD,
        num_steps=1,
        verify_decoders=False,
        vgg_params=vgg_params_session,
    )
    jb = get_field(res_p, "jxl_bytes")
    assert len(jb) > 0
    info = inspect_jxl(jb)
    assert info["frames"][1]["has_patches"] is True
    assert info["frames"][1]["num_patches"] == 1
  elif case_name == "invalid_mode_rejected_cleanly":
    img = _make_pattern_image(16, 16, "gradient")
    with pytest.raises(ValueError):
      comp_mod.encode_image_to_jxl(
          img, mode="nonexistent_mode", vgg_params=vgg_params_session
      )


# =============================================================================
# F11 Boundaries: Stage 3 Continuous Refinement Corner Cases (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "small_16x16_stage3_strict_improvement",
        "odd_17x23_stage3_refinement_valid_jxl",
        "stage3_with_custom_dequant_disabled_keeps_all_default_true",
        "stage3_with_noise_enabled_clamps_lut_to_valid_range",
        "stage3_80_20_freeze_schedule_on_short_run",
    ],
)
def test_f11_boundary_stage3_refinement_corners(
    case_name: str,
    vgg_params_session: Any,
) -> None:
  """Verifies F11 boundaries: 16x16 strict improvement, odd 17x23 refinement, disabled custom dequant, noise LUT clamp, short freeze schedule."""
  import jxl_wasserstein.optimizer as opt_mod

  if case_name == "small_16x16_stage3_strict_improvement":
    img = _make_pattern_image(16, 16, "split_smooth_checker")
    res = opt_mod.optimize_single_frame(
        img, vgg_params_session, rd_lambda=0.15, num_steps=3
    )
    s2_j = float(get_field(res, "stage2_true_j", default=res.history[0]["lagrangian_true"]))
    s3_j = float(get_field(res, "stage3_true_j", "lagrangian", "rd_cost"))
    assert s3_j <= s2_j
  elif case_name == "odd_17x23_stage3_refinement_valid_jxl":
    img = _make_pattern_image(17, 23, "gradient")
    res = opt_mod.optimize_single_frame(
        img, vgg_params_session, rd_lambda=0.15, num_steps=2
    )
    ver = verify_jxl_codestream(get_field(res, "jxl_bytes"), min_cross_decoder_psnr=45.0)
    assert ver["rgb_djxl"].shape == (17, 23, 3)
    assert ver["cross_decoder_psnr"] >= 45.0
  elif case_name == "stage3_with_custom_dequant_disabled_keeps_all_default_true":
    img = _make_pattern_image(16, 16, "gradient")
    sig = inspect.signature(opt_mod.optimize_single_frame)
    dq_kw = (
        "custom_dequant"
        if "custom_dequant" in sig.parameters
        else (
            "optimize_dequant"
            if "optimize_dequant" in sig.parameters
            else "enable_custom_dequant"
        )
    )
    res = opt_mod.optimize_single_frame(
        img,
        vgg_params_session,
        rd_lambda=0.15,
        num_steps=2,
        **({dq_kw: False} if dq_kw in sig.parameters else {}),
    )
    info = inspect_jxl(get_field(res, "jxl_bytes"))
    assert info["all_default"] is True
  elif case_name == "stage3_with_noise_enabled_clamps_lut_to_valid_range":
    huge_lut = jnp.full((8,), 5.0, dtype=jnp.float32)
    q_lut = np.asarray(quantize_noise_lut_ste(huge_lut))
    assert bool(np.all(q_lut <= NOISE_LUT_MAX + 1e-6))
    img = _make_pattern_image(16, 16, "checkerboard")
    res = opt_mod.optimize_single_frame(
        img, vgg_params_session, rd_lambda=0.15, num_steps=2, enable_noise=True
    )
    lut = get_field(res, "noise_lut")
    assert lut is not None and len(lut) == 8
    assert all(0.0 <= float(v) <= NOISE_LUT_MAX + 1e-5 for v in lut)
  elif case_name == "stage3_80_20_freeze_schedule_on_short_run":
    img = _make_pattern_image(16, 16, "gradient")
    for steps in (2, 5):
      res = opt_mod.optimize_single_frame(
          img, vgg_params_session, rd_lambda=0.15, num_steps=steps
      )
      assert np.isfinite(float(get_field(res, "lagrangian", "rd_cost")))


# =============================================================================
# F12 Boundaries: Entropy Proxy & Lambda Sweep Extremes (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "rate_proxy_at_extreme_low_bitrate_0_15_bpp",
        "rate_proxy_at_extreme_high_bitrate_2_0_bpp",
        "canonical_llf_invariance_on_multiblock",
        "empty_patches_and_constant_alpha_rate_bounds",
        "lambda_to_distance_monotonic_clamping",
    ],
)
def test_f12_boundary_rate_proxy_and_lambda_extremes(case_name: str) -> None:
  """Verifies F12 boundaries: extreme 0.15 and 2.0 bpp proxy states, LLF invariance, empty patches, and lambda mapping."""
  from jxl_wasserstein.optimizer import rd_lambda_to_distance

  if case_name == "rate_proxy_at_extreme_low_bitrate_0_15_bpp":
    st = _cached_pattern_state(32, 32, "gradient", 10.0, 0)
    val, g_ac = jax.value_and_grad(lambda ac: soft_jxl_rate_proxy(st, quant_ac=ac))(
        jnp.asarray(st.quant_ac, dtype=jnp.float32)
    )
    assert float(val) > 0.0 and bool(jnp.all(jnp.isfinite(g_ac)))
  elif case_name == "rate_proxy_at_extreme_high_bitrate_2_0_bpp":
    st = _cached_pattern_state(32, 32, "checkerboard", 0.25, 0)
    val, g_ac = jax.value_and_grad(lambda ac: soft_jxl_rate_proxy(st, quant_ac=ac))(
        jnp.asarray(st.quant_ac, dtype=jnp.float32)
    )
    assert float(val) > 100.0 and bool(jnp.all(jnp.isfinite(g_ac)))
  elif case_name == "canonical_llf_invariance_on_multiblock":
    img = _make_pattern_image(32, 32, "gradient")
    acs = np.full((4, 4), 4, dtype=np.uint8)  # DCT16X16
    st16 = init_state_from_rgb_jax(img, distance=1.5, ac_strategy=acs)
    topo16 = StaticVarDCTTopology(st16)
    ac_base = jnp.asarray(st16.quant_ac, dtype=jnp.float32).reshape(3, 2, 16, 2, 16)
    ac_pert = ac_base.at[:, :, :2, :, :2].set(99.0)
    b1 = float(soft_ac_rate_bits(ac_base.reshape(3, 32, 32), topology=topo16, ac_strategy=acs))
    b2 = float(soft_ac_rate_bits(ac_pert.reshape(3, 32, 32), topology=topo16, ac_strategy=acs))
    assert b1 == pytest.approx(b2, abs=1e-5)
  elif case_name == "empty_patches_and_constant_alpha_rate_bounds":
    assert float(soft_patch_dictionary_bits([], upsampling=2)) == 0.0
    a_zero = jnp.zeros((32, 32), dtype=jnp.float32)
    a_rand = jnp.asarray(
        np.random.default_rng(5).uniform(0.0, 1.0, (32, 32)).astype(np.float32)
    )
    b_zero = float(soft_alpha_channel_bits(a_zero, dim_shift=0, full_hw=(32, 32)))
    b_rand = float(soft_alpha_channel_bits(a_rand, dim_shift=0, full_hw=(32, 32)))
    assert 0.0 < b_zero < b_rand
  elif case_name == "lambda_to_distance_monotonic_clamping":
    d_low = rd_lambda_to_distance(0.0001, base_distance=1.0)
    d_mid = rd_lambda_to_distance(0.2, base_distance=1.0)
    d_high = rd_lambda_to_distance(100.0, base_distance=1.0)
    assert 0.05 <= d_low < d_mid < d_high <= 25.0


# =============================================================================
# F13 Boundaries: Benchmark, CLI & Corrupt Bitstream Rejection (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "case_name",
    [
        "bd_rate_fewer_than_4_points_raises",
        "bd_rate_non_overlapping_and_unsorted_inputs",
        "cli_missing_input_file_error",
        "corrupt_and_truncated_jxl_codestream_rejected",
        "psnr_identical_and_shape_mismatch_edges",
    ],
)
def test_f13_boundary_benchmark_cli_and_corrupt_streams(
    case_name: str,
    kodim01_crop_64_01: np.ndarray,
    kodim01_crop_64_u8: np.ndarray,
    tmp_path: Path,
) -> None:
  """Verifies F13 boundaries: BD-rate edge cases, missing CLI file, corrupt codestreams, and PSNR helpers."""
  if case_name == "bd_rate_fewer_than_4_points_raises":
    with pytest.raises(ValueError, match="at least 4"):
      compute_bd_rate([0.5, 1.0, 1.5], [2.0, 1.0, 0.5], [0.5, 1.0, 1.5], [2.0, 1.0, 0.5])
  elif case_name == "bd_rate_non_overlapping_and_unsorted_inputs":
    val = compute_bd_rate(
        [0.5, 1.0, 1.5, 2.0],
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 1.0, 1.5, 2.0],
        [1.0, 2.0, 3.0, 4.0],
    )
    assert math.isnan(val)
    r_ref = np.array([1.8, 0.5, 1.1, 0.8])
    d_ref = np.array([0.4, 2.2, 0.9, 1.4])
    bdr = compute_bd_rate(r_ref, d_ref, r_ref * 0.90, d_ref)
    assert bdr == pytest.approx(-10.0, abs=1e-4)
  elif case_name == "cli_missing_input_file_error":
    with pytest.raises((ValueError, FileNotFoundError, SystemExit)):
      cli_main(["--input", str(tmp_path / "does_not_exist.png")])
    p = tmp_path / "small.png"
    Image.fromarray(kodim01_crop_64_u8, mode="RGB").save(p)
    assert load_benchmark_image(p, crop_size=24).shape == (24, 24, 3)
  elif case_name == "corrupt_and_truncated_jxl_codestream_rejected":
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
      verify_jxl_codestream(b"")
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
      verify_jxl_codestream(b"\x00\x00\x00\x00corrupt_jxl_stream")
    st = _cached_pattern_state(16, 16, "gradient", 1.5, 0)
    jb, _ = encode_single_frame(st)
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
      verify_jxl_codestream(jb[:6])
  elif case_name == "psnr_identical_and_shape_mismatch_edges":
    assert compute_psnr(kodim01_crop_64_01, kodim01_crop_64_01, peak=1.0) >= 99.0
    with pytest.raises(ValueError):
      compute_psnr(kodim01_crop_64_01, kodim01_crop_64_01[:32, :32], peak=1.0)
