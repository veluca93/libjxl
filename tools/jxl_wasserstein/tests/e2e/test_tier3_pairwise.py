"""Tier 3 E2E Test Suite: Cross-Feature Pairwise Combinations (22 test cases across F1-F13, >= 13 required)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest
from scipy.stats import spearmanr

from jxl_wasserstein.cli import main as cli_main
from jxl_wasserstein.cpp_bridge import (
    PATCH_BLEND_ADD,
    PATCH_BLEND_REPLACE,
    PatchEntry,
    compute_psnr,
    encode_alpha_composite,
    encode_patches_composite,
    encode_single_frame,
    inspect_jxl,
    verify_jxl_codestream,
)
from jxl_wasserstein.entropy_proxy import (
    soft_alpha_composite_rate_proxy,
    soft_jxl_rate_proxy,
    soft_patch_composite_rate_proxy,
)
from jxl_wasserstein.forward_model import (
    get_default_dequant_matrices,
    init_state_from_rgb_jax,
)

from conftest import (
    forbid_libjxl_per_block_choices,
    get_field,
)


def _extract_stage2_and_stage3_j(res: Any) -> tuple[float, float]:
  s2_j = get_field(res, "stage2_true_j")
  s3_j = get_field(res, "stage3_true_j")
  if s2_j is not None and s3_j is not None:
    return float(s2_j), float(s3_j)
  hist = get_field(res, "history")
  assert hist and len(hist) >= 1
  j0 = float(hist[0].get("lagrangian_true", hist[0].get("rd_loss")))
  j_best = float(get_field(res, "lagrangian", "rd_cost"))
  return j0, j_best


def _with_custom_dequant_kw(fn: Any, enable_custom_dq: bool) -> dict[str, Any]:
  sig = inspect.signature(fn)
  for kw in ("custom_dequant", "optimize_dequant", "enable_custom_dequant"):
    if kw in sig.parameters:
      return {kw: enable_custom_dq}
  return {}


# =============================================================================
# Group 1: 3-Stage Multi-Block AcStrategy (F2, F3, F8, F9, F10) x Custom DequantMatrices (F6, F7, F11)
#          x NoiseParams in Single-Frame Mode (4 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "custom_dequant,enable_noise",
    [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ],
)
def test_tier3_single_frame_3stage_multiblock_custom_dequant_noise_combo(
    custom_dequant: bool,
    enable_noise: bool,
    mixed_image_64_01: np.ndarray,
    vgg_params_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verifies pairwise combination of 3-stage multi-block AcStrategy + custom DequantMatrices + NoiseParams."""
  from jxl_wasserstein.optimizer import optimize_single_frame

  extra_kw = _with_custom_dequant_kw(optimize_single_frame, custom_dequant)
  with forbid_libjxl_per_block_choices(monkeypatch):
    res = optimize_single_frame(
        mixed_image_64_01,
        vgg_params_session,
        rd_lambda=0.15,
        num_steps=3,
        lr=0.06,
        enable_noise=enable_noise,
        **extra_kw,
    )

  st = get_field(res, "state", "main_state")
  acs = np.asarray(
      get_field(res, "stage2_ac_strategy")
      if get_field(res, "stage2_ac_strategy") is not None
      else st.ac_strategy
  )
  assert len(np.unique(acs)) >= 2
  assert bool(np.any(acs != 0))

  s2_j, s3_j = _extract_stage2_and_stage3_j(res)
  assert s3_j < s2_j, f"Expected stage3_true_j ({s3_j:.6f}) < stage2_true_j ({s2_j:.6f})"

  jxl_bytes = get_field(res, "jxl_bytes")
  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  assert ver["cross_decoder_psnr"] >= 45.0
  info = ver["inspect"]
  assert info["num_frames"] == 1
  assert info["frames"][0]["has_noise"] is enable_noise
  assert info["all_default"] is (not custom_dequant)

  if not enable_noise:
    psnr_parity = float(
        get_field(res, "psnr_jax_vs_djxl", "psnr_vs_djxl", "psnr")
    )
    assert psnr_parity >= 40.0


# =============================================================================
# Group 2: XYB PatchDictionary Extra-Alpha-Channel Blending (F4, F5) Across U in {2,4,8}
#          x D in {0,1,2,3} x BlendMode in {5 (kBlendBelow), 7 (kAlphaWeightedAddBelow)}
#          x Custom DequantMatrices x NoiseParams (6 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "upsampling,dim_shift,alpha_blend_mode,custom_dequant,enable_noise",
    [
        (2, 0, 5, True, False),
        (2, 1, 5, True, True),
        (4, 2, 5, True, False),
        (8, 3, 5, False, False),
        (2, 0, 7, True, False),
        (4, 1, 7, True, True),
    ],
)
def test_tier3_xyb_patch_alpha_upsampling_dimshift_blend_dequant_noise_combo(
    upsampling: int,
    dim_shift: int,
    alpha_blend_mode: int,
    custom_dequant: bool,
    enable_noise: bool,
    composite_smooth_sharp_64_01: np.ndarray,
    vgg_params_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verifies XYB PatchDictionary alpha blending across U, dim_shift, blend mode, custom dequant, and noise."""
  from jxl_wasserstein.compositing import optimize_alpha_composite

  sig = inspect.signature(optimize_alpha_composite)
  extra_kw = _with_custom_dequant_kw(optimize_alpha_composite, custom_dequant)
  if "alpha_blend_mode" in sig.parameters:
    extra_kw["alpha_blend_mode"] = alpha_blend_mode
  elif "blend_mode" in sig.parameters:
    extra_kw["blend_mode"] = alpha_blend_mode

  with forbid_libjxl_per_block_choices(monkeypatch):
    res = optimize_alpha_composite(
        composite_smooth_sharp_64_01,
        upsampling=upsampling,
        dim_shift=dim_shift,
        distance=0.45,
        lambda_rd=0.05,
        num_steps=2,
        enable_noise=enable_noise,
        vgg_params=vgg_params_session,
        **extra_kw,
    )

  s2_j, s3_j = _extract_stage2_and_stage3_j(res)
  assert s3_j < s2_j, f"Expected stage3_true_j ({s3_j:.6f}) < stage2_true_j ({s2_j:.6f})"

  jxl_bytes = get_field(res, "jxl_bytes")
  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  assert ver["cross_decoder_psnr"] >= 45.0

  info = ver["inspect"]
  assert info["num_frames"] == 2
  assert len(info["extra_channels"]) >= 1
  assert info["extra_channels"][0]["dim_shift"] == dim_shift
  assert info["frames"][0]["upsampling"] == upsampling
  assert info["frames"][0]["save_before_color_transform"] is True
  assert info["frames"][0]["effective_ec_upsampling"][0] == max(
      upsampling, 1 << dim_shift
  )
  assert info["frames"][1]["has_patches"] is True
  assert info["frames"][1]["num_patches"] >= 1
  assert alpha_blend_mode in info["frames"][1]["patch_blend_modes"]
  assert info["all_default"] is (not custom_dequant)

  if enable_noise:
    assert info["frames"][1]["has_noise"] is True
  else:
    djxl_rgb = ver["rgb_djxl"].astype(np.float32) / 255.0
    recon_rgb = np.asarray(get_field(res, "recon_rgb", "recon_jax_rgb"))
    assert compute_psnr(recon_rgb, djxl_rgb, peak=1.0) >= 40.0


# =============================================================================
# Group 3: Discrete XYB PatchDictionary (kReplace & kAdd) Across U in {2,4,8}
#          x 3-Stage Rate-WD Selection x Custom DequantMatrices x NoiseParams (4 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "upsampling,blend_mode,custom_dequant,enable_noise",
    [
        (2, PATCH_BLEND_REPLACE, True, False),
        (2, PATCH_BLEND_ADD, True, True),
        (4, PATCH_BLEND_REPLACE, True, False),
        (8, PATCH_BLEND_ADD, False, False),
    ],
)
def test_tier3_patch_composite_upsampling_blend_dequant_noise_combo(
    upsampling: int,
    blend_mode: int,
    custom_dequant: bool,
    enable_noise: bool,
    composite_smooth_sharp_64_01: np.ndarray,
    vgg_params_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verifies ReferenceOnly (U in {2,4,8}) + PatchDictionary (kReplace/kAdd) + custom DequantMatrices + NoiseParams."""
  from jxl_wasserstein.compositing import optimize_patch_composite

  extra_kw = _with_custom_dequant_kw(optimize_patch_composite, custom_dequant)
  with forbid_libjxl_per_block_choices(monkeypatch):
    res = optimize_patch_composite(
        composite_smooth_sharp_64_01,
        upsampling=upsampling,
        blend_mode=blend_mode,
        distance=0.45,
        lambda_rd=0.05,
        num_steps=2,
        enable_noise=enable_noise,
        vgg_params=vgg_params_session,
        **extra_kw,
    )

  s2_j, s3_j = _extract_stage2_and_stage3_j(res)
  assert s3_j < s2_j

  jxl_bytes = get_field(res, "jxl_bytes")
  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  assert ver["cross_decoder_psnr"] >= 45.0

  info = ver["inspect"]
  assert info["num_frames"] == 2
  assert info["frames"][0]["upsampling"] == upsampling
  assert info["frames"][0]["save_before_color_transform"] is True
  assert info["frames"][1]["has_patches"] is True
  assert info["frames"][1]["num_patches"] >= 1
  assert info["all_default"] is (not custom_dequant)

  if enable_noise:
    assert info["frames"][1]["has_noise"] is True
  else:
    djxl_rgb = ver["rgb_djxl"].astype(np.float32) / 255.0
    recon_rgb = np.asarray(get_field(res, "recon_rgb", "recon_jax_rgb"))
    assert compute_psnr(recon_rgb, djxl_rgb, peak=1.0) >= 40.0


# =============================================================================
# Group 4: Differentiable Entropy Proxy Spearman rho >= 0.95 Across Modes x Custom DequantMatrices (3 test cases)
# =============================================================================


@pytest.mark.parametrize("stream_mode", ["single", "patches", "alpha"])
def test_tier3_entropy_proxy_spearman_across_modes_with_custom_dequant(
    stream_mode: str,
    kodim01_crop_64_01: np.ndarray,
) -> None:
  """Verifies Spearman rank correlation >= 0.95 across single, patches, and XYB patch-alpha with custom DequantMatrices."""
  distances = (0.35, 0.55, 0.85, 1.3, 2.0, 3.2, 5.0, 8.0)
  dq_def = get_default_dequant_matrices()
  custom_dq = {
      int(s): (np.asarray(m, dtype=np.float32) * 1.15).astype(np.float32)
      for s, m in dq_def.items()
  }
  proxy_rates: list[float] = []
  true_rates: list[float] = []

  if stream_mode == "single":
    for d in distances:
      st = init_state_from_rgb_jax(kodim01_crop_64_01, distance=d)
      st.dequant_matrices = {k: v.copy() for k, v in custom_dq.items()}
      _, stats = encode_single_frame(st)
      proxy_rates.append(float(soft_jxl_rate_proxy(st)))
      true_rates.append(float(stats["total_bits"]))
  elif stream_mode == "patches":
    ref_crop = kodim01_crop_64_01[::2, ::2]
    patches = [PatchEntry(0, 0, 0, 32, 32, 0, 0, PATCH_BLEND_REPLACE)]
    for d in distances:
      ref_st = init_state_from_rgb_jax(ref_crop, distance=d * 1.2)
      main_st = init_state_from_rgb_jax(kodim01_crop_64_01, distance=d)
      main_st.dequant_matrices = {k: v.copy() for k, v in custom_dq.items()}
      jb, _ = encode_patches_composite(
          ref_st, main_st, upsampling=2, patches=patches
      )
      p_bits = float(
          soft_patch_composite_rate_proxy(
              ref_st, main_st, patches=patches, upsampling=2
          )
      )
      proxy_rates.append(p_bits)
      true_rates.append(float(len(jb) * 8))
  else:
    ref_crop = kodim01_crop_64_01[::2, ::2]
    alpha_u8 = np.zeros((64, 64), dtype=np.uint8)
    alpha_u8[16:48, 16:48] = 255
    alpha_01 = jnp.asarray(alpha_u8, dtype=jnp.float32) / 255.0
    for d in distances:
      ref_st = init_state_from_rgb_jax(ref_crop, distance=d * 1.2)
      main_st = init_state_from_rgb_jax(kodim01_crop_64_01, distance=d)
      main_st.dequant_matrices = {k: v.copy() for k, v in custom_dq.items()}
      jb, _ = encode_alpha_composite(
          ref_st, main_st, upsampling=2, dim_shift=0, alpha_u8=alpha_u8
      )
      p_bits = float(
          soft_alpha_composite_rate_proxy(
              ref_st, main_st, alpha_01=alpha_01, upsampling=2, dim_shift=0
          )
      )
      proxy_rates.append(p_bits)
      true_rates.append(float(len(jb) * 8))

  rho = float(spearmanr(proxy_rates, true_rates).statistic)
  assert rho >= 0.95, f"Mode {stream_mode} Spearman rho={rho:.4f} < 0.95"


# =============================================================================
# Group 5: CLI End-to-End Execution Across Modes, Custom Dequant, Noise, Verify & JSON Out (5 test cases)
# =============================================================================


@pytest.mark.parametrize(
    "cli_case,extra_flags",
    [
        ("cli_single_3stage_custom_dequant", ["--mode", "single", "--distance", "2.0", "--steps", "2"]),
        (
            "cli_single_with_noise_and_custom_dequant",
            ["--mode", "single", "--distance", "2.0", "--steps", "2", "--enable-noise"],
        ),
        (
            "cli_patches_u2_rate_wd",
            ["--mode", "patches", "--upsampling", "2", "--distance", "0.8", "--steps", "2"],
        ),
        (
            "cli_alpha_u2_d1_xyb_patch_dictionary",
            ["--mode", "alpha", "--upsampling", "2", "--dim-shift", "1", "--distance", "0.8", "--steps", "2"],
        ),
        (
            "cli_auto_mode_selection",
            ["--mode", "auto", "--distance", "0.5", "--steps", "2"],
        ),
    ],
)
def test_tier3_cli_modes_custom_dequant_verify_and_json_out(
    cli_case: str,
    extra_flags: list[str],
    composite_smooth_sharp_64_u8: np.ndarray,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verifies CLI end-to-end execution across single, patches, alpha, auto, custom DequantMatrices, and noise."""
  in_png = tmp_path / f"{cli_case}_in.png"
  out_jxl = tmp_path / f"{cli_case}_out.jxl"
  json_path = tmp_path / f"{cli_case}_report.json"
  Image.fromarray(composite_smooth_sharp_64_u8, mode="RGB").save(in_png)

  argv = [
      "--input",
      str(in_png),
      "--output",
      str(out_jxl),
      "--verify",
      "--json-out",
      str(json_path),
      *extra_flags,
  ]
  with forbid_libjxl_per_block_choices(monkeypatch):
    rc = cli_main(argv)

  assert rc == 0
  assert out_jxl.is_file() and out_jxl.stat().st_size > 0
  assert json_path.is_file()

  report = json.loads(json_path.read_text(encoding="utf-8"))
  assert report["file_bytes"] == out_jxl.stat().st_size
  assert report["bpp"] > 0.0
  assert report["wd"] >= 0.0
  assert report["cross_decoder_psnr"] >= 45.0

  info = inspect_jxl(out_jxl.read_bytes())
  assert info["all_default"] is False
  if "--enable-noise" in extra_flags:
    assert info["frames"][-1]["has_noise"] is True
  if cli_case == "cli_alpha_u2_d1_xyb_patch_dictionary":
    assert info["frames"][0]["save_before_color_transform"] is True
    assert info["frames"][1]["has_patches"] is True
