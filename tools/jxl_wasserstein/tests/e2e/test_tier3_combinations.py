"""Tier 3 E2E Test Suite: Cross-Feature Pairwise Combinations (>= 15 test cases)."""

from __future__ import annotations

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
    decode_with_djxl,
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
    init_state_from_rgb_jax,
    reconstruct_alpha_composite,
    reconstruct_patch_composite,
    reconstruct_single_frame,
)

from conftest import get_field


# =============================================================================
# Group 1: Multi-Block AcStrategy + CfL + EPF + NoiseParams in Single-Frame Mode (2 tests)
# =============================================================================


@pytest.mark.parametrize("enable_noise", [False, True])
def test_tier3_single_frame_multiblock_cfl_epf_noise_combo(
    enable_noise: bool,
    mixed_image_64_01: np.ndarray,
    vgg_params_session: Any,
) -> None:
  """Verifies multi-block AcStrategy + CfL + EPF + optional NoiseParams in single-frame mode."""
  from jxl_wasserstein.optimizer import (
      optimize_single_frame,
      select_block_params_jax,
  )

  blk = select_block_params_jax(
      mixed_image_64_01, distance=1.2, allow_multiblock=True
  )
  assert len(np.unique(blk["ac_strategy"])) >= 3
  assert len(np.unique(blk["epf_sharpness"])) >= 2
  assert bool(np.any(blk["ytox_map"] != 0) or np.any(blk["ytob_map"] != 0))

  res = optimize_single_frame(
      mixed_image_64_01,
      vgg_params_session,
      rd_lambda=0.15,
      num_steps=4,
      lr=0.06,
      enable_noise=enable_noise,
      allow_multiblock=True,
      checkpoint_interval=2,
  )
  jxl_bytes = get_field(res, "jxl_bytes")
  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  assert ver["cross_decoder_psnr"] >= 45.0
  info = ver["inspect"]
  assert info["num_frames"] == 1
  assert info["frames"][0]["has_noise"] is enable_noise

  if not enable_noise:
    psnr_parity = float(
        get_field(res, "psnr_jax_vs_djxl", "psnr_vs_djxl", "psnr")
    )
    assert psnr_parity >= 40.0


# =============================================================================
# Group 2: Multi-Res ReferenceOnly (U in {2,4,8}) + XYB PatchDictionary (kReplace & kAdd)
#          + NoiseParams + Cross-Decoder Verification (6 tests)
# =============================================================================


@pytest.mark.parametrize(
    "upsampling,blend_mode,enable_noise",
    [
        (2, PATCH_BLEND_REPLACE, False),
        (2, PATCH_BLEND_ADD, True),
        (4, PATCH_BLEND_REPLACE, False),
        (4, PATCH_BLEND_ADD, False),
        (8, PATCH_BLEND_REPLACE, True),
        (8, PATCH_BLEND_ADD, False),
    ],
)
def test_tier3_patch_composite_upsampling_blend_noise_combo(
    upsampling: int,
    blend_mode: int,
    enable_noise: bool,
    composite_smooth_sharp_64_01: np.ndarray,
    vgg_params_session: Any,
) -> None:
  """Verifies ReferenceOnly (U in {2,4,8}) + PatchDictionary (kReplace/kAdd) + NoiseParams + decoders."""
  from jxl_wasserstein.compositing import optimize_patch_composite

  res = optimize_patch_composite(
      composite_smooth_sharp_64_01,
      upsampling=upsampling,
      blend_mode=blend_mode,
      distance=0.45,
      lambda_rd=0.05,
      num_steps=2,
      enable_noise=enable_noise,
      vgg_params=vgg_params_session,
  )
  jxl_bytes = get_field(res, "jxl_bytes")
  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  assert ver["cross_decoder_psnr"] >= 45.0

  info = ver["inspect"]
  assert info["num_frames"] == 2
  assert info["frames"][0]["upsampling"] == upsampling
  assert info["frames"][0]["save_before_color_transform"] is True
  assert info["frames"][1]["has_patches"] is True
  assert info["frames"][1]["num_patches"] >= 1
  if enable_noise:
    assert info["frames"][1]["has_noise"] is True
  else:
    djxl_rgb = ver["rgb_djxl"].astype(np.float32) / 255.0
    recon_rgb = np.asarray(get_field(res, "recon_rgb", "recon_jax_rgb"))
    assert compute_psnr(recon_rgb, djxl_rgb, peak=1.0) >= 40.0


# =============================================================================
# Group 3: Multi-Res ReferenceOnly (U in {2,4,8}) + ExtraChannel::kAlpha (D in {0,1,2,3})
#          + NoiseParams + Cross-Decoder Verification (6 tests)
# =============================================================================


@pytest.mark.parametrize(
    "upsampling,dim_shift,enable_noise",
    [
        (2, 0, False),
        (2, 1, True),
        (2, 2, False),
        (2, 3, False),
        (4, 1, False),
        (8, 1, True),
    ],
)
def test_tier3_alpha_composite_upsampling_dimshift_noise_combo(
    upsampling: int,
    dim_shift: int,
    enable_noise: bool,
    composite_smooth_sharp_64_01: np.ndarray,
    vgg_params_session: Any,
) -> None:
  """Verifies ReferenceOnly (U in {2,4,8}) + ExtraChannel::kAlpha (D in {0..3}) + NoiseParams + decoders."""
  from jxl_wasserstein.compositing import optimize_alpha_composite

  res = optimize_alpha_composite(
      composite_smooth_sharp_64_01,
      upsampling=upsampling,
      dim_shift=dim_shift,
      distance=0.45,
      lambda_rd=0.05,
      num_steps=2,
      enable_noise=enable_noise,
      vgg_params=vgg_params_session,
  )
  jxl_bytes = get_field(res, "jxl_bytes")
  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  assert ver["cross_decoder_psnr"] >= 45.0

  info = ver["inspect"]
  assert info["num_frames"] == 2
  assert len(info["extra_channels"]) == 1
  assert info["extra_channels"][0]["dim_shift"] == dim_shift
  assert info["frames"][0]["upsampling"] == upsampling
  assert info["frames"][0]["save_before_color_transform"] is False
  assert info["frames"][0]["ec_upsampling"][0] == max(1, upsampling >> dim_shift)
  if enable_noise:
    assert info["frames"][1]["has_noise"] is True
  else:
    djxl_rgb = ver["rgb_djxl"].astype(np.float32) / 255.0
    recon_rgb = np.asarray(get_field(res, "recon_rgb", "recon_jax_rgb"))
    assert compute_psnr(recon_rgb, djxl_rgb, peak=1.0) >= 40.0


# =============================================================================
# Group 4: Entropy Proxy Spearman rho >= 0.95 Across Single, Patch, and Alpha Bitstreams (3 tests)
# =============================================================================


@pytest.mark.parametrize("stream_mode", ["single", "patches", "alpha"])
def test_tier3_entropy_proxy_spearman_across_modes(
    stream_mode: str,
    kodim01_crop_64_01: np.ndarray,
) -> None:
  """Verifies Spearman rank correlation >= 0.95 between soft rate proxy and true libjxl bits across modes."""
  distances = (0.35, 0.55, 0.85, 1.3, 2.0, 3.2, 5.0, 8.0)
  proxy_rates: list[float] = []
  true_rates: list[float] = []

  if stream_mode == "single":
    for d in distances:
      st = init_state_from_rgb_jax(kodim01_crop_64_01, distance=d)
      _, stats = encode_single_frame(st)
      proxy_rates.append(float(soft_jxl_rate_proxy(st)))
      true_rates.append(float(stats["total_bits"]))
  elif stream_mode == "patches":
    ref_crop = kodim01_crop_64_01[::2, ::2]
    patches = [PatchEntry(0, 0, 0, 32, 32, 0, 0, PATCH_BLEND_REPLACE)]
    for d in distances:
      ref_st = init_state_from_rgb_jax(ref_crop, distance=d * 1.2)
      main_st = init_state_from_rgb_jax(kodim01_crop_64_01, distance=d)
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
# Group 5: CLI End-to-End Execution Across Modes, Noise, Verify, and JSON Out (5 tests)
# =============================================================================


@pytest.mark.parametrize(
    "cli_case,extra_flags",
    [
        ("single", ["--mode", "single", "--distance", "2.0", "--steps", "2"]),
        (
            "single_with_noise",
            [
                "--mode",
                "single",
                "--distance",
                "2.0",
                "--steps",
                "2",
                "--enable-noise",
            ],
        ),
        (
            "patches",
            [
                "--mode",
                "patches",
                "--upsampling",
                "2",
                "--distance",
                "0.8",
                "--steps",
                "2",
            ],
        ),
        (
            "alpha",
            [
                "--mode",
                "alpha",
                "--upsampling",
                "2",
                "--dim-shift",
                "1",
                "--distance",
                "0.8",
                "--steps",
                "2",
            ],
        ),
        (
            "auto",
            ["--mode", "auto", "--distance", "0.5", "--steps", "2"],
        ),
    ],
)
def test_tier3_cli_modes_verify_and_json_out(
    cli_case: str,
    extra_flags: list[str],
    composite_smooth_sharp_64_u8: np.ndarray,
    tmp_path: Path,
) -> None:
  """Verifies CLI end-to-end execution across single, patches, alpha, auto, noise, verify, and json-out."""
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
  rc = cli_main(argv)
  assert rc == 0
  assert out_jxl.is_file() and out_jxl.stat().st_size > 0
  assert json_path.is_file()

  report = json.loads(json_path.read_text(encoding="utf-8"))
  assert report["file_bytes"] == out_jxl.stat().st_size
  assert report["bpp"] > 0.0
  assert report["wd"] >= 0.0
  assert report["cross_decoder_psnr"] >= 45.0
  if "--enable-noise" in extra_flags:
    assert report["header_info"]["frames"][-1]["has_noise"] is True
