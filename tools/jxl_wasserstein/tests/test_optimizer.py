"""Tests for M4 pure-JAX VarDCT Rate-WD optimizer (`optimizer.py`).

Verifies:
1. Pure-JAX per-block discrete selection (`select_ac_strategy_jax`,
   `select_quant_and_epf_jax`, `select_cfl_jax`, `select_block_params_jax`)
   produces multi-block strategies (`DCT16X16`, `DCT32X32`, `DCT16X8`, `DCT8X16`,
   `DCT8X8`) and valid `raw_quant_field`, `epf_sharpness`, and CfL maps without
   ever calling `libjxl`/`cjxl` (`dump_init_state` monkeypatched to raise).
2. Two-stage Adam+STE optimization (`optimize_single_frame`) strictly reduces the
   Rate-WD Lagrangian `J = WD + rd_lambda * bpp` compared to step 0 and produces
   standards-compliant JXL codestreams verified by both `djxl` and `jxl_cli`.
3. Monotonic 4-point `rd_lambda` sweep produces monotonically decreasing bitrates
   (`bpp`) as `rd_lambda` increases.
4. Procedural noise synthesis optimization (`enable_noise=True`) produces a valid
   codestream with `has_noise=True` and strictly lower Wasserstein Distortion than
   `enable_noise=False` on a noisy textured test image.
5. Single-compile dynamic-scale Kodak Rate-WD initializer and optimizer
   (`init_pure_jax_wd_state`, `optimize_rate_wd_point`) strictly reduces WD vs
   pure-JAX step 0 and produces valid codestreams.
"""

from __future__ import annotations

import numpy as np
import pytest

import jxl_wasserstein.cpp_bridge as cpp_bridge
from jxl_wasserstein.cpp_bridge import (
    encode_single_frame,
    inspect_jxl,
    verify_jxl_codestream,
)
from jxl_wasserstein.forward_model import init_state_from_rgb_jax
from jxl_wasserstein.optimizer import (
    init_pure_jax_wd_state,
    optimize_rate_wd_point,
    optimize_single_frame,
    select_block_params_jax,
    stage1_explore_ac_candidates,
    stage2_select_non_overlapping_ac_strategy,
    validate_ac_strategy_tiling,
)
from jxl_wasserstein.vgg import load_vgg16_params


@pytest.fixture(scope="module")
def vgg_params():
  return load_vgg16_params()


@pytest.fixture(autouse=True)
def forbid_dump_init_state(monkeypatch):
  """Enforces Constraint R2: `dump_init_state` must NEVER be called."""

  def _forbidden(*args, **kwargs):
    raise AssertionError(
        "Constraint R2 violation: dump_init_state must NEVER be called!"
    )

  monkeypatch.setattr(cpp_bridge, "dump_init_state", _forbidden)


def _make_mixed_test_image(h: int = 64, w: int = 64) -> np.ndarray:
  """Creates a 64x64 image with smooth sky (32x32), gradient (16x16), sharp diagonal edges, and texture."""
  rng = np.random.default_rng(42)
  yy, xx = np.meshgrid(
      np.linspace(0.0, 1.0, h, dtype=np.float32),
      np.linspace(0.0, 1.0, w, dtype=np.float32),
      indexing="ij",
  )
  img = np.zeros((h, w, 3), dtype=np.float32)
  img[:, :, 0] = 0.25 + 0.10 * yy
  img[:, :, 1] = 0.45 + 0.12 * xx
  img[:, :, 2] = 0.70 - 0.08 * yy

  diag_mask = (yy - 0.5) > 0.85 * (xx - 0.05)
  img[h // 2 :, : w // 2, 0] = np.where(
      diag_mask[h // 2 :, : w // 2], 0.85, 0.15
  )
  img[h // 2 :, : w // 2, 1] = np.where(
      diag_mask[h // 2 :, : w // 2], 0.25, 0.75
  )
  img[h // 2 :, : w // 2, 2] = np.where(
      diag_mask[h // 2 :, : w // 2], 0.20, 0.80
  )

  tex = 0.08 * np.sin(2.0 * np.pi * xx * 11.0) * np.cos(
      2.0 * np.pi * yy * 9.0
  ) + rng.normal(0.0, 0.035, size=(h, w)).astype(np.float32)
  img[h // 2 :, w // 2 :, 0] = np.clip(
      0.55 + tex[h // 2 :, w // 2 :], 0.0, 1.0
  )
  img[h // 2 :, w // 2 :, 1] = np.clip(
      0.50 + 0.9 * tex[h // 2 :, w // 2 :], 0.0, 1.0
  )
  img[h // 2 :, w // 2 :, 2] = np.clip(
      0.42 + 0.7 * tex[h // 2 :, w // 2 :], 0.0, 1.0
  )
  return np.clip(img, 0.0, 1.0)


def _make_noisy_textured_image(h: int = 64, w: int = 64) -> np.ndarray:
  """Creates a fine-grained natural film-grain textured image for noise synthesis testing."""
  rng = np.random.default_rng(123)
  yy, xx = np.meshgrid(
      np.linspace(0.0, 1.0, h, dtype=np.float32),
      np.linspace(0.0, 1.0, w, dtype=np.float32),
      indexing="ij",
  )
  base_r = 0.48 + 0.18 * np.sin(2.0 * np.pi * yy * 1.5)
  base_g = 0.46 + 0.15 * np.cos(2.0 * np.pi * xx * 1.5)
  base_b = 0.42 + 0.12 * np.sin(2.0 * np.pi * (xx + yy))
  base = np.stack([base_r, base_g, base_b], axis=-1).astype(np.float32)

  grain_luma = rng.normal(0.0, 0.055, size=(h, w, 1)).astype(np.float32)
  grain_chroma = rng.normal(0.0, 0.015, size=(h, w, 3)).astype(np.float32)
  return np.clip(base + grain_luma + grain_chroma, 0.0, 1.0)


def test_pure_jax_block_params_selection_and_codestream_validity():
  """Tests pure-JAX AcStrategy, quant field, EPF sharpness, and CfL selection without libjxl."""
  img_mixed = _make_mixed_test_image(64, 64)
  blk = select_block_params_jax(img_mixed, distance=1.2, allow_multiblock=True)

  unique_acs = np.unique(blk["ac_strategy"])
  assert len(unique_acs) >= 3, f"Expected multiple AcStrategy types, got {unique_acs}"
  assert blk["raw_quant_field"].shape == (8, 8)
  assert np.all((blk["raw_quant_field"] >= 1) & (blk["raw_quant_field"] <= 256))
  assert blk["epf_sharpness"].shape == (8, 8)
  assert np.all((blk["epf_sharpness"] >= 0) & (blk["epf_sharpness"] <= 7))
  assert blk["ytox_map"].shape == (1, 1)
  assert blk["ytob_map"].shape == (1, 1)

  st0 = init_state_from_rgb_jax(
      img_mixed,
      distance=1.2,
      ac_strategy=blk["ac_strategy"],
      raw_quant_field=blk["raw_quant_field"],
      epf_sharpness=blk["epf_sharpness"],
      ytox_map=blk["ytox_map"],
      ytob_map=blk["ytob_map"],
      epf_iters=blk["epf_iters"],
  )
  jxl0, _ = encode_single_frame(st0)
  ver0 = verify_jxl_codestream(jxl0, min_cross_decoder_psnr=45.0)
  assert ver0["cross_decoder_psnr"] >= 45.0


def test_two_stage_adam_ste_reduces_lagrangian(vgg_params):
  """Tests that `optimize_single_frame` strictly reduces the Rate-WD Lagrangian J vs step 0."""
  img_mixed = _make_mixed_test_image(64, 64)
  res_opt = optimize_single_frame(
      img_mixed,
      vgg_params,
      rd_lambda=0.15,
      num_steps=10,
      lr=0.08,
      enable_noise=False,
      checkpoint_interval=2,
  )
  j0 = res_opt.history[0]["lagrangian_true"]
  assert res_opt.lagrangian < j0, (
      f"Expected strict Lagrangian reduction: {res_opt.lagrangian} vs {j0}"
  )
  assert res_opt.psnr_jax_vs_djxl >= 38.0
  ver = verify_jxl_codestream(res_opt.jxl_bytes, min_cross_decoder_psnr=45.0)
  assert ver["cross_decoder_psnr"] >= 45.0


def test_monotonic_four_point_lambda_sweep(vgg_params):
  """Tests that a 4-point sweep of `rd_lambda` yields monotonically decreasing bitrates."""
  img_mixed = _make_mixed_test_image(64, 64)
  lambdas = [0.02, 0.08, 0.32, 1.28]
  sweep_bpps = []
  for lam in lambdas:
    r_lam = optimize_single_frame(
        img_mixed,
        vgg_params,
        rd_lambda=lam,
        num_steps=8,
        lr=0.08,
        enable_noise=False,
        checkpoint_interval=4,
    )
    sweep_bpps.append(r_lam.bpp)

  for i in range(len(sweep_bpps) - 1):
    assert sweep_bpps[i] >= sweep_bpps[i + 1], (
        f"Non-monotonic bpp sweep across {lambdas}: {sweep_bpps}"
    )


def test_noise_synthesis_optimization_improves_wd_on_noisy_texture(vgg_params):
  """Tests that `enable_noise=True` sets `has_noise=True` and reduces WD on noisy textures."""
  img_noisy = _make_noisy_textured_image(64, 64)
  res_no_noise = optimize_single_frame(
      img_noisy,
      vgg_params,
      distance=3.8,
      rd_lambda=0.25,
      scale_distance_with_lambda=False,
      num_steps=8,
      lr=0.08,
      enable_noise=False,
      checkpoint_interval=4,
  )
  res_with_noise = optimize_single_frame(
      img_noisy,
      vgg_params,
      distance=3.8,
      rd_lambda=0.25,
      scale_distance_with_lambda=False,
      num_steps=8,
      lr=0.08,
      enable_noise=True,
      checkpoint_interval=4,
  )
  info_noise = inspect_jxl(res_with_noise.jxl_bytes)
  frame0_hdr = info_noise["frames"][0]
  assert frame0_hdr.get("has_noise") is True
  assert res_with_noise.noise_lut is not None and any(
      x > 0.0 for x in res_with_noise.noise_lut
  )
  assert res_with_noise.wd < res_no_noise.wd, (
      f"Expected enable_noise=True WD ({res_with_noise.wd}) < "
      f"enable_noise=False WD ({res_no_noise.wd})"
  )


def test_single_compile_kodak_rate_wd_point_optimizer(vgg_params):
  """Tests `init_pure_jax_wd_state` and `optimize_rate_wd_point` multi-transform solver."""
  img_mixed = _make_mixed_test_image(64, 64)
  st, topo, u_ac, u_dc = init_pure_jax_wd_state(img_mixed, distance=1.5)
  assert np.any(st.ac_strategy != 0)
  assert len(np.unique(st.ac_strategy)) >= 2
  assert u_ac.shape == (3, 64, 64)
  assert u_dc.shape == (3, 8, 8)

  res = optimize_rate_wd_point(
      img_mixed,
      distance=1.5,
      vgg_params=vgg_params,
      num_steps=6,
      lr_ac=0.08,
      lr_dc=0.03,
      rate_weight=0.15,
      verify_decoders=True,
  )
  assert res.lagrangian <= res.stage2_true_j
  assert res["cross_decoder_psnr"] >= 45.0
  assert len(res["jxl_bytes"]) > 0
  assert res.all_default is False


def test_three_stage_pipeline_and_custom_dequant_matrices(vgg_params):
  """Tests Stage 1 candidate exploration, Stage 2 non-overlapping DP, and Stage 3 custom dequant."""
  img_mixed = _make_mixed_test_image(64, 64)
  cands = stage1_explore_ac_candidates(
      img_mixed,
      vgg_params=vgg_params,
      distance=1.5,
      rd_lambda=0.15,
      steps=2,
      allow_multiblock=True,
  )
  assert len(cands) >= 6
  for s_id, entry in cands.items():
    assert entry["j_map"].shape == (8, 8)
    assert entry["j_score"] < entry["j_init"], (
        f"Strategy {s_id} did not reduce J in Stage 1: {entry['j_score']} vs {entry['j_init']}"
    )

  acs_grid = stage2_select_non_overlapping_ac_strategy(cands, h8=8, w8=8)
  assert validate_ac_strategy_tiling(acs_grid)
  assert np.any(acs_grid != 0)
  assert len(np.unique(acs_grid)) >= 3

  res = optimize_single_frame(
      img_mixed,
      vgg_params=vgg_params,
      rd_lambda=0.15,
      num_steps=6,
      opt_custom_dequant=True,
  )
  assert res.stage3_true_j < res.stage2_true_j
  assert res.all_default is False
  info = inspect_jxl(res.jxl_bytes)
  assert info["all_default"] is False


def test_zero_and_random_initialization_convergence(vgg_params):
  """Verifies that optimization starting from 0 or random noise converges and produces valid JXL bitstreams."""
  img = _make_mixed_test_image(64, 64)
  res_zero = optimize_single_frame(
      img,
      vgg_params=vgg_params,
      distance=1.5,
      rd_lambda=0.12,
      init_mode="zero",
      num_steps=35,
      verify_decoders=True,
  )
  assert res_zero.wd_final < res_zero.wd_init * 0.45
  assert res_zero.stage3_true_j < res_zero.stage2_true_j
  assert res_zero.psnr_jax_vs_djxl >= 40.0
  assert res_zero.cross_decoder_psnr >= 45.0

  res_rand = optimize_single_frame(
      img,
      vgg_params=vgg_params,
      distance=1.5,
      rd_lambda=0.12,
      init_mode="random",
      seed=7,
      num_steps=35,
      verify_decoders=True,
  )
  assert res_rand.wd_final < res_rand.wd_init * 0.45
  assert res_rand.stage3_true_j < res_rand.stage2_true_j
  assert res_rand.psnr_jax_vs_djxl >= 40.0
  assert res_rand.cross_decoder_psnr >= 45.0



