"""Regression tests verifying all 9 fixes for 8-bit SDR reconstruction & R-D pipeline."""

from __future__ import annotations

import gc
import inspect
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import pytest

from jxl_wasserstein import benchmark, cli, compositing, entropy_proxy, forward_model, optimizer, wasserstein
from jxl_wasserstein.cpp_bridge import (
    ACS_COVERED_BLOCKS_X,
    ACS_COVERED_BLOCKS_Y,
    CJXL_BIN,
    PATCH_BLEND_ADD,
    PATCH_BLEND_ALPHA_BELOW,
    PATCH_BLEND_MUL_ADD_BELOW,
    PatchEntry,
    compute_psnr,
    decode_with_djxl,
    encode_alpha_composite,
    encode_patches_composite,
    encode_single_frame,
    inspect_jxl,
    verify_jxl_codestream,
)
from jxl_wasserstein.vgg import load_vgg16_params


@pytest.fixture(scope="module")
def vgg_params():
  return load_vgg16_params()


def _make_rich_sdr_test_image(h: int = 128, w: int = 128) -> np.ndarray:
  yy, xx = np.meshgrid(
      np.linspace(0.0, 1.0, h, dtype=np.float32),
      np.linspace(0.0, 1.0, w, dtype=np.float32),
      indexing="ij",
  )
  rng = np.random.default_rng(123)
  # Smooth gradient + correlated chroma + sharp high-contrast edges & fine texture
  luma = 0.48 + 0.25 * np.sin(2.0 * np.pi * yy) * np.cos(2.0 * np.pi * xx)
  edge = ((xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.08).astype(np.float32)
  tex = 0.18 * np.sin(28.0 * np.pi * xx) * np.cos(24.0 * np.pi * yy) * edge
  r = luma + 0.14 * yy + tex + rng.normal(0.0, 0.008, (h, w)).astype(np.float32)
  g = luma - 0.05 * xx + 0.8 * tex + rng.normal(0.0, 0.008, (h, w)).astype(np.float32)
  b = luma - 0.16 * yy + 0.5 * tex + rng.normal(0.0, 0.008, (h, w)).astype(np.float32)
  rgb = np.stack([r, g, b], axis=-1)
  return np.round(np.clip(rgb, 0.02, 0.98) * 255.0).astype(np.uint8)


def test_fix1_no_nonlinear_distance_inflation_at_high_d():
  """Fix 1: `init_state_from_rgb_jax` calibrates global_scale ~ 5111/d without 49.5x power-law explosion."""
  img = _make_rich_sdr_test_image(64, 64)
  st_42 = forward_model.init_state_from_rgb_jax(img, distance=4.2)
  st_85 = forward_model.init_state_from_rgb_jax(img, distance=8.5)

  # Previously d=4.2 inflated to d_eff=7.92 and d=8.5 inflated to d_eff=52 (gs ~ 98)
  assert abs(st_42.global_scale - round(5111.0 / 4.2)) <= 25
  assert abs(st_85.global_scale - round(5111.0 / 8.5)) <= 15
  assert st_85.global_scale >= 550
  assert int(np.count_nonzero(st_85.quant_ac)) > 50

  jb_85, _ = encode_single_frame(st_85)
  dec_85 = decode_with_djxl(jb_85)
  psnr_85 = compute_psnr(img, dec_85, peak=255.0)
  assert psnr_85 >= 29.0, f"Expected clean PSNR >= 29 dB at d=8.5, got {psnr_85:.2f} dB"


def test_fix2_epf_aq_cfl_enabled_and_broadcast_epf_inv_sigma():
  """Fix 2: `compositing` initializes EPF >= 1, adaptive raw_quant_field, CfL, and broadcast EPF gather."""
  img = _make_rich_sdr_test_image(128, 128)
  st = compositing._init_frame_state_with_aq_epf_cfl(img, distance=2.0)
  assert st.epf_iters >= 1
  assert len(np.unique(st.raw_quant_field)) >= 2
  assert np.any(st.ytox_map != 0) or np.any(st.ytob_map != 0)

  # Verify broadcast-reshape _epf_pixel_inv_sigma matches exact block indexing on odd dimensions
  inv_grid = jnp.linspace(0.5, 4.5, 3 * 4, dtype=jnp.float32).reshape(3, 4)
  h_odd, w_odd = 19, 27
  eff, row_px = forward_model._epf_pixel_inv_sigma(inv_grid, h_odd, w_odd, 1.65, 0.66)
  by = jnp.arange(h_odd) // 8
  bx = jnp.arange(w_odd) // 8
  expected_row_px = inv_grid[by[:, None], bx[None, :]]
  np.testing.assert_allclose(np.asarray(row_px), np.asarray(expected_row_px), rtol=1e-6, atol=1e-6)
  assert eff.shape == (h_odd, w_odd)

  grad_grid = jax.grad(lambda g: jnp.sum(forward_model.epf_stage1(jnp.ones((3, h_odd, w_odd)) * 0.3, g)))(inv_grid)
  assert bool(jnp.all(jnp.isfinite(grad_grid)))


def test_fix3_default_log2_sigma_is_1_0_across_pipeline():
  """Fix 3: Default `log2_sigma` is 1.0 (sigma = 2 px) across wasserstein, optimizer, compositing, and CLI."""
  fns = [
      wasserstein.compute_wd,
      wasserstein.compute_spatial_wd_map,
      optimizer.stage1_explore_ac_candidates,
      optimizer.optimize_single_frame,
      compositing.optimize_single_frame_baseline,
      compositing.optimize_patch_composite,
      compositing.optimize_alpha_composite,
      compositing.select_best_mode,
      compositing.encode_image_to_jxl,
  ]
  for fn in fns:
    sig = inspect.signature(fn)
    assert "log2_sigma" in sig.parameters, f"Missing log2_sigma in {fn.__name__}"
    assert float(sig.parameters["log2_sigma"].default) == 1.0, (
        f"Expected log2_sigma=1.0 in {fn.__name__}, got {sig.parameters['log2_sigma'].default}"
    )

  parsed = cli.build_parser().parse_args(["-i", "in.png", "-o", "out.jxl"])
  assert float(parsed.log2_sigma) == 1.0


def test_fix4_high_bitrate_auto_mode_no_foldback(vgg_params):
  """Fix 4: `calibrate_rd_lambda` increases with `d`, and `mode='auto'` does not fold back at d=0.18 vs d=1.30."""
  d_vals = [0.18, 0.38, 0.72, 1.30, 2.40, 4.20]
  lams = [compositing.calibrate_rd_lambda(d) for d in d_vals]
  assert all(lams[i] < lams[i + 1] for i in range(len(lams) - 1))

  img = _make_rich_sdr_test_image(64, 64)
  res_018 = compositing.encode_image_to_jxl(
      img, distance=0.18, mode="auto", steps=2, verify_decoders=False, vgg_params=vgg_params
  )
  res_130 = compositing.encode_image_to_jxl(
      img, distance=1.30, mode="auto", steps=2, verify_decoders=False, vgg_params=vgg_params
  )
  assert res_018.wd < res_130.wd, (
      f"High-bitrate fold-back detected: WD(d=0.18)={res_018.wd:.4f} >= WD(d=1.30)={res_130.wd:.4f}"
  )
  assert res_018.bpp > res_130.bpp, (
      f"High-bitrate fold-back detected: bpp(d=0.18)={res_018.bpp:.3f} <= bpp(d=1.30)={res_130.bpp:.3f}"
  )


def test_fix5_cfl_counter_adjustment_and_continuous_rq_coupling():
  """Fix 5: CfL changes counter-adjust X/B AC coefficients and continuous `rq` scales AC smoothly before rounding."""
  img = _make_rich_sdr_test_image(64, 64)
  base_st = compositing._init_frame_state_with_aq_epf_cfl(img, distance=1.2)
  # Ensure strong non-zero Y AC energy
  ac_float = base_st.quant_ac.astype(np.float32)
  dc_float = base_st.dc_quant.astype(np.float32)

  cand_st = base_st.copy()
  cand_st.ytox_map = np.clip(base_st.ytox_map.astype(np.int32) + 24, -128, 127).astype(np.int8)
  cand_st.ytob_map = np.clip(base_st.ytob_map.astype(np.int32) - 20, -128, 127).astype(np.int8)

  no_adj_st = cand_st.copy()
  ac_adj, dc_adj = compositing._couple_candidate_ac_dc_np(
      ac_float,
      dc_float,
      cand_st,
      base_st,
      None,
      None,
      False,
      rq_cont=base_st.raw_quant_field.astype(np.float32),
  )
  # Soft reconstruction (before integer rounding) is nearly exact (>= 45 dB vs ~14.5 dB unadjusted)
  rec_base_soft = np.asarray(
      forward_model.reconstruct_single_frame(base_st, hard=False, epf_iters_override=0)
  )
  rec_cand_soft = np.asarray(
      forward_model.reconstruct_single_frame(
          cand_st, quant_ac=ac_adj, dc_quant=dc_adj, hard=False, epf_iters_override=0
      )
  )
  psnr_cfl_soft = compute_psnr(rec_base_soft, rec_cand_soft, peak=1.0)
  assert psnr_cfl_soft >= 45.0, (
      f"Expected soft CfL counter-adjustment PSNR >= 45 dB, got {psnr_cfl_soft:.2f} dB"
  )

  # Integer-rounded reconstruction improves by > 20 dB over unadjusted CfL update
  cand_st.quant_ac = np.round(ac_adj).astype(np.int32)
  cand_st.dc_quant = np.round(dc_adj).astype(np.int32)
  rec_base = np.asarray(
      forward_model.reconstruct_single_frame(base_st, hard=True, epf_iters_override=0)
  )
  rec_noadj = np.asarray(
      forward_model.reconstruct_single_frame(no_adj_st, hard=True, epf_iters_override=0)
  )
  rec_cand = np.asarray(
      forward_model.reconstruct_single_frame(cand_st, hard=True, epf_iters_override=0)
  )
  psnr_noadj = compute_psnr(rec_base, rec_noadj, peak=1.0)
  psnr_cfl = compute_psnr(rec_base, rec_cand, peak=1.0)
  assert psnr_cfl >= 35.0 and (psnr_cfl - psnr_noadj) >= 20.0, (
      f"Expected hard CfL counter-adjustment >= 35 dB and +20 dB over unadjusted "
      f"({psnr_noadj:.2f} dB), got {psnr_cfl:.2f} dB"
  )

  # Continuous rq_cont +0.4 (rounds to same int rq, but continuous coupling scales AC by +0.4/rq)
  rq_frac = base_st.raw_quant_field.astype(np.float32) + 0.4
  cand_rq = base_st.copy()
  cand_rq.raw_quant_field = np.round(rq_frac).astype(np.int32)
  ac_frac, _ = compositing._couple_candidate_ac_dc_np(
      ac_float * 4.0,
      dc_float,
      cand_rq,
      base_st,
      None,
      None,
      False,
      rq_cont=rq_frac,
  )
  ac_unscaled, _ = compositing._couple_candidate_ac_dc_np(
      ac_float * 4.0,
      dc_float,
      cand_rq,
      base_st,
      None,
      None,
      False,
      rq_cont=base_st.raw_quant_field.astype(np.float32),
  )
  assert not np.allclose(ac_frac, ac_unscaled)

  # Ensure hardcoded `* 1.06` AC inflation is gone from optimize_single_frame_baseline
  src_single = inspect.getsource(compositing.optimize_single_frame_baseline)
  assert "1.06" not in src_single


def test_fix6_discover_sdr_corpus_filters_to_40_pure_8bit_sdr_images():
  """Fix 6: `discover_sdr_corpus()` returns the exact 40 pure 8-bit SDR images and drops 16-bit & ClassE/HDR."""
  entries = benchmark.discover_sdr_corpus()
  assert len(entries) == 40
  names = [n for n, _ in entries]
  assert "info_flower_foveon.ppm.png" in names
  for name, path in entries:
    assert not name.startswith("ClassE_")
    assert "hdr." not in name.lower()
    assert "_10bit_" not in name.lower() and "_12bit_" not in name.lower()
    assert benchmark.is_pure_8bit_sdr_image(path)


def test_fix7_dequant_parameterization_selection_by_estimated_frame_bits(vgg_params):
  """Fix 7: `select_dequant_mode_for_frame` chooses none (<16kb), compact (16..96kb), full_bands (>=96kb)."""
  assert compositing.select_dequant_mode_for_frame(64, 64, 2.0) == "none"
  assert compositing.select_dequant_mode_for_frame(256, 256, 1.5) == "compact"
  assert compositing.select_dequant_mode_for_frame(1024, 1024, 1.0) == "full_bands"

  # On 64x64 (< 16 kbits), default custom_dequant=None emits 1-bit default header (all_default == True)
  img64 = _make_rich_sdr_test_image(64, 64)
  res64 = compositing.optimize_single_frame_baseline(
      img64, distance=2.0, steps=2, custom_dequant=None, verify_decoders=False, vgg_params=vgg_params
  )
  assert inspect_jxl(res64.jxl_bytes)["all_default"] is True


def test_fix8_multi_cluster_entropy_proxy_and_cpp_block_ctx_clustering():
  """Fix 8: Multi-cluster AC token rate proxy is differentiable and C++ helper clusters block contexts."""
  img = _make_rich_sdr_test_image(128, 128)
  acs8 = np.zeros((16, 16), dtype=np.uint8)
  acs8[0:2, 0:2] = 4  # DCT16X16 (cy=2, cx=2)
  acs8[2:4, 0:4] = 11  # DCT16X32 (cy=2, cx=4)
  acs8[4, 4] = 1      # IDENTITY
  acs8[4, 5] = 14     # AFV0
  st = compositing._init_frame_state_with_aq_epf_cfl(img, distance=1.5, ac_strategy=acs8)

  q_ac = jnp.asarray(st.quant_ac, dtype=jnp.float32)
  rq = jnp.asarray(st.raw_quant_field, dtype=jnp.float32)
  bits_global = float(entropy_proxy.soft_jxl_rate_proxy(st, quant_ac=q_ac, raw_quant_field=rq, cluster_blend=0.0))
  bits_clustered = float(entropy_proxy.soft_jxl_rate_proxy(st, quant_ac=q_ac, raw_quant_field=rq, cluster_blend=1.0))
  assert np.isfinite(bits_global) and np.isfinite(bits_clustered)
  assert bits_clustered > 0.0

  g_ac = jax.grad(
      lambda ac: entropy_proxy.soft_jxl_rate_proxy(st, quant_ac=ac, raw_quant_field=rq, cluster_blend=1.0)
  )(q_ac)
  assert bool(jnp.all(jnp.isfinite(g_ac))) and float(jnp.linalg.norm(g_ac)) > 0.0

  jb, _ = encode_single_frame(st)
  info = inspect_jxl(jb)
  assert int(info.get("num_block_ctxs", 0)) > 1


def test_fix9_all_le_32x32_ac_strategies_0_to_17_parity_and_64x64_tile_boundary():
  """Fix 9: Strategies 0..17 reconstruct matching djxl >= 40 dB / jxl-rs >= 45 dB and never cross 64x64 tiles."""
  img = _make_rich_sdr_test_image(128, 128)
  h8, w8 = 16, 16
  acs = np.zeros((h8, w8), dtype=np.uint8)
  # Place every single AcStrategy in 0..17 inside the 8x8-block (64x64 px) CfL tiles:
  # Tile (0, 0): rows 0..7, cols 0..7
  acs[0:4, 0:4] = 5    # DCT32X32 (cy=4, cx=4)
  acs[0:4, 4:6] = 10   # DCT32X16 (cy=4, cx=2)
  acs[0:2, 6:8] = 4    # DCT16X16 (cy=2, cx=2)
  acs[2:4, 6] = 6      # DCT16X8  (cy=2, cx=1)
  acs[2, 7] = 1        # IDENTITY (cy=1, cx=1)
  acs[3, 7] = 2        # DCT2X2   (cy=1, cx=1)
  acs[4:6, 0:4] = 11   # DCT16X32 (cy=2, cx=4)
  acs[6, 0:4] = 9      # DCT8X32  (cy=1, cx=4)
  acs[7, 0:2] = 7      # DCT8X16  (cy=1, cx=2)
  acs[7, 2] = 3        # DCT4X4   (cy=1, cx=1)
  acs[7, 3] = 12       # DCT4X8   (cy=1, cx=1)
  # Tile (0, 1): rows 0..7, cols 8..15
  acs[0:4, 8] = 8      # DCT32X8  (cy=4, cx=1)
  acs[0, 9] = 13       # DCT8X4   (cy=1, cx=1)
  acs[0, 10] = 14      # AFV0     (cy=1, cx=1)
  acs[0, 11] = 15      # AFV1     (cy=1, cx=1)
  acs[0, 12] = 16      # AFV2     (cy=1, cx=1)
  acs[0, 13] = 17      # AFV3     (cy=1, cx=1)

  assert set(range(18)).issubset(set(int(x) for x in np.unique(acs)))
  st = forward_model.init_state_from_rgb_jax(img, distance=1.0, ac_strategy=acs, epf_iters=1, gab=True)
  jb, _ = encode_single_frame(st)
  ver = verify_jxl_codestream(jb, min_cross_decoder_psnr=45.0)
  assert ver["cross_decoder_psnr"] >= 45.0

  jax_rgb = np.asarray(forward_model.reconstruct_single_frame(st, hard=True))
  djxl_rgb = ver["rgb_djxl"].astype(np.float32) / 255.0
  psnr_parity = compute_psnr(jax_rgb, djxl_rgb, peak=1.0)
  assert psnr_parity >= 40.0, f"Expected JAX vs djxl PSNR >= 40 dB across strategies 0..17, got {psnr_parity:.2f} dB"

  # Verify Stage 2 DP strongly prefers multi-block across boundary if unconstrained, yet never crosses 8-block CfL tile!
  fake_cands = {0: {"j_map": np.ones((16, 16), dtype=np.float32) * 10.0}}
  fake_cands[5] = {"j_map": np.ones((16, 16), dtype=np.float32) * 0.1}   # 4x4 DCT32X32
  fake_cands[18] = {"j_map": np.ones((16, 16), dtype=np.float32) * 0.01} # >=64x64 must be ignored
  dp_grid, dp_anc = compositing.stage2_select_non_overlapping_ac_strategy_compositing(fake_cands, 16, 16)
  assert int(np.max(dp_grid)) <= 17
  bys, bxs = np.nonzero(dp_anc)
  for by, bx in zip(bys.tolist(), bxs.tolist()):
    s_id = int(dp_grid[by, bx])
    cy, cx = ACS_COVERED_BLOCKS_Y[s_id], ACS_COVERED_BLOCKS_X[s_id]
    assert (by // 8) == ((by + cy - 1) // 8)
    assert (bx // 8) == ((bx + cx - 1) // 8)


def _max_u8_err(jax_rgb_01: np.ndarray, jxl_bytes: bytes) -> float:
  djxl_u8 = decode_with_djxl(jxl_bytes).astype(np.float32)
  jax_u8 = np.rint(np.clip(np.asarray(jax_rgb_01, dtype=np.float32) * 255.0, 0.0, 255.0))
  return float(np.max(np.abs(jax_u8 - djxl_u8)))


def test_decoder_proxy_max_u8_error_across_modes_and_transforms():
  """Verifies JAX differentiable decoder proxy L_inf error <= 2/255 vs djxl across 0..17, odd dims, patches, and alpha."""
  img = _make_rich_sdr_test_image(128, 128)
  acs = np.zeros((16, 16), dtype=np.uint8)
  acs[0:4, 0:4] = 5
  acs[0:4, 4:6] = 10
  acs[0:2, 6:8] = 4
  acs[2:4, 6] = 6
  acs[2, 7] = 1
  acs[3, 7] = 2
  acs[4:6, 0:4] = 11
  acs[6, 0:4] = 9
  acs[7, 0:2] = 7
  acs[7, 2] = 3
  acs[7, 3] = 12
  acs[0:4, 8] = 8
  acs[0, 9] = 13
  acs[0, 10] = 14
  acs[0, 11] = 15
  acs[0, 12] = 16
  acs[0, 13] = 17

  # 1. Mixed AcStrategy 0..17 + custom dequant ("compact" and "full_bands")
  rng = np.random.default_rng(42)
  for dq_mode, dof in [("compact", 9), ("full_bands", 33)]:
    st = compositing._init_frame_state_with_aq_epf_cfl(img, distance=1.0, ac_strategy=acs)
    dq_vec = rng.uniform(-0.15, 0.15, size=(dof,)).astype(np.float32)
    compositing._apply_custom_dequant_to_state(st, dq_vec, custom_dequant=dq_mode)
    jb, _ = encode_single_frame(st)
    rec = np.asarray(forward_model.reconstruct_single_frame(st, hard=True))
    fast_rec = np.asarray(compositing._fast_reconstruct_single_frame(st))
    assert _max_u8_err(rec, jb) <= 2.0
    assert float(np.max(np.abs(fast_rec - rec)) * 255.0) <= 0.05

  # 2. Odd-dimension high-rate (d=0.18, epf=0, gab=False) & mid-rate (d=1.5, epf=1, gab=True)
  img_odd = img[:75, :83]
  for d_val in (0.18, 1.5):
    st_odd = compositing._init_frame_state_with_aq_epf_cfl(img_odd, distance=d_val)
    jb_odd, _ = encode_single_frame(st_odd)
    rec_odd = np.asarray(forward_model.reconstruct_single_frame(st_odd, hard=True))
    assert _max_u8_err(rec_odd, jb_odd) <= 2.0

  # 3. Single full-frame patches (kAdd) and XYB patch-alpha / alphamuladd (dim_shift=0, 1, 2)
  for u in (2, 4):
    ref_s = compositing._init_frame_state_with_aq_epf_cfl(img[::u, ::u], distance=1.2)
    main_s = compositing._init_frame_state_with_aq_epf_cfl(img, distance=0.8, ac_strategy=acs)
    ref_s_add = compositing._init_frame_state_with_aq_epf_cfl(img[::u, ::u] // 2, distance=1.2)
    main_s_add = compositing._init_frame_state_with_aq_epf_cfl(img // 3, distance=0.8, ac_strategy=acs)
    patches = [PatchEntry(0, 0, 0, 128, 128, 0, 0, PATCH_BLEND_ADD)]
    jb_p, _ = encode_patches_composite(ref_s_add, main_s_add, u, patches)
    rec_p = np.asarray(forward_model.reconstruct_patch_composite(ref_s_add, main_s_add, u, patches, hard=True))
    fast_p = np.asarray(compositing._fast_reconstruct_patch_composite(ref_s_add, main_s_add, u, patches))
    assert _max_u8_err(rec_p, jb_p) <= 2.0
    assert float(np.max(np.abs(fast_p - rec_p)) * 255.0) <= 0.05

    for ds in (0, 1, 2):
      ah, aw = (128 + (1 << ds) - 1) >> ds, (128 + (1 << ds) - 1) >> ds
      yy, xx = np.meshgrid(np.linspace(0.0, 1.0, ah), np.linspace(0.0, 1.0, aw), indexing="ij")
      alpha_01 = np.clip(0.5 + 0.5 * np.sin(3.0 * np.pi * yy) * np.cos(3.0 * np.pi * xx), 0.0, 1.0).astype(np.float32)
      alpha_u8 = np.rint(alpha_01 * 255.0).astype(np.uint8)
      for a_bm, r_st, m_st in (
          (PATCH_BLEND_ALPHA_BELOW, ref_s, main_s),
          (PATCH_BLEND_MUL_ADD_BELOW, ref_s_add, main_s_add),
      ):
        jb_a, _ = encode_alpha_composite(
            r_st, m_st, upsampling=u, alpha_u8=alpha_u8, dim_shift=ds, alpha_blend_mode=a_bm
        )
        rec_a = np.asarray(
            forward_model.reconstruct_alpha_composite(
                r_st, m_st, upsampling=u, dim_shift=ds, alpha_01=alpha_01, alpha_blend_mode=a_bm, hard=True
            )
        )
        fast_a = np.asarray(
            compositing._fast_reconstruct_alpha_composite(
                r_st, m_st, upsampling=u, dim_shift=ds, alpha_01=alpha_01, alpha_blend_mode=a_bm
            )
        )
        assert _max_u8_err(rec_a, jb_a) <= 2.0
        assert float(np.max(np.abs(fast_a - rec_a)) * 255.0) <= 0.05


def test_high_rate_encode_achieves_strong_butteraugli_and_ssimulacra2(vgg_params):
  """Verifies highest-rate (d=0.18) encodes achieve high PSNR, SSIMULACRA2, and sub-threshold Butteraugli."""
  repo_root = Path(CJXL_BIN).resolve().parents[2]
  kodak_img = repo_root / "image-corpus/qoi_benchmark_suite/photo_kodak/kodim23.png"
  if kodak_img.exists():
    full_u8 = np.asarray(Image.open(kodak_img).convert("RGB"), dtype=np.uint8)
    img = np.ascontiguousarray(full_u8[160:288, 288:416])
  else:
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, 128, dtype=np.float32), np.linspace(0.0, 1.0, 128, dtype=np.float32), indexing="ij")
    luma = 0.48 + 0.25 * np.sin(2.0 * np.pi * yy) * np.cos(2.0 * np.pi * xx)
    edge = (0.5 + 0.5 * np.tanh(30.0 * (0.08 - ((xx - 0.5) ** 2 + (yy - 0.5) ** 2)))).astype(np.float32)
    tex = 0.16 * np.sin(20.0 * np.pi * xx) * np.cos(18.0 * np.pi * yy) * edge
    rgb = np.stack([luma + 0.14 * yy + tex, luma - 0.05 * xx + 0.8 * tex, luma - 0.16 * yy + 0.5 * tex], axis=-1)
    img = np.round(np.clip(rgb, 0.02, 0.98) * 255.0).astype(np.uint8)

  res = compositing.encode_image_to_jxl(
      img, distance=0.18, mode="auto", steps=8, verify_decoders=True, vgg_params=vgg_params
  )
  dec_u8 = decode_with_djxl(res.jxl_bytes)
  psnr = float(compute_psnr(img, dec_u8, peak=255.0))
  assert psnr >= 44.5, f"Expected PSNR >= 44.5 dB at d=0.18, got {psnr:.2f} dB"
  assert _max_u8_err(np.asarray(res.recon_rgb), res.jxl_bytes) <= 2.0

  tools_dir = Path(CJXL_BIN).parent
  ssim2_bin = tools_dir / "ssimulacra2"
  ba_bin = tools_dir / "butteraugli_main"
  with tempfile.TemporaryDirectory(prefix="jxlw_hr_test_") as td:
    p_orig = Path(td) / "orig.png"
    p_dec = Path(td) / "dec.png"
    Image.fromarray(img, mode="RGB").save(p_orig)
    Image.fromarray(dec_u8, mode="RGB").save(p_dec)

    if ssim2_bin.exists():
      r_s2 = subprocess.run([str(ssim2_bin), str(p_orig), str(p_dec)], capture_output=True, text=True, check=True)
      ssim2 = float(r_s2.stdout.strip().splitlines()[-1])
      assert ssim2 >= 93.0, f"Expected SSIMULACRA2 >= 93.0 at d=0.18, got {ssim2:.2f}"

    if ba_bin.exists():
      r_ba = subprocess.run([str(ba_bin), str(p_orig), str(p_dec)], capture_output=True, text=True, check=True)
      ba_lines = [ln.strip() for ln in r_ba.stdout.strip().splitlines() if ln.strip()]
      ba_max = float(ba_lines[0])
      ba_3 = float(ba_lines[1].split(":")[-1].strip())
      assert ba_max < 1.0, f"Expected Butteraugli max < 1.0 at d=0.18, got {ba_max:.3f}"
      assert ba_3 < 0.45, f"Expected Butteraugli 3-norm < 0.45 at d=0.18, got {ba_3:.3f}"


def test_flower_foveon_auto_selects_multires_and_noise(vgg_params):
  """Verifies mode='auto' selects multi-resolution (U>1) + noise synthesis on flower_foveon."""
  repo_root = Path(CJXL_BIN).resolve().parents[2]
  flower_img = repo_root / "image-corpus/imagecompression.info.rgb8/flower_foveon.ppm.png"
  if flower_img.exists():
    full_u8 = np.asarray(Image.open(flower_img).convert("RGB"), dtype=np.uint8)
    # 512x768 crop spanning the sharp petal boundary and noisy dark bokeh background
    crop_u8 = np.ascontiguousarray(full_u8[256:768, 1280:2048])
  else:
    rng = np.random.default_rng(42)
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 512, dtype=np.float32),
        np.linspace(0.0, 1.0, 512, dtype=np.float32),
        indexing="ij",
    )
    fg = (xx < 0.38).astype(np.float32)
    bg_rgb = np.stack([0.14 + 0.03 * yy, 0.20 + 0.04 * yy, 0.10 + 0.02 * xx], axis=-1)
    fg_rgb = np.stack(
        [
            0.78 + 0.14 * np.sin(26.0 * np.pi * xx) * np.cos(22.0 * np.pi * yy),
            0.22 + 0.08 * np.cos(20.0 * np.pi * yy),
            0.48 + 0.10 * np.sin(24.0 * np.pi * xx),
        ],
        axis=-1,
    )
    base = fg[:, :, None] * fg_rgb + (1.0 - fg[:, :, None]) * bg_rgb
    grain = rng.normal(0.0, 0.012, size=base.shape).astype(np.float32)
    crop_u8 = np.round(np.clip(base + grain, 0.01, 0.99) * 255.0).astype(np.uint8)

  res = compositing.encode_image_to_jxl(
      crop_u8,
      distance=1.30,
      mode="auto",
      steps=2,
      verify_decoders=True,
      vgg_params=vgg_params,
  )
  assert res.mode_selected in ("patches", "alpha"), (
      f"Expected multi-res mode ('patches' or 'alpha') on flower_foveon, got {res.mode_selected}"
  )
  assert res.upsampling > 1, f"Expected upsampling > 1, got {res.upsampling}"
  assert res.has_noise is True, "Expected automatic noise synthesis to be enabled"
  assert res.noise_lut is not None and max(res.noise_lut) > 0.0
  assert res.psnr_djxl_vs_jxlrs >= 45.0
  single_cands = [c for c in res.all_candidates if c["mode"] == "single"]
  assert single_cands, "Expected single-frame baseline in all_candidates"
  assert res.rd_cost < float(single_cands[0]["rd_cost"]), (
      f"Expected selected multi-res rd_cost ({res.rd_cost:.5f}) < "
      f"single rd_cost ({single_cands[0]['rd_cost']:.5f})"
  )


def test_large_width_gt_2048_dim_shift_3_alpha_non_black_djxl_decode():
  """Verifies >2048px (>1 modular group) + dim_shift=3 non-palettized alpha decodes non-black in djxl."""
  h, w = 256, 2268
  yy, xx = np.meshgrid(
      np.linspace(0.0, 1.0, h, dtype=np.float32),
      np.linspace(0.0, 1.0, w, dtype=np.float32),
      indexing="ij",
  )
  r = 0.55 + 0.30 * np.sin(4.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
  g = 0.35 + 0.22 * np.cos(3.0 * np.pi * xx + yy)
  b = 0.50 - 0.25 * np.sin(2.0 * np.pi * yy)
  img_wide = np.round(np.clip(np.stack([r, g, b], axis=-1), 0.04, 0.96) * 255.0).astype(np.uint8)

  ref_s = compositing._init_frame_state_with_aq_epf_cfl(img_wide[::4, ::4], distance=4.2)
  main_s = compositing._init_frame_state_with_aq_epf_cfl(img_wide, distance=3.0)

  ah, aw = (h + 7) >> 3, (w + 7) >> 3
  ay, ax = np.meshgrid(
      np.linspace(0.0, 1.0, ah, dtype=np.float32),
      np.linspace(0.0, 1.0, aw, dtype=np.float32),
      indexing="ij",
  )
  alpha_01 = np.clip(0.5 + 0.48 * np.sin(5.0 * np.pi * ax) * np.cos(3.0 * np.pi * ay), 0.0, 1.0).astype(np.float32)
  alpha_u8 = np.rint(alpha_01 * 255.0).astype(np.uint8)
  # Must have >16 distinct alpha values so Modular Palette does not collapse groups to 1
  assert len(np.unique(alpha_u8)) >= 32

  jb, _ = encode_alpha_composite(
      ref_s,
      main_s,
      upsampling=4,
      alpha_u8=alpha_u8,
      dim_shift=3,
      alpha_blend_mode=PATCH_BLEND_ALPHA_BELOW,
  )
  ver = verify_jxl_codestream(jb, min_cross_decoder_psnr=45.0)
  rgb_djxl = ver["rgb_djxl"]
  assert float(rgb_djxl.mean()) > 80.0, f"djxl decoded black/dark frame on w={w}, ds=3: mean={rgb_djxl.mean():.2f}"
  psnr_vs_orig = float(compute_psnr(img_wide, rgb_djxl, peak=255.0))
  assert psnr_vs_orig >= 34.0, f"Expected PSNR >= 34 dB on w={w}, ds=3, got {psnr_vs_orig:.2f} dB"

  rec_a = np.asarray(
      forward_model.reconstruct_alpha_composite(
          ref_s,
          main_s,
          upsampling=4,
          dim_shift=3,
          alpha_01=alpha_u8.astype(np.float32) / 255.0,
          alpha_blend_mode=PATCH_BLEND_ALPHA_BELOW,
          hard=True,
      )
  )
  assert _max_u8_err(rec_a, jb) <= 2.0


def test_multistep_alpha_u4_ds3_no_inpaint_discoloration_and_zero_init_convergence(vgg_params):
  """Verifies multi-step alpha (U=4, ds=3, noise) has clean petal color & zero-init converges cleanly."""
  repo_root = Path(CJXL_BIN).resolve().parents[2]
  flower_img = repo_root / "image-corpus/imagecompression.info.rgb8/flower_foveon.ppm.png"
  if flower_img.exists():
    full_u8 = np.asarray(Image.open(flower_img).convert("RGB"), dtype=np.uint8)
    crop_u8 = np.ascontiguousarray(full_u8[512:768, 1024:1280])
  else:
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 256, dtype=np.float32),
        np.linspace(0.0, 1.0, 256, dtype=np.float32),
        indexing="ij",
    )
    petal = ((xx < 0.65) & (yy < 0.70)).astype(np.float32)
    fg = np.stack([0.74 + 0.08 * np.sin(18.0 * xx), 0.19 + 0.03 * yy, 0.67 + 0.06 * np.cos(16.0 * yy)], axis=-1)
    bg = np.stack([0.16 + 0.03 * yy, 0.24 + 0.04 * xx, 0.10 + 0.02 * yy], axis=-1)
    crop_u8 = np.round(np.clip(petal[..., None] * fg + (1.0 - petal[..., None]) * bg, 0.01, 0.99) * 255.0).astype(
        np.uint8
    )

  orig_petal_rgb = crop_u8[:64, :64].mean(axis=(0, 1))

  res_dct = compositing.optimize_alpha_composite(
      crop_u8,
      upsampling=4,
      dim_shift=3,
      distance=6.21,
      enable_noise=True,
      steps=32,
      init_mode="dct",
      verify_decoders=True,
      vgg_params=vgg_params,
  )
  dec_dct = decode_with_djxl(res_dct.jxl_bytes)
  assert res_dct.psnr_db >= 31.0, f"Expected DCT-init PSNR >= 31 dB at d=6.21, got {res_dct.psnr_db:.2f} dB"
  assert res_dct.stage3_true_j < res_dct.stage2_true_j
  assert res_dct.psnr_djxl_vs_jxlrs >= 45.0
  dct_petal_err = float(np.max(np.abs(dec_dct[:64, :64].mean(axis=(0, 1)) - orig_petal_rgb)))
  assert dct_petal_err <= 8.0, f"Harmonic-inpaint discoloration on petal: max channel shift {dct_petal_err:.2f}"

  res_zero = compositing.optimize_alpha_composite(
      crop_u8,
      upsampling=4,
      dim_shift=3,
      distance=6.21,
      enable_noise=True,
      steps=40,
      init_mode="zero",
      verify_decoders=True,
      vgg_params=vgg_params,
  )
  dec_zero = decode_with_djxl(res_zero.jxl_bytes)
  assert res_zero.psnr_db >= 25.0, f"Expected zero-init PSNR >= 25 dB at d=6.21, got {res_zero.psnr_db:.2f} dB"
  assert res_zero.stage3_true_j < 0.25 * res_zero.stage2_true_j
  assert res_zero.psnr_djxl_vs_jxlrs >= 45.0
  zero_petal_err = float(np.max(np.abs(dec_zero[:64, :64].mean(axis=(0, 1)) - orig_petal_rgb)))
  assert zero_petal_err <= 15.0, f"Zero-init petal chroma failed to converge: max shift {zero_petal_err:.2f}"


def test_w8_or_h8_equals_3_quant_ac_shape_disambiguation(vgg_params):
  """Regression test: h8=3 or w8=3 (including 24x24 and 124x132 U=8 -> 16x17) never mis-transposes 5D quant_ac."""
  # 1. Direct 5D shape disambiguation for (h8=2, w8=3), (h8=3, w8=2), and (h8=3, w8=3)
  for h8, w8 in ((2, 3), (3, 2), (3, 3)):
    img_hw = _make_rich_sdr_test_image(h8 * 8, w8 * 8)
    acs = np.zeros((h8, w8), dtype=np.uint8)
    acs[0:2, 0:2] = 4  # DCT16X16 multi-block so resolved_topo.multi_groups is non-empty
    st = compositing._init_frame_state_with_aq_epf_cfl(img_hw, distance=2.4, ac_strategy=acs)
    q_3hw = jnp.asarray(st.quant_ac, dtype=jnp.float32)
    q_5d_chw = q_3hw.reshape(3, h8, 8, w8, 8).transpose(0, 1, 3, 2, 4)
    bits_3d = float(entropy_proxy.soft_jxl_rate_proxy(st, quant_ac=q_3hw, ac_strategy=acs))
    bits_5d = float(entropy_proxy.soft_jxl_rate_proxy(st, quant_ac=q_5d_chw, ac_strategy=acs))
    assert np.isfinite(bits_3d) and abs(bits_3d - bits_5d) <= 1e-4
    smap_res = entropy_proxy.compute_spatial_rate_map_bits(st, {"quant_ac": q_5d_chw, "ac_strategy": acs})
    smap = np.asarray(smap_res["block_bits"])
    assert smap.shape == (h8, w8) and np.all(np.isfinite(smap))

  # 2. Full end-to-end 124x132 auto encode (U=8 reference frame is 16x17 -> h8=2, w8=3)
  img_124_132 = _make_rich_sdr_test_image(124, 132)
  res = compositing.encode_image_to_jxl(
      img_124_132,
      distance=2.4,
      mode="auto",
      upsampling="auto",
      steps=1,
      verify_decoders=True,
      vgg_params=vgg_params,
  )
  assert res.psnr_djxl_vs_jxlrs >= 45.0


def test_repeated_auto_encodes_release_transient_gpu_caches(vgg_params):
  """Regression test: repeated `encode_image_to_jxl(..., mode='auto')` calls do not retain Stage 1 / VGG caches."""
  img = _make_rich_sdr_test_image(128, 128)
  live_counts = []
  for d in (4.2, 2.4, 1.3):
    res = compositing.encode_image_to_jxl(
        img,
        distance=d,
        mode="auto",
        upsampling="auto",
        steps=1,
        verify_decoders=False,
        vgg_params=vgg_params,
    )
    assert len(res.jxl_bytes) > 0
    del res
    gc.collect()
    assert len(optimizer._STAGE1_CACHE) == 0
    assert len(compositing._TARGET_FEATS_CACHE) == 0
    assert len(compositing._STAGE1_AC_CANDIDATE_CACHE) == 0
    assert len(compositing._STAGE1_MULTIRES_CACHE) == 0
    live_3hw = sum(
        1 for a in jax.live_arrays() if a.ndim == 3 and a.shape == (3, 128, 128)
    )
    live_counts.append(live_3hw)

  # Live (3, H, W) buffer count must not grow monotonically across distance sweeps
  assert live_counts[-1] <= live_counts[0] + 2, f"Leaked (3, H, W) buffers across encodes: {live_counts}"

