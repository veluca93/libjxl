"""Tests for M4 multi-resolution XYB patches, sRGB alpha blending, and RD selector (`compositing.py`).

Verifies:
1. `extract_patch_rectangles` decomposes a boolean block mask into minimal
   non-overlapping 8x8-aligned `PatchEntry` rectangles.
2. `optimize_patch_composite` across `upsampling in {2, 4, 8}` and
   `blend_mode in {PATCH_BLEND_REPLACE (1), PATCH_BLEND_ADD (2)}` produces valid
   2-frame JXL codestreams verified by both `djxl` and `jxl_cli` (`PSNR >= 45 dB`),
   with finite gradients and JAX vs `djxl` parity `>= 45 dB`.
3. `optimize_alpha_composite` across `upsampling in {2, 4, 8}` and
   `dim_shift in {0, 1, 2, 3}` produces valid 2-frame JXL codestreams with
   `extra_channel_bits=[8]`, non-zero `alpha_logit` gradient norm, and JAX vs
   `djxl` parity `>= 45 dB`.
4. `select_best_mode` and `encode_image_to_jxl` automatically evaluate candidate
   topologies by true RD Lagrangian `J = WD + lambda_rd * bpp_true`, and on a
   synthetic mixed smooth-background + sharp-foreground image both `"patches"`
   and `"alpha"` strictly beat `"single"`.
5. Constraint R2 compliance: `dump_init_state` is monkeypatched to raise an error
   throughout all tests.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jxl_wasserstein.compositing as compositing
import jxl_wasserstein.cpp_bridge as cpp_bridge
from jxl_wasserstein.compositing import (
    EncodeResult,
    compute_detail_mask_jax,
    downsample_rgb_jax,
    encode_image_to_jxl,
    extract_patch_rectangles,
    optimize_alpha_composite,
    optimize_patch_composite,
    select_best_mode,
)
from jxl_wasserstein.cpp_bridge import (
    PATCH_BLEND_ADD,
    PATCH_BLEND_REPLACE,
    compute_psnr,
    inspect_jxl,
    verify_jxl_codestream,
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


def _make_smooth_bg_sharp_fg_image(h: int = 64, w: int = 64) -> np.ndarray:
  """Creates a 64x64 image with smooth sinusoidal background and a sharp 16x16 foreground patch."""
  yy, xx = np.meshgrid(
      np.linspace(0, 1, h, dtype=np.float32),
      np.linspace(0, 1, w, dtype=np.float32),
      indexing="ij",
  )
  rng = np.random.default_rng(42)
  bg = np.stack(
      [
          0.45
          + 0.20 * np.sin(2 * np.pi * yy)
          + 0.03 * np.sin(6 * np.pi * yy) * np.cos(6 * np.pi * xx),
          0.50 + 0.18 * np.cos(2 * np.pi * xx) + 0.03 * np.sin(6 * np.pi * xx),
          0.55 + 0.15 * np.sin(2 * np.pi * (yy + xx)),
      ],
      axis=-1,
  ).astype(np.float32)
  img = bg + rng.normal(0.0, 0.012, size=bg.shape).astype(np.float32)
  chk = np.sign(
      np.sin(20 * np.pi * yy[24:40, 24:40])
      * np.cos(20 * np.pi * xx[24:40, 24:40])
  )[..., None]
  img[24:40, 24:40] = np.clip(0.5 + 0.35 * chk, 0.05, 0.95)
  return np.clip(img, 0.02, 0.98)


def test_downsample_and_extract_patch_rectangles():
  """Tests area-average downsampling and greedy rectangle merging."""
  img = _make_smooth_bg_sharp_fg_image(64, 64)
  for u, expected_hw in ((1, (64, 64)), (2, (32, 32)), (4, (16, 16)), (8, (8, 8))):
    down = downsample_rgb_jax(img, u)
    assert down.shape == (*expected_hw, 3)

  with pytest.raises(ValueError):
    downsample_rgb_jax(img, 3)

  mask = np.ones((8, 8), dtype=bool)
  mask[2:6, 2:6] = False
  rects = extract_patch_rectangles(mask, 64, 64, PATCH_BLEND_REPLACE)
  assert len(rects) == 4
  covered = np.zeros((64, 64), dtype=int)
  for r in rects:
    assert r.x0 % 8 == 0 and r.y0 % 8 == 0
    covered[r.y : r.y + r.ysize, r.x : r.x + r.xsize] += 1
  assert np.all(covered[16:48, 16:48] == 0)
  assert int(np.sum(covered == 1)) == 64 * 64 - 32 * 32

  detail = compute_detail_mask_jax(img, upsampling=2, distance=0.5)
  assert detail["fg_mask_blocks"].shape == (8, 8)
  assert np.any(detail["fg_mask_blocks"][3:5, 3:5])


@pytest.mark.parametrize("upsampling", [2, 4, 8])
@pytest.mark.parametrize("blend_mode", [PATCH_BLEND_REPLACE, PATCH_BLEND_ADD])
def test_optimize_patch_composite_all_upsamplings_and_blend_modes(
    upsampling, blend_mode, vgg_params
):
  """Tests `optimize_patch_composite` across U in {2, 4, 8} and blend_mode in {1, 2}."""
  img64 = _make_smooth_bg_sharp_fg_image(64, 64)
  res = optimize_patch_composite(
      img64,
      upsampling=upsampling,
      blend_mode=blend_mode,
      distance=0.35,
      lambda_rd=0.05,
      num_steps=3,
      vgg_params=vgg_params,
  )
  assert isinstance(res, EncodeResult)
  assert res.mode_selected == "patches"
  assert res.upsampling == upsampling
  assert res.num_patches >= 1
  assert res.psnr_djxl_vs_jxlrs >= 45.0
  assert res.psnr_vs_djxl >= 45.0

  ver = verify_jxl_codestream(res["jxl_bytes"], min_cross_decoder_psnr=45.0)
  djxl_rgb = ver["rgb_djxl"].astype(np.float32) / 255.0
  psnr_parity = compute_psnr(res["recon_rgb"], djxl_rgb, peak=1.0)
  assert psnr_parity >= 45.0
  for k, g in res["last_grads"].items():
    assert bool(jnp.all(jnp.isfinite(g))), f"Non-finite gradient for {k}"


@pytest.mark.parametrize(
    "upsampling,dim_shift",
    [(2, 0), (2, 1), (2, 2), (2, 3), (4, 1), (8, 1)],
)
def test_optimize_alpha_composite_upsamplings_and_dim_shifts(
    upsampling, dim_shift, vgg_params
):
  """Tests `optimize_alpha_composite` across U in {2, 4, 8} and dim_shift in {0, 1, 2, 3}."""
  img64 = _make_smooth_bg_sharp_fg_image(64, 64)
  res_a = optimize_alpha_composite(
      img64,
      upsampling=upsampling,
      dim_shift=dim_shift,
      distance=0.35,
      lambda_rd=0.05,
      num_steps=3,
      vgg_params=vgg_params,
  )
  assert isinstance(res_a, EncodeResult)
  assert res_a.mode_selected == "alpha"
  assert res_a.upsampling == upsampling
  assert res_a.dim_shift == dim_shift
  assert res_a.header_info.get("extra_channel_bits") == [8]
  assert res_a.psnr_djxl_vs_jxlrs >= 45.0
  assert res_a.psnr_vs_djxl >= 45.0

  grad_alpha_norm = float(jnp.linalg.norm(res_a["last_grads"]["alpha_logit"]))
  assert grad_alpha_norm > 0.0, (
      f"Zero alpha_logit gradient norm for U={upsampling}, D={dim_shift}"
  )


def test_select_best_mode_and_encode_image_to_jxl(vgg_params):
  """Tests that `select_best_mode` and `encode_image_to_jxl` beat single-frame on mixed detail."""
  img64 = _make_smooth_bg_sharp_fg_image(64, 64)
  best_all = select_best_mode(
      img64,
      candidate_modes=("single", "patches", "alpha"),
      upsampling_candidates=(2, 4),
      dim_shift_candidates=(1,),
      blend_mode_candidates=(PATCH_BLEND_REPLACE,),
      distance=0.35,
      lambda_rd=0.18,
      num_steps=3,
      vgg_params=vgg_params,
  )
  assert isinstance(best_all, EncodeResult)
  c_single = [c for c in best_all["all_candidates"] if c["mode"] == "single"][0]
  c_patches = min(
      [c for c in best_all["all_candidates"] if c["mode"] == "patches"],
      key=lambda x: x["rd_cost"],
  )
  c_alpha = min(
      [c for c in best_all["all_candidates"] if c["mode"] == "alpha"],
      key=lambda x: x["rd_cost"],
  )
  assert c_patches["rd_cost"] < c_single["rd_cost"], (
      f"Expected patches RD cost ({c_patches['rd_cost']:.5f}) < "
      f"single ({c_single['rd_cost']:.5f})"
  )
  assert c_alpha["rd_cost"] < c_single["rd_cost"], (
      f"Expected alpha RD cost ({c_alpha['rd_cost']:.5f}) < "
      f"single ({c_single['rd_cost']:.5f})"
  )
  assert best_all.mode_selected in ("patches", "alpha")
  assert best_all.psnr_djxl_vs_jxlrs >= 45.0

  # Verify high-level encode_image_to_jxl entry point with uint8 input and noise synthesis
  img64_u8 = np.round(img64 * 255.0).astype(np.uint8)
  enc_res = encode_image_to_jxl(
      img64_u8,
      rd_lambda=0.12,
      distance=0.5,
      mode="patches",
      upsampling=2,
      patch_blend_mode="replace",
      enable_noise=True,
      num_steps=2,
      verify_decoders=True,
      vgg_params=vgg_params,
  )
  assert isinstance(enc_res, EncodeResult)
  assert enc_res.mode_selected == "patches"
  assert enc_res.has_noise is True
  assert enc_res.psnr_djxl_vs_jxlrs >= 45.0


def test_f3_no_static_mse_heuristic_and_stage1_rate_wd_maps(
    monkeypatch, vgg_params
):
  """Verifies F3: patch/alpha compositors never call `compute_detail_mask_jax` and populate Stage 1/2 Rate-WD maps."""
  img64 = _make_smooth_bg_sharp_fg_image(64, 64)

  def _forbid_static_detail(*args, **kwargs):
    raise AssertionError(
        "F3 violation: compute_detail_mask_jax must not be called by compositors"
    )

  monkeypatch.setattr(
      compositing, "compute_detail_mask_jax", _forbid_static_detail
  )

  res_p = optimize_patch_composite(
      img64,
      upsampling=2,
      blend_mode="replace",
      distance=0.35,
      lambda_rd=0.05,
      num_steps=2,
      vgg_params=vgg_params,
  )
  assert res_p.spatial_delta_j_map is not None
  assert res_p.spatial_delta_j_map.shape == (8, 8)
  assert res_p.stage1_main_j_map is not None
  assert res_p.stage1_ref_j_map is not None
  assert np.allclose(
      res_p.spatial_delta_j_map,
      res_p.stage1_main_j_map - res_p.stage1_ref_j_map,
      atol=1e-6,
  )
  assert res_p.stage1_candidates is not None and len(res_p.stage1_candidates) >= 1
  assert res_p.stage2_ac_strategy is not None
  assert res_p.stage2_ac_strategy.shape == (8, 8)

  res_a = optimize_alpha_composite(
      img64,
      upsampling=2,
      dim_shift=1,
      distance=0.35,
      lambda_rd=0.05,
      num_steps=2,
      vgg_params=vgg_params,
  )
  assert res_a.spatial_delta_j_map is not None
  assert res_a.spatial_delta_j_map.shape == (8, 8)
  assert res_a.stage1_candidates is not None and len(res_a.stage1_candidates) >= 1
  assert res_a.stage2_ac_strategy is not None
  assert res_a.stage2_ac_strategy.shape == (8, 8)


@pytest.mark.parametrize("alpha_blend_mode", [4, 5, 6, 7])
def test_r2_xyb_patch_dictionary_alpha_blending_and_post_patch_noise(
    alpha_blend_mode, vgg_params
):
  """Verifies R2: Frame 0 saved in XYB (`save_before_color_transform=True`), Frame 1 uses `PatchDictionary` (`has_patches=True`)."""
  img64 = _make_smooth_bg_sharp_fg_image(64, 64)
  enable_noise = alpha_blend_mode == 5
  res_a = optimize_alpha_composite(
      img64,
      upsampling=2,
      dim_shift=1,
      alpha_blend_mode=alpha_blend_mode,
      enable_noise=enable_noise,
      distance=0.35,
      lambda_rd=0.05,
      num_steps=2,
      vgg_params=vgg_params,
  )
  assert res_a.alpha_blend_mode == alpha_blend_mode
  assert res_a.psnr_djxl_vs_jxlrs >= 45.0
  assert res_a.psnr_vs_djxl >= 45.0

  info = inspect_jxl(res_a.jxl_bytes)
  assert info["num_frames"] == 2
  assert info["num_extra_channels"] >= 1
  f0, f1 = info["frames"][0], info["frames"][1]
  assert f0["frame_type"] in (1, "kReferenceOnly")
  assert f0["save_before_color_transform"] is True
  assert f0["effective_ec_upsampling"] == [max(2, 1 << 1)]
  assert f1["frame_type"] in (0, "kRegularFrame")
  assert f1["has_patches"] is True
  assert alpha_blend_mode in f1["patch_blend_modes"]
  if enable_noise:
    assert res_a.has_noise is True
    assert f1["has_noise"] is True
    assert res_a.noise_lut is not None and len(res_a.noise_lut) == 8


def test_f9_f10_f11_stage3_strictly_improves_j_and_custom_dequant(vgg_params):
  """Verifies F9/F10/F11: `stage3_true_j < stage2_true_j` and `inspect_jxl` `all_default` reflects `custom_dequant`."""
  img64 = _make_smooth_bg_sharp_fg_image(64, 64)

  res_p_custom = optimize_patch_composite(
      img64,
      upsampling=2,
      blend_mode="replace",
      custom_dequant=True,
      distance=0.35,
      lambda_rd=0.05,
      num_steps=2,
      vgg_params=vgg_params,
  )
  assert res_p_custom.stage3_true_j < res_p_custom.stage2_true_j
  assert inspect_jxl(res_p_custom.jxl_bytes)["all_default"] is False

  res_p_def = optimize_patch_composite(
      img64,
      upsampling=2,
      blend_mode="replace",
      custom_dequant=False,
      distance=0.35,
      lambda_rd=0.05,
      num_steps=2,
      vgg_params=vgg_params,
  )
  assert res_p_def.stage3_true_j < res_p_def.stage2_true_j
  assert inspect_jxl(res_p_def.jxl_bytes)["all_default"] is True

  res_a_custom = optimize_alpha_composite(
      img64,
      upsampling=2,
      dim_shift=1,
      custom_dequant=True,
      distance=0.35,
      lambda_rd=0.05,
      num_steps=2,
      vgg_params=vgg_params,
  )
  assert res_a_custom.stage3_true_j < res_a_custom.stage2_true_j
  assert inspect_jxl(res_a_custom.jxl_bytes)["all_default"] is False

  res_a_def = optimize_alpha_composite(
      img64,
      upsampling=2,
      dim_shift=1,
      custom_dequant=False,
      distance=0.35,
      lambda_rd=0.05,
      num_steps=2,
      vgg_params=vgg_params,
  )
  assert res_a_def.stage3_true_j < res_a_def.stage2_true_j
  assert inspect_jxl(res_a_def.jxl_bytes)["all_default"] is True


def test_stage2_auto_scaling_and_zero_random_init(vgg_params):
  """Verifies Stage 1/2 automatic scaling selection over U in {2, 4, 8} and zero/random init."""
  img64 = _make_smooth_bg_sharp_fg_image(64, 64)

  res_patch_auto = optimize_patch_composite(
      img64,
      upsampling="auto",
      blend_mode=PATCH_BLEND_REPLACE,
      distance=0.5,
      lambda_rd=0.06,
      num_steps=3,
      vgg_params=vgg_params,
  )
  assert res_patch_auto.upsampling in (2, 4, 8)
  assert [c["upsampling"] for c in res_patch_auto.all_candidates] == [2, 4, 8]
  assert res_patch_auto.stage3_true_j < res_patch_auto.stage2_true_j
  assert res_patch_auto.ref_state is not None and bool(res_patch_auto.ref_state.dequant_matrices)
  assert res_patch_auto.main_state is not None and bool(res_patch_auto.main_state.dequant_matrices)

  res_alpha_auto = optimize_alpha_composite(
      img64,
      upsampling="auto",
      dim_shift="auto",
      distance=0.5,
      lambda_rd=0.06,
      num_steps=3,
      vgg_params=vgg_params,
  )
  assert res_alpha_auto.upsampling in (2, 4, 8)
  assert res_alpha_auto.dim_shift in (0, 1, 2, 3)
  assert len(res_alpha_auto.all_candidates) == 12
  assert res_alpha_auto.stage3_true_j < res_alpha_auto.stage2_true_j

  res_single_zero = encode_image_to_jxl(
      img64,
      mode="single",
      distance=1.5,
      rd_lambda=0.08,
      init_mode="zero",
      num_steps=30,
      vgg_params=vgg_params,
  )
  assert res_single_zero.stage3_true_j < res_single_zero.stage2_true_j
  assert res_single_zero.psnr_jax_vs_djxl >= 40.0
  assert res_single_zero.cross_decoder_psnr >= 45.0


