"""Unit tests for deterministic JXL Xorshift128Plus PRNG and differentiable NoiseParams synthesis."""

from __future__ import annotations

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest

from jxl_wasserstein.cpp_bridge import (
    PATCH_BLEND_ADD,
    PatchEntry,
    compute_psnr,
    decode_with_djxl,
    decode_with_jxl_rs,
    encode_alpha_composite,
    encode_patches_composite,
    encode_single_frame,
    inspect_jxl,
    verify_jxl_codestream,
)
from jxl_wasserstein.forward_model import (
    init_state_from_rgb_jax,
    reconstruct_alpha_composite,
    reconstruct_patch_composite,
    reconstruct_single_frame,
)
from jxl_wasserstein.noise import (
    NOISE_LUT_MAX,
    NOISE_PRECISION,
    Xorshift128Plus,
    convolve_noise_planes,
    eval_noise_strength_lut,
    generate_raw_noise_planes_np,
    quantize_noise_lut_ste,
    synthesize_jxl_noise,
)

KODAK_DIR = Path(
    "/usr/local/google/home/veluca/libjxl/image-corpus/qoi_benchmark_suite/photo_kodak"
)


def _load_kodak_crop(name: str = "kodim01.png", w: int = 128, h: int = 96) -> np.ndarray:
  img = Image.open(KODAK_DIR / name).convert("RGB")
  iw, ih = img.size
  x0 = (iw - w) // 2
  y0 = (ih - h) // 2
  return np.asarray(img.crop((x0, y0, x0 + w, y0 + h)), dtype=np.float32) / 255.0


def test_xorshift128plus_deterministic_seeding_and_batch_generation() -> None:
  rng_a = Xorshift128Plus(1, 0, 0, 0)
  rng_b = Xorshift128Plus(1, 0, 0, 0)
  u64_a = rng_a.fill_u64()
  u64_b = rng_b.fill_u64()
  assert u64_a.shape == (8,)
  assert u64_a.dtype == np.uint64
  np.testing.assert_array_equal(u64_a, u64_b)

  f32_batch = rng_a.fill_f32_batch(subtract_one=False)
  assert f32_batch.shape == (16,)
  assert f32_batch.dtype == np.float32
  assert bool(np.all((f32_batch >= 1.0) & (f32_batch < 2.0)))

  f32_sub = rng_b.fill_f32_batch(subtract_one=True)
  assert bool(np.all((f32_sub >= 0.0) & (f32_sub < 1.0)))
  np.testing.assert_allclose(f32_batch - 1.0, f32_sub, atol=1e-7)

  # Single-seed constructor
  rng_single1 = Xorshift128Plus(123456789)
  rng_single2 = Xorshift128Plus(123456789)
  np.testing.assert_array_equal(rng_single1.fill_u64(), rng_single2.fill_u64())

  # Distinct tile/frame seeds are uncorrelated
  planes_10 = generate_raw_noise_planes_np(64, 64, visible_frame_index=1, nonvisible_frame_index=0)
  planes_01 = generate_raw_noise_planes_np(64, 64, visible_frame_index=0, nonvisible_frame_index=1)
  corr = float(np.corrcoef(planes_10.ravel(), planes_01.ravel())[0, 1])
  assert abs(corr) < 0.05


def test_convolve_noise_planes_zero_mean_and_impulse_response() -> None:
  # Constant plane -> exact 0.0 everywhere since 24 * 0.16 - 3.84 == 0.0
  const_plane = jnp.full((3, 16, 16), 1.5, dtype=jnp.float32)
  conv_const = np.asarray(convolve_noise_planes(const_plane))
  np.testing.assert_allclose(conv_const, 0.0, atol=1e-6)

  # Interior impulse response at (4, 4) of a 9x9 plane
  impulse = np.zeros((3, 9, 9), dtype=np.float32)
  impulse[0, 4, 4] = 1.0
  conv_imp = np.asarray(convolve_noise_planes(impulse))[0]
  assert np.isclose(conv_imp[4, 4], -3.84, atol=1e-6)
  for dy in range(-2, 3):
    for dx in range(-2, 3):
      if dy == 0 and dx == 0:
        continue
      assert np.isclose(conv_imp[4 + dy, 4 + dx], 0.16, atol=1e-6)
  # Outside 5x5 window around (4, 4) is 0.0
  assert np.isclose(conv_imp[0, 0], 0.0, atol=1e-6)
  assert np.isclose(conv_imp[8, 8], 0.0, atol=1e-6)

  # Corner impulse response at (0, 0) reflects across symmetric border=2
  corner = np.zeros((3, 7, 7), dtype=np.float32)
  corner[0, 0, 0] = 1.0
  conv_corner = np.asarray(convolve_noise_planes(corner))[0]
  # In padded coordinates, (0,0) appears at padded[2,2], padded[1,2], padded[2,1], padded[1,1] (4 times in 5x5 window around (0,0): center + 3 reflected neighbors)
  expected_00 = -3.84 + 3.0 * 0.16
  assert np.isclose(conv_corner[0, 0], expected_00, atol=1e-6)


def test_eval_noise_strength_lut_knots_clamping_and_gradients() -> None:
  lut = jnp.array([0.02, 0.08, 0.15, 0.25, 0.40, 0.55, 0.70, 0.90], dtype=jnp.float32)
  knots = jnp.arange(8, dtype=jnp.float32) / 6.0
  vals = np.asarray(eval_noise_strength_lut(knots, lut))
  np.testing.assert_allclose(vals, np.asarray(lut), atol=1e-6)

  # Midpoints between knots
  mids = (jnp.arange(7, dtype=jnp.float32) + 0.5) / 6.0
  mid_vals = np.asarray(eval_noise_strength_lut(mids, lut))
  expected_mids = 0.5 * (np.asarray(lut[:-1]) + np.asarray(lut[1:]))
  np.testing.assert_allclose(mid_vals, expected_mids, atol=1e-6)

  # Negative input clamps to lut[0], input >= 7/6 clamps to lut[7]
  extremes = jnp.array([-0.5, 0.0, 7.0 / 6.0, 2.5], dtype=jnp.float32)
  ext_vals = np.asarray(eval_noise_strength_lut(extremes, lut))
  np.testing.assert_allclose(
      ext_vals, [float(lut[0]), float(lut[0]), float(lut[7]), float(lut[7])], atol=1e-6
  )

  # STE LUT quantization
  raw_lut = jnp.array([-0.1, 0.0, 0.1234, 0.5, 0.75, 0.9, 0.9999, 1.2], dtype=jnp.float32)
  q_lut = np.asarray(quantize_noise_lut_ste(raw_lut))
  assert bool(np.all((q_lut >= 0.0) & (q_lut <= NOISE_LUT_MAX + 1e-7)))
  np.testing.assert_allclose(q_lut * NOISE_PRECISION, np.round(q_lut * NOISE_PRECISION), atol=1e-5)

  grad_lut = np.asarray(jax.grad(lambda x: jnp.sum(quantize_noise_lut_ste(x)))(raw_lut))
  # Interior entries in (0, NOISE_LUT_MAX) have exact unit STE derivative
  np.testing.assert_allclose(grad_lut[2:6], 1.0, atol=1e-6)


def test_synthesize_jxl_noise_zero_lut_identity_and_layout_invariance() -> None:
  rng = np.random.default_rng(42)
  xyb_chw = jnp.asarray(rng.uniform(0.05, 0.8, size=(3, 48, 64)).astype(np.float32))
  xyb_hwc = jnp.moveaxis(xyb_chw, 0, -1)

  zero_lut = jnp.zeros(8, dtype=jnp.float32)
  out_zero = synthesize_jxl_noise(xyb_chw, zero_lut)
  np.testing.assert_allclose(np.asarray(out_zero), np.asarray(xyb_chw), atol=1e-7)

  lut = jnp.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18], dtype=jnp.float32)
  out_chw = synthesize_jxl_noise(xyb_chw, lut)
  out_hwc = synthesize_jxl_noise(xyb_hwc, lut)
  assert out_chw.shape == (3, 48, 64)
  assert out_hwc.shape == (48, 64, 3)
  np.testing.assert_allclose(
      np.asarray(out_chw), np.moveaxis(np.asarray(out_hwc), -1, 0), atol=1e-6
  )


def test_synthesize_jxl_noise_jit_and_gradients_wrt_xyb_and_lut() -> None:
  rng = np.random.default_rng(7)
  xyb = jnp.asarray(rng.uniform(0.1, 0.6, size=(3, 32, 32)).astype(np.float32))
  target = jnp.asarray(rng.uniform(0.1, 0.6, size=(3, 32, 32)).astype(np.float32))
  lut = jnp.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16], dtype=jnp.float32)

  @jax.jit
  def loss_and_grad(x: jax.Array, l: jax.Array):
    def _loss(x_in: jax.Array, l_in: jax.Array) -> jax.Array:
      noisy = synthesize_jxl_noise(x_in, l_in, ste_quantize_lut=True)
      return jnp.mean(jnp.square(noisy - target))

    return jax.value_and_grad(_loss, argnums=(0, 1))(x, l)

  loss_val, (grad_xyb, grad_lut) = loss_and_grad(xyb, lut)
  assert np.isfinite(float(loss_val))
  assert bool(np.all(np.isfinite(np.asarray(grad_xyb))))
  assert bool(np.all(np.isfinite(np.asarray(grad_lut))))
  assert float(np.linalg.norm(np.asarray(grad_xyb))) > 1e-6
  assert float(np.linalg.norm(np.asarray(grad_lut))) > 1e-5

  # Central finite-difference check on lut
  eps = 1e-3
  fd_grad = np.zeros(8, dtype=np.float32)
  for i in range(8):
    e_vec = jnp.zeros(8, dtype=jnp.float32).at[i].set(eps)
    l_plus = jnp.mean(
        jnp.square(synthesize_jxl_noise(xyb, lut + e_vec, ste_quantize_lut=False) - target)
    )
    l_minus = jnp.mean(
        jnp.square(synthesize_jxl_noise(xyb, lut - e_vec, ste_quantize_lut=False) - target)
    )
    fd_grad[i] = float(l_plus - l_minus) / (2.0 * eps)
  np.testing.assert_allclose(np.asarray(grad_lut), fd_grad, rtol=2e-2, atol=1e-5)


@pytest.mark.parametrize("size_wh", [(128, 96), (67, 53), (300, 190)])
def test_synthesize_jxl_noise_matches_djxl_single_frame_and_odd_multigroup(
    size_wh: tuple[int, int],
) -> None:
  w, h = size_wh
  rgb_01 = _load_kodak_crop("kodim01.png", w=w, h=h)
  state = init_state_from_rgb_jax(rgb_01, distance=1.5)
  noise_lut = [float(k) / 1024.0 for k in (24, 32, 40, 48, 56, 64, 72, 80)]

  jxl_no_noise, _ = encode_single_frame(state, noise_lut=None)
  jxl_with_noise, _ = encode_single_frame(state, noise_lut=noise_lut)

  info = inspect_jxl(jxl_with_noise)
  assert info["frames"][0]["has_noise"] is True
  np.testing.assert_allclose(info["frames"][0]["noise_lut"], noise_lut, atol=1e-6)

  djxl_no_noise = decode_with_djxl(jxl_no_noise).astype(np.float32) / 255.0
  ver_with_noise = verify_jxl_codestream(jxl_with_noise, min_cross_decoder_psnr=45.0)
  djxl_with_noise = ver_with_noise["rgb_djxl"].astype(np.float32) / 255.0

  jax_no_noise = np.asarray(reconstruct_single_frame(state, include_noise=False))
  jax_with_noise = np.asarray(
      reconstruct_single_frame(state, include_noise=True, noise_lut=noise_lut)
  )

  psnr_vs_djxl = compute_psnr(jax_with_noise, djxl_with_noise, peak=1.0)
  assert psnr_vs_djxl >= 45.0, f"Expected PSNR >= 45 dB for {size_wh}, got {psnr_vs_djxl:.2f} dB"

  # Verify the added noise delta is highly correlated with djxl's noise delta
  delta_djxl = (djxl_with_noise - djxl_no_noise).ravel()
  delta_jax = (jax_with_noise - jax_no_noise).ravel()
  corr = float(np.corrcoef(delta_djxl, delta_jax)[0, 1])
  assert corr >= 0.99, f"Expected noise delta corr >= 0.99 for {size_wh}, got {corr:.5f}"


def test_synthesize_jxl_noise_matches_djxl_multiframe_patches_and_alpha() -> None:
  rgb_01 = _load_kodak_crop("kodim02.png", w=128, h=128)
  rgb_low = np.asarray(
      Image.fromarray(np.round(rgb_01 * 255.0).astype(np.uint8)).resize(
          (64, 64), Image.Resampling.BICUBIC
      ),
      dtype=np.float32,
  ) / 255.0

  ref_state = init_state_from_rgb_jax(rgb_low * 0.65, distance=2.0)
  main_state = init_state_from_rgb_jax(rgb_01 * 0.35, distance=1.5)
  noise_lut = [float(k) / 1024.0 for k in (32, 40, 48, 56, 64, 72, 80, 88)]

  # 1. Single full-frame kAdd patch composite with noise
  patches = [
      PatchEntry(ref=0, x0=0, y0=0, xsize=128, ysize=128, x=0, y=0, blend_mode=PATCH_BLEND_ADD)
  ]
  jxl_p_clean, _ = encode_patches_composite(ref_state, main_state, upsampling=2, patches=patches)
  jxl_p_noisy, _ = encode_patches_composite(
      ref_state, main_state, upsampling=2, patches=patches, noise_lut=noise_lut
  )
  djxl_p_clean = decode_with_djxl(jxl_p_clean).astype(np.float32) / 255.0
  ver_p_noisy = verify_jxl_codestream(jxl_p_noisy, min_cross_decoder_psnr=45.0)
  djxl_p_noisy = ver_p_noisy["rgb_djxl"].astype(np.float32) / 255.0

  jax_p_clean = np.asarray(
      reconstruct_patch_composite(ref_state, main_state, upsampling=2, patches=patches, include_noise=False)
  )
  jax_p_noisy = np.asarray(
      reconstruct_patch_composite(
          ref_state,
          main_state,
          upsampling=2,
          patches=patches,
          include_noise=True,
          noise_lut=noise_lut,
          visible_frame_index=1,
          nonvisible_frame_index=0,
      )
  )
  jax_p_wrong_seed = np.asarray(
      reconstruct_patch_composite(
          ref_state,
          main_state,
          upsampling=2,
          patches=patches,
          include_noise=True,
          noise_lut=noise_lut,
          visible_frame_index=0,
          nonvisible_frame_index=1,
      )
  )

  psnr_p = compute_psnr(jax_p_noisy, djxl_p_noisy, peak=1.0)
  assert psnr_p >= 45.0
  corr_p = float(
      np.corrcoef((djxl_p_noisy - djxl_p_clean).ravel(), (jax_p_noisy - jax_p_clean).ravel())[0, 1]
  )
  corr_wrong = float(
      np.corrcoef(
          (djxl_p_noisy - djxl_p_clean).ravel(), (jax_p_wrong_seed - jax_p_clean).ravel()
      )[0, 1]
  )
  assert corr_p >= 0.99
  assert abs(corr_wrong) < 0.05

  # 2. Alpha composite with noise
  alpha_u8 = np.zeros((64, 64), dtype=np.uint8)
  alpha_u8[:, 32:] = 255
  alpha_01 = alpha_u8.astype(np.float32) / 255.0

  jxl_a_clean, _ = encode_alpha_composite(
      ref_state, main_state, upsampling=2, dim_shift=1, alpha_u8=alpha_u8
  )
  jxl_a_noisy, _ = encode_alpha_composite(
      ref_state, main_state, upsampling=2, dim_shift=1, alpha_u8=alpha_u8, noise_lut=noise_lut
  )
  djxl_a_clean = decode_with_djxl(jxl_a_clean).astype(np.float32) / 255.0
  ver_a_noisy = verify_jxl_codestream(jxl_a_noisy, min_cross_decoder_psnr=45.0)
  djxl_a_noisy = ver_a_noisy["rgb_djxl"].astype(np.float32) / 255.0

  jax_a_clean = np.asarray(
      reconstruct_alpha_composite(
          ref_state, main_state, upsampling=2, dim_shift=1, alpha_01=alpha_01, include_noise=False
      )
  )
  jax_a_noisy = np.asarray(
      reconstruct_alpha_composite(
          ref_state,
          main_state,
          upsampling=2,
          dim_shift=1,
          alpha_01=alpha_01,
          include_noise=True,
          noise_lut=noise_lut,
      )
  )
  psnr_a = compute_psnr(jax_a_noisy, djxl_a_noisy, peak=1.0)
  assert psnr_a >= 45.0
  corr_a = float(
      np.corrcoef((djxl_a_noisy - djxl_a_clean).ravel(), (jax_a_noisy - jax_a_clean).ravel())[0, 1]
  )
  assert corr_a >= 0.99
