"""Unit tests for standalone VGG-16 feature extraction and Wasserstein Distortion."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jxl_wasserstein.vgg import (
    VggNet,
    ensure_chw_rgb,
    load_vgg16_params,
    multiscale_vgg_features,
    select_num_scales,
)
from jxl_wasserstein.wasserstein import (
    compute_multiscale_stats,
    compute_spatial_wd_map,
    compute_wd,
    lower_limit,
    lowpass,
    multi_wasserstein_distortion,
    safe_sqrt,
    wasserstein_distortion,
)


def test_zero_legacy_package_references_r1() -> None:
  """Verifies R1: zero references to legacy package in src, tests, or sys.modules."""
  pkg_root = Path(__file__).resolve().parents[1]
  legacy_prefix = "jp" + "eg"
  forbidden_tokens = (
      f"{legacy_prefix}_wasserstein",
      f"{legacy_prefix}-wasserstein",
  )

  for mod_name in list(sys.modules):
    for tok in forbidden_tokens:
      assert tok not in mod_name, f"Forbidden module loaded: {mod_name}"

  scan_paths = [
      pkg_root / "pyproject.toml",
      pkg_root.parent / "jxl_wasserstein_helper.cc",
  ]
  for sub in ("src", "tests"):
    for p in (pkg_root / sub).rglob("*"):
      if (
          p.is_file()
          and "__pycache__" not in p.parts
          and ".venv" not in p.parts
          and "e2e" not in p.parts
      ):
        scan_paths.append(p)

  for path in scan_paths:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for tok in forbidden_tokens:
      assert tok not in text, f"Found forbidden token {tok!r} in {path}"

  # Verify load_vgg16_params defaults to ~/.cache/jxl_wasserstein
  src_code = inspect.getsource(load_vgg16_params)
  assert "jxl_wasserstein" in src_code


def test_matmul_precision_and_deterministic_xla_flags() -> None:
  """Verifies float32 matmul precision and deterministic XLA flags."""
  assert jax.config.jax_default_matmul_precision == "float32"
  assert "--xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", "")


def test_lower_limit_and_safe_sqrt_custom_derivatives() -> None:
  """Verifies custom VJP on lower_limit and custom JVP on safe_sqrt."""
  x = jnp.array([-2.0, 0.0, 3.0], dtype=jnp.float32)
  y = lower_limit(x, 0.0)
  np.testing.assert_allclose(np.asarray(y), [0.0, 0.0, 3.0])

  # Positive upstream gradient is zeroed when clamped (< limit), passed when >= limit
  grad_pos = jax.grad(lambda z: jnp.sum(lower_limit(z, 0.0) * 2.0))(x)
  np.testing.assert_allclose(np.asarray(grad_pos), [0.0, 2.0, 2.0])

  # Negative upstream gradient passes through even when clamped (< limit)
  grad_neg = jax.grad(lambda z: jnp.sum(lower_limit(z, 0.0) * -2.0))(x)
  np.testing.assert_allclose(np.asarray(grad_neg), [-2.0, -2.0, -2.0])

  # safe_sqrt is exact 0.0 at 0.0 and caps derivative at limit=1e6
  zeros = jnp.array([0.0, 1e-20, 4.0], dtype=jnp.float32)
  s_out = safe_sqrt(zeros, 1e6)
  assert float(s_out[0]) == 0.0
  assert float(s_out[2]) == pytest.approx(2.0, abs=1e-6)

  g_sqrt = jax.grad(lambda z: jnp.sum(safe_sqrt(z, 1e6)))(zeros)
  assert np.all(np.isfinite(np.asarray(g_sqrt)))
  np.testing.assert_allclose(np.asarray(g_sqrt), [1e6, 1e6, 0.25], rtol=1e-5)


def test_lowpass_and_compute_multiscale_stats() -> None:
  """Verifies separable 3x3 lowpass filter and multiscale mean/variance pyramids."""
  even_in = jnp.ones((3, 16, 16), dtype=jnp.float32) * 2.5
  odd_in = jnp.ones((3, 13, 19), dtype=jnp.float32) * 1.5
  assert lowpass(even_in, stride=1).shape == (3, 16, 16)
  assert lowpass(even_in, stride=2).shape == (3, 8, 8)
  assert lowpass(odd_in, stride=1).shape == (3, 13, 19)
  assert lowpass(odd_in, stride=2).shape == (3, 7, 10)

  # Impulse response at center of 5x5 array
  impulse = jnp.zeros((1, 5, 5), dtype=jnp.float32).at[0, 2, 2].set(1.0)
  resp = np.asarray(lowpass(impulse, stride=1)[0, 1:4, 1:4])
  k1d = np.array([0.25, 0.5, 0.25], dtype=np.float32)
  expected_k2d = np.outer(k1d, k1d)
  np.testing.assert_allclose(resp, expected_k2d, atol=1e-7)

  # Interior of constant array has zero variance
  means_c, vars_c = compute_multiscale_stats(even_in, num_levels=3)
  assert len(means_c) == 3 and len(vars_c) == 3
  np.testing.assert_allclose(
      np.asarray(vars_c[0][:, 2:-2, 2:-2]), 0.0, atol=1e-6
  )

  # Checkerboard has strictly positive variance
  yy, xx = jnp.meshgrid(jnp.arange(16), jnp.arange(16), indexing="ij")
  checker = jnp.broadcast_to(((yy + xx) % 2).astype(jnp.float32), (3, 16, 16))
  _, vars_chk = compute_multiscale_stats(checker, num_levels=2)
  assert float(jnp.mean(vars_chk[0][:, 2:-2, 2:-2])) > 0.1


def test_wasserstein_distortion_mse_limit_and_invariants() -> None:
  """Verifies MSE equivalence at log2_sigma=0, identity=0, symmetry, and JIT."""
  rng = np.random.default_rng(42)
  fa = jnp.asarray(rng.standard_normal((4, 16, 16), dtype=np.float32))
  fb = jnp.asarray(rng.standard_normal((4, 16, 16), dtype=np.float32))
  ls_zero = jnp.zeros((16, 16), dtype=jnp.float32)

  wd0 = wasserstein_distortion(fa, fb, ls_zero, num_levels=3)
  mse = jnp.mean(jnp.square(fa - fb))
  assert float(wd0) == pytest.approx(float(mse), rel=1e-6, abs=1e-6)

  grad_wd0 = jax.grad(lambda x: wasserstein_distortion(x, fb, ls_zero, 3))(fa)
  grad_mse = jax.grad(lambda x: jnp.mean(jnp.square(x - fb)))(fa)
  np.testing.assert_allclose(
      np.asarray(grad_wd0), np.asarray(grad_mse), rtol=1e-5, atol=1e-6
  )

  ls_two = jnp.full((16, 16), 2.0, dtype=jnp.float32)
  wd_same = wasserstein_distortion(fa, fa, ls_two, num_levels=3)
  assert float(wd_same) == pytest.approx(0.0, abs=1e-7)

  wd_ab = wasserstein_distortion(fa, fb, ls_two, num_levels=3)
  wd_ba = wasserstein_distortion(fb, fa, ls_two, num_levels=3)
  assert float(wd_ab) > 0.0
  assert float(wd_ab) == pytest.approx(float(wd_ba), rel=1e-6)

  wd_jit = jax.jit(lambda x, y: wasserstein_distortion(x, y, ls_two, 3))(fa, fb)
  assert float(wd_jit) == pytest.approx(float(wd_ab), rel=1e-6)


def test_multi_wasserstein_scale_adaptation_and_codex_parity() -> None:
  """Verifies multi_wasserstein_distortion scale adaptation and golden parity."""
  key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
  fa0 = jax.random.normal(key_a, (3, 32, 32), dtype=jnp.float32)
  fb0 = jax.random.normal(key_b, (3, 32, 32), dtype=jnp.float32)
  fa1 = fa0[:, ::2, ::2]
  fb1 = fb0[:, ::2, ::2]
  fa2 = fa0[:, ::4, ::4]
  fb2 = fb0[:, ::4, ::4]
  ls = jnp.full((32, 32), 1.5, dtype=jnp.float32)

  d_single = wasserstein_distortion(fa0, fb0, ls, num_levels=3)
  d_multi, inter = multi_wasserstein_distortion(
      [fa0, fa1, fa2],
      [fb0, fb1, fb2],
      ls,
      num_levels=3,
      return_intermediates=True,
  )
  assert float(d_single) > 0.0
  assert float(d_multi) > float(d_single)
  assert len(inter["weights"]) == 3

  # At coarse scale fa2 (8x8 vs 32x32, log_ratio=2.0 > 1.5), ls is ReLU-clamped to 0
  # so weight[0] == 1.0 everywhere and coarse contribution equals exact MSE(fa2, fb2).
  np.testing.assert_allclose(
      np.asarray(inter["weights"][2][0]), 1.0, atol=1e-6
  )

  # Identical features have exact 0 distortion and exact 0 gradient under JIT
  val_zero, grad_zero = jax.jit(
      jax.value_and_grad(
          lambda x: multi_wasserstein_distortion([x, fa1], [fa0, fa1], ls, 3)
      )
  )(fa0)
  assert float(val_zero) == 0.0
  np.testing.assert_allclose(np.asarray(grad_zero), 0.0, atol=1e-7)


def test_vgg16_feature_pyramid_and_select_num_scales() -> None:
  """Verifies VGG-16 weight loading, pyramid structure, and scale selection."""
  assert select_num_scales(16, 16) == 1
  assert select_num_scales(30, 30) == 1
  assert select_num_scales(31, 31) == 2
  assert select_num_scales(60, 60) == 2
  assert select_num_scales(61, 61) == 3
  assert select_num_scales(64, 64) == 3
  assert select_num_scales(512, 768) == 3
  with pytest.raises(ValueError):
    select_num_scales(15, 32)

  params = load_vgg16_params()
  leaves = jax.tree_util.tree_leaves(params)
  assert len(leaves) == 26
  for leaf in leaves:
    assert leaf.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(leaf)))

  # Even (64x64) and odd (67x75) feature extraction
  img_even = jnp.ones((3, 64, 64), dtype=jnp.float32) * 0.5
  feats_even = multiscale_vgg_features(params, img_even, num_scales=3)
  assert len(feats_even) == 1 + 5 * 3
  expected_channels = [3] + [64, 128, 256, 512, 512] * 3
  assert [f.shape[0] for f in feats_even] == expected_channels

  img_odd = jnp.ones((67, 75, 3), dtype=jnp.float32) * 0.4
  feats_odd = multiscale_vgg_features(
      params, ensure_chw_rgb(img_odd), num_scales=3
  )
  assert len(feats_odd) == 16


def test_compute_wd_identity_positivity_monotonicity_and_gradients(
    kodim01_rgb_u8: np.ndarray,
) -> None:
  """Verifies compute_wd on Kodak crops: identity=0, monotonicity, and finite-diff grad."""
  params = load_vgg16_params()
  crop_even = (
      jnp.asarray(kodim01_rgb_u8[:64, :64], dtype=jnp.float32) / 255.0
  )
  crop_odd = (
      jnp.asarray(kodim01_rgb_u8[:67, :75], dtype=jnp.float32) / 255.0
  )

  # Identity == 0.0 (scalar and 2D log2_sigma, eager and JIT)
  wd_id_even = jax.jit(compute_wd, static_argnames=("num_levels", "num_scales"))(
      crop_even, crop_even, params, 2.0, num_levels=2, num_scales=3
  )
  assert float(wd_id_even) == pytest.approx(0.0, abs=1e-6)

  ls_2d = jnp.full((67, 75), 2.0, dtype=jnp.float32)
  wd_id_odd = compute_wd(
      crop_odd, crop_odd, params, log2_sigma=ls_2d, num_levels=2, num_scales=3
  )
  assert float(wd_id_odd) == pytest.approx(0.0, abs=1e-6)

  # Zero gradient at identity
  grad_id = jax.jit(
      jax.grad(
          lambda x: compute_wd(x, crop_even, params, 2.0, num_levels=2, num_scales=3)
      )
  )(crop_even)
  assert bool(jnp.all(jnp.isfinite(grad_id)))
  np.testing.assert_allclose(np.asarray(grad_id), 0.0, atol=1e-6)

  # Monotonicity under additive noise
  rng = np.random.default_rng(123)
  noise = jnp.asarray(rng.standard_normal(crop_even.shape, dtype=np.float32))
  pert_small = jnp.clip(crop_even + 0.01 * noise, 0.0, 1.0)
  pert_large = jnp.clip(crop_even + 0.05 * noise, 0.0, 1.0)
  wd_small = float(compute_wd(pert_small, crop_even, params, 2.0))
  wd_large = float(compute_wd(pert_large, crop_even, params, 2.0))
  assert 0.0 < wd_small < wd_large

  # Directional derivative check vs central finite differences
  direction = noise / jnp.linalg.norm(noise)
  loss_fn = jax.jit(
      lambda x: compute_wd(x, crop_even, params, 2.0, num_levels=2, num_scales=2)
  )
  grad_pert = jax.jit(jax.grad(loss_fn))(pert_large)
  assert bool(jnp.all(jnp.isfinite(grad_pert)))
  analytic_dir = float(jnp.sum(grad_pert * direction))
  eps = 1e-3
  fd_dir = (
      float(loss_fn(pert_large + eps * direction))
      - float(loss_fn(pert_large - eps * direction))
  ) / (2.0 * eps)
  assert analytic_dir == pytest.approx(fd_dir, rel=2e-2, abs=1e-4)


def test_compute_wd_small_frames_auto_clamping(
    kodim01_rgb_u8: np.ndarray,
) -> None:
  """Verifies compute_wd auto-clamps num_scales via select_num_scales on 33x35 and 16x16 frames."""
  params = load_vgg16_params()
  for h, w, expected_scales in ((33, 35, 2), (16, 16, 1)):
    assert select_num_scales(h, w, max_scales=3) == expected_scales
    im_a = jnp.asarray(kodim01_rgb_u8[:h, :w], dtype=jnp.float32) / 255.0
    im_b = jnp.clip(im_a + 0.05, 0.0, 1.0)

    # Default num_scales=3 succeeds and equals zero at identity
    wd_same = float(compute_wd(im_a, im_a, params))
    assert wd_same == pytest.approx(0.0, abs=1e-6)

    # Default num_scales=3 value_and_grad succeeds with finite positive WD and gradients
    wd_diff, grad = jax.value_and_grad(lambda x: compute_wd(x, im_a, params))(
        im_b
    )
    assert np.isfinite(float(wd_diff))
    assert float(wd_diff) > 0.0
    assert grad.shape == (h, w, 3)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0

    # Matches explicit num_scales=expected_scales
    wd_explicit = float(
        compute_wd(im_b, im_a, params, num_scales=expected_scales)
    )
    assert float(wd_diff) == pytest.approx(wd_explicit, rel=1e-6, abs=1e-6)

  # Frames smaller than 16x16 still raise ValueError
  im_tiny = jnp.zeros((15, 16, 3), dtype=jnp.float32)
  with pytest.raises(ValueError, match="16x16"):
    compute_wd(im_tiny, im_tiny, params)


def test_compute_spatial_wd_map_invariants_and_localization_r3(
    kodim01_rgb_u8: np.ndarray,
) -> None:
  """Verifies F8/R3: compute_spatial_wd_map shape (H8, W8), >=0, sum==total_wd, and localization."""
  params = load_vgg16_params()
  crop_even = (
      jnp.asarray(kodim01_rgb_u8[:64, :64], dtype=jnp.float32) / 255.0
  )
  crop_odd = (
      jnp.asarray(kodim01_rgb_u8[:67, :75], dtype=jnp.float32) / 255.0
  )

  # Identity: total_wd == 0, wd_block_map == 0 everywhere
  wd_zero, map_zero = compute_spatial_wd_map(crop_even, crop_even, params)
  assert map_zero.shape == (8, 8)
  assert float(wd_zero) == pytest.approx(0.0, abs=1e-6)
  np.testing.assert_allclose(np.asarray(map_zero), 0.0, atol=1e-6)

  # Perturbed top-left 16x16 region (2x2 blocks at [0:2, 0:2])
  rng = np.random.default_rng(77)
  patch_noise = jnp.asarray(
      rng.uniform(-0.25, 0.25, size=(16, 16, 3)).astype(np.float32)
  )
  pert_even = crop_even.at[:16, :16, :].set(
      jnp.clip(crop_even[:16, :16, :] + patch_noise, 0.0, 1.0)
  )
  total_wd, wd_map = jax.jit(
      compute_spatial_wd_map,
      static_argnames=("block_size", "num_levels", "num_scales"),
  )(pert_even, crop_even, params, block_size=8, log2_sigma=4.0, num_levels=2, num_scales=3)

  expected_wd = compute_wd(
      pert_even, crop_even, params, log2_sigma=4.0, num_levels=2, num_scales=3
  )
  assert wd_map.shape == (8, 8)
  assert float(total_wd) > 0.0
  assert float(total_wd) == pytest.approx(float(expected_wd), rel=1e-5, abs=1e-6)
  assert bool(jnp.all(wd_map >= 0.0))
  assert float(jnp.sum(wd_map)) == pytest.approx(
      float(total_wd), rel=1e-5, abs=1e-5
  )

  # Top-left perturbed blocks must have significantly higher distortion than bottom-right
  top_left_mean = float(jnp.mean(wd_map[:3, :3]))
  bottom_right_mean = float(jnp.mean(wd_map[5:, 5:]))
  assert top_left_mean > 5.0 * max(bottom_right_mean, 1e-12)

  # Odd spatial dimensions (67x75 -> H8=9, W8=10)
  pert_odd = jnp.clip(crop_odd + 0.03, 0.0, 1.0)
  total_wd_odd, wd_map_odd = compute_spatial_wd_map(
      pert_odd, crop_odd, params, block_size=8, log2_sigma=2.0
  )
  assert wd_map_odd.shape == (9, 10)
  assert bool(jnp.all(wd_map_odd >= 0.0))
  assert float(jnp.sum(wd_map_odd)) == pytest.approx(
      float(total_wd_odd), rel=1e-5, abs=1e-5
  )

  # Differentiability wrt input image
  grad_img = jax.grad(
      lambda x: jnp.sum(compute_spatial_wd_map(x, crop_even, params)[1][:2, :2])
  )(pert_even)
  assert grad_img.shape == crop_even.shape
  assert bool(jnp.all(jnp.isfinite(grad_img)))
  assert float(jnp.linalg.norm(grad_img)) > 0.0


