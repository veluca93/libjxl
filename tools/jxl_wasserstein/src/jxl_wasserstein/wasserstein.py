"""Multi-scale Wasserstein Distortion metric in JAX (arXiv:2412.00505 / CoDeX parity)."""

from __future__ import annotations

import collections
from collections.abc import Sequence
import functools
from typing import Any

import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")


@jax.custom_vjp
def lower_limit(inputs: jax.Array, limit: float | jax.Array) -> jax.Array:
  """Clamps `inputs` from below at `limit`, passing negative upstream gradients."""
  return jnp.maximum(inputs, limit)


def _lower_limit_fwd(
    inputs: jax.Array,
    limit: float | jax.Array,
) -> tuple[jax.Array, tuple[jax.Array]]:
  return jnp.maximum(inputs, limit), (inputs >= limit,)


def _lower_limit_bwd(
    res: tuple[jax.Array],
    grad: jax.Array,
) -> tuple[jax.Array, None]:
  (limit_inactive,) = res
  pass_through_if = jnp.logical_or(limit_inactive, grad < 0.0)
  return jnp.where(pass_through_if, grad, 0.0), None


lower_limit.defvjp(_lower_limit_fwd, _lower_limit_bwd)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def safe_sqrt(x: jax.Array, limit: float = 1e6) -> jax.Array:
  """Computes `jnp.sqrt(x)` with exact zero forward and capped JVP derivative."""
  del limit
  return jnp.sqrt(x)


@safe_sqrt.defjvp
def _safe_sqrt_jvp(
    limit: float,
    primals: tuple[jax.Array],
    tangents: tuple[jax.Array],
) -> tuple[jax.Array, jax.Array]:
  (x,) = primals
  (x_dot,) = tangents
  sqrt_dot = jnp.minimum(0.5 / jnp.sqrt(x), limit) * x_dot
  return safe_sqrt(x, limit), sqrt_dot


def lowpass(inputs: jax.Array, stride: int) -> jax.Array:
  """Applies a separable 3x3 [0.25, 0.5, 0.25] lowpass filter on (C, H, W) arrays."""
  arr = jnp.asarray(inputs)
  if arr.ndim != 3:
    raise ValueError(f"Expected 3D (C, H, W) array for lowpass, got {arr.shape}")
  if stride < 1:
    raise ValueError(f"stride must be >= 1, got {stride}")
  kernel_1d = jnp.array([0.25, 0.5, 0.25], dtype=arr.dtype)
  kernel_2d = jnp.outer(kernel_1d, kernel_1d)[None, None]
  return jax.lax.conv(
      jnp.expand_dims(arr, 1),
      kernel_2d,
      window_strides=(stride, stride),
      padding="same",
      precision=jax.lax.Precision.HIGHEST,
  ).squeeze(1)


def compute_multiscale_stats(
    features: jax.Array,
    num_levels: int,
) -> tuple[list[jax.Array], list[jax.Array]]:
  """Computes multi-scale local means and variances of a (C, H, W) feature array."""
  feat = jnp.asarray(features)
  if feat.ndim != 3:
    raise ValueError(f"Expected 3D (C, H, W) features, got {feat.shape}")
  if num_levels < 0:
    raise ValueError(f"num_levels must be >= 0, got {num_levels}")
  squared = jnp.square(feat)
  means: list[jax.Array] = []
  variances: list[jax.Array] = []
  for _ in range(num_levels):
    m = lowpass(feat, stride=1)
    p = lowpass(squared, stride=1)
    means.append(m)
    variances.append(p - jnp.square(m))
    feat = m[..., ::2, ::2]
    squared = p[..., ::2, ::2]
  return means, variances


def wasserstein_distortion(
    fa: jax.Array,
    fb: jax.Array,
    log2_sigma: jax.Array,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_intermediates: bool = False,
) -> jax.Array | tuple[jax.Array, dict[str, Any]]:
  """Evaluates Wasserstein Distortion between two (C, H, W) feature arrays."""
  fa_arr = jnp.asarray(fa)
  fb_arr = jnp.asarray(fb)
  ls_curr = jnp.asarray(log2_sigma)
  if fa_arr.shape != fb_arr.shape:
    raise ValueError(
        f"`fa` and `fb` must have same shape, got {fa_arr.shape} vs {fb_arr.shape}"
    )
  if fa_arr.ndim != 3:
    raise ValueError(f"Expected 3D (C, H, W) feature arrays, got {fa_arr.shape}")
  if ls_curr.shape != fa_arr.shape[-2:]:
    raise ValueError(
        f"`fa` and `log2_sigma` must have same spatial shape, got "
        f"{fa_arr.shape[-2:]} vs {ls_curr.shape}"
    )
  if num_levels < 0:
    raise ValueError(f"num_levels must be >= 0, got {num_levels}")

  means_a, variances_a = compute_multiscale_stats(fa_arr, num_levels)
  means_b, variances_b = compute_multiscale_stats(fb_arr, num_levels)

  wd_maps = [jnp.square(fa_arr - fb_arr)]
  for ma, mb, va, vb in zip(means_a, means_b, variances_a, variances_b):
    va_clamped = lower_limit(va, 0.0)
    vb_clamped = lower_limit(vb, 0.0)
    sa = safe_sqrt(va_clamped, sqrt_grad_limit)
    sb = safe_sqrt(vb_clamped, sqrt_grad_limit)
    wd_maps.append(jnp.square(ma - mb) + jnp.square(sa - sb))

  dist = jnp.zeros((), dtype=fa_arr.dtype)
  intermediates: dict[str, Any] = collections.defaultdict(list)
  intermediates["wd_maps"] = wd_maps
  for i, wd_map in enumerate(wd_maps):
    weight = jax.nn.relu(1.0 - jnp.abs(ls_curr - i))
    intermediates["weights"].append(weight)
    dist = dist + jnp.mean(weight * wd_map)
    if i > 0:
      ls_curr = lowpass(ls_curr[None], stride=2).squeeze(0)

  if return_intermediates:
    return dist, dict(intermediates)
  return dist


def multi_wasserstein_distortion(
    features_a: Sequence[jax.Array],
    features_b: Sequence[jax.Array],
    log2_sigma: jax.Array,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_intermediates: bool = False,
) -> jax.Array | tuple[jax.Array, dict[str, Any]]:
  """Sums Wasserstein Distortion across multiple feature arrays of two images."""
  if len(features_a) != len(features_b):
    raise ValueError(
        f"`features_a` and `features_b` must have same length, got "
        f"{len(features_a)} vs {len(features_b)}"
    )
  if len(features_a) == 0:
    raise ValueError("`features_a` and `features_b` must be non-empty")
  ls_base = jnp.asarray(log2_sigma)
  if ls_base.ndim != 2:
    raise ValueError(f"Expected 2D (H, W) `log2_sigma`, got {ls_base.shape}")

  dist = jnp.zeros((), dtype=ls_base.dtype)
  intermediates: dict[str, list[Any]] = collections.defaultdict(list)
  for fa, fb in zip(features_a, features_b):
    if fa.shape != fb.shape:
      raise ValueError(
          f"Feature arrays must have identical shapes, got {fa.shape} vs {fb.shape}"
      )
    ls = jax.image.resize(
        ls_base, fa.shape[-2:], method="linear", antialias=True
    )
    log_ratio_h = jnp.log2(ls_base.shape[-2] / fa.shape[-2])
    log_ratio_w = jnp.log2(ls_base.shape[-1] / fa.shape[-1])
    mean_log_ratio = (log_ratio_h + log_ratio_w) / 2.0
    ls = jax.nn.relu(ls - mean_log_ratio)
    d, s = wasserstein_distortion(
        fa,
        fb,
        ls,
        num_levels=num_levels,
        sqrt_grad_limit=sqrt_grad_limit,
        return_intermediates=True,
    )
    dist = dist + d
    for k, v in s.items():
      intermediates[k].append(v)

  if return_intermediates:
    return dist, dict(intermediates)
  return dist


def compute_wd(
    image_a_01: jax.Array,
    image_b_01: jax.Array,
    vgg_params: Any,
    log2_sigma: float | jax.Array = 2.0,
    num_levels: int = 2,
    num_scales: int = 3,
) -> jax.Array:
  """Computes multi-scale VGG-16 Wasserstein Distortion on (H, W, 3) sRGB [0, 1] images."""
  from jxl_wasserstein.vgg import multiscale_vgg_features, select_num_scales

  img_a = jnp.asarray(image_a_01, dtype=jnp.float32)
  img_b = jnp.asarray(image_b_01, dtype=jnp.float32)
  if img_a.ndim != 3 or img_a.shape[2] != 3:
    raise ValueError(f"Expected image_a_01 shape (H, W, 3), got {img_a.shape}")
  if img_a.shape != img_b.shape:
    raise ValueError(
        f"image_a_01 and image_b_01 must have same shape, got {img_a.shape} vs {img_b.shape}"
    )
  if num_levels < 0:
    raise ValueError(f"num_levels must be >= 0, got {num_levels}")
  if num_scales < 1:
    raise ValueError(f"num_scales must be >= 1, got {num_scales}")

  h, w = img_a.shape[0], img_a.shape[1]
  ls_arr = jnp.asarray(log2_sigma, dtype=jnp.float32)
  if ls_arr.ndim == 0:
    ls_map = jnp.full((h, w), ls_arr, dtype=jnp.float32)
  elif ls_arr.ndim == 2:
    if ls_arr.shape != (h, w):
      raise ValueError(
          f"2D log2_sigma shape {ls_arr.shape} must match image (H, W) = {(h, w)}"
      )
    ls_map = ls_arr
  else:
    raise ValueError(
        f"log2_sigma must be scalar or 2D (H, W), got shape {ls_arr.shape}"
    )

  eff_scales = select_num_scales(h, w, max_scales=num_scales)
  features_a = multiscale_vgg_features(
      vgg_params, jnp.transpose(img_a, (2, 0, 1)), num_scales=eff_scales
  )
  features_b = multiscale_vgg_features(
      vgg_params, jnp.transpose(img_b, (2, 0, 1)), num_scales=eff_scales
  )
  return multi_wasserstein_distortion(
      features_a, features_b, ls_map, num_levels=num_levels
  )


def compute_spatial_wd_map(
    img_a: jax.Array,
    img_b: jax.Array,
    vgg_params: Any,
    block_size: int = 8,
    log2_sigma: float | jax.Array = 4.0,
    num_levels: int = 2,
    num_scales: int = 3,
) -> tuple[jax.Array, jax.Array]:
  """Computes total scalar WD and a non-negative per-block `(H8, W8)` spatial WD map.

  Projects all multi-scale VGG feature and pooling-level Wasserstein Distortion
  maps onto the `(ceil(H / block_size), ceil(W / block_size))` block grid such
  that `jnp.sum(wd_block_map)` equals `total_wd`.
  """
  from jxl_wasserstein.vgg import multiscale_vgg_features, select_num_scales

  arr_a = jnp.asarray(img_a, dtype=jnp.float32)
  arr_b = jnp.asarray(img_b, dtype=jnp.float32)
  if arr_a.ndim != 3 or arr_a.shape[2] != 3:
    raise ValueError(f"Expected img_a shape (H, W, 3), got {arr_a.shape}")
  if arr_a.shape != arr_b.shape:
    raise ValueError(
        f"img_a and img_b must have same shape, got {arr_a.shape} vs {arr_b.shape}"
    )
  if block_size < 1:
    raise ValueError(f"block_size must be >= 1, got {block_size}")
  if num_levels < 0:
    raise ValueError(f"num_levels must be >= 0, got {num_levels}")
  if num_scales < 1:
    raise ValueError(f"num_scales must be >= 1, got {num_scales}")

  h, w = arr_a.shape[0], arr_a.shape[1]
  h8 = (h + block_size - 1) // block_size
  w8 = (w + block_size - 1) // block_size

  ls_arr = jnp.asarray(log2_sigma, dtype=jnp.float32)
  if ls_arr.ndim == 0:
    ls_map = jnp.full((h, w), ls_arr, dtype=jnp.float32)
  elif ls_arr.ndim == 2:
    if ls_arr.shape != (h, w):
      raise ValueError(
          f"2D log2_sigma shape {ls_arr.shape} must match image (H, W) = {(h, w)}"
      )
    ls_map = ls_arr
  else:
    raise ValueError(
        f"log2_sigma must be scalar or 2D (H, W), got shape {ls_arr.shape}"
    )

  eff_scales = select_num_scales(h, w, max_scales=num_scales)
  features_a = multiscale_vgg_features(
      vgg_params, jnp.transpose(arr_a, (2, 0, 1)), num_scales=eff_scales
  )
  features_b = multiscale_vgg_features(
      vgg_params, jnp.transpose(arr_b, (2, 0, 1)), num_scales=eff_scales
  )
  total_wd, intermediates = multi_wasserstein_distortion(
      features_a,
      features_b,
      ls_map,
      num_levels=num_levels,
      return_intermediates=True,
  )

  wd_block_map = jnp.zeros((h8, w8), dtype=jnp.float32)
  num_blocks = float(max(1, h8 * w8))
  for wd_maps_f, weights_f in zip(
      intermediates["wd_maps"], intermediates["weights"]
  ):
    for wd_map, weight in zip(wd_maps_f, weights_f):
      spatial_2d = weight * jnp.mean(wd_map, axis=0)
      term_scalar = jnp.mean(spatial_2d)
      resized = jax.image.resize(
          spatial_2d, (h8, w8), method="linear", antialias=True
      )
      resized_pos = jnp.maximum(resized, 0.0)
      resized_sum = jnp.sum(resized_pos)
      norm_scale = jax.lax.stop_gradient(
          jnp.where(
              resized_sum > 1e-12,
              term_scalar / jnp.maximum(resized_sum, 1e-12),
              1.0 / num_blocks,
          )
      )
      term_offset = jax.lax.stop_gradient(
          jnp.where(resized_sum > 1e-12, 0.0, term_scalar / num_blocks)
      )
      term_block = resized * norm_scale + term_offset
      wd_block_map = wd_block_map + term_block

  wd_block_map_pos = jnp.maximum(wd_block_map, 0.0)
  map_sum = jnp.sum(wd_block_map_pos)
  total_scale = jax.lax.stop_gradient(
      jnp.where(
          map_sum > 1e-12,
          total_wd / jnp.maximum(map_sum, 1e-12),
          1.0,
      )
  )
  total_offset = jax.lax.stop_gradient(
      jnp.where(
          map_sum > 1e-12,
          wd_block_map_pos - wd_block_map,
          jnp.full((h8, w8), total_wd / num_blocks, dtype=jnp.float32)
          - wd_block_map,
      )
  )
  wd_block_map = (wd_block_map + total_offset) * total_scale
  return total_wd, wd_block_map


__all__ = [
    "compute_multiscale_stats",
    "compute_spatial_wd_map",
    "compute_wd",
    "lower_limit",
    "lowpass",
    "multi_wasserstein_distortion",
    "safe_sqrt",
    "wasserstein_distortion",
]

