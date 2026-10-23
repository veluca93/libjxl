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


def _single_wd_chunk(
    fa: jax.Array,
    fb: jax.Array,
    ls_curr: jax.Array,
    num_levels: int,
    sqrt_grad_limit: float,
) -> jax.Array:
  feat_a = fa
  feat_b = fb
  sq_a = jnp.square(feat_a)
  sq_b = jnp.square(feat_b)
  dist = jnp.zeros((), dtype=fa.dtype)
  ls_i = ls_curr

  for i in range(int(num_levels) + 1):
    if i == 0:
      wd_map = jnp.square(fa - fb)
    else:
      ma = lowpass(feat_a, stride=1)
      pa = lowpass(sq_a, stride=1)
      va = pa - jnp.square(ma)
      mb = lowpass(feat_b, stride=1)
      pb = lowpass(sq_b, stride=1)
      vb = pb - jnp.square(mb)
      sa = safe_sqrt(lower_limit(va, 0.0), sqrt_grad_limit)
      sb = safe_sqrt(lower_limit(vb, 0.0), sqrt_grad_limit)
      wd_map = jnp.square(ma - mb) + jnp.square(sa - sb)
      if i < int(num_levels):
        feat_a = ma[..., ::2, ::2]
        sq_a = pa[..., ::2, ::2]
        feat_b = mb[..., ::2, ::2]
        sq_b = pb[..., ::2, ::2]

    weight = jax.nn.relu(1.0 - jnp.abs(ls_i - i))
    dist = dist + jnp.mean(weight * wd_map)
    if i > 0 and i < int(num_levels):
      ls_i = lowpass(ls_i[None], stride=2).squeeze(0)

  return dist


@functools.partial(jax.checkpoint, prevent_cse=False, static_argnums=(3, 4))
def _checkpointed_single_wd(
    fa: jax.Array,
    fb: jax.Array,
    ls_base: jax.Array,
    num_levels: int,
    sqrt_grad_limit: float,
) -> jax.Array:
  """Checkpointed scalar WD for a single feature pair, chunked across channels."""
  ls_curr = jax.image.resize(
      ls_base, fa.shape[-2:], method="linear", antialias=True
  )
  log_ratio_h = jnp.log2(ls_base.shape[-2] / fa.shape[-2])
  log_ratio_w = jnp.log2(ls_base.shape[-1] / fa.shape[-1])
  ls_curr = jax.nn.relu(ls_curr - 0.5 * (log_ratio_h + log_ratio_w))

  c_dim, fh, fw = fa.shape
  if c_dim > 8 and c_dim % 8 == 0 and fh * fw >= 65536:
    n_chunks = c_dim // 8
    fa_chunks = fa.reshape(n_chunks, 8, fh, fw)
    fb_chunks = fb.reshape(n_chunks, 8, fh, fw)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def _scan_body(acc: jax.Array, pair: tuple[jax.Array, jax.Array]):
      ca, cb = pair
      d_c = _single_wd_chunk(ca, cb, ls_curr, num_levels, sqrt_grad_limit)
      return acc + d_c, None

    total_c, _ = jax.lax.scan(
        _scan_body,
        jnp.zeros((), dtype=fa.dtype),
        (fa_chunks, fb_chunks),
    )
    return total_c / float(n_chunks)

  return _single_wd_chunk(fa, fb, ls_curr, num_levels, sqrt_grad_limit)


@functools.partial(
    jax.checkpoint, prevent_cse=False, static_argnums=(3, 4, 5, 6, 7)
)
def _checkpointed_single_spatial_wd(
    fa: jax.Array,
    fb: jax.Array,
    ls_base: jax.Array,
    h8: int,
    w8: int,
    num_levels: int,
    sqrt_grad_limit: float,
    add_blk0_w2: bool,
) -> tuple[jax.Array, jax.Array]:
  """Checkpointed scalar + `(h8, w8)` spatial WD contribution, chunked across channels."""
  ls_curr = jax.image.resize(
      ls_base, fa.shape[-2:], method="linear", antialias=True
  )
  log_ratio_h = jnp.log2(ls_base.shape[-2] / fa.shape[-2])
  log_ratio_w = jnp.log2(ls_base.shape[-1] / fa.shape[-1])
  ls_curr = jax.nn.relu(ls_curr - 0.5 * (log_ratio_h + log_ratio_w))

  c_dim, fh, fw = fa.shape
  num_blocks = float(max(1, h8 * w8))

  def _spatial_levels_mean_2d(
      ca: jax.Array, cb: jax.Array
  ) -> list[jax.Array]:
    feat_a = ca
    feat_b = cb
    sq_a = jnp.square(feat_a)
    sq_b = jnp.square(feat_b)
    means_2d: list[jax.Array] = []
    for i in range(int(num_levels) + 1):
      if i == 0:
        wd_m = jnp.square(ca - cb)
      else:
        ma = lowpass(feat_a, stride=1)
        pa = lowpass(sq_a, stride=1)
        va = pa - jnp.square(ma)
        mb = lowpass(feat_b, stride=1)
        pb = lowpass(sq_b, stride=1)
        vb = pb - jnp.square(mb)
        sa = safe_sqrt(lower_limit(va, 0.0), sqrt_grad_limit)
        sb = safe_sqrt(lower_limit(vb, 0.0), sqrt_grad_limit)
        wd_m = jnp.square(ma - mb) + jnp.square(sa - sb)
        if i < int(num_levels):
          feat_a = ma[..., ::2, ::2]
          sq_a = pa[..., ::2, ::2]
          feat_b = mb[..., ::2, ::2]
          sq_b = pb[..., ::2, ::2]
      means_2d.append(jnp.mean(wd_m, axis=0))
    return means_2d

  if c_dim > 8 and c_dim % 8 == 0 and fh * fw >= 65536:
    n_chunks = c_dim // 8
    fa_chunks = fa.reshape(n_chunks, 8, fh, fw)
    fb_chunks = fb.reshape(n_chunks, 8, fh, fw)
    init_m2d = [
        jnp.zeros((fh >> max(0, i - 1), fw >> max(0, i - 1)), dtype=fa.dtype)
        for i in range(int(num_levels) + 1)
    ]
    # Note: stride-2 slicing `ma[..., ::2, ::2]` has ceil(dim / 2) shape:
    init_m2d = []
    sh_i, sw_i = fh, fw
    for i in range(int(num_levels) + 1):
      if i >= 2:
        sh_i = (sh_i + 1) // 2
        sw_i = (sw_i + 1) // 2
      init_m2d.append(jnp.zeros((sh_i, sw_i), dtype=fa.dtype))

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def _scan_sp(acc_m2d: list[jax.Array], pair: tuple[jax.Array, jax.Array]):
      ca, cb = pair
      cur_m2d = _spatial_levels_mean_2d(ca, cb)
      return [a + b for a, b in zip(acc_m2d, cur_m2d)], None

    sum_m2d, _ = jax.lax.scan(
        _scan_sp, init_m2d, (fa_chunks, fb_chunks)
    )
    level_means_2d = [m / float(n_chunks) for m in sum_m2d]
  else:
    level_means_2d = _spatial_levels_mean_2d(fa, fb)

  dist = jnp.zeros((), dtype=fa.dtype)
  blk_acc = jnp.zeros((h8, w8), dtype=jnp.float32)
  ls_i = ls_curr
  for i, mean_wd_2d in enumerate(level_means_2d):
    weight = jax.nn.relu(1.0 - jnp.abs(ls_i - i))
    spatial_2d = weight * mean_wd_2d
    term_scalar = jnp.mean(spatial_2d)
    dist = dist + term_scalar
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
    blk_acc = blk_acc + (resized * norm_scale + term_offset)
    if i > 0 and i < int(num_levels):
      ls_i = lowpass(ls_i[None], stride=2).squeeze(0)

  if add_blk0_w2:
    pad_h = h8 * 8 - fh
    pad_w = w8 * 8 - fw
    f_r = (
        jnp.pad(fa, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        if (pad_h > 0 or pad_w > 0)
        else fa
    )
    f_t = (
        jnp.pad(fb, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        if (pad_h > 0 or pad_w > 0)
        else fb
    )
    f_r_blk = jnp.transpose(f_r.reshape(c_dim, h8, 8, w8, 8), (1, 3, 0, 2, 4))
    f_t_blk = jnp.transpose(f_t.reshape(c_dim, h8, 8, w8, 8), (1, 3, 0, 2, 4))
    mu_r = jnp.mean(f_r_blk, axis=(3, 4))
    mu_t = jnp.mean(f_t_blk, axis=(3, 4))
    sigma_r = jnp.sqrt(jnp.var(f_r_blk, axis=(3, 4)) + 1e-7)
    sigma_t = jnp.sqrt(jnp.var(f_t_blk, axis=(3, 4)) + 1e-7)
    w2_blk = jnp.mean(
        jnp.square(mu_r - mu_t) + jnp.square(sigma_r - sigma_t), axis=2
    )
    blk_acc = blk_acc + (0.25 / num_blocks) * w2_blk

  return dist, blk_acc


@functools.partial(jax.checkpoint, prevent_cse=False, static_argnums=(4, 5, 6))
def _checkpointed_vgg_block_wd_target(
    nhwc_in: jax.Array,
    block_params: Any,
    target_feat: jax.Array,
    ls_map: jax.Array,
    num_features: int,
    num_layers: int,
    num_levels: int,
) -> tuple[jax.Array, jax.Array]:
  import flax.linen as nn
  from jxl_wasserstein.vgg import VggBlock

  blk_out = VggBlock(num_features=num_features, num_layers=num_layers).apply(
      {"params": block_params}, nhwc_in
  )
  fa = jnp.transpose(jnp.squeeze(blk_out, axis=0), (2, 0, 1))
  d = _checkpointed_single_wd(fa, target_feat, ls_map, int(num_levels), 1e6)
  pooled = nn.avg_pool(
      blk_out, window_shape=(2, 2), strides=(2, 2), padding="VALID"
  )
  return pooled, d


@functools.partial(
    jax.checkpoint, prevent_cse=False, static_argnums=(4, 5, 6, 7, 8, 9)
)
def _checkpointed_vgg_block_spatial_wd_target(
    nhwc_in: jax.Array,
    block_params: Any,
    target_feat: jax.Array,
    ls_map: jax.Array,
    h8: int,
    w8: int,
    num_features: int,
    num_layers: int,
    num_levels: int,
    add_blk0_w2: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  import flax.linen as nn
  from jxl_wasserstein.vgg import VggBlock

  blk_out = VggBlock(num_features=num_features, num_layers=num_layers).apply(
      {"params": block_params}, nhwc_in
  )
  fa = jnp.transpose(jnp.squeeze(blk_out, axis=0), (2, 0, 1))
  d, blk_c = _checkpointed_single_spatial_wd(
      fa, target_feat, ls_map, h8, w8, int(num_levels), 1e6, bool(add_blk0_w2)
  )
  pooled = nn.avg_pool(
      blk_out, window_shape=(2, 2), strides=(2, 2), padding="VALID"
  )
  return pooled, d, blk_c


def stream_vgg_wd_against_targets(
    vgg_params: Any,
    recon_chw: jax.Array,
    target_features: Sequence[jax.Array],
    ls_map: jax.Array,
    eff_scales: int,
    num_levels: int,
) -> jax.Array:
  """Streams multi-scale VGG-16 + WD block-by-block with rematerialization."""
  from jxl_wasserstein.vgg import (
      _VGG_BLOCKS,
      IMAGENET_MEAN,
      IMAGENET_STD,
      ensure_chw_rgb,
      multiscale_vgg_features,
  )

  if not isinstance(vgg_params, dict) or "params" not in vgg_params:
    recon_feats = multiscale_vgg_features(
        vgg_params, recon_chw, num_scales=eff_scales
    )
    return multi_wasserstein_distortion(
        recon_feats, target_features, ls_map, num_levels=num_levels
    )

  image = ensure_chw_rgb(jnp.asarray(recon_chw, dtype=jnp.float32))
  mean = jnp.array(IMAGENET_MEAN, dtype=image.dtype)[:, None, None]
  std = jnp.array(IMAGENET_STD, dtype=image.dtype)[:, None, None]
  image = (image - mean) / std

  dist = _checkpointed_single_wd(
      image, target_features[0], ls_map, int(num_levels), 1e6
  )
  p_blocks = vgg_params["params"]
  for scale_idx in range(int(eff_scales)):
    if scale_idx > 0:
      image = lowpass(image, stride=2)
    nhwc = jnp.transpose(image, (1, 2, 0))[None, ...]
    for blk_idx, (num_features, num_layers) in enumerate(_VGG_BLOCKS):
      feat_idx = 1 + scale_idx * 5 + blk_idx
      nhwc, d_blk = _checkpointed_vgg_block_wd_target(
          nhwc,
          p_blocks[f"VggBlock_{blk_idx}"],
          target_features[feat_idx],
          ls_map,
          num_features,
          num_layers,
          int(num_levels),
      )
      dist = dist + d_blk
  return dist


def stream_vgg_spatial_wd_against_targets(
    vgg_params: Any,
    recon_chw: jax.Array,
    target_features: Sequence[jax.Array],
    ls_map: jax.Array,
    h8: int,
    w8: int,
    eff_scales: int,
    num_levels: int,
) -> tuple[jax.Array, jax.Array]:
  """Streams multi-scale VGG-16 + `(h8, w8)` spatial WD map block-by-block."""
  from jxl_wasserstein.vgg import (
      _VGG_BLOCKS,
      IMAGENET_MEAN,
      IMAGENET_STD,
      ensure_chw_rgb,
  )

  image = ensure_chw_rgb(jnp.asarray(recon_chw, dtype=jnp.float32))
  mean = jnp.array(IMAGENET_MEAN, dtype=image.dtype)[:, None, None]
  std = jnp.array(IMAGENET_STD, dtype=image.dtype)[:, None, None]
  image = (image - mean) / std

  wd_total, wd_block_map = _checkpointed_single_spatial_wd(
      image, target_features[0], ls_map, h8, w8, int(num_levels), 1e6, False
  )
  p_blocks = vgg_params["params"]
  for scale_idx in range(int(eff_scales)):
    if scale_idx > 0:
      image = lowpass(image, stride=2)
    nhwc = jnp.transpose(image, (1, 2, 0))[None, ...]
    for blk_idx, (num_features, num_layers) in enumerate(_VGG_BLOCKS):
      feat_idx = 1 + scale_idx * 5 + blk_idx
      add_w2 = bool(int(num_levels) < 2 and scale_idx == 0 and blk_idx == 0)
      nhwc, d_blk, blk_c = _checkpointed_vgg_block_spatial_wd_target(
          nhwc,
          p_blocks[f"VggBlock_{blk_idx}"],
          target_features[feat_idx],
          ls_map,
          h8,
          w8,
          num_features,
          num_layers,
          int(num_levels),
          add_w2,
      )
      wd_total = wd_total + d_blk
      wd_block_map = wd_block_map + blk_c

  num_blocks = float(max(1, h8 * w8))
  wd_block_map_pos = jnp.maximum(wd_block_map, 0.0)
  map_sum = jnp.sum(wd_block_map_pos)
  total_scale = jax.lax.stop_gradient(
      jnp.where(
          map_sum > 1e-12,
          wd_total / jnp.maximum(map_sum, 1e-12),
          1.0,
      )
  )
  total_offset = jax.lax.stop_gradient(
      jnp.where(
          map_sum > 1e-12,
          wd_block_map_pos - wd_block_map,
          jnp.full((h8, w8), wd_total / num_blocks, dtype=jnp.float32)
          - wd_block_map,
      )
  )
  wd_map = (wd_block_map + total_offset) * total_scale
  return wd_total, wd_map


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

  if not return_intermediates:
    dist = jnp.zeros((), dtype=ls_base.dtype)
    for fa, fb in zip(features_a, features_b):
      if fa.shape != fb.shape:
        raise ValueError(
            f"Feature arrays must have identical shapes, got {fa.shape} vs {fb.shape}"
        )
      dist = dist + _checkpointed_single_wd(
          fa, fb, ls_base, int(num_levels), float(sqrt_grad_limit)
      )
    return dist

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


@functools.partial(jax.jit, static_argnames=("eff_scales", "num_levels"))
def _jit_compute_wd_impl(
    img_a: jax.Array,
    img_b: jax.Array,
    vgg_params: Any,
    ls_map: jax.Array,
    eff_scales: int,
    num_levels: int,
) -> jax.Array:
  from jxl_wasserstein.vgg import multiscale_vgg_features

  features_a = multiscale_vgg_features(
      vgg_params, jnp.transpose(img_a, (2, 0, 1)), num_scales=eff_scales
  )
  features_b = multiscale_vgg_features(
      vgg_params, jnp.transpose(img_b, (2, 0, 1)), num_scales=eff_scales
  )
  return multi_wasserstein_distortion(
      features_a, features_b, ls_map, num_levels=num_levels
  )


def compute_wd(
    image_a_01: jax.Array,
    image_b_01: jax.Array,
    vgg_params: Any,
    log2_sigma: float | jax.Array = 1.0,
    num_levels: int = 2,
    num_scales: int = 3,
) -> jax.Array:
  """Computes multi-scale VGG-16 Wasserstein Distortion on (H, W, 3) sRGB [0, 1] images."""
  from jxl_wasserstein.vgg import select_num_scales

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
  return _jit_compute_wd_impl(
      img_a, img_b, vgg_params, ls_map, int(eff_scales), int(num_levels)
  )


def compute_spatial_wd_map(
    img_a: jax.Array,
    img_b: jax.Array,
    vgg_params: Any,
    block_size: int = 8,
    log2_sigma: float | jax.Array = 1.0,
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

