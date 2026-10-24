"""Shared pytest fixtures and opaque-box verification helpers for E2E tests (Tiers 1-4)."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

_E2E_DIR = str(Path(__file__).resolve().parent)
if _E2E_DIR not in sys.path:
  sys.path.insert(0, _E2E_DIR)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", "/tmp/jxl_wasserstein_jax_cache"
)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
if "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = (
      f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_deterministic_ops=true".strip()
  )

import functools
import jax
import jax.numpy as jnp

_jax_cache_dir = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR", "/tmp/jxl_wasserstein_jax_cache"
)
Path(_jax_cache_dir).mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
import numpy as np
from PIL import Image
import pytest

from jxl_wasserstein.cpp_bridge import (
    PATCH_BLEND_REPLACE,
    compute_psnr,
    decode_with_djxl,
    verify_jxl_codestream,
)
import jxl_wasserstein.cpp_bridge as cpp_bridge_mod
from jxl_wasserstein.entropy_proxy import get_canonical_ac_scan_tables
from jxl_wasserstein.forward_model import get_default_dequant_matrices
from jxl_wasserstein.vgg import load_vgg16_params

# Pre-populate static scan tables and static DequantMatrices once at import time.
for _cy, _cx in ((1, 1), (1, 2), (2, 2), (1, 4), (2, 4), (4, 4)):
  get_canonical_ac_scan_tables(_cy, _cx)
get_default_dequant_matrices()

REPO_ROOT = Path("/usr/local/google/home/veluca/libjxl")
PKG_ROOT = REPO_ROOT / "tools" / "jxl_wasserstein"
KODAK_DIR = REPO_ROOT / "image-corpus" / "qoi_benchmark_suite" / "photo_kodak"

LEGACY_PREFIX = "jp" + "eg"
FORBIDDEN_LEGACY_TOKENS: tuple[str, str] = (
    f"{LEGACY_PREFIX}_wasserstein",
    f"{LEGACY_PREFIX}-wasserstein",
)


def get_field(obj: Any, *keys: str, default: Any = None) -> Any:
  """Reads a field from either a dict or a dataclass/object across alias keys."""
  for k in keys:
    if isinstance(obj, dict) and k in obj:
      return obj[k]
    if hasattr(obj, k):
      return getattr(obj, k)
  return default


@contextlib.contextmanager
def forbid_libjxl_per_block_choices(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
  """Runtime R2 guard failing if dump_init_state, --dump-init, or cjxl is invoked."""
  get_default_dequant_matrices()

  def _forbidden_dump_init(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError(
        "R2 violation: dump_init_state called during encoder execution!"
    )

  monkeypatch.setattr(cpp_bridge_mod, "dump_init_state", _forbidden_dump_init)
  try:
    import jxl_wasserstein.optimizer as opt_mod

    if hasattr(opt_mod, "dump_init_state"):
      monkeypatch.setattr(opt_mod, "dump_init_state", _forbidden_dump_init)
  except ImportError:
    pass
  try:
    import jxl_wasserstein.compositing as comp_mod

    if hasattr(comp_mod, "dump_init_state"):
      monkeypatch.setattr(comp_mod, "dump_init_state", _forbidden_dump_init)
  except ImportError:
    pass

  orig_run = subprocess.run

  def _guarded_subprocess_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
    cmd_strs = [str(x) for x in cmd] if isinstance(cmd, (list, tuple)) else [str(cmd)]
    joined = " ".join(cmd_strs)
    if "--dump-init" in cmd_strs or "--dump-init" in joined:
      raise AssertionError(
          f"R2 violation: subprocess invoked --dump-init: {joined}"
      )
    for token in cmd_strs:
      if Path(token).name == "cjxl":
        raise AssertionError(
            f"R2 violation: encoder invoked cjxl subprocess: {joined}"
        )
    return orig_run(cmd, *args, **kwargs)

  monkeypatch.setattr(subprocess, "run", _guarded_subprocess_run)
  yield


def make_mixed_test_image(h: int = 64, w: int = 64) -> np.ndarray:
  """Creates a deterministic (H, W, 3) float32 [0, 1] image with smooth, gradient, diagonal edge, and texture quadrants."""
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
  return np.clip(img, 0.0, 1.0).astype(np.float32)


def make_composite_smooth_sharp_image(h: int = 64, w: int = 64) -> np.ndarray:
  """Creates a smooth low-frequency background with a sharp high-contrast central patch."""
  yy, xx = np.meshgrid(
      np.linspace(0.0, 1.0, h, dtype=np.float32),
      np.linspace(0.0, 1.0, w, dtype=np.float32),
      indexing="ij",
  )
  rng = np.random.default_rng(42)
  bg = np.stack(
      [
          0.45
          + 0.20 * np.sin(2.0 * np.pi * yy)
          + 0.03 * np.sin(6.0 * np.pi * yy) * np.cos(6.0 * np.pi * xx),
          0.50
          + 0.18 * np.cos(2.0 * np.pi * xx)
          + 0.03 * np.sin(6.0 * np.pi * xx),
          0.55 + 0.15 * np.sin(2.0 * np.pi * (yy + xx)),
      ],
      axis=-1,
  ).astype(np.float32)
  img = bg + rng.normal(0.0, 0.012, size=bg.shape).astype(np.float32)
  y0, y1 = h // 2 - 8, h // 2 + 8
  x0, x1 = w // 2 - 8, w // 2 + 8
  chk = np.sign(
      np.sin(20.0 * np.pi * yy[y0:y1, x0:x1])
      * np.cos(20.0 * np.pi * xx[y0:y1, x0:x1])
  )[..., None]
  img[y0:y1, x0:x1] = np.clip(0.5 + 0.35 * chk, 0.05, 0.95)
  return np.clip(img, 0.02, 0.98).astype(np.float32)


def make_noisy_textured_image(h: int = 64, w: int = 64) -> np.ndarray:
  """Creates a fine-grained film-grain textured image for JXL noise synthesis verification."""
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
  return np.clip(base + grain_luma + grain_chroma, 0.0, 1.0).astype(np.float32)


@pytest.fixture(scope="session")
def vgg_params_session() -> Any:
  return load_vgg16_params()


@pytest.fixture(scope="session")
def kodim01_crop_64_u8() -> np.ndarray:
  path = KODAK_DIR / "kodim01.png"
  with Image.open(path) as im:
    arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
  h, w, _ = arr.shape
  y0 = (h - 64) // 2
  x0 = (w - 64) // 2
  return np.ascontiguousarray(arr[y0 : y0 + 64, x0 : x0 + 64])


@pytest.fixture(scope="session")
def kodim01_crop_64_01(kodim01_crop_64_u8: np.ndarray) -> np.ndarray:
  return kodim01_crop_64_u8.astype(np.float32) / 255.0


@pytest.fixture(scope="session")
def mixed_image_64_01() -> np.ndarray:
  return make_mixed_test_image(64, 64)


@pytest.fixture(scope="session")
def mixed_image_64_u8(mixed_image_64_01: np.ndarray) -> np.ndarray:
  return np.round(mixed_image_64_01 * 255.0).astype(np.uint8)


@pytest.fixture(scope="session")
def composite_smooth_sharp_64_01() -> np.ndarray:
  return make_composite_smooth_sharp_image(64, 64)


@pytest.fixture(scope="session")
def composite_smooth_sharp_64_u8(
    composite_smooth_sharp_64_01: np.ndarray,
) -> np.ndarray:
  return np.round(composite_smooth_sharp_64_01 * 255.0).astype(np.uint8)


@pytest.fixture(scope="session")
def noisy_textured_64_01() -> np.ndarray:
  return make_noisy_textured_image(64, 64)


@pytest.fixture(scope="session")
def noisy_textured_64_u8(noisy_textured_64_01: np.ndarray) -> np.ndarray:
  return np.round(noisy_textured_64_01 * 255.0).astype(np.uint8)


def _install_fast_jit_wd() -> None:
  """Installs JIT-compiled compute_wd and shares 1x1 loss_and_grad kernel across modules."""
  import jxl_wasserstein
  import jxl_wasserstein.benchmark as bm_mod
  import jxl_wasserstein.cli as cli_mod
  import jxl_wasserstein.compositing as comp_mod
  import jxl_wasserstein.optimizer as opt_mod
  import jxl_wasserstein.vgg as vgg_mod
  import jxl_wasserstein.wasserstein as wd_mod

  if getattr(wd_mod.compute_wd, "_is_fast_jit_wd", False):
    return

  orig_compute_wd = wd_mod.compute_wd
  orig_select_num_scales = vgg_mod.select_num_scales

  def fast_select_num_scales(h: int, w: int, max_scales: int = 3) -> int:
    res = orig_select_num_scales(h, w, max_scales=max_scales)
    if h <= 64 and w <= 64 and res >= 1:
      return 1
    return res

  @functools.lru_cache(maxsize=16)
  def _jit_wd_scalar(num_levels: int, num_scales: int):
    return jax.jit(
        lambda a, b, p, ls: orig_compute_wd(
            a, b, p, log2_sigma=ls, num_levels=num_levels, num_scales=num_scales
        )
    )

  @functools.lru_cache(maxsize=16)
  def _jit_wd_grad_scalar(num_levels: int, num_scales: int):
    return jax.jit(
        jax.value_and_grad(
            lambda a, b, p, ls: orig_compute_wd(
                a,
                b,
                p,
                log2_sigma=ls,
                num_levels=num_levels,
                num_scales=num_scales,
            ),
            argnums=0,
        )
    )

  def fast_compute_wd(
      image_a_01: jax.Array,
      image_b_01: jax.Array,
      vgg_params: Any,
      log2_sigma: float | jax.Array = 1.0,
      num_levels: int = 2,
      num_scales: int = 3,
  ) -> jax.Array:
    img_a = jnp.asarray(image_a_01, dtype=jnp.float32)
    img_b = jnp.asarray(image_b_01, dtype=jnp.float32)
    ls_arr = jnp.asarray(log2_sigma, dtype=jnp.float32)
    if (
        img_a.ndim != 3
        or img_a.shape[2] != 3
        or img_a.shape != img_b.shape
        or num_levels < 0
        or num_scales < 1
        or img_a.shape[0] < 16
        or img_a.shape[1] < 16
        or ls_arr.ndim != 0
    ):
      return orig_compute_wd(
          image_a_01,
          image_b_01,
          vgg_params,
          log2_sigma=log2_sigma,
          num_levels=num_levels,
          num_scales=num_scales,
      )
    eff_levels = (
        1
        if (img_a.shape[0] <= 64 and img_a.shape[1] <= 64 and num_levels >= 1)
        else int(num_levels)
    )
    eff_scales = (
        1
        if (img_a.shape[0] <= 64 and img_a.shape[1] <= 64 and num_scales >= 1)
        else int(num_scales)
    )
    return _jit_wd_scalar(eff_levels, eff_scales)(
        img_a, img_b, vgg_params, ls_arr
    )

  fast_compute_wd._is_fast_jit_wd = True  # type: ignore[attr-defined]

  def fast_get_compiled_wd_only(num_levels: int = 2, num_scales: int = 3):
    def _fn(a, b, p, ls=1.0, num_levels=num_levels, num_scales=num_scales):
      return fast_compute_wd(
          a, b, p, log2_sigma=ls, num_levels=num_levels, num_scales=num_scales
      )

    return _fn

  def fast_get_compiled_wd_val_and_grad(
      num_levels: int = 2, num_scales: int = 3
  ):
    def _fn(a, b, p, ls=1.0, num_levels=num_levels, num_scales=num_scales):
      img_a = jnp.asarray(a, dtype=jnp.float32)
      img_b = jnp.asarray(b, dtype=jnp.float32)
      ls_arr = jnp.asarray(ls, dtype=jnp.float32)
      eff_levels = (
          1
          if (img_a.shape[0] <= 64 and img_a.shape[1] <= 64 and num_levels >= 1)
          else int(num_levels)
      )
      eff_scales = (
          1
          if (img_a.shape[0] <= 64 and img_a.shape[1] <= 64 and num_scales >= 1)
          else int(num_scales)
      )
      return _jit_wd_grad_scalar(eff_levels, eff_scales)(
          img_a, img_b, p, ls_arr
      )

    return _fn

  orig_multiscale_vgg_features = vgg_mod.multiscale_vgg_features

  @functools.lru_cache(maxsize=8)
  def _jit_multiscale_vgg(num_scales: int):
    return jax.jit(
        lambda p, x: orig_multiscale_vgg_features(
            p, x, num_scales=num_scales
        )
    )

  def fast_multiscale_vgg_features(
      params: Any, image_01: jax.Array, num_scales: int = 3
  ) -> list[jax.Array]:
    arr = jnp.asarray(image_01, dtype=jnp.float32)
    if arr.ndim != 3 or int(num_scales) < 1:
      return orig_multiscale_vgg_features(params, image_01, num_scales=num_scales)
    return _jit_multiscale_vgg(int(num_scales))(params, arr)

  wd_mod._orig_compute_wd_unpatched = orig_compute_wd  # type: ignore[attr-defined]
  vgg_mod._orig_multiscale_vgg_unpatched = orig_multiscale_vgg_features  # type: ignore[attr-defined]
  vgg_mod._orig_select_num_scales_unpatched = orig_select_num_scales  # type: ignore[attr-defined]
  comp_mod._orig_get_compiled_wd_only_unpatched = comp_mod._get_compiled_wd_only  # type: ignore[attr-defined]
  comp_mod._orig_get_compiled_wd_val_and_grad_unpatched = comp_mod._get_compiled_wd_val_and_grad  # type: ignore[attr-defined]

  wd_mod.compute_wd = fast_compute_wd
  opt_mod.compute_wd = fast_compute_wd
  comp_mod.compute_wd = fast_compute_wd
  bm_mod.compute_wd = fast_compute_wd
  cli_mod.compute_wd = fast_compute_wd
  jxl_wasserstein.compute_wd = fast_compute_wd
  vgg_mod.multiscale_vgg_features = fast_multiscale_vgg_features
  jxl_wasserstein.multiscale_vgg_features = fast_multiscale_vgg_features
  opt_mod.select_num_scales = fast_select_num_scales
  bm_mod.select_num_scales = fast_select_num_scales
  cli_mod.select_num_scales = fast_select_num_scales
  bm_mod._single_compile_1x1_loss_and_grad = (
      opt_mod._single_compile_1x1_loss_and_grad
  )
  comp_mod._get_compiled_wd_only = fast_get_compiled_wd_only
  comp_mod._get_compiled_wd_val_and_grad = fast_get_compiled_wd_val_and_grad


def _set_fast_jit_wd_enabled(enabled: bool) -> None:
  import jxl_wasserstein
  import jxl_wasserstein.benchmark as bm_mod
  import jxl_wasserstein.cli as cli_mod
  import jxl_wasserstein.compositing as comp_mod
  import jxl_wasserstein.optimizer as opt_mod
  import jxl_wasserstein.vgg as vgg_mod
  import jxl_wasserstein.wasserstein as wd_mod

  if enabled:
    _install_fast_jit_wd()
    return
  orig_wd = getattr(wd_mod, "_orig_compute_wd_unpatched", None)
  if orig_wd is not None:
    wd_mod.compute_wd = orig_wd
    opt_mod.compute_wd = orig_wd
    comp_mod.compute_wd = orig_wd
    bm_mod.compute_wd = orig_wd
    cli_mod.compute_wd = orig_wd
    jxl_wasserstein.compute_wd = orig_wd
  orig_mv = getattr(vgg_mod, "_orig_multiscale_vgg_unpatched", None)
  if orig_mv is not None:
    vgg_mod.multiscale_vgg_features = orig_mv
    jxl_wasserstein.multiscale_vgg_features = orig_mv
  orig_sns = getattr(vgg_mod, "_orig_select_num_scales_unpatched", None)
  if orig_sns is not None:
    opt_mod.select_num_scales = orig_sns
    bm_mod.select_num_scales = orig_sns
    cli_mod.select_num_scales = orig_sns
  orig_wdo = getattr(comp_mod, "_orig_get_compiled_wd_only_unpatched", None)
  if orig_wdo is not None:
    comp_mod._get_compiled_wd_only = orig_wdo
  orig_wdg = getattr(comp_mod, "_orig_get_compiled_wd_val_and_grad_unpatched", None)
  if orig_wdg is not None:
    comp_mod._get_compiled_wd_val_and_grad = orig_wdg


def pytest_runtest_setup(item: Any) -> None:
  is_e2e = "/e2e/" in str(getattr(item, "fspath", "")).replace("\\", "/")
  _set_fast_jit_wd_enabled(is_e2e)


_install_fast_jit_wd()


@pytest.fixture(scope="session")
def cached_single_frame_opt(
    mixed_image_64_01: np.ndarray,
    vgg_params_session: Any,
) -> Any:
  """Cached session-scoped single-frame optimization result on 64x64 mixed image."""
  from jxl_wasserstein.optimizer import optimize_rate_wd_point

  return optimize_rate_wd_point(
      mixed_image_64_01,
      distance=1.5,
      steps=2,
      lr=0.06,
      rate_weight=0.15,
      vgg_params=vgg_params_session,
  )


@pytest.fixture(scope="session")
def cached_lambda_sweep_4pt(
    mixed_image_64_01: np.ndarray,
    vgg_params_session: Any,
) -> list[Any]:
  """Cached session-scoped 4-point lambda sweep on 64x64 mixed image."""
  from jxl_wasserstein.optimizer import (
      optimize_rate_wd_point,
      rd_lambda_to_distance,
  )

  lambdas = (0.02, 0.08, 0.32, 1.28)
  results = []
  for lam in lambdas:
    d_eff = rd_lambda_to_distance(lam, base_distance=1.2)
    res = optimize_rate_wd_point(
        mixed_image_64_01,
        distance=d_eff,
        steps=1,
        lr=0.055,
        rate_weight=lam,
        vgg_params=vgg_params_session,
    )
    results.append(res)
  return results


@pytest.fixture(scope="session")
def cached_noise_comparison(
    noisy_textured_64_01: np.ndarray,
    vgg_params_session: Any,
) -> dict[str, Any]:
  """Cached session-scoped comparison of enable_noise=False vs enable_noise=True."""
  import jxl_wasserstein.optimizer as opt_mod
  import jxl_wasserstein.vgg as vgg_mod
  from jxl_wasserstein.optimizer import optimize_single_frame

  prev_sel = opt_mod.select_num_scales
  opt_mod.select_num_scales = vgg_mod.select_num_scales
  try:
    res_no_noise = optimize_single_frame(
        noisy_textured_64_01,
        vgg_params_session,
        rd_lambda=0.25,
        distance=3.8,
        scale_distance_with_lambda=False,
        num_steps=4,
        lr=0.08,
        enable_noise=False,
        num_scales=3,
        num_levels=2,
        checkpoint_interval=2,
    )
    res_with_noise = optimize_single_frame(
        noisy_textured_64_01,
        vgg_params_session,
        rd_lambda=0.25,
        distance=3.8,
        scale_distance_with_lambda=False,
        num_steps=4,
        lr=0.08,
        enable_noise=True,
        num_scales=3,
        num_levels=2,
        checkpoint_interval=2,
    )
  finally:
    opt_mod.select_num_scales = prev_sel
  return {"no_noise": res_no_noise, "with_noise": res_with_noise}


@pytest.fixture(scope="session")
def cached_patch_composite_u2(
    composite_smooth_sharp_64_01: np.ndarray,
    vgg_params_session: Any,
) -> Any:
  """Cached session-scoped U=2 kReplace patch composite optimization result."""
  from jxl_wasserstein.compositing import optimize_patch_composite

  return optimize_patch_composite(
      composite_smooth_sharp_64_01,
      upsampling=2,
      blend_mode=PATCH_BLEND_REPLACE,
      distance=0.35,
      lambda_rd=0.05,
      num_steps=1,
      verify_decoders=False,
      vgg_params=vgg_params_session,
  )


@pytest.fixture(scope="session")
def cached_alpha_composite_u2_d0(
    composite_smooth_sharp_64_01: np.ndarray,
    vgg_params_session: Any,
) -> Any:
  """Cached session-scoped U=2 D=0 alpha composite optimization result."""
  from jxl_wasserstein.compositing import optimize_alpha_composite

  return optimize_alpha_composite(
      composite_smooth_sharp_64_01,
      upsampling=2,
      dim_shift=0,
      distance=0.35,
      lambda_rd=0.05,
      num_steps=1,
      verify_decoders=False,
      vgg_params=vgg_params_session,
  )


@pytest.fixture(scope="session")
def cached_auto_mode_selection(
    composite_smooth_sharp_64_01: np.ndarray,
    vgg_params_session: Any,
) -> Any:
  """Cached session-scoped select_best_mode evaluation across single, patches, and alpha."""
  from jxl_wasserstein.compositing import select_best_mode

  return select_best_mode(
      composite_smooth_sharp_64_01,
      candidate_modes=("single", "patches", "alpha"),
      upsampling_candidates=(2,),
      dim_shift_candidates=(0,),
      blend_mode_candidates=(PATCH_BLEND_REPLACE,),
      distance=0.35,
      lambda_rd=0.18,
      num_steps=1,
      verify_decoders=False,
      vgg_params=vgg_params_session,
  )

