"""Verification suite for the differentiable JXL token/context entropy proxy (`entropy_proxy.py`).

Tests:
1. Pure-JAX state sweep Spearman rank correlation (rho >= 0.95) across `0.15-2.0+ bpp`
   on Kodak crops (`kodim01`, `kodim02`, `kodim03`) via `init_state_from_rgb_jax`
   for `total_bits` (and `file_bytes * 8`), `ac_bits`, and `dc_bits`.
2. Coefficient scaling & deadzone thresholding monotonicity and Spearman rho >= 0.98.
3. Isolated DC (`dc_quant`, `ytox_dc`, `ytob_dc`) and AC metadata (`raw_quant_field`,
   `ytox_map`, `ytob_map`) spatial perturbations (Spearman rho >= 0.95).
4. Multi-block `AcStrategy` (`DCT16X16`, `DCT32X32`, `DCT16X8`, `DCT8X16`, `IDENTITY`,
   `DCT4X4`, `AFV0`) Spearman rho >= 0.95 and canonical LLF invariance (0.0 rate change
   and 0.0 gradient on canonical LLF entries of `quant_ac`).
5. Noise header delta (`80.0` bits matching `LayerType::Noise`).
6. Multi-resolution `PatchDictionary` (`soft_patch_dictionary_bits`,
   `soft_patch_composite_rate_proxy` vs `encode_patches_composite`) and Modular
   `ExtraChannel::kAlpha` (`soft_alpha_channel_bits`, `soft_alpha_composite_rate_proxy`
   vs `encode_alpha_composite` across `dim_shift in {0, 1, 2, 3}`) Spearman rho >= 0.95.
7. Gradient health (`|q| == 0`, `|q| > 0`, `ste_round`, and all continuous latents:
   `quant_ac`, `dc_quant`, `raw_quant_field`, `ytox_map`, `ytob_map`, `ytox_dc`,
   `ytob_dc`, `noise_lut`, `alpha_01`) and `@jax.jit` execution speed.
"""

from __future__ import annotations

import copy
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from jxl_wasserstein.cpp_bridge import (
    PATCH_BLEND_ADD,
    JxlVarDCTState,
    PatchEntry,
    encode_alpha_composite,
    encode_patches_composite,
    encode_single_frame,
)
from jxl_wasserstein.entropy_proxy import (
    RateBreakdown,
    compute_natural_coeff_order,
    compute_spatial_rate_map_bits,
    get_first_block_and_llf_masks,
    normalize_quant_ac_shape,
    soft_ac_meta_rate_bits,
    soft_ac_metadata_bits,
    soft_ac_rate_bits,
    soft_alpha_channel_bits,
    soft_alpha_composite_rate_proxy,
    soft_clamped_gradient_2d,
    soft_dc_rate_bits,
    soft_dequant_matrices_rate_bits,
    soft_header_rate_bits,
    soft_jxl_rate_proxy,
    soft_noise_header_bits,
    soft_patch_composite_rate_proxy,
    soft_patch_dictionary_bits,
    ste_round,
)
from jxl_wasserstein.forward_model import (
    ACS_NAMES,
    K_DC_QUANT,
    StaticVarDCTTopology,
    get_default_dequant_matrices,
    init_state_from_rgb_jax,
)

ACS_DCT = ACS_NAMES.index("DCT")
ACS_IDENTITY = ACS_NAMES.index("IDENTITY")
ACS_DCT4X4 = ACS_NAMES.index("DCT4X4")
ACS_DCT16X16 = ACS_NAMES.index("DCT16X16")
ACS_DCT32X32 = ACS_NAMES.index("DCT32X32")
ACS_DCT16X8 = ACS_NAMES.index("DCT16X8")
ACS_DCT8X16 = ACS_NAMES.index("DCT8X16")
ACS_AFV0 = ACS_NAMES.index("AFV0")

DEFAULT_NOISE_LUT: tuple[float, ...] = (
    0.008,
    0.012,
    0.016,
    0.020,
    0.024,
    0.026,
    0.028,
    0.030,
)

KODAK_DIR = Path(
    "/usr/local/google/home/veluca/libjxl/image-corpus/qoi_benchmark_suite/photo_kodak"
)


def _load_kodak_crop(
    name: str,
    crop_h: int = 128,
    crop_w: int = 128,
    y0: int = 64,
    x0: int = 64,
) -> np.ndarray:
  """Loads a nonlinear sRGB float32 [0, 1] crop from the Kodak corpus."""
  path = KODAK_DIR / f"{name}.png"
  if path.exists():
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(img[y0 : y0 + crop_h, x0 : x0 + crop_w])
  seed = sum(ord(ch) for ch in name)
  rng = np.random.default_rng(seed)
  yy, xx = np.meshgrid(
      np.linspace(0.0, 1.0, crop_h, dtype=np.float32),
      np.linspace(0.0, 1.0, crop_w, dtype=np.float32),
      indexing="ij",
  )
  r = 0.5 + 0.28 * np.sin(2.0 * np.pi * (3.0 * xx + 2.0 * yy))
  g = 0.5 + 0.24 * np.cos(2.0 * np.pi * (5.0 * xx - 3.0 * yy))
  b = 0.5 + 0.20 * np.sin(2.0 * np.pi * (2.0 * xx - 4.0 * yy))
  base = np.stack([r, g, b], axis=-1) + rng.normal(
      0.0, 0.06, size=(crop_h, crop_w, 3)
  )
  return np.clip(base, 0.0, 1.0).astype(np.float32)


def _replace_state_arrays(
    state: JxlVarDCTState,
    quant_ac: np.ndarray | None = None,
    dc_quant: np.ndarray | None = None,
    raw_quant_field: np.ndarray | None = None,
    ytox_map: np.ndarray | None = None,
    ytob_map: np.ndarray | None = None,
) -> JxlVarDCTState:
  st = copy.deepcopy(state)
  if quant_ac is not None:
    q_arr = np.asarray(quant_ac, dtype=np.int32)
    if q_arr.ndim == 5:
      # Convert (3, bh, bw, 8, 8) -> (3, bh*8, bw*8)
      c, bh, bw, _, _ = q_arr.shape
      q_arr = q_arr.transpose(0, 1, 3, 2, 4).reshape(c, bh * 8, bw * 8)
    st.quant_ac = q_arr
  if dc_quant is not None:
    st.dc_quant = np.asarray(dc_quant, dtype=np.int32)
  if raw_quant_field is not None:
    st.raw_quant_field = np.asarray(raw_quant_field, dtype=np.int32)
  if ytox_map is not None:
    st.ytox_map = np.asarray(ytox_map, dtype=np.int8)
  if ytob_map is not None:
    st.ytob_map = np.asarray(ytob_map, dtype=np.int8)
  return st


def test_pure_jax_state_sweep_spearman_correlation() -> None:
  """1. Pure-JAX state sweep Spearman rank correlation (rho >= 0.95) across 0.15-2.0+ bpp."""
  crops = ["kodim01", "kodim02", "kodim03"]
  # Distances spanning ~0.12 bpp (d=36.0) to ~2.8 bpp (d=0.35)
  distances = [
      36.0,
      24.0,
      16.0,
      10.0,
      6.5,
      4.0,
      2.5,
      1.6,
      1.0,
      0.7,
      0.5,
      0.35,
  ]

  pooled_true_total: list[float] = []
  pooled_true_file_bits: list[float] = []
  pooled_proxy_total: list[float] = []
  pooled_true_ac: list[float] = []
  pooled_proxy_ac: list[float] = []
  pooled_true_dc: list[float] = []
  pooled_proxy_dc: list[float] = []
  pooled_true_bpp: list[float] = []

  for name in crops:
    rgb = _load_kodak_crop(name, crop_h=128, crop_w=128)
    num_pixels = float(rgb.shape[0] * rgb.shape[1])

    img_true_total: list[float] = []
    img_true_file_bits: list[float] = []
    img_proxy_total: list[float] = []
    img_true_ac: list[float] = []
    img_proxy_ac: list[float] = []
    img_true_dc: list[float] = []
    img_proxy_dc: list[float] = []

    for d in distances:
      st = init_state_from_rgb_jax(rgb, distance=d)
      _, stats = encode_single_frame(st)
      bd = soft_jxl_rate_proxy(st, return_breakdown=True)
      assert isinstance(bd, RateBreakdown)

      true_tot = float(stats["total_bits"])
      true_f_bits = float(stats["file_bytes"] * 8)
      true_ac_tok = float(stats["ac_token_bits"])
      true_dc = float(stats["dc_bits"])

      proxy_tot = float(bd.total_bits)
      proxy_ac = float(bd.ac_bits)
      proxy_dc = float(bd.dc_bits)

      img_true_total.append(true_tot)
      img_true_file_bits.append(true_f_bits)
      img_proxy_total.append(proxy_tot)
      img_true_ac.append(true_ac_tok)
      img_proxy_ac.append(proxy_ac)
      img_true_dc.append(true_dc)
      img_proxy_dc.append(proxy_dc)

      pooled_true_total.append(true_tot)
      pooled_true_file_bits.append(true_f_bits)
      pooled_proxy_total.append(proxy_tot)
      pooled_true_ac.append(true_ac_tok)
      pooled_proxy_ac.append(proxy_ac)
      pooled_true_dc.append(true_dc)
      pooled_proxy_dc.append(proxy_dc)
      pooled_true_bpp.append(true_f_bits / num_pixels)

    rho_img_tot = float(spearmanr(img_true_total, img_proxy_total).statistic)
    rho_img_file = float(
        spearmanr(img_true_file_bits, img_proxy_total).statistic
    )
    rho_img_ac = float(spearmanr(img_true_ac, img_proxy_ac).statistic)
    rho_img_dc = float(spearmanr(img_true_dc, img_proxy_dc).statistic)

    assert rho_img_tot >= 0.95, f"{name} total_bits rho={rho_img_tot:.4f}"
    assert rho_img_file >= 0.95, f"{name} file_bits rho={rho_img_file:.4f}"
    assert rho_img_ac >= 0.95, f"{name} ac_bits rho={rho_img_ac:.4f}"
    assert rho_img_dc >= 0.95, f"{name} dc_bits rho={rho_img_dc:.4f}"

  # Verify bpp span covers [0.15, 2.0] bpp
  assert min(pooled_true_bpp) <= 0.16, f"min bpp={min(pooled_true_bpp):.3f}"
  assert max(pooled_true_bpp) >= 2.00, f"max bpp={max(pooled_true_bpp):.3f}"

  rho_pool_tot = float(
      spearmanr(pooled_true_total, pooled_proxy_total).statistic
  )
  rho_pool_file = float(
      spearmanr(pooled_true_file_bits, pooled_proxy_total).statistic
  )
  rho_pool_ac = float(spearmanr(pooled_true_ac, pooled_proxy_ac).statistic)
  rho_pool_dc = float(spearmanr(pooled_true_dc, pooled_proxy_dc).statistic)

  assert rho_pool_tot >= 0.95, f"Pooled total_bits rho={rho_pool_tot:.4f}"
  assert rho_pool_file >= 0.95, f"Pooled file_bits rho={rho_pool_file:.4f}"
  assert rho_pool_ac >= 0.95, f"Pooled ac_bits rho={rho_pool_ac:.4f}"
  assert rho_pool_dc >= 0.95, f"Pooled dc_bits rho={rho_pool_dc:.4f}"


def test_ac_scaling_and_deadzone_thresholding_monotonicity() -> None:
  """2. Coefficient scaling & deadzone thresholding monotonicity (Spearman rho >= 0.98)."""
  rgb = _load_kodak_crop("kodim01", crop_h=128, crop_w=128)
  base_st = init_state_from_rgb_jax(rgb, distance=1.0)
  topo = StaticVarDCTTopology(base_st)
  base_q = jnp.asarray(base_st.quant_ac, dtype=jnp.float32)

  # 2a. AC magnitude scaling sweep
  scales = [0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0, 1.20, 1.45, 1.75]
  true_scale_bits: list[float] = []
  proxy_scale_bits: list[float] = []
  proxy_cont_scale_bits: list[float] = []

  for alpha in scales:
    q_cont = base_q * alpha
    q_int = np.round(np.asarray(q_cont)).astype(np.int32)
    st = _replace_state_arrays(base_st, quant_ac=q_int)
    _, stats = encode_single_frame(st)
    true_scale_bits.append(
        float(stats["ac_token_bits"] + stats["layer_bits"]["Ac"])
    )

    p_ste = float(soft_ac_rate_bits(ste_round(q_cont), topology=topo))
    p_cont = float(soft_ac_rate_bits(q_cont, topology=topo))
    proxy_scale_bits.append(p_ste)
    proxy_cont_scale_bits.append(p_cont)

  rho_scale_ste = float(spearmanr(true_scale_bits, proxy_scale_bits).statistic)
  rho_scale_cont = float(
      spearmanr(true_scale_bits, proxy_cont_scale_bits).statistic
  )
  assert rho_scale_ste >= 0.98, f"AC scaling STE rho={rho_scale_ste:.4f}"
  assert rho_scale_cont >= 0.98, f"AC scaling cont rho={rho_scale_cont:.4f}"

  # Verify strict monotonicity of continuous AC rate w.r.t. scale factor alpha
  for i in range(len(proxy_cont_scale_bits) - 1):
    assert proxy_cont_scale_bits[i + 1] > proxy_cont_scale_bits[i], (
        f"Non-monotonic AC scaling at {scales[i]} -> {scales[i + 1]}: "
        f"{proxy_cont_scale_bits[i]} vs {proxy_cont_scale_bits[i + 1]}"
    )

  # 2b. Deadzone thresholding sweep (half-integer thresholds to avoid round-to-even parity flip)
  thresholds = [0.0, 0.8, 1.8, 2.8, 3.8, 5.8, 8.8, 12.8, 18.8]
  true_dz_bits: list[float] = []
  proxy_dz_bits: list[float] = []

  for th in thresholds:
    q_dz = jnp.sign(base_q) * jnp.maximum(0.0, jnp.abs(base_q) - th)
    q_int = np.round(np.asarray(q_dz)).astype(np.int32)
    st = _replace_state_arrays(base_st, quant_ac=q_int)
    _, stats = encode_single_frame(st)
    true_dz_bits.append(
        float(stats["ac_token_bits"] + stats["layer_bits"]["Ac"])
    )
    proxy_dz_bits.append(float(soft_ac_rate_bits(q_dz, topology=topo)))

  rho_dz = float(spearmanr(true_dz_bits, proxy_dz_bits).statistic)
  assert rho_dz >= 0.98, f"Deadzone sweep rho={rho_dz:.4f}"
  for i in range(len(proxy_dz_bits) - 1):
    assert proxy_dz_bits[i + 1] < proxy_dz_bits[i], (
        f"Non-monotonic deadzone at {thresholds[i]} -> {thresholds[i + 1]}: "
        f"{proxy_dz_bits[i]} vs {proxy_dz_bits[i + 1]}"
    )


def test_isolated_dc_and_ac_metadata_perturbations() -> None:
  """3. Isolated DC (`dc_quant`) & AC metadata (`raw_quant_field`, `ytox_map`, `ytob_map`) perturbations."""
  rgb = _load_kodak_crop("kodim02", crop_h=128, crop_w=128)
  base_st = init_state_from_rgb_jax(rgb, distance=1.5)
  rng = np.random.default_rng(1234)

  # 3a. Isolated DC perturbations (varying spatial smoothness and amplitude)
  dc_sigmas = [0.0, 1.0, 2.0, 4.0, 8.0, 14.0, 22.0, 35.0, 55.0, 85.0]
  true_dc_list: list[float] = []
  proxy_dc_list: list[float] = []

  noise_dc = rng.standard_normal(base_st.dc_quant.shape).astype(np.float32)
  for sig in dc_sigmas:
    dc_pert = np.round(
        base_st.dc_quant.astype(np.float32) + sig * noise_dc
    ).astype(np.int32)
    st = _replace_state_arrays(base_st, dc_quant=dc_pert)
    _, stats = encode_single_frame(st)
    true_dc_list.append(float(stats["dc_bits"]))
    proxy_dc_list.append(
        float(
            soft_dc_rate_bits(
                jnp.array(st.dc_quant, dtype=jnp.float32),
                ytox_dc=st.ytox_dc,
                ytob_dc=st.ytob_dc,
                ac_strategy=st.ac_strategy,
                color_factor=st.color_factor,
            )
        )
    )

  rho_dc_pert = float(spearmanr(true_dc_list, proxy_dc_list).statistic)
  assert rho_dc_pert >= 0.95, f"Isolated DC perturbation rho={rho_dc_pert:.4f}"

  # 3b. Isolated AC metadata perturbations (`raw_quant_field`, `ytox_map`, `ytob_map`)
  meta_amps = [0, 1, 2, 4, 8, 12, 18, 26, 38]
  true_meta_list: list[float] = []
  proxy_meta_list: list[float] = []

  qf_base = np.full_like(base_st.raw_quant_field, 16, dtype=np.int32)
  for amp in meta_amps:
    if amp == 0:
      qf_mod = qf_base.copy()
      yx_mod = np.zeros_like(base_st.ytox_map, dtype=np.int8)
      yb_mod = np.zeros_like(base_st.ytob_map, dtype=np.int8)
    else:
      qf_mod = np.clip(
          qf_base + rng.integers(-amp, amp + 1, size=qf_base.shape), 1, 255
      ).astype(np.int32)
      yx_mod = np.clip(
          rng.integers(-amp, amp + 1, size=base_st.ytox_map.shape), -64, 64
      ).astype(np.int8)
      yb_mod = np.clip(
          rng.integers(-amp, amp + 1, size=base_st.ytob_map.shape), -64, 64
      ).astype(np.int8)

    st = _replace_state_arrays(
        base_st, raw_quant_field=qf_mod, ytox_map=yx_mod, ytob_map=yb_mod
    )
    _, stats = encode_single_frame(st)
    true_meta_list.append(
        float(stats["ac_meta_bits"] + stats["layer_bits"]["ModularGlobal"])
    )
    proxy_meta_list.append(
        float(
            soft_ac_meta_rate_bits(
                jnp.array(st.raw_quant_field, dtype=jnp.float32),
                jnp.array(st.ytox_map, dtype=jnp.float32),
                jnp.array(st.ytob_map, dtype=jnp.float32),
                ac_strategy=st.ac_strategy,
                epf_sharpness=jnp.array(st.epf_sharpness, dtype=jnp.float32),
                global_scale=st.global_scale,
            )
        )
    )

  rho_meta_pert = float(spearmanr(true_meta_list, proxy_meta_list).statistic)
  assert (
      rho_meta_pert >= 0.95
  ), f"Isolated AC metadata perturbation rho={rho_meta_pert:.4f}"
  assert np.isclose(
      float(
          soft_ac_metadata_bits(
              jnp.array(base_st.raw_quant_field, dtype=jnp.float32),
              jnp.array(base_st.ytox_map, dtype=jnp.float32),
              jnp.array(base_st.ytob_map, dtype=jnp.float32),
              ac_strategy=base_st.ac_strategy,
          )
      ),
      float(
          soft_ac_meta_rate_bits(
              jnp.array(base_st.raw_quant_field, dtype=jnp.float32),
              jnp.array(base_st.ytox_map, dtype=jnp.float32),
              jnp.array(base_st.ytob_map, dtype=jnp.float32),
              ac_strategy=base_st.ac_strategy,
          )
      ),
  )


def test_multiblock_ac_strategy_and_canonical_llf_invariance() -> None:
  """4. Multi-block AcStrategy Spearman rho >= 0.95 & canonical LLF invariance."""
  rgb = _load_kodak_crop("kodim03", crop_h=128, crop_w=128)

  # Construct a mixed multi-block AcStrategy grid on 16x16 blocks (128x128 px)
  acs = np.full((16, 16), ACS_DCT, dtype=np.uint8)
  # 32x32 block at (0..3, 0..3)
  acs[0:4, 0:4] = ACS_DCT32X32
  # 16x16 block at (0..1, 4..5)
  acs[0:2, 4:6] = ACS_DCT16X16
  # DCT16X8 block (cy=2, cx=1) at (2..3, 4)
  acs[2:4, 4] = ACS_DCT16X8
  # DCT8X16 block (cy=1, cx=2) at (4, 0..1)
  acs[4, 0:2] = ACS_DCT8X16
  # 1x1 variants
  acs[6, 6] = ACS_DCT4X4
  acs[6, 7] = ACS_IDENTITY
  acs[7, 6] = ACS_AFV0

  first_block_mask, llf_mask = get_first_block_and_llf_masks(acs)
  # Total LLF slots = 16*16 = 256 (exactly 1 LLF slot per 8x8 block across the grid)
  assert int(np.sum(llf_mask)) == 16 * 16
  assert int(np.sum(first_block_mask)) == (
      16 * 16 - (16 - 1) - (4 - 1) - (2 - 1) - (2 - 1)
  )

  # 4a. Verify canonical LLF invariance: perturbing LLF entries of `quant_ac`
  # produces 0.0 rate change and 0.0 gradient on LLF entries
  st_base = init_state_from_rgb_jax(rgb, distance=1.2, ac_strategy=acs)
  topo = StaticVarDCTTopology(st_base)
  q_base_5d = normalize_quant_ac_shape(
      jnp.array(st_base.quant_ac, dtype=jnp.float32)
  )
  llf_mask_5d = jnp.array(llf_mask[None, :, :, :, :], dtype=jnp.float32)

  # Add arbitrary large perturbations to all canonical LLF positions
  q_llf_perturbed = q_base_5d + llf_mask_5d * 777.0

  rate_base_topo = soft_ac_rate_bits(q_base_5d, topology=topo)
  rate_pert_topo = soft_ac_rate_bits(q_llf_perturbed, topology=topo)
  assert float(jnp.abs(rate_base_topo - rate_pert_topo)) == 0.0

  # Also verify with raw `ac_strategy` (without pre-built `topology`)
  rate_base_acs = soft_ac_rate_bits(q_base_5d, ac_strategy=acs)
  rate_pert_acs = soft_ac_rate_bits(q_llf_perturbed, ac_strategy=acs)
  assert float(jnp.abs(rate_base_acs - rate_pert_acs)) == 0.0
  assert np.isclose(float(rate_base_topo), float(rate_base_acs), rtol=1e-5)

  # Verify gradient on canonical LLF slots is identically 0.0
  grad_q = jax.grad(lambda q: soft_ac_rate_bits(q, topology=topo))(
      q_llf_perturbed
  )
  max_llf_grad = float(jnp.max(jnp.abs(grad_q * llf_mask_5d)))
  assert max_llf_grad == 0.0, f"Expected 0.0 LLF gradient, got {max_llf_grad}"

  # Verify non-LLF gradient is strictly non-zero on all non-LLF slots
  non_llf_mask_5d = 1.0 - llf_mask_5d
  min_non_llf_grad = float(jnp.min(jnp.abs(grad_q) + llf_mask_5d * 1.0))
  assert (
      min_non_llf_grad > 0.0
  ), f"Expected non-zero AC gradient on all non-LLF slots, got {min_non_llf_grad}"
  assert float(jnp.sum(jnp.abs(grad_q * non_llf_mask_5d))) > 0.0

  # 4b. Multi-block distance sweep Spearman correlation vs libjxl encode_single_frame
  distances = [14.0, 8.0, 5.0, 3.0, 2.0, 1.3, 0.9, 0.6, 0.4]
  true_tot_list: list[float] = []
  proxy_tot_list: list[float] = []
  true_ac_list: list[float] = []
  proxy_ac_list: list[float] = []

  for d in distances:
    st = init_state_from_rgb_jax(rgb, distance=d, ac_strategy=acs)
    _, stats = encode_single_frame(st)
    bd = soft_jxl_rate_proxy(st, topology=topo, return_breakdown=True)
    assert isinstance(bd, RateBreakdown)
    true_tot_list.append(float(stats["total_bits"]))
    proxy_tot_list.append(float(bd.total_bits))
    true_ac_list.append(float(stats["ac_token_bits"]))
    proxy_ac_list.append(float(bd.ac_bits))

  rho_mb_tot = float(spearmanr(true_tot_list, proxy_tot_list).statistic)
  rho_mb_ac = float(spearmanr(true_ac_list, proxy_ac_list).statistic)
  assert rho_mb_tot >= 0.95, f"Multi-block total_bits rho={rho_mb_tot:.4f}"
  assert rho_mb_ac >= 0.95, f"Multi-block ac_bits rho={rho_mb_ac:.4f}"


def test_noise_header_delta_80_bits() -> None:
  """5. Verify enable_noise=True vs False adds 80.0 bits matching LayerType::Noise."""
  rgb = _load_kodak_crop("kodim01", crop_h=64, crop_w=64)
  st = init_state_from_rgb_jax(rgb, distance=1.5)

  _, stats_no_noise = encode_single_frame(st, noise_lut=None)
  _, stats_noise = encode_single_frame(st, noise_lut=DEFAULT_NOISE_LUT)
  true_noise_delta = float(
      stats_noise["layer_bits"]["Noise"]
      - stats_no_noise["layer_bits"]["Noise"]
  )
  assert true_noise_delta == 80.0

  bd_no_noise = soft_jxl_rate_proxy(
      st, enable_noise=False, return_breakdown=True
  )
  bd_noise = soft_jxl_rate_proxy(
      st,
      enable_noise=True,
      noise_lut=DEFAULT_NOISE_LUT,
      return_breakdown=True,
  )
  assert isinstance(bd_no_noise, RateBreakdown)
  assert isinstance(bd_noise, RateBreakdown)

  assert float(bd_no_noise.noise_bits) == 0.0
  assert np.isclose(float(bd_noise.noise_bits), 80.0, atol=1e-2)
  assert np.isclose(
      float(bd_noise.total_bits - bd_no_noise.total_bits), 80.0, atol=1e-2
  )


def test_patch_dictionary_and_alpha_composite_sweeps() -> None:
  """6. Multi-resolution PatchDictionary & Modular ExtraChannel::kAlpha composite sweeps."""
  rgb512 = _load_kodak_crop("kodim01", crop_h=512, crop_w=512, y0=0, x0=0)
  rgb = rgb512[:128, :128]

  # 6a. Single full-frame kAdd PatchDictionary sweep (varying frame dimensions, upsampling, distances)
  true_patch_dict_bits: list[float] = []
  proxy_patch_dict_bits: list[float] = []
  true_patch_comp_bits: list[float] = []
  proxy_patch_comp_bits: list[float] = []

  configs = [
      (16, 2, 4.0, 2.5),
      (32, 2, 3.0, 2.0),
      (64, 4, 2.5, 1.6),
      (128, 4, 2.0, 1.2),
      (256, 8, 1.5, 0.9),
      (512, 8, 1.2, 0.7),
  ]

  for sz, upsampling, ref_d, main_d in configs:
    rgb_crop = rgb512[:sz, :sz]
    h_ref, w_ref = sz // upsampling, sz // upsampling
    rgb_ref = np.asarray(
        jax.image.resize(
            jnp.asarray(rgb_crop), (h_ref, w_ref, 3), method="bilinear"
        )
    )
    ref_st = init_state_from_rgb_jax(rgb_ref * 0.6, distance=ref_d)
    main_st = init_state_from_rgb_jax(rgb_crop * 0.4, distance=main_d)

    patches = [
        PatchEntry(
            ref=0,
            x0=0,
            y0=0,
            xsize=sz,
            ysize=sz,
            x=0,
            y=0,
            blend_mode=PATCH_BLEND_ADD,
        )
    ]

    _, stats_p = encode_patches_composite(
        ref_state=ref_st,
        main_state=main_st,
        upsampling=upsampling,
        patches=patches,
    )
    p_bits = float(soft_patch_dictionary_bits(patches, upsampling=upsampling))
    c_bd = soft_patch_composite_rate_proxy(
        ref_st, main_st, patches, upsampling=upsampling, return_breakdown=True
    )
    assert isinstance(c_bd, RateBreakdown)

    true_patch_dict_bits.append(float(stats_p["layer_bits"]["Dictionary"]))
    proxy_patch_dict_bits.append(p_bits)
    true_patch_comp_bits.append(float(stats_p["file_bytes"] * 8))
    proxy_patch_comp_bits.append(float(c_bd.total_bits))

  rho_patch_dict = float(
      spearmanr(true_patch_dict_bits, proxy_patch_dict_bits).statistic
  )
  rho_patch_comp = float(
      spearmanr(true_patch_comp_bits, proxy_patch_comp_bits).statistic
  )
  assert (
      rho_patch_dict >= 0.95
  ), f"PatchDictionary bits rho={rho_patch_dict:.4f}"
  assert (
      rho_patch_comp >= 0.95
  ), f"Patch composite total_bits rho={rho_patch_comp:.4f}"

  # 6b. Modular ExtraChannel::kAlpha sweep across dim_shift in {0, 1, 2, 3}
  rgb_ref_2x = np.asarray(
      jax.image.resize(jnp.asarray(rgb), (64, 64, 3), method="bilinear")
  )
  ref_st_a = init_state_from_rgb_jax(rgb_ref_2x, distance=2.5)
  main_st_a = init_state_from_rgb_jax(rgb, distance=1.5)

  _, base_2frame_stats = encode_patches_composite(
      ref_state=ref_st_a,
      main_state=main_st_a,
      upsampling=2,
      patches=None,
  )
  base_2frame_bits = float(base_2frame_stats["file_bytes"] * 8)

  pooled_true_alpha: list[float] = []
  pooled_proxy_alpha: list[float] = []
  pooled_true_alpha_comp: list[float] = []
  pooled_proxy_alpha_comp: list[float] = []

  for ds in [0, 1, 2, 3]:
    h_a = 128 >> ds
    w_a = 128 >> ds
    y = jnp.linspace(0.0, 1.0, h_a)[:, None]
    x = jnp.linspace(0.0, 1.0, w_a)[None, :]
    rng = np.random.default_rng(42 + ds)

    patterns = [
        jnp.zeros((h_a, w_a), dtype=jnp.float32),
        jnp.broadcast_to((x > 0.5).astype(jnp.float32), (h_a, w_a)),
        jnp.clip(0.5 * (x + y), 0.0, 1.0),
        jnp.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2.0 * 0.18**2)),
        jnp.clip(
            0.5
            + 0.35 * jnp.sin(20.0 * x + 12.0 * y) * jnp.cos(16.0 * y - 8.0 * x)
            + jnp.asarray(
                rng.normal(0.0, 0.04, size=(h_a, w_a)), dtype=jnp.float32
            ),
            0.0,
            1.0,
        ),
        jnp.asarray(rng.uniform(0.0, 1.0, size=(h_a, w_a)), dtype=jnp.float32),
    ]

    ds_true_alpha: list[float] = []
    ds_proxy_alpha: list[float] = []
    for alpha_01 in patterns:
      alpha_u8 = np.round(
          np.clip(np.asarray(alpha_01), 0.0, 1.0) * 255.0
      ).astype(np.uint8)
      _, stats_a = encode_alpha_composite(
          ref_state=ref_st_a,
          main_state=main_st_a,
          upsampling=2,
          dim_shift=ds,
          alpha_u8=alpha_u8,
      )
      t_comp = float(stats_a["file_bytes"] * 8)
      t_alpha = max(t_comp - base_2frame_bits, 40.0)
      p_alpha = float(soft_alpha_channel_bits(alpha_01))
      ac_bd = soft_alpha_composite_rate_proxy(
          ref_st_a,
          main_st_a,
          alpha_01,
          upsampling=2,
          dim_shift=0,
          return_breakdown=True,
      )
      assert isinstance(ac_bd, RateBreakdown)

      ds_true_alpha.append(t_alpha)
      ds_proxy_alpha.append(p_alpha)
      pooled_true_alpha.append(t_alpha)
      pooled_proxy_alpha.append(p_alpha)
      pooled_true_alpha_comp.append(t_comp)
      pooled_proxy_alpha_comp.append(float(ac_bd.total_bits))

    rho_ds = float(spearmanr(ds_true_alpha, ds_proxy_alpha).statistic)
    assert rho_ds >= 0.95, f"Alpha bits rho at dim_shift={ds} is {rho_ds:.4f}"

  rho_pool_alpha = float(
      spearmanr(pooled_true_alpha, pooled_proxy_alpha).statistic
  )
  rho_pool_alpha_comp = float(
      spearmanr(pooled_true_alpha_comp, pooled_proxy_alpha_comp).statistic
  )
  assert (
      rho_pool_alpha >= 0.95
  ), f"Pooled alpha_bits across dim_shift={{0,1,2,3}} rho={rho_pool_alpha:.4f}"
  assert (
      rho_pool_alpha_comp >= 0.95
  ), f"Pooled alpha composite total_bits rho={rho_pool_alpha_comp:.4f}"


def test_gradient_health_and_jit_speed() -> None:
  """7. Gradient health (|q|==0, |q|>0, STE, all continuous latents) & @jax.jit speed."""
  rgb = _load_kodak_crop("kodim01", crop_h=64, crop_w=64)
  st = init_state_from_rgb_jax(rgb, distance=1.5)
  topo = StaticVarDCTTopology(st)

  # 7a. Verify ste_round preserves signbit in (-0.5, 0) and gives non-zero radial gradient
  probe = jnp.array(
      [-2.3, -1.0, -0.4, -0.1, -0.0, 0.0, 0.1, 0.4, 1.0, 2.3],
      dtype=jnp.float32,
  )
  ste_vals = ste_round(probe)
  assert bool(jnp.signbit(ste_vals[2])) is True
  assert bool(jnp.signbit(ste_vals[3])) is True
  assert bool(jnp.signbit(ste_vals[6])) is False

  # 7b. Verify radial derivative dR/d|q| > 0 at |q| == 0 and under ste_round
  q_5d = normalize_quant_ac_shape(jnp.array(st.quant_ac, dtype=jnp.float32))
  q_zero = jnp.zeros_like(q_5d)
  grad_at_pos_zero = jax.grad(lambda q: soft_ac_rate_bits(q, topology=topo))(
      q_zero
  )
  assert bool(jnp.all(jnp.isfinite(grad_at_pos_zero)))
  non_llf_grad_pos_zero = grad_at_pos_zero.reshape(3, 8, 8, 64)[:, :, :, 1:]
  assert float(jnp.min(non_llf_grad_pos_zero)) > 0.0

  # At -0.0, gradient w.r.t. q is strictly negative (pushing toward 0)
  q_neg_zero = -jnp.zeros_like(q_zero)
  grad_at_neg_zero = jax.grad(lambda q: soft_ac_rate_bits(q, topology=topo))(
      q_neg_zero
  )
  assert bool(jnp.all(jnp.isfinite(grad_at_neg_zero)))
  assert float(jnp.max(grad_at_neg_zero.reshape(3, 8, 8, 64)[:, :, :, 1:])) < 0.0

  # Under ste_round on continuous coefficients including sub-deadzone (-0.4, +0.4)
  rng = np.random.default_rng(77)
  sub_dz = jnp.asarray(
      rng.uniform(-0.4, 0.4, size=q_5d.shape), dtype=jnp.float32
  )
  # Ensure non-zero continuous values everywhere
  sub_dz = jnp.where(jnp.abs(sub_dz) < 0.05, 0.15, sub_dz)
  q_cont = q_5d + sub_dz
  grad_ste = jax.grad(
      lambda q: soft_ac_rate_bits(ste_round(q), topology=topo)
  )(q_cont)
  assert bool(jnp.all(jnp.isfinite(grad_ste)))
  non_llf_q = q_cont.reshape(3, 8, 8, 64)[:, :, :, 1:]
  non_llf_g = grad_ste.reshape(3, 8, 8, 64)[:, :, :, 1:]
  expected_sign = jnp.where(jnp.signbit(non_llf_q), -1.0, 1.0)
  radial_deriv = non_llf_g * expected_sign
  assert float(jnp.min(radial_deriv)) > 0.0

  # 7c. Joint @jax.jit value_and_grad across all continuous latents
  dc_q = jnp.array(st.dc_quant, dtype=jnp.float32) + 0.15
  qf = jnp.array(st.raw_quant_field, dtype=jnp.float32) + 0.25
  cfx = jnp.array(st.ytox_map, dtype=jnp.float32) + 0.1
  cfb = jnp.array(st.ytob_map, dtype=jnp.float32) - 0.1
  ytox_dc = jnp.float32(float(st.ytox_dc) + 0.2)
  ytob_dc = jnp.float32(float(st.ytob_dc) - 0.2)
  noise_lut = jnp.asarray(DEFAULT_NOISE_LUT, dtype=jnp.float32)
  yy, xx = jnp.meshgrid(
      jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32),
      jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32),
      indexing="ij",
  )
  alpha_01 = jnp.clip(0.5 + 0.35 * jnp.sin(6.0 * xx + 4.0 * yy), 0.0, 1.0)

  @jax.jit
  def loss_fn(params: dict[str, jax.Array]) -> jax.Array:
    return soft_jxl_rate_proxy(
        st,
        quant_ac=ste_round(params["quant_ac"]),
        dc_quant=params["dc_quant"],
        raw_quant_field=params["raw_quant_field"],
        ytox_map=params["ytox_map"],
        ytob_map=params["ytob_map"],
        ytox_dc=params["ytox_dc"],
        ytob_dc=params["ytob_dc"],
        enable_noise=True,
        noise_lut=params["noise_lut"],
        alpha_01=params["alpha_01"],
        dim_shift=0,
        topology=topo,
        return_breakdown=False,
    )

  params = {
      "quant_ac": q_cont,
      "dc_quant": dc_q,
      "raw_quant_field": qf,
      "ytox_map": cfx,
      "ytob_map": cfb,
      "ytox_dc": ytox_dc,
      "ytob_dc": ytob_dc,
      "noise_lut": noise_lut,
      "alpha_01": alpha_01,
  }

  val_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
  val, grads = val_and_grad_fn(params)
  jax.block_until_ready(val)

  assert bool(jnp.isfinite(val))
  assert float(val) > 0.0

  for k, g in grads.items():
    assert bool(jnp.all(jnp.isfinite(g))), f"Non-finite gradient for {k}"
    grad_norm = float(jnp.linalg.norm(g))
    assert grad_norm > 0.0, f"Zero gradient norm for {k}"

  # Benchmark compiled @jax.jit step speed
  t0 = time.perf_counter()
  num_iters = 10
  for _ in range(num_iters):
    v_iter, _ = val_and_grad_fn(params)
    jax.block_until_ready(v_iter)
  elapsed_ms = (time.perf_counter() - t0) * 1000.0 / num_iters
  assert elapsed_ms < 250.0, f"Compiled JIT step too slow: {elapsed_ms:.2f} ms"

  # Also verify compute_natural_coeff_order and soft_clamped_gradient_2d directly
  order_8x8 = compute_natural_coeff_order(1, 1)
  assert order_8x8.shape == (64,)
  assert int(order_8x8[0]) == 0
  assert len(set(order_8x8.tolist())) == 64

  pred_2d = soft_clamped_gradient_2d(dc_q, tau=0.05)
  assert pred_2d.shape == dc_q.shape
  assert bool(jnp.all(jnp.isfinite(pred_2d)))

  hdr_bits = soft_header_rate_bits(height=64, width=64)
  assert float(hdr_bits) > 0.0
  assert float(soft_noise_header_bits(enable_noise=True)) == 80.0
  assert float(soft_noise_header_bits(enable_noise=False)) == 0.0


def test_first_call_jit_cache_no_tracer_leak() -> None:
  """8. Regression test: cold-cache first-call @jax.jit must never leak DynamicJaxprTracer."""
  from jxl_wasserstein import entropy_proxy as ep_mod

  rgb = _load_kodak_crop("kodim01", crop_h=64, crop_w=64)
  st_1x1 = init_state_from_rgb_jax(rgb, distance=1.5)
  q_zero = jnp.zeros((3, 8, 8, 8, 8), dtype=jnp.float32)
  q_sub_dz = jnp.full((3, 8, 8, 8, 8), 0.25, dtype=jnp.float32)

  # 8a. 1x1 (topology=None) cold-cache @jax.jit on soft_ac_rate_bits (called twice across 2 JITs)
  ep_mod._NATURAL_AC_ORDER_CACHE.clear()
  ep_mod._ACS_TOPO_CACHE.clear()

  jit_ac_1x1_a = jax.jit(jax.grad(lambda q: soft_ac_rate_bits(q, topology=None)))
  jit_ac_1x1_b = jax.jit(
      jax.grad(lambda q: soft_ac_rate_bits(ste_round(q), topology=None))
  )
  g_1x1_a1 = jit_ac_1x1_a(q_zero)
  g_1x1_a2 = jit_ac_1x1_a(q_sub_dz)
  g_1x1_b1 = jit_ac_1x1_b(q_sub_dz)
  g_1x1_b2 = jit_ac_1x1_b(-q_sub_dz)

  for g in (g_1x1_a1, g_1x1_a2, g_1x1_b1, g_1x1_b2):
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g)) > 0.0
    # Canonical 1x1 LLF at [:, :, :, 0, 0] has identically 0.0 gradient
    assert float(jnp.max(jnp.abs(g[:, :, :, 0, 0]))) == 0.0

  # 8b. 1x1 cold-cache @jax.jit on soft_jxl_rate_proxy (called twice)
  ep_mod._NATURAL_AC_ORDER_CACHE.clear()
  ep_mod._ACS_TOPO_CACHE.clear()

  jit_proxy_1x1_a = jax.jit(
      jax.grad(lambda q: soft_jxl_rate_proxy(st_1x1, quant_ac_override=q))
  )
  jit_proxy_1x1_b = jax.jit(
      jax.grad(
          lambda q: soft_jxl_rate_proxy(st_1x1, quant_ac=ste_round(q))
      )
  )
  gp_1x1_1 = jit_proxy_1x1_a(q_zero)
  gp_1x1_2 = jit_proxy_1x1_a(q_sub_dz)
  gp_1x1_3 = jit_proxy_1x1_b(q_sub_dz)
  gp_1x1_4 = jit_proxy_1x1_b(-q_sub_dz)

  for g in (gp_1x1_1, gp_1x1_2, gp_1x1_3, gp_1x1_4):
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g)) > 0.0

  # 8c. Multi-block AcStrategy cold-cache @jax.jit on soft_ac_rate_bits (called twice)
  acs_mb = np.full((8, 8), ACS_DCT, dtype=np.uint8)
  acs_mb[0:4, 0:4] = ACS_DCT32X32
  acs_mb[0:2, 4:6] = ACS_DCT16X16
  acs_mb[2:4, 4] = ACS_DCT16X8
  acs_mb[4, 0:2] = ACS_DCT8X16
  acs_mb[6, 6] = ACS_DCT4X4
  acs_mb[6, 7] = ACS_IDENTITY
  acs_mb[7, 6] = ACS_AFV0
  st_mb = init_state_from_rgb_jax(rgb, distance=1.5, ac_strategy=acs_mb)
  _, llf_mb = get_first_block_and_llf_masks(acs_mb)
  llf_mb_5d = jnp.array(llf_mb[None, :, :, :, :], dtype=jnp.bool_)

  ep_mod._NATURAL_AC_ORDER_CACHE.clear()
  ep_mod._ACS_TOPO_CACHE.clear()

  jit_ac_mb_a = jax.jit(
      jax.grad(lambda q: soft_ac_rate_bits(q, ac_strategy=acs_mb))
  )
  jit_ac_mb_b = jax.jit(
      jax.grad(lambda q: soft_ac_rate_bits(ste_round(q), ac_strategy=acs_mb))
  )
  g_mb_a1 = jit_ac_mb_a(q_zero)
  g_mb_a2 = jit_ac_mb_a(q_sub_dz)
  g_mb_b1 = jit_ac_mb_b(q_sub_dz)
  g_mb_b2 = jit_ac_mb_b(-q_sub_dz)

  for g in (g_mb_a1, g_mb_a2, g_mb_b1, g_mb_b2):
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g)) > 0.0
    assert float(jnp.max(jnp.abs(jnp.where(llf_mb_5d, g, 0.0)))) == 0.0
    assert float(jnp.min(jnp.where(~llf_mb_5d, jnp.abs(g), jnp.inf))) > 0.0

  # 8d. Multi-block AcStrategy cold-cache @jax.jit on soft_jxl_rate_proxy (called twice)
  ep_mod._NATURAL_AC_ORDER_CACHE.clear()
  ep_mod._ACS_TOPO_CACHE.clear()

  jit_proxy_mb_a = jax.jit(
      jax.grad(
          lambda q: soft_jxl_rate_proxy(
              st_mb, quant_ac_override=q, ac_strategy=acs_mb
          )
      )
  )
  jit_proxy_mb_b = jax.jit(
      jax.grad(
          lambda q: soft_jxl_rate_proxy(
              st_mb, quant_ac=ste_round(q), ac_strategy=acs_mb
          )
      )
  )
  gp_mb_1 = jit_proxy_mb_a(q_zero)
  gp_mb_2 = jit_proxy_mb_a(q_sub_dz)
  gp_mb_3 = jit_proxy_mb_b(q_sub_dz)
  gp_mb_4 = jit_proxy_mb_b(-q_sub_dz)

  for g in (gp_mb_1, gp_mb_2, gp_mb_3, gp_mb_4):
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g)) > 0.0

  # Verify no jax.core.Tracer instances were stored in module-level caches
  for cached_tuple in ep_mod._NATURAL_AC_ORDER_CACHE.values():
    for arr in cached_tuple:
      assert not isinstance(arr, jax.core.Tracer)
      assert isinstance(arr, np.ndarray)
  for cached_topo in ep_mod._ACS_TOPO_CACHE.values():
    assert not isinstance(cached_topo.by_1x1, jax.core.Tracer)
    assert not isinstance(cached_topo.bx_1x1, jax.core.Tracer)
    assert isinstance(cached_topo.by_1x1, np.ndarray)
    assert isinstance(cached_topo.bx_1x1, np.ndarray)
    for grp in cached_topo.multi_groups:
      assert not isinstance(grp["by"], jax.core.Tracer)
      assert not isinstance(grp["bx"], jax.core.Tracer)
      assert isinstance(grp["by"], np.ndarray)
      assert isinstance(grp["bx"], np.ndarray)


def test_compute_spatial_rate_map_bits_sum_and_footprint_scaling_r3() -> None:
  """9. R3: Verify compute_spatial_rate_map_bits sum conservation and footprint metadata scaling."""
  rgb = _load_kodak_crop("kodim01", crop_h=128, crop_w=128)
  acs = np.full((16, 16), ACS_DCT, dtype=np.uint8)
  acs[0:4, 0:4] = ACS_DCT32X32
  acs[0:2, 4:6] = ACS_DCT16X16
  acs[2:4, 4] = ACS_DCT16X8
  acs[4, 0:2] = ACS_DCT8X16

  st = init_state_from_rgb_jax(rgb, distance=1.5, ac_strategy=acs)
  topo = StaticVarDCTTopology(st)
  bd = soft_jxl_rate_proxy(st, topology=topo, return_breakdown=True)
  assert isinstance(bd, RateBreakdown)

  rate_map = compute_spatial_rate_map_bits(st, topology=topo)
  for k in ("block_bits", "ac_block_bits", "dc_block_bits", "meta_block_bits"):
    assert rate_map[k].shape == (16, 16)
    assert bool(jnp.all(jnp.isfinite(rate_map[k])))

  # Sum conservation within < 1e-3
  assert float(jnp.abs(jnp.sum(rate_map["ac_block_bits"]) - bd.ac_bits)) < 1e-3
  assert float(jnp.abs(jnp.sum(rate_map["dc_block_bits"]) - bd.dc_bits)) < 1e-3
  assert float(jnp.abs(jnp.sum(rate_map["meta_block_bits"]) - bd.ac_meta_bits)) < 1e-3
  assert (
      float(
          jnp.abs(
              jnp.sum(rate_map["block_bits"])
              - (rate_map["total_bits"] - bd.header_bits)
          )
      )
      < 1e-3
  )

  # Larger transform footprints (DCT32X32, DCT16X16) have lower per-pixel/per-block metadata rate than DCT8
  meta_map = np.asarray(rate_map["meta_block_bits"])
  mean_meta_dct32 = float(np.mean(meta_map[0:4, 0:4]))
  mean_meta_dct16 = float(np.mean(meta_map[0:2, 4:6]))
  mean_meta_dct8 = float(np.mean(meta_map[8:16, 8:16]))
  assert mean_meta_dct32 < mean_meta_dct16 < mean_meta_dct8, (
      f"Expected DCT32X32 ({mean_meta_dct32}) < DCT16X16 ({mean_meta_dct16}) < DCT8 ({mean_meta_dct8})"
  )

  # Also verify across uniform grids of DCT8, DCT16X16, and DCT32X32
  acs_all_dct8 = np.full((8, 8), ACS_DCT, dtype=np.uint8)
  acs_all_dct16 = np.full((8, 8), ACS_DCT16X16, dtype=np.uint8)
  acs_all_dct32 = np.full((8, 8), ACS_DCT32X32, dtype=np.uint8)
  rgb64 = _load_kodak_crop("kodim01", crop_h=64, crop_w=64)
  st8 = init_state_from_rgb_jax(rgb64, distance=1.5, ac_strategy=acs_all_dct8)
  st16 = init_state_from_rgb_jax(rgb64, distance=1.5, ac_strategy=acs_all_dct16)
  st32 = init_state_from_rgb_jax(rgb64, distance=1.5, ac_strategy=acs_all_dct32)

  m8 = float(jnp.mean(compute_spatial_rate_map_bits(st8)["meta_block_bits"]))
  m16 = float(jnp.mean(compute_spatial_rate_map_bits(st16)["meta_block_bits"]))
  m32 = float(jnp.mean(compute_spatial_rate_map_bits(st32)["meta_block_bits"]))
  assert m32 < m16 < m8, f"Expected uniform m32 ({m32}) < m16 ({m16}) < m8 ({m8})"


def test_soft_dequant_matrices_rate_bits_and_gradients_r5() -> None:
  """10. R5: Verify soft_dequant_matrices_rate_bits overhead and gradients w.r.t. dequant_matrices and dc_quant."""
  rgb = _load_kodak_crop("kodim01", crop_h=64, crop_w=64)
  st = init_state_from_rgb_jax(rgb, distance=1.5)
  default_dq = get_default_dequant_matrices()

  # Default tables cost 0.0 extra header bits
  assert float(soft_dequant_matrices_rate_bits()) == 0.0
  dq_18 = jnp.stack(
      [
          jax.image.resize(
              jnp.asarray(default_dq[s], dtype=jnp.float32), (3, 8, 8), method="linear"
          ).reshape(3, 64)
          for s in range(18)
      ],
      axis=0,
  )
  def_dc = jnp.asarray(K_DC_QUANT, dtype=jnp.float32)
  assert np.isclose(
      float(soft_dequant_matrices_rate_bits(dq_18, def_dc, ac_strategy=st.ac_strategy)),
      0.0,
      atol=1e-4,
  )

  # Custom scaled dequant_matrices and dc_quant incur positive rate overhead
  custom_dq = dq_18 * 1.35
  custom_dc = def_dc * 1.25
  rate_custom = float(
      soft_dequant_matrices_rate_bits(custom_dq, custom_dc, ac_strategy=st.ac_strategy)
  )
  assert rate_custom > 100.0

  bd_def = soft_jxl_rate_proxy(st, return_breakdown=True)
  bd_cust = soft_jxl_rate_proxy(
      st,
      dequant_matrices=custom_dq,
      dc_quant_weights=custom_dc,
      return_breakdown=True,
  )
  assert isinstance(bd_def, RateBreakdown)
  assert isinstance(bd_cust, RateBreakdown)
  assert np.isclose(
      float(bd_cust.header_bits - bd_def.header_bits), rate_custom, rtol=1e-4
  )

  # Finite and non-zero gradients wrt both dequant_matrices and dc_quant
  grad_fn = jax.jit(
      jax.grad(
          lambda dq_in, dc_in: soft_jxl_rate_proxy(
              st, dequant_matrices=dq_in, dc_quant=dc_in
          ),
          argnums=(0, 1),
      )
  )
  g_dq, g_dc = grad_fn(custom_dq, custom_dc)
  assert bool(jnp.all(jnp.isfinite(g_dq)))
  assert bool(jnp.all(jnp.isfinite(g_dc)))
  assert float(jnp.linalg.norm(g_dq)) > 1e-3
  assert float(jnp.linalg.norm(g_dc)) > 1e-1


