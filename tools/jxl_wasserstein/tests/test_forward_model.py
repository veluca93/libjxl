"""Unit tests for differentiable JXL VarDCT, upsampling, XYB patch, and alpha compositing forward model."""

from __future__ import annotations

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest

from jxl_wasserstein.cpp_bridge import (
    ACS_COVERED_BLOCKS_X,
    ACS_COVERED_BLOCKS_Y,
    PATCH_BLEND_ADD,
    PatchEntry,
    compute_psnr,
    decode_with_djxl,
    dump_init_state,
    encode_alpha_composite,
    encode_patches_composite,
    encode_single_frame,
    verify_jxl_codestream,
)
from jxl_wasserstein.forward_model import (
    K_DC_QUANT,
    StaticVarDCTTopology,
    adaptive_dc_smoothing,
    adjust_quant_bias_hard,
    adjust_quant_bias_ste,
    compute_epf_inv_sigma_grid,
    dc_from_lowest_frequencies_for_strategy,
    epf_stage0,
    epf_stage1,
    epf_stage2,
    gaborish_3x3,
    gaborish_inverse_3x3,
    get_default_dequant_matrices,
    get_default_inv_dequant_matrices,
    init_state_from_rgb_jax,
    linear_to_srgb,
    lowest_frequencies_from_dc_for_strategy,
    project_custom_dequant_matrices,
    reconstruct_alpha_composite,
    reconstruct_alpha_composite_xyb,
    reconstruct_patch_composite,
    reconstruct_single_frame,
    reconstruct_single_frame_xyb,
    srgb_to_linear,
    srgb_to_xyb,
    ste_dequant_matrices,
    ste_quantize_alpha_01,
    transform_from_pixels_for_strategy,
    transform_to_pixels_for_strategy,
    upsample_jxl,
    xyb_to_srgb,
)

KODAK_DIR = Path(
    "/usr/local/google/home/veluca/libjxl/image-corpus/qoi_benchmark_suite/photo_kodak"
)


def _load_kodak_crop(name: str = "kodim01.png", w: int = 128, h: int = 128) -> np.ndarray:
  img = Image.open(KODAK_DIR / name).convert("RGB")
  iw, ih = img.size
  x0 = (iw - w) // 2
  y0 = (ih - h) // 2
  return np.asarray(img.crop((x0, y0, x0 + w, y0 + h)), dtype=np.float32) / 255.0


def test_color_conversions_xyb_srgb_roundtrip_and_gradients() -> None:
  rng = np.random.default_rng(101)
  srgb_hwc = jnp.asarray(rng.uniform(0.0, 1.0, size=(32, 40, 3)).astype(np.float32))
  lin = srgb_to_linear(srgb_hwc)
  srgb_rt1 = linear_to_srgb(lin)
  assert compute_psnr(np.asarray(srgb_hwc), np.asarray(srgb_rt1), peak=1.0) >= 90.0

  xyb_hwc = srgb_to_xyb(srgb_hwc)
  srgb_rt2 = xyb_to_srgb(xyb_hwc)
  assert xyb_hwc.shape == (32, 40, 3)
  assert srgb_rt2.shape == (32, 40, 3)
  assert compute_psnr(np.asarray(srgb_hwc), np.asarray(srgb_rt2), peak=1.0) >= 80.0

  # CHW layout equivalence
  srgb_chw = jnp.moveaxis(srgb_hwc, -1, 0)
  xyb_chw = srgb_to_xyb(srgb_chw)
  srgb_rt_chw = xyb_to_srgb(xyb_chw)
  np.testing.assert_allclose(
      np.asarray(srgb_rt_chw), np.moveaxis(np.asarray(srgb_rt2), -1, 0), atol=1e-6
  )

  # Finite non-zero gradients
  grad_srgb = np.asarray(jax.grad(lambda x: jnp.mean(jnp.square(xyb_to_srgb(srgb_to_xyb(x)))))(srgb_hwc))
  assert bool(np.all(np.isfinite(grad_srgb)))
  assert float(np.linalg.norm(grad_srgb)) > 1e-4


@pytest.mark.parametrize("strategy_id", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
def test_dct_idct_and_llf_from_dc_across_ac_strategies(strategy_id: int) -> None:
  rng = np.random.default_rng(200 + strategy_id)
  cx = ACS_COVERED_BLOCKS_X[strategy_id]
  cy = ACS_COVERED_BLOCKS_Y[strategy_id]
  cy_l, cx_l = min(cy, cx), max(cy, cx)

  pixels = jnp.asarray(rng.normal(size=(3, 8 * cy, 8 * cx)).astype(np.float32))
  coeffs = transform_from_pixels_for_strategy(strategy_id, pixels)
  assert coeffs.shape == (3, 8 * cy_l, 8 * cx_l)

  pixels_rt = transform_to_pixels_for_strategy(strategy_id, coeffs)
  assert pixels_rt.shape == (3, 8 * cy, 8 * cx)
  np.testing.assert_allclose(np.asarray(pixels_rt), np.asarray(pixels), atol=1e-4, rtol=1e-4)

  dc_block = dc_from_lowest_frequencies_for_strategy(strategy_id, coeffs)
  assert dc_block.shape == (3, cy, cx)
  llf_rt = lowest_frequencies_from_dc_for_strategy(strategy_id, dc_block)
  assert llf_rt.shape == (3, cy_l, cx_l)
  np.testing.assert_allclose(
      np.asarray(llf_rt), np.asarray(coeffs[:, :cy_l, :cx_l]), atol=1e-5, rtol=1e-5
  )


def test_dequant_dc_adaptive_dc_smoothing_and_dequant_ac() -> None:
  # AdjustQuantBias hard & STE
  q = jnp.zeros((3, 5), dtype=jnp.float32)
  q = q.at[:, 0].set(-2.0).at[:, 1].set(-1.0).at[:, 2].set(0.0).at[:, 3].set(1.0).at[:, 4].set(2.0)
  hard_out = np.asarray(adjust_quant_bias_hard(q))
  assert np.all(hard_out[:, 2] == 0.0)
  assert np.all(np.abs(hard_out[:, 1]) < 1.0)
  assert np.all(np.abs(hard_out[:, 3]) < 1.0)
  assert np.all(np.abs(hard_out[:, 4]) == 2.0 - 0.145 / 2.0)

  ste_out = np.asarray(adjust_quant_bias_ste(q, hard=False))
  np.testing.assert_allclose(ste_out, hard_out, atol=1e-6)
  grad_q = np.asarray(jax.grad(lambda x: jnp.sum(adjust_quant_bias_ste(x, hard=False)))(q))
  np.testing.assert_allclose(grad_q, 1.0, atol=1e-6)

  # AdaptiveDCSmoothing preserves outer border and smooths small interior steps
  dc = jnp.ones((3, 6, 6), dtype=jnp.float32) * 0.5
  dc = dc.at[:, 2, 2].set(0.5001)
  mul_dc = jnp.array([1e-3, 1e-3, 1e-3], dtype=jnp.float32)
  dc_sm = np.asarray(adaptive_dc_smoothing(dc, mul_dc))
  np.testing.assert_allclose(dc_sm[:, 0, :], 0.5, atol=1e-7)
  np.testing.assert_allclose(dc_sm[:, -1, :], 0.5, atol=1e-7)
  np.testing.assert_allclose(dc_sm[:, :, 0], 0.5, atol=1e-7)
  np.testing.assert_allclose(dc_sm[:, :, -1], 0.5, atol=1e-7)
  assert abs(dc_sm[1, 2, 2] - 0.5) < abs(0.5001 - 0.5)


def test_gaborish_and_epf_stages_0_1_2_differentiability() -> None:
  rng = np.random.default_rng(303)
  xyb = jnp.asarray((0.35 + 0.002 * rng.normal(size=(3, 32, 32))).astype(np.float32))

  # Constant plane is invariant under Gaborish, inverse Gaborish, and EPF stages
  const_xyb = jnp.full((3, 16, 16), 0.35, dtype=jnp.float32)
  np.testing.assert_allclose(np.asarray(gaborish_3x3(const_xyb)), np.asarray(const_xyb), atol=1e-6)
  np.testing.assert_allclose(
      np.asarray(gaborish_inverse_3x3(const_xyb)), np.asarray(const_xyb), atol=1e-6
  )

  raw_q = jnp.full((4, 4), 12.0, dtype=jnp.float32)
  sharp = jnp.full((4, 4), 4, dtype=jnp.int32)

  def _pipeline(x: jax.Array, rq: jax.Array) -> jax.Array:
    y = gaborish_3x3(x)
    inv_sig = compute_epf_inv_sigma_grid(rq, sharp, global_scale=4096)
    y = epf_stage0(y, inv_sig)
    y = epf_stage1(y, inv_sig)
    y = epf_stage2(y, inv_sig)
    return jnp.mean(jnp.square(y))

  val, (g_x, g_rq) = jax.jit(jax.value_and_grad(_pipeline, argnums=(0, 1)))(xyb, raw_q)
  assert np.isfinite(float(val))
  assert bool(np.all(np.isfinite(np.asarray(g_x))))
  assert bool(np.all(np.isfinite(np.asarray(g_rq))))
  assert float(np.linalg.norm(np.asarray(g_x))) > 1e-4
  assert float(np.linalg.norm(np.asarray(g_rq))) > 1e-7


@pytest.mark.parametrize(
    "img_name,size_wh,distance,effort",
    [
        ("kodim01.png", (128, 128), 0.5, 3),
        ("kodim01.png", (128, 128), 1.0, 7),
        ("kodim02.png", (128, 128), 2.0, 7),
        ("kodim02.png", (128, 128), 4.0, 7),
        ("kodim03.png", (67, 53), 1.5, 7),
        ("kodim01.png", (260, 196), 1.5, 3),
    ],
)
def test_reconstruct_single_frame_matches_djxl_across_distances_and_dimensions(
    img_name: str,
    size_wh: tuple[int, int],
    distance: float,
    effort: int,
    tmp_path: Path,
) -> None:
  w, h = size_wh
  crop_01 = _load_kodak_crop(img_name, w=w, h=h)
  jxl_path = tmp_path / "dump.jxl"
  state = dump_init_state(crop_01, distance=distance, effort=effort, output_jxl=jxl_path)

  djxl_srgb = decode_with_djxl(jxl_path).astype(np.float32) / 255.0
  jax_srgb = np.asarray(reconstruct_single_frame(state, include_noise=False))
  assert jax_srgb.shape == (h, w, 3)

  psnr_val = compute_psnr(jax_srgb, djxl_srgb, peak=1.0)
  assert psnr_val >= 48.0, (
      f"Single-frame PSNR {psnr_val:.2f} dB < 48 dB for {img_name} {size_wh} d={distance} e={effort}"
  )


def test_pure_jax_init_state_from_rgb_jax_without_libjxl_choices() -> None:
  dq = get_default_dequant_matrices()
  inv_dq = get_default_inv_dequant_matrices()
  assert len(dq) == 27
  assert len(inv_dq) == 27

  crop_01 = _load_kodak_crop("kodim01.png", w=128, h=128)
  h8, w8 = 16, 16
  custom_acs = np.zeros((h8, w8), dtype=np.uint8)
  custom_acs[0:2, 0:2] = 4  # DCT16X16
  custom_acs[2, 0] = 2  # DCT2X2
  custom_acs[2, 1] = 3  # DCT4X4
  custom_acs[4:6, 0] = 6  # DCT16X8
  custom_acs[0, 4:6] = 7  # DCT8X16
  custom_acs[3, 0] = 14  # AFV0

  for d in (0.5, 1.5, 4.0):
    state = init_state_from_rgb_jax(crop_01, distance=d, ac_strategy=custom_acs)
    jxl_bytes, stats = encode_single_frame(state)
    assert stats["file_bytes"] > 0
    ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
    djxl_srgb = ver["rgb_djxl"].astype(np.float32) / 255.0
    jax_srgb = np.asarray(reconstruct_single_frame(state, hard=True))
    psnr_parity = compute_psnr(jax_srgb, djxl_srgb, peak=1.0)
    assert psnr_parity >= 48.0


def test_upsample_jxl_2x_4x_8x_with_5x5_min_max_clamping() -> None:
  # Step edge from 0.2 to 0.8 must never overshoot outside [0.2, 0.8]
  step = np.full((3, 16, 16), 0.2, dtype=np.float32)
  step[:, :, 8:] = 0.8
  step_j = jnp.asarray(step)

  for u in (1, 2, 4, 8):
    out = np.asarray(upsample_jxl(step_j, u, out_hw=(16 * u - 1, 16 * u - 2) if u > 1 else None))
    if u > 1:
      assert out.shape == (3, 16 * u - 1, 16 * u - 2)
    assert float(np.min(out)) >= 0.2 - 1e-6
    assert float(np.max(out)) <= 0.8 + 1e-6

  grad_in = np.asarray(jax.grad(lambda x: jnp.mean(jnp.square(upsample_jxl(x, 4))))(step_j))
  assert bool(np.all(np.isfinite(grad_in)))
  assert float(np.linalg.norm(grad_in)) > 1e-4


@pytest.mark.parametrize("upsampling", [2, 4, 8])
@pytest.mark.parametrize("explicit_patch", [False, True])
def test_reconstruct_patch_composite_matches_djxl_all_upsampling_and_blend_modes(
    upsampling: int,
    explicit_patch: bool,
) -> None:
  rgb_01 = _load_kodak_crop("kodim02.png", w=128, h=128)
  h_low = (128 + upsampling - 1) // upsampling
  w_low = (128 + upsampling - 1) // upsampling
  rgb_low = np.asarray(
      Image.fromarray(np.round(rgb_01 * 255.0).astype(np.uint8)).resize(
          (w_low, h_low), Image.Resampling.BICUBIC
      ),
      dtype=np.float32,
  ) / 255.0

  ref_state = init_state_from_rgb_jax(rgb_low * 0.6, distance=2.0)
  main_state = init_state_from_rgb_jax(rgb_01 * 0.4, distance=1.5)
  patches = (
      [PatchEntry(ref=0, x0=0, y0=0, xsize=128, ysize=128, x=0, y=0, blend_mode=PATCH_BLEND_ADD)]
      if explicit_patch
      else None
  )

  jxl_bytes, _ = encode_patches_composite(ref_state, main_state, upsampling=upsampling, patches=patches)
  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  djxl_srgb = ver["rgb_djxl"].astype(np.float32) / 255.0

  jax_srgb = np.asarray(
      reconstruct_patch_composite(ref_state, main_state, upsampling=upsampling, patches=patches)
  )
  assert jax_srgb.shape == (128, 128, 3)
  psnr_val = compute_psnr(jax_srgb, djxl_srgb, peak=1.0)
  assert psnr_val >= 48.0, (
      f"Patch composite PSNR {psnr_val:.2f} dB < 48 dB for U={upsampling} explicit={explicit_patch}"
  )

  if upsampling == 2 and explicit_patch:
    with pytest.raises(ValueError):
      reconstruct_patch_composite(
          ref_state,
          main_state,
          upsampling=2,
          patches=[PatchEntry(ref=0, x0=8, y0=8, xsize=40, ysize=32, x=8, y=8, blend_mode=PATCH_BLEND_ADD)],
      )
    with pytest.raises(ValueError):
      reconstruct_patch_composite(
          ref_state,
          main_state,
          upsampling=2,
          patches=[PatchEntry(ref=0, x0=0, y0=0, xsize=128, ysize=128, x=0, y=0, blend_mode=1)],
      )


@pytest.mark.parametrize("upsampling", [2, 4, 8])
@pytest.mark.parametrize("dim_shift", [0, 1, 2, 3])
def test_reconstruct_alpha_composite_matches_djxl_all_upsampling_and_dim_shifts(
    upsampling: int,
    dim_shift: int,
) -> None:
  rgb_01 = _load_kodak_crop("kodim03.png", w=64, h=64)
  h_low = (64 + upsampling - 1) // upsampling
  w_low = (64 + upsampling - 1) // upsampling
  rgb_low = np.asarray(
      Image.fromarray(np.round(rgb_01 * 255.0).astype(np.uint8)).resize(
          (w_low, h_low), Image.Resampling.BICUBIC
      ),
      dtype=np.float32,
  ) / 255.0

  ref_state = init_state_from_rgb_jax(rgb_low, distance=2.0)
  main_state = init_state_from_rgb_jax(rgb_01, distance=1.2)

  h_a = (64 + (1 << dim_shift) - 1) // (1 << dim_shift)
  w_a = (64 + (1 << dim_shift) - 1) // (1 << dim_shift)
  yy, xx = np.meshgrid(
      np.linspace(0.0, 1.0, h_a, dtype=np.float32),
      np.linspace(0.0, 1.0, w_a, dtype=np.float32),
      indexing="ij",
  )
  alpha_01 = np.clip(0.5 + 0.5 * np.sin(3.0 * xx) * np.cos(3.0 * yy), 0.0, 1.0)
  alpha_u8 = np.round(alpha_01 * 255.0).astype(np.uint8)

  jxl_bytes, _ = encode_alpha_composite(
      ref_state, main_state, upsampling=upsampling, dim_shift=dim_shift, alpha_u8=alpha_u8
  )
  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  djxl_srgb = ver["rgb_djxl"].astype(np.float32) / 255.0

  jax_srgb = np.asarray(
      reconstruct_alpha_composite(
          ref_state,
          main_state,
          upsampling=upsampling,
          dim_shift=dim_shift,
          alpha_01=alpha_u8.astype(np.float32) / 255.0,
      )
  )
  assert jax_srgb.shape == (64, 64, 3)
  psnr_val = compute_psnr(jax_srgb, djxl_srgb, peak=1.0)
  assert psnr_val >= 48.0, (
      f"Alpha composite PSNR {psnr_val:.2f} dB < 48 dB for U={upsampling} D={dim_shift}"
  )


def test_odd_dimensions_and_end_to_end_differentiability() -> None:
  # Odd non-multiple-of-8 dimensions: 67x53 (H=53, W=67)
  h, w = 53, 67
  rgb_01 = _load_kodak_crop("kodim01.png", w=w, h=h)
  u = 4
  dim_shift = 2
  h_low = (h + u - 1) // u
  w_low = (w + u - 1) // u
  rgb_low = np.asarray(
      Image.fromarray(np.round(rgb_01 * 255.0).astype(np.uint8)).resize(
          (w_low, h_low), Image.Resampling.BICUBIC
      ),
      dtype=np.float32,
  ) / 255.0

  ref_state = init_state_from_rgb_jax(rgb_low, distance=2.0)
  main_state = init_state_from_rgb_jax(rgb_01, distance=1.5)

  h_a = (h + (1 << dim_shift) - 1) // (1 << dim_shift)
  w_a = (w + (1 << dim_shift) - 1) // (1 << dim_shift)
  alpha_u8 = np.full((h_a, w_a), 140, dtype=np.uint8)
  alpha_01 = jnp.asarray(alpha_u8.astype(np.float32) / 255.0)

  jxl_bytes, _ = encode_alpha_composite(
      ref_state, main_state, upsampling=u, dim_shift=dim_shift, alpha_u8=alpha_u8
  )
  djxl_srgb = decode_with_djxl(jxl_bytes).astype(np.float32) / 255.0
  jax_srgb = np.asarray(
      reconstruct_alpha_composite(
          ref_state, main_state, upsampling=u, dim_shift=dim_shift, alpha_01=alpha_01
      )
  )
  assert jax_srgb.shape == (h, w, 3)
  assert compute_psnr(jax_srgb, djxl_srgb, peak=1.0) >= 48.0

  # Verify JIT value_and_grad wrt ref, main, and alpha_01
  ref_topo = StaticVarDCTTopology(ref_state)
  main_topo = StaticVarDCTTopology(main_state)
  target = jnp.asarray(rgb_01, dtype=jnp.float32)

  ref_ac = jnp.asarray(ref_state.quant_ac, dtype=jnp.float32)
  ref_dc = jnp.asarray(ref_state.dc_quant, dtype=jnp.float32)
  ref_rq = jnp.asarray(ref_state.raw_quant_field, dtype=jnp.float32)
  ref_yx = jnp.asarray(ref_state.ytox_map, dtype=jnp.float32)
  ref_yb = jnp.asarray(ref_state.ytob_map, dtype=jnp.float32)

  main_ac = jnp.asarray(main_state.quant_ac, dtype=jnp.float32)
  main_dc = jnp.asarray(main_state.dc_quant, dtype=jnp.float32)
  main_rq = jnp.asarray(main_state.raw_quant_field, dtype=jnp.float32)
  main_yx = jnp.asarray(main_state.ytox_map, dtype=jnp.float32)
  main_yb = jnp.asarray(main_state.ytob_map, dtype=jnp.float32)

  def _loss_alpha(r_ac, r_dc, m_ac, m_dc, a_01):
    pred = reconstruct_alpha_composite(
        ref_topo,
        main_topo,
        upsampling=u,
        dim_shift=dim_shift,
        alpha_01=a_01,
        ref_quant_ac=r_ac,
        ref_dc_quant=r_dc,
        ref_raw_quant_field=ref_rq,
        ref_ytox_map=ref_yx,
        ref_ytob_map=ref_yb,
        main_quant_ac=m_ac,
        main_dc_quant=m_dc,
        main_raw_quant_field=main_rq,
        main_ytox_map=main_yx,
        main_ytob_map=main_yb,
    )
    return jnp.mean(jnp.square(pred - target))

  loss_val, grads = jax.jit(jax.value_and_grad(_loss_alpha, argnums=(0, 1, 2, 3, 4)))(
      ref_ac, ref_dc, main_ac, main_dc, alpha_01
  )
  assert np.isfinite(float(loss_val))
  for g in grads:
    g_np = np.asarray(g)
    assert bool(np.all(np.isfinite(g_np)))
    assert float(np.linalg.norm(g_np)) > 1e-6


def test_pure_python_default_dequant_matrices_matches_libjxl_without_dump_init_state_r1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """R1: Verify pure-Python get_default_dequant_matrices() matches libjxl < 1e-5 without dump_init_state."""
  from jxl_wasserstein import cpp_bridge

  # Reference state from libjxl binary before monkeypatching
  ref_st = dump_init_state(np.zeros((16, 16, 3), dtype=np.uint8), distance=1.0, effort=3)

  def _forbid_dump(*args, **kwargs):
    raise RuntimeError("dump_init_state must not be called by get_default_dequant_matrices()")

  monkeypatch.setattr(cpp_bridge, "dump_init_state", _forbid_dump)
  get_default_dequant_matrices.cache_clear()
  get_default_inv_dequant_matrices.cache_clear()

  dq_py = get_default_dequant_matrices()
  assert len(dq_py) == 27
  for s in range(18):
    mat_py = dq_py[s]
    mat_cpp = ref_st.dequant_matrices[s]
    assert mat_py.shape == mat_cpp.shape
    rel_err = float(
        np.max(np.abs(mat_py - mat_cpp) / np.maximum(np.abs(mat_cpp), 1e-12))
    )
    assert rel_err < 1e-5, f"Strategy {s} relative error {rel_err} >= 1e-5"

  for s in range(27):
    mat_py = dq_py[s]
    mat_cpp = ref_st.dequant_matrices[s]
    assert mat_py.shape == mat_cpp.shape
    rel_err = float(
        np.max(np.abs(mat_py - mat_cpp) / np.maximum(np.abs(mat_cpp), 1e-12))
    )
    assert rel_err < 1e-5, f"Strategy {s} relative error {rel_err} >= 1e-5"


def test_reconstruct_alpha_composite_xyb_and_post_patch_noise_r2() -> None:
  """R2: Verify XYB PatchDictionary alpha blending and post-patch XYB noise synthesis."""
  rgb_01 = _load_kodak_crop("kodim03.png", w=64, h=64)
  rgb_low = _load_kodak_crop("kodim03.png", w=32, h=32)
  ref_state = init_state_from_rgb_jax(rgb_low, distance=2.0)
  main_state = init_state_from_rgb_jax(rgb_01, distance=1.2)
  alpha_01 = jnp.full((32, 32), 0.6, dtype=jnp.float32)

  comp_xyb = reconstruct_alpha_composite_xyb(
      ref_state,
      main_state,
      upsampling=2,
      dim_shift=1,
      alpha_01=alpha_01,
      alpha_blend_mode=5,
  )
  assert comp_xyb.shape == (3, 64, 64)
  srgb_no_noise = reconstruct_alpha_composite(
      ref_state,
      main_state,
      upsampling=2,
      dim_shift=1,
      alpha_01=alpha_01,
      enable_noise=False,
  )
  np.testing.assert_allclose(
      np.asarray(srgb_no_noise),
      np.asarray(xyb_to_srgb(jnp.moveaxis(comp_xyb, 0, -1))),
      atol=1e-5,
  )

  noise_lut = (0.01, 0.015, 0.02, 0.025, 0.03, 0.03, 0.03, 0.03)
  srgb_with_noise = reconstruct_alpha_composite(
      ref_state,
      main_state,
      upsampling=2,
      dim_shift=1,
      alpha_01=alpha_01,
      enable_noise=True,
      noise_lut=noise_lut,
  )
  assert srgb_with_noise.shape == (64, 64, 3)
  assert float(jnp.mean(jnp.abs(srgb_with_noise - srgb_no_noise))) > 1e-4

  # Verify upsampling=1 (full-res reference frame) also works in pure JAX
  srgb_u1 = reconstruct_alpha_composite(
      main_state,
      main_state,
      upsampling=1,
      dim_shift=1,
      alpha_01=alpha_01,
  )
  assert srgb_u1.shape == (64, 64, 3)


def test_multiblock_ac_strategy_forward_model_matches_djxl_and_gradients_r4() -> None:
  """R4: Verify reconstruct_single_frame on mixed AcStrategy grids (DCT8..DCT32X32, DCT4X4, DCT2X2, IDENTITY, AFV0..3)."""
  crop_01 = _load_kodak_crop("kodim02.png", w=128, h=128)
  acs = np.zeros((16, 16), dtype=np.uint8)
  acs[0:4, 0:4] = 5  # DCT32X32
  acs[0:2, 4:6] = 4  # DCT16X16
  acs[2:4, 4] = 6  # DCT16X8
  acs[4, 0:2] = 7  # DCT8X16
  acs[6, 0] = 1  # IDENTITY
  acs[6, 1] = 2  # DCT2X2
  acs[6, 2] = 3  # DCT4X4
  acs[7, 0] = 14  # AFV0
  acs[7, 1] = 15  # AFV1
  acs[7, 2] = 16  # AFV2
  acs[7, 3] = 17  # AFV3

  state = init_state_from_rgb_jax(crop_01, distance=1.2, ac_strategy=acs)
  jxl_bytes, _ = encode_single_frame(state)
  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  djxl_srgb = ver["rgb_djxl"].astype(np.float32) / 255.0
  jax_srgb = np.asarray(reconstruct_single_frame(state, hard=True))
  psnr_val = compute_psnr(jax_srgb, djxl_srgb, peak=1.0)
  assert psnr_val >= 48.0, f"Mixed multi-block AcStrategy PSNR {psnr_val:.2f} dB < 48 dB"

  topo = StaticVarDCTTopology(state)
  q_ac = jnp.asarray(state.quant_ac, dtype=jnp.float32)
  grad_q = np.asarray(
      jax.jit(
          jax.grad(
              lambda q: jnp.sum(
                  jnp.square(
                      reconstruct_single_frame_xyb(topo, quant_ac=q, epf_iters_override=0, gab_override=False)
                  )
              )
          )
      )(q_ac + 0.5)
  )
  assert bool(np.all(np.isfinite(grad_q)))
  assert float(np.linalg.norm(grad_q)) > 1e-4


def test_custom_dequant_matrices_and_dc_quant_forward_and_ste_projection_r5() -> None:
  """R5: Verify custom dequant_matrices scaling doubles AC amplitude, gradients, and STE projection."""
  crop_01 = _load_kodak_crop("kodim01.png", w=64, h=64)
  state = init_state_from_rgb_jax(crop_01, distance=1.5)
  topo = StaticVarDCTTopology(state)

  # Compare AC contribution in XYB (before sRGB clamping, with gab=False, epf_iters=0)
  q_zero = jnp.zeros_like(jnp.asarray(state.quant_ac, dtype=jnp.float32))
  xyb_dc_only = reconstruct_single_frame_xyb(
      topo, quant_ac=q_zero, gab_override=False, epf_iters_override=0
  )
  dq_default = {s: jnp.asarray(m, dtype=jnp.float32) for s, m in get_default_dequant_matrices().items()}
  dq_double = {s: 2.0 * m for s, m in dq_default.items()}

  xyb_1x = reconstruct_single_frame_xyb(
      topo, dequant_matrices=dq_default, gab_override=False, epf_iters_override=0
  )
  xyb_2x = reconstruct_single_frame_xyb(
      topo, dequant_matrices=dq_double, gab_override=False, epf_iters_override=0
  )
  ac_diff_1x = np.asarray(xyb_1x - xyb_dc_only)
  ac_diff_2x = np.asarray(xyb_2x - xyb_dc_only)
  np.testing.assert_allclose(ac_diff_2x, 2.0 * ac_diff_1x, rtol=1e-4, atol=1e-5)

  # Also verify with (18, 3, 64) tensor representation and custom (3,) dc_quant weights
  dq_18 = jnp.stack(
      [
          jax.image.resize(dq_default[s], (3, 8, 8), method="linear").reshape(3, 64)
          for s in range(18)
      ],
      axis=0,
  )
  dc_w = jnp.asarray(K_DC_QUANT, dtype=jnp.float32) * 1.15
  target = jnp.asarray(crop_01, dtype=jnp.float32)

  def _loss_dq_dc(dq_param: jax.Array, dc_param: jax.Array) -> jax.Array:
    dq_ste, dc_ste = ste_dequant_matrices(dq_param, dc_param)
    recon = reconstruct_single_frame(
        topo,
        params_override={"dequant_matrices": dq_ste, "dc_quant": dc_ste},
    )
    return jnp.mean(jnp.square(recon - target))

  loss_val, (g_dq, g_dc) = jax.jit(jax.value_and_grad(_loss_dq_dc, argnums=(0, 1)))(
      dq_18 * 1.25, dc_w
  )
  assert np.isfinite(float(loss_val))
  assert bool(np.all(np.isfinite(np.asarray(g_dq))))
  assert bool(np.all(np.isfinite(np.asarray(g_dc))))
  assert float(np.linalg.norm(np.asarray(g_dq))) > 1e-6
  assert float(np.linalg.norm(np.asarray(g_dc))) > 1e-4

  # Verify project_custom_dequant_matrices / ste_dequant_matrices enforces positivity and bounds
  # while preserving straight-through gradients
  bad_dq = dq_18.at[0, 0, :8].set(-5.0).at[0, 1, :8].set(500.0)
  bad_dc = jnp.array([-0.1, 50.0, 0.0], dtype=jnp.float32)
  proj_dq, proj_dc = project_custom_dequant_matrices(bad_dq, bad_dc)
  assert float(jnp.min(proj_dq)) > 0.0
  assert float(jnp.max(proj_dq)) <= 10.0
  assert float(jnp.min(proj_dc)) > 0.0
  assert float(jnp.max(proj_dc)) <= 1.0

  ste_dq, ste_dc = ste_dequant_matrices(bad_dq, bad_dc)
  np.testing.assert_allclose(np.asarray(ste_dq), np.asarray(proj_dq), atol=1e-6)
  np.testing.assert_allclose(np.asarray(ste_dc), np.asarray(proj_dc), atol=1e-6)

  grad_ste_dq, grad_ste_dc = jax.grad(
      lambda dq_in, dc_in: jnp.sum(ste_dequant_matrices(dq_in, dc_in)[0])
      + jnp.sum(ste_dequant_matrices(dq_in, dc_in)[1]),
      argnums=(0, 1),
  )(bad_dq, bad_dc)
  np.testing.assert_allclose(np.asarray(grad_ste_dq), 1.0, atol=1e-6)
  np.testing.assert_allclose(np.asarray(grad_ste_dc), 1.0, atol=1e-6)

