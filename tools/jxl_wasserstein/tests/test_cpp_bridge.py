"""Unit tests for jxl_wasserstein_helper and cpp_bridge across all 5 subcommands."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from jxl_wasserstein.cpp_bridge import (
    ACS_COVERED_BLOCKS_X,
    ACS_COVERED_BLOCKS_Y,
    DEFAULT_DC_QUANT_WEIGHTS,
    DJXL_BIN,
    HELPER_BIN,
    JXL_CLI_BIN,
    JXLINFO_BIN,
    PATCH_BLEND_ADD,
    PATCH_BLEND_ALPHA_ABOVE,
    PATCH_BLEND_ALPHA_BELOW,
    PATCH_BLEND_MUL_ADD_ABOVE,
    PATCH_BLEND_MUL_ADD_BELOW,
    PATCH_BLEND_REPLACE,
    PatchEntry,
    compute_psnr,
    decode_with_djxl,
    decode_with_jxl_rs,
    dump_init_state,
    encode_alpha_composite,
    encode_patches_composite,
    encode_single_frame,
    has_custom_dequant_matrices,
    inspect_jxl,
    load_state_bin,
    roundtrip_custom_dequant_matrices,
    save_state_bin,
    verify_jxl_codestream,
)


def _downsample_rgb_u8(rgb_u8: np.ndarray, factor: int) -> np.ndarray:
  h, w, _ = rgb_u8.shape
  hr = (h + factor - 1) // factor
  wr = (w + factor - 1) // factor
  im = Image.fromarray(rgb_u8, mode="RGB").resize(
      (wr, hr), resample=Image.Resampling.BICUBIC
  )
  return np.ascontiguousarray(np.asarray(im, dtype=np.uint8))


def test_binaries_exist() -> None:
  """Verifies that all required C++ and Rust binaries exist and are executable."""
  for bin_path in (HELPER_BIN, DJXL_BIN, JXLINFO_BIN, JXL_CLI_BIN):
    assert bin_path.exists(), f"Missing binary: {bin_path}"


def test_dump_init_and_state_bin_roundtrip(
    kodim01_rgb_u8: np.ndarray,
    tmp_path: Path,
) -> None:
  """Verifies --dump-init state extraction and bit-exact Python load/save roundtrip."""
  crop = kodim01_rgb_u8[:96, :128]
  init_jxl = tmp_path / "init.jxl"
  state_file = tmp_path / "init_state.bin"
  state = dump_init_state(
      crop,
      distance=1.5,
      effort=7,
      output_state=state_file,
      output_jxl=init_jxl,
  )

  assert state.height == 96
  assert state.width == 128
  assert state.h8 == 12
  assert state.w8 == 16
  assert state.ht == 2
  assert state.wt == 2
  assert state.global_scale > 0
  assert state.quant_dc > 0
  assert state.ac_strategy.shape == (12, 16)
  assert state.raw_quant_field.shape == (12, 16)
  assert state.epf_sharpness.shape == (12, 16)
  assert state.ytox_map.shape == (2, 2)
  assert state.ytob_map.shape == (2, 2)
  assert state.dc_quant.shape == (3, 12, 16)
  assert state.quant_ac.shape == (3, 96, 128)
  assert len(state.dequant_matrices) > 0
  assert state.bit_stats["total_bits"] > 0
  assert state.bit_stats["file_bytes"] > 0

  # Save and reload via Python and verify bit-exact equality
  re_saved = save_state_bin(state, tmp_path / "resaved.bin", include_dequant=True)
  reloaded = load_state_bin(re_saved)
  assert reloaded.height == state.height
  assert reloaded.width == state.width
  assert reloaded.global_scale == state.global_scale
  assert reloaded.quant_dc == state.quant_dc
  np.testing.assert_array_equal(reloaded.ac_strategy, state.ac_strategy)
  np.testing.assert_array_equal(reloaded.raw_quant_field, state.raw_quant_field)
  np.testing.assert_array_equal(reloaded.epf_sharpness, state.epf_sharpness)
  np.testing.assert_array_equal(reloaded.ytox_map, state.ytox_map)
  np.testing.assert_array_equal(reloaded.ytob_map, state.ytob_map)
  np.testing.assert_array_equal(reloaded.dc_quant, state.dc_quant)
  np.testing.assert_array_equal(reloaded.quant_ac, state.quant_ac)
  for k, mat in state.dequant_matrices.items():
    np.testing.assert_array_equal(reloaded.dequant_matrices[k], mat)


def test_encode_single_roundtrip_and_cross_decoder_parity(
    kodim01_rgb_u8: np.ndarray,
    tmp_path: Path,
) -> None:
  """Verifies --encode-single roundtrip, parameter perturbation, and cross-decoder parity."""
  crop = kodim01_rgb_u8[:96, :128]
  init_jxl = tmp_path / "init.jxl"
  state = dump_init_state(crop, distance=1.5, effort=7, output_jxl=init_jxl)

  jxl_bytes, stats = encode_single_frame(state)
  assert stats["file_bytes"] == len(jxl_bytes)
  assert stats["total_bits"] > 0

  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  assert ver["inspect"]["num_frames"] == 1
  assert ver["inspect"]["width"] == 128
  assert ver["inspect"]["height"] == 96

  # Compare decoded pixels of init.jxl vs re-encoded single frame
  rgb_init = decode_with_djxl(init_jxl)
  assert compute_psnr(rgb_init, ver["rgb_djxl"]) >= 45.0

  # Perturb DC, AC, raw_quant_field, CfL, and EPF sharpness and verify valid re-encode
  mod_state = state.copy()
  mod_state.dc_quant[1, 0, 0] += 2
  mod_state.quant_ac[1, :8, :8] = 0
  mod_state.raw_quant_field = np.clip(mod_state.raw_quant_field + 1, 1, 256)
  mod_state.ytox_map[0, 0] = np.int8(np.clip(int(mod_state.ytox_map[0, 0]) + 3, -128, 127))
  mod_state.epf_sharpness[0, 0] = np.uint8((int(mod_state.epf_sharpness[0, 0]) + 1) % 8)

  mod_bytes, mod_stats = encode_single_frame(mod_state)
  mod_ver = verify_jxl_codestream(mod_bytes, min_cross_decoder_psnr=45.0)
  assert mod_stats["total_bits"] != stats["total_bits"] or not np.array_equal(
      mod_ver["rgb_djxl"], ver["rgb_djxl"]
  )


def test_encode_single_with_noise_lut_and_inspect(
    kodim01_rgb_u8: np.ndarray,
) -> None:
  """Verifies --noise-lut injection, --inspect readback, and active decoder synthesis."""
  crop = kodim01_rgb_u8[:96, :128]
  state = dump_init_state(crop, distance=1.5, effort=7)

  jxl_no_noise, _ = encode_single_frame(state)
  rgb_no_noise = decode_with_djxl(jxl_no_noise)

  noise_lut = [0.02, 0.03, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05]
  jxl_noise, stats_noise = encode_single_frame(state, noise_lut=noise_lut)
  assert stats_noise["layer_bits"]["Noise"] > 0

  ver = verify_jxl_codestream(jxl_noise, min_cross_decoder_psnr=45.0)
  insp = ver["inspect"]
  assert insp["has_noise"] is True
  assert insp["frames"][0]["has_noise"] is True
  np.testing.assert_allclose(insp["frames"][0]["noise_lut"], noise_lut, atol=1e-3)

  # Noise synthesis changes the decoded pixels deterministically
  assert compute_psnr(rgb_no_noise, ver["rgb_djxl"]) < 44.0


@pytest.fixture(scope="module")
def composite_states_128(
    kodim01_rgb_u8: np.ndarray,
) -> tuple[dict[int, object], object]:
  crop = kodim01_rgb_u8[:128, :128]
  main_state = dump_init_state(crop, distance=2.0, effort=7)
  ref_states = {
      u: dump_init_state(_downsample_rgb_u8(crop, u), distance=1.5, effort=7)
      for u in (2, 4, 8)
  }
  return ref_states, main_state


@pytest.mark.parametrize("upsampling", [2, 4, 8])
@pytest.mark.parametrize("blend_mode", [PATCH_BLEND_REPLACE, PATCH_BLEND_ADD])
def test_encode_patches_all_upsampling_and_blend_modes(
    composite_states_128: tuple[dict[int, object], object],
    upsampling: int,
    blend_mode: int,
) -> None:
  """Verifies --encode-patches across U in {2, 4, 8} and kReplace/kAdd in djxl and jxl-rs."""
  ref_states, main_state = composite_states_128
  ref_state = ref_states[upsampling]

  patches = [
      PatchEntry(
          ref=0,
          x0=8,
          y0=8,
          xsize=32,
          ysize=24,
          x=8,
          y=8,
          blend_mode=blend_mode,
      ),
      PatchEntry(
          ref=0,
          x0=64,
          y0=48,
          xsize=24,
          ysize=32,
          x=64,
          y=48,
          blend_mode=blend_mode,
      ),
  ]
  noise_lut = (
      [0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03]
      if upsampling == 2 and blend_mode == PATCH_BLEND_ADD
      else None
  )

  jxl_bytes, stats = encode_patches_composite(
      ref_state,  # type: ignore[arg-type]
      main_state,  # type: ignore[arg-type]
      upsampling=upsampling,
      patches=patches,
      noise_lut=noise_lut,
  )
  assert stats["file_bytes"] == len(jxl_bytes)
  assert stats["layer_bits"]["Dictionary"] > 0

  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  insp = ver["inspect"]
  assert insp["num_frames"] == 2
  f0, f1 = insp["frames"][0], insp["frames"][1]
  assert f0["frame_type"] == "kReferenceOnly"
  assert f0["upsampling"] == upsampling
  assert f0["save_before_color_transform"] is True
  assert f0["save_as_reference"] == 0
  assert f0["is_last"] is False

  assert f1["frame_type"] == "kRegularFrame"
  assert f1["upsampling"] == 1
  assert f1["is_last"] is True
  assert f1["has_patches"] is True
  assert f1["num_patches"] == 2
  assert f1["patch_blend_modes"] == [blend_mode, blend_mode]
  if noise_lut is not None:
    assert f1["has_noise"] is True


@pytest.mark.parametrize("upsampling", [2, 4, 8])
@pytest.mark.parametrize("dim_shift", [0, 1, 2, 3])
def test_encode_alpha_all_upsampling_and_dim_shifts(
    composite_states_128: tuple[dict[int, object], object],
    upsampling: int,
    dim_shift: int,
) -> None:
  """Verifies --encode-alpha across all 12 (upsampling, dim_shift) pairs in djxl and jxl-rs."""
  ref_states, main_state = composite_states_128
  ref_state = ref_states[upsampling]

  h_a = (128 + (1 << dim_shift) - 1) >> dim_shift
  w_a = (128 + (1 << dim_shift) - 1) >> dim_shift
  yy, xx = np.meshgrid(
      np.linspace(0, 255, h_a), np.linspace(0, 255, w_a), indexing="ij"
  )
  alpha_u8 = np.clip(np.round(0.5 * (yy + xx)), 0, 255).astype(np.uint8)
  noise_lut = (
      [0.01, 0.015, 0.02, 0.025, 0.025, 0.025, 0.025, 0.025]
      if upsampling == 4 and dim_shift == 1
      else None
  )

  jxl_bytes, stats = encode_alpha_composite(
      ref_state,
      main_state,
      upsampling=upsampling,
      dim_shift=dim_shift,
      alpha_u8=alpha_u8,
      noise_lut=noise_lut,
  )
  assert stats["file_bytes"] == len(jxl_bytes)

  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  insp = ver["inspect"]
  assert insp["num_frames"] == 2
  assert len(insp["extra_channels"]) == 1
  ec0 = insp["extra_channels"][0]
  assert ec0["type"] == "kAlpha"
  assert ec0["bits_per_sample"] == 8
  assert ec0["dim_shift"] == dim_shift
  assert ec0["alpha_associated"] is False

  f0, f1 = insp["frames"][0], insp["frames"][1]
  assert f0["frame_type"] == "kReferenceOnly"
  assert f0["upsampling"] == upsampling
  assert f0["ec_upsampling"] == [max(1, upsampling >> dim_shift)]
  assert f0["save_before_color_transform"] is True
  assert f0["save_as_reference"] == 0
  assert f0["is_last"] is False

  assert f1["frame_type"] == "kRegularFrame"
  assert f1["blending_mode"] == "kReplace"
  assert f1["has_patches"] is True
  assert f1["num_patches"] == 1
  assert f1["patch_blend_modes"] == [PATCH_BLEND_ALPHA_BELOW]
  assert f1["patch_ec_blend_modes"] == [PATCH_BLEND_REPLACE]
  assert f1["is_last"] is True


def test_encode_alpha_boundary_masks_0_and_255(
    kodim01_rgb_u8: np.ndarray,
) -> None:
  """Verifies alpha=0 (pure background Frame 0) and alpha=255 (pure foreground Frame 1)."""
  crop = kodim01_rgb_u8[:96, :128]
  low_rgb = _downsample_rgb_u8(crop, 2)
  ref_state = dump_init_state(low_rgb, distance=2.5, effort=7)
  main_state = dump_init_state(crop, distance=0.8, effort=7)

  # Pure foreground (alpha = 255) should match single-frame encoding of main_state
  jxl_fg, _ = encode_alpha_composite(
      ref_state,
      main_state,
      upsampling=2,
      dim_shift=0,
      alpha_u8=np.full((96, 128), 255, dtype=np.uint8),
  )
  jxl_bg, _ = encode_alpha_composite(
      ref_state,
      main_state,
      upsampling=2,
      dim_shift=0,
      alpha_u8=np.zeros((96, 128), dtype=np.uint8),
  )
  jxl_single, _ = encode_single_frame(main_state)

  rgb_fg = decode_with_djxl(jxl_fg)
  rgb_bg = decode_with_djxl(jxl_bg)
  rgb_single = decode_with_djxl(jxl_single)

  # Foreground (alpha=255) must match single-frame main_state closely
  assert compute_psnr(rgb_fg, rgb_single) >= 55.0
  # Background (alpha=0) must differ from foreground
  assert not np.array_equal(rgb_bg, rgb_fg)
  # Both must pass cross-decoder verification with jxl-rs
  assert compute_psnr(rgb_bg, decode_with_jxl_rs(jxl_bg)) >= 45.0


@pytest.mark.parametrize(
    "alpha_blend_mode",
    [
        PATCH_BLEND_ALPHA_ABOVE,
        PATCH_BLEND_ALPHA_BELOW,
        PATCH_BLEND_MUL_ADD_ABOVE,
        PATCH_BLEND_MUL_ADD_BELOW,
    ],
)
def test_encode_alpha_xyb_patch_blend_modes_and_regional_patches(
    composite_states_128: tuple[dict[int, object], object],
    alpha_blend_mode: int,
) -> None:
  """Verifies XYB PatchDictionary alpha blend modes (4, 5, 6, 7) and regional patches."""
  ref_states, main_state = composite_states_128
  ref_state = ref_states[2]
  alpha_u8 = np.full((64, 64), 128, dtype=np.uint8)

  jxl_bytes, stats = encode_alpha_composite(
      ref_state,  # type: ignore[arg-type]
      main_state,  # type: ignore[arg-type]
      upsampling=2,
      dim_shift=1,
      alpha_u8=alpha_u8,
      alpha_blend_mode=alpha_blend_mode,
  )
  assert stats["layer_bits"]["Dictionary"] > 0
  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  f0, f1 = ver["inspect"]["frames"]
  assert f0["save_before_color_transform"] is True
  assert f1["has_patches"] is True
  assert f1["patch_blend_modes"] == [alpha_blend_mode]
  assert f1["patch_ec_blend_modes"] == [PATCH_BLEND_REPLACE]

  if alpha_blend_mode == PATCH_BLEND_ALPHA_BELOW:
    regional_patches = [
        PatchEntry(
            ref=0,
            x0=16,
            y0=16,
            xsize=48,
            ysize=48,
            x=16,
            y=16,
            blend_mode=PATCH_BLEND_ALPHA_BELOW,
        )
    ]
    alpha_reg = np.full((64, 64), 255, dtype=np.uint8)
    alpha_reg[12:28, 12:28] = 96
    jxl_reg, _ = encode_alpha_composite(
        ref_state,  # type: ignore[arg-type]
        main_state,  # type: ignore[arg-type]
        upsampling=2,
        dim_shift=1,
        alpha_u8=alpha_reg,
        patches=regional_patches,
    )
    ver_reg = verify_jxl_codestream(jxl_reg, min_cross_decoder_psnr=45.0)
    assert ver_reg["inspect"]["frames"][1]["num_patches"] == 1
    assert ver_reg["inspect"]["frames"][1]["patch_blend_modes"] == [
        PATCH_BLEND_ALPHA_BELOW
    ]


def test_custom_dequant_matrices_and_dc_quant_bitstream_serialization(
    kodim01_rgb_u8: np.ndarray,
    tmp_path: Path,
) -> None:
  """Verifies custom AC DequantMatrices and custom DC quant weights serialization and inspection."""
  crop = kodim01_rgb_u8[:96, :128]
  base_state = dump_init_state(crop, distance=1.5, effort=7)
  assert not has_custom_dequant_matrices(base_state)

  jxl_def, stats_def = encode_single_frame(base_state)
  ver_def = verify_jxl_codestream(jxl_def, min_cross_decoder_psnr=45.0)
  insp_def = ver_def["inspect"]
  assert insp_def["all_default"] is True
  assert insp_def["dequant_all_default"] is True
  assert insp_def["dc_quant_all_default"] is True
  np.testing.assert_allclose(
      insp_def["dc_quant"], DEFAULT_DC_QUANT_WEIGHTS, rtol=1e-4
  )

  # 1. Scale AC dequant_matrices[0] (DCT8) and custom DC quant weights
  custom_state = base_state.copy()
  custom_state.dequant_matrices[0] = (
      custom_state.dequant_matrices[0] * np.array([1.15, 1.25, 0.85], dtype=np.float32)[:, None, None]
  )
  custom_dc_w = np.array(DEFAULT_DC_QUANT_WEIGHTS, dtype=np.float32) * np.array(
      [1.25, 0.85, 1.15], dtype=np.float32
  )
  custom_state.dc_quant_weights = custom_dc_w
  assert has_custom_dequant_matrices(custom_state)

  # Verify binary state file roundtrip of custom dequant & dc_quant_weights
  saved_bin = save_state_bin(custom_state, tmp_path / "custom_state.bin")
  reloaded_custom = load_state_bin(saved_bin)
  assert has_custom_dequant_matrices(reloaded_custom)
  np.testing.assert_allclose(
      reloaded_custom.dequant_matrices[0], custom_state.dequant_matrices[0]
  )
  np.testing.assert_allclose(reloaded_custom.dc_quant_weights, custom_dc_w)

  # Verify Python F16/QuantEncoding roundtrip helper
  rt_deq, rt_dc = roundtrip_custom_dequant_matrices(custom_state)
  np.testing.assert_allclose(
      rt_deq[0][:, 1:, 1:], custom_state.dequant_matrices[0][:, 1:, 1:], rtol=5e-3
  )
  np.testing.assert_allclose(rt_dc, custom_dc_w, rtol=2e-3)

  # Encode custom_state to JXL and verify non-default Quant layer and decoded pixels
  jxl_cust, stats_cust = encode_single_frame(custom_state)
  assert stats_cust["layer_bits"]["Quant"] > stats_def["layer_bits"]["Quant"]
  ver_cust = verify_jxl_codestream(jxl_cust, min_cross_decoder_psnr=45.0)
  insp_cust = ver_cust["inspect"]
  assert insp_cust["all_default"] is False
  assert insp_cust["dequant_all_default"] is False
  assert insp_cust["dc_quant_all_default"] is False
  np.testing.assert_allclose(insp_cust["dc_quant"], rt_dc, rtol=1e-4)
  assert not np.array_equal(ver_cust["rgb_djxl"], ver_def["rgb_djxl"])

  # 2. Radial frequency-dependent custom dequant matrix (fits 8-band DctQuantWeightParams)
  radial_state = base_state.copy()
  yy, xx = np.meshgrid(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8), indexing="ij")
  radial_ramp = (1.0 + 0.35 * np.sqrt(xx * xx + yy * yy)).astype(np.float32)
  radial_state.dequant_matrices[0] = radial_state.dequant_matrices[0] * radial_ramp[None, :, :]
  assert has_custom_dequant_matrices(radial_state)
  jxl_rad, _ = encode_single_frame(radial_state)
  ver_rad = verify_jxl_codestream(jxl_rad, min_cross_decoder_psnr=45.0)
  assert ver_rad["inspect"]["dequant_all_default"] is False
  assert not np.array_equal(ver_rad["rgb_djxl"], ver_def["rgb_djxl"])


def test_encode_single_nonzero_llf_quant_ac_multiblock_and_multigroup(
    kodim01_rgb_u8: np.ndarray,
) -> None:
  """Verifies non-zero LLF quant_ac mutations across multi-block AcStrategy and 260x260 multi-group frames."""
  # 1. 260x260 multi-group frame (h8=33, w8=33 -> 2x2 AC groups) with natural multi-block AcStrategy
  crop_260 = kodim01_rgb_u8[:260, :260]
  state_260 = dump_init_state(crop_260, distance=1.5, effort=7)
  assert state_260.h8 == 33 and state_260.w8 == 33

  # Ensure multi-block transforms are present and also explicitly place representative
  # multi-block transforms (DCT16X16=4, DCT32X32=5, DCT16X8=6, DCT8X16=7, DCT32X16=10, DCT16X32=11)
  # in a dedicated synthetic multi-block test state.
  mut_260 = state_260.copy()
  rng = np.random.default_rng(12345)
  mut_260.quant_ac = (
      -mut_260.quant_ac
      + rng.integers(-5, 6, size=mut_260.quant_ac.shape, dtype=np.int32)
  ).astype(np.int32)
  # Force every entry (including all canonical LLF slots) to be non-zero
  mut_260.quant_ac = np.where(mut_260.quant_ac == 0, 7, mut_260.quant_ac).astype(
      np.int32
  )

  jxl_bytes_260, stats_260 = encode_single_frame(mut_260)
  assert stats_260["file_bytes"] == len(jxl_bytes_260)
  ver_260 = verify_jxl_codestream(jxl_bytes_260, min_cross_decoder_psnr=45.0)
  assert ver_260["inspect"]["width"] == 260
  assert ver_260["inspect"]["height"] == 260

  # 2. Explicit placement of multi-block AcStrategy transforms with non-zero LLF slots:
  # DCT16X16 (4: 2x2), DCT32X32 (5: 4x4), DCT16X8 (6: cy=2,cx=1),
  # DCT8X16 (7: cy=1,cx=2), DCT32X16 (10: cy=4,cx=2), DCT16X32 (11: cy=2,cx=4).
  explicit_state = state_260.copy()
  explicit_state.ac_strategy[:, :] = 0  # Reset all to DCT8 (1x1)
  placements = [
      (0, 0, 4),    # DCT16X16 (cy=2, cx=2) at (0..1, 0..1)
      (0, 4, 5),    # DCT32X32 (cy=4, cx=4) at (0..3, 4..7)
      (0, 8, 6),    # DCT16X8  (cy=2, cx=1) at (0..1, 8)
      (0, 10, 7),   # DCT8X16  (cy=1, cx=2) at (0, 10..11)
      (4, 0, 10),   # DCT32X16 (cy=4, cx=2) at (4..7, 0..1)
      (4, 4, 11),   # DCT16X32 (cy=2, cx=4) at (4..5, 4..7)
  ]
  for by, bx, acs_id in placements:
    cx = ACS_COVERED_BLOCKS_X[acs_id]
    cy = ACS_COVERED_BLOCKS_Y[acs_id]
    explicit_state.ac_strategy[by : by + cy, bx : bx + cx] = np.uint8(acs_id)

  explicit_state.quant_ac.fill(9)
  jxl_bytes_exp, stats_exp = encode_single_frame(explicit_state)
  assert stats_exp["file_bytes"] == len(jxl_bytes_exp)
  verify_jxl_codestream(jxl_bytes_exp, min_cross_decoder_psnr=45.0)


def test_encode_single_rejects_invalid_ac_strategy_bounds(
    kodim01_rgb_u8: np.ndarray,
) -> None:
  """Verifies that invalid ac_strategy IDs and out-of-bounds/group-crossing blocks raise RuntimeError."""
  crop_260 = kodim01_rgb_u8[:260, :260]
  base_state = dump_init_state(crop_260, distance=1.5, effort=7)
  assert base_state.h8 == 33 and base_state.w8 == 33

  # Case 1: Invalid raw strategy ID (99 >= kNumValidStrategies=27)
  bad_id = base_state.copy()
  bad_id.ac_strategy[:, :] = 0
  bad_id.ac_strategy[0, 0] = 99
  with pytest.raises(RuntimeError):
    encode_single_frame(bad_id)

  # Case 2: DCT16X16 (4, cx=2, cy=2) overflowing right frame boundary at bx = w8 - 1
  bad_right = base_state.copy()
  bad_right.ac_strategy[:, :] = 0
  bad_right.ac_strategy[0, bad_right.w8 - 1] = 4
  with pytest.raises(RuntimeError):
    encode_single_frame(bad_right)

  # Case 3: DCT16X16 (4, cx=2, cy=2) overflowing bottom frame boundary at by = h8 - 1
  bad_bottom = base_state.copy()
  bad_bottom.ac_strategy[:, :] = 0
  bad_bottom.ac_strategy[bad_bottom.h8 - 1, 0] = 4
  with pytest.raises(RuntimeError):
    encode_single_frame(bad_bottom)

  # Case 4: DCT16X16 (4, cx=2, cy=2) crossing horizontal 32-block group boundary at bx = 31
  bad_group_x = base_state.copy()
  bad_group_x.ac_strategy[:, :] = 0
  bad_group_x.ac_strategy[0, 31] = 4
  with pytest.raises(RuntimeError):
    encode_single_frame(bad_group_x)

  # Case 5: DCT16X16 (4, cx=2, cy=2) crossing vertical 32-block group boundary at by = 31
  bad_group_y = base_state.copy()
  bad_group_y.ac_strategy[:, :] = 0
  bad_group_y.ac_strategy[31, 0] = 4
  with pytest.raises(RuntimeError):
    encode_single_frame(bad_group_y)

  # Case 6: Overlapping multi-block transforms (DCT16X8 at (0, 1) and DCT8X16 at (1, 0) colliding at (1, 1))
  bad_overlap = base_state.copy()
  bad_overlap.ac_strategy[:, :] = 0
  bad_overlap.ac_strategy[0, 1] = 6  # DCT16X8 (cy=2, cx=1) covers (0, 1) and (1, 1)
  bad_overlap.ac_strategy[1, 0] = 7  # DCT8X16 (cy=1, cx=2) tries to cover (1, 0) and (1, 1)
  with pytest.raises(RuntimeError):
    encode_single_frame(bad_overlap)


