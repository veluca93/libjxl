"""Tier 1 E2E Test Suite: Category-Partition Feature Coverage (F1-F13, 5 test cases per feature, 65 total)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest
from scipy.stats import spearmanr

from jxl_wasserstein.benchmark import (
    DEFAULT_BENCHMARK_DISTANCES,
    compute_bd_rate,
    encode_cjxl_baseline,
    evaluate_image_rd_curve,
    run_kodak_benchmark,
)
from jxl_wasserstein.cli import main as cli_main
from jxl_wasserstein.cpp_bridge import (
    CJXL_BIN,
    DJXL_BIN,
    HELPER_BIN,
    JXL_CLI_BIN,
    JXLINFO_BIN,
    PATCH_BLEND_ADD,
    PATCH_BLEND_REPLACE,
    PatchEntry,
    compute_psnr,
    decode_with_djxl,
    dump_init_state,
    encode_alpha_composite,
    encode_patches_composite,
    encode_single_frame,
    inspect_jxl,
    load_state_bin,
    save_state_bin,
    verify_jxl_codestream,
)
import jxl_wasserstein.cpp_bridge as cpp_bridge_mod
from jxl_wasserstein.entropy_proxy import (
    RateBreakdown,
    soft_alpha_composite_rate_proxy,
    soft_jxl_rate_proxy,
    soft_patch_composite_rate_proxy,
)
import jxl_wasserstein.entropy_proxy as ep_mod
from jxl_wasserstein.forward_model import (
    StaticVarDCTTopology,
    get_default_dequant_matrices,
    get_default_inv_dequant_matrices,
    init_state_from_rgb_jax,
    reconstruct_alpha_composite,
    reconstruct_patch_composite,
    reconstruct_single_frame,
    reconstruct_single_frame_xyb,
)
import jxl_wasserstein.forward_model as fm_mod
from jxl_wasserstein.wasserstein import compute_wd
import jxl_wasserstein.wasserstein as wd_mod

from conftest import (
    FORBIDDEN_LEGACY_TOKENS,
    PKG_ROOT,
    REPO_ROOT,
    forbid_libjxl_per_block_choices,
    get_field,
)


def _call_encode_alpha(
    ref_state: Any,
    main_state: Any,
    upsampling: int,
    dim_shift: int,
    alpha_u8: np.ndarray,
    alpha_blend_mode: int = 5,
    patches: list[PatchEntry] | None = None,
    noise_lut: list[float] | None = None,
) -> tuple[bytes, dict[str, Any]]:
  """Calls encode_alpha_composite handling optional alpha_blend_mode / patches kwargs."""
  sig = inspect.signature(encode_alpha_composite)
  kwargs: dict[str, Any] = {
      "ref_state": ref_state,
      "main_state": main_state,
      "upsampling": upsampling,
      "dim_shift": dim_shift,
      "alpha_u8": alpha_u8,
  }
  if "alpha_blend_mode" in sig.parameters:
    kwargs["alpha_blend_mode"] = alpha_blend_mode
  elif "blend_mode" in sig.parameters:
    kwargs["blend_mode"] = alpha_blend_mode
  if patches is not None and "patches" in sig.parameters:
    kwargs["patches"] = patches
  if noise_lut is not None and "noise_lut" in sig.parameters:
    kwargs["noise_lut"] = noise_lut
  return encode_alpha_composite(**kwargs)


def _call_reconstruct_alpha(
    ref_state: Any,
    main_state: Any,
    alpha_01: jax.Array,
    upsampling: int = 2,
    dim_shift: int = 0,
    alpha_blend_mode: int = 5,
    include_noise: bool = False,
    noise_lut: Any = None,
    hard: bool = True,
) -> jax.Array:
  """Calls reconstruct_alpha_composite across positional/keyword conventions."""
  sig = inspect.signature(reconstruct_alpha_composite)
  kwargs: dict[str, Any] = {
      "ref_state": ref_state,
      "main_state": main_state,
      "alpha_01": alpha_01,
      "upsampling": upsampling,
      "hard": hard,
      "include_noise": include_noise,
  }
  if "dim_shift" in sig.parameters:
    kwargs["dim_shift"] = dim_shift
  if "alpha_dim_shift" in sig.parameters:
    kwargs["alpha_dim_shift"] = dim_shift
  if "alpha_blend_mode" in sig.parameters:
    kwargs["alpha_blend_mode"] = alpha_blend_mode
  elif "blend_mode" in sig.parameters:
    kwargs["blend_mode"] = alpha_blend_mode
  if noise_lut is not None and "noise_lut" in sig.parameters:
    kwargs["noise_lut"] = noise_lut
  return reconstruct_alpha_composite(**kwargs)


def _set_custom_dc_quant(state: Any, dc_quant_3: tuple[float, float, float]) -> None:
  """Attaches custom DC quantization step sizes to a JxlVarDCTState."""
  arr = np.asarray(dc_quant_3, dtype=np.float32)
  for attr in ("custom_dc_quant", "dc_quant_weights", "dc_dequant", "dc_quant_step"):
    setattr(state, attr, arr)


def _make_custom_ac_state(base_state: Any, scale_factor: float = 1.25) -> Any:
  """Creates a copy of base_state with scaled custom AC dequant_matrices."""
  st = base_state.copy()
  dq_default = get_default_dequant_matrices()
  st.dequant_matrices = {
      int(s): (np.asarray(mat, dtype=np.float32) * float(scale_factor)).astype(
          np.float32
      )
      for s, mat in dq_default.items()
  }
  if hasattr(st, "has_custom_dequant"):
    st.has_custom_dequant = True
  return st


def _extract_ac_strategy(res: Any) -> np.ndarray:
  """Extracts the discrete AcStrategy array from an EncodeResult / OptimizationResult."""
  acs = get_field(res, "stage2_ac_strategy")
  if acs is not None:
    return np.asarray(acs)
  st = get_field(res, "state", "main_state")
  assert st is not None, f"Could not extract state from result {type(res)}"
  return np.asarray(st.ac_strategy)


def _extract_stage2_and_stage3_j(res: Any) -> tuple[float, float]:
  """Extracts (stage2_true_j, stage3_true_j) from an EncodeResult / OptimizationResult."""
  s2_j = get_field(res, "stage2_true_j")
  s3_j = get_field(res, "stage3_true_j")
  if s2_j is not None and s3_j is not None:
    return float(s2_j), float(s3_j)
  hist = get_field(res, "history")
  assert hist and len(hist) >= 1, "Expected non-empty optimization history"
  j0 = float(hist[0].get("lagrangian_true", hist[0].get("rd_loss")))
  j_best = float(get_field(res, "lagrangian", "rd_cost"))
  return j0, j_best


# =============================================================================
# F1: Repository & Git Hygiene (R1) — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "check_name",
    [
        "package_and_binary_layout",
        "zero_legacy_sys_modules",
        "zero_legacy_source_references",
        "gitignore_and_exclude_hygiene",
        "zero_tracked_agents_archive",
    ],
)
def test_f1_repository_and_git_hygiene(check_name: str) -> None:
  """Verifies F1 (R1): standalone package layout, zero legacy package refs, and clean git ignore hygiene."""
  if check_name == "package_and_binary_layout":
    assert (PKG_ROOT / "pyproject.toml").is_file()
    assert (PKG_ROOT / ".venv" / "bin" / "python").exists()
    assert (PKG_ROOT / "src" / "jxl_wasserstein" / "__init__.py").is_file()
    for bin_path in (HELPER_BIN, DJXL_BIN, JXLINFO_BIN, JXL_CLI_BIN, CJXL_BIN):
      assert bin_path.exists(), f"Missing required binary: {bin_path}"
  elif check_name == "zero_legacy_sys_modules":
    for mod_name in list(sys.modules):
      for tok in FORBIDDEN_LEGACY_TOKENS:
        assert tok not in mod_name, f"Forbidden legacy module loaded: {mod_name}"
  elif check_name == "zero_legacy_source_references":
    scan_paths = [
        PKG_ROOT / "pyproject.toml",
        REPO_ROOT / "tools" / "jxl_wasserstein_helper.cc",
    ]
    for sub in ("src", "tests"):
      for p in (PKG_ROOT / sub).rglob("*"):
        if (
            p.is_file()
            and "__pycache__" not in p.parts
            and ".venv" not in p.parts
        ):
          scan_paths.append(p)
    for path in scan_paths:
      text = path.read_text(encoding="utf-8", errors="ignore")
      for tok in FORBIDDEN_LEGACY_TOKENS:
        assert tok not in text, f"Found forbidden token {tok!r} in {path}"
  elif check_name == "gitignore_and_exclude_hygiene":
    gitignore_txt = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    exclude_path = REPO_ROOT / ".git" / "info" / "exclude"
    exclude_txt = (
        exclude_path.read_text(encoding="utf-8") if exclude_path.is_file() else ""
    )
    combined = gitignore_txt + "\n" + exclude_txt
    for pattern in (".agents", "ORIGINAL_REQUEST.md", "PROJECT.md", "TEST_INFRA.md", "TEST_READY.md"):
      assert pattern in combined, f"Expected {pattern!r} in .gitignore or .git/info/exclude"
    res = subprocess.run(
        ["git", "check-ignore", ".agents/teamwork/ORIGINAL_REQUEST.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, ".agents/ must be ignored by git"
  elif check_name == "zero_tracked_agents_archive":
    res = subprocess.run(
        ["git", "ls-files", ".agents/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "", (
        f"Expected zero git-tracked files under .agents/, got: {res.stdout}"
    )


# =============================================================================
# F2: Pure-Python Library DequantMatrices Initialization (R1) — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "check_name",
    [
        "no_dump_init_invocation",
        "all_18_strategy_slots_present",
        "dct8_and_identity_parity_vs_libjxl",
        "multiblock_dct16_dct32_dct8x16_parity",
        "inv_dequant_matrices_consistency",
    ],
)
def test_f2_pure_python_default_dequant_matrices(
    check_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verifies F2 (R1): get_default_dequant_matrices() computes libjxl defaults in pure Python without dump_init_state."""
  if check_name == "no_dump_init_invocation":
    # Reset any module-level cache on forward_model so cold initialization is tested
    for cache_attr in ("_DEFAULT_DEQUANT_CACHE", "_DEFAULT_INV_DEQUANT_CACHE"):
      if hasattr(fm_mod, cache_attr):
        monkeypatch.setattr(fm_mod, cache_attr, None)
    if hasattr(get_default_dequant_matrices, "cache_clear"):
      get_default_dequant_matrices.cache_clear()  # type: ignore[attr-defined]
    with forbid_libjxl_per_block_choices(monkeypatch):
      dq = get_default_dequant_matrices()
      assert len(dq) >= 18
  elif check_name == "all_18_strategy_slots_present":
    dq = get_default_dequant_matrices()
    for sid in range(18):
      assert sid in dq, f"Missing strategy {sid} in get_default_dequant_matrices()"
      mat = dq[sid]
      cy = cpp_bridge_mod.ACS_COVERED_BLOCKS_Y[sid]
      cx = cpp_bridge_mod.ACS_COVERED_BLOCKS_X[sid]
      assert mat.shape == (3, 8 * min(cy, cx), 8 * max(cy, cx))
      assert bool(np.all(np.isfinite(mat))) and bool(np.all(mat > 0.0))
  elif check_name == "dct8_and_identity_parity_vs_libjxl":
    dq_py = get_default_dequant_matrices()
    ref_state = dump_init_state(np.zeros((8, 8, 3), dtype=np.float32), distance=1.0, effort=3)
    for sid in (0, 1, 2, 3):
      assert sid in ref_state.dequant_matrices
      np.testing.assert_allclose(
          dq_py[sid], ref_state.dequant_matrices[sid], rtol=1e-5, atol=1e-6
      )
  elif check_name == "multiblock_dct16_dct32_dct8x16_parity":
    dq_py = get_default_dequant_matrices()
    ref_state = dump_init_state(np.zeros((8, 8, 3), dtype=np.float32), distance=1.0, effort=3)
    for sid in (4, 5, 6, 7, 14, 15, 16, 17):
      assert sid in ref_state.dequant_matrices
      np.testing.assert_allclose(
          dq_py[sid], ref_state.dequant_matrices[sid], rtol=1e-5, atol=1e-6
      )
  elif check_name == "inv_dequant_matrices_consistency":
    dq = get_default_dequant_matrices()
    inv_dq = get_default_inv_dequant_matrices()
    for sid in range(18):
      prod = dq[sid] * inv_dq[sid]
      # Non-LLF AC slots should satisfy dq * inv_dq == 1.0
      assert np.allclose(prod[:, 2:, 2:], 1.0, rtol=1e-5, atol=1e-5)


# =============================================================================
# F3: Removal of Static Pre-Opt Variance/MSE Heuristics & ac_strategy=0 Hardcoding — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "check_name",
    [
        "no_dump_init_or_cjxl_in_single_frame_opt",
        "no_dump_init_or_cjxl_in_multires_opt",
        "ac_strategy_not_hardcoded_zero_in_wd_init",
        "ac_strategy_driven_by_stage1_rate_wd_not_static_variance",
        "patch_and_alpha_driven_by_stage1_rate_wd_not_static_mse",
    ],
)
def test_f3_no_static_heuristics_or_hardcoded_acs0(
    check_name: str,
    mixed_image_64_01: np.ndarray,
    composite_smooth_sharp_64_01: np.ndarray,
    vgg_params_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verifies F3 (R1, R3, R5): zero dump_init/cjxl calls, zero static variance/MSE heuristics, and no ac_strategy=0 hardcoding."""
  import jxl_wasserstein.compositing as comp_mod
  import jxl_wasserstein.optimizer as opt_mod

  if check_name == "no_dump_init_or_cjxl_in_single_frame_opt":
    with forbid_libjxl_per_block_choices(monkeypatch):
      res = opt_mod.optimize_single_frame(
          mixed_image_64_01[:32, :32],
          vgg_params_session,
          rd_lambda=0.15,
          num_steps=2,
      )
      assert len(get_field(res, "jxl_bytes")) > 0
  elif check_name == "no_dump_init_or_cjxl_in_multires_opt":
    with forbid_libjxl_per_block_choices(monkeypatch):
      for mode in ("patches", "alpha", "auto"):
        res = comp_mod.encode_image_to_jxl(
            composite_smooth_sharp_64_01[:32, :32],
            mode=mode,
            upsampling=2,
            dim_shift=1,
            distance=0.8,
            lambda_rd=0.12,
            num_steps=1,
            verify_decoders=False,
            vgg_params=vgg_params_session,
        )
        assert len(get_field(res, "jxl_bytes")) > 0
  elif check_name == "ac_strategy_not_hardcoded_zero_in_wd_init":
    res = opt_mod.optimize_rate_wd_point(
        mixed_image_64_01,
        distance=1.5,
        steps=2,
        vgg_params=vgg_params_session,
    )
    acs = _extract_ac_strategy(res)
    assert bool(np.any(acs != 0)), "Expected non-zero AcStrategy transforms on mixed image"
    assert len(np.unique(acs)) >= 2, f"Expected >= 2 distinct AcStrategy types, got {np.unique(acs)}"
  elif check_name == "ac_strategy_driven_by_stage1_rate_wd_not_static_variance":
    if hasattr(opt_mod, "_compute_block_activity_metrics_jax"):
      monkeypatch.setattr(
          opt_mod,
          "_compute_block_activity_metrics_jax",
          lambda *a, **kw: (_ for _ in ()).throw(
              AssertionError("Static pixel variance heuristic must not be called")
          ),
      )
    res = opt_mod.optimize_single_frame(
        mixed_image_64_01,
        vgg_params_session,
        rd_lambda=0.15,
        num_steps=2,
    )
    s1 = get_field(res, "stage1_candidates")
    assert s1 is not None and len(s1) >= 2
    acs = _extract_ac_strategy(res)
    assert len(np.unique(acs)) >= 2
  elif check_name == "patch_and_alpha_driven_by_stage1_rate_wd_not_static_mse":
    if hasattr(comp_mod, "compute_detail_mask_jax"):
      monkeypatch.setattr(
          comp_mod,
          "compute_detail_mask_jax",
          lambda *a, **kw: (_ for _ in ()).throw(
              AssertionError("Static pixel MSE heuristic must not be called")
          ),
      )
    res_p = comp_mod.optimize_patch_composite(
        composite_smooth_sharp_64_01[:32, :32],
        upsampling=2,
        blend_mode=PATCH_BLEND_REPLACE,
        distance=0.5,
        lambda_rd=0.05,
        num_steps=1,
        verify_decoders=False,
        vgg_params=vgg_params_session,
    )
    assert len(get_field(res_p, "patches") or []) >= 1
    res_a = comp_mod.optimize_alpha_composite(
        composite_smooth_sharp_64_01[:32, :32],
        upsampling=2,
        dim_shift=0,
        distance=0.5,
        lambda_rd=0.05,
        num_steps=1,
        verify_decoders=False,
        vgg_params=vgg_params_session,
    )
    assert len(get_field(res_a, "jxl_bytes")) > 0


# =============================================================================
# F4: C++ XYB PatchDictionary Extra-Alpha-Channel Encoding (R2) — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "case_name,upsampling,dim_shift,alpha_blend_mode",
    [
        ("alpha_u2_d0_blend_below_default", 2, 0, 5),
        ("alpha_u4_d1_xyb_patches", 4, 1, 5),
        ("alpha_u8_d2_xyb_patches", 8, 2, 5),
        ("alpha_weighted_add_below_mode_7", 2, 0, 7),
        ("opaque_output_alpha_restoration", 4, 2, 5),
    ],
)
def test_f4_cpp_xyb_patch_dictionary_alpha_encoding(
    case_name: str,
    upsampling: int,
    dim_shift: int,
    alpha_blend_mode: int,
    kodim01_crop_64_01: np.ndarray,
) -> None:
  """Verifies F4 (R2): C++ --encode-alpha blends Frame 0 and Frame 1 in XYB via PatchDictionary."""
  h, w, _ = kodim01_crop_64_01.shape
  ref_crop = kodim01_crop_64_01[::upsampling, ::upsampling]
  ref_state = init_state_from_rgb_jax(ref_crop, distance=1.5)
  main_state = init_state_from_rgb_jax(kodim01_crop_64_01, distance=1.2)
  step_a = 1 << dim_shift
  alpha_u8 = np.zeros((h // step_a, w // step_a), dtype=np.uint8)
  alpha_u8[
      max(1, (h // step_a) // 4) : 3 * (h // step_a) // 4,
      max(1, (w // step_a) // 4) : 3 * (w // step_a) // 4,
  ] = 255

  jxl_bytes, stats = _call_encode_alpha(
      ref_state,
      main_state,
      upsampling=upsampling,
      dim_shift=dim_shift,
      alpha_u8=alpha_u8,
      alpha_blend_mode=alpha_blend_mode,
  )
  assert stats["file_bytes"] == len(jxl_bytes)
  ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
  assert ver["cross_decoder_psnr"] >= 45.0

  info = ver["inspect"]
  assert info["num_frames"] == 2
  assert info["frames"][0]["upsampling"] == upsampling
  assert info["frames"][0]["save_before_color_transform"] is True
  assert info["frames"][1]["has_patches"] is True
  assert info["frames"][1]["num_patches"] >= 1
  assert alpha_blend_mode in info["frames"][1]["patch_blend_modes"]
  assert info["extra_channels"][0]["dim_shift"] == dim_shift

  if case_name == "opaque_output_alpha_restoration":
    rgb_dec = decode_with_djxl(jxl_bytes, check_opaque_alpha=True)
    assert rgb_dec.shape == (h, w, 3)


# =============================================================================
# F5: JAX XYB PatchDictionary Alpha Forward Model & Render Pipeline Ordering (R2) — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "case_name,upsampling,dim_shift,alpha_blend_mode,include_noise",
    [
        ("xyb_blend_below_u2_d0_vs_djxl", 2, 0, 5, False),
        ("xyb_blend_below_u4_d1_vs_djxl", 4, 1, 5, False),
        ("xyb_blend_below_u8_d2_vs_djxl", 8, 2, 5, False),
        ("xyb_alpha_weighted_add_below_mode7_vs_djxl", 2, 0, 7, False),
        ("xyb_noise_after_patches_and_alpha_gradients", 2, 1, 5, True),
    ],
)
def test_f5_jax_xyb_patch_alpha_forward_model(
    case_name: str,
    upsampling: int,
    dim_shift: int,
    alpha_blend_mode: int,
    include_noise: bool,
    composite_smooth_sharp_64_01: np.ndarray,
) -> None:
  """Verifies F5 (R2): JAX XYB PatchDictionary alpha forward model matches djxl within PSNR >= 40 dB."""
  h, w, _ = composite_smooth_sharp_64_01.shape
  ref_rgb = composite_smooth_sharp_64_01[::upsampling, ::upsampling]
  ref_state = init_state_from_rgb_jax(ref_rgb, distance=1.2, epf_iters=0, gab=True)
  main_state = init_state_from_rgb_jax(
      composite_smooth_sharp_64_01, distance=1.0, epf_iters=0, gab=True
  )
  step_a = 1 << dim_shift
  alpha_u8 = np.zeros((h // step_a, w // step_a), dtype=np.uint8)
  alpha_u8[(h // step_a) // 4 : 3 * (h // step_a) // 4, (w // step_a) // 4 : 3 * (w // step_a) // 4] = 200
  alpha_01 = jnp.asarray(alpha_u8, dtype=jnp.float32) / 255.0

  if not include_noise:
    jxl_bytes, _ = _call_encode_alpha(
        ref_state,
        main_state,
        upsampling=upsampling,
        dim_shift=dim_shift,
        alpha_u8=alpha_u8,
        alpha_blend_mode=alpha_blend_mode,
    )
    djxl_rgb = decode_with_djxl(jxl_bytes).astype(np.float32) / 255.0
    jax_rgb = np.asarray(
        _call_reconstruct_alpha(
            ref_state,
            main_state,
            alpha_01=alpha_01,
            upsampling=upsampling,
            dim_shift=dim_shift,
            alpha_blend_mode=alpha_blend_mode,
            hard=True,
        )
    )
    psnr = compute_psnr(jax_rgb, djxl_rgb, peak=1.0)
    assert psnr >= 40.0, f"{case_name}: expected PSNR >= 40 dB vs djxl, got {psnr:.2f} dB"
  else:
    lut = [0.015] * 8
    jax_noisy = np.asarray(
        _call_reconstruct_alpha(
            ref_state,
            main_state,
            alpha_01=alpha_01,
            upsampling=upsampling,
            dim_shift=dim_shift,
            alpha_blend_mode=alpha_blend_mode,
            include_noise=True,
            noise_lut=lut,
            hard=True,
        )
    )
    jax_clean = np.asarray(
        _call_reconstruct_alpha(
            ref_state,
            main_state,
            alpha_01=alpha_01,
            upsampling=upsampling,
            dim_shift=dim_shift,
            alpha_blend_mode=alpha_blend_mode,
            include_noise=False,
            hard=True,
        )
    )
    assert float(np.mean(np.abs(jax_noisy - jax_clean))) > 1e-4
    grad_a = np.asarray(
        jax.grad(
            lambda a: jnp.mean(
                jnp.square(
                    _call_reconstruct_alpha(
                        ref_state,
                        main_state,
                        alpha_01=a,
                        upsampling=upsampling,
                        dim_shift=dim_shift,
                        alpha_blend_mode=alpha_blend_mode,
                        hard=False,
                    )
                )
            )
        )(alpha_01)
    )
    assert bool(np.all(np.isfinite(grad_a)))
    assert float(np.linalg.norm(grad_a)) > 0.0


# =============================================================================
# F6: C++ Custom DequantMatrices Bitstream Serialization & --inspect Support (R4) — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "check_name",
    [
        "default_matrices_inspect_all_default_true",
        "custom_dc_quant_serialization_and_inspect",
        "custom_ac_dct8_dequant_serialization_and_inspect",
        "custom_multiblock_dequant_serialization",
        "state_bin_custom_dequant_roundtrip",
    ],
)
def test_f6_cpp_custom_dequant_matrices_serialization(
    check_name: str,
    kodim01_crop_64_01: np.ndarray,
    tmp_path: Path,
) -> None:
  """Verifies F6 (R4): C++ serialization of custom DC and AC DequantMatrices and --inspect reporting."""
  crop32 = kodim01_crop_64_01[:32, :32]
  base_state = init_state_from_rgb_jax(crop32, distance=1.5)

  if check_name == "default_matrices_inspect_all_default_true":
    def_state = base_state.copy()
    def_state.dequant_matrices = {}
    jxl_bytes, _ = encode_single_frame(def_state)
    info = inspect_jxl(jxl_bytes)
    assert info["all_default"] is True
    assert info["dequant_all_default"] is True
    assert info["dc_quant_all_default"] is True
  elif check_name == "custom_dc_quant_serialization_and_inspect":
    st = base_state.copy()
    _set_custom_dc_quant(st, (1.0 / 3200.0, 1.0 / 400.0, 1.0 / 200.0))
    jxl_bytes, _ = encode_single_frame(st)
    ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
    info = ver["inspect"]
    assert info["dc_quant_all_default"] is False
    assert info["all_default"] is False
  elif check_name == "custom_ac_dct8_dequant_serialization_and_inspect":
    st = _make_custom_ac_state(base_state, scale_factor=1.30)
    jxl_bytes, _ = encode_single_frame(st)
    ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
    info = ver["inspect"]
    assert info["dequant_all_default"] is False
    assert info["all_default"] is False
  elif check_name == "custom_multiblock_dequant_serialization":
    acs = np.zeros((4, 4), dtype=np.uint8)
    acs[0:2, 0:2] = 4  # DCT16X16
    acs[2, 2] = 3  # DCT4X4
    st_mb = init_state_from_rgb_jax(crop32, distance=1.5, ac_strategy=acs)
    st_mb = _make_custom_ac_state(st_mb, scale_factor=1.20)
    jxl_bytes, _ = encode_single_frame(st_mb)
    ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
    assert ver["inspect"]["all_default"] is False
    assert ver["cross_decoder_psnr"] >= 45.0
  elif check_name == "state_bin_custom_dequant_roundtrip":
    st = _make_custom_ac_state(base_state, scale_factor=1.35)
    _set_custom_dc_quant(st, (1.0 / 3500.0, 1.0 / 450.0, 1.0 / 220.0))
    bin_path = save_state_bin(st, tmp_path / "custom_dq.bin", include_dequant=True)
    reloaded = load_state_bin(bin_path)
    assert 0 in reloaded.dequant_matrices
    np.testing.assert_allclose(
        reloaded.dequant_matrices[0], st.dequant_matrices[0], rtol=1e-5, atol=1e-6
    )


# =============================================================================
# F7: Differentiable Custom DequantMatrices Forward Model & Rate Proxy (R4, R5) — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "check_name",
    [
        "project_custom_dequant_matrices_parity",
        "forward_single_frame_custom_dequant_vs_djxl",
        "forward_custom_dequant_gradients",
        "soft_dequant_matrices_rate_bits_and_grad",
        "multires_composite_with_custom_dequant_vs_djxl",
    ],
)
def test_f7_differentiable_custom_dequant_forward_and_rate(
    check_name: str,
    kodim01_crop_64_01: np.ndarray,
) -> None:
  """Verifies F7 (R4, R5): differentiable custom DequantMatrices in forward_model.py and entropy_proxy.py."""
  crop32 = kodim01_crop_64_01[:32, :32]
  base_state = init_state_from_rgb_jax(crop32, distance=1.4, epf_iters=0, gab=True)

  if check_name == "project_custom_dequant_matrices_parity":
    proj_fn = getattr(fm_mod, "project_custom_dequant_matrices", None)
    assert proj_fn is not None, "Expected project_custom_dequant_matrices in forward_model.py"
    st = _make_custom_ac_state(base_state, scale_factor=1.25)
    proj_dq, proj_dc = proj_fn(st.dequant_matrices)
    assert isinstance(proj_dq, dict) and 0 in proj_dq
    assert bool(np.all(np.isfinite(proj_dq[0]))) and bool(np.all(proj_dq[0] > 0.0))
    assert proj_dc.shape == (3,) and bool(np.all(proj_dc > 0.0))
  elif check_name == "forward_single_frame_custom_dequant_vs_djxl":
    st = _make_custom_ac_state(base_state, scale_factor=1.25)
    proj_fn = getattr(fm_mod, "project_custom_dequant_matrices", None)
    if proj_fn is not None:
      st.dequant_matrices, _ = proj_fn(st.dequant_matrices)
    jxl_bytes, _ = encode_single_frame(st)
    info = inspect_jxl(jxl_bytes)
    assert info["all_default"] is False
    djxl_rgb = decode_with_djxl(jxl_bytes).astype(np.float32) / 255.0
    jax_rgb = np.asarray(reconstruct_single_frame(st, hard=True))
    psnr = compute_psnr(jax_rgb, djxl_rgb, peak=1.0)
    assert psnr >= 40.0, f"Expected PSNR >= 40 dB with custom DequantMatrices, got {psnr:.2f} dB"
  elif check_name == "forward_custom_dequant_gradients":
    dq0 = jnp.asarray(base_state.dequant_matrices[0], dtype=jnp.float32)

    def _loss_dq(dq_mat0: jax.Array) -> jax.Array:
      sig = inspect.signature(reconstruct_single_frame_xyb)
      if "dequant_matrices" in sig.parameters:
        xyb = reconstruct_single_frame_xyb(
            base_state, dequant_matrices={0: dq_mat0}, hard=False
        )
      elif "params_override" in sig.parameters:
        xyb = reconstruct_single_frame_xyb(
            base_state,
            params_override={"dequant_matrices": {0: dq_mat0}},
            hard=False,
        )
      else:
        st_tmp = base_state.copy()
        st_tmp.dequant_matrices = dict(base_state.dequant_matrices)
        st_tmp.dequant_matrices[0] = dq_mat0  # type: ignore[assignment]
        xyb = reconstruct_single_frame_xyb(st_tmp, hard=False)
      return jnp.mean(jnp.square(xyb))

    g_dq = np.asarray(jax.grad(_loss_dq)(dq0))
    assert bool(np.all(np.isfinite(g_dq)))
    assert float(np.linalg.norm(g_dq)) > 0.0
  elif check_name == "soft_dequant_matrices_rate_bits_and_grad":
    rate_fn = getattr(ep_mod, "soft_dequant_matrices_rate_bits", None)
    assert rate_fn is not None, "Expected soft_dequant_matrices_rate_bits in entropy_proxy.py"
    dq0_def = jnp.asarray(get_default_dequant_matrices()[0], dtype=jnp.float32)
    dq0_cust = dq0_def * 1.35
    b_def = float(rate_fn({0: dq0_def}))
    val_cust, grad_cust = jax.value_and_grad(lambda m: rate_fn({0: m}))(dq0_cust)
    assert float(val_cust) > b_def
    assert bool(jnp.all(jnp.isfinite(grad_cust)))
    assert float(np.linalg.norm(grad_cust)) > 0.0
  elif check_name == "multires_composite_with_custom_dequant_vs_djxl":
    ref_st = _make_custom_ac_state(
        init_state_from_rgb_jax(crop32[::2, ::2], distance=1.5, epf_iters=0), 1.20
    )
    main_st = _make_custom_ac_state(base_state, 1.25)
    proj_fn = getattr(fm_mod, "project_custom_dequant_matrices", None)
    if proj_fn is not None:
      ref_st.dequant_matrices, _ = proj_fn(ref_st.dequant_matrices)
      main_st.dequant_matrices, _ = proj_fn(main_st.dequant_matrices)
    patches = [PatchEntry(0, 0, 0, 16, 16, 0, 0, PATCH_BLEND_REPLACE)]
    jb, _ = encode_patches_composite(ref_st, main_st, upsampling=2, patches=patches)
    djxl_rgb = decode_with_djxl(jb).astype(np.float32) / 255.0
    jax_rgb = np.asarray(
        reconstruct_patch_composite(ref_st, main_st, 2, patches, hard=True)
    )
    assert compute_psnr(jax_rgb, djxl_rgb, peak=1.0) >= 40.0


# =============================================================================
# F8: Spatial Per-Block WD Map & Per-Footprint Rate Proxy Map (R3) — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "check_name",
    [
        "spatial_wd_map_shape_and_nonnegativity",
        "spatial_wd_map_exact_additivity",
        "spatial_wd_map_localizes_perturbation",
        "spatial_rate_map_shape_and_keys",
        "spatial_rate_map_additivity_and_multiblock_savings",
    ],
)
def test_f8_spatial_wd_and_rate_proxy_maps(
    check_name: str,
    kodim01_crop_64_01: np.ndarray,
    vgg_params_session: Any,
) -> None:
  """Verifies F8 (R3): compute_spatial_wd_map and compute_spatial_rate_map_bits additivity and localization."""
  spatial_wd_fn = getattr(wd_mod, "compute_spatial_wd_map", None)
  spatial_rate_fn = getattr(ep_mod, "compute_spatial_rate_map_bits", None)

  if check_name.startswith("spatial_wd_map"):
    assert spatial_wd_fn is not None, "Expected compute_spatial_wd_map in wasserstein.py"
    img_a = jnp.asarray(kodim01_crop_64_01, dtype=jnp.float32)
    img_b = jnp.clip(img_a + 0.05, 0.0, 1.0)
    total_wd, wd_block_map = spatial_wd_fn(img_a, img_b, vgg_params_session)
    if check_name == "spatial_wd_map_shape_and_nonnegativity":
      assert wd_block_map.shape == (8, 8)
      assert bool(jnp.all(jnp.isfinite(wd_block_map)))
      assert bool(jnp.all(wd_block_map >= 0.0))
    elif check_name == "spatial_wd_map_exact_additivity":
      assert float(jnp.sum(wd_block_map)) == pytest.approx(
          float(total_wd), rel=1e-5, abs=1e-6
      )
      assert float(total_wd) > 0.0
    elif check_name == "spatial_wd_map_localizes_perturbation":
      pert_br = np.array(kodim01_crop_64_01, copy=True)
      pert_br[48:, 48:, :] = np.clip(1.0 - pert_br[48:, 48:, :], 0.0, 1.0)
      _, wd_map_loc = spatial_wd_fn(
          img_a, jnp.asarray(pert_br, dtype=jnp.float32), vgg_params_session
      )
      br_mean = float(jnp.mean(wd_map_loc[6:, 6:]))
      tl_mean = float(jnp.mean(wd_map_loc[:2, :2]))
      assert br_mean > tl_mean * 1.5, (
          f"Expected bottom-right perturbed blocks ({br_mean}) > top-left ({tl_mean})"
      )
  else:
    assert spatial_rate_fn is not None, "Expected compute_spatial_rate_map_bits in entropy_proxy.py"
    crop32 = kodim01_crop_64_01[:32, :32]
    state8 = init_state_from_rgb_jax(crop32, distance=1.5)
    rmap8 = spatial_rate_fn(state8)
    if check_name == "spatial_rate_map_shape_and_keys":
      for k in ("block_bits", "ac_block_bits", "dc_block_bits", "meta_block_bits", "total_bits", "total_bpp"):
        assert k in rmap8, f"Missing key {k!r} in compute_spatial_rate_map_bits output"
      assert rmap8["block_bits"].shape == (4, 4)
      assert bool(jnp.all(jnp.isfinite(rmap8["block_bits"])))
      assert float(rmap8["total_bits"]) > 0.0
    elif check_name == "spatial_rate_map_additivity_and_multiblock_savings":
      block_sum = float(jnp.sum(rmap8["block_bits"]))
      comp_sum = float(
          jnp.sum(
              rmap8["ac_block_bits"]
              + rmap8["dc_block_bits"]
              + rmap8["meta_block_bits"]
          )
      )
      assert block_sum == pytest.approx(comp_sum, rel=1e-4, abs=1e-3)
      smooth32 = np.full((32, 32, 3), 0.5, dtype=np.float32)
      st_s8 = init_state_from_rgb_jax(
          smooth32, distance=1.5, ac_strategy=np.zeros((4, 4), dtype=np.uint8)
      )
      st_s16 = init_state_from_rgb_jax(
          smooth32, distance=1.5, ac_strategy=np.full((4, 4), 4, dtype=np.uint8)
      )
      r_s8 = spatial_rate_fn(st_s8)
      r_s16 = spatial_rate_fn(st_s16)
      meta_dc_8 = float(jnp.sum(r_s8["meta_block_bits"] + r_s8["dc_block_bits"]))
      meta_dc_16 = float(jnp.sum(r_s16["meta_block_bits"] + r_s16["dc_block_bits"]))
      assert meta_dc_16 < meta_dc_8


# =============================================================================
# F9: Stage 1 Candidate Exploration with Preliminary Rate-WD Optimization (R3) — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "check_name",
    [
        "stage1_explores_multiple_ac_strategy_candidates",
        "stage1_preliminary_optimization_lowers_candidate_loss",
        "stage1_records_spatial_j_maps_per_candidate",
        "stage1_explores_downsampling_ratios_2_4_8",
        "stage1_explores_single_patches_and_alpha_modes",
    ],
)
def test_f9_stage1_candidate_exploration(
    check_name: str,
    mixed_image_64_01: np.ndarray,
    composite_smooth_sharp_64_01: np.ndarray,
    vgg_params_session: Any,
) -> None:
  """Verifies F9 (R3): Stage 1 evaluates and optimizes candidate block shapes and downsampling/blending topologies."""
  import jxl_wasserstein.compositing as comp_mod
  import jxl_wasserstein.optimizer as opt_mod

  if check_name in (
      "stage1_explores_multiple_ac_strategy_candidates",
      "stage1_preliminary_optimization_lowers_candidate_loss",
      "stage1_records_spatial_j_maps_per_candidate",
  ):
    res = opt_mod.optimize_single_frame(
        mixed_image_64_01[:32, :32],
        vgg_params_session,
        rd_lambda=0.15,
        num_steps=2,
    )
    s1 = get_field(res, "stage1_candidates")
    assert s1 is not None, "Expected stage1_candidates on single-frame result"
    cand_items = list(s1.values()) if isinstance(s1, dict) else list(s1)
    assert len(cand_items) >= 3, f"Expected >= 3 Stage 1 candidates, got {len(cand_items)}"

    if check_name == "stage1_explores_multiple_ac_strategy_candidates":
      strat_ids = {
          get_field(c, "strategy_id", "ac_strategy_id", "strategy", "name")
          for c in cand_items
      }
      assert len(strat_ids) >= 3
    elif check_name == "stage1_preliminary_optimization_lowers_candidate_loss":
      for c in cand_items:
        j_score = float(get_field(c, "j_score", "rd_cost", "lagrangian"))
        assert np.isfinite(j_score) and j_score > 0.0
        j_init = get_field(c, "j_init", "initial_j_score")
        if j_init is not None:
          assert j_score <= float(j_init) + 1e-4
    elif check_name == "stage1_records_spatial_j_maps_per_candidate":
      for c in cand_items:
        j_map = get_field(c, "j_map", "spatial_j_map", "cost_map")
        assert j_map is not None
        arr_map = np.asarray(j_map)
        assert arr_map.shape == (4, 4)
        assert bool(np.all(np.isfinite(arr_map)))
  elif check_name == "stage1_explores_downsampling_ratios_2_4_8":
    res = comp_mod.select_best_mode(
        composite_smooth_sharp_64_01,
        candidate_modes=("patches",),
        upsampling_candidates=(2, 4, 8),
        blend_mode_candidates=(PATCH_BLEND_REPLACE,),
        distance=0.5,
        lambda_rd=0.12,
        num_steps=1,
        verify_decoders=False,
        vgg_params=vgg_params_session,
    )
    cands = get_field(res, "stage1_candidates", "all_candidates")
    assert cands is not None and len(cands) >= 3
  elif check_name == "stage1_explores_single_patches_and_alpha_modes":
    res = comp_mod.select_best_mode(
        composite_smooth_sharp_64_01,
        candidate_modes=("single", "patches", "alpha"),
        upsampling_candidates=(2,),
        dim_shift_candidates=(0,),
        blend_mode_candidates=(PATCH_BLEND_REPLACE,),
        distance=0.45,
        lambda_rd=0.15,
        num_steps=1,
        verify_decoders=False,
        vgg_params=vgg_params_session,
    )
    all_cands = get_field(res, "all_candidates", "stage1_candidates")
    assert all_cands is not None
    cand_list = list(all_cands.values()) if isinstance(all_cands, dict) else list(all_cands)
    modes_seen = {get_field(c, "mode") for c in cand_list}
    assert {"single", "patches", "alpha"}.issubset(modes_seen)


# =============================================================================
# F10: Stage 2 Discrete AcStrategy DP, Patch Placement & Mode Selection (R3) — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "check_name",
    [
        "stage2_dp_valid_non_overlapping_ac_strategy_grid",
        "stage2_selects_multiple_distinct_transforms_on_mixed_image",
        "stage2_selects_patches_from_rate_wd_advantage_map",
        "stage2_initializes_xyb_alpha_from_rate_wd_advantage_map",
        "stage2_materializes_unrefined_true_j",
    ],
)
def test_f10_stage2_discrete_decisions(
    check_name: str,
    mixed_image_64_01: np.ndarray,
    composite_smooth_sharp_64_01: np.ndarray,
    vgg_params_session: Any,
) -> None:
  """Verifies F10 (R3): Stage 2 non-overlapping AcStrategy DP, patch placement, alpha init, and stage2_true_j."""
  import jxl_wasserstein.compositing as comp_mod
  import jxl_wasserstein.optimizer as opt_mod

  if check_name in (
      "stage2_dp_valid_non_overlapping_ac_strategy_grid",
      "stage2_selects_multiple_distinct_transforms_on_mixed_image",
      "stage2_materializes_unrefined_true_j",
  ):
    res = opt_mod.optimize_single_frame(
        mixed_image_64_01,
        vgg_params_session,
        rd_lambda=0.15,
        num_steps=2,
    )
    acs = _extract_ac_strategy(res)
    if check_name == "stage2_dp_valid_non_overlapping_ac_strategy_grid":
      assert acs.shape == (8, 8)
      st = get_field(res, "state", "main_state")
      topo = StaticVarDCTTopology(st)
      assert topo is not None
    elif check_name == "stage2_selects_multiple_distinct_transforms_on_mixed_image":
      assert len(np.unique(acs)) >= 2
      assert bool(np.any(acs != 0))
    elif check_name == "stage2_materializes_unrefined_true_j":
      s2_bpp = get_field(res, "stage2_true_bpp", "step0_bpp")
      s2_wd = get_field(res, "stage2_wd", "step0_wd")
      s2_j, _ = _extract_stage2_and_stage3_j(res)
      assert s2_bpp is not None and float(s2_bpp) > 0.0
      assert s2_wd is not None and float(s2_wd) >= 0.0
      assert np.isfinite(s2_j) and s2_j > 0.0
  elif check_name == "stage2_selects_patches_from_rate_wd_advantage_map":
    res_p = comp_mod.optimize_patch_composite(
        composite_smooth_sharp_64_01,
        upsampling=2,
        blend_mode=PATCH_BLEND_REPLACE,
        distance=0.45,
        lambda_rd=0.08,
        num_steps=1,
        verify_decoders=False,
        vgg_params=vgg_params_session,
    )
    patches = get_field(res_p, "patches")
    assert patches is not None and len(patches) >= 1
    for p in patches:
      assert p.blend_mode in (PATCH_BLEND_REPLACE, PATCH_BLEND_ADD)
  elif check_name == "stage2_initializes_xyb_alpha_from_rate_wd_advantage_map":
    res_a = comp_mod.optimize_alpha_composite(
        composite_smooth_sharp_64_01,
        upsampling=2,
        dim_shift=0,
        distance=0.45,
        lambda_rd=0.08,
        num_steps=1,
        verify_decoders=False,
        vgg_params=vgg_params_session,
    )
    alpha_arr = get_field(res_a, "alpha_01", "alpha_u8")
    assert alpha_arr is not None
    alpha_f = np.asarray(alpha_arr, dtype=np.float32)
    if alpha_f.max() > 1.5:
      alpha_f = alpha_f / 255.0
    center_mean = float(np.mean(alpha_f[24:40, 24:40]))
    corner_mean = float(np.mean(alpha_f[:16, :16]))
    assert center_mean > corner_mean, (
        f"Expected higher alpha in sharp center ({center_mean:.3f}) than smooth corner ({corner_mean:.3f})"
    )


# =============================================================================
# F11: Stage 3 Joint Continuous Refinement & Custom DequantMatrices Gain (R3, R4) — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "check_name",
    [
        "stage3_strictly_improves_true_lagrangian_single_frame",
        "stage3_strictly_improves_true_lagrangian_patches_and_alpha",
        "stage3_emits_custom_dequant_matrices_in_bitstream",
        "stage3_custom_dequant_beats_fixed_default_matrices",
        "stage3_refines_cfl_quant_field_alpha_and_noise_lut",
    ],
)
def test_f11_stage3_continuous_refinement_and_custom_dequant(
    check_name: str,
    mixed_image_64_01: np.ndarray,
    composite_smooth_sharp_64_01: np.ndarray,
    noisy_textured_64_01: np.ndarray,
    vgg_params_session: Any,
) -> None:
  """Verifies F11 (R3, R4): stage3_true_j < stage2_true_j and custom DequantMatrices Lagrangian improvement."""
  import jxl_wasserstein.compositing as comp_mod
  import jxl_wasserstein.optimizer as opt_mod

  if check_name == "stage3_strictly_improves_true_lagrangian_single_frame":
    res = opt_mod.optimize_single_frame(
        mixed_image_64_01,
        vgg_params_session,
        rd_lambda=0.15,
        num_steps=4,
    )
    s2_j, s3_j = _extract_stage2_and_stage3_j(res)
    assert s3_j < s2_j, f"Expected stage3_true_j ({s3_j:.6f}) < stage2_true_j ({s2_j:.6f})"
  elif check_name == "stage3_strictly_improves_true_lagrangian_patches_and_alpha":
    res_p = comp_mod.optimize_patch_composite(
        composite_smooth_sharp_64_01,
        upsampling=2,
        blend_mode=PATCH_BLEND_REPLACE,
        distance=0.45,
        lambda_rd=0.06,
        num_steps=3,
        verify_decoders=False,
        vgg_params=vgg_params_session,
    )
    s2_jp, s3_jp = _extract_stage2_and_stage3_j(res_p)
    assert s3_jp < s2_jp, f"Patches: expected stage3_true_j ({s3_jp:.6f}) < stage2_true_j ({s2_jp:.6f})"

    res_a = comp_mod.optimize_alpha_composite(
        composite_smooth_sharp_64_01,
        upsampling=2,
        dim_shift=0,
        distance=0.45,
        lambda_rd=0.06,
        num_steps=3,
        verify_decoders=False,
        vgg_params=vgg_params_session,
    )
    s2_ja, s3_ja = _extract_stage2_and_stage3_j(res_a)
    assert s3_ja < s2_ja, f"Alpha: expected stage3_true_j ({s3_ja:.6f}) < stage2_true_j ({s2_ja:.6f})"
  elif check_name == "stage3_emits_custom_dequant_matrices_in_bitstream":
    res = opt_mod.optimize_single_frame(
        mixed_image_64_01[:32, :32],
        vgg_params_session,
        rd_lambda=0.15,
        num_steps=3,
    )
    info = inspect_jxl(get_field(res, "jxl_bytes"))
    assert info["all_default"] is False
  elif check_name == "stage3_custom_dequant_beats_fixed_default_matrices":
    sig = inspect.signature(opt_mod.optimize_single_frame)
    dq_kw = (
        "custom_dequant"
        if "custom_dequant" in sig.parameters
        else (
            "optimize_dequant"
            if "optimize_dequant" in sig.parameters
            else "enable_custom_dequant"
        )
    )
    res_cust = opt_mod.optimize_single_frame(
        mixed_image_64_01[:32, :32],
        vgg_params_session,
        rd_lambda=0.15,
        num_steps=4,
        **({dq_kw: True} if dq_kw in sig.parameters else {}),
    )
    res_fixed = opt_mod.optimize_single_frame(
        mixed_image_64_01[:32, :32],
        vgg_params_session,
        rd_lambda=0.15,
        num_steps=4,
        **({dq_kw: False} if dq_kw in sig.parameters else {}),
    )
    _, j_cust = _extract_stage2_and_stage3_j(res_cust)
    _, j_fixed = _extract_stage2_and_stage3_j(res_fixed)
    assert j_cust < j_fixed, (
        f"Expected custom DequantMatrices J ({j_cust:.6f}) < fixed default J ({j_fixed:.6f})"
    )
  elif check_name == "stage3_refines_cfl_quant_field_alpha_and_noise_lut":
    res_n = opt_mod.optimize_single_frame(
        noisy_textured_64_01[:32, :32],
        vgg_params_session,
        rd_lambda=0.20,
        num_steps=3,
        enable_noise=True,
    )
    info_n = inspect_jxl(get_field(res_n, "jxl_bytes"))
    assert info_n["frames"][0]["has_noise"] is True
    assert max(info_n["frames"][0]["noise_lut"]) > 0.001


# =============================================================================
# F12: Differentiable Entropy Proxy Rank Correlation & Lambda Monotonicity (R5) — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "check_name",
    [
        "spearman_rank_correlation_across_0_15_to_2_0_bpp",
        "spearman_correlation_with_multiblock_and_custom_dequant",
        "monotonic_4pt_lambda_sweep_bitrates",
        "rate_breakdown_components_and_gradients",
        "patch_and_alpha_composite_rate_proxies",
    ],
)
def test_f12_entropy_proxy_correlation_and_lambda_monotonicity(
    check_name: str,
    kodim01_crop_64_01: np.ndarray,
    cached_lambda_sweep_4pt: list[Any],
) -> None:
  """Verifies F12 (R5): Spearman rank correlation >= 0.95 across 0.15-2.0 bpp and monotonic 4-point lambda sweep."""
  crop32 = kodim01_crop_64_01[:32, :32]
  distances = (0.35, 0.55, 0.85, 1.3, 2.0, 3.2, 5.0, 8.5)

  if check_name == "spearman_rank_correlation_across_0_15_to_2_0_bpp":
    proxy_bits, true_bits, bpps = [], [], []
    for d in distances:
      st = init_state_from_rgb_jax(kodim01_crop_64_01, distance=d)
      jb, stats = encode_single_frame(st)
      proxy_bits.append(float(soft_jxl_rate_proxy(st)))
      true_bits.append(float(stats["total_bits"]))
      bpps.append(len(jb) * 8.0 / float(64 * 64))
    assert min(bpps) <= 0.45 and max(bpps) >= 1.5
    rho = float(spearmanr(proxy_bits, true_bits).statistic)
    assert rho >= 0.95, f"Expected Spearman rho >= 0.95, got {rho:.4f}"
  elif check_name == "spearman_correlation_with_multiblock_and_custom_dequant":
    acs = np.zeros((4, 4), dtype=np.uint8)
    acs[0:2, 0:2] = 4  # DCT16X16
    proxy_bits, true_bits = [], []
    for d in distances:
      st = init_state_from_rgb_jax(crop32, distance=d, ac_strategy=acs)
      st = _make_custom_ac_state(st, scale_factor=1.15)
      _, stats = encode_single_frame(st)
      proxy_bits.append(float(soft_jxl_rate_proxy(st)))
      true_bits.append(float(stats["total_bits"]))
    rho = float(spearmanr(proxy_bits, true_bits).statistic)
    assert rho >= 0.95, f"Expected Spearman rho >= 0.95 on custom dequant states, got {rho:.4f}"
  elif check_name == "monotonic_4pt_lambda_sweep_bitrates":
    bpps = [float(get_field(r, "bpp")) for r in cached_lambda_sweep_4pt]
    assert len(bpps) == 4
    for i in range(3):
      assert bpps[i] >= bpps[i + 1], f"Non-monotonic bpp sweep at {i}: {bpps}"
    assert bpps[0] > bpps[-1]
  elif check_name == "rate_breakdown_components_and_gradients":
    st = init_state_from_rgb_jax(crop32, distance=1.5)
    bd = soft_jxl_rate_proxy(st, return_breakdown=True)
    assert isinstance(bd, RateBreakdown)
    assert float(bd.total_bits) > 0.0
    assert float(bd.ac_bits) > 0.0 and float(bd.dc_bits) > 0.0
    q_ac = jnp.asarray(st.quant_ac, dtype=jnp.float32)
    g_ac = np.asarray(jax.grad(lambda ac: soft_jxl_rate_proxy(st, quant_ac=ac))(q_ac))
    assert bool(np.all(np.isfinite(g_ac))) and float(np.linalg.norm(g_ac)) > 0.0
  elif check_name == "patch_and_alpha_composite_rate_proxies":
    ref_st = init_state_from_rgb_jax(crop32[::2, ::2], distance=1.8)
    main_st = init_state_from_rgb_jax(crop32, distance=1.2)
    patches = [PatchEntry(0, 0, 0, 16, 16, 0, 0, PATCH_BLEND_REPLACE)]
    p_bits = float(
        soft_patch_composite_rate_proxy(ref_st, main_st, patches=patches, upsampling=2)
    )
    alpha_01 = jnp.full((32, 32), 0.5, dtype=jnp.float32)
    a_bits = float(
        soft_alpha_composite_rate_proxy(
            ref_st, main_st, alpha_01=alpha_01, upsampling=2, dim_shift=0
        )
    )
    assert p_bits > 0.0 and a_bits > 0.0


# =============================================================================
# F13: Kodak R-D Benchmark Negative Mean BD-Rate vs cjxl -e 7 & Cross-Decoder Interop (R5) — 5 test cases
# =============================================================================


@pytest.mark.parametrize(
    "check_name",
    [
        "bd_rate_analytical_shifts",
        "cjxl_e7_baseline_wrapper",
        "evaluate_image_rd_curve_uses_3stage_and_custom_dequant",
        "cli_end_to_end_encode_and_json_report",
        "kodak_benchmark_negative_mean_bd_rate",
    ],
)
def test_f13_kodak_benchmark_and_cross_decoder_interop(
    check_name: str,
    kodim01_crop_64_u8: np.ndarray,
    vgg_params_session: Any,
    tmp_path: Path,
) -> None:
  """Verifies F13 (R5): Kodak R-D benchmark negative BD-rate vs cjxl -e 7, 3-stage + custom dequant, and cross-decoder interop."""
  if check_name == "bd_rate_analytical_shifts":
    rate_ref = np.array([0.3, 0.6, 1.0, 1.5], dtype=np.float64)
    dist_ref = np.array([2.5, 1.4, 0.7, 0.35], dtype=np.float64)
    assert compute_bd_rate(rate_ref, dist_ref, rate_ref, dist_ref) == pytest.approx(
        0.0, abs=1e-6
    )
    assert compute_bd_rate(
        rate_ref, dist_ref, rate_ref * 0.85, dist_ref
    ) == pytest.approx(-15.0, abs=1e-4)
    assert compute_bd_rate(
        rate_ref, dist_ref, rate_ref * 1.10, dist_ref
    ) == pytest.approx(10.0, abs=1e-4)
  elif check_name == "cjxl_e7_baseline_wrapper":
    jxl_b, dec_u8 = encode_cjxl_baseline(kodim01_crop_64_u8, distance=2.0, effort=7)
    assert len(jxl_b) > 0
    assert dec_u8.shape == kodim01_crop_64_u8.shape
    assert compute_psnr(kodim01_crop_64_u8, dec_u8, peak=255.0) >= 25.0
  elif check_name == "evaluate_image_rd_curve_uses_3stage_and_custom_dequant":
    res = evaluate_image_rd_curve(
        kodim01_crop_64_u8,
        image_name="kodim01_64",
        distances=(1.2, 2.2, 4.0, 7.0),
        steps=2,
        verify=True,
        vgg_params=vgg_params_session,
    )
    assert len(res.cjxl_points) == 4
    assert len(res.opt_points) == 4
    for pt in res.opt_points:
      assert pt.cross_decoder_psnr is not None and pt.cross_decoder_psnr >= 45.0
      jxl_b = get_field(pt, "jxl_bytes")
      if jxl_b is not None:
        info = inspect_jxl(jxl_b)
        assert info["all_default"] is False
      acs = get_field(pt, "ac_strategy", "stage2_ac_strategy")
      if acs is not None:
        assert bool(np.any(np.asarray(acs) != 0))
  elif check_name == "cli_end_to_end_encode_and_json_report":
    in_png = tmp_path / "in.png"
    out_jxl = tmp_path / "out.jxl"
    json_out = tmp_path / "stats.json"
    Image.fromarray(kodim01_crop_64_u8, mode="RGB").save(in_png)
    rc = cli_main(
        [
            "--input",
            str(in_png),
            "--output",
            str(out_jxl),
            "--distance",
            "2.0",
            "--steps",
            "2",
            "--verify",
            "--json-out",
            str(json_out),
        ]
    )
    assert rc == 0
    assert out_jxl.is_file() and out_jxl.stat().st_size > 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["cross_decoder_psnr"] >= 45.0
    info = inspect_jxl(out_jxl.read_bytes())
    assert info["all_default"] is False
  elif check_name == "kodak_benchmark_negative_mean_bd_rate":
    report = run_kodak_benchmark(
        images=("kodim01.png",),
        crop_size=128,
        distances=DEFAULT_BENCHMARK_DISTANCES,
        steps=4,
        mode="single",
        effort=7,
        verify=True,
    )
    assert len(report.per_image) == 1
    assert report.mean_bd_rate_opt_pct < 0.0
