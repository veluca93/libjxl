"""Tier 4 E2E Test Suite: Real-World Application & Kodak R-D Benchmark Scenarios (8 scenarios, >= 7 required)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import spearmanr

from jxl_wasserstein.benchmark import (
    DEFAULT_BENCHMARK_DISTANCES,
    format_benchmark_table,
    run_kodak_benchmark,
)
from jxl_wasserstein.cpp_bridge import (
    PATCH_BLEND_ADD,
    compute_psnr,
    decode_with_djxl,
    encode_single_frame,
    inspect_jxl,
    verify_jxl_codestream,
)
from jxl_wasserstein.entropy_proxy import soft_jxl_rate_proxy
import jxl_wasserstein.entropy_proxy as ep_mod
from jxl_wasserstein.forward_model import (
    get_default_dequant_matrices,
    init_state_from_rgb_jax,
    reconstruct_single_frame,
)
import jxl_wasserstein.forward_model as fm_mod
import jxl_wasserstein.wasserstein as wd_mod

from conftest import (
    FORBIDDEN_LEGACY_TOKENS,
    PKG_ROOT,
    REPO_ROOT,
    forbid_libjxl_per_block_choices,
    get_field,
)


def _extract_stage2_and_stage3_j(res: Any) -> tuple[float, float]:
  s2_j = get_field(res, "stage2_true_j")
  s3_j = get_field(res, "stage3_true_j")
  if s2_j is not None and s3_j is not None:
    return float(s2_j), float(s3_j)
  hist = get_field(res, "history")
  assert hist and len(hist) >= 1
  j0 = float(hist[0].get("lagrangian_true", hist[0].get("rd_loss")))
  j_best = float(get_field(res, "lagrangian", "rd_cost"))
  return j0, j_best


def test_scenario1_xyb_patch_alpha_blending_and_cross_decoder_interoperability(
    composite_smooth_sharp_64_01: np.ndarray,
    vgg_params_session: Any,
) -> None:
  """Scenario 1 (R2): XYB PatchDictionary alpha-blending across U in {2,4,8} and dim_shift in {0..3} with djxl & jxl_cli interop."""
  from jxl_wasserstein.compositing import optimize_alpha_composite

  for upsampling, dim_shift in ((2, 0), (2, 1), (4, 2), (8, 3)):
    res_a = optimize_alpha_composite(
        composite_smooth_sharp_64_01,
        upsampling=upsampling,
        dim_shift=dim_shift,
        distance=0.45,
        lambda_rd=0.05,
        num_steps=2,
        enable_noise=False,
        vgg_params=vgg_params_session,
    )
    jxl_bytes = get_field(res_a, "jxl_bytes")
    ver = verify_jxl_codestream(jxl_bytes, min_cross_decoder_psnr=45.0)
    assert ver["cross_decoder_psnr"] >= 45.0

    info = ver["inspect"]
    assert info["num_frames"] == 2
    assert info["frames"][0]["upsampling"] == upsampling
    assert info["frames"][0]["save_before_color_transform"] is True
    assert info["frames"][1]["has_patches"] is True
    assert info["frames"][1]["num_patches"] >= 1
    assert any(
        m in (4, 5, 6, 7) for m in info["frames"][1]["patch_blend_modes"]
    ), f"Expected alpha-aware PatchBlendMode in {{4,5,6,7}}, got {info['frames'][1]['patch_blend_modes']}"
    assert info["extra_channels"][0]["dim_shift"] == dim_shift

    djxl_rgb = ver["rgb_djxl"].astype(np.float32) / 255.0
    recon_rgb = np.asarray(get_field(res_a, "recon_rgb", "recon_jax_rgb"))
    assert compute_psnr(recon_rgb, djxl_rgb, peak=1.0) >= 40.0


def test_scenario2_three_stage_global_optimization_and_wd_block_selection(
    mixed_image_64_01: np.ndarray,
    vgg_params_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Scenario 2 (R3): 3-stage pipeline candidate exploration, Stage 2 multi-transform AcStrategy DP, and Stage 3 strict improvement."""
  from jxl_wasserstein.optimizer import optimize_single_frame

  spatial_wd_fn = getattr(wd_mod, "compute_spatial_wd_map", None)
  spatial_rate_fn = getattr(ep_mod, "compute_spatial_rate_map_bits", None)
  assert spatial_wd_fn is not None and spatial_rate_fn is not None

  img_a = jnp.asarray(mixed_image_64_01, dtype=jnp.float32)
  img_b = jnp.clip(img_a + 0.04, 0.0, 1.0)
  tot_wd, wd_map = spatial_wd_fn(img_a, img_b, vgg_params_session)
  assert wd_map.shape == (8, 8) and float(jnp.sum(wd_map)) == pytest.approx(
      float(tot_wd), rel=1e-5, abs=1e-6
  )

  with forbid_libjxl_per_block_choices(monkeypatch):
    res = optimize_single_frame(
        mixed_image_64_01,
        vgg_params_session,
        rd_lambda=0.15,
        num_steps=4,
    )

  # 1. Stage 1 candidate exploration
  s1 = get_field(res, "stage1_candidates")
  assert s1 is not None and len(s1) >= 3

  # 2. Stage 2 non-overlapping multi-transform AcStrategy grid
  st = get_field(res, "state", "main_state")
  acs = np.asarray(
      get_field(res, "stage2_ac_strategy")
      if get_field(res, "stage2_ac_strategy") is not None
      else st.ac_strategy
  )
  assert acs.shape == (8, 8)
  assert len(np.unique(acs)) >= 2
  assert bool(np.any(acs != 0))

  # 3. Stage 3 strict improvement of true bitstream Lagrangian
  s2_j, s3_j = _extract_stage2_and_stage3_j(res)
  assert s3_j < s2_j, (
      f"Expected Stage 3 true Lagrangian ({s3_j:.6f}) < Stage 2 true Lagrangian ({s2_j:.6f})"
  )


def test_scenario3_custom_dequant_matrices_optimization_and_serialization(
    mixed_image_64_01: np.ndarray,
    vgg_params_session: Any,
) -> None:
  """Scenario 3 (R4): Custom DequantMatrices optimization lowers Rate-WD Lagrangian over fixed defaults and serializes all_default == False."""
  from jxl_wasserstein.optimizer import optimize_single_frame

  sig = inspect.signature(optimize_single_frame)
  dq_kw = (
      "custom_dequant"
      if "custom_dequant" in sig.parameters
      else (
          "optimize_dequant"
          if "optimize_dequant" in sig.parameters
          else "enable_custom_dequant"
      )
  )

  res_cust = optimize_single_frame(
      mixed_image_64_01[:32, :32],
      vgg_params_session,
      rd_lambda=0.15,
      num_steps=4,
      **({dq_kw: True} if dq_kw in sig.parameters else {}),
  )
  res_fixed = optimize_single_frame(
      mixed_image_64_01[:32, :32],
      vgg_params_session,
      rd_lambda=0.15,
      num_steps=4,
      **({dq_kw: False} if dq_kw in sig.parameters else {}),
  )

  jxl_cust = get_field(res_cust, "jxl_bytes")
  ver_cust = verify_jxl_codestream(jxl_cust, min_cross_decoder_psnr=45.0)
  assert ver_cust["cross_decoder_psnr"] >= 45.0
  assert ver_cust["inspect"]["all_default"] is False

  st_cust = get_field(res_cust, "state", "main_state")
  djxl_rgb = decode_with_djxl(jxl_cust).astype(np.float32) / 255.0
  jax_rgb = np.asarray(reconstruct_single_frame(st_cust, hard=True))
  assert compute_psnr(jax_rgb, djxl_rgb, peak=1.0) >= 40.0

  _, j_cust = _extract_stage2_and_stage3_j(res_cust)
  _, j_fixed = _extract_stage2_and_stage3_j(res_fixed)
  assert j_cust < j_fixed, (
      f"Expected custom DequantMatrices Lagrangian ({j_cust:.6f}) < fixed default ({j_fixed:.6f})"
  )


def test_scenario4_multires_auto_mode_selection_on_mixed_composite_scene(
    composite_smooth_sharp_64_01: np.ndarray,
    cached_auto_mode_selection: Any,
    vgg_params_session: Any,
) -> None:
  """Scenario 4 (R2, R3): Automatic 3-stage RD mode selection on smooth-background + sharp-foreground scene."""
  from jxl_wasserstein.compositing import encode_image_to_jxl

  mode_best = get_field(
      cached_auto_mode_selection, "mode", "mode_selected", "mode_used"
  )
  assert mode_best in ("patches", "alpha")

  all_cands = get_field(cached_auto_mode_selection, "all_candidates")
  assert all_cands is not None and len(all_cands) >= 3
  c_single = [c for c in all_cands if c["mode"] == "single"][0]
  c_patches = min(
      [c for c in all_cands if c["mode"] == "patches"],
      key=lambda x: x["rd_cost"],
  )
  c_alpha = min(
      [c for c in all_cands if c["mode"] == "alpha"],
      key=lambda x: x["rd_cost"],
  )
  assert c_patches["rd_cost"] < c_single["rd_cost"]
  assert c_alpha["rd_cost"] < c_single["rd_cost"]

  res_auto = encode_image_to_jxl(
      composite_smooth_sharp_64_01,
      mode="auto",
      distance=0.35,
      lambda_rd=0.18,
      num_steps=2,
      vgg_params=vgg_params_session,
  )
  sel_mode = get_field(res_auto, "mode", "mode_selected", "mode_used")
  assert sel_mode in ("patches", "alpha")
  ver = verify_jxl_codestream(
      get_field(res_auto, "jxl_bytes"), min_cross_decoder_psnr=45.0
  )
  assert ver["inspect"]["num_frames"] == 2
  assert ver["inspect"]["frames"][0]["save_before_color_transform"] is True
  assert ver["inspect"]["frames"][1]["has_patches"] is True


def test_scenario5_entropy_proxy_spearman_correlation_and_4pt_lambda_sweep(
    kodim01_crop_64_01: np.ndarray,
    cached_lambda_sweep_4pt: list[Any],
) -> None:
  """Scenario 5 (R5): Differentiable entropy proxy Spearman rho >= 0.95 across 0.15-2.0 bpp and monotonic 4-point lambda sweep."""
  distances = (0.35, 0.55, 0.85, 1.3, 2.0, 3.2, 5.0, 8.5, 14.0, 38.0)
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

  sweep_bpps = [float(get_field(r, "bpp")) for r in cached_lambda_sweep_4pt]
  assert len(sweep_bpps) == 4
  for i in range(3):
    assert (
        sweep_bpps[i] >= sweep_bpps[i + 1]
    ), f"Bitrate not monotonically non-increasing at index {i}: {sweep_bpps}"
  assert sweep_bpps[0] > sweep_bpps[-1]


def test_scenario6_textured_photography_noise_synthesis_and_cross_decoder_matrix(
    composite_smooth_sharp_64_01: np.ndarray,
    cached_noise_comparison: dict[str, Any],
    cached_single_frame_opt: Any,
    cached_patch_composite_u2: Any,
    cached_alpha_composite_u2_d0: Any,
    vgg_params_session: Any,
) -> None:
  """Scenario 6 (R2, R4, R5): JXL NoiseParams synthesis lowers WD and all modes pass djxl & jxl_cli interop."""
  from jxl_wasserstein.compositing import (
      optimize_alpha_composite,
      optimize_patch_composite,
  )

  res_no_noise = cached_noise_comparison["no_noise"]
  res_with_noise = cached_noise_comparison["with_noise"]
  assert float(get_field(res_with_noise, "wd")) < float(
      get_field(res_no_noise, "wd")
  )
  jxl_noisy = get_field(res_with_noise, "jxl_bytes")
  info_noisy = inspect_jxl(jxl_noisy)
  assert info_noisy["frames"][0]["has_noise"] is True
  assert max(info_noisy["frames"][0]["noise_lut"]) > 0.001
  assert verify_jxl_codestream(jxl_noisy, min_cross_decoder_psnr=45.0)["cross_decoder_psnr"] >= 45.0

  # Single-frame mode
  ver_single = verify_jxl_codestream(
      get_field(cached_single_frame_opt, "jxl_bytes"), min_cross_decoder_psnr=45.0
  )
  assert ver_single["cross_decoder_psnr"] >= 45.0
  assert float(get_field(cached_single_frame_opt, "psnr_jax_vs_djxl", "psnr_vs_djxl", "psnr")) >= 40.0

  # Patch composite modes (U=2 kAdd + U=4 kAdd)
  for res_p in (
      cached_patch_composite_u2,
      optimize_patch_composite(
          composite_smooth_sharp_64_01,
          upsampling=4,
          blend_mode=PATCH_BLEND_ADD,
          distance=0.45,
          lambda_rd=0.05,
          num_steps=2,
          vgg_params=vgg_params_session,
      ),
  ):
    ver_p = verify_jxl_codestream(
        get_field(res_p, "jxl_bytes"), min_cross_decoder_psnr=45.0
    )
    assert ver_p["cross_decoder_psnr"] >= 45.0
    djxl_rgb = ver_p["rgb_djxl"].astype(np.float32) / 255.0
    recon_rgb = np.asarray(get_field(res_p, "recon_rgb", "recon_jax_rgb"))
    assert compute_psnr(recon_rgb, djxl_rgb, peak=1.0) >= 40.0

  # Alpha composite modes (U=2 D=0 + U=4 D=2)
  for res_a in (
      cached_alpha_composite_u2_d0,
      optimize_alpha_composite(
          composite_smooth_sharp_64_01,
          upsampling=4,
          dim_shift=2,
          distance=0.45,
          lambda_rd=0.05,
          num_steps=2,
          vgg_params=vgg_params_session,
      ),
  ):
    ver_a = verify_jxl_codestream(
        get_field(res_a, "jxl_bytes"), min_cross_decoder_psnr=45.0
    )
    assert ver_a["cross_decoder_psnr"] >= 45.0
    assert ver_a["inspect"]["frames"][0]["save_before_color_transform"] is True
    assert ver_a["inspect"]["frames"][1]["has_patches"] is True
    djxl_rgb = ver_a["rgb_djxl"].astype(np.float32) / 255.0
    recon_rgb = np.asarray(get_field(res_a, "recon_rgb", "recon_jax_rgb"))
    assert compute_psnr(recon_rgb, djxl_rgb, peak=1.0) >= 40.0


def test_scenario7_kodak_rd_benchmark_negative_mean_bd_rate_vs_cjxl_e7(
    tmp_path: Path,
) -> None:
  """Scenario 7 (R3, R4, R5): Kodak R-D benchmark across 4 bitrates with 3-stage WD + custom DequantMatrices achieves negative mean BD-rate vs cjxl -e 7."""
  json_report_path = tmp_path / "kodak_report.json"
  report = run_kodak_benchmark(
      images=("kodim01.png", "kodim08.png"),
      crop_size=256,
      distances=DEFAULT_BENCHMARK_DISTANCES,
      steps=8,
      mode="single",
      effort=7,
      verify=True,
      json_out=json_report_path,
  )

  table_str = format_benchmark_table(report)
  assert "kodim01.png" in table_str and "kodim08.png" in table_str
  assert json_report_path.is_file()
  json_data = json.loads(json_report_path.read_text(encoding="utf-8"))
  assert len(json_data["per_image"]) == 2

  for img_res in report.per_image:
    assert len(img_res.cjxl_points) == 4
    assert len(img_res.opt_points) == 4
    assert img_res.bd_rate_opt_pct < 0.0
    assert img_res.bd_rate_opt_pct < img_res.bd_rate_step0_pct
    for pt in img_res.opt_points:
      assert pt.cross_decoder_psnr is not None and pt.cross_decoder_psnr >= 45.0
      jxl_b = get_field(pt, "jxl_bytes")
      if jxl_b is not None:
        info = inspect_jxl(jxl_b)
        assert info["all_default"] is False
      acs = get_field(pt, "ac_strategy", "stage2_ac_strategy")
      if acs is not None:
        assert bool(np.any(np.asarray(acs) != 0))

  assert report.mean_bd_rate_opt_pct < 0.0
  assert report.mean_bd_rate_opt_pct < report.mean_bd_rate_step0_pct


def test_scenario8_standalone_r1_hygiene_and_zero_heuristics_end_to_end_audit(
    composite_smooth_sharp_64_01: np.ndarray,
    vgg_params_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Scenario 8 (R1): Standalone repository hygiene, git ignore audit, pure-Python DequantMatrices, and runtime R1/R3 guard across all modes."""
  from jxl_wasserstein.compositing import encode_image_to_jxl
  from jxl_wasserstein.optimizer import optimize_single_frame

  # 1. Zero legacy package references
  for mod_name in list(sys.modules):
    for tok in FORBIDDEN_LEGACY_TOKENS:
      assert tok not in mod_name

  files_to_scan = [
      PKG_ROOT / "pyproject.toml",
      REPO_ROOT / "tools" / "jxl_wasserstein_helper.cc",
  ]
  for sub in ("src", "tests"):
    for p in (PKG_ROOT / sub).rglob("*"):
      if p.is_file() and "__pycache__" not in p.parts and ".venv" not in p.parts:
        files_to_scan.append(p)
  for fpath in files_to_scan:
    content = fpath.read_text(encoding="utf-8", errors="ignore")
    for tok in FORBIDDEN_LEGACY_TOKENS:
      assert tok not in content, f"R1 violation in {fpath}: found {tok!r}"

  # 2. Zero tracked files under .agents/
  ls_res = subprocess.run(
      ["git", "ls-files", ".agents/"],
      cwd=REPO_ROOT,
      capture_output=True,
      text=True,
      check=True,
  )
  assert ls_res.stdout.strip() == ""

  # 3. Runtime guard: pure-Python get_default_dequant_matrices() + all modes run without dump_init_state or cjxl
  for cache_attr in ("_DEFAULT_DEQUANT_CACHE", "_DEFAULT_INV_DEQUANT_CACHE"):
    if hasattr(fm_mod, cache_attr):
      monkeypatch.setattr(fm_mod, cache_attr, None)

  with forbid_libjxl_per_block_choices(monkeypatch):
    dq = get_default_dequant_matrices()
    assert len(dq) >= 18

    res_single = optimize_single_frame(
        composite_smooth_sharp_64_01[:32, :32],
        vgg_params_session,
        rd_lambda=0.15,
        num_steps=2,
    )
    assert len(get_field(res_single, "jxl_bytes")) > 0

    for mode in ("single", "patches", "alpha", "auto"):
      res_m = encode_image_to_jxl(
          composite_smooth_sharp_64_01[:32, :32],
          mode=mode,
          upsampling=2,
          dim_shift=1,
          distance=0.6,
          lambda_rd=0.12,
          num_steps=1,
          vgg_params=vgg_params_session,
      )
      assert len(get_field(res_m, "jxl_bytes")) > 0
