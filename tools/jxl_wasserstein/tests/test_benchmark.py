"""Tests for M4 Kodak R-D benchmark (`benchmark.py`) and CLI (`cli.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from jxl_wasserstein import benchmark, cli, cpp_bridge


def test_compute_bd_rate_analytical_shift() -> None:
  """Verifies analytical BD-rate shift properties and input validation."""
  dist = np.array([0.3, 0.7, 1.4, 2.5], dtype=np.float64)
  rate_ref = np.array([2.2, 1.4, 0.85, 0.45], dtype=np.float64)

  # 15% lower rate at identical distortions -> -15.0% BD-rate
  bdr_minus_15 = benchmark.compute_bd_rate(
      rate_ref, dist, 0.85 * rate_ref, dist
  )
  assert abs(bdr_minus_15 - (-15.0)) < 1e-4

  # Identical R-D curves -> 0.0% BD-rate
  bdr_zero = benchmark.compute_bd_rate(rate_ref, dist, rate_ref, dist)
  assert abs(bdr_zero) < 1e-6

  # 10% higher rate at identical distortions -> +10.0% BD-rate
  bdr_plus_10 = benchmark.compute_bd_rate(
      rate_ref, dist, 1.10 * rate_ref, dist
  )
  assert abs(bdr_plus_10 - 10.0) < 1e-4

  # Fewer than 4 points raises ValueError
  with pytest.raises(ValueError, match="at least 4"):
    benchmark.compute_bd_rate(rate_ref[:3], dist[:3], rate_ref[:3], dist[:3])

  # Non-overlapping distortion ranges return NaN
  dist_disjoint = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float64)
  assert np.isnan(
      benchmark.compute_bd_rate(rate_ref, dist, rate_ref, dist_disjoint)
  )


def test_cli_single_image_encode_and_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies CLI single-image encoding, cross-decoder verification, and R2 compliance."""

  def _forbid_dump_init(*args, **kwargs):
    raise AssertionError(
        "R2 violation: dump_init_state must NEVER be called by the encoder!"
    )

  monkeypatch.setattr(cpp_bridge, "dump_init_state", _forbid_dump_init)

  kodim01_path = benchmark.DEFAULT_KODAK_DIR / "kodim01.png"
  crop_u8 = benchmark.load_benchmark_image(kodim01_path, crop_size=64)
  in_png = tmp_path / "kodim01_64.png"
  out_jxl = tmp_path / "kodim01_64.jxl"
  json_path = tmp_path / "kodim01_64.json"
  Image.fromarray(crop_u8, mode="RGB").save(in_png)

  rc = cli.main(
      [
          "--input",
          str(in_png),
          "--output",
          str(out_jxl),
          "--distance",
          "2.4",
          "--steps",
          "4",
          "--verify",
          "--json-out",
          str(json_path),
      ]
  )
  assert rc == 0
  assert out_jxl.exists()
  assert out_jxl.stat().st_size > 0
  assert json_path.exists()

  report = json.loads(json_path.read_text(encoding="utf-8"))
  assert report["width"] == 64
  assert report["height"] == 64
  assert report["file_bytes"] == out_jxl.stat().st_size
  assert report["bpp"] > 0.0
  assert report["wd"] > 0.0
  assert report["cross_decoder_psnr"] is not None
  assert float(report["cross_decoder_psnr"]) >= 45.0


def test_kodak_rd_curve_negative_bd_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verifies negative BD-rate vs cjxl -e 7 and optimization improvement on Kodak."""

  def _forbid_dump_init(*args, **kwargs):
    raise AssertionError(
        "R2 violation: dump_init_state must NEVER be called by the encoder!"
    )

  monkeypatch.setattr(cpp_bridge, "dump_init_state", _forbid_dump_init)

  img_path = benchmark.DEFAULT_KODAK_DIR / "kodim08.png"
  rgb = benchmark.load_benchmark_image(img_path, crop_size=256)
  distances = [1.4, 2.4, 4.0, 6.8]

  res = benchmark.evaluate_image_rd_curve(
      rgb,
      image_name="kodim08.png",
      distances=distances,
      steps=8,
      mode="single",
      effort=7,
      verify=True,
  )

  opt_bpps = [p.bpp for p in res.opt_points]
  opt_wds = [p.wd for p in res.opt_points]

  # Monotonicity across increasing distance
  for i in range(len(distances) - 1):
    assert (
        opt_bpps[i] > opt_bpps[i + 1]
    ), f"Expected decreasing bpp with distance: {opt_bpps}"
    assert (
        opt_wds[i] < opt_wds[i + 1]
    ), f"Expected increasing WD with distance: {opt_wds}"

  # Cross-decoder PSNR check
  assert res.opt_points[0].cross_decoder_psnr is not None
  assert res.opt_points[0].cross_decoder_psnr >= 45.0

  # Strict optimization improvement over step 0 and negative BD-rate vs cjxl -e 7
  assert (
      res.bd_rate_opt_pct < res.bd_rate_step0_pct
  ), f"Expected opt BD-rate ({res.bd_rate_opt_pct:.2f}%) < step0 ({res.bd_rate_step0_pct:.2f}%)"
  assert (
      res.bd_rate_opt_pct < 0.0
  ), f"Expected negative BD-rate vs cjxl -e 7, got {res.bd_rate_opt_pct:+.2f}%"
