#!/usr/bin/env python3
"""8-bit SDR corpus Rate-WD / PSNR / SSIMULACRA2 / Butteraugli comparison runner.

Discovers strictly the 40 pure 8-bit SDR images in `image-corpus/` (filtering out
all 16-bit images and `ClassE_*` / HDR images) and compares `jxl_wasserstein`
against `cjxl -e 7` across target distances.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import jax.numpy as jnp
import numpy as np
from PIL import Image

from jxl_wasserstein.benchmark import (
    DEFAULT_CORPUS_ROOT,
    compute_bd_rate,
    discover_sdr_corpus,
    encode_cjxl_baseline,
    is_pure_8bit_sdr_image,
    load_benchmark_image,
)
from jxl_wasserstein.compositing import encode_image_to_jxl
from jxl_wasserstein.cpp_bridge import (
    DJXL_BIN,
    compute_psnr,
    decode_with_djxl,
    verify_jxl_codestream,
)
from jxl_wasserstein.vgg import load_vgg16_params
from jxl_wasserstein.wasserstein import compute_wd

BUILD_TOOLS_DIR = DJXL_BIN.parent
SSIM2_BIN = BUILD_TOOLS_DIR / "ssimulacra2"
BUTTERAUGLI_BIN = BUILD_TOOLS_DIR / "butteraugli_main"

DEFAULT_COMPARISON_DISTANCES: tuple[float, ...] = (
    0.18,
    0.38,
    0.72,
    1.30,
    2.40,
    4.20,
    8.50,
)


def measure_external_metrics(
    ref_u8: np.ndarray, dec_u8: np.ndarray
) -> dict[str, float | None]:
  """Computes SSIMULACRA2 and Butteraugli (max & 3-norm) when binaries are available."""
  ssim2: float | None = None
  ba_max: float | None = None
  ba3: float | None = None
  if not SSIM2_BIN.is_file() and not BUTTERAUGLI_BIN.is_file():
    return {"ssimulacra2": None, "ba_max": None, "ba3": None}

  with tempfile.TemporaryDirectory(prefix="jxl_wd_metrics_") as tmp:
    tdir = Path(tmp)
    ref_png = tdir / "ref.png"
    dec_png = tdir / "dec.png"
    Image.fromarray(np.asarray(ref_u8, dtype=np.uint8), mode="RGB").save(ref_png)
    Image.fromarray(np.asarray(dec_u8, dtype=np.uint8), mode="RGB").save(dec_png)

    if SSIM2_BIN.is_file():
      res_s = subprocess.run(
          [str(SSIM2_BIN), str(ref_png), str(dec_png)],
          capture_output=True,
          text=True,
          check=False,
      )
      if res_s.returncode == 0 and res_s.stdout.strip():
        try:
          ssim2 = float(res_s.stdout.strip().splitlines()[-1])
        except ValueError:
          ssim2 = None

    if BUTTERAUGLI_BIN.is_file():
      res_b = subprocess.run(
          [str(BUTTERAUGLI_BIN), str(ref_png), str(dec_png)],
          capture_output=True,
          text=True,
          check=False,
      )
      if res_b.returncode == 0 and res_b.stdout.strip():
        lines = [ln.strip() for ln in res_b.stdout.strip().splitlines() if ln.strip()]
        if lines:
          try:
            ba_max = float(lines[0])
          except ValueError:
            ba_max = None
        for ln in lines[1:]:
          if ln.startswith("3-norm:"):
            try:
              ba3 = float(ln.split(":", 1)[1].strip())
            except ValueError:
              ba3 = None
  return {"ssimulacra2": ssim2, "ba_max": ba_max, "ba3": ba3}


def run_corpus_comparison(
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
    *,
    distances: Sequence[float] = DEFAULT_COMPARISON_DISTANCES,
    mode: str = "auto",
    steps: int = 12,
    crop_size: int | None = None,
    max_images: int | None = None,
    name_filter: str | None = None,
    effort: int = 7,
    log2_sigma: float = 1.0,
    verify_decoders: bool = True,
    csv_out: str | Path | None = None,
    json_out: str | Path | None = None,
) -> dict[str, Any]:
  """Runs R-D comparison across the 8-bit SDR corpus against `cjxl -e <effort>`."""
  entries = discover_sdr_corpus(corpus_root)
  if name_filter:
    flt = name_filter.lower()
    entries = [(n, p) for (n, p) in entries if flt in n.lower()]
  if max_images is not None and max_images > 0:
    entries = entries[: int(max_images)]

  vgg_params = load_vgg16_params()
  rows: list[dict[str, Any]] = []
  per_image_bdr: list[dict[str, Any]] = []

  for img_idx, (img_label, img_path) in enumerate(entries, start=1):
    rgb_u8 = load_benchmark_image(img_path, crop_size=crop_size)
    h, w, _ = rgb_u8.shape
    ref_01 = jnp.asarray(rgb_u8, dtype=jnp.float32) / 255.0

    c_rates: list[float] = []
    c_wds: list[float] = []
    w_rates: list[float] = []
    w_wds: list[float] = []

    for d in distances:
      d_f = float(d)
      t0 = time.time()
      cb, cdec = encode_cjxl_baseline(rgb_u8, d_f, effort=effort)
      t_cjxl = time.time() - t0
      c_bpp = len(cb) * 8.0 / float(h * w)
      c_wd = float(
          compute_wd(
              ref_01,
              jnp.asarray(cdec, dtype=jnp.float32) / 255.0,
              vgg_params,
              log2_sigma=log2_sigma,
          )
      )
      c_psnr = compute_psnr(rgb_u8, cdec, peak=255.0)
      c_ext = measure_external_metrics(rgb_u8, cdec)
      c_rates.append(c_bpp)
      c_wds.append(c_wd)
      rows.append({
          "image": img_label,
          "codec": f"cjxl_e{effort}",
          "mode_selected": "cjxl",
          "distance": d_f,
          "bytes": len(cb),
          "bpp": c_bpp,
          "wd": c_wd,
          "psnr_db": c_psnr,
          "ssimulacra2": c_ext["ssimulacra2"],
          "ba3": c_ext["ba3"],
          "elapsed_s": t_cjxl,
      })

      t1 = time.time()
      res = encode_image_to_jxl(
          rgb_u8,
          distance=d_f,
          mode=mode,
          steps=steps,
          verify_decoders=verify_decoders,
          log2_sigma=log2_sigma,
          vgg_params=vgg_params,
          effort=effort,
      )
      t_wdd = time.time() - t1
      jb = res.jxl_bytes
      wdec = decode_with_djxl(jb)
      if verify_decoders:
        verify_jxl_codestream(jb, min_cross_decoder_psnr=45.0)
      w_bpp = len(jb) * 8.0 / float(h * w)
      w_wd = float(
          compute_wd(
              ref_01,
              jnp.asarray(wdec, dtype=jnp.float32) / 255.0,
              vgg_params,
              log2_sigma=log2_sigma,
          )
      )
      w_psnr = compute_psnr(rgb_u8, wdec, peak=255.0)
      w_ext = measure_external_metrics(rgb_u8, wdec)
      w_rates.append(w_bpp)
      w_wds.append(w_wd)
      rows.append({
          "image": img_label,
          "codec": "jxl_wasserstein",
          "mode_selected": res.mode_selected,
          "distance": d_f,
          "bytes": len(jb),
          "bpp": w_bpp,
          "wd": w_wd,
          "psnr_db": w_psnr,
          "ssimulacra2": w_ext["ssimulacra2"],
          "ba3": w_ext["ba3"],
          "elapsed_s": t_wdd,
      })
      print(
          f"[{img_idx}/{len(entries)}] {img_label} d={d_f:.2f}: "
          f"wdd({res.mode_selected}) bpp={w_bpp:.3f} WD={w_wd:.4f} "
          f"PSNR={w_psnr:.2f} SSIM2={w_ext['ssimulacra2']} BA3={w_ext['ba3']} | "
          f"cjxl bpp={c_bpp:.3f} WD={c_wd:.4f} PSNR={c_psnr:.2f}"
      )

    bdr = (
        compute_bd_rate(c_rates, c_wds, w_rates, w_wds)
        if len(distances) >= 4
        else float("nan")
    )
    per_image_bdr.append({"image": img_label, "bd_rate_wd_pct": bdr})

  finite_bdrs = [
      x["bd_rate_wd_pct"]
      for x in per_image_bdr
      if np.isfinite(x["bd_rate_wd_pct"])
  ]
  report = {
      "corpus_root": str(corpus_root),
      "num_images": len(entries),
      "distances": [float(d) for d in distances],
      "mode": mode,
      "steps": int(steps),
      "log2_sigma": float(log2_sigma),
      "mean_bd_rate_wd_pct": (
          float(np.mean(finite_bdrs)) if finite_bdrs else float("nan")
      ),
      "per_image": per_image_bdr,
      "rows": rows,
  }

  if csv_out is not None and rows:
    csv_path = Path(csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
      writer.writeheader()
      writer.writerows(rows)

  if json_out is not None:
    json_path = Path(json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

  return report


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Run 8-bit SDR corpus comparison for jxl_wasserstein vs cjxl."
  )
  parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
  parser.add_argument(
      "--list-only",
      action="store_true",
      default=False,
      help="List discovered pure 8-bit SDR corpus images and exit.",
  )
  parser.add_argument(
      "--distances",
      type=float,
      nargs="+",
      default=list(DEFAULT_COMPARISON_DISTANCES),
  )
  parser.add_argument(
      "--mode", choices=("single", "patches", "alpha", "auto"), default="auto"
  )
  parser.add_argument("--steps", type=int, default=12)
  parser.add_argument("--crop-size", type=int, default=None)
  parser.add_argument("--max-images", type=int, default=None)
  parser.add_argument("--filter", dest="name_filter", type=str, default=None)
  parser.add_argument("--effort", type=int, default=7)
  parser.add_argument("--log2-sigma", type=float, default=1.0)
  parser.add_argument("--csv-out", type=Path, default=None)
  parser.add_argument("--json-out", type=Path, default=None)
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  if args.list_only:
    entries = discover_sdr_corpus(args.corpus_root)
    for name, path in entries:
      print(f"{name}\t{path}")
    print(f"Total pure 8-bit SDR images: {len(entries)}")
    return 0

  run_corpus_comparison(
      corpus_root=args.corpus_root,
      distances=args.distances,
      mode=args.mode,
      steps=args.steps,
      crop_size=args.crop_size,
      max_images=args.max_images,
      name_filter=args.name_filter,
      effort=args.effort,
      log2_sigma=args.log2_sigma,
      csv_out=args.csv_out,
      json_out=args.json_out,
  )
  return 0


__all__ = [
    "DEFAULT_COMPARISON_DISTANCES",
    "discover_sdr_corpus",
    "is_pure_8bit_sdr_image",
    "measure_external_metrics",
    "run_corpus_comparison",
    "main",
]


if __name__ == "__main__":
  raise SystemExit(main())
