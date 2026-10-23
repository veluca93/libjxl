"""Command-line interface (`jxl-wasserstein`) for single-image encoding and Kodak benchmarking."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from PIL import Image

from jxl_wasserstein.benchmark import (
    DEFAULT_BENCHMARK_DISTANCES,
    DEFAULT_KODAK_DIR,
    DEFAULT_KODAK_SUBSET,
    format_benchmark_table,
    load_benchmark_image,
    run_kodak_benchmark,
)
from jxl_wasserstein.cpp_bridge import (
    PATCH_BLEND_ADD,
    PATCH_BLEND_REPLACE,
    compute_psnr,
    decode_with_djxl,
    inspect_jxl,
    verify_jxl_codestream,
)
from jxl_wasserstein.optimizer import (
    init_pure_jax_wd_state,
    optimize_rate_wd_point,
)
from jxl_wasserstein.vgg import (
    load_vgg16_params,
    multiscale_vgg_features,
    select_num_scales,
)
from jxl_wasserstein.wasserstein import compute_wd


def _parse_upsampling(val: str | int) -> int | str:
  s = str(val).strip().lower()
  if s == "auto":
    return "auto"
  u = int(s)
  if u not in (1, 2, 4, 8):
    raise argparse.ArgumentTypeError(
        f"--upsampling must be one of auto, 1, 2, 4, 8 (got {val!r})"
    )
  return u


def _parse_dim_shift(val: str | int) -> int | str:
  s = str(val).strip().lower()
  if s == "auto":
    return "auto"
  ds = int(s)
  if ds not in (0, 1, 2, 3):
    raise argparse.ArgumentTypeError(
        f"--dim-shift must be one of auto, 0, 1, 2, 3 (got {val!r})"
    )
  return ds


def _parse_patch_blend_mode(val: str | int) -> int | str:
  s = str(val).strip().lower()
  if s == "auto":
    return "auto"
  if s in ("replace", "1"):
    return PATCH_BLEND_REPLACE
  if s in ("add", "2"):
    return PATCH_BLEND_ADD
  raise argparse.ArgumentTypeError(
      f"--patch-blend-mode must be auto, replace, add, 1, or 2 (got {val!r})"
  )


def _parse_custom_dequant(val: str | bool) -> str | bool:
  if isinstance(val, bool):
    return val
  s = str(val).strip().lower()
  if s in ("auto", "none", "compact", "full_bands"):
    return s
  if s in ("true", "1", "yes", "on"):
    return True
  if s in ("false", "0", "no", "off"):
    return False
  raise argparse.ArgumentTypeError(
      "--custom-dequant must be auto, none, compact, full_bands, true, or"
      f" false (got {val!r})"
  )


def build_parser() -> argparse.ArgumentParser:
  """Builds the argument parser for the `jxl-wasserstein` CLI."""
  parser = argparse.ArgumentParser(
      prog="jxl-wasserstein",
      description=(
          "Gradient-based JPEG XL Wasserstein Distortion encoder and Kodak"
          " R-D benchmark runner."
      ),
  )
  parser.add_argument(
      "--input",
      "-i",
      type=Path,
      default=None,
      help="Input RGB image path (required unless --benchmark-kodak is set).",
  )
  parser.add_argument(
      "--output",
      "-o",
      type=Path,
      default=None,
      help="Output .jxl bitstream file path.",
  )
  parser.add_argument(
      "--distance",
      "-d",
      type=float,
      default=1.5,
      help="Target Butteraugli-like quantization distance d > 0 (default: 1.5).",
  )
  parser.add_argument(
      "--lambda",
      "--lam",
      "--rd-lambda",
      dest="lambda_rate",
      type=float,
      default=None,
      help="Explicit Rate-WD trade-off lambda > 0 (default: calibrated from -d).",
  )
  parser.add_argument(
      "--mode",
      choices=("single", "patches", "alpha", "auto"),
      default="single",
      help="Encoding topology mode (default: single).",
  )
  parser.add_argument(
      "--upsampling",
      type=_parse_upsampling,
      default=2,
      help="Reference frame upsampling factor: auto, 1, 2, 4, or 8 (default: 2).",
  )
  parser.add_argument(
      "--dim-shift",
      type=_parse_dim_shift,
      default=0,
      help="Alpha channel downsampling shift: auto, 0, 1, 2, or 3 (default: 0).",
  )
  parser.add_argument(
      "--patch-blend-mode",
      type=_parse_patch_blend_mode,
      default="replace",
      help="Patch blend mode for patches mode: auto, replace (1), or add (2).",
  )
  parser.add_argument(
      "--enable-noise",
      "--noise",
      dest="enable_noise",
      action="store_true",
      default=None,
      help=(
          "Enable differentiable JXL NoiseParams LUT synthesis (default: auto"
          " in --mode auto, disabled otherwise)."
      ),
  )
  parser.add_argument(
      "--no-noise",
      dest="enable_noise",
      action="store_false",
      help="Explicitly disable JXL NoiseParams LUT synthesis.",
  )
  parser.add_argument(
      "--custom-dequant",
      dest="custom_dequant",
      type=_parse_custom_dequant,
      default=None,
      help=(
          "Custom DequantMatrices parameterization: auto, none, compact,"
          " full_bands, true, or false (default: auto)."
      ),
  )
  parser.add_argument(
      "--no-custom-dequant",
      dest="custom_dequant",
      action="store_const",
      const="none",
      help="Disable custom DequantMatrices (use 1-bit default weights).",
  )
  parser.add_argument(
      "--log2-sigma",
      type=float,
      default=1.0,
      help="Log2 Gaussian window sigma for Wasserstein Distortion (default: 1.0).",
  )
  parser.add_argument(
      "--init-mode",
      choices=("dct", "zero", "random"),
      default="dct",
      help="Coefficient initialization mode: dct, zero, or random (default: dct).",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="Random seed when --init-mode=random (default: 0).",
  )
  parser.add_argument(
      "--steps",
      "--num-steps",
      dest="steps",
      type=int,
      default=None,
      help="Number of Adam+STE optimization steps (default: 8 for dct, 40 for zero/random).",
  )
  parser.add_argument(
      "--lr",
      type=float,
      default=0.055,
      help="Adam learning rate for Rate-WD optimization (default: 0.055).",
  )
  parser.add_argument(
      "--crop-size",
      type=int,
      default=None,
      help="Optional center crop size in pixels.",
  )
  parser.add_argument(
      "--verify",
      action="store_true",
      default=False,
      help="Verify emitted .jxl bitstream with djxl, jxlinfo, and jxl_cli.",
  )
  parser.add_argument(
      "--json-out",
      type=Path,
      default=None,
      help="Path to write JSON encoding or benchmark report.",
  )
  parser.add_argument(
      "--benchmark-kodak",
      action="store_true",
      default=False,
      help="Run Kodak Rate-WD curve and BD-rate benchmark against cjxl -e 7.",
  )
  parser.add_argument(
      "--kodak-dir",
      type=Path,
      default=DEFAULT_KODAK_DIR,
      help="Directory containing Kodak PNG images.",
  )
  parser.add_argument(
      "--max-images",
      type=int,
      default=4,
      help="Maximum number of Kodak images to evaluate (default: 4).",
  )
  parser.add_argument(
      "--distances",
      type=float,
      nargs="+",
      default=list(DEFAULT_BENCHMARK_DISTANCES),
      help="List of >= 4 distances for Kodak benchmark.",
  )
  return parser


def _encode_single_image_cli(args: argparse.Namespace) -> dict[str, Any]:
  """Executes single-image encoding according to parsed CLI flags."""
  if args.input is None:
    raise ValueError("--input / -i is required when --benchmark-kodak is not set.")
  in_path = Path(args.input)
  if not in_path.exists():
    raise FileNotFoundError(f"Input image not found: {in_path}")

  if args.crop_size is not None and args.crop_size > 0:
    rgb_u8 = load_benchmark_image(in_path, crop_size=args.crop_size)
  else:
    rgb_u8 = np.asarray(Image.open(in_path).convert("RGB"), dtype=np.uint8)

  from jxl_wasserstein.compositing import (
      calibrate_rd_lambda,
      encode_image_to_jxl,
  )

  h, w, _ = rgb_u8.shape
  distance = float(args.distance)
  explicit_lam = (
      float(args.lambda_rate) if args.lambda_rate is not None else None
  )
  lam_rate = (
      explicit_lam
      if explicit_lam is not None
      else calibrate_rd_lambda(distance)
  )
  log2_sigma = float(getattr(args, "log2_sigma", 1.0))
  custom_dequant = getattr(args, "custom_dequant", None)
  mode = str(args.mode).lower()
  upsampling = args.upsampling
  if upsampling == "auto":
    u_val: int | str = "auto"
  elif int(upsampling) == 1:
    u_val = 2
  else:
    u_val = int(upsampling)

  dim_shift = args.dim_shift
  ds_val: int | str = (
      "auto" if str(dim_shift).lower() == "auto" else int(dim_shift)
  )
  init_mode = str(getattr(args, "init_mode", "dct")).strip().lower()
  seed = int(getattr(args, "seed", 0))
  eff_steps = (
      int(args.steps)
      if args.steps is not None
      else (40 if init_mode in ("zero", "random") else 8)
  )

  vgg_params = load_vgg16_params()
  ref_01 = jnp.asarray(rgb_u8, dtype=jnp.float32) / 255.0

  if (
      mode == "single"
      and not bool(args.enable_noise)
      and custom_dequant in (None, True)
  ):
    pad_h = ((h + 7) // 8) * 8 - h
    pad_w = ((w + 7) // 8) * 8 - w
    ref_pad = (
        jnp.pad(ref_01, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        if (pad_h > 0 or pad_w > 0)
        else ref_01
    )
    feats_ref = tuple(
        multiscale_vgg_features(
            vgg_params,
            jnp.moveaxis(ref_pad, -1, 0),
            num_scales=select_num_scales(ref_pad.shape[0], ref_pad.shape[1]),
        )
    )
    st, topo, u_ac, u_dc = init_pure_jax_wd_state(
        rgb_u8,
        distance,
        vgg_params=vgg_params,
        lambda_rate=lam_rate,
        init_mode=init_mode,
        seed=seed,
    )
    details = optimize_rate_wd_point(
        rgb_u8,
        st,
        topo,
        u_ac,
        u_dc,
        feats_ref,
        vgg_params,
        distance,
        steps=eff_steps,
        verify_decoder=bool(args.verify),
        lr=float(args.lr),
        lambda_rate=lam_rate,
        return_details=True,
        init_mode=init_mode,
        seed=seed,
    )
    assert isinstance(details, dict)
    jxl_bytes = details["best_jb"]
    bpp = float(details["best_bpp"])
    wd = float(details["best_wd"])
    psnr_db = float(details["best_psnr"])
    mode_selected = "single"
    u_selected = 1
    ds_selected = 0
    has_noise = False
  else:
    res = encode_image_to_jxl(
        srgb_hwc=rgb_u8,
        mode=mode,
        upsampling=u_val,
        patch_blend_mode=args.patch_blend_mode,
        dim_shift=ds_val,
        distance=distance,
        lambda_rd=explicit_lam,
        num_steps=eff_steps,
        enable_noise=args.enable_noise,
        custom_dequant=custom_dequant,
        log2_sigma=log2_sigma,
        vgg_params=vgg_params,
        init_mode=init_mode,
        seed=seed,
    )
    jxl_bytes = res["jxl_bytes"]
    dec_u8 = decode_with_djxl(jxl_bytes)
    bpp = len(jxl_bytes) * 8.0 / float(h * w)
    wd = float(
        compute_wd(
            ref_01,
            jnp.asarray(dec_u8, dtype=jnp.float32) / 255.0,
            vgg_params,
            log2_sigma=log2_sigma,
        )
    )
    psnr_db = compute_psnr(rgb_u8, dec_u8, peak=255.0)
    mode_selected = str(res.get("mode", mode))
    u_selected = int(
        res.get(
            "upsampling",
            1
            if mode_selected == "single"
            else (2 if u_val == "auto" else int(u_val)),
        )
    )
    ds_selected = int(
        res.get(
            "dim_shift",
            (0 if ds_val == "auto" else int(ds_val))
            if mode_selected == "alpha"
            else 0,
        )
    )
    has_noise = bool(res.get("has_noise", bool(args.enable_noise)))

  if args.output is not None:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(jxl_bytes)

  verify_info: dict[str, Any] | None = None
  cross_psnr: float | None = None
  if args.verify:
    verify_info = verify_jxl_codestream(jxl_bytes)
    cross_psnr = float(verify_info["cross_decoder_psnr"])

  header_info = inspect_jxl(jxl_bytes)
  summary: dict[str, Any] = {
      "input": str(in_path),
      "output": str(args.output) if args.output is not None else None,
      "width": int(w),
      "height": int(h),
      "distance": distance,
      "lambda_rate": lam_rate,
      "steps": eff_steps,
      "init_mode": init_mode,
      "seed": seed,
      "mode": mode_selected,
      "upsampling": u_selected,
      "dim_shift": ds_selected,
      "enable_noise": has_noise,
      "file_bytes": len(jxl_bytes),
      "bpp": bpp,
      "wd": wd,
      "psnr_db": psnr_db,
      "cross_decoder_psnr": cross_psnr,
      "header_info": header_info,
  }
  if args.json_out is not None:
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  return summary


def main(argv: Sequence[str] | None = None) -> int:
  """Main CLI entry point for `jxl-wasserstein`."""
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.benchmark_kodak:
    max_imgs = int(args.max_images) if args.max_images is not None else None
    images = (
        DEFAULT_KODAK_SUBSET
        if (max_imgs is not None and max_imgs <= len(DEFAULT_KODAK_SUBSET))
        else None
    )
    eff_bench_steps = int(args.steps) if args.steps is not None else 8
    report = run_kodak_benchmark(
        kodak_dir=args.kodak_dir,
        images=images,
        max_images=max_imgs,
        crop_size=args.crop_size if args.crop_size is not None else 256,
        distances=args.distances,
        steps=eff_bench_steps,
        mode=args.mode,
        enable_noise=args.enable_noise,
        verify=args.verify,
        json_out=args.json_out,
    )
    print(format_benchmark_table(report))
    return 0

  if args.input is None or (args.output is None and args.json_out is None):
    parser.error(
        "Single-image mode requires --input (-i) and --output (-o) (or --json-out)."
    )

  summary = _encode_single_image_cli(args)
  c_str = (
      f", cross_decoder_psnr={summary['cross_decoder_psnr']:.2f} dB"
      if summary["cross_decoder_psnr"] is not None
      else ""
  )
  print(
      f"[jxl-wasserstein] mode={summary['mode']} "
      f"bytes={summary['file_bytes']} bpp={summary['bpp']:.4f} "
      f"WD={summary['wd']:.4f} PSNR={summary['psnr_db']:.2f} dB{c_str}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
