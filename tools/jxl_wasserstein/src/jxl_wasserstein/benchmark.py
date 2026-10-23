"""Kodak R-D benchmark runner and Bjontegaard Delta Rate (BD-rate) evaluator vs cjxl -e 7.

Implements:
- `compute_bd_rate`: Cubic polynomial Bjontegaard Delta Rate (%) over
  (Wasserstein Distortion, log(bitrate)). Negative values indicate bitrate
  reduction by the test encoder at equal Wasserstein Distortion.
- `encode_cjxl_baseline`: External reference baseline wrapper (`cjxl -d <d> -e 7`),
  used strictly as the reference competitor in BD-rate evaluation.
- Re-exports `init_pure_jax_wd_state`, `_single_compile_1x1_loss_and_grad`, and
  `optimize_rate_wd_point` from `jxl_wasserstein.optimizer` (3-stage Rate-WD
  pipeline with Stage 1 exploration, Stage 2 AcStrategy DP, and Stage 3 custom
  DequantMatrices refinement).
- `load_benchmark_image`, `evaluate_image_rd_curve`, `run_kodak_benchmark`,
  `run_benchmark`, and `format_benchmark_table`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import jax.numpy as jnp
import numpy as np
from PIL import Image

from jxl_wasserstein.cpp_bridge import (
    CJXL_BIN,
    compute_psnr,
    decode_with_djxl,
    inspect_jxl,
    verify_jxl_codestream,
)
from jxl_wasserstein.entropy_proxy import get_canonical_ac_scan_tables
from jxl_wasserstein.optimizer import (
    _single_compile_1x1_loss_and_grad,
    init_pure_jax_wd_state,
    optimize_rate_wd_point,
)
from jxl_wasserstein.vgg import (
    load_vgg16_params,
    multiscale_vgg_features,
    select_num_scales,
)
from jxl_wasserstein.wasserstein import compute_wd

_CANDIDATE_ROOTS = (
    Path("/usr/local/google/home/veluca/libjxl/image-corpus"),
    Path.home() / "image-corpus",
    Path("/home/luca/image-corpus"),
)
DEFAULT_CORPUS_ROOT = next((p for p in _CANDIDATE_ROOTS if p.is_dir()), _CANDIDATE_ROOTS[0])
DEFAULT_KODAK_DIR = (
    DEFAULT_CORPUS_ROOT / "qoi_benchmark_suite" / "photo_kodak"
)
DEFAULT_BENCHMARK_DISTANCES: tuple[float, ...] = (1.4, 2.4, 4.0, 6.8)
DEFAULT_KODAK_SUBSET: tuple[str, ...] = (
    "kodim01.png",
    "kodim04.png",
    "kodim08.png",
    "kodim15.png",
)

# Pre-warm canonical scan table cache outside @jax.jit so module-level caches
# hold concrete arrays rather than JAX tracers.
for _cy, _cx in ((1, 1), (1, 2), (2, 2), (1, 4), (2, 4), (4, 4)):
  get_canonical_ac_scan_tables(_cy, _cx)


@dataclass
class RDPoint:
  """Single Rate-Wasserstein Distortion operating point."""

  distance: float
  bpp: float
  wd: float
  psnr_db: float
  num_bytes: int
  elapsed_s: float = 0.0
  mode_used: str = "single"
  cross_decoder_psnr: float | None = None
  jxl_bytes: bytes | None = None
  ac_strategy: np.ndarray | None = None
  stage2_ac_strategy: np.ndarray | None = None
  all_default: bool | None = None

  def to_dict(self) -> dict[str, Any]:
    d: dict[str, Any] = {
        "distance": float(self.distance),
        "bpp": float(self.bpp),
        "wd": float(self.wd),
        "psnr_db": float(self.psnr_db),
        "num_bytes": int(self.num_bytes),
        "elapsed_s": float(self.elapsed_s),
        "mode_used": str(self.mode_used),
        "cross_decoder_psnr": (
            float(self.cross_decoder_psnr)
            if self.cross_decoder_psnr is not None
            else None
        ),
        "all_default": self.all_default,
    }
    if self.ac_strategy is not None:
      d["ac_strategy"] = np.asarray(self.ac_strategy).tolist()
    return d


@dataclass
class ImageRDResult:
  """Per-image Rate-WD curve comparison against `cjxl -e 7`."""

  image_name: str
  width: int
  height: int
  cjxl_points: list[RDPoint]
  step0_points: list[RDPoint]
  opt_points: list[RDPoint]
  bd_rate_step0_pct: float
  bd_rate_opt_pct: float
  bd_rate_improvement_pct: float

  def to_dict(self) -> dict[str, Any]:
    return {
        "image_name": str(self.image_name),
        "width": int(self.width),
        "height": int(self.height),
        "cjxl_points": [p.to_dict() for p in self.cjxl_points],
        "step0_points": [p.to_dict() for p in self.step0_points],
        "opt_points": [p.to_dict() for p in self.opt_points],
        "bd_rate_step0_pct": float(self.bd_rate_step0_pct),
        "bd_rate_opt_pct": float(self.bd_rate_opt_pct),
        "bd_rate_improvement_pct": float(self.bd_rate_improvement_pct),
    }


@dataclass
class KodakBenchmarkReport:
  """Aggregate Kodak R-D benchmark report."""

  corpus_dir: str
  crop_size: int | None
  distances: list[float]
  steps: int
  mode: str
  effort: int
  per_image: list[ImageRDResult]
  mean_bd_rate_step0_pct: float
  mean_bd_rate_opt_pct: float
  mean_bd_rate_improvement_pct: float
  median_bd_rate_opt_pct: float
  total_elapsed_s: float

  def to_dict(self) -> dict[str, Any]:
    return {
        "corpus_dir": str(self.corpus_dir),
        "crop_size": int(self.crop_size) if self.crop_size is not None else None,
        "distances": [float(d) for d in self.distances],
        "steps": int(self.steps),
        "mode": str(self.mode),
        "effort": int(self.effort),
        "per_image": [r.to_dict() for r in self.per_image],
        "mean_bd_rate_step0_pct": float(self.mean_bd_rate_step0_pct),
        "mean_bd_rate_opt_pct": float(self.mean_bd_rate_opt_pct),
        "mean_bd_rate_improvement_pct": float(self.mean_bd_rate_improvement_pct),
        "median_bd_rate_opt_pct": float(self.median_bd_rate_opt_pct),
        "total_elapsed_s": float(self.total_elapsed_s),
    }


def compute_bd_rate(
    rate_ref: Sequence[float] | np.ndarray,
    dist_ref: Sequence[float] | np.ndarray,
    rate_test: Sequence[float] | np.ndarray,
    dist_test: Sequence[float] | np.ndarray,
) -> float:
  """Bjontegaard Delta rate (BD-rate %) using cubic polynomial fit in (WD, log(rate)).

  Negative value = bitrate reduction by test codec at equal Wasserstein Distortion.
  """
  r1 = np.asarray(rate_ref, dtype=np.float64).reshape(-1)
  d1 = np.asarray(dist_ref, dtype=np.float64).reshape(-1)
  r2 = np.asarray(rate_test, dtype=np.float64).reshape(-1)
  d2 = np.asarray(dist_test, dtype=np.float64).reshape(-1)
  if r1.size != d1.size or r2.size != d2.size:
    raise ValueError("Rate and distortion arrays must have matching lengths.")
  if r1.size < 4 or r2.size < 4:
    raise ValueError("compute_bd_rate requires at least 4 R-D points per curve.")
  if not (
      np.all(np.isfinite(r1))
      and np.all(np.isfinite(d1))
      and np.all(np.isfinite(r2))
      and np.all(np.isfinite(d2))
  ):
    raise ValueError("Rate and distortion arrays must be finite.")
  if np.any(r1 <= 0.0) or np.any(r2 <= 0.0):
    raise ValueError("Rates must be strictly positive.")

  o1 = np.argsort(d1)
  o2 = np.argsort(d2)
  d1, lr1 = d1[o1], np.log(np.maximum(r1[o1], 1e-12))
  d2, lr2 = d2[o2], np.log(np.maximum(r2[o2], 1e-12))
  lr1 = np.minimum.accumulate(lr1)
  lr2 = np.minimum.accumulate(lr2)

  d_min = max(float(d1[0]), float(d2[0]))
  d_max = min(float(d1[-1]), float(d2[-1]))
  if d_max <= d_min:
    return float("nan")

  mid = 0.5 * (d_max + d_min)
  half = 0.5 * (d_max - d_min)
  t1 = (d1 - mid) / half
  t2 = (d2 - mid) / half
  p1 = np.polyfit(t1, lr1, deg=3)
  p2 = np.polyfit(t2, lr2, deg=3)
  avg1 = float(p1[3] + p1[1] / 3.0)
  avg2 = float(p2[3] + p2[1] / 3.0)
  return float(math.expm1(avg2 - avg1) * 100.0)


bd_rate = compute_bd_rate


def _eval_bdr_flexible(
    r_ref: list[float],
    d_ref: list[float],
    r_tst: list[float],
    d_tst: list[float],
) -> float:
  """Evaluates BD-rate, interpolating to 4 log-linear points when 2 <= N < 4."""
  if len(r_ref) >= 4 and len(r_tst) >= 4:
    return compute_bd_rate(r_ref, d_ref, r_tst, d_tst)
  if len(r_ref) < 2 or len(r_tst) < 2:
    return float("nan")
  r1 = np.asarray(r_ref, dtype=np.float64)
  d1 = np.asarray(d_ref, dtype=np.float64)
  r2 = np.asarray(r_tst, dtype=np.float64)
  d2 = np.asarray(d_tst, dtype=np.float64)
  t1 = np.linspace(0.0, 1.0, r1.size)
  t2 = np.linspace(0.0, 1.0, r2.size)
  t4 = np.linspace(0.0, 1.0, 4)
  r1_4 = np.exp(np.interp(t4, t1, np.log(np.maximum(r1, 1e-12))))
  d1_4 = np.interp(t4, t1, d1)
  r2_4 = np.exp(np.interp(t4, t2, np.log(np.maximum(r2, 1e-12))))
  d2_4 = np.interp(t4, t2, d2)
  return compute_bd_rate(r1_4, d1_4, r2_4, d2_4)


def encode_cjxl_baseline(
    rgb_u8: np.ndarray, distance: float, effort: int = 7
) -> tuple[bytes, np.ndarray]:
  """Encodes rgb_u8 with external baseline `cjxl -d <distance> -e <effort>`.

  Strictly used ONLY as the reference competitor in BD-rate benchmarking.
  """
  with tempfile.TemporaryDirectory(prefix="cjxl_base_") as tmpdir:
    tmp = Path(tmpdir)
    in_png = tmp / "in.png"
    out_jxl = tmp / "out.jxl"
    Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8), mode="RGB").save(in_png)
    cmd = [
        str(CJXL_BIN),
        str(in_png),
        str(out_jxl),
        "-d",
        f"{float(distance):.6g}",
        "-e",
        str(int(effort)),
        "--quiet",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    jxl_bytes = out_jxl.read_bytes()
    dec_u8 = decode_with_djxl(jxl_bytes)
    return jxl_bytes, dec_u8


def is_pure_8bit_sdr_image(path: str | Path) -> bool:
  """Returns True iff `path` is a pure 8-bit SDR image (excluding >8-bit and ClassE/HDR images)."""
  p = Path(path)
  if not p.is_file():
    return False
  parent_low = p.parent.name.lower()
  if "16" in parent_low:
    return False
  name = p.name
  name_low = name.lower()
  if (
      name.startswith("ClassE_")
      or name_low.startswith("classe_")
      or name_low.startswith("hdr.")
      or "_hdr" in name_low
      or "_10bit_" in name_low
      or "_12bit_" in name_low
      or "_16bit_" in name_low
      or "_10b_" in name_low
      or "_12b_" in name_low
      or "_16b_" in name_low
  ):
    return False
  suffix = p.suffix.lower()
  if suffix == ".png":
    try:
      with p.open("rb") as f:
        hdr = f.read(29)
      if len(hdr) < 26 or hdr[:8] != b"\x89PNG\r\n\x1a\n" or hdr[12:16] != b"IHDR":
        return False
      bit_depth = int(hdr[24])
      return bit_depth == 8
    except OSError:
      return False
  if suffix in (".ppm", ".pgm"):
    try:
      with p.open("rb") as f:
        tokens: list[bytes] = []
        while len(tokens) < 4:
          line = f.readline()
          if not line:
            break
          line = line.split(b"#", 1)[0].strip()
          if line:
            tokens.extend(line.split())
      if len(tokens) < 4:
        return False
      maxval = int(tokens[3])
      return 0 < maxval <= 255
    except (OSError, ValueError):
      return False
  if suffix in (".jpg", ".jpeg"):
    return True
  return False


def discover_sdr_corpus(
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
) -> list[tuple[str, Path]]:
  """Discovers the 40 pure 8-bit SDR corpus images, filtering out all 16-bit and ClassE_*/HDR images."""
  root = Path(corpus_root)
  entries: list[tuple[str, Path]] = []

  info_dir = root / "imagecompression.info.rgb8"
  if info_dir.is_dir():
    for p in sorted(info_dir.iterdir()):
      if is_pure_8bit_sdr_image(p):
        entries.append((f"info_{p.name}", p))

  jxl_dir = root / "jpeg_xl_png"
  if jxl_dir.is_dir():
    for p in sorted(jxl_dir.iterdir()):
      if is_pure_8bit_sdr_image(p):
        entries.append((p.name, p))

  return entries


def load_benchmark_image(
    path: str | Path, crop_size: int | None = 256
) -> np.ndarray:
  """Loads an RGB image as uint8 (H, W, 3) and optionally extracts a centered crop."""
  rgb_full = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
  if crop_size is None or crop_size <= 0:
    return rgb_full.copy()
  h, w, _ = rgb_full.shape
  cs_h = min(int(crop_size), h)
  cs_w = min(int(crop_size), w)
  y0 = (h - cs_h) // 2
  x0 = (w - cs_w) // 2
  return rgb_full[y0 : y0 + cs_h, x0 : x0 + cs_w].copy()


def evaluate_image_rd_curve(
    rgb_u8: np.ndarray,
    image_name: str = "image",
    distances: Sequence[float] = DEFAULT_BENCHMARK_DISTANCES,
    *,
    steps: int = 8,
    num_steps: int | None = None,
    mode: str = "single",
    effort: int = 7,
    enable_noise: bool = False,
    verify: bool = False,
    verify_decoders: bool | None = None,
    verify_all_points: bool = True,
    vgg_params: Any = None,
) -> ImageRDResult:
  """Evaluates cjxl baseline, pure-JAX step-0, and optimized Rate-WD curves on one image."""
  eff_steps = int(steps if num_steps is None else num_steps)
  eff_verify = bool(verify if verify_decoders is None else verify_decoders)
  rgb = np.asarray(rgb_u8, dtype=np.uint8)
  h, w, _ = rgb.shape
  if vgg_params is None:
    vgg_params = load_vgg16_params()

  ref_01 = jnp.asarray(rgb, dtype=jnp.float32) / 255.0
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

  cjxl_points: list[RDPoint] = []
  step0_points: list[RDPoint] = []
  opt_points: list[RDPoint] = []

  for idx_d, d in enumerate(distances):
    d_f = float(d)
    t_c0 = time.time()
    cb, cdec = encode_cjxl_baseline(rgb, d_f, effort=effort)
    t_cjxl = time.time() - t_c0
    c_bpp = len(cb) * 8.0 / float(h * w)
    c_wd = float(
        compute_wd(
            ref_01, jnp.asarray(cdec, dtype=jnp.float32) / 255.0, vgg_params
        )
    )
    c_psnr = compute_psnr(rgb, cdec, peak=255.0)
    cjxl_points.append(
        RDPoint(
            distance=d_f,
            bpp=c_bpp,
            wd=c_wd,
            psnr_db=c_psnr,
            num_bytes=len(cb),
            elapsed_s=t_cjxl,
            mode_used="cjxl",
            cross_decoder_psnr=None,
            jxl_bytes=cb,
        )
    )

    do_verify = bool(eff_verify and (verify_all_points or idx_d == 0))
    if mode.lower() == "single" and not enable_noise:
      st, topo, u_ac, u_dc = init_pure_jax_wd_state(
          rgb, d_f, vgg_params=vgg_params
      )
      details = optimize_rate_wd_point(
          rgb,
          st,
          topo,
          u_ac,
          u_dc,
          feats_ref,
          vgg_params,
          d_f,
          steps=eff_steps,
          verify_decoder=do_verify,
          return_details=True,
      )
      assert isinstance(details, dict)
      acs_arr = np.asarray(details["ac_strategy"], dtype=np.uint8)
      step0_points.append(
          RDPoint(
              distance=d_f,
              bpp=float(details["bpp0"]),
              wd=float(details["wd0"]),
              psnr_db=float(details["psnr0"]),
              num_bytes=len(details["jb0"]),
              elapsed_s=float(details["elapsed0_s"]),
              mode_used="single",
              cross_decoder_psnr=None,
              jxl_bytes=details["jb0"],
              ac_strategy=acs_arr,
              stage2_ac_strategy=acs_arr,
              all_default=bool(details.get("all_default", False)),
          )
      )
      opt_points.append(
          RDPoint(
              distance=d_f,
              bpp=float(details["best_bpp"]),
              wd=float(details["best_wd"]),
              psnr_db=float(details["best_psnr"]),
              num_bytes=len(details["best_jb"]),
              elapsed_s=float(details["elapsed_opt_s"]),
              mode_used="single",
              cross_decoder_psnr=details["cross_psnr"],
              jxl_bytes=details["best_jb"],
              ac_strategy=acs_arr,
              stage2_ac_strategy=acs_arr,
              all_default=bool(details.get("all_default", False)),
          )
      )
    else:
      from jxl_wasserstein.compositing import (
          calibrate_rd_lambda,
          encode_image_to_jxl,
      )

      lam_f = calibrate_rd_lambda(d_f)
      t0 = time.time()
      res0 = encode_image_to_jxl(
          rgb,
          mode=mode,
          distance=d_f,
          lambda_rd=lam_f,
          num_steps=0,
          enable_noise=enable_noise,
          vgg_params=vgg_params,
          effort=effort,
      )
      t_s0 = time.time() - t0
      dec0 = decode_with_djxl(res0["jxl_bytes"])
      wd0 = float(
          compute_wd(
              ref_01, jnp.asarray(dec0, dtype=jnp.float32) / 255.0, vgg_params
          )
      )
      step0_points.append(
          RDPoint(
              distance=d_f,
              bpp=len(res0["jxl_bytes"]) * 8.0 / float(h * w),
              wd=wd0,
              psnr_db=compute_psnr(rgb, dec0, peak=255.0),
              num_bytes=len(res0["jxl_bytes"]),
              elapsed_s=t_s0,
              mode_used=str(res0.get("mode", mode)),
              cross_decoder_psnr=None,
              jxl_bytes=res0["jxl_bytes"],
          )
      )

      t1 = time.time()
      res_opt = encode_image_to_jxl(
          rgb,
          mode=mode,
          distance=d_f,
          lambda_rd=lam_f,
          num_steps=max(1, eff_steps),
          enable_noise=enable_noise,
          vgg_params=vgg_params,
          effort=effort,
      )
      t_s1 = time.time() - t1
      jb_opt = res_opt["jxl_bytes"]
      dec_opt = decode_with_djxl(jb_opt)
      wd_opt = float(
          compute_wd(
              ref_01,
              jnp.asarray(dec_opt, dtype=jnp.float32) / 255.0,
              vgg_params,
          )
      )
      cpsnr = (
          float(verify_jxl_codestream(jb_opt)["cross_decoder_psnr"])
          if do_verify
          else None
      )
      info_opt = inspect_jxl(jb_opt)
      st_opt = res_opt.get("state", res_opt.get("main_state"))
      acs_opt = (
          np.asarray(st_opt.ac_strategy, dtype=np.uint8)
          if st_opt is not None and hasattr(st_opt, "ac_strategy")
          else None
      )
      opt_points.append(
          RDPoint(
              distance=d_f,
              bpp=len(jb_opt) * 8.0 / float(h * w),
              wd=wd_opt,
              psnr_db=compute_psnr(rgb, dec_opt, peak=255.0),
              num_bytes=len(jb_opt),
              elapsed_s=t_s1,
              mode_used=str(res_opt.get("mode", mode)),
              cross_decoder_psnr=cpsnr,
              jxl_bytes=jb_opt,
              ac_strategy=acs_opt,
              stage2_ac_strategy=acs_opt,
              all_default=bool(info_opt.get("all_default", False)),
          )
      )

  c_bpps = [p.bpp for p in cjxl_points]
  c_wds = [p.wd for p in cjxl_points]
  s0_bpps = [p.bpp for p in step0_points]
  s0_wds = [p.wd for p in step0_points]
  opt_bpps = [p.bpp for p in opt_points]
  opt_wds = [p.wd for p in opt_points]

  bdr0 = _eval_bdr_flexible(c_bpps, c_wds, s0_bpps, s0_wds)
  bdr_opt = _eval_bdr_flexible(c_bpps, c_wds, opt_bpps, opt_wds)
  return ImageRDResult(
      image_name=image_name,
      width=w,
      height=h,
      cjxl_points=cjxl_points,
      step0_points=step0_points,
      opt_points=opt_points,
      bd_rate_step0_pct=bdr0,
      bd_rate_opt_pct=bdr_opt,
      bd_rate_improvement_pct=bdr_opt - bdr0,
  )


def run_kodak_benchmark(
    kodak_dir: str | Path = DEFAULT_KODAK_DIR,
    *,
    images: Sequence[str] | None = DEFAULT_KODAK_SUBSET,
    max_images: int | None = None,
    crop_size: int | None = 256,
    max_dim: int | None = None,
    distances: Sequence[float] = DEFAULT_BENCHMARK_DISTANCES,
    steps: int = 8,
    num_steps: int | None = None,
    mode: str = "single",
    effort: int = 7,
    enable_noise: bool = False,
    verify: bool = True,
    verify_decoders: bool | None = None,
    verify_all_points: bool = True,
    vgg_params: Any = None,
    json_out: str | Path | None = None,
) -> KodakBenchmarkReport:
  """Runs Rate-WD benchmark across Kodak images and computes BD-rate vs cjxl -e 7."""
  t_start = time.time()
  eff_steps = int(steps if num_steps is None else num_steps)
  eff_verify = bool(verify if verify_decoders is None else verify_decoders)
  eff_crop = int(max_dim) if max_dim is not None else crop_size
  kdir = Path(kodak_dir)
  if images is None:
    img_list = sorted(p.name for p in kdir.glob("kodim*.png"))
    if not img_list:
      img_list = sorted(p.name for p in kdir.glob("*.png"))
  else:
    img_list = list(images)

  if max_images is not None and max_images > 0:
    if images is not None and max_images > len(img_list):
      all_imgs = sorted(p.name for p in kdir.glob("kodim*.png"))
      if all_imgs:
        img_list = all_imgs[: int(max_images)]
    else:
      img_list = img_list[: int(max_images)]

  if vgg_params is None:
    vgg_params = load_vgg16_params()
  per_image: list[ImageRDResult] = []
  for img_name in img_list:
    p_cand = Path(img_name)
    if p_cand.is_absolute():
      img_path = p_cand
    else:
      img_path = kdir / img_name
      if not img_path.exists() and (kdir / f"{img_name}.png").exists():
        img_path = kdir / f"{img_name}.png"
    rgb = load_benchmark_image(img_path, crop_size=eff_crop)
    res = evaluate_image_rd_curve(
        rgb,
        image_name=img_path.name,
        distances=distances,
        steps=eff_steps,
        mode=mode,
        effort=effort,
        enable_noise=enable_noise,
        verify=eff_verify,
        verify_all_points=verify_all_points,
        vgg_params=vgg_params,
    )
    per_image.append(res)

  bdr0_vals = [r.bd_rate_step0_pct for r in per_image]
  bdr_opt_vals = [r.bd_rate_opt_pct for r in per_image]
  mean_bdr0 = float(np.mean(bdr0_vals)) if bdr0_vals else float("nan")
  mean_bdr_opt = float(np.mean(bdr_opt_vals)) if bdr_opt_vals else float("nan")
  median_bdr_opt = (
      float(np.median(bdr_opt_vals)) if bdr_opt_vals else float("nan")
  )
  elapsed = time.time() - t_start

  report = KodakBenchmarkReport(
      corpus_dir=str(kdir),
      crop_size=eff_crop,
      distances=[float(d) for d in distances],
      steps=eff_steps,
      mode=mode,
      effort=int(effort),
      per_image=per_image,
      mean_bd_rate_step0_pct=mean_bdr0,
      mean_bd_rate_opt_pct=mean_bdr_opt,
      mean_bd_rate_improvement_pct=mean_bdr_opt - mean_bdr0,
      median_bd_rate_opt_pct=median_bdr_opt,
      total_elapsed_s=elapsed,
  )
  if json_out is not None:
    out_path = Path(json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
  return report


def run_benchmark(
    path_or_dir: str | Path = DEFAULT_KODAK_DIR,
    **kwargs: Any,
) -> KodakBenchmarkReport:
  """Wrapper around `run_kodak_benchmark` accepting a directory or single image path."""
  p = Path(path_or_dir)
  if p.is_file():
    return run_kodak_benchmark(
        kodak_dir=p.parent,
        images=[p.name],
        **kwargs,
    )
  return run_kodak_benchmark(kodak_dir=p, **kwargs)


def format_benchmark_table(report: KodakBenchmarkReport) -> str:
  """Formats a human-readable summary table for a `KodakBenchmarkReport`."""
  lines: list[str] = []
  lines.append("=" * 84)
  lines.append(
      f"Kodak Rate-WD Benchmark vs cjxl -e {report.effort} "
      f"(mode={report.mode}, steps={report.steps}, crop={report.crop_size})"
  )
  lines.append("=" * 84)
  lines.append(
      f"{'Image':<14} | {'Step-0 BD-Rate':>15} | {'Opt BD-Rate':>14} | "
      f"{'Opt Delta':>12} | {'Min Cross-PSNR':>15}"
  )
  lines.append("-" * 84)
  for r in report.per_image:
    cpsnrs = [
        p.cross_decoder_psnr
        for p in r.opt_points
        if p.cross_decoder_psnr is not None
    ]
    c_str = f"{min(cpsnrs):.2f} dB" if cpsnrs else "n/a"
    lines.append(
        f"{r.image_name:<14} | {r.bd_rate_step0_pct:>+14.2f}% | "
        f"{r.bd_rate_opt_pct:>+13.2f}% | {r.bd_rate_improvement_pct:>+11.2f}% | "
        f"{c_str:>15}"
    )
    c_pts = [(round(p.bpp, 4), round(p.wd, 4)) for p in r.cjxl_points]
    s0_pts = [(round(p.bpp, 4), round(p.wd, 4)) for p in r.step0_points]
    opt_pts = [(round(p.bpp, 4), round(p.wd, 4)) for p in r.opt_points]
    lines.append(f"  cjxl -e {report.effort} : {c_pts}")
    lines.append(f"  JAX step0 : {s0_pts}")
    lines.append(f"  JAX opt   : {opt_pts}")
  lines.append("-" * 84)
  lines.append(
      f"Mean Step-0 BD-rate vs cjxl -e {report.effort}    : "
      f"{report.mean_bd_rate_step0_pct:+.2f}%"
  )
  lines.append(
      f"Mean Optimized BD-rate vs cjxl -e {report.effort} : "
      f"{report.mean_bd_rate_opt_pct:+.2f}%"
  )
  lines.append(
      f"Median Optimized BD-rate vs cjxl -e {report.effort}: "
      f"{report.median_bd_rate_opt_pct:+.2f}%"
  )
  lines.append(
      f"Optimization BD-rate improvement    : "
      f"{report.mean_bd_rate_improvement_pct:+.2f}%"
  )
  lines.append(f"Total elapsed wall time             : {report.total_elapsed_s:.1f}s")
  lines.append("=" * 84)
  return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
  """Console script entry point for `jxl-wasserstein-benchmark`."""
  import sys
  from jxl_wasserstein import cli

  args = list(sys.argv[1:] if argv is None else argv)
  if "--benchmark-kodak" not in args:
    args = ["--benchmark-kodak", *args]
  return cli.main(args)


__all__ = [
    "DEFAULT_CORPUS_ROOT",
    "DEFAULT_KODAK_DIR",
    "DEFAULT_BENCHMARK_DISTANCES",
    "DEFAULT_KODAK_SUBSET",
    "RDPoint",
    "ImageRDResult",
    "KodakBenchmarkReport",
    "compute_bd_rate",
    "bd_rate",
    "encode_cjxl_baseline",
    "is_pure_8bit_sdr_image",
    "discover_sdr_corpus",
    "load_benchmark_image",
    "init_pure_jax_wd_state",
    "optimize_rate_wd_point",
    "evaluate_image_rd_curve",
    "run_kodak_benchmark",
    "run_benchmark",
    "format_benchmark_table",
    "main",
]


if __name__ == "__main__":
  raise SystemExit(main())

