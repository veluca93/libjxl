// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_group.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/memory_manager_internal.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_group.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/common.h"  // kMaxNumPasses
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_transforms-inl.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/image.h"
#include "lib/jxl/quantizer-inl.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/simd_util.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::CopySignToAbs;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Round;
using hwy::HWY_NAMESPACE::Sub;

// NOTE: caller takes care of extracting quant from rect of RawQuantField.
void QuantizeBlockAC(const Quantizer& quantizer, const bool error_diffusion,
                     size_t c, float qm_multiplier, AcStrategyType quant_kind,
                     size_t xsize, size_t ysize, const float zero_thres[8],
                     float eob_scale, const float* JXL_RESTRICT block_in,
                     int32_t quant, int32_t* JXL_RESTRICT block_out) {
  const float* JXL_RESTRICT qm = quantizer.InvDequantMatrix(quant_kind, c);
  const float qac = quantizer.Scale() * quant * qm_multiplier;

  float quad_thres[4] = {zero_thres[0], zero_thres[1], zero_thres[2],
                         zero_thres[3]};
  if (c != 1 && xsize * ysize >= 4) {
    if (eob_scale > 0.0f) {
      const float sub =
          Clamp1(0.0045f * static_cast<float>(xsize * ysize), 0.0f, 0.055f);
      quad_thres[0] = std::max(0.525f, quad_thres[0] - sub);
      for (int i = 1; i < 3; ++i) {
        quad_thres[i] = std::max(0.57f, quad_thres[i] - 0.35f * sub);
      }
    } else {
      const float sub = 0.00744f * static_cast<float>(xsize * ysize);
      for (int i = 0; i < 4; ++i) {
        quad_thres[i] = std::max(0.50f, quad_thres[i] - sub);
      }
    }
  }

  HWY_CAPPED(float, kBlockDim) df;
  HWY_CAPPED(int32_t, kBlockDim) di;
  const auto quantv = Set(df, qac);
  const auto one = Set(df, 1.0f);
  const auto half = Set(df, 0.5f);
  const auto lim1 = Set(df, 2.2f);
  const auto lim2 = Set(df, 4.2f);

  HWY_ALIGN float t0_row[AcStrategy::kMaxBlockDim];
  HWY_ALIGN float bias_row[AcStrategy::kMaxBlockDim];
  const float bias_scale = Clamp1(1.35f * eob_scale, 0.0f, 1.0f);

  for (size_t v8 = 0; v8 < kBlockDim; ++v8) {
    if ((v8 & 1) == 0) {
      for (size_t u8 = 0; u8 < kBlockDim; ++u8) {
        const size_t hfix =
            (static_cast<size_t>(v8 >= 4) * 2 + static_cast<size_t>(u8 >= 4));
        const uint8_t s = kCoeffBand[v8][u8];
        const float t0 = quad_thres[hfix] + zero_thres[4 + s];
        const float min_bias = (c == 1 && s == 3) ? -0.010f : -0.004f;
        const float bias =
            bias_scale * Clamp1(0.42f * (t0 - 0.568f), min_bias, 0.040f);
        for (size_t ix = 0; ix < xsize; ++ix) {
          t0_row[u8 * xsize + ix] = t0;
          bias_row[u8 * xsize + ix] = bias;
        }
      }
    }
    for (size_t iy = 0; iy < ysize; ++iy) {
      const size_t y = v8 * ysize + iy;
      const size_t off = y * kBlockDim * xsize;
      for (size_t x = 0; x < xsize * kBlockDim; x += Lanes(df)) {
        const auto t0 = Load(df, t0_row + x);
        const auto half_bias = Mul(Load(df, bias_row + x), half);
        const auto q = Mul(Load(df, qm + off + x), quantv);
        const auto in = Load(df, block_in + off + x);
        const auto val = Mul(q, in);
        const auto abs_val = Abs(val);
        const auto nzero_mask = Ge(abs_val, t0);
        const auto eff_bias =
            Add(IfThenElseZero(Lt(abs_val, lim1), half_bias),
                IfThenElseZero(Lt(abs_val, lim2), half_bias));
        const auto r_abs = Max(one, Round(Sub(abs_val, eff_bias)));
        const auto v = ConvertTo(
            di, IfThenElseZero(nzero_mask, CopySignToAbs(r_abs, val)));
        Store(v, di, block_out + off + x);
      }
    }
  }

  // Backward EOB / trailing zero-run RDO along the natural zig-zag scan order.
  // In AC tokenization, the scan terminates immediately after the last non-zero
  // coefficient (when --nzeros == 0). Thus, a trailing +/-1 coefficient
  // preceded by a run of Z zeros costs (Z + 1) extra tokens compared to
  // terminating at the previous non-zero coefficient.
  if (eob_scale > 0.0f &&
      ((1 << static_cast<size_t>(quant_kind)) & kPartialBlockKinds) == 0 &&
      xsize >= ysize) {
    const size_t cx = xsize;
    const size_t cy = ysize;
    const size_t xs = cx / cy;
    const size_t xsm = xs - 1;
    const size_t xss = CeilLog2Nonzero(xs);
    const size_t dim = cx * kBlockDim;
    const size_t min_keep_nz =
        (eob_scale < 0.35f) ? (2 * cx * cy) : ((c == 1) ? 2 : 1);
    const float max_bonus = ((c == 1) ? 0.045f : 0.055f) * eob_scale;
    const float cand_max_bonus = 1.35f * max_bonus;
    const float run_slope = ((c == 1) ? 0.0032f : 0.0040f) * eob_scale;

    constexpr size_t kMaxCands = 8;
    struct TailCand {
      int pos;
      float norm_excess;
      uint8_t band;
      size_t zeros_before;
    };
    TailCand cands[kMaxCands];
    size_t num_cands = 0;
    bool hit_barrier = false;
    size_t kept_nz = 0;
    size_t zeros_since_last_nz = 0;
    int pending_interior_pos = -1;

    auto visit_coeff = [&](size_t x, size_t y) -> bool {
      const size_t pos = y * dim + x;
      const int32_t q = block_out[pos];
      if (q == 0) {
        if (num_cands > 0 && !hit_barrier) {
          ++cands[num_cands - 1].zeros_before;
        }
        ++zeros_since_last_nz;
        return true;
      }
      if (pending_interior_pos >= 0) {
        if (zeros_since_last_nz >= 5) {
          block_out[pending_interior_pos] = 0;
          --kept_nz;
        }
        pending_interior_pos = -1;
      }
      const size_t v8 = y / cy;
      const size_t u8 = x / cx;
      const uint8_t s = kCoeffBand[v8][u8];
      const size_t hfix =
          (static_cast<size_t>(v8 >= 4) * 2 + static_cast<size_t>(u8 >= 4));
      const float t0 = quad_thres[hfix] + zero_thres[4 + s];
      const float abs_val = std::abs(block_in[pos] * (qm[pos] * qac));

      if (!hit_barrier && num_cands < kMaxCands) {
        if ((q == 1 || q == -1) && s <= 2 && zero_thres[4 + s] >= -0.004f) {
          const float excess = std::max(0.0f, abs_val - t0);
          const float band_scale =
              ((s == 0) ? 1.0f : ((s == 1) ? 0.80f : 0.45f)) *
              Clamp1(1.0f + 2.5f * zero_thres[4 + s], 0.80f, 1.25f);
          const float norm_excess = excess / band_scale;
          if (norm_excess < cand_max_bonus) {
            cands[num_cands++] = {static_cast<int>(pos), norm_excess, s, 0};
            zeros_since_last_nz = 0;
            return true;
          }
        }
        hit_barrier = true;
      }

      // Sparse-tail level transition |q|=2 -> 1 when preceded/followed by
      // long zero runs in UHF/HF (s <= 1).
      if ((q == 2 || q == -2) && s <= 1 && zeros_since_last_nz >= 3 &&
          zero_thres[4 + s] >= 0.0f) {
        const float bias =
            bias_scale * Clamp1(0.42f * (t0 - 0.568f), -0.004f, 0.040f);
        const float level2_thres = 1.50f + bias + 0.022f * eob_scale;
        if (abs_val < level2_thres) {
          block_out[pos] = (q > 0) ? 1 : -1;
        }
      } else if (hit_barrier && (q == 1 || q == -1) && s == 0 && hfix == 3 &&
                 zeros_since_last_nz >= 5 && zero_thres[4] > 0.002f) {
        const float excess = abs_val - t0;
        if (excess < 0.012f * eob_scale) {
          pending_interior_pos = static_cast<int>(pos);
        }
      }

      zeros_since_last_nz = 0;
      ++kept_nz;
      return (kept_nz < min_keep_nz + 4);
    };

    bool keep_scanning = true;
    for (size_t i = 0; i < dim - 1 && keep_scanning; ++i) {
      for (size_t jj = 0; jj <= i; ++jj) {
        const size_t j = i - jj;
        size_t x = dim - 1 - (i - j);
        size_t y = dim - 1 - j;
        if (i & 1) std::swap(x, y);
        if ((y & xsm) != 0) continue;
        y >>= xss;
        if (!visit_coeff(x, y)) {
          keep_scanning = false;
          break;
        }
      }
    }
    for (size_t ii = 0; ii < dim && keep_scanning; ++ii) {
      const size_t i = dim - 1 - ii;
      for (size_t jj = 0; jj <= i; ++jj) {
        const size_t j = i - jj;
        size_t x = j;
        size_t y = i - j;
        if (i & 1) std::swap(x, y);
        if ((y & xsm) != 0) continue;
        y >>= xss;
        if (x < cx && y < cy) continue;
        if (!visit_coeff(x, y)) {
          keep_scanning = false;
          break;
        }
      }
    }

    if (kept_nz + num_cands > min_keep_nz) {
      const size_t max_k =
          std::min(num_cands, (kept_nz + num_cands) - min_keep_nz);
      // Viterbi DP for optimal EOB cutoff position best_m in [0, max_k].
      bool reachable[kMaxCands + 1] = {true};
      float dp_gain[kMaxCands + 1] = {0.0f};
      size_t best_m = 0;
      float best_gain = 0.0f;
      for (size_t m = 1; m <= max_k; ++m) {
        for (size_t prev_m = 0; prev_m < m; ++prev_m) {
          if (!reachable[prev_m]) continue;
          if (m - prev_m == 1 && cands[prev_m].norm_excess >= max_bonus) {
            continue;
          }
          float group_excess = 0.0f;
          size_t group_zeros = 0;
          uint8_t max_band = 0;
          for (size_t j = prev_m; j < m; ++j) {
            group_excess += cands[j].norm_excess;
            group_zeros += cands[j].zeros_before;
            max_band = std::max(max_band, cands[j].band);
          }
          group_zeros += (m - 1 - prev_m);
          const size_t min_run = (max_band == 2) ? 5 : 2;
          const float group_cap =
              max_bonus * (1.0f + 0.35f * static_cast<float>(m - 1 - prev_m));
          if (group_zeros >= min_run) {
            const float allowed = std::min(
                group_cap,
                run_slope * static_cast<float>(group_zeros - min_run + 1));
            if (group_excess < allowed) {
              reachable[m] = true;
              const float gain = dp_gain[prev_m] + (allowed - group_excess);
              if (gain > dp_gain[m]) {
                dp_gain[m] = gain;
              }
            }
          }
        }
        if (reachable[m] && dp_gain[m] >= best_gain) {
          best_gain = dp_gain[m];
          best_m = m;
        }
      }
      for (size_t i = 0; i < best_m; ++i) {
        block_out[cands[i].pos] = 0;
      }
    } else if (kept_nz == 0 && num_cands == 0 && c == 1 && cx * cy <= 4) {
      float best_abs = 0.46f;
      int best_pos = -1;
      int32_t best_sign = 1;
      for (size_t y = 0; y < 4 * cy; ++y) {
        for (size_t x = 0; x < 4 * cx; ++x) {
          if (x < cx && y < cy) continue;
          const size_t pos = y * dim + x;
          const float val = block_in[pos] * (qm[pos] * qac);
          const float abs_val = std::abs(val);
          if (abs_val > best_abs) {
            best_abs = abs_val;
            best_pos = static_cast<int>(pos);
            best_sign = (val >= 0.0f) ? 1 : -1;
          }
        }
      }
      if (best_pos >= 0) {
        block_out[best_pos] = best_sign;
      }
    }
  }
}

// NOTE: caller takes care of extracting quant from rect of RawQuantField.
void QuantizeRoundtripYBlockAC(PassesEncoderState* enc_state, const size_t size,
                               const Quantizer& quantizer,
                               const bool error_diffusion,
                               AcStrategyType quant_kind, size_t xsize,
                               size_t ysize, float zero_thres[3][8],
                               float eob_scale, const float* JXL_RESTRICT biases,
                               const int32_t quant, float* JXL_RESTRICT inout,
                               int32_t* JXL_RESTRICT quantized) {
  if (enc_state->cparams.speed_tier > SpeedTier::kHare) {
    zero_thres[1][0] = 0.56f;
    zero_thres[1][1] = 0.62f;
    zero_thres[1][2] = 0.62f;
    zero_thres[1][3] = 0.62f;
    for (size_t s = 0; s < 4; ++s) {
      zero_thres[1][4 + s] = 0.0f;
    }
  }
  QuantizeBlockAC(quantizer, error_diffusion, 1, 1.0f, quant_kind, xsize, ysize,
                  zero_thres[1], eob_scale, inout + size, quant,
                  quantized + size);

  const float* JXL_RESTRICT dequant_matrix =
      quantizer.DequantMatrix(quant_kind, 1);

  HWY_CAPPED(float, kDCTBlockSize) df;
  HWY_CAPPED(int32_t, kDCTBlockSize) di;
  const auto inv_qac = Set(df, quantizer.inv_quant_ac(quant));
  for (size_t k = 0; k < kDCTBlockSize * xsize * ysize; k += Lanes(df)) {
    const auto oquant = Load(di, quantized + size + k);
    const auto adj_quant = AdjustQuantBias(di, 1, oquant, biases);
    const auto dequantm = Load(df, dequant_matrix + k);
    Store(Mul(Mul(adj_quant, dequantm), inv_qac), df, inout + size + k);
  }
}

Status ComputeCoefficients(size_t group_idx, PassesEncoderState* enc_state,
                           const Image3F& opsin, const Rect& rect,
                           Image3F* dc) {
  JxlMemoryManager* memory_manager = opsin.memory_manager();
  const Rect block_group_rect =
      enc_state->shared.frame_dim.BlockGroupRect(group_idx);
  const Rect cmap_rect(
      block_group_rect.x0() / kColorTileDimInBlocks,
      block_group_rect.y0() / kColorTileDimInBlocks,
      DivCeil(block_group_rect.xsize(), kColorTileDimInBlocks),
      DivCeil(block_group_rect.ysize(), kColorTileDimInBlocks));
  const Rect group_rect =
      enc_state->shared.frame_dim.GroupRect(group_idx).Translate(rect.x0(),
                                                                 rect.y0());

  const size_t xsize_blocks = block_group_rect.xsize();
  const size_t ysize_blocks = block_group_rect.ysize();

  const size_t dc_stride = static_cast<size_t>(dc->PixelsPerRow());
  const size_t opsin_stride = static_cast<size_t>(opsin.PixelsPerRow());

  const ImageI& full_quant_field = enc_state->shared.raw_quant_field;
  const CompressParams& cparams = enc_state->cparams;
  const size_t num_passes = enc_state->progressive_splitter.GetNumPasses();
  JXL_ENSURE(num_passes > 0);
  const bool has_band_qf =
      cparams.speed_tier <= SpeedTier::kHare &&
      !cparams.disable_perceptual_optimizations && num_passes == 1 &&
      cparams.butteraugli_distance < 3.5f && cparams.resampling <= 1 &&
      cparams.gaborish != Override::kOff &&
      xsize_blocks * ysize_blocks > 64 &&
      enc_state->band_qf[0][0].xsize() > 0 &&
      enc_state->band_qf[0][0].ysize() > 0;

  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;

  // TODO(veluca): consider strategies to reduce this memory.
  size_t mem_bytes = 3 * AcStrategy::kMaxCoeffArea * sizeof(int32_t);
  JXL_ASSIGN_OR_RETURN(auto mem,
                       AlignedMemory::Create(memory_manager, mem_bytes));
  size_t fmem_bytes =
      (5 * AcStrategy::kMaxCoeffArea + dct_scratch_size) * sizeof(float);
  JXL_ASSIGN_OR_RETURN(auto fmem,
                       AlignedMemory::Create(memory_manager, fmem_bytes));
  float* JXL_RESTRICT scratch_space =
      fmem.address<float>() + 3 * AcStrategy::kMaxCoeffArea;
  {
    // Only use error diffusion in Squirrel mode or slower.
    const bool error_diffusion = cparams.speed_tier <= SpeedTier::kSquirrel;
    constexpr HWY_CAPPED(float, kDCTBlockSize) d;

    int32_t* JXL_RESTRICT coeffs[3][kMaxNumPasses] = {};
    for (size_t i = 0; i < num_passes; i++) {
      // TODO(veluca): 16-bit quantized coeffs are not implemented yet.
      JXL_ENSURE(enc_state->coeffs[i]->Type() == ACType::k32);
      for (size_t c = 0; c < 3; c++) {
        coeffs[c][i] = enc_state->coeffs[i]->PlaneRow(c, group_idx, 0).ptr32;
      }
    }

    HWY_ALIGN float* coeffs_in = fmem.address<float>();
    HWY_ALIGN int32_t* quantized = mem.address<int32_t>();

    for (size_t by = 0; by < ysize_blocks; ++by) {
      const int32_t* JXL_RESTRICT row_quant_ac =
          block_group_rect.ConstRow(full_quant_field, by);
      size_t ty = by / kColorTileDimInBlocks;
      const int8_t* JXL_RESTRICT row_cmap[3] = {
          cmap_rect.ConstRow(enc_state->shared.cmap.ytox_map, ty),
          nullptr,
          cmap_rect.ConstRow(enc_state->shared.cmap.ytob_map, ty),
      };
      const float* JXL_RESTRICT opsin_rows[3] = {
          group_rect.ConstPlaneRow(opsin, 0, by * kBlockDim),
          group_rect.ConstPlaneRow(opsin, 1, by * kBlockDim),
          group_rect.ConstPlaneRow(opsin, 2, by * kBlockDim),
      };
      float* JXL_RESTRICT dc_rows[3] = {
          block_group_rect.PlaneRow(dc, 0, by),
          block_group_rect.PlaneRow(dc, 1, by),
          block_group_rect.PlaneRow(dc, 2, by),
      };
      AcStrategyRow ac_strategy_row =
          enc_state->shared.ac_strategy.ConstRow(block_group_rect, by);
      for (size_t tx = 0; tx < DivCeil(xsize_blocks, kColorTileDimInBlocks);
           tx++) {
        const auto x_factor =
            Set(d, enc_state->shared.cmap.base().YtoXRatio(row_cmap[0][tx]));
        const auto b_factor =
            Set(d, enc_state->shared.cmap.base().YtoBRatio(row_cmap[2][tx]));
        for (size_t bx = tx * kColorTileDimInBlocks;
             bx < xsize_blocks && bx < (tx + 1) * kColorTileDimInBlocks; ++bx) {
          const AcStrategy acs = ac_strategy_row[bx];
          if (!acs.IsFirstBlock()) continue;

          const size_t cov_x = acs.covered_blocks_x();
          const size_t cov_y = acs.covered_blocks_y();
          size_t xblocks = cov_x;
          size_t yblocks = cov_y;

          CoefficientLayout(&yblocks, &xblocks);

          size_t size = kDCTBlockSize * xblocks * yblocks;

          const int32_t quant_ac = row_quant_ac[bx];
          const float q_block = enc_state->shared.quantizer.Scale() * quant_ac;

          // Gather per-band target quantizer q_{c, s}(T) committed during RDO
          // and derive jpegli-style per-band and HF-activity deadzones.
          const float d_target = cparams.butteraugli_distance;
          const float dz_w =
              has_band_qf
                  ? Clamp1((d_target - 0.35f) / (d_target + 0.55f) * 1.55f,
                           0.0f, 1.0f)
                  : 0.0f;
          const float eob_scale =
              (has_band_qf && cparams.speed_tier <= SpeedTier::kHare)
                  ? Clamp1((d_target - 0.28f) / (d_target + 0.50f) * 1.45f,
                           0.0f, 1.0f)
                  : 0.0f;
          float band_q[3][4];
          for (size_t c = 0; c < 3; ++c) {
            for (size_t s = 0; s < 4; ++s) {
              band_q[c][s] =
                  has_band_qf
                      ? block_group_rect.ConstRow(enc_state->band_qf[c][s],
                                                  by)[bx]
                      : q_block;
            }
          }
          if (!has_band_qf) {
            if (cparams.speed_tier <= SpeedTier::kHare &&
                enc_state->band_qf[1][0].xsize() > 0 &&
                enc_state->band_qf[1][0].ysize() > 0) {
              band_q[1][0] =
                  block_group_rect.ConstRow(enc_state->band_qf[1][0], by)[bx];
            } else {
              uint32_t bits = 0;
              memcpy(&bits, &band_q[1][0], sizeof(bits));
              bits &= ~31u;
              memcpy(&band_q[1][0], &bits, sizeof(bits));
            }
          }
          float zero_thres[3][8];
          ComputeBlockThresholds(acs.Strategy(), xblocks, yblocks, q_block,
                                 band_q, d_target, dz_w, eob_scale, zero_thres);

          // DCT all channels, roundtrip-quantize Y and set DC.
          for (size_t c : {0, 1, 2}) {
            TransformFromPixels(acs.Strategy(), opsin_rows[c] + bx * kBlockDim,
                                opsin_stride, coeffs_in + c * size,
                                scratch_space);
          }
          DCFromLowestFrequencies(acs.Strategy(), coeffs_in + size,
                                  dc_rows[1] + bx, dc_stride, scratch_space);

          QuantizeRoundtripYBlockAC(
              enc_state, size, enc_state->shared.quantizer, error_diffusion,
              acs.Strategy(), xblocks, yblocks, zero_thres, eob_scale,
              kDefaultQuantBias, quant_ac, coeffs_in, quantized);

          // Unapply color correlation
          for (size_t k = 0; k < size; k += Lanes(d)) {
            const auto in_x = Load(d, coeffs_in + k);
            const auto in_y = Load(d, coeffs_in + size + k);
            const auto in_b = Load(d, coeffs_in + 2 * size + k);
            const auto out_x = NegMulAdd(x_factor, in_y, in_x);
            const auto out_b = NegMulAdd(b_factor, in_y, in_b);
            Store(out_x, d, coeffs_in + k);
            Store(out_b, d, coeffs_in + 2 * size + k);
          }

          // Quantize X and B channels and set DC.
          for (size_t c : {0, 2}) {
            QuantizeBlockAC(enc_state->shared.quantizer, error_diffusion, c,
                            c == 0 ? enc_state->x_qm_multiplier
                                   : enc_state->b_qm_multiplier,
                            acs.Strategy(), xblocks, yblocks, zero_thres[c],
                            eob_scale, coeffs_in + c * size, quant_ac,
                            quantized + c * size);
            DCFromLowestFrequencies(acs.Strategy(), coeffs_in + c * size,
                                    dc_rows[c] + bx, dc_stride, scratch_space);
          }
          for (size_t c = 0; c < 3; c++) {
            enc_state->progressive_splitter.SplitACCoefficients(
                quantized + c * size, acs, bx, by, coeffs[c]);
            for (size_t p = 0; p < num_passes; p++) {
              coeffs[c][p] += size;
            }
          }
        }
      }
    }
  }
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(ComputeCoefficients);
Status ComputeCoefficients(size_t group_idx, PassesEncoderState* enc_state,
                           const Image3F& opsin, const Rect& rect,
                           Image3F* dc) {
  return HWY_DYNAMIC_DISPATCH(ComputeCoefficients)(group_idx, enc_state, opsin,
                                                   rect, dc);
}

Status EncodeGroupTokenizedCoefficients(size_t group_idx, size_t pass_idx,
                                        size_t histogram_idx,
                                        const PassesEncoderState& enc_state,
                                        BitWriter* writer, AuxOut* aux_out) {
  // Select which histogram to use among those of the current pass.
  const size_t num_histograms = enc_state.shared.num_histograms;
  // num_histograms is 0 only for lossless.
  JXL_ENSURE(num_histograms == 0 || histogram_idx < num_histograms);
  size_t histo_selector_bits = CeilLog2Nonzero(num_histograms);

  if (histo_selector_bits != 0) {
    JXL_RETURN_IF_ERROR(
        writer->WithMaxBits(histo_selector_bits, LayerType::Ac, aux_out, [&] {
          writer->Write(histo_selector_bits, histogram_idx);
          return true;
        }));
  }
  size_t context_offset =
      histogram_idx * enc_state.shared.block_ctx_map.NumACContexts();
  JXL_RETURN_IF_ERROR(
      WriteTokens(enc_state.passes[pass_idx].ac_tokens[group_idx],
                  enc_state.passes[pass_idx].codes, context_offset, writer,
                  LayerType::AcTokens, aux_out));

  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
