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
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/pack_signed.h"

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

static JXL_INLINE size_t TrellisNzBucket(size_t shifted_n) {
  if (shifted_n <= 1) return 0;
  if (shifted_n == 2) return 1;
  if (shifted_n <= 4) return 2;
  if (shifted_n <= 8) return 3;
  if (shifted_n <= 12) return 4;
  if (shifted_n <= 20) return 5;
  if (shifted_n <= 32) return 6;
  return 7;
}

int32_t TrellisQuantizeBlockAC(const Quantizer& quantizer,
                               const AcTokenCostModel& tcm, size_t c,
                               float qm_multiplier, AcStrategyType quant_kind,
                               size_t xsize, size_t ysize, size_t ord,
                               size_t covered_blocks,
                               size_t log2_covered_blocks,
                               int32_t predicted_nzeros,
                               const float quad_thres[4],
                               const float zero_thres[8], float eob_scale,
                               const float* JXL_RESTRICT block_in,
                               int32_t quant, int32_t* JXL_RESTRICT block_out) {
  static constexpr uint16_t kNzBucketCtxOffset[8] = {0,   31,  62,  93,
                                                     123, 152, 180, 206};
  static constexpr size_t kMaxTrellisActive = 64;
  const size_t size = covered_blocks * kDCTBlockSize;
  const size_t dim = xsize * kBlockDim;
  const coeff_order_t* JXL_RESTRICT order =
      &tcm.coeff_orders[CoeffOrderOffset(ord, c)];
  const size_t block_ctx = tcm.block_ctx_map.Context(/*dc_idx=*/0, quant, ord, c);
  const uint32_t nzero_ctx =
      tcm.block_ctx_map.NonZeroContext(predicted_nzeros, block_ctx);
  const uint8_t cl_nzero = tcm.cluster_of_ctx[nzero_ctx];
  const size_t histo_offset =
      tcm.block_ctx_map.ZeroDensityContextsOffset(block_ctx);
  const float* JXL_RESTRICT qm = quantizer.InvDequantMatrix(quant_kind, c);
  const float qac = quantizer.Scale() * quant * qm_multiplier;
  const float b_c = kDefaultQuantBias[c];

  auto a_hat = [b_c](int32_t m) -> float {
    return (m == 1)
               ? b_c
               : (static_cast<float>(m) - 0.145f / static_cast<float>(m));
  };
  const bool is_partial_block =
      ((1 << static_cast<size_t>(quant_kind)) & kPartialBlockKinds) != 0;
  auto rescue_quad0 = [&]() -> int32_t {
    if (is_partial_block || c != 1 || covered_blocks > 4 || eob_scale <= 0.0f) {
      return 0;
    }
    float best_abs = 0.46f;
    int best_pos = -1;
    int32_t best_sign = 1;
    for (size_t y = 0; y < 4 * ysize; ++y) {
      for (size_t x = 0; x < 4 * xsize; ++x) {
        if (x < xsize && y < ysize) continue;
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
      return 1;
    }
    return 0;
  };

  struct ActiveCoeff {
    uint32_t k;
    uint32_t pos;
    int32_t q_lo;
    int32_t q_hi;
    float d_lo;
    float d_hi;
  };
  std::vector<ActiveCoeff> all_active;
  all_active.reserve(64);

  static constexpr float kBandDistWeight[4] = {0.92f, 1.00f, 1.24f, 1.65f};
  const float sparse_w = Clamp1((eob_scale - 0.32f) / 0.42f, 0.0f, 1.0f);
  const float base_ref = (c == 1) ? 2.78f : 2.98f;
  const float eff_ref_bits = base_ref * (1.0f + 0.35f * eob_scale);
  const float kQuadTilt = 0.50f;
  const float base_h = 0.58f;
  const float t_ref = 0.60f;
  const float denom_ref = b_c * (2.0f * t_ref - b_c);
  size_t scalar_lf_nz = 0;

  for (size_t k = covered_blocks; k < size; ++k) {
    const coeff_order_t pos = order[k];
    const size_t y = pos / dim;
    const size_t x = pos % dim;
    const size_t v8 = y / ysize;
    const size_t u8 = x / xsize;
    const size_t hfix =
        (static_cast<size_t>(v8 >= 4) * 2 + static_cast<size_t>(u8 >= 4));
    const uint8_t s = kCoeffBand[v8][u8];
    const float t0_full = quad_thres[hfix] + zero_thres[4 + s];
    const float x_val = block_in[pos] * (qm[pos] * qac);
    const float a = std::abs(x_val);
    if (block_out[pos] != 0 && (is_partial_block ? (k <= 8) : (s >= 2))) {
      ++scalar_lf_nz;
    }
    if (block_out[pos] == 0 && a < std::max(0.49f, t0_full - 0.04f)) {
      continue;
    }
    const float raw_h = (hfix == 0) ? 0.58f : ((c == 1) ? 0.64f : 0.62f);
    const float perc_t0 =
        std::max(0.50f, t0_full - (1.0f - kQuadTilt) * (raw_h - base_h));
    const float t_ratio = t_ref / perc_t0;
    const float w_base = (eff_ref_bits / denom_ref) * (t_ratio * t_ratio);
    const float band_w = is_partial_block ? 1.0f : kBandDistWeight[s];
    const float w_zero = w_base * (1.0f + sparse_w * (band_w - 1.0f));
    const int32_t sgn = (x_val >= 0.0f) ? 1 : -1;
    int32_t q_lo = sgn;
    int32_t q_hi = sgn;
    if (a >= 1.40f) {
      const int32_t m_hi =
          std::max<int32_t>(1, static_cast<int32_t>(std::floor(a + 0.56f)));
      const int32_t m_lo = std::max<int32_t>(1, m_hi - 1);
      q_lo = sgn * m_lo;
      q_hi = sgn * m_hi;
    }
    const float d_one_base = b_c * (b_c - 2.0f * a);
    const float d_zero_step = (w_zero - w_base) * d_one_base;
    const float ah_lo = a_hat(std::abs(q_lo));
    const float d_lo = d_zero_step + w_base * ah_lo * (ah_lo - 2.0f * a);
    float d_hi = d_lo;
    if (q_hi != q_lo) {
      const float ah_hi = a_hat(std::abs(q_hi));
      d_hi = d_zero_step + w_base * ah_hi * (ah_hi - 2.0f * a);
    }
    all_active.push_back(
        {static_cast<uint32_t>(k), pos, q_lo, q_hi, d_lo, d_hi});
  }

  const size_t total_active = all_active.size();
  if (total_active == 0) {
    return rescue_quad0();
  }

  size_t prefix_active = 0;
  size_t prefix_nzeros = 0;
  size_t prefix_last_nz_k = 0;
  if (total_active > kMaxTrellisActive) {
    prefix_active = total_active - kMaxTrellisActive;
    for (size_t idx = 0; idx < prefix_active; ++idx) {
      const ActiveCoeff& ac = all_active[idx];
      if (block_out[ac.pos] != 0 && ac.q_hi != ac.q_lo) {
        const size_t rem_est = total_active - idx;
        const size_t b_n = TrellisNzBucket((rem_est + covered_blocks - 1) >>
                                           log2_covered_blocks);
        const size_t kf = kCoeffFreqContext[ac.k >> log2_covered_blocks];
        const size_t prev =
            (ac.k > covered_blocks && block_out[order[ac.k - 1]] != 0) ? 1 : 0;
        const uint8_t cl = tcm.cluster_of_ctx[histo_offset +
                                              (kNzBucketCtxOffset[b_n] + kf) * 2 +
                                              prev];
        const float c_lo = ac.d_lo + tcm.ValBits(cl, PackSigned(ac.q_lo));
        const float c_hi = ac.d_hi + tcm.ValBits(cl, PackSigned(ac.q_hi));
        block_out[ac.pos] = (c_hi < c_lo) ? ac.q_hi : ac.q_lo;
      }
    }
    const size_t prefix_end_k = all_active[prefix_active].k;
    for (size_t k = covered_blocks; k < prefix_end_k; ++k) {
      if (block_out[order[k]] != 0) {
        ++prefix_nzeros;
        prefix_last_nz_k = k;
      }
    }
  }

  const ActiveCoeff* active = all_active.data() + prefix_active;
  const size_t M = total_active - prefix_active;
  float dp0[kMaxTrellisActive + 1] = {};
  float dp1[kMaxTrellisActive + 1] = {};
  uint64_t chose_nz0[kMaxTrellisActive] = {};
  uint64_t chose_nz1[kMaxTrellisActive] = {};

  {
    const ActiveCoeff& last_ac = active[M - 1];
    const size_t kf = kCoeffFreqContext[last_ac.k >> log2_covered_blocks];
    for (size_t p = 0; p < 2; ++p) {
      const uint8_t cl =
          tcm.cluster_of_ctx[histo_offset + (kNzBucketCtxOffset[0] + kf) * 2 + p];
      const float c_lo = last_ac.d_lo + tcm.ValBits(cl, PackSigned(last_ac.q_lo));
      const float c_hi =
          (last_ac.q_hi == last_ac.q_lo)
              ? c_lo
              : (last_ac.d_hi + tcm.ValBits(cl, PackSigned(last_ac.q_hi)));
      const float best_c = std::min(c_lo, c_hi);
      if (p == 0) {
        dp0[1] = best_c;
      } else {
        dp1[1] = best_c;
      }
    }
    chose_nz0[M - 1] = 1ULL;
    chose_nz1[M - 1] = 1ULL;
  }

  for (int i = static_cast<int>(M) - 2; i >= 0; --i) {
    const ActiveCoeff& ac = active[i];
    const size_t k_cur = ac.k;
    const size_t k_next = active[i + 1].k;
    const size_t gap = k_next - k_cur - 1;
    float gap_bits[8][2] = {};
    if (gap >= 1) {
      const size_t kf1 = kCoeffFreqContext[(k_cur + 1) >> log2_covered_blocks];
      for (size_t b = 0; b < 8; ++b) {
        if (kNzBucketCtxOffset[b] + kf1 >= 229) break;
        const size_t base = histo_offset + (kNzBucketCtxOffset[b] + kf1) * 2;
        gap_bits[b][0] = tcm.ValBits(tcm.cluster_of_ctx[base + 0], 0);
        gap_bits[b][1] = tcm.ValBits(tcm.cluster_of_ctx[base + 1], 0);
      }
      for (size_t k = k_cur + 2; k < k_next; ++k) {
        const size_t kf = kCoeffFreqContext[k >> log2_covered_blocks];
        for (size_t b = 0; b < 8; ++b) {
          if (kNzBucketCtxOffset[b] + kf >= 229) break;
          const size_t base = histo_offset + (kNzBucketCtxOffset[b] + kf) * 2;
          const float bz = tcm.ValBits(tcm.cluster_of_ctx[base + 0], 0);
          gap_bits[b][0] += bz;
          gap_bits[b][1] += bz;
        }
      }
    }

    float zero_bits_cur[8][2] = {};
    float nz_cost_cur[8][2] = {};
    const size_t kf_cur = kCoeffFreqContext[k_cur >> log2_covered_blocks];
    const uint32_t u_lo = PackSigned(ac.q_lo);
    const uint32_t u_hi = PackSigned(ac.q_hi);
    for (size_t b = 0; b < 8; ++b) {
      if (kNzBucketCtxOffset[b] + kf_cur >= 229) break;
      for (size_t p = 0; p < 2; ++p) {
        const uint8_t cl =
            tcm.cluster_of_ctx[histo_offset +
                               (kNzBucketCtxOffset[b] + kf_cur) * 2 + p];
        zero_bits_cur[b][p] = tcm.ValBits(cl, 0);
        const float c_lo = ac.d_lo + tcm.ValBits(cl, u_lo);
        const float c_hi =
            (ac.q_hi == ac.q_lo) ? c_lo : (ac.d_hi + tcm.ValBits(cl, u_hi));
        nz_cost_cur[b][p] = std::min(c_lo, c_hi);
      }
    }

    const size_t max_n_next = M - 1 - static_cast<size_t>(i);
    const size_t max_n_cur = max_n_next + 1;
    uint64_t mask0 = 0;
    uint64_t mask1 = 0;
    for (size_t n = max_n_cur; n >= 1; --n) {
      const size_t b_n =
          TrellisNzBucket((n + covered_blocks - 1) >> log2_covered_blocks);
      float tail_nz = 0.0f;
      if (n >= 2) {
        if (gap == 0) {
          tail_nz = dp1[n - 1];
        } else {
          const size_t b_nm1 = TrellisNzBucket((n - 1 + covered_blocks - 1) >>
                                               log2_covered_blocks);
          tail_nz = gap_bits[b_nm1][1] + dp0[n - 1];
        }
      }
      const float c0_nz = nz_cost_cur[b_n][0] + tail_nz;
      const float c1_nz = nz_cost_cur[b_n][1] + tail_nz;
      const uint64_t bit = 1ULL << (n - 1);
      if (n <= max_n_next) {
        const float tail_z =
            (gap == 0) ? dp0[n] : (gap_bits[b_n][0] + dp0[n]);
        const float c0_z = zero_bits_cur[b_n][0] + tail_z;
        const float c1_z = zero_bits_cur[b_n][1] + tail_z;
        if (c0_nz < c0_z) mask0 |= bit;
        if (c1_nz < c1_z) mask1 |= bit;
        dp0[n] = std::min(c0_nz, c0_z);
        dp1[n] = std::min(c1_nz, c1_z);
      } else {
        mask0 |= bit;
        mask1 |= bit;
        dp0[n] = c0_nz;
        dp1[n] = c1_nz;
      }
    }
    chose_nz0[i] = mask0;
    chose_nz1[i] = mask1;
  }

  const size_t gap_start =
      (prefix_nzeros > 0) ? (prefix_last_nz_k + 1) : covered_blocks;
  const size_t pre_gap = active[0].k - gap_start;
  float pre_gap_bits[8][2] = {};
  if (pre_gap >= 1) {
    const size_t kf1 = kCoeffFreqContext[gap_start >> log2_covered_blocks];
    for (size_t b = 0; b < 8; ++b) {
      if (kNzBucketCtxOffset[b] + kf1 >= 229) break;
      const size_t base = histo_offset + (kNzBucketCtxOffset[b] + kf1) * 2;
      pre_gap_bits[b][0] = tcm.ValBits(tcm.cluster_of_ctx[base + 0], 0);
      pre_gap_bits[b][1] = tcm.ValBits(tcm.cluster_of_ctx[base + 1], 0);
    }
    for (size_t k = gap_start + 1; k < active[0].k; ++k) {
      const size_t kf = kCoeffFreqContext[k >> log2_covered_blocks];
      for (size_t b = 0; b < 8; ++b) {
        if (kNzBucketCtxOffset[b] + kf >= 229) break;
        const size_t base = histo_offset + (kNzBucketCtxOffset[b] + kf) * 2;
        const float bz = tcm.ValBits(tcm.cluster_of_ctx[base + 0], 0);
        pre_gap_bits[b][0] += bz;
        pre_gap_bits[b][1] += bz;
      }
    }
  }

  const float flat_penalty =
      (prefix_nzeros == 0 && scalar_lf_nz > 0)
          ? (sparse_w * ((c == 1) ? 1.15f : 0.35f) *
             static_cast<float>(std::min<size_t>(2, scalar_lf_nz)))
          : 0.0f;
  float best_total_cost =
      tcm.ValBits(cl_nzero, static_cast<uint32_t>(prefix_nzeros)) +
      flat_penalty;
  size_t best_n = 0;
  for (size_t n = 1; n <= M; ++n) {
    const size_t total_n = prefix_nzeros + n;
    const size_t b_n =
        TrellisNzBucket((n + covered_blocks - 1) >> log2_covered_blocks);
    const size_t prev_start =
        (prefix_nzeros > 0) ? 1 : ((total_n > size / 16) ? 0 : 1);
    float ac_cost = 0.0f;
    if (pre_gap == 0) {
      ac_cost = (prev_start == 0) ? dp0[n] : dp1[n];
    } else {
      ac_cost = pre_gap_bits[b_n][prev_start] + dp0[n];
    }
    const float tot_cost =
        tcm.ValBits(cl_nzero, static_cast<uint32_t>(total_n)) + ac_cost;
    if (tot_cost < best_total_cost) {
      best_total_cost = tot_cost;
      best_n = n;
    }
  }

  for (size_t i = 0; i < M; ++i) {
    block_out[active[i].pos] = 0;
  }
  if (best_n > 0) {
    size_t cur_n = best_n;
    const size_t total_n = prefix_nzeros + best_n;
    const size_t prev_start =
        (prefix_nzeros > 0) ? 1 : ((total_n > size / 16) ? 0 : 1);
    size_t cur_prev = (pre_gap == 0) ? prev_start : 0;
    for (size_t i = 0; i < M && cur_n > 0; ++i) {
      const uint64_t mask = (cur_prev == 0) ? chose_nz0[i] : chose_nz1[i];
      const bool chose = ((mask >> (cur_n - 1)) & 1ULL) != 0;
      if (chose) {
        const size_t b_n =
            TrellisNzBucket((cur_n + covered_blocks - 1) >> log2_covered_blocks);
        const size_t kf = kCoeffFreqContext[active[i].k >> log2_covered_blocks];
        const uint8_t cl = tcm.cluster_of_ctx[histo_offset +
                                              (kNzBucketCtxOffset[b_n] + kf) * 2 +
                                              cur_prev];
        const float c_lo =
            active[i].d_lo + tcm.ValBits(cl, PackSigned(active[i].q_lo));
        const float c_hi =
            (active[i].q_hi == active[i].q_lo)
                ? c_lo
                : (active[i].d_hi + tcm.ValBits(cl, PackSigned(active[i].q_hi)));
        block_out[active[i].pos] =
            (c_hi < c_lo) ? active[i].q_hi : active[i].q_lo;
        cur_n -= 1;
        cur_prev = (i + 1 < M && active[i + 1].k == active[i].k + 1) ? 1 : 0;
      } else {
        cur_prev = 0;
      }
    }
  }

  int32_t final_nzeros = static_cast<int32_t>(prefix_nzeros + best_n);
  if (final_nzeros == 0) {
    final_nzeros = rescue_quad0();
  }
  return final_nzeros;
}

// NOTE: caller takes care of extracting quant from rect of RawQuantField.
void QuantizeBlockAC(const Quantizer& quantizer, const bool error_diffusion,
                     size_t c, float qm_multiplier, AcStrategyType quant_kind,
                     size_t xsize, size_t ysize, const float zero_thres[8],
                     float eob_scale, const float* JXL_RESTRICT block_in,
                     int32_t quant, int32_t* JXL_RESTRICT block_out,
                     const AcTokenCostModel* tcm = nullptr, size_t ord = 0,
                     size_t covered_blocks = 1, size_t log2_covered_blocks = 0,
                     int32_t predicted_nzeros = 32,
                     int32_t* out_nzeros = nullptr) {
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

  if (tcm != nullptr && eob_scale > 0.0f && xsize >= ysize) {
    const int32_t nz = TrellisQuantizeBlockAC(
        quantizer, *tcm, c, qm_multiplier, quant_kind, xsize, ysize, ord,
        covered_blocks, log2_covered_blocks, predicted_nzeros, quad_thres,
        zero_thres, eob_scale, block_in, quant, block_out);
    if (out_nzeros != nullptr) *out_nzeros = nz;
    return;
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
                               int32_t* JXL_RESTRICT quantized,
                               const AcTokenCostModel* tcm = nullptr,
                               size_t ord = 0, size_t covered_blocks = 1,
                               size_t log2_covered_blocks = 0,
                               int32_t predicted_nzeros = 32,
                               int32_t* out_nzeros = nullptr) {
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
                  quantized + size, tcm, ord, covered_blocks,
                  log2_covered_blocks, predicted_nzeros, out_nzeros);

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
      enc_state->shared.frame_dim.xsize_blocks *
              enc_state->shared.frame_dim.ysize_blocks >
          64 &&
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
    const AcTokenCostModel* JXL_RESTRICT tcm =
        has_band_qf ? enc_state->token_cost_model.get() : nullptr;
    Image3I group_nzeros;
    if (tcm != nullptr) {
      JXL_ASSIGN_OR_RETURN(
          group_nzeros,
          Image3I::Create(memory_manager, xsize_blocks, ysize_blocks));
    }

    for (size_t by = 0; by < ysize_blocks; ++by) {
      int32_t* JXL_RESTRICT row_nzeros[3] = {
          tcm ? group_nzeros.PlaneRow(0, by) : nullptr,
          tcm ? group_nzeros.PlaneRow(1, by) : nullptr,
          tcm ? group_nzeros.PlaneRow(2, by) : nullptr,
      };
      const int32_t* JXL_RESTRICT row_nzeros_top[3] = {
          (tcm && by > 0) ? group_nzeros.ConstPlaneRow(0, by - 1) : nullptr,
          (tcm && by > 0) ? group_nzeros.ConstPlaneRow(1, by - 1) : nullptr,
          (tcm && by > 0) ? group_nzeros.ConstPlaneRow(2, by - 1) : nullptr,
      };
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
            uint32_t act_bits = 0;
            if (cparams.speed_tier <= SpeedTier::kHare &&
                enc_state->band_qf[1][0].xsize() > 0 &&
                enc_state->band_qf[1][0].ysize() > 0) {
              const float bq10 =
                  block_group_rect.ConstRow(enc_state->band_qf[1][0], by)[bx];
              memcpy(&act_bits, &bq10, sizeof(act_bits));
              act_bits &= 31u;
            }
            uint32_t bits = 0;
            memcpy(&bits, &band_q[1][0], sizeof(bits));
            bits = (bits & ~31u) | act_bits;
            memcpy(&band_q[1][0], &bits, sizeof(bits));
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

          const size_t covered_blocks = cov_x * cov_y;
          const size_t log2_covered_blocks =
              Num0BitsBelowLS1Bit_Nonzero(covered_blocks);
          const size_t ord = kStrategyOrder[acs.RawStrategy()];
          const int32_t pred_nz_y =
              (tcm != nullptr)
                  ? PredictFromTopAndLeft(row_nzeros_top[1], row_nzeros[1], bx,
                                          32)
                  : 32;
          int32_t final_nz_y = 0;

          QuantizeRoundtripYBlockAC(
              enc_state, size, enc_state->shared.quantizer, error_diffusion,
              acs.Strategy(), xblocks, yblocks, zero_thres, eob_scale,
              kDefaultQuantBias, quant_ac, coeffs_in, quantized, tcm, ord,
              covered_blocks, log2_covered_blocks, pred_nz_y, &final_nz_y);
          if (tcm != nullptr) {
            const int32_t shifted_nz_y = static_cast<int32_t>(
                (final_nz_y + covered_blocks - 1) >> log2_covered_blocks);
            for (size_t iy = 0; iy < cov_y; ++iy) {
              int32_t* JXL_RESTRICT r = group_nzeros.PlaneRow(1, by + iy);
              for (size_t ix = 0; ix < cov_x; ++ix) {
                r[bx + ix] = shifted_nz_y;
              }
            }
          }

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
            const int32_t pred_nz_c =
                (tcm != nullptr)
                    ? PredictFromTopAndLeft(row_nzeros_top[c], row_nzeros[c],
                                            bx, 32)
                    : 32;
            int32_t final_nz_c = 0;
            QuantizeBlockAC(enc_state->shared.quantizer, error_diffusion, c,
                            c == 0 ? enc_state->x_qm_multiplier
                                   : enc_state->b_qm_multiplier,
                            acs.Strategy(), xblocks, yblocks, zero_thres[c],
                            eob_scale, coeffs_in + c * size, quant_ac,
                            quantized + c * size, tcm, ord, covered_blocks,
                            log2_covered_blocks, pred_nz_c, &final_nz_c);
            if (tcm != nullptr) {
              const int32_t shifted_nz_c = static_cast<int32_t>(
                  (final_nz_c + covered_blocks - 1) >> log2_covered_blocks);
              for (size_t iy = 0; iy < cov_y; ++iy) {
                int32_t* JXL_RESTRICT r = group_nzeros.PlaneRow(c, by + iy);
                for (size_t ix = 0; ix < cov_x; ++ix) {
                  r[bx + ix] = shifted_nz_c;
                }
              }
            }
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
