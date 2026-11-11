// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_rdoq.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_heuristics.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/epf.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/pack_signed.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/simd_util.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_rdoq.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/dec_transforms-inl.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/quantizer-inl.h"

#ifndef LIB_JXL_ENC_RDOQ_ONCE_INTERNAL_
#define LIB_JXL_ENC_RDOQ_ONCE_INTERNAL_
namespace jxl {
namespace {

constexpr float kKappaPix = 0.40f;
constexpr float kTileArea = 64.0f;

JXL_INLINE int32_t ClampQF(int32_t qf) {
  return Clamp1<int32_t>(qf, 1, Quantizer::kQuantMax);
}

// Target quantization weight per block matching AdjustQuantField aggregation:
// mean/max interpolation for B >= 4, max for B == 2, exact value for B == 1.
float ComputeBlockTargetQ(const ImageF& initial_quant_field, size_t bx,
                          size_t by, size_t bw, size_t bh,
                          float butteraugli_distance) {
  float qmax = initial_quant_field.Row(by)[bx];
  float qsum = 0.0f;
  for (size_t sy = 0; sy < bh; ++sy) {
    const float* row = initial_quant_field.Row(by + sy);
    for (size_t sx = 0; sx < bw; ++sx) {
      float v = row[bx + sx];
      qsum += v;
      if (v > qmax) qmax = v;
    }
  }
  const size_t num_blocks = bw * bh;
  if (num_blocks < 4) {
    return qmax;
  }
  float mean_max_mixer = 1.0f;
  static constexpr float kLimit = 1.54138f;
  static constexpr float kMul = 0.56391f;
  if (butteraugli_distance > kLimit) {
    mean_max_mixer =
        std::max(0.0f, 1.0f - (butteraugli_distance - kLimit) * kMul);
  }
  float qmean = qsum / static_cast<float>(num_blocks);
  return mean_max_mixer * qmax + (1.0f - mean_max_mixer) * qmean;
}

// Exact AC coefficient dequantization bias model matching Quantizer /
// AdjustQuantBias:
//   0 -> 0, +-1 -> +-biases[c], |k| >= 2 -> k - biases[3] / k.
JXL_INLINE float DebiasCoeffLevel(size_t c, int32_t k) {
  if (k == 0) return 0.0f;
  if (k == 1) return kDefaultQuantBias[c];
  if (k == -1) return -kDefaultQuantBias[c];
  float fk = static_cast<float>(k);
  return fk - kDefaultQuantBias[3] / fk;
}

// Perceptual channel parameters for pixel L8-quad distortion matching
// enc_ac_strategy.cc.
constexpr float kMaskOffset[3] = {12.0f, 0.0f, 4.0f};
constexpr float kChannelWeightL8[3] = {
    20441628.0f,  // pow(8.2, 8) = 20441628.0...
    1.0f,
    1.26677008f,  // pow(1.03, 8)
};

// Candidate strategies considered inside an 8x8 pixel block (1x1 block).
constexpr std::array<AcStrategyType, 10> kStrategies1x1 = {
    AcStrategyType::DCT,    AcStrategyType::DCT4X4, AcStrategyType::DCT2X2,
    AcStrategyType::DCT4X8, AcStrategyType::DCT8X4, AcStrategyType::IDENTITY,
    AcStrategyType::AFV0,   AcStrategyType::AFV1,   AcStrategyType::AFV2,
    AcStrategyType::AFV3,
};

// Banding window centered around z_ref[p].
struct TrellisBand {
  std::vector<int32_t> z_ref;
  std::vector<int32_t> z_min;
  std::vector<int32_t> z_max;
};

struct BlockTrellisWorkspace {
  std::vector<float> u;
  std::vector<float> mse_scale;
  std::vector<float> cfl_extra_k1;
  std::vector<float> cfl_extra_k2;
  std::vector<int32_t> cand_k1;
  std::vector<int32_t> cand_k2;
  std::vector<float> dist0;
  std::vector<float> dist1;
  std::vector<float> dist2;
  std::vector<float> dp_curr;
  std::vector<float> dp_next;
  std::vector<int32_t> back_k;
  TrellisBand band;
};

int32_t RoundToNearestInt(float v) {
  return static_cast<int32_t>(std::nearbyint(v));
}

struct UnmaskedHFInfo {
  bool active = false;
  float hf_max_u = 0.0f;
};

UnmaskedHFInfo DetectUnmaskedHF(AcStrategyType acs_type, const float* coeffs_in,
                                const float* inv, float qtarget_qm) {
  UnmaskedHFInfo info;
  const AcStrategy acs = AcStrategy::FromRawStrategy(acs_type);
  size_t cx = acs.covered_blocks_x();
  size_t cy = acs.covered_blocks_y();
  CoefficientLayout(&cy, &cx);
  if (acs_type != AcStrategyType::DCT && cx * cy < 2) return info;
  const size_t num_cols = cx * kBlockDim;
  const size_t num_rows = cy * kBlockDim;
  float lf_mf_max = 0.0f;
  float hf_max = 0.0f;
  for (size_t r = 0; r < num_rows; ++r) {
    for (size_t col = 0; col < num_cols; ++col) {
      if (r < cy && col < cx) continue;  // Skip LLF / DC.
      const size_t idx = r * num_cols + col;
      const float iv = inv[idx];
      if (iv <= 0.0f) continue;
      const float au = std::abs(coeffs_in[idx] * iv * qtarget_qm);
      if (r >= 3 * cy || col >= 3 * cx) {
        if (au > hf_max) hf_max = au;
      } else {
        if (au > lf_mf_max) lf_mf_max = au;
      }
    }
  }
  info.hf_max_u = hf_max;
  if (lf_mf_max < 0.35f && hf_max >= 0.32f && hf_max > 2.2f * lf_mf_max) {
    info.active = true;
  }
  return info;
}

// Computes exact backward-DP trellis RDOQ for one canonical transform block on
// channel `c` using target-normalized coefficients and rate model.
// `nzeros_out` returns total nonzero AC coefficients in the block.
RDOQTrellisResult RunTrellisRDOQCore(
    size_t c, AcStrategyType acs_type, const float* coeffs_in,
    const Quantizer& quantizer, float qm_multiplier, int32_t qf, float q_target,
    int32_t predicted_nzeros, size_t block_ctx, const BlockCtxMap& bcm,
    const coeff_order_t* order, const RDORateModel& rate_model, float lambda,
    int band_width, int multi_level, bool non_perceptual_mse,
    float jnd_zero_floor, float unmasked_lf_dist_mul,
    float unmasked_hf_dist_mul, const float* coupled_x_coeffs,
    const float* coupled_b_coeffs, float coupled_x_factor,
    float coupled_b_factor, const float* coupled_x_inv,
    const float* coupled_b_inv, float coupled_x_qm_mul, float coupled_b_qm_mul,
    float coupled_x_scale, float coupled_b_scale, bool enable_cfl_coupling,
    BlockTrellisWorkspace* ws, int32_t* quantized_out) {
  const AcStrategy acs = AcStrategy::FromRawStrategy(acs_type);
  size_t cx = acs.covered_blocks_x();
  size_t cy = acs.covered_blocks_y();
  CoefficientLayout(&cy, &cx);
  const size_t covered_blocks = cx * cy;
  const size_t log2_covered_blocks =
      Num0BitsBelowLS1Bit_Nonzero(covered_blocks);
  const size_t size = covered_blocks * kDCTBlockSize;

  std::fill_n(quantized_out, size, 0);

  const float* inv = quantizer.InvDequantMatrix(acs_type, c);
  const float* deq = quantizer.DequantMatrix(acs_type, c);
  const int32_t qf_clamped = ClampQF(qf);
  const float qac = quantizer.Scale() * static_cast<float>(qf_clamped);
  const float qac_qm = qac * qm_multiplier;
  const float qtarget_qm = q_target * qm_multiplier;
  const float s_base = (qac > 0.0f) ? (q_target / qac) : 1.0f;

  const uint32_t nz_ctx =
      bcm.NonZeroContext(static_cast<uint32_t>(std::max(0, predicted_nzeros)),
                         static_cast<uint32_t>(block_ctx));
  const size_t zd_offset =
      bcm.ZeroDensityContextsOffset(static_cast<uint32_t>(block_ctx));

  if (ws->u.size() < size) {
    ws->u.resize(size);
    ws->mse_scale.resize(size);
    ws->dist0.resize(size);
  }
  if (ws->cand_k1.size() < size) {
    ws->dist1.resize(size);
    ws->dist2.resize(size);
    ws->cand_k1.resize(size);
    ws->cand_k2.resize(size);
    ws->band.z_ref.resize(size + 1);
    ws->band.z_min.resize(size + 1);
    ws->band.z_max.resize(size + 1);
  }
  if (enable_cfl_coupling && ws->cfl_extra_k1.size() < size) {
    ws->cfl_extra_k1.resize(size);
    ws->cfl_extra_k2.resize(size);
  }

  // Convert coeffs to target-normalized representation in canonical order.
  const float sqrt_b = std::sqrt(static_cast<float>(covered_blocks));

  const size_t num_cols = cx * kBlockDim;
  const bool is_pure_dct =
      (acs_type == AcStrategyType::DCT || covered_blocks >= 2);
  bool has_mf_hf = false;
  if (is_pure_dct && unmasked_lf_dist_mul > 1.0f && !non_perceptual_mse &&
      predicted_nzeros <= 1) {
    for (size_t p = covered_blocks; p < size; ++p) {
      const coeff_order_t idx = order[p];
      const size_t r_pos = idx / num_cols;
      const size_t c_pos = idx % num_cols;
      if (r_pos >= 2 * cy || c_pos >= 2 * cx) {
        if (inv[idx] > 0.0f && std::abs(coeffs_in[idx] * inv[idx] *
                                        qtarget_qm) >= 0.52f * s_base) {
          has_mf_hf = true;
          break;
        }
      }
    }
  }
  const float eff_lf_dist_mul =
      (is_pure_dct && unmasked_lf_dist_mul > 1.0f && !non_perceptual_mse &&
       predicted_nzeros <= 1 && !has_mf_hf)
          ? unmasked_lf_dist_mul
          : 1.0f;
  const UnmaskedHFInfo unmasked_hf =
      (!non_perceptual_mse && unmasked_hf_dist_mul > 1.0f)
          ? DetectUnmaskedHF(acs_type, coeffs_in, inv, qtarget_qm)
          : UnmaskedHFInfo{};

  size_t last_active_p = covered_blocks - 1;
  float tail_dist_sum = 0.0f;
  float all_zero_dist_sum = 0.0f;

  for (size_t p = covered_blocks; p < size; ++p) {
    const coeff_order_t idx = order[p];
    const size_t r_pos = idx / num_cols;
    const size_t c_pos = idx % num_cols;
    const float w_dist =
        (unmasked_hf.active && (r_pos >= 3 * cy || c_pos >= 3 * cx))
            ? unmasked_hf_dist_mul
            : ((r_pos < 2 * cy && c_pos < 2 * cx) ? eff_lf_dist_mul : 1.0f);
    float mse_mul = 1.0f;
    if (non_perceptual_mse && covered_blocks > 1) {
      const float norm_rc =
          (r_pos == 0 ? 2.0f : 1.0f) * (c_pos == 0 ? 2.0f : 1.0f);
      mse_mul = sqrt_b * std::sqrt(norm_rc);
    } else if (non_perceptual_mse) {
      const float norm_rc =
          (r_pos == 0 ? 2.0f : 1.0f) * (c_pos == 0 ? 2.0f : 1.0f);
      mse_mul = std::sqrt(norm_rc);
    }

    const float u_val = coeffs_in[idx] * (inv[idx] * qtarget_qm) * mse_mul;
    ws->u[p] = u_val;
    ws->mse_scale[p] = mse_mul;
    const float abs_u = std::abs(u_val);
    const float abs_u_unscaled = (mse_mul > 0.0f) ? (abs_u / mse_mul) : abs_u;
    const float d0_freq = w_dist * (u_val * u_val);

    // Optional cross-channel CfL coupling at kTortoise on channel Y (c == 1).
    auto chroma_extra_cost = [&](int32_t k_y) -> float {
      if (!enable_cfl_coupling) return 0.0f;
      float recon_y_raw = 0.0f;
      if (k_y != 0 && qac_qm > 0.0f) {
        recon_y_raw = DebiasCoeffLevel(1, k_y) * deq[idx] / qac_qm;
      }
      auto eval_ch = [&](const float* ch_in, const float* ch_inv,
                         float ch_qm_mul, float ch_scale, float factor,
                         size_t ch_idx) -> float {
        (void)ch_scale;
        if (factor == 0.0f || ch_in == nullptr || ch_inv == nullptr ||
            ch_inv[idx] <= 0.0f) {
          return 0.0f;
        }
        const float res = ch_in[idx] - factor * recon_y_raw;
        const float u_ch = res * (ch_inv[idx] * q_target * ch_qm_mul) * mse_mul;
        const float eff_s_ch = s_base * mse_mul;
        const float u_over_s = (eff_s_ch > 0.0f) ? (u_ch / eff_s_ch) : 0.0f;
        int32_t k_ch = RoundToNearestInt(u_over_s);
        const float abs_uch_unscaled =
            (mse_mul > 0.0f) ? (std::abs(u_ch) / mse_mul) : std::abs(u_ch);
        const float ch_dz =
            std::max(jnd_zero_floor, 0.5f * s_base * kDefaultQuantBias[ch_idx]);
        if (abs_uch_unscaled < ch_dz) k_ch = 0;
        const float d_ch0 = u_ch * u_ch;
        if (k_ch == 0) return d_ch0;
        const float diff_ch = u_ch - eff_s_ch * DebiasCoeffLevel(ch_idx, k_ch);
        const float d_chk = diff_ch * diff_ch + lambda * 4.5f;
        return std::min(d_ch0, d_chk);
      };
      return eval_ch(coupled_x_coeffs, coupled_x_inv, coupled_x_qm_mul,
                     coupled_x_scale, coupled_x_factor, 0) +
             eval_ch(coupled_b_coeffs, coupled_b_inv, coupled_b_qm_mul,
                     coupled_b_scale, coupled_b_factor, 2);
    };

    const float extra0 = enable_cfl_coupling ? chroma_extra_cost(0) : 0.0f;
    ws->dist0[p] = d0_freq;
    all_zero_dist_sum += d0_freq;

    // Candidate 1 and Candidate 2 around u / (s * mse_mul).
    const float eff_s = s_base * mse_mul;
    const float scaled_in =
        (eff_s > 0.0f && inv[idx] > 0.0f) ? (u_val / eff_s) : 0.0f;
    const float abs_scaled = std::abs(scaled_in);
    const float eff_u_thresh = std::max(jnd_zero_floor, 0.46f * s_base);

    int32_t k1 = 0;
    int32_t k2 = 0;
    if (inv[idx] > 0.0f && abs_u_unscaled > eff_u_thresh) {
      int32_t k_near = RoundToNearestInt(scaled_in);
      if (k_near == 0) {
        k_near = (scaled_in >= 0.0f) ? 1 : -1;
      }
      k1 = k_near;
      if (multi_level > 1 && abs_scaled >= 0.35f) {
        if (std::abs(k_near) == 1) {
          k2 = (k_near > 0) ? 2 : -2;
        } else {
          float resid = scaled_in - static_cast<float>(k_near);
          k2 = (resid >= 0.0f) ? (k_near + 1) : (k_near - 1);
          if (k2 == 0) {
            k2 = (k_near > 0) ? (k_near + 1) : (k_near - 1);
          }
        }
      }
    }
    if (k1 != 0 && abs_u_unscaled < jnd_zero_floor) {
      k1 = 0;
      k2 = 0;
    }

    ws->cand_k1[p] = k1;
    ws->cand_k2[p] = k2;

    if (k1 != 0) {
      last_active_p = p;
      const float r1 = u_val - eff_s * DebiasCoeffLevel(c, k1);
      ws->dist1[p] = w_dist * (r1 * r1);
      if (enable_cfl_coupling) {
        ws->cfl_extra_k1[p] = Clamp1(0.25f * (chroma_extra_cost(k1) - extra0),
                                     -1.5f * lambda, 1.5f * lambda);
      }
    } else {
      ws->dist1[p] = std::numeric_limits<float>::infinity();
      if (enable_cfl_coupling) {
        ws->cfl_extra_k1[p] = 0.0f;
      }
    }
    if (k2 != 0) {
      const float r2 = u_val - eff_s * DebiasCoeffLevel(c, k2);
      ws->dist2[p] = w_dist * (r2 * r2);
      if (enable_cfl_coupling) {
        ws->cfl_extra_k2[p] = Clamp1(0.25f * (chroma_extra_cost(k2) - extra0),
                                     -1.5f * lambda, 1.5f * lambda);
      }
    } else {
      ws->dist2[p] = std::numeric_limits<float>::infinity();
      if (enable_cfl_coupling) {
        ws->cfl_extra_k2[p] = 0.0f;
      }
    }
  }

  const float r_zero_block = lambda * rate_model.TokenCost(nz_ctx, 0);
  if (last_active_p < covered_blocks) {
    return RDOQTrellisResult{all_zero_dist_sum + r_zero_block,
                             all_zero_dist_sum, rate_model.TokenCost(nz_ctx, 0),
                             0};
  }

  for (size_t p = last_active_p + 1; p < size; ++p) {
    tail_dist_sum += ws->dist0[p];
  }

  // Reference nonzero count from backward local RDO for narrow-band trellis.
  const size_t dp_len = last_active_p + 1;
  ws->band.z_ref[dp_len] = 0;
  for (size_t p = last_active_p; p >= covered_blocks; --p) {
    const int32_t z_next = ws->band.z_ref[p + 1];
    const int32_t k1 = ws->cand_k1[p];
    bool nz_ref = false;
    if (k1 != 0) {
      const size_t ctx1 =
          zd_offset + ZeroDensityContext(static_cast<size_t>(z_next + 1), p,
                                         covered_blocks, log2_covered_blocks,
                                         0);
      const float cfl1 = enable_cfl_coupling ? ws->cfl_extra_k1[p] : 0.0f;
      const float j1_est = ws->dist1[p] + cfl1 +
                           lambda * rate_model.TokenCost(ctx1, PackSigned(k1));
      float r0_est = 0.0f;
      if (z_next > 0) {
        const size_t ctx0 =
            zd_offset + ZeroDensityContext(static_cast<size_t>(z_next), p,
                                           covered_blocks, log2_covered_blocks,
                                           0);
        r0_est = rate_model.TokenCost(ctx0, 0);
      }
      const float j0_est = ws->dist0[p] + lambda * r0_est;
      nz_ref = (j1_est < j0_est);
    }
    ws->band.z_ref[p] = z_next + (nz_ref ? 1 : 0);
  }

  // For full precision when dp_len is modest relative to band_width, allow full
  // search up to max feasible nonzeros [0, size - p].
  const int32_t band_half = std::max(1, band_width / 2);
  ws->band.z_min[dp_len] = 0;
  ws->band.z_max[dp_len] = 0;
  for (size_t p = covered_blocks; p <= last_active_p; ++p) {
    const int32_t max_possible = static_cast<int32_t>(dp_len - p);
    int32_t lo = std::max<int32_t>(0, ws->band.z_ref[p] - band_half);
    int32_t hi = std::min<int32_t>(max_possible, ws->band.z_ref[p] + band_half);
    if (ws->band.z_ref[p] <= band_width) lo = 0;
    ws->band.z_min[p] = lo;
    ws->band.z_max[p] = hi;
  }

  // Ensure monotonic feasibility along p -> p+1:
  // Any z in [z_min[p], z_max[p]] should be able to transition to z or z-1 in
  // [z_min[p+1], z_max[p+1]] (with z_ref already feasible).
  const int32_t width = band_width + band_half + 2;
  const size_t state_stride = static_cast<size_t>(width) * 2;
  const size_t back_needed = dp_len * state_stride;
  if (ws->dp_curr.size() < state_stride) {
    ws->dp_curr.resize(state_stride);
    ws->dp_next.resize(state_stride);
  }
  if (ws->back_k.size() < back_needed) {
    ws->back_k.resize(back_needed);
  }

  constexpr float kInf = std::numeric_limits<float>::infinity();
  std::fill(ws->dp_next.begin(), ws->dp_next.begin() + state_stride, kInf);
  // Base state at p = dp_len: z = 0 is valid with prev in {0, 1}.
  ws->dp_next[0 * 2 + 0] = tail_dist_sum;
  ws->dp_next[0 * 2 + 1] = tail_dist_sum;

  for (size_t p = last_active_p; p >= covered_blocks; --p) {
    const int32_t z_lo = ws->band.z_min[p];
    const int32_t z_hi = ws->band.z_max[p];
    const int32_t zn_lo = ws->band.z_min[p + 1];
    const int32_t zn_hi = ws->band.z_max[p + 1];
    std::fill(ws->dp_curr.begin(), ws->dp_curr.begin() + state_stride, kInf);

    const float d0 = ws->dist0[p];
    const float d1 = ws->dist1[p];
    const float d2 = ws->dist2[p];
    const int32_t k1 = ws->cand_k1[p];
    const int32_t k2 = ws->cand_k2[p];
    const bool has_k1 = (k1 != 0);
    const bool has_k2 = (k2 != 0);

    const uint32_t u_zero = 0;
    const uint32_t u_k1 = has_k1 ? PackSigned(k1) : 0;
    const uint32_t u_k2 = has_k2 ? PackSigned(k2) : 0;

    int32_t* back_row = ws->back_k.data() + p * state_stride;

    for (int32_t z = z_lo; z <= z_hi; ++z) {
      const size_t zr = static_cast<size_t>(z - z_lo);
      if (zr >= static_cast<size_t>(width)) break;

      if (z == 0) {
        // When remaining nonzeros z == 0, the decoder loop has stopped:
        // only k = 0 is allowed, 0 bits coded.
        const float v0 = (zn_lo == 0) ? (d0 + ws->dp_next[0]) : kInf;
        const float v1 = (zn_lo == 0) ? (d0 + ws->dp_next[1]) : kInf;
        ws->dp_curr[zr * 2 + 0] = v0;
        ws->dp_curr[zr * 2 + 1] = v1;
        back_row[zr * 2 + 0] = 0;
        back_row[zr * 2 + 1] = 0;
        continue;
      }

      for (size_t prev_in = 0; prev_in < 2; ++prev_in) {
        const size_t ctx =
            zd_offset + ZeroDensityContext(static_cast<size_t>(z), p,
                                           covered_blocks, log2_covered_blocks,
                                           prev_in);
        float best_j = kInf;
        int32_t best_k = 0;

        // Branch 1: k = 0 -> z_next = z, prev_out = 0.
        if (z <= static_cast<int32_t>(last_active_p - p) && z >= zn_lo &&
            z <= zn_hi) {
          const size_t zn_rel = static_cast<size_t>(z - zn_lo);
          if (zn_rel < static_cast<size_t>(width)) {
            const float next_cost = ws->dp_next[zn_rel * 2 + 0];
            if (std::isfinite(next_cost)) {
              const float j0 =
                  d0 + lambda * rate_model.TokenCost(ctx, u_zero) + next_cost;
              if (j0 < best_j) {
                best_j = j0;
                best_k = 0;
              }
            }
          }
        }

        // Branch 2: k != 0 -> z_next = z - 1, prev_out = 1.
        const int32_t z_minus = z - 1;
        if (has_k1 && z_minus >= zn_lo && z_minus <= zn_hi) {
          const size_t zn_rel = static_cast<size_t>(z_minus - zn_lo);
          if (zn_rel < static_cast<size_t>(width)) {
            const float next_cost = ws->dp_next[zn_rel * 2 + 1];
            if (std::isfinite(next_cost)) {
              const float cfl1 =
                  enable_cfl_coupling ? ws->cfl_extra_k1[p] : 0.0f;
              const float j1 = d1 + cfl1 +
                               lambda * rate_model.TokenCost(ctx, u_k1) +
                               next_cost;
              if (j1 < best_j) {
                best_j = j1;
                best_k = k1;
              }
              if (has_k2) {
                const float cfl2 =
                    enable_cfl_coupling ? ws->cfl_extra_k2[p] : 0.0f;
                const float j2 = d2 + cfl2 +
                                 lambda * rate_model.TokenCost(ctx, u_k2) +
                                 next_cost;
                if (j2 < best_j) {
                  best_j = j2;
                  best_k = k2;
                }
              }
            }
          }
        }

        ws->dp_curr[zr * 2 + prev_in] = best_j;
        back_row[zr * 2 + prev_in] = best_k;
      }
    }
    ws->dp_curr.swap(ws->dp_next);
  }

  // Pick optimal total nonzeros at p = covered_blocks with NonZeroContext cost.
  const int32_t z_b_lo = ws->band.z_min[covered_blocks];
  const int32_t z_b_hi = ws->band.z_max[covered_blocks];
  float best_total = all_zero_dist_sum + r_zero_block;
  int32_t best_nz = 0;
  for (int32_t nz = z_b_lo; nz <= z_b_hi; ++nz) {
    const size_t zr = static_cast<size_t>(nz - z_b_lo);
    if (zr >= static_cast<size_t>(width)) break;
    const size_t prev_init =
        (static_cast<size_t>(nz) > size / 16) ? size_t{0} : size_t{1};
    const float body_j = ws->dp_next[zr * 2 + prev_init];
    if (!std::isfinite(body_j)) continue;
    const float nz_bits =
        rate_model.TokenCost(nz_ctx, static_cast<uint32_t>(nz));
    const float total_j = body_j + lambda * nz_bits;
    if (total_j < best_total) {
      best_total = total_j;
      best_nz = nz;
    }
  }

  // Backtrack optimal levels and recompute exact dist & rate_bits.
  int32_t z_cur = best_nz;
  size_t prev_cur =
      (static_cast<size_t>(best_nz) > size / 16) ? size_t{0} : size_t{1};
  float exact_dist = tail_dist_sum;
  float exact_rate =
      rate_model.TokenCost(nz_ctx, static_cast<uint32_t>(best_nz));

  if (best_nz == 0) {
    return RDOQTrellisResult{all_zero_dist_sum + r_zero_block,
                             all_zero_dist_sum, rate_model.TokenCost(nz_ctx, 0),
                             0};
  }

  for (size_t p = covered_blocks; p <= last_active_p; ++p) {
    if (z_cur == 0) {
      exact_dist += ws->dist0[p];
      continue;
    }
    const int32_t z_lo = ws->band.z_min[p];
    const size_t zr = static_cast<size_t>(z_cur - z_lo);
    const int32_t k = ws->back_k[p * state_stride + zr * 2 + prev_cur];
    const size_t ctx =
        zd_offset + ZeroDensityContext(static_cast<size_t>(z_cur), p,
                                       covered_blocks, log2_covered_blocks,
                                       prev_cur);
    exact_rate += rate_model.TokenCost(ctx, PackSigned(k));
    if (k == 0) {
      exact_dist += ws->dist0[p];
      prev_cur = 0;
    } else {
      quantized_out[order[p]] = k;
      if (k == ws->cand_k1[p]) {
        exact_dist += ws->dist1[p];
      } else if (k == ws->cand_k2[p]) {
        exact_dist += ws->dist2[p];
      } else {
        const coeff_order_t idx = order[p];
        const size_t r_pos = idx / num_cols;
        const size_t c_pos = idx % num_cols;
        const float w_dist =
            (unmasked_hf.active && (r_pos >= 3 * cy || c_pos >= 3 * cx))
                ? unmasked_hf_dist_mul
                : ((r_pos < 2 * cy && c_pos < 2 * cx) ? eff_lf_dist_mul : 1.0f);
        const float eff_s = s_base * ws->mse_scale[p];
        const float r = ws->u[p] - eff_s * DebiasCoeffLevel(c, k);
        exact_dist += w_dist * (r * r);
      }
      prev_cur = 1;
      --z_cur;
    }
  }
  JXL_DASSERT(z_cur == 0);
  return RDOQTrellisResult{exact_dist + lambda * exact_rate, exact_dist,
                           exact_rate, best_nz};
}

// Fast exact-rate O(size) RDO quantization used during tile search at e7 to
// avoid inner DP overhead while scoring exact J = D_freq + lambda * R_ac
// with genuine ZeroDensityContext + NonZeroContext bit costs.
RDOQTrellisResult RunFastExactRDOBlock(
    size_t c, AcStrategyType acs_type, const float* coeffs_in,
    const Quantizer& quantizer, float qm_multiplier, int32_t qf, float q_target,
    int32_t predicted_nzeros, size_t block_ctx, const BlockCtxMap& bcm,
    const coeff_order_t* order, const RDORateModel& rate_model, float lambda,
    bool non_perceptual_mse, float unmasked_hf_dist_mul,
    BlockTrellisWorkspace* ws, int32_t* quantized_out) {
  const AcStrategy acs = AcStrategy::FromRawStrategy(acs_type);
  size_t cx = acs.covered_blocks_x();
  size_t cy = acs.covered_blocks_y();
  CoefficientLayout(&cy, &cx);
  const size_t covered_blocks = cx * cy;
  const size_t log2_covered_blocks =
      Num0BitsBelowLS1Bit_Nonzero(covered_blocks);
  const size_t size = covered_blocks * kDCTBlockSize;

  std::fill_n(quantized_out, size, 0);

  const float* inv = quantizer.InvDequantMatrix(acs_type, c);
  const int32_t qf_clamped = ClampQF(qf);
  const float qac = quantizer.Scale() * static_cast<float>(qf_clamped);
  const float qtarget_qm = q_target * qm_multiplier;
  const float s_base = (qac > 0.0f) ? (q_target / qac) : 1.0f;

  const uint32_t nz_ctx =
      bcm.NonZeroContext(static_cast<uint32_t>(std::max(0, predicted_nzeros)),
                         static_cast<uint32_t>(block_ctx));
  const size_t zd_offset =
      bcm.ZeroDensityContextsOffset(static_cast<uint32_t>(block_ctx));

  if (ws->u.size() < size) {
    ws->u.resize(size);
    ws->mse_scale.resize(size);
    ws->dist0.resize(size);
  }

  const float sqrt_b = std::sqrt(static_cast<float>(covered_blocks));
  const size_t num_cols = cx * kBlockDim;
  const float deadzone = 0.5f * s_base * kDefaultQuantBias[c];

  const UnmaskedHFInfo unmasked_hf =
      (!non_perceptual_mse && unmasked_hf_dist_mul > 1.0f)
          ? DetectUnmaskedHF(acs_type, coeffs_in, inv, qtarget_qm)
          : UnmaskedHFInfo{};
  auto coeff_w_dist = [&](size_t r, size_t col) -> float {
    return (unmasked_hf.active && (r >= 3 * cy || col >= 3 * cx))
               ? unmasked_hf_dist_mul
               : 1.0f;
  };

  size_t nz = 0;
  size_t last_nz_p = 0;
  float total_dist0 = 0.0f;
  for (size_t p = covered_blocks; p < size; ++p) {
    const coeff_order_t idx = order[p];
    const size_t r = idx / num_cols;
    const size_t col = idx % num_cols;
    float mse_mul = 1.0f;
    if (non_perceptual_mse) {
      const float norm_rc = (r == 0 ? 2.0f : 1.0f) * (col == 0 ? 2.0f : 1.0f);
      mse_mul = (covered_blocks > 1 ? sqrt_b : 1.0f) * std::sqrt(norm_rc);
    }
    const float u_val = coeffs_in[idx] * (inv[idx] * qtarget_qm) * mse_mul;
    ws->u[p] = u_val;
    ws->mse_scale[p] = mse_mul;
    const float w_dist = coeff_w_dist(r, col);
    const float d0 = w_dist * (u_val * u_val);
    ws->dist0[p] = d0;
    total_dist0 += d0;

    if (inv[idx] > 0.0f) {
      const float abs_u_unscaled =
          (mse_mul > 0.0f) ? (std::abs(u_val) / mse_mul) : std::abs(u_val);
      if (abs_u_unscaled >= deadzone) {
        int32_t k = RoundToNearestInt(u_val / (s_base * mse_mul));
        if (k == 0) k = (u_val >= 0.0f) ? 1 : -1;
        quantized_out[idx] = k;
        ++nz;
        last_nz_p = p;
      }
    }
  }

  const float r0 = rate_model.TokenCost(nz_ctx, 0);
  if (nz == 0) {
    return RDOQTrellisResult{total_dist0 + lambda * r0, total_dist0, r0, 0};
  }

  // Backward RDO filter pass tracking running rem_nz so sparse multi-block
  // transforms do not keep sub-RDO nonzeros during tile selection.
  size_t back_nz = 0;
  for (size_t p = last_nz_p; p >= covered_blocks; --p) {
    const coeff_order_t idx = order[p];
    int32_t k = quantized_out[idx];
    if (k == 0) continue;
    const size_t r = idx / num_cols;
    const size_t col = idx % num_cols;
    const float w_dist = coeff_w_dist(r, col);
    const size_t ctx1 =
        zd_offset + ZeroDensityContext(back_nz + 1, p, covered_blocks,
                                       log2_covered_blocks, 0);
    const float eff_s = s_base * ws->mse_scale[p];
    const float diff = ws->u[p] - eff_s * DebiasCoeffLevel(c, k);
    float best_j = w_dist * (diff * diff) +
                   lambda * rate_model.TokenCost(ctx1, PackSigned(k));
    if (std::abs(k) > 1) {
      const int32_t k_alt = (k > 0) ? (k - 1) : (k + 1);
      const float diff_alt = ws->u[p] - eff_s * DebiasCoeffLevel(c, k_alt);
      const float j_alt =
          w_dist * (diff_alt * diff_alt) +
          lambda * rate_model.TokenCost(ctx1, PackSigned(k_alt));
      if (j_alt < best_j) {
        k = k_alt;
        best_j = j_alt;
      }
    }
    float r0_cost = 0.0f;
    if (back_nz > 0) {
      const size_t ctx0 =
          zd_offset + ZeroDensityContext(back_nz, p, covered_blocks,
                                         log2_covered_blocks, 0);
      r0_cost = rate_model.TokenCost(ctx0, 0);
    }
    const float j0 = ws->dist0[p] + lambda * r0_cost;
    if (j0 <= best_j) {
      quantized_out[idx] = 0;
    } else {
      quantized_out[idx] = k;
      ++back_nz;
    }
  }
  nz = back_nz;
  if (nz == 0) {
    return RDOQTrellisResult{total_dist0 + lambda * r0, total_dist0, r0, 0};
  }
  while (last_nz_p > covered_blocks && quantized_out[order[last_nz_p]] == 0) {
    --last_nz_p;
  }

  // Refine threshold decisions with actual sequential context bit costs and
  // test chopping small trailing +-1 nonzeros when tail zero savings win.
  size_t rem_nz = nz;
  size_t prev = (nz > size / 16) ? size_t{0} : size_t{1};
  float total_dist = 0.0f;
  float total_rate = rate_model.TokenCost(nz_ctx, static_cast<uint32_t>(nz));

  for (size_t p = covered_blocks; p <= last_nz_p; ++p) {
    const coeff_order_t idx = order[p];
    int32_t k = quantized_out[idx];
    const size_t ctx =
        zd_offset + ZeroDensityContext(rem_nz, p, covered_blocks,
                                       log2_covered_blocks, prev);
    if (k != 0) {
      const size_t r = idx / num_cols;
      const size_t col = idx % num_cols;
      const float w_dist = coeff_w_dist(r, col);
      const float eff_s = s_base * ws->mse_scale[p];
      const float diff = ws->u[p] - eff_s * DebiasCoeffLevel(c, k);
      const float dk = w_dist * (diff * diff);
      const float rk = rate_model.TokenCost(ctx, PackSigned(k));
      // Consider +/-1 magnitude refinement in-place without changing nzeros.
      if (std::abs(k) > 1) {
        int32_t k_alt = (k > 0) ? (k - 1) : (k + 1);
        const float diff_alt = ws->u[p] - eff_s * DebiasCoeffLevel(c, k_alt);
        const float dk_alt = w_dist * (diff_alt * diff_alt);
        const float rk_alt = rate_model.TokenCost(ctx, PackSigned(k_alt));
        if (dk_alt + lambda * rk_alt < dk + lambda * rk) {
          k = k_alt;
          quantized_out[idx] = k;
          total_dist += dk_alt;
          total_rate += rk_alt;
        } else {
          total_dist += dk;
          total_rate += rk;
        }
      } else {
        // Test dropping the very last nonzero coefficient if rem_nz == 1.
        if (rem_nz == 1 && ws->dist0[p] < dk + lambda * rk) {
          quantized_out[idx] = 0;
          total_dist += ws->dist0[p];
          --nz;
          --rem_nz;
          break;
        }
        total_dist += dk;
        total_rate += rk;
      }
      prev = 1;
      --rem_nz;
    } else {
      total_dist += ws->dist0[p];
      total_rate += rate_model.TokenCost(ctx, 0);
      prev = 0;
    }
  }
  for (size_t p = last_nz_p + 1; p < size; ++p) {
    total_dist += ws->dist0[p];
  }
  // Adjust NonZeroContext token if last nonzero was pruned.
  if (rem_nz == 0 && nz == 0) {
    return RDOQTrellisResult{total_dist0 + lambda * r0, total_dist0, r0, 0};
  }
  return RDOQTrellisResult{total_dist + lambda * total_rate, total_dist,
                           total_rate, static_cast<int32_t>(nz)};
}

}  // namespace
}  // namespace jxl
#endif  // LIB_JXL_ENC_RDOQ_ONCE_INTERNAL_

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

using hwy::HWY_NAMESPACE::Load;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::NegMulAdd;
using hwy::HWY_NAMESPACE::Set;
using hwy::HWY_NAMESPACE::Store;

// Computes exact reconstructed AC coefficients for channel `c` into
// `recon_coeffs` given quantized integer levels `quantized[0..size)`.
void ReconstructQuantizedCoeffs(size_t c, AcStrategyType acs_type,
                                const Quantizer& quantizer, float qm_multiplier,
                                int32_t qf, const int32_t* quantized,
                                float* recon_coeffs) {
  const AcStrategy acs = AcStrategy::FromRawStrategy(acs_type);
  const size_t size =
      acs.covered_blocks_x() * acs.covered_blocks_y() * kDCTBlockSize;
  const float* deq = quantizer.DequantMatrix(acs_type, c);
  const float inv_qac_qm = quantizer.inv_quant_ac(ClampQF(qf)) / qm_multiplier;
  for (size_t i = 0; i < size; ++i) {
    int32_t k = quantized[i];
    recon_coeffs[i] =
        (k == 0) ? 0.0f : (DebiasCoeffLevel(c, k) * deq[i] * inv_qac_qm);
  }
}

// Helper computing closed-form weighted CfL slope and optimizing discrete
// [-128, 127] CfL map coordinate over tile blocks using quantized Y_hat.
int8_t OptimizeCfLTilesDiscrete(
    const float* const* cand_orig_x_or_b, const float* const* cand_recon_y,
    const size_t* cand_sizes, const AcStrategyType* cand_types,
    size_t num_cands, const Quantizer& quantizer, size_t channel, float qm_mul,
    float q_tile, int32_t qf_tile, float base_corr, int32_t pred_m,
    const RDORateModel& rate_model, float lambda, bool tile_unmasked_hf,
    float unmasked_hf_dist_mul) {
  constexpr float kInvColorFactor = 1.0f / kDefaultColorFactor;
  const double hf_w =
      tile_unmasked_hf ? static_cast<double>(2.0f * unmasked_hf_dist_mul) : 1.0;
  double num = 0.0;
  double den = 0.0;
  for (size_t b = 0; b < num_cands; ++b) {
    const float* inv = quantizer.InvDequantMatrix(cand_types[b], channel);
    const float* inv_y = tile_unmasked_hf
                             ? quantizer.InvDequantMatrix(cand_types[b], 1)
                             : nullptr;
    const AcStrategy acs = AcStrategy::FromRawStrategy(cand_types[b]);
    size_t cx = acs.covered_blocks_x();
    size_t cy = acs.covered_blocks_y();
    CoefficientLayout(&cy, &cx);
    const size_t num_cols = cx * kBlockDim;
    const float* ry = cand_recon_y[b];
    const float* oc = cand_orig_x_or_b[b];
    const size_t sz = cand_sizes[b];
    for (size_t i = 0; i < sz; ++i) {
      float eff_inv_qm = inv[i] * qm_mul;
      double w_bin = 1.0;
      if (tile_unmasked_hf) {
        const size_t r_pos = i / num_cols;
        const size_t c_pos = i % num_cols;
        if (r_pos < cy && c_pos < cx) continue;
        if (r_pos >= 3 * cy || c_pos >= 3 * cx) {
          w_bin = hf_w;
          eff_inv_qm = std::max(eff_inv_qm, 0.25f * inv_y[i]);
        }
      }
      if (eff_inv_qm <= 0.0f) continue;
      const float w = eff_inv_qm * q_tile;
      const double w2 = w_bin * static_cast<double>(w) * static_cast<double>(w);
      const float a = ry[i] * kInvColorFactor;
      const float res0 = oc[i] - base_corr * ry[i];
      num += w2 * static_cast<double>(a) * static_cast<double>(res0);
      den += w2 * static_cast<double>(a) * static_cast<double>(a);
    }
  }
  const int32_t ls_m =
      (den > 1e-12) ? static_cast<int32_t>(
                          std::lround(Clamp1<double>(num / den, -128.0, 127.0)))
                    : 0;

  std::array<int32_t, 7> m_cands = {0,    pred_m,   ls_m - 2, ls_m - 1,
                                    ls_m, ls_m + 1, ls_m + 2};
  int32_t best_m = 0;
  double best_j = std::numeric_limits<double>::infinity();
  const float s_q = (quantizer.Scale() * qf_tile > 0.0f)
                        ? (q_tile / (quantizer.Scale() *
                                     static_cast<float>(ClampQF(qf_tile))))
                        : 1.0f;

  for (size_t ci = 0; ci < m_cands.size(); ++ci) {
    const int32_t m = Clamp1<int32_t>(m_cands[ci], -128, 127);
    bool dup = false;
    for (size_t j = 0; j < ci; ++j) {
      if (Clamp1<int32_t>(m_cands[j], -128, 127) == m) {
        dup = true;
        break;
      }
    }
    if (dup) continue;
    const float factor = base_corr + static_cast<float>(m) * kInvColorFactor;
    double dist = 0.0;
    double approx_ac_bits = 0.0;
    for (size_t b = 0; b < num_cands; ++b) {
      const float* inv = quantizer.InvDequantMatrix(cand_types[b], channel);
      const float* inv_y = tile_unmasked_hf
                               ? quantizer.InvDequantMatrix(cand_types[b], 1)
                               : nullptr;
      const AcStrategy acs = AcStrategy::FromRawStrategy(cand_types[b]);
      size_t cx = acs.covered_blocks_x();
      size_t cy = acs.covered_blocks_y();
      CoefficientLayout(&cy, &cx);
      const size_t num_cols = cx * kBlockDim;
      const float* ry = cand_recon_y[b];
      const float* oc = cand_orig_x_or_b[b];
      const size_t sz = cand_sizes[b];
      const double sparse_hf_bits =
          (tile_unmasked_hf && sz > kDCTBlockSize)
              ? (2.5 * std::log2(static_cast<double>(sz / kDCTBlockSize)))
              : 0.0;
      for (size_t i = 0; i < sz; ++i) {
        float eff_inv_qm = inv[i] * qm_mul;
        double w_bin = 1.0;
        if (tile_unmasked_hf) {
          const size_t r_pos = i / num_cols;
          const size_t c_pos = i % num_cols;
          if (r_pos < cy && c_pos < cx) continue;
          if (r_pos >= 3 * cy || c_pos >= 3 * cx) {
            w_bin = hf_w;
            eff_inv_qm = std::max(eff_inv_qm, 0.25f * inv_y[i]);
          }
        }
        if (eff_inv_qm <= 0.0f) continue;
        const float res = oc[i] - factor * ry[i];
        const float u = res * (eff_inv_qm * q_tile);
        const float scaled = u / s_q;
        if (std::abs(scaled) < 0.58f) {
          dist += w_bin * static_cast<double>(u) * u;
        } else {
          const int32_t k = RoundToNearestInt(scaled);
          const float r = u - s_q * DebiasCoeffLevel(channel, k);
          dist += w_bin * static_cast<double>(r) * r;
          approx_ac_bits +=
              2.2 + sparse_hf_bits + 0.6 * std::log2(std::abs(k) + 1.0);
        }
      }
    }
    const double j =
        dist + lambda * (approx_ac_bits + rate_model.CflCost(m, pred_m));
    if (j < best_j) {
      best_j = j;
      best_m = m;
    }
  }
  return static_cast<int8_t>(best_m);
}

// ============================================================================
// ProcessAllTilesRDO_HWY
// ============================================================================
Status ProcessAllTilesRDO_HWY(const CompressParams& cparams,
                              const FrameHeader& frame_header,
                              const Image3F& opsin, const Rect& rect,
                              const ImageF& initial_quant_field,
                              const ImageF& initial_quant_masking1x1,
                              const RDORateModel& rate_model, ThreadPool* pool,
                              PassesEncoderState* enc_state, AuxOut* aux_out) {
  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = shared.frame_dim;
  const Quantizer& quantizer = shared.quantizer;
  const BlockCtxMap& bcm = shared.block_ctx_map;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();

  const size_t xsize_blocks = frame_dim.xsize_blocks;
  const size_t ysize_blocks = frame_dim.ysize_blocks;
  const size_t n_tiles_x = DivCeil(xsize_blocks, kEncTileDimInBlocks);
  const size_t n_tiles_y = DivCeil(ysize_blocks, kEncTileDimInBlocks);
  const size_t num_tiles = n_tiles_x * n_tiles_y;

  const float butteraugli = cparams.butteraugli_distance;
  const bool is_progressive =
      (enc_state->progressive_splitter.GetNumPasses() > 1);
  const float upsample_lambda_mul =
      (frame_header.upsampling > 1 && butteraugli >= 4.0f)
          ? std::pow(static_cast<float>(frame_header.upsampling), 0.58f)
          : 1.0f;
  const float prog_lambda_mul = is_progressive ? 0.88f : 1.0f;
  const float lambda =
      ComputeRDOLambda(butteraugli) * upsample_lambda_mul * prog_lambda_mul;
  const bool non_perceptual = cparams.disable_perceptual_optimizations;
  const float unmasked_hf_dist_mul = !non_perceptual ? 14.0f : 1.0f;
  const bool is_e7 = cparams.speed_tier >= SpeedTier::kSquirrel;
  const bool is_e9 = cparams.speed_tier <= SpeedTier::kTortoise;
  const int tile_band = is_e9 ? 8 : 4;
  const int tile_multi_level = is_e9 ? 2 : 1;

  const coeff_order_t* coeff_orders0 = shared.coeff_orders.data();
  const size_t opsin_stride = opsin.PixelsPerRow();

  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;
  const size_t tile_coeff_area =
      kEncTileDimInBlocks * kEncTileDimInBlocks * kDCTBlockSize;

  struct TileDpChoice {
    bool is_split = false;
    bool horiz_split = false;
    uint8_t split_pos = 0;
    AcStrategyType type = AcStrategyType::DCT;
    int32_t qf = 1;
  };
  struct TileChosenBlock {
    uint8_t x, y;
    AcStrategyType type;
    int32_t qf;
  };
  struct TileThreadMem {
    AlignedMemory float_buf;
    AlignedMemory int_buf;
    BlockTrellisWorkspace trellis_ws;
    std::vector<float> y_recon_tile;
    std::vector<float> x_orig_tile;
    std::vector<float> b_orig_tile;
    std::vector<float> win_recon_y;
    std::vector<float> win_orig_x;
    std::vector<float> win_orig_b;
    std::vector<float> dp_cost;
    std::vector<TileDpChoice> dp_choice;
    std::vector<TileChosenBlock> winning_blocks;
  };
  std::vector<TileThreadMem> thread_mems;

  const auto prepare_threads = [&](size_t num_threads) -> Status {
    thread_mems.resize(num_threads);
    const size_t f_elems =
        8 * tile_coeff_area + 3 * tile_coeff_area + dct_scratch_size;
    const size_t i_elems = 3 * tile_coeff_area;
    for (size_t t = 0; t < num_threads; ++t) {
      JXL_ASSIGN_OR_RETURN(
          thread_mems[t].float_buf,
          AlignedMemory::Create(memory_manager, f_elems * sizeof(float)));
      JXL_ASSIGN_OR_RETURN(
          thread_mems[t].int_buf,
          AlignedMemory::Create(memory_manager, i_elems * sizeof(int32_t)));
      thread_mems[t].y_recon_tile.resize(tile_coeff_area);
      thread_mems[t].x_orig_tile.resize(tile_coeff_area);
      thread_mems[t].b_orig_tile.resize(tile_coeff_area);
      thread_mems[t].win_recon_y.resize(tile_coeff_area);
      thread_mems[t].win_orig_x.resize(tile_coeff_area);
      thread_mems[t].win_orig_b.resize(tile_coeff_area);
      thread_mems[t].dp_cost.resize(9 * 9 * 9 * 9);
      thread_mems[t].dp_choice.resize(9 * 9 * 9 * 9);
    }
    return true;
  };

  struct BlockEvalResult {
    AcStrategyType type = AcStrategyType::DCT;
    int32_t qf = 1;
    float rd_cost = std::numeric_limits<float>::infinity();
    bool valid = false;
  };

  const auto process_tile = [&](uint32_t tid, size_t thread) -> Status {
    const size_t tx = tid % n_tiles_x;
    const size_t ty = tid / n_tiles_x;
    const size_t bx0 = tx * kEncTileDimInBlocks;
    const size_t by0 = ty * kEncTileDimInBlocks;
    const size_t bx1 = std::min(bx0 + kEncTileDimInBlocks, xsize_blocks);
    const size_t by1 = std::min(by0 + kEncTileDimInBlocks, ysize_blocks);
    const size_t tw = bx1 - bx0;
    const size_t th = by1 - by0;

    float* fbase = thread_mems[thread].float_buf.address<float>();
    float* orig_block = fbase;                            // 0..3
    float* res_block = orig_block + 3 * tile_coeff_area;  // 3..6
    float* err_pix = res_block + 3 * tile_coeff_area;     // 6..9
    float* err_work = err_pix + 3 * tile_coeff_area;      // 9..10
    float* recon_y = err_work + tile_coeff_area;          // 10..11
    float* dct_scratch = recon_y + tile_coeff_area;

    int32_t* quantized_buf = thread_mems[thread].int_buf.address<int32_t>();
    BlockTrellisWorkspace& ws = thread_mems[thread].trellis_ws;

    const size_t map_x_idx = tx;
    const size_t map_y_idx = ty;

    // Step A: Initial (mx, mb) estimate from quantized 8x8 DCT Y_hat across
    // all 1x1 blocks in the tile.
    float q_tile_sum = 0.0f;
    for (size_t sy = 0; sy < th; ++sy) {
      for (size_t sx = 0; sx < tw; ++sx) {
        q_tile_sum += initial_quant_field.Row(by0 + sy)[bx0 + sx];
      }
    }
    const float q_tile = q_tile_sum / static_cast<float>(tw * th);
    const int32_t qf_tile = ClampQF(
        static_cast<int32_t>(std::lround(q_tile * quantizer.InvGlobalScale())));

    UnmaskedHFInfo tile_hf_y;
    if (!non_perceptual && tw == 8 && th == 8) {
      const float q_tile64 =
          ComputeBlockTargetQ(initial_quant_field, bx0, by0, 8, 8, butteraugli);
      TransformFromPixels(
          AcStrategyType::DCT64X64,
          rect.ConstPlaneRow(opsin, 1, by0 * kBlockDim) + bx0 * kBlockDim,
          opsin_stride, orig_block, dct_scratch);
      tile_hf_y = DetectUnmaskedHF(
          AcStrategyType::DCT64X64, orig_block,
          quantizer.InvDequantMatrix(AcStrategyType::DCT64X64, 1), q_tile64);
    }

    TileThreadMem& tmem = thread_mems[thread];
    std::vector<float>& y_recon_tile = tmem.y_recon_tile;
    std::vector<float>& x_orig_tile = tmem.x_orig_tile;
    std::vector<float>& b_orig_tile = tmem.b_orig_tile;
    std::array<const float*, 64> x_ptrs;
    std::array<const float*, 64> b_ptrs;
    std::array<const float*, 64> ry_ptrs;
    std::array<size_t, 64> b_sizes;
    std::array<AcStrategyType, 64> b_types;
    size_t n_1x1 = 0;

    for (size_t sy = 0; sy < th; ++sy) {
      const size_t py = (by0 + sy) * kBlockDim;
      for (size_t sx = 0; sx < tw; ++sx) {
        const size_t px = (bx0 + sx) * kBlockDim;
        float* ob_y = y_recon_tile.data() + n_1x1 * kDCTBlockSize;
        float* ob_x = x_orig_tile.data() + n_1x1 * kDCTBlockSize;
        float* ob_b = b_orig_tile.data() + n_1x1 * kDCTBlockSize;
        TransformFromPixels(AcStrategyType::DCT,
                            rect.ConstPlaneRow(opsin, 0, py) + px, opsin_stride,
                            ob_x, dct_scratch);
        TransformFromPixels(AcStrategyType::DCT,
                            rect.ConstPlaneRow(opsin, 1, py) + px, opsin_stride,
                            recon_y, dct_scratch);
        TransformFromPixels(AcStrategyType::DCT,
                            rect.ConstPlaneRow(opsin, 2, py) + px, opsin_stride,
                            ob_b, dct_scratch);
        const float q_sub = initial_quant_field.Row(by0 + sy)[bx0 + sx];
        int32_t qf_sub = ClampQF(static_cast<int32_t>(
            std::lround(q_sub * quantizer.InvGlobalScale())));
        if (tile_hf_y.active) {
          const UnmaskedHFInfo hf_sub = DetectUnmaskedHF(
              AcStrategyType::DCT, recon_y,
              quantizer.InvDequantMatrix(AcStrategyType::DCT, 1), q_sub);
          if (hf_sub.active && hf_sub.hf_max_u > 1e-6f) {
            const float q_hf =
                q_sub * Clamp1(1.45f / hf_sub.hf_max_u, 1.20f, 2.6f);
            qf_sub = ClampQF(static_cast<int32_t>(
                std::lround(q_hf * quantizer.InvGlobalScale())));
          }
        }
        const size_t ord0 = CoeffOrderOffset(kStrategyOrder[0], 1);
        const size_t bctx_y = bcm.Context(0, qf_sub, kStrategyOrder[0], 1);
        RunFastExactRDOBlock(
            1, AcStrategyType::DCT, recon_y, quantizer, 1.0f, qf_sub, q_sub, 3,
            bctx_y, bcm, coeff_orders0 + ord0, rate_model, lambda,
            non_perceptual, unmasked_hf_dist_mul, &ws, quantized_buf);
        ReconstructQuantizedCoeffs(1, AcStrategyType::DCT, quantizer, 1.0f,
                                   qf_sub, quantized_buf, ob_y);
        x_ptrs[n_1x1] = ob_x;
        b_ptrs[n_1x1] = ob_b;
        ry_ptrs[n_1x1] = ob_y;
        b_sizes[n_1x1] = kDCTBlockSize;
        b_types[n_1x1] = AcStrategyType::DCT;
        ++n_1x1;
      }
    }

    int8_t mx = OptimizeCfLTilesDiscrete(
        x_ptrs.data(), ry_ptrs.data(), b_sizes.data(), b_types.data(), n_1x1,
        quantizer, 0, enc_state->x_qm_multiplier, q_tile, qf_tile,
        shared.cmap.base().GetBaseCorrelationX(), 0, rate_model, lambda,
        tile_hf_y.active, unmasked_hf_dist_mul);
    int8_t mb = OptimizeCfLTilesDiscrete(
        b_ptrs.data(), ry_ptrs.data(), b_sizes.data(), b_types.data(), n_1x1,
        quantizer, 2, enc_state->b_qm_multiplier, q_tile, qf_tile,
        shared.cmap.base().GetBaseCorrelationB(), 0, rate_model, lambda,
        tile_hf_y.active, unmasked_hf_dist_mul);

    // Candidate evaluator for any sub-rectangle (rx, ry, rw, rh) inside tile.
    auto eval_candidate = [&](size_t rx, size_t ry_loc, AcStrategyType type,
                              int8_t cur_mx, int8_t cur_mb) -> BlockEvalResult {
      BlockEvalResult res;
      const AcStrategy acs = AcStrategy::FromRawStrategy(type);
      const size_t rw = acs.covered_blocks_x();
      const size_t rh = acs.covered_blocks_y();
      if (rx + rw > tw || ry_loc + rh > th) return res;
      if (is_progressive &&
          (type == AcStrategyType::IDENTITY || type == AcStrategyType::AFV0 ||
           type == AcStrategyType::AFV1 || type == AcStrategyType::AFV2 ||
           type == AcStrategyType::AFV3)) {
        return res;
      }

      const size_t abs_bx = bx0 + rx;
      const size_t abs_by = by0 + ry_loc;
      const size_t size = rw * rh * kDCTBlockSize;
      const size_t py = abs_by * kBlockDim;
      const size_t px = abs_bx * kBlockDim;

      const float q_target = ComputeBlockTargetQ(initial_quant_field, abs_bx,
                                                 abs_by, rw, rh, butteraugli);
      int32_t qf = ClampQF(static_cast<int32_t>(
          std::lround(q_target * quantizer.InvGlobalScale())));

      const uint8_t raw_acs = static_cast<uint8_t>(type);
      const int ord = kStrategyOrder[raw_acs];
      const int32_t pred_nz =
          rate_model.has_empirical_ac
              ? 3
              : static_cast<int32_t>(std::min<size_t>(64, rw * rh * 3));

      // Transform Y, X, B from opsin pixels.
      for (size_t c = 0; c < 3; ++c) {
        TransformFromPixels(type, rect.ConstPlaneRow(opsin, c, py) + px,
                            opsin_stride, orig_block + c * size, dct_scratch);
      }

      if (!non_perceptual && rw * rh >= 2) {
        const UnmaskedHFInfo hf_y =
            DetectUnmaskedHF(type, orig_block + size,
                             quantizer.InvDequantMatrix(type, 1), q_target);
        if (hf_y.active && hf_y.hf_max_u > 1e-6f) {
          const float q_hf =
              q_target * Clamp1(1.45f / hf_y.hf_max_u, 1.20f, 2.6f);
          qf = ClampQF(static_cast<int32_t>(
              std::lround(q_hf * quantizer.InvGlobalScale())));
        }
      }

      // Quantize Y using trellis / fast exact RDO.
      const size_t bctx_y = bcm.Context(0, qf, ord, 1);
      int32_t* qy = quantized_buf + size;
      RDOQTrellisResult ry_res =
          is_e7 ? RunFastExactRDOBlock(1, type, orig_block + size, quantizer,
                                       1.0f, qf, q_target, pred_nz, bctx_y, bcm,
                                       coeff_orders0 + CoeffOrderOffset(ord, 1),
                                       rate_model, lambda, non_perceptual,
                                       unmasked_hf_dist_mul, &ws, qy)
                : RunTrellisRDOQCore(
                      1, type, orig_block + size, quantizer, 1.0f, qf, q_target,
                      pred_nz, bctx_y, bcm,
                      coeff_orders0 + CoeffOrderOffset(ord, 1), rate_model,
                      lambda, tile_band, tile_multi_level, non_perceptual, 0.0f,
                      1.0f, unmasked_hf_dist_mul, nullptr, nullptr, 0.0f, 0.0f,
                      nullptr, nullptr, 1.0f, 1.0f, 1.0f, 1.0f, false, &ws, qy);

      ReconstructQuantizedCoeffs(1, type, quantizer, 1.0f, qf, qy, recon_y);

      // Subtract CfL prediction using quantized Y_hat for X and B.
      const float x_fac = shared.cmap.base().YtoXRatio(cur_mx);
      const float b_fac = shared.cmap.base().YtoBRatio(cur_mb);
      for (size_t i = 0; i < size; ++i) {
        res_block[i] = orig_block[i] - x_fac * recon_y[i];
        res_block[2 * size + i] = orig_block[2 * size + i] - b_fac * recon_y[i];
        res_block[size + i] = orig_block[size + i];
      }

      const size_t bctx_x = bcm.Context(0, qf, ord, 0);
      const size_t bctx_b = bcm.Context(0, qf, ord, 2);
      RDOQTrellisResult rx_res =
          is_e7 ? RunFastExactRDOBlock(0, type, res_block, quantizer,
                                       enc_state->x_qm_multiplier, qf, q_target,
                                       pred_nz, bctx_x, bcm,
                                       coeff_orders0 + CoeffOrderOffset(ord, 0),
                                       rate_model, lambda, non_perceptual,
                                       unmasked_hf_dist_mul, &ws, quantized_buf)
                : RunTrellisRDOQCore(
                      0, type, res_block, quantizer, enc_state->x_qm_multiplier,
                      qf, q_target, pred_nz, bctx_x, bcm,
                      coeff_orders0 + CoeffOrderOffset(ord, 0), rate_model,
                      lambda, tile_band, tile_multi_level, non_perceptual, 0.0f,
                      1.0f, unmasked_hf_dist_mul, nullptr, nullptr, 0.0f, 0.0f,
                      nullptr, nullptr, 1.0f, 1.0f, 1.0f, 1.0f, false, &ws,
                      quantized_buf);

      RDOQTrellisResult rb_res =
          is_e7 ? RunFastExactRDOBlock(
                      2, type, res_block + 2 * size, quantizer,
                      enc_state->b_qm_multiplier, qf, q_target, pred_nz, bctx_b,
                      bcm, coeff_orders0 + CoeffOrderOffset(ord, 2), rate_model,
                      lambda, non_perceptual, unmasked_hf_dist_mul, &ws,
                      quantized_buf + 2 * size)
                : RunTrellisRDOQCore(
                      2, type, res_block + 2 * size, quantizer,
                      enc_state->b_qm_multiplier, qf, q_target, pred_nz, bctx_b,
                      bcm, coeff_orders0 + CoeffOrderOffset(ord, 2), rate_model,
                      lambda, tile_band, tile_multi_level, non_perceptual, 0.0f,
                      1.0f, unmasked_hf_dist_mul, nullptr, nullptr, 0.0f, 0.0f,
                      nullptr, nullptr, 1.0f, 1.0f, 1.0f, 1.0f, false, &ws,
                      quantized_buf + 2 * size);

      float d_freq =
          non_perceptual ? 0.0f : (rx_res.dist + ry_res.dist + rb_res.dist);
      const float r_ac = ry_res.rate_bits + rx_res.rate_bits + rb_res.rate_bits;
      float d_pix = 0.0f;

      {
        // Compute reconstructed residual errors for X, Y, B into `res_block`,
        // zero canonical LLF positions, and inverse-transform to `err_pix`.
        ReconstructQuantizedCoeffs(0, type, quantizer,
                                   enc_state->x_qm_multiplier, qf,
                                   quantized_buf, err_work);
        for (size_t i = 0; i < size; ++i) res_block[i] -= err_work[i];
        // Note for Y: res_block[size..2*size) held orig_y!
        for (size_t i = 0; i < size; ++i) res_block[size + i] -= recon_y[i];
        ReconstructQuantizedCoeffs(2, type, quantizer,
                                   enc_state->b_qm_multiplier, qf,
                                   quantized_buf + 2 * size, err_work);
        for (size_t i = 0; i < size; ++i)
          res_block[2 * size + i] -= err_work[i];
        size_t lcx = acs.covered_blocks_x();
        size_t lcy = acs.covered_blocks_y();
        CoefficientLayout(&lcy, &lcx);
        const size_t px_s = rw * kBlockDim;
        for (size_t c = 0; c < 3; ++c) {
          std::copy_n(res_block + c * size, size, err_work);
          for (size_t iy = 0; iy < lcy; ++iy) {
            for (size_t ix = 0; ix < lcx; ++ix) {
              err_work[iy * lcx * kBlockDim + ix] = 0.0f;
            }
          }
          TransformToPixels(type, err_work, err_pix + c * size, px_s,
                            dct_scratch);
        }
        // Accumulate L8-quad (perceptual) / MSE (non-perceptual) over all
        // 8x8 sub-blocks.
        const bool has_mask = (initial_quant_masking1x1.xsize() > 0 &&
                               initial_quant_masking1x1.ysize() > 0);
        const size_t mask_x_max =
            has_mask ? (initial_quant_masking1x1.xsize() - 1) : 0;
        const size_t mask_y_max =
            has_mask ? (initial_quant_masking1x1.ysize() - 1) : 0;
        if (non_perceptual) {
          for (size_t sy = 0; sy < rh; ++sy) {
            const float* q_row = initial_quant_field.Row(abs_by + sy);
            for (size_t sx = 0; sx < rw; ++sx) {
              const float q_sub = q_row[abs_bx + sx];
              double e2 = 0.0;
              for (size_t c = 0; c < 3; ++c) {
                const float* ep = err_pix + c * size;
                for (size_t dy = 0; dy < kBlockDim; ++dy) {
                  const float* er =
                      ep + (sy * kBlockDim + dy) * px_s + sx * kBlockDim;
                  for (size_t dx = 0; dx < kBlockDim; ++dx) {
                    e2 += static_cast<double>(er[dx]) * er[dx];
                  }
                }
              }
              const double qs = 1000.0 * static_cast<double>(q_sub);
              d_pix += static_cast<float>((1.0 / 16.0) * qs * qs * e2);
            }
          }
        } else {
          const double x_ring_w =
              (rw * rh >= 2)
                  ? static_cast<double>(
                        1.0f +
                        std::min(3.0f, static_cast<float>(rw * rh) * 0.125f))
                  : 1.0;
          const double cw0 =
              static_cast<double>(kChannelWeightL8[0]) * x_ring_w;
          const double cw1 = static_cast<double>(kChannelWeightL8[1]);
          const double cw2 = static_cast<double>(kChannelWeightL8[2]);
          const double cw[3] = {cw0, cw1, cw2};
          const float inv_d = 1.0f / std::max(0.25f, butteraugli);
          const float d_scale = std::min(1.0f, lambda / 0.0465f);
          double total_sum8 = 0.0;
          for (size_t sy = 0; sy < rh; ++sy) {
            const float* q_row = initial_quant_field.Row(abs_by + sy);
            for (size_t sx = 0; sx < rw; ++sx) {
              const float q_sub = q_row[abs_bx + sx];
              for (size_t dy = 0; dy < kBlockDim; ++dy) {
                const size_t mpy =
                    std::min((abs_by + sy) * kBlockDim + dy, mask_y_max);
                const float* mrow =
                    has_mask ? initial_quant_masking1x1.ConstRow(mpy) : nullptr;
                for (size_t dx = 0; dx < kBlockDim; ++dx) {
                  const float mask_val =
                      has_mask ? mrow[std::min((abs_bx + sx) * kBlockDim + dx,
                                               mask_x_max)]
                               : (1.0f / (q_sub + 0.001f));
                  const size_t off =
                      (sy * kBlockDim + dy) * px_s + sx * kBlockDim + dx;
                  for (size_t c = 0; c < 3; ++c) {
                    const float sc =
                        (mask_val + kMaskOffset[c]) * err_pix[c * size + off];
                    const float sc2 = sc * sc;
                    const float sc4 = sc2 * sc2;
                    total_sum8 += cw[c] * static_cast<double>(sc4 * sc4);
                  }
                }
              }
            }
          }
          const double mean8 =
              total_sum8 / static_cast<double>(rw * rh * kDCTBlockSize);
          d_pix = d_scale * inv_d * static_cast<float>(rw * rh) *
                  static_cast<float>(std::sqrt(std::sqrt(std::sqrt(mean8))));
        }
        // Equalize distortion scale against canonical DCT8 CSF weights across
        // all non-DCT8 strategies (1x1 sub-8x8 transforms like DCT4X4/AFV as
        // well as multi-block transforms and IDENTITY).
        if (!non_perceptual &&
            (tile_hf_y.active ||
             !(rw == 1 && rh == 1 && type == AcStrategyType::DCT))) {
          const float dct8_hf_w =
              tile_hf_y.active ? (2.0f * unmasked_hf_dist_mul) : 1.0f;
          std::array<float, 3> d_dct8 = {0.0f, 0.0f, 0.0f};
          for (size_t c = 0; c < 3; ++c) {
            const float qm_c =
                (c == 0) ? enc_state->x_qm_multiplier
                         : ((c == 2) ? enc_state->b_qm_multiplier : 1.0f);
            const float* inv_dct8 =
                quantizer.InvDequantMatrix(AcStrategyType::DCT, c);
            const float inv0 = inv_dct8[1];
            float dist_c = 0.0f;
            for (size_t sy = 0; sy < rh; ++sy) {
              const float* q_row = initial_quant_field.Row(abs_by + sy);
              for (size_t sx = 0; sx < rw; ++sx) {
                const float q_sub = q_row[abs_bx + sx];
                const float w_scale = q_sub * qm_c;
                const float* sub_pix = err_pix + c * size +
                                       (sy * kBlockDim) * px_s + sx * kBlockDim;
                TransformFromPixels(AcStrategyType::DCT, sub_pix, px_s,
                                    err_work, dct_scratch);
                for (size_t i = 0; i < kDCTBlockSize; ++i) {
                  const float inv_w = (i == 0)
                                          ? (0.5f * kSqrt2 * inv0 * w_scale)
                                          : (inv_dct8[i] * w_scale);
                  const float u_err = err_work[i] * inv_w;
                  const float hw = (tile_hf_y.active &&
                                    (i / kBlockDim >= 3 || i % kBlockDim >= 3))
                                       ? dct8_hf_w
                                       : 1.0f;
                  dist_c += hw * u_err * u_err;
                }
              }
            }
            d_dct8[c] = dist_c;
          }
          if (rw == 1 && rh == 1 && !tile_hf_y.active) {
            d_freq = d_dct8[0] + d_dct8[1] + d_dct8[2];
          } else {
            d_freq = std::max(rx_res.dist, d_dct8[0]) +
                     std::max(ry_res.dist, d_dct8[1]) +
                     std::max(rb_res.dist, d_dct8[2]);
          }
        }
      }

      const float r_meta =
          rate_model.AcsCost(raw_acs, 0) + rate_model.QfDeltaCost(qf, qf_tile);
      res.type = type;
      res.qf = qf;
      res.rd_cost = (non_perceptual ? d_pix : (d_freq + kKappaPix * d_pix)) +
                    lambda * (r_ac + r_meta);
      res.valid = true;
      return res;
    };

    auto solve_tile_dp = [&](int8_t cur_mx, int8_t cur_mb,
                             std::vector<TileChosenBlock>* out_blocks) {
      out_blocks->clear();
      // For each valid sub-rectangle [x0, x1) x [y0, y1) inside [0, tw) x [0,
      // th): Index (x0, x1, y0, y1) in 9^4 <= 6561 table -> only compute over
      // x1>x0, y1>y0 (<= 1296).
      auto rect_idx = [](size_t x0, size_t x1, size_t y0, size_t y1) -> size_t {
        return (((y0 * 9 + y1) * 9 + x0) * 9 + x1);
      };
      std::vector<float>& dp_cost = tmem.dp_cost;
      std::fill(dp_cost.begin(), dp_cost.end(),
                std::numeric_limits<float>::infinity());
      std::vector<TileDpChoice>& dp_choice = tmem.dp_choice;
      std::fill(dp_choice.begin(), dp_choice.end(), TileDpChoice{});

      // Helper recording single block evaluation at (x0, y0, rw, rh).
      auto try_rect_strategy = [&](size_t x0, size_t y0, AcStrategyType type) {
        const AcStrategy acs = AcStrategy::FromRawStrategy(type);
        const size_t x1 = x0 + acs.covered_blocks_x();
        const size_t y1 = y0 + acs.covered_blocks_y();
        if (x1 > tw || y1 > th) return;
        const BlockEvalResult r = eval_candidate(x0, y0, type, cur_mx, cur_mb);
        if (!r.valid) return;
        const size_t idx = rect_idx(x0, x1, y0, y1);
        if (r.rd_cost < dp_cost[idx]) {
          dp_cost[idx] = r.rd_cost;
          dp_choice[idx] = TileDpChoice{false, false, 0, r.type, r.qf};
        }
      };

      const bool use_dyadic = is_e7 && (tw == 8) && (th == 8);
      if (use_dyadic) {
        // Exact 225-state Dyadic Guillotine Partition DP for e7 on full 8x8
        // tiles. Leaf transforms over aligned dyadic sub-rectangles.
        for (size_t y0 = 0; y0 < 8; ++y0) {
          for (size_t x0 = 0; x0 < 8; ++x0) {
            for (AcStrategyType t : kStrategies1x1) {
              try_rect_strategy(x0, y0, t);
            }
          }
        }
        constexpr std::array<AcStrategyType, 11> kDyadicMulti = {
            AcStrategyType::DCT16X8,  AcStrategyType::DCT8X16,
            AcStrategyType::DCT16X16, AcStrategyType::DCT32X8,
            AcStrategyType::DCT8X32,  AcStrategyType::DCT32X16,
            AcStrategyType::DCT16X32, AcStrategyType::DCT32X32,
            AcStrategyType::DCT64X32, AcStrategyType::DCT32X64,
            AcStrategyType::DCT64X64,
        };
        for (AcStrategyType t : kDyadicMulti) {
          const AcStrategy acs = AcStrategy::FromRawStrategy(t);
          const size_t w = acs.covered_blocks_x();
          const size_t h = acs.covered_blocks_y();
          for (size_t y0 = 0; y0 + h <= 8; y0 += h) {
            for (size_t x0 = 0; x0 + w <= 8; x0 += w) {
              try_rect_strategy(x0, y0, t);
            }
          }
        }
        // Combine dyadic sub-rectangles in increasing area order.
        for (size_t h : {size_t{1}, size_t{2}, size_t{4}, size_t{8}}) {
          for (size_t w : {size_t{1}, size_t{2}, size_t{4}, size_t{8}}) {
            if (w == 1 && h == 1) continue;
            for (size_t y0 = 0; y0 < 8; y0 += h) {
              for (size_t x0 = 0; x0 < 8; x0 += w) {
                const size_t idx = rect_idx(x0, x0 + w, y0, y0 + h);
                if (w >= 2) {
                  const size_t xm = x0 + w / 2;
                  const float sp = dp_cost[rect_idx(x0, xm, y0, y0 + h)] +
                                   dp_cost[rect_idx(xm, x0 + w, y0, y0 + h)];
                  if (sp < dp_cost[idx]) {
                    dp_cost[idx] = sp;
                    dp_choice[idx] =
                        TileDpChoice{true, false, static_cast<uint8_t>(xm),
                                     AcStrategyType::DCT, 1};
                  }
                }
                if (h >= 2) {
                  const size_t ym = y0 + h / 2;
                  const float sp = dp_cost[rect_idx(x0, x0 + w, y0, ym)] +
                                   dp_cost[rect_idx(x0, x0 + w, ym, y0 + h)];
                  if (sp < dp_cost[idx]) {
                    dp_cost[idx] = sp;
                    dp_choice[idx] =
                        TileDpChoice{true, true, static_cast<uint8_t>(ym),
                                     AcStrategyType::DCT, 1};
                  }
                }
              }
            }
          }
        }
      } else {
        // Exact Full Guillotine Partition DP (all [x0, x1) x [y0, y1) inside
        // [0, tw) x [0, th)) with shifted multi-block strides matching e8 vs e9
        // (and partial edge tiles).
        for (size_t y0 = 0; y0 < th; ++y0) {
          for (size_t x0 = 0; x0 < tw; ++x0) {
            for (AcStrategyType t : kStrategies1x1) {
              try_rect_strategy(x0, y0, t);
            }
          }
        }
        const size_t s2 = 1;
        const size_t s4 = is_e9 ? 1 : 2;
        const size_t s8 = is_e9 ? 2 : 4;
        struct ShiftedDef {
          AcStrategyType t;
          size_t step_x, step_y;
        };
        const std::array<ShiftedDef, 11> kShifted = {{
            {AcStrategyType::DCT16X8, s2, s2},
            {AcStrategyType::DCT8X16, s2, s2},
            {AcStrategyType::DCT16X16, s2, s2},
            {AcStrategyType::DCT32X8, s4, 1},
            {AcStrategyType::DCT8X32, 1, s4},
            {AcStrategyType::DCT32X16, s4, s4},
            {AcStrategyType::DCT16X32, s4, s4},
            {AcStrategyType::DCT32X32, s4, s4},
            {AcStrategyType::DCT64X32, 1, s8},
            {AcStrategyType::DCT32X64, s8, 1},
            {AcStrategyType::DCT64X64, 1, 1},
        }};
        for (const auto& sd : kShifted) {
          const AcStrategy acs = AcStrategy::FromRawStrategy(sd.t);
          const size_t w = acs.covered_blocks_x();
          const size_t h = acs.covered_blocks_y();
          const size_t sx = is_e7 ? w : sd.step_x;
          const size_t sy = is_e7 ? h : sd.step_y;
          for (size_t y0 = 0; y0 + h <= th; y0 += sy) {
            for (size_t x0 = 0; x0 + w <= tw; x0 += sx) {
              try_rect_strategy(x0, y0, sd.t);
            }
          }
        }

        // Solve general Guillotine DP by increasing width and height.
        for (size_t h = 1; h <= th; ++h) {
          for (size_t w = 1; w <= tw; ++w) {
            if (w == 1 && h == 1) continue;
            for (size_t y0 = 0; y0 + h <= th; ++y0) {
              const size_t y1 = y0 + h;
              for (size_t x0 = 0; x0 + w <= tw; ++x0) {
                const size_t x1 = x0 + w;
                const size_t idx = rect_idx(x0, x1, y0, y1);
                float best = dp_cost[idx];
                TileDpChoice best_ch = dp_choice[idx];
                for (size_t xm = x0 + 1; xm < x1; ++xm) {
                  const float sp = dp_cost[rect_idx(x0, xm, y0, y1)] +
                                   dp_cost[rect_idx(xm, x1, y0, y1)];
                  if (sp < best) {
                    best = sp;
                    best_ch =
                        TileDpChoice{true, false, static_cast<uint8_t>(xm),
                                     AcStrategyType::DCT, 1};
                  }
                }
                for (size_t ym = y0 + 1; ym < y1; ++ym) {
                  const float sp = dp_cost[rect_idx(x0, x1, y0, ym)] +
                                   dp_cost[rect_idx(x0, x1, ym, y1)];
                  if (sp < best) {
                    best = sp;
                    best_ch = TileDpChoice{true, true, static_cast<uint8_t>(ym),
                                           AcStrategyType::DCT, 1};
                  }
                }
                dp_cost[idx] = best;
                dp_choice[idx] = best_ch;
              }
            }
          }
        }
      }

      // Recursively extract winning partition.
      auto extract = [&](auto& self, size_t x0, size_t x1, size_t y0,
                         size_t y1) -> void {
        const TileDpChoice& ch = dp_choice[rect_idx(x0, x1, y0, y1)];
        if (!ch.is_split) {
          out_blocks->push_back(TileChosenBlock{static_cast<uint8_t>(x0),
                                                static_cast<uint8_t>(y0),
                                                ch.type, ch.qf});
        } else if (ch.horiz_split) {
          self(self, x0, x1, y0, ch.split_pos);
          self(self, x0, x1, ch.split_pos, y1);
        } else {
          self(self, x0, ch.split_pos, y0, y1);
          self(self, ch.split_pos, x1, y0, y1);
        }
      };
      extract(extract, 0, tw, 0, th);
    };

    std::vector<TileChosenBlock>& winning_blocks = tmem.winning_blocks;
    solve_tile_dp(mx, mb, &winning_blocks);

    // Step D: Re-optimize (mx, mb) over the winning tile partition using actual
    // winning quantized Y_hat.
    std::vector<float>& win_recon_y = tmem.win_recon_y;
    std::vector<float>& win_orig_x = tmem.win_orig_x;
    std::vector<float>& win_orig_b = tmem.win_orig_b;
    std::array<const float*, 64> w_x_ptrs, w_b_ptrs, w_ry_ptrs;
    std::array<size_t, 64> w_sizes;
    std::array<AcStrategyType, 64> w_types;
    size_t w_off = 0;
    size_t w_cnt = 0;
    for (const TileChosenBlock& cb : winning_blocks) {
      const AcStrategy acs = AcStrategy::FromRawStrategy(cb.type);
      const size_t rw = acs.covered_blocks_x();
      const size_t rh = acs.covered_blocks_y();
      const size_t sz = rw * rh * kDCTBlockSize;
      const size_t px = (bx0 + cb.x) * kBlockDim;
      const size_t py = (by0 + cb.y) * kBlockDim;
      const float q_target = ComputeBlockTargetQ(
          initial_quant_field, bx0 + cb.x, by0 + cb.y, rw, rh, butteraugli);
      const int ord = kStrategyOrder[static_cast<uint8_t>(cb.type)];
      TransformFromPixels(cb.type, rect.ConstPlaneRow(opsin, 0, py) + px,
                          opsin_stride, win_orig_x.data() + w_off, dct_scratch);
      TransformFromPixels(cb.type, rect.ConstPlaneRow(opsin, 1, py) + px,
                          opsin_stride, orig_block, dct_scratch);
      TransformFromPixels(cb.type, rect.ConstPlaneRow(opsin, 2, py) + px,
                          opsin_stride, win_orig_b.data() + w_off, dct_scratch);
      const size_t bctx_y = bcm.Context(0, cb.qf, ord, 1);
      const int32_t pred_nz_d =
          rate_model.has_empirical_ac
              ? 3
              : static_cast<int32_t>(std::min<size_t>(64, rw * rh * 3));
      RunFastExactRDOBlock(
          1, cb.type, orig_block, quantizer, 1.0f, cb.qf, q_target, pred_nz_d,
          bctx_y, bcm, coeff_orders0 + CoeffOrderOffset(ord, 1), rate_model,
          lambda, non_perceptual, unmasked_hf_dist_mul, &ws, quantized_buf);
      ReconstructQuantizedCoeffs(1, cb.type, quantizer, 1.0f, cb.qf,
                                 quantized_buf, win_recon_y.data() + w_off);
      w_x_ptrs[w_cnt] = win_orig_x.data() + w_off;
      w_b_ptrs[w_cnt] = win_orig_b.data() + w_off;
      w_ry_ptrs[w_cnt] = win_recon_y.data() + w_off;
      w_sizes[w_cnt] = sz;
      w_types[w_cnt] = cb.type;
      w_off += sz;
      ++w_cnt;
    }

    const int8_t mx_refined = OptimizeCfLTilesDiscrete(
        w_x_ptrs.data(), w_ry_ptrs.data(), w_sizes.data(), w_types.data(),
        w_cnt, quantizer, 0, enc_state->x_qm_multiplier, q_tile, qf_tile,
        shared.cmap.base().GetBaseCorrelationX(), 0, rate_model, lambda,
        tile_hf_y.active, unmasked_hf_dist_mul);
    const int8_t mb_refined = OptimizeCfLTilesDiscrete(
        w_b_ptrs.data(), w_ry_ptrs.data(), w_sizes.data(), w_types.data(),
        w_cnt, quantizer, 2, enc_state->b_qm_multiplier, q_tile, qf_tile,
        shared.cmap.base().GetBaseCorrelationB(), 0, rate_model, lambda,
        tile_hf_y.active, unmasked_hf_dist_mul);

    const int cfl_re_dp_thresh = tile_hf_y.active ? 1 : (is_e7 ? 6 : 3);
    if (std::abs(mx_refined - mx) >= cfl_re_dp_thresh ||
        std::abs(mb_refined - mb) >= cfl_re_dp_thresh) {
      mx = mx_refined;
      mb = mb_refined;
      solve_tile_dp(mx, mb, &winning_blocks);
    } else {
      mx = mx_refined;
      mb = mb_refined;
    }

    // Commit winning tile AC strategies, initial raw_quant_field, and CfL map.
    shared.cmap.ytox_map.Row(map_y_idx)[map_x_idx] = mx;
    shared.cmap.ytob_map.Row(map_y_idx)[map_x_idx] = mb;
    for (const TileChosenBlock& cb : winning_blocks) {
      JXL_RETURN_IF_ERROR(
          shared.ac_strategy.Set(bx0 + cb.x, by0 + cb.y, cb.type));
      const AcStrategy acs = AcStrategy::FromRawStrategy(cb.type);
      for (size_t iy = 0; iy < acs.covered_blocks_y(); ++iy) {
        int32_t* qrow = shared.raw_quant_field.Row(by0 + cb.y + iy);
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ++ix) {
          qrow[bx0 + cb.x + ix] = cb.qf;
        }
      }
    }
    return true;
  };

  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_tiles, prepare_threads,
                                process_tile, "ProcessAllTilesRDO"));

  if (aux_out != nullptr) {
    const AcStrategyImage& ac_strategy = shared.ac_strategy;
    aux_out->num_small_blocks =
        ac_strategy.CountBlocks(AcStrategyType::IDENTITY) +
        ac_strategy.CountBlocks(AcStrategyType::DCT2X2) +
        ac_strategy.CountBlocks(AcStrategyType::DCT4X4);
    aux_out->num_dct4x8_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT4X8) +
        ac_strategy.CountBlocks(AcStrategyType::DCT8X4);
    aux_out->num_afv_blocks = ac_strategy.CountBlocks(AcStrategyType::AFV0) +
                              ac_strategy.CountBlocks(AcStrategyType::AFV1) +
                              ac_strategy.CountBlocks(AcStrategyType::AFV2) +
                              ac_strategy.CountBlocks(AcStrategyType::AFV3);
    aux_out->num_dct8_blocks = ac_strategy.CountBlocks(AcStrategyType::DCT);
    aux_out->num_dct8x16_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT8X16) +
        ac_strategy.CountBlocks(AcStrategyType::DCT16X8);
    aux_out->num_dct8x32_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT8X32) +
        ac_strategy.CountBlocks(AcStrategyType::DCT32X8);
    aux_out->num_dct16_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT16X16);
    aux_out->num_dct16x32_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT16X32) +
        ac_strategy.CountBlocks(AcStrategyType::DCT32X16);
    aux_out->num_dct32_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT32X32);
    aux_out->num_dct32x64_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT32X64) +
        ac_strategy.CountBlocks(AcStrategyType::DCT64X32);
    aux_out->num_dct64_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT64X64);
  }
  return true;
}

// ============================================================================
// ComputeCoefficientsRDOQ_HWY
// ============================================================================
Status ComputeCoefficientsRDOQ_HWY(const FrameHeader& frame_header,
                                   const Image3F& opsin, const Rect& rect,
                                   const ImageF& initial_quant_field,
                                   const ImageF& initial_quant_masking1x1,
                                   const RDORateModel& rate_model,
                                   ThreadPool* pool,
                                   PassesEncoderState* enc_state) {
  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = shared.frame_dim;
  const Quantizer& quantizer = shared.quantizer;
  const BlockCtxMap& bcm = shared.block_ctx_map;
  const CompressParams& cparams = enc_state->cparams;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();

  const size_t num_passes = enc_state->progressive_splitter.GetNumPasses();
  JXL_ENSURE(num_passes > 0);
  const float butteraugli = cparams.butteraugli_distance;
  const float upsample_lambda_mul =
      (frame_header.upsampling > 1 && butteraugli >= 4.0f)
          ? std::pow(static_cast<float>(frame_header.upsampling), 0.58f)
          : 1.0f;
  const float prog_lambda_mul = (num_passes > 1) ? 0.88f : 1.0f;
  const float lambda =
      ComputeRDOLambda(butteraugli) * upsample_lambda_mul * prog_lambda_mul;
  const bool non_perceptual = cparams.disable_perceptual_optimizations;
  const float unmasked_lf_dist_mul = !non_perceptual ? 1.6f : 1.0f;
  const float unmasked_hf_dist_mul = !non_perceptual ? 14.0f : 1.0f;

  const int band_width = (cparams.speed_tier <= SpeedTier::kTortoise) ? 24
                         : (cparams.speed_tier <= SpeedTier::kKitten) ? 8
                                                                      : 6;
  const int multi_level = 2;
  const bool is_e8_plus = (cparams.speed_tier <= SpeedTier::kKitten);
  const bool enable_cfl_coupling = true;
  const size_t opsin_stride = opsin.PixelsPerRow();
  const coeff_order_t* coeff_orders0 = shared.coeff_orders.data();

  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;

  struct CoeffThreadMem {
    AlignedMemory float_buf;
    AlignedMemory int_buf;
    Image3I num_nzeroes;
    BlockTrellisWorkspace ws;
  };
  std::vector<CoeffThreadMem> thread_mems;

  const auto prepare_threads = [&](size_t num_threads) -> Status {
    thread_mems.resize(num_threads);
    const size_t f_elems = 10 * AcStrategy::kMaxCoeffArea + dct_scratch_size;
    const size_t i_elems = 6 * AcStrategy::kMaxCoeffArea;
    for (size_t t = 0; t < num_threads; ++t) {
      JXL_ASSIGN_OR_RETURN(
          thread_mems[t].float_buf,
          AlignedMemory::Create(memory_manager, f_elems * sizeof(float)));
      JXL_ASSIGN_OR_RETURN(
          thread_mems[t].int_buf,
          AlignedMemory::Create(memory_manager, i_elems * sizeof(int32_t)));
      JXL_ASSIGN_OR_RETURN(thread_mems[t].num_nzeroes,
                           Image3I::Create(memory_manager, kGroupDimInBlocks,
                                           kGroupDimInBlocks));
    }
    return true;
  };

  constexpr HWY_CAPPED(float, kDCTBlockSize) d_vec;

  const auto process_group = [&](uint32_t group_idx, size_t thread) -> Status {
    const Rect block_group_rect = frame_dim.BlockGroupRect(group_idx);
    const Rect cmap_rect(
        block_group_rect.x0() / kColorTileDimInBlocks,
        block_group_rect.y0() / kColorTileDimInBlocks,
        DivCeil(block_group_rect.xsize(), kColorTileDimInBlocks),
        DivCeil(block_group_rect.ysize(), kColorTileDimInBlocks));
    const Rect group_rect =
        frame_dim.GroupRect(group_idx).Translate(rect.x0(), rect.y0());

    const size_t xsize_blocks = block_group_rect.xsize();
    const size_t ysize_blocks = block_group_rect.ysize();

    float* fbase = thread_mems[thread].float_buf.address<float>();
    float* orig_coeffs = fbase;                                       // 0..3
    float* work_coeff = orig_coeffs + 3 * AcStrategy::kMaxCoeffArea;  // 3..4
    float* err_pix = work_coeff + AcStrategy::kMaxCoeffArea;          // 4..7
    float* cand_recon_y = err_pix + 3 * AcStrategy::kMaxCoeffArea;    // 7..8
    float* res_xb = cand_recon_y + AcStrategy::kMaxCoeffArea;         // 8..10
    float* dct_scratch = res_xb + 2 * AcStrategy::kMaxCoeffArea;      // 10..

    int32_t* quantized =
        thread_mems[thread].int_buf.address<int32_t>();  // 0..3
    int32_t* best_quantized =
        quantized + 3 * AcStrategy::kMaxCoeffArea;  // 3..6
    Image3I& num_nzeroes = thread_mems[thread].num_nzeroes;
    const size_t nzeros_stride = num_nzeroes.PixelsPerRow();
    BlockTrellisWorkspace& ws = thread_mems[thread].ws;

    int32_t* coeffs_out[3][kMaxNumPasses] = {};
    for (size_t p = 0; p < num_passes; ++p) {
      JXL_ENSURE(enc_state->coeffs[p]->Type() == ACType::k32);
      for (size_t c = 0; c < 3; ++c) {
        coeffs_out[c][p] =
            enc_state->coeffs[p]->PlaneRow(c, group_idx, 0).ptr32;
      }
    }

    const YCbCrChromaSubsampling& cs = frame_header.chroma_subsampling;
    int32_t prev_qf = -1;

    for (size_t by = 0; by < ysize_blocks; ++by) {
      const size_t abs_by = block_group_rect.y0() + by;
      const size_t ty = by / kColorTileDimInBlocks;
      const int8_t* row_cmap_x = cmap_rect.ConstRow(shared.cmap.ytox_map, ty);
      const int8_t* row_cmap_b = cmap_rect.ConstRow(shared.cmap.ytob_map, ty);
      const float* opsin_rows[3] = {
          group_rect.ConstPlaneRow(opsin, 0, by * kBlockDim),
          group_rect.ConstPlaneRow(opsin, 1, by * kBlockDim),
          group_rect.ConstPlaneRow(opsin, 2, by * kBlockDim),
      };
      const uint8_t* row_qdc =
          shared.quant_dc.ConstRow(abs_by) + block_group_rect.x0();
      AcStrategyRow acs_row = shared.ac_strategy.ConstRow(block_group_rect, by);

      int32_t* row_nzeros[3] = {
          num_nzeroes.PlaneRow(0, by),
          num_nzeroes.PlaneRow(1, by),
          num_nzeroes.PlaneRow(2, by),
      };
      const int32_t* row_nzeros_top[3] = {
          by == 0 ? nullptr : num_nzeroes.ConstPlaneRow(0, by - 1),
          by == 0 ? nullptr : num_nzeroes.ConstPlaneRow(1, by - 1),
          by == 0 ? nullptr : num_nzeroes.ConstPlaneRow(2, by - 1),
      };

      for (size_t tx = 0; tx < DivCeil(xsize_blocks, kColorTileDimInBlocks);
           ++tx) {
        const float x_factor = shared.cmap.base().YtoXRatio(row_cmap_x[tx]);
        const float b_factor = shared.cmap.base().YtoBRatio(row_cmap_b[tx]);
        const auto x_factor_v = Set(d_vec, x_factor);
        const auto b_factor_v = Set(d_vec, b_factor);

        for (size_t bx = tx * kColorTileDimInBlocks;
             bx < xsize_blocks && bx < (tx + 1) * kColorTileDimInBlocks; ++bx) {
          const AcStrategy acs = acs_row[bx];
          if (!acs.IsFirstBlock()) continue;

          const size_t abs_bx = block_group_rect.x0() + bx;
          size_t xblocks = acs.covered_blocks_x();
          size_t yblocks = acs.covered_blocks_y();
          CoefficientLayout(&yblocks, &xblocks);
          const size_t covered_blocks = xblocks * yblocks;
          const size_t log2_covered_blocks =
              Num0BitsBelowLS1Bit_Nonzero(covered_blocks);
          const size_t size = kDCTBlockSize * covered_blocks;

          for (size_t c : {0, 1, 2}) {
            TransformFromPixels(acs.Strategy(), opsin_rows[c] + bx * kBlockDim,
                                opsin_stride, orig_coeffs + c * size,
                                dct_scratch);
          }

          const float q_target = ComputeBlockTargetQ(
              initial_quant_field, abs_bx, abs_by, acs.covered_blocks_x(),
              acs.covered_blocks_y(), butteraugli);
          const int32_t qf_ideal = ClampQF(static_cast<int32_t>(
              std::lround(q_target * quantizer.InvGlobalScale())));
          if (prev_qf < 0) {
            prev_qf = qf_ideal;
          }

          const int ord = kStrategyOrder[acs.RawStrategy()];
          const int32_t pred_nz[3] = {
              PredictFromTopAndLeft(row_nzeros_top[0], row_nzeros[0], bx, 32),
              PredictFromTopAndLeft(row_nzeros_top[1], row_nzeros[1], bx, 32),
              PredictFromTopAndLeft(row_nzeros_top[2], row_nzeros[2], bx, 32),
          };

          // Clamp AQ smoothing candidates to avoid large mismatch against
          // local qf_ideal (e.g. smooth gradient next to high-activity region).
          const int32_t qf_min_safe = ClampQF(std::min(
              qf_ideal, static_cast<int32_t>(
                            std::ceil(static_cast<float>(qf_ideal) / 1.08f))));
          const int32_t qf_max_safe = ClampQF(
              std::max(qf_ideal, static_cast<int32_t>(std::lround(
                                     static_cast<float>(qf_ideal) * 1.20f))));

          int32_t qf_hf = qf_ideal;
          if (!non_perceptual) {
            const UnmaskedHFInfo hf_y = DetectUnmaskedHF(
                acs.Strategy(), orig_coeffs + size,
                quantizer.InvDequantMatrix(acs.Strategy(), 1), q_target);
            if (hf_y.active && hf_y.hf_max_u > 1e-6f) {
              const float q_hf =
                  q_target * Clamp1(1.45f / hf_y.hf_max_u, 1.20f, 2.6f);
              qf_hf = ClampQF(static_cast<int32_t>(
                  std::lround(q_hf * quantizer.InvGlobalScale())));
            }
          }

          // Build deadzone-widening AQ candidate list for qf.
          std::array<int32_t, 5> qf_cands;
          size_t num_qf_cands = 0;
          qf_cands[num_qf_cands++] = qf_ideal;
          qf_cands[num_qf_cands++] =
              Clamp1<int32_t>(prev_qf, qf_min_safe, qf_max_safe);
          qf_cands[num_qf_cands++] = Clamp1<int32_t>(
              ClampQF((qf_ideal + prev_qf + 1) / 2), qf_min_safe, qf_max_safe);
          if (is_e8_plus) {
            qf_cands[num_qf_cands++] =
                Clamp1<int32_t>(ClampQF((2 * prev_qf + qf_ideal + 1) / 3),
                                qf_min_safe, qf_max_safe);
          }
          if (qf_hf != qf_ideal) {
            qf_cands[num_qf_cands++] = qf_hf;
          }

          int32_t best_qf = qf_ideal;
          float best_total_j = std::numeric_limits<float>::infinity();
          int32_t best_nz_y = 0, best_nz_x = 0, best_nz_b = 0;

          const float* inv_x = quantizer.InvDequantMatrix(acs.Strategy(), 0);
          const float* inv_b = quantizer.InvDequantMatrix(acs.Strategy(), 2);

          for (size_t qi = 0; qi < num_qf_cands; ++qi) {
            const int32_t cand_qf = ClampQF(qf_cands[qi]);
            bool dup = false;
            for (size_t pj = 0; pj < qi; ++pj) {
              if (ClampQF(qf_cands[pj]) == cand_qf) {
                dup = true;
                break;
              }
            }
            if (dup) continue;

            const float qac_cand =
                quantizer.Scale() * static_cast<float>(cand_qf);
            const float q_eff = (cand_qf == qf_ideal)
                                    ? std::max(q_target, 0.92f * qac_cand)
                                    : q_target;
            const float s_cand = (qac_cand > 0.0f) ? (q_eff / qac_cand) : 1.0f;
            // When keeping a finer prev_qf (s_cand < 1) in perceptual mode,
            // widen the effective deadzone on sub-JND noise so AQ avoids paying
            // delta_qf bits where it wouldn't make a visible difference.
            const float jnd_floor = (!non_perceptual && s_cand < 0.999f &&
                                     cand_qf != qf_ideal && cand_qf != qf_hf)
                                        ? 0.48f
                                        : 0.0f;

            const uint8_t qdc_val = row_qdc[bx];
            const size_t bctx_y = bcm.Context(qdc_val, cand_qf, ord, 1);
            const size_t bctx_x = bcm.Context(qdc_val, cand_qf, ord, 0);
            const size_t bctx_b = bcm.Context(qdc_val, cand_qf, ord, 2);

            const RDOQTrellisResult res_y = RunTrellisRDOQCore(
                1, acs.Strategy(), orig_coeffs + size, quantizer, 1.0f, cand_qf,
                q_eff, pred_nz[1], bctx_y, bcm,
                coeff_orders0 + CoeffOrderOffset(ord, 1), rate_model, lambda,
                band_width, multi_level, non_perceptual, jnd_floor,
                unmasked_lf_dist_mul, unmasked_hf_dist_mul, orig_coeffs,
                orig_coeffs + 2 * size, x_factor, b_factor, inv_x, inv_b,
                enc_state->x_qm_multiplier, enc_state->b_qm_multiplier, s_cand,
                s_cand, enable_cfl_coupling, &ws, quantized + size);

            ReconstructQuantizedCoeffs(1, acs.Strategy(), quantizer, 1.0f,
                                       cand_qf, quantized + size, cand_recon_y);

            // Unapply CfL using quantized Y_hat.
            for (size_t k = 0; k < size; k += Lanes(d_vec)) {
              const auto in_x = Load(d_vec, orig_coeffs + k);
              const auto in_b = Load(d_vec, orig_coeffs + 2 * size + k);
              const auto ry = Load(d_vec, cand_recon_y + k);
              Store(NegMulAdd(x_factor_v, ry, in_x), d_vec, res_xb + k);
              Store(NegMulAdd(b_factor_v, ry, in_b), d_vec, res_xb + size + k);
            }

            const RDOQTrellisResult res_x = RunTrellisRDOQCore(
                0, acs.Strategy(), res_xb, quantizer,
                enc_state->x_qm_multiplier, cand_qf, q_eff, pred_nz[0], bctx_x,
                bcm, coeff_orders0 + CoeffOrderOffset(ord, 0), rate_model,
                lambda, band_width, multi_level, non_perceptual, jnd_floor,
                unmasked_lf_dist_mul, unmasked_hf_dist_mul, nullptr, nullptr,
                0.0f, 0.0f, nullptr, nullptr, 1.0f, 1.0f, 1.0f, 1.0f, false,
                &ws, quantized);

            const RDOQTrellisResult res_b = RunTrellisRDOQCore(
                2, acs.Strategy(), res_xb + size, quantizer,
                enc_state->b_qm_multiplier, cand_qf, q_eff, pred_nz[2], bctx_b,
                bcm, coeff_orders0 + CoeffOrderOffset(ord, 2), rate_model,
                lambda, band_width, multi_level, non_perceptual, jnd_floor,
                unmasked_lf_dist_mul, unmasked_hf_dist_mul, nullptr, nullptr,
                0.0f, 0.0f, nullptr, nullptr, 1.0f, 1.0f, 1.0f, 1.0f, false,
                &ws, quantized + 2 * size);

            float d_pix = 0.0f;
            float cand_freq = res_y.dist + res_x.dist + res_b.dist;
            float cand_rate =
                res_y.rate_bits + res_x.rate_bits + res_b.rate_bits;

            {
              // Transform per-channel residual error coefficients into
              // spatial pixel errors (`err_pix`) for L8-quad / MSE scoring.
              size_t lcx = acs.covered_blocks_x();
              size_t lcy = acs.covered_blocks_y();
              CoefficientLayout(&lcy, &lcx);
              const size_t px_s = acs.covered_blocks_x() * kBlockDim;
              for (size_t ch : {0, 1, 2}) {
                if (ch == 1) {
                  for (size_t i = 0; i < size; ++i) {
                    work_coeff[i] = orig_coeffs[size + i] - cand_recon_y[i];
                  }
                } else {
                  const float qm_m = (ch == 0) ? enc_state->x_qm_multiplier
                                               : enc_state->b_qm_multiplier;
                  const float* res_src = (ch == 0) ? res_xb : (res_xb + size);
                  ReconstructQuantizedCoeffs(ch, acs.Strategy(), quantizer,
                                             qm_m, cand_qf,
                                             quantized + ch * size, work_coeff);
                  for (size_t i = 0; i < size; ++i) {
                    work_coeff[i] = res_src[i] - work_coeff[i];
                  }
                }
                for (size_t iy = 0; iy < lcy; ++iy) {
                  for (size_t ix = 0; ix < lcx; ++ix) {
                    work_coeff[iy * lcx * kBlockDim + ix] = 0.0f;
                  }
                }
                TransformToPixels(acs.Strategy(), work_coeff,
                                  err_pix + ch * size, px_s, dct_scratch);
              }
              const bool has_mask = (initial_quant_masking1x1.xsize() > 0 &&
                                     initial_quant_masking1x1.ysize() > 0);
              const size_t mask_x_max =
                  has_mask ? (initial_quant_masking1x1.xsize() - 1) : 0;
              const size_t mask_y_max =
                  has_mask ? (initial_quant_masking1x1.ysize() - 1) : 0;
              const size_t bw = acs.covered_blocks_x();
              const size_t bh = acs.covered_blocks_y();
              if (non_perceptual) {
                for (size_t sy = 0; sy < bh; ++sy) {
                  const float* q_row = initial_quant_field.Row(abs_by + sy);
                  for (size_t sx = 0; sx < bw; ++sx) {
                    const float q_sub = q_row[abs_bx + sx];
                    double e2 = 0.0;
                    for (size_t c = 0; c < 3; ++c) {
                      const float* ep = err_pix + c * size;
                      for (size_t dy = 0; dy < kBlockDim; ++dy) {
                        const float* er =
                            ep + (sy * kBlockDim + dy) * px_s + sx * kBlockDim;
                        for (size_t dx = 0; dx < kBlockDim; ++dx) {
                          e2 += static_cast<double>(er[dx]) * er[dx];
                        }
                      }
                    }
                    const double qs = 1000.0 * static_cast<double>(q_sub);
                    d_pix += static_cast<float>((1.0 / 16.0) * qs * qs * e2);
                  }
                }
              } else {
                const double x_ring_w =
                    (bw * bh >= 2)
                        ? static_cast<double>(
                              1.0f +
                              std::min(3.0f,
                                       static_cast<float>(bw * bh) * 0.125f))
                        : 1.0;
                const double cw0 =
                    static_cast<double>(kChannelWeightL8[0]) * x_ring_w;
                const double cw1 = static_cast<double>(kChannelWeightL8[1]);
                const double cw2 = static_cast<double>(kChannelWeightL8[2]);
                const double cw[3] = {cw0, cw1, cw2};
                const float inv_d = 1.0f / std::max(0.25f, butteraugli);
                const float d_scale = std::min(1.0f, lambda / 0.0465f);
                double total_sum8 = 0.0;
                for (size_t sy = 0; sy < bh; ++sy) {
                  const float* q_row = initial_quant_field.Row(abs_by + sy);
                  for (size_t sx = 0; sx < bw; ++sx) {
                    const float q_sub = q_row[abs_bx + sx];
                    for (size_t dy = 0; dy < kBlockDim; ++dy) {
                      const size_t mpy =
                          std::min((abs_by + sy) * kBlockDim + dy, mask_y_max);
                      const float* mrow =
                          has_mask ? initial_quant_masking1x1.ConstRow(mpy)
                                   : nullptr;
                      for (size_t dx = 0; dx < kBlockDim; ++dx) {
                        const float mask_val =
                            has_mask
                                ? mrow[std::min((abs_bx + sx) * kBlockDim + dx,
                                                mask_x_max)]
                                : (1.0f / (q_sub + 0.001f));
                        const size_t off =
                            (sy * kBlockDim + dy) * px_s + sx * kBlockDim + dx;
                        for (size_t c = 0; c < 3; ++c) {
                          const float sc = (mask_val + kMaskOffset[c]) *
                                           err_pix[c * size + off];
                          const float sc2 = sc * sc;
                          const float sc4 = sc2 * sc2;
                          total_sum8 += cw[c] * static_cast<double>(sc4 * sc4);
                        }
                      }
                    }
                  }
                }
                const double mean8 =
                    total_sum8 / static_cast<double>(bw * bh * kDCTBlockSize);
                d_pix =
                    d_scale * inv_d * static_cast<float>(bw * bh) *
                    static_cast<float>(std::sqrt(std::sqrt(std::sqrt(mean8))));
              }
            }

            const float cand_j =
                (non_perceptual ? d_pix : (cand_freq + kKappaPix * d_pix)) +
                lambda * (cand_rate + rate_model.QfDeltaCost(cand_qf, prev_qf));
            if (cand_j < best_total_j) {
              best_total_j = cand_j;
              best_qf = cand_qf;
              best_nz_y = res_y.nzeros;
              best_nz_x = res_x.nzeros;
              best_nz_b = res_b.nzeros;
              std::copy_n(quantized, 3 * size, best_quantized);
            }
          }

          prev_qf = best_qf;

          // Write winning qf across ALL covered_blocks_y * covered_blocks_x
          // cells in shared.raw_quant_field.
          for (size_t iy = 0; iy < acs.covered_blocks_y(); ++iy) {
            int32_t* qrow = shared.raw_quant_field.Row(abs_by + iy);
            for (size_t ix = 0; ix < acs.covered_blocks_x(); ++ix) {
              qrow[abs_bx + ix] = best_qf;
            }
          }

          // Update group-local num_nzeroes identically to TokenizeCoefficients.
          const int32_t nz_arr[3] = {best_nz_x, best_nz_y, best_nz_b};
          for (size_t c = 0; c < 3; ++c) {
            const int32_t shifted =
                (covered_blocks == 1)
                    ? nz_arr[c]
                    : static_cast<int32_t>((nz_arr[c] + covered_blocks - 1) >>
                                           log2_covered_blocks);
            for (size_t iy = 0; iy < acs.covered_blocks_y(); ++iy) {
              for (size_t ix = 0; ix < acs.covered_blocks_x(); ++ix) {
                num_nzeroes.PlaneRow(c, by + iy)[bx + ix] = shifted;
              }
            }
            (void)nzeros_stride;
          }
          (void)cs;

          // Split into progressive passes in enc_state->coeffs.
          for (size_t c = 0; c < 3; ++c) {
            enc_state->progressive_splitter.SplitACCoefficients(
                best_quantized + c * size, acs, bx, by, coeffs_out[c]);
            for (size_t p = 0; p < num_passes; ++p) {
              coeffs_out[c][p] += size;
            }
          }
        }
      }
    }
    return true;
  };

  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_groups, prepare_threads,
                                process_group, "ComputeCoefficientsRDOQ"));
  return true;
}

Status ComputeDCFromOpsin_HWY(const Image3F& opsin, const Rect& rect,
                              const AcStrategyImage& ac_strategy,
                              ThreadPool* pool, Image3F* dc) {
  JxlMemoryManager* memory_manager = dc->memory_manager();
  const size_t xsize_blocks = dc->xsize();
  const size_t ysize_blocks = dc->ysize();
  const size_t opsin_stride = opsin.PixelsPerRow();
  const size_t dc_stride = dc->PixelsPerRow();
  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;
  const size_t f_elems = AcStrategy::kMaxCoeffArea + dct_scratch_size;

  std::vector<AlignedMemory> thread_mems;
  const auto prepare_threads = [&](size_t num_threads) -> Status {
    thread_mems.resize(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
      JXL_ASSIGN_OR_RETURN(
          thread_mems[t],
          AlignedMemory::Create(memory_manager, f_elems * sizeof(float)));
    }
    return true;
  };

  const auto process_row = [&](uint32_t by, size_t thread) -> Status {
    float* block = thread_mems[thread].address<float>();
    float* scratch = block + AcStrategy::kMaxCoeffArea;
    AcStrategyRow acs_row = ac_strategy.ConstRow(by);
    const size_t py = by * kBlockDim;
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      const AcStrategy acs = acs_row[bx];
      if (!acs.IsFirstBlock()) continue;
      const size_t px = bx * kBlockDim;
      for (size_t c = 0; c < 3; ++c) {
        TransformFromPixels(acs.Strategy(),
                            rect.ConstPlaneRow(opsin, c, py) + px, opsin_stride,
                            block, scratch);
        DCFromLowestFrequencies(acs.Strategy(), block, dc->PlaneRow(c, by) + bx,
                                dc_stride, scratch);
      }
    }
    return true;
  };

  return RunOnPool(pool, 0, ysize_blocks, prepare_threads, process_row,
                   "ComputeDCFromOpsin");
}

}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
namespace {
HWY_EXPORT(ProcessAllTilesRDO_HWY);
HWY_EXPORT(ComputeCoefficientsRDOQ_HWY);
HWY_EXPORT(ComputeDCFromOpsin_HWY);
}  // namespace

float ComputeRDOLambda(float butteraugli_distance) {
  const float d = Clamp1(butteraugli_distance, 0.12f, 12.0f);
  return 0.0465f * std::pow(d, -0.44f);
}

// ============================================================================
// RDORateModel implementation
// ============================================================================

void RDORateModel::InitPreset(const BlockCtxMap& bcm,
                              float butteraugli_distance,
                              uint32_t decoding_speed_tier) {
  (void)decoding_speed_tier;
  has_empirical_ac = false;
  uint_configs.clear();
  context_map.clear();
  sym_bits.clear();
  unseen_base = 12.0f;

  num_ac_contexts = bcm.NumACContexts();
  fast_cost.assign(num_ac_contexts * kFastTableSize, 0.0f);

  const size_t num_block_ctxs = bcm.num_ctxs;
  const size_t nz_ctx_total = num_block_ctxs * kNonZeroBuckets;

  // NonZeroContext LUT [0 .. num_block_ctxs * kNonZeroBuckets - 1].
  for (size_t bucket = 0; bucket < kNonZeroBuckets; ++bucket) {
    const float pred = (bucket < 8)
                           ? static_cast<float>(bucket)
                           : static_cast<float>((bucket - 4) * 2) + 0.5f;
    const float mu = std::max(0.25f, pred);
    const float sigma =
        1.8f + 0.42f * mu + 0.35f * Clamp1(butteraugli_distance, 0.2f, 8.0f);
    std::array<float, kFastTableSize> probs;
    double sum = 0.0;
    for (size_t u = 0; u < kFastTableSize; ++u) {
      double p = std::exp(-std::abs(static_cast<double>(u) - mu) / sigma);
      if (u == 0 && bucket <= 2) {
        p *= (bucket == 0 ? 1.8 : (bucket == 1 ? 1.3 : 1.1));
      }
      probs[u] = static_cast<float>(p);
      sum += p;
    }
    const double inv_sum = 1.0 / sum;
    for (size_t bc = 0; bc < num_block_ctxs; ++bc) {
      const size_t ctx = bucket * num_block_ctxs + bc;
      for (size_t u = 0; u < kFastTableSize; ++u) {
        const double prob = std::max(1e-5, probs[u] * inv_sum);
        fast_cost[ctx * kFastTableSize + u] =
            static_cast<float>(-std::log2(prob));
      }
    }
  }

  // Compute geometric-mean covered_blocks per block context to avoid
  // over-pricing zero runs in grp == 0 on multi-block transforms (B >= 2).
  std::vector<double> eff_b_log_sum(num_block_ctxs, 0.0);
  std::vector<int> eff_b_cnt(num_block_ctxs, 0);
  const size_t num_qf_idx = bcm.qf_thresholds.size() + 1;
  for (size_t c_idx = 0; c_idx < 3; ++c_idx) {
    for (size_t ord = 0; ord < kNumOrders; ++ord) {
      const size_t cov_b =
          kCoeffOrderOffset[3 * ord + 1] - kCoeffOrderOffset[3 * ord];
      const double log_b = std::log(static_cast<double>(cov_b));
      for (size_t qf_i = 0; qf_i < num_qf_idx; ++qf_i) {
        for (size_t dc_i = 0; dc_i < bcm.num_dc_ctxs; ++dc_i) {
          const size_t idx = ((c_idx * kNumOrders + ord) * num_qf_idx + qf_i) *
                                 bcm.num_dc_ctxs +
                             dc_i;
          if (idx < bcm.ctx_map.size()) {
            const size_t bc = bcm.ctx_map[idx];
            if (bc < num_block_ctxs) {
              eff_b_log_sum[bc] += log_b;
              eff_b_cnt[bc] += 1;
            }
          }
        }
      }
    }
  }
  std::vector<float> eff_b(num_block_ctxs, 1.0f);
  for (size_t bc = 0; bc < num_block_ctxs; ++bc) {
    if (eff_b_cnt[bc] > 0) {
      eff_b[bc] = static_cast<float>(
          std::exp(eff_b_log_sum[bc] / static_cast<double>(eff_b_cnt[bc])));
    }
  }

  // ZeroDensityContext LUT [nz_ctx_total .. num_ac_contexts - 1].
  static constexpr int kNzBucketStarts[8] = {0, 31, 62, 93, 123, 152, 180, 206};
  static constexpr float kNzRep[8] = {1.0f,  2.0f,  3.5f,  6.2f,
                                      10.5f, 16.0f, 26.0f, 44.0f};
  std::array<float, 63> spec_w;
  std::array<float, 63> spec_tail;
  for (int j = 0; j < 63; ++j) {
    spec_w[j] = std::pow(static_cast<float>(j) + 2.0f, -1.25f);
  }
  spec_tail[62] = spec_w[62];
  for (int j = 61; j >= 0; --j) {
    spec_tail[j] = spec_w[j] + spec_tail[j + 1];
  }
  for (size_t zd = 0; zd < kZeroDensityContextCount; ++zd) {
    const size_t prev = zd & 1;
    const int comb = static_cast<int>(zd >> 1);
    int grp = 0;
    for (int g = 7; g >= 0; --g) {
      if (comb >= kNzBucketStarts[g]) {
        grp = g;
        break;
      }
    }
    const int freq_c = Clamp1(comb - kNzBucketStarts[grp], 0, 30);
    int k_pos = 0;
    if (freq_c < 15) {
      k_pos = freq_c;
    } else if (freq_c < 23) {
      k_pos = 15 + 2 * (freq_c - 15);
    } else {
      k_pos = std::min(62, 31 + 4 * (freq_c - 23) + 1);
    }
    const float h1 = spec_w[k_pos] / std::max(1e-6f, spec_tail[k_pos]);
    const float rem = std::max(2.0f, 64.0f - static_cast<float>(k_pos + 1));
    const float h_eff = (grp >= 5) ? std::max(h1, 0.75f / rem) : h1;
    float p_any = 1.0f - std::pow(std::max(0.0f, 1.0f - h_eff), kNzRep[grp]);
    if (grp >= 6) {
      p_any = std::max(p_any, std::min(0.91f, 0.80f * kNzRep[grp] / rem));
    }
    const float p_nz = Clamp1(p_any * (prev ? 1.22f : 0.88f), 0.03f, 0.91f);
    const double alpha = 2.35 + 0.035 * static_cast<double>(freq_c);

    std::array<double, kFastTableSize> nz_norm;
    nz_norm[0] = 0.0;
    double nz_weight_sum = 0.0;
    for (size_t u = 1; u < kFastTableSize; ++u) {
      const size_t mag = (u + 1) >> 1;
      const double w = 0.5 * std::pow(static_cast<double>(mag) + 0.85, -alpha);
      nz_norm[u] = w;
      nz_weight_sum += w;
    }
    const double inv_nz_wsum = 1.0 / nz_weight_sum;
    for (size_t u = 1; u < kFastTableSize; ++u) {
      nz_norm[u] *= inv_nz_wsum;
    }

    for (size_t bc = 0; bc < num_block_ctxs; ++bc) {
      const size_t ctx = nz_ctx_total + bc * kZeroDensityContextCount + zd;
      const float p_nz_bc =
          (grp == 0 && eff_b[bc] > 1.05f)
              ? Clamp1(p_nz * std::pow(eff_b[bc], -0.65f), 0.0015f, 0.91f)
              : p_nz;
      const double p0_prob = std::max(1e-6, 1.0 - static_cast<double>(p_nz_bc));
      fast_cost[ctx * kFastTableSize + 0] =
          static_cast<float>(-std::log2(p0_prob));
      for (size_t u = 1; u < kFastTableSize; ++u) {
        const double prob =
            std::max(1e-6, static_cast<double>(p_nz_bc) * nz_norm[u]);
        fast_cost[ctx * kFastTableSize + u] =
            static_cast<float>(-std::log2(prob));
      }
    }
  }

  // Preset kACMeta cost tables.
  for (size_t ctx = 0; ctx < 4; ++ctx) {
    for (size_t s = 0; s < AcStrategy::kNumValidStrategies; ++s) {
      float base = 4.0f;
      const auto t = static_cast<AcStrategyType>(s);
      if (t == AcStrategyType::DCT) {
        base = 1.8f;
      } else if (t == AcStrategyType::DCT16X16 ||
                 t == AcStrategyType::DCT8X16 || t == AcStrategyType::DCT16X8) {
        base = 3.2f;
      } else if (t == AcStrategyType::DCT4X8 || t == AcStrategyType::DCT8X4) {
        base = 4.2f;
      } else if (t == AcStrategyType::DCT2X2 || t == AcStrategyType::DCT4X4 ||
                 t == AcStrategyType::AFV0 || t == AcStrategyType::AFV1 ||
                 t == AcStrategyType::AFV2 || t == AcStrategyType::AFV3) {
        base = 4.5f;
      } else if (t == AcStrategyType::IDENTITY) {
        base = 5.4f;
      }
      acs_cost_table[ctx][s] = (ctx == 1) ? std::max(0.8f, base - 1.0f) : base;
    }
    qf_zero_cost[ctx] = 0.75f;
    qf_nonzero_base[ctx] = 3.4f;
    for (size_t e = 0; e < 8; ++e) {
      epf_cost_table[ctx][e] = (ctx == 3 && e == 4) ? 1.5f : 3.2f;
    }
  }
  cfl_zero_cost = 0.8f;
  cfl_nonzero_base = 3.3f;
}

void RDORateModel::InitFromEmpiricalPass(const BlockCtxMap& bcm,
                                         const EntropyEncodingData& codes,
                                         const AcStrategyImage& ac_strategy,
                                         const ImageI& raw_quant_field,
                                         const ColorCorrelationMap& cmap,
                                         const ImageB& epf_sharpness,
                                         const FrameDimensions& frame_dim) {
  InitPreset(bcm, 1.0f, 0);
  const std::vector<float> prior_fast_cost = fast_cost;
  has_empirical_ac = !codes.encoding_info.empty();
  uint_configs = codes.uint_config;
  context_map = codes.context_map;

  sym_bits.resize(codes.encoding_info.size());
  std::vector<std::vector<uint8_t>> sym_seen(codes.encoding_info.size());
  for (size_t h = 0; h < codes.encoding_info.size(); ++h) {
    const auto& info = codes.encoding_info[h];
    sym_bits[h].resize(info.size());
    sym_seen[h].assign(info.size(), 0);
    for (size_t sym = 0; sym < info.size(); ++sym) {
      if (codes.use_prefix_code) {
        const bool seen = (info[sym].depth > 0);
        sym_seen[h][sym] = seen ? 1 : 0;
        sym_bits[h][sym] = seen ? static_cast<float>(info[sym].depth) : 14.0f;
      } else {
        const bool seen = (info[sym].freq_ > 0);
        sym_seen[h][sym] = seen ? 1 : 0;
        sym_bits[h][sym] =
            seen ? (static_cast<float>(ANS_LOG_TAB_SIZE) -
                    std::log2(static_cast<float>(info[sym].freq_)))
                 : 14.0f;
      }
    }
  }

  for (size_t ctx = 0; ctx < num_ac_contexts; ++ctx) {
    const size_t h = (ctx < context_map.size()) ? context_map[ctx] : 0;
    if (h >= sym_bits.size() || h >= uint_configs.size()) continue;
    const HybridUintConfig& cfg = uint_configs[h];
    const auto& h_bits = sym_bits[h];
    const auto& h_seen = sym_seen[h];
    for (uint32_t u = 0; u < kFastTableSize; ++u) {
      uint32_t tok = 0, nbits = 0, bits = 0;
      cfg.Encode(u, &tok, &nbits, &bits);
      float c_bits;
      if (tok < h_bits.size() && h_seen[tok]) {
        c_bits = h_bits[tok];
      } else {
        const float prior_c = prior_fast_cost[ctx * kFastTableSize + u];
        c_bits = std::min(12.5f, prior_c + 1.5f);
      }
      fast_cost[ctx * kFastTableSize + u] = c_bits + static_cast<float>(nbits);
    }
  }

  // Update kACMeta empirical costs with Laplace smoothing and a DCT prior.
  std::array<std::array<double, AcStrategy::kNumValidStrategies>, 2> acs_cnt{};
  for (size_t m = 0; m < 2; ++m) acs_cnt[m].fill(1.0);
  acs_cnt[0][static_cast<size_t>(AcStrategyType::DCT)] = 8.0;
  acs_cnt[1][static_cast<size_t>(AcStrategyType::DCT)] = 12.0;
  std::array<double, 2> acs_tot{34.0, 38.0};

  double qf_z = 2.0, qf_nz = 2.0;
  int32_t prev_qf = -1;
  uint8_t prev_acs = 0;
  for (size_t by = 0; by < frame_dim.ysize_blocks; ++by) {
    AcStrategyRow acs_row = ac_strategy.ConstRow(by);
    const int32_t* qrow = raw_quant_field.ConstRow(by);
    for (size_t bx = 0; bx < frame_dim.xsize_blocks; ++bx) {
      const AcStrategy acs = acs_row[bx];
      if (!acs.IsFirstBlock()) continue;
      const uint8_t s = acs.RawStrategy();
      const size_t match = (s == prev_acs) ? 1 : 0;
      acs_cnt[match][s] += 1.0;
      acs_tot[match] += 1.0;
      prev_acs = s;

      if (prev_qf >= 0) {
        if (qrow[bx] == prev_qf)
          qf_z += 1.0;
        else
          qf_nz += 1.0;
      }
      prev_qf = qrow[bx];
    }
  }
  for (size_t ctx = 0; ctx < 4; ++ctx) {
    const size_t m = (ctx == 1) ? 1 : 0;
    for (size_t s = 0; s < AcStrategy::kNumValidStrategies; ++s) {
      acs_cost_table[ctx][s] =
          static_cast<float>(-std::log2(acs_cnt[m][s] / acs_tot[m]));
    }
    const double p_z = qf_z / (qf_z + qf_nz);
    qf_zero_cost[ctx] = static_cast<float>(-std::log2(p_z));
    qf_nonzero_base[ctx] = static_cast<float>(-std::log2(1.0 - p_z) + 1.5);
  }

  double cfl_z = 2.0, cfl_nz = 2.0;
  int8_t prev_x = 0, prev_b = 0;
  for (size_t ty = 0; ty < cmap.ytox_map.ysize(); ++ty) {
    const int8_t* xr = cmap.ytox_map.ConstRow(ty);
    const int8_t* br = cmap.ytob_map.ConstRow(ty);
    for (size_t tx = 0; tx < cmap.ytox_map.xsize(); ++tx) {
      if (xr[tx] == prev_x)
        cfl_z += 1.0;
      else
        cfl_nz += 1.0;
      if (br[tx] == prev_b)
        cfl_z += 1.0;
      else
        cfl_nz += 1.0;
      prev_x = xr[tx];
      prev_b = br[tx];
    }
  }
  const double pcfl_z = cfl_z / (cfl_z + cfl_nz);
  cfl_zero_cost = static_cast<float>(-std::log2(pcfl_z));
  cfl_nonzero_base = static_cast<float>(-std::log2(1.0 - pcfl_z) + 1.5);

  std::array<std::array<double, 8>, 4> epf_cnt{};
  std::array<double, 4> epf_tot{8.0, 8.0, 8.0, 8.0};
  for (size_t m = 0; m < 4; ++m) epf_cnt[m].fill(1.0);
  for (size_t by = 0; by < frame_dim.ysize_blocks; ++by) {
    const uint8_t* row = epf_sharpness.ConstRow(by);
    const uint8_t* trow = (by > 0) ? epf_sharpness.ConstRow(by - 1) : row;
    for (size_t bx = 0; bx < frame_dim.xsize_blocks; ++bx) {
      const uint8_t e = Clamp1<uint8_t>(row[bx], 0, 7);
      const uint8_t top = (by > 0) ? trow[bx] : 4;
      const uint8_t left = (bx > 0) ? row[bx - 1] : 4;
      const size_t ctx = ((e == top) ? 1 : 0) + ((e == left) ? 2 : 0);
      epf_cnt[ctx][e] += 1.0;
      epf_tot[ctx] += 1.0;
    }
  }
  for (size_t ctx = 0; ctx < 4; ++ctx) {
    for (size_t e = 0; e < 8; ++e) {
      epf_cost_table[ctx][e] =
          static_cast<float>(-std::log2(epf_cnt[ctx][e] / epf_tot[ctx]));
    }
  }
}

float RDORateModel::SlowTokenCost(size_t ctx, uint32_t u_val) const {
  if (ctx >= num_ac_contexts) ctx = 0;
  if (has_empirical_ac) {
    const size_t h = (ctx < context_map.size()) ? context_map[ctx] : 0;
    if (h < sym_bits.size() && h < uint_configs.size()) {
      uint32_t tok = 0, nbits = 0, bits = 0;
      uint_configs[h].Encode(u_val, &tok, &nbits, &bits);
      const float base =
          (tok < sym_bits[h].size()) ? sym_bits[h][tok] : unseen_base;
      return base + static_cast<float>(nbits);
    }
  }
  const float base = fast_cost.empty()
                         ? unseen_base
                         : fast_cost[ctx * kFastTableSize + kFastTableSize - 1];
  return base + 2.0f * static_cast<float>(
                           FloorLog2Nonzero(u_val / kFastTableSize + 1) + 1);
}

float RDORateModel::AcsCost(uint8_t raw_acs, uint8_t prev_raw_acs) const {
  const uint8_t s =
      Clamp1<uint8_t>(raw_acs, 0, AcStrategy::kNumValidStrategies - 1);
  if (raw_acs == prev_raw_acs) return acs_cost_table[1][s];
  return acs_cost_table[0][s];
}

float RDORateModel::QfDeltaCost(int32_t qf, int32_t prev_qf) const {
  if (qf == prev_qf) return qf_zero_cost[0];
  const int32_t diff = std::abs(qf - prev_qf);
  return qf_nonzero_base[0] + 2.0f * std::log2(static_cast<float>(diff) + 1.0f);
}

float RDORateModel::CflCost(int32_t m, int32_t pred_m) const {
  if (m == pred_m) return cfl_zero_cost;
  const int32_t diff = std::abs(m - pred_m);
  return cfl_nonzero_base + 2.0f * std::log2(static_cast<float>(diff) + 1.0f);
}

float RDORateModel::EpfCost(uint8_t epf, uint8_t top_epf,
                            uint8_t left_epf) const {
  const uint8_t e = Clamp1<uint8_t>(epf, 0, 7);
  if (e == top_epf && e == left_epf) return epf_cost_table[3][e] * 0.4f;
  if (e == top_epf || e == left_epf) return epf_cost_table[1][e] * 0.7f;
  return epf_cost_table[0][e] + 1.2f;
}

// ============================================================================
// Public entry points
// ============================================================================

Status ProcessAllTilesRDO(const CompressParams& cparams,
                          const FrameHeader& frame_header, const Image3F& opsin,
                          const Rect& rect, const ImageF& initial_quant_field,
                          const ImageF& initial_quant_masking1x1,
                          const RDORateModel& rate_model, ThreadPool* pool,
                          PassesEncoderState* enc_state, AuxOut* aux_out) {
  return HWY_DYNAMIC_DISPATCH(ProcessAllTilesRDO_HWY)(
      cparams, frame_header, opsin, rect, initial_quant_field,
      initial_quant_masking1x1, rate_model, pool, enc_state, aux_out);
}

Status ComputeCoefficientsRDOQ(const FrameHeader& frame_header,
                               const Image3F& opsin, const Rect& rect,
                               const ImageF& initial_quant_field,
                               const ImageF& initial_quant_masking1x1,
                               const RDORateModel& rate_model, ThreadPool* pool,
                               PassesEncoderState* enc_state) {
  return HWY_DYNAMIC_DISPATCH(ComputeCoefficientsRDOQ_HWY)(
      frame_header, opsin, rect, initial_quant_field, initial_quant_masking1x1,
      rate_model, pool, enc_state);
}

Status ComputeDCFromOpsin(const Image3F& opsin, const Rect& rect,
                          const AcStrategyImage& ac_strategy, ThreadPool* pool,
                          Image3F* dc) {
  return HWY_DYNAMIC_DISPATCH(ComputeDCFromOpsin_HWY)(opsin, rect, ac_strategy,
                                                      pool, dc);
}

Status ComputeEPFSharpnessRDO(const FrameHeader& frame_header,
                              const Image3F& orig_opsin, const Rect& rect,
                              const RDORateModel& rate_model, ThreadPool* pool,
                              PassesEncoderState* enc_state) {
  (void)rect;
  const CompressParams& cparams = enc_state->cparams;
  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = shared.frame_dim;
  const ImageF& initial_quant_masking1x1 = enc_state->initial_quant_masking1x1;
  ImageB& epf_sharpness = shared.epf_sharpness;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();

  if (cparams.butteraugli_distance < kMinButteraugliForDynamicAR ||
      cparams.disable_perceptual_optimizations ||
      frame_header.loop_filter.epf_iters == 0 || frame_header.upsampling > 1) {
    FillPlane(static_cast<uint8_t>(4), &epf_sharpness, Rect(epf_sharpness));
    return true;
  }

  std::vector<uint8_t> epf_steps;
  if (cparams.butteraugli_distance > 4.5f) {
    epf_steps = std::vector<uint8_t>{0, 4};
  } else {
    epf_steps = (cparams.speed_tier <= SpeedTier::kKitten)
                    ? std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 7}
                    : std::vector<uint8_t>{0, 2, 4, 7};
  }

  const bool has_mask = (initial_quant_masking1x1.xsize() > 0 &&
                         initial_quant_masking1x1.ysize() > 0);
  const size_t mask_w = initial_quant_masking1x1.xsize();
  const size_t mask_h = initial_quant_masking1x1.ysize();
  const size_t xb = frame_dim.xsize_blocks;
  const size_t yb = frame_dim.ysize_blocks;

  static constexpr double kChanW[3] = {12.339445295782363, 1.0, 0.2};

  std::array<ImageF, 8> error_images;
  for (uint8_t val : epf_steps) {
    FillPlane(val, &epf_sharpness, Rect(epf_sharpness));
    JXL_ASSIGN_OR_RETURN(
        Image3F decoded,
        ReconstructImage(frame_header, shared, enc_state->coeffs, pool));
    JXL_ASSIGN_OR_RETURN(error_images[val],
                         ImageF::Create(memory_manager, xb, yb));
    auto compute_row = [&](uint32_t by, size_t /*thread*/) -> Status {
      float* erow = error_images[val].Row(by);
      const size_t py0 = by * kBlockDim;
      const size_t py1 = std::min(py0 + kBlockDim, orig_opsin.ysize());
      for (size_t bx = 0; bx < xb; ++bx) {
        const size_t px0 = bx * kBlockDim;
        const size_t px1 = std::min(px0 + kBlockDim, orig_opsin.xsize());
        double e2[3] = {0.0, 0.0, 0.0};
        for (size_t y = py0; y < py1; ++y) {
          const float* mr =
              has_mask
                  ? initial_quant_masking1x1.ConstRow(std::min(y, mask_h - 1))
                  : nullptr;
          const float* a0 = orig_opsin.ConstPlaneRow(0, y);
          const float* a1 = orig_opsin.ConstPlaneRow(1, y);
          const float* a2 = orig_opsin.ConstPlaneRow(2, y);
          const float* b0 = decoded.ConstPlaneRow(0, y);
          const float* b1 = decoded.ConstPlaneRow(1, y);
          const float* b2 = decoded.ConstPlaneRow(2, y);
          for (size_t x = px0; x < px1; ++x) {
            const float m = has_mask ? mr[std::min(x, mask_w - 1)] : 1.0f;
            const float m2 = m * m;
            const float d0 = a0[x] - b0[x];
            const float d1 = a1[x] - b1[x];
            const float d2 = a2[x] - b2[x];
            e2[0] += m2 * d0 * d0;
            e2[1] += m2 * d1 * d1;
            e2[2] += m2 * d2 * d2;
          }
        }
        erow[bx] = static_cast<float>(kChanW[0] * e2[0] + kChanW[1] * e2[1] +
                                      kChanW[2] * e2[2]);
      }
      return true;
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, yb, ThreadPool::NoInit, compute_row,
                                  "EPFErrorBlocks"));
  }

  if (cparams.butteraugli_distance <= 4.5f) {
    ScaleImage(0.99f, &error_images[0]);
  }
  double sum_err4 = 0.0;
  for (size_t by = 0; by < yb; ++by) {
    const float* er4 = error_images[4].ConstRow(by);
    for (size_t bx = 0; bx < xb; ++bx) {
      sum_err4 += static_cast<double>(er4[bx]);
    }
  }
  const double mean_err4 =
      std::max(1e-8, sum_err4 / static_cast<double>(xb * yb));
  const double lambda_epf =
      (0.065 / Clamp1<double>(cparams.butteraugli_distance, 0.5, 5.0)) *
      mean_err4;

  // Empirical EPF conditional counts updated after Pass 1.
  std::array<std::array<double, 8>, 4> emp_cnt{};
  std::array<double, 4> emp_tot{8.0, 8.0, 8.0, 8.0};
  for (size_t ctx = 0; ctx < 4; ++ctx) emp_cnt[ctx].fill(1.0);
  double emp_rate[4][8];
  for (size_t ctx = 0; ctx < 4; ++ctx) {
    for (uint8_t s : epf_steps) {
      uint8_t fake_top = (ctx & 1) ? s : static_cast<uint8_t>((s + 1) & 7);
      uint8_t fake_left = (ctx & 2) ? s : static_cast<uint8_t>((s + 2) & 7);
      emp_rate[ctx][s] = rate_model.EpfCost(s, fake_top, fake_left);
    }
  }

  const bool use_viterbi = (cparams.speed_tier <= SpeedTier::kKitten);
  for (int pass = 0; pass < 2; ++pass) {
    if (pass == 1) {
      for (size_t ctx = 0; ctx < 4; ++ctx) {
        for (size_t e = 0; e < 8; ++e) {
          emp_rate[ctx][e] = -std::log2(emp_cnt[ctx][e] / emp_tot[ctx]);
        }
      }
    }

    if (!use_viterbi || pass == 0) {
      // Causal raster RDO sweep.
      for (size_t by = 0; by < yb; ++by) {
        uint8_t* row = epf_sharpness.Row(by);
        const uint8_t* trow = (by > 0) ? epf_sharpness.ConstRow(by - 1) : row;
        for (size_t bx = 0; bx < xb; ++bx) {
          const uint8_t top = (by > 0) ? trow[bx] : 4;
          const uint8_t left = (bx > 0) ? row[bx - 1] : 4;
          uint8_t best_s = epf_steps[0];
          double best_j = std::numeric_limits<double>::infinity();
          for (uint8_t s : epf_steps) {
            const size_t ctx = ((s == top) ? 1 : 0) + ((s == left) ? 2 : 0);
            const double d = error_images[s].Row(by)[bx];
            const double j = d + lambda_epf * emp_rate[ctx][s];
            if (j < best_j) {
              best_j = j;
              best_s = s;
            }
          }
          row[bx] = best_s;
          if (pass == 0) {
            const size_t ctx =
                ((best_s == top) ? 1 : 0) + ((best_s == left) ? 2 : 0);
            emp_cnt[ctx][best_s] += 1.0;
            emp_tot[ctx] += 1.0;
          }
        }
      }
    } else {
      // Exact 8-state row-wise Viterbi DP for e8/e9.
      const size_t ns = epf_steps.size();
      std::vector<double> prev_cost(ns), curr_cost(ns);
      std::vector<uint8_t> back(xb * ns);
      for (size_t by = 0; by < yb; ++by) {
        uint8_t* row = epf_sharpness.Row(by);
        const uint8_t* trow = (by > 0) ? epf_sharpness.ConstRow(by - 1) : row;
        for (size_t si = 0; si < ns; ++si) {
          const uint8_t s = epf_steps[si];
          const uint8_t top = (by > 0) ? trow[0] : 4;
          const size_t ctx = ((s == top) ? 1 : 0) + ((s == 4) ? 2 : 0);
          prev_cost[si] =
              error_images[s].Row(by)[0] + lambda_epf * emp_rate[ctx][s];
        }
        for (size_t bx = 1; bx < xb; ++bx) {
          const uint8_t top = (by > 0) ? trow[bx] : 4;
          for (size_t si = 0; si < ns; ++si) {
            const uint8_t s = epf_steps[si];
            const double err = error_images[s].Row(by)[bx];
            double best = std::numeric_limits<double>::infinity();
            uint8_t best_prev = 0;
            for (size_t pi = 0; pi < ns; ++pi) {
              const uint8_t left = epf_steps[pi];
              const size_t ctx = ((s == top) ? 1 : 0) + ((s == left) ? 2 : 0);
              const double c_val =
                  prev_cost[pi] + err + lambda_epf * emp_rate[ctx][s];
              if (c_val < best) {
                best = c_val;
                best_prev = static_cast<uint8_t>(pi);
              }
            }
            curr_cost[si] = best;
            back[bx * ns + si] = best_prev;
          }
          prev_cost.swap(curr_cost);
        }
        size_t best_si = 0;
        for (size_t si = 1; si < ns; ++si) {
          if (prev_cost[si] < prev_cost[best_si]) best_si = si;
        }
        for (size_t bx = xb; bx-- > 0;) {
          row[bx] = epf_steps[best_si];
          if (bx > 0) best_si = back[bx * ns + best_si];
        }
      }
    }
  }
  return true;
}

RDOQTrellisResult RunTrellisRDOQSingleBlockForTest(
    size_t channel, AcStrategyType acs_type, const float* coeffs_in,
    const Quantizer& quantizer, float qm_multiplier, int32_t qf, float q_target,
    int32_t predicted_nzeros, size_t block_ctx, const BlockCtxMap& bcm,
    const coeff_order_t* order, const RDORateModel& rate_model, float lambda,
    int band_width, int multi_level, int32_t* quantized_out) {
  BlockTrellisWorkspace ws;
  return RunTrellisRDOQCore(
      channel, acs_type, coeffs_in, quantizer, qm_multiplier, qf, q_target,
      predicted_nzeros, block_ctx, bcm, order, rate_model, lambda, band_width,
      multi_level, /*non_perceptual_mse=*/false, /*jnd_zero_floor=*/0.0f,
      /*unmasked_lf_dist_mul=*/1.0f, /*unmasked_hf_dist_mul=*/1.0f, nullptr,
      nullptr, 0.0f, 0.0f, nullptr, nullptr, 1.0f, 1.0f, 1.0f, 1.0f,
      /*enable_cfl_coupling=*/false, &ws, quantized_out);
}

}  // namespace jxl
#endif  // HWY_ONCE
