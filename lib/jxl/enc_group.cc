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

static constexpr size_t kMaxNzeroDelta = 6;
static constexpr size_t kBandWidth = 2 * kMaxNzeroDelta + 2;  // 14
static constexpr size_t kBandStride = 2 * kBandWidth;         // 28
static constexpr size_t kMaxRdoqWindow = 1024;
static constexpr size_t kMaxCandidates = 5;

static JXL_INLINE float ReconstructedLevel(int32_t level, float bias_one) {
  const int32_t abs_level = std::abs(level);
  if (abs_level == 0) return 0.0f;
  if (abs_level == 1) return (level > 0) ? bias_one : -bias_one;
  const float f = static_cast<float>(level);
  return f - 0.145f / f;
}

// Counts distinct 1/1024-quantized XYB triples in the transform footprint.
// Returns true if distinct colors <= 24 * eff_blocks. Uses at most 1 KB stack.
static bool BlockHasFewColors(const Image3F* orig_opsin,
                              const float* const JXL_RESTRICT opsin_rows[3],
                              size_t opsin_stride, size_t frame_px0,
                              size_t frame_py0, size_t bx_px_offset,
                              size_t cov_x, size_t cov_y) {
  constexpr size_t kFewColorsPerBlock = 24;
  const size_t step_x = std::max<size_t>(1, cov_x / 2);
  const size_t step_y = std::max<size_t>(1, cov_y / 2);
  const size_t eff_blocks = (cov_x / step_x) * (cov_y / step_y);
  const size_t limit = kFewColorsPerBlock * eff_blocks;

  uint32_t table[256];
  const size_t table_size = (eff_blocks == 1) ? 64 : 256;
  const uint32_t table_mask = static_cast<uint32_t>(table_size - 1);
  const int hash_shift = (eff_blocks == 1) ? 26 : 24;
  memset(table, 0xFF, table_size * sizeof(uint32_t));

  const size_t width = cov_x * kBlockDim;
  const size_t height = cov_y * kBlockDim;
  const bool use_orig =
      (orig_opsin != nullptr && frame_px0 < orig_opsin->xsize() &&
       frame_py0 < orig_opsin->ysize());
  const size_t max_x =
      use_orig ? std::min(width, orig_opsin->xsize() - frame_px0) : width;
  const size_t max_y =
      use_orig ? std::min(height, orig_opsin->ysize() - frame_py0) : height;
  if (max_x == 0 || max_y == 0) return false;

  size_t distinct = 0;
  for (size_t y = 0; y < max_y; y += step_y) {
    const float* JXL_RESTRICT rx =
        use_orig ? (orig_opsin->ConstPlaneRow(0, frame_py0 + y) + frame_px0)
                 : (opsin_rows[0] + y * opsin_stride + bx_px_offset);
    const float* JXL_RESTRICT ry =
        use_orig ? (orig_opsin->ConstPlaneRow(1, frame_py0 + y) + frame_px0)
                 : (opsin_rows[1] + y * opsin_stride + bx_px_offset);
    const float* JXL_RESTRICT rb =
        use_orig ? (orig_opsin->ConstPlaneRow(2, frame_py0 + y) + frame_px0)
                 : (opsin_rows[2] + y * opsin_stride + bx_px_offset);
    for (size_t x = 0; x < max_x; x += step_x) {
      const uint32_t qx =
          static_cast<uint32_t>(static_cast<int32_t>(rx[x] * 1024.0f + 512.5f)) &
          0x3FFu;
      const uint32_t qy =
          static_cast<uint32_t>(static_cast<int32_t>(ry[x] * 1024.0f + 512.5f)) &
          0x7FFu;
      const uint32_t qb =
          static_cast<uint32_t>(static_cast<int32_t>(rb[x] * 1024.0f + 512.5f)) &
          0x7FFu;
      uint32_t key = qx | (qy << 10) | (qb << 21);
      if (key == 0xFFFFFFFFu) key = 0xFFFFFFFEu;

      uint32_t slot = (key * 0x9E3779B1u) >> hash_shift;
      while (true) {
        const uint32_t cur = table[slot];
        if (cur == key) break;
        if (cur == 0xFFFFFFFFu) {
          table[slot] = key;
          if (++distinct > limit) return false;
          break;
        }
        slot = (slot + 1u) & table_mask;
      }
    }
  }
  return true;
}

static JXL_INLINE size_t BuildRdoqCandidates(
    int32_t current, float ideal, float abs_ideal, bool allow_zero_to_one,
    bool allow_round_up, int32_t out_cands[kMaxCandidates]) {
  out_cands[0] = 0;
  size_t count = 1;
  const int32_t sgn = (ideal >= 0.0f) ? 1 : -1;

  auto push_unique = [&](int32_t cand) {
    for (size_t i = 0; i < count; ++i) {
      if (out_cands[i] == cand) return;
    }
    if (count < kMaxCandidates) {
      out_cands[count++] = cand;
    }
  };

  if (current == 0) {
    if (allow_zero_to_one) {
      push_unique(sgn);
    }
    return count;
  }

  push_unique(current);
  const int32_t cur_sgn = (current > 0) ? 1 : -1;
  const int32_t q_abs = std::abs(current);
  if (q_abs >= 2) {
    push_unique(cur_sgn * (q_abs - 1));
  }
  if (q_abs >= 3) {
    push_unique(cur_sgn * (q_abs - 2));
  }
  if (allow_round_up && abs_ideal + 0.56f >= static_cast<float>(q_abs + 1)) {
    push_unique(cur_sgn * (q_abs + 1));
  }
  return count;
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
                               bool few_colors,
                               const float* JXL_RESTRICT block_in,
                               int32_t quant, int32_t* JXL_RESTRICT block_out,
                               uint8_t* JXL_RESTRICT choices_buf) {
  const size_t size = covered_blocks * kDCTBlockSize;
  const size_t dim = xsize * kBlockDim;
  const coeff_order_t* JXL_RESTRICT order =
      &tcm.coeff_orders[CoeffOrderOffset(ord, c)];
  const size_t block_ctx =
      tcm.block_ctx_map.Context(/*dc_idx=*/0, quant, ord, c);
  const uint32_t nzero_ctx =
      tcm.block_ctx_map.NonZeroContext(predicted_nzeros, block_ctx);
  const uint8_t cl_nzero = tcm.cluster_of_ctx[nzero_ctx];
  const size_t histo_offset =
      tcm.block_ctx_map.ZeroDensityContextsOffset(block_ctx);
  const float* JXL_RESTRICT qm = quantizer.InvDequantMatrix(quant_kind, c);
  const float qac = quantizer.Scale() * quant * qm_multiplier;
  const float b_c = kDefaultQuantBias[c];
  const float restore_floor = few_colors ? 0.50f : (0.5f * b_c);

  const bool is_partial_block =
      ((1 << static_cast<size_t>(quant_kind)) & kPartialBlockKinds) != 0;
  auto rescue_quad0 = [&]() -> int32_t {
    if (is_partial_block || c != 1 || covered_blocks > 4 || eob_scale <= 0.0f) {
      return 0;
    }
    float best_abs = few_colors ? 0.54f : 0.46f;
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

  const size_t window_cap =
      (covered_blocks == 1)
          ? 63
          : std::min<size_t>(64 * covered_blocks, kMaxRdoqWindow);
  const size_t restore_len =
      (covered_blocks == 1) ? 20 : std::min<size_t>(24 * covered_blocks, 256);
  const size_t search_end = std::min<size_t>(size, covered_blocks + window_cap);
  if (search_end <= covered_blocks) {
    int32_t nz = 0;
    for (size_t k = covered_blocks; k < size; ++k) {
      nz += static_cast<int32_t>(block_out[order[k]] != 0);
    }
    return nz;
  }
  const size_t window_len = search_end - covered_blocks;

  size_t suffix_nzeros = 0;
  for (size_t k = search_end; k < size; ++k) {
    suffix_nzeros += static_cast<size_t>(block_out[order[k]] != 0);
  }
  size_t window_orig_nzeros = 0;
  size_t scalar_lf_nz = 0;
  bool any_near_dz_cand = false;
  for (size_t wi = 0; wi < window_len; ++wi) {
    const size_t k = covered_blocks + wi;
    const size_t pos = order[k];
    const size_t y = pos / dim;
    const size_t x = pos % dim;
    const size_t v8 = y / ysize;
    const size_t u8 = x / xsize;
    const uint8_t s = kCoeffBand[v8][u8];
    if (block_out[pos] != 0) {
      ++window_orig_nzeros;
      if (is_partial_block ? (k <= 8) : (s >= 2)) {
        ++scalar_lf_nz;
      }
    } else if (!any_near_dz_cand) {
      const size_t hfix =
          (static_cast<size_t>(v8 >= 4) * 2 + static_cast<size_t>(u8 >= 4));
      const float t0_full = quad_thres[hfix] + zero_thres[4 + s];
      const float near_dz_floor = std::max(0.49f, t0_full - 0.04f);
      const float abs_ideal = std::abs(block_in[pos] * (qm[pos] * qac));
      const bool can_restore =
          (c == 1)
              ? (wi < restore_len)
              : ((wi < restore_len / 2) &&
                 (is_partial_block ? (k <= 8) : (s >= 2)));
      if ((can_restore && abs_ideal >= restore_floor) ||
          (!few_colors && abs_ideal >= near_dz_floor)) {
        any_near_dz_cand = true;
      }
    }
  }
  if (suffix_nzeros == 0 && window_orig_nzeros == 0 && !any_near_dz_cand) {
    return rescue_quad0();
  }

  float suffix_cost[2] = {0.0f, 0.0f};
  if (suffix_nzeros > 0) {
    for (size_t init_prev = 0; init_prev < 2; ++init_prev) {
      size_t rem = suffix_nzeros;
      size_t prev = init_prev;
      float bits = 0.0f;
      for (size_t k = search_end; k < size && rem > 0; ++k) {
        const int32_t coeff = block_out[order[k]];
        const uint32_t ctx =
            static_cast<uint32_t>(histo_offset) +
            ZeroDensityContext(rem, k, covered_blocks, log2_covered_blocks,
                               prev);
        const uint8_t cl = tcm.cluster_of_ctx[ctx];
        bits += tcm.ValBits(cl, PackSigned(coeff));
        prev = (coeff != 0) ? 1 : 0;
        rem -= prev;
      }
      suffix_cost[init_prev] = bits;
    }
  }

  static constexpr float kBandDistWeight[4] = {0.92f, 1.00f, 1.24f, 1.65f};
  const float sparse_w = Clamp1((eob_scale - 0.32f) / 0.42f, 0.0f, 1.0f);
  const float base_ref = (c == 1) ? 2.78f : 2.98f;
  const float eff_ref_bits = base_ref * (1.0f + 0.35f * eob_scale);
  const float kQuadTilt = 0.50f;
  const float base_h = 0.58f;
  const float t_ref = 0.60f;
  const float denom_ref = b_c * (2.0f * t_ref - b_c);

  constexpr float kInf = 1e30f;
  float cost_buf[2][kBandStride];
  for (size_t i = 0; i < kBandStride; ++i) {
    cost_buf[0][i] = kInf;
    cost_buf[1][i] = kInf;
  }
  size_t next_row = 0;
  cost_buf[next_row][0] = suffix_cost[0];
  cost_buf[next_row][1] = suffix_cost[1];
  size_t next_min = suffix_nzeros;
  size_t next_max = suffix_nzeros;
  size_t original_after_nzeros = 0;

  for (size_t rev = 0; rev < window_len; ++rev) {
    const size_t wi = window_len - 1 - rev;
    const size_t k = covered_blocks + wi;
    const size_t pos = order[k];
    const int32_t current = block_out[pos];
    const float ideal = block_in[pos] * (qm[pos] * qac);
    const float abs_ideal = std::abs(ideal);

    const size_t y = pos / dim;
    const size_t x = pos % dim;
    const size_t v8 = y / ysize;
    const size_t u8 = x / xsize;
    const size_t hfix =
        (static_cast<size_t>(v8 >= 4) * 2 + static_cast<size_t>(u8 >= 4));
    const uint8_t s = kCoeffBand[v8][u8];
    const float t0_full = quad_thres[hfix] + zero_thres[4 + s];
    const float near_dz_floor = std::max(0.49f, t0_full - 0.04f);
    const bool can_restore =
        (c == 1)
            ? (wi < restore_len)
            : ((wi < restore_len / 2) &&
               (is_partial_block ? (k <= 8) : (s >= 2)));
    const bool allow_zero_to_one =
        (can_restore && abs_ideal >= restore_floor) ||
        (!few_colors && abs_ideal >= near_dz_floor);
    const bool allow_round_up = !(few_colors && s <= 1);
    float restore_rate_mul = 1.0f;
    if (current == 0) {
      if (few_colors) {
        restore_rate_mul = (c == 1) ? 2.0f : 2.5f;
      } else if (c != 1 && abs_ideal < near_dz_floor) {
        restore_rate_mul = (c == 0) ? 1.15f : 1.25f;
      }
    }

    int32_t candidates[kMaxCandidates];
    const size_t cand_count = BuildRdoqCandidates(
        current, ideal, abs_ideal, allow_zero_to_one, allow_round_up,
        candidates);

    float dist_cost[kMaxCandidates];
    uint32_t packed_sym[kMaxCandidates];
    dist_cost[0] = 0.0f;
    packed_sym[0] = 0u;
    if (cand_count > 1) {
      const float raw_h = (hfix == 0) ? 0.58f : ((c == 1) ? 0.64f : 0.62f);
      const float perc_t0 =
          std::max(0.50f, t0_full - (1.0f - kQuadTilt) * (raw_h - base_h));
      const float t_ratio = t_ref / perc_t0;
      const float w_base = (eff_ref_bits / denom_ref) * (t_ratio * t_ratio);
      const float band_w = is_partial_block ? 1.0f : kBandDistWeight[s];
      const float w_zero = w_base * (1.0f + sparse_w * (band_w - 1.0f));
      const float d_one_base = b_c * (b_c - 2.0f * abs_ideal);
      const float d_zero_step = (w_zero - w_base) * d_one_base;
      for (size_t ci = 1; ci < cand_count; ++ci) {
        const int32_t level = candidates[ci];
        const float deq = ReconstructedLevel(level, b_c);
        dist_cost[ci] = d_zero_step + w_base * deq * (deq - 2.0f * ideal);
        packed_sym[ci] = PackSigned(level);
      }
    }

    const size_t processed_after = window_len - 1 - wi;
    const size_t min_rem =
        suffix_nzeros +
        ((original_after_nzeros > kMaxNzeroDelta)
             ? (original_after_nzeros - kMaxNzeroDelta)
             : 0);
    const size_t max_rem =
        suffix_nzeros +
        std::min(original_after_nzeros + kMaxNzeroDelta, processed_after);
    const size_t current_min = min_rem;
    const size_t current_max =
        std::min(max_rem + 1, suffix_nzeros + window_len);

    const size_t cur_row = 1 - next_row;
    float* JXL_RESTRICT cur_dp = cost_buf[cur_row];
    const float* JXL_RESTRICT next_dp = cost_buf[next_row];
    for (size_t i = 0; i < kBandStride; ++i) {
      cur_dp[i] = kInf;
    }
    uint8_t* JXL_RESTRICT choice_row = choices_buf + wi * kBandStride;

    uint32_t cached_ctx0 = 0xFFFFFFFFu;
    float zero_local[2] = {0.0f, 0.0f};
    float nz_local[2] = {kInf, kInf};
    uint8_t nz_best_ci[2] = {0, 0};

    for (size_t rem = current_min; rem <= current_max; ++rem) {
      const size_t band_offset = (rem - current_min) * 2;
      if (rem == 0) {
        if (next_min == 0) {
          const float tail0 = next_dp[0];
          cur_dp[band_offset + 0] = tail0;
          cur_dp[band_offset + 1] = tail0;
          choice_row[band_offset + 0] = 0;
          choice_row[band_offset + 1] = 0;
        }
        continue;
      }

      const uint32_t ctx0 =
          static_cast<uint32_t>(histo_offset) +
          ZeroDensityContext(rem, k, covered_blocks, log2_covered_blocks, 0);
      if (ctx0 != cached_ctx0) {
        cached_ctx0 = ctx0;
        tcm.ValBitsPair(ctx0, 0, &zero_local[0], &zero_local[1]);
        nz_local[0] = kInf;
        nz_local[1] = kInf;
        nz_best_ci[0] = 0;
        nz_best_ci[1] = 0;
        for (size_t ci = 1; ci < cand_count; ++ci) {
          float b0 = 0.0f;
          float b1 = 0.0f;
          tcm.ValBitsPair(ctx0, packed_sym[ci], &b0, &b1);
          const float c0 = dist_cost[ci] + b0 * restore_rate_mul;
          const float c1 = dist_cost[ci] + b1 * restore_rate_mul;
          if (c0 < nz_local[0]) {
            nz_local[0] = c0;
            nz_best_ci[0] = static_cast<uint8_t>(ci);
          }
          if (c1 < nz_local[1]) {
            nz_local[1] = c1;
            nz_best_ci[1] = static_cast<uint8_t>(ci);
          }
        }
      }

      float z_tail = kInf;
      if (rem >= next_min && rem <= next_max) {
        z_tail = next_dp[(rem - next_min) * 2 + 0];
      }
      float nz_tail = kInf;
      if (cand_count > 1 && (rem - 1) >= next_min && (rem - 1) <= next_max) {
        nz_tail = next_dp[(rem - 1 - next_min) * 2 + 1];
      }

      for (size_t prev = 0; prev < 2; ++prev) {
        float best_c = kInf;
        uint8_t best_ci = 0;
        if (z_tail < kInf) {
          best_c = zero_local[prev] + z_tail;
        }
        if (nz_tail < kInf) {
          const float c_nz = nz_local[prev] + nz_tail;
          if (c_nz < best_c) {
            best_c = c_nz;
            best_ci = nz_best_ci[prev];
          }
        }
        cur_dp[band_offset + prev] = best_c;
        choice_row[band_offset + prev] = best_ci;
      }
    }

    next_row = cur_row;
    next_min = current_min;
    next_max = current_max;
    original_after_nzeros += static_cast<size_t>(current != 0);
  }

  const float flat_penalty =
      (suffix_nzeros == 0 && scalar_lf_nz > 0)
          ? (sparse_w * ((c == 1) ? 1.15f : 0.35f) *
             static_cast<float>(std::min<size_t>(2, scalar_lf_nz)))
          : 0.0f;

  const float* JXL_RESTRICT final_dp = cost_buf[next_row];
  float best_total_cost = kInf;
  size_t best_remaining = suffix_nzeros + window_orig_nzeros;
  for (size_t rem = next_min; rem <= next_max; ++rem) {
    const size_t init_prev = (rem > size / 16) ? 0 : 1;
    const float path_cost = final_dp[(rem - next_min) * 2 + init_prev];
    if (path_cost >= kInf * 0.5f) continue;
    float nzero_bits = tcm.ValBits(cl_nzero, static_cast<uint32_t>(rem));
    if (rem == 0) {
      nzero_bits += flat_penalty;
    }
    const float total_cost = path_cost + nzero_bits;
    if (total_cost < best_total_cost) {
      best_total_cost = total_cost;
      best_remaining = rem;
    }
  }
  if (best_total_cost >= kInf * 0.5f) {
    return static_cast<int32_t>(suffix_nzeros + window_orig_nzeros);
  }

  size_t cur_rem = best_remaining;
  size_t cur_prev = (best_remaining > size / 16) ? 0 : 1;
  size_t orig_after = window_orig_nzeros;
  for (size_t wi = 0; wi < window_len; ++wi) {
    const size_t k = covered_blocks + wi;
    const size_t pos = order[k];
    const int32_t current = block_out[pos];
    orig_after -= static_cast<size_t>(current != 0);

    if (cur_rem == 0) {
      block_out[pos] = 0;
      cur_prev = 0;
      continue;
    }
    const size_t row_min =
        suffix_nzeros +
        ((orig_after > kMaxNzeroDelta) ? (orig_after - kMaxNzeroDelta) : 0);
    const size_t choice_idx =
        wi * kBandStride + (cur_rem - row_min) * 2 + cur_prev;
    const uint8_t ci = choices_buf[choice_idx];
    if (ci == 0) {
      block_out[pos] = 0;
      cur_prev = 0;
    } else {
      const float ideal = block_in[pos] * (qm[pos] * qac);
      int32_t chosen = current;
      if (current == 0) {
        chosen = (ideal >= 0.0f) ? 1 : -1;
      } else {
        const size_t y = pos / dim;
        const size_t x = pos % dim;
        const uint8_t s = kCoeffBand[y / ysize][x / xsize];
        const bool allow_round_up = !(few_colors && s <= 1);
        const float abs_ideal = std::abs(ideal);
        int32_t candidates[kMaxCandidates];
        const size_t cand_count = BuildRdoqCandidates(
            current, ideal, abs_ideal, false, allow_round_up, candidates);
        if (ci < cand_count) chosen = candidates[ci];
      }
      block_out[pos] = chosen;
      const size_t is_nz = static_cast<size_t>(chosen != 0);
      cur_rem -= is_nz;
      cur_prev = is_nz;
    }
  }

  int32_t final_nzeros = static_cast<int32_t>(best_remaining);
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
                     int32_t* out_nzeros = nullptr,
                     uint8_t* JXL_RESTRICT rdoq_choices = nullptr,
                     bool few_colors = false) {
  CoefficientLayout(&ysize, &xsize);
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

  if (tcm != nullptr && eob_scale > 0.0f && rdoq_choices != nullptr) {
    const int32_t nz = TrellisQuantizeBlockAC(
        quantizer, *tcm, c, qm_multiplier, quant_kind, xsize, ysize, ord,
        covered_blocks, log2_covered_blocks, predicted_nzeros, quad_thres,
        zero_thres, eob_scale, few_colors, block_in, quant, block_out,
        rdoq_choices);
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
                               int32_t* out_nzeros = nullptr,
                               uint8_t* JXL_RESTRICT rdoq_choices = nullptr,
                               bool few_colors = false) {
  CoefficientLayout(&ysize, &xsize);
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
                  log2_covered_blocks, predicted_nzeros, out_nzeros,
                  rdoq_choices, few_colors);

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
  const size_t quant_ints = 3 * AcStrategy::kMaxCoeffArea;
  const size_t rdoq_choice_bytes = kMaxRdoqWindow * kBandStride;
  size_t mem_bytes = quant_ints * sizeof(int32_t) + rdoq_choice_bytes;
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
    uint8_t* JXL_RESTRICT rdoq_choices =
        reinterpret_cast<uint8_t*>(quantized + quant_ints);
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
          const size_t frame_px0 = (block_group_rect.x0() + bx) * kBlockDim;
          const size_t frame_py0 = (block_group_rect.y0() + by) * kBlockDim;
          const bool few_colors =
              (tcm != nullptr && eob_scale > 0.0f)
                  ? BlockHasFewColors(enc_state->orig_opsin, opsin_rows,
                                      opsin_stride, frame_px0, frame_py0,
                                      bx * kBlockDim, cov_x, cov_y)
                  : false;

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
              covered_blocks, log2_covered_blocks, pred_nz_y, &final_nz_y,
              rdoq_choices, few_colors);
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
                            log2_covered_blocks, pred_nz_c, &final_nz_c,
                            rdoq_choices, few_colors);
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
