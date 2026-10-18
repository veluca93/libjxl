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

#ifndef LIB_JXL_ENC_GROUP_CACHE_TYPES_H_
#define LIB_JXL_ENC_GROUP_CACHE_TYPES_H_

#include <array>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/cms/opsin_params.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image_ops.h"

namespace jxl {

struct AcBlockEvalSlot {
  int32_t quant_ac = -1;
  uint8_t use_rdoq = 0;
  int16_t pred_nz[3] = {-1, -1, -1};
  int32_t shifted_nz[3] = {0, 0, 0};
  float bits = 0.0f;
  std::vector<float> ac_err_pixels;
};

struct AcTransformCache {
  uint8_t bx = 0;
  uint8_t by = 0;
  uint8_t cov_x = 1;
  uint8_t cov_y = 1;
  uint8_t xblocks = 1;
  uint8_t yblocks = 1;
  uint8_t acs_raw = 0;
  bool few_colors = false;
  int32_t base_quant_ac = 1;
  int32_t cur_quant_ac = 1;
  uint32_t act_bits = 0;
  uint32_t tile_idx = 0;
  float ytox = 0.0f;
  float ytob = 0.0f;
  float base_band_q[3][4] = {};
  std::vector<float> src_dct[3];
  static constexpr size_t kNumSlots = 4;
  AcBlockEvalSlot slots[kNumSlots];
  uint8_t next_slot = 0;
  uint8_t cur_slot = 0;
  uint8_t cand_slot = 0;
};

struct AcGroupBlockCache {
  size_t group_idx = 0;
  size_t gx0 = 0;
  size_t gy0 = 0;
  size_t gw_blocks = 0;
  size_t gh_blocks = 0;
  size_t px0 = 0;
  size_t py0 = 0;
  size_t pw_px = 0;
  size_t ph_px = 0;
  std::vector<float> base_pre_gab[3];
  std::vector<AcTransformCache> transforms;
};

}  // namespace jxl
#endif  // LIB_JXL_ENC_GROUP_CACHE_TYPES_H_

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

Status InitAcGroupCacheImpl(size_t group_idx,
                            const PassesEncoderState* enc_state,
                            const Image3F& opsin, const Rect& rect,
                            size_t tiles_x, size_t tile_blocks,
                            AcGroupBlockCache* cache) {
  JxlMemoryManager* memory_manager = opsin.memory_manager();
  const PassesSharedState& shared = enc_state->shared;
  const Rect block_group_rect = shared.frame_dim.BlockGroupRect(group_idx);
  const size_t gx0 = block_group_rect.x0();
  const size_t gy0 = block_group_rect.y0();
  const size_t gw_blocks = block_group_rect.xsize();
  const size_t gh_blocks = block_group_rect.ysize();
  const size_t px0 = gx0 * kBlockDim;
  const size_t py0 = gy0 * kBlockDim;
  const size_t xsize = shared.frame_dim.xsize;
  const size_t ysize = shared.frame_dim.ysize;
  const size_t pw_px = std::min(xsize - px0, gw_blocks * kBlockDim);
  const size_t ph_px = std::min(ysize - py0, gh_blocks * kBlockDim);

  cache->group_idx = group_idx;
  cache->gx0 = gx0;
  cache->gy0 = gy0;
  cache->gw_blocks = gw_blocks;
  cache->gh_blocks = gh_blocks;
  cache->px0 = px0;
  cache->py0 = py0;
  cache->pw_px = pw_px;
  cache->ph_px = ph_px;

  const Rect group_rect =
      shared.frame_dim.GroupRect(group_idx).Translate(rect.x0(), rect.y0());
  const Rect cmap_rect(gx0 / kColorTileDimInBlocks, gy0 / kColorTileDimInBlocks,
                       DivCeil(gw_blocks, kColorTileDimInBlocks),
                       DivCeil(gh_blocks, kColorTileDimInBlocks));
  const size_t opsin_stride = static_cast<size_t>(opsin.PixelsPerRow());
  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;
  const size_t fmem_bytes =
      (2 * AcStrategy::kMaxCoeffArea + dct_scratch_size) * sizeof(float);
  JXL_ASSIGN_OR_RETURN(auto fmem,
                       AlignedMemory::Create(memory_manager, fmem_bytes));
  HWY_ALIGN float* JXL_RESTRICT dct_buf = fmem.address<float>();
  HWY_ALIGN float* JXL_RESTRICT px_buf = dct_buf + AcStrategy::kMaxCoeffArea;
  float* JXL_RESTRICT scratch_space = px_buf + AcStrategy::kMaxCoeffArea;

  std::vector<float> dc_src[3];
  std::vector<float> dc_err[3];
  for (size_t c = 0; c < 3; ++c) {
    dc_src[c].assign(gw_blocks * gh_blocks, 0.0f);
    dc_err[c].assign(gw_blocks * gh_blocks, 0.0f);
    cache->base_pre_gab[c].resize(pw_px * ph_px);
    for (size_t y = 0; y < ph_px; ++y) {
      const float* JXL_RESTRICT src_row = group_rect.ConstPlaneRow(opsin, c, y);
      memcpy(cache->base_pre_gab[c].data() + y * pw_px, src_row,
             pw_px * sizeof(float));
    }
  }

  size_t num_transforms = 0;
  for (size_t by = 0; by < gh_blocks; ++by) {
    AcStrategyRow acs_row = shared.ac_strategy.ConstRow(block_group_rect, by);
    for (size_t bx = 0; bx < gw_blocks; ++bx) {
      if (acs_row[bx].IsFirstBlock()) ++num_transforms;
    }
  }
  cache->transforms.clear();
  cache->transforms.reserve(num_transforms);

  const float d_target = enc_state->cparams.butteraugli_distance;
  const float eob_scale =
      (enc_state->cparams.speed_tier <= SpeedTier::kHare)
          ? Clamp1((d_target - 0.28f) / (d_target + 0.50f) * 1.45f, 0.0f, 1.0f)
          : 0.0f;

  for (size_t by = 0; by < gh_blocks; ++by) {
    const int32_t* JXL_RESTRICT row_quant_ac =
        block_group_rect.ConstRow(shared.raw_quant_field, by);
    const size_t ty = by / kColorTileDimInBlocks;
    const int8_t* JXL_RESTRICT row_cmap_x =
        cmap_rect.ConstRow(shared.cmap.ytox_map, ty);
    const int8_t* JXL_RESTRICT row_cmap_b =
        cmap_rect.ConstRow(shared.cmap.ytob_map, ty);
    const float* JXL_RESTRICT opsin_rows[3] = {
        group_rect.ConstPlaneRow(opsin, 0, by * kBlockDim),
        group_rect.ConstPlaneRow(opsin, 1, by * kBlockDim),
        group_rect.ConstPlaneRow(opsin, 2, by * kBlockDim),
    };
    AcStrategyRow acs_row = shared.ac_strategy.ConstRow(block_group_rect, by);
    for (size_t bx = 0; bx < gw_blocks; ++bx) {
      const AcStrategy acs = acs_row[bx];
      if (!acs.IsFirstBlock()) continue;
      const size_t tx = bx / kColorTileDimInBlocks;
      const size_t cov_x = acs.covered_blocks_x();
      const size_t cov_y = acs.covered_blocks_y();
      size_t xblocks = cov_x;
      size_t yblocks = cov_y;
      CoefficientLayout(&yblocks, &xblocks);
      const size_t size = kDCTBlockSize * xblocks * yblocks;

      AcTransformCache tr;
      tr.bx = static_cast<uint8_t>(bx);
      tr.by = static_cast<uint8_t>(by);
      tr.cov_x = static_cast<uint8_t>(cov_x);
      tr.cov_y = static_cast<uint8_t>(cov_y);
      tr.xblocks = static_cast<uint8_t>(xblocks);
      tr.yblocks = static_cast<uint8_t>(yblocks);
      tr.acs_raw = acs.RawStrategy();
      tr.tile_idx = static_cast<uint32_t>(
          ((gy0 + by) / tile_blocks) * tiles_x + (gx0 + bx) / tile_blocks);
      tr.base_quant_ac = row_quant_ac[bx];
      tr.cur_quant_ac = tr.base_quant_ac;
      tr.ytox = shared.cmap.base().YtoXRatio(row_cmap_x[tx]);
      tr.ytob = shared.cmap.base().YtoBRatio(row_cmap_b[tx]);
      for (size_t c = 0; c < 3; ++c) {
        for (size_t s = 0; s < 4; ++s) {
          tr.base_band_q[c][s] =
              block_group_rect.ConstRow(enc_state->band_qf[c][s], by)[bx];
        }
      }
      uint32_t bq10_bits = 0;
      memcpy(&bq10_bits, &tr.base_band_q[1][0], sizeof(bq10_bits));
      tr.act_bits = bq10_bits & 31u;
      tr.few_colors =
          (eob_scale > 0.0f)
              ? BlockHasFewColors(enc_state->orig_opsin, opsin_rows,
                                  opsin_stride, (gx0 + bx) * kBlockDim,
                                  (gy0 + by) * kBlockDim, bx * kBlockDim, cov_x,
                                  cov_y)
              : false;
      for (size_t c = 0; c < 3; ++c) {
        TransformFromPixels(acs.Strategy(), opsin_rows[c] + bx * kBlockDim,
                            opsin_stride, dct_buf, scratch_space);
        tr.src_dct[c].assign(dct_buf, dct_buf + size);
        DCFromLowestFrequencies(acs.Strategy(), dct_buf,
                                dc_src[c].data() + by * gw_blocks + bx,
                                gw_blocks, scratch_space);
      }
      cache->transforms.push_back(std::move(tr));
    }
  }

  const float mul =
      (enc_state->cparams.speed_tier < SpeedTier::kFalcon) ? 0.5f : 1.0f;
  const float step_x = shared.quantizer.GetDcStep(0) * mul;
  const float step_y = shared.quantizer.GetDcStep(1) * mul;
  const float step_b = shared.quantizer.GetDcStep(2) * mul;
  const float inv_step_x = shared.quantizer.GetInvDcStep(0) / mul;
  const float inv_step_y = shared.quantizer.GetInvDcStep(1) / mul;
  const float inv_step_b = shared.quantizer.GetInvDcStep(2) / mul;
  const float cfl_x = shared.cmap.base().DCFactors()[0];
  const float cfl_b = shared.cmap.base().DCFactors()[2];
  for (size_t i = 0; i < gw_blocks * gh_blocks; ++i) {
    const float sy = dc_src[1][i];
    const float qy = std::round(sy * inv_step_y);
    const float dy = qy * step_y;
    dc_err[1][i] = dy - sy;
    const float sx = dc_src[0][i];
    const float qx = std::round((sx - dy * cfl_x) * inv_step_x);
    const float dx = qx * step_x + dy * cfl_x;
    dc_err[0][i] = dx - sx;
    const float sb = dc_src[2][i];
    const float qb = std::round((sb - dy * cfl_b) * inv_step_b);
    const float db = qb * step_b + dy * cfl_b;
    dc_err[2][i] = db - sb;
  }

  for (const AcTransformCache& tr : cache->transforms) {
    const AcStrategy acs = AcStrategy::FromRawStrategy(tr.acs_raw);
    const size_t size = kDCTBlockSize * tr.xblocks * tr.yblocks;
    const size_t cw_px = tr.cov_x * kBlockDim;
    const size_t ch_px = tr.cov_y * kBlockDim;
    const size_t lx0 = tr.bx * kBlockDim;
    const size_t ly0 = tr.by * kBlockDim;
    const size_t lw = std::min(pw_px - lx0, cw_px);
    const size_t lh = std::min(ph_px - ly0, ch_px);
    for (size_t c = 0; c < 3; ++c) {
      memset(dct_buf, 0, size * sizeof(float));
      LowestFrequenciesFromDC(acs.Strategy(),
                              dc_err[c].data() + tr.by * gw_blocks + tr.bx,
                              gw_blocks, dct_buf, scratch_space);
      TransformToPixels(acs.Strategy(), dct_buf, px_buf, cw_px, scratch_space);
      float* JXL_RESTRICT base_p =
          cache->base_pre_gab[c].data() + ly0 * pw_px + lx0;
      for (size_t y = 0; y < lh; ++y) {
        for (size_t x = 0; x < lw; ++x) {
          base_p[y * pw_px + x] += px_buf[y * cw_px + x];
        }
      }
    }
  }
  return true;
}

Status EvalAcGroupCacheImpl(const PassesEncoderState* enc_state,
                            AcGroupBlockCache* cache, int32_t qf_step_override,
                            bool use_rdoq, bool commit_cur_slot,
                            bool only_write_changed, Image3F* out_pre_gab,
                            float* out_tile_bits, float* out_group_bits,
                            Rect* out_dirty_px) {
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  const PassesSharedState& shared = enc_state->shared;
  const AcTokenCostModel* JXL_RESTRICT tcm = enc_state->token_cost_model.get();
  const size_t gw_blocks = cache->gw_blocks;
  const size_t gh_blocks = cache->gh_blocks;
  const size_t pw_px = cache->pw_px;
  const size_t ph_px = cache->ph_px;

  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;
  const size_t quant_ints = 3 * AcStrategy::kMaxCoeffArea;
  const size_t rdoq_choice_bytes = kMaxRdoqWindow * kBandStride;
  const size_t mem_bytes = quant_ints * sizeof(int32_t) + rdoq_choice_bytes;
  JXL_ASSIGN_OR_RETURN(auto mem,
                       AlignedMemory::Create(memory_manager, mem_bytes));
  const size_t fmem_bytes =
      (5 * AcStrategy::kMaxCoeffArea + dct_scratch_size) * sizeof(float);
  JXL_ASSIGN_OR_RETURN(auto fmem,
                       AlignedMemory::Create(memory_manager, fmem_bytes));
  HWY_ALIGN int32_t* JXL_RESTRICT quantized = mem.address<int32_t>();
  uint8_t* JXL_RESTRICT rdoq_choices =
      reinterpret_cast<uint8_t*>(quantized + quant_ints);
  HWY_ALIGN float* JXL_RESTRICT coeffs_in = fmem.address<float>();
  HWY_ALIGN float* JXL_RESTRICT err_dct =
      coeffs_in + 3 * AcStrategy::kMaxCoeffArea;
  HWY_ALIGN float* JXL_RESTRICT px_buf = err_dct + AcStrategy::kMaxCoeffArea;
  float* JXL_RESTRICT scratch_space = px_buf + AcStrategy::kMaxCoeffArea;

  JXL_ASSIGN_OR_RETURN(Image3I group_nzeros,
                       Image3I::Create(memory_manager, gw_blocks, gh_blocks));

  const float d_target = enc_state->cparams.butteraugli_distance;
  const float dz_w =
      Clamp1((d_target - 0.35f) / (d_target + 0.55f) * 1.55f, 0.0f, 1.0f);
  const float eob_scale =
      (enc_state->cparams.speed_tier <= SpeedTier::kHare)
          ? Clamp1((d_target - 0.28f) / (d_target + 0.50f) * 1.45f, 0.0f, 1.0f)
          : 0.0f;
  const bool error_diffusion =
      enc_state->cparams.speed_tier <= SpeedTier::kSquirrel;

  size_t dirty_x0 = cache->px0 + pw_px;
  size_t dirty_y0 = cache->py0 + ph_px;
  size_t dirty_x1 = cache->px0;
  size_t dirty_y1 = cache->py0;
  double total_group_bits = 0.0;

  for (AcTransformCache& tr : cache->transforms) {
    const size_t bx = tr.bx;
    const size_t by = tr.by;
    const size_t cov_x = tr.cov_x;
    const size_t cov_y = tr.cov_y;
    const size_t xblocks = tr.xblocks;
    const size_t yblocks = tr.yblocks;
    const size_t covered_blocks = cov_x * cov_y;
    const size_t log2_covered_blocks =
        Num0BitsBelowLS1Bit_Nonzero(covered_blocks);
    const size_t size = kDCTBlockSize * covered_blocks;
    const AcStrategy acs = AcStrategy::FromRawStrategy(tr.acs_raw);
    const size_t ord = kStrategyOrder[tr.acs_raw];
    const int32_t quant_ac =
        Clamp1<int32_t>(tr.cur_quant_ac + qf_step_override, 1, 255);

    int32_t pred_nz[3];
    for (size_t c = 0; c < 3; ++c) {
      const int32_t* JXL_RESTRICT row_nz = group_nzeros.ConstPlaneRow(c, by);
      const int32_t* JXL_RESTRICT row_nz_top =
          (by > 0) ? group_nzeros.ConstPlaneRow(c, by - 1) : nullptr;
      pred_nz[c] = PredictFromTopAndLeft(row_nz_top, row_nz, bx, 32);
    }

    int hit_slot = -1;
    for (size_t s = 0; s < AcTransformCache::kNumSlots; ++s) {
      const AcBlockEvalSlot& slot = tr.slots[s];
      if (slot.quant_ac == quant_ac &&
          slot.use_rdoq == static_cast<uint8_t>(use_rdoq) &&
          slot.pred_nz[0] == static_cast<int16_t>(pred_nz[0]) &&
          slot.pred_nz[1] == static_cast<int16_t>(pred_nz[1]) &&
          slot.pred_nz[2] == static_cast<int16_t>(pred_nz[2])) {
        hit_slot = static_cast<int>(s);
        break;
      }
    }

    uint8_t slot_idx = 0;
    if (hit_slot >= 0) {
      slot_idx = static_cast<uint8_t>(hit_slot);
    } else {
      slot_idx = tr.next_slot;
      if (!commit_cur_slot && slot_idx == tr.cur_slot) {
        slot_idx = (slot_idx + 1) % AcTransformCache::kNumSlots;
      }
      tr.next_slot = (slot_idx + 1) % AcTransformCache::kNumSlots;
      AcBlockEvalSlot& slot = tr.slots[slot_idx];
      slot.quant_ac = quant_ac;
      slot.use_rdoq = static_cast<uint8_t>(use_rdoq);
      for (size_t c = 0; c < 3; ++c) {
        slot.pred_nz[c] = static_cast<int16_t>(pred_nz[c]);
      }

      const float ratio =
          static_cast<float>(quant_ac) / static_cast<float>(tr.base_quant_ac);
      float band_q[3][4];
      for (size_t c = 0; c < 3; ++c) {
        for (size_t s = 0; s < 4; ++s) {
          band_q[c][s] = tr.base_band_q[c][s] * ratio;
        }
      }
      uint32_t bq10_bits = 0;
      memcpy(&bq10_bits, &band_q[1][0], sizeof(bq10_bits));
      bq10_bits = (bq10_bits & ~31u) | tr.act_bits;
      memcpy(&band_q[1][0], &bq10_bits, sizeof(bq10_bits));

      const float q_block = shared.quantizer.Scale() * quant_ac;
      float zero_thres[3][8];
      ComputeBlockThresholds(acs.Strategy(), xblocks, yblocks, q_block, band_q,
                             d_target, dz_w, eob_scale, zero_thres);

      for (size_t c = 0; c < 3; ++c) {
        memcpy(coeffs_in + c * size, tr.src_dct[c].data(),
               size * sizeof(float));
      }

      const AcTokenCostModel* tcm_q = use_rdoq ? tcm : nullptr;
      uint8_t* rdoq_ptr = use_rdoq ? rdoq_choices : nullptr;
      int32_t final_nz[3] = {0, 0, 0};

      QuantizeRoundtripYBlockAC(
          const_cast<PassesEncoderState*>(enc_state), size, shared.quantizer,
          error_diffusion, acs.Strategy(), xblocks, yblocks, zero_thres,
          eob_scale, kDefaultQuantBias, quant_ac, coeffs_in, quantized, tcm_q,
          ord, covered_blocks, log2_covered_blocks, pred_nz[1], &final_nz[1],
          rdoq_ptr, tr.few_colors);

      constexpr HWY_CAPPED(float, kDCTBlockSize) df;
      const auto x_factor = Set(df, tr.ytox);
      const auto b_factor = Set(df, tr.ytob);
      for (size_t k = 0; k < size; k += Lanes(df)) {
        const auto in_x = Load(df, coeffs_in + k);
        const auto in_y = Load(df, coeffs_in + size + k);
        const auto in_b = Load(df, coeffs_in + 2 * size + k);
        Store(NegMulAdd(x_factor, in_y, in_x), df, coeffs_in + k);
        Store(NegMulAdd(b_factor, in_y, in_b), df, coeffs_in + 2 * size + k);
      }

      for (size_t c : {0, 2}) {
        QuantizeBlockAC(
            shared.quantizer, error_diffusion, c,
            c == 0 ? enc_state->x_qm_multiplier : enc_state->b_qm_multiplier,
            acs.Strategy(), xblocks, yblocks, zero_thres[c], eob_scale,
            coeffs_in + c * size, quant_ac, quantized + c * size, tcm_q, ord,
            covered_blocks, log2_covered_blocks, pred_nz[c], &final_nz[c],
            rdoq_ptr, tr.few_colors);
      }

      if (!use_rdoq) {
        for (size_t c = 0; c < 3; ++c) {
          const coeff_order_t* JXL_RESTRICT order =
              &tcm->coeff_orders[CoeffOrderOffset(ord, c)];
          int32_t nz = 0;
          for (size_t k = covered_blocks; k < size; ++k) {
            nz += static_cast<int32_t>(quantized[c * size + order[k]] != 0);
          }
          final_nz[c] = nz;
        }
      }

      float blk_bits = 0.0f;
      for (size_t c : {1, 0, 2}) {
        const coeff_order_t* JXL_RESTRICT order =
            &tcm->coeff_orders[CoeffOrderOffset(ord, c)];
        const size_t block_ctx =
            tcm->block_ctx_map.Context(/*dc_idx=*/0, quant_ac, ord, c);
        const uint32_t nzero_ctx =
            tcm->block_ctx_map.NonZeroContext(pred_nz[c], block_ctx);
        const int32_t nz = final_nz[c];
        blk_bits += tcm->ValBits(tcm->cluster_of_ctx[nzero_ctx],
                                 static_cast<uint32_t>(nz));
        const size_t histo_offset =
            tcm->block_ctx_map.ZeroDensityContextsOffset(block_ctx);
        size_t prev = (nz > static_cast<int32_t>(size / 16)) ? 0 : 1;
        size_t rem_nz = static_cast<size_t>(nz);
        for (size_t k = covered_blocks; k < size && rem_nz != 0; ++k) {
          const int32_t coeff = quantized[c * size + order[k]];
          const size_t ctx =
              histo_offset + ZeroDensityContext(rem_nz, k, covered_blocks,
                                                log2_covered_blocks, prev);
          blk_bits += tcm->ValBits(tcm->cluster_of_ctx[ctx], PackSigned(coeff));
          prev = (coeff != 0) ? 1 : 0;
          rem_nz -= prev;
        }
      }
      slot.bits = blk_bits;
      for (size_t c = 0; c < 3; ++c) {
        slot.shifted_nz[c] = static_cast<int32_t>(
            (final_nz[c] + covered_blocks - 1) >> log2_covered_blocks);
      }

      const size_t cw_px = cov_x * kBlockDim;
      const size_t ch_px = cov_y * kBlockDim;
      slot.ac_err_pixels.resize(3 * cw_px * ch_px);

      for (size_t k = 0; k < size; ++k) {
        err_dct[k] = coeffs_in[size + k] - tr.src_dct[1][k];
      }
      for (size_t y = 0; y < yblocks; ++y) {
        for (size_t x = 0; x < xblocks; ++x) {
          err_dct[y * kBlockDim * xblocks + x] = 0.0f;
        }
      }
      TransformToPixels(acs.Strategy(), err_dct, px_buf, cw_px, scratch_space);
      memcpy(slot.ac_err_pixels.data() + cw_px * ch_px, px_buf,
             cw_px * ch_px * sizeof(float));

      constexpr HWY_CAPPED(int32_t, kDCTBlockSize) di;
      for (size_t c : {0, 2}) {
        const float qm_mul =
            (c == 0) ? enc_state->x_qm_multiplier : enc_state->b_qm_multiplier;
        const auto inv_qv =
            Set(df, shared.quantizer.inv_quant_ac(quant_ac) / qm_mul);
        const float* JXL_RESTRICT dequant_matrix =
            shared.quantizer.DequantMatrix(acs.Strategy(), c);
        for (size_t k = 0; k < size; k += Lanes(df)) {
          const auto oq = Load(di, quantized + c * size + k);
          const auto adj = AdjustQuantBias(di, c, oq, kDefaultQuantBias);
          const auto dm = Load(df, dequant_matrix + k);
          const auto deq_cc = Mul(Mul(adj, dm), inv_qv);
          const auto res = Load(df, coeffs_in + c * size + k);
          Store(Sub(deq_cc, res), df, err_dct + k);
        }
        for (size_t y = 0; y < yblocks; ++y) {
          for (size_t x = 0; x < xblocks; ++x) {
            err_dct[y * kBlockDim * xblocks + x] = 0.0f;
          }
        }
        TransformToPixels(acs.Strategy(), err_dct, px_buf, cw_px,
                          scratch_space);
        memcpy(slot.ac_err_pixels.data() + c * cw_px * ch_px, px_buf,
               cw_px * ch_px * sizeof(float));
      }
    }

    const AcBlockEvalSlot& slot = tr.slots[slot_idx];
    for (size_t c = 0; c < 3; ++c) {
      const int32_t snz = slot.shifted_nz[c];
      for (size_t iy = 0; iy < cov_y; ++iy) {
        int32_t* JXL_RESTRICT r = group_nzeros.PlaneRow(c, by + iy);
        for (size_t ix = 0; ix < cov_x; ++ix) {
          r[bx + ix] = snz;
        }
      }
    }

    total_group_bits += static_cast<double>(slot.bits);
    if (out_tile_bits != nullptr) {
      out_tile_bits[tr.tile_idx] += slot.bits;
    }

    const bool changed = !only_write_changed || (slot_idx != tr.cur_slot);
    tr.cand_slot = slot_idx;
    if (commit_cur_slot) {
      tr.cur_slot = slot_idx;
    }

    if (out_pre_gab != nullptr && changed) {
      const size_t cw_px = cov_x * kBlockDim;
      const size_t ch_px = cov_y * kBlockDim;
      const size_t lx0 = bx * kBlockDim;
      const size_t ly0 = by * kBlockDim;
      const size_t lw = std::min(pw_px - lx0, cw_px);
      const size_t lh = std::min(ph_px - ly0, ch_px);
      const size_t fx0 = cache->px0 + lx0;
      const size_t fy0 = cache->py0 + ly0;
      dirty_x0 = std::min(dirty_x0, fx0);
      dirty_y0 = std::min(dirty_y0, fy0);
      dirty_x1 = std::max(dirty_x1, fx0 + lw);
      dirty_y1 = std::max(dirty_y1, fy0 + lh);
      for (size_t c = 0; c < 3; ++c) {
        const float* JXL_RESTRICT base_p =
            cache->base_pre_gab[c].data() + ly0 * pw_px + lx0;
        const float* JXL_RESTRICT err_p =
            slot.ac_err_pixels.data() + c * cw_px * ch_px;
        for (size_t y = 0; y < lh; ++y) {
          float* JXL_RESTRICT dst = out_pre_gab->PlaneRow(c, fy0 + y) + fx0;
          for (size_t x = 0; x < lw; ++x) {
            dst[x] = base_p[y * pw_px + x] + err_p[y * cw_px + x];
          }
        }
      }
    }
  }

  if (out_group_bits != nullptr) {
    *out_group_bits = static_cast<float>(total_group_bits);
  }
  if (out_dirty_px != nullptr) {
    if (dirty_x1 > dirty_x0 && dirty_y1 > dirty_y0) {
      *out_dirty_px =
          Rect(dirty_x0, dirty_y0, dirty_x1 - dirty_x0, dirty_y1 - dirty_y0);
    } else {
      *out_dirty_px = Rect(0, 0, 0, 0);
    }
  }
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include <atomic>

namespace jxl {
HWY_EXPORT(ComputeCoefficients);
HWY_EXPORT(InitAcGroupCacheImpl);
HWY_EXPORT(EvalAcGroupCacheImpl);
Status ComputeCoefficients(size_t group_idx, PassesEncoderState* enc_state,
                           const Image3F& opsin, const Rect& rect,
                           Image3F* dc) {
  return HWY_DYNAMIC_DISPATCH(ComputeCoefficients)(group_idx, enc_state, opsin,
                                                   rect, dc);
}

namespace {

constexpr double kSsim2Weights[108] = {
    0.0,
    0.0007376606707406586,
    0.0,
    0.0,
    0.0007793481682867309,
    0.0,
    0.0,
    0.0004371155730107379,
    0.0,
    1.1041726426657346,
    0.00066284834129271,
    0.00015231632783718752,
    0.0,
    0.0016406437456599754,
    0.0,
    1.8422455520539298,
    11.441172603757666,
    0.0,
    0.0007989109436015163,
    0.000176816438078653,
    0.0,
    1.8787594979546387,
    10.94906990605142,
    0.0,
    0.0007289346991508072,
    0.9677937080626833,
    0.0,
    0.00014003424285435884,
    0.9981766977854967,
    0.00031949755934435053,
    0.0004550992113792063,
    0.0,
    0.0,
    0.0013648766163243398,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.466890328078848,
    0.0,
    17.445833984131262,
    0.0006235601634041466,
    0.0,
    0.0,
    6.683678146179332,
    0.00037724407979611296,
    1.027889937768264,
    225.20515300849274,
    0.0,
    0.0,
    19.213238186143016,
    0.0011401524586618361,
    0.001237755635509985,
    176.39317598450694,
    0.0,
    0.0,
    24.43300999870476,
    0.28520802612117757,
    0.0004485436923833408,
    0.0,
    0.0,
    0.0,
    34.77906344483772,
    44.835625328877896,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0008680556573291698,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0005313191874358747,
    0.0,
    0.00016533814161379112,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0004179171803251336,
    0.0017290828234722833,
    0.0,
    0.0020827005846636437,
    0.0,
    0.0,
    8.826982764996862,
    23.19243343998926,
    0.0,
    95.1080498811086,
    0.9863978034400682,
    0.9834382792465353,
    0.0012286405048278493,
    171.2667255897307,
    0.9807858872435379,
    0.0,
    0.0,
    0.0,
    0.0005130064588990679,
    0.0,
    0.00010854057858411537,
};

constexpr float kCvvdpLogLum[32] = {
    -2.30102992f, -2.09777093f, -1.89451194f, -1.69125295f, -1.48799384f,
    -1.28473485f, -1.08147585f, -0.87821674f, -0.67495775f, -0.47169870f,
    -0.26843968f, -0.06518064f, 0.13807839f,  0.34133741f,  0.54459643f,
    0.74785548f,  0.95111454f,  1.15437353f,  1.35763264f,  1.56089163f,
    1.76415062f,  1.96740973f,  2.17066860f,  2.37392783f,  2.57718682f,
    2.78044581f,  2.98370481f,  3.18696380f,  3.39022279f,  3.59348202f,
    3.79674101f,  4.00000000f,
};

constexpr float kCvvdpLogCsf[10][3][32] = {
    {
        {-3.9115868f, -3.6279184f, -3.3471669f, -3.0709100f, -2.7997575f,
         -2.5339674f, -2.2736796f, -2.0190129f, -1.7701128f, -1.5271881f,
         -1.2905534f, -1.0606860f, -0.8383038f, -0.6244686f, -0.4207079f,
         -0.2291263f, -0.0524345f, 0.1062224f,  0.2437609f,  0.3579663f,
         0.4483165f,  0.5163421f,  0.5652562f,  0.5990798f,  0.6217617f,
         0.6366066f,  0.6452460f,  0.6426774f,  0.6144616f,  0.5485542f,
         0.4442597f,  0.3084537f},
        {-0.6989700f, -0.6989700f, -0.6989700f, -0.6989700f, -0.6989700f,
         -0.6989700f, -0.5181862f, 0.0663240f,  0.1648103f,  0.2630289f,
         0.3608231f,  0.4579473f,  0.5540207f,  0.6484633f,  0.7404155f,
         0.8286557f,  0.9115598f,  0.9871763f,  1.0535042f,  1.1089779f,
         1.1529797f,  1.1860733f,  1.2097902f,  1.2261229f,  1.2370332f,
         1.2441641f,  1.2487555f,  1.2516823f,  1.2535361f,  1.2547053f,
         1.2554409f,  1.2559028f},
        {-1.0120667f, -0.9328088f, -0.8535553f, -0.7743085f, -0.6950726f,
         -0.6158539f, -0.5366629f, -0.4575159f, -0.3784391f, -0.2994741f,
         -0.2206871f, -0.1421825f, -0.0641246f, 0.0132305f,  0.0894898f,
         0.1640631f,  0.2360957f,  0.3044164f,  0.3675510f,  0.4238736f,
         0.4719359f,  0.5108959f,  0.5408239f,  0.5626749f,  0.5779493f,
         0.5882674f,  0.5950646f,  0.5994644f,  0.6022791f,  0.6040659f,
         0.6051946f,  0.6059053f},
    },
    {
        {-1.7440390f, -1.5073504f, -1.2760881f, -1.0502638f, -0.8298956f,
         -0.6150106f, -0.4056508f, -0.2018809f, -0.0038002f, 0.1884395f,
         0.3746070f,  0.5543532f,  0.7271580f,  0.8922600f,  1.0485766f,
         1.1946355f,  1.3285655f,  1.4482297f,  1.5515767f,  1.6371969f,
         1.7048825f,  1.7558689f,  1.7925758f,  1.8179951f,  1.8350639f,
         1.8462406f,  1.8525184f,  1.8484445f,  1.8192752f,  1.7527663f,
         1.6480932f,  1.5120489f},
        {-0.6989700f, -0.2054516f, 0.1081825f,  0.2070770f,  0.3059454f,
         0.4047721f,  0.5035326f,  0.6021873f,  0.7006736f,  0.7988922f,
         0.8966864f,  0.9938105f,  1.0898840f,  1.1843266f,  1.2762787f,
         1.3645189f,  1.4474231f,  1.5230396f,  1.5893675f,  1.6448411f,
         1.6888429f,  1.7219366f,  1.7456535f,  1.7619861f,  1.7728965f,
         1.7800274f,  1.7846187f,  1.7875456f,  1.7893994f,  1.7905686f,
         1.7913042f,  1.7917661f},
        {-0.4746738f, -0.3954160f, -0.3161624f, -0.2369157f, -0.1576797f,
         -0.0784611f, 0.0007299f,  0.0798769f,  0.1589538f,  0.2379187f,
         0.3167058f,  0.3952104f,  0.4732682f,  0.5506234f,  0.6268826f,
         0.7014559f,  0.7734886f,  0.8418092f,  0.9049439f,  0.9612665f,
         1.0093286f,  1.0482887f,  1.0782167f,  1.1000677f,  1.1153422f,
         1.1256603f,  1.1324574f,  1.1368573f,  1.1396719f,  1.1414588f,
         1.1425874f,  1.1432981f},
    },
    {
        {-0.6764267f, -0.4710318f, -0.2710610f, -0.0765237f, 0.1125657f,
         0.2961850f,  0.4742999f,  0.6468585f,  0.8137814f,  0.9749482f,
         1.1301776f,  1.2791984f,  1.4216113f,  1.5568390f,  1.6840719f,
         1.8022246f,  1.9099410f,  2.0057021f,  2.0880893f,  2.1561865f,
         2.2099785f,  2.2505196f,  2.2797434f,  2.3000108f,  2.3136385f,
         2.3225667f,  2.3273968f,  2.3224001f,  2.2926462f,  2.2257689f,
         2.1208638f,  1.9846738f},
        {0.2073214f, 0.3062424f, 0.4051532f, 0.5040477f, 0.6029160f,
         0.7017429f, 0.8005033f, 0.8991580f, 0.9976443f, 1.0958628f,
         1.1936570f, 1.2907812f, 1.3868546f, 1.4812974f, 1.5732494f,
         1.6614896f, 1.7443938f, 1.8200103f, 1.8863382f, 1.9418118f,
         1.9858137f, 2.0189074f, 2.0426241f, 2.0589567f, 2.0698672f,
         2.0769981f, 2.0815893f, 2.0845164f, 2.0863701f, 2.0875393f,
         2.0882748f, 2.0887369f},
        {-0.1796570f, -0.1003992f, -0.0211456f, 0.0581012f, 0.1373371f,
         0.2165557f,  0.2957468f,  0.3748938f,  0.4539706f, 0.5329356f,
         0.6117226f,  0.6902272f,  0.7682851f,  0.8456402f, 0.9218995f,
         0.9964728f,  1.0685054f,  1.1368261f,  1.1999607f, 1.2562832f,
         1.3043456f,  1.3433056f,  1.3732335f,  1.3950846f, 1.4103590f,
         1.4206771f,  1.4274742f,  1.4318741f,  1.4346888f, 1.4364756f,
         1.4376043f,  1.4383149f},
    },
    {
        {0.1370946f, 0.3111936f, 0.4798719f, 0.6431218f, 0.8009320f,
         0.9532852f, 1.1001552f, 1.2415025f, 1.3772675f, 1.5073615f,
         1.6316526f, 1.7499479f, 1.8619689f, 1.9673224f, 2.0654713f,
         2.1557180f, 2.2372209f, 2.3090791f, 2.3705064f, 2.4210805f,
         2.4609788f, 2.4910745f, 2.5128152f, 2.5279306f, 2.5381176f,
         2.5447973f, 2.5481797f, 2.5422601f, 2.5119218f, 2.4446758f,
         2.3395390f, 2.2032035f},
        {0.4660442f, 0.5649653f, 0.6638760f, 0.7627705f, 0.8616389f,
         0.9604657f, 1.0592261f, 1.1578808f, 1.2563671f, 1.3545857f,
         1.4523799f, 1.5495041f, 1.6455775f, 1.7400201f, 1.8319722f,
         1.9202124f, 2.0031166f, 2.0787332f, 2.1450610f, 2.2005346f,
         2.2445365f, 2.2776301f, 2.3013470f, 2.3176796f, 2.3285901f,
         2.3357210f, 2.3403122f, 2.3432391f, 2.3450928f, 2.3462622f,
         2.3469977f, 2.3474595f},
        {0.0793351f, 0.1585929f, 0.2378465f, 0.3170932f, 0.3963292f,
         0.4755478f, 0.5547388f, 0.6338859f, 0.7129627f, 0.7919277f,
         0.8707147f, 0.9492193f, 1.0272772f, 1.1046323f, 1.1808916f,
         1.2554648f, 1.3274975f, 1.3958182f, 1.4589528f, 1.5152754f,
         1.5633376f, 1.6022977f, 1.6322257f, 1.6540766f, 1.6693511f,
         1.6796692f, 1.6864663f, 1.6908662f, 1.6936809f, 1.6954677f,
         1.6965964f, 1.6973070f},
    },
    {
        {0.6754775f, 0.8182807f, 0.9556662f, 1.0876286f, 1.2141597f,
         1.3352470f, 1.4508720f, 1.5610079f, 1.6656150f, 1.7646361f,
         1.8579891f, 1.9455590f, 2.0271880f, 2.1026672f, 2.1717324f,
         2.2340731f, 2.2893623f, 2.3373175f, 2.3777849f, 2.4108359f,
         2.4368406f, 2.4564912f, 2.4707484f, 2.4807121f, 2.4874581f,
         2.4918892f, 2.4938240f, 2.4869817f, 2.4560591f, 2.3884445f,
         2.2830760f, 2.1465952f},
        {0.6538805f, 0.7528015f, 0.8517123f, 0.9506068f, 1.0494751f,
         1.1483020f, 1.2470624f, 1.3457171f, 1.4442034f, 1.5424220f,
         1.6402162f, 1.7373404f, 1.8334138f, 1.9278564f, 2.0198085f,
         2.1080488f, 2.1909529f, 2.2665694f, 2.3328973f, 2.3883709f,
         2.4323727f, 2.4654664f, 2.4891833f, 2.5055160f, 2.5164263f,
         2.5235572f, 2.5281486f, 2.5310754f, 2.5329292f, 2.5340984f,
         2.5348339f, 2.5352959f},
        {0.2775347f, 0.3567925f, 0.4360461f, 0.5152929f, 0.5945288f,
         0.6737474f, 0.7529385f, 0.8320855f, 0.9111623f, 0.9901273f,
         1.0689143f, 1.1474189f, 1.2254768f, 1.3028319f, 1.3790912f,
         1.4536645f, 1.5256972f, 1.5940178f, 1.6571524f, 1.7134750f,
         1.7615372f, 1.8004973f, 1.8304253f, 1.8522763f, 1.8675507f,
         1.8778688f, 1.8846659f, 1.8890658f, 1.8918806f, 1.8936673f,
         1.8947959f, 1.8955067f},
    },
    {
        {0.9693742f, 1.0808815f, 1.1869743f, 1.2876493f, 1.3829011f,
         1.4727223f, 1.5571025f, 1.6360271f, 1.7094763f, 1.7774244f,
         1.8398392f, 1.8966837f, 1.9479209f, 1.9935258f, 2.0335071f,
         2.0679415f, 2.0970174f, 2.1210696f, 2.1405772f, 2.1561050f,
         2.1682160f, 2.1774215f, 2.1841955f, 2.1890072f, 2.1923125f,
         2.1944949f, 2.1949822f, 2.1872173f, 2.1557104f, 2.0877275f,
         1.9821279f, 1.8455028f},
        {0.7302759f, 0.8291969f, 0.9281077f, 1.0270021f, 1.1258705f,
         1.2246974f, 1.3234578f, 1.4221125f, 1.5205988f, 1.6188174f,
         1.7166115f, 1.8137358f, 1.9098092f, 2.0042519f, 2.0962040f,
         2.1844441f, 2.2673482f, 2.3429648f, 2.4092926f, 2.4647664f,
         2.5087682f, 2.5418617f, 2.5655786f, 2.5819112f, 2.5928218f,
         2.5999526f, 2.6045438f, 2.6074707f, 2.6093246f, 2.6104938f,
         2.6112293f, 2.6116912f},
        {0.3713476f, 0.4506054f, 0.5298590f, 0.6091057f, 0.6883417f,
         0.7675603f, 0.8467514f, 0.9258983f, 1.0049752f, 1.0839401f,
         1.1627272f, 1.2412318f, 1.3192897f, 1.3966448f, 1.4729041f,
         1.5474773f, 1.6195100f, 1.6878307f, 1.7509652f, 1.8072879f,
         1.8553501f, 1.8943101f, 1.9242382f, 1.9460891f, 1.9613635f,
         1.9716817f, 1.9784788f, 1.9828787f, 1.9856934f, 1.9874801f,
         1.9886088f, 1.9893195f},
    },
    {
        {1.0536571f, 1.1338684f, 1.2086686f, 1.2780561f, 1.3420287f,
         1.4005840f, 1.4537192f, 1.5014324f, 1.5437238f, 1.5805990f,
         1.6120756f, 1.6381946f, 1.6590400f, 1.6802212f, 1.6991292f,
         1.7291838f, 1.7590048f, 1.7870416f, 1.8126379f, 1.8351169f,
         1.8539436f, 1.8688992f, 1.8801632f, 1.8882456f, 1.8938170f,
         1.8975157f, 1.8989939f, 1.8918668f, 1.8607665f, 1.7930413f,
         1.6876047f, 1.5510836f},
        {0.7028861f, 0.8018071f, 0.9007179f, 0.9996124f, 1.0984808f,
         1.1973076f, 1.2960681f, 1.3947227f, 1.4932090f, 1.5914276f,
         1.6892218f, 1.7863459f, 1.8824194f, 1.9768621f, 2.0688142f,
         2.1570544f, 2.2399585f, 2.3155751f, 2.3819029f, 2.4373765f,
         2.4813783f, 2.5144720f, 2.5381889f, 2.5545215f, 2.5654320f,
         2.5725629f, 2.5771541f, 2.5800810f, 2.5819349f, 2.5831041f,
         2.5838396f, 2.5843015f},
        {0.3470458f, 0.4263036f, 0.5055572f, 0.5848039f, 0.6640399f,
         0.7432585f, 0.8224496f, 0.9015966f, 0.9806734f, 1.0596384f,
         1.1384254f, 1.2169300f, 1.2949879f, 1.3723430f, 1.4486023f,
         1.5231756f, 1.5952082f, 1.6635289f, 1.7266635f, 1.7829861f,
         1.8310483f, 1.8700084f, 1.8999364f, 1.9217874f, 1.9370618f,
         1.9473799f, 1.9541770f, 1.9585769f, 1.9613916f, 1.9631784f,
         1.9643071f, 1.9650178f},
    },
    {
        {0.9510262f, 0.9999416f, 1.0434492f, 1.0815492f, 1.1142425f,
         1.1415318f, 1.1634222f, 1.1811666f, 1.2082567f, 1.2362776f,
         1.2694281f, 1.3024370f, 1.3352227f, 1.3676581f, 1.3995496f,
         1.4306101f, 1.4604310f, 1.4884679f, 1.5140642f, 1.5365431f,
         1.5553699f, 1.5703256f, 1.5815897f, 1.5896722f, 1.5952437f,
         1.5989425f, 1.6004209f, 1.5932940f, 1.5621940f, 1.4944696f,
         1.3890342f, 1.2525155f},
        {0.6186906f, 0.7176116f, 0.8165224f, 0.9154168f, 1.0142853f,
         1.1131120f, 1.2118725f, 1.3105271f, 1.4090134f, 1.5072320f,
         1.6050263f, 1.7021504f, 1.7982238f, 1.8926665f, 1.9846186f,
         2.0728589f, 2.1557630f, 2.2313796f, 2.2977074f, 2.3531809f,
         2.3971827f, 2.4302764f, 2.4539934f, 2.4703260f, 2.4812365f,
         2.4883674f, 2.4929586f, 2.4958854f, 2.4977394f, 2.4989086f,
         2.4996441f, 2.5001059f},
        {0.2487745f, 0.3280323f, 0.4072858f, 0.4865326f, 0.5657685f,
         0.6449871f, 0.7241782f, 0.8033252f, 0.8824020f, 0.9613670f,
         1.0401540f, 1.1186587f, 1.1967165f, 1.2740716f, 1.3503309f,
         1.4249042f, 1.4969368f, 1.5652575f, 1.6283921f, 1.6847147f,
         1.7327769f, 1.7717370f, 1.8016650f, 1.8235159f, 1.8387904f,
         1.8491086f, 1.8559057f, 1.8603055f, 1.8631202f, 1.8649070f,
         1.8660357f, 1.8667463f},
    },
    {
        {0.6670897f, 0.6847092f, 0.7039792f, 0.7364048f, 0.7692864f,
         0.8026533f, 0.8360064f, 0.8693374f, 0.9026332f, 0.9358728f,
         0.9690232f, 1.0020322f, 1.0348179f, 1.0672533f, 1.0991448f,
         1.1302054f, 1.1600263f, 1.1880631f, 1.2136595f, 1.2361385f,
         1.2549653f, 1.2699210f, 1.2811852f, 1.2892677f, 1.2948393f,
         1.2985383f, 1.3000168f, 1.2928903f, 1.2617907f, 1.1940669f,
         1.0886330f, 0.9521169f},
        {0.5029513f, 0.6018723f, 0.7007831f, 0.7996776f, 0.8985460f,
         0.9973728f, 1.0961332f, 1.1947879f, 1.2932742f, 1.3914928f,
         1.4892870f, 1.5864112f, 1.6824846f, 1.7769272f, 1.8688794f,
         1.9571195f, 2.0400238f, 2.1156403f, 2.1819682f, 2.2374417f,
         2.2814436f, 2.3145372f, 2.3382542f, 2.3545868f, 2.3654971f,
         2.3726279f, 2.3772194f, 2.3801462f, 2.3819999f, 2.3831691f,
         2.3839049f, 2.3843667f},
        {0.1096632f, 0.1889210f, 0.2681746f, 0.3474213f, 0.4266573f,
         0.5058759f, 0.5850669f, 0.6642140f, 0.7432908f, 0.8222557f,
         0.9010428f, 0.9795474f, 1.0576053f, 1.1349604f, 1.2112197f,
         1.2857929f, 1.3578256f, 1.4261463f, 1.4892808f, 1.5456035f,
         1.5936657f, 1.6326257f, 1.6625538f, 1.6844047f, 1.6996791f,
         1.7099973f, 1.7167944f, 1.7211943f, 1.7240089f, 1.7257957f,
         1.7269244f, 1.7276351f},
    },
    {
        {0.5404148f, 0.5738014f, 0.6071858f, 0.6405669f, 0.6739426f,
         0.7073095f, 0.7406626f, 0.7739936f, 0.8072893f, 0.8405289f,
         0.8736793f, 0.9066883f, 0.9394740f, 0.9719095f, 1.0038010f,
         1.0348614f, 1.0646825f, 1.0927193f, 1.1183157f, 1.1407948f,
         1.1596215f, 1.1745772f, 1.1858414f, 1.1939241f, 1.1994957f,
         1.2031946f, 1.2046733f, 1.1975467f, 1.1664473f, 1.0987238f,
         0.9932902f, 0.8567749f},
        {0.4612012f, 0.5601222f, 0.6590330f, 0.7579275f, 0.8567958f,
         0.9556227f, 1.0543830f, 1.1530378f, 1.2515241f, 1.3497427f,
         1.4475368f, 1.5446610f, 1.6407344f, 1.7351772f, 1.8271292f,
         1.9153694f, 1.9982736f, 2.0738902f, 2.1402180f, 2.1956916f,
         2.2396934f, 2.2727871f, 2.2965040f, 2.3128366f, 2.3237469f,
         2.3308778f, 2.3354692f, 2.3383961f, 2.3402498f, 2.3414192f,
         2.3421547f, 2.3426166f},
        {0.0593085f, 0.1385663f, 0.2178199f, 0.2970666f, 0.3763025f,
         0.4555212f, 0.5347123f, 0.6138592f, 0.6929361f, 0.7719010f,
         0.8506880f, 0.9291927f, 1.0072505f, 1.0846057f, 1.1608649f,
         1.2354382f, 1.3074709f, 1.3757915f, 1.4389261f, 1.4952488f,
         1.5433110f, 1.5822710f, 1.6121991f, 1.6340500f, 1.6493244f,
         1.6596426f, 1.6664397f, 1.6708395f, 1.6736543f, 1.6754410f,
         1.6765697f, 1.6772804f},
    },
};

inline float FastCbrtPositive(float x) {
  if (x <= 0.0f) return 0.0f;
  uint32_t bits = 0;
  memcpy(&bits, &x, sizeof(bits));
  bits = (bits & 0x7fffffffu) / 3u + 709958130u;
  float s = 0.0f;
  memcpy(&s, &bits, sizeof(s));
  for (int i = 0; i < 2; ++i) {
    const float s3 = s * s * s;
    s *= (2.0f * x + s3) / (2.0f * s3 + x);
  }
  return s;
}

void XybRowToLinearRgb(const float* JXL_RESTRICT row_x,
                       const float* JXL_RESTRICT row_y,
                       const float* JXL_RESTRICT row_b, size_t len,
                       const Matrix3x3& inv_m, const float biases[3],
                       const float biases_cbrt[3], float inv_scale,
                       float* JXL_RESTRICT out_r, float* JXL_RESTRICT out_g,
                       float* JXL_RESTRICT out_b) {
  const float m00 = inv_m[0][0] * inv_scale;
  const float m01 = inv_m[0][1] * inv_scale;
  const float m02 = inv_m[0][2] * inv_scale;
  const float m10 = inv_m[1][0] * inv_scale;
  const float m11 = inv_m[1][1] * inv_scale;
  const float m12 = inv_m[1][2] * inv_scale;
  const float m20 = inv_m[2][0] * inv_scale;
  const float m21 = inv_m[2][1] * inv_scale;
  const float m22 = inv_m[2][2] * inv_scale;
  const float br = biases[0];
  const float bg = biases[1];
  const float bb = biases[2];
  const float cbr = biases_cbrt[0];
  const float cbg = biases_cbrt[1];
  const float cbb = biases_cbrt[2];
  for (size_t i = 0; i < len; ++i) {
    const float x = row_x[i];
    const float y = row_y[i];
    const float b = row_b[i];
    const float gr = (y + x) - cbr;
    const float gg = (y - x) - cbg;
    const float gb = b - cbb;
    const float mr = gr * gr * gr + br;
    const float mg = gg * gg * gg + bg;
    const float mb = gb * gb * gb + bb;
    out_r[i] = m00 * mr + m01 * mg + m02 * mb;
    out_g[i] = m10 * mr + m11 * mg + m12 * mb;
    out_b[i] = m20 * mr + m21 * mg + m22 * mb;
  }
}

float EstimateGroupQfBits(const AcGroupBlockCache& cache,
                          const AcTokenCostModel& tcm, int32_t qf_step_override,
                          float* out_tile_bits) {
  const float qf_scale = Clamp1(tcm.qf_block_bits / 1.50f, 0.50f, 2.00f);
  int32_t prev_qf = 0;
  double total_bits = 0.0;
  for (const AcTransformCache& tr : cache.transforms) {
    const int32_t q =
        Clamp1<int32_t>(tr.cur_quant_ac + qf_step_override, 1, 255) - 1;
    const uint32_t a = static_cast<uint32_t>(std::abs(q - prev_qf));
    prev_qf = q;
    float bits =
        (a == 0)
            ? 0.65f
            : (a == 1 ? 2.40f : 2.40f + 1.25f * std::log2(static_cast<float>(a)));
    bits *= qf_scale;
    total_bits += static_cast<double>(bits);
    if (out_tile_bits != nullptr) {
      out_tile_bits[tr.tile_idx] += bits;
    }
  }
  return static_cast<float>(total_bits);
}

void ReconstructLinearRgb(const FrameHeader& frame_header,
                          const PassesSharedState& shared,
                          const Image3F& pre_gab, const Image3F* feat_xyb,
                          const Rect& dirty_rect, Image3F* out_lin) {
  const size_t xsize = shared.frame_dim.xsize;
  const size_t ysize = shared.frame_dim.ysize;
  if (dirty_rect.xsize() == 0 || dirty_rect.ysize() == 0) return;
  const size_t x0 = std::min(dirty_rect.x0(), xsize);
  const size_t y0 = std::min(dirty_rect.y0(), ysize);
  const size_t x1 = std::min(dirty_rect.x0() + dirty_rect.xsize(), xsize);
  const size_t y1 = std::min(dirty_rect.y0() + dirty_rect.ysize(), ysize);
  if (x0 >= x1 || y0 >= y1) return;
  const size_t len = x1 - x0;

  const auto& im = shared.metadata->transform_data.opsin_inverse_matrix;
  float biases_cbrt[3];
  for (size_t c = 0; c < 3; ++c) {
    biases_cbrt[c] = std::cbrt(im.opsin_biases[c]);
  }
  const float inv_scale = 255.0f / shared.metadata->m.IntensityTarget();

  const bool do_gab = frame_header.loop_filter.gab;
  float gab_w[3][3] = {};
  if (do_gab) {
    const LoopFilter& lf = frame_header.loop_filter;
    const float raw_w1[3] = {lf.gab_x_weight1, lf.gab_y_weight1,
                             lf.gab_b_weight1};
    const float raw_w2[3] = {lf.gab_x_weight2, lf.gab_y_weight2,
                             lf.gab_b_weight2};
    for (size_t c = 0; c < 3; ++c) {
      const float div = 1.0f + 4.0f * (raw_w1[c] + raw_w2[c]);
      const float mul = 1.0f / div;
      gab_w[c][0] = mul;
      gab_w[c][1] = raw_w1[c] * mul;
      gab_w[c][2] = raw_w2[c] * mul;
    }
  }

  std::vector<float> xyb_row[3];
  for (size_t c = 0; c < 3; ++c) {
    xyb_row[c].resize(len);
  }

  for (size_t y = y0; y < y1; ++y) {
    const size_t yt = (y > 0) ? y - 1 : 0;
    const size_t yb = (y + 1 < ysize) ? y + 1 : ysize - 1;
    for (size_t c = 0; c < 3; ++c) {
      float* JXL_RESTRICT dst = xyb_row[c].data();
      if (do_gab) {
        const float* JXL_RESTRICT rt = pre_gab.ConstPlaneRow(c, yt);
        const float* JXL_RESTRICT rm = pre_gab.ConstPlaneRow(c, y);
        const float* JXL_RESTRICT rb = pre_gab.ConstPlaneRow(c, yb);
        const float w0 = gab_w[c][0];
        const float w1 = gab_w[c][1];
        const float w2 = gab_w[c][2];
        for (size_t i = 0; i < len; ++i) {
          const size_t x = x0 + i;
          const size_t xl = (x > 0) ? x - 1 : 0;
          const size_t xr = (x + 1 < xsize) ? x + 1 : xsize - 1;
          const float s0 = rm[x];
          const float s1 = rm[xl] + rm[xr] + rt[x] + rb[x];
          const float s2 = rt[xl] + rt[xr] + rb[xl] + rb[xr];
          dst[i] = s0 * w0 + s1 * w1 + s2 * w2;
        }
      } else {
        const float* JXL_RESTRICT rm = pre_gab.ConstPlaneRow(c, y) + x0;
        memcpy(dst, rm, len * sizeof(float));
      }
      if (feat_xyb != nullptr) {
        const float* JXL_RESTRICT rf = feat_xyb->ConstPlaneRow(c, y) + x0;
        for (size_t i = 0; i < len; ++i) {
          dst[i] += rf[i];
        }
      }
    }
    XybRowToLinearRgb(xyb_row[0].data(), xyb_row[1].data(), xyb_row[2].data(),
                      len, im.inverse_matrix, im.opsin_biases, biases_cbrt,
                      inv_scale, out_lin->PlaneRow(0, y) + x0,
                      out_lin->PlaneRow(1, y) + x0,
                      out_lin->PlaneRow(2, y) + x0);
  }
}

class Ssim2IncrementalArbiter {
 public:
  static constexpr size_t kMaxScales = 6;
  static constexpr size_t kRadius = 5;
  static constexpr size_t kTaps = 11;
  static constexpr size_t kTile = 8;

  struct Checkpoint {
    struct RgbPatch {
      size_t scale;
      size_t x0, y0, x1, y1;
      size_t offset;
    };
    struct TilePatch {
      size_t scale;
      size_t tile_idx;
      double s1[9];
      double s4[9];
    };
    std::vector<RgbPatch> rgb_patches;
    std::vector<float> rgb_xyb_data;
    std::vector<TilePatch> tile_patches;
    double prev_total_s1[kMaxScales][9] = {};
    double prev_total_s4[kMaxScales][9] = {};
  };

  void Init(const Image3F& src_lin) {
    w0_ = src_lin.xsize();
    h0_ = src_lin.ysize();
    InitKernel();
    scales_.clear();
    size_t w = w0_;
    size_t h = h0_;
    while (scales_.size() < kMaxScales) {
      ScaleState sc;
      sc.w = w;
      sc.h = h;
      sc.tw = DivCeil(w, kTile);
      sc.th = DivCeil(h, kTile);
      for (size_t c = 0; c < 3; ++c) {
        sc.src_rgb[c].resize(w * h);
        sc.rec_rgb[c].resize(w * h);
        sc.src_xyb[c].resize(w * h);
        sc.rec_xyb[c].resize(w * h);
        sc.blur_src[c].resize(w * h);
        sc.blur_src_sq[c].resize(w * h);
      }
      for (size_t m = 0; m < 9; ++m) {
        sc.tile_s1[m].assign(sc.tw * sc.th, 0.0);
        sc.tile_s4[m].assign(sc.tw * sc.th, 0.0);
      }
      scales_.push_back(std::move(sc));
      if (w < 8 || h < 8 || scales_.size() == kMaxScales) break;
      w = (w + 1) / 2;
      h = (h + 1) / 2;
    }

    for (size_t c = 0; c < 3; ++c) {
      for (size_t y = 0; y < h0_; ++y) {
        memcpy(scales_[0].src_rgb[c].data() + y * w0_,
               src_lin.ConstPlaneRow(c, y), w0_ * sizeof(float));
      }
    }
    for (size_t s = 1; s < scales_.size(); ++s) {
      DownsampleRect(scales_[s - 1].src_rgb, scales_[s - 1].w, scales_[s - 1].h,
                     0, 0, scales_[s].w, scales_[s].h, scales_[s].w,
                     scales_[s].src_rgb);
    }

    std::vector<float> h_buf;
    std::vector<float> src_sq;
    for (size_t s = 0; s < scales_.size(); ++s) {
      ScaleState& sc = scales_[s];
      ConvertRgbToPosXybRect(sc.src_rgb, sc.w, 0, 0, sc.w, sc.h, sc.src_xyb);
      src_sq.resize(sc.w * sc.h);
      for (size_t c = 0; c < 3; ++c) {
        BlurFullPlane(sc.src_xyb[c].data(), sc.w, sc.h, &h_buf,
                      sc.blur_src[c].data());
        for (size_t i = 0; i < sc.w * sc.h; ++i) {
          const float v = sc.src_xyb[c][i];
          src_sq[i] = v * v;
        }
        BlurFullPlane(src_sq.data(), sc.w, sc.h, &h_buf,
                      sc.blur_src_sq[c].data());
      }
    }
  }

  double ScoreFull(const Image3F& rec_lin) {
    for (size_t c = 0; c < 3; ++c) {
      for (size_t y = 0; y < h0_; ++y) {
        memcpy(scales_[0].rec_rgb[c].data() + y * w0_,
               rec_lin.ConstPlaneRow(c, y), w0_ * sizeof(float));
      }
    }
    for (size_t s = 0; s < scales_.size(); ++s) {
      ScaleState& sc = scales_[s];
      if (s > 0) {
        DownsampleRect(scales_[s - 1].rec_rgb, scales_[s - 1].w,
                       scales_[s - 1].h, 0, 0, sc.w, sc.h, sc.w, sc.rec_rgb);
      }
      ConvertRgbToPosXybRect(sc.rec_rgb, sc.w, 0, 0, sc.w, sc.h, sc.rec_xyb);
      PoolScaleTiles(s, 0, 0, sc.tw, sc.th, nullptr);
    }
    return Deficit();
  }

  double UpdateCheckpointed(const Image3F& rec_lin, const Rect& dirty,
                            Checkpoint* cp) {
    cp->rgb_patches.clear();
    cp->rgb_xyb_data.clear();
    cp->tile_patches.clear();
    for (size_t s = 0; s < scales_.size(); ++s) {
      for (size_t m = 0; m < 9; ++m) {
        cp->prev_total_s1[s][m] = scales_[s].total_s1[m];
        cp->prev_total_s4[s][m] = scales_[s].total_s4[m];
      }
    }
    if (dirty.xsize() == 0 || dirty.ysize() == 0) return Deficit();
    size_t rx0 = std::min(dirty.x0(), w0_);
    size_t ry0 = std::min(dirty.y0(), h0_);
    size_t rx1 = std::min(dirty.x0() + dirty.xsize(), w0_);
    size_t ry1 = std::min(dirty.y0() + dirty.ysize(), h0_);
    if (rx0 >= rx1 || ry0 >= ry1) return Deficit();

    for (size_t s = 0; s < scales_.size(); ++s) {
      ScaleState& sc = scales_[s];
      SaveRgbXybPatch(s, rx0, ry0, rx1, ry1, cp);
      if (s == 0) {
        for (size_t c = 0; c < 3; ++c) {
          for (size_t y = ry0; y < ry1; ++y) {
            memcpy(sc.rec_rgb[c].data() + y * w0_ + rx0,
                   rec_lin.ConstPlaneRow(c, y) + rx0,
                   (rx1 - rx0) * sizeof(float));
          }
        }
      } else {
        DownsampleRect(scales_[s - 1].rec_rgb, scales_[s - 1].w,
                       scales_[s - 1].h, rx0, ry0, rx1, ry1, sc.w, sc.rec_rgb);
      }
      ConvertRgbToPosXybRect(sc.rec_rgb, sc.w, rx0, ry0, rx1, ry1, sc.rec_xyb);

      const size_t bx0 = (rx0 > kRadius) ? rx0 - kRadius : 0;
      const size_t by0 = (ry0 > kRadius) ? ry0 - kRadius : 0;
      const size_t bx1 = std::min(sc.w, rx1 + kRadius);
      const size_t by1 = std::min(sc.h, ry1 + kRadius);
      const size_t tx0 = bx0 / kTile;
      const size_t ty0 = by0 / kTile;
      const size_t tx1 = std::min(sc.tw, DivCeil(bx1, kTile));
      const size_t ty1 = std::min(sc.th, DivCeil(by1, kTile));
      PoolScaleTiles(s, tx0, ty0, tx1, ty1, cp);

      if (s + 1 < scales_.size()) {
        rx0 = rx0 / 2;
        ry0 = ry0 / 2;
        rx1 = std::min(scales_[s + 1].w, (rx1 + 1) / 2);
        ry1 = std::min(scales_[s + 1].h, (ry1 + 1) / 2);
      }
    }
    return Deficit();
  }

  void Rollback(const Checkpoint& cp) {
    for (const auto& p : cp.rgb_patches) {
      ScaleState& sc = scales_[p.scale];
      const size_t len = p.x1 - p.x0;
      size_t at = p.offset;
      for (size_t c = 0; c < 3; ++c) {
        for (size_t y = p.y0; y < p.y1; ++y) {
          memcpy(sc.rec_rgb[c].data() + y * sc.w + p.x0,
                 cp.rgb_xyb_data.data() + at, len * sizeof(float));
          at += len;
        }
      }
      for (size_t c = 0; c < 3; ++c) {
        for (size_t y = p.y0; y < p.y1; ++y) {
          memcpy(sc.rec_xyb[c].data() + y * sc.w + p.x0,
                 cp.rgb_xyb_data.data() + at, len * sizeof(float));
          at += len;
        }
      }
    }
    for (size_t i = cp.tile_patches.size(); i > 0; --i) {
      const auto& tp = cp.tile_patches[i - 1];
      ScaleState& sc = scales_[tp.scale];
      for (size_t m = 0; m < 9; ++m) {
        sc.tile_s1[m][tp.tile_idx] = tp.s1[m];
        sc.tile_s4[m][tp.tile_idx] = tp.s4[m];
      }
    }
    for (size_t s = 0; s < scales_.size(); ++s) {
      for (size_t m = 0; m < 9; ++m) {
        scales_[s].total_s1[m] = cp.prev_total_s1[s][m];
        scales_[s].total_s4[m] = cp.prev_total_s4[s][m];
      }
    }
  }

  double Deficit() const {
    const double sum = WeightedSum();
    const double u = sum * 0.9562382616834844;
    const double g = 2.326765642916932 * u - 0.020884521182843837 * u * u +
                     6.248496625763138e-05 * u * u * u;
    if (g <= 0.0) return 0.0;
    return 10.0 * std::pow(g, 0.6276336467831387);
  }

  std::vector<float> ErrorMap() {
    std::vector<float> full_map(w0_ * h0_, 0.0f);
    const size_t num_scales = scales_.size();
    const double sum = WeightedSum();
    const double u = sum * 0.9562382616834844;
    const double g = 2.326765642916932 * u - 0.020884521182843837 * u * u +
                     6.248496625763138e-05 * u * u * u;
    if (g <= 0.0) return full_map;
    const double dg = (2.326765642916932 - 2.0 * 0.020884521182843837 * u +
                       3.0 * 6.248496625763138e-05 * u * u) *
                      0.9562382616834844;
    const double slope =
        10.0 * 0.6276336467831387 * std::pow(g, 0.6276336467831387 - 1.0) * dg;
    if (slope <= 0.0) return full_map;

    std::vector<float> scale_acc;
    std::vector<float> h_rec;
    std::vector<float> h_rec_sq;
    std::vector<float> h_src_rec;
    std::vector<float> v_rec;
    std::vector<float> v_rec_sq;
    std::vector<float> v_src_rec;
    std::vector<float> tmp_sq;
    std::vector<float> tmp_sr;

    for (size_t s = 0; s < num_scales; ++s) {
      const ScaleState& sc = scales_[s];
      const size_t count = sc.w * sc.h;
      const double n = static_cast<double>(count);
      scale_acc.assign(count, 0.0f);
      tmp_sq.resize(count);
      tmp_sr.resize(count);

      for (size_t c = 0; c < 3; ++c) {
        const size_t wbase = (c * num_scales + s) * 6;
        double lin_w[3] = {};
        double fourth_w[3] = {};
        bool any_active = false;
        for (size_t m = 0; m < 3; ++m) {
          const double w1 = kSsim2Weights[wbase + m];
          const double w4 = kSsim2Weights[wbase + 3 + m];
          if (w1 == 0.0 && w4 == 0.0) continue;
          any_active = true;
          lin_w[m] = w1 / n;
          const double s4 = sc.total_s4[3 * c + m];
          if (w4 > 0.0 && s4 > 0.0) {
            const double n4 = std::sqrt(std::sqrt(s4 / n));
            if (n4 > 0.0) {
              fourth_w[m] = w4 / (4.0 * n * n4 * n4 * n4);
            }
          }
        }
        if (!any_active) continue;

        for (size_t i = 0; i < count; ++i) {
          const float r = sc.rec_xyb[c][i];
          tmp_sq[i] = r * r;
          tmp_sr[i] = sc.src_xyb[c][i] * r;
        }
        BlurFullPlane(sc.rec_xyb[c].data(), sc.w, sc.h, &h_rec, &v_rec);
        BlurFullPlane(tmp_sq.data(), sc.w, sc.h, &h_rec_sq, &v_rec_sq);
        BlurFullPlane(tmp_sr.data(), sc.w, sc.h, &h_src_rec, &v_src_rec);

        for (size_t i = 0; i < count; ++i) {
          const float m1 = sc.blur_src[c][i];
          const float m2 = v_rec[i];
          const float s11 = sc.blur_src_sq[c][i];
          const float s22 = v_rec_sq[i];
          const float s12 = v_src_rec[i];
          const float dm = m1 - m2;
          constexpr float kC2 = 0.0009f;
          const float d_ssim =
              std::max(0.0f, 1.0f - (1.0f - dm * dm) *
                                        (2.0f * (s12 - m1 * m2) + kC2) /
                                        ((s11 - m1 * m1) + (s22 - m2 * m2) + kC2));
          const float er =
              (1.0f + std::abs(sc.rec_xyb[c][i] - m2)) /
                  (1.0f + std::abs(sc.src_xyb[c][i] - m1)) -
              1.0f;
          const float d_art = std::max(0.0f, er);
          const float d_det = std::max(0.0f, -er);
          const double vals[3] = {static_cast<double>(d_ssim),
                                  static_cast<double>(d_art),
                                  static_cast<double>(d_det)};
          double contrib = 0.0;
          for (size_t m = 0; m < 3; ++m) {
            const double v = vals[m];
            const double v2 = v * v;
            contrib += lin_w[m] * v + fourth_w[m] * v2 * v2;
          }
          scale_acc[i] += static_cast<float>(contrib);
        }
      }

      const size_t step = size_t{1} << s;
      const float share = 1.0f / static_cast<float>(step * step);
      for (size_t sy = 0; sy < sc.h; ++sy) {
        const size_t fy0 = sy * step;
        const size_t fy1 = std::min(h0_, fy0 + step);
        for (size_t sx = 0; sx < sc.w; ++sx) {
          const float v = scale_acc[sy * sc.w + sx] * share;
          if (v == 0.0f) continue;
          const size_t fx0 = sx * step;
          const size_t fx1 = std::min(w0_, fx0 + step);
          for (size_t fy = fy0; fy < fy1; ++fy) {
            for (size_t fx = fx0; fx < fx1; ++fx) {
              full_map[fy * w0_ + fx] += v;
            }
          }
        }
      }
    }

    const float scale = static_cast<float>(slope * static_cast<double>(w0_ * h0_));
    for (float& v : full_map) {
      v *= scale;
    }
    return full_map;
  }

 private:
  struct ScaleState {
    size_t w = 0;
    size_t h = 0;
    size_t tw = 0;
    size_t th = 0;
    std::vector<float> src_rgb[3];
    std::vector<float> rec_rgb[3];
    std::vector<float> src_xyb[3];
    std::vector<float> rec_xyb[3];
    std::vector<float> blur_src[3];
    std::vector<float> blur_src_sq[3];
    std::vector<double> tile_s1[9];
    std::vector<double> tile_s4[9];
    double total_s1[9] = {};
    double total_s4[9] = {};
  };

  void InitKernel() {
    const double sigma = 1.5;
    const double radius = static_cast<double>(kRadius);
    const double pi_div_2r = 3.14159265358979323846 / (2.0 * radius);
    const double omega[3] = {pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r};
    const double p[3] = {
        1.0 / std::tan(0.5 * omega[0]),
        -1.0 / std::tan(0.5 * omega[1]),
        1.0 / std::tan(0.5 * omega[2]),
    };
    const double r[3] = {
        p[0] * p[0] / std::sin(omega[0]),
        -p[1] * p[1] / std::sin(omega[1]),
        p[2] * p[2] / std::sin(omega[2]),
    };
    double rho[3];
    for (size_t j = 0; j < 3; ++j) {
      rho[j] = std::exp(-0.5 * sigma * sigma * omega[j] * omega[j]) / radius;
    }
    const double d13 = p[0] * r[1] - r[0] * p[1];
    const double d35 = p[1] * r[2] - r[1] * p[2];
    const double d51 = p[2] * r[0] - r[2] * p[0];
    const double zeta15 = d35 / d13;
    const double zeta35 = d51 / d13;
    const double a[3][3] = {
        {p[0], p[1], p[2]},
        {r[0], r[1], r[2]},
        {zeta15, zeta35, 1.0},
    };
    const double g[3] = {
        1.0,
        radius * radius - sigma * sigma,
        zeta15 * rho[0] + zeta35 * rho[1] + rho[2],
    };
    const double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                       a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                       a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    double beta[3];
    for (size_t col = 0; col < 3; ++col) {
      double m[3][3];
      for (size_t row = 0; row < 3; ++row) {
        for (size_t c = 0; c < 3; ++c) {
          m[row][c] = (c == col) ? g[row] : a[row][c];
        }
      }
      beta[col] = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                   m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                   m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) /
                  det;
    }
    for (size_t i = 0; i < kTaps; ++i) {
      const double x = static_cast<double>(i) - radius;
      double sum = 0.0;
      for (size_t j = 0; j < 3; ++j) {
        sum += beta[j] * std::cos(omega[j] * x);
      }
      kernel_[i] = static_cast<float>(sum);
    }
  }

  static void DownsampleRect(const std::vector<float> prev[3], size_t pw,
                             size_t ph, size_t rx0, size_t ry0, size_t rx1,
                             size_t ry1, size_t nw,
                             std::vector<float> next[3]) {
    for (size_t c = 0; c < 3; ++c) {
      const float* JXL_RESTRICT src = prev[c].data();
      float* JXL_RESTRICT dst = next[c].data();
      for (size_t y = ry0; y < ry1; ++y) {
        const size_t sy0 = 2 * y;
        const size_t sy1 = std::min(sy0 + 1, ph - 1);
        const float* JXL_RESTRICT r0 = src + sy0 * pw;
        const float* JXL_RESTRICT r1 = src + sy1 * pw;
        float* JXL_RESTRICT out = dst + y * nw;
        for (size_t x = rx0; x < rx1; ++x) {
          const size_t sx0 = 2 * x;
          const size_t sx1 = std::min(sx0 + 1, pw - 1);
          out[x] = 0.25f * (r0[sx0] + r0[sx1] + r1[sx0] + r1[sx1]);
        }
      }
    }
  }

  static void ConvertRgbToPosXybRect(const std::vector<float> rgb[3], size_t w,
                                     size_t rx0, size_t ry0, size_t rx1,
                                     size_t ry1,
                                     std::vector<float> pos_xyb[3]) {
    constexpr float kBias = 0.0037930734f;
    constexpr float kNegBiasCbrt = -0.1559542f;
    for (size_t y = ry0; y < ry1; ++y) {
      const float* JXL_RESTRICT rr = rgb[0].data() + y * w;
      const float* JXL_RESTRICT rg = rgb[1].data() + y * w;
      const float* JXL_RESTRICT rb = rgb[2].data() + y * w;
      float* JXL_RESTRICT ox = pos_xyb[0].data() + y * w;
      float* JXL_RESTRICT oy = pos_xyb[1].data() + y * w;
      float* JXL_RESTRICT ob = pos_xyb[2].data() + y * w;
      for (size_t x = rx0; x < rx1; ++x) {
        const float r = rr[x];
        const float g = rg[x];
        const float b = rb[x];
        const float ml =
            std::max(0.0f, 0.30f * r + 0.622f * g + 0.078f * b + kBias);
        const float mm =
            std::max(0.0f, 0.23f * r + 0.692f * g + 0.078f * b + kBias);
        const float ms = std::max(
            0.0f, 0.24342269f * r + 0.20476745f * g + 0.55180986f * b + kBias);
        const float l = FastCbrtPositive(ml) + kNegBiasCbrt;
        const float m = FastCbrtPositive(mm) + kNegBiasCbrt;
        const float s = FastCbrtPositive(ms) + kNegBiasCbrt;
        const float px = 0.5f * (l - m);
        const float py = 0.5f * (l + m);
        ox[x] = px * 14.0f + 0.42f;
        oy[x] = py + 0.01f;
        ob[x] = (s - py) + 0.55f;
      }
    }
  }

  void BlurFullPlane(const float* JXL_RESTRICT src, size_t w, size_t h,
                     std::vector<float>* h_buf, float* JXL_RESTRICT dst) const {
    h_buf->resize(w * h);
    float* JXL_RESTRICT hb = h_buf->data();
    for (size_t y = 0; y < h; ++y) {
      const float* JXL_RESTRICT sr = src + y * w;
      float* JXL_RESTRICT hr = hb + y * w;
      for (size_t x = 0; x < w; ++x) {
        const size_t d0 = (x < kRadius) ? kRadius - x : 0;
        const size_t d1 =
            (x + kRadius < w) ? kTaps : kRadius + (w - x);
        float sum = 0.0f;
        for (size_t d = d0; d < d1; ++d) {
          sum += kernel_[d] * sr[x + d - kRadius];
        }
        hr[x] = sum;
      }
    }
    for (size_t y = 0; y < h; ++y) {
      const size_t d0 = (y < kRadius) ? kRadius - y : 0;
      const size_t d1 = (y + kRadius < h) ? kTaps : kRadius + (h - y);
      float* JXL_RESTRICT dr = dst + y * w;
      for (size_t x = 0; x < w; ++x) {
        float sum = 0.0f;
        for (size_t d = d0; d < d1; ++d) {
          sum += kernel_[d] * hb[(y + d - kRadius) * w + x];
        }
        dr[x] = sum;
      }
    }
  }

  void BlurFullPlane(const float* JXL_RESTRICT src, size_t w, size_t h,
                     std::vector<float>* h_buf,
                     std::vector<float>* v_buf) const {
    v_buf->resize(w * h);
    BlurFullPlane(src, w, h, h_buf, v_buf->data());
  }

  void SaveRgbXybPatch(size_t s, size_t rx0, size_t ry0, size_t rx1, size_t ry1,
                       Checkpoint* cp) const {
    if (rx0 >= rx1 || ry0 >= ry1) return;
    const ScaleState& sc = scales_[s];
    const size_t len = rx1 - rx0;
    const size_t offset = cp->rgb_xyb_data.size();
    cp->rgb_patches.push_back({s, rx0, ry0, rx1, ry1, offset});
    cp->rgb_xyb_data.resize(offset + 6 * len * (ry1 - ry0));
    size_t at = offset;
    for (size_t c = 0; c < 3; ++c) {
      for (size_t y = ry0; y < ry1; ++y) {
        memcpy(cp->rgb_xyb_data.data() + at,
               sc.rec_rgb[c].data() + y * sc.w + rx0, len * sizeof(float));
        at += len;
      }
    }
    for (size_t c = 0; c < 3; ++c) {
      for (size_t y = ry0; y < ry1; ++y) {
        memcpy(cp->rgb_xyb_data.data() + at,
               sc.rec_xyb[c].data() + y * sc.w + rx0, len * sizeof(float));
        at += len;
      }
    }
  }

  void PoolScaleTiles(size_t s, size_t tx0, size_t ty0, size_t tx1, size_t ty1,
                      Checkpoint* cp) {
    if (tx0 >= tx1 || ty0 >= ty1) return;
    ScaleState& sc = scales_[s];
    const size_t w = sc.w;
    const size_t h = sc.h;
    const size_t px0 = tx0 * kTile;
    const size_t py0 = ty0 * kTile;
    const size_t px1 = std::min(w, tx1 * kTile);
    const size_t py1 = std::min(h, ty1 * kTile);
    const size_t pw = px1 - px0;
    const size_t hy0 = (py0 > kRadius) ? py0 - kRadius : 0;
    const size_t hy1 = std::min(h, py1 + kRadius);
    const size_t hh = hy1 - hy0;

    if (cp != nullptr) {
      for (size_t ty = ty0; ty < ty1; ++ty) {
        for (size_t tx = tx0; tx < tx1; ++tx) {
          const size_t tidx = ty * sc.tw + tx;
          Checkpoint::TilePatch tp;
          tp.scale = s;
          tp.tile_idx = tidx;
          for (size_t m = 0; m < 9; ++m) {
            tp.s1[m] = sc.tile_s1[m][tidx];
            tp.s4[m] = sc.tile_s4[m][tidx];
          }
          cp->tile_patches.push_back(tp);
        }
      }
    }

    h_rec_.resize(pw * hh);
    h_rec_sq_.resize(pw * hh);
    h_src_rec_.resize(pw * hh);
    v_rec_.resize(pw);
    v_rec_sq_.resize(pw);
    v_src_rec_.resize(pw);

    const size_t n_tiles_sub = (tx1 - tx0) * (ty1 - ty0);
    sub_s1_.assign(9 * n_tiles_sub, 0.0);
    sub_s4_.assign(9 * n_tiles_sub, 0.0);

    for (size_t c = 0; c < 3; ++c) {
      const float* JXL_RESTRICT src_p = sc.src_xyb[c].data();
      const float* JXL_RESTRICT rec_p = sc.rec_xyb[c].data();
      for (size_t hy = hy0; hy < hy1; ++hy) {
        const float* JXL_RESTRICT sr = src_p + hy * w;
        const float* JXL_RESTRICT rr = rec_p + hy * w;
        float* JXL_RESTRICT hr = h_rec_.data() + (hy - hy0) * pw;
        float* JXL_RESTRICT hrsq = h_rec_sq_.data() + (hy - hy0) * pw;
        float* JXL_RESTRICT hrsr = h_src_rec_.data() + (hy - hy0) * pw;
        for (size_t i = 0; i < pw; ++i) {
          const size_t x = px0 + i;
          const size_t d0 = (x < kRadius) ? kRadius - x : 0;
          const size_t d1 =
              (x + kRadius < w) ? kTaps : kRadius + (w - x);
          float sum_r = 0.0f;
          float sum_sq = 0.0f;
          float sum_sr = 0.0f;
          for (size_t d = d0; d < d1; ++d) {
            const float kw = kernel_[d];
            const size_t sx = x + d - kRadius;
            const float rv = rr[sx];
            sum_r += kw * rv;
            sum_sq += kw * rv * rv;
            sum_sr += kw * sr[sx] * rv;
          }
          hr[i] = sum_r;
          hrsq[i] = sum_sq;
          hrsr[i] = sum_sr;
        }
      }

      for (size_t y = py0; y < py1; ++y) {
        const size_t d0 = (y < kRadius) ? kRadius - y : 0;
        const size_t d1 = (y + kRadius < h) ? kTaps : kRadius + (h - y);
        for (size_t i = 0; i < pw; ++i) {
          float sum_r = 0.0f;
          float sum_sq = 0.0f;
          float sum_sr = 0.0f;
          for (size_t d = d0; d < d1; ++d) {
            const float kw = kernel_[d];
            const size_t hrow = (y + d - kRadius - hy0) * pw + i;
            sum_r += kw * h_rec_[hrow];
            sum_sq += kw * h_rec_sq_[hrow];
            sum_sr += kw * h_src_rec_[hrow];
          }
          v_rec_[i] = sum_r;
          v_rec_sq_[i] = sum_sq;
          v_src_rec_[i] = sum_sr;
        }

        const size_t lty = (y / kTile) - ty0;
        const float* JXL_RESTRICT b_src = sc.blur_src[c].data() + y * w + px0;
        const float* JXL_RESTRICT b_src_sq =
            sc.blur_src_sq[c].data() + y * w + px0;
        const float* JXL_RESTRICT s_row = src_p + y * w + px0;
        const float* JXL_RESTRICT r_row = rec_p + y * w + px0;

        for (size_t i = 0; i < pw; ++i) {
          const size_t ltx = ((px0 + i) / kTile) - tx0;
          const size_t sub_idx = (lty * (tx1 - tx0) + ltx) * 9 + 3 * c;
          const float m1 = b_src[i];
          const float m2 = v_rec_[i];
          const float s11 = b_src_sq[i];
          const float s22 = v_rec_sq_[i];
          const float s12 = v_src_rec_[i];
          const float dm = m1 - m2;
          constexpr float kC2 = 0.0009f;
          const float d_ssim =
              std::max(0.0f, 1.0f - (1.0f - dm * dm) *
                                        (2.0f * (s12 - m1 * m2) + kC2) /
                                        ((s11 - m1 * m1) + (s22 - m2 * m2) + kC2));
          const float er =
              (1.0f + std::abs(r_row[i] - m2)) /
                  (1.0f + std::abs(s_row[i] - m1)) -
              1.0f;
          const float d_art = std::max(0.0f, er);
          const float d_det = std::max(0.0f, -er);
          const double vals[3] = {static_cast<double>(d_ssim),
                                  static_cast<double>(d_art),
                                  static_cast<double>(d_det)};
          for (size_t m = 0; m < 3; ++m) {
            const double v = vals[m];
            const double v2 = v * v;
            sub_s1_[sub_idx + m] += v;
            sub_s4_[sub_idx + m] += v2 * v2;
          }
        }
      }
    }

    for (size_t lty = 0; lty < ty1 - ty0; ++lty) {
      for (size_t ltx = 0; ltx < tx1 - tx0; ++ltx) {
        const size_t tidx = (ty0 + lty) * sc.tw + (tx0 + ltx);
        const size_t sub_base = (lty * (tx1 - tx0) + ltx) * 9;
        for (size_t m = 0; m < 9; ++m) {
          sc.tile_s1[m][tidx] = sub_s1_[sub_base + m];
          sc.tile_s4[m][tidx] = sub_s4_[sub_base + m];
        }
      }
    }

    for (size_t m = 0; m < 9; ++m) {
      double tot1 = 0.0;
      double tot4 = 0.0;
      for (size_t t = 0; t < sc.tw * sc.th; ++t) {
        tot1 += sc.tile_s1[m][t];
        tot4 += sc.tile_s4[m][t];
      }
      sc.total_s1[m] = tot1;
      sc.total_s4[m] = tot4;
    }
  }

  double WeightedSum() const {
    const size_t num_scales = scales_.size();
    double sum = 0.0;
    for (size_t c = 0; c < 3; ++c) {
      for (size_t s = 0; s < num_scales; ++s) {
        const ScaleState& sc = scales_[s];
        const double inv_n = 1.0 / static_cast<double>(sc.w * sc.h);
        const size_t wbase = (c * num_scales + s) * 6;
        for (size_t m = 0; m < 3; ++m) {
          const double w1 = kSsim2Weights[wbase + m];
          const double w4 = kSsim2Weights[wbase + 3 + m];
          if (w1 == 0.0 && w4 == 0.0) continue;
          const double m1 = sc.total_s1[3 * c + m] * inv_n;
          const double m4 = sc.total_s4[3 * c + m] * inv_n;
          const double n4 = std::sqrt(std::sqrt(m4));
          sum += w1 * m1 + w4 * n4;
        }
      }
    }
    return sum;
  }

  size_t w0_ = 0;
  size_t h0_ = 0;
  float kernel_[kTaps] = {};
  std::vector<ScaleState> scales_;
  std::vector<float> h_rec_;
  std::vector<float> h_rec_sq_;
  std::vector<float> h_src_rec_;
  std::vector<float> v_rec_;
  std::vector<float> v_rec_sq_;
  std::vector<float> v_src_rec_;
  std::vector<double> sub_s1_;
  std::vector<double> sub_s4_;
};

class CvvdpIncrementalArbiter {
 public:
  static constexpr size_t kRadius = 6;
  static constexpr size_t kTaps = 13;
  static constexpr size_t kTile = 8;
  static constexpr size_t kAlign = 8;
  static constexpr float kEpsilon = 1e-5f;
  static constexpr float kMaskP = 2.2643552f;

  struct Checkpoint {
    struct GaussPatch {
      size_t level;
      size_t x0, y0, x1, y1;
      size_t offset;
    };
    struct TilePatch {
      size_t level;
      size_t tile_idx;
      double sums[3];
    };
    std::vector<GaussPatch> gauss_patches;
    std::vector<float> gauss_data;
    std::vector<TilePatch> tile_patches;
    double base[3] = {};
  };

  void Init(const Image3F& src_lin) {
    w0_ = src_lin.xsize();
    h0_ = src_lin.ysize();
    InitParamsAndTables();
    const size_t height = PyramidHeight(w0_, h0_);
    levels_.clear();
    levels_.resize(height + 1);

    size_t w = w0_;
    size_t h = h0_;
    for (size_t k = 0; k <= height; ++k) {
      LevelState& lv = levels_[k];
      lv.w = w;
      lv.h = h;
      lv.is_base = (k == height);
      lv.tiles_x = DivCeil(w, kTile);
      lv.tiles_y = DivCeil(h, kTile);
      for (size_t c = 0; c < 3; ++c) {
        lv.ref_contrast[c].resize(w * h, 0.0f);
        lv.ref_sens[c].resize(lv.is_base ? 1 : (w * h), 0.0f);
        lv.test_gauss[c].resize(w * h, 0.0f);
      }
      if (!lv.is_base) {
        lv.tiles.assign(lv.tiles_x * lv.tiles_y, {{0.0, 0.0, 0.0}});
        lv.row_sums.assign(lv.tiles_y, {{0.0, 0.0, 0.0}});
      }
      w = DivCeil(w, 2);
      h = DivCeil(h, 2);
    }

    std::vector<float> cur_dkl[3];
    std::vector<float> next_dkl[3];
    std::vector<float> exp_dkl[3];
    std::vector<float> tmp_buf;
    for (size_t c = 0; c < 3; ++c) {
      cur_dkl[c].resize(w0_ * h0_);
    }
    for (size_t y = 0; y < h0_; ++y) {
      RgbRowToDkl(src_lin.ConstPlaneRow(0, y), src_lin.ConstPlaneRow(1, y),
                  src_lin.ConstPlaneRow(2, y), w0_,
                  cur_dkl[0].data() + y * w0_, cur_dkl[1].data() + y * w0_,
                  cur_dkl[2].data() + y * w0_);
    }

    for (size_t k = 0; k <= height; ++k) {
      LevelState& lv = levels_[k];
      const size_t count = lv.w * lv.h;
      const size_t csf_idx = lv.is_base ? 9 : std::min<size_t>(k, 8);
      if (lv.is_base) {
        const float bg = MeanLuminance(cur_dkl[0].data(), count);
        float s[3];
        CsfAtLuminance(kCvvdpLogCsf[csf_idx], bg, s);
        for (size_t c = 0; c < 3; ++c) {
          lv.ref_sens[c][0] = s[c];
          for (size_t i = 0; i < count; ++i) {
            lv.ref_contrast[c][i] =
                std::min(cur_dkl[c][i] / bg, 1000.0f) * s[c];
          }
        }
      } else {
        const size_t nw = levels_[k + 1].w;
        const size_t nh = levels_[k + 1].h;
        for (size_t c = 0; c < 3; ++c) {
          next_dkl[c].resize(nw * nh);
          exp_dkl[c].resize(count);
          ReduceFull(cur_dkl[c].data(), lv.w, lv.h, nw, nh, &tmp_buf,
                     next_dkl[c].data());
          ExpandFull(next_dkl[c].data(), nw, nh, lv.w, lv.h, &tmp_buf,
                     exp_dkl[c].data());
        }
        const float band_scale = (k == 0) ? 1.0f : 2.0f;
        const float ch_gain[3] = {1.0f, 1.45f, 1.0f};
        for (size_t i = 0; i < count; ++i) {
          const float bg = std::max(exp_dkl[0][i], 0.01f);
          float s[3];
          CsfAtLuminance(kCvvdpLogCsf[csf_idx], bg, s);
          for (size_t c = 0; c < 3; ++c) {
            const float sens = s[c] * ch_gain[c];
            lv.ref_sens[c][i] = sens;
            lv.ref_contrast[c][i] =
                std::min((cur_dkl[c][i] - exp_dkl[c][i]) / bg, 1000.0f) *
                band_scale * sens;
          }
        }
        for (size_t c = 0; c < 3; ++c) {
          cur_dkl[c].swap(next_dkl[c]);
        }
      }
    }
  }

  double ScoreFull(const Image3F& test_lin) {
    ApplyRect(test_lin, Rect(0, 0, w0_, h0_), nullptr);
    return Distortion();
  }

  double UpdateCheckpointed(const Image3F& test_lin, const Rect& dirty,
                            Checkpoint* cp) {
    cp->gauss_patches.clear();
    cp->gauss_data.clear();
    cp->tile_patches.clear();
    for (size_t c = 0; c < 3; ++c) {
      cp->base[c] = base_sums_[c];
    }
    ApplyRect(test_lin, dirty, cp);
    return Distortion();
  }

  void Rollback(const Checkpoint& cp) {
    for (const auto& gp : cp.gauss_patches) {
      LevelState& lv = levels_[gp.level];
      const size_t len = gp.x1 - gp.x0;
      size_t at = gp.offset;
      for (size_t c = 0; c < 3; ++c) {
        for (size_t y = gp.y0; y < gp.y1; ++y) {
          memcpy(lv.test_gauss[c].data() + y * lv.w + gp.x0,
                 cp.gauss_data.data() + at, len * sizeof(float));
          at += len;
        }
      }
    }
    for (size_t i = cp.tile_patches.size(); i > 0; --i) {
      const auto& tp = cp.tile_patches[i - 1];
      LevelState& lv = levels_[tp.level];
      lv.tiles[tp.tile_idx] = {{tp.sums[0], tp.sums[1], tp.sums[2]}};
    }
    for (const auto& tp : cp.tile_patches) {
      LevelState& lv = levels_[tp.level];
      SumTileRow(&lv, tp.tile_idx / lv.tiles_x);
    }
    for (size_t c = 0; c < 3; ++c) {
      base_sums_[c] = cp.base[c];
    }
  }

  double Distortion() const {
    constexpr double kBaseWeight[3] = {0.0036334486212581396,
                                       1.6627724170684814, 4.11874532699585};
    constexpr double kImageIntegration = 0.577918291091919;
    double per_channel[3] = {0.0, 0.0, 0.0};
    for (size_t k = 0; k < levels_.size(); ++k) {
      const LevelState& lv = levels_[k];
      double sq[3] = {0.0, 0.0, 0.0};
      if (lv.is_base) {
        for (size_t c = 0; c < 3; ++c) sq[c] = base_sums_[c];
      } else {
        for (const auto& row : lv.row_sums) {
          for (size_t c = 0; c < 3; ++c) sq[c] += row[c];
        }
      }
      const double inv_n = 1.0 / static_cast<double>(lv.w * lv.h);
      for (size_t c = 0; c < 3; ++c) {
        const double rms = SafePow(sq[c] * inv_n, 0.5);
        const double weighted = lv.is_base ? (rms * kBaseWeight[c]) : rms;
        per_channel[c] += SafePow(weighted, 4.0);
      }
    }
    double total = 0.0;
    for (size_t c = 0; c < 3; ++c) {
      total += SafePow(SafePow(per_channel[c], 0.25), 4.0);
    }
    return SafePow(total, 0.25) * kImageIntegration;
  }

 private:
  struct PowerTable {
    float scale[256] = {};
    float poly[11] = {};
  };

  struct LevelState {
    size_t w = 0;
    size_t h = 0;
    size_t tiles_x = 0;
    size_t tiles_y = 0;
    bool is_base = false;
    std::vector<float> ref_contrast[3];
    std::vector<float> ref_sens[3];
    std::vector<float> test_gauss[3];
    std::vector<std::array<double, 3>> tiles;
    std::vector<std::array<double, 3>> row_sums;
  };

  void InitParamsAndTables() {
    const float peak = 500.0f;
    const float black_lum = peak / 1351.3513f;
    const float refl = 250.0f * 0.005f / 3.14159265358979323846f;
    disp_scale_ = peak - black_lum;
    disp_black_ = black_lum + refl;

    float ksum = 0.0f;
    for (size_t i = 0; i < kTaps; ++i) {
      const float d = static_cast<float>(i) - static_cast<float>(kRadius);
      blur_kernel_[i] = std::exp(-(d * d) / 18.0f);
      ksum += blur_kernel_[i];
    }
    for (size_t i = 0; i < kTaps; ++i) {
      blur_kernel_[i] /= ksum;
    }

    constexpr float kMaskQ[3] = {1.3026227f, 2.8885908f, 3.6807714f};
    mask_c_ = std::pow(10.0f, -0.7954971f);
    dmax_ = std::pow(10.0f, 2.5642455f);
    for (size_t c = 0; c < 3; ++c) {
      mask_zero_[c] = std::pow(kEpsilon, kMaskQ[c]);
      InitPowerTable(kMaskQ[c], &pow_q_[c]);
    }
    diff_zero_ = std::pow(kEpsilon, kMaskP);
    InitPowerTable(kMaskP, &pow_p_);

    constexpr float kXcmLog[3][3] = {
        {-0.18950105f, -5.962151f, -4.318346f},
        {2.5655594f, 0.34406713f, -2.7196465f},
        {3.8118372f, -1.0051706f, -0.51933765f},
    };
    for (size_t r = 0; r < 3; ++r) {
      for (size_t c = 0; c < 3; ++c) {
        xcm_[r][c] = std::pow(2.0f, kXcmLog[r][c]);
      }
    }
  }

  static void InitPowerTable(float exponent, PowerTable* tbl) {
    const double p = static_cast<double>(exponent);
    for (int i = 0; i < 256; ++i) {
      tbl->scale[i] =
          static_cast<float>(std::pow(2.0, static_cast<double>(i - 127) * p));
    }
    tbl->poly[0] = 1.0f;
    double coeff = 1.0;
    for (int i = 1; i < 11; ++i) {
      coeff *= (p - static_cast<double>(i - 1)) / static_cast<double>(i);
      tbl->poly[i] = static_cast<float>(coeff);
    }
  }

  static inline float EvalPow(float x, const PowerTable& tbl) {
    uint32_t bits = 0;
    memcpy(&bits, &x, sizeof(bits));
    bits += (0x3f800000u - 0x3f3504f3u);
    const uint32_t exp_idx = (bits >> 23) & 255u;
    const uint32_t m_bits = (bits & 0x007fffffu) + 0x3f3504f3u;
    float mant = 1.0f;
    memcpy(&mant, &m_bits, sizeof(mant));
    const float t = mant - 1.0f;
    float poly = tbl.poly[10];
    for (int i = 9; i >= 0; --i) {
      poly = poly * t + tbl.poly[i];
    }
    return poly * tbl.scale[exp_idx];
  }

  static inline double SafePow(double v, double e) {
    constexpr double kEps = 1e-5;
    return std::max(0.0, std::pow(v + kEps, e) - std::pow(kEps, e));
  }

  static size_t PyramidHeight(size_t w, size_t h) {
    constexpr float kPpd = 98.79242f;
    size_t min_dim = std::min(w, h);
    size_t ilog2 = 0;
    while ((size_t{1} << (ilog2 + 1)) <= min_dim) ++ilog2;
    const size_t max_h = std::min<size_t>((ilog2 > 0) ? ilog2 - 1 : 0, 14);
    for (size_t i = 0; i <= max_h; ++i) {
      const float freq =
          (i == 0)
              ? (kPpd * 0.5f)
              : (kPpd * 0.5f * 0.3228f / static_cast<float>(size_t{1} << (i - 1)));
      if (freq <= 0.2f) {
        return std::min(max_h, i + 1);
      }
    }
    return max_h;
  }

  void RgbRowToDkl(const float* JXL_RESTRICT rr, const float* JXL_RESTRICT rg,
                   const float* JXL_RESTRICT rb, size_t len,
                   float* JXL_RESTRICT d0, float* JXL_RESTRICT d1,
                   float* JXL_RESTRICT d2) const {
    for (size_t i = 0; i < len; ++i) {
      const float r = Clamp1(rr[i], 0.0f, 1.0f) * disp_scale_ + disp_black_;
      const float g = Clamp1(rg[i], 0.0f, 1.0f) * disp_scale_ + disp_black_;
      const float b = Clamp1(rb[i], 0.0f, 1.0f) * disp_scale_ + disp_black_;
      d0[i] = 0.23320122f * r + 0.72883075f * g + 0.08899587f * b;
      d1[i] = 0.12762076f * r - 0.08706811f * g - 0.036777392f * b;
      d2[i] = -0.2148225f * r - 0.62625366f * g + 0.8514033f * b;
    }
  }

  static float MeanLuminance(const float* JXL_RESTRICT p, size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
      sum += static_cast<double>(std::max(p[i], 0.01f));
    }
    return static_cast<float>(sum / static_cast<double>(count));
  }

  static void CsfAtLuminance(const float csf[3][32], float lum, float out[3]) {
    const float l10 = std::log10(std::max(lum, 1e-6f));
    const float pos = Clamp1(
        (l10 - kCvvdpLogLum[0]) * 31.0f / (kCvvdpLogLum[31] - kCvvdpLogLum[0]),
        0.0f, 31.0f);
    const size_t idx = std::min<size_t>(static_cast<size_t>(pos), 30);
    const float t = pos - static_cast<float>(idx);
    const float corr = std::pow(10.0f, -0.27974233f / 20.0f);
    for (size_t c = 0; c < 3; ++c) {
      const float v = csf[c][idx] * (1.0f - t) + csf[c][idx + 1] * t;
      out[c] = std::pow(10.0f, v) * corr;
    }
  }

  static inline size_t ReflectInclusive(ptrdiff_t i, size_t n) {
    if (n <= 1) return 0;
    if (i < 0) return static_cast<size_t>(-i - 1);
    if (static_cast<size_t>(i) >= n) return 2 * n - static_cast<size_t>(i) - 1;
    return static_cast<size_t>(i);
  }

  static inline size_t ReflectExclusive(ptrdiff_t i, size_t n) {
    if (n <= 1) return 0;
    if (i < 0) return static_cast<size_t>(-i);
    if (static_cast<size_t>(i) >= n) return 2 * n - static_cast<size_t>(i) - 2;
    return static_cast<size_t>(i);
  }

  static void ReduceFull(const float* JXL_RESTRICT src, size_t w, size_t h,
                         size_t nw, size_t nh, std::vector<float>* tmp,
                         float* JXL_RESTRICT dst) {
    tmp->resize(w);
    float* JXL_RESTRICT t = tmp->data();
    for (size_t y = 0; y < nh; ++y) {
      const size_t r0 = ReflectInclusive(static_cast<ptrdiff_t>(2 * y) - 2, h);
      const size_t r1 = ReflectInclusive(static_cast<ptrdiff_t>(2 * y) - 1, h);
      const size_t r2 = ReflectInclusive(static_cast<ptrdiff_t>(2 * y), h);
      const size_t r3 = ReflectInclusive(static_cast<ptrdiff_t>(2 * y) + 1, h);
      const size_t r4 = ReflectInclusive(static_cast<ptrdiff_t>(2 * y) + 2, h);
      for (size_t x = 0; x < w; ++x) {
        t[x] = 0.05f * (src[r0 * w + x] + src[r4 * w + x]) +
               0.25f * (src[r1 * w + x] + src[r3 * w + x]) +
               0.40f * src[r2 * w + x];
      }
      for (size_t x = 0; x < nw; ++x) {
        const ptrdiff_t xx = static_cast<ptrdiff_t>(2 * x);
        const float v =
            0.05f * (t[ReflectInclusive(xx - 2, w)] +
                     t[ReflectInclusive(xx + 2, w)]) +
            0.25f * (t[ReflectInclusive(xx - 1, w)] +
                     t[ReflectInclusive(xx + 1, w)]) +
            0.40f * t[ReflectInclusive(xx, w)];
        dst[y * nw + x] = v;
      }
    }
  }

  static void ExpandFull(const float* JXL_RESTRICT src, size_t sw, size_t sh,
                         size_t w, size_t h, std::vector<float>* tmp,
                         float* JXL_RESTRICT dst) {
    tmp->resize(sw);
    float* JXL_RESTRICT t = tmp->data();
    for (size_t y = 0; y < h; ++y) {
      const size_t sy = y / 2;
      const size_t sb = (sy > 0) ? sy - 1 : 0;
      const size_t sa = std::min(sy + 1, sh - 1);
      for (size_t cx = 0; cx < sw; ++cx) {
        t[cx] = (y % 2 == 0)
                    ? (0.1f * (src[sb * sw + cx] + src[sa * sw + cx]) +
                       0.8f * src[sy * sw + cx])
                    : (0.5f * (src[sy * sw + cx] + src[sa * sw + cx]));
      }
      for (size_t x = 0; x < w; ++x) {
        const size_t sx = x / 2;
        const size_t sxb = (sx > 0) ? sx - 1 : 0;
        const size_t sxa = std::min(sx + 1, sw - 1);
        dst[y * w + x] =
            (x % 2 == 0) ? (0.1f * (t[sxb] + t[sxa]) + 0.8f * t[sx])
                         : (0.5f * (t[sx] + t[sxa]));
      }
    }
  }

  void SaveGaussPatch(size_t k, size_t x0, size_t y0, size_t x1, size_t y1,
                      Checkpoint* cp) const {
    if (cp == nullptr || x0 >= x1 || y0 >= y1) return;
    const LevelState& lv = levels_[k];
    const size_t len = x1 - x0;
    const size_t offset = cp->gauss_data.size();
    cp->gauss_patches.push_back({k, x0, y0, x1, y1, offset});
    cp->gauss_data.resize(offset + 3 * len * (y1 - y0));
    size_t at = offset;
    for (size_t c = 0; c < 3; ++c) {
      for (size_t y = y0; y < y1; ++y) {
        memcpy(cp->gauss_data.data() + at,
               lv.test_gauss[c].data() + y * lv.w + x0, len * sizeof(float));
        at += len;
      }
    }
  }

  static void SumTileRow(LevelState* lv, size_t ty) {
    double sums[3] = {0.0, 0.0, 0.0};
    for (size_t tx = 0; tx < lv->tiles_x; ++tx) {
      const auto& t = lv->tiles[ty * lv->tiles_x + tx];
      for (size_t c = 0; c < 3; ++c) sums[c] += t[c];
    }
    lv->row_sums[ty] = {{sums[0], sums[1], sums[2]}};
  }

  void ApplyRect(const Image3F& test_lin, const Rect& rect, Checkpoint* cp) {
    if (rect.xsize() == 0 || rect.ysize() == 0) return;
    size_t dx0 = std::min(rect.x0(), w0_);
    size_t dy0 = std::min(rect.y0(), h0_);
    size_t dx1 = std::min(rect.x0() + rect.xsize(), w0_);
    size_t dy1 = std::min(rect.y0() + rect.ysize(), h0_);
    if (dx0 >= dx1 || dy0 >= dy1) return;

    dx0 = (dx0 / kAlign) * kAlign;
    dx1 = std::min(w0_, DivCeil(dx1, kAlign) * kAlign);

    SaveGaussPatch(0, dx0, dy0, dx1, dy1, cp);
    {
      LevelState& lv0 = levels_[0];
      const size_t len = dx1 - dx0;
      for (size_t y = dy0; y < dy1; ++y) {
        RgbRowToDkl(test_lin.ConstPlaneRow(0, y) + dx0,
                    test_lin.ConstPlaneRow(1, y) + dx0,
                    test_lin.ConstPlaneRow(2, y) + dx0, len,
                    lv0.test_gauss[0].data() + y * lv0.w + dx0,
                    lv0.test_gauss[1].data() + y * lv0.w + dx0,
                    lv0.test_gauss[2].data() + y * lv0.w + dx0);
      }
    }

    const size_t base_idx = levels_.size() - 1;
    for (size_t k = 0; k < base_idx; ++k) {
      const size_t w = levels_[k].w;
      const size_t h = levels_[k].h;
      const size_t nw = levels_[k + 1].w;
      const size_t nh = levels_[k + 1].h;

      const size_t cx0 = (dx0 >= 2) ? (dx0 - 2) / 2 : 0;
      const size_t cy0 = (dy0 >= 2) ? (dy0 - 2) / 2 : 0;
      const size_t cx1 = std::min(nw, DivCeil(dx1, 2) + 1);
      const size_t cy1 = std::min(nh, DivCeil(dy1, 2) + 1);

      SaveGaussPatch(k + 1, cx0, cy0, cx1, cy1, cp);
      ReduceRegion(k, cx0, cy0, cx1, cy1);

      const size_t ex0 = (2 * cx0 >= 2) ? (2 * cx0 - 2) : 0;
      const size_t ey0 = (2 * cy0 >= 2) ? (2 * cy0 - 2) : 0;
      const size_t ex1 = std::min(w, 2 * cx1 + 2);
      const size_t ey1 = std::min(h, 2 * cy1 + 2);

      size_t chx0 = std::min(dx0, ex0);
      size_t chy0 = std::min(dy0, ey0);
      size_t chx1 = std::max(dx1, ex1);
      size_t chy1 = std::max(dy1, ey1);

      const bool blur = (w > kRadius && h > kRadius);
      if (blur) {
        chx0 = (chx0 > kRadius) ? chx0 - kRadius : 0;
        chy0 = (chy0 > kRadius) ? chy0 - kRadius : 0;
        chx1 = std::min(w, chx1 + kRadius);
        chy1 = std::min(h, chy1 + kRadius);
      }

      const size_t tx0 = chx0 / kTile;
      const size_t ty0 = chy0 / kTile;
      const size_t tx1 = std::min(levels_[k].tiles_x, DivCeil(chx1, kTile));
      const size_t ty1 = std::min(levels_[k].tiles_y, DivCeil(chy1, kTile));

      if (cp != nullptr) {
        const LevelState& lv = levels_[k];
        for (size_t ty = ty0; ty < ty1; ++ty) {
          for (size_t tx = tx0; tx < tx1; ++tx) {
            const size_t tidx = ty * lv.tiles_x + tx;
            cp->tile_patches.push_back(
                {k,
                 tidx,
                 {lv.tiles[tidx][0], lv.tiles[tidx][1], lv.tiles[tidx][2]}});
          }
        }
      }

      PoolRegion(k, tx0, ty0, tx1, ty1);
      dx0 = cx0;
      dy0 = cy0;
      dx1 = cx1;
      dy1 = cy1;
    }

    ScoreBase();
  }

  void ReduceRegion(size_t k, size_t cx0, size_t cy0, size_t cx1, size_t cy1) {
    if (cx0 >= cx1 || cy0 >= cy1) return;
    const LevelState& src = levels_[k];
    LevelState& dst = levels_[k + 1];
    const size_t w = src.w;
    const size_t h = src.h;
    const size_t lo = (2 * cx0 >= 2) ? (2 * cx0 - 2) : 0;
    const size_t hi = std::min(w, 2 * cx1 + 1);
    scratch_tmp_.resize(hi - lo);
    float* JXL_RESTRICT t = scratch_tmp_.data();

    for (size_t c = 0; c < 3; ++c) {
      const float* JXL_RESTRICT s = src.test_gauss[c].data();
      for (size_t y = cy0; y < cy1; ++y) {
        const size_t r0 = ReflectInclusive(static_cast<ptrdiff_t>(2 * y) - 2, h);
        const size_t r1 = ReflectInclusive(static_cast<ptrdiff_t>(2 * y) - 1, h);
        const size_t r2 = ReflectInclusive(static_cast<ptrdiff_t>(2 * y), h);
        const size_t r3 = ReflectInclusive(static_cast<ptrdiff_t>(2 * y) + 1, h);
        const size_t r4 = ReflectInclusive(static_cast<ptrdiff_t>(2 * y) + 2, h);
        for (size_t xx = lo; xx < hi; ++xx) {
          t[xx - lo] = 0.05f * (s[r0 * w + xx] + s[r4 * w + xx]) +
                       0.25f * (s[r1 * w + xx] + s[r3 * w + xx]) +
                       0.40f * s[r2 * w + xx];
        }
        float* JXL_RESTRICT out = dst.test_gauss[c].data() + y * dst.w;
        for (size_t x = cx0; x < cx1; ++x) {
          const ptrdiff_t xx = static_cast<ptrdiff_t>(2 * x);
          out[x] = 0.05f * (t[ReflectInclusive(xx - 2, w) - lo] +
                            t[ReflectInclusive(xx + 2, w) - lo]) +
                   0.25f * (t[ReflectInclusive(xx - 1, w) - lo] +
                            t[ReflectInclusive(xx + 1, w) - lo]) +
                   0.40f * t[ReflectInclusive(xx, w) - lo];
        }
      }
    }
  }

  void ExpandRegion(size_t k_coarse, size_t w, size_t h, size_t ax0, size_t ay0,
                    size_t ax1, size_t ay1, std::vector<float> out[3]) {
    const LevelState& coarse = levels_[k_coarse];
    const size_t sw = coarse.w;
    const size_t sh = coarse.h;
    const size_t cx0 = (ax0 / 2 > 0) ? (ax0 / 2 - 1) : 0;
    const size_t cx1 = std::min(sw, (ax1 - 1) / 2 + 2);
    const size_t aw = ax1 - ax0;
    const size_t ah = ay1 - ay0;
    scratch_tmp_.resize(sw);
    float* JXL_RESTRICT t = scratch_tmp_.data();

    for (size_t c = 0; c < 3; ++c) {
      out[c].resize(aw * ah);
      const float* JXL_RESTRICT src = coarse.test_gauss[c].data();
      for (size_t y = ay0; y < ay1; ++y) {
        const size_t sy = y / 2;
        const size_t sb = (sy > 0) ? sy - 1 : 0;
        const size_t sa = std::min(sy + 1, sh - 1);
        for (size_t cx = cx0; cx < cx1; ++cx) {
          t[cx] = (y % 2 == 0)
                      ? (0.1f * (src[sb * sw + cx] + src[sa * sw + cx]) +
                         0.8f * src[sy * sw + cx])
                      : (0.5f * (src[sy * sw + cx] + src[sa * sw + cx]));
        }
        float* JXL_RESTRICT dst = out[c].data() + (y - ay0) * aw;
        for (size_t x = ax0; x < ax1; ++x) {
          const size_t sx = x / 2;
          const size_t sxb = (sx > 0) ? sx - 1 : 0;
          const size_t sxa = std::min(sx + 1, sw - 1);
          dst[x - ax0] =
              (x % 2 == 0) ? (0.1f * (t[sxb] + t[sxa]) + 0.8f * t[sx])
                           : (0.5f * (t[sx] + t[sxa]));
        }
      }
    }
  }

  void PoolRegion(size_t k, size_t tx0, size_t ty0, size_t tx1, size_t ty1) {
    if (tx0 >= tx1 || ty0 >= ty1) return;
    LevelState& lv = levels_[k];
    const size_t w = lv.w;
    const size_t h = lv.h;
    const size_t px0 = tx0 * kTile;
    const size_t py0 = ty0 * kTile;
    const size_t px1 = std::min(w, tx1 * kTile);
    const size_t py1 = std::min(h, ty1 * kTile);
    const size_t tw = px1 - px0;
    const size_t th = py1 - py0;
    const bool blur = (w > kRadius && h > kRadius);

    size_t ax0 = blur ? ((px0 > kRadius) ? px0 - kRadius : 0) : px0;
    size_t ay0 = blur ? ((py0 > kRadius) ? py0 - kRadius : 0) : py0;
    size_t ax1 = blur ? std::min(w, px1 + kRadius) : px1;
    size_t ay1 = blur ? std::min(h, py1 + kRadius) : py1;
    ax0 = (ax0 / kAlign) * kAlign;
    ax1 = std::min(w, DivCeil(ax1, kAlign) * kAlign);
    const size_t aw = ax1 - ax0;
    const size_t ah = ay1 - ay0;

    ExpandRegion(k + 1, w, h, ax0, ay0, ax1, ay1, scratch_exp_);
    for (size_t c = 0; c < 3; ++c) {
      scratch_con_[c].resize(aw * ah);
      scratch_mask_[c].resize(aw * ah);
      scratch_blur_[c].resize(tw * th);
    }

    const float scale = (k == 0) ? 1.0f : 2.0f;
    for (size_t y = 0; y < ah; ++y) {
      const size_t src_off = (ay0 + y) * w + ax0;
      const size_t a_off = y * aw;
      const float* JXL_RESTRICT exp0 = scratch_exp_[0].data() + a_off;
      for (size_t c = 0; c < 3; ++c) {
        const float* JXL_RESTRICT g = lv.test_gauss[c].data() + src_off;
        const float* JXL_RESTRICT expc = scratch_exp_[c].data() + a_off;
        const float* JXL_RESTRICT sens = lv.ref_sens[c].data() + src_off;
        const float* JXL_RESTRICT refc = lv.ref_contrast[c].data() + src_off;
        float* JXL_RESTRICT con = scratch_con_[c].data() + a_off;
        float* JXL_RESTRICT msk = scratch_mask_[c].data() + a_off;
        for (size_t x = 0; x < aw; ++x) {
          const float bg = std::max(exp0[x], 0.01f);
          const float wc =
              std::min((g[x] - expc[x]) / bg, 1000.0f) * scale * sens[x];
          con[x] = wc;
          msk[x] = std::min(std::abs(wc), std::abs(refc[x]));
        }
      }
    }

    if (blur) {
      const size_t ry0 = (py0 > kRadius) ? py0 - kRadius : 0;
      const size_t ry1 = std::min(h, py1 + kRadius);
      const size_t rh = ry1 - ry0;
      scratch_horiz_.resize(rh * tw);
      scratch_pad_.resize(tw + 2 * kRadius);
      for (size_t c = 0; c < 3; ++c) {
        const float* JXL_RESTRICT msk = scratch_mask_[c].data();
        for (size_t y = ry0; y < ry1; ++y) {
          for (size_t i = 0; i < tw + 2 * kRadius; ++i) {
            const size_t sx = ReflectExclusive(
                static_cast<ptrdiff_t>(px0 + i) - static_cast<ptrdiff_t>(kRadius),
                w);
            scratch_pad_[i] = msk[(y - ay0) * aw + (sx - ax0)];
          }
          float* JXL_RESTRICT hr = scratch_horiz_.data() + (y - ry0) * tw;
          for (size_t x = 0; x < tw; ++x) {
            float sum = 0.0f;
            for (size_t d = 0; d < kTaps; ++d) {
              sum += blur_kernel_[d] * scratch_pad_[x + d];
            }
            hr[x] = sum;
          }
        }
        float* JXL_RESTRICT blr = scratch_blur_[c].data();
        for (size_t y = py0; y < py1; ++y) {
          float* JXL_RESTRICT br = blr + (y - py0) * tw;
          for (size_t x = 0; x < tw; ++x) {
            float sum = 0.0f;
            for (size_t d = 0; d < kTaps; ++d) {
              const size_t sy = ReflectExclusive(
                  static_cast<ptrdiff_t>(y + d) - static_cast<ptrdiff_t>(kRadius),
                  h);
              sum += blur_kernel_[d] * scratch_horiz_[(sy - ry0) * tw + x];
            }
            br[x] = sum;
          }
        }
      }
    } else {
      for (size_t c = 0; c < 3; ++c) {
        for (size_t y = py0; y < py1; ++y) {
          memcpy(scratch_blur_[c].data() + (y - py0) * tw,
                 scratch_mask_[c].data() + (y - ay0) * aw + (px0 - ax0),
                 tw * sizeof(float));
        }
      }
    }

    for (size_t ty = ty0; ty < ty1; ++ty) {
      for (size_t tx = tx0; tx < tx1; ++tx) {
        const size_t x0 = tx * kTile;
        const size_t x1 = std::min(w, (tx + 1) * kTile);
        const size_t y0 = ty * kTile;
        const size_t y1 = std::min(h, (ty + 1) * kTile);
        double sums[3] = {0.0, 0.0, 0.0};
        for (size_t y = y0; y < y1; ++y) {
          const size_t c_at = (y - ay0) * aw + (x0 - ax0);
          const size_t b_at = (y - py0) * tw + (x0 - px0);
          const size_t r_at = y * w + x0;
          const size_t len = x1 - x0;
          for (size_t i = 0; i < len; ++i) {
            float act[3];
            for (size_t c = 0; c < 3; ++c) {
              act[c] = std::max(
                  0.0f,
                  EvalPow(scratch_blur_[c][b_at + i] * mask_c_ + kEpsilon,
                          pow_q_[c]) -
                      mask_zero_[c]);
            }
            for (size_t c = 0; c < 3; ++c) {
              const float mval = act[0] * xcm_[0][c] + act[1] * xcm_[1][c] +
                                 act[2] * xcm_[2][c];
              const float diff = std::abs(scratch_con_[c][c_at + i] -
                                          lv.ref_contrast[c][r_at + i]);
              const float du =
                  std::max(0.0f, EvalPow(diff + kEpsilon, pow_p_) - diff_zero_) /
                  (1.0f + mval);
              const double d =
                  static_cast<double>(dmax_ * du / (dmax_ + du));
              sums[c] += d * d + 2.0 * static_cast<double>(kEpsilon) * d;
            }
          }
        }
        lv.tiles[ty * lv.tiles_x + tx] = {{sums[0], sums[1], sums[2]}};
      }
      SumTileRow(&lv, ty);
    }
  }

  void ScoreBase() {
    const LevelState& lv = levels_.back();
    const size_t count = lv.w * lv.h;
    const float bg = MeanLuminance(lv.test_gauss[0].data(), count);
    for (size_t c = 0; c < 3; ++c) {
      const float sens = lv.ref_sens[c][0];
      double sum = 0.0;
      for (size_t i = 0; i < count; ++i) {
        const double d = static_cast<double>(
            std::abs(std::min(lv.test_gauss[c][i] / bg, 1000.0f) * sens -
                     lv.ref_contrast[c][i]));
        sum += d * d + 2.0 * static_cast<double>(kEpsilon) * d;
      }
      base_sums_[c] = sum;
    }
  }

  size_t w0_ = 0;
  size_t h0_ = 0;
  float disp_scale_ = 0.0f;
  float disp_black_ = 0.0f;
  float blur_kernel_[kTaps] = {};
  float mask_c_ = 0.0f;
  float dmax_ = 0.0f;
  float mask_zero_[3] = {};
  float diff_zero_ = 0.0f;
  float xcm_[3][3] = {};
  PowerTable pow_q_[3];
  PowerTable pow_p_;
  std::vector<LevelState> levels_;
  double base_sums_[3] = {};

  std::vector<float> scratch_tmp_;
  std::vector<float> scratch_exp_[3];
  std::vector<float> scratch_con_[3];
  std::vector<float> scratch_mask_[3];
  std::vector<float> scratch_blur_[3];
  std::vector<float> scratch_horiz_;
  std::vector<float> scratch_pad_;
};

}  // namespace

Status RefineQuantFieldClosedLoop(const FrameHeader& frame_header,
                                  const Image3F& opsin, const Rect& rect,
                                  ThreadPool* pool,
                                  PassesEncoderState* enc_state) {
  const CompressParams& cparams = enc_state->cparams;
  if (cparams.speed_tier > SpeedTier::kSquirrel ||
      cparams.disable_perceptual_optimizations ||
      (frame_header.flags & FrameHeader::kNoise) != 0 ||
      cparams.dots == Override::kOn ||
      cparams.butteraugli_distance >= kMinButteraugliForDots ||
      enc_state->token_cost_model == nullptr ||
      enc_state->token_cost_model->small_val_bits.empty() ||
      enc_state->band_qf[0][0].xsize() == 0) {
    return true;
  }

  PassesSharedState& shared = enc_state->shared;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  const size_t xsize = shared.frame_dim.xsize;
  const size_t ysize = shared.frame_dim.ysize;
  if (xsize < 8 || ysize < 8 || enc_state->orig_opsin == nullptr ||
      enc_state->orig_opsin->xsize() < xsize ||
      enc_state->orig_opsin->ysize() < ysize) {
    return true;
  }

  constexpr size_t kTileBlocks = 4;
  const size_t tiles_x = DivCeil(shared.frame_dim.xsize_blocks, kTileBlocks);
  const size_t tiles_y = DivCeil(shared.frame_dim.ysize_blocks, kTileBlocks);
  const size_t num_tiles = tiles_x * tiles_y;
  const size_t num_groups = shared.frame_dim.num_groups;

  std::vector<AcGroupBlockCache> group_caches(num_groups);
  std::atomic<bool> init_ok{true};
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, static_cast<uint32_t>(num_groups), ThreadPool::NoInit,
      [&](uint32_t g, size_t /*thread*/) -> Status {
        if (!HWY_DYNAMIC_DISPATCH(InitAcGroupCacheImpl)(
                g, enc_state, opsin, rect, tiles_x, kTileBlocks,
                &group_caches[g])) {
          init_ok.store(false);
        }
        return true;
      },
      "InitAcGroupCache"));
  JXL_ENSURE(init_ok.load());

  JXL_ASSIGN_OR_RETURN(Image3F src_lin,
                       Image3F::Create(memory_manager, xsize, ysize));
  {
    const auto& im = shared.metadata->transform_data.opsin_inverse_matrix;
    float biases_cbrt[3];
    for (size_t c = 0; c < 3; ++c) {
      biases_cbrt[c] = std::cbrt(im.opsin_biases[c]);
    }
    const float inv_scale = 255.0f / shared.metadata->m.IntensityTarget();
    for (size_t y = 0; y < ysize; ++y) {
      XybRowToLinearRgb(enc_state->orig_opsin->ConstPlaneRow(0, y),
                        enc_state->orig_opsin->ConstPlaneRow(1, y),
                        enc_state->orig_opsin->ConstPlaneRow(2, y), xsize,
                        im.inverse_matrix, im.opsin_biases, biases_cbrt,
                        inv_scale, src_lin.PlaneRow(0, y),
                        src_lin.PlaneRow(1, y), src_lin.PlaneRow(2, y));
    }
  }

  Image3F feat_xyb;
  const bool has_patches = shared.image_features.patches.HasAny();
  const bool has_splines = shared.image_features.splines.HasAny();
  if (has_patches || has_splines) {
    JXL_ASSIGN_OR_RETURN(feat_xyb,
                         Image3F::Create(memory_manager, xsize, ysize));
    ZeroFillImage(&feat_xyb);
    if (has_patches) {
      const std::vector<ExtraChannelInfo> empty_ec;
      for (size_t y = 0; y < ysize; ++y) {
        float* rows[3] = {feat_xyb.PlaneRow(0, y), feat_xyb.PlaneRow(1, y),
                          feat_xyb.PlaneRow(2, y)};
        JXL_RETURN_IF_ERROR(shared.image_features.patches.AddOneRow(
            rows, y, 0, xsize, empty_ec));
      }
    }
    if (has_splines) {
      shared.image_features.splines.AddTo(&feat_xyb, Rect(0, 0, xsize, ysize));
    }
  }
  const Image3F* feat_ptr = (has_patches || has_splines) ? &feat_xyb : nullptr;

  auto ssim2 = jxl::make_unique<Ssim2IncrementalArbiter>();
  auto cvvdp = jxl::make_unique<CvvdpIncrementalArbiter>();
  ssim2->Init(src_lin);
  cvvdp->Init(src_lin);

  JXL_ASSIGN_OR_RETURN(Image3F cur_pre_gab,
                       Image3F::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(Image3F cand_pre_gab,
                       Image3F::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(Image3F cur_lin,
                       Image3F::Create(memory_manager, xsize, ysize));

  const AcTokenCostModel& tcm = *enc_state->token_cost_model;
  auto eval_all_groups = [&](int step_mode, bool use_rdoq, bool commit_cur,
                             Image3F* out_pre_gab,
                             std::vector<float>* out_tile_bits,
                             std::vector<float>* out_group_bits,
                             double* out_total_bits) -> Status {
    if (out_tile_bits != nullptr) out_tile_bits->assign(num_tiles, 0.0f);
    if (out_group_bits != nullptr) out_group_bits->assign(num_groups, 0.0f);
    std::vector<float> grp_bits(num_groups, 0.0f);
    std::atomic<bool> eval_ok{true};
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, static_cast<uint32_t>(num_groups), ThreadPool::NoInit,
        [&](uint32_t g, size_t /*thread*/) -> Status {
          const int32_t qf_step =
              (step_mode == 1)
                  ? 1
                  : (step_mode == -1 ? ((g % 2 == 0) ? -1 : 0) : 0);
          float ac_bits = 0.0f;
          float* tile_ptr =
              (out_tile_bits != nullptr) ? out_tile_bits->data() : nullptr;
          if (!HWY_DYNAMIC_DISPATCH(EvalAcGroupCacheImpl)(
                  enc_state, &group_caches[g], qf_step, use_rdoq, commit_cur,
                  /*only_write_changed=*/false, out_pre_gab, tile_ptr, &ac_bits,
                  nullptr)) {
            eval_ok.store(false);
            return true;
          }
          const float qf_bits =
              EstimateGroupQfBits(group_caches[g], tcm, qf_step, tile_ptr);
          grp_bits[g] = ac_bits + qf_bits;
          return true;
        },
        "EvalAcGroups"));
    JXL_ENSURE(eval_ok.load());
    double sum = 0.0;
    for (size_t g = 0; g < num_groups; ++g) {
      sum += static_cast<double>(grp_bits[g]);
      if (out_group_bits != nullptr) (*out_group_bits)[g] = grp_bits[g];
    }
    *out_total_bits = sum;
    return true;
  };

  std::vector<float> plain_tile_bits;
  std::vector<float> fine_tile_bits;
  std::vector<float> cur_tile_bits;
  std::vector<float> cur_group_bits;
  double plain_bits = 0.0;
  double fine_bits = 0.0;
  double coarse_bits = 0.0;
  double cur_bits = 0.0;

  JXL_RETURN_IF_ERROR(eval_all_groups(0, false, false, &cand_pre_gab,
                                      &plain_tile_bits, nullptr, &plain_bits));
  ReconstructLinearRgb(frame_header, shared, cand_pre_gab, feat_ptr,
                       Rect(0, 0, xsize, ysize), &cur_lin);
  const double plain_d_ss2 = ssim2->ScoreFull(cur_lin);
  const double plain_d_cv = cvvdp->ScoreFull(cur_lin);

  JXL_RETURN_IF_ERROR(eval_all_groups(1, false, false, &cand_pre_gab,
                                      &fine_tile_bits, nullptr, &fine_bits));
  ReconstructLinearRgb(frame_header, shared, cand_pre_gab, feat_ptr,
                       Rect(0, 0, xsize, ysize), &cur_lin);
  const double fine_d_ss2 = ssim2->ScoreFull(cur_lin);
  const double fine_d_cv = cvvdp->ScoreFull(cur_lin);

  JXL_RETURN_IF_ERROR(eval_all_groups(-1, false, false, &cand_pre_gab, nullptr,
                                      nullptr, &coarse_bits));
  ReconstructLinearRgb(frame_header, shared, cand_pre_gab, feat_ptr,
                       Rect(0, 0, xsize, ysize), &cur_lin);
  const double coarse_d_ss2 = ssim2->ScoreFull(cur_lin);
  const double coarse_d_cv = cvvdp->ScoreFull(cur_lin);

  JXL_RETURN_IF_ERROR(eval_all_groups(0, true, true, &cur_pre_gab,
                                      &cur_tile_bits, &cur_group_bits,
                                      &cur_bits));
  JXL_RETURN_IF_ERROR(CopyImageTo(cur_pre_gab, &cand_pre_gab));
  ReconstructLinearRgb(frame_header, shared, cur_pre_gab, feat_ptr,
                       Rect(0, 0, xsize, ysize), &cur_lin);
  double cur_d_ss2 = ssim2->ScoreFull(cur_lin);
  double cur_d_cv = cvvdp->ScoreFull(cur_lin);

  const double ref_bits = std::max(cur_bits, 1.0);
  auto rate_term = [ref_bits](double bits) -> double {
    return 100.0 * std::log(std::max(bits, 1.0) / ref_bits);
  };
  const double r_plain = rate_term(plain_bits);
  const double r_fine = rate_term(fine_bits);
  const double r_coarse = rate_term(coarse_bits);

  const double s_fine_ss2 =
      (r_fine > r_plain + 1e-4)
          ? std::max(0.0, (plain_d_ss2 - fine_d_ss2) / (r_fine - r_plain))
          : 0.0;
  const double s_coarse_ss2 =
      (r_plain > r_coarse + 1e-4)
          ? std::max(0.0, (coarse_d_ss2 - plain_d_ss2) / (r_plain - r_coarse))
          : 0.0;
  const double s_fine_cv =
      (r_fine > r_plain + 1e-4)
          ? std::max(0.0, (plain_d_cv - fine_d_cv) / (r_fine - r_plain))
          : 0.0;
  const double s_coarse_cv =
      (r_plain > r_coarse + 1e-4)
          ? std::max(0.0, (coarse_d_cv - plain_d_cv) / (r_plain - r_coarse))
          : 0.0;

  const double slope_ss2 =
      0.85 * ((s_fine_ss2 > 0.0 && s_coarse_ss2 > 0.0)
                  ? 0.5 * (s_fine_ss2 + s_coarse_ss2)
                  : std::max(s_fine_ss2, s_coarse_ss2));
  const double slope_cv =
      0.85 * ((s_fine_cv > 0.0 && s_coarse_cv > 0.0)
                  ? 0.5 * (s_fine_cv + s_coarse_cv)
                  : std::max(s_fine_cv, s_coarse_cv));
  if (slope_ss2 <= 1e-8 && slope_cv <= 1e-8) {
    return true;
  }

  std::vector<size_t> tile_group(num_tiles, num_groups);
  std::vector<std::vector<size_t>> tile_tr_indices(num_tiles);
  for (size_t g = 0; g < num_groups; ++g) {
    const auto& trs = group_caches[g].transforms;
    for (size_t i = 0; i < trs.size(); ++i) {
      const uint32_t t = trs[i].tile_idx;
      tile_group[t] = g;
      tile_tr_indices[t].push_back(i);
    }
  }

  constexpr size_t kMaxIters = 2;
  constexpr size_t kMaxGreedy = 240;
  constexpr size_t kMaxPerGroup = 12;
  const size_t max_moves = std::min<size_t>(
      kMaxGreedy, std::max<size_t>(1, static_cast<size_t>(0.25f * num_tiles)));
  std::vector<uint8_t> tile_step(num_tiles, 0);
  size_t total_moves = 0;

  Ssim2IncrementalArbiter::Checkpoint cp_ss2;
  CvvdpIncrementalArbiter::Checkpoint cp_cv;
  std::vector<float> saved_lin;

  for (size_t iter = 0; iter < kMaxIters && total_moves < max_moves; ++iter) {
    const std::vector<float> err_map = ssim2->ErrorMap();
    std::vector<double> tile_err(num_tiles, 0.0);
    double total_emap = 0.0;
    for (size_t y = 0; y < ysize; ++y) {
      const size_t ty = (y / kBlockDim) / kTileBlocks;
      for (size_t x = 0; x < xsize; ++x) {
        const size_t tx = (x / kBlockDim) / kTileBlocks;
        const double v = static_cast<double>(err_map[y * xsize + x]);
        tile_err[ty * tiles_x + tx] += v;
        total_emap += v;
      }
    }
    const double scale_ss2 =
        (total_emap > 1e-9 && plain_d_ss2 > fine_d_ss2)
            ? ((plain_d_ss2 - fine_d_ss2) / total_emap)
            : 0.0;

    struct CandTile {
      double score;
      size_t tile;
      size_t group;
    };
    std::vector<CandTile> all_cands;
    all_cands.reserve(num_tiles);
    for (size_t t = 0; t < num_tiles; ++t) {
      if (tile_step[t] != 0 || tile_tr_indices[t].empty()) continue;
      const size_t g = tile_group[t];
      bool can_step = false;
      for (size_t idx : tile_tr_indices[t]) {
        if (group_caches[g].transforms[idx].cur_quant_ac < 255) {
          can_step = true;
          break;
        }
      }
      if (!can_step) continue;

      const double dbits =
          static_cast<double>(std::max(0.0f, fine_tile_bits[t] - plain_tile_bits[t]));
      const double drate = rate_term(cur_bits + dbits) - rate_term(cur_bits);
      const double est_gain = tile_err[t] * scale_ss2;
      const double proxy_j = est_gain - slope_ss2 * drate;
      if (proxy_j > 0.0) {
        all_cands.push_back({proxy_j, t, g});
      }
    }
    if (all_cands.empty()) break;

    std::sort(all_cands.begin(), all_cands.end(),
              [](const CandTile& a, const CandTile& b) {
                return (a.score != b.score) ? (a.score > b.score)
                                            : (a.tile < b.tile);
              });

    std::vector<std::vector<size_t>> group_queues(num_groups);
    for (const CandTile& c : all_cands) {
      if (group_queues[c.group].size() < kMaxPerGroup) {
        group_queues[c.group].push_back(c.tile);
      }
    }
    std::vector<size_t> group_q_pos(num_groups, 0);

    while (total_moves < max_moves) {
      struct BatchItem {
        size_t group;
        size_t tile;
        float cand_bits;
        Rect dirty_px;
        std::vector<int32_t> old_quant;
      };
      std::vector<BatchItem> batch;
      for (size_t g = 0; g < num_groups; ++g) {
        if (group_q_pos[g] < group_queues[g].size()) {
          const size_t t = group_queues[g][group_q_pos[g]++];
          BatchItem item;
          item.group = g;
          item.tile = t;
          item.cand_bits = 0.0f;
          item.dirty_px = Rect(0, 0, 0, 0);
          item.old_quant.reserve(tile_tr_indices[t].size());
          for (size_t idx : tile_tr_indices[t]) {
            int32_t& q = group_caches[g].transforms[idx].cur_quant_ac;
            item.old_quant.push_back(q);
            q = std::min<int32_t>(q + 1, 255);
          }
          batch.push_back(std::move(item));
        }
      }
      if (batch.empty()) break;

      std::atomic<bool> batch_ok{true};
      JXL_RETURN_IF_ERROR(RunOnPool(
          pool, 0, static_cast<uint32_t>(batch.size()), ThreadPool::NoInit,
          [&](uint32_t bi, size_t /*thread*/) -> Status {
            BatchItem& item = batch[bi];
            AcGroupBlockCache& cache = group_caches[item.group];
            float ac_bits = 0.0f;
            if (!HWY_DYNAMIC_DISPATCH(EvalAcGroupCacheImpl)(
                    enc_state, &cache, /*qf_step_override=*/0,
                    /*use_rdoq=*/true, /*commit_cur_slot=*/false,
                    /*only_write_changed=*/true, &cand_pre_gab, nullptr,
                    &ac_bits, &item.dirty_px)) {
              batch_ok.store(false);
              return true;
            }
            const float qf_bits = EstimateGroupQfBits(cache, tcm, 0, nullptr);
            item.cand_bits = ac_bits + qf_bits;
            return true;
          },
          "EvalCandBatch"));
      JXL_ENSURE(batch_ok.load());

      for (BatchItem& item : batch) {
        const size_t g = item.group;
        const size_t t = item.tile;
        if (total_moves >= max_moves ||
            (item.dirty_px.xsize() == 0 &&
             item.cand_bits >= cur_group_bits[g])) {
          if (item.dirty_px.xsize() != 0) {
            for (size_t c = 0; c < 3; ++c) {
              for (size_t y = 0; y < item.dirty_px.ysize(); ++y) {
                memcpy(cand_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       cur_pre_gab.ConstPlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       item.dirty_px.xsize() * sizeof(float));
              }
            }
          }
          for (size_t k = 0; k < tile_tr_indices[t].size(); ++k) {
            group_caches[g].transforms[tile_tr_indices[t][k]].cur_quant_ac =
                item.old_quant[k];
          }
          continue;
        }

        Rect dirty_lin = item.dirty_px;
        if (dirty_lin.xsize() != 0 && dirty_lin.ysize() != 0) {
          const size_t lx0 = (dirty_lin.x0() > 0) ? dirty_lin.x0() - 1 : 0;
          const size_t ly0 = (dirty_lin.y0() > 0) ? dirty_lin.y0() - 1 : 0;
          const size_t lx1 =
              std::min(xsize, dirty_lin.x0() + dirty_lin.xsize() + 1);
          const size_t ly1 =
              std::min(ysize, dirty_lin.y0() + dirty_lin.ysize() + 1);
          dirty_lin = Rect(lx0, ly0, lx1 - lx0, ly1 - ly0);

          const size_t lw = dirty_lin.xsize();
          const size_t lh = dirty_lin.ysize();
          saved_lin.resize(3 * lw * lh);
          for (size_t c = 0; c < 3; ++c) {
            for (size_t y = 0; y < lh; ++y) {
              memcpy(saved_lin.data() + (c * lh + y) * lw,
                     cur_lin.ConstPlaneRow(c, dirty_lin.y0() + y) +
                         dirty_lin.x0(),
                     lw * sizeof(float));
            }
          }
          ReconstructLinearRgb(frame_header, shared, cand_pre_gab, feat_ptr,
                               dirty_lin, &cur_lin);
        }

        const double d_ss2 =
            (dirty_lin.xsize() != 0)
                ? ssim2->UpdateCheckpointed(cur_lin, dirty_lin, &cp_ss2)
                : cur_d_ss2;
        const double d_cv =
            (dirty_lin.xsize() != 0)
                ? cvvdp->UpdateCheckpointed(cur_lin, dirty_lin, &cp_cv)
                : cur_d_cv;

        const double new_bits =
            cur_bits + static_cast<double>(item.cand_bits - cur_group_bits[g]);
        const double delta_rate = rate_term(new_bits) - rate_term(cur_bits);
        const double gain_ss2 = cur_d_ss2 - d_ss2;
        const double gain_cv = cur_d_cv - d_cv;

        const double norm_ss2 =
            (slope_ss2 > 1e-8) ? (gain_ss2 / slope_ss2) : 0.0;
        const double norm_cv = (slope_cv > 1e-8) ? (gain_cv / slope_cv) : 0.0;
        const double w_ss2 = (slope_ss2 > 1e-8 && slope_cv > 1e-8)
                                 ? 0.5
                                 : (slope_ss2 > 1e-8 ? 1.0 : 0.0);
        const double w_cv = (slope_ss2 > 1e-8 && slope_cv > 1e-8)
                                ? 0.5
                                : (slope_cv > 1e-8 ? 1.0 : 0.0);
        const double j_gain = w_cv * norm_cv + w_ss2 * norm_ss2 - delta_rate;
        const bool cv_ok = (slope_cv <= 1e-8) || (gain_cv >= -1e-4);

        if (j_gain > 1e-6 && cv_ok) {
          for (AcTransformCache& tr : group_caches[g].transforms) {
            tr.cur_slot = tr.cand_slot;
          }
          if (item.dirty_px.xsize() != 0) {
            for (size_t c = 0; c < 3; ++c) {
              for (size_t y = 0; y < item.dirty_px.ysize(); ++y) {
                memcpy(cur_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       cand_pre_gab.ConstPlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       item.dirty_px.xsize() * sizeof(float));
              }
            }
          }
          cur_bits = new_bits;
          cur_group_bits[g] = item.cand_bits;
          cur_d_ss2 = d_ss2;
          cur_d_cv = d_cv;
          tile_step[t] = 1;
          ++total_moves;
        } else {
          if (dirty_lin.xsize() != 0) {
            ssim2->Rollback(cp_ss2);
            cvvdp->Rollback(cp_cv);
            const size_t lw = dirty_lin.xsize();
            const size_t lh = dirty_lin.ysize();
            for (size_t c = 0; c < 3; ++c) {
              for (size_t y = 0; y < lh; ++y) {
                memcpy(cur_lin.PlaneRow(c, dirty_lin.y0() + y) + dirty_lin.x0(),
                       saved_lin.data() + (c * lh + y) * lw,
                       lw * sizeof(float));
              }
            }
            for (size_t c = 0; c < 3; ++c) {
              for (size_t y = 0; y < item.dirty_px.ysize(); ++y) {
                memcpy(cand_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       cur_pre_gab.ConstPlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       item.dirty_px.xsize() * sizeof(float));
              }
            }
          }
          for (size_t k = 0; k < tile_tr_indices[t].size(); ++k) {
            group_caches[g].transforms[tile_tr_indices[t][k]].cur_quant_ac =
                item.old_quant[k];
          }
        }
      }
    }
  }

  if (total_moves > 0) {
    for (size_t g = 0; g < num_groups; ++g) {
      const Rect block_group_rect = shared.frame_dim.BlockGroupRect(g);
      for (const AcTransformCache& tr : group_caches[g].transforms) {
        if (tr.cur_quant_ac == tr.base_quant_ac) continue;
        const int32_t q_new = tr.cur_quant_ac;
        const float ratio =
            static_cast<float>(q_new) / static_cast<float>(tr.base_quant_ac);
        for (size_t iy = 0; iy < tr.cov_y; ++iy) {
          int32_t* JXL_RESTRICT qf_row =
              block_group_rect.Row(&shared.raw_quant_field, tr.by + iy);
          for (size_t ix = 0; ix < tr.cov_x; ++ix) {
            qf_row[tr.bx + ix] = q_new;
          }
        }
        for (size_t c = 0; c < 3; ++c) {
          for (size_t s = 0; s < 4; ++s) {
            for (size_t iy = 0; iy < tr.cov_y; ++iy) {
              float* JXL_RESTRICT bqf_row =
                  block_group_rect.Row(&enc_state->band_qf[c][s], tr.by + iy);
              for (size_t ix = 0; ix < tr.cov_x; ++ix) {
                float v = tr.base_band_q[c][s] * ratio;
                if (c == 1 && s == 0 && iy == 0 && ix == 0) {
                  uint32_t bits = 0;
                  memcpy(&bits, &v, sizeof(bits));
                  bits = (bits & ~31u) | tr.act_bits;
                  memcpy(&v, &bits, sizeof(bits));
                }
                bqf_row[tr.bx + ix] = v;
              }
            }
          }
        }
      }
    }
  }

  return true;
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
