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
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/epf.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/loop_filter.h"

namespace jxl {

struct EpfWorkspace {
  AlignedMemory mem;
  size_t cap_floats = 0;
};

struct AcsEvalWorkspace {
  AlignedMemory mem;
  size_t cap_mem_bytes = 0;
  AlignedMemory fmem;
  size_t cap_fmem_bytes = 0;
};

struct AcBlockEvalSlot {
  int32_t quant_ac = -1;
  uint8_t use_rdoq = 0;
  int8_t rdoq_level = 0;
  int8_t ytox_q = 0;
  int8_t ytob_q = 0;
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
  uint8_t dc_idx = 0;
  bool few_colors = false;
  bool acs_changed = false;
  int8_t cur_rdoq_level = 0;
  int8_t cur_ytox_q = 0;
  int8_t cur_ytob_q = 0;
  int32_t base_quant_ac = 1;
  int32_t cur_quant_ac = 1;
  uint32_t act_bits = 0;
  uint32_t tile_idx = 0;
  uint32_t cfl_tile_idx = 0;
  float base_band_q[3][4] = {};
  std::vector<float> src_dct[3];
  static constexpr size_t kNumSlots = 6;
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

struct AcsSubSpec {
  uint8_t bx = 0;
  uint8_t by = 0;
  uint8_t acs_raw = 0;
  int8_t rdoq_lvl = 0;
  int8_t ytox_q = 0;
  int8_t ytob_q = 0;
  int32_t quant_ac = 1;
  uint32_t act_bits = 0;
  float band_q[3][4] = {};
};

}  // namespace jxl
#endif  // LIB_JXL_ENC_GROUP_CACHE_TYPES_H_

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::AbsDiff;
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::ApproximateReciprocal;
using hwy::HWY_NAMESPACE::CopySignToAbs;
using hwy::HWY_NAMESPACE::Div;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Round;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::Vec;
using hwy::HWY_NAMESPACE::ZeroIfNegative;

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
    bool allow_round_up, float round_up_margin,
    int32_t out_cands[kMaxCandidates]) {
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
  if (allow_round_up &&
      abs_ideal + round_up_margin >= static_cast<float>(q_abs + 1)) {
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
                               bool few_colors, int8_t rdoq_level,
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
  const float aq_vfloor =
      (rdoq_level == -2) ? 0.66f : ((rdoq_level == -3) ? 0.76f : 0.86f);
  const float restore_floor =
      (rdoq_level <= -2)
          ? aq_vfloor
          : (few_colors ? ((rdoq_level > 0) ? 0.48f : 0.50f)
                        : (((rdoq_level > 0) ? 0.45f : 0.5f) * b_c));
  const float round_up_margin = (rdoq_level > 0) ? 0.60f : 0.56f;
  const float dz_pull =
      (rdoq_level <= -2)
          ? 0.0f
          : ((rdoq_level > 0) ? 0.065f : ((rdoq_level < 0) ? 0.015f : 0.04f));
  const float dz_min =
      (rdoq_level <= -2)
          ? aq_vfloor
          : ((rdoq_level > 0) ? 0.455f : ((rdoq_level < 0) ? 0.515f : 0.49f));

  const bool is_partial_block =
      ((1 << static_cast<size_t>(quant_kind)) & kPartialBlockKinds) != 0;
  auto rescue_quad0 = [&]() -> int32_t {
    if (is_partial_block || c != 1 || covered_blocks > 4 || eob_scale <= 0.0f) {
      return 0;
    }
    const float rescue_mul =
        (rdoq_level <= -2)
            ? ((rdoq_level == -2) ? 1.25f
                                  : ((rdoq_level == -3) ? 1.42f : 1.60f))
            : 1.0f;
    float best_abs = (few_colors ? 0.54f : 0.46f) * rescue_mul;
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
      const float near_dz_floor = std::max(dz_min, t0_full - dz_pull);
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

  static constexpr float kBandDistWeight[4] = {0.92f, 1.00f, 1.27829313f,
                                               1.55410683f};
  const float sparse_w = Clamp1((eob_scale - 0.32f) / 0.42f, 0.0f, 1.0f);
  const float base_ref = (c == 1) ? 2.79515505f : 3.55426836f;
  float rdoq_ref_mul = 1.0f;
  if (rdoq_level > 0) {
    rdoq_ref_mul = 1.46380079f;
  } else if (rdoq_level == -1) {
    rdoq_ref_mul = 0.705992103f;
  } else if (rdoq_level <= -2) {
    rdoq_ref_mul = (rdoq_level == -2)
                       ? 0.558125257f
                       : ((rdoq_level == -3) ? 0.435468376f : 0.334956259f);
  }
  const float eff_ref_bits =
      base_ref * (1.0f + 0.353990942f * eob_scale) * rdoq_ref_mul;
  constexpr float kQuadTilt = 0.494345099f;
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
    const float near_dz_floor = std::max(dz_min, t0_full - dz_pull);
    const bool can_restore =
        (c == 1)
            ? (wi < restore_len)
            : ((wi < restore_len / 2) &&
               (is_partial_block ? (k <= 8) : (s >= 2)));
    const bool allow_zero_to_one =
        (can_restore && abs_ideal >= restore_floor) ||
        (!few_colors && abs_ideal >= near_dz_floor);
    const bool allow_round_up =
        (rdoq_level > -2) && !(few_colors && s <= 1);
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
        round_up_margin, candidates);

    float dist_cost[kMaxCandidates];
    uint32_t packed_sym[kMaxCandidates];
    dist_cost[0] = 0.0f;
    packed_sym[0] = 0u;
    if (cand_count > 1) {
      const float raw_h = (hfix == 0) ? 0.58f : ((c == 1) ? 0.64f : 0.62f);
      const float perc_t0 =
          std::max(0.50f, t0_full - (1.0f - kQuadTilt) * (raw_h - base_h));
      const float t_ratio = t_ref / perc_t0;
      float w_base = (eff_ref_bits / denom_ref) * (t_ratio * t_ratio);
      if (!is_partial_block && rdoq_level < 0) {
        if (hfix == 0 && s >= 2) {
          w_base *= 1.12768173f;
        } else if (hfix == 3) {
          w_base *= 0.900313497f;
        }
      }
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
        const bool allow_round_up =
            (rdoq_level > -2) && !(few_colors && s <= 1);
        const float abs_ideal = std::abs(ideal);
        int32_t candidates[kMaxCandidates];
        const size_t cand_count = BuildRdoqCandidates(
            current, ideal, abs_ideal, false, allow_round_up, round_up_margin,
            candidates);
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

// Pre-DCT POCS continuation for clipped white/black SDR regions: allows
// Gibbs ringing lobes adjacent to saturated pixels to push freely beyond the
// SDR [0, 1] display clip along the gray axis (Y and B shifted identically so
// CfL carries B at zero chroma cost).
static bool ContinueSaturatedBlock(
    const PassesEncoderState* enc_state, AcStrategy acs, size_t cov_x,
    size_t cov_y, size_t xblocks, size_t yblocks, int32_t quant_ac,
    const float* const JXL_RESTRICT opsin_rows[3], size_t opsin_stride,
    size_t bx_px_offset, size_t frame_px0, size_t frame_py0,
    float* JXL_RESTRICT scratch_space, float* JXL_RESTRICT clip_work,
    float* JXL_RESTRICT shift_out) {
  const size_t covered = cov_x * cov_y;
  if (covered > 16) return false;
  if (enc_state->cparams.speed_tier > SpeedTier::kSquirrel ||
      enc_state->cparams.disable_perceptual_optimizations ||
      enc_state->cparams.noise == Override::kOn ||
      enc_state->cparams.photon_noise_iso > 0 ||
      !enc_state->cparams.manual_noise.empty()) {
    return false;
  }
  const PassesSharedState& shared = enc_state->shared;
  if (shared.metadata == nullptr ||
      shared.metadata->m.bit_depth.floating_point_sample ||
      shared.metadata->m.color_encoding.IsGray() ||
      !shared.metadata->m.color_encoding.IsSRGB()) {
    return false;
  }
  const Image3F* orig_opsin = enc_state->orig_opsin;
  if (orig_opsin == nullptr || orig_opsin->xsize() == 0 ||
      orig_opsin->ysize() == 0) {
    return false;
  }

  const auto& im = shared.metadata->transform_data.opsin_inverse_matrix;
  const auto& M = im.inverse_matrix;
  const float det =
      M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
      M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
      M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  if (std::abs(det) < 1e-12f) return false;
  const float s =
      (shared.metadata->m.IntensityTarget() / 255.0f) / det;
  const float dm0 =
      s * ((M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
           (M[0][1] * M[2][2] - M[0][2] * M[2][1]) +
           (M[0][1] * M[1][2] - M[0][2] * M[1][1]));
  const float dm1 =
      s * (-(M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
           (M[0][0] * M[2][2] - M[0][2] * M[2][0]) -
           (M[0][0] * M[1][2] - M[0][2] * M[1][0]));
  const float dm2 =
      s * ((M[1][0] * M[2][1] - M[1][1] * M[2][0]) -
           (M[0][0] * M[2][1] - M[0][1] * M[2][0]) +
           (M[0][0] * M[1][1] - M[0][1] * M[1][0]));
  const float g0 = std::cbrt(std::max(0.0f, dm0 + im.opsin_biases[0])) -
                   std::cbrt(im.opsin_biases[0]);
  const float g1 = std::cbrt(std::max(0.0f, dm1 + im.opsin_biases[1])) -
                   std::cbrt(im.opsin_biases[1]);
  const float g2 = std::cbrt(std::max(0.0f, dm2 + im.opsin_biases[2])) -
                   std::cbrt(im.opsin_biases[2]);
  const float x_white = 0.5f * (g0 - g1);
  const float y_white = 0.5f * (g0 + g1);
  const float b_white = g2;

  const size_t pw = cov_x * kBlockDim;
  const size_t ph = cov_y * kBlockDim;
  const size_t n = pw * ph;
  const size_t ox_lim = orig_opsin->xsize() - 1;
  const size_t oy_lim = orig_opsin->ysize() - 1;
  constexpr float kYTol = 0.002f;
  constexpr float kChromaTol = 0.0025f;

  constexpr size_t kClipArea = 16 * kDCTBlockSize;
  float* JXL_RESTRICT src_y = clip_work;
  float* JXL_RESTRICT tgt_y = src_y + kClipArea;
  float* JXL_RESTRICT coeff = tgt_y + kClipArea;
  float* JXL_RESTRICT recon = coeff + kClipArea;
  uint8_t* JXL_RESTRICT cls = reinterpret_cast<uint8_t*>(recon + kClipArea);

  size_t saturated = 0;
  bool any_white = false;
  bool any_black = false;
  bool any_normal = false;
  for (size_t y = 0; y < ph; ++y) {
    const size_t fy = std::min(frame_py0 + y, oy_lim);
    const float* JXL_RESTRICT rx = orig_opsin->ConstPlaneRow(0, fy);
    const float* JXL_RESTRICT ry = orig_opsin->ConstPlaneRow(1, fy);
    const float* JXL_RESTRICT rb = orig_opsin->ConstPlaneRow(2, fy);
    for (size_t x = 0; x < pw; ++x) {
      const size_t fx = std::min(frame_px0 + x, ox_lim);
      const float vx = rx[fx];
      const float vy = ry[fx];
      const float vb = rb[fx];
      uint8_t c = 0;
      if (vy >= y_white - kYTol && std::abs(vx - x_white) <= kChromaTol &&
          vb >= b_white - kChromaTol) {
        c = 1;
        any_white = true;
      } else if (vy <= kYTol && std::abs(vx) <= kChromaTol &&
                 vb <= kChromaTol) {
        c = 2;
        any_black = true;
      } else {
        any_normal = true;
      }
      saturated += static_cast<size_t>(c != 0);
      cls[y * pw + x] = c;
    }
  }
  if ((!any_normal && (any_white != any_black)) || saturated * 50 < n) {
    return false;
  }

  for (size_t y = 0; y < ph; ++y) {
    const float* JXL_RESTRICT row =
        opsin_rows[1] + bx_px_offset + y * opsin_stride;
    memcpy(src_y + y * pw, row, pw * sizeof(float));
  }
  memcpy(tgt_y, src_y, n * sizeof(float));

  const float* JXL_RESTRICT qm =
      shared.quantizer.InvDequantMatrix(acs.Strategy(), 1);
  const float* JXL_RESTRICT deqm =
      shared.quantizer.DequantMatrix(acs.Strategy(), 1);
  const float qac = shared.quantizer.Scale() * quant_ac;
  const float inv_qac = shared.quantizer.inv_quant_ac(quant_ac);
  const float bias_one = kDefaultQuantBias[1];
  const size_t wc = xblocks * kBlockDim;
  constexpr float kMaxShift = 0.025f;

  for (size_t iter = 0; iter < 3; ++iter) {
    TransformFromPixels(acs.Strategy(), tgt_y, pw, coeff, scratch_space);
    for (size_t cy = 0; cy < yblocks * kBlockDim; ++cy) {
      const size_t x_start = (cy < yblocks) ? xblocks : 0;
      for (size_t cx = x_start; cx < wc; ++cx) {
        const size_t pos = cy * wc + cx;
        const float val = coeff[pos] * (qm[pos] * qac);
        const float abs_v = std::abs(val);
        int32_t q = 0;
        if (abs_v >= 0.58f) {
          const int32_t mag = static_cast<int32_t>(std::round(abs_v));
          q = (val >= 0.0f) ? mag : -mag;
        }
        coeff[pos] = ReconstructedLevel(q, bias_one) * deqm[pos] * inv_qac;
      }
    }
    TransformToPixels(acs.Strategy(), coeff, recon, pw, scratch_space);
    for (size_t i = 0; i < n; ++i) {
      if (cls[i] == 1) {
        tgt_y[i] = Clamp1(recon[i], src_y[i], src_y[i] + kMaxShift);
      } else if (cls[i] == 2) {
        tgt_y[i] = Clamp1(recon[i], src_y[i] - kMaxShift, src_y[i]);
      }
    }
  }

  bool any_shift = false;
  for (size_t i = 0; i < n; ++i) {
    const float d = tgt_y[i] - src_y[i];
    shift_out[i] = d;
    if (d != 0.0f) any_shift = true;
  }
  return any_shift;
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
                     bool few_colors = false, int8_t rdoq_level = 0) {
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
  constexpr float scalar_bias_slope = 0.449189574f;
  const float bias_scale = Clamp1(1.35f * eob_scale, 0.0f, 1.0f);
  const float dz_shift =
      (rdoq_level > 0) ? -0.025f : ((rdoq_level == -1) ? 0.028f : 0.0f);
  const float aq_dz_mul =
      (rdoq_level == -2) ? 1.20f : ((rdoq_level == -3) ? 1.36f : 1.54f);
  const float aq_round_bias =
      (rdoq_level == -2) ? 0.045f : ((rdoq_level == -3) ? 0.085f : 0.130f);

  for (size_t v8 = 0; v8 < kBlockDim; ++v8) {
    if ((v8 & 1) == 0) {
      for (size_t u8 = 0; u8 < kBlockDim; ++u8) {
        const size_t hfix =
            (static_cast<size_t>(v8 >= 4) * 2 + static_cast<size_t>(u8 >= 4));
        const uint8_t s = kCoeffBand[v8][u8];
        const float base_t0 = quad_thres[hfix] + zero_thres[4 + s];
        const float min_bias = (c == 1 && s == 3) ? -0.010f : -0.004f;
        float t0 = 0.0f;
        float bias = 0.0f;
        if (rdoq_level <= -2) {
          t0 = Clamp1(base_t0 * aq_dz_mul, 0.485f, 1.18f);
          bias = bias_scale *
                 (Clamp1(scalar_bias_slope * (base_t0 - 0.568f), min_bias,
                         0.040f) +
                  aq_round_bias);
        } else {
          t0 = Clamp1(base_t0 + dz_shift, 0.485f, 0.78f);
          bias = bias_scale *
                 Clamp1(scalar_bias_slope * (t0 - 0.568f), min_bias, 0.040f);
        }
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
        zero_thres, eob_scale, few_colors, rdoq_level, block_in, quant,
        block_out, rdoq_choices);
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
        const float bias = bias_scale *
                           Clamp1(scalar_bias_slope * (t0 - 0.568f), -0.004f,
                                  0.040f);
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
                               bool few_colors = false, int8_t rdoq_level = 0) {
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
                  rdoq_choices, few_colors, rdoq_level);

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

  const size_t dc_stride =
      (dc != nullptr) ? static_cast<size_t>(dc->PixelsPerRow()) : 0;
  const size_t opsin_stride = static_cast<size_t>(opsin.PixelsPerRow());

  const ImageI& full_quant_field = enc_state->shared.raw_quant_field;
  const ImageI& clip_quant_field =
      (enc_state->base_raw_quant_field.xsize() >=
           block_group_rect.x0() + xsize_blocks &&
       enc_state->base_raw_quant_field.ysize() >=
           block_group_rect.y0() + ysize_blocks)
          ? enc_state->base_raw_quant_field
          : full_quant_field;
  const CompressParams& cparams = enc_state->cparams;
  const size_t num_passes = enc_state->progressive_splitter.GetNumPasses();
  JXL_ENSURE(num_passes > 0);
  const bool has_band_qf =
      cparams.speed_tier <= SpeedTier::kHare &&
      !cparams.disable_perceptual_optimizations && num_passes == 1 &&
      cparams.butteraugli_distance <= 15.0f && cparams.resampling <= 1 &&
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
  constexpr size_t kClipArea = 16 * kDCTBlockSize;
  size_t fmem_bytes =
      (5 * AcStrategy::kMaxCoeffArea + dct_scratch_size + 7 * kClipArea) *
      sizeof(float);
  JXL_ASSIGN_OR_RETURN(auto fmem,
                       AlignedMemory::Create(memory_manager, fmem_bytes));
  float* JXL_RESTRICT scratch_space =
      fmem.address<float>() + 3 * AcStrategy::kMaxCoeffArea;
  float* JXL_RESTRICT clip_work =
      fmem.address<float>() + 5 * AcStrategy::kMaxCoeffArea + dct_scratch_size;
  float* JXL_RESTRICT clip_shift = clip_work + 5 * kClipArea;
  float* JXL_RESTRICT shifted_px = clip_shift + kClipArea;
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
    const bool has_rdoq_field =
        (tcm != nullptr &&
         enc_state->rdoq_level_field.xsize() >=
             block_group_rect.x0() + xsize_blocks &&
         enc_state->rdoq_level_field.ysize() >=
             block_group_rect.y0() + ysize_blocks);

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
      const int32_t* JXL_RESTRICT row_clip_quant_ac =
          block_group_rect.ConstRow(clip_quant_field, by);
      const int8_t* JXL_RESTRICT row_rdoq_lvl =
          has_rdoq_field
              ? block_group_rect.ConstRow(enc_state->rdoq_level_field, by)
              : nullptr;
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
          (dc != nullptr) ? block_group_rect.PlaneRow(dc, 0, by) : nullptr,
          (dc != nullptr) ? block_group_rect.PlaneRow(dc, 1, by) : nullptr,
          (dc != nullptr) ? block_group_rect.PlaneRow(dc, 2, by) : nullptr,
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
          const bool has_clip_shift = ContinueSaturatedBlock(
              enc_state, acs, cov_x, cov_y, xblocks, yblocks,
              row_clip_quant_ac[bx], opsin_rows, opsin_stride, bx * kBlockDim,
              frame_px0, frame_py0, scratch_space, clip_work, clip_shift);
          if (has_clip_shift) {
            const size_t pw = cov_x * kBlockDim;
            const size_t ph = cov_y * kBlockDim;
            TransformFromPixels(acs.Strategy(), opsin_rows[0] + bx * kBlockDim,
                                opsin_stride, coeffs_in, scratch_space);
            for (size_t c : {1, 2}) {
              for (size_t iy = 0; iy < ph; ++iy) {
                const float* JXL_RESTRICT row =
                    opsin_rows[c] + bx * kBlockDim + iy * opsin_stride;
                const float* JXL_RESTRICT sh = clip_shift + iy * pw;
                float* JXL_RESTRICT dst = shifted_px + iy * pw;
                for (size_t ix = 0; ix < pw; ++ix) {
                  dst[ix] = row[ix] + sh[ix];
                }
              }
              TransformFromPixels(acs.Strategy(), shifted_px, pw,
                                  coeffs_in + c * size, scratch_space);
            }
          } else {
            for (size_t c : {0, 1, 2}) {
              TransformFromPixels(acs.Strategy(),
                                  opsin_rows[c] + bx * kBlockDim, opsin_stride,
                                  coeffs_in + c * size, scratch_space);
            }
          }
          if (dc != nullptr) {
            DCFromLowestFrequencies(acs.Strategy(), coeffs_in + size,
                                    dc_rows[1] + bx, dc_stride, scratch_space);
          }

          const size_t covered_blocks = cov_x * cov_y;
          const size_t log2_covered_blocks =
              Num0BitsBelowLS1Bit_Nonzero(covered_blocks);
          const size_t ord = kStrategyOrder[acs.RawStrategy()];
          const int32_t pred_nz_y =
              (tcm != nullptr)
                  ? PredictFromTopAndLeft(row_nzeros_top[1], row_nzeros[1], bx,
                                          32)
                  : 32;
          const int8_t rdoq_lvl =
              (row_rdoq_lvl != nullptr) ? row_rdoq_lvl[bx] : 0;
          int32_t final_nz_y = 0;

          QuantizeRoundtripYBlockAC(
              enc_state, size, enc_state->shared.quantizer, error_diffusion,
              acs.Strategy(), xblocks, yblocks, zero_thres, eob_scale,
              kDefaultQuantBias, quant_ac, coeffs_in, quantized, tcm, ord,
              covered_blocks, log2_covered_blocks, pred_nz_y, &final_nz_y,
              rdoq_choices, few_colors, rdoq_lvl);
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
                            rdoq_choices, few_colors, rdoq_lvl);
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
            if (dc != nullptr) {
              DCFromLowestFrequencies(acs.Strategy(), coeffs_in + c * size,
                                      dc_rows[c] + bx, dc_stride,
                                      scratch_space);
            }
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
  constexpr size_t kClipArea = 16 * kDCTBlockSize;
  const size_t fmem_bytes =
      (2 * AcStrategy::kMaxCoeffArea + dct_scratch_size + 7 * kClipArea) *
      sizeof(float);
  JXL_ASSIGN_OR_RETURN(auto fmem,
                       AlignedMemory::Create(memory_manager, fmem_bytes));
  HWY_ALIGN float* JXL_RESTRICT dct_buf = fmem.address<float>();
  HWY_ALIGN float* JXL_RESTRICT px_buf = dct_buf + AcStrategy::kMaxCoeffArea;
  float* JXL_RESTRICT scratch_space = px_buf + AcStrategy::kMaxCoeffArea;
  float* JXL_RESTRICT clip_work = scratch_space + dct_scratch_size;
  float* JXL_RESTRICT clip_shift = clip_work + 5 * kClipArea;
  float* JXL_RESTRICT shifted_px = clip_shift + kClipArea;

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

  const size_t cfl_tiles_x =
      DivCeil(shared.frame_dim.xsize_blocks, kColorTileDimInBlocks);
  const bool has_quant_dc =
      shared.quant_dc.xsize() >= gx0 + gw_blocks &&
      shared.quant_dc.ysize() >= gy0 + gh_blocks;
  for (size_t by = 0; by < gh_blocks; ++by) {
    const int32_t* JXL_RESTRICT row_quant_ac =
        block_group_rect.ConstRow(shared.raw_quant_field, by);
    const uint8_t* JXL_RESTRICT row_qdc =
        has_quant_dc ? block_group_rect.ConstRow(shared.quant_dc, by) : nullptr;
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
      tr.dc_idx = (row_qdc != nullptr) ? row_qdc[bx] : 0;
      tr.tile_idx = static_cast<uint32_t>(
          ((gy0 + by) / tile_blocks) * tiles_x + (gx0 + bx) / tile_blocks);
      tr.cfl_tile_idx = static_cast<uint32_t>(
          ((gy0 + by) / kColorTileDimInBlocks) * cfl_tiles_x +
          (gx0 + bx) / kColorTileDimInBlocks);
      tr.base_quant_ac = row_quant_ac[bx];
      tr.cur_quant_ac = tr.base_quant_ac;
      tr.cur_ytox_q = row_cmap_x[tx];
      tr.cur_ytob_q = row_cmap_b[tx];
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
      const bool has_clip_shift = ContinueSaturatedBlock(
          enc_state, acs, cov_x, cov_y, xblocks, yblocks, tr.base_quant_ac,
          opsin_rows, opsin_stride, bx * kBlockDim, (gx0 + bx) * kBlockDim,
          (gy0 + by) * kBlockDim, scratch_space, clip_work, clip_shift);
      if (has_clip_shift) {
        const size_t pw = cov_x * kBlockDim;
        const size_t ph = cov_y * kBlockDim;
        const size_t lx0 = bx * kBlockDim;
        const size_t ly0 = by * kBlockDim;
        const size_t lw = std::min(pw_px - lx0, pw);
        const size_t lh = std::min(ph_px - ly0, ph);
        for (size_t c = 0; c < 3; ++c) {
          if (c == 0) {
            TransformFromPixels(acs.Strategy(), opsin_rows[0] + bx * kBlockDim,
                                opsin_stride, dct_buf, scratch_space);
          } else {
            for (size_t iy = 0; iy < ph; ++iy) {
              const float* JXL_RESTRICT row =
                  opsin_rows[c] + bx * kBlockDim + iy * opsin_stride;
              const float* JXL_RESTRICT sh = clip_shift + iy * pw;
              float* JXL_RESTRICT dst = shifted_px + iy * pw;
              for (size_t ix = 0; ix < pw; ++ix) {
                dst[ix] = row[ix] + sh[ix];
              }
            }
            float* JXL_RESTRICT base_p =
                cache->base_pre_gab[c].data() + ly0 * pw_px + lx0;
            for (size_t iy = 0; iy < lh; ++iy) {
              const float* JXL_RESTRICT sh = clip_shift + iy * pw;
              for (size_t ix = 0; ix < lw; ++ix) {
                base_p[iy * pw_px + ix] += sh[ix];
              }
            }
            TransformFromPixels(acs.Strategy(), shifted_px, pw, dct_buf,
                                scratch_space);
          }
          tr.src_dct[c].assign(dct_buf, dct_buf + size);
          DCFromLowestFrequencies(acs.Strategy(), dct_buf,
                                  dc_src[c].data() + by * gw_blocks + bx,
                                  gw_blocks, scratch_space);
        }
      } else {
        for (size_t c = 0; c < 3; ++c) {
          TransformFromPixels(acs.Strategy(), opsin_rows[c] + bx * kBlockDim,
                              opsin_stride, dct_buf, scratch_space);
          tr.src_dct[c].assign(dct_buf, dct_buf + size);
          DCFromLowestFrequencies(acs.Strategy(), dct_buf,
                                  dc_src[c].data() + by * gw_blocks + bx,
                                  gw_blocks, scratch_space);
        }
      }
      cache->transforms.push_back(std::move(tr));
    }
  }

  if (shared.dc_storage.xsize() >= gx0 + gw_blocks &&
      shared.dc_storage.ysize() >= gy0 + gh_blocks) {
    for (size_t c = 0; c < 3; ++c) {
      for (size_t by = 0; by < gh_blocks; ++by) {
        const float* JXL_RESTRICT dec_dc_row =
            block_group_rect.ConstPlaneRow(shared.dc_storage, c, by);
        for (size_t bx = 0; bx < gw_blocks; ++bx) {
          const size_t i = by * gw_blocks + bx;
          dc_err[c][i] = dec_dc_row[bx] - dc_src[c][i];
        }
      }
    }
  } else {
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

    const int8_t eff_rdoq_level = use_rdoq ? tr.cur_rdoq_level : 0;
    int hit_slot = -1;
    for (size_t s = 0; s < AcTransformCache::kNumSlots; ++s) {
      const AcBlockEvalSlot& slot = tr.slots[s];
      if (slot.quant_ac == quant_ac &&
          slot.use_rdoq == static_cast<uint8_t>(use_rdoq) &&
          slot.rdoq_level == eff_rdoq_level &&
          slot.ytox_q == tr.cur_ytox_q &&
          slot.ytob_q == tr.cur_ytob_q &&
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
      slot.rdoq_level = eff_rdoq_level;
      slot.ytox_q = tr.cur_ytox_q;
      slot.ytob_q = tr.cur_ytob_q;
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
          rdoq_ptr, tr.few_colors, eff_rdoq_level);

      constexpr HWY_CAPPED(float, kDCTBlockSize) df;
      const auto x_factor =
          Set(df, shared.cmap.base().YtoXRatio(tr.cur_ytox_q));
      const auto b_factor =
          Set(df, shared.cmap.base().YtoBRatio(tr.cur_ytob_q));
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
            rdoq_ptr, tr.few_colors, eff_rdoq_level);
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
            tcm->block_ctx_map.Context(tr.dc_idx, quant_ac, ord, c);
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

Status EvalAcsBoxCandidateImpl(const PassesEncoderState* enc_state,
                               const Image3F& opsin, const Rect& rect,
                               size_t tiles_x, size_t tile_blocks,
                               const AcGroupBlockCache& cache, size_t bx0,
                               size_t by0, size_t bw, size_t bh,
                               size_t num_specs, const AcsSubSpec* specs,
                               std::vector<AcTransformCache>* out_trs,
                               std::vector<float> out_base_patch[3],
                               float* out_local_ac_bits,
                               Image3F* out_local_pre_gab,
                               AcsEvalWorkspace* eval_ws) {
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  const PassesSharedState& shared = enc_state->shared;
  const AcTokenCostModel* JXL_RESTRICT tcm = enc_state->token_cost_model.get();
  const size_t gx0 = cache.gx0;
  const size_t gy0 = cache.gy0;
  const size_t gw_blocks = cache.gw_blocks;
  const size_t gh_blocks = cache.gh_blocks;
  const size_t pw_px = cache.pw_px;
  const size_t ph_px = cache.ph_px;
  const Rect block_group_rect = shared.frame_dim.BlockGroupRect(cache.group_idx);
  const Rect group_rect =
      shared.frame_dim.GroupRect(cache.group_idx).Translate(rect.x0(), rect.y0());
  const size_t opsin_stride = static_cast<size_t>(opsin.PixelsPerRow());

  constexpr size_t kMaxTrArea = 64 * kDCTBlockSize;
  constexpr size_t kClipArea = 16 * kDCTBlockSize;
  constexpr size_t kNzStride = 33;
  constexpr size_t kNzPlane = kNzStride * kNzStride;
  constexpr size_t kNzInts = 3 * kNzPlane;
  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;
  const size_t quant_ints = 3 * kMaxTrArea;
  const size_t rdoq_choice_bytes = kMaxRdoqWindow * kBandStride;
  const size_t mem_bytes =
      (quant_ints + kNzInts) * sizeof(int32_t) + rdoq_choice_bytes;
  const size_t fmem_bytes =
      (9 * kMaxTrArea + 7 * kClipArea + dct_scratch_size + 384) * sizeof(float);

  AlignedMemory local_mem;
  AlignedMemory local_fmem;
  int32_t* mem_base = nullptr;
  float* fmem_base = nullptr;
  if (eval_ws != nullptr) {
    if (eval_ws->cap_mem_bytes < mem_bytes) {
      JXL_ASSIGN_OR_RETURN(eval_ws->mem,
                           AlignedMemory::Create(memory_manager, mem_bytes));
      eval_ws->cap_mem_bytes = mem_bytes;
    }
    if (eval_ws->cap_fmem_bytes < fmem_bytes) {
      JXL_ASSIGN_OR_RETURN(eval_ws->fmem,
                           AlignedMemory::Create(memory_manager, fmem_bytes));
      eval_ws->cap_fmem_bytes = fmem_bytes;
    }
    mem_base = eval_ws->mem.address<int32_t>();
    fmem_base = eval_ws->fmem.address<float>();
  } else {
    JXL_ASSIGN_OR_RETURN(local_mem,
                         AlignedMemory::Create(memory_manager, mem_bytes));
    JXL_ASSIGN_OR_RETURN(local_fmem,
                         AlignedMemory::Create(memory_manager, fmem_bytes));
    mem_base = local_mem.address<int32_t>();
    fmem_base = local_fmem.address<float>();
  }

  HWY_ALIGN int32_t* JXL_RESTRICT quantized = mem_base;
  int32_t* JXL_RESTRICT local_nz = quantized + quant_ints;
  uint8_t* JXL_RESTRICT rdoq_choices =
      reinterpret_cast<uint8_t*>(local_nz + kNzInts);
  HWY_ALIGN float* JXL_RESTRICT coeffs_in = fmem_base;
  HWY_ALIGN float* JXL_RESTRICT err_dct = coeffs_in + 3 * kMaxTrArea;
  HWY_ALIGN float* JXL_RESTRICT px_buf = err_dct + kMaxTrArea;
  HWY_ALIGN float* JXL_RESTRICT base_sub = px_buf + kMaxTrArea;
  HWY_ALIGN float* JXL_RESTRICT src_y_copy = base_sub + 3 * kMaxTrArea;
  float* JXL_RESTRICT scratch_space = src_y_copy + kMaxTrArea;
  float* JXL_RESTRICT clip_work = scratch_space + dct_scratch_size;
  float* JXL_RESTRICT clip_shift = clip_work + 5 * kClipArea;
  float* JXL_RESTRICT shifted_px = clip_shift + kClipArea;
  float* JXL_RESTRICT dc_src_sub = shifted_px + kClipArea;
  float* JXL_RESTRICT dc_err_sub = dc_src_sub + 192;

  const size_t box_x0_px = bx0 * kBlockDim;
  const size_t box_y0_px = by0 * kBlockDim;
  const size_t box_w_px =
      (box_x0_px < pw_px) ? std::min(pw_px - box_x0_px, bw * kBlockDim) : 0;
  const size_t box_h_px =
      (box_y0_px < ph_px) ? std::min(ph_px - box_y0_px, bh * kBlockDim) : 0;
  if (out_base_patch != nullptr) {
    for (size_t c = 0; c < 3; ++c) {
      out_base_patch[c].assign(box_w_px * box_h_px, 0.0f);
    }
  }
  if (out_trs != nullptr) {
    out_trs->clear();
    out_trs->reserve(num_specs);
  }
  if (out_local_ac_bits != nullptr) {
    *out_local_ac_bits = 0.0f;
  }

  memset(local_nz, 0, kNzInts * sizeof(int32_t));
  if (out_local_pre_gab != nullptr && tcm != nullptr &&
      tcm->pass1_nzeros.xsize() >= gx0 + gw_blocks &&
      tcm->pass1_nzeros.ysize() >= gy0 + gh_blocks) {
    for (size_t c = 0; c < 3; ++c) {
      int32_t* JXL_RESTRICT nz_c = local_nz + c * kNzPlane;
      if (by0 > 0) {
        const int32_t* r_top =
            tcm->pass1_nzeros.ConstPlaneRow(c, gy0 + by0 - 1) + gx0;
        for (size_t ix = 0; ix < bw; ++ix) {
          nz_c[1 + ix] = r_top[bx0 + ix];
        }
      }
      if (bx0 > 0) {
        for (size_t iy = 0; iy < bh; ++iy) {
          nz_c[(1 + iy) * kNzStride] =
              tcm->pass1_nzeros.ConstPlaneRow(c, gy0 + by0 + iy)[gx0 + bx0 - 1];
        }
      }
    }
  }

  const float d_target = enc_state->cparams.butteraugli_distance;
  const float dz_w =
      Clamp1((d_target - 0.35f) / (d_target + 0.55f) * 1.55f, 0.0f, 1.0f);
  const float eob_scale =
      (enc_state->cparams.speed_tier <= SpeedTier::kHare)
          ? Clamp1((d_target - 0.28f) / (d_target + 0.50f) * 1.45f, 0.0f, 1.0f)
          : 0.0f;
  const bool error_diffusion =
      enc_state->cparams.speed_tier <= SpeedTier::kSquirrel;
  const size_t cfl_tiles_x =
      DivCeil(shared.frame_dim.xsize_blocks, kColorTileDimInBlocks);
  const bool has_quant_dc =
      shared.quant_dc.xsize() >= gx0 + gw_blocks &&
      shared.quant_dc.ysize() >= gy0 + gh_blocks;
  const bool has_base_qf =
      enc_state->base_raw_quant_field.xsize() >= gx0 + gw_blocks &&
      enc_state->base_raw_quant_field.ysize() >= gy0 + gh_blocks;

  for (size_t k = 0; k < num_specs; ++k) {
    const AcsSubSpec& sp = specs[k];
    const size_t sbx = sp.bx;
    const size_t sby = sp.by;
    const AcStrategy acs = AcStrategy::FromRawStrategy(sp.acs_raw);
    const size_t cov_x = acs.covered_blocks_x();
    const size_t cov_y = acs.covered_blocks_y();
    size_t xblocks = cov_x;
    size_t yblocks = cov_y;
    CoefficientLayout(&yblocks, &xblocks);
    const size_t covered_blocks = cov_x * cov_y;
    JXL_ENSURE(covered_blocks <= 64);
    const size_t size = kDCTBlockSize * xblocks * yblocks;
    const size_t cw_px = cov_x * kBlockDim;
    const size_t ch_px = cov_y * kBlockDim;
    const size_t lx0 = sbx * kBlockDim;
    const size_t ly0 = sby * kBlockDim;
    const size_t lw = (lx0 < pw_px) ? std::min(pw_px - lx0, cw_px) : 0;
    const size_t lh = (ly0 < ph_px) ? std::min(ph_px - ly0, ch_px) : 0;

    const float* JXL_RESTRICT opsin_rows[3] = {
        group_rect.ConstPlaneRow(opsin, 0, sby * kBlockDim),
        group_rect.ConstPlaneRow(opsin, 1, sby * kBlockDim),
        group_rect.ConstPlaneRow(opsin, 2, sby * kBlockDim),
    };
    const uint8_t dc_idx =
        has_quant_dc ? block_group_rect.ConstRow(shared.quant_dc, sby)[sbx] : 0;
    const bool few_colors =
        (eob_scale > 0.0f)
            ? BlockHasFewColors(enc_state->orig_opsin, opsin_rows, opsin_stride,
                                (gx0 + sbx) * kBlockDim,
                                (gy0 + sby) * kBlockDim, lx0, cov_x, cov_y)
            : false;
    const int32_t clip_q =
        has_base_qf
            ? block_group_rect.ConstRow(enc_state->base_raw_quant_field,
                                        sby)[sbx]
            : sp.quant_ac;
    const bool has_clip_shift = ContinueSaturatedBlock(
        enc_state, acs, cov_x, cov_y, xblocks, yblocks, clip_q, opsin_rows,
        opsin_stride, lx0, (gx0 + sbx) * kBlockDim, (gy0 + sby) * kBlockDim,
        scratch_space, clip_work, clip_shift);

    memset(dc_src_sub, 0, 192 * sizeof(float));
    memset(dc_err_sub, 0, 192 * sizeof(float));
    for (size_t c = 0; c < 3; ++c) {
      float* JXL_RESTRICT b_dst = base_sub + c * kMaxTrArea;
      float* JXL_RESTRICT dc_src_c = dc_src_sub + c * 64;
      float* JXL_RESTRICT dc_err_c = dc_err_sub + c * 64;
      for (size_t iy = 0; iy < lh; ++iy) {
        const float* JXL_RESTRICT src_r =
            opsin_rows[c] + lx0 + iy * opsin_stride;
        memcpy(b_dst + iy * cw_px, src_r, lw * sizeof(float));
      }
      if (has_clip_shift && c > 0) {
        for (size_t iy = 0; iy < ch_px; ++iy) {
          const float* JXL_RESTRICT row =
              opsin_rows[c] + lx0 + iy * opsin_stride;
          const float* JXL_RESTRICT sh = clip_shift + iy * cw_px;
          float* JXL_RESTRICT dst = shifted_px + iy * cw_px;
          for (size_t ix = 0; ix < cw_px; ++ix) {
            dst[ix] = row[ix] + sh[ix];
          }
        }
        for (size_t iy = 0; iy < lh; ++iy) {
          const float* JXL_RESTRICT sh = clip_shift + iy * cw_px;
          for (size_t ix = 0; ix < lw; ++ix) {
            b_dst[iy * cw_px + ix] += sh[ix];
          }
        }
        TransformFromPixels(acs.Strategy(), shifted_px, cw_px,
                            coeffs_in + c * size, scratch_space);
      } else {
        TransformFromPixels(acs.Strategy(), opsin_rows[c] + lx0, opsin_stride,
                            coeffs_in + c * size, scratch_space);
      }
      DCFromLowestFrequencies(acs.Strategy(), coeffs_in + c * size,
                              dc_src_c, cov_x, scratch_space);
      for (size_t iy = 0; iy < cov_y; ++iy) {
        const float* JXL_RESTRICT dec_dc_r =
            block_group_rect.ConstPlaneRow(shared.dc_storage, c, sby + iy);
        for (size_t ix = 0; ix < cov_x; ++ix) {
          dc_err_c[iy * cov_x + ix] =
              dec_dc_r[sbx + ix] - dc_src_c[iy * cov_x + ix];
        }
      }
      memset(err_dct, 0, size * sizeof(float));
      LowestFrequenciesFromDC(acs.Strategy(), dc_err_c, cov_x, err_dct,
                              scratch_space);
      TransformToPixels(acs.Strategy(), err_dct, px_buf, cw_px, scratch_space);
      for (size_t iy = 0; iy < lh; ++iy) {
        for (size_t ix = 0; ix < lw; ++ix) {
          b_dst[iy * cw_px + ix] += px_buf[iy * cw_px + ix];
        }
      }
      if (out_base_patch != nullptr) {
        const size_t ox = (sbx - bx0) * kBlockDim;
        const size_t oy = (sby - by0) * kBlockDim;
        for (size_t iy = 0; iy < lh && oy + iy < box_h_px; ++iy) {
          memcpy(out_base_patch[c].data() + (oy + iy) * box_w_px + ox,
                 b_dst + iy * cw_px,
                 std::min(lw, box_w_px - ox) * sizeof(float));
        }
      }
    }

    if (out_trs != nullptr) {
      AcTransformCache tr;
      tr.bx = static_cast<uint8_t>(sbx);
      tr.by = static_cast<uint8_t>(sby);
      tr.cov_x = static_cast<uint8_t>(cov_x);
      tr.cov_y = static_cast<uint8_t>(cov_y);
      tr.xblocks = static_cast<uint8_t>(xblocks);
      tr.yblocks = static_cast<uint8_t>(yblocks);
      tr.acs_raw = sp.acs_raw;
      tr.dc_idx = dc_idx;
      tr.few_colors = few_colors;
      tr.acs_changed = true;
      tr.cur_rdoq_level = sp.rdoq_lvl;
      tr.cur_ytox_q = sp.ytox_q;
      tr.cur_ytob_q = sp.ytob_q;
      tr.base_quant_ac = sp.quant_ac;
      tr.cur_quant_ac = sp.quant_ac;
      tr.act_bits = sp.act_bits;
      tr.tile_idx = static_cast<uint32_t>(
          ((gy0 + sby) / tile_blocks) * tiles_x + (gx0 + sbx) / tile_blocks);
      tr.cfl_tile_idx = static_cast<uint32_t>(
          ((gy0 + sby) / kColorTileDimInBlocks) * cfl_tiles_x +
          (gx0 + sbx) / kColorTileDimInBlocks);
      for (size_t c = 0; c < 3; ++c) {
        for (size_t s = 0; s < 4; ++s) {
          tr.base_band_q[c][s] = sp.band_q[c][s];
        }
        tr.src_dct[c].assign(coeffs_in + c * size, coeffs_in + (c + 1) * size);
      }
      tr.cur_slot = static_cast<uint8_t>(AcTransformCache::kNumSlots);
      out_trs->push_back(std::move(tr));
    }

    if (out_local_pre_gab != nullptr && tcm != nullptr) {
      const size_t log2_covered_blocks =
          Num0BitsBelowLS1Bit_Nonzero(covered_blocks);
      const size_t ord = kStrategyOrder[sp.acs_raw];
      const int32_t quant_ac = Clamp1<int32_t>(sp.quant_ac, 1, 255);
      float band_q[3][4];
      for (size_t c = 0; c < 3; ++c) {
        for (size_t s = 0; s < 4; ++s) {
          band_q[c][s] = sp.band_q[c][s];
        }
      }
      uint32_t bq10_bits = 0;
      memcpy(&bq10_bits, &band_q[1][0], sizeof(bq10_bits));
      bq10_bits = (bq10_bits & ~31u) | sp.act_bits;
      memcpy(&band_q[1][0], &bq10_bits, sizeof(bq10_bits));
      const float q_block = shared.quantizer.Scale() * quant_ac;
      float zero_thres[3][8];
      ComputeBlockThresholds(acs.Strategy(), xblocks, yblocks, q_block, band_q,
                             d_target, dz_w, eob_scale, zero_thres);

      memcpy(src_y_copy, coeffs_in + size, size * sizeof(float));

      const size_t ly = 1 + (sby - by0);
      const size_t lx = 1 + (sbx - bx0);
      int32_t pred_nz[3];
      for (size_t c = 0; c < 3; ++c) {
        const int32_t* JXL_RESTRICT nz_c = local_nz + c * kNzPlane;
        if (sbx == 0 && sby == 0) {
          pred_nz[c] = 32;
        } else if (sby == 0) {
          pred_nz[c] = nz_c[ly * kNzStride + (lx - 1)];
        } else if (sbx == 0) {
          pred_nz[c] = nz_c[(ly - 1) * kNzStride + lx];
        } else {
          pred_nz[c] = (nz_c[ly * kNzStride + (lx - 1)] +
                        nz_c[(ly - 1) * kNzStride + lx] + 1) >>
                       1;
        }
      }

      int32_t final_nz[3] = {0, 0, 0};
      QuantizeRoundtripYBlockAC(
          const_cast<PassesEncoderState*>(enc_state), size, shared.quantizer,
          error_diffusion, acs.Strategy(), xblocks, yblocks, zero_thres,
          eob_scale, kDefaultQuantBias, quant_ac, coeffs_in, quantized, tcm,
          ord, covered_blocks, log2_covered_blocks, pred_nz[1], &final_nz[1],
          rdoq_choices, few_colors, sp.rdoq_lvl);

      constexpr HWY_CAPPED(float, kDCTBlockSize) df;
      const auto x_factor = Set(df, shared.cmap.base().YtoXRatio(sp.ytox_q));
      const auto b_factor = Set(df, shared.cmap.base().YtoBRatio(sp.ytob_q));
      for (size_t i = 0; i < size; i += Lanes(df)) {
        const auto in_x = Load(df, coeffs_in + i);
        const auto in_y = Load(df, coeffs_in + size + i);
        const auto in_b = Load(df, coeffs_in + 2 * size + i);
        Store(NegMulAdd(x_factor, in_y, in_x), df, coeffs_in + i);
        Store(NegMulAdd(b_factor, in_y, in_b), df, coeffs_in + 2 * size + i);
      }

      for (size_t c : {0, 2}) {
        QuantizeBlockAC(
            shared.quantizer, error_diffusion, c,
            c == 0 ? enc_state->x_qm_multiplier : enc_state->b_qm_multiplier,
            acs.Strategy(), xblocks, yblocks, zero_thres[c], eob_scale,
            coeffs_in + c * size, quant_ac, quantized + c * size, tcm, ord,
            covered_blocks, log2_covered_blocks, pred_nz[c], &final_nz[c],
            rdoq_choices, few_colors, sp.rdoq_lvl);
      }

      float blk_bits = 0.0f;
      for (size_t c : {1, 0, 2}) {
        const coeff_order_t* JXL_RESTRICT order =
            &tcm->coeff_orders[CoeffOrderOffset(ord, c)];
        const size_t block_ctx =
            tcm->block_ctx_map.Context(dc_idx, quant_ac, ord, c);
        const uint32_t nzero_ctx =
            tcm->block_ctx_map.NonZeroContext(pred_nz[c], block_ctx);
        const int32_t nz = final_nz[c];
        blk_bits +=
            tcm->ValBits(tcm->cluster_of_ctx[nzero_ctx], static_cast<uint32_t>(nz));
        const size_t histo_offset =
            tcm->block_ctx_map.ZeroDensityContextsOffset(block_ctx);
        size_t prev = (nz > static_cast<int32_t>(size / 16)) ? 0 : 1;
        size_t rem_nz = static_cast<size_t>(nz);
        for (size_t i = covered_blocks; i < size && rem_nz != 0; ++i) {
          const int32_t coeff = quantized[c * size + order[i]];
          const size_t ctx =
              histo_offset + ZeroDensityContext(rem_nz, i, covered_blocks,
                                                log2_covered_blocks, prev);
          blk_bits += tcm->ValBits(tcm->cluster_of_ctx[ctx], PackSigned(coeff));
          prev = (coeff != 0) ? 1 : 0;
          rem_nz -= prev;
        }
        const int32_t snz = static_cast<int32_t>(
            (final_nz[c] + covered_blocks - 1) >> log2_covered_blocks);
        int32_t* JXL_RESTRICT nz_c = local_nz + c * kNzPlane;
        for (size_t iy = 0; iy < cov_y; ++iy) {
          for (size_t ix = 0; ix < cov_x; ++ix) {
            nz_c[(ly + iy) * kNzStride + (lx + ix)] = snz;
          }
        }
      }
      if (out_local_ac_bits != nullptr) {
        *out_local_ac_bits += blk_bits;
      }

      for (size_t i = 0; i < size; ++i) {
        err_dct[i] = coeffs_in[size + i] - src_y_copy[i];
      }
      for (size_t y = 0; y < yblocks; ++y) {
        for (size_t x = 0; x < xblocks; ++x) {
          err_dct[y * kBlockDim * xblocks + x] = 0.0f;
        }
      }
      TransformToPixels(acs.Strategy(), err_dct, px_buf, cw_px, scratch_space);
      for (size_t y = 0; y < lh; ++y) {
        float* JXL_RESTRICT dst =
            out_local_pre_gab->PlaneRow(1, cache.py0 + ly0 + y) + cache.px0 + lx0;
        const float* JXL_RESTRICT b_src =
            base_sub + kMaxTrArea + y * cw_px;
        for (size_t x = 0; x < lw; ++x) {
          dst[x] = b_src[x] + px_buf[y * cw_px + x];
        }
      }

      constexpr HWY_CAPPED(int32_t, kDCTBlockSize) di;
      for (size_t c : {0, 2}) {
        const float qm_mul =
            (c == 0) ? enc_state->x_qm_multiplier : enc_state->b_qm_multiplier;
        const auto inv_qv =
            Set(df, shared.quantizer.inv_quant_ac(quant_ac) / qm_mul);
        const float* JXL_RESTRICT dequant_matrix =
            shared.quantizer.DequantMatrix(acs.Strategy(), c);
        for (size_t i = 0; i < size; i += Lanes(df)) {
          const auto oq = Load(di, quantized + c * size + i);
          const auto adj = AdjustQuantBias(di, c, oq, kDefaultQuantBias);
          const auto dm = Load(df, dequant_matrix + i);
          const auto deq_cc = Mul(Mul(adj, dm), inv_qv);
          const auto res = Load(df, coeffs_in + c * size + i);
          Store(Sub(deq_cc, res), df, err_dct + i);
        }
        for (size_t y = 0; y < yblocks; ++y) {
          for (size_t x = 0; x < xblocks; ++x) {
            err_dct[y * kBlockDim * xblocks + x] = 0.0f;
          }
        }
        TransformToPixels(acs.Strategy(), err_dct, px_buf, cw_px, scratch_space);
        for (size_t y = 0; y < lh; ++y) {
          float* JXL_RESTRICT dst =
              out_local_pre_gab->PlaneRow(c, cache.py0 + ly0 + y) + cache.px0 + lx0;
          const float* JXL_RESTRICT b_src =
              base_sub + c * kMaxTrArea + y * cw_px;
          for (size_t x = 0; x < lw; ++x) {
            dst[x] = b_src[x] + px_buf[y * cw_px + x];
          }
        }
      }
    }
  }
  return true;
}

void ApplyTransformDcDeltaImpl(AcStrategyType strategy, size_t cov_x,
                               size_t cov_y, const float* JXL_RESTRICT dc_delta,
                               size_t dc_stride, float* JXL_RESTRICT dct_buf,
                               float* JXL_RESTRICT px_buf,
                               float* JXL_RESTRICT scratch_space) {
  size_t xblocks = cov_x;
  size_t yblocks = cov_y;
  CoefficientLayout(&yblocks, &xblocks);
  memset(dct_buf, 0, kDCTBlockSize * xblocks * yblocks * sizeof(float));
  LowestFrequenciesFromDC(strategy, dc_delta, dc_stride, dct_buf,
                          scratch_space);
  TransformToPixels(strategy, dct_buf, px_buf, cov_x * kBlockDim,
                    scratch_space);
}

using DF8 = HWY_CAPPED(float, 8);

JXL_INLINE Vec<DF8> EpfWeightVec(Vec<DF8> sad, Vec<DF8> inv_sigma) {
  return ZeroIfNegative(MulAdd(sad, inv_sigma, Set(DF8(), 1.0f)));
}

static inline void ExpandMirrorRange(size_t a0, size_t a1, size_t r, size_t n,
                                     size_t* b0, size_t* b1) {
  if (n <= 2 * r && (a0 < r || a1 + r > n)) {
    *b0 = 0;
    *b1 = n;
    return;
  }
  size_t lo = (a0 > r) ? (a0 - r) : 0;
  size_t hi = std::min(n, a1 + r);
  if (a0 < r) hi = std::max(hi, std::min(n, r - a0));
  if (a1 + r > n) {
    const size_t over = a1 + r - n;
    lo = std::min(lo, (n > over) ? (n - over) : 0);
  }
  *b0 = lo;
  *b1 = hi;
}

Status ReconstructXybPostFilterImpl(
    JxlMemoryManager* memory_manager, const LoopFilter& lf,
    const Image3F& pre_gab, const ImageF& sigma, size_t xsize, size_t ysize,
    size_t ox0, size_t oy0, size_t ox1, size_t oy1, EpfWorkspace* ws,
    const float* out_ptrs[3], size_t* out_stride, size_t* out_y0,
    size_t* out_min_px) {
  size_t stage_kinds[3] = {};
  size_t stage_rad[3] = {};
  size_t num_stages = 0;
  if (lf.epf_iters >= 3) {
    stage_kinds[num_stages] = 0;
    stage_rad[num_stages++] = 3;
  }
  if (lf.epf_iters >= 1) {
    stage_kinds[num_stages] = 1;
    stage_rad[num_stages++] = 2;
  }
  if (lf.epf_iters >= 2) {
    stage_kinds[num_stages] = 2;
    stage_rad[num_stages++] = 1;
  }

  size_t sy0[4] = {};
  size_t sy1[4] = {};
  size_t bx0[4] = {};
  size_t bx1[4] = {};
  sy0[num_stages] = oy0;
  sy1[num_stages] = oy1;
  bx0[num_stages] = ox0 / kBlockDim;
  bx1[num_stages] = DivCeil(ox1, kBlockDim);
  for (size_t k = num_stages; k > 0; --k) {
    const size_t r = stage_rad[k - 1];
    ExpandMirrorRange(sy0[k], sy1[k], r, ysize, &sy0[k - 1], &sy1[k - 1]);
    size_t need_x0 = 0;
    size_t need_x1 = 0;
    ExpandMirrorRange(bx0[k] * kBlockDim,
                      std::min(xsize, bx1[k] * kBlockDim), r, xsize, &need_x0,
                      &need_x1);
    bx0[k - 1] = need_x0 / kBlockDim;
    bx1[k - 1] = DivCeil(need_x1, kBlockDim);
  }

  const size_t min_bx = bx0[0];
  const size_t max_bx = bx1[0];
  const size_t num_bx = max_bx - min_bx;
  const size_t min_px = min_bx * kBlockDim;
  const size_t min_y = sy0[0];
  const size_t max_y = sy1[0];
  const size_t num_y = max_y - min_y;
  constexpr size_t kPadX = 16;
  const size_t stride = RoundUpTo(num_bx * kBlockDim + 2 * kPadX, 16);
  const size_t buf_floats = 3 * num_y * stride;
  const size_t total_floats = 2 * buf_floats;
  if (ws->cap_floats < total_floats) {
    JXL_ASSIGN_OR_RETURN(
        ws->mem,
        AlignedMemory::Create(memory_manager, total_floats * sizeof(float)));
    ws->cap_floats = total_floats;
  }
  float* JXL_RESTRICT buf_a = ws->mem.address<float>();
  float* JXL_RESTRICT buf_b = buf_a + buf_floats;
  auto row_origin = [num_y, min_y, stride](float* base, size_t c,
                                           size_t y) -> float* {
    return base + (c * num_y + (y - min_y)) * stride + kPadX;
  };

  // Stage G: Gaborish (or raw copy) into buf_a on [min_px, valid_x1) x [min_y, max_y).
  const size_t valid_x1 = std::min(xsize, max_bx * kBlockDim);
  const size_t gab_len = valid_x1 - min_px;
  float gab_w[3][3] = {};
  if (lf.gab) {
    const float raw_w1[3] = {lf.gab_x_weight1, lf.gab_y_weight1,
                             lf.gab_b_weight1};
    const float raw_w2[3] = {lf.gab_x_weight2, lf.gab_y_weight2,
                             lf.gab_b_weight2};
    for (size_t c = 0; c < 3; ++c) {
      const float mul = 1.0f / (1.0f + 4.0f * (raw_w1[c] + raw_w2[c]));
      gab_w[c][0] = mul;
      gab_w[c][1] = raw_w1[c] * mul;
      gab_w[c][2] = raw_w2[c] * mul;
    }
  }
  for (size_t y = min_y; y < max_y; ++y) {
    const size_t yt = static_cast<size_t>(
        Mirror(static_cast<int64_t>(y) - 1, static_cast<int64_t>(ysize)));
    const size_t yb = static_cast<size_t>(
        Mirror(static_cast<int64_t>(y) + 1, static_cast<int64_t>(ysize)));
    for (size_t c = 0; c < 3; ++c) {
      float* JXL_RESTRICT dst = row_origin(buf_a, c, y);
      if (lf.gab) {
        const float* JXL_RESTRICT rt = pre_gab.ConstPlaneRow(c, yt);
        const float* JXL_RESTRICT rm = pre_gab.ConstPlaneRow(c, y);
        const float* JXL_RESTRICT rb = pre_gab.ConstPlaneRow(c, yb);
        const float w0 = gab_w[c][0];
        const float w1 = gab_w[c][1];
        const float w2 = gab_w[c][2];
        for (size_t i = 0; i < gab_len; ++i) {
          const size_t x = min_px + i;
          const size_t xl = (x > 0) ? (x - 1) : 0;
          const size_t xr = (x + 1 < xsize) ? (x + 1) : (xsize - 1);
          const float s0 = rm[x];
          const float s1 = rm[xl] + rm[xr] + rt[x] + rb[x];
          const float s2 = rt[xl] + rt[xr] + rb[xl] + rb[xr];
          dst[i] = s0 * w0 + s1 * w1 + s2 * w2;
        }
      } else {
        memcpy(dst, pre_gab.ConstPlaneRow(c, y) + min_px,
               gab_len * sizeof(float));
      }
    }
  }

  DF8 df;
  float* cur_in = buf_a;
  float* cur_out = buf_b;
  for (size_t st = 0; st < num_stages; ++st) {
    const size_t kind = stage_kinds[st];
    const size_t rad = stage_rad[st];
    const size_t in_y0 = sy0[st];
    const size_t in_y1 = sy1[st];
    const size_t out_y_lo = sy0[st + 1];
    const size_t out_y_hi = sy1[st + 1];
    const size_t in_bx_lo = bx0[st];
    const size_t in_bx_hi = bx1[st];
    const size_t out_bx_lo = bx0[st + 1];
    const size_t out_bx_hi = bx1[st + 1];

    // Fill left/right symmetric mirror borders on cur_in where touching frame edges.
    if (in_bx_lo == 0 || in_bx_hi * kBlockDim >= xsize) {
      for (size_t y = in_y0; y < in_y1; ++y) {
        for (size_t c = 0; c < 3; ++c) {
          float* JXL_RESTRICT ro = row_origin(cur_in, c, y);
          if (in_bx_lo == 0) {
            for (size_t ix = 1; ix <= kBlockDim; ++ix) {
              const size_t mx = static_cast<size_t>(Mirror(
                  -static_cast<int64_t>(ix), static_cast<int64_t>(xsize)));
              ro[-static_cast<ptrdiff_t>(ix)] = ro[mx];
            }
          }
          if (in_bx_hi * kBlockDim >= xsize) {
            const size_t max_fill = out_bx_hi * kBlockDim + kBlockDim;
            for (size_t x = xsize; x < max_fill; ++x) {
              const size_t mx = static_cast<size_t>(Mirror(
                  static_cast<int64_t>(x), static_cast<int64_t>(xsize)));
              ro[x - min_px] = ro[mx - min_px];
            }
          }
        }
      }
    }

    const float sm =
        (kind == 0)
            ? (lf.epf_pass0_sigma_scale * 1.65f)
            : (kind == 1 ? 1.65f : (lf.epf_pass2_sigma_scale * 1.65f));
    const float bsm = sm * lf.epf_border_sad_mul;
    HWY_ALIGN const float sad_mul_center[kBlockDim] = {bsm, sm, sm, sm,
                                                       sm,  sm, sm, bsm};
    HWY_ALIGN const float sad_mul_border[kBlockDim] = {bsm, bsm, bsm, bsm,
                                                       bsm, bsm, bsm, bsm};

    for (size_t y = out_y_lo; y < out_y_hi; ++y) {
      const float* JXL_RESTRICT row_sigma =
          sigma.ConstRow(y / kBlockDim + kSigmaPadding);
      const float* sad_mul =
          (y % kBlockDim == 0 || y % kBlockDim == kBlockDim - 1)
              ? sad_mul_border
              : sad_mul_center;
      const float* JXL_RESTRICT in_rows[3][7] = {};
      for (int iy = -static_cast<int>(rad); iy <= static_cast<int>(rad); ++iy) {
        const size_t my = static_cast<size_t>(Mirror(
            static_cast<int64_t>(y) + iy, static_cast<int64_t>(ysize)));
        for (size_t c = 0; c < 3; ++c) {
          in_rows[c][iy + static_cast<int>(rad)] = row_origin(cur_in, c, my);
        }
      }
      float* JXL_RESTRICT out_rows[3] = {
          row_origin(cur_out, 0, y),
          row_origin(cur_out, 1, y),
          row_origin(cur_out, 2, y),
      };

      for (size_t bx = out_bx_lo; bx < out_bx_hi; ++bx) {
        const ptrdiff_t dx0 =
            static_cast<ptrdiff_t>((bx - min_bx) * kBlockDim);
        const float sigma_val = row_sigma[bx + kSigmaPadding];
        if (sigma_val < kMinSigma) {
          for (size_t lane = 0; lane < kBlockDim; lane += Lanes(df)) {
            const ptrdiff_t dx = dx0 + static_cast<ptrdiff_t>(lane);
            for (size_t c = 0; c < 3; ++c) {
              Store(Load(df, in_rows[c][rad] + dx), df, out_rows[c] + dx);
            }
          }
          continue;
        }

        if (kind == 0) {
          constexpr std::array<int, 2> sads_off[12] = {
              {{-2, 0}}, {{-1, -1}}, {{-1, 0}}, {{-1, 1}}, {{0, -2}}, {{0, -1}},
              {{0, 1}},  {{0, 2}},   {{1, -1}}, {{1, 0}},  {{1, 1}},  {{2, 0}},
          };
          constexpr std::array<int, 2> plus_off[5] = {
              {{0, 0}}, {{-1, 0}}, {{0, -1}}, {{1, 0}}, {{0, 1}},
          };
          for (size_t lane = 0; lane < kBlockDim; lane += Lanes(df)) {
            const ptrdiff_t dx = dx0 + static_cast<ptrdiff_t>(lane);
            const auto inv_sigma =
                Mul(Set(df, sigma_val), Load(df, sad_mul + lane));
            Vec<DF8> sads[12];
            for (size_t i = 0; i < 12; ++i) sads[i] = Zero(df);
            for (size_t c = 0; c < 3; ++c) {
              const auto scale = Set(df, lf.epf_channel_scale[c]);
              for (size_t i = 0; i < 12; ++i) {
                auto sad = Zero(df);
                for (const auto& off : plus_off) {
                  const auto r11 =
                      LoadU(df, in_rows[c][3 + off[0]] + dx + off[1]);
                  const auto c11 =
                      LoadU(df, in_rows[c][3 + sads_off[i][0] + off[0]] + dx +
                                    sads_off[i][1] + off[1]);
                  sad = Add(sad, AbsDiff(r11, c11));
                }
                sads[i] = MulAdd(sad, scale, sads[i]);
              }
            }
            auto X = Load(df, in_rows[0][3] + dx);
            auto Y = Load(df, in_rows[1][3] + dx);
            auto B = Load(df, in_rows[2][3] + dx);
            auto w = Set(df, 1.0f);
            for (size_t i = 0; i < 12; ++i) {
              const int ry = 3 + sads_off[i][0];
              const ptrdiff_t cx_off = dx + sads_off[i][1];
              const auto weight = EpfWeightVec(sads[i], inv_sigma);
              w = Add(w, weight);
              X = MulAdd(weight, LoadU(df, in_rows[0][ry] + cx_off), X);
              Y = MulAdd(weight, LoadU(df, in_rows[1][ry] + cx_off), Y);
              B = MulAdd(weight, LoadU(df, in_rows[2][ry] + cx_off), B);
            }
#if JXL_HIGH_PRECISION
            const auto inv_w = Div(Set(df, 1.0f), w);
#else
            const auto inv_w = ApproximateReciprocal(w);
#endif
            Store(Mul(X, inv_w), df, out_rows[0] + dx);
            Store(Mul(Y, inv_w), df, out_rows[1] + dx);
            Store(Mul(B, inv_w), df, out_rows[2] + dx);
          }
        } else if (kind == 1) {
          for (size_t lane = 0; lane < kBlockDim; lane += Lanes(df)) {
            const ptrdiff_t dx = dx0 + static_cast<ptrdiff_t>(lane);
            const auto inv_sigma =
                Mul(Set(df, sigma_val), Load(df, sad_mul + lane));
            auto sad0 = Zero(df);
            auto sad1 = Zero(df);
            auto sad2 = Zero(df);
            auto sad3 = Zero(df);
            for (size_t c = 0; c < 3; ++c) {
              const auto p20 = Load(df, in_rows[c][0] + dx);
              const auto p21 = Load(df, in_rows[c][1] + dx);
              auto sad0c = AbsDiff(p20, p21);
              const auto p11 = LoadU(df, in_rows[c][1] + dx - 1);
              auto sad1c = AbsDiff(p11, p21);
              const auto p31 = LoadU(df, in_rows[c][1] + dx + 1);
              auto sad2c = AbsDiff(p31, p21);
              const auto p02 = LoadU(df, in_rows[c][2] + dx - 2);
              const auto p12 = LoadU(df, in_rows[c][2] + dx - 1);
              sad1c = Add(sad1c, AbsDiff(p02, p12));
              sad0c = Add(sad0c, AbsDiff(p11, p12));
              const auto p22 = Load(df, in_rows[c][2] + dx);
              auto t = AbsDiff(p12, p22);
              sad1c = Add(sad1c, t);
              sad2c = Add(sad2c, t);
              t = AbsDiff(p22, p21);
              auto sad3c = t;
              sad0c = Add(sad0c, t);
              const auto p32 = LoadU(df, in_rows[c][2] + dx + 1);
              sad0c = Add(sad0c, AbsDiff(p31, p32));
              t = AbsDiff(p22, p32);
              sad1c = Add(sad1c, t);
              sad2c = Add(sad2c, t);
              const auto p42 = LoadU(df, in_rows[c][2] + dx + 2);
              sad2c = Add(sad2c, AbsDiff(p42, p32));
              const auto p13 = LoadU(df, in_rows[c][3] + dx - 1);
              sad3c = Add(sad3c, AbsDiff(p13, p12));
              const auto p23 = Load(df, in_rows[c][3] + dx);
              t = AbsDiff(p22, p23);
              sad0c = Add(sad0c, t);
              sad3c = Add(sad3c, t);
              sad1c = Add(sad1c, AbsDiff(p13, p23));
              const auto p33 = LoadU(df, in_rows[c][3] + dx + 1);
              sad2c = Add(sad2c, AbsDiff(p33, p23));
              sad3c = Add(sad3c, AbsDiff(p33, p32));
              const auto p24 = Load(df, in_rows[c][4] + dx);
              sad3c = Add(sad3c, AbsDiff(p24, p23));

              const auto scale = Set(df, lf.epf_channel_scale[c]);
              sad0 = MulAdd(sad0c, scale, sad0);
              sad1 = MulAdd(sad1c, scale, sad1);
              sad2 = MulAdd(sad2c, scale, sad2);
              sad3 = MulAdd(sad3c, scale, sad3);
            }
            auto X = Load(df, in_rows[0][2] + dx);
            auto Y = Load(df, in_rows[1][2] + dx);
            auto B = Load(df, in_rows[2][2] + dx);
            auto w = Set(df, 1.0f);
            const auto w0 = EpfWeightVec(sad0, inv_sigma);
            w = Add(w, w0);
            X = MulAdd(w0, Load(df, in_rows[0][1] + dx), X);
            Y = MulAdd(w0, Load(df, in_rows[1][1] + dx), Y);
            B = MulAdd(w0, Load(df, in_rows[2][1] + dx), B);
            const auto w1 = EpfWeightVec(sad1, inv_sigma);
            w = Add(w, w1);
            X = MulAdd(w1, LoadU(df, in_rows[0][2] + dx - 1), X);
            Y = MulAdd(w1, LoadU(df, in_rows[1][2] + dx - 1), Y);
            B = MulAdd(w1, LoadU(df, in_rows[2][2] + dx - 1), B);
            const auto w2 = EpfWeightVec(sad2, inv_sigma);
            w = Add(w, w2);
            X = MulAdd(w2, LoadU(df, in_rows[0][2] + dx + 1), X);
            Y = MulAdd(w2, LoadU(df, in_rows[1][2] + dx + 1), Y);
            B = MulAdd(w2, LoadU(df, in_rows[2][2] + dx + 1), B);
            const auto w3 = EpfWeightVec(sad3, inv_sigma);
            w = Add(w, w3);
            X = MulAdd(w3, Load(df, in_rows[0][3] + dx), X);
            Y = MulAdd(w3, Load(df, in_rows[1][3] + dx), Y);
            B = MulAdd(w3, Load(df, in_rows[2][3] + dx), B);
#if JXL_HIGH_PRECISION
            const auto inv_w = Div(Set(df, 1.0f), w);
#else
            const auto inv_w = ApproximateReciprocal(w);
#endif
            Store(Mul(X, inv_w), df, out_rows[0] + dx);
            Store(Mul(Y, inv_w), df, out_rows[1] + dx);
            Store(Mul(B, inv_w), df, out_rows[2] + dx);
          }
        } else {
          const auto sc0 = Set(df, lf.epf_channel_scale[0]);
          const auto sc1 = Set(df, lf.epf_channel_scale[1]);
          const auto sc2 = Set(df, lf.epf_channel_scale[2]);
          for (size_t lane = 0; lane < kBlockDim; lane += Lanes(df)) {
            const ptrdiff_t dx = dx0 + static_cast<ptrdiff_t>(lane);
            const auto inv_sigma =
                Mul(Set(df, sigma_val), Load(df, sad_mul + lane));
            const auto rx = Load(df, in_rows[0][1] + dx);
            const auto ry = Load(df, in_rows[1][1] + dx);
            const auto rb = Load(df, in_rows[2][1] + dx);
            auto X = rx;
            auto Y = ry;
            auto B = rb;
            auto w = Set(df, 1.0f);
            auto add_nb = [&](int r_off, ptrdiff_t x_off, bool aligned) {
              const auto cx = aligned ? Load(df, in_rows[0][1 + r_off] + x_off)
                                      : LoadU(df, in_rows[0][1 + r_off] + x_off);
              const auto cy = aligned ? Load(df, in_rows[1][1 + r_off] + x_off)
                                      : LoadU(df, in_rows[1][1 + r_off] + x_off);
              const auto cb = aligned ? Load(df, in_rows[2][1 + r_off] + x_off)
                                      : LoadU(df, in_rows[2][1 + r_off] + x_off);
              auto sad = Mul(AbsDiff(cx, rx), sc0);
              sad = MulAdd(AbsDiff(cy, ry), sc1, sad);
              sad = MulAdd(AbsDiff(cb, rb), sc2, sad);
              const auto wt = EpfWeightVec(sad, inv_sigma);
              w = Add(w, wt);
              X = MulAdd(wt, cx, X);
              Y = MulAdd(wt, cy, Y);
              B = MulAdd(wt, cb, B);
            };
            add_nb(-1, dx, true);
            add_nb(0, dx - 1, false);
            add_nb(0, dx + 1, false);
            add_nb(1, dx, true);
#if JXL_HIGH_PRECISION
            const auto inv_w = Div(Set(df, 1.0f), w);
#else
            const auto inv_w = ApproximateReciprocal(w);
#endif
            Store(Mul(X, inv_w), df, out_rows[0] + dx);
            Store(Mul(Y, inv_w), df, out_rows[1] + dx);
            Store(Mul(B, inv_w), df, out_rows[2] + dx);
          }
        }
      }
    }
    std::swap(cur_in, cur_out);
  }

  for (size_t c = 0; c < 3; ++c) {
    out_ptrs[c] = row_origin(cur_in, c, 0 + min_y);
  }
  *out_stride = stride;
  *out_y0 = min_y;
  *out_min_px = min_px;
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include <atomic>

#include "lib/jxl/enc_modular.h"
#include "lib/jxl/modular/encoding/context_predict.h"

namespace jxl {
HWY_EXPORT(ComputeCoefficients);
HWY_EXPORT(InitAcGroupCacheImpl);
HWY_EXPORT(EvalAcGroupCacheImpl);
HWY_EXPORT(EvalAcsBoxCandidateImpl);
HWY_EXPORT(ApplyTransformDcDeltaImpl);
HWY_EXPORT(ReconstructXybPostFilterImpl);
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
                       bool clamp_sdr, float* JXL_RESTRICT out_r,
                       float* JXL_RESTRICT out_g, float* JXL_RESTRICT out_b) {
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
    float r = m00 * mr + m01 * mg + m02 * mb;
    float g = m10 * mr + m11 * mg + m12 * mb;
    float bl = m20 * mr + m21 * mg + m22 * mb;
    if (clamp_sdr) {
      r = Clamp1(r, 0.0f, 1.0f);
      g = Clamp1(g, 0.0f, 1.0f);
      bl = Clamp1(bl, 0.0f, 1.0f);
    }
    out_r[i] = r;
    out_g[i] = g;
    out_b[i] = bl;
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

struct EpfSigmaState {
  ImageF sigma;
  size_t xsize_blocks = 0;
  size_t ysize_blocks = 0;

  Status Init(JxlMemoryManager* memory_manager, size_t xb, size_t yb) {
    xsize_blocks = xb;
    ysize_blocks = yb;
    JXL_ASSIGN_OR_RETURN(sigma, ImageF::Create(memory_manager,
                                               xb + 2 * kSigmaPadding,
                                               yb + 2 * kSigmaPadding));
    ZeroFillImage(&sigma);
    return true;
  }

  void UpdateRect(const LoopFilter& lf, const AcStrategyImage& ac_strategy,
                  float quant_scale, const ImageI& quant_field,
                  const ImageB& sharpness, const Rect& block_rect) {
    if (lf.epf_iters == 0 || block_rect.xsize() == 0 ||
        block_rect.ysize() == 0) {
      return;
    }
    const size_t sigma_stride = sigma.PixelsPerRow();
    const size_t sharpness_stride = sharpness.PixelsPerRow();
    for (size_t by = 0; by < block_rect.ysize(); ++by) {
      float* JXL_RESTRICT sigma_row = block_rect.Row(&sigma, by);
      const uint8_t* JXL_RESTRICT sharp_row =
          block_rect.ConstRow(sharpness, by);
      AcStrategyRow acs_row = ac_strategy.ConstRow(block_rect, by);
      const int32_t* JXL_RESTRICT row_quant =
          block_rect.ConstRow(quant_field, by);
      for (size_t bx = 0; bx < block_rect.xsize(); ++bx) {
        const AcStrategy acs = acs_row[bx];
        if (!acs.IsFirstBlock()) continue;
        const size_t llf_x = acs.covered_blocks_x();
        const size_t llf_y = acs.covered_blocks_y();
        const float sigma_quant =
            lf.epf_quant_mul / (quant_scale * row_quant[bx] * kInvSigmaNum);
        for (size_t iy = 0; iy < llf_y; ++iy) {
          for (size_t ix = 0; ix < llf_x; ++ix) {
            const uint8_t s =
                sharp_row[bx + ix + iy * sharpness_stride];
            const float sig = std::min(-1e-4f, sigma_quant * lf.epf_sharp_lut[s]);
            sigma_row[bx + ix + kSigmaPadding +
                      (iy + kSigmaPadding) * sigma_stride] = 1.0f / sig;
          }
        }
        if (bx + block_rect.x0() == 0) {
          for (size_t iy = 0; iy < llf_y; ++iy) {
            float* p =
                sigma_row + kSigmaPadding + (iy + kSigmaPadding) * sigma_stride;
            for (size_t i = 0; i < kSigmaBorder; ++i) {
              *(p - 1 - i) = p[i];
            }
          }
        }
        if (bx + block_rect.x0() + llf_x == xsize_blocks) {
          for (size_t iy = 0; iy < llf_y; ++iy) {
            float* p = sigma_row + kSigmaPadding + bx + llf_x +
                       (iy + kSigmaPadding) * sigma_stride;
            for (size_t i = 0; i < kSigmaBorder; ++i) {
              p[i] = *(p - 1 - i);
            }
          }
        }
        const size_t offset_before =
            (bx + block_rect.x0() == 0)
                ? (kSigmaPadding - kSigmaBorder)
                : (bx + kSigmaPadding);
        const size_t offset_after =
            (bx + block_rect.x0() + llf_x == xsize_blocks)
                ? (kSigmaPadding + llf_x + bx + kSigmaBorder)
                : (kSigmaPadding + llf_x + bx);
        const size_t num = offset_after - offset_before;
        if (by + block_rect.y0() == 0) {
          for (size_t iy = 0; iy < kSigmaBorder; ++iy) {
            memcpy(sigma_row + offset_before +
                       (kSigmaPadding - 1 - iy) * sigma_stride,
                   sigma_row + offset_before +
                       (kSigmaPadding + iy) * sigma_stride,
                   num * sizeof(*sigma_row));
          }
        }
        if (by + block_rect.y0() + llf_y == ysize_blocks) {
          for (size_t iy = 0; iy < kSigmaBorder; ++iy) {
            memcpy(sigma_row + offset_before +
                       (llf_y + kSigmaPadding + iy) * sigma_stride,
                   sigma_row + offset_before +
                       (llf_y + kSigmaPadding - 1 - iy) * sigma_stride,
                   num * sizeof(*sigma_row));
          }
        }
      }
    }
  }
};

Status ReconstructLinearRgb(JxlMemoryManager* memory_manager,
                            const LoopFilter& lf,
                            const PassesSharedState& shared,
                            const Image3F& pre_gab, const Image3F* feat_xyb,
                            const EpfSigmaState* epf_sigma, EpfWorkspace* ws,
                            const Rect& dirty_rect, Image3F* out_lin,
                            Image3F* out_xyb = nullptr) {
  const size_t xsize = shared.frame_dim.xsize;
  const size_t ysize = shared.frame_dim.ysize;
  if (dirty_rect.xsize() == 0 || dirty_rect.ysize() == 0) return true;
  const size_t x0 = std::min(dirty_rect.x0(), xsize);
  const size_t y0 = std::min(dirty_rect.y0(), ysize);
  const size_t x1 = std::min(dirty_rect.x0() + dirty_rect.xsize(), xsize);
  const size_t y1 = std::min(dirty_rect.y0() + dirty_rect.ysize(), ysize);
  if (x0 >= x1 || y0 >= y1) return true;
  const size_t len = x1 - x0;

  const auto& im = shared.metadata->transform_data.opsin_inverse_matrix;
  float biases_cbrt[3];
  for (size_t c = 0; c < 3; ++c) {
    biases_cbrt[c] = std::cbrt(im.opsin_biases[c]);
  }
  const float inv_scale = 255.0f / shared.metadata->m.IntensityTarget();
  const bool clamp_sdr = !shared.metadata->m.bit_depth.floating_point_sample;

  if (lf.epf_iters > 0 && epf_sigma != nullptr && ws != nullptr) {
    const float* out_ptrs[3] = {};
    size_t out_stride = 0;
    size_t out_y0 = 0;
    size_t out_min_px = 0;
    JXL_RETURN_IF_ERROR(HWY_DYNAMIC_DISPATCH(ReconstructXybPostFilterImpl)(
        memory_manager, lf, pre_gab, epf_sigma->sigma, xsize, ysize, x0, y0, x1,
        y1, ws, out_ptrs, &out_stride, &out_y0, &out_min_px));
    std::vector<float> xyb_row[3];
    if (feat_xyb != nullptr) {
      for (size_t c = 0; c < 3; ++c) xyb_row[c].resize(len);
    }
    for (size_t y = y0; y < y1; ++y) {
      const size_t ry = y - out_y0;
      const size_t rx = x0 - out_min_px;
      const float* row_x = out_ptrs[0] + ry * out_stride + rx;
      const float* row_y = out_ptrs[1] + ry * out_stride + rx;
      const float* row_b = out_ptrs[2] + ry * out_stride + rx;
      if (feat_xyb != nullptr) {
        for (size_t c = 0; c < 3; ++c) {
          const float* src_r = out_ptrs[c] + ry * out_stride + rx;
          const float* rf = feat_xyb->ConstPlaneRow(c, y) + x0;
          float* dst_r = xyb_row[c].data();
          for (size_t i = 0; i < len; ++i) dst_r[i] = src_r[i] + rf[i];
        }
        row_x = xyb_row[0].data();
        row_y = xyb_row[1].data();
        row_b = xyb_row[2].data();
      }
      if (out_xyb != nullptr) {
        memcpy(out_xyb->PlaneRow(0, y) + x0, row_x, len * sizeof(float));
        memcpy(out_xyb->PlaneRow(1, y) + x0, row_y, len * sizeof(float));
        memcpy(out_xyb->PlaneRow(2, y) + x0, row_b, len * sizeof(float));
      }
      if (out_lin != nullptr) {
        XybRowToLinearRgb(row_x, row_y, row_b, len, im.inverse_matrix,
                          im.opsin_biases, biases_cbrt, inv_scale, clamp_sdr,
                          out_lin->PlaneRow(0, y) + x0,
                          out_lin->PlaneRow(1, y) + x0,
                          out_lin->PlaneRow(2, y) + x0);
      }
    }
    return true;
  }

  const bool do_gab = lf.gab;
  float gab_w[3][3] = {};
  if (do_gab) {
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
      if (out_xyb != nullptr) {
        memcpy(out_xyb->PlaneRow(c, y) + x0, dst, len * sizeof(float));
      }
    }
    if (out_lin != nullptr) {
      XybRowToLinearRgb(xyb_row[0].data(), xyb_row[1].data(), xyb_row[2].data(),
                        len, im.inverse_matrix, im.opsin_biases, biases_cbrt,
                        inv_scale, clamp_sdr, out_lin->PlaneRow(0, y) + x0,
                        out_lin->PlaneRow(1, y) + x0,
                        out_lin->PlaneRow(2, y) + x0);
    }
  }
  return true;
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

  void AccumulateBlockError(size_t bx_w, size_t by_h, double weight,
                            double* out_blk) const {
    if (weight <= 0.0) return;
    const size_t num_scales = scales_.size();
    const double sum = WeightedSum();
    const double u = sum * 0.9562382616834844;
    const double g = 2.326765642916932 * u - 0.020884521182843837 * u * u +
                     6.248496625763138e-05 * u * u * u;
    if (g <= 0.0) return;
    const double dg = (2.326765642916932 - 2.0 * 0.020884521182843837 * u +
                       3.0 * 6.248496625763138e-05 * u * u) *
                      0.9562382616834844;
    const double slope =
        10.0 * 0.6276336467831387 * std::pow(g, 0.6276336467831387 - 1.0) * dg;
    if (slope <= 0.0) return;
    const double mul_all = weight * slope;

    for (size_t s = 0; s < num_scales; ++s) {
      const ScaleState& sc = scales_[s];
      const double n = static_cast<double>(sc.w * sc.h);
      double w1_m[9] = {};
      double w4_m[9] = {};
      bool any = false;
      for (size_t c = 0; c < 3; ++c) {
        const size_t wbase = (c * num_scales + s) * 6;
        for (size_t m = 0; m < 3; ++m) {
          const double w1 = kSsim2Weights[wbase + m];
          const double w4 = kSsim2Weights[wbase + 3 + m];
          if (w1 != 0.0) {
            w1_m[3 * c + m] = mul_all * (w1 / n);
            any = true;
          }
          const double tot4 = sc.total_s4[3 * c + m];
          if (w4 > 0.0 && tot4 > 0.0) {
            const double n4 = std::sqrt(std::sqrt(tot4 / n));
            if (n4 > 0.0) {
              w4_m[3 * c + m] = mul_all * (w4 / (4.0 * n * n4 * n4 * n4));
              any = true;
            }
          }
        }
      }
      if (!any) continue;
      for (size_t ty = 0; ty < sc.th; ++ty) {
        const size_t py0 = (ty * kTile) << s;
        const size_t py1 = std::min(h0_, ((ty + 1) * kTile) << s);
        const size_t by0 = std::min(by_h, py0 / kBlockDim);
        const size_t by1 = std::min(by_h, DivCeil(py1, kBlockDim));
        if (by0 >= by1) continue;
        for (size_t tx = 0; tx < sc.tw; ++tx) {
          const size_t px0 = (tx * kTile) << s;
          const size_t px1 = std::min(w0_, ((tx + 1) * kTile) << s);
          const size_t bx0 = std::min(bx_w, px0 / kBlockDim);
          const size_t bx1 = std::min(bx_w, DivCeil(px1, kBlockDim));
          if (bx0 >= bx1) continue;
          const size_t tidx = ty * sc.tw + tx;
          double tval = 0.0;
          for (size_t m = 0; m < 9; ++m) {
            tval += w1_m[m] * sc.tile_s1[m][tidx] +
                    w4_m[m] * sc.tile_s4[m][tidx];
          }
          const double share =
              tval / static_cast<double>((bx1 - bx0) * (by1 - by0));
          for (size_t by = by0; by < by1; ++by) {
            for (size_t bx = bx0; bx < bx1; ++bx) {
              out_blk[by * bx_w + bx] += share;
            }
          }
        }
      }
    }
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

  void AccumulateBlockError(size_t bx_w, size_t by_h, double weight,
                            double* out_blk) const {
    if (weight <= 0.0 || levels_.size() < 2) return;
    constexpr double kBaseWeight[3] = {0.0036334486212581396,
                                       1.6627724170684814, 4.11874532699585};
    constexpr double kImageIntegration = 0.577918291091919;
    constexpr double kEps = 1e-5;
    double per_channel[3] = {0.0, 0.0, 0.0};
    std::vector<std::array<double, 3>> level_sq(levels_.size());
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
      level_sq[k] = {{sq[0], sq[1], sq[2]}};
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
    if (total <= 1e-12) return;
    const double d_dist_d_tot =
        0.25 * std::pow(total + kEps, -0.75) * kImageIntegration * weight;

    for (size_t k = 0; k + 1 < levels_.size(); ++k) {
      const LevelState& lv = levels_[k];
      const double inv_n = 1.0 / static_cast<double>(lv.w * lv.h);
      double d_sq[3] = {};
      bool any = false;
      for (size_t c = 0; c < 3; ++c) {
        const double v = level_sq[k][c] * inv_n;
        const double rms = SafePow(v, 0.5);
        if (rms <= 1e-9) continue;
        const double d_rms_d_sq = 0.5 * std::pow(v + kEps, -0.5) * inv_n;
        const double d_pc_d_rms = 4.0 * std::pow(rms + kEps, 3.0);
        d_sq[c] = d_dist_d_tot * d_pc_d_rms * d_rms_d_sq;
        any = true;
      }
      if (!any) continue;
      for (size_t ty = 0; ty < lv.tiles_y; ++ty) {
        const size_t py0 = (ty * kTile) << k;
        const size_t py1 = std::min(h0_, ((ty + 1) * kTile) << k);
        const size_t by0 = std::min(by_h, py0 / kBlockDim);
        const size_t by1 = std::min(by_h, DivCeil(py1, kBlockDim));
        if (by0 >= by1) continue;
        for (size_t tx = 0; tx < lv.tiles_x; ++tx) {
          const size_t px0 = (tx * kTile) << k;
          const size_t px1 = std::min(w0_, ((tx + 1) * kTile) << k);
          const size_t bx0 = std::min(bx_w, px0 / kBlockDim);
          const size_t bx1 = std::min(bx_w, DivCeil(px1, kBlockDim));
          if (bx0 >= bx1) continue;
          const auto& t = lv.tiles[ty * lv.tiles_x + tx];
          const double tval =
              d_sq[0] * t[0] + d_sq[1] * t[1] + d_sq[2] * t[2];
          const double share =
              tval / static_cast<double>((bx1 - bx0) * (by1 - by0));
          for (size_t by = by0; by < by1; ++by) {
            for (size_t bx = bx0; bx < bx1; ++bx) {
              out_blk[by * bx_w + bx] += share;
            }
          }
        }
      }
    }
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

class Ba3IncrementalArbiter {
 public:
  static constexpr size_t kTile = 8;
  static constexpr size_t kHalo = 5;

  struct Checkpoint {
    size_t px0 = 0, py0 = 0, px1 = 0, py1 = 0;
    std::vector<float> xyb_patch;
    struct TilePatch {
      size_t tile_idx;
      double d3;
      double d1;
    };
    std::vector<TilePatch> tile_patches;
    double prev_total_d3 = 0.0;
    double prev_total_d1 = 0.0;
  };

  void Init(const Image3F& src_lin) {
    w0_ = src_lin.xsize();
    h0_ = src_lin.ysize();
    tw_ = DivCeil(w0_, kTile);
    th_ = DivCeil(h0_, kTile);
    const size_t n = w0_ * h0_;
    for (size_t c = 0; c < 3; ++c) {
      ref_xyb_[c].resize(n);
      test_xyb_[c].resize(n, 0.0f);
      ref_mf_[c].resize(n);
      ref_lf_[c].resize(n);
    }
    for (size_t c = 0; c < 2; ++c) {
      ref_uhf_[c].resize(n);
      ref_hf_[c].resize(n);
    }
    mask_ac_.resize(n);
    mask_dc_.resize(n);
    tile_d3_.assign(tw_ * th_, 0.0);
    tile_d1_.assign(tw_ * th_, 0.0);
    total_d3_ = 0.0;
    total_d1_ = 0.0;

    for (size_t y = 0; y < h0_; ++y) {
      RgbRowToBaXyb(src_lin.ConstPlaneRow(0, y), src_lin.ConstPlaneRow(1, y),
                    src_lin.ConstPlaneRow(2, y), w0_,
                    ref_xyb_[0].data() + y * w0_,
                    ref_xyb_[1].data() + y * w0_,
                    ref_xyb_[2].data() + y * w0_);
    }

    std::vector<float> g0(n);
    std::vector<float> g1(n);
    std::vector<float> g2(n);
    std::vector<float> tmp(n);
    std::vector<float> raw_mask(n, 0.0f);
    for (size_t c = 0; c < 3; ++c) {
      BlurSep5(ref_xyb_[c].data(), w0_, h0_, 0, 0, w0_, h0_, 1, &tmp,
               g1.data());
      BlurSep5(ref_xyb_[c].data(), w0_, h0_, 0, 0, w0_, h0_, 2, &tmp,
               g2.data());
      for (size_t i = 0; i < n; ++i) {
        ref_lf_[c][i] = g2[i];
        ref_mf_[c][i] = g1[i] - g2[i];
      }
      if (c < 2) {
        BlurSep3(ref_xyb_[c].data(), w0_, h0_, 0, 0, w0_, h0_, &tmp, g0.data());
        const float m_w = (c == 0) ? 2.5f : 1.0f;
        for (size_t i = 0; i < n; ++i) {
          const float uhf = ref_xyb_[c][i] - g0[i];
          const float hf = g0[i] - g1[i];
          ref_uhf_[c][i] = uhf;
          ref_hf_[c][i] = hf;
          const float a = std::sqrt(uhf * uhf + hf * hf);
          raw_mask[i] += m_w * a;
        }
      }
    }
    BlurSep5(raw_mask.data(), w0_, h0_, 0, 0, w0_, h0_, 2, &tmp, g2.data());
    constexpr double kGlobalScale = 0.0709;
    for (size_t i = 0; i < n; ++i) {
      const double delta = static_cast<double>(g2[i]);
      const double ca = 2.5485944793 / (0.451936922203 * delta + 0.829591754942);
      const double ra = kGlobalScale * (1.0 + ca);
      mask_ac_[i] = static_cast<float>(ra * ra);
      const double cd = 0.505054525019 / (3.87449418804 * delta + 0.20025578522);
      const double rd = kGlobalScale * (1.0 + cd);
      mask_dc_[i] = static_cast<float>(rd * rd);
    }
  }

  double ScoreFull(const Image3F& test_lin) {
    for (size_t y = 0; y < h0_; ++y) {
      RgbRowToBaXyb(test_lin.ConstPlaneRow(0, y), test_lin.ConstPlaneRow(1, y),
                    test_lin.ConstPlaneRow(2, y), w0_,
                    test_xyb_[0].data() + y * w0_,
                    test_xyb_[1].data() + y * w0_,
                    test_xyb_[2].data() + y * w0_);
    }
    PoolTiles(0, 0, tw_, th_, nullptr);
    return Score();
  }

  double UpdateCheckpointed(const Image3F& test_lin, const Rect& dirty,
                            Checkpoint* cp) {
    cp->tile_patches.clear();
    cp->xyb_patch.clear();
    cp->prev_total_d3 = total_d3_;
    cp->prev_total_d1 = total_d1_;
    cp->px0 = cp->py0 = cp->px1 = cp->py1 = 0;
    if (dirty.xsize() == 0 || dirty.ysize() == 0) return Score();
    const size_t rx0 = std::min(dirty.x0(), w0_);
    const size_t ry0 = std::min(dirty.y0(), h0_);
    const size_t rx1 = std::min(dirty.x0() + dirty.xsize(), w0_);
    const size_t ry1 = std::min(dirty.y0() + dirty.ysize(), h0_);
    if (rx0 >= rx1 || ry0 >= ry1) return Score();

    cp->px0 = rx0;
    cp->py0 = ry0;
    cp->px1 = rx1;
    cp->py1 = ry1;
    const size_t len = rx1 - rx0;
    cp->xyb_patch.resize(3 * len * (ry1 - ry0));
    size_t at = 0;
    for (size_t c = 0; c < 3; ++c) {
      for (size_t y = ry0; y < ry1; ++y) {
        memcpy(cp->xyb_patch.data() + at, test_xyb_[c].data() + y * w0_ + rx0,
               len * sizeof(float));
        at += len;
      }
    }
    for (size_t y = ry0; y < ry1; ++y) {
      RgbRowToBaXyb(test_lin.ConstPlaneRow(0, y) + rx0,
                    test_lin.ConstPlaneRow(1, y) + rx0,
                    test_lin.ConstPlaneRow(2, y) + rx0, len,
                    test_xyb_[0].data() + y * w0_ + rx0,
                    test_xyb_[1].data() + y * w0_ + rx0,
                    test_xyb_[2].data() + y * w0_ + rx0);
    }

    const size_t bx0 = (rx0 > kHalo) ? (rx0 - kHalo) : 0;
    const size_t by0 = (ry0 > kHalo) ? (ry0 - kHalo) : 0;
    const size_t bx1 = std::min(w0_, rx1 + kHalo);
    const size_t by1 = std::min(h0_, ry1 + kHalo);
    PoolTiles(bx0 / kTile, by0 / kTile, std::min(tw_, DivCeil(bx1, kTile)),
              std::min(th_, DivCeil(by1, kTile)), cp);
    return Score();
  }

  void Rollback(const Checkpoint& cp) {
    if (cp.px1 > cp.px0 && cp.py1 > cp.py0) {
      const size_t len = cp.px1 - cp.px0;
      size_t at = 0;
      for (size_t c = 0; c < 3; ++c) {
        for (size_t y = cp.py0; y < cp.py1; ++y) {
          memcpy(test_xyb_[c].data() + y * w0_ + cp.px0,
                 cp.xyb_patch.data() + at, len * sizeof(float));
          at += len;
        }
      }
    }
    for (size_t i = cp.tile_patches.size(); i > 0; --i) {
      const auto& tp = cp.tile_patches[i - 1];
      tile_d3_[tp.tile_idx] = tp.d3;
      tile_d1_[tp.tile_idx] = tp.d1;
    }
    total_d3_ = cp.prev_total_d3;
    total_d1_ = cp.prev_total_d1;
  }

  double Score() const {
    const double inv_n = 1.0 / static_cast<double>(w0_ * h0_);
    const double d3_norm = std::cbrt(std::max(0.0, total_d3_) * inv_n);
    const double d1_norm = std::max(0.0, total_d1_) * inv_n;
    return 0.55 * d3_norm + 0.45 * d1_norm;
  }

  void AccumulateBlockError(size_t bx_w, size_t by_h, double weight,
                            double* out_blk) const {
    if (weight <= 0.0 || (total_d3_ <= 1e-12 && total_d1_ <= 1e-12)) return;
    const double inv_n = 1.0 / static_cast<double>(w0_ * h0_);
    const double s3 = std::cbrt(std::max(0.0, total_d3_) * inv_n);
    const double w_d3 =
        (s3 > 1e-9) ? (0.55 * weight * inv_n / (3.0 * s3 * s3)) : 0.0;
    const double w_d1 = 0.45 * weight * inv_n;
    const size_t lim_y = std::min(by_h, th_);
    const size_t lim_x = std::min(bx_w, tw_);
    for (size_t ty = 0; ty < lim_y; ++ty) {
      for (size_t tx = 0; tx < lim_x; ++tx) {
        const size_t tidx = ty * tw_ + tx;
        out_blk[ty * bx_w + tx] += w_d3 * tile_d3_[tidx] + w_d1 * tile_d1_[tidx];
      }
    }
  }

 private:
  static inline size_t ClampCoord(ptrdiff_t v, size_t n) {
    if (v < 0) return static_cast<size_t>(Mirror(v, static_cast<int64_t>(n)));
    if (static_cast<size_t>(v) >= n) {
      return static_cast<size_t>(Mirror(v, static_cast<int64_t>(n)));
    }
    return static_cast<size_t>(v);
  }

  static void RgbRowToBaXyb(const float* JXL_RESTRICT r,
                            const float* JXL_RESTRICT g,
                            const float* JXL_RESTRICT b, size_t len,
                            float* JXL_RESTRICT ox, float* JXL_RESTRICT oy,
                            float* JXL_RESTRICT ob) {
    constexpr float kRetMul = 19.245013259874995f;
    constexpr float kRetAdd = -23.16046239805755f;
    constexpr float kBias = 9.9710635769299145f;
    for (size_t i = 0; i < len; ++i) {
      const float sr = std::max(0.0f, r[i]) * 255.0f;
      const float sg = std::max(0.0f, g[i]) * 255.0f;
      const float sb = std::max(0.0f, b[i]) * 255.0f;
      const float a0 = std::max(
          1.75574836f,
          0.29956550f * sr + 0.63373088f * sg + 0.07770562f * sb + 1.75574836f);
      const float a1 = std::max(
          1.75574836f,
          0.22158691f * sr + 0.69391388f * sg + 0.09873136f * sb + 1.75574836f);
      const float a2 = std::max(
          12.2264547f,
          0.02f * sr + 0.02f * sg + 0.20480129f * sb + 12.2264547f);
      const float g0 = kRetMul * std::log(a0 + kBias) + kRetAdd;
      const float g1 = kRetMul * std::log(a1 + kBias) + kRetAdd;
      const float g2 = kRetMul * std::log(a2 + kBias) + kRetAdd;
      ox[i] = g0 - g1;
      oy[i] = g0 + g1;
      ob[i] = g2;
    }
  }

  static void BlurSep3(const float* src, size_t w, size_t h, size_t x0,
                       size_t y0, size_t x1, size_t y1,
                       std::vector<float>* tmp, float* dst) {
    const size_t bw = x1 - x0;
    const size_t hy0 = (y0 > 1) ? (y0 - 1) : 0;
    const size_t hy1 = std::min(h, y1 + 1);
    const size_t hh = hy1 - hy0;
    tmp->resize(bw * hh);
    for (size_t y = hy0; y < hy1; ++y) {
      const float* row = src + y * w;
      float* trow = tmp->data() + (y - hy0) * bw;
      for (size_t x = x0; x < x1; ++x) {
        const size_t xl = (x > 0) ? (x - 1) : 0;
        const size_t xr = (x + 1 < w) ? (x + 1) : (w - 1);
        trow[x - x0] = 0.20f * (row[xl] + row[xr]) + 0.60f * row[x];
      }
    }
    for (size_t y = y0; y < y1; ++y) {
      const size_t yt = (y > 0 ? y - 1 : 0) - hy0;
      const size_t ym = y - hy0;
      const size_t yb = (y + 1 < h ? y + 1 : h - 1) - hy0;
      const float* rt = tmp->data() + yt * bw;
      const float* rm = tmp->data() + ym * bw;
      const float* rb = tmp->data() + yb * bw;
      float* drow = dst + (y - y0) * bw;
      for (size_t i = 0; i < bw; ++i) {
        drow[i] = 0.20f * (rt[i] + rb[i]) + 0.60f * rm[i];
      }
    }
  }

  static void BlurSep5(const float* src, size_t w, size_t h, size_t x0,
                       size_t y0, size_t x1, size_t y1, ptrdiff_t step,
                       std::vector<float>* tmp, float* dst) {
    const size_t bw = x1 - x0;
    const size_t rad = static_cast<size_t>(2 * step);
    size_t hy0 = 0;
    size_t hy1 = h;
    if (h > 2 * rad) {
      hy0 = (y0 > rad) ? (y0 - rad) : 0;
      hy1 = std::min(h, y1 + rad);
      if (y0 < rad) hy1 = std::max(hy1, std::min(h, rad));
      if (y1 + rad > h) hy0 = std::min(hy0, h - rad);
    }
    const size_t hh = hy1 - hy0;
    tmp->resize(bw * hh);
    constexpr float kW0 = 0.38774f;
    constexpr float kW1 = 0.24477f;
    constexpr float kW2 = 0.06136f;
    for (size_t y = hy0; y < hy1; ++y) {
      const float* row = src + y * w;
      float* trow = tmp->data() + (y - hy0) * bw;
      for (size_t x = x0; x < x1; ++x) {
        const ptrdiff_t px = static_cast<ptrdiff_t>(x);
        const float s1 =
            row[ClampCoord(px - step, w)] + row[ClampCoord(px + step, w)];
        const float s2 = row[ClampCoord(px - 2 * step, w)] +
                         row[ClampCoord(px + 2 * step, w)];
        trow[x - x0] = kW0 * row[x] + kW1 * s1 + kW2 * s2;
      }
    }
    for (size_t y = y0; y < y1; ++y) {
      const ptrdiff_t py = static_cast<ptrdiff_t>(y);
      const float* r_2 = tmp->data() + (ClampCoord(py - 2 * step, h) - hy0) * bw;
      const float* r_1 = tmp->data() + (ClampCoord(py - step, h) - hy0) * bw;
      const float* r0 = tmp->data() + (y - hy0) * bw;
      const float* r1 = tmp->data() + (ClampCoord(py + step, h) - hy0) * bw;
      const float* r2 = tmp->data() + (ClampCoord(py + 2 * step, h) - hy0) * bw;
      float* drow = dst + (y - y0) * bw;
      for (size_t i = 0; i < bw; ++i) {
        drow[i] = kW0 * r0[i] + kW1 * (r_1[i] + r1[i]) + kW2 * (r_2[i] + r2[i]);
      }
    }
  }

  static inline float AsymDiffSq(float v0, float v1, float w_sym,
                                 float w_asym) {
    const float d = v0 - v1;
    float tot = w_sym * d * d;
    const float a0 = std::abs(v0);
    const float too_small = 0.68f * a0;
    const float too_big = 1.04f * a0;
    float ex = 0.0f;
    if (v0 < 0.0f) {
      if (v1 > -too_small) {
        ex = v1 + too_small;
      } else if (v1 < -too_big) {
        ex = -v1 - too_big;
      }
    } else {
      if (v1 < too_small) {
        ex = too_small - v1;
      } else if (v1 > too_big) {
        ex = v1 - too_big;
      }
    }
    return tot + w_asym * ex * ex;
  }

  void PoolTiles(size_t tx0, size_t ty0, size_t tx1, size_t ty1,
                 Checkpoint* cp) {
    if (tx0 >= tx1 || ty0 >= ty1) return;
    const size_t px0 = tx0 * kTile;
    const size_t py0 = ty0 * kTile;
    const size_t px1 = std::min(w0_, tx1 * kTile);
    const size_t py1 = std::min(h0_, ty1 * kTile);
    const size_t ex0 = (px0 > 0) ? (px0 - 1) : 0;
    const size_t ey0 = (py0 > 0) ? (py0 - 1) : 0;
    const size_t ex1 = std::min(w0_, px1 + 1);
    const size_t ey1 = std::min(h0_, py1 + 1);
    const size_t ew = ex1 - ex0;
    const size_t eh = ey1 - ey0;

    for (size_t c = 0; c < 3; ++c) {
      sc_g1_[c].resize(ew * eh);
      sc_g2_[c].resize(ew * eh);
      BlurSep5(test_xyb_[c].data(), w0_, h0_, ex0, ey0, ex1, ey1, 1, &sc_tmp_,
               sc_g1_[c].data());
      BlurSep5(test_xyb_[c].data(), w0_, h0_, ex0, ey0, ex1, ey1, 2, &sc_tmp_,
               sc_g2_[c].data());
      if (c < 2) {
        sc_g0_[c].resize(ew * eh);
        sc_uhf_[c].resize(ew * eh);
        sc_hf_[c].resize(ew * eh);
        BlurSep3(test_xyb_[c].data(), w0_, h0_, ex0, ey0, ex1, ey1, &sc_tmp_,
                 sc_g0_[c].data());
        for (size_t y = ey0; y < ey1; ++y) {
          const size_t roff = (y - ey0) * ew;
          const float* tx_r = test_xyb_[c].data() + y * w0_ + ex0;
          for (size_t i = 0; i < ew; ++i) {
            sc_uhf_[c][roff + i] = tx_r[i] - sc_g0_[c][roff + i];
            sc_hf_[c][roff + i] = sc_g0_[c][roff + i] - sc_g1_[c][roff + i];
          }
        }
      }
    }

    constexpr float kXMul = 29.5f;
    for (size_t ty = ty0; ty < ty1; ++ty) {
      const size_t y_lo = ty * kTile;
      const size_t y_hi = std::min(h0_, y_lo + kTile);
      for (size_t tx = tx0; tx < tx1; ++tx) {
        const size_t x_lo = tx * kTile;
        const size_t x_hi = std::min(w0_, x_lo + kTile);
        const size_t tidx = ty * tw_ + tx;
        if (cp != nullptr) {
          cp->tile_patches.push_back({tidx, tile_d3_[tidx], tile_d1_[tidx]});
        }
        double sum_d3 = 0.0;
        double sum_d1 = 0.0;
        for (size_t y = y_lo; y < y_hi; ++y) {
          const size_t sy = y - ey0;
          const size_t syt = (y > ey0) ? (sy - 1) : sy;
          const size_t syb = (y + 1 < ey1) ? (sy + 1) : sy;
          const size_t ry_t = (y > 0) ? (y - 1) : 0;
          const size_t ry_b = (y + 1 < h0_) ? (y + 1) : (h0_ - 1);
          for (size_t x = x_lo; x < x_hi; ++x) {
            const size_t idx = y * w0_ + x;
            const size_t sx = x - ex0;
            const size_t sxl = (x > ex0) ? (sx - 1) : sx;
            const size_t sxr = (x + 1 < ex1) ? (sx + 1) : sx;
            const size_t rx_l = (x > 0) ? (x - 1) : 0;
            const size_t rx_r = (x + 1 < w0_) ? (x + 1) : (w0_ - 1);
            const size_t sidx = sy * ew + sx;

            float ac_c[3] = {0.0f, 0.0f, 0.0f};
            float dc_c[3] = {0.0f, 0.0f, 0.0f};
            for (size_t c = 0; c < 3; ++c) {
              const float dlf = ref_lf_[c][idx] - sc_g2_[c][sidx];
              const float dmf =
                  ref_mf_[c][idx] - (sc_g1_[c][sidx] - sc_g2_[c][sidx]);
              const float w_lf = (c == 0) ? 1.0f : (c == 1 ? 0.55f : 0.35f);
              const float w_mf = (c == 0) ? 1.15f : (c == 1 ? 1.0f : 0.25f);
              dc_c[c] = w_lf * dlf * dlf;
              ac_c[c] = w_mf * dmf * dmf;
            }
            for (size_t c = 0; c < 2; ++c) {
              const float w_h = (c == 0) ? 1.35f : 1.75f;
              const float w_u = (c == 0) ? 1.10f : 1.95f;
              ac_c[c] += AsymDiffSq(ref_hf_[c][idx], sc_hf_[c][sidx], w_h,
                                    1.15f * w_h);
              ac_c[c] += AsymDiffSq(ref_uhf_[c][idx], sc_uhf_[c][sidx], w_u,
                                    1.35f * w_u);
            }
            // 4-direction Malta-like UHF+HF line contrast on Y.
            const float r_dx =
                ref_uhf_[1][y * w0_ + rx_r] - ref_uhf_[1][y * w0_ + rx_l];
            const float t_dx =
                sc_uhf_[1][sy * ew + sxr] - sc_uhf_[1][sy * ew + sxl];
            const float r_dy =
                ref_uhf_[1][ry_b * w0_ + x] - ref_uhf_[1][ry_t * w0_ + x];
            const float t_dy =
                sc_uhf_[1][syb * ew + sx] - sc_uhf_[1][syt * ew + sx];
            const float r_d1 =
                ref_hf_[1][ry_b * w0_ + rx_r] - ref_hf_[1][ry_t * w0_ + rx_l];
            const float t_d1 =
                sc_hf_[1][syb * ew + sxr] - sc_hf_[1][syt * ew + sxl];
            const float r_d2 =
                ref_hf_[1][ry_b * w0_ + rx_l] - ref_hf_[1][ry_t * w0_ + rx_r];
            const float t_d2 =
                sc_hf_[1][syb * ew + sxl] - sc_hf_[1][syt * ew + sxr];
            ac_c[1] += AsymDiffSq(r_dx, t_dx, 0.32f, 0.28f) +
                       AsymDiffSq(r_dy, t_dy, 0.32f, 0.28f) +
                       AsymDiffSq(r_d1, t_d1, 0.22f, 0.18f) +
                       AsymDiffSq(r_d2, t_d2, 0.22f, 0.18f);

            const float d2 =
                mask_dc_[idx] * (kXMul * dc_c[0] + dc_c[1] + dc_c[2]) +
                mask_ac_[idx] * (kXMul * ac_c[0] + ac_c[1] + ac_c[2]);
            const double d = std::sqrt(static_cast<double>(std::max(0.0f, d2)));
            sum_d3 += d * d * d;
            sum_d1 += d;
          }
        }
        total_d3_ += (sum_d3 - tile_d3_[tidx]);
        total_d1_ += (sum_d1 - tile_d1_[tidx]);
        tile_d3_[tidx] = sum_d3;
        tile_d1_[tidx] = sum_d1;
      }
    }
  }

  size_t w0_ = 0;
  size_t h0_ = 0;
  size_t tw_ = 0;
  size_t th_ = 0;
  std::vector<float> ref_xyb_[3];
  std::vector<float> test_xyb_[3];
  std::vector<float> ref_uhf_[2];
  std::vector<float> ref_hf_[2];
  std::vector<float> ref_mf_[3];
  std::vector<float> ref_lf_[3];
  std::vector<float> mask_ac_;
  std::vector<float> mask_dc_;
  std::vector<double> tile_d3_;
  std::vector<double> tile_d1_;
  double total_d3_ = 0.0;
  double total_d1_ = 0.0;

  std::vector<float> sc_tmp_;
  std::vector<float> sc_g0_[2];
  std::vector<float> sc_g1_[3];
  std::vector<float> sc_g2_[3];
  std::vector<float> sc_uhf_[2];
  std::vector<float> sc_hf_[2];
};

class WdProxyIncrementalArbiter {
 public:
  static constexpr size_t kNumScales = 3;
  static constexpr size_t kNumFeats = 12;
  static constexpr size_t kTile = 8;

  struct Checkpoint {
    struct SrgbPatch {
      size_t scale;
      size_t x0, y0, x1, y1;
      size_t offset;
    };
    struct TileSave {
      uint8_t scale;
      uint32_t idx;
      double w2;
    };
    std::vector<SrgbPatch> srgb_patches;
    std::vector<float> srgb_data;
    std::vector<TileSave> tile_saves;
    double prev_total_w2[kNumScales] = {};
  };

  void Init(const Image3F& src_lin) {
    w0_ = src_lin.xsize();
    h0_ = src_lin.ysize();
    size_t w = w0_;
    size_t h = h0_;
    for (size_t s = 0; s < kNumScales; ++s) {
      ScaleData& sd = scales_[s];
      sd.w = w;
      sd.h = h;
      sd.tw = DivCeil(w, kTile);
      sd.th = DivCeil(h, kTile);
      for (size_t c = 0; c < 3; ++c) {
        sd.ref_srgb[c].resize(w * h);
        sd.test_srgb[c].resize(w * h, 0.0f);
      }
      sd.tile_w2.assign(sd.tw * sd.th, 0.0);
      sd.total_w2 = 0.0;
      w = std::max<size_t>(1, (w + 1) / 2);
      h = std::max<size_t>(1, (h + 1) / 2);
    }

    for (size_t y = 0; y < h0_; ++y) {
      RgbRowToNormSrgb(src_lin.ConstPlaneRow(0, y), src_lin.ConstPlaneRow(1, y),
                       src_lin.ConstPlaneRow(2, y), w0_,
                       scales_[0].ref_srgb[0].data() + y * w0_,
                       scales_[0].ref_srgb[1].data() + y * w0_,
                       scales_[0].ref_srgb[2].data() + y * w0_);
    }
    for (size_t s = 1; s < kNumScales; ++s) {
      DownsampleLowpass3x3(scales_[s - 1].ref_srgb, scales_[s - 1].w,
                           scales_[s - 1].h, 0, 0, scales_[s].w, scales_[s].h,
                           scales_[s].w, &sc_ds_tmp_, scales_[s].ref_srgb);
    }
  }

  double ScoreFull(const Image3F& test_lin) {
    for (size_t y = 0; y < h0_; ++y) {
      RgbRowToNormSrgb(test_lin.ConstPlaneRow(0, y),
                       test_lin.ConstPlaneRow(1, y),
                       test_lin.ConstPlaneRow(2, y), w0_,
                       scales_[0].test_srgb[0].data() + y * w0_,
                       scales_[0].test_srgb[1].data() + y * w0_,
                       scales_[0].test_srgb[2].data() + y * w0_);
    }
    for (size_t s = 0; s < kNumScales; ++s) {
      ScaleData& sd = scales_[s];
      if (s > 0) {
        DownsampleLowpass3x3(scales_[s - 1].test_srgb, scales_[s - 1].w,
                             scales_[s - 1].h, 0, 0, sd.w, sd.h, sd.w,
                             &sc_ds_tmp_, sd.test_srgb);
      }
      PoolScaleTiles(s, 0, 0, sd.tw, sd.th, nullptr);
    }
    return Score();
  }

  double UpdateCheckpointed(const Image3F& test_lin, const Rect& dirty,
                            Checkpoint* cp) {
    cp->srgb_patches.clear();
    cp->srgb_data.clear();
    cp->tile_saves.clear();
    for (size_t s = 0; s < kNumScales; ++s) {
      cp->prev_total_w2[s] = scales_[s].total_w2;
    }
    if (dirty.xsize() == 0 || dirty.ysize() == 0) return Score();
    size_t rx0 = std::min(dirty.x0(), w0_);
    size_t ry0 = std::min(dirty.y0(), h0_);
    size_t rx1 = std::min(dirty.x0() + dirty.xsize(), w0_);
    size_t ry1 = std::min(dirty.y0() + dirty.ysize(), h0_);
    if (rx0 >= rx1 || ry0 >= ry1) return Score();

    for (size_t s = 0; s < kNumScales; ++s) {
      ScaleData& sd = scales_[s];
      if (s == 0) {
        SaveSrgbPatch(0, rx0, ry0, rx1, ry1, cp);
        const size_t len = rx1 - rx0;
        for (size_t y = ry0; y < ry1; ++y) {
          RgbRowToNormSrgb(test_lin.ConstPlaneRow(0, y) + rx0,
                           test_lin.ConstPlaneRow(1, y) + rx0,
                           test_lin.ConstPlaneRow(2, y) + rx0, len,
                           sd.test_srgb[0].data() + y * w0_ + rx0,
                           sd.test_srgb[1].data() + y * w0_ + rx0,
                           sd.test_srgb[2].data() + y * w0_ + rx0);
        }
      } else {
        const size_t ox0 = (rx0 > 1) ? ((rx0 - 1) / 2) : 0;
        const size_t oy0 = (ry0 > 1) ? ((ry0 - 1) / 2) : 0;
        const size_t ox1 = std::min(sd.w, rx1 / 2 + 1);
        const size_t oy1 = std::min(sd.h, ry1 / 2 + 1);
        rx0 = ox0;
        ry0 = oy0;
        rx1 = ox1;
        ry1 = oy1;
        if (rx0 >= rx1 || ry0 >= ry1) break;
        SaveSrgbPatch(s, rx0, ry0, rx1, ry1, cp);
        DownsampleLowpass3x3(scales_[s - 1].test_srgb, scales_[s - 1].w,
                             scales_[s - 1].h, rx0, ry0, rx1, ry1, sd.w,
                             &sc_ds_tmp_, sd.test_srgb);
      }
      const size_t bx0 = (rx0 > 2) ? (rx0 - 2) : 0;
      const size_t by0 = (ry0 > 2) ? (ry0 - 2) : 0;
      const size_t bx1 = std::min(sd.w, rx1 + 2);
      const size_t by1 = std::min(sd.h, ry1 + 2);
      PoolScaleTiles(s, bx0 / kTile, by0 / kTile,
                     std::min(sd.tw, DivCeil(bx1, kTile)),
                     std::min(sd.th, DivCeil(by1, kTile)), cp);
    }
    return Score();
  }

  void Rollback(const Checkpoint& cp) {
    for (const auto& sp : cp.srgb_patches) {
      ScaleData& sd = scales_[sp.scale];
      const size_t len = sp.x1 - sp.x0;
      size_t at = sp.offset;
      for (size_t c = 0; c < 3; ++c) {
        for (size_t y = sp.y0; y < sp.y1; ++y) {
          memcpy(sd.test_srgb[c].data() + y * sd.w + sp.x0,
                 cp.srgb_data.data() + at, len * sizeof(float));
          at += len;
        }
      }
    }
    for (size_t i = cp.tile_saves.size(); i > 0; --i) {
      const auto& ts = cp.tile_saves[i - 1];
      scales_[ts.scale].tile_w2[ts.idx] = ts.w2;
    }
    for (size_t s = 0; s < kNumScales; ++s) {
      scales_[s].total_w2 = cp.prev_total_w2[s];
    }
  }

  double Score() const {
    constexpr double kScaleW[kNumScales] = {0.42, 0.35, 0.23};
    double tot = 0.0;
    for (size_t s = 0; s < kNumScales; ++s) {
      const ScaleData& sd = scales_[s];
      const double inv_n =
          1.0 / static_cast<double>(std::max<size_t>(1, sd.w * sd.h));
      tot += kScaleW[s] * std::max(0.0, sd.total_w2) * inv_n;
    }
    return tot;
  }

  void AccumulateBlockError(size_t bx_w, size_t by_h, double weight,
                            double* out_blk) const {
    if (weight <= 0.0) return;
    constexpr double kScaleW[kNumScales] = {0.42, 0.35, 0.23};
    for (size_t s = 0; s < kNumScales; ++s) {
      const ScaleData& sd = scales_[s];
      if (sd.total_w2 <= 1e-12) continue;
      const double d_w2 =
          weight * kScaleW[s] /
          static_cast<double>(std::max<size_t>(1, sd.w * sd.h));
      for (size_t ty = 0; ty < sd.th; ++ty) {
        const size_t fy0 = (ty * kTile) << s;
        const size_t fy1 = std::min(h0_, ((ty + 1) * kTile) << s);
        const size_t by0 = std::min(by_h, fy0 / kBlockDim);
        const size_t by1 = std::min(by_h, DivCeil(fy1, kBlockDim));
        if (by0 >= by1) continue;
        for (size_t tx = 0; tx < sd.tw; ++tx) {
          const size_t fx0 = (tx * kTile) << s;
          const size_t fx1 = std::min(w0_, ((tx + 1) * kTile) << s);
          const size_t bx0 = std::min(bx_w, fx0 / kBlockDim);
          const size_t bx1 = std::min(bx_w, DivCeil(fx1, kBlockDim));
          if (bx0 >= bx1) continue;
          const double val = d_w2 * sd.tile_w2[ty * sd.tw + tx];
          const double share =
              val / static_cast<double>((bx1 - bx0) * (by1 - by0));
          for (size_t by = by0; by < by1; ++by) {
            for (size_t bx = bx0; bx < bx1; ++bx) {
              out_blk[by * bx_w + bx] += share;
            }
          }
        }
      }
    }
  }

 private:
  struct ScaleData {
    size_t w = 0;
    size_t h = 0;
    size_t tw = 0;
    size_t th = 0;
    std::vector<float> ref_srgb[3];
    std::vector<float> test_srgb[3];
    std::vector<double> tile_w2;
    double total_w2 = 0.0;
  };

  static inline float LinearToSrgb01(float v) {
    const float c = Clamp1(v, 0.0f, 1.0f);
    if (c <= 0.0031308f) return 12.92f * c;
    const float s4 = std::sqrt(std::sqrt(c));
    return 1.055f * FastCbrtPositive(c * s4) - 0.055f;
  }

  static void RgbRowToNormSrgb(const float* JXL_RESTRICT r,
                               const float* JXL_RESTRICT g,
                               const float* JXL_RESTRICT b, size_t len,
                               float* JXL_RESTRICT sr, float* JXL_RESTRICT sg,
                               float* JXL_RESTRICT sb) {
    constexpr float kInvStd0 = 1.0f / 0.229f;
    constexpr float kInvStd1 = 1.0f / 0.224f;
    constexpr float kInvStd2 = 1.0f / 0.225f;
    for (size_t i = 0; i < len; ++i) {
      sr[i] = LinearToSrgb01(r[i]) * kInvStd0;
      sg[i] = LinearToSrgb01(g[i]) * kInvStd1;
      sb[i] = LinearToSrgb01(b[i]) * kInvStd2;
    }
  }

  // Matches wasserstein.py `lowpass(inputs, stride=2)` with zero-padded borders.
  static void DownsampleLowpass3x3(const std::vector<float> src[3], size_t sw,
                                   size_t sh, size_t ox0, size_t oy0,
                                   size_t ox1, size_t oy1, size_t dw,
                                   std::vector<float>* tmp_h,
                                   std::vector<float> dst[3]) {
    if (ox0 >= ox1 || oy0 >= oy1) return;
    const size_t sy_lo = (2 * oy0 > 0) ? (2 * oy0 - 1) : 0;
    const size_t sy_hi = std::min(sh, 2 * (oy1 - 1) + 2);
    const size_t ow = ox1 - ox0;
    const size_t sh_sub = sy_hi - sy_lo;
    tmp_h->resize(ow * sh_sub);
    for (size_t c = 0; c < 3; ++c) {
      const float* s = src[c].data();
      for (size_t sy = sy_lo; sy < sy_hi; ++sy) {
        const float* srow = s + sy * sw;
        float* hrow = tmp_h->data() + (sy - sy_lo) * ow;
        for (size_t ox = ox0; ox < ox1; ++ox) {
          const size_t sx = 2 * ox;
          const float vl = (sx > 0) ? srow[sx - 1] : 0.0f;
          const float vc = (sx < sw) ? srow[sx] : 0.0f;
          const float vr = (sx + 1 < sw) ? srow[sx + 1] : 0.0f;
          hrow[ox - ox0] = 0.25f * (vl + vr) + 0.50f * vc;
        }
      }
      float* d = dst[c].data();
      for (size_t oy = oy0; oy < oy1; ++oy) {
        const size_t sy = 2 * oy;
        const float* rt =
            (sy > 0) ? (tmp_h->data() + (sy - 1 - sy_lo) * ow) : nullptr;
        const float* rc =
            (sy < sh) ? (tmp_h->data() + (sy - sy_lo) * ow) : nullptr;
        const float* rb =
            (sy + 1 < sh) ? (tmp_h->data() + (sy + 1 - sy_lo) * ow) : nullptr;
        float* drow = d + oy * dw;
        for (size_t i = 0; i < ow; ++i) {
          const float vt = (rt != nullptr) ? rt[i] : 0.0f;
          const float vc = (rc != nullptr) ? rc[i] : 0.0f;
          const float vb = (rb != nullptr) ? rb[i] : 0.0f;
          drow[ox0 + i] = 0.25f * (vt + vb) + 0.50f * vc;
        }
      }
    }
  }

  static void ExtractFeatsRect(const std::vector<float> srgb[3], size_t w,
                               size_t h, size_t ex0, size_t ey0, size_t ex1,
                               size_t ey1, float* out_f) {
    const size_t ew = ex1 - ex0;
    const float* R = srgb[0].data();
    const float* G = srgb[1].data();
    const float* B = srgb[2].data();
    auto lum_at = [&](size_t x, size_t y) -> float {
      const size_t idx = y * w + x;
      return 0.299f * R[idx] + 0.587f * G[idx] + 0.114f * B[idx];
    };
    for (size_t y = ey0; y < ey1; ++y) {
      const size_t yt = (y > 0) ? (y - 1) : 0;
      const size_t yb = (y + 1 < h) ? (y + 1) : (h - 1);
      for (size_t x = ex0; x < ex1; ++x) {
        const size_t xl = (x > 0) ? (x - 1) : 0;
        const size_t xr = (x + 1 < w) ? (x + 1) : (w - 1);
        const size_t idx = y * w + x;
        float* f = out_f + ((y - ey0) * ew + (x - ex0)) * kNumFeats;
        const float rv = R[idx];
        const float gv = G[idx];
        const float bv = B[idx];
        const float lc = 0.299f * rv + 0.587f * gv + 0.114f * bv;
        const float ll = lum_at(xl, y);
        const float lr = lum_at(xr, y);
        const float lt = lum_at(x, yt);
        const float lb = lum_at(x, yb);
        const float ltl = lum_at(xl, yt);
        const float ltr = lum_at(xr, yt);
        const float lbl = lum_at(xl, yb);
        const float lbr = lum_at(xr, yb);
        const float dx = lr - ll;
        const float dy = lb - lt;
        const float d1 = 0.70710678f * (lbr - ltl);
        const float d2 = 0.70710678f * (lbl - ltr);
        const float lap = lc - 0.25f * (ll + lr + lt + lb);
        const float rg_dx =
            0.5f * ((R[y * w + xr] - G[y * w + xr]) -
                    (R[y * w + xl] - G[y * w + xl]));
        const float rg_dy =
            0.5f * ((R[yb * w + x] - G[yb * w + x]) -
                    (R[yt * w + x] - G[yt * w + x]));
        const float by_dx =
            0.5f * ((0.5f * (R[y * w + xr] + G[y * w + xr]) - B[y * w + xr]) -
                    (0.5f * (R[y * w + xl] + G[y * w + xl]) - B[y * w + xl]));
        const float by_dy =
            0.5f * ((0.5f * (R[yb * w + x] + G[yb * w + x]) - B[yb * w + x]) -
                    (0.5f * (R[yt * w + x] + G[yt * w + x]) - B[yt * w + x]));
        f[0] = rv;
        f[1] = gv;
        f[2] = bv;
        f[3] = std::max(0.0f, dx);
        f[4] = std::max(0.0f, -dx);
        f[5] = std::max(0.0f, dy);
        f[6] = std::max(0.0f, -dy);
        f[7] = std::max(0.0f, d1);
        f[8] = std::max(0.0f, -d1);
        f[9] = std::max(0.0f, d2);
        f[10] = std::max(0.0f, -d2);
        f[11] = std::abs(lap) +
                0.35f * (std::abs(rg_dx) + std::abs(rg_dy) +
                         std::abs(by_dx) + std::abs(by_dy));
      }
    }
  }

  void SaveSrgbPatch(size_t s, size_t x0, size_t y0, size_t x1, size_t y1,
                     Checkpoint* cp) const {
    if (cp == nullptr || x0 >= x1 || y0 >= y1) return;
    const ScaleData& sd = scales_[s];
    const size_t len = x1 - x0;
    const size_t off = cp->srgb_data.size();
    cp->srgb_patches.push_back({s, x0, y0, x1, y1, off});
    cp->srgb_data.resize(off + 3 * len * (y1 - y0));
    size_t at = off;
    for (size_t c = 0; c < 3; ++c) {
      for (size_t y = y0; y < y1; ++y) {
        memcpy(cp->srgb_data.data() + at,
               sd.test_srgb[c].data() + y * sd.w + x0, len * sizeof(float));
        at += len;
      }
    }
  }

  void PoolScaleTiles(size_t s, size_t tx0, size_t ty0, size_t tx1, size_t ty1,
                      Checkpoint* cp) {
    if (tx0 >= tx1 || ty0 >= ty1) return;
    ScaleData& sd = scales_[s];
    const size_t px0 = tx0 * kTile;
    const size_t py0 = ty0 * kTile;
    const size_t px1 = std::min(sd.w, tx1 * kTile);
    const size_t py1 = std::min(sd.h, ty1 * kTile);
    if (px0 >= px1 || py0 >= py1) return;

    const size_t ex0 = (px0 > 0) ? (px0 - 1) : 0;
    const size_t ey0 = (py0 > 0) ? (py0 - 1) : 0;
    const size_t ex1 = std::min(sd.w, px1 + 1);
    const size_t ey1 = std::min(sd.h, py1 + 1);
    const size_t ew = ex1 - ex0;
    const size_t eh = ey1 - ey0;
    const size_t pw = px1 - px0;

    sc_ref_f_.resize(ew * eh * kNumFeats);
    sc_test_f_.resize(ew * eh * kNumFeats);
    ExtractFeatsRect(sd.ref_srgb, sd.w, sd.h, ex0, ey0, ex1, ey1,
                     sc_ref_f_.data());
    ExtractFeatsRect(sd.test_srgb, sd.w, sd.h, ex0, ey0, ex1, ey1,
                     sc_test_f_.data());

    sc_hm_r_.resize(pw * eh * kNumFeats);
    sc_hp_r_.resize(pw * eh * kNumFeats);
    sc_hm_t_.resize(pw * eh * kNumFeats);
    sc_hp_t_.resize(pw * eh * kNumFeats);

    for (size_t y = ey0; y < ey1; ++y) {
      const size_t ey = y - ey0;
      for (size_t x = px0; x < px1; ++x) {
        const size_t sx = x - ex0;
        const float* rl =
            (x > 0) ? (sc_ref_f_.data() + (ey * ew + sx - 1) * kNumFeats)
                    : nullptr;
        const float* rc = sc_ref_f_.data() + (ey * ew + sx) * kNumFeats;
        const float* rr =
            (x + 1 < sd.w) ? (sc_ref_f_.data() + (ey * ew + sx + 1) * kNumFeats)
                           : nullptr;
        const float* tl =
            (x > 0) ? (sc_test_f_.data() + (ey * ew + sx - 1) * kNumFeats)
                    : nullptr;
        const float* tc = sc_test_f_.data() + (ey * ew + sx) * kNumFeats;
        const float* tr =
            (x + 1 < sd.w)
                ? (sc_test_f_.data() + (ey * ew + sx + 1) * kNumFeats)
                : nullptr;
        const size_t hoff = (ey * pw + (x - px0)) * kNumFeats;
        for (size_t k = 0; k < kNumFeats; ++k) {
          const float r_l = (rl != nullptr) ? rl[k] : 0.0f;
          const float r_c = rc[k];
          const float r_r = (rr != nullptr) ? rr[k] : 0.0f;
          const float t_l = (tl != nullptr) ? tl[k] : 0.0f;
          const float t_c = tc[k];
          const float t_r = (tr != nullptr) ? tr[k] : 0.0f;
          sc_hm_r_[hoff + k] = 0.25f * (r_l + r_r) + 0.50f * r_c;
          sc_hp_r_[hoff + k] =
              0.25f * (r_l * r_l + r_r * r_r) + 0.50f * (r_c * r_c);
          sc_hm_t_[hoff + k] = 0.25f * (t_l + t_r) + 0.50f * t_c;
          sc_hp_t_[hoff + k] =
              0.25f * (t_l * t_l + t_r * t_r) + 0.50f * (t_c * t_c);
        }
      }
    }

    constexpr float kFeatW[kNumFeats] = {
        1.00f, 1.00f, 1.00f, 1.35f, 1.35f, 1.35f,
        1.35f, 1.25f, 1.25f, 1.25f, 1.25f, 1.40f,
    };
    const float pt_mix = (s == 0) ? 0.0f : 0.65f;
    const float wd_mix = 1.0f - pt_mix;

    for (size_t ty = ty0; ty < ty1; ++ty) {
      const size_t y_lo = ty * kTile;
      const size_t y_hi = std::min(sd.h, y_lo + kTile);
      for (size_t tx = tx0; tx < tx1; ++tx) {
        const size_t x_lo = tx * kTile;
        const size_t x_hi = std::min(sd.w, x_lo + kTile);
        const size_t tidx = ty * sd.tw + tx;
        if (cp != nullptr) {
          cp->tile_saves.push_back(
              {static_cast<uint8_t>(s), static_cast<uint32_t>(tidx),
               sd.tile_w2[tidx]});
        }
        double sum_w2 = 0.0;
        for (size_t y = y_lo; y < y_hi; ++y) {
          const size_t ey = y - ey0;
          const bool has_t = (y > 0);
          const bool has_b = (y + 1 < sd.h);
          for (size_t x = x_lo; x < x_hi; ++x) {
            const size_t px = x - px0;
            const size_t sx = x - ex0;
            const size_t c_off = (ey * pw + px) * kNumFeats;
            const size_t t_off = has_t ? ((ey - 1) * pw + px) * kNumFeats : 0;
            const size_t b_off = has_b ? ((ey + 1) * pw + px) * kNumFeats : 0;
            const float* f_ref = sc_ref_f_.data() + (ey * ew + sx) * kNumFeats;
            const float* f_tst = sc_test_f_.data() + (ey * ew + sx) * kNumFeats;
            float pix_loss = 0.0f;
            for (size_t k = 0; k < kNumFeats; ++k) {
              const float mr =
                  0.50f * sc_hm_r_[c_off + k] +
                  0.25f * ((has_t ? sc_hm_r_[t_off + k] : 0.0f) +
                           (has_b ? sc_hm_r_[b_off + k] : 0.0f));
              const float pr =
                  0.50f * sc_hp_r_[c_off + k] +
                  0.25f * ((has_t ? sc_hp_r_[t_off + k] : 0.0f) +
                           (has_b ? sc_hp_r_[b_off + k] : 0.0f));
              const float mt =
                  0.50f * sc_hm_t_[c_off + k] +
                  0.25f * ((has_t ? sc_hm_t_[t_off + k] : 0.0f) +
                           (has_b ? sc_hm_t_[b_off + k] : 0.0f));
              const float pt =
                  0.50f * sc_hp_t_[c_off + k] +
                  0.25f * ((has_t ? sc_hp_t_[t_off + k] : 0.0f) +
                           (has_b ? sc_hp_t_[b_off + k] : 0.0f));
              const float sr_v = std::sqrt(std::max(0.0f, pr - mr * mr));
              const float st_v = std::sqrt(std::max(0.0f, pt - mt * mt));
              const float dm = mr - mt;
              const float ds = sr_v - st_v;
              float l_k = wd_mix * (dm * dm + ds * ds);
              if (pt_mix > 0.0f) {
                const float df = f_ref[k] - f_tst[k];
                l_k += pt_mix * df * df;
              }
              pix_loss += kFeatW[k] * l_k;
            }
            sum_w2 += static_cast<double>(pix_loss);
          }
        }
        sd.total_w2 += (sum_w2 - sd.tile_w2[tidx]);
        sd.tile_w2[tidx] = sum_w2;
      }
    }
  }

  size_t w0_ = 0;
  size_t h0_ = 0;
  ScaleData scales_[kNumScales];
  std::vector<float> sc_ds_tmp_;
  std::vector<float> sc_ref_f_;
  std::vector<float> sc_test_f_;
  std::vector<float> sc_hm_r_;
  std::vector<float> sc_hp_r_;
  std::vector<float> sc_hm_t_;
  std::vector<float> sc_hp_t_;
};

float ComputeLocalBlockEpfError(const Image3F& a, const Image3F& b,
                                const ImageF& mask1x1, size_t by, size_t bx,
                                float asym_edge_deficit_w,
                                float* JXL_RESTRICT out_bdry_err,
                                float* JXL_RESTRICT out_grad_a,
                                float* JXL_RESTRICT out_grad_b) {
  const Rect rect(bx * kBlockDim, by * kBlockDim, kBlockDim, kBlockDim,
                  a.xsize(), a.ysize());
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  const size_t x0 = rect.x0();
  const size_t y0 = rect.y0();
  const bool has_left = (out_bdry_err != nullptr) && (x0 > 0);
  const bool has_right = (out_bdry_err != nullptr) && (x0 + xsize < a.xsize());
  const bool has_top = (out_bdry_err != nullptr) && (y0 > 0);
  const bool has_bottom = (out_bdry_err != nullptr) && (y0 + ysize < a.ysize());
  float err2_masked[3] = {0.0f, 0.0f, 0.0f};
  float err2_unmasked[3] = {0.0f, 0.0f, 0.0f};
  float bdry_err2[3] = {0.0f, 0.0f, 0.0f};
  float grad_a2[3] = {0.0f, 0.0f, 0.0f};
  float grad_b2[3] = {0.0f, 0.0f, 0.0f};
  float sum_mask2 = 0.0f;
  const bool full_8x8 = (xsize == 8 && ysize == 8);
  float cell_a[3][4] = {};
  float cell_b[3][4] = {};
  float cell_a2[3][4] = {};
  float cell_b2[3][4] = {};
  float cell_e2[3][4] = {};
  static constexpr float kSsimScale[3] = {14.0f, 1.0f, 1.0f};
  for (size_t y = 0; y < ysize; ++y) {
    const float* row_a[3] = {
        rect.ConstPlaneRow(a, 0, y),
        rect.ConstPlaneRow(a, 1, y),
        rect.ConstPlaneRow(a, 2, y),
    };
    const float* row_b[3] = {
        rect.ConstPlaneRow(b, 0, y),
        rect.ConstPlaneRow(b, 1, y),
        rect.ConstPlaneRow(b, 2, y),
    };
    const float* row_a_next[3] = {
        (y + 1 < ysize || has_bottom) ? a.ConstPlaneRow(0, y0 + y + 1) + x0
                                      : nullptr,
        (y + 1 < ysize || has_bottom) ? a.ConstPlaneRow(1, y0 + y + 1) + x0
                                      : nullptr,
        (y + 1 < ysize || has_bottom) ? a.ConstPlaneRow(2, y0 + y + 1) + x0
                                      : nullptr,
    };
    const float* row_b_next[3] = {
        (y + 1 < ysize || has_bottom) ? b.ConstPlaneRow(0, y0 + y + 1) + x0
                                      : nullptr,
        (y + 1 < ysize || has_bottom) ? b.ConstPlaneRow(1, y0 + y + 1) + x0
                                      : nullptr,
        (y + 1 < ysize || has_bottom) ? b.ConstPlaneRow(2, y0 + y + 1) + x0
                                      : nullptr,
    };
    const float* row_a_prev[3] = {
        (y == 0 && has_top) ? a.ConstPlaneRow(0, y0 - 1) + x0 : nullptr,
        (y == 0 && has_top) ? a.ConstPlaneRow(1, y0 - 1) + x0 : nullptr,
        (y == 0 && has_top) ? a.ConstPlaneRow(2, y0 - 1) + x0 : nullptr,
    };
    const float* row_b_prev[3] = {
        (y == 0 && has_top) ? b.ConstPlaneRow(0, y0 - 1) + x0 : nullptr,
        (y == 0 && has_top) ? b.ConstPlaneRow(1, y0 - 1) + x0 : nullptr,
        (y == 0 && has_top) ? b.ConstPlaneRow(2, y0 - 1) + x0 : nullptr,
    };
    const float* row_mask = rect.ConstRow(mask1x1, y);
    for (size_t x = 0; x < xsize; ++x) {
      const float mask = row_mask[x];
      const float mask2 = mask * mask;
      sum_mask2 += mask2;
      if (full_8x8) {
        const size_t q = ((y >> 2) << 1) + (x >> 2);
        const float oa[3] = {kSsimScale[0] * row_a[0][x], row_a[1][x],
                             row_a[2][x] - row_a[1][x]};
        const float ob[3] = {kSsimScale[0] * row_b[0][x], row_b[1][x],
                             row_b[2][x] - row_b[1][x]};
        for (int c = 0; c < 3; ++c) {
          const float oe = oa[c] - ob[c];
          cell_a[c][q] += oa[c];
          cell_b[c][q] += ob[c];
          cell_a2[c][q] += oa[c] * oa[c];
          cell_b2[c][q] += ob[c] * ob[c];
          cell_e2[c][q] += oe * oe;
        }
      }
      for (int i = 0; i < 3; ++i) {
        const float diff = row_a[i][x] - row_b[i][x];
        const float diff2 = diff * diff;
        err2_masked[i] += mask2 * diff2;
        err2_unmasked[i] += diff2;
        if (x + 1 < xsize) {
          const float ga = row_a[i][x + 1] - row_a[i][x];
          const float gb = row_b[i][x + 1] - row_b[i][x];
          grad_a2[i] += ga * ga;
          grad_b2[i] += gb * gb;
        }
        if (y + 1 < ysize) {
          const float ga = row_a_next[i][x] - row_a[i][x];
          const float gb = row_b_next[i][x] - row_b[i][x];
          grad_a2[i] += ga * ga;
          grad_b2[i] += gb * gb;
        }
        if (x == 0 && has_left) {
          const float db = diff - (row_a[i][-1] - row_b[i][-1]);
          bdry_err2[i] += mask2 * db * db;
        }
        if (x + 1 == xsize && has_right) {
          const float db = diff - (row_a[i][xsize] - row_b[i][xsize]);
          bdry_err2[i] += mask2 * db * db;
        }
        if (y == 0 && has_top) {
          const float db = diff - (row_a_prev[i][x] - row_b_prev[i][x]);
          bdry_err2[i] += mask2 * db * db;
        }
        if (y + 1 == ysize && has_bottom) {
          const float db = diff - (row_a_next[i][x] - row_b_next[i][x]);
          bdry_err2[i] += mask2 * db * db;
        }
      }
    }
  }
  const float mean_mask2 = sum_mask2 / static_cast<float>(xsize * ysize);
  constexpr float kUnmaskedMix = 0.22f;
  float err2[3];
  for (int i = 0; i < 3; ++i) {
    err2[i] = (1.0f - kUnmaskedMix) * err2_masked[i] +
              kUnmaskedMix * mean_mask2 * err2_unmasked[i];
  }
  static constexpr float kW[3] = {12.3394453f, 1.0f, 0.2f};
  if (out_bdry_err != nullptr) {
    *out_bdry_err =
        kW[0] * bdry_err2[0] + kW[1] * bdry_err2[1] + kW[2] * bdry_err2[2];
  }
  if (out_grad_a != nullptr && out_grad_b != nullptr) {
    *out_grad_a = mean_mask2 * (kW[0] * grad_a2[0] + kW[1] * grad_a2[1] +
                                kW[2] * grad_a2[2]);
    *out_grad_b = mean_mask2 * (kW[0] * grad_b2[0] + kW[1] * grad_b2[1] +
                                kW[2] * grad_b2[2]);
  }
  float retval = kW[0] * err2[0] + kW[1] * err2[1] + kW[2] * err2[2];
  if (full_8x8) {
    static constexpr float kW_ssim[3] = {0.06f, 1.00f, 0.45f};
    float ssim_err = 0.0f;
    for (int c = 0; c < 3; ++c) {
      float mu_a[4];
      float mu_b[4];
      float c_err = 0.0f;
      for (size_t q = 0; q < 4; ++q) {
        mu_a[q] = 0.0625f * cell_a[c][q];
        mu_b[q] = 0.0625f * cell_b[c][q];
        const float mu_e = mu_a[q] - mu_b[q];
        const float sig_a2 =
            std::max(0.0f, 0.0625f * cell_a2[c][q] - mu_a[q] * mu_a[q]);
        const float sig_b2 =
            std::max(0.0f, 0.0625f * cell_b2[c][q] - mu_b[q] * mu_b[q]);
        const float sig_e2 =
            std::max(0.0f, 0.0625f * cell_e2[c][q] - mu_e * mu_e);
        const float denom = sig_a2 + sig_b2 + 0.0009f;
        const float two_cov_c2 = std::max(0.0f, denom - sig_e2);
        const float d_ssim = (sig_e2 + mu_e * mu_e * two_cov_c2) / denom;
        const float sd_a = std::sqrt(sig_a2);
        const float sd_b = std::sqrt(sig_b2);
        const float edge_d = (sd_b - sd_a) / (1.0f + sd_a);
        const float edge_pen =
            (edge_d < 0.0f) ? (asym_edge_deficit_w * edge_d * edge_d)
                            : (0.45f * edge_d * edge_d);
        c_err += 16.0f * (900.0f * d_ssim + 750.0f * edge_pen);
      }
      const float mu_a8 = 0.25f * (mu_a[0] + mu_a[1] + mu_a[2] + mu_a[3]);
      const float mu_b8 = 0.25f * (mu_b[0] + mu_b[1] + mu_b[2] + mu_b[3]);
      const float mu_e8 = mu_a8 - mu_b8;
      float s_a2 = 0.0f;
      float s_b2 = 0.0f;
      float s_e2 = 0.0f;
      for (size_t q = 0; q < 4; ++q) {
        const float da = mu_a[q] - mu_a8;
        const float db = mu_b[q] - mu_b8;
        const float de = da - db;
        s_a2 += da * da;
        s_b2 += db * db;
        s_e2 += de * de;
      }
      const float denom8 = 0.25f * (s_a2 + s_b2) + 0.0009f;
      const float two_cov8 = std::max(0.0f, denom8 - 0.25f * s_e2);
      const float d_ssim8 = (0.25f * s_e2 + mu_e8 * mu_e8 * two_cov8) / denom8;
      c_err = 0.45f * c_err + 0.55f * (64.0f * 900.0f * d_ssim8);
      ssim_err += kW_ssim[c] * c_err;
    }
    retval = 0.45f * retval + 0.55f * ssim_err;
  }
  return retval;
}

double EstimateEpfSharpnessBits(const ImageB& sharp, size_t xb, size_t yb) {
  if (xb == 0 || yb == 0) return 0.0;
  static constexpr uint8_t kLut[8] = {0, 0, 1, 1, 2, 2, 3, 3};
  size_t val_cnt[8] = {};
  size_t ctx4[4][8] = {};
  size_t tot4[4] = {};
  size_t ctx16[16][8] = {};
  size_t tot16[16] = {};
  for (size_t by = 0; by < yb; ++by) {
    const uint8_t* row = sharp.ConstRow(by);
    const uint8_t* prev = (by > 0) ? sharp.ConstRow(by - 1) : nullptr;
    for (size_t bx = 0; bx < xb; ++bx) {
      const uint8_t v = row[bx] & 7u;
      const uint8_t t = (by > 0) ? (prev[bx] & 7u) : 0;
      const uint8_t l = (bx > 0) ? (row[bx - 1] & 7u) : 0;
      const size_t c4 = (t > 0 ? 2u : 0u) + (l > 0 ? 1u : 0u);
      const size_t c16 = (static_cast<size_t>(kLut[t]) << 2) | kLut[l];
      ++val_cnt[v];
      ++ctx4[c4][v];
      ++tot4[c4];
      ++ctx16[c16][v];
      ++tot16[c16];
    }
  }
  size_t num_unique = 0;
  for (size_t v = 0; v < 8; ++v) {
    if (val_cnt[v] > 0) ++num_unique;
  }
  if (num_unique <= 1) return 0.0;

  double bits4 = 96.0;
  for (size_t c = 0; c < 4; ++c) {
    if (tot4[c] == 0) continue;
    const double inv = 1.0 / static_cast<double>(tot4[c]);
    for (size_t v = 0; v < 8; ++v) {
      if (ctx4[c][v] > 0) {
        bits4 += 2.0 - 1.25 * static_cast<double>(ctx4[c][v]) *
                           std::log2(static_cast<double>(ctx4[c][v]) * inv);
      }
    }
  }
  double bits16 = 136.0;
  for (size_t c = 0; c < 16; ++c) {
    if (tot16[c] == 0) continue;
    const double inv = 1.0 / static_cast<double>(tot16[c]);
    for (size_t v = 0; v < 8; ++v) {
      if (ctx16[c][v] > 0) {
        bits16 += 2.0 - 1.25 * static_cast<double>(ctx16[c][v]) *
                            std::log2(static_cast<double>(ctx16[c][v]) * inv);
      }
    }
  }
  return std::min(bits4, bits16);
}

double EstimateLoopFilterBits(const LoopFilter& lf) {
  const bool is_default =
      lf.gab && !lf.gab_custom && lf.epf_iters == 2 &&
      !lf.epf_sharp_custom && !lf.epf_weight_custom &&
      !lf.epf_sigma_custom && lf.extensions == 0;
  if (is_default) return 1.0;
  double bits = 2.0;  // all_default=0 + gab
  if (lf.gab) {
    bits += 1.0 + (lf.gab_custom ? 96.0 : 0.0);
  }
  bits += 2.0;  // epf_iters (2 bits)
  if (lf.epf_iters > 0) {
    bits += 1.0 + (lf.epf_sharp_custom ? 128.0 : 0.0);
    bits += 1.0 + (lf.epf_weight_custom ? 80.0 : 0.0);
    bits += 1.0 + (lf.epf_sigma_custom ? 64.0 : 0.0);
  }
  bits += 2.0;  // BeginExtensions(0)
  return bits;
}

inline int32_t CflTileGradPred(const ImageSB& map, size_t x, size_t y) {
  const size_t lx = x & 31u;
  const size_t ly = y & 31u;
  if (lx == 0 && ly == 0) return 0;
  const int8_t* row_cur = map.ConstRow(y);
  const int8_t* row_top = (ly > 0) ? map.ConstRow(y - 1) : nullptr;
  if (lx == 0) return static_cast<int32_t>(row_top[x]);
  if (ly == 0) return static_cast<int32_t>(row_cur[x - 1]);
  const int32_t left = static_cast<int32_t>(row_cur[x - 1]);
  const int32_t top = static_cast<int32_t>(row_top[x]);
  const int32_t topleft = static_cast<int32_t>(row_top[x - 1]);
  const int32_t m = std::min(top, left);
  const int32_t M = std::max(top, left);
  return std::min(M, std::max(m, top + left - topleft));
}

inline void AccumCflHybrid420(int32_t signed_res, uint32_t hist[64],
                              uint64_t* extra_bits) {
  const uint32_t u = static_cast<uint32_t>(
      (signed_res >= 0) ? (2 * signed_res) : (-2 * signed_res - 1));
  uint32_t token = u;
  if (u >= 16u) {
    const uint32_t n = FloorLog2Nonzero(u);
    const uint32_t nbits = n - 2u;
    *extra_bits += nbits;
    token =
        std::min<uint32_t>(63u, 16u + ((n - 4u) << 2u) + ((u >> nbits) & 3u));
  }
  hist[token]++;
}

double EstimateCflMapBits(const ImageSB& map, size_t w, size_t h) {
  if (w == 0 || h == 0) return 0.0;
  const int8_t v0 = map.ConstRow(0)[0];
  bool all_const = true;
  uint32_t hist_z[64] = {};
  uint32_t hist_g[64] = {};
  uint64_t extra_z = 0;
  uint64_t extra_g = 0;
  for (size_t y = 0; y < h; ++y) {
    const int8_t* row = map.ConstRow(y);
    for (size_t x = 0; x < w; ++x) {
      const int32_t cur = static_cast<int32_t>(row[x]);
      if (row[x] != v0) all_const = false;
      AccumCflHybrid420(cur, hist_z, &extra_z);
      AccumCflHybrid420(cur - CflTileGradPred(map, x, y), hist_g, &extra_g);
    }
  }
  if (all_const) {
    return (v0 == 0) ? 2.0
                     : (8.0 + std::log2(1.0 + std::abs(static_cast<int>(v0))));
  }
  const double n = static_cast<double>(w * h);
  auto calc_bits = [&](const uint32_t hist[64], uint64_t extra) -> double {
    double b = 8.0 + static_cast<double>(extra) + n * std::log2(n);
    for (size_t i = 0; i < 64; ++i) {
      if (hist[i] > 0) {
        b += 1.5 - static_cast<double>(hist[i]) *
                       std::log2(static_cast<double>(hist[i]));
      }
    }
    return b;
  };
  return std::min(calc_bits(hist_z, extra_z), calc_bits(hist_g, extra_g));
}

void ApplyLocalGabBox(bool gab, const float gab_w_f[3][3],
                      const Image3F* feat_ptr, size_t xsize, size_t ysize,
                      const Image3F& cur_pre_gab, const Image3F& patch_src,
                      size_t fx0, size_t fy0, size_t lw, size_t lh,
                      Image3F* dst_xyb) {
  for (size_t iy = 0; iy < lh; ++iy) {
    const size_t y = fy0 + iy;
    const size_t yt = static_cast<size_t>(
        Mirror(static_cast<int64_t>(y) - 1, static_cast<int64_t>(ysize)));
    const size_t yb = static_cast<size_t>(
        Mirror(static_cast<int64_t>(y) + 1, static_cast<int64_t>(ysize)));
    const bool yt_in = (yt >= fy0 && yt < fy0 + lh);
    const bool yb_in = (yb >= fy0 && yb < fy0 + lh);
    for (size_t c = 0; c < 3; ++c) {
      float* JXL_RESTRICT out_r = dst_xyb->PlaneRow(c, y) + fx0;
      if (!gab) {
        memcpy(out_r, patch_src.ConstPlaneRow(c, y) + fx0, lw * sizeof(float));
        continue;
      }
      const float* rt = yt_in ? patch_src.ConstPlaneRow(c, yt)
                              : cur_pre_gab.ConstPlaneRow(c, yt);
      const float* rm_p = patch_src.ConstPlaneRow(c, y);
      const float* rm_c = cur_pre_gab.ConstPlaneRow(c, y);
      const float* rb = yb_in ? patch_src.ConstPlaneRow(c, yb)
                              : cur_pre_gab.ConstPlaneRow(c, yb);
      const float w0 = gab_w_f[c][0];
      const float w1 = gab_w_f[c][1];
      const float w2 = gab_w_f[c][2];
      for (size_t ix = 0; ix < lw; ++ix) {
        const size_t x = fx0 + ix;
        const size_t xl = (x > 0) ? (x - 1) : 0;
        const size_t xr = (x + 1 < xsize) ? (x + 1) : (xsize - 1);
        const bool xl_in = (xl >= fx0 && xl < fx0 + lw);
        const bool xr_in = (xr >= fx0 && xr < fx0 + lw);
        const float s0 = rm_p[x];
        const float s1 = (xl_in ? rm_p[xl] : rm_c[xl]) +
                         (xr_in ? rm_p[xr] : rm_c[xr]) + rt[x] + rb[x];
        const float tl = (yt_in && xl_in) ? patch_src.ConstPlaneRow(c, yt)[xl]
                                          : cur_pre_gab.ConstPlaneRow(c, yt)[xl];
        const float tr = (yt_in && xr_in) ? patch_src.ConstPlaneRow(c, yt)[xr]
                                          : cur_pre_gab.ConstPlaneRow(c, yt)[xr];
        const float bl = (yb_in && xl_in) ? patch_src.ConstPlaneRow(c, yb)[xl]
                                          : cur_pre_gab.ConstPlaneRow(c, yb)[xl];
        const float br = (yb_in && xr_in) ? patch_src.ConstPlaneRow(c, yb)[xr]
                                          : cur_pre_gab.ConstPlaneRow(c, yb)[xr];
        out_r[ix] = s0 * w0 + s1 * w1 + (tl + tr + bl + br) * w2;
      }
    }
  }
  if (feat_ptr != nullptr) {
    for (size_t c = 0; c < 3; ++c) {
      for (size_t iy = 0; iy < lh; ++iy) {
        float* JXL_RESTRICT out_r = dst_xyb->PlaneRow(c, fy0 + iy) + fx0;
        const float* JXL_RESTRICT rf =
            feat_ptr->ConstPlaneRow(c, fy0 + iy) + fx0;
        for (size_t ix = 0; ix < lw; ++ix) out_r[ix] += rf[ix];
      }
    }
  }
}

struct GroupAcsDpComponent {
  size_t group = 0;
  uint8_t bx0 = 0;
  uint8_t by0 = 0;
  uint8_t bw = 1;
  uint8_t bh = 1;
  double dp_gain = 0.0;
  uint32_t new_tables_mask = 0;
  bool committed = false;
  std::vector<AcsSubSpec> specs;
};

struct GroupAcsDpWorkspace {
  static constexpr size_t kMaxIv = 32 * 33 / 2;
  AcsEvalWorkspace eval_ws;
  uint16_t iv_idx[32][33] = {};
  std::vector<float> cost_flat;
  std::vector<uint16_t> split_flat;
  uint32_t best_1x1_spec_idx[32][32] = {};
  float best_1x1_cost[32][32] = {};
  float prefix_base[33][33] = {};
  float prefix_cur_dist[33][33] = {};
  uint32_t leaf_spec_idx[32][32][9][9] = {};
  float pre_leaf_cost[32][32][9][9] = {};
  uint32_t pre_leaf_spec_idx[32][32][9][9] = {};
  int32_t blk_q[32][32] = {};
  int8_t blk_rdoq[32][32] = {};
  uint32_t blk_act[32][32] = {};
  float blk_bq[32][32][3][4] = {};
  float w_4m[32][32] = {};
  float cur_blk_dist[32][32] = {};
  float cur_blk_grad[32][32] = {};
  uint16_t cur_tr_at[32][32] = {};
  uint8_t dp_cov[32][32] = {};
  uint16_t dp_spec_at[32][32] = {};
  uint8_t diff_blk[32][32] = {};

  struct CandBlock {
    uint8_t x0, y0, w, h;
    uint32_t spec_idx;
    float gain;
    float rate;
  };
  std::vector<CandBlock> cands;

  struct LocalCand {
    uint64_t mask;
    float gain;
    float rate;
    uint32_t spec_idx;
  };
  LocalCand local_cands[352] = {};

  struct SubRect {
    uint8_t x0, y0, w, h;
  };
  SubRect stack[512] = {};
  uint16_t ng_chosen[64] = {};
  uint16_t ng_cur_chosen[64] = {};

  std::vector<float> grad_samples;
  std::vector<float> cur_tr_costs;
  std::vector<AcsSubSpec> spec_pool;
  std::vector<float> spec_cost_pool;
  std::vector<uint32_t> dp_spec_indices;
};

Status SolveClosedLoopGroupAcsDP(
    const PassesEncoderState* enc_state, const Image3F& opsin, const Rect& rect,
    size_t tiles_x, size_t tile_blocks, const AcGroupBlockCache& cache,
    bool gab, const float gab_w_f[3][3], const Image3F* feat_ptr,
    const Image3F& cur_pre_gab, Image3F* plain_pre_gab, Image3F* fine_pre_gab,
    const ImageSB& rdo_ytox_map, const ImageSB& rdo_ytob_map,
    const double* blk_err_cur, double mean_blk_err_2f,
    const size_t quant_table_counts[kNumQuantTables],
    GroupAcsDpWorkspace* ws, std::vector<GroupAcsDpComponent>* out_comps) {
  out_comps->clear();
  const size_t rx = cache.gw_blocks;
  const size_t ry = cache.gh_blocks;
  if (rx == 0 || ry == 0 || cache.transforms.empty()) return true;

  const PassesSharedState& shared = enc_state->shared;
  const CompressParams& cparams = enc_state->cparams;
  const AcTokenCostModel& tcm = *enc_state->token_cost_model;
  const size_t xsize = shared.frame_dim.xsize;
  const size_t ysize = shared.frame_dim.ysize;
  const size_t xsize_blocks = shared.frame_dim.xsize_blocks;
  const size_t ysize_blocks = shared.frame_dim.ysize_blocks;
  const size_t total_frame_blocks = xsize_blocks * ysize_blocks;
  const Rect bgr = shared.frame_dim.BlockGroupRect(cache.group_idx);
  const auto& qm_encs = shared.matrices.encodings();
  const bool has_rdo_cfl = (rdo_ytox_map.xsize() > 0 && rdo_ytob_map.xsize() > 0);

  ws->cost_flat.assign(
      GroupAcsDpWorkspace::kMaxIv * GroupAcsDpWorkspace::kMaxIv, 1e30f);
  ws->split_flat.assign(
      GroupAcsDpWorkspace::kMaxIv * GroupAcsDpWorkspace::kMaxIv, 0);
  ws->spec_pool.clear();
  ws->spec_cost_pool.clear();
  ws->spec_pool.reserve(4096);
  ws->spec_cost_pool.reserve(4096);
  ws->cands.clear();
  ws->cands.reserve(4096);

  for (size_t y = 0; y < ry; ++y) {
    for (size_t x = 0; x < rx; ++x) {
      ws->cur_tr_at[y][x] = 0xFFFFu;
      for (size_t h = 0; h <= 8; ++h) {
        for (size_t w = 0; w <= 8; ++w) {
          ws->leaf_spec_idx[y][x][h][w] = 0xFFFFFFFFu;
          ws->pre_leaf_cost[y][x][h][w] = 1e30f;
          ws->pre_leaf_spec_idx[y][x][h][w] = 0xFFFFFFFFu;
        }
      }
    }
  }

  for (size_t ti = 0; ti < cache.transforms.size(); ++ti) {
    const AcTransformCache& tr = cache.transforms[ti];
    const float ratio = static_cast<float>(tr.cur_quant_ac) /
                        static_cast<float>(tr.base_quant_ac);
    for (size_t iy = 0; iy < tr.cov_y; ++iy) {
      for (size_t ix = 0; ix < tr.cov_x; ++ix) {
        const size_t y = tr.by + iy;
        const size_t x = tr.bx + ix;
        if (y >= ry || x >= rx) continue;
        ws->cur_tr_at[y][x] = static_cast<uint16_t>(ti);
        ws->blk_q[y][x] = tr.cur_quant_ac;
        ws->blk_rdoq[y][x] = tr.cur_rdoq_level;
        if (ix == 0 && iy == 0) {
          ws->blk_act[y][x] = tr.act_bits;
          for (size_t c = 0; c < 3; ++c) {
            for (size_t s = 0; s < 4; ++s) {
              ws->blk_bq[y][x][c][s] = tr.base_band_q[c][s] * ratio;
            }
          }
        } else {
          const float bq10_raw =
              bgr.ConstRow(enc_state->band_qf[1][0], y)[x];
          uint32_t abits = 0;
          memcpy(&abits, &bq10_raw, sizeof(abits));
          ws->blk_act[y][x] = abits & 31u;
          for (size_t c = 0; c < 3; ++c) {
            for (size_t s = 0; s < 4; ++s) {
              ws->blk_bq[y][x][c][s] =
                  bgr.ConstRow(enc_state->band_qf[c][s], y)[x] * ratio;
            }
          }
        }
      }
    }
  }

  constexpr float epf_bdry_w_f = 0.076615788f;
  ws->grad_samples.clear();
  ws->grad_samples.reserve(rx * ry);
  const double inv_mean_be = 1.0 / std::max(1e-12, mean_blk_err_2f);
  for (size_t y = 0; y < ry; ++y) {
    for (size_t x = 0; x < rx; ++x) {
      const double be =
          blk_err_cur[(cache.gy0 + y) * xsize_blocks + cache.gx0 + x];
      const float w4 = Clamp1(
          static_cast<float>(0.65 + 0.35 * (be * inv_mean_be)), 0.55f, 1.65f);
      ws->w_4m[y][x] = w4;
      float bdry_b = 0.0f;
      float grad_a = 0.0f;
      const float eloc = ComputeLocalBlockEpfError(
          *enc_state->orig_opsin, *fine_pre_gab,
          enc_state->initial_quant_masking1x1, cache.gy0 + y, cache.gx0 + x,
          1.20f, &bdry_b, &grad_a, nullptr);
      ws->cur_blk_dist[y][x] = w4 * (eloc + epf_bdry_w_f * bdry_b);
      ws->cur_blk_grad[y][x] = grad_a;
      ws->grad_samples.push_back(grad_a);
    }
  }
  float grad_p70 = 0.0f;
  if (!ws->grad_samples.empty()) {
    const size_t idx70 = (ws->grad_samples.size() * 70) / 100;
    std::nth_element(ws->grad_samples.begin(),
                     ws->grad_samples.begin() + idx70,
                     ws->grad_samples.end());
    grad_p70 = ws->grad_samples[idx70];
  }

  for (size_t x = 0; x <= rx; ++x) ws->prefix_cur_dist[0][x] = 0.0f;
  for (size_t y = 0; y < ry; ++y) {
    float rsum = 0.0f;
    ws->prefix_cur_dist[y + 1][0] = 0.0f;
    for (size_t x = 0; x < rx; ++x) {
      rsum += ws->cur_blk_dist[y][x];
      ws->prefix_cur_dist[y + 1][x + 1] =
          ws->prefix_cur_dist[y][x + 1] + rsum;
    }
  }
  auto rect_cur_dist = [&](size_t x0, size_t y0, size_t w, size_t h) -> float {
    return ws->prefix_cur_dist[y0 + h][x0 + w] -
           ws->prefix_cur_dist[y0][x0 + w] -
           ws->prefix_cur_dist[y0 + h][x0] + ws->prefix_cur_dist[y0][x0];
  };

  const float d_target = cparams.butteraugli_distance;
  const float qf_unit = 0.65f * Clamp1(tcm.qf_block_bits / 1.50f, 0.50f, 2.00f);
  const float lam_bits = 0.837818801f * tcm.bits_to_cost_mul;
  const float expected_table_uses = Clamp1(
      static_cast<float>(total_frame_blocks) / 192.0f, 1.25f, 5.5f);

  auto push_spec = [&](const AcsSubSpec& sp, float cost) -> uint32_t {
    const uint32_t idx = static_cast<uint32_t>(ws->spec_pool.size());
    ws->spec_pool.push_back(sp);
    ws->spec_cost_pool.push_back(cost);
    return idx;
  };

  auto make_existing_spec = [&](const AcTransformCache& tr) -> AcsSubSpec {
    AcsSubSpec sp;
    sp.bx = tr.bx;
    sp.by = tr.by;
    sp.acs_raw = tr.acs_raw;
    sp.rdoq_lvl = tr.cur_rdoq_level;
    sp.ytox_q = tr.cur_ytox_q;
    sp.ytob_q = tr.cur_ytob_q;
    sp.quant_ac = tr.cur_quant_ac;
    sp.act_bits = tr.act_bits;
    const float ratio = static_cast<float>(tr.cur_quant_ac) /
                        static_cast<float>(tr.base_quant_ac);
    for (size_t c = 0; c < 3; ++c) {
      for (size_t s = 0; s < 4; ++s) {
        sp.band_q[c][s] = tr.base_band_q[c][s] * ratio;
      }
    }
    return sp;
  };

  auto make_cand_spec = [&](size_t x0, size_t y0, size_t w, size_t h,
                            AcStrategyType type) -> AcsSubSpec {
    static constexpr float kBandRescaleMix[4] = {0.72f, 0.84f, 0.96f, 1.00f};
    AcsSubSpec sp;
    sp.bx = static_cast<uint8_t>(x0);
    sp.by = static_cast<uint8_t>(y0);
    sp.acs_raw = static_cast<uint8_t>(type);
    const size_t cfl_tx = (cache.gx0 + x0) / kColorTileDimInBlocks;
    const size_t cfl_ty = (cache.gy0 + y0) / kColorTileDimInBlocks;
    sp.ytox_q = has_rdo_cfl ? rdo_ytox_map.ConstRow(cfl_ty)[cfl_tx]
                            : shared.cmap.ytox_map.ConstRow(cfl_ty)[cfl_tx];
    sp.ytob_q = has_rdo_cfl ? rdo_ytob_map.ConstRow(cfl_ty)[cfl_tx]
                            : shared.cmap.ytob_map.ConstRow(cfl_ty)[cfl_tx];

    int32_t q_max = 1;
    double q_sq_sum = 0.0;
    int8_t rdoq_best = -127;
    uint32_t best_act = ws->blk_act[y0][x0];
    size_t distinct_trs = 0;
    uint16_t first_tr = ws->cur_tr_at[y0][x0];
    for (size_t iy = 0; iy < h; ++iy) {
      for (size_t ix = 0; ix < w; ++ix) {
        const int32_t qv = ws->blk_q[y0 + iy][x0 + ix];
        q_max = std::max(q_max, qv);
        q_sq_sum += static_cast<double>(qv) * qv;
        rdoq_best = std::max(rdoq_best, ws->blk_rdoq[y0 + iy][x0 + ix]);
        const uint32_t a = ws->blk_act[y0 + iy][x0 + ix];
        if (((a >> 1) & 15u) > ((best_act >> 1) & 15u)) best_act = a;
        if (ws->cur_tr_at[y0 + iy][x0 + ix] != first_tr) ++distinct_trs;
      }
    }
    const float inv_area = 1.0f / static_cast<float>(w * h);
    const float q_rms = std::sqrt(static_cast<float>(q_sq_sum) * inv_area);
    float q_pool = static_cast<float>(q_max);
    if (w * h >= 4 && d_target > 1.54138f) {
      const float mixer =
          std::max(0.0f, 1.0f - (d_target - 1.54138f) * 0.56391f);
      q_pool = mixer * static_cast<float>(q_max) + (1.0f - mixer) * q_rms;
    }
    sp.quant_ac =
        Clamp1<int32_t>(static_cast<int32_t>(std::lround(q_pool)), 1, 255);
    sp.rdoq_lvl = (w * h > 1 && distinct_trs > 0)
                      ? std::max<int8_t>(rdoq_best, -1)
                      : rdoq_best;
    sp.act_bits = best_act;

    const float q_step_ratio =
        static_cast<float>(sp.quant_ac) / std::max(q_pool, 1.0f);
    for (size_t c = 0; c < 3; ++c) {
      for (size_t s = 0; s < 4; ++s) {
        float sq = 0.0f;
        for (size_t iy = 0; iy < h; ++iy) {
          for (size_t ix = 0; ix < w; ++ix) {
            const float old_q =
                std::max(1.0f, static_cast<float>(ws->blk_q[y0 + iy][x0 + ix]));
            const float rescale = q_pool / old_q;
            const float b_rescale =
                (rescale > 1.0f)
                    ? (1.0f + kBandRescaleMix[s] * (rescale - 1.0f))
                    : rescale;
            const float v = ws->blk_bq[y0 + iy][x0 + ix][c][s] * b_rescale;
            sq += v * v;
          }
        }
        sp.band_q[c][s] = std::sqrt(sq * inv_area) * q_step_ratio;
      }
    }
    return sp;
  };

  auto calc_spec_overhead_bits = [&](uint8_t acs_raw) -> float {
    float bits = qf_unit;
    if (acs_raw < 27) bits += tcm.acs_strategy_bits[acs_raw];
    const size_t ti = static_cast<size_t>(kAcStrategyToQuantTableMap[acs_raw]);
    if (ti >= 1 && ti < kNumQuantTables && ti < qm_encs.size() &&
        quant_table_counts[ti] == 0 &&
        qm_encs[ti].mode == QuantEncoding::kQuantModeDCT) {
      const float full_tax =
          3.0f + 48.0f * static_cast<float>(qm_encs[ti].dct_params.num_distance_bands);
      bits += full_tax / expected_table_uses;
    }
    return bits;
  };

  auto eval_cand_cost = [&](size_t x0, size_t y0, size_t w, size_t h,
                            const AcsSubSpec& sp, float* out_cost) -> Status {
    float ac_b = 0.0f;
    JXL_RETURN_IF_ERROR(HWY_DYNAMIC_DISPATCH(EvalAcsBoxCandidateImpl)(
        enc_state, opsin, rect, tiles_x, tile_blocks, cache, x0, y0, w, h, 1,
        &sp, nullptr, nullptr, &ac_b, plain_pre_gab, &ws->eval_ws));
    const size_t fx0 = cache.px0 + x0 * kBlockDim;
    const size_t fy0 = cache.py0 + y0 * kBlockDim;
    const size_t bw_px = std::min(cache.pw_px - x0 * kBlockDim, w * kBlockDim);
    const size_t bh_px = std::min(cache.ph_px - y0 * kBlockDim, h * kBlockDim);
    ApplyLocalGabBox(gab, gab_w_f, feat_ptr, xsize, ysize, cur_pre_gab,
                     *plain_pre_gab, fx0, fy0, bw_px, bh_px, fine_pre_gab);
    float dist = 0.0f;
    for (size_t iy = 0; iy < h; ++iy) {
      for (size_t ix = 0; ix < w; ++ix) {
        float bdry_b = 0.0f;
        const float eloc = ComputeLocalBlockEpfError(
            *enc_state->orig_opsin, *fine_pre_gab,
            enc_state->initial_quant_masking1x1, cache.gy0 + y0 + iy,
            cache.gx0 + x0 + ix, 1.20f, &bdry_b, nullptr, nullptr);
        dist += ws->w_4m[y0 + iy][x0 + ix] * (eloc + epf_bdry_w_f * bdry_b);
      }
    }
    ApplyLocalGabBox(gab, gab_w_f, feat_ptr, xsize, ysize, cur_pre_gab,
                     cur_pre_gab, fx0, fy0, bw_px, bh_px, fine_pre_gab);
    const float cur_d_rect = rect_cur_dist(x0, y0, w, h);
    const float hyst =
        0.966556609f *
        (0.025f * std::max(1.0f, cur_d_rect) +
         0.14f * lam_bits * std::sqrt(static_cast<float>(w * h)));
    *out_cost =
        dist + lam_bits * (ac_b + calc_spec_overhead_bits(sp.acs_raw)) + hyst;
    return true;
  };

  // Pre-price all existing transforms in cache.transforms (0 extra DCTs).
  ws->cur_tr_costs.assign(cache.transforms.size(), 0.0f);
  for (size_t ti = 0; ti < cache.transforms.size(); ++ti) {
    const AcTransformCache& tr = cache.transforms[ti];
    const float dist = rect_cur_dist(tr.bx, tr.by, tr.cov_x, tr.cov_y);
    const float ac_b = tr.slots[tr.cur_slot].bits;
    const float cost =
        dist + lam_bits * (ac_b + calc_spec_overhead_bits(tr.acs_raw));
    ws->cur_tr_costs[ti] = cost;
    if (tr.cov_x <= 8 && tr.cov_y <= 8 &&
        (tr.cov_x > 1 || tr.cov_y > 1) && tr.bx < rx && tr.by < ry) {
      const AcsSubSpec sp = make_existing_spec(tr);
      const uint32_t sidx = push_spec(sp, cost);
      ws->pre_leaf_cost[tr.by][tr.bx][tr.cov_y][tr.cov_x] = cost;
      ws->pre_leaf_spec_idx[tr.by][tr.bx][tr.cov_y][tr.cov_x] = sidx;
    }
  }

  // Price 1x1 strategies across [0, rx) x [0, ry).
  static constexpr AcStrategyType kSub8Types[] = {
      AcStrategyType::DCT4X4, AcStrategyType::DCT4X8, AcStrategyType::DCT8X4,
      AcStrategyType::AFV0,   AcStrategyType::AFV1,   AcStrategyType::AFV2,
      AcStrategyType::AFV3,   AcStrategyType::DCT2X2,
  };
  for (size_t y0 = 0; y0 < ry; ++y0) {
    for (size_t x0 = 0; x0 < rx; ++x0) {
      float best_c = 1e30f;
      uint32_t best_idx = 0xFFFFFFFFu;
      uint8_t existing_1x1_raw = 255;
      const uint16_t cur_ti = ws->cur_tr_at[y0][x0];
      if (cur_ti < cache.transforms.size()) {
        const AcTransformCache& ctr = cache.transforms[cur_ti];
        if (ctr.bx == x0 && ctr.by == y0 && ctr.cov_x == 1 && ctr.cov_y == 1) {
          existing_1x1_raw = ctr.acs_raw;
          best_c = ws->cur_tr_costs[cur_ti];
          best_idx = push_spec(make_existing_spec(ctr), best_c);
        }
      }
      if (existing_1x1_raw != static_cast<uint8_t>(AcStrategyType::DCT)) {
        const AcsSubSpec sp = make_cand_spec(x0, y0, 1, 1, AcStrategyType::DCT);
        float c_dct = 1e30f;
        JXL_RETURN_IF_ERROR(eval_cand_cost(x0, y0, 1, 1, sp, &c_dct));
        if (c_dct < best_c) {
          best_c = c_dct;
          best_idx = push_spec(sp, c_dct);
        }
      }
      const bool try_sub8 =
          (ws->w_4m[y0][x0] > 0.98f) || (ws->cur_blk_grad[y0][x0] >= grad_p70) ||
          (existing_1x1_raw != 255 &&
           existing_1x1_raw != static_cast<uint8_t>(AcStrategyType::DCT));
      if (try_sub8) {
        for (AcStrategyType st_type : kSub8Types) {
          if (existing_1x1_raw == static_cast<uint8_t>(st_type)) continue;
          const AcsSubSpec sp = make_cand_spec(x0, y0, 1, 1, st_type);
          float c_cand = 1e30f;
          JXL_RETURN_IF_ERROR(eval_cand_cost(x0, y0, 1, 1, sp, &c_cand));
          if (c_cand < best_c) {
            best_c = c_cand;
            best_idx = push_spec(sp, c_cand);
          }
        }
        if (cparams.decoding_speed_tier < 4 &&
            ws->cur_blk_grad[y0][x0] >= grad_p70 &&
            existing_1x1_raw != static_cast<uint8_t>(AcStrategyType::IDENTITY)) {
          const AcsSubSpec sp =
              make_cand_spec(x0, y0, 1, 1, AcStrategyType::IDENTITY);
          float c_id = 1e30f;
          JXL_RETURN_IF_ERROR(eval_cand_cost(x0, y0, 1, 1, sp, &c_id));
          if (c_id < best_c) {
            best_c = c_id;
            best_idx = push_spec(sp, c_id);
          }
        }
      }
      ws->best_1x1_cost[y0][x0] = best_c;
      ws->best_1x1_spec_idx[y0][x0] = best_idx;
    }
  }

  for (size_t x = 0; x <= rx; ++x) ws->prefix_base[0][x] = 0.0f;
  for (size_t y = 0; y < ry; ++y) {
    float rsum = 0.0f;
    ws->prefix_base[y + 1][0] = 0.0f;
    for (size_t x = 0; x < rx; ++x) {
      rsum += ws->best_1x1_cost[y][x];
      ws->prefix_base[y + 1][x + 1] = ws->prefix_base[y][x + 1] + rsum;
    }
  }
  auto rect_base_sum = [&](size_t x0, size_t y0, size_t w, size_t h) -> float {
    return ws->prefix_base[y0 + h][x0 + w] - ws->prefix_base[y0][x0 + w] -
           ws->prefix_base[y0 + h][x0] + ws->prefix_base[y0][x0];
  };

  // Map 1D intervals [pos, pos + len) in [0, max_dim) to compact [0, next_iv).
  const size_t max_dim = std::max<size_t>(1, std::max(rx, ry));
  uint16_t next_iv = 0;
  for (size_t pos = 0; pos < max_dim; ++pos) {
    for (size_t len = 1; pos + len <= max_dim; ++len) {
      ws->iv_idx[pos][len] = next_iv++;
    }
  }
  const size_t iv_stride = next_iv;
  auto dp_cost = [&](uint16_t iy, uint16_t ix) -> float& {
    return ws->cost_flat[static_cast<size_t>(iy) * iv_stride + ix];
  };
  auto dp_split = [&](uint16_t iy, uint16_t ix) -> uint16_t& {
    return ws->split_flat[static_cast<size_t>(iy) * iv_stride + ix];
  };

  const bool allow_16 = (cparams.decoding_speed_tier <= 4);
  const bool allow_16x32 = allow_16;
  const bool allow_32x32 = (cparams.decoding_speed_tier < 4);
  const bool allow_64x32 = (cparams.decoding_speed_tier <= 1);
  const bool allow_64x64 = (cparams.decoding_speed_tier < 4);

  auto solve_non_guillotine_subrect = [&](size_t rx0, size_t ry0, size_t rw,
                                          size_t rh, float dp_sub_cost,
                                          size_t* out_best_num,
                                          float* out_best_gain) -> void {
    *out_best_num = 0;
    const float sub_base = rect_base_sum(rx0, ry0, rw, rh);
    const float dp_sub_gain = std::max(0.0f, sub_base - dp_sub_cost);
    *out_best_gain = dp_sub_gain;

    size_t n_loc = 0;
    for (const auto& cb : ws->cands) {
      if (cb.x0 >= rx0 && cb.y0 >= ry0 && cb.x0 + cb.w <= rx0 + rw &&
          cb.y0 + cb.h <= ry0 + rh && !(cb.w == rw && cb.h == rh) &&
          n_loc < 352) {
        uint64_t mask = 0;
        const uint64_t row_bits =
            ((uint64_t{1} << cb.w) - 1) << (cb.x0 - rx0);
        for (size_t dy = 0; dy < cb.h; ++dy) {
          mask |= (row_bits << ((cb.y0 - ry0 + dy) * 8));
        }
        ws->local_cands[n_loc++] = {mask, cb.gain, cb.rate, cb.spec_idx};
      }
    }
    if (n_loc <= 1) return;

    std::sort(ws->local_cands, ws->local_cands + n_loc,
              [](const GroupAcsDpWorkspace::LocalCand& a,
                 const GroupAcsDpWorkspace::LocalCand& b) {
                return a.rate > b.rate;
              });
    float best_ng_gain = dp_sub_gain + 1e-3f;
    size_t cur_num = 0;
    size_t nodes = 0;
    constexpr size_t kMaxNodes = 4096;

    auto dfs = [&](auto& self, size_t idx, uint64_t used,
                   float cur_gain) -> void {
      if (cur_gain > best_ng_gain) {
        best_ng_gain = cur_gain;
        *out_best_num = cur_num;
        *out_best_gain = cur_gain;
        for (size_t i = 0; i < cur_num; ++i) {
          ws->ng_chosen[i] = ws->ng_cur_chosen[i];
        }
      }
      while (idx < n_loc && (ws->local_cands[idx].mask & used) != 0) {
        ++idx;
      }
      if (idx >= n_loc || ++nodes > kMaxNodes) return;
      uint64_t avail_mask = 0;
      float avail_gain = 0.0f;
      for (size_t k = idx; k < n_loc; ++k) {
        if ((ws->local_cands[k].mask & used) == 0) {
          avail_mask |= ws->local_cands[k].mask;
          avail_gain += ws->local_cands[k].gain;
        }
      }
      const float max_possible =
          cur_gain +
          std::min(avail_gain,
                   ws->local_cands[idx].rate *
                       static_cast<float>(hwy::PopCount(avail_mask)));
      if (max_possible <= best_ng_gain) return;

      const auto& lc = ws->local_cands[idx];
      ws->ng_cur_chosen[cur_num++] = static_cast<uint16_t>(idx);
      self(self, idx + 1, used | lc.mask, cur_gain + lc.gain);
      --cur_num;
      if (nodes <= kMaxNodes) {
        self(self, idx + 1, used, cur_gain);
      }
    };
    dfs(dfs, 0, 0, 0.0f);
  };

  struct PinSeg {
    uint8_t a, b;
  };
  static constexpr uint8_t kSegLens[3] = {1, 2, 4};
  constexpr float kInf = 1e30f;

  for (size_t h = 1; h <= ry; ++h) {
    PinSeg y_segs[9];
    size_t num_y_segs = 0;
    if (h >= 3 && h <= 12) {
      for (uint8_t s1 : kSegLens) {
        for (uint8_t s2 : kSegLens) {
          for (uint8_t s3 : kSegLens) {
            if (static_cast<size_t>(s1 + s2 + s3) == h) {
              y_segs[num_y_segs++] = {s1, static_cast<uint8_t>(s1 + s2)};
            }
          }
        }
      }
    }

    for (size_t w = 1; w <= rx; ++w) {
      if (w == 1 && h == 1) {
        for (size_t y0 = 0; y0 < ry; ++y0) {
          const uint16_t iy = ws->iv_idx[y0][1];
          for (size_t x0 = 0; x0 < rx; ++x0) {
            const uint16_t ix = ws->iv_idx[x0][1];
            dp_cost(iy, ix) = ws->best_1x1_cost[y0][x0];
            dp_split(iy, ix) = 0;
            ws->leaf_spec_idx[y0][x0][1][1] = ws->best_1x1_spec_idx[y0][x0];
          }
        }
        continue;
      }

      PinSeg x_segs[9];
      size_t num_x_segs = 0;
      if (w >= 3 && w <= 12 && num_y_segs > 0) {
        for (uint8_t s1 : kSegLens) {
          for (uint8_t s2 : kSegLens) {
            for (uint8_t s3 : kSegLens) {
              if (static_cast<size_t>(s1 + s2 + s3) == w) {
                x_segs[num_x_segs++] = {s1, static_cast<uint8_t>(s1 + s2)};
              }
            }
          }
        }
      }

      // Strictly <= 64x64 (w <= 8, h <= 8; NEVER DCT128!).
      bool has_direct = false;
      AcStrategyType direct_type = AcStrategyType::DCT;
      if (w == 1 && h == 2 && allow_16) {
        has_direct = true;
        direct_type = AcStrategyType::DCT16X8;
      } else if (w == 2 && h == 1 && allow_16) {
        has_direct = true;
        direct_type = AcStrategyType::DCT8X16;
      } else if (w == 2 && h == 2 && allow_16) {
        has_direct = true;
        direct_type = AcStrategyType::DCT16X16;
      } else if (w == 4 && h == 1 && allow_16x32) {
        has_direct = true;
        direct_type = AcStrategyType::DCT8X32;
      } else if (w == 1 && h == 4 && allow_16x32) {
        has_direct = true;
        direct_type = AcStrategyType::DCT32X8;
      } else if (w == 4 && h == 2 && allow_16x32) {
        has_direct = true;
        direct_type = AcStrategyType::DCT16X32;
      } else if (w == 2 && h == 4 && allow_16x32) {
        has_direct = true;
        direct_type = AcStrategyType::DCT32X16;
      } else if (w == 4 && h == 4 && allow_32x32) {
        has_direct = true;
        direct_type = AcStrategyType::DCT32X32;
      } else if (w == 4 && h == 8 && allow_64x32) {
        has_direct = true;
        direct_type = AcStrategyType::DCT64X32;
      } else if (w == 8 && h == 4 && allow_64x32) {
        has_direct = true;
        direct_type = AcStrategyType::DCT32X64;
      } else if (w == 8 && h == 8 && allow_64x64) {
        has_direct = true;
        direct_type = AcStrategyType::DCT64X64;
      }

      for (size_t y0 = 0; y0 + h <= ry; ++y0) {
        const uint16_t iy = ws->iv_idx[y0][h];
        for (size_t x0 = 0; x0 + w <= rx; ++x0) {
          const uint16_t ix = ws->iv_idx[x0][w];
          float best_c = kInf;
          uint16_t best_s = 1;

          for (size_t k = 1; k < w; ++k) {
            const float c = dp_cost(iy, ws->iv_idx[x0][k]) +
                            dp_cost(iy, ws->iv_idx[x0 + k][w - k]);
            if (c < best_c) {
              best_c = c;
              best_s = static_cast<uint16_t>(k);
            }
          }
          for (size_t k = 1; k < h; ++k) {
            const float c = dp_cost(ws->iv_idx[y0][k], ix) +
                            dp_cost(ws->iv_idx[y0 + k][h - k], ix);
            if (c < best_c) {
              best_c = c;
              best_s = static_cast<uint16_t>(32 + k);
            }
          }
          for (size_t yi = 0; yi < num_y_segs; ++yi) {
            const uint8_t ya = y_segs[yi].a;
            const uint8_t yb = y_segs[yi].b;
            const uint16_t iy_top_a = ws->iv_idx[y0][ya];
            const uint16_t iy_top_b = ws->iv_idx[y0][yb];
            const uint16_t iy_bot_a = ws->iv_idx[y0 + ya][h - ya];
            const uint16_t iy_bot_b = ws->iv_idx[y0 + yb][h - yb];
            const uint16_t iy_mid = ws->iv_idx[y0 + ya][yb - ya];
            for (size_t xi = 0; xi < num_x_segs; ++xi) {
              const uint8_t xa = x_segs[xi].a;
              const uint8_t xb = x_segs[xi].b;
              const float c_mid =
                  dp_cost(iy_mid, ws->iv_idx[x0 + xa][xb - xa]);
              const float c_cw =
                  dp_cost(iy_top_a, ws->iv_idx[x0][xb]) +
                  dp_cost(iy_top_b, ws->iv_idx[x0 + xb][w - xb]) +
                  dp_cost(iy_bot_b, ws->iv_idx[x0 + xa][w - xa]) +
                  dp_cost(iy_bot_a, ws->iv_idx[x0][xa]) + c_mid;
              if (c_cw < best_c) {
                best_c = c_cw;
                best_s = static_cast<uint16_t>(
                    0x4000u | (static_cast<uint16_t>(xa) << 10) |
                    (static_cast<uint16_t>(xb - xa) << 7) |
                    (static_cast<uint16_t>(ya) << 4) |
                    static_cast<uint16_t>(yb - ya));
              }
              const float c_ccw =
                  dp_cost(iy_top_b, ws->iv_idx[x0][xa]) +
                  dp_cost(iy_top_a, ws->iv_idx[x0 + xa][w - xa]) +
                  dp_cost(iy_bot_a, ws->iv_idx[x0 + xb][w - xb]) +
                  dp_cost(iy_bot_b, ws->iv_idx[x0][xb]) + c_mid;
              if (c_ccw < best_c) {
                best_c = c_ccw;
                best_s = static_cast<uint16_t>(
                    0x6000u | (static_cast<uint16_t>(xa) << 10) |
                    (static_cast<uint16_t>(xb - xa) << 7) |
                    (static_cast<uint16_t>(ya) << 4) |
                    static_cast<uint16_t>(yb - ya));
              }
            }
          }

          if (has_direct && cache.px0 + (x0 + w) * kBlockDim <= xsize &&
              cache.py0 + (y0 + h) * kBlockDim <= ysize) {
            const float base_sum = rect_base_sum(x0, y0, w, h);
            float cand_c = kInf;
            uint32_t cand_sidx = 0xFFFFFFFFu;
            if (ws->pre_leaf_spec_idx[y0][x0][h][w] != 0xFFFFFFFFu) {
              cand_c = ws->pre_leaf_cost[y0][x0][h][w];
              cand_sidx = ws->pre_leaf_spec_idx[y0][x0][h][w];
            } else {
              int32_t q_min = 255;
              int32_t q_max = 1;
              for (size_t iy_q = 0; iy_q < h; ++iy_q) {
                for (size_t ix_q = 0; ix_q < w; ++ix_q) {
                  const int32_t qv = ws->blk_q[y0 + iy_q][x0 + ix_q];
                  q_min = std::min(q_min, qv);
                  q_max = std::max(q_max, qv);
                }
              }
              const bool q_homog =
                  (q_max - q_min) <= std::max<int32_t>(2, q_max / 3);
              const bool is_8x32 = (w == 4 && h == 1) || (w == 1 && h == 4);
              const bool even_xy = ((x0 & 1) == 0 && (y0 & 1) == 0);
              const bool eval_direct =
                  q_homog &&
                  ((w * h <= 4) ||
                   (is_8x32 ? (even_xy || best_c < base_sum - 1e-4f)
                            : ((w * h <= 16)
                                   ? (even_xy || best_c < base_sum * 0.97f)
                                   : (even_xy || (rx <= 8 && ry <= 8)))));
              if (eval_direct) {
                const AcsSubSpec sp = make_cand_spec(x0, y0, w, h, direct_type);
                JXL_RETURN_IF_ERROR(
                    eval_cand_cost(x0, y0, w, h, sp, &cand_c));
                if (cand_c < best_c) {
                  cand_sidx = push_spec(sp, cand_c);
                }
              }
            }
            if (cand_c < best_c && cand_sidx != 0xFFFFFFFFu) {
              best_c = cand_c;
              best_s = 0;
              ws->leaf_spec_idx[y0][x0][h][w] = cand_sidx;
              const float gain = base_sum - cand_c;
              if (gain > 0.0f && ws->cands.size() < 6144) {
                ws->cands.push_back({
                    static_cast<uint8_t>(x0),
                    static_cast<uint8_t>(y0),
                    static_cast<uint8_t>(w),
                    static_cast<uint8_t>(h),
                    cand_sidx,
                    gain,
                    gain / static_cast<float>(w * h),
                });
              }
            }
          }

          if (((w == 8 && h == 8) ||
               (w == rx && h == ry && rx <= 8 && ry <= 8)) &&
              ((x0 % 4 == 0 && y0 % 4 == 0) || (w == rx && h == ry))) {
            size_t ng_num = 0;
            float ng_gain = 0.0f;
            solve_non_guillotine_subrect(x0, y0, w, h, best_c, &ng_num,
                                         &ng_gain);
            if (ng_num > 0) {
              best_c = rect_base_sum(x0, y0, w, h) - ng_gain;
              best_s = 0x3FFFu;
            }
          }

          dp_cost(iy, ix) = best_c;
          dp_split(iy, ix) = best_s;
        }
      }
    }
  }

  // Reconstruct the globally optimal group partition into dp_spec_indices.
  ws->dp_spec_indices.clear();
  ws->dp_spec_indices.reserve(rx * ry);
  size_t sp_top = 0;
  ws->stack[sp_top++] = {0, 0, static_cast<uint8_t>(rx),
                         static_cast<uint8_t>(ry)};
  while (sp_top > 0) {
    const auto cur = ws->stack[--sp_top];
    const uint16_t iy = ws->iv_idx[cur.y0][cur.h];
    const uint16_t ix = ws->iv_idx[cur.x0][cur.w];
    const uint16_t s = dp_split(iy, ix);
    if (s == 0) {
      const uint32_t sidx = ws->leaf_spec_idx[cur.y0][cur.x0][cur.h][cur.w];
      if (sidx == 0xFFFFFFFFu) return true;
      ws->dp_spec_indices.push_back(sidx);
      continue;
    }
    const bool already_checked_root_ng =
        (cur.w == rx && cur.h == ry && rx <= 8 && ry <= 8 && s != 0x3FFFu);
    if (cur.w <= 8 && cur.h <= 8 && !already_checked_root_ng &&
        (s == 0x3FFFu || (cur.w >= 4 && cur.h >= 4))) {
      size_t ng_num = 0;
      float ng_gain = 0.0f;
      const float compare_cost = (s == 0x3FFFu) ? kInf : dp_cost(iy, ix);
      solve_non_guillotine_subrect(cur.x0, cur.y0, cur.w, cur.h, compare_cost,
                                   &ng_num, &ng_gain);
      if (ng_num > 0) {
        uint64_t covered = 0;
        for (size_t i = 0; i < ng_num; ++i) {
          const auto& lc = ws->local_cands[ws->ng_chosen[i]];
          covered |= lc.mask;
          ws->dp_spec_indices.push_back(lc.spec_idx);
        }
        for (size_t dy = 0; dy < cur.h; ++dy) {
          for (size_t dx = 0; dx < cur.w; ++dx) {
            if ((covered & (uint64_t{1} << (dy * 8 + dx))) == 0) {
              ws->dp_spec_indices.push_back(
                  ws->best_1x1_spec_idx[cur.y0 + dy][cur.x0 + dx]);
            }
          }
        }
        continue;
      }
    }
    if (s < 32) {
      const uint8_t k = static_cast<uint8_t>(s);
      ws->stack[sp_top++] = {static_cast<uint8_t>(cur.x0 + k), cur.y0,
                             static_cast<uint8_t>(cur.w - k), cur.h};
      ws->stack[sp_top++] = {cur.x0, cur.y0, k, cur.h};
    } else if (s < 64) {
      const uint8_t k = static_cast<uint8_t>(s - 32);
      ws->stack[sp_top++] = {cur.x0, static_cast<uint8_t>(cur.y0 + k), cur.w,
                             static_cast<uint8_t>(cur.h - k)};
      ws->stack[sp_top++] = {cur.x0, cur.y0, cur.w, k};
    } else if ((s & 0x4000u) != 0) {
      const bool ccw = (s & 0x2000u) != 0;
      const uint8_t xa = static_cast<uint8_t>((s >> 10) & 7u);
      const uint8_t xb = static_cast<uint8_t>(xa + ((s >> 7) & 7u));
      const uint8_t ya = static_cast<uint8_t>((s >> 4) & 7u);
      const uint8_t yb = static_cast<uint8_t>(ya + (s & 7u));
      ws->stack[sp_top++] = {static_cast<uint8_t>(cur.x0 + xa),
                             static_cast<uint8_t>(cur.y0 + ya),
                             static_cast<uint8_t>(xb - xa),
                             static_cast<uint8_t>(yb - ya)};
      if (!ccw) {
        ws->stack[sp_top++] = {cur.x0, static_cast<uint8_t>(cur.y0 + ya), xa,
                               static_cast<uint8_t>(cur.h - ya)};
        ws->stack[sp_top++] = {static_cast<uint8_t>(cur.x0 + xa),
                               static_cast<uint8_t>(cur.y0 + yb),
                               static_cast<uint8_t>(cur.w - xa),
                               static_cast<uint8_t>(cur.h - yb)};
        ws->stack[sp_top++] = {static_cast<uint8_t>(cur.x0 + xb), cur.y0,
                               static_cast<uint8_t>(cur.w - xb), yb};
        ws->stack[sp_top++] = {cur.x0, cur.y0, xb, ya};
      } else {
        ws->stack[sp_top++] = {cur.x0, static_cast<uint8_t>(cur.y0 + yb), xb,
                               static_cast<uint8_t>(cur.h - yb)};
        ws->stack[sp_top++] = {static_cast<uint8_t>(cur.x0 + xb),
                               static_cast<uint8_t>(cur.y0 + ya),
                               static_cast<uint8_t>(cur.w - xb),
                               static_cast<uint8_t>(cur.h - ya)};
        ws->stack[sp_top++] = {static_cast<uint8_t>(cur.x0 + xa), cur.y0,
                               static_cast<uint8_t>(cur.w - xa), ya};
        ws->stack[sp_top++] = {cur.x0, cur.y0, xa, yb};
      }
    } else {
      return true;
    }
  }

  std::sort(ws->dp_spec_indices.begin(), ws->dp_spec_indices.end(),
            [&](uint32_t ia, uint32_t ib) {
              const AcsSubSpec& a = ws->spec_pool[ia];
              const AcsSubSpec& b = ws->spec_pool[ib];
              return (a.by != b.by) ? (a.by < b.by) : (a.bx < b.bx);
            });

  memset(ws->dp_cov, 0, sizeof(ws->dp_cov));
  size_t total_cov = 0;
  for (size_t k = 0; k < ws->dp_spec_indices.size(); ++k) {
    const AcsSubSpec& sp = ws->spec_pool[ws->dp_spec_indices[k]];
    const AcStrategy acs = AcStrategy::FromRawStrategy(sp.acs_raw);
    const size_t cw = acs.covered_blocks_x();
    const size_t ch = acs.covered_blocks_y();
    if (sp.bx + cw > rx || sp.by + ch > ry) return true;
    for (size_t iy = 0; iy < ch; ++iy) {
      for (size_t ix = 0; ix < cw; ++ix) {
        if (ws->dp_cov[sp.by + iy][sp.bx + ix] != 0) return true;
        ws->dp_cov[sp.by + iy][sp.bx + ix] = 1;
        ws->dp_spec_at[sp.by + iy][sp.bx + ix] = static_cast<uint16_t>(k);
        ++total_cov;
      }
    }
  }
  if (total_cov != rx * ry) return true;

  // Identify minimal pairwise-disjoint self-contained difference rectangles.
  struct BoxRect {
    uint8_t x0, y0, x1, y1;
  };
  std::vector<BoxRect> boxes;
  for (size_t y = 0; y < ry; ++y) {
    for (size_t x = 0; x < rx; ++x) {
      const uint16_t cti = ws->cur_tr_at[y][x];
      const AcsSubSpec& dsp =
          ws->spec_pool[ws->dp_spec_indices[ws->dp_spec_at[y][x]]];
      bool same = false;
      if (cti < cache.transforms.size()) {
        const AcTransformCache& ctr = cache.transforms[cti];
        same = (ctr.bx == dsp.bx && ctr.by == dsp.by &&
                ctr.acs_raw == dsp.acs_raw);
      }
      ws->diff_blk[y][x] = same ? 0 : 1;
      if (!same) {
        boxes.push_back({static_cast<uint8_t>(x), static_cast<uint8_t>(y),
                         static_cast<uint8_t>(x + 1),
                         static_cast<uint8_t>(y + 1)});
      }
    }
  }
  if (boxes.empty()) return true;

  bool changed = true;
  while (changed) {
    changed = false;
    for (BoxRect& b : boxes) {
      for (const AcTransformCache& tr : cache.transforms) {
        if (tr.bx < b.x1 && tr.bx + tr.cov_x > b.x0 && tr.by < b.y1 &&
            tr.by + tr.cov_y > b.y0) {
          if (tr.bx < b.x0) {
            b.x0 = tr.bx;
            changed = true;
          }
          if (tr.by < b.y0) {
            b.y0 = tr.by;
            changed = true;
          }
          if (tr.bx + tr.cov_x > b.x1) {
            b.x1 = static_cast<uint8_t>(tr.bx + tr.cov_x);
            changed = true;
          }
          if (tr.by + tr.cov_y > b.y1) {
            b.y1 = static_cast<uint8_t>(tr.by + tr.cov_y);
            changed = true;
          }
        }
      }
      for (uint32_t sidx : ws->dp_spec_indices) {
        const AcsSubSpec& sp = ws->spec_pool[sidx];
        const AcStrategy acs = AcStrategy::FromRawStrategy(sp.acs_raw);
        const uint8_t sx1 = static_cast<uint8_t>(sp.bx + acs.covered_blocks_x());
        const uint8_t sy1 = static_cast<uint8_t>(sp.by + acs.covered_blocks_y());
        if (sp.bx < b.x1 && sx1 > b.x0 && sp.by < b.y1 && sy1 > b.y0) {
          if (sp.bx < b.x0) {
            b.x0 = sp.bx;
            changed = true;
          }
          if (sp.by < b.y0) {
            b.y0 = sp.by;
            changed = true;
          }
          if (sx1 > b.x1) {
            b.x1 = sx1;
            changed = true;
          }
          if (sy1 > b.y1) {
            b.y1 = sy1;
            changed = true;
          }
        }
      }
    }
    for (size_t i = 0; i < boxes.size(); ++i) {
      for (size_t j = i + 1; j < boxes.size();) {
        if (boxes[i].x0 < boxes[j].x1 && boxes[j].x0 < boxes[i].x1 &&
            boxes[i].y0 < boxes[j].y1 && boxes[j].y0 < boxes[i].y1) {
          boxes[i].x0 = std::min(boxes[i].x0, boxes[j].x0);
          boxes[i].y0 = std::min(boxes[i].y0, boxes[j].y0);
          boxes[i].x1 = std::max(boxes[i].x1, boxes[j].x1);
          boxes[i].y1 = std::max(boxes[i].y1, boxes[j].y1);
          boxes.erase(boxes.begin() + j);
          changed = true;
        } else {
          ++j;
        }
      }
    }
  }

  for (const BoxRect& b : boxes) {
    double cur_cost = 0.0;
    for (size_t ti = 0; ti < cache.transforms.size(); ++ti) {
      const AcTransformCache& tr = cache.transforms[ti];
      if (tr.bx >= b.x0 && tr.bx < b.x1 && tr.by >= b.y0 && tr.by < b.y1) {
        cur_cost += ws->cur_tr_costs[ti];
      }
    }
    GroupAcsDpComponent comp;
    comp.group = cache.group_idx;
    comp.bx0 = b.x0;
    comp.by0 = b.y0;
    comp.bw = static_cast<uint8_t>(b.x1 - b.x0);
    comp.bh = static_cast<uint8_t>(b.y1 - b.y0);
    double new_cost = 0.0;
    for (uint32_t sidx : ws->dp_spec_indices) {
      const AcsSubSpec& sp = ws->spec_pool[sidx];
      if (sp.bx >= b.x0 && sp.bx < b.x1 && sp.by >= b.y0 && sp.by < b.y1) {
        comp.specs.push_back(sp);
        new_cost += ws->spec_cost_pool[sidx];
        const size_t qt =
            static_cast<size_t>(kAcStrategyToQuantTableMap[sp.acs_raw]);
        if (qt >= 1 && qt < kNumQuantTables && qt < qm_encs.size() &&
            quant_table_counts[qt] == 0 &&
            qm_encs[qt].mode == QuantEncoding::kQuantModeDCT) {
          comp.new_tables_mask |= (1u << static_cast<uint32_t>(qt));
        }
      }
    }
    comp.dp_gain = cur_cost - new_cost;
    if (!comp.specs.empty() && comp.dp_gain > 1e-4) {
      out_comps->push_back(std::move(comp));
    }
  }

  return true;
}

}  // namespace

Status RefineQuantFieldClosedLoop(
    FrameHeader& frame_header, const Image3F& opsin, const Rect& rect,
    ThreadPool* pool, PassesEncoderState* enc_state,
    ModularFrameEncoder* modular_frame_encoder) {
  const CompressParams& cparams = enc_state->cparams;
  enc_state->rdoq_level_field = ImageSB();
  if (cparams.speed_tier > SpeedTier::kSquirrel ||
      cparams.disable_perceptual_optimizations || cparams.max_error_mode ||
      frame_header.frame_type == FrameType::kDCFrame ||
      frame_header.color_transform != ColorTransform::kXYB ||
      !frame_header.chroma_subsampling.Is444() ||
      frame_header.upsampling != 1 ||
      (frame_header.flags & FrameHeader::kNoise) != 0 ||
      cparams.dots == Override::kOn ||
      cparams.butteraugli_distance > 15.0f ||
      enc_state->streaming_mode ||
      enc_state->token_cost_model == nullptr ||
      enc_state->token_cost_model->small_val_bits.empty() ||
      enc_state->band_qf[0][0].xsize() == 0) {
    return true;
  }

  PassesSharedState& shared = enc_state->shared;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  const size_t xsize = shared.frame_dim.xsize;
  const size_t ysize = shared.frame_dim.ysize;
  const size_t xsize_blocks = shared.frame_dim.xsize_blocks;
  const size_t ysize_blocks = shared.frame_dim.ysize_blocks;
  if (xsize < 8 || ysize < 8 || enc_state->orig_opsin == nullptr ||
      enc_state->orig_opsin->xsize() < xsize ||
      enc_state->orig_opsin->ysize() < ysize) {
    return true;
  }

  constexpr size_t kTileBlocks = 4;
  const size_t tiles_x = DivCeil(xsize_blocks, kTileBlocks);
  const size_t tiles_y = DivCeil(ysize_blocks, kTileBlocks);
  const size_t num_tiles = tiles_x * tiles_y;
  const size_t num_groups = shared.frame_dim.num_groups;
  const size_t num_blocks = xsize_blocks * ysize_blocks;

  struct ClosedLoopRdoState {
    std::vector<AcGroupBlockCache> group_caches;
    Image3F src_lin;
    Image3F feat_xyb;
    Image3F plain_pre_gab;
    Image3F fine_pre_gab;
    Image3F coarse_pre_gab;
    Image3F cur_pre_gab;
    Image3F cand_pre_gab;
    Image3F batch_pre_gab;
    Image3F cur_lin;
    Image3F post_xyb;
    ImageI rdo_quant_field;
    ImageB rdo_sharpness;
    ImageB tmp_sharp;
    ImageB best_cand_sharp;
    ImageB cand_sharp;
    std::array<ImageF, 8> ar_err_imgs;
    ImageF ar_bdry_err0;
    ImageF ar_target_grad;
    size_t ar_histo[9][8] = {};
    size_t ar_totals[9] = {};
    size_t ar_bctx_histo[4][8] = {};
    size_t ar_bctx_totals[4] = {};
    size_t ar_final_bctx_histo[4][8] = {};
    size_t ar_final_bctx_totals[4] = {};
    size_t ar_val_counts[8] = {};
    float ar_mul[27] = {};
    WdProxyIncrementalArbiter wd;
    Ssim2IncrementalArbiter ssim2;
    Ba3IncrementalArbiter ba3;
    CvvdpIncrementalArbiter cvvdp;
    EpfSigmaState epf_sigma;
    EpfWorkspace epf_ws;
    WdProxyIncrementalArbiter::Checkpoint cp_wd;
    Ssim2IncrementalArbiter::Checkpoint cp_ss2;
    Ba3IncrementalArbiter::Checkpoint cp_ba3;
    CvvdpIncrementalArbiter::Checkpoint cp_cv;
    std::vector<float> plain_tile_bits;
    std::vector<float> fine_tile_bits;
    std::vector<float> cur_tile_bits;
    std::vector<float> cur_group_bits;
    std::vector<double> tmp_metric_blk;
    std::vector<double> blk_err_cur;
    std::vector<double> tile_err;
    std::vector<size_t> tile_group;
    std::vector<std::vector<size_t>> tile_tr_indices;
    std::vector<int8_t> tile_step;
    std::vector<int8_t> tile_rdoq;
    ImageSB rdo_ytox_map;
    ImageSB rdo_ytob_map;
    std::vector<size_t> cfl_tile_group;
    std::vector<std::vector<size_t>> cfl_tile_trs;
    std::vector<double> cfl_tile_err;
    std::vector<uint8_t> cfl_moved[2];
    std::vector<float> saved_lin;
    std::vector<uint8_t> saved_sharp;
    std::vector<AcTransformCache> acs_scratch_trs;
    Image3F dc_unsmoothed;
    std::vector<uint32_t> blk_to_tr;
    std::vector<uint8_t> dc_touched_blk;
    LoopFilter anchor_lf;
    LoopFilter best_lf;
    LoopFilter base_gab_lf;
    std::vector<LoopFilter> cand_lfs;
  };
  auto st_ptr = jxl::make_unique<ClosedLoopRdoState>();
  ClosedLoopRdoState& st = *st_ptr;
  auto& group_caches = st.group_caches;
  group_caches.resize(num_groups);
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

  Image3F& src_lin = st.src_lin;
  JXL_ASSIGN_OR_RETURN(src_lin, Image3F::Create(memory_manager, xsize, ysize));
  {
    const auto& im = shared.metadata->transform_data.opsin_inverse_matrix;
    float biases_cbrt[3];
    for (size_t c = 0; c < 3; ++c) {
      biases_cbrt[c] = std::cbrt(im.opsin_biases[c]);
    }
    const float inv_scale = 255.0f / shared.metadata->m.IntensityTarget();
    const bool clamp_sdr = !shared.metadata->m.bit_depth.floating_point_sample;
    for (size_t y = 0; y < ysize; ++y) {
      XybRowToLinearRgb(enc_state->orig_opsin->ConstPlaneRow(0, y),
                        enc_state->orig_opsin->ConstPlaneRow(1, y),
                        enc_state->orig_opsin->ConstPlaneRow(2, y), xsize,
                        im.inverse_matrix, im.opsin_biases, biases_cbrt,
                        inv_scale, clamp_sdr, src_lin.PlaneRow(0, y),
                        src_lin.PlaneRow(1, y), src_lin.PlaneRow(2, y));
    }
  }

  Image3F& feat_xyb = st.feat_xyb;
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

  auto* wd = &st.wd;
  auto* ssim2 = &st.ssim2;
  auto* ba3 = &st.ba3;
  auto* cvvdp = &st.cvvdp;
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, 4, ThreadPool::NoInit,
      [&](uint32_t m, size_t /*thread*/) -> Status {
        if (m == 0) {
          wd->Init(src_lin);
        } else if (m == 1) {
          ssim2->Init(src_lin);
        } else if (m == 2) {
          ba3->Init(src_lin);
        } else {
          cvvdp->Init(src_lin);
        }
        return true;
      },
      "Init4Metrics"));

  auto* epf_sigma = &st.epf_sigma;
  JXL_RETURN_IF_ERROR(
      epf_sigma->Init(memory_manager, xsize_blocks, ysize_blocks));
  auto* epf_ws = &st.epf_ws;

  ImageI& rdo_quant_field = st.rdo_quant_field;
  ImageB& rdo_sharpness = st.rdo_sharpness;
  ImageB& tmp_sharp = st.tmp_sharp;
  ImageB& best_cand_sharp = st.best_cand_sharp;
  ImageB& cand_sharp = st.cand_sharp;
  JXL_ASSIGN_OR_RETURN(rdo_quant_field,
                       ImageI::Create(memory_manager, xsize_blocks,
                                      ysize_blocks));
  JXL_RETURN_IF_ERROR(CopyImageTo(shared.raw_quant_field, &rdo_quant_field));
  JXL_ASSIGN_OR_RETURN(rdo_sharpness,
                       ImageB::Create(memory_manager, xsize_blocks,
                                      ysize_blocks));
  JXL_ASSIGN_OR_RETURN(tmp_sharp,
                       ImageB::Create(memory_manager, xsize_blocks,
                                      ysize_blocks));
  JXL_ASSIGN_OR_RETURN(best_cand_sharp,
                       ImageB::Create(memory_manager, xsize_blocks,
                                      ysize_blocks));
  JXL_ASSIGN_OR_RETURN(cand_sharp,
                       ImageB::Create(memory_manager, xsize_blocks,
                                      ysize_blocks));

  Image3F& plain_pre_gab = st.plain_pre_gab;
  Image3F& fine_pre_gab = st.fine_pre_gab;
  Image3F& coarse_pre_gab = st.coarse_pre_gab;
  Image3F& cur_pre_gab = st.cur_pre_gab;
  Image3F& cand_pre_gab = st.cand_pre_gab;
  Image3F& batch_pre_gab = st.batch_pre_gab;
  Image3F& cur_lin = st.cur_lin;
  JXL_ASSIGN_OR_RETURN(plain_pre_gab,
                       Image3F::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(fine_pre_gab,
                       Image3F::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(coarse_pre_gab,
                       Image3F::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(cur_pre_gab,
                       Image3F::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(cand_pre_gab,
                       Image3F::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(batch_pre_gab,
                       Image3F::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(cur_lin,
                       Image3F::Create(memory_manager, xsize, ysize));

  const Rect full_px_rect(0, 0, xsize, ysize);
  const Rect full_blk_rect(0, 0, xsize_blocks, ysize_blocks);
  const float quant_scale = shared.quantizer.Scale();
  LoopFilter& lf = frame_header.loop_filter;

  auto solve_ar_sharpness = [&](const LoopFilter& cur_lf,
                                const Image3F& pre_gab_src,
                                const ImageI& qf_src, float asym_def_w,
                                float tex_mul, float c3_mul,
                                bool allow_const_collapse,
                                ImageB* out_sharp) -> Status {
    if (cur_lf.epf_iters == 0) {
      FillPlane(static_cast<uint8_t>(0), out_sharp, Rect(*out_sharp));
      return true;
    }
    if (cparams.butteraugli_distance < kMinButteraugliForDynamicAR ||
        cparams.speed_tier > SpeedTier::kWombat ||
        !cparams.use_full_image_heuristics ||
        enc_state->initial_quant_masking1x1.xsize() < xsize ||
        enc_state->initial_quant_masking1x1.ysize() < ysize) {
      FillPlane(static_cast<uint8_t>(4), out_sharp, Rect(*out_sharp));
      return true;
    }

    if (st.post_xyb.xsize() == 0) {
      JXL_ASSIGN_OR_RETURN(st.post_xyb,
                           Image3F::Create(memory_manager, xsize, ysize));
      JXL_ASSIGN_OR_RETURN(
          st.ar_bdry_err0,
          ImageF::Create(memory_manager, xsize_blocks, ysize_blocks));
      JXL_ASSIGN_OR_RETURN(
          st.ar_target_grad,
          ImageF::Create(memory_manager, xsize_blocks, ysize_blocks));
    }

    uint8_t epf_steps[3] = {0, 2, 5};
    size_t num_steps = 3;
    if (cparams.butteraugli_distance <= 1.6f) {
      epf_steps[2] = 6;
    }
    size_t epf_steps_lut[8] = {};
    for (size_t i = 0; i < num_steps; ++i) {
      epf_steps_lut[epf_steps[i]] = i;
    }

    const float clamped_butteraugli =
        std::min(5.0f, cparams.butteraugli_distance);
    constexpr float kBdryWeight = 0.08f;
    const float tex_weight_7 = 0.012f * tex_mul;
    const float tex_weight_2 =
        0.008f * tex_mul *
        Clamp1((1.15f - clamped_butteraugli) / 0.65f, 0.0f, 1.0f);

    for (size_t si = 0; si < num_steps; ++si) {
      const uint8_t val = epf_steps[si];
      if (st.ar_err_imgs[val].xsize() == 0) {
        JXL_ASSIGN_OR_RETURN(
            st.ar_err_imgs[val],
            ImageF::Create(memory_manager, xsize_blocks, ysize_blocks));
      }
      FillPlane(val, &tmp_sharp, Rect(tmp_sharp));
      epf_sigma->UpdateRect(cur_lf, shared.ac_strategy, quant_scale, qf_src,
                            tmp_sharp, full_blk_rect);
      JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
          memory_manager, cur_lf, shared, pre_gab_src, feat_ptr, epf_sigma,
          epf_ws, full_px_rect, /*out_lin=*/nullptr,
          /*out_xyb=*/&st.post_xyb));
      for (size_t by = 0; by < ysize_blocks; ++by) {
        float* JXL_RESTRICT error_row = st.ar_err_imgs[val].Row(by);
        float* JXL_RESTRICT bdry0_row = st.ar_bdry_err0.Row(by);
        float* JXL_RESTRICT target_grad_row = st.ar_target_grad.Row(by);
        for (size_t bx = 0; bx < xsize_blocks; ++bx) {
          float bdry_b = 0.0f;
          float grad_a = 0.0f;
          float grad_b = 0.0f;
          float err = ComputeLocalBlockEpfError(
              *enc_state->orig_opsin, st.post_xyb,
              enc_state->initial_quant_masking1x1, by, bx, asym_def_w, &bdry_b,
              &grad_a, &grad_b);
          if (val == 0) {
            bdry0_row[bx] = bdry_b;
            target_grad_row[bx] = std::min(grad_a, grad_b);
          } else {
            const float tw = (val >= 7) ? tex_weight_7 : tex_weight_2;
            err += kBdryWeight * (bdry_b - bdry0_row[bx]) +
                   tw * std::max(0.0f, target_grad_row[bx] - grad_b);
          }
          error_row[bx] = err;
        }
      }
    }

    double sum_sqrt_d0 = 0.0;
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* row0 = st.ar_err_imgs[0].ConstRow(by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        sum_sqrt_d0 += std::sqrt(std::max(0.0f, row0[bx]));
      }
    }
    const float mean_sqrt_d0 =
        static_cast<float>(sum_sqrt_d0 / std::max<size_t>(1, num_blocks));
    const float ref_d0 = mean_sqrt_d0 * mean_sqrt_d0;
    constexpr float kLocalRef = 0.84f;
    const float global_ref = 0.15f / std::pow(clamped_butteraugli, 0.35f);

    memset(st.ar_histo, 0, sizeof(st.ar_histo));
    for (size_t i = 0; i < 9; ++i) st.ar_totals[i] = 1;
    memset(st.ar_bctx_histo, 0, sizeof(st.ar_bctx_histo));
    for (size_t i = 0; i < 4; ++i) st.ar_bctx_totals[i] = 1;

    static constexpr float kFavorNoSmoothing = 0.99f;
    for (size_t by = 0; by < ysize_blocks; ++by) {
      uint8_t* JXL_RESTRICT out_row = out_sharp->Row(by);
      const uint8_t* JXL_RESTRICT prev_row =
          out_sharp->ConstRow(by > 0 ? by - 1 : 0);
      const float* row0 = st.ar_err_imgs[0].ConstRow(by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        const float d_ref = kLocalRef * row0[bx] + global_ref * ref_d0;
        uint8_t best_val = 0;
        float best_error = 1e30f;
        const uint8_t top_val = (by > 0) ? prev_row[bx] : 0;
        const uint8_t left_val = (bx > 0) ? out_row[bx - 1] : 0;
        const float top_mul =
            0.985f * (top_val == 0 ? kFavorNoSmoothing : 1.0f);
        const float left_mul =
            0.985f * (left_val == 0 ? kFavorNoSmoothing : 1.0f);
        const float top_error =
            st.ar_err_imgs[top_val].ConstRow(by)[bx] + d_ref * (top_mul - 1.0f);
        const float left_error = st.ar_err_imgs[left_val].ConstRow(by)[bx] +
                                 d_ref * (left_mul - 1.0f);
        for (size_t si = 0; si < num_steps; ++si) {
          const uint8_t val = epf_steps[si];
          const float m = (val == 0) ? kFavorNoSmoothing : 1.0f;
          const float error =
              st.ar_err_imgs[val].ConstRow(by)[bx] + d_ref * (m - 1.0f);
          if (error < best_error) {
            best_val = val;
            best_error = error;
          }
        }
        if (best_error < std::min(top_error, left_error)) {
          out_row[bx] = best_val;
        } else if (top_error < left_error) {
          out_row[bx] = top_val;
        } else {
          out_row[bx] = left_val;
        }
        const size_t context =
            epf_steps_lut[top_val] * 3 + epf_steps_lut[left_val];
        const size_t bctx = (top_val > 0 ? 2u : 0u) + (left_val > 0 ? 1u : 0u);
        ++st.ar_histo[context][out_row[bx]];
        ++st.ar_totals[context];
        ++st.ar_bctx_histo[bctx][out_row[bx]];
        ++st.ar_bctx_totals[bctx];
      }
    }

    constexpr float kC3Base = 0.980171988f;
    constexpr float kC3Clamp = 0.859703389f;
    const float c3 =
        c3_mul * std::max(kC3Clamp, std::pow(kC3Base, clamped_butteraugli));
    constexpr float kC5 = 0.108769036f;
    memset(st.ar_mul, 0, sizeof(st.ar_mul));
    for (size_t ti = 0; ti < num_steps; ++ti) {
      const uint8_t top_val = epf_steps[ti];
      for (size_t li = 0; li < num_steps; ++li) {
        const uint8_t left_val = epf_steps[li];
        for (size_t vi = 0; vi < num_steps; ++vi) {
          const uint8_t val = epf_steps[vi];
          const size_t context =
              epf_steps_lut[top_val] * 3 + epf_steps_lut[left_val];
          const size_t bctx =
              (top_val > 0 ? 2u : 0u) + (left_val > 0 ? 1u : 0u);
          const size_t mulix = epf_steps_lut[val] + 3 * context;
          const float ratio_9 = static_cast<float>(st.ar_histo[context][val]) /
                                static_cast<float>(st.ar_totals[context]);
          const float ratio_4 =
              static_cast<float>(st.ar_bctx_histo[bctx][val]) /
              static_cast<float>(st.ar_bctx_totals[bctx]);
          const float ratio = 0.65f * ratio_9 + 0.35f * ratio_4;
          st.ar_mul[mulix] =
              1.0f / (1.0f + kC5 * std::log1p(ratio) / clamped_butteraugli);
          if (val == top_val && val == left_val) {
            st.ar_mul[mulix] *= 0.988f;
          } else if (val == top_val || val == left_val) {
            st.ar_mul[mulix] *= 0.994f;
          }
          if (val == 0) {
            st.ar_mul[mulix] *= c3;
          }
        }
      }
    }

    double total_mod_err = 0.0;
    double sum_d_ref = 0.0;
    memset(st.ar_final_bctx_histo, 0, sizeof(st.ar_final_bctx_histo));
    memset(st.ar_final_bctx_totals, 0, sizeof(st.ar_final_bctx_totals));
    memset(st.ar_val_counts, 0, sizeof(st.ar_val_counts));
    for (size_t by = 0; by < ysize_blocks; ++by) {
      uint8_t* JXL_RESTRICT out_row = out_sharp->Row(by);
      const uint8_t* JXL_RESTRICT prev_row =
          out_sharp->ConstRow(by > 0 ? by - 1 : 0);
      const float* row0 = st.ar_err_imgs[0].ConstRow(by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        const float d_ref = kLocalRef * row0[bx] + global_ref * ref_d0;
        uint8_t best_val = 0;
        float best_error = 1e30f;
        const uint8_t top_val = (by > 0) ? prev_row[bx] : 0;
        const uint8_t left_val = (bx > 0) ? out_row[bx - 1] : 0;
        const size_t context =
            epf_steps_lut[top_val] * 3 + epf_steps_lut[left_val];
        for (size_t si = 0; si < num_steps; ++si) {
          const uint8_t val = epf_steps[si];
          const size_t mulix = epf_steps_lut[val] + 3 * context;
          const float error = st.ar_err_imgs[val].ConstRow(by)[bx] +
                              d_ref * (st.ar_mul[mulix] - 1.0f);
          if (error < best_error) {
            best_val = val;
            best_error = error;
          }
        }
        out_row[bx] = best_val;
        total_mod_err += st.ar_err_imgs[best_val].ConstRow(by)[bx];
        sum_d_ref += d_ref;
        const size_t bctx = (top_val > 0 ? 2u : 0u) + (left_val > 0 ? 1u : 0u);
        ++st.ar_final_bctx_histo[bctx][best_val];
        ++st.ar_final_bctx_totals[bctx];
        ++st.ar_val_counts[best_val];
      }
    }

    size_t num_unique_vals = 0;
    for (size_t si = 0; si < num_steps; ++si) {
      if (st.ar_val_counts[epf_steps[si]] > 0) ++num_unique_vals;
    }
    if (allow_const_collapse && num_unique_vals > 1 && num_blocks > 0) {
      double epf_bits = 14.0;
      for (size_t bctx = 0; bctx < 4; ++bctx) {
        if (st.ar_final_bctx_totals[bctx] == 0) continue;
        const double inv_tot =
            1.0 / static_cast<double>(st.ar_final_bctx_totals[bctx]);
        for (size_t si = 0; si < num_steps; ++si) {
          const size_t cnt = st.ar_final_bctx_histo[bctx][epf_steps[si]];
          if (cnt > 0) {
            epf_bits -= static_cast<double>(cnt) *
                        std::log2(static_cast<double>(cnt) * inv_tot);
          }
        }
      }
      const double mean_d_ref = sum_d_ref / static_cast<double>(num_blocks);
      const double lambda_bit =
          (0.018 / static_cast<double>(clamped_butteraugli)) * mean_d_ref;
      uint8_t best_const_val = 0;
      double best_const_err = 1e30;
      for (size_t si = 0; si < num_steps; ++si) {
        const uint8_t val = epf_steps[si];
        double c_err = 0.0;
        for (size_t by = 0; by < ysize_blocks; ++by) {
          const float* erow = st.ar_err_imgs[val].ConstRow(by);
          for (size_t bx = 0; bx < xsize_blocks; ++bx) {
            c_err += erow[bx];
          }
        }
        if (c_err < best_const_err) {
          best_const_err = c_err;
          best_const_val = val;
        }
      }
      if (best_const_err <= total_mod_err + lambda_bit * epf_bits) {
        FillPlane(best_const_val, out_sharp, Rect(*out_sharp));
      }
    }
    return true;
  };

  auto score_full_4m = [&](const Image3F& lin,
                           std::array<double, 4>* out_d) -> Status {
    return RunOnPool(
        pool, 0, 4, ThreadPool::NoInit,
        [&](uint32_t m, size_t /*thread*/) -> Status {
          if (m == 0) {
            (*out_d)[0] = wd->ScoreFull(lin);
          } else if (m == 1) {
            (*out_d)[1] = ssim2->ScoreFull(lin);
          } else if (m == 2) {
            (*out_d)[2] = ba3->ScoreFull(lin);
          } else {
            (*out_d)[3] = cvvdp->ScoreFull(lin);
          }
          return true;
        },
        "ScoreFull4M");
  };

  auto update_4m_checkpointed = [&](const Image3F& lin, const Rect& dirty,
                                    std::array<double, 4>* out_d) -> Status {
    return RunOnPool(
        pool, 0, 4, ThreadPool::NoInit,
        [&](uint32_t m, size_t /*thread*/) -> Status {
          if (m == 0) {
            (*out_d)[0] = wd->UpdateCheckpointed(lin, dirty, &st.cp_wd);
          } else if (m == 1) {
            (*out_d)[1] = ssim2->UpdateCheckpointed(lin, dirty, &st.cp_ss2);
          } else if (m == 2) {
            (*out_d)[2] = ba3->UpdateCheckpointed(lin, dirty, &st.cp_ba3);
          } else {
            (*out_d)[3] = cvvdp->UpdateCheckpointed(lin, dirty, &st.cp_cv);
          }
          return true;
        },
        "Update4M");
  };

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

  auto& plain_tile_bits = st.plain_tile_bits;
  auto& fine_tile_bits = st.fine_tile_bits;
  auto& cur_tile_bits = st.cur_tile_bits;
  auto& cur_group_bits = st.cur_group_bits;
  double plain_bits = 0.0;
  double fine_bits = 0.0;
  double coarse_bits = 0.0;
  double cur_bits = 0.0;

  JXL_RETURN_IF_ERROR(eval_all_groups(0, false, false, &plain_pre_gab,
                                      &plain_tile_bits, nullptr, &plain_bits));
  JXL_RETURN_IF_ERROR(eval_all_groups(1, false, false, &fine_pre_gab,
                                      &fine_tile_bits, nullptr, &fine_bits));
  JXL_RETURN_IF_ERROR(eval_all_groups(-1, false, false, &coarse_pre_gab,
                                      nullptr, nullptr, &coarse_bits));
  JXL_RETURN_IF_ERROR(eval_all_groups(0, true, true, &cur_pre_gab,
                                      &cur_tile_bits, &cur_group_bits,
                                      &cur_bits));
  JXL_RETURN_IF_ERROR(CopyImageTo(cur_pre_gab, &cand_pre_gab));
  JXL_RETURN_IF_ERROR(CopyImageTo(cur_pre_gab, &batch_pre_gab));

  // Stage 0: Warm-start per-block epf_sharpness on post-RDOQ reconstruction.
  JXL_RETURN_IF_ERROR(solve_ar_sharpness(
      lf, cur_pre_gab, rdo_quant_field, 1.20f, 1.0f, 1.0f,
      /*allow_const_collapse=*/true, &rdo_sharpness));
  double cur_sh_bits =
      (lf.epf_iters > 0)
          ? EstimateEpfSharpnessBits(rdo_sharpness, xsize_blocks, ysize_blocks)
          : 0.0;
  const double init_lf_bits = EstimateLoopFilterBits(lf);
  const double ref_bits = std::max(cur_bits + init_lf_bits + cur_sh_bits, 1.0);
  auto rate_term = [ref_bits](double bits) -> double {
    return 100.0 * std::log(std::max(bits, 1.0) / ref_bits);
  };

  // Stage 1: Calibrate 4-metric RD slopes at fixed initial epf_sigma so AC
  // qf sensitivity measures genuine coefficient fidelity after Gaborish + EPF
  // rather than epf_sigma shrinkage side effects.
  std::array<double, 4> plain_d = {};
  std::array<double, 4> fine_d = {};
  std::array<double, 4> coarse_d = {};
  std::array<double, 4> cur_d = {};
  std::array<double, 4> slope_m = {};
  std::array<double, 4> weight_m = {};

  epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale, rdo_quant_field,
                        rdo_sharpness, full_blk_rect);
  JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
      memory_manager, lf, shared, plain_pre_gab, feat_ptr, epf_sigma, epf_ws,
      full_px_rect, &cur_lin));
  JXL_RETURN_IF_ERROR(score_full_4m(cur_lin, &plain_d));

  JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
      memory_manager, lf, shared, fine_pre_gab, feat_ptr, epf_sigma, epf_ws,
      full_px_rect, &cur_lin));
  JXL_RETURN_IF_ERROR(score_full_4m(cur_lin, &fine_d));

  JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
      memory_manager, lf, shared, coarse_pre_gab, feat_ptr, epf_sigma, epf_ws,
      full_px_rect, &cur_lin));
  JXL_RETURN_IF_ERROR(score_full_4m(cur_lin, &coarse_d));

  JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
      memory_manager, lf, shared, cur_pre_gab, feat_ptr, epf_sigma, epf_ws,
      full_px_rect, &cur_lin));
  JXL_RETURN_IF_ERROR(score_full_4m(cur_lin, &cur_d));

  constexpr double kAcsEpfBdryWD = 0.076615785663558644;
  constexpr double kP2gLocLamMul = 0.82305822832816966;
  {
    static constexpr double kMetricPrior[4] = {
        0.17409245382991032, 0.29944174292572223, 0.22814790689226977,
        0.29831789635209766};
    const double r_plain = rate_term(plain_bits + cur_sh_bits);
    const double r_fine = rate_term(fine_bits + cur_sh_bits);
    const double r_coarse = rate_term(coarse_bits + cur_sh_bits);
    const float d_clamped =
        Clamp1(cparams.butteraugli_distance, 0.35f, 15.0f);
    const float s_d =
        Clamp1(std::log(d_clamped * (1.0f / 1.85f)), -1.45f, 1.85f);
    const double slope_scale_eff =
        0.92420335414208032 *
        std::exp(-0.0067694249999999999 * static_cast<double>(s_d));
    size_t active_cnt = 0;
    for (size_t m = 0; m < 4; ++m) {
      const double sf =
          (r_fine > r_plain + 1e-4)
              ? std::max(0.0, (plain_d[m] - fine_d[m]) / (r_fine - r_plain))
              : 0.0;
      const double sc =
          (r_plain > r_coarse + 1e-4)
              ? std::max(0.0, (coarse_d[m] - plain_d[m]) / (r_plain - r_coarse))
              : 0.0;
      slope_m[m] = slope_scale_eff *
                   ((sf > 0.0 && sc > 0.0) ? 0.5 * (sf + sc) : std::max(sf, sc));
      if (slope_m[m] > 1e-9) ++active_cnt;
    }
    if (active_cnt == 0) {
      if (lf.epf_iters == 0) {
        FillPlane(static_cast<uint8_t>(0), &shared.epf_sharpness,
                  Rect(shared.epf_sharpness));
      } else {
        JXL_RETURN_IF_ERROR(CopyImageTo(rdo_sharpness, &shared.epf_sharpness));
      }
      enc_state->epf_optimized_in_rdo = true;
      return true;
    }
    double w_sum = 0.0;
    for (size_t m = 0; m < 4; ++m) {
      weight_m[m] = (slope_m[m] > 1e-9) ? kMetricPrior[m] : 0.0;
      w_sum += weight_m[m];
    }
    if (w_sum > 0.0) {
      for (size_t m = 0; m < 4; ++m) weight_m[m] /= w_sum;
    }
  }

  auto& tmp_metric_blk = st.tmp_metric_blk;
  tmp_metric_blk.assign(num_blocks, 0.0);
  auto accumulate_4m_spatial_gain = [&](double* out_blk,
                                        bool weight_by_step1) {
    std::fill(out_blk, out_blk + num_blocks, 0.0);
    for (size_t m = 0; m < 4; ++m) {
      if (weight_m[m] <= 0.0) continue;
      const double step_gain =
          weight_by_step1
              ? (std::max(0.0, plain_d[m] - fine_d[m]) / slope_m[m])
              : 1.0;
      if (step_gain <= 1e-12) continue;
      std::fill(tmp_metric_blk.begin(), tmp_metric_blk.end(), 0.0);
      if (m == 0) {
        wd->AccumulateBlockError(xsize_blocks, ysize_blocks, 1.0,
                                 tmp_metric_blk.data());
      } else if (m == 1) {
        ssim2->AccumulateBlockError(xsize_blocks, ysize_blocks, 1.0,
                                    tmp_metric_blk.data());
      } else if (m == 2) {
        ba3->AccumulateBlockError(xsize_blocks, ysize_blocks, 1.0,
                                  tmp_metric_blk.data());
      } else {
        cvvdp->AccumulateBlockError(xsize_blocks, ysize_blocks, 1.0,
                                    tmp_metric_blk.data());
      }
      double sum_m = 0.0;
      for (size_t b = 0; b < num_blocks; ++b) sum_m += tmp_metric_blk[b];
      if (sum_m <= 1e-12) continue;
      const double scale = (weight_m[m] * step_gain) / sum_m;
      for (size_t b = 0; b < num_blocks; ++b) {
        out_blk[b] += scale * tmp_metric_blk[b];
      }
    }
  };

  auto& tile_group = st.tile_group;
  auto& tile_tr_indices = st.tile_tr_indices;
  tile_group.assign(num_tiles, num_groups);
  tile_tr_indices.resize(num_tiles);
  for (size_t g = 0; g < num_groups; ++g) {
    const auto& trs = group_caches[g].transforms;
    for (size_t i = 0; i < trs.size(); ++i) {
      const uint32_t t = trs[i].tile_idx;
      tile_group[t] = g;
      tile_tr_indices[t].push_back(i);
    }
  }

  // Stage 2: Bidirectional {-1, +1, +2} closed-loop qf refinement through
  // active dirty-rect EPF (Pass 2A reclaims bits on masked q>=6 tiles under
  // unanimous 4-metric RD agreement; Pass 2B invests +1/+2 steps on salient tiles).
  constexpr size_t kMaxPlusIters = 2;
  constexpr size_t kMaxPlusGreedy = 280;
  constexpr size_t kMaxMinusGreedy = 180;
  constexpr size_t kMaxPlusPerGroup = 14;
  constexpr size_t kMaxMinusPerGroup = 10;
  const size_t max_plus_moves = std::min<size_t>(
      kMaxPlusGreedy,
      std::max<size_t>(1, static_cast<size_t>(0.30f * num_tiles)));
  const size_t max_minus_moves = std::min<size_t>(
      kMaxMinusGreedy,
      std::max<size_t>(1, static_cast<size_t>(0.20f * num_tiles)));
  auto& tile_step = st.tile_step;
  auto& tile_rdoq = st.tile_rdoq;
  tile_step.assign(num_tiles, 0);
  tile_rdoq.assign(num_tiles, 0);
  size_t minus_moves = 0;
  size_t plus_moves = 0;
  size_t total_moves = 0;
  size_t rdoq_minus_moves = 0;
  size_t rdoq_plus_moves = 0;
  size_t rdoq_moves = 0;

  auto& cp_wd = st.cp_wd;
  auto& cp_ss2 = st.cp_ss2;
  auto& cp_ba3 = st.cp_ba3;
  auto& cp_cv = st.cp_cv;
  auto& saved_lin = st.saved_lin;
  auto& blk_err_cur = st.blk_err_cur;
  auto& tile_err = st.tile_err;
  blk_err_cur.assign(num_blocks, 0.0);

  size_t lf_pad = lf.Padding();
  static constexpr size_t kSigmaPadPerIter[4] = {0, 0, 2, 6};

  struct CandTile {
    double score;
    size_t tile;
    size_t group;
  };
  enum class Stage2MoveKind : uint8_t {
    kUnsignalDip = 0,
    kUnsignalTile = 1,
    kRdoqMinus = 2,
    kQfMinus = 3,
    kQfPlus = 4,
    kRdoqPlus = 5,
    kTrRdoqMinus = 6,
    kTrRdoqPlus = 7,
    kTrQfPlus = 8,
    kTrUnsignalDip = 9,
  };
  struct Stage2QueueEntry {
    uint32_t target_idx;
    uint16_t tr_count;
    int16_t ref_q;
    Stage2MoveKind kind;
  };
  auto map_alpha_to_rdoq = [](float alpha) -> int8_t {
    if (alpha < 1.23f) return -1;
    if (alpha < 1.38f) return -2;
    if (alpha < 1.55f) return -3;
    return -4;
  };
  size_t unsignal_moves = 0;

  auto run_stage2_queue =
      [&](size_t max_moves_cap,
          const std::vector<std::vector<Stage2QueueEntry>>& queues,
          size_t* pass_moves) -> Status {
    std::vector<size_t> group_q_pos(num_groups, 0);
    while (*pass_moves < max_moves_cap) {
      struct SavedTrState {
        uint32_t tr_idx;
        int32_t old_q;
        int8_t old_rl;
      };
      struct BatchItem {
        size_t group;
        uint32_t target_idx;
        Stage2MoveKind kind;
        float cand_bits;
        Rect dirty_px;
        std::vector<SavedTrState> saved_trs;
        bool any_qf_changed;
        bool any_rl_changed;
      };
      std::vector<BatchItem> batch;
      for (size_t g = 0; g < num_groups; ++g) {
        auto& trs = group_caches[g].transforms;
        while (group_q_pos[g] < queues[g].size()) {
          const Stage2QueueEntry qe = queues[g][group_q_pos[g]++];
          BatchItem item;
          item.group = g;
          item.target_idx = qe.target_idx;
          item.kind = qe.kind;
          item.cand_bits = 0.0f;
          item.dirty_px = Rect(0, 0, 0, 0);
          item.any_qf_changed = false;
          item.any_rl_changed = false;

          if (qe.kind == Stage2MoveKind::kUnsignalDip) {
            const size_t i0 = qe.target_idx;
            const size_t i1 = i0 + qe.tr_count - 1;
            if (qe.tr_count == 0 || i1 >= trs.size()) continue;
            const int32_t q_low = trs[i0].cur_quant_ac;
            bool valid_run = true;
            for (size_t i = i0; i <= i1; ++i) {
              if (trs[i].cur_quant_ac != q_low || trs[i].cur_rdoq_level != 0) {
                valid_run = false;
                break;
              }
            }
            if (!valid_run) continue;
            const int32_t q_l = (i0 > 0) ? trs[i0 - 1].cur_quant_ac : 0;
            const int32_t q_r =
                (i1 + 1 < trs.size()) ? trs[i1 + 1].cur_quant_ac : 0;
            int32_t q_tgt = 0;
            if (q_l > q_low && q_r > q_low) {
              q_tgt = std::min(q_l, q_r);
            } else if (std::max(q_l, q_r) > q_low && qe.tr_count <= 2) {
              q_tgt = std::max(q_l, q_r);
            } else {
              continue;
            }
            const float alpha =
                static_cast<float>(q_tgt) / static_cast<float>(q_low);
            if (alpha < 1.05f || alpha > 1.72f) continue;
            const int8_t cand_rl = map_alpha_to_rdoq(alpha);
            item.saved_trs.reserve(qe.tr_count);
            for (size_t i = i0; i <= i1; ++i) {
              item.saved_trs.push_back(
                  {static_cast<uint32_t>(i), trs[i].cur_quant_ac,
                   trs[i].cur_rdoq_level});
              trs[i].cur_quant_ac = q_tgt;
              trs[i].cur_rdoq_level = cand_rl;
            }
            item.any_qf_changed = true;
            item.any_rl_changed = true;
          } else if (qe.kind == Stage2MoveKind::kUnsignalTile) {
            const size_t t = qe.target_idx;
            if (tile_tr_indices[t].empty()) continue;
            bool can_unsig = true;
            for (size_t idx : tile_tr_indices[t]) {
              if (trs[idx].cur_rdoq_level != 0 ||
                  trs[idx].cur_quant_ac >= qe.ref_q) {
                can_unsig = false;
                break;
              }
            }
            if (!can_unsig) continue;
            item.saved_trs.reserve(tile_tr_indices[t].size());
            for (size_t idx : tile_tr_indices[t]) {
              const int32_t oq = trs[idx].cur_quant_ac;
              const int32_t nq = std::min<int32_t>(oq + 1, qe.ref_q);
              const float alpha =
                  static_cast<float>(nq) / static_cast<float>(oq);
              const int8_t nrl = map_alpha_to_rdoq(alpha);
              item.saved_trs.push_back(
                  {static_cast<uint32_t>(idx), oq, trs[idx].cur_rdoq_level});
              if (nq != oq) item.any_qf_changed = true;
              if (nrl != trs[idx].cur_rdoq_level) item.any_rl_changed = true;
              trs[idx].cur_quant_ac = nq;
              trs[idx].cur_rdoq_level = nrl;
            }
          } else if (qe.kind == Stage2MoveKind::kTrRdoqMinus ||
                     qe.kind == Stage2MoveKind::kTrRdoqPlus ||
                     qe.kind == Stage2MoveKind::kTrQfPlus ||
                     qe.kind == Stage2MoveKind::kTrUnsignalDip) {
            const size_t idx = qe.target_idx;
            if (idx >= trs.size()) continue;
            const int32_t oq = trs[idx].cur_quant_ac;
            const int8_t orl = trs[idx].cur_rdoq_level;
            int32_t nq = oq;
            int8_t nrl = orl;
            if (qe.kind == Stage2MoveKind::kTrRdoqMinus) {
              nrl = static_cast<int8_t>(std::max<int>(-4, orl - 1));
            } else if (qe.kind == Stage2MoveKind::kTrRdoqPlus) {
              if (orl == 0) nrl = 1;
            } else if (qe.kind == Stage2MoveKind::kTrUnsignalDip) {
              if (orl == 0 && oq < qe.ref_q) {
                const int32_t cand_q = qe.ref_q;
                const float alpha =
                    static_cast<float>(cand_q) / static_cast<float>(oq);
                if (alpha >= 1.05f && alpha <= 1.72f) {
                  nq = cand_q;
                  nrl = map_alpha_to_rdoq(alpha);
                }
              }
            } else {
              nq = std::min<int32_t>(oq + 1, 255);
            }
            item.saved_trs.push_back({static_cast<uint32_t>(idx), oq, orl});
            if (nq != oq) item.any_qf_changed = true;
            if (nrl != orl) item.any_rl_changed = true;
            trs[idx].cur_quant_ac = nq;
            trs[idx].cur_rdoq_level = nrl;
          } else {
            const size_t t = qe.target_idx;
            item.saved_trs.reserve(tile_tr_indices[t].size());
            for (size_t idx : tile_tr_indices[t]) {
              const int32_t oq = trs[idx].cur_quant_ac;
              const int8_t orl = trs[idx].cur_rdoq_level;
              int32_t nq = oq;
              int8_t nrl = orl;
              if (qe.kind == Stage2MoveKind::kRdoqMinus) {
                nrl = static_cast<int8_t>(std::max<int>(-4, orl - 1));
              } else if (qe.kind == Stage2MoveKind::kRdoqPlus) {
                if (orl == 0) nrl = 1;
              } else if (qe.kind == Stage2MoveKind::kQfMinus) {
                nq = std::max<int32_t>(oq - 1, 5);
                if (nq != oq && orl < 0) {
                  nrl = static_cast<int8_t>(orl + 1);
                }
              } else if (qe.kind == Stage2MoveKind::kQfPlus) {
                nq = std::min<int32_t>(oq + 1, 255);
              }
              item.saved_trs.push_back({static_cast<uint32_t>(idx), oq, orl});
              if (nq != oq) item.any_qf_changed = true;
              if (nrl != orl) item.any_rl_changed = true;
              trs[idx].cur_quant_ac = nq;
              trs[idx].cur_rdoq_level = nrl;
            }
          }

          if (!item.any_qf_changed && !item.any_rl_changed) {
            for (const SavedTrState& s : item.saved_trs) {
              trs[s.tr_idx].cur_quant_ac = s.old_q;
              trs[s.tr_idx].cur_rdoq_level = s.old_rl;
            }
            continue;
          }
          batch.push_back(std::move(item));
          break;
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
                    /*only_write_changed=*/true, &batch_pre_gab, nullptr,
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
        const Rect bgr = shared.frame_dim.BlockGroupRect(g);
        const double new_bits =
            cur_bits + static_cast<double>(item.cand_bits - cur_group_bits[g]);
        const double delta_rate = rate_term(new_bits + cur_sh_bits) -
                                  rate_term(cur_bits + cur_sh_bits);

        const bool is_unsig = (item.kind == Stage2MoveKind::kUnsignalDip ||
                               item.kind == Stage2MoveKind::kUnsignalTile ||
                               item.kind == Stage2MoveKind::kTrUnsignalDip);
        const bool is_tr_move = (item.kind == Stage2MoveKind::kTrRdoqMinus ||
                                 item.kind == Stage2MoveKind::kTrRdoqPlus ||
                                 item.kind == Stage2MoveKind::kTrQfPlus ||
                                 item.kind == Stage2MoveKind::kTrUnsignalDip);
        const bool is_rdoq_only = (item.kind == Stage2MoveKind::kRdoqMinus ||
                                   item.kind == Stage2MoveKind::kRdoqPlus ||
                                   item.kind == Stage2MoveKind::kTrRdoqMinus ||
                                   item.kind == Stage2MoveKind::kTrRdoqPlus);
        const bool is_minus = (item.kind == Stage2MoveKind::kRdoqMinus ||
                               item.kind == Stage2MoveKind::kQfMinus ||
                               item.kind == Stage2MoveKind::kTrRdoqMinus);
        const bool has_epf_sigma_change =
            (lf.epf_iters > 0) && item.any_qf_changed;

        bool skip_early = (*pass_moves >= max_moves_cap);
        if (!skip_early) {
          if (is_rdoq_only &&
              std::abs(item.cand_bits - cur_group_bits[g]) < 1e-3f) {
            skip_early = true;
          } else if (item.kind == Stage2MoveKind::kTrRdoqMinus) {
            if (item.cand_bits - cur_group_bits[g] > -1.2f) skip_early = true;
          } else if (is_minus) {
            const double req_save =
                (item.kind == Stage2MoveKind::kRdoqMinus) ? -0.001 : -0.002;
            if (delta_rate >= req_save) skip_early = true;
          } else if (is_unsig) {
            if (delta_rate > 0.12 ||
                (item.dirty_px.xsize() == 0 && !has_epf_sigma_change &&
                 delta_rate >= -1e-4)) {
              skip_early = true;
            }
          } else {
            if (item.dirty_px.xsize() == 0 && !has_epf_sigma_change &&
                delta_rate >= 0.0) {
              skip_early = true;
            }
          }
        }

        if (skip_early) {
          if (item.dirty_px.xsize() != 0 && item.dirty_px.ysize() != 0) {
            for (size_t c = 0; c < 3; ++c) {
              for (size_t y = 0; y < item.dirty_px.ysize(); ++y) {
                memcpy(batch_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       cur_pre_gab.ConstPlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       item.dirty_px.xsize() * sizeof(float));
              }
            }
          }
          for (const SavedTrState& s : item.saved_trs) {
            AcTransformCache& tr = group_caches[g].transforms[s.tr_idx];
            tr.cur_quant_ac = s.old_q;
            tr.cur_rdoq_level = s.old_rl;
          }
          continue;
        }

        Rect sigma_upd_rect(0, 0, 0, 0);
        if (item.any_qf_changed) {
          size_t sbx0 = xsize_blocks;
          size_t sby0 = ysize_blocks;
          size_t sbx1 = 0;
          size_t sby1 = 0;
          for (const SavedTrState& s : item.saved_trs) {
            const AcTransformCache& tr = group_caches[g].transforms[s.tr_idx];
            if (tr.cur_quant_ac == s.old_q) continue;
            for (size_t iy = 0; iy < tr.cov_y; ++iy) {
              int32_t* JXL_RESTRICT qrow =
                  bgr.Row(&rdo_quant_field, tr.by + iy);
              for (size_t ix = 0; ix < tr.cov_x; ++ix) {
                qrow[tr.bx + ix] = tr.cur_quant_ac;
              }
            }
            sbx0 = std::min(sbx0, bgr.x0() + tr.bx);
            sby0 = std::min(sby0, bgr.y0() + tr.by);
            sbx1 = std::max(sbx1, bgr.x0() + tr.bx + tr.cov_x);
            sby1 = std::max(sby1, bgr.y0() + tr.by + tr.cov_y);
          }
          if (sbx0 < sbx1 && sby0 < sby1) {
            sigma_upd_rect = Rect(sbx0, sby0, sbx1 - sbx0, sby1 - sby0);
            if (lf.epf_iters > 0) {
              epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale,
                                    rdo_quant_field, rdo_sharpness,
                                    sigma_upd_rect);
            }
          }
        }

        size_t dlx0 = xsize;
        size_t dly0 = ysize;
        size_t dlx1 = 0;
        size_t dly1 = 0;
        if (item.dirty_px.xsize() != 0 && item.dirty_px.ysize() != 0) {
          for (size_t c = 0; c < 3; ++c) {
            for (size_t y = 0; y < item.dirty_px.ysize(); ++y) {
              memcpy(cand_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                         item.dirty_px.x0(),
                     batch_pre_gab.ConstPlaneRow(c, item.dirty_px.y0() + y) +
                         item.dirty_px.x0(),
                     item.dirty_px.xsize() * sizeof(float));
            }
          }
          dlx0 = std::min(
              dlx0,
              (item.dirty_px.x0() > lf_pad) ? (item.dirty_px.x0() - lf_pad)
                                            : 0);
          dly0 = std::min(
              dly0,
              (item.dirty_px.y0() > lf_pad) ? (item.dirty_px.y0() - lf_pad)
                                            : 0);
          dlx1 = std::max(
              dlx1,
              std::min(xsize,
                       item.dirty_px.x0() + item.dirty_px.xsize() + lf_pad));
          dly1 = std::max(
              dly1,
              std::min(ysize,
                       item.dirty_px.y0() + item.dirty_px.ysize() + lf_pad));
        }
        if (has_epf_sigma_change && sigma_upd_rect.xsize() != 0) {
          const size_t spad = kSigmaPadPerIter[lf.epf_iters];
          const size_t px0 = sigma_upd_rect.x0() * kBlockDim;
          const size_t py0 = sigma_upd_rect.y0() * kBlockDim;
          const size_t px1 = std::min(
              xsize, (sigma_upd_rect.x0() + sigma_upd_rect.xsize()) * kBlockDim);
          const size_t py1 = std::min(
              ysize, (sigma_upd_rect.y0() + sigma_upd_rect.ysize()) * kBlockDim);
          dlx0 = std::min(dlx0, (px0 > spad) ? (px0 - spad) : 0);
          dly0 = std::min(dly0, (py0 > spad) ? (py0 - spad) : 0);
          dlx1 = std::max(dlx1, std::min(xsize, px1 + spad));
          dly1 = std::max(dly1, std::min(ysize, py1 + spad));
        }
        Rect dirty_lin(0, 0, 0, 0);
        if (dlx0 < dlx1 && dly0 < dly1) {
          dirty_lin = Rect(dlx0, dly0, dlx1 - dlx0, dly1 - dly0);
        }

        if (dirty_lin.xsize() != 0 && dirty_lin.ysize() != 0) {
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
          JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
              memory_manager, lf, shared, cand_pre_gab, feat_ptr, epf_sigma,
              epf_ws, dirty_lin, &cur_lin));
        }

        std::array<double, 4> cand_d = cur_d;
        if (dirty_lin.xsize() != 0) {
          JXL_RETURN_IF_ERROR(
              update_4m_checkpointed(cur_lin, dirty_lin, &cand_d));
        }

        double norm_sum = 0.0;
        bool pareto_ok = true;
        for (size_t m = 0; m < 4; ++m) {
          if (weight_m[m] <= 0.0) continue;
          const double nm = (cur_d[m] - cand_d[m]) / slope_m[m];
          norm_sum += weight_m[m] * nm;
          if (is_minus || (is_unsig && delta_rate < 0.0)) {
            if (nm - delta_rate < 0.0) {
              pareto_ok = false;
            }
          } else if (is_unsig) {
            const double floor_u = is_tr_move ? 0.005 : 0.015;
            if (nm < -0.15594302155620929 * std::max(floor_u, delta_rate)) {
              pareto_ok = false;
            }
          } else {
            const double floor_r =
                is_tr_move ? (is_rdoq_only ? 0.003 : 0.005)
                           : (is_rdoq_only ? 0.015 : 0.025);
            if (nm < -0.25990503592701547 *
                         std::max(floor_r, std::abs(delta_rate))) {
              pareto_ok = false;
            }
          }
        }
        const double j_gain = norm_sum - delta_rate;
        double min_gain = 1e-6;
        if (item.kind == Stage2MoveKind::kRdoqMinus) {
          min_gain = 0.002;
        } else if (item.kind == Stage2MoveKind::kTrRdoqMinus) {
          min_gain = 0.0004;
        } else if (item.kind == Stage2MoveKind::kTrRdoqPlus) {
          min_gain = 0.0003;
        } else if (item.kind == Stage2MoveKind::kTrQfPlus) {
          min_gain = 0.0005;
        } else if (item.kind == Stage2MoveKind::kTrUnsignalDip) {
          min_gain = (delta_rate < 0.0) ? 0.0006 : 0.0010;
        } else if (item.kind == Stage2MoveKind::kQfMinus) {
          min_gain = 0.004;
        } else if (is_unsig) {
          min_gain = (delta_rate < 0.0) ? 0.002 : 0.003;
        }

        if (j_gain > min_gain && pareto_ok) {
          for (AcTransformCache& tr : group_caches[g].transforms) {
            tr.cur_slot = tr.cand_slot;
          }
          if (item.dirty_px.xsize() != 0 && item.dirty_px.ysize() != 0) {
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
          cur_d = cand_d;
          ++(*pass_moves);
          if (item.any_qf_changed) ++total_moves;
          if (item.any_rl_changed) ++rdoq_moves;
          if (item.kind == Stage2MoveKind::kRdoqMinus) {
            tile_rdoq[item.target_idx] -= 1;
          } else if (item.kind == Stage2MoveKind::kRdoqPlus) {
            tile_rdoq[item.target_idx] = 1;
          } else if (item.kind == Stage2MoveKind::kQfMinus) {
            tile_step[item.target_idx] -= 1;
            if (tile_rdoq[item.target_idx] < 0) {
              tile_rdoq[item.target_idx] += 1;
            }
          } else if (item.kind == Stage2MoveKind::kQfPlus) {
            tile_step[item.target_idx] += 1;
          }
        } else {
          if (dirty_lin.xsize() != 0) {
            wd->Rollback(cp_wd);
            ssim2->Rollback(cp_ss2);
            ba3->Rollback(cp_ba3);
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
          }
          if (item.dirty_px.xsize() != 0 && item.dirty_px.ysize() != 0) {
            for (size_t c = 0; c < 3; ++c) {
              for (size_t y = 0; y < item.dirty_px.ysize(); ++y) {
                const float* src_r =
                    cur_pre_gab.ConstPlaneRow(c, item.dirty_px.y0() + y) +
                    item.dirty_px.x0();
                memcpy(cand_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       src_r, item.dirty_px.xsize() * sizeof(float));
                memcpy(batch_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       src_r, item.dirty_px.xsize() * sizeof(float));
              }
            }
          }
          for (const SavedTrState& s : item.saved_trs) {
            AcTransformCache& tr = group_caches[g].transforms[s.tr_idx];
            tr.cur_quant_ac = s.old_q;
            tr.cur_rdoq_level = s.old_rl;
          }
          if (item.any_qf_changed && sigma_upd_rect.xsize() != 0) {
            for (const SavedTrState& s : item.saved_trs) {
              const AcTransformCache& tr = group_caches[g].transforms[s.tr_idx];
              for (size_t iy = 0; iy < tr.cov_y; ++iy) {
                int32_t* JXL_RESTRICT qrow =
                    bgr.Row(&rdo_quant_field, tr.by + iy);
                for (size_t ix = 0; ix < tr.cov_x; ++ix) {
                  qrow[tr.bx + ix] = s.old_q;
                }
              }
            }
            if (lf.epf_iters > 0) {
              epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale,
                                    rdo_quant_field, rdo_sharpness,
                                    sigma_upd_rect);
            }
          }
        }
      }
    }
    return true;
  };

  const float d_target_s2 = cparams.butteraugli_distance;
  const float eob_scale_s2 =
      Clamp1((d_target_s2 - 0.28f) / (d_target_s2 + 0.50f) * 1.45f, 0.0f, 1.0f);

  auto find_2d_dip_target_q = [&](size_t g,
                                  const AcTransformCache& tr) -> int16_t {
    if (tr.cur_rdoq_level != 0) return 0;
    const size_t cov = static_cast<size_t>(tr.cov_x) * tr.cov_y;
    if (cov == 0 || cov > 4) return 0;
    const Rect bgr = shared.frame_dim.BlockGroupRect(g);
    const size_t gx = bgr.x0() + tr.bx;
    const size_t gy = bgr.y0() + tr.by;
    const size_t gw = tr.cov_x;
    const size_t gh = tr.cov_y;
    const int32_t q_cur = tr.cur_quant_ac;
    size_t num_nbr = 0;
    size_t num_hi = 0;
    int32_t min_hi = 256;
    if (gx > 0) {
      for (size_t iy = 0; iy < gh && gy + iy < ysize_blocks; ++iy) {
        const int32_t qn = rdo_quant_field.ConstRow(gy + iy)[gx - 1];
        ++num_nbr;
        if (qn > q_cur) {
          ++num_hi;
          min_hi = std::min(min_hi, qn);
        }
      }
    }
    if (gx + gw < xsize_blocks) {
      for (size_t iy = 0; iy < gh && gy + iy < ysize_blocks; ++iy) {
        const int32_t qn = rdo_quant_field.ConstRow(gy + iy)[gx + gw];
        ++num_nbr;
        if (qn > q_cur) {
          ++num_hi;
          min_hi = std::min(min_hi, qn);
        }
      }
    }
    if (gy > 0) {
      for (size_t ix = 0; ix < gw && gx + ix < xsize_blocks; ++ix) {
        const int32_t qn = rdo_quant_field.ConstRow(gy - 1)[gx + ix];
        ++num_nbr;
        if (qn > q_cur) {
          ++num_hi;
          min_hi = std::min(min_hi, qn);
        }
      }
    }
    if (gy + gh < ysize_blocks) {
      for (size_t ix = 0; ix < gw && gx + ix < xsize_blocks; ++ix) {
        const int32_t qn = rdo_quant_field.ConstRow(gy + gh)[gx + ix];
        ++num_nbr;
        if (qn > q_cur) {
          ++num_hi;
          min_hi = std::min(min_hi, qn);
        }
      }
    }
    if (num_nbr < 2 || num_hi * 4 < 3 * num_nbr || min_hi > 255) return 0;
    const int32_t max_alpha_q =
        static_cast<int32_t>(std::floor(1.70f * static_cast<float>(q_cur)));
    const int32_t q_tgt = std::min(min_hi, max_alpha_q);
    if (q_tgt <= q_cur) return 0;
    const float alpha = static_cast<float>(q_tgt) / static_cast<float>(q_cur);
    if (alpha < 1.05f || alpha > 1.72f) return 0;
    return static_cast<int16_t>(q_tgt);
  };

  // Pass 2A0: Unsignal pre-existing AQ dips & high-AQ plateaus by raising
  // signaled quant_ac (collapsing out_qf Modular transitions & shrinking
  // epf_sigma blur) while applying matching virtual AQ quantization
  // (rdoq_level in {-4, -3, -2, -1}).
  if (eob_scale_s2 > 0.0f) {
    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
    tile_err.assign(num_tiles, 0.0);
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const size_t ty = by / kTileBlocks;
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        const size_t tx = bx / kTileBlocks;
        tile_err[ty * tiles_x + tx] += blk_err_cur[by * xsize_blocks + bx];
      }
    }
    auto qf_jump_cost = [](int32_t qa, int32_t qb) -> double {
      const uint32_t a = static_cast<uint32_t>(std::abs(qa - qb));
      return (a == 0)
                 ? 0.65
                 : (a == 1 ? 2.40 : 2.40 + 1.25 * std::log2(static_cast<double>(a)));
    };

    std::vector<std::vector<Stage2QueueEntry>> unsig_queues(num_groups);
    std::vector<int32_t> q_scratch;
    for (size_t g = 0; g < num_groups; ++g) {
      const auto& trs = group_caches[g].transforms;
      if (trs.empty()) continue;
      q_scratch.resize(trs.size());
      for (size_t i = 0; i < trs.size(); ++i) q_scratch[i] = trs[i].cur_quant_ac;
      std::nth_element(q_scratch.begin(), q_scratch.begin() + q_scratch.size() / 2,
                       q_scratch.end());
      const int32_t q_med = q_scratch[q_scratch.size() / 2];

      struct DipCand {
        double score;
        uint32_t i0;
        uint16_t cnt;
      };
      std::vector<DipCand> dips;
      size_t i = 0;
      while (i < trs.size()) {
        const int32_t q_low = trs[i].cur_quant_ac;
        size_t j = i;
        while (j + 1 < trs.size() && trs[j + 1].cur_quant_ac == q_low) ++j;
        const size_t len = j - i + 1;
        if (len <= 4) {
          const int32_t q_l = (i > 0) ? trs[i - 1].cur_quant_ac : 0;
          const int32_t q_r =
              (j + 1 < trs.size()) ? trs[j + 1].cur_quant_ac : 0;
          int32_t q_tgt = 0;
          if (q_l > q_low && q_r > q_low) {
            q_tgt = std::min(q_l, q_r);
          } else if (std::max(q_l, q_r) > q_low && len <= 2) {
            q_tgt = std::max(q_l, q_r);
          }
          if (q_tgt > q_low) {
            const float alpha =
                static_cast<float>(q_tgt) / static_cast<float>(q_low);
            if (alpha >= 1.05f && alpha <= 1.72f) {
              double save = 0.0;
              const int32_t ql_prev = (i > 0) ? q_l : 1;
              save += qf_jump_cost(q_low, ql_prev) - qf_jump_cost(q_tgt, ql_prev);
              if (j + 1 < trs.size()) {
                save += qf_jump_cost(q_r, q_low) - qf_jump_cost(q_r, q_tgt);
              }
              const double score = save + 1.5 * (1.0 - 1.0 / alpha);
              dips.push_back(
                  {score, static_cast<uint32_t>(i), static_cast<uint16_t>(len)});
            }
          }
        }
        i = j + 1;
      }
      std::sort(dips.begin(), dips.end(),
                [](const DipCand& a, const DipCand& b) {
                  return (a.score != b.score) ? (a.score > b.score)
                                              : (a.i0 < b.i0);
                });
      for (size_t k = 0; k < std::min<size_t>(12, dips.size()); ++k) {
        unsig_queues[g].push_back(
            {dips[k].i0, dips[k].cnt, 0, Stage2MoveKind::kUnsignalDip});
      }

      if (q_med >= 4) {
        std::vector<CandTile> hi_aq_tiles;
        for (size_t t = 0; t < num_tiles; ++t) {
          if (tile_group[t] != g || tile_tr_indices[t].empty()) continue;
          bool all_below_med = true;
          int32_t min_q = 255;
          for (size_t idx : tile_tr_indices[t]) {
            const int32_t qt = trs[idx].cur_quant_ac;
            if (qt >= q_med) {
              all_below_med = false;
              break;
            }
            min_q = std::min(min_q, qt);
          }
          if (!all_below_med || tile_err[t] <= 1e-12) continue;
          const double sc =
              tile_err[t] / std::max(1.0, static_cast<double>(min_q));
          hi_aq_tiles.push_back({sc, t, g});
        }
        std::sort(hi_aq_tiles.begin(), hi_aq_tiles.end(),
                  [](const CandTile& a, const CandTile& b) {
                    return (a.score != b.score) ? (a.score > b.score)
                                                : (a.tile < b.tile);
                  });
        for (size_t k = 0; k < std::min<size_t>(6, hi_aq_tiles.size()); ++k) {
          unsig_queues[g].push_back(
              {static_cast<uint32_t>(hi_aq_tiles[k].tile), 0,
               static_cast<int16_t>(q_med), Stage2MoveKind::kUnsignalTile});
        }
      }
    }
    const size_t max_unsig_moves = std::min<size_t>(
        240, std::max<size_t>(4, static_cast<size_t>(0.35f * num_tiles)));
    JXL_RETURN_IF_ERROR(
        run_stage2_queue(max_unsig_moves, unsig_queues, &unsignal_moves));

    // Also unsignal 2D-surrounded sub-tile AQ dips inside mixed 32x32 tiles
    // that were not contiguous 1D runs in `transforms` order.
    struct TrDipCand {
      double score;
      uint32_t tr_idx;
      int16_t ref_q;
    };
    std::vector<std::vector<Stage2QueueEntry>> tr_unsig_queues(num_groups);
    bool any_tr_dip = false;
    for (size_t g = 0; g < num_groups; ++g) {
      const Rect bgr = shared.frame_dim.BlockGroupRect(g);
      const auto& trs = group_caches[g].transforms;
      std::vector<TrDipCand> tdips;
      for (size_t i = 0; i < trs.size(); ++i) {
        const AcTransformCache& tr = trs[i];
        if (tile_tr_indices[tr.tile_idx].size() < 2) continue;
        const int16_t q_tgt = find_2d_dip_target_q(g, tr);
        if (q_tgt <= 0) continue;
        double tr_err = 0.0;
        for (size_t iy = 0; iy < tr.cov_y; ++iy) {
          for (size_t ix = 0; ix < tr.cov_x; ++ix) {
            tr_err += blk_err_cur[(bgr.y0() + tr.by + iy) * xsize_blocks +
                                  (bgr.x0() + tr.bx + ix)];
          }
        }
        const double alpha =
            static_cast<double>(q_tgt) / static_cast<double>(tr.cur_quant_ac);
        const double sc = (1.0 - 1.0 / alpha) + 0.05 / (0.01 + tr_err);
        tdips.push_back({sc, static_cast<uint32_t>(i), q_tgt});
      }
      if (tdips.empty()) continue;
      std::sort(tdips.begin(), tdips.end(),
                [](const TrDipCand& a, const TrDipCand& b) {
                  return (a.score != b.score) ? (a.score > b.score)
                                              : (a.tr_idx < b.tr_idx);
                });
      for (size_t k = 0; k < std::min<size_t>(10, tdips.size()); ++k) {
        tr_unsig_queues[g].push_back(
            {tdips[k].tr_idx, 0, tdips[k].ref_q,
             Stage2MoveKind::kTrUnsignalDip});
        any_tr_dip = true;
      }
    }
    if (any_tr_dip) {
      size_t tr_unsig_moves = 0;
      const size_t max_tr_unsig =
          std::min<size_t>(240, std::max<size_t>(4, num_tiles));
      JXL_RETURN_IF_ERROR(
          run_stage2_queue(max_tr_unsig, tr_unsig_queues, &tr_unsig_moves));
      unsignal_moves += tr_unsig_moves;
    }
  }

  // Pass 2C: Increase quantization on masked tiles WITHOUT signaling higher AQ
  // (rdoq_level in {-1, -2} at zero out_qf cost and zero epf_sigma inflation)
  // before testing signaled qf - 1 in Pass 2A.
  if (eob_scale_s2 > 0.0f) {
    const size_t max_rdoq_minus = std::min<size_t>(
        220, std::max<size_t>(1, static_cast<size_t>(0.25f * num_tiles)));
    for (int iter_rm = 0; iter_rm < 2 && rdoq_minus_moves < max_rdoq_minus;
         ++iter_rm) {
      const size_t prev_rm = rdoq_minus_moves;
      accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
      tile_err.assign(num_tiles, 0.0);
      for (size_t by = 0; by < ysize_blocks; ++by) {
        const size_t ty = by / kTileBlocks;
        for (size_t bx = 0; bx < xsize_blocks; ++bx) {
          const size_t tx = bx / kTileBlocks;
          tile_err[ty * tiles_x + tx] += blk_err_cur[by * xsize_blocks + bx];
        }
      }
      std::vector<CandTile> rdoq_minus_cands;
      rdoq_minus_cands.reserve(num_tiles);
      for (size_t t = 0; t < num_tiles; ++t) {
        if (tile_tr_indices[t].empty() || tile_step[t] > 0) continue;
        if (tile_rdoq[t] != static_cast<int8_t>(-iter_rm)) continue;
        if (cur_tile_bits[t] <= 4.0f) continue;
        const size_t g = tile_group[t];
        const double save_frac = (iter_rm == 0) ? 0.08 : 0.06;
        const double dbits_est =
            save_frac * static_cast<double>(cur_tile_bits[t]);
        const double drate_save =
            rate_term(cur_bits + cur_sh_bits) -
            rate_term(std::max(1.0, cur_bits + cur_sh_bits - dbits_est));
        const double proxy_j = drate_save - 0.45 * tile_err[t];
        if (proxy_j > 0.001) {
          rdoq_minus_cands.push_back({proxy_j, t, g});
        }
      }
      if (rdoq_minus_cands.empty()) break;
      std::sort(rdoq_minus_cands.begin(), rdoq_minus_cands.end(),
                [](const CandTile& a, const CandTile& b) {
                  return (a.score != b.score) ? (a.score > b.score)
                                              : (a.tile < b.tile);
                });
      std::vector<std::vector<Stage2QueueEntry>> group_queues(num_groups);
      for (const CandTile& c : rdoq_minus_cands) {
        if (group_queues[c.group].size() < 12) {
          group_queues[c.group].push_back(
              {static_cast<uint32_t>(c.tile), 0, 0,
               Stage2MoveKind::kRdoqMinus});
        }
      }
      JXL_RETURN_IF_ERROR(
          run_stage2_queue(max_rdoq_minus, group_queues, &rdoq_minus_moves));
      if (rdoq_minus_moves == prev_rm) break;
    }
  }

  // Pass 2A: Signal higher AQ (cur_quant_ac - 1) on q >= 6 tiles ONLY when
  // signaled AQ beats Pass 2C's un-signaled quantization even after paying
  // out_qf bits and live epf_sigma blur under unanimous 4-metric agreement.
  {
    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
    tile_err.assign(num_tiles, 0.0);
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const size_t ty = by / kTileBlocks;
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        const size_t tx = bx / kTileBlocks;
        tile_err[ty * tiles_x + tx] += blk_err_cur[by * xsize_blocks + bx];
      }
    }
    std::vector<CandTile> minus_cands;
    minus_cands.reserve(num_tiles);
    for (size_t t = 0; t < num_tiles; ++t) {
      if (tile_tr_indices[t].empty() || tile_step[t] != 0) continue;
      const size_t g = tile_group[t];
      bool all_above_floor = true;
      for (size_t idx : tile_tr_indices[t]) {
        if (group_caches[g].transforms[idx].cur_quant_ac < 6) {
          all_above_floor = false;
          break;
        }
      }
      if (!all_above_floor) continue;
      const double rdoq_disc = (tile_rdoq[t] < 0) ? 0.65 : 1.0;
      const double dbits_save =
          rdoq_disc *
          static_cast<double>(std::min(
              cur_tile_bits[t],
              std::max(0.0f, fine_tile_bits[t] - plain_tile_bits[t])));
      if (dbits_save <= 1.5) continue;
      const double drate_save =
          rate_term(cur_bits + cur_sh_bits) -
          rate_term(std::max(1.0, cur_bits + cur_sh_bits - dbits_save));
      const double proxy_j = 0.85 * drate_save - tile_err[t];
      if (proxy_j > 0.002) {
        minus_cands.push_back({proxy_j, t, g});
      }
    }
    if (!minus_cands.empty()) {
      std::sort(minus_cands.begin(), minus_cands.end(),
                [](const CandTile& a, const CandTile& b) {
                  return (a.score != b.score) ? (a.score > b.score)
                                              : (a.tile < b.tile);
                });
      std::vector<std::vector<Stage2QueueEntry>> group_queues(num_groups);
      for (const CandTile& c : minus_cands) {
        if (group_queues[c.group].size() < kMaxMinusPerGroup) {
          group_queues[c.group].push_back(
              {static_cast<uint32_t>(c.tile), 0, 0, Stage2MoveKind::kQfMinus});
        }
      }
      JXL_RETURN_IF_ERROR(
          run_stage2_queue(max_minus_moves, group_queues, &minus_moves));
    }
  }

  // Pass 2B: Invest +1 and +2 qf steps on high-salience tiles (with live
  // epf_sigma shrinkage reward).
  for (size_t iter = 0; iter < kMaxPlusIters && plus_moves < max_plus_moves;
       ++iter) {
    const size_t prev_plus = plus_moves;
    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
    tile_err.assign(num_tiles, 0.0);
    double total_blk_gain = 0.0;
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const size_t ty = by / kTileBlocks;
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        const size_t tx = bx / kTileBlocks;
        const double v = blk_err_cur[by * xsize_blocks + bx];
        tile_err[ty * tiles_x + tx] += v;
        total_blk_gain += v;
      }
    }
    if (total_blk_gain <= 1e-12) break;

    std::vector<CandTile> all_cands;
    all_cands.reserve(num_tiles);
    for (size_t t = 0; t < num_tiles; ++t) {
      if (tile_tr_indices[t].empty()) continue;
      if (tile_step[t] < 0 || tile_rdoq[t] < 0 ||
          tile_step[t] > static_cast<int8_t>(iter)) {
        continue;
      }
      const size_t g = tile_group[t];
      bool can_step = false;
      for (size_t idx : tile_tr_indices[t]) {
        if (group_caches[g].transforms[idx].cur_quant_ac < 255) {
          can_step = true;
          break;
        }
      }
      if (!can_step) continue;
      const double step_mul = (tile_step[t] > 0) ? 1.15 : 1.0;
      const double dbits =
          step_mul * static_cast<double>(
                         std::max(0.0f, fine_tile_bits[t] - plain_tile_bits[t]));
      const double drate = rate_term(cur_bits + cur_sh_bits + dbits) -
                           rate_term(cur_bits + cur_sh_bits);
      const double proxy_j = tile_err[t] - drate;
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

    std::vector<std::vector<Stage2QueueEntry>> group_queues(num_groups);
    for (const CandTile& c : all_cands) {
      if (group_queues[c.group].size() < kMaxPlusPerGroup) {
        group_queues[c.group].push_back(
            {static_cast<uint32_t>(c.tile), 0, 0, Stage2MoveKind::kQfPlus});
      }
    }
    JXL_RETURN_IF_ERROR(
        run_stage2_queue(max_plus_moves, group_queues, &plus_moves));
    if (plus_moves == prev_plus) break;
  }

  // Pass 2D: Zero-bitstream-overhead +1 RDOQ fidelity boost on salient tiles.
  if (eob_scale_s2 > 0.0f) {
    const size_t max_rdoq_plus = std::min<size_t>(
        260, std::max<size_t>(1, static_cast<size_t>(0.30f * num_tiles)));
    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
    tile_err.assign(num_tiles, 0.0);
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const size_t ty = by / kTileBlocks;
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        const size_t tx = bx / kTileBlocks;
        tile_err[ty * tiles_x + tx] += blk_err_cur[by * xsize_blocks + bx];
      }
    }
    std::vector<CandTile> rdoq_plus_cands;
    rdoq_plus_cands.reserve(num_tiles);
    for (size_t t = 0; t < num_tiles; ++t) {
      if (tile_tr_indices[t].empty() || tile_rdoq[t] != 0 || tile_step[t] < 0) {
        continue;
      }
      const size_t g = tile_group[t];
      const double dbits_est =
          0.35 * static_cast<double>(
                     std::max(2.0f, fine_tile_bits[t] - plain_tile_bits[t]));
      const double drate = rate_term(cur_bits + cur_sh_bits + dbits_est) -
                           rate_term(cur_bits + cur_sh_bits);
      const double proxy_j = 0.55 * tile_err[t] - drate;
      if (proxy_j > 0.0) {
        rdoq_plus_cands.push_back({proxy_j, t, g});
      }
    }
    if (!rdoq_plus_cands.empty()) {
      std::sort(rdoq_plus_cands.begin(), rdoq_plus_cands.end(),
                [](const CandTile& a, const CandTile& b) {
                  return (a.score != b.score) ? (a.score > b.score)
                                              : (a.tile < b.tile);
                });
      std::vector<std::vector<Stage2QueueEntry>> group_queues(num_groups);
      for (const CandTile& c : rdoq_plus_cands) {
        if (group_queues[c.group].size() < 12) {
          group_queues[c.group].push_back(
              {static_cast<uint32_t>(c.tile), 0, 0, Stage2MoveKind::kRdoqPlus});
        }
      }
      JXL_RETURN_IF_ERROR(
          run_stage2_queue(max_rdoq_plus, group_queues, &rdoq_plus_moves));
    }
  }

  // Pass 2D_sub: Sub-tile per-transform refinement inside heterogeneous 32x32
  // tiles (tile_tr_indices[t].size() >= 2), allowing smooth/masked transforms
  // adjacent to sharp edges to drop rdoq_level without blurring the edge, and
  // allowing isolated sharp-edge transforms to receive +1 rdoq_level or +1
  // quant_ac (with local epf_sigma shrinkage) without wasting bits on adjacent
  // smooth blocks.
  if (eob_scale_s2 > 0.0f && num_blocks >= 64) {
    struct CandTransform {
      double score;
      uint32_t tr_idx;
      size_t group;
    };

    // 1) Un-signaled per-transform virtual AQ reclaim (kTrRdoqMinus) under
    // unanimous 4-metric agreement.
    size_t tr_rm_moves = 0;
    const size_t max_tr_rm = std::min<size_t>(
        480, std::max<size_t>(4, static_cast<size_t>(1.5f * num_tiles)));
    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
    std::vector<CandTransform> tr_rm_cands;
    for (size_t g = 0; g < num_groups; ++g) {
      const Rect bgr = shared.frame_dim.BlockGroupRect(g);
      const auto& trs = group_caches[g].transforms;
      for (size_t i = 0; i < trs.size(); ++i) {
        const AcTransformCache& tr = trs[i];
        if (tile_tr_indices[tr.tile_idx].size() < 2) continue;
        if (tr.cur_rdoq_level != 0 || tile_step[tr.tile_idx] > 0) continue;
        const float tr_bits = tr.slots[tr.cur_slot].bits;
        if (tr_bits < 4.5f) continue;
        double tr_err = 0.0;
        for (size_t iy = 0; iy < tr.cov_y; ++iy) {
          for (size_t ix = 0; ix < tr.cov_x; ++ix) {
            tr_err +=
                blk_err_cur[(bgr.y0() + tr.by + iy) * xsize_blocks +
                            (bgr.x0() + tr.bx + ix)];
          }
        }
        const double dbits_est = 0.10 * static_cast<double>(tr_bits);
        const double drate_save =
            rate_term(cur_bits + cur_sh_bits) -
            rate_term(std::max(1.0, cur_bits + cur_sh_bits - dbits_est));
        const double proxy_j = drate_save - 0.50 * tr_err;
        if (proxy_j > 0.0002) {
          tr_rm_cands.push_back({proxy_j, static_cast<uint32_t>(i), g});
        }
      }
    }
    if (!tr_rm_cands.empty()) {
      std::sort(tr_rm_cands.begin(), tr_rm_cands.end(),
                [](const CandTransform& a, const CandTransform& b) {
                  if (a.score != b.score) return a.score > b.score;
                  return (a.group != b.group) ? (a.group < b.group)
                                              : (a.tr_idx < b.tr_idx);
                });
      std::vector<std::vector<Stage2QueueEntry>> group_queues(num_groups);
      for (const CandTransform& c : tr_rm_cands) {
        if (group_queues[c.group].size() < 20) {
          group_queues[c.group].push_back(
              {c.tr_idx, 0, 0, Stage2MoveKind::kTrRdoqMinus});
        }
      }
      JXL_RETURN_IF_ERROR(
          run_stage2_queue(max_tr_rm, group_queues, &tr_rm_moves));
    }

    // 2) Zero-header per-transform RDOQ fidelity boost (kTrRdoqPlus) on salient
    // edge/contour transforms inside mixed tiles.
    size_t tr_rp_moves = 0;
    const size_t max_tr_rp = std::min<size_t>(
        400, std::max<size_t>(4, static_cast<size_t>(1.25f * num_tiles)));
    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
    double sum_blk_err = 0.0;
    for (size_t b = 0; b < num_blocks; ++b) sum_blk_err += blk_err_cur[b];
    const double mean_blk_err =
        sum_blk_err / static_cast<double>(std::max<size_t>(1, num_blocks));
    if (mean_blk_err > 1e-12) {
      std::vector<CandTransform> tr_rp_cands;
      for (size_t g = 0; g < num_groups; ++g) {
        const Rect bgr = shared.frame_dim.BlockGroupRect(g);
        const auto& trs = group_caches[g].transforms;
        for (size_t i = 0; i < trs.size(); ++i) {
          const AcTransformCache& tr = trs[i];
          if (tile_tr_indices[tr.tile_idx].size() < 2) continue;
          if (tr.cur_rdoq_level != 0 || tile_step[tr.tile_idx] < 0) continue;
          const size_t cov = static_cast<size_t>(tr.cov_x) * tr.cov_y;
          if (cov > 4) continue;
          double tr_err = 0.0;
          for (size_t iy = 0; iy < tr.cov_y; ++iy) {
            for (size_t ix = 0; ix < tr.cov_x; ++ix) {
              tr_err +=
                  blk_err_cur[(bgr.y0() + tr.by + iy) * xsize_blocks +
                              (bgr.x0() + tr.bx + ix)];
            }
          }
          if (tr_err <= 1.25 * mean_blk_err * static_cast<double>(cov)) {
            continue;
          }
          const float tr_bits = tr.slots[tr.cur_slot].bits;
          const double dbits_est =
              std::max(1.5, 0.14 * static_cast<double>(tr_bits));
          const double drate = rate_term(cur_bits + cur_sh_bits + dbits_est) -
                               rate_term(cur_bits + cur_sh_bits);
          const double proxy_j = 0.60 * tr_err - drate;
          if (proxy_j > 0.0002) {
            tr_rp_cands.push_back({proxy_j, static_cast<uint32_t>(i), g});
          }
        }
      }
      if (!tr_rp_cands.empty()) {
        std::sort(tr_rp_cands.begin(), tr_rp_cands.end(),
                  [](const CandTransform& a, const CandTransform& b) {
                    if (a.score != b.score) return a.score > b.score;
                    return (a.group != b.group) ? (a.group < b.group)
                                                : (a.tr_idx < b.tr_idx);
                  });
        std::vector<std::vector<Stage2QueueEntry>> group_queues(num_groups);
        for (const CandTransform& c : tr_rp_cands) {
          if (group_queues[c.group].size() < 18) {
            group_queues[c.group].push_back(
                {c.tr_idx, 0, 0, Stage2MoveKind::kTrRdoqPlus});
          }
        }
        JXL_RETURN_IF_ERROR(
            run_stage2_queue(max_tr_rp, group_queues, &tr_rp_moves));
      }
    }

    // 3) Targeted +1 quant_ac on top salient edge transforms inside mixed tiles
    // (pricing exact out_qf transition bits + localized epf_sigma shrinkage).
    size_t tr_qp_moves = 0;
    const size_t max_tr_qp = std::min<size_t>(
        280, std::max<size_t>(4, num_tiles));
    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
    sum_blk_err = 0.0;
    for (size_t b = 0; b < num_blocks; ++b) sum_blk_err += blk_err_cur[b];
    const double mean_blk_err2 =
        sum_blk_err / static_cast<double>(std::max<size_t>(1, num_blocks));
    if (mean_blk_err2 > 1e-12) {
      std::vector<CandTransform> tr_qp_cands;
      for (size_t g = 0; g < num_groups; ++g) {
        const Rect bgr = shared.frame_dim.BlockGroupRect(g);
        const auto& trs = group_caches[g].transforms;
        for (size_t i = 0; i < trs.size(); ++i) {
          const AcTransformCache& tr = trs[i];
          if (tile_tr_indices[tr.tile_idx].size() < 2) continue;
          if (tr.cur_quant_ac >= 255 || tr.cur_rdoq_level < 0 ||
              tile_step[tr.tile_idx] < 0 || tile_step[tr.tile_idx] >= 2) {
            continue;
          }
          const size_t cov = static_cast<size_t>(tr.cov_x) * tr.cov_y;
          if (cov > 4) continue;
          double tr_err = 0.0;
          for (size_t iy = 0; iy < tr.cov_y; ++iy) {
            for (size_t ix = 0; ix < tr.cov_x; ++ix) {
              tr_err +=
                  blk_err_cur[(bgr.y0() + tr.by + iy) * xsize_blocks +
                              (bgr.x0() + tr.bx + ix)];
            }
          }
          if (tr_err <= 1.65 * mean_blk_err2 * static_cast<double>(cov)) {
            continue;
          }
          const float tr_bits = tr.slots[tr.cur_slot].bits;
          const double dbits_est =
              3.5 + 0.22 * static_cast<double>(std::max(2.0f, tr_bits));
          const double drate = rate_term(cur_bits + cur_sh_bits + dbits_est) -
                               rate_term(cur_bits + cur_sh_bits);
          const double proxy_j = tr_err - drate;
          if (proxy_j > 0.0004) {
            tr_qp_cands.push_back({proxy_j, static_cast<uint32_t>(i), g});
          }
        }
      }
      if (!tr_qp_cands.empty()) {
        std::sort(tr_qp_cands.begin(), tr_qp_cands.end(),
                  [](const CandTransform& a, const CandTransform& b) {
                    if (a.score != b.score) return a.score > b.score;
                    return (a.group != b.group) ? (a.group < b.group)
                                                : (a.tr_idx < b.tr_idx);
                  });
        std::vector<std::vector<Stage2QueueEntry>> group_queues(num_groups);
        for (const CandTransform& c : tr_qp_cands) {
          if (group_queues[c.group].size() < 12) {
            group_queues[c.group].push_back(
                {c.tr_idx, 0, 0, Stage2MoveKind::kTrQfPlus});
          }
        }
        JXL_RETURN_IF_ERROR(
            run_stage2_queue(max_tr_qp, group_queues, &tr_qp_moves));
      }
    }
  }

  // Pass 2E: Closed-loop 64x64-tile CfL (ytox_map, ytob_map) refinement.
  // Reclaims ACMetadata Modular CfL bits on noisy/masked tiles via predictor
  // snapping under unanimous 4-metric agreement, then refines ytox/ytob on
  // top-error tiles through post-RDOQ Y AC and active EPF.
  size_t cfl_moves = 0;
  const size_t cfl_tiles_x = DivCeil(xsize_blocks, kColorTileDimInBlocks);
  const size_t cfl_tiles_y = DivCeil(ysize_blocks, kColorTileDimInBlocks);
  const size_t num_cfl_tiles = cfl_tiles_x * cfl_tiles_y;
  if (num_cfl_tiles > 0 && shared.cmap.ytox_map.xsize() >= cfl_tiles_x &&
      shared.cmap.ytox_map.ysize() >= cfl_tiles_y &&
      shared.cmap.ytob_map.xsize() >= cfl_tiles_x &&
      shared.cmap.ytob_map.ysize() >= cfl_tiles_y) {
    JXL_ASSIGN_OR_RETURN(
        st.rdo_ytox_map,
        ImageSB::Create(memory_manager, shared.cmap.ytox_map.xsize(),
                        shared.cmap.ytox_map.ysize()));
    JXL_ASSIGN_OR_RETURN(
        st.rdo_ytob_map,
        ImageSB::Create(memory_manager, shared.cmap.ytob_map.xsize(),
                        shared.cmap.ytob_map.ysize()));
    JXL_RETURN_IF_ERROR(CopyImageTo(shared.cmap.ytox_map, &st.rdo_ytox_map));
    JXL_RETURN_IF_ERROR(CopyImageTo(shared.cmap.ytob_map, &st.rdo_ytob_map));

    st.cfl_tile_group.assign(num_cfl_tiles, num_groups);
    st.cfl_tile_trs.assign(num_cfl_tiles, {});
    st.cfl_moved[0].assign(num_cfl_tiles, 0);
    st.cfl_moved[1].assign(num_cfl_tiles, 0);
    for (size_t g = 0; g < num_groups; ++g) {
      const auto& trs = group_caches[g].transforms;
      for (size_t i = 0; i < trs.size(); ++i) {
        const uint32_t ct = trs[i].cfl_tile_idx;
        if (ct < num_cfl_tiles) {
          st.cfl_tile_group[ct] = g;
          st.cfl_tile_trs[ct].push_back(i);
        }
      }
    }

    const double init_cfl_bits =
        EstimateCflMapBits(st.rdo_ytox_map, cfl_tiles_x, cfl_tiles_y) +
        EstimateCflMapBits(st.rdo_ytob_map, cfl_tiles_x, cfl_tiles_y);
    double cur_cfl_bits = init_cfl_bits;

    struct CflQueueEntry {
      size_t ct;
      int8_t cand_q;
    };
    auto run_cfl_batch_queue =
        [&](int chan, bool is_reclaim, size_t max_moves_cap,
            const std::vector<std::vector<CflQueueEntry>>& queues,
            size_t* pass_moves) -> Status {
      ImageSB* cmap_ptr = (chan == 0) ? &st.rdo_ytox_map : &st.rdo_ytob_map;
      std::vector<size_t> group_q_pos(num_groups, 0);
      while (*pass_moves < max_moves_cap) {
        struct CflBatchItem {
          size_t group;
          size_t ct;
          int8_t old_q;
          int8_t cand_q;
          float cand_bits;
          Rect dirty_px;
        };
        std::vector<CflBatchItem> batch;
        for (size_t g = 0; g < num_groups; ++g) {
          while (group_q_pos[g] < queues[g].size()) {
            const CflQueueEntry qe = queues[g][group_q_pos[g]++];
            const size_t ct = qe.ct;
            if (!is_reclaim && st.cfl_moved[chan][ct] != 0) continue;
            const size_t ct_x = ct % cfl_tiles_x;
            const size_t ct_y = ct / cfl_tiles_x;
            const int8_t cur_q = cmap_ptr->ConstRow(ct_y)[ct_x];
            int8_t target_q = qe.cand_q;
            if (is_reclaim) {
              const int32_t gp = CflTileGradPred(*cmap_ptr, ct_x, ct_y);
              target_q = (cur_q != static_cast<int8_t>(gp))
                             ? static_cast<int8_t>(gp)
                             : static_cast<int8_t>(0);
              if (target_q == cur_q ||
                  std::abs(static_cast<int>(cur_q) -
                           static_cast<int>(target_q)) > 6) {
                continue;
              }
            } else if (target_q == cur_q) {
              continue;
            }
            CflBatchItem item{g,    ct,   cur_q,           target_q,
                              0.0f, Rect(0, 0, 0, 0)};
            for (size_t idx : st.cfl_tile_trs[ct]) {
              if (chan == 0) {
                group_caches[g].transforms[idx].cur_ytox_q = target_q;
              } else {
                group_caches[g].transforms[idx].cur_ytob_q = target_q;
              }
            }
            batch.push_back(item);
            break;
          }
        }
        if (batch.empty()) break;

        std::atomic<bool> batch_ok{true};
        JXL_RETURN_IF_ERROR(RunOnPool(
            pool, 0, static_cast<uint32_t>(batch.size()), ThreadPool::NoInit,
            [&](uint32_t bi, size_t /*thread*/) -> Status {
              CflBatchItem& item = batch[bi];
              AcGroupBlockCache& cache = group_caches[item.group];
              float ac_bits = 0.0f;
              if (!HWY_DYNAMIC_DISPATCH(EvalAcGroupCacheImpl)(
                      enc_state, &cache, /*qf_step_override=*/0,
                      /*use_rdoq=*/true, /*commit_cur_slot=*/false,
                      /*only_write_changed=*/true, &batch_pre_gab, nullptr,
                      &ac_bits, &item.dirty_px)) {
                batch_ok.store(false);
                return true;
              }
              const float qf_bits = EstimateGroupQfBits(cache, tcm, 0, nullptr);
              item.cand_bits = ac_bits + qf_bits;
              return true;
            },
            "EvalCflCandBatch"));
        JXL_ENSURE(batch_ok.load());

        for (CflBatchItem& item : batch) {
          const size_t g = item.group;
          const size_t ct = item.ct;
          const size_t ct_x = ct % cfl_tiles_x;
          const size_t ct_y = ct / cfl_tiles_x;
          cmap_ptr->Row(ct_y)[ct_x] = item.cand_q;
          const double new_cfl_bits =
              EstimateCflMapBits(st.rdo_ytox_map, cfl_tiles_x, cfl_tiles_y) +
              EstimateCflMapBits(st.rdo_ytob_map, cfl_tiles_x, cfl_tiles_y);
          const double new_bits =
              cur_bits + static_cast<double>(item.cand_bits - cur_group_bits[g]);
          const double cur_tot =
              std::max(1.0, cur_bits + cur_sh_bits + (cur_cfl_bits - init_cfl_bits));
          const double new_tot =
              std::max(1.0, new_bits + cur_sh_bits + (new_cfl_bits - init_cfl_bits));
          const double delta_rate = rate_term(new_tot) - rate_term(cur_tot);

          const bool skip_early =
              (*pass_moves >= max_moves_cap) ||
              (is_reclaim
                   ? (delta_rate >= -0.001)
                   : (item.dirty_px.xsize() == 0 && delta_rate >= -1e-4));
          if (skip_early) {
            cmap_ptr->Row(ct_y)[ct_x] = item.old_q;
            if (item.dirty_px.xsize() != 0 && item.dirty_px.ysize() != 0) {
              for (size_t c = 0; c < 3; ++c) {
                for (size_t y = 0; y < item.dirty_px.ysize(); ++y) {
                  memcpy(batch_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                             item.dirty_px.x0(),
                         cur_pre_gab.ConstPlaneRow(c, item.dirty_px.y0() + y) +
                             item.dirty_px.x0(),
                         item.dirty_px.xsize() * sizeof(float));
                }
              }
            }
            for (size_t idx : st.cfl_tile_trs[ct]) {
              if (chan == 0) {
                group_caches[g].transforms[idx].cur_ytox_q = item.old_q;
              } else {
                group_caches[g].transforms[idx].cur_ytob_q = item.old_q;
              }
            }
            continue;
          }

          Rect dirty_lin(0, 0, 0, 0);
          if (item.dirty_px.xsize() != 0 && item.dirty_px.ysize() != 0) {
            for (size_t c = 0; c < 3; ++c) {
              for (size_t y = 0; y < item.dirty_px.ysize(); ++y) {
                memcpy(cand_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       batch_pre_gab.ConstPlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       item.dirty_px.xsize() * sizeof(float));
              }
            }
            const size_t lx0 = (item.dirty_px.x0() > lf_pad)
                                   ? (item.dirty_px.x0() - lf_pad)
                                   : 0;
            const size_t ly0 = (item.dirty_px.y0() > lf_pad)
                                   ? (item.dirty_px.y0() - lf_pad)
                                   : 0;
            const size_t lx1 = std::min(
                xsize, item.dirty_px.x0() + item.dirty_px.xsize() + lf_pad);
            const size_t ly1 = std::min(
                ysize, item.dirty_px.y0() + item.dirty_px.ysize() + lf_pad);
            if (lx0 < lx1 && ly0 < ly1) {
              dirty_lin = Rect(lx0, ly0, lx1 - lx0, ly1 - ly0);
            }
          }

          if (dirty_lin.xsize() != 0 && dirty_lin.ysize() != 0) {
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
            JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
                memory_manager, lf, shared, cand_pre_gab, feat_ptr, epf_sigma,
                epf_ws, dirty_lin, &cur_lin));
          }

          std::array<double, 4> cand_d = cur_d;
          if (dirty_lin.xsize() != 0) {
            JXL_RETURN_IF_ERROR(
                update_4m_checkpointed(cur_lin, dirty_lin, &cand_d));
          }

          double norm_sum = 0.0;
          bool pareto_ok = true;
          for (size_t m = 0; m < 4; ++m) {
            if (weight_m[m] <= 0.0) continue;
            const double nm = (cur_d[m] - cand_d[m]) / slope_m[m];
            norm_sum += weight_m[m] * nm;
            if (is_reclaim) {
              if (nm - delta_rate < 0.0) pareto_ok = false;
            } else {
              if (nm < -0.20792402874161239 *
                           std::max(0.015, std::abs(delta_rate))) {
                pareto_ok = false;
              }
            }
          }
          const double j_gain = norm_sum - delta_rate;
          const double min_gain = is_reclaim ? 0.002 : 0.003;

          if (j_gain > min_gain && pareto_ok) {
            for (AcTransformCache& tr : group_caches[g].transforms) {
              tr.cur_slot = tr.cand_slot;
            }
            if (item.dirty_px.xsize() != 0 && item.dirty_px.ysize() != 0) {
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
            cur_cfl_bits = new_cfl_bits;
            cur_d = cand_d;
            st.cfl_moved[chan][ct] = 1;
            ++(*pass_moves);
            ++cfl_moves;
          } else {
            cmap_ptr->Row(ct_y)[ct_x] = item.old_q;
            if (dirty_lin.xsize() != 0) {
              wd->Rollback(cp_wd);
              ssim2->Rollback(cp_ss2);
              ba3->Rollback(cp_ba3);
              cvvdp->Rollback(cp_cv);
              const size_t lw = dirty_lin.xsize();
              const size_t lh = dirty_lin.ysize();
              for (size_t c = 0; c < 3; ++c) {
                for (size_t y = 0; y < lh; ++y) {
                  memcpy(
                      cur_lin.PlaneRow(c, dirty_lin.y0() + y) + dirty_lin.x0(),
                      saved_lin.data() + (c * lh + y) * lw, lw * sizeof(float));
                }
              }
            }
            if (item.dirty_px.xsize() != 0 && item.dirty_px.ysize() != 0) {
              for (size_t c = 0; c < 3; ++c) {
                for (size_t y = 0; y < item.dirty_px.ysize(); ++y) {
                  const float* src_r =
                      cur_pre_gab.ConstPlaneRow(c, item.dirty_px.y0() + y) +
                      item.dirty_px.x0();
                  memcpy(cand_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                             item.dirty_px.x0(),
                         src_r, item.dirty_px.xsize() * sizeof(float));
                  memcpy(batch_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                             item.dirty_px.x0(),
                         src_r, item.dirty_px.xsize() * sizeof(float));
                }
              }
            }
            for (size_t idx : st.cfl_tile_trs[ct]) {
              if (chan == 0) {
                group_caches[g].transforms[idx].cur_ytox_q = item.old_q;
              } else {
                group_caches[g].transforms[idx].cur_ytob_q = item.old_q;
              }
            }
          }
        }
      }
      return true;
    };

    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/false);
    st.cfl_tile_err.assign(num_cfl_tiles, 0.0);
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const size_t ct_y = by / kColorTileDimInBlocks;
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        const size_t ct_x = bx / kColorTileDimInBlocks;
        st.cfl_tile_err[ct_y * cfl_tiles_x + ct_x] +=
            blk_err_cur[by * xsize_blocks + bx];
      }
    }

    // Step 2E.1: Snap noisy/masked CfL multipliers to predictor/zero when all
    // 4 metrics agree RD improves.
    for (int chan = 0; chan < 2; ++chan) {
      const ImageSB& cmap_ref = (chan == 0) ? st.rdo_ytox_map : st.rdo_ytob_map;
      std::vector<CandTile> reclaim_cands;
      reclaim_cands.reserve(num_cfl_tiles);
      for (size_t ct = 0; ct < num_cfl_tiles; ++ct) {
        if (st.cfl_tile_trs[ct].empty() || st.cfl_tile_group[ct] >= num_groups) {
          continue;
        }
        const size_t ct_x = ct % cfl_tiles_x;
        const size_t ct_y = ct / cfl_tiles_x;
        const int32_t cur_q = static_cast<int32_t>(cmap_ref.ConstRow(ct_y)[ct_x]);
        const int32_t gp = CflTileGradPred(cmap_ref, ct_x, ct_y);
        const int32_t tgt = (cur_q != gp) ? gp : 0;
        const int32_t diff = std::abs(cur_q - tgt);
        if (diff == 0 || diff > 6) continue;
        const double score = 1.0 / (0.01 + st.cfl_tile_err[ct]);
        reclaim_cands.push_back({score, ct, st.cfl_tile_group[ct]});
      }
      if (!reclaim_cands.empty()) {
        std::sort(reclaim_cands.begin(), reclaim_cands.end(),
                  [](const CandTile& a, const CandTile& b) {
                    return (a.score != b.score) ? (a.score > b.score)
                                                : (a.tile < b.tile);
                  });
        std::vector<std::vector<CflQueueEntry>> queues(num_groups);
        for (const CandTile& c : reclaim_cands) {
          if (queues[c.group].size() < 6) {
            queues[c.group].push_back({c.tile, 0});
          }
        }
        size_t reclaim_moves = 0;
        JXL_RETURN_IF_ERROR(run_cfl_batch_queue(
            chan, /*is_reclaim=*/true, num_cfl_tiles, queues, &reclaim_moves));
      }
    }

    // Step 2E.2: Refine ytox (±2) and ytob (±3) on top salient 64x64 tiles.
    std::vector<CandTile> salient_cfl;
    salient_cfl.reserve(num_cfl_tiles);
    for (size_t ct = 0; ct < num_cfl_tiles; ++ct) {
      if (st.cfl_tile_trs[ct].empty() || st.cfl_tile_group[ct] >= num_groups) {
        continue;
      }
      if (st.cfl_tile_err[ct] > 1e-12) {
        salient_cfl.push_back({st.cfl_tile_err[ct], ct, st.cfl_tile_group[ct]});
      }
    }
    if (!salient_cfl.empty()) {
      std::sort(salient_cfl.begin(), salient_cfl.end(),
                [](const CandTile& a, const CandTile& b) {
                  return (a.score != b.score) ? (a.score > b.score)
                                              : (a.tile < b.tile);
                });
      const size_t max_cfl_ref = std::min<size_t>(
          64, std::max<size_t>(1, static_cast<size_t>(0.35f * num_cfl_tiles)));
      for (int chan = 0; chan < 2; ++chan) {
        const ImageSB& cmap_ref =
            (chan == 0) ? st.rdo_ytox_map : st.rdo_ytob_map;
        const int32_t step_mag = (chan == 0) ? 2 : 3;
        for (int dir : {-1, 1}) {
          std::vector<std::vector<CflQueueEntry>> queues(num_groups);
          for (const CandTile& c : salient_cfl) {
            if (queues[c.group].size() >= 5) continue;
            if (st.cfl_moved[chan][c.tile] != 0) continue;
            const size_t ct_x = c.tile % cfl_tiles_x;
            const size_t ct_y = c.tile / cfl_tiles_x;
            const int32_t cur_q =
                static_cast<int32_t>(cmap_ref.ConstRow(ct_y)[ct_x]);
            const int8_t cand_q = static_cast<int8_t>(
                Clamp1<int32_t>(cur_q + dir * step_mag, -128, 127));
            if (cand_q == static_cast<int8_t>(cur_q)) continue;
            queues[c.group].push_back({c.tile, cand_q});
          }
          size_t ref_moves = 0;
          JXL_RETURN_IF_ERROR(run_cfl_batch_queue(
              chan, /*is_reclaim=*/false, max_cfl_ref, queues, &ref_moves));
        }
      }
    }

    if (cfl_moves > 0) {
      cur_bits = std::max(1.0, cur_bits + (cur_cfl_bits - init_cfl_bits));
    }
  }

  // Sync rdo_quant_field with winning Stage 2 cur_quant_ac values before Pass 2F & Stage 3.
  if (total_moves > 0) {
    for (size_t g = 0; g < num_groups; ++g) {
      const Rect bgr = shared.frame_dim.BlockGroupRect(g);
      for (const AcTransformCache& tr : group_caches[g].transforms) {
        if (tr.cur_quant_ac == tr.base_quant_ac) continue;
        for (size_t iy = 0; iy < tr.cov_y; ++iy) {
          int32_t* JXL_RESTRICT qrow = bgr.Row(&rdo_quant_field, tr.by + iy);
          for (size_t ix = 0; ix < tr.cov_x; ++ix) {
            qrow[tr.bx + ix] = tr.cur_quant_ac;
          }
        }
      }
    }
    epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale, rdo_quant_field,
                          rdo_sharpness, full_blk_rect);
  }

  // Pass 2F: Full 256x256-group (32x32-block) closed-loop ac_strategy DP
  // (Guillotine + 5-rect CW/CCW Pinwheels + 8x8 Non-Guillotine B&B over all
  // legal transforms <= 64x64 at arbitrary (x0, y0) shifts), followed by
  // cold-start QuantTable multi-component joint verification, per-component
  // 4-metric checkpointed verification, and salient 2x2 / 1x1 rescue.
  size_t acs_moves = 0;
  if (enc_state->initial_quant_masking1x1.xsize() >= xsize &&
      enc_state->initial_quant_masking1x1.ysize() >= ysize) {
    size_t quant_table_counts[kNumQuantTables] = {};
    for (size_t g = 0; g < num_groups; ++g) {
      for (const AcTransformCache& tr : group_caches[g].transforms) {
        const size_t t_idx =
            static_cast<size_t>(kAcStrategyToQuantTableMap[tr.acs_raw]);
        if (t_idx < kNumQuantTables) quant_table_counts[t_idx]++;
      }
    }
    const auto& qm_encs = shared.matrices.encodings();
    auto calc_table_tax_bits =
        [&](const size_t counts[kNumQuantTables]) -> double {
      double tax = 0.0;
      for (size_t t = 1; t < kNumQuantTables && t < qm_encs.size(); ++t) {
        if (counts[t] == 0) continue;
        if (qm_encs[t].mode == QuantEncoding::kQuantModeDCT) {
          tax += 3.0 + 48.0 * static_cast<double>(
                                  qm_encs[t].dct_params.num_distance_bands);
        }
      }
      return tax;
    };
    const double init_table_tax_bits = calc_table_tax_bits(quant_table_counts);
    double cur_table_tax_bits = init_table_tax_bits;
    double cum_acs_sym_delta_bits = 0.0;

    float gab_w_f[3][3] = {};
    if (lf.gab) {
      const float rw1[3] = {lf.gab_x_weight1, lf.gab_y_weight1,
                            lf.gab_b_weight1};
      const float rw2[3] = {lf.gab_x_weight2, lf.gab_y_weight2,
                            lf.gab_b_weight2};
      for (size_t c = 0; c < 3; ++c) {
        const float mul = 1.0f / (1.0f + 4.0f * (rw1[c] + rw2[c]));
        gab_w_f[c][0] = mul;
        gab_w_f[c][1] = rw1[c] * mul;
        gab_w_f[c][2] = rw2[c] * mul;
      }
    }

    auto apply_local_gab_box = [&](const Image3F& patch_src, size_t fx0,
                                   size_t fy0, size_t lw, size_t lh,
                                   Image3F* dst_xyb) {
      ApplyLocalGabBox(lf.gab, gab_w_f, feat_ptr, xsize, ysize, cur_pre_gab,
                       patch_src, fx0, fy0, lw, lh, dst_xyb);
    };

    auto swap_box_transforms = [](std::vector<AcTransformCache>* trs,
                                  size_t bx0, size_t by0, size_t bw, size_t bh,
                                  std::vector<AcTransformCache>* repl,
                                  std::vector<AcTransformCache>* scratch) {
      std::vector<AcTransformCache> extracted;
      extracted.reserve(16);
      scratch->clear();
      scratch->reserve(trs->size() + repl->size());
      size_t ri = 0;
      for (AcTransformCache& t : *trs) {
        if (t.bx >= bx0 && t.bx < bx0 + bw && t.by >= by0 && t.by < by0 + bh) {
          extracted.push_back(std::move(t));
          continue;
        }
        while (ri < repl->size() &&
               ((*repl)[ri].by < t.by ||
                ((*repl)[ri].by == t.by && (*repl)[ri].bx < t.bx))) {
          scratch->push_back(std::move((*repl)[ri++]));
        }
        scratch->push_back(std::move(t));
      }
      while (ri < repl->size()) {
        scratch->push_back(std::move((*repl)[ri++]));
      }
      trs->swap(*scratch);
      repl->swap(extracted);
    };

    struct AcsLayoutDef {
      uint8_t count = 0;
      uint8_t dx[4] = {};
      uint8_t dy[4] = {};
      AcStrategyType types[4] = {};
    };

    struct AcsBoxQueueEntry {
      uint8_t bx0 = 0;
      uint8_t by0 = 0;
      uint8_t bw = 1;
      uint8_t bh = 1;
      bool is_merge = false;
      std::vector<AcsSubSpec> pre_specs;
      std::vector<AcsLayoutDef> layouts;
    };

    // Seed fine_pre_gab with current post-Gaborish XYB so 1-px neighbor border
    // in ComputeLocalBlockEpfError is valid everywhere across all groups.
    for (size_t g = 0; g < num_groups; ++g) {
      const AcGroupBlockCache& cache = group_caches[g];
      if (cache.pw_px > 0 && cache.ph_px > 0) {
        apply_local_gab_box(cur_pre_gab, cache.px0, cache.py0, cache.pw_px,
                            cache.ph_px, &fine_pre_gab);
      }
    }

    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/false);
    double sum_blk_err_2f = 0.0;
    for (double v : blk_err_cur) sum_blk_err_2f += v;
    double mean_blk_err_2f =
        sum_blk_err_2f / static_cast<double>(std::max<size_t>(1, num_blocks));

    std::vector<GroupAcsDpWorkspace> dp_ws;
    std::vector<std::vector<GroupAcsDpComponent>> group_dp_comps(num_groups);

    // Stage I: Solve full 32x32 closed-loop group ac_strategy DP in 4-color
    // checkerboard waves so concurrent groups are separated by >= 256 pixels.
    const size_t groups_x = std::max<size_t>(1, shared.frame_dim.xsize_groups);
    for (size_t wave = 0; wave < 4; ++wave) {
      std::vector<size_t> wave_groups;
      wave_groups.reserve((num_groups + 3) / 4);
      for (size_t g = 0; g < num_groups; ++g) {
        const size_t gx = g % groups_x;
        const size_t gy = g / groups_x;
        if (((gx & 1u) + 2u * (gy & 1u)) == wave) {
          wave_groups.push_back(g);
        }
      }
      if (wave_groups.empty()) continue;
      JXL_RETURN_IF_ERROR(RunOnPool(
          pool, 0, static_cast<uint32_t>(wave_groups.size()),
          [&](size_t num_threads) -> Status {
            const size_t want = std::max<size_t>(1, num_threads);
            if (dp_ws.size() < want) dp_ws.resize(want);
            return true;
          },
          [&](uint32_t wi, size_t thread) -> Status {
            const size_t g = wave_groups[wi];
            return SolveClosedLoopGroupAcsDP(
                enc_state, opsin, rect, tiles_x, kTileBlocks, group_caches[g],
                lf.gab, gab_w_f, feat_ptr, cur_pre_gab, &plain_pre_gab,
                &fine_pre_gab, st.rdo_ytox_map, st.rdo_ytob_map,
                blk_err_cur.data(), mean_blk_err_2f, quant_table_counts,
                &dp_ws[thread], &group_dp_comps[g]);
          },
          "SolveClosedLoopGroupAcsDP"));
    }

    // Stage II.A: Cold-start QuantTable multi-component joint verification.
    // When >= 2 DP components across the frame want an unused DCT QuantTable
    // (paying a single +3 + 48*bands header tax once), evaluate them jointly.
    if (dp_ws.empty()) dp_ws.resize(1);
    for (size_t qt = 1; qt < kNumQuantTables && qt < qm_encs.size(); ++qt) {
      if (quant_table_counts[qt] != 0 ||
          qm_encs[qt].mode != QuantEncoding::kQuantModeDCT) {
        continue;
      }
      std::vector<GroupAcsDpComponent*> joint_comps;
      for (size_t g = 0; g < num_groups; ++g) {
        const AcGroupBlockCache& cache = group_caches[g];
        for (GroupAcsDpComponent& comp : group_dp_comps[g]) {
          if (comp.committed ||
              (comp.new_tables_mask & (1u << static_cast<uint32_t>(qt))) == 0) {
            continue;
          }
          if (comp.bx0 + comp.bw > cache.gw_blocks ||
              comp.by0 + comp.bh > cache.gh_blocks) {
            continue;
          }
          size_t sum_cov = 0;
          for (const AcTransformCache& tr : cache.transforms) {
            if (tr.bx >= comp.bx0 && tr.bx < comp.bx0 + comp.bw &&
                tr.by >= comp.by0 && tr.by < comp.by0 + comp.bh &&
                tr.bx + tr.cov_x <= comp.bx0 + comp.bw &&
                tr.by + tr.cov_y <= comp.by0 + comp.bh) {
              sum_cov += static_cast<size_t>(tr.cov_x) * tr.cov_y;
            }
          }
          if (sum_cov == static_cast<size_t>(comp.bw) * comp.bh) {
            joint_comps.push_back(&comp);
          }
        }
      }
      if (joint_comps.size() < 2) continue;

      struct JointCompUndo {
        GroupAcsDpComponent* comp = nullptr;
        std::vector<AcTransformCache> saved_trs;
        std::vector<float> saved_base_patch[3];
        double old_acs_sym_bits = 0.0;
        double new_acs_sym_bits = 0.0;
      };
      std::vector<JointCompUndo> undos;
      undos.reserve(joint_comps.size());
      std::vector<size_t> aff_groups;
      bool joint_eval_ok = true;

      for (GroupAcsDpComponent* comp : joint_comps) {
        AcGroupBlockCache& cache = group_caches[comp->group];
        const size_t bx0 = comp->bx0;
        const size_t by0 = comp->by0;
        const size_t bw = comp->bw;
        const size_t bh = comp->bh;
        const size_t box_w_px =
            std::min(cache.pw_px - bx0 * kBlockDim, bw * kBlockDim);
        const size_t box_h_px =
            std::min(cache.ph_px - by0 * kBlockDim, bh * kBlockDim);

        double old_acs_bits = 0.0;
        for (const AcTransformCache& tr : cache.transforms) {
          if (tr.bx >= bx0 && tr.bx < bx0 + bw && tr.by >= by0 &&
              tr.by < by0 + bh && tr.acs_raw < 27) {
            old_acs_bits += tcm.acs_strategy_bits[tr.acs_raw];
          }
        }

        std::vector<AcTransformCache> new_trs;
        std::vector<float> new_base_patch[3];
        if (!HWY_DYNAMIC_DISPATCH(EvalAcsBoxCandidateImpl)(
                enc_state, opsin, rect, tiles_x, kTileBlocks, cache, bx0, by0,
                bw, bh, comp->specs.size(), comp->specs.data(), &new_trs,
                new_base_patch, nullptr, nullptr, &dp_ws[0].eval_ws)) {
          joint_eval_ok = false;
          break;
        }

        JointCompUndo u;
        u.comp = comp;
        u.old_acs_sym_bits = old_acs_bits;
        for (const AcTransformCache& tr : new_trs) {
          if (tr.acs_raw < 27) {
            u.new_acs_sym_bits += tcm.acs_strategy_bits[tr.acs_raw];
          }
        }
        for (size_t c = 0; c < 3; ++c) {
          u.saved_base_patch[c].resize(box_w_px * box_h_px);
          for (size_t iy = 0; iy < box_h_px; ++iy) {
            float* JXL_RESTRICT bp =
                cache.base_pre_gab[c].data() +
                (by0 * kBlockDim + iy) * cache.pw_px + bx0 * kBlockDim;
            memcpy(u.saved_base_patch[c].data() + iy * box_w_px, bp,
                   box_w_px * sizeof(float));
            memcpy(bp, new_base_patch[c].data() + iy * box_w_px,
                   box_w_px * sizeof(float));
          }
        }
        swap_box_transforms(&cache.transforms, bx0, by0, bw, bh, &new_trs,
                            &st.acs_scratch_trs);
        u.saved_trs = std::move(new_trs);

        const Rect box_blk_rect(cache.gx0 + bx0, cache.gy0 + by0, bw, bh);
        for (const AcsSubSpec& sp : comp->specs) {
          const AcStrategy acs = AcStrategy::FromRawStrategy(sp.acs_raw);
          JXL_RETURN_IF_ERROR(shared.ac_strategy.Set(
              cache.gx0 + sp.bx, cache.gy0 + sp.by, acs.Strategy()));
          for (size_t iy = 0; iy < acs.covered_blocks_y(); ++iy) {
            int32_t* JXL_RESTRICT qrow =
                rdo_quant_field.Row(cache.gy0 + sp.by + iy);
            for (size_t ix = 0; ix < acs.covered_blocks_x(); ++ix) {
              qrow[cache.gx0 + sp.bx + ix] = sp.quant_ac;
            }
          }
        }
        epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale,
                              rdo_quant_field, rdo_sharpness, box_blk_rect);
        if (std::find(aff_groups.begin(), aff_groups.end(), comp->group) ==
            aff_groups.end()) {
          aff_groups.push_back(comp->group);
        }
        undos.push_back(std::move(u));
      }

      std::vector<Rect> aff_dirty(aff_groups.size());
      std::vector<float> aff_cand_bits(aff_groups.size(), 0.0f);
      if (joint_eval_ok) {
        for (size_t gi = 0; gi < aff_groups.size(); ++gi) {
          const size_t g = aff_groups[gi];
          AcGroupBlockCache& cache = group_caches[g];
          float ac_bits = 0.0f;
          if (!HWY_DYNAMIC_DISPATCH(EvalAcGroupCacheImpl)(
                  enc_state, &cache, /*qf_step_override=*/0, /*use_rdoq=*/true,
                  /*commit_cur_slot=*/false, /*only_write_changed=*/true,
                  &batch_pre_gab, nullptr, &ac_bits, &aff_dirty[gi])) {
            joint_eval_ok = false;
            break;
          }
          aff_cand_bits[gi] =
              ac_bits + EstimateGroupQfBits(cache, tcm, 0, nullptr);
          if (aff_dirty[gi].xsize() != 0 && aff_dirty[gi].ysize() != 0) {
            for (size_t c = 0; c < 3; ++c) {
              for (size_t y = 0; y < aff_dirty[gi].ysize(); ++y) {
                memcpy(cand_pre_gab.PlaneRow(c, aff_dirty[gi].y0() + y) +
                           aff_dirty[gi].x0(),
                       batch_pre_gab.ConstPlaneRow(c, aff_dirty[gi].y0() + y) +
                           aff_dirty[gi].x0(),
                       aff_dirty[gi].xsize() * sizeof(float));
              }
            }
          }
        }
      }
      JXL_ENSURE(joint_eval_ok);

      size_t cand_table_counts[kNumQuantTables];
      memcpy(cand_table_counts, quant_table_counts, sizeof(quant_table_counts));
      double delta_acs_sym = 0.0;
      size_t px_x0 = xsize;
      size_t px_y0 = ysize;
      size_t px_x1 = 0;
      size_t px_y1 = 0;
      for (const JointCompUndo& u : undos) {
        delta_acs_sym += (u.new_acs_sym_bits - u.old_acs_sym_bits);
        for (const AcTransformCache& tr : u.saved_trs) {
          const size_t ti =
              static_cast<size_t>(kAcStrategyToQuantTableMap[tr.acs_raw]);
          if (ti < kNumQuantTables && cand_table_counts[ti] > 0) {
            cand_table_counts[ti]--;
          }
        }
        for (const AcsSubSpec& sp : u.comp->specs) {
          const size_t ti =
              static_cast<size_t>(kAcStrategyToQuantTableMap[sp.acs_raw]);
          if (ti < kNumQuantTables) cand_table_counts[ti]++;
        }
        const AcGroupBlockCache& cache = group_caches[u.comp->group];
        const size_t bx0_px = cache.px0 + u.comp->bx0 * kBlockDim;
        const size_t by0_px = cache.py0 + u.comp->by0 * kBlockDim;
        const size_t bw_px = std::min(cache.pw_px - u.comp->bx0 * kBlockDim,
                                      static_cast<size_t>(u.comp->bw) * kBlockDim);
        const size_t bh_px = std::min(cache.ph_px - u.comp->by0 * kBlockDim,
                                      static_cast<size_t>(u.comp->bh) * kBlockDim);
        px_x0 = std::min(px_x0, bx0_px);
        px_y0 = std::min(px_y0, by0_px);
        px_x1 = std::max(px_x1, bx0_px + bw_px);
        px_y1 = std::max(px_y1, by0_px + bh_px);
      }
      double new_ac_qf_bits = cur_bits;
      for (size_t gi = 0; gi < aff_groups.size(); ++gi) {
        const size_t g = aff_groups[gi];
        new_ac_qf_bits +=
            static_cast<double>(aff_cand_bits[gi] - cur_group_bits[g]);
        if (aff_dirty[gi].xsize() != 0 && aff_dirty[gi].ysize() != 0) {
          px_x0 = std::min(px_x0, aff_dirty[gi].x0());
          px_y0 = std::min(px_y0, aff_dirty[gi].y0());
          px_x1 =
              std::max(px_x1, aff_dirty[gi].x0() + aff_dirty[gi].xsize());
          px_y1 =
              std::max(px_y1, aff_dirty[gi].y0() + aff_dirty[gi].ysize());
        }
      }
      const double cand_table_tax_bits =
          calc_table_tax_bits(cand_table_counts);
      const double cur_tot = std::max(
          1.0, cur_bits + cur_sh_bits + cum_acs_sym_delta_bits +
                   (cur_table_tax_bits - init_table_tax_bits));
      const double new_tot = std::max(
          1.0, new_ac_qf_bits + cur_sh_bits + cum_acs_sym_delta_bits +
                   delta_acs_sym +
                   (cand_table_tax_bits - init_table_tax_bits));
      const double delta_rate = rate_term(new_tot) - rate_term(cur_tot);

      const size_t lx0 = (px_x0 > lf_pad) ? (px_x0 - lf_pad) : 0;
      const size_t ly0 = (px_y0 > lf_pad) ? (px_y0 - lf_pad) : 0;
      const size_t lx1 = std::min(xsize, px_x1 + lf_pad);
      const size_t ly1 = std::min(ysize, px_y1 + lf_pad);
      const Rect dirty_lin(lx0, ly0, lx1 - lx0, ly1 - ly0);

      const size_t lw = dirty_lin.xsize();
      const size_t lh = dirty_lin.ysize();
      saved_lin.resize(3 * lw * lh);
      for (size_t c = 0; c < 3; ++c) {
        for (size_t y = 0; y < lh; ++y) {
          memcpy(saved_lin.data() + (c * lh + y) * lw,
                 cur_lin.ConstPlaneRow(c, dirty_lin.y0() + y) + dirty_lin.x0(),
                 lw * sizeof(float));
        }
      }
      JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
          memory_manager, lf, shared, cand_pre_gab, feat_ptr, epf_sigma, epf_ws,
          dirty_lin, &cur_lin));
      std::array<double, 4> cand_d = cur_d;
      JXL_RETURN_IF_ERROR(update_4m_checkpointed(cur_lin, dirty_lin, &cand_d));

      double norm_sum = 0.0;
      bool pareto_ok = (delta_rate <= 0.35);
      for (size_t m = 0; m < 4; ++m) {
        if (weight_m[m] <= 0.0) continue;
        const double nm = (cur_d[m] - cand_d[m]) / slope_m[m];
        norm_sum += weight_m[m] * nm;
        if (delta_rate < 0.0) {
          if (nm - delta_rate < 0.0) pareto_ok = false;
        } else {
          if (nm < -0.18713162586745113 * std::max(0.008, delta_rate)) {
            pareto_ok = false;
          }
        }
      }
      const double j_gain = norm_sum - delta_rate;
      if (j_gain > 0.0018 && pareto_ok) {
        for (size_t gi = 0; gi < aff_groups.size(); ++gi) {
          const size_t g = aff_groups[gi];
          for (AcTransformCache& tr : group_caches[g].transforms) {
            tr.cur_slot = tr.cand_slot;
          }
          if (aff_dirty[gi].xsize() != 0 && aff_dirty[gi].ysize() != 0) {
            for (size_t c = 0; c < 3; ++c) {
              for (size_t y = 0; y < aff_dirty[gi].ysize(); ++y) {
                memcpy(cur_pre_gab.PlaneRow(c, aff_dirty[gi].y0() + y) +
                           aff_dirty[gi].x0(),
                       cand_pre_gab.ConstPlaneRow(c, aff_dirty[gi].y0() + y) +
                           aff_dirty[gi].x0(),
                       aff_dirty[gi].xsize() * sizeof(float));
              }
            }
          }
          cur_group_bits[g] = aff_cand_bits[gi];
        }
        for (JointCompUndo& u : undos) {
          const AcGroupBlockCache& cache = group_caches[u.comp->group];
          const size_t bw_px =
              std::min(cache.pw_px - u.comp->bx0 * kBlockDim,
                       static_cast<size_t>(u.comp->bw) * kBlockDim);
          const size_t bh_px =
              std::min(cache.ph_px - u.comp->by0 * kBlockDim,
                       static_cast<size_t>(u.comp->bh) * kBlockDim);
          apply_local_gab_box(cur_pre_gab,
                              cache.px0 + u.comp->bx0 * kBlockDim,
                              cache.py0 + u.comp->by0 * kBlockDim, bw_px, bh_px,
                              &fine_pre_gab);
          u.comp->committed = true;
          ++acs_moves;
        }
        cur_bits = new_ac_qf_bits;
        cum_acs_sym_delta_bits += delta_acs_sym;
        cur_table_tax_bits = cand_table_tax_bits;
        memcpy(quant_table_counts, cand_table_counts,
               sizeof(quant_table_counts));
        cur_d = cand_d;
      } else {
        wd->Rollback(st.cp_wd);
        ssim2->Rollback(st.cp_ss2);
        ba3->Rollback(st.cp_ba3);
        cvvdp->Rollback(st.cp_cv);
        for (size_t c = 0; c < 3; ++c) {
          for (size_t y = 0; y < lh; ++y) {
            memcpy(cur_lin.PlaneRow(c, dirty_lin.y0() + y) + dirty_lin.x0(),
                   saved_lin.data() + (c * lh + y) * lw, lw * sizeof(float));
          }
        }
        for (size_t gi = 0; gi < aff_groups.size(); ++gi) {
          if (aff_dirty[gi].xsize() != 0 && aff_dirty[gi].ysize() != 0) {
            for (size_t c = 0; c < 3; ++c) {
              for (size_t y = 0; y < aff_dirty[gi].ysize(); ++y) {
                const float* src_r =
                    cur_pre_gab.ConstPlaneRow(c, aff_dirty[gi].y0() + y) +
                    aff_dirty[gi].x0();
                memcpy(cand_pre_gab.PlaneRow(c, aff_dirty[gi].y0() + y) +
                           aff_dirty[gi].x0(),
                       src_r, aff_dirty[gi].xsize() * sizeof(float));
                memcpy(batch_pre_gab.PlaneRow(c, aff_dirty[gi].y0() + y) +
                           aff_dirty[gi].x0(),
                       src_r, aff_dirty[gi].xsize() * sizeof(float));
              }
            }
          }
        }
        for (size_t ui = undos.size(); ui > 0; --ui) {
          JointCompUndo& u = undos[ui - 1];
          AcGroupBlockCache& cache = group_caches[u.comp->group];
          const size_t bx0 = u.comp->bx0;
          const size_t by0 = u.comp->by0;
          const size_t bw = u.comp->bw;
          const size_t bh = u.comp->bh;
          const size_t box_w_px =
              std::min(cache.pw_px - bx0 * kBlockDim, bw * kBlockDim);
          const size_t box_h_px =
              std::min(cache.ph_px - by0 * kBlockDim, bh * kBlockDim);
          swap_box_transforms(&cache.transforms, bx0, by0, bw, bh, &u.saved_trs,
                              &st.acs_scratch_trs);
          for (size_t c = 0; c < 3; ++c) {
            for (size_t iy = 0; iy < box_h_px; ++iy) {
              float* JXL_RESTRICT bp =
                  cache.base_pre_gab[c].data() +
                  (by0 * kBlockDim + iy) * cache.pw_px + bx0 * kBlockDim;
              memcpy(bp, u.saved_base_patch[c].data() + iy * box_w_px,
                     box_w_px * sizeof(float));
            }
          }
          for (const AcTransformCache& tr : cache.transforms) {
            if (tr.bx < bx0 || tr.bx >= bx0 + bw || tr.by < by0 ||
                tr.by >= by0 + bh) {
              continue;
            }
            const AcStrategy acs = AcStrategy::FromRawStrategy(tr.acs_raw);
            JXL_RETURN_IF_ERROR(shared.ac_strategy.Set(
                cache.gx0 + tr.bx, cache.gy0 + tr.by, acs.Strategy()));
            for (size_t iy = 0; iy < tr.cov_y; ++iy) {
              int32_t* JXL_RESTRICT qrow =
                  rdo_quant_field.Row(cache.gy0 + tr.by + iy);
              for (size_t ix = 0; ix < tr.cov_x; ++ix) {
                qrow[cache.gx0 + tr.bx + ix] = tr.cur_quant_ac;
              }
            }
          }
          epf_sigma->UpdateRect(
              lf, shared.ac_strategy, quant_scale, rdo_quant_field,
              rdo_sharpness, Rect(cache.gx0 + bx0, cache.gy0 + by0, bw, bh));
        }
      }
    }

    auto run_acs_box_queue =
        [&](size_t max_moves_cap,
            const std::vector<std::vector<AcsBoxQueueEntry>>& queues,
            size_t* pass_moves) -> Status {
      std::vector<size_t> group_q_pos(num_groups, 0);
      while (*pass_moves < max_moves_cap) {
        struct AcsBatchItem {
          size_t group = 0;
          uint8_t bx0 = 0;
          uint8_t by0 = 0;
          uint8_t bw = 1;
          uint8_t bh = 1;
          bool is_merge = false;
          bool valid = false;
          std::vector<AcsSubSpec> cand_specs;
          std::vector<AcTransformCache> saved_trs;
          std::vector<float> saved_base_patch[3];
          float cand_ac_qf_bits = 0.0f;
          double old_acs_sym_bits = 0.0;
          double new_acs_sym_bits = 0.0;
          Rect dirty_px{0, 0, 0, 0};
        };
        std::vector<AcsBatchItem> batch;
        batch.reserve(num_groups);
        for (size_t g = 0; g < num_groups; ++g) {
          const AcGroupBlockCache& cache = group_caches[g];
          while (group_q_pos[g] < queues[g].size()) {
            const AcsBoxQueueEntry& qe = queues[g][group_q_pos[g]++];
            if (qe.bx0 + qe.bw > cache.gw_blocks ||
                qe.by0 + qe.bh > cache.gh_blocks) {
              continue;
            }
            if (qe.pre_specs.empty() && qe.bw * qe.bh > 1 &&
                (cache.px0 + (qe.bx0 + qe.bw) * kBlockDim > xsize ||
                 cache.py0 + (qe.by0 + qe.bh) * kBlockDim > ysize)) {
              continue;
            }
            // Verify box is still cleanly partitioned by current transforms.
            size_t sum_cov = 0;
            for (const AcTransformCache& tr : cache.transforms) {
              if (tr.bx >= qe.bx0 && tr.bx < qe.bx0 + qe.bw &&
                  tr.by >= qe.by0 && tr.by < qe.by0 + qe.bh &&
                  tr.bx + tr.cov_x <= qe.bx0 + qe.bw &&
                  tr.by + tr.cov_y <= qe.by0 + qe.bh) {
                sum_cov += static_cast<size_t>(tr.cov_x) * tr.cov_y;
              }
            }
            if (sum_cov != static_cast<size_t>(qe.bw) * qe.bh) continue;
            AcsBatchItem item;
            item.group = g;
            item.bx0 = qe.bx0;
            item.by0 = qe.by0;
            item.bw = qe.bw;
            item.bh = qe.bh;
            item.is_merge = qe.is_merge;
            batch.push_back(std::move(item));
            break;
          }
        }
        if (batch.empty()) break;

        std::vector<const AcsBoxQueueEntry*> batch_qe(batch.size(), nullptr);
        for (size_t bi = 0; bi < batch.size(); ++bi) {
          const size_t g = batch[bi].group;
          batch_qe[bi] = &queues[g][group_q_pos[g] - 1];
        }

        std::atomic<bool> batch_ok{true};
        JXL_RETURN_IF_ERROR(RunOnPool(
            pool, 0, static_cast<uint32_t>(batch.size()),
            [&](size_t num_threads) -> Status {
              const size_t want = std::max<size_t>(1, num_threads);
              if (dp_ws.size() < want) dp_ws.resize(want);
              return true;
            },
            [&](uint32_t bi, size_t thread) -> Status {
              AcsBatchItem& item = batch[bi];
              const AcsBoxQueueEntry& qe = *batch_qe[bi];
              AcGroupBlockCache& cache = group_caches[item.group];
              AcsEvalWorkspace* eval_ws = &dp_ws[thread].eval_ws;
              const size_t bx0 = item.bx0;
              const size_t by0 = item.by0;
              const size_t bw = item.bw;
              const size_t bh = item.bh;

              int8_t box_ytox = 0;
              int8_t box_ytob = 0;
              double old_ac_bits = 0.0;
              double old_acs_bits = 0.0;
              size_t old_tr_cnt = 0;
              uint8_t cur_only_raw = 255;
              for (const AcTransformCache& tr : cache.transforms) {
                if (tr.bx < bx0 || tr.bx >= bx0 + bw || tr.by < by0 ||
                    tr.by >= by0 + bh) {
                  continue;
                }
                if (old_tr_cnt == 0) {
                  box_ytox = tr.cur_ytox_q;
                  box_ytob = tr.cur_ytob_q;
                  cur_only_raw = tr.acs_raw;
                } else {
                  cur_only_raw = 255;
                }
                ++old_tr_cnt;
                old_ac_bits += tr.slots[tr.cur_slot].bits;
                if (tr.acs_raw < 27) {
                  old_acs_bits += tcm.acs_strategy_bits[tr.acs_raw];
                }
              }
              if (old_tr_cnt == 0) return true;
              item.old_acs_sym_bits = old_acs_bits;

              const size_t fx0 = cache.px0 + bx0 * kBlockDim;
              const size_t fy0 = cache.py0 + by0 * kBlockDim;
              const size_t box_w_px =
                  std::min(cache.pw_px - bx0 * kBlockDim, bw * kBlockDim);
              const size_t box_h_px =
                  std::min(cache.ph_px - by0 * kBlockDim, bh * kBlockDim);

              if (!qe.pre_specs.empty()) {
                item.cand_specs = qe.pre_specs;
              } else {
                if (bw > 4 || bh > 4) return true;
                const Rect bgr = shared.frame_dim.BlockGroupRect(item.group);
                int32_t blk_q[4][4] = {};
                int8_t blk_rdoq[4][4] = {};
                uint32_t blk_act[4][4] = {};
                float blk_bq[4][4][3][4] = {};
                for (const AcTransformCache& tr : cache.transforms) {
                  if (tr.bx < bx0 || tr.bx >= bx0 + bw || tr.by < by0 ||
                      tr.by >= by0 + bh) {
                    continue;
                  }
                  const float ratio = static_cast<float>(tr.cur_quant_ac) /
                                      static_cast<float>(tr.base_quant_ac);
                  for (size_t iy = 0; iy < tr.cov_y; ++iy) {
                    for (size_t ix = 0; ix < tr.cov_x; ++ix) {
                      const size_t ly = (tr.by - by0) + iy;
                      const size_t lx = (tr.bx - bx0) + ix;
                      blk_q[ly][lx] = tr.cur_quant_ac;
                      blk_rdoq[ly][lx] = tr.cur_rdoq_level;
                      if (ix == 0 && iy == 0) {
                        blk_act[ly][lx] = tr.act_bits;
                        for (size_t c = 0; c < 3; ++c) {
                          for (size_t s = 0; s < 4; ++s) {
                            blk_bq[ly][lx][c][s] = tr.base_band_q[c][s] * ratio;
                          }
                        }
                      } else {
                        const float bq10_raw =
                            bgr.ConstRow(enc_state->band_qf[1][0],
                                         tr.by + iy)[tr.bx + ix];
                        uint32_t abits = 0;
                        memcpy(&abits, &bq10_raw, sizeof(abits));
                        blk_act[ly][lx] = abits & 31u;
                        for (size_t c = 0; c < 3; ++c) {
                          for (size_t s = 0; s < 4; ++s) {
                            blk_bq[ly][lx][c][s] =
                                bgr.ConstRow(enc_state->band_qf[c][s],
                                             tr.by + iy)[tr.bx + ix] *
                                ratio;
                          }
                        }
                      }
                    }
                  }
                }

                const bool has_rdo_cfl = (st.rdo_ytox_map.xsize() > 0 &&
                                          st.rdo_ytob_map.xsize() > 0);
                auto fill_specs_for_layout = [&](const AcsLayoutDef& ldef,
                                                 AcsSubSpec out_sp[4]) -> bool {
                  for (size_t k = 0; k < ldef.count; ++k) {
                    const AcStrategy acs =
                        AcStrategy::FromRawStrategy(ldef.types[k]);
                    const size_t scw = acs.covered_blocks_x();
                    const size_t sch = acs.covered_blocks_y();
                    const size_t dx = ldef.dx[k];
                    const size_t dy = ldef.dy[k];
                    if (dx + scw > bw || dy + sch > bh) return false;
                    AcsSubSpec& sp = out_sp[k];
                    sp.bx = static_cast<uint8_t>(bx0 + dx);
                    sp.by = static_cast<uint8_t>(by0 + dy);
                    sp.acs_raw = acs.RawStrategy();
                    const size_t ctx =
                        (cache.gx0 + sp.bx) / kColorTileDimInBlocks;
                    const size_t cty =
                        (cache.gy0 + sp.by) / kColorTileDimInBlocks;
                    sp.ytox_q =
                        (has_rdo_cfl && ctx < st.rdo_ytox_map.xsize() &&
                         cty < st.rdo_ytox_map.ysize())
                            ? st.rdo_ytox_map.ConstRow(cty)[ctx]
                            : box_ytox;
                    sp.ytob_q =
                        (has_rdo_cfl && ctx < st.rdo_ytob_map.xsize() &&
                         cty < st.rdo_ytob_map.ysize())
                            ? st.rdo_ytob_map.ConstRow(cty)[ctx]
                            : box_ytob;
                    int32_t q_max = 1;
                    double q_sum = 0.0;
                    int8_t rdoq_best = -127;
                    uint32_t best_act = blk_act[dy][dx];
                    for (size_t iy = 0; iy < sch; ++iy) {
                      for (size_t ix = 0; ix < scw; ++ix) {
                        q_max = std::max(q_max, blk_q[dy + iy][dx + ix]);
                        q_sum += blk_q[dy + iy][dx + ix];
                        rdoq_best =
                            std::max(rdoq_best, blk_rdoq[dy + iy][dx + ix]);
                        const uint32_t a = blk_act[dy + iy][dx + ix];
                        if (((a >> 1) & 15u) > ((best_act >> 1) & 15u)) {
                          best_act = a;
                        }
                      }
                    }
                    const float inv_area =
                        1.0f / static_cast<float>(scw * sch);
                    const float q_mean =
                        std::max(1.0f, static_cast<float>(q_sum) * inv_area);
                    sp.quant_ac = q_max;
                    sp.rdoq_lvl = (scw * sch > 1 && qe.is_merge)
                                      ? std::max<int8_t>(rdoq_best, -1)
                                      : rdoq_best;
                    sp.act_bits = best_act;
                    const float q_adj = static_cast<float>(q_max) / q_mean;
                    for (size_t c = 0; c < 3; ++c) {
                      for (size_t s = 0; s < 4; ++s) {
                        float sq = 0.0f;
                        for (size_t iy = 0; iy < sch; ++iy) {
                          for (size_t ix = 0; ix < scw; ++ix) {
                            const float v = blk_bq[dy + iy][dx + ix][c][s];
                            sq += v * v;
                          }
                        }
                        sp.band_q[c][s] = std::sqrt(sq * inv_area) * q_adj;
                      }
                    }
                  }
                  return true;
                };

                int best_li = -1;
                if (qe.layouts.size() == 1) {
                  const AcsLayoutDef& l0 = qe.layouts[0];
                  AcsSubSpec tmp_sp[4] = {};
                  if (!(old_tr_cnt == 1 && l0.count == 1 &&
                        static_cast<uint8_t>(l0.types[0]) == cur_only_raw) &&
                      fill_specs_for_layout(l0, tmp_sp)) {
                    item.cand_specs.assign(tmp_sp, tmp_sp + l0.count);
                    best_li = 0;
                  }
                } else {
                  apply_local_gab_box(cur_pre_gab, fx0, fy0, box_w_px, box_h_px,
                                      &fine_pre_gab);
                  double cur_loc_err = 0.0;
                  for (size_t iy = 0; iy < bh; ++iy) {
                    for (size_t ix = 0; ix < bw; ++ix) {
                      float bdry_b = 0.0f;
                      cur_loc_err += ComputeLocalBlockEpfError(
                          *enc_state->orig_opsin, fine_pre_gab,
                          enc_state->initial_quant_masking1x1,
                          cache.gy0 + by0 + iy, cache.gx0 + bx0 + ix, 1.20f,
                          &bdry_b, nullptr, nullptr);
                      cur_loc_err += kAcsEpfBdryWD * bdry_b;
                    }
                  }
                  const float qf_unit =
                      0.65f * Clamp1(tcm.qf_block_bits / 1.50f, 0.50f, 2.00f);
                  const double cur_loc_bits =
                      old_ac_bits + old_acs_bits + old_tr_cnt * qf_unit;
                  double best_loc_score = -0.30 * std::max(1.0, cur_loc_err);

                  for (size_t li = 0; li < qe.layouts.size(); ++li) {
                    const AcsLayoutDef& ldef = qe.layouts[li];
                    if (old_tr_cnt == 1 && ldef.count == 1 &&
                        static_cast<uint8_t>(ldef.types[0]) == cur_only_raw) {
                      continue;
                    }
                    if (old_tr_cnt == 4 && ldef.count == 4 &&
                        ldef.types[0] == AcStrategyType::DCT) {
                      bool all_dct8 = true;
                      for (const AcTransformCache& tr : cache.transforms) {
                        if (tr.bx >= bx0 && tr.bx < bx0 + bw && tr.by >= by0 &&
                            tr.by < by0 + bh &&
                            tr.acs_raw !=
                                static_cast<uint8_t>(AcStrategyType::DCT)) {
                          all_dct8 = false;
                          break;
                        }
                      }
                      if (all_dct8) continue;
                    }
                    AcsSubSpec tmp_sp[4] = {};
                    if (!fill_specs_for_layout(ldef, tmp_sp)) continue;
                    float cand_ac_local = 0.0f;
                    if (!HWY_DYNAMIC_DISPATCH(EvalAcsBoxCandidateImpl)(
                            enc_state, opsin, rect, tiles_x, kTileBlocks, cache,
                            bx0, by0, bw, bh, ldef.count, tmp_sp, nullptr,
                            nullptr, &cand_ac_local, &plain_pre_gab, eval_ws)) {
                      batch_ok.store(false);
                      return true;
                    }
                    apply_local_gab_box(plain_pre_gab, fx0, fy0, box_w_px,
                                        box_h_px, &fine_pre_gab);
                    double cand_loc_err = 0.0;
                    for (size_t iy = 0; iy < bh; ++iy) {
                      for (size_t ix = 0; ix < bw; ++ix) {
                        float bdry_b = 0.0f;
                        cand_loc_err += ComputeLocalBlockEpfError(
                            *enc_state->orig_opsin, fine_pre_gab,
                            enc_state->initial_quant_masking1x1,
                            cache.gy0 + by0 + iy, cache.gx0 + bx0 + ix, 1.20f,
                            &bdry_b, nullptr, nullptr);
                        cand_loc_err += kAcsEpfBdryWD * bdry_b;
                      }
                    }
                    double cand_acs_b = 0.0;
                    size_t tmp_t_cnt[kNumQuantTables] = {};
                    for (size_t k = 0; k < ldef.count; ++k) {
                      if (tmp_sp[k].acs_raw < 27) {
                        cand_acs_b += tcm.acs_strategy_bits[tmp_sp[k].acs_raw];
                      }
                      const size_t ti = static_cast<size_t>(
                          kAcStrategyToQuantTableMap[tmp_sp[k].acs_raw]);
                      if (ti < kNumQuantTables) tmp_t_cnt[ti]++;
                    }
                    double tax_pen = 0.0;
                    for (size_t t = 1;
                         t < kNumQuantTables && t < qm_encs.size(); ++t) {
                      if (tmp_t_cnt[t] > 0 && quant_table_counts[t] == 0 &&
                          qm_encs[t].mode == QuantEncoding::kQuantModeDCT) {
                        tax_pen +=
                            3.0 +
                            48.0 * static_cast<double>(
                                       qm_encs[t].dct_params.num_distance_bands);
                      }
                    }
                    const double cand_loc_bits =
                        cand_ac_local + cand_acs_b + ldef.count * qf_unit +
                        tax_pen;
                    const double loc_score =
                        (cur_loc_err - cand_loc_err) -
                        kP2gLocLamMul *
                            static_cast<double>(tcm.bits_to_cost_mul) *
                            (cand_loc_bits - cur_loc_bits);
                    if (loc_score > best_loc_score) {
                      best_loc_score = loc_score;
                      best_li = static_cast<int>(li);
                      item.cand_specs.assign(tmp_sp, tmp_sp + ldef.count);
                    }
                  }
                  apply_local_gab_box(cur_pre_gab, fx0, fy0, box_w_px, box_h_px,
                                      &fine_pre_gab);
                }
                if (best_li < 0) item.cand_specs.clear();
              }
              if (item.cand_specs.empty()) return true;

              std::vector<AcTransformCache> new_trs;
              std::vector<float> new_base_patch[3];
              if (!HWY_DYNAMIC_DISPATCH(EvalAcsBoxCandidateImpl)(
                      enc_state, opsin, rect, tiles_x, kTileBlocks, cache, bx0,
                      by0, bw, bh, item.cand_specs.size(),
                      item.cand_specs.data(), &new_trs, new_base_patch, nullptr,
                      nullptr, eval_ws)) {
                batch_ok.store(false);
                return true;
              }
              item.new_acs_sym_bits = 0.0;
              for (const AcTransformCache& tr : new_trs) {
                if (tr.acs_raw < 27) {
                  item.new_acs_sym_bits += tcm.acs_strategy_bits[tr.acs_raw];
                }
              }

              for (size_t c = 0; c < 3; ++c) {
                item.saved_base_patch[c].resize(box_w_px * box_h_px);
                for (size_t iy = 0; iy < box_h_px; ++iy) {
                  float* JXL_RESTRICT bp =
                      cache.base_pre_gab[c].data() +
                      (by0 * kBlockDim + iy) * cache.pw_px + bx0 * kBlockDim;
                  memcpy(item.saved_base_patch[c].data() + iy * box_w_px, bp,
                         box_w_px * sizeof(float));
                  memcpy(bp, new_base_patch[c].data() + iy * box_w_px,
                         box_w_px * sizeof(float));
                }
              }

              std::vector<AcTransformCache> local_scratch;
              swap_box_transforms(&cache.transforms, bx0, by0, bw, bh, &new_trs,
                                  &local_scratch);
              item.saved_trs = std::move(new_trs);

              float ac_bits = 0.0f;
              if (!HWY_DYNAMIC_DISPATCH(EvalAcGroupCacheImpl)(
                      enc_state, &cache, /*qf_step_override=*/0,
                      /*use_rdoq=*/true, /*commit_cur_slot=*/false,
                      /*only_write_changed=*/true, &batch_pre_gab, nullptr,
                      &ac_bits, &item.dirty_px)) {
                batch_ok.store(false);
                return true;
              }
              const float qf_bits = EstimateGroupQfBits(cache, tcm, 0, nullptr);
              item.cand_ac_qf_bits = ac_bits + qf_bits;
              item.valid = true;
              return true;
            },
            "EvalAcsBoxBatch"));
        JXL_ENSURE(batch_ok.load());

        for (AcsBatchItem& item : batch) {
          if (!item.valid) continue;
          const size_t g = item.group;
          AcGroupBlockCache& cache = group_caches[g];
          const size_t bx0 = item.bx0;
          const size_t by0 = item.by0;
          const size_t bw = item.bw;
          const size_t bh = item.bh;
          const size_t box_w_px =
              std::min(cache.pw_px - bx0 * kBlockDim, bw * kBlockDim);
          const size_t box_h_px =
              std::min(cache.ph_px - by0 * kBlockDim, bh * kBlockDim);

          size_t cand_table_counts[kNumQuantTables];
          memcpy(cand_table_counts, quant_table_counts,
                 sizeof(quant_table_counts));
          for (const AcTransformCache& tr : item.saved_trs) {
            const size_t ti =
                static_cast<size_t>(kAcStrategyToQuantTableMap[tr.acs_raw]);
            if (ti < kNumQuantTables && cand_table_counts[ti] > 0) {
              cand_table_counts[ti]--;
            }
          }
          for (const AcsSubSpec& sp : item.cand_specs) {
            const size_t ti =
                static_cast<size_t>(kAcStrategyToQuantTableMap[sp.acs_raw]);
            if (ti < kNumQuantTables) cand_table_counts[ti]++;
          }
          const double cand_table_tax_bits =
              calc_table_tax_bits(cand_table_counts);
          const double delta_acs_sym =
              item.new_acs_sym_bits - item.old_acs_sym_bits;
          const double new_ac_qf_bits =
              cur_bits +
              static_cast<double>(item.cand_ac_qf_bits - cur_group_bits[g]);
          const double cur_tot = std::max(
              1.0, cur_bits + cur_sh_bits + cum_acs_sym_delta_bits +
                       (cur_table_tax_bits - init_table_tax_bits));
          const double new_tot = std::max(
              1.0, new_ac_qf_bits + cur_sh_bits + cum_acs_sym_delta_bits +
                       delta_acs_sym +
                       (cand_table_tax_bits - init_table_tax_bits));
          const double delta_rate = rate_term(new_tot) - rate_term(cur_tot);

          auto rollback_cache_only = [&]() {
            if (item.dirty_px.xsize() != 0 && item.dirty_px.ysize() != 0) {
              for (size_t c = 0; c < 3; ++c) {
                for (size_t y = 0; y < item.dirty_px.ysize(); ++y) {
                  memcpy(batch_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                             item.dirty_px.x0(),
                         cur_pre_gab.ConstPlaneRow(c, item.dirty_px.y0() + y) +
                             item.dirty_px.x0(),
                         item.dirty_px.xsize() * sizeof(float));
                }
              }
            }
            swap_box_transforms(&cache.transforms, bx0, by0, bw, bh,
                                &item.saved_trs, &st.acs_scratch_trs);
            for (size_t c = 0; c < 3; ++c) {
              for (size_t iy = 0; iy < box_h_px; ++iy) {
                float* JXL_RESTRICT bp =
                    cache.base_pre_gab[c].data() +
                    (by0 * kBlockDim + iy) * cache.pw_px + bx0 * kBlockDim;
                memcpy(bp, item.saved_base_patch[c].data() + iy * box_w_px,
                       box_w_px * sizeof(float));
              }
            }
          };

          const double max_drate =
              item.is_merge
                  ? 0.015
                  : (bw * bh == 1
                         ? 0.08
                         : (0.08 + 0.015 * std::min<double>(bw * bh, 32)));
          if (*pass_moves >= max_moves_cap || delta_rate > max_drate) {
            rollback_cache_only();
            continue;
          }

          // Update shared.ac_strategy, rdo_quant_field, and epf_sigma on box.
          const Rect box_blk_rect(cache.gx0 + bx0, cache.gy0 + by0, bw, bh);
          for (const AcsSubSpec& sp : item.cand_specs) {
            const AcStrategy acs = AcStrategy::FromRawStrategy(sp.acs_raw);
            JXL_RETURN_IF_ERROR(shared.ac_strategy.Set(
                cache.gx0 + sp.bx, cache.gy0 + sp.by, acs.Strategy()));
            for (size_t iy = 0; iy < acs.covered_blocks_y(); ++iy) {
              int32_t* JXL_RESTRICT qrow =
                  rdo_quant_field.Row(cache.gy0 + sp.by + iy);
              for (size_t ix = 0; ix < acs.covered_blocks_x(); ++ix) {
                qrow[cache.gx0 + sp.bx + ix] = sp.quant_ac;
              }
            }
          }
          epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale,
                                rdo_quant_field, rdo_sharpness, box_blk_rect);

          if (item.dirty_px.xsize() != 0 && item.dirty_px.ysize() != 0) {
            for (size_t c = 0; c < 3; ++c) {
              for (size_t y = 0; y < item.dirty_px.ysize(); ++y) {
                memcpy(cand_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       batch_pre_gab.ConstPlaneRow(c, item.dirty_px.y0() + y) +
                           item.dirty_px.x0(),
                       item.dirty_px.xsize() * sizeof(float));
              }
            }
          }

          const size_t px_x0 =
              (item.dirty_px.xsize() != 0)
                  ? std::min(item.dirty_px.x0(), cache.px0 + bx0 * kBlockDim)
                  : (cache.px0 + bx0 * kBlockDim);
          const size_t px_y0 =
              (item.dirty_px.ysize() != 0)
                  ? std::min(item.dirty_px.y0(), cache.py0 + by0 * kBlockDim)
                  : (cache.py0 + by0 * kBlockDim);
          const size_t px_x1 =
              (item.dirty_px.xsize() != 0)
                  ? std::max(item.dirty_px.x0() + item.dirty_px.xsize(),
                             cache.px0 + bx0 * kBlockDim + box_w_px)
                  : (cache.px0 + bx0 * kBlockDim + box_w_px);
          const size_t px_y1 =
              (item.dirty_px.ysize() != 0)
                  ? std::max(item.dirty_px.y0() + item.dirty_px.ysize(),
                             cache.py0 + by0 * kBlockDim + box_h_px)
                  : (cache.py0 + by0 * kBlockDim + box_h_px);
          const size_t lx0 = (px_x0 > lf_pad) ? (px_x0 - lf_pad) : 0;
          const size_t ly0 = (px_y0 > lf_pad) ? (px_y0 - lf_pad) : 0;
          const size_t lx1 = std::min(xsize, px_x1 + lf_pad);
          const size_t ly1 = std::min(ysize, px_y1 + lf_pad);
          const Rect dirty_lin(lx0, ly0, lx1 - lx0, ly1 - ly0);

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
          JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
              memory_manager, lf, shared, cand_pre_gab, feat_ptr, epf_sigma,
              epf_ws, dirty_lin, &cur_lin));

          std::array<double, 4> cand_d = cur_d;
          JXL_RETURN_IF_ERROR(
              update_4m_checkpointed(cur_lin, dirty_lin, &cand_d));

          double norm_sum = 0.0;
          bool pareto_ok = true;
          for (size_t m = 0; m < 4; ++m) {
            if (weight_m[m] <= 0.0) continue;
            const double nm = (cur_d[m] - cand_d[m]) / slope_m[m];
            norm_sum += weight_m[m] * nm;
            if (delta_rate < 0.0) {
              if (nm - delta_rate < 0.0) pareto_ok = false;
            } else {
              if (nm < -0.18713162586745113 * std::max(0.008, delta_rate)) {
                pareto_ok = false;
              }
            }
          }
          const double j_gain = norm_sum - delta_rate;
          const double min_gain =
              (delta_rate < 0.0) ? 0.0008 : (bw * bh == 1 ? 0.0010 : 0.0014);

          if (j_gain > min_gain && pareto_ok) {
            for (AcTransformCache& tr : cache.transforms) {
              tr.cur_slot = tr.cand_slot;
            }
            if (item.dirty_px.xsize() != 0 && item.dirty_px.ysize() != 0) {
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
            apply_local_gab_box(cur_pre_gab, cache.px0 + bx0 * kBlockDim,
                                cache.py0 + by0 * kBlockDim, box_w_px, box_h_px,
                                &fine_pre_gab);
            cur_bits = new_ac_qf_bits;
            cur_group_bits[g] = item.cand_ac_qf_bits;
            cum_acs_sym_delta_bits += delta_acs_sym;
            cur_table_tax_bits = cand_table_tax_bits;
            memcpy(quant_table_counts, cand_table_counts,
                   sizeof(quant_table_counts));
            cur_d = cand_d;
            ++(*pass_moves);
            ++acs_moves;
          } else {
            wd->Rollback(st.cp_wd);
            ssim2->Rollback(st.cp_ss2);
            ba3->Rollback(st.cp_ba3);
            cvvdp->Rollback(st.cp_cv);
            for (size_t c = 0; c < 3; ++c) {
              for (size_t y = 0; y < lh; ++y) {
                memcpy(cur_lin.PlaneRow(c, dirty_lin.y0() + y) + dirty_lin.x0(),
                       saved_lin.data() + (c * lh + y) * lw, lw * sizeof(float));
              }
            }
            if (item.dirty_px.xsize() != 0 && item.dirty_px.ysize() != 0) {
              for (size_t c = 0; c < 3; ++c) {
                for (size_t y = 0; y < item.dirty_px.ysize(); ++y) {
                  const float* src_r =
                      cur_pre_gab.ConstPlaneRow(c, item.dirty_px.y0() + y) +
                      item.dirty_px.x0();
                  memcpy(cand_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                             item.dirty_px.x0(),
                         src_r, item.dirty_px.xsize() * sizeof(float));
                  memcpy(batch_pre_gab.PlaneRow(c, item.dirty_px.y0() + y) +
                             item.dirty_px.x0(),
                         src_r, item.dirty_px.xsize() * sizeof(float));
                }
              }
            }
            swap_box_transforms(&cache.transforms, bx0, by0, bw, bh,
                                &item.saved_trs, &st.acs_scratch_trs);
            for (size_t c = 0; c < 3; ++c) {
              for (size_t iy = 0; iy < box_h_px; ++iy) {
                float* JXL_RESTRICT bp =
                    cache.base_pre_gab[c].data() +
                    (by0 * kBlockDim + iy) * cache.pw_px + bx0 * kBlockDim;
                memcpy(bp, item.saved_base_patch[c].data() + iy * box_w_px,
                       box_w_px * sizeof(float));
              }
            }
            // Restore shared.ac_strategy, rdo_quant_field, and epf_sigma.
            for (const AcTransformCache& tr : cache.transforms) {
              if (tr.bx < bx0 || tr.bx >= bx0 + bw || tr.by < by0 ||
                  tr.by >= by0 + bh) {
                continue;
              }
              const AcStrategy acs = AcStrategy::FromRawStrategy(tr.acs_raw);
              JXL_RETURN_IF_ERROR(shared.ac_strategy.Set(
                  cache.gx0 + tr.bx, cache.gy0 + tr.by, acs.Strategy()));
              for (size_t iy = 0; iy < tr.cov_y; ++iy) {
                int32_t* JXL_RESTRICT qrow =
                    rdo_quant_field.Row(cache.gy0 + tr.by + iy);
                for (size_t ix = 0; ix < tr.cov_x; ++ix) {
                  qrow[cache.gx0 + tr.bx + ix] = tr.cur_quant_ac;
                }
              }
            }
            epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale,
                                  rdo_quant_field, rdo_sharpness, box_blk_rect);
          }
        }
      }
      return true;
    };

    // Stage II.B: Verify uncommitted DP difference components sorted by dp_gain
    // through exact 4-metric checkpointed arbiter.
    {
      std::vector<const GroupAcsDpComponent*> pending_dp;
      for (size_t g = 0; g < num_groups; ++g) {
        for (const GroupAcsDpComponent& comp : group_dp_comps[g]) {
          if (!comp.committed && !comp.specs.empty()) {
            pending_dp.push_back(&comp);
          }
        }
      }
      if (!pending_dp.empty()) {
        std::sort(pending_dp.begin(), pending_dp.end(),
                  [](const GroupAcsDpComponent* a,
                     const GroupAcsDpComponent* b) {
                    if (a->dp_gain != b->dp_gain) return a->dp_gain > b->dp_gain;
                    if (a->group != b->group) return a->group < b->group;
                    return (a->by0 != b->by0) ? (a->by0 < b->by0)
                                              : (a->bx0 < b->bx0);
                  });
        std::vector<std::vector<AcsBoxQueueEntry>> dp_queues(num_groups);
        for (const GroupAcsDpComponent* c : pending_dp) {
          if (dp_queues[c->group].size() >= 32) continue;
          AcsBoxQueueEntry qe;
          qe.bx0 = c->bx0;
          qe.by0 = c->by0;
          qe.bw = c->bw;
          qe.bh = c->bh;
          qe.is_merge = false;
          qe.pre_specs = c->specs;
          dp_queues[c->group].push_back(std::move(qe));
        }
        size_t dp_moves = 0;
        JXL_RETURN_IF_ERROR(
            run_acs_box_queue(num_blocks, dp_queues, &dp_moves));
      }
    }

    // Stage II.C: Salient 2x2 and 1x1 4-metric rescue pass for top residual
    // edge/contour blocks where perceptual 4-metric error disagrees with XYB L2.
    if (acs_moves > 0) {
      accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/false);
      sum_blk_err_2f = 0.0;
      for (double v : blk_err_cur) sum_blk_err_2f += v;
      mean_blk_err_2f =
          sum_blk_err_2f / static_cast<double>(std::max<size_t>(1, num_blocks));
    }
    {
      const AcsLayoutDef kLayoutDct8{1, {0}, {0}, {AcStrategyType::DCT}};
      const AcsLayoutDef kLayoutDct16{1, {0}, {0}, {AcStrategyType::DCT16X16}};
      const std::vector<AcsLayoutDef> kLayouts2x2 = {
          kLayoutDct16,
          {2,
           {0, 0},
           {0, 1},
           {AcStrategyType::DCT8X16, AcStrategyType::DCT8X16}},
          {2,
           {0, 1},
           {0, 0},
           {AcStrategyType::DCT16X8, AcStrategyType::DCT16X8}},
          {4,
           {0, 1, 0, 1},
           {0, 0, 1, 1},
           {AcStrategyType::DCT, AcStrategyType::DCT, AcStrategyType::DCT,
            AcStrategyType::DCT}},
      };
      std::vector<AcsLayoutDef> kLayouts1x1 = {
          kLayoutDct8,
          {1, {0}, {0}, {AcStrategyType::DCT4X4}},
          {1, {0}, {0}, {AcStrategyType::DCT4X8}},
          {1, {0}, {0}, {AcStrategyType::DCT8X4}},
          {1, {0}, {0}, {AcStrategyType::AFV0}},
          {1, {0}, {0}, {AcStrategyType::AFV1}},
          {1, {0}, {0}, {AcStrategyType::AFV2}},
          {1, {0}, {0}, {AcStrategyType::AFV3}},
          {1, {0}, {0}, {AcStrategyType::DCT2X2}},
      };
      if (cparams.decoding_speed_tier < 4) {
        kLayouts1x1.push_back({1, {0}, {0}, {AcStrategyType::IDENTITY}});
      }

      struct ScoredAcsBox {
        double score;
        size_t group;
        uint8_t bx0;
        uint8_t by0;
        uint8_t bw;
        uint8_t bh;
      };

      // Compact residual reclaim pass for 1-bit symbol savings skipped by DP
      // hysteresis (sub-8x8 -> DCT8 and near-uniform 2x2 -> DCT16) under the
      // unanimous 4-metric gate.
      size_t reclaim_moves = 0;
      const double dct8_sym_bits =
          tcm.acs_strategy_bits[static_cast<size_t>(AcStrategyType::DCT)];
      std::vector<ScoredAcsBox> rec_1x1;
      for (size_t g = 0; g < num_groups; ++g) {
        const AcGroupBlockCache& cache = group_caches[g];
        for (const AcTransformCache& tr : cache.transforms) {
          if (tr.cov_x != 1 || tr.cov_y != 1 ||
              tr.acs_raw == static_cast<uint8_t>(AcStrategyType::DCT) ||
              tr.acs_raw >= 27) {
            continue;
          }
          const double sym_save =
              tcm.acs_strategy_bits[tr.acs_raw] - dct8_sym_bits;
          if (sym_save <= 0.4) continue;
          const double be =
              blk_err_cur[(cache.gy0 + tr.by) * xsize_blocks + cache.gx0 +
                          tr.bx];
          const double sc = sym_save - 0.15 * be;
          if (sc > 0.0) {
            rec_1x1.push_back({sc, g, tr.bx, tr.by, 1, 1});
          }
        }
      }
      if (!rec_1x1.empty()) {
        std::sort(rec_1x1.begin(), rec_1x1.end(),
                  [](const ScoredAcsBox& a, const ScoredAcsBox& b) {
                    if (a.score != b.score) return a.score > b.score;
                    if (a.group != b.group) return a.group < b.group;
                    return (a.by0 != b.by0) ? (a.by0 < b.by0) : (a.bx0 < b.bx0);
                  });
        std::vector<std::vector<AcsBoxQueueEntry>> queues(num_groups);
        const std::vector<AcsLayoutDef> kOnlyDct8 = {kLayoutDct8};
        for (const ScoredAcsBox& c : rec_1x1) {
          if (queues[c.group].size() >= 14) continue;
          queues[c.group].push_back(
              {c.bx0, c.by0, 1, 1, /*is_merge=*/true, {}, kOnlyDct8});
        }
        size_t r1_moves = 0;
        const size_t cap_r1 = std::max<size_t>(6, num_groups * 10);
        JXL_RETURN_IF_ERROR(run_acs_box_queue(cap_r1, queues, &r1_moves));
        reclaim_moves += r1_moves;
      }

      const size_t qt_dct16 = static_cast<size_t>(
          kAcStrategyToQuantTableMap[static_cast<size_t>(
              AcStrategyType::DCT16X16)]);
      if (qt_dct16 == 0 ||
          (qt_dct16 < kNumQuantTables && quant_table_counts[qt_dct16] > 0)) {
        const double dct16_sym_bits = tcm.acs_strategy_bits[static_cast<size_t>(
            AcStrategyType::DCT16X16)];
        std::vector<ScoredAcsBox> rec_2x2;
        for (size_t g = 0; g < num_groups; ++g) {
          const AcGroupBlockCache& cache = group_caches[g];
          for (size_t by0 = 0; by0 + 2 <= cache.gh_blocks; by0 += 2) {
            if (cache.py0 + (by0 + 2) * kBlockDim > ysize) continue;
            for (size_t bx0 = 0; bx0 + 2 <= cache.gw_blocks; bx0 += 2) {
              if (cache.px0 + (bx0 + 2) * kBlockDim > xsize) continue;
              size_t sum_cov = 0;
              size_t tr_cnt = 0;
              int32_t q_min = std::numeric_limits<int32_t>::max();
              int32_t q_max = 0;
              double old_sym = 0.0;
              for (const AcTransformCache& tr : cache.transforms) {
                if (tr.bx >= bx0 && tr.bx < bx0 + 2 && tr.by >= by0 &&
                    tr.by < by0 + 2 && tr.bx + tr.cov_x <= bx0 + 2 &&
                    tr.by + tr.cov_y <= by0 + 2) {
                  sum_cov += static_cast<size_t>(tr.cov_x) * tr.cov_y;
                  ++tr_cnt;
                  q_min = std::min(q_min, tr.cur_quant_ac);
                  q_max = std::max(q_max, tr.cur_quant_ac);
                  if (tr.acs_raw < 27) {
                    old_sym += tcm.acs_strategy_bits[tr.acs_raw];
                  }
                }
              }
              if (sum_cov != 4 || tr_cnt < 2 || q_max - q_min > 1) continue;
              double box_err = 0.0;
              for (size_t iy = 0; iy < 2; ++iy) {
                for (size_t ix = 0; ix < 2; ++ix) {
                  box_err += blk_err_cur[(cache.gy0 + by0 + iy) * xsize_blocks +
                                         cache.gx0 + bx0 + ix];
                }
              }
              const double sc = (old_sym - dct16_sym_bits) - 0.10 * box_err;
              if (sc > 0.0) {
                rec_2x2.push_back({sc, g, static_cast<uint8_t>(bx0),
                                   static_cast<uint8_t>(by0), 2, 2});
              }
            }
          }
        }
        if (!rec_2x2.empty()) {
          std::sort(rec_2x2.begin(), rec_2x2.end(),
                    [](const ScoredAcsBox& a, const ScoredAcsBox& b) {
                      if (a.score != b.score) return a.score > b.score;
                      if (a.group != b.group) return a.group < b.group;
                      return (a.by0 != b.by0) ? (a.by0 < b.by0)
                                              : (a.bx0 < b.bx0);
                    });
          std::vector<std::vector<AcsBoxQueueEntry>> queues(num_groups);
          const std::vector<AcsLayoutDef> kOnlyDct16 = {kLayoutDct16};
          for (const ScoredAcsBox& c : rec_2x2) {
            if (queues[c.group].size() >= 14) continue;
            queues[c.group].push_back(
                {c.bx0, c.by0, 2, 2, /*is_merge=*/true, {}, kOnlyDct16});
          }
          size_t r2_moves = 0;
          const size_t cap_r2 = std::max<size_t>(4, num_groups * 8);
          JXL_RETURN_IF_ERROR(run_acs_box_queue(cap_r2, queues, &r2_moves));
          reclaim_moves += r2_moves;
        }
      }
      if (reclaim_moves > 0) {
        accumulate_4m_spatial_gain(blk_err_cur.data(),
                                   /*weight_by_step1=*/false);
        sum_blk_err_2f = 0.0;
        for (double v : blk_err_cur) sum_blk_err_2f += v;
        mean_blk_err_2f =
            sum_blk_err_2f / static_cast<double>(std::max<size_t>(1, num_blocks));
      }

      std::vector<ScoredAcsBox> sal_2x2;
      for (size_t g = 0; g < num_groups; ++g) {
        const AcGroupBlockCache& cache = group_caches[g];
        for (size_t by0 = 0; by0 + 2 <= cache.gh_blocks; by0 += 2) {
          if (cache.py0 + (by0 + 2) * kBlockDim > ysize) continue;
          for (size_t bx0 = 0; bx0 + 2 <= cache.gw_blocks; bx0 += 2) {
            if (cache.px0 + (bx0 + 2) * kBlockDim > xsize) continue;
            size_t sum_cov = 0;
            for (const AcTransformCache& tr : cache.transforms) {
              if (tr.bx >= bx0 && tr.bx < bx0 + 2 && tr.by >= by0 &&
                  tr.by < by0 + 2 && tr.bx + tr.cov_x <= bx0 + 2 &&
                  tr.by + tr.cov_y <= by0 + 2) {
                sum_cov += static_cast<size_t>(tr.cov_x) * tr.cov_y;
              }
            }
            if (sum_cov != 4) continue;
            double box_err = 0.0;
            for (size_t iy = 0; iy < 2; ++iy) {
              for (size_t ix = 0; ix < 2; ++ix) {
                box_err += blk_err_cur[(cache.gy0 + by0 + iy) * xsize_blocks +
                                       cache.gx0 + bx0 + ix];
              }
            }
            if (box_err > 1.25 * 4.0 * mean_blk_err_2f) {
              sal_2x2.push_back({box_err * 0.50, g, static_cast<uint8_t>(bx0),
                                 static_cast<uint8_t>(by0), 2, 2});
            }
          }
        }
      }
      if (!sal_2x2.empty()) {
        std::sort(sal_2x2.begin(), sal_2x2.end(),
                  [](const ScoredAcsBox& a, const ScoredAcsBox& b) {
                    if (a.score != b.score) return a.score > b.score;
                    if (a.group != b.group) return a.group < b.group;
                    return (a.by0 != b.by0) ? (a.by0 < b.by0) : (a.bx0 < b.bx0);
                  });
        std::vector<std::vector<AcsBoxQueueEntry>> queues(num_groups);
        for (const ScoredAcsBox& c : sal_2x2) {
          if (queues[c.group].size() >= 12) continue;
          queues[c.group].push_back(
              {c.bx0, c.by0, 2, 2, /*is_merge=*/false, {}, kLayouts2x2});
        }
        size_t sal_multi_moves = 0;
        const size_t cap_multi = std::max<size_t>(4, num_groups * 8);
        JXL_RETURN_IF_ERROR(
            run_acs_box_queue(cap_multi, queues, &sal_multi_moves));
      }

      std::vector<ScoredAcsBox> sal_1x1;
      for (size_t g = 0; g < num_groups; ++g) {
        const AcGroupBlockCache& cache = group_caches[g];
        for (const AcTransformCache& tr : cache.transforms) {
          if (tr.cov_x != 1 || tr.cov_y != 1) continue;
          const double be =
              blk_err_cur[(cache.gy0 + tr.by) * xsize_blocks + cache.gx0 +
                          tr.bx];
          if (be > 1.25 * mean_blk_err_2f) {
            sal_1x1.push_back({be, g, tr.bx, tr.by, 1, 1});
          }
        }
      }
      if (!sal_1x1.empty()) {
        std::sort(sal_1x1.begin(), sal_1x1.end(),
                  [](const ScoredAcsBox& a, const ScoredAcsBox& b) {
                    if (a.score != b.score) return a.score > b.score;
                    if (a.group != b.group) return a.group < b.group;
                    return (a.by0 != b.by0) ? (a.by0 < b.by0) : (a.bx0 < b.bx0);
                  });
        std::vector<std::vector<AcsBoxQueueEntry>> queues(num_groups);
        for (const ScoredAcsBox& c : sal_1x1) {
          if (queues[c.group].size() >= 14) continue;
          queues[c.group].push_back(
              {c.bx0, c.by0, 1, 1, /*is_merge=*/false, {}, kLayouts1x1});
        }
        size_t sal_1x1_moves = 0;
        const size_t cap_1x1 = std::max<size_t>(6, num_groups * 10);
        JXL_RETURN_IF_ERROR(
            run_acs_box_queue(cap_1x1, queues, &sal_1x1_moves));
      }
    }

    if (acs_moves > 0) {
      cur_bits = std::max(
          1.0, cur_bits + cum_acs_sym_delta_bits +
                   (cur_table_tax_bits - init_table_tax_bits));
    }
  }

  // Pass 2G: Closed-loop quantized VarDCTDC residual refinement (±1/±2 steps
  // on Y/X/B), strictly preserving Predictor::Gradient + TreeMode::kGradientOnly
  // while accounting for exact ClampedGradient residual entropy, DequantDC
  // color correlation, AdaptiveDCSmoothing 3x3 stencil, AC LLF pixel shifts,
  // Gaborish + EPF, and the 4-metric checkpointed arbiter.
  size_t dc_moves = 0;
  st.dc_touched_blk.assign(num_blocks, 0);
  if (modular_frame_encoder != nullptr && !enc_state->streaming_mode &&
      !(frame_header.flags & FrameHeader::kUseDcFrame) &&
      frame_header.dc_level == 0 &&
      frame_header.chroma_subsampling.Is444() &&
      frame_header.color_transform == ColorTransform::kXYB &&
      shared.block_ctx_map.num_dc_ctxs <= 1 &&
      shared.dc_storage.xsize() >= xsize_blocks &&
      shared.dc_storage.ysize() >= ysize_blocks &&
      enc_state->initial_quant_masking1x1.xsize() >= xsize &&
      enc_state->initial_quant_masking1x1.ysize() >= ysize) {
    const size_t num_dc_groups = shared.frame_dim.num_dc_groups;
    bool dc_streams_ok = (num_dc_groups > 0) &&
                         (modular_frame_encoder->extra_dc_precision.size() >=
                          num_dc_groups);
    for (size_t dg = 0; dg < num_dc_groups && dc_streams_ok; ++dg) {
      const Rect dcr = shared.frame_dim.DCGroupRect(dg);
      Image* dc_img = modular_frame_encoder->VarDCTDCImage(dg);
      if (dc_img == nullptr || dc_img->channel.size() < 3 ||
          dc_img->channel[0].w != dcr.xsize() ||
          dc_img->channel[0].h != dcr.ysize()) {
        dc_streams_ok = false;
      }
    }

    if (dc_streams_ok) {
      const bool clamp_i16 =
          shared.metadata != nullptr &&
          shared.metadata->m.modular_16_bit_buffer_sufficient;
      const bool smooth_dc =
          !(frame_header.flags & FrameHeader::kSkipAdaptiveDCSmoothing) &&
          xsize_blocks > 2 && ysize_blocks > 2;
      const float* dc_factors = shared.quantizer.MulDC();
      const float* cfl_dc = shared.cmap.base().DCFactors();

      JXL_ASSIGN_OR_RETURN(
          st.dc_unsmoothed,
          Image3F::Create(memory_manager, xsize_blocks, ysize_blocks));
      for (size_t dg = 0; dg < num_dc_groups; ++dg) {
        const Rect dcr = shared.frame_dim.DCGroupRect(dg);
        const Image* dc_img = modular_frame_encoder->VarDCTDCImage(dg);
        const float mul_deq =
            1.0f /
            static_cast<float>(1u << modular_frame_encoder->extra_dc_precision[dg]);
        const float fac_x = dc_factors[0] * mul_deq;
        const float fac_y = dc_factors[1] * mul_deq;
        const float fac_b = dc_factors[2] * mul_deq;
        for (size_t ly = 0; ly < dcr.ysize(); ++ly) {
          const int32_t* qy_r = dc_img->channel[0].plane.ConstRow(ly);
          const int32_t* qx_r = dc_img->channel[1].plane.ConstRow(ly);
          const int32_t* qb_r = dc_img->channel[2].plane.ConstRow(ly);
          float* ux_r = st.dc_unsmoothed.PlaneRow(0, dcr.y0() + ly) + dcr.x0();
          float* uy_r = st.dc_unsmoothed.PlaneRow(1, dcr.y0() + ly) + dcr.x0();
          float* ub_r = st.dc_unsmoothed.PlaneRow(2, dcr.y0() + ly) + dcr.x0();
          for (size_t lx = 0; lx < dcr.xsize(); ++lx) {
            const float uy = static_cast<float>(qy_r[lx]) * fac_y;
            uy_r[lx] = uy;
            ux_r[lx] = static_cast<float>(qx_r[lx]) * fac_x + uy * cfl_dc[0];
            ub_r[lx] = static_cast<float>(qb_r[lx]) * fac_b + uy * cfl_dc[2];
          }
        }
      }

      auto eval_smoothed_dc_at = [&](size_t ux, size_t uy, float out_dc[3]) {
        if (!smooth_dc || ux == 0 || ux + 1 >= xsize_blocks || uy == 0 ||
            uy + 1 >= ysize_blocks) {
          for (size_t c = 0; c < 3; ++c) {
            out_dc[c] = st.dc_unsmoothed.ConstPlaneRow(c, uy)[ux];
          }
          return;
        }
        constexpr float kW1 = 0.20345139757231578f;
        constexpr float kW2 = 0.0334829185968739f;
        constexpr float kW0 = 1.0f - 4.0f * (kW1 + kW2);
        float mc[3];
        float sm[3];
        float gap = 0.5f;
        for (size_t c = 0; c < 3; ++c) {
          const float* rt = st.dc_unsmoothed.ConstPlaneRow(c, uy - 1);
          const float* rm = st.dc_unsmoothed.ConstPlaneRow(c, uy);
          const float* rb = st.dc_unsmoothed.ConstPlaneRow(c, uy + 1);
          mc[c] = rm[ux];
          const float side = rm[ux - 1] + rm[ux + 1] + rt[ux] + rb[ux];
          const float corner =
              rt[ux - 1] + rt[ux + 1] + rb[ux - 1] + rb[ux + 1];
          sm[c] = mc[c] * kW0 + side * kW1 + corner * kW2;
          gap = std::max(gap, std::abs((mc[c] - sm[c]) / dc_factors[c]));
        }
        const float factor = std::max(0.0f, 3.0f - 4.0f * gap);
        for (size_t c = 0; c < 3; ++c) {
          out_dc[c] = mc[c] + factor * (sm[c] - mc[c]);
        }
      };

      // Learn pilot non-WP Modular MA tree over VarDCTDC and build exact
      // HybridUint(4,1,2) per-leaf and per-channel residual histograms.
      struct DcSingleHist {
        uint32_t hist[128] = {};
        uint32_t tot = 0;
      };
      struct DcStencilPt {
        uint8_t mc_p;
        uint16_t px;
        uint16_t py;
      };
      struct DcReclaimCand {
        double score;
        uint32_t dg;
        uint16_t bx;
        uint16_t by;
        uint8_t mc;
        int8_t dq;
      };
      struct SalDcBlk {
        double err;
        uint32_t dg;
        uint16_t bx;
        uint16_t by;
      };
      struct Pass2GDcState {
        Tree dc_pilot_tree;
        std::vector<std::array<FlatTree, 3>> dc_flat_trees;
        std::vector<std::array<size_t, 3>> dc_num_props;
        Properties dc_props_scratch;
        std::vector<DcSingleHist> dc_leaf_hists;
        std::vector<DcSingleHist> dc_chan_hists;
        AlignedMemory dc_fmem;
        float gab_w_2g[3][3] = {};
        DcStencilPt pts[17] = {};
        float delta_dc[3][3][3] = {};
        float tr_dc_diff[64] = {};
        float new_u[3] = {};
        float new_dec[3][3][3] = {};
        float u_tmp[3] = {};
        float dec_tmp[3][3][3] = {};
        std::vector<DcReclaimCand> dc_reclaims;
        std::vector<SalDcBlk> sal_dc_blks;
      };
      auto p2g_ptr = jxl::make_unique<Pass2GDcState>();
      Pass2GDcState& p2g = *p2g_ptr;

      JXL_RETURN_IF_ERROR(
          modular_frame_encoder->LearnVarDCTDCTree(&p2g.dc_pilot_tree));
      const size_t num_dc_leaves =
          std::max<size_t>(1, (p2g.dc_pilot_tree.size() + 1) / 2);
      p2g.dc_flat_trees.resize(num_dc_groups);
      p2g.dc_num_props.resize(num_dc_groups);
      size_t max_dcr_w = 1;
      size_t max_dc_props = kNumNonrefProperties;
      for (size_t dg = 0; dg < num_dc_groups; ++dg) {
        const Rect dcr = shared.frame_dim.DCGroupRect(dg);
        max_dcr_w = std::max(max_dcr_w, dcr.xsize());
        const size_t stream_id =
            ModularStreamId::VarDCTDC(dg).ID(shared.frame_dim);
        for (size_t mc = 0; mc < 3; ++mc) {
          std::array<pixel_type, kNumStaticProperties> static_props = {
              {static_cast<pixel_type>(mc),
               static_cast<pixel_type>(stream_id)}};
          bool use_wp = false;
          bool is_wp_only = false;
          bool is_grad_only = false;
          p2g.dc_flat_trees[dg][mc] = FilterTree(
              p2g.dc_pilot_tree, static_props, &p2g.dc_num_props[dg][mc],
              &use_wp, &is_wp_only, &is_grad_only);
          JXL_ENSURE(!use_wp);
          max_dc_props = std::max(max_dc_props, p2g.dc_num_props[dg][mc]);
        }
      }

      p2g.dc_props_scratch.assign(max_dc_props, 0);
      JXL_ASSIGN_OR_RETURN(
          Channel dc_ref_scratch,
          Channel::Create(
              memory_manager,
              std::max<size_t>(1, max_dc_props - kNumNonrefProperties),
              max_dcr_w));

      auto eval_dc_tree_pixel = [&](size_t dg, size_t mc_eval, size_t px,
                                    size_t py, size_t* out_ctx) -> int32_t {
        const Image* dc_img = modular_frame_encoder->VarDCTDCImage(dg);
        const Channel& ch = dc_img->channel[mc_eval];
        const size_t w = ch.w;
        const ptrdiff_t onerow = ch.plane.PixelsPerRow();
        const pixel_type* JXL_RESTRICT row = ch.Row(py);
        const size_t nprops = p2g.dc_num_props[dg][mc_eval];
        if (p2g.dc_props_scratch.size() != nprops) {
          p2g.dc_props_scratch.resize(nprops);
        }
        p2g.dc_props_scratch[0] = static_cast<pixel_type>(mc_eval);
        p2g.dc_props_scratch[1] = static_cast<pixel_type>(
            ModularStreamId::VarDCTDC(dg).ID(shared.frame_dim));
        p2g.dc_props_scratch[2] = static_cast<pixel_type>(py);
        if (px == 0) {
          p2g.dc_props_scratch[9] = 0;
        } else {
          const size_t xp = px - 1;
          const pixel_type* JXL_RESTRICT prow =
              (py > 0) ? ch.Row(py - 1) : nullptr;
          const pixel_type_w left_p =
              (xp > 0) ? row[xp - 1] : (py > 0 ? prow[0] : 0);
          const pixel_type_w top_p = (py > 0) ? prow[xp] : left_p;
          const pixel_type_w tl_p =
              (xp > 0 && py > 0) ? prow[xp - 1] : left_p;
          p2g.dc_props_scratch[9] =
              static_cast<PropertyVal>(left_p + top_p - tl_p);
        }
        const size_t num_extra = nprops - kNumNonrefProperties;
        dc_ref_scratch.w = num_extra;
        if (num_extra > 0) {
          pixel_type* JXL_RESTRICT rp = dc_ref_scratch.Row(px);
          for (size_t k = 0; k < num_extra; ++k) rp[k] = 0;
          size_t offset = 0;
          for (int32_t j = static_cast<int32_t>(mc_eval) - 1;
               j >= 0 && offset < num_extra; --j) {
            const Channel& ref_ch = dc_img->channel[j];
            if (ref_ch.w != ch.w || ref_ch.h != ch.h ||
                ref_ch.hshift != ch.hshift || ref_ch.vshift != ch.vshift) {
              continue;
            }
            const pixel_type* JXL_RESTRICT rpp = ref_ch.Row(py);
            const pixel_type* JXL_RESTRICT rpprev =
                ref_ch.Row(py > 0 ? py - 1 : 0);
            const pixel_type_w v = rpp[px];
            rp[offset + 0] = static_cast<pixel_type>(std::abs(v));
            rp[offset + 1] = static_cast<pixel_type>(v);
            const pixel_type_w vleft = (px > 0) ? rpp[px - 1] : 0;
            const pixel_type_w vtop = (py > 0) ? rpprev[px] : vleft;
            const pixel_type_w vtl =
                (px > 0 && py > 0) ? rpprev[px - 1] : vleft;
            const pixel_type_w vpred = ClampedGradient(vleft, vtop, vtl);
            rp[offset + 2] = static_cast<pixel_type>(std::abs(v - vpred));
            rp[offset + 3] = static_cast<pixel_type>(v - vpred);
            offset += kExtraPropsPerChannel;
          }
        }
        MATreeLookup lookup(p2g.dc_flat_trees[dg][mc_eval]);
        const PredictionResult res = PredictTreeNoWP(
            &p2g.dc_props_scratch, w, row + px, onerow, static_cast<int>(px),
            static_cast<int>(py), lookup, dc_ref_scratch);
        if (out_ctx != nullptr) {
          *out_ctx = std::min<size_t>(static_cast<size_t>(res.context),
                                      num_dc_leaves - 1);
        }
        const int32_t mult = std::max<int32_t>(1, res.multiplier);
        return static_cast<int32_t>((row[px] - res.guess) / mult);
      };

      p2g.dc_leaf_hists.resize(num_dc_leaves);
      p2g.dc_chan_hists.resize(3);
      const HybridUintConfig dc_uint_cfg(4, 1, 2);

      for (size_t dg = 0; dg < num_dc_groups; ++dg) {
        const Rect dcr = shared.frame_dim.DCGroupRect(dg);
        const Image* dc_img = modular_frame_encoder->VarDCTDCImage(dg);
        for (size_t mc = 0; mc < 3; ++mc) {
          const Channel& ch = dc_img->channel[mc];
#if JXL_IS_DEBUG_BUILD
          const size_t nprops_dbg = p2g.dc_num_props[dg][mc];
          Properties dbg_props(nprops_dbg, 0);
          std::array<pixel_type, kNumStaticProperties> dbg_sp = {
              {static_cast<pixel_type>(mc),
               static_cast<pixel_type>(
                   ModularStreamId::VarDCTDC(dg).ID(shared.frame_dim))}};
          JXL_ASSIGN_OR_RETURN(
              Channel dbg_refs,
              Channel::Create(memory_manager, nprops_dbg - kNumNonrefProperties,
                              ch.w));
          MATreeLookup dbg_lookup(p2g.dc_flat_trees[dg][mc]);
#endif
          for (size_t ly = 0; ly < dcr.ysize(); ++ly) {
#if JXL_IS_DEBUG_BUILD
            PrecomputeReferences(ch, ly, *dc_img, static_cast<uint32_t>(mc),
                                 &dbg_refs);
            InitPropsRow(&dbg_props, dbg_sp, static_cast<int>(ly));
#endif
            for (size_t lx = 0; lx < dcr.xsize(); ++lx) {
              size_t ctx = 0;
              const int32_t res = eval_dc_tree_pixel(dg, mc, lx, ly, &ctx);
#if JXL_IS_DEBUG_BUILD
              const PredictionResult dbg_pr = PredictTreeNoWP(
                  &dbg_props, ch.w, ch.Row(ly) + lx, ch.plane.PixelsPerRow(),
                  static_cast<int>(lx), static_cast<int>(ly), dbg_lookup,
                  dbg_refs);
              const size_t dbg_ctx = std::min<size_t>(
                  static_cast<size_t>(dbg_pr.context), num_dc_leaves - 1);
              const int32_t dbg_res = static_cast<int32_t>(
                  (ch.Row(ly)[lx] - dbg_pr.guess) /
                  std::max<int32_t>(1, dbg_pr.multiplier));
              JXL_DASSERT(ctx == dbg_ctx);
              JXL_DASSERT(res == dbg_res);
#endif
              uint32_t tok = 0;
              uint32_t nb = 0;
              uint32_t eb = 0;
              dc_uint_cfg.Encode(PackSigned(res), &tok, &nb, &eb);
              tok = std::min<uint32_t>(tok, 127u);
              p2g.dc_leaf_hists[ctx].hist[tok]++;
              p2g.dc_leaf_hists[ctx].tot++;
              p2g.dc_chan_hists[mc].hist[tok]++;
              p2g.dc_chan_hists[mc].tot++;
            }
          }
        }
      }

      // Build O(1) block -> (group, transform_idx) lookup table.
      st.blk_to_tr.assign(num_blocks, 0xFFFFFFFFu);
      for (size_t g = 0; g < num_groups; ++g) {
        const AcGroupBlockCache& cache = group_caches[g];
        for (size_t ti = 0; ti < cache.transforms.size(); ++ti) {
          const AcTransformCache& tr = cache.transforms[ti];
          const uint32_t code =
              (static_cast<uint32_t>(g) << 16) | static_cast<uint32_t>(ti);
          for (size_t iy = 0; iy < tr.cov_y; ++iy) {
            for (size_t ix = 0; ix < tr.cov_x; ++ix) {
              st.blk_to_tr[(cache.gy0 + tr.by + iy) * xsize_blocks + cache.gx0 +
                           tr.bx + ix] = code;
            }
          }
        }
      }

      const size_t dct_scratch_size =
          3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;
      JXL_ASSIGN_OR_RETURN(
          p2g.dc_fmem,
          AlignedMemory::Create(
              memory_manager,
              (2 * AcStrategy::kMaxCoeffArea + dct_scratch_size) *
                  sizeof(float)));
      float* JXL_RESTRICT dc_dct_buf = p2g.dc_fmem.address<float>();
      float* JXL_RESTRICT dc_px_buf = dc_dct_buf + AcStrategy::kMaxCoeffArea;
      float* JXL_RESTRICT dc_scratch = dc_px_buf + AcStrategy::kMaxCoeffArea;

      if (lf.gab) {
        const float rw1[3] = {lf.gab_x_weight1, lf.gab_y_weight1,
                              lf.gab_b_weight1};
        const float rw2[3] = {lf.gab_x_weight2, lf.gab_y_weight2,
                              lf.gab_b_weight2};
        for (size_t c = 0; c < 3; ++c) {
          const float mul = 1.0f / (1.0f + 4.0f * (rw1[c] + rw2[c]));
          p2g.gab_w_2g[c][0] = mul;
          p2g.gab_w_2g[c][1] = rw1[c] * mul;
          p2g.gab_w_2g[c][2] = rw2[c] * mul;
        }
      }
      auto apply_gab_rect_2g = [&](const Image3F& src_xyb, size_t fx0,
                                   size_t fy0, size_t lw, size_t lh,
                                   Image3F* dst_xyb) {
        for (size_t iy = 0; iy < lh; ++iy) {
          const size_t y = fy0 + iy;
          const size_t yt = static_cast<size_t>(
              Mirror(static_cast<int64_t>(y) - 1, static_cast<int64_t>(ysize)));
          const size_t yb = static_cast<size_t>(
              Mirror(static_cast<int64_t>(y) + 1, static_cast<int64_t>(ysize)));
          for (size_t c = 0; c < 3; ++c) {
            float* JXL_RESTRICT out_r = dst_xyb->PlaneRow(c, y) + fx0;
            if (!lf.gab) {
              memcpy(out_r, src_xyb.ConstPlaneRow(c, y) + fx0,
                     lw * sizeof(float));
            } else {
              const float* rt = src_xyb.ConstPlaneRow(c, yt);
              const float* rm = src_xyb.ConstPlaneRow(c, y);
              const float* rb = src_xyb.ConstPlaneRow(c, yb);
              const float w0 = p2g.gab_w_2g[c][0];
              const float w1 = p2g.gab_w_2g[c][1];
              const float w2 = p2g.gab_w_2g[c][2];
              for (size_t ix = 0; ix < lw; ++ix) {
                const size_t x = fx0 + ix;
                const size_t xl = (x > 0) ? (x - 1) : 0;
                const size_t xr = (x + 1 < xsize) ? (x + 1) : (xsize - 1);
                out_r[ix] = rm[x] * w0 +
                            (rm[xl] + rm[xr] + rt[x] + rb[x]) * w1 +
                            (rt[xl] + rt[xr] + rb[xl] + rb[xr]) * w2;
              }
            }
            if (feat_ptr != nullptr) {
              const float* JXL_RESTRICT rf =
                  feat_ptr->ConstPlaneRow(c, y) + fx0;
              for (size_t ix = 0; ix < lw; ++ix) out_r[ix] += rf[ix];
            }
          }
        }
      };

      JXL_RETURN_IF_ERROR(CopyImageTo(cur_pre_gab, &plain_pre_gab));
      JXL_RETURN_IF_ERROR(CopyImageTo(cur_pre_gab, &cand_pre_gab));
      apply_gab_rect_2g(cur_pre_gab, 0, 0, xsize, ysize, &fine_pre_gab);

      double cum_dc_delta_bits = 0.0;

      auto collect_dc_stencil = [&](size_t dg, size_t mc, size_t lx, size_t ly,
                                    DcStencilPt out_pts[17], size_t* out_np) {
        const Rect dcr = shared.frame_dim.DCGroupRect(dg);
        const ptrdiff_t w = static_cast<ptrdiff_t>(dcr.xsize());
        const ptrdiff_t h = static_cast<ptrdiff_t>(dcr.ysize());
        const ptrdiff_t bx0 = static_cast<ptrdiff_t>(lx);
        const ptrdiff_t by0 = static_cast<ptrdiff_t>(ly);
        static constexpr int kSelfOffsets[9][2] = {
            {0, 0},  {1, 0},  {2, 0}, {-2, 1}, {-1, 1},
            {0, 1},  {1, 1},  {2, 1}, {0, 2}};
        size_t np = 0;
        for (const auto& off : kSelfOffsets) {
          const ptrdiff_t px = bx0 + off[0];
          const ptrdiff_t py = by0 + off[1];
          if (px >= 0 && px < w && py >= 0 && py < h) {
            out_pts[np++] = {static_cast<uint8_t>(mc),
                             static_cast<uint16_t>(px),
                             static_cast<uint16_t>(py)};
          }
        }
        for (size_t mc2 = mc + 1; mc2 < 3; ++mc2) {
          const size_t min_needed =
              kNumNonrefProperties +
              (mc2 - 1 - mc) * kExtraPropsPerChannel;
          if (p2g.dc_num_props[dg][mc2] <= min_needed) continue;
          for (ptrdiff_t dy = 0; dy <= 1; ++dy) {
            const ptrdiff_t py = by0 + dy;
            if (py >= h) continue;
            for (ptrdiff_t dx = 0; dx <= 1; ++dx) {
              const ptrdiff_t px = bx0 + dx;
              if (px < w) {
                out_pts[np++] = {static_cast<uint8_t>(mc2),
                                 static_cast<uint16_t>(px),
                                 static_cast<uint16_t>(py)};
              }
            }
          }
        }
        *out_np = np;
      };

      auto eval_dc_token_bits = [&](size_t mc_p, size_t ctx, int32_t r_val,
                                    bool allow_new_sym,
                                    double* out_bits) -> bool {
        uint32_t tok = 0;
        uint32_t nb = 0;
        uint32_t eb = 0;
        dc_uint_cfg.Encode(PackSigned(r_val), &tok, &nb, &eb);
        tok = std::min<uint32_t>(tok, 127u);
        const DcSingleHist& ch_h = p2g.dc_chan_hists[mc_p];
        const DcSingleHist& lf_h = p2g.dc_leaf_hists[ctx];
        if (!allow_new_sym && ch_h.hist[tok] == 0) return false;
        const double p_chan =
            (static_cast<double>(ch_h.hist[tok]) + 0.5) /
            (static_cast<double>(ch_h.tot) + 8.0);
        const double p_leaf =
            (static_cast<double>(lf_h.hist[tok]) + 6.0 * p_chan) /
            (static_cast<double>(lf_h.tot) + 6.0);
        const double new_sym_pen = (ch_h.hist[tok] == 0) ? 2.0 : 0.0;
        *out_bits = static_cast<double>(nb) - std::log2(p_leaf) + new_sym_pen;
        return true;
      };

      auto price_dc_step = [&](size_t dg, size_t mc, size_t lx, size_t ly,
                               int32_t dq, bool allow_new_sym,
                               double* out_delta_bits) -> bool {
        Image* dc_img = modular_frame_encoder->VarDCTDCImage(dg);
        Plane<pixel_type>& pl = dc_img->channel[mc].plane;
        const int32_t q_old = pl.ConstRow(ly)[lx];
        const int64_t q_cand64 = static_cast<int64_t>(q_old) + dq;
        if (clamp_i16 && (q_cand64 < -32768 || q_cand64 > 32767)) return false;
        const int32_t q_new = static_cast<int32_t>(q_cand64);

        size_t np = 0;
        collect_dc_stencil(dg, mc, lx, ly, p2g.pts, &np);

        double old_bits = 0.0;
        for (size_t i = 0; i < np; ++i) {
          size_t ctx = 0;
          const int32_t r_val = eval_dc_tree_pixel(
              dg, p2g.pts[i].mc_p, p2g.pts[i].px, p2g.pts[i].py, &ctx);
          double b = 0.0;
          eval_dc_token_bits(p2g.pts[i].mc_p, ctx, r_val,
                             /*allow_new_sym=*/true, &b);
          old_bits += b;
        }

        pl.Row(ly)[lx] = q_new;
        bool ok = true;
        double new_bits = 0.0;
        for (size_t i = 0; i < np; ++i) {
          size_t ctx = 0;
          const int32_t r_val = eval_dc_tree_pixel(
              dg, p2g.pts[i].mc_p, p2g.pts[i].px, p2g.pts[i].py, &ctx);
          double b = 0.0;
          if (!eval_dc_token_bits(p2g.pts[i].mc_p, ctx, r_val, allow_new_sym,
                                  &b)) {
            ok = false;
            break;
          }
          new_bits += b;
        }
        pl.Row(ly)[lx] = q_old;
        if (!ok) return false;
        *out_delta_bits = new_bits - old_bits;
        return true;
      };

      auto commit_dc_hist_step = [&](size_t dg, size_t mc, size_t lx, size_t ly,
                                     int32_t q_new) {
        Plane<pixel_type>& pl =
            modular_frame_encoder->VarDCTDCImage(dg)->channel[mc].plane;
        size_t np = 0;
        collect_dc_stencil(dg, mc, lx, ly, p2g.pts, &np);
        for (size_t i = 0; i < np; ++i) {
          size_t ctx = 0;
          const int32_t r_val = eval_dc_tree_pixel(
              dg, p2g.pts[i].mc_p, p2g.pts[i].px, p2g.pts[i].py, &ctx);
          uint32_t tok = 0;
          uint32_t nb = 0;
          uint32_t eb = 0;
          dc_uint_cfg.Encode(PackSigned(r_val), &tok, &nb, &eb);
          tok = std::min<uint32_t>(tok, 127u);
          if (p2g.dc_leaf_hists[ctx].hist[tok] > 0) {
            p2g.dc_leaf_hists[ctx].hist[tok]--;
          }
          if (p2g.dc_leaf_hists[ctx].tot > 0) p2g.dc_leaf_hists[ctx].tot--;
          if (p2g.dc_chan_hists[p2g.pts[i].mc_p].hist[tok] > 0) {
            p2g.dc_chan_hists[p2g.pts[i].mc_p].hist[tok]--;
          }
          if (p2g.dc_chan_hists[p2g.pts[i].mc_p].tot > 0) {
            p2g.dc_chan_hists[p2g.pts[i].mc_p].tot--;
          }
        }
        pl.Row(ly)[lx] = q_new;
        for (size_t i = 0; i < np; ++i) {
          size_t ctx = 0;
          const int32_t r_val = eval_dc_tree_pixel(
              dg, p2g.pts[i].mc_p, p2g.pts[i].px, p2g.pts[i].py, &ctx);
          uint32_t tok = 0;
          uint32_t nb = 0;
          uint32_t eb = 0;
          dc_uint_cfg.Encode(PackSigned(r_val), &tok, &nb, &eb);
          tok = std::min<uint32_t>(tok, 127u);
          p2g.dc_leaf_hists[ctx].hist[tok]++;
          p2g.dc_leaf_hists[ctx].tot++;
          p2g.dc_chan_hists[p2g.pts[i].mc_p].hist[tok]++;
          p2g.dc_chan_hists[p2g.pts[i].mc_p].tot++;
        }
      };

      // Evaluates or commits the pixel-domain LLF shift induced by changing
      // q(bx, by) on modular channel `mc` by `dq`.
      auto apply_dc_perturbation =
          [&](size_t dg, size_t bx, size_t by, size_t mc, int32_t dq,
              Image3F* target_pre_gab, bool commit_state, Rect* out_px_box,
              float out_new_u[3], float out_new_dec[3][3][3]) -> bool {
        const float mul_deq =
            1.0f /
            static_cast<float>(1u << modular_frame_encoder->extra_dc_precision[dg]);
        float old_u[3];
        for (size_t c = 0; c < 3; ++c) {
          old_u[c] = st.dc_unsmoothed.ConstPlaneRow(c, by)[bx];
          out_new_u[c] = old_u[c];
        }
        if (mc == 0) {
          const float du_y = static_cast<float>(dq) * (dc_factors[1] * mul_deq);
          out_new_u[1] += du_y;
          out_new_u[0] += du_y * cfl_dc[0];
          out_new_u[2] += du_y * cfl_dc[2];
        } else if (mc == 1) {
          out_new_u[0] += static_cast<float>(dq) * (dc_factors[0] * mul_deq);
        } else {
          out_new_u[2] += static_cast<float>(dq) * (dc_factors[2] * mul_deq);
        }
        for (size_t c = 0; c < 3; ++c) {
          st.dc_unsmoothed.PlaneRow(c, by)[bx] = out_new_u[c];
        }

        const size_t sx0 = smooth_dc ? (bx > 0 ? bx - 1 : 0) : bx;
        const size_t sy0 = smooth_dc ? (by > 0 ? by - 1 : 0) : by;
        const size_t sx1 = smooth_dc ? std::min(xsize_blocks, bx + 2) : (bx + 1);
        const size_t sy1 = smooth_dc ? std::min(ysize_blocks, by + 2) : (by + 1);

        memset(p2g.delta_dc, 0, sizeof(p2g.delta_dc));
        uint32_t aff_trs[9];
        size_t num_aff = 0;
        for (size_t sy = sy0; sy < sy1; ++sy) {
          for (size_t sx = sx0; sx < sx1; ++sx) {
            float ndc[3];
            eval_smoothed_dc_at(sx, sy, ndc);
            for (size_t c = 0; c < 3; ++c) {
              out_new_dec[c][sy - sy0][sx - sx0] = ndc[c];
              p2g.delta_dc[c][sy - sy0][sx - sx0] =
                  ndc[c] - shared.dc_storage.ConstPlaneRow(c, sy)[sx];
            }
            const uint32_t code = st.blk_to_tr[sy * xsize_blocks + sx];
            if (code == 0xFFFFFFFFu) continue;
            bool seen = false;
            for (size_t k = 0; k < num_aff; ++k) {
              if (aff_trs[k] == code) {
                seen = true;
                break;
              }
            }
            if (!seen && num_aff < 9) aff_trs[num_aff++] = code;
          }
        }
        if (!commit_state) {
          for (size_t c = 0; c < 3; ++c) {
            st.dc_unsmoothed.PlaneRow(c, by)[bx] = old_u[c];
          }
        }
        if (num_aff == 0) return false;

        size_t px_x0 = xsize;
        size_t px_y0 = ysize;
        size_t px_x1 = 0;
        size_t px_y1 = 0;
        for (size_t k = 0; k < num_aff; ++k) {
          const size_t g = aff_trs[k] >> 16;
          const size_t ti = aff_trs[k] & 0xFFFFu;
          AcGroupBlockCache& cache = group_caches[g];
          const AcTransformCache& tr = cache.transforms[ti];
          const size_t t_bx0 = cache.gx0 + tr.bx;
          const size_t t_by0 = cache.gy0 + tr.by;
          const size_t cw_px = tr.cov_x * kBlockDim;
          const size_t lx0 = tr.bx * kBlockDim;
          const size_t ly0 = tr.by * kBlockDim;
          const size_t lw = std::min(cache.pw_px - lx0, cw_px);
          const size_t lh = std::min(cache.ph_px - ly0, tr.cov_y * kBlockDim);
          const size_t fx0 = cache.px0 + lx0;
          const size_t fy0 = cache.py0 + ly0;
          px_x0 = std::min(px_x0, fx0);
          px_y0 = std::min(px_y0, fy0);
          px_x1 = std::max(px_x1, fx0 + lw);
          px_y1 = std::max(px_y1, fy0 + lh);

          const AcStrategy acs = AcStrategy::FromRawStrategy(tr.acs_raw);
          for (size_t c = 0; c < 3; ++c) {
            memset(p2g.tr_dc_diff, 0, tr.cov_x * tr.cov_y * sizeof(float));
            bool non_zero = false;
            for (size_t sy = std::max(sy0, t_by0);
                 sy < std::min(sy1, t_by0 + tr.cov_y); ++sy) {
              for (size_t sx = std::max(sx0, t_bx0);
                   sx < std::min(sx1, t_bx0 + tr.cov_x); ++sx) {
                const float dv = p2g.delta_dc[c][sy - sy0][sx - sx0];
                if (std::abs(dv) > 1e-9f) {
                  p2g.tr_dc_diff[(sy - t_by0) * tr.cov_x + (sx - t_bx0)] = dv;
                  non_zero = true;
                }
              }
            }
            if (!non_zero) continue;
            HWY_DYNAMIC_DISPATCH(ApplyTransformDcDeltaImpl)(
                acs.Strategy(), tr.cov_x, tr.cov_y, p2g.tr_dc_diff, tr.cov_x,
                dc_dct_buf, dc_px_buf, dc_scratch);
            if (target_pre_gab != nullptr) {
              for (size_t iy = 0; iy < lh; ++iy) {
                float* JXL_RESTRICT dst =
                    target_pre_gab->PlaneRow(c, fy0 + iy) + fx0;
                const float* JXL_RESTRICT dp = dc_px_buf + iy * cw_px;
                for (size_t ix = 0; ix < lw; ++ix) dst[ix] += dp[ix];
              }
            }
            if (commit_state) {
              float* JXL_RESTRICT bp =
                  cache.base_pre_gab[c].data() + ly0 * cache.pw_px + lx0;
              for (size_t iy = 0; iy < lh; ++iy) {
                const float* JXL_RESTRICT dp = dc_px_buf + iy * cw_px;
                for (size_t ix = 0; ix < lw; ++ix) {
                  bp[iy * cache.pw_px + ix] += dp[ix];
                }
              }
            }
          }
        }
        if (commit_state) {
          for (size_t sy = sy0; sy < sy1; ++sy) {
            for (size_t sx = sx0; sx < sx1; ++sx) {
              for (size_t c = 0; c < 3; ++c) {
                shared.dc_storage.PlaneRow(c, sy)[sx] =
                    out_new_dec[c][sy - sy0][sx - sx0];
              }
            }
          }
        }
        if (px_x0 >= px_x1 || px_y0 >= px_y1) return false;
        *out_px_box = Rect(px_x0, px_y0, px_x1 - px_x0, px_y1 - px_y0);
        return true;
      };

      auto eval_exact_4m_dc_move = [&](size_t dg, size_t bx, size_t by,
                                       size_t lx, size_t ly, size_t mc,
                                       int32_t dq, double delta_dc_bits,
                                       bool require_unanimous) -> StatusOr<bool> {
        const double cur_tot =
            std::max(1.0, cur_bits + cur_sh_bits + cum_dc_delta_bits);
        const double new_tot = std::max(
            1.0, cur_bits + cur_sh_bits + cum_dc_delta_bits + delta_dc_bits);
        const double delta_rate = rate_term(new_tot) - rate_term(cur_tot);

        Rect px_box;
        if (!apply_dc_perturbation(dg, bx, by, mc, dq, &cand_pre_gab,
                                   /*commit_state=*/false, &px_box, p2g.new_u,
                                   p2g.new_dec)) {
          return false;
        }
        const size_t lx0 = (px_box.x0() > lf_pad) ? (px_box.x0() - lf_pad) : 0;
        const size_t ly0 = (px_box.y0() > lf_pad) ? (px_box.y0() - lf_pad) : 0;
        const size_t lx1 = std::min(xsize, px_box.x0() + px_box.xsize() + lf_pad);
        const size_t ly1 = std::min(ysize, px_box.y0() + px_box.ysize() + lf_pad);
        const Rect dirty_lin(lx0, ly0, lx1 - lx0, ly1 - ly0);

        const size_t lw = dirty_lin.xsize();
        const size_t lh = dirty_lin.ysize();
        saved_lin.resize(3 * lw * lh);
        for (size_t c = 0; c < 3; ++c) {
          for (size_t y = 0; y < lh; ++y) {
            memcpy(saved_lin.data() + (c * lh + y) * lw,
                   cur_lin.ConstPlaneRow(c, dirty_lin.y0() + y) + dirty_lin.x0(),
                   lw * sizeof(float));
          }
        }
        JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
            memory_manager, lf, shared, cand_pre_gab, feat_ptr, epf_sigma,
            epf_ws, dirty_lin, &cur_lin));
        std::array<double, 4> cand_d = cur_d;
        JXL_RETURN_IF_ERROR(update_4m_checkpointed(cur_lin, dirty_lin, &cand_d));

        double norm_sum = 0.0;
        bool pareto_ok = true;
        for (size_t m = 0; m < 4; ++m) {
          if (weight_m[m] <= 0.0) continue;
          const double nm = (cur_d[m] - cand_d[m]) / slope_m[m];
          norm_sum += weight_m[m] * nm;
          if (require_unanimous || delta_rate < 0.0) {
            if (nm - delta_rate < 0.0) pareto_ok = false;
          } else {
            if (nm < -0.15594302155620929 * std::max(0.006, delta_rate)) {
              pareto_ok = false;
            }
          }
        }
        const double j_gain = norm_sum - delta_rate;
        const double min_gain = (delta_rate < 0.0) ? 0.0005 : 0.0007;

        if (j_gain > min_gain && pareto_ok) {
          const int32_t q_old =
              modular_frame_encoder->VarDCTDCImage(dg)
                  ->channel[mc]
                  .plane.ConstRow(ly)[lx];
          commit_dc_hist_step(dg, mc, lx, ly, q_old + dq);
          Rect unused_box;
          apply_dc_perturbation(dg, bx, by, mc, dq, /*target_pre_gab=*/nullptr,
                                /*commit_state=*/true, &unused_box, p2g.u_tmp,
                                p2g.dec_tmp);
          for (size_t c = 0; c < 3; ++c) {
            for (size_t y = 0; y < px_box.ysize(); ++y) {
              const float* src_r =
                  cand_pre_gab.ConstPlaneRow(c, px_box.y0() + y) + px_box.x0();
              memcpy(cur_pre_gab.PlaneRow(c, px_box.y0() + y) + px_box.x0(),
                     src_r, px_box.xsize() * sizeof(float));
              memcpy(plain_pre_gab.PlaneRow(c, px_box.y0() + y) + px_box.x0(),
                     src_r, px_box.xsize() * sizeof(float));
              memcpy(batch_pre_gab.PlaneRow(c, px_box.y0() + y) + px_box.x0(),
                     src_r, px_box.xsize() * sizeof(float));
            }
          }
          const size_t gx0 = (px_box.x0() > 0) ? (px_box.x0() - 1) : 0;
          const size_t gy0 = (px_box.y0() > 0) ? (px_box.y0() - 1) : 0;
          const size_t gx1 = std::min(xsize, px_box.x0() + px_box.xsize() + 1);
          const size_t gy1 = std::min(ysize, px_box.y0() + px_box.ysize() + 1);
          apply_gab_rect_2g(cur_pre_gab, gx0, gy0, gx1 - gx0, gy1 - gy0,
                            &fine_pre_gab);
          const size_t tsx0 = smooth_dc ? (bx > 0 ? bx - 1 : 0) : bx;
          const size_t tsy0 = smooth_dc ? (by > 0 ? by - 1 : 0) : by;
          const size_t tsx1 =
              smooth_dc ? std::min(xsize_blocks, bx + 2) : (bx + 1);
          const size_t tsy1 =
              smooth_dc ? std::min(ysize_blocks, by + 2) : (by + 1);
          for (size_t tsy = tsy0; tsy < tsy1; ++tsy) {
            for (size_t tsx = tsx0; tsx < tsx1; ++tsx) {
              st.dc_touched_blk[tsy * xsize_blocks + tsx] = 1;
            }
          }
          cum_dc_delta_bits += delta_dc_bits;
          cur_d = cand_d;
          ++dc_moves;
          return true;
        }

        wd->Rollback(st.cp_wd);
        ssim2->Rollback(st.cp_ss2);
        ba3->Rollback(st.cp_ba3);
        cvvdp->Rollback(st.cp_cv);
        for (size_t c = 0; c < 3; ++c) {
          for (size_t y = 0; y < lh; ++y) {
            memcpy(cur_lin.PlaneRow(c, dirty_lin.y0() + y) + dirty_lin.x0(),
                   saved_lin.data() + (c * lh + y) * lw, lw * sizeof(float));
          }
          for (size_t y = 0; y < px_box.ysize(); ++y) {
            memcpy(cand_pre_gab.PlaneRow(c, px_box.y0() + y) + px_box.x0(),
                   cur_pre_gab.ConstPlaneRow(c, px_box.y0() + y) + px_box.x0(),
                   px_box.xsize() * sizeof(float));
          }
        }
        return false;
      };

      accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/false);
      double sum_be_2g = 0.0;
      for (double v : blk_err_cur) sum_be_2g += v;
      const double mean_be_2g =
          sum_be_2g / static_cast<double>(std::max<size_t>(1, num_blocks));

      // Step 2G.1: Unanimous-4-metric DC residual snap bit-reclaim pass.
      for (size_t dg = 0; dg < num_dc_groups; ++dg) {
        const Rect dcr = shared.frame_dim.DCGroupRect(dg);
        for (size_t ly = 0; ly < dcr.ysize(); ++ly) {
          const size_t by = dcr.y0() + ly;
          for (size_t lx = 0; lx < dcr.xsize(); ++lx) {
            const size_t bx = dcr.x0() + lx;
            const double be = blk_err_cur[by * xsize_blocks + bx];
            if (be > 1.35 * mean_be_2g) continue;
            for (size_t mc = 0; mc < 3; ++mc) {
              int8_t best_dq = 0;
              double best_dbits = -1.15;
              for (int32_t dq : {-3, -2, -1, 1, 2, 3}) {
                double dbits = 0.0;
                if (price_dc_step(dg, mc, lx, ly, dq, /*allow_new_sym=*/false,
                                  &dbits) &&
                    dbits < best_dbits) {
                  best_dbits = dbits;
                  best_dq = static_cast<int8_t>(dq);
                }
              }
              if (best_dq != 0) {
                const double sc = (-best_dbits) / (0.001 + be);
                p2g.dc_reclaims.push_back(
                    {sc, static_cast<uint32_t>(dg), static_cast<uint16_t>(bx),
                     static_cast<uint16_t>(by), static_cast<uint8_t>(mc),
                     best_dq});
              }
            }
          }
        }
      }
      if (!p2g.dc_reclaims.empty()) {
        std::sort(p2g.dc_reclaims.begin(), p2g.dc_reclaims.end(),
                  [](const DcReclaimCand& a, const DcReclaimCand& b) {
                    if (a.score != b.score) return a.score > b.score;
                    if (a.by != b.by) return a.by < b.by;
                    if (a.bx != b.bx) return a.bx < b.bx;
                    return a.mc < b.mc;
                  });
        const size_t max_rec_trials = std::min<size_t>(
            p2g.dc_reclaims.size(), std::max<size_t>(96, num_groups * 56));
        for (size_t ri = 0; ri < max_rec_trials; ++ri) {
          const DcReclaimCand& rc = p2g.dc_reclaims[ri];
          const Rect dcr = shared.frame_dim.DCGroupRect(rc.dg);
          const size_t lx = rc.bx - dcr.x0();
          const size_t ly = rc.by - dcr.y0();
          double dbits = 0.0;
          if (!price_dc_step(rc.dg, rc.mc, lx, ly, rc.dq,
                             /*allow_new_sym=*/false, &dbits) ||
              dbits > -1.00) {
            continue;
          }
          JXL_ASSIGN_OR_RETURN(
              bool ok, eval_exact_4m_dc_move(rc.dg, rc.bx, rc.by, lx, ly, rc.mc,
                                             rc.dq, dbits,
                                             /*require_unanimous=*/true));
          (void)ok;
        }
      }

      // Step 2G.2: Salient 8x8 block Y/X/B quantized DC refinement.
      if (dc_moves > 0) {
        accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/false);
      }
      for (size_t dg = 0; dg < num_dc_groups; ++dg) {
        const Rect dcr = shared.frame_dim.DCGroupRect(dg);
        for (size_t ly = 0; ly < dcr.ysize(); ++ly) {
          const size_t by = dcr.y0() + ly;
          for (size_t lx = 0; lx < dcr.xsize(); ++lx) {
            const size_t bx = dcr.x0() + lx;
            const double be = blk_err_cur[by * xsize_blocks + bx];
            if (be > 1.15 * mean_be_2g) {
              p2g.sal_dc_blks.push_back({be, static_cast<uint32_t>(dg),
                                         static_cast<uint16_t>(bx),
                                         static_cast<uint16_t>(by)});
            }
          }
        }
      }
      if (!p2g.sal_dc_blks.empty()) {
        std::sort(p2g.sal_dc_blks.begin(), p2g.sal_dc_blks.end(),
                  [](const SalDcBlk& a, const SalDcBlk& b) {
                    if (a.err != b.err) return a.err > b.err;
                    return (a.by != b.by) ? (a.by < b.by) : (a.bx < b.bx);
                  });
        const size_t max_sal_dc = std::min<size_t>(
            p2g.sal_dc_blks.size(), std::max<size_t>(80, num_groups * 44));
        for (size_t si = 0; si < max_sal_dc; ++si) {
          const SalDcBlk& sb = p2g.sal_dc_blks[si];
          const Rect dcr = shared.frame_dim.DCGroupRect(sb.dg);
          const size_t lx = sb.bx - dcr.x0();
          const size_t ly = sb.by - dcr.y0();

          for (size_t mc = 0; mc < 3; ++mc) {
            int32_t best_dq = 0;
            double best_dbits = 0.0;
            double best_loc_j = 0.0;

            for (int32_t dq : {-3, -2, -1, 1, 2, 3}) {
              double dbits = 0.0;
              if (!price_dc_step(sb.dg, mc, lx, ly, dq, /*allow_new_sym=*/true,
                                 &dbits) ||
                  dbits > 5.5) {
                continue;
              }
              Rect px_box;
              if (!apply_dc_perturbation(sb.dg, sb.bx, sb.by, mc, dq,
                                         &plain_pre_gab,
                                         /*commit_state=*/false, &px_box,
                                         p2g.new_u, p2g.new_dec)) {
                continue;
              }
              const size_t gx0 = (px_box.x0() > 0) ? (px_box.x0() - 1) : 0;
              const size_t gy0 = (px_box.y0() > 0) ? (px_box.y0() - 1) : 0;
              const size_t gx1 =
                  std::min(xsize, px_box.x0() + px_box.xsize() + 1);
              const size_t gy1 =
                  std::min(ysize, px_box.y0() + px_box.ysize() + 1);
              const size_t b_x0 = gx0 / kBlockDim;
              const size_t b_y0 = gy0 / kBlockDim;
              const size_t b_x1 = std::min(xsize_blocks, DivCeil(gx1, kBlockDim));
              const size_t b_y1 = std::min(ysize_blocks, DivCeil(gy1, kBlockDim));

              double cur_loc_err = 0.0;
              for (size_t ey = b_y0; ey < b_y1; ++ey) {
                for (size_t ex = b_x0; ex < b_x1; ++ex) {
                  float bdry_b = 0.0f;
                  cur_loc_err += ComputeLocalBlockEpfError(
                      *enc_state->orig_opsin, fine_pre_gab,
                      enc_state->initial_quant_masking1x1, ey, ex, 1.20f,
                      &bdry_b, nullptr, nullptr);
                  cur_loc_err += kAcsEpfBdryWD * bdry_b;
                }
              }

              apply_gab_rect_2g(plain_pre_gab, gx0, gy0, gx1 - gx0, gy1 - gy0,
                                &fine_pre_gab);
              double cand_loc_err = 0.0;
              for (size_t ey = b_y0; ey < b_y1; ++ey) {
                for (size_t ex = b_x0; ex < b_x1; ++ex) {
                  float bdry_b = 0.0f;
                  cand_loc_err += ComputeLocalBlockEpfError(
                      *enc_state->orig_opsin, fine_pre_gab,
                      enc_state->initial_quant_masking1x1, ey, ex, 1.20f,
                      &bdry_b, nullptr, nullptr);
                  cand_loc_err += kAcsEpfBdryWD * bdry_b;
                }
              }

              // Restore plain_pre_gab and fine_pre_gab.
              for (size_t c = 0; c < 3; ++c) {
                for (size_t y = 0; y < px_box.ysize(); ++y) {
                  memcpy(plain_pre_gab.PlaneRow(c, px_box.y0() + y) + px_box.x0(),
                         cur_pre_gab.ConstPlaneRow(c, px_box.y0() + y) +
                             px_box.x0(),
                         px_box.xsize() * sizeof(float));
                }
              }
              apply_gab_rect_2g(cur_pre_gab, gx0, gy0, gx1 - gx0, gy1 - gy0,
                                &fine_pre_gab);

              const double loc_j =
                  (cur_loc_err - cand_loc_err) -
                  kP2gLocLamMul *
                      static_cast<double>(tcm.bits_to_cost_mul) * dbits;
              if (loc_j > best_loc_j) {
                best_loc_j = loc_j;
                best_dq = dq;
                best_dbits = dbits;
              }
            }

            if (best_dq != 0) {
              JXL_ASSIGN_OR_RETURN(
                  bool ok, eval_exact_4m_dc_move(sb.dg, sb.bx, sb.by, lx, ly,
                                                 mc, best_dq, best_dbits,
                                                 /*require_unanimous=*/false));
              (void)ok;
            }
          }
        }
      }

      if (dc_moves > 0) {
        cur_bits = std::max(1.0, cur_bits + cum_dc_delta_bits);
      }
    }
  }

  // Pass 2H: Coupled post-(CfL + ACS + DC) per-transform AC quantization &
  // RDOQ polish. Re-optimizes transforms whose basis changed in Pass 2F
  // (acs_changed, initialized at cur_rdoq_level == 0) or whose chroma/LLF
  // residuals shifted in Pass 2E (CfL) or Pass 2G (VarDCTDC).
  if ((cfl_moves > 0 || acs_moves > 0 || dc_moves > 0) &&
      eob_scale_s2 > 0.0f && num_blocks >= 64) {
    JXL_RETURN_IF_ERROR(CopyImageTo(cur_pre_gab, &cand_pre_gab));
    JXL_RETURN_IF_ERROR(CopyImageTo(cur_pre_gab, &batch_pre_gab));
    if (acs_moves > 0) {
      for (size_t t = 0; t < num_tiles; ++t) tile_tr_indices[t].clear();
      for (size_t g = 0; g < num_groups; ++g) {
        const auto& trs = group_caches[g].transforms;
        for (size_t i = 0; i < trs.size(); ++i) {
          if (trs[i].tile_idx < num_tiles) {
            tile_tr_indices[trs[i].tile_idx].push_back(i);
          }
        }
      }
    }

    auto is_tr_touched = [&](size_t g, const AcTransformCache& tr) -> bool {
      if (tr.acs_changed) return true;
      if (cfl_moves > 0 && tr.cfl_tile_idx < num_cfl_tiles &&
          !st.cfl_moved[0].empty() && !st.cfl_moved[1].empty() &&
          (st.cfl_moved[0][tr.cfl_tile_idx] != 0 ||
           st.cfl_moved[1][tr.cfl_tile_idx] != 0)) {
        return true;
      }
      if (dc_moves > 0) {
        const Rect bgr = shared.frame_dim.BlockGroupRect(g);
        for (size_t iy = 0; iy < tr.cov_y; ++iy) {
          for (size_t ix = 0; ix < tr.cov_x; ++ix) {
            if (st.dc_touched_blk[(bgr.y0() + tr.by + iy) * xsize_blocks +
                                  (bgr.x0() + tr.bx + ix)] != 0) {
              return true;
            }
          }
        }
      }
      return false;
    };

    struct CandTr2H {
      double score;
      uint32_t tr_idx;
      size_t group;
      int16_t ref_q;
    };

    // Step 2H.0: Unsignal 2D-surrounded sub-tile AQ dips created or exposed by
    // Pass 2F/2G.
    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
    {
      std::vector<CandTr2H> dip_cands;
      for (size_t g = 0; g < num_groups; ++g) {
        const Rect bgr = shared.frame_dim.BlockGroupRect(g);
        const auto& trs = group_caches[g].transforms;
        for (size_t i = 0; i < trs.size(); ++i) {
          const AcTransformCache& tr = trs[i];
          if (!is_tr_touched(g, tr)) continue;
          const int16_t q_tgt = find_2d_dip_target_q(g, tr);
          if (q_tgt <= 0) continue;
          double tr_err = 0.0;
          for (size_t iy = 0; iy < tr.cov_y; ++iy) {
            for (size_t ix = 0; ix < tr.cov_x; ++ix) {
              tr_err += blk_err_cur[(bgr.y0() + tr.by + iy) * xsize_blocks +
                                    (bgr.x0() + tr.bx + ix)];
            }
          }
          const double alpha =
              static_cast<double>(q_tgt) / static_cast<double>(tr.cur_quant_ac);
          const double sc = (1.0 - 1.0 / alpha) + 0.05 / (0.01 + tr_err);
          dip_cands.push_back(
              {sc, static_cast<uint32_t>(i), g, q_tgt});
        }
      }
      if (!dip_cands.empty()) {
        std::sort(dip_cands.begin(), dip_cands.end(),
                  [](const CandTr2H& a, const CandTr2H& b) {
                    if (a.score != b.score) return a.score > b.score;
                    return (a.group != b.group) ? (a.group < b.group)
                                                : (a.tr_idx < b.tr_idx);
                  });
        std::vector<std::vector<Stage2QueueEntry>> queues(num_groups);
        for (const CandTr2H& c : dip_cands) {
          if (queues[c.group].size() < 12) {
            queues[c.group].push_back(
                {c.tr_idx, 0, c.ref_q, Stage2MoveKind::kTrUnsignalDip});
          }
        }
        size_t dip_moves_2h = 0;
        const size_t cap = std::min<size_t>(180, std::max<size_t>(4, num_tiles));
        JXL_RETURN_IF_ERROR(run_stage2_queue(cap, queues, &dip_moves_2h));
        unsignal_moves += dip_moves_2h;
      }
    }

    // Step 2H.1: Un-signaled virtual AQ bit reclaim (kTrRdoqMinus) under
    // unanimous 4-metric agreement on touched transforms.
    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
    {
      std::vector<CandTr2H> rm_cands;
      for (size_t g = 0; g < num_groups; ++g) {
        const Rect bgr = shared.frame_dim.BlockGroupRect(g);
        const auto& trs = group_caches[g].transforms;
        for (size_t i = 0; i < trs.size(); ++i) {
          const AcTransformCache& tr = trs[i];
          if (!is_tr_touched(g, tr)) continue;
          if (tr.cur_rdoq_level > 0 || tr.cur_rdoq_level <= -2) continue;
          if (tr.acs_changed && tr.cur_rdoq_level != 0) continue;
          const float tr_bits = tr.slots[tr.cur_slot].bits;
          if (tr_bits < 4.5f) continue;
          double tr_err = 0.0;
          for (size_t iy = 0; iy < tr.cov_y; ++iy) {
            for (size_t ix = 0; ix < tr.cov_x; ++ix) {
              tr_err += blk_err_cur[(bgr.y0() + tr.by + iy) * xsize_blocks +
                                    (bgr.x0() + tr.bx + ix)];
            }
          }
          const double dbits_est = 0.10 * static_cast<double>(tr_bits);
          const double drate_save =
              rate_term(cur_bits + cur_sh_bits) -
              rate_term(std::max(1.0, cur_bits + cur_sh_bits - dbits_est));
          const double proxy_j = drate_save - 0.50 * tr_err;
          if (proxy_j > 0.0002) {
            rm_cands.push_back({proxy_j, static_cast<uint32_t>(i), g, 0});
          }
        }
      }
      if (!rm_cands.empty()) {
        std::sort(rm_cands.begin(), rm_cands.end(),
                  [](const CandTr2H& a, const CandTr2H& b) {
                    if (a.score != b.score) return a.score > b.score;
                    return (a.group != b.group) ? (a.group < b.group)
                                                : (a.tr_idx < b.tr_idx);
                  });
        std::vector<std::vector<Stage2QueueEntry>> queues(num_groups);
        for (const CandTr2H& c : rm_cands) {
          if (queues[c.group].size() < 16) {
            queues[c.group].push_back(
                {c.tr_idx, 0, 0, Stage2MoveKind::kTrRdoqMinus});
          }
        }
        size_t rm_moves_2h = 0;
        const size_t cap = std::min<size_t>(
            320, std::max<size_t>(4, num_tiles));
        JXL_RETURN_IF_ERROR(run_stage2_queue(cap, queues, &rm_moves_2h));
      }
    }

    // Step 2H.2: Zero-header +1 RDOQ fidelity boost (kTrRdoqPlus) on salient
    // touched edge/contour transforms (e.g. newly split DCT4X4/AFV/anisotropic).
    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
    double sum_be_2h = 0.0;
    for (size_t b = 0; b < num_blocks; ++b) sum_be_2h += blk_err_cur[b];
    const double mean_be_2h =
        sum_be_2h / static_cast<double>(std::max<size_t>(1, num_blocks));
    if (mean_be_2h > 1e-12) {
      std::vector<CandTr2H> rp_cands;
      for (size_t g = 0; g < num_groups; ++g) {
        const Rect bgr = shared.frame_dim.BlockGroupRect(g);
        const auto& trs = group_caches[g].transforms;
        for (size_t i = 0; i < trs.size(); ++i) {
          const AcTransformCache& tr = trs[i];
          if (!is_tr_touched(g, tr)) continue;
          if (tr.cur_rdoq_level != 0) continue;
          const size_t cov = static_cast<size_t>(tr.cov_x) * tr.cov_y;
          if (cov > 4) continue;
          double tr_err = 0.0;
          for (size_t iy = 0; iy < tr.cov_y; ++iy) {
            for (size_t ix = 0; ix < tr.cov_x; ++ix) {
              tr_err += blk_err_cur[(bgr.y0() + tr.by + iy) * xsize_blocks +
                                    (bgr.x0() + tr.bx + ix)];
            }
          }
          if (tr_err <= 1.20 * mean_be_2h * static_cast<double>(cov)) continue;
          const float tr_bits = tr.slots[tr.cur_slot].bits;
          const double dbits_est =
              std::max(1.5, 0.14 * static_cast<double>(tr_bits));
          const double drate = rate_term(cur_bits + cur_sh_bits + dbits_est) -
                               rate_term(cur_bits + cur_sh_bits);
          const double proxy_j = 0.60 * tr_err - drate;
          if (proxy_j > 0.0002) {
            rp_cands.push_back({proxy_j, static_cast<uint32_t>(i), g, 0});
          }
        }
      }
      if (!rp_cands.empty()) {
        std::sort(rp_cands.begin(), rp_cands.end(),
                  [](const CandTr2H& a, const CandTr2H& b) {
                    if (a.score != b.score) return a.score > b.score;
                    return (a.group != b.group) ? (a.group < b.group)
                                                : (a.tr_idx < b.tr_idx);
                  });
        std::vector<std::vector<Stage2QueueEntry>> queues(num_groups);
        for (const CandTr2H& c : rp_cands) {
          if (queues[c.group].size() < 16) {
            queues[c.group].push_back(
                {c.tr_idx, 0, 0, Stage2MoveKind::kTrRdoqPlus});
          }
        }
        size_t rp_moves_2h = 0;
        const size_t cap = std::min<size_t>(
            280, std::max<size_t>(4, num_tiles));
        JXL_RETURN_IF_ERROR(run_stage2_queue(cap, queues, &rp_moves_2h));
      }
    }

    // Step 2H.3: Targeted +1 quant_ac + local epf_sigma shrinkage on top
    // salient touched edge transforms.
    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
    sum_be_2h = 0.0;
    for (size_t b = 0; b < num_blocks; ++b) sum_be_2h += blk_err_cur[b];
    const double mean_be2_2h =
        sum_be_2h / static_cast<double>(std::max<size_t>(1, num_blocks));
    if (mean_be2_2h > 1e-12) {
      std::vector<CandTr2H> qp_cands;
      for (size_t g = 0; g < num_groups; ++g) {
        const Rect bgr = shared.frame_dim.BlockGroupRect(g);
        const auto& trs = group_caches[g].transforms;
        for (size_t i = 0; i < trs.size(); ++i) {
          const AcTransformCache& tr = trs[i];
          if (!is_tr_touched(g, tr)) continue;
          if (tr.cur_quant_ac >= 255 || tr.cur_rdoq_level < 0) continue;
          const size_t cov = static_cast<size_t>(tr.cov_x) * tr.cov_y;
          if (cov > 4) continue;
          double tr_err = 0.0;
          for (size_t iy = 0; iy < tr.cov_y; ++iy) {
            for (size_t ix = 0; ix < tr.cov_x; ++ix) {
              tr_err += blk_err_cur[(bgr.y0() + tr.by + iy) * xsize_blocks +
                                    (bgr.x0() + tr.bx + ix)];
            }
          }
          if (tr_err <= 1.55 * mean_be2_2h * static_cast<double>(cov)) continue;
          const float tr_bits = tr.slots[tr.cur_slot].bits;
          const double dbits_est =
              3.5 + 0.22 * static_cast<double>(std::max(2.0f, tr_bits));
          const double drate = rate_term(cur_bits + cur_sh_bits + dbits_est) -
                               rate_term(cur_bits + cur_sh_bits);
          const double proxy_j = tr_err - drate;
          if (proxy_j > 0.0004) {
            qp_cands.push_back({proxy_j, static_cast<uint32_t>(i), g, 0});
          }
        }
      }
      if (!qp_cands.empty()) {
        std::sort(qp_cands.begin(), qp_cands.end(),
                  [](const CandTr2H& a, const CandTr2H& b) {
                    if (a.score != b.score) return a.score > b.score;
                    return (a.group != b.group) ? (a.group < b.group)
                                                : (a.tr_idx < b.tr_idx);
                  });
        std::vector<std::vector<Stage2QueueEntry>> queues(num_groups);
        for (const CandTr2H& c : qp_cands) {
          if (queues[c.group].size() < 10) {
            queues[c.group].push_back(
                {c.tr_idx, 0, 0, Stage2MoveKind::kTrQfPlus});
          }
        }
        size_t qp_moves_2h = 0;
        const size_t cap = std::min<size_t>(
            180, std::max<size_t>(4, num_tiles));
        JXL_RETURN_IF_ERROR(run_stage2_queue(cap, queues, &qp_moves_2h));
      }
    }
  }

  // Stage 3: Principled 4-metric closed-loop LoopFilter + epf_sharpness RDO.
  bool s3_changed = false;
  const bool allow_lf_rdo =
      (cparams.epf == -1) && (cparams.decoding_speed_tier < 3);
  if ((lf.epf_iters > 0 || allow_lf_rdo) && cparams.decoding_speed_tier < 3) {
    // Part 3A: Jointly compete candidate LoopFilter configs (epf_iters,
    // F16-exact gab_custom) + adaptive AR maps + unimodal uniform {0..7}
    // sharpness levels directly under the 4-metric arbiter + exact header bits.
    if (lf.epf_iters > 0 &&
        (total_moves > 0 || rdoq_moves > 0 || cfl_moves > 0 || acs_moves > 0 ||
         dc_moves > 0)) {
      JXL_RETURN_IF_ERROR(solve_ar_sharpness(
          lf, cur_pre_gab, rdo_quant_field, 1.20f, 1.0f, 1.0f,
          /*allow_const_collapse=*/true, &rdo_sharpness));
      cur_sh_bits =
          EstimateEpfSharpnessBits(rdo_sharpness, xsize_blocks, ysize_blocks);
      epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale,
                            rdo_quant_field, rdo_sharpness, full_blk_rect);
      JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
          memory_manager, lf, shared, cur_pre_gab, feat_ptr, epf_sigma, epf_ws,
          full_px_rect, &cur_lin));
      JXL_RETURN_IF_ERROR(score_full_4m(cur_lin, &cur_d));
      s3_changed = true;
    } else if (lf.epf_iters == 0) {
      FillPlane(static_cast<uint8_t>(0), &rdo_sharpness, Rect(rdo_sharpness));
      cur_sh_bits = 0.0;
    }

    st.anchor_lf = lf;
    st.best_lf = lf;
    const std::array<double, 4> anchor_d = cur_d;
    const double anchor_lf_bits = EstimateLoopFilterBits(st.anchor_lf);
    const double anchor_tot_oh = anchor_lf_bits + cur_sh_bits;
    JXL_RETURN_IF_ERROR(CopyImageTo(rdo_sharpness, &best_cand_sharp));
    double best_global_j = 0.0;
    double best_global_sh_bits = cur_sh_bits;
    bool global_changed = false;

    auto eval_lf_sharpness_cand = [&](const LoopFilter& c_lf,
                                      const ImageB& cand_s,
                                      double* out_j) -> Status {
      const double c_sh_bits =
          (c_lf.epf_iters > 0)
              ? EstimateEpfSharpnessBits(cand_s, xsize_blocks, ysize_blocks)
              : 0.0;
      const double c_tot_oh = EstimateLoopFilterBits(c_lf) + c_sh_bits;
      epf_sigma->UpdateRect(c_lf, shared.ac_strategy, quant_scale,
                            rdo_quant_field, cand_s, full_blk_rect);
      JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
          memory_manager, c_lf, shared, cur_pre_gab, feat_ptr, epf_sigma,
          epf_ws, full_px_rect, &cur_lin));
      std::array<double, 4> c_d = {};
      JXL_RETURN_IF_ERROR(score_full_4m(cur_lin, &c_d));
      const double delta_rate = rate_term(cur_bits + c_tot_oh) -
                                rate_term(cur_bits + anchor_tot_oh);
      double j_gain = 0.0;
      double min_rd_m = 1e30;
      for (size_t m = 0; m < 4; ++m) {
        if (weight_m[m] <= 0.0) continue;
        const double rd_m = (anchor_d[m] - c_d[m]) / slope_m[m] - delta_rate;
        j_gain += weight_m[m] * rd_m;
        min_rd_m = std::min(min_rd_m, rd_m);
      }
      if (out_j != nullptr) *out_j = j_gain;
      const double pareto_tol = -std::min(0.25, 0.35 * std::max(0.0, j_gain));
      if (j_gain > best_global_j + 0.010 && min_rd_m >= pareto_tol) {
        best_global_j = j_gain;
        best_global_sh_bits = c_sh_bits;
        st.best_lf = c_lf;
        JXL_RETURN_IF_ERROR(CopyImageTo(cand_s, &best_cand_sharp));
        global_changed = true;
      }
      return true;
    };

    auto& cand_lfs = st.cand_lfs;
    cand_lfs.clear();
    cand_lfs.push_back(st.anchor_lf);
    if (allow_lf_rdo) {
      const uint32_t max_epf = (cparams.decoding_speed_tier == 2) ? 2u : 3u;
      const float dist = cparams.butteraugli_distance;
      auto push_std_iters = [&](uint32_t iters) {
        if (iters > max_epf) return;
        if (!st.anchor_lf.epf_sigma_custom && st.anchor_lf.epf_iters == iters) {
          return;
        }
        LoopFilter clf = st.anchor_lf;
        clf.epf_iters = iters;
        clf.epf_sigma_custom = false;
        clf.epf_quant_mul = 0.46f;
        clf.epf_pass0_sigma_scale = 0.9f;
        clf.epf_pass2_sigma_scale = 6.5f;
        clf.epf_border_sad_mul = 0.6666666666666666f;
        cand_lfs.push_back(clf);
      };
      if (dist <= 2.2f) push_std_iters(0);
      if (dist >= 0.45f && dist <= 5.0f) push_std_iters(1);
      if (dist >= 1.1f) push_std_iters(2);
      if (dist >= 2.8f) push_std_iters(3);
      if (num_blocks >= 256 && !st.anchor_lf.epf_sigma_custom &&
          dist >= 3.0f && max_epf >= 3) {
        st.base_gab_lf = st.anchor_lf;
        st.base_gab_lf.epf_iters = 3;
        st.base_gab_lf.epf_sigma_custom = true;
        st.base_gab_lf.epf_quant_mul = 0.4375f;
        st.base_gab_lf.epf_pass0_sigma_scale = 3.0f;
        st.base_gab_lf.epf_pass2_sigma_scale = 6.5f;
        st.base_gab_lf.epf_border_sad_mul = 0.5f;
        cand_lfs.push_back(st.base_gab_lf);
      }
    }

    for (const LoopFilter& c_lf : cand_lfs) {
      if (c_lf.epf_iters == 0) {
        FillPlane(static_cast<uint8_t>(0), &cand_sharp, Rect(cand_sharp));
        JXL_RETURN_IF_ERROR(eval_lf_sharpness_cand(c_lf, cand_sharp, nullptr));
        continue;
      }
      JXL_RETURN_IF_ERROR(solve_ar_sharpness(
          c_lf, cur_pre_gab, rdo_quant_field, 1.20f, 1.0f, 1.0f,
          /*allow_const_collapse=*/false, &cand_sharp));
      JXL_RETURN_IF_ERROR(eval_lf_sharpness_cand(c_lf, cand_sharp, nullptr));

      static constexpr uint8_t kCoarseU[3] = {2, 4, 6};
      uint8_t best_u = 4;
      double best_u_j = -1e30;
      if (!allow_lf_rdo) {
        double j0 = 0.0;
        FillPlane(static_cast<uint8_t>(0), &cand_sharp, Rect(cand_sharp));
        JXL_RETURN_IF_ERROR(eval_lf_sharpness_cand(c_lf, cand_sharp, &j0));
        best_u = 0;
        best_u_j = j0;
      }
      for (uint8_t u : kCoarseU) {
        double ju = 0.0;
        FillPlane(u, &cand_sharp, Rect(cand_sharp));
        JXL_RETURN_IF_ERROR(eval_lf_sharpness_cand(c_lf, cand_sharp, &ju));
        if (ju > best_u_j) {
          best_u_j = ju;
          best_u = u;
        }
      }
      if (best_u >= 1) {
        const uint8_t u_lo = best_u - 1;
        if (u_lo >= (allow_lf_rdo ? 1 : 0) && u_lo != 2 && u_lo != 4) {
          FillPlane(u_lo, &cand_sharp, Rect(cand_sharp));
          JXL_RETURN_IF_ERROR(
              eval_lf_sharpness_cand(c_lf, cand_sharp, nullptr));
        }
      }
      if (best_u + 1 <= 7) {
        const uint8_t u_hi = best_u + 1;
        if (u_hi != 2 && u_hi != 4 && u_hi != 6) {
          FillPlane(u_hi, &cand_sharp, Rect(cand_sharp));
          JXL_RETURN_IF_ERROR(
              eval_lf_sharpness_cand(c_lf, cand_sharp, nullptr));
        }
      }
    }

    // Closed-loop F16-exact post-quantization gab_custom de-blurring:
    // counteracts uncompensated decoder Gaborish blur where mid/high-frequency
    // AC pre-sharpening was deadzoned to zero, paying exact +96 header bits.
    if (cparams.gaborish == Override::kDefault && st.best_lf.gab &&
        !st.best_lf.gab_custom && num_blocks >= 256 &&
        cparams.butteraugli_distance >= 1.2f) {
      st.base_gab_lf = st.best_lf;
      JXL_RETURN_IF_ERROR(CopyImageTo(best_cand_sharp, &cand_sharp));

      LoopFilter glf1 = st.base_gab_lf;
      glf1.gab_custom = true;
      glf1.gab_x_weight1 = 112.0f / 1024.0f;
      glf1.gab_x_weight2 = 60.0f / 1024.0f;
      glf1.gab_y_weight1 = 100.0f / 1024.0f;
      glf1.gab_y_weight2 = 53.0f / 1024.0f;
      glf1.gab_b_weight1 = 100.0f / 1024.0f;
      glf1.gab_b_weight2 = 53.0f / 1024.0f;
      JXL_RETURN_IF_ERROR(eval_lf_sharpness_cand(glf1, cand_sharp, nullptr));

      if (cparams.butteraugli_distance >= 2.4f) {
        LoopFilter glf2 = st.base_gab_lf;
        glf2.gab_custom = true;
        glf2.gab_x_weight1 = 108.0f / 1024.0f;
        glf2.gab_x_weight2 = 58.0f / 1024.0f;
        glf2.gab_y_weight1 = 85.0f / 1024.0f;
        glf2.gab_y_weight2 = 45.0f / 1024.0f;
        glf2.gab_b_weight1 = 90.0f / 1024.0f;
        glf2.gab_b_weight2 = 48.0f / 1024.0f;
        JXL_RETURN_IF_ERROR(eval_lf_sharpness_cand(glf2, cand_sharp, nullptr));
      }
    }

    if (global_changed) {
      lf = st.best_lf;
      lf_pad = lf.Padding();
      JXL_RETURN_IF_ERROR(CopyImageTo(best_cand_sharp, &rdo_sharpness));
      cur_sh_bits = best_global_sh_bits;
      s3_changed = true;
    }

    // Sync epf_sigma, cur_lin, and all 4 incremental arbiters to Part 3A winner.
    if (lf.epf_iters > 0) {
      epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale,
                            rdo_quant_field, rdo_sharpness, full_blk_rect);
    }
    JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
        memory_manager, lf, shared, cur_pre_gab, feat_ptr, epf_sigma, epf_ws,
        full_px_rect, &cur_lin));
    JXL_RETURN_IF_ERROR(score_full_4m(cur_lin, &cur_d));

    // Part 3B: Closed-loop dirty-rect per-tile epf_sharpness refinement.
    // Preserves exact EstimateEpfSharpnessBits when already adaptive; when
    // starting from uniform sharpness, explores tile splits under ClampedGradient
    // residual bit pricing and commits ONLY under unanimous 4-metric gain.
    if (lf.epf_iters > 0 && num_tiles >= 4) {
      const bool started_uniform = (cur_sh_bits == 0.0);
      const uint8_t u0_sharp =
          started_uniform ? rdo_sharpness.ConstRow(0)[0] : 0;
      const std::array<double, 4> pre_3b_d = cur_d;
      auto eval_3b_sh_bits = [&](const ImageB& sharp_img,
                                 double* out_raw) -> double {
        const double std_b =
            EstimateEpfSharpnessBits(sharp_img, xsize_blocks, ysize_blocks);
        if (!started_uniform || std_b == 0.0) {
          *out_raw = std_b;
          return std_b;
        }
        size_t grad_cnt[15] = {};
        for (size_t by = 0; by < ysize_blocks; ++by) {
          const uint8_t* row = sharp_img.ConstRow(by);
          const uint8_t* prev = (by > 0) ? sharp_img.ConstRow(by - 1) : nullptr;
          const size_t ly = by & 31u;
          for (size_t bx = 0; bx < xsize_blocks; ++bx) {
            const size_t lx = bx & 31u;
            int32_t gp = 0;
            if (lx == 0 && ly == 0) {
              gp = 0;
            } else if (lx == 0) {
              gp = static_cast<int32_t>(prev[bx] & 7u);
            } else if (ly == 0) {
              gp = static_cast<int32_t>(row[bx - 1] & 7u);
            } else {
              const int32_t gt = static_cast<int32_t>(prev[bx] & 7u);
              const int32_t gl = static_cast<int32_t>(row[bx - 1] & 7u);
              const int32_t gtl = static_cast<int32_t>(prev[bx - 1] & 7u);
              gp = Clamp1(gt + gl - gtl, std::min(gt, gl), std::max(gt, gl));
            }
            const int32_t gr =
                Clamp1(static_cast<int32_t>(row[bx] & 7u) - gp, -7, 7);
            ++grad_cnt[gr + 7];
          }
        }
        double b_grad = 36.0;
        const double inv_n =
            1.0 / static_cast<double>(xsize_blocks * ysize_blocks);
        for (size_t r = 0; r < 15; ++r) {
          if (grad_cnt[r] > 0) {
            b_grad += 1.5 - 1.15 * static_cast<double>(grad_cnt[r]) *
                                std::log2(static_cast<double>(grad_cnt[r]) *
                                          inv_n);
          }
        }
        *out_raw = std::min(std_b, b_grad);
        return std::max(10.0, *out_raw - 96.0);
      };
      double cur_eff_sh_bits = cur_sh_bits;
      size_t accepted_3b = 0;

      const size_t cur_sigma_px_pad = kSigmaPadPerIter[lf.epf_iters];
      const double cur_lf_bits = EstimateLoopFilterBits(lf);
      accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/false);
      tile_err.assign(num_tiles, 0.0);
      for (size_t by = 0; by < ysize_blocks; ++by) {
        const size_t ty = by / kTileBlocks;
        for (size_t bx = 0; bx < xsize_blocks; ++bx) {
          const size_t tx = bx / kTileBlocks;
          tile_err[ty * tiles_x + tx] += blk_err_cur[by * xsize_blocks + bx];
        }
      }
      struct EpfTileRank {
        double err;
        size_t tile;
      };
      std::vector<EpfTileRank> epf_ranks;
      epf_ranks.reserve(num_tiles);
      for (size_t t = 0; t < num_tiles; ++t) {
        if (tile_err[t] > 1e-12) epf_ranks.push_back({tile_err[t], t});
      }
      std::sort(epf_ranks.begin(), epf_ranks.end(),
                [](const EpfTileRank& a, const EpfTileRank& b) {
                  return (a.err != b.err) ? (a.err > b.err) : (a.tile < b.tile);
                });

      const size_t max_epf_tiles =
          std::min<size_t>(started_uniform ? 48 : 64, epf_ranks.size());
      auto& saved_sharp = st.saved_sharp;
      for (size_t ri = 0; ri < max_epf_tiles; ++ri) {
        const size_t t = epf_ranks[ri].tile;
        const size_t tx = t % tiles_x;
        const size_t ty = t / tiles_x;
        const size_t bx0 = tx * kTileBlocks;
        const size_t by0 = ty * kTileBlocks;
        const size_t bw = std::min<size_t>(kTileBlocks, xsize_blocks - bx0);
        const size_t bh = std::min<size_t>(kTileBlocks, ysize_blocks - by0);
        if (bw == 0 || bh == 0) continue;

        saved_sharp.resize(bw * bh);
        size_t sum_s = 0;
        for (size_t iy = 0; iy < bh; ++iy) {
          const uint8_t* srow = rdo_sharpness.ConstRow(by0 + iy) + bx0;
          for (size_t ix = 0; ix < bw; ++ix) {
            const uint8_t sv = srow[ix];
            saved_sharp[iy * bw + ix] = sv;
            sum_s += sv;
          }
        }
        const int mean_s = static_cast<int>(
            (sum_s + (bw * bh) / 2) / std::max<size_t>(1, bw * bh));
        uint8_t tile_s_cands[7] = {};
        size_t num_s_cands = 0;
        auto push_s_cand = [&](int v) {
          const uint8_t uv = static_cast<uint8_t>(Clamp1(v, 0, 7));
          for (size_t k = 0; k < num_s_cands; ++k) {
            if (tile_s_cands[k] == uv) return;
          }
          tile_s_cands[num_s_cands++] = uv;
        };
        push_s_cand(0);
        push_s_cand(std::max(1, mean_s - 2));
        push_s_cand(std::max(1, mean_s - 1));
        push_s_cand(mean_s);
        push_s_cand(std::min(7, mean_s + 1));
        push_s_cand(std::min(7, mean_s + 2));
        if (started_uniform) push_s_cand(7);

        // Include up to 7 blocks top/left so UpdateRect visits IsFirstBlock()
        // of any transform (up to 64x64 = 8x8 blocks) covering [bx0..bx0+bw).
        const size_t sx0 = (bx0 > 7) ? (bx0 - 7) : 0;
        const size_t sy0 = (by0 > 7) ? (by0 - 7) : 0;
        const Rect sigma_upd_rect(sx0, sy0, bx0 + bw - sx0, by0 + bh - sy0);

        const size_t px0 = bx0 * kBlockDim;
        const size_t py0 = by0 * kBlockDim;
        const size_t px1 = std::min(xsize, (bx0 + bw) * kBlockDim);
        const size_t py1 = std::min(ysize, (by0 + bh) * kBlockDim);
        const size_t lx0 =
            (px0 > cur_sigma_px_pad) ? (px0 - cur_sigma_px_pad) : 0;
        const size_t ly0 =
            (py0 > cur_sigma_px_pad) ? (py0 - cur_sigma_px_pad) : 0;
        const size_t lx1 = std::min(xsize, px1 + cur_sigma_px_pad);
        const size_t ly1 = std::min(ysize, py1 + cur_sigma_px_pad);
        if (lx0 >= lx1 || ly0 >= ly1) continue;
        const Rect dirty_lin(lx0, ly0, lx1 - lx0, ly1 - ly0);

        const size_t lw = dirty_lin.xsize();
        const size_t lh = dirty_lin.ysize();
        saved_lin.resize(3 * lw * lh);
        for (size_t c = 0; c < 3; ++c) {
          for (size_t y = 0; y < lh; ++y) {
            memcpy(saved_lin.data() + (c * lh + y) * lw,
                   cur_lin.ConstPlaneRow(c, dirty_lin.y0() + y) + dirty_lin.x0(),
                   lw * sizeof(float));
          }
        }

        bool has_best_s = false;
        uint8_t best_s_val = 0;
        double best_s_j = 0.002;
        double best_s_raw_bits = cur_sh_bits;
        double best_s_eff_bits = cur_eff_sh_bits;

        for (size_t sci = 0; sci < num_s_cands; ++sci) {
          const uint8_t s_val = tile_s_cands[sci];
          bool same_as_cur = true;
          for (size_t k = 0; k < bw * bh; ++k) {
            if (saved_sharp[k] != s_val) {
              same_as_cur = false;
              break;
            }
          }
          if (same_as_cur) continue;

          for (size_t iy = 0; iy < bh; ++iy) {
            uint8_t* srow = rdo_sharpness.Row(by0 + iy) + bx0;
            for (size_t ix = 0; ix < bw; ++ix) srow[ix] = s_val;
          }
          double cand_raw_bits = 0.0;
          const double cand_eff_bits =
              eval_3b_sh_bits(rdo_sharpness, &cand_raw_bits);
          const double delta_rate =
              rate_term(cur_bits + cur_lf_bits + cand_eff_bits) -
              rate_term(cur_bits + cur_lf_bits + cur_eff_sh_bits);

          epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale,
                                rdo_quant_field, rdo_sharpness, sigma_upd_rect);
          JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
              memory_manager, lf, shared, cur_pre_gab, feat_ptr, epf_sigma,
              epf_ws, dirty_lin, &cur_lin));

          std::array<double, 4> cand_d = {};
          JXL_RETURN_IF_ERROR(
              update_4m_checkpointed(cur_lin, dirty_lin, &cand_d));

          wd->Rollback(cp_wd);
          ssim2->Rollback(cp_ss2);
          ba3->Rollback(cp_ba3);
          cvvdp->Rollback(cp_cv);
          for (size_t c = 0; c < 3; ++c) {
            for (size_t y = 0; y < lh; ++y) {
              memcpy(cur_lin.PlaneRow(c, dirty_lin.y0() + y) + dirty_lin.x0(),
                     saved_lin.data() + (c * lh + y) * lw, lw * sizeof(float));
            }
          }

          double j_gain = 0.0;
          double min_rd_m = 1e30;
          for (size_t m = 0; m < 4; ++m) {
            if (weight_m[m] <= 0.0) continue;
            const double nm = (cur_d[m] - cand_d[m]) / slope_m[m];
            const double rd_m = nm - delta_rate;
            j_gain += weight_m[m] * rd_m;
            min_rd_m = std::min(min_rd_m, rd_m);
          }
          const double pareto_tol =
              started_uniform ? 0.0
                              : -std::min(0.020, 0.35 * std::max(0.0, j_gain));
          if (j_gain > best_s_j && min_rd_m >= pareto_tol) {
            has_best_s = true;
            best_s_val = s_val;
            best_s_j = j_gain;
            best_s_raw_bits = cand_raw_bits;
            best_s_eff_bits = cand_eff_bits;
          }
        }

        if (has_best_s) {
          for (size_t iy = 0; iy < bh; ++iy) {
            uint8_t* srow = rdo_sharpness.Row(by0 + iy) + bx0;
            for (size_t ix = 0; ix < bw; ++ix) srow[ix] = best_s_val;
          }
          epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale,
                                rdo_quant_field, rdo_sharpness, sigma_upd_rect);
          JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
              memory_manager, lf, shared, cur_pre_gab, feat_ptr, epf_sigma,
              epf_ws, dirty_lin, &cur_lin));
          JXL_RETURN_IF_ERROR(
              update_4m_checkpointed(cur_lin, dirty_lin, &cur_d));
          cur_sh_bits = best_s_raw_bits;
          cur_eff_sh_bits = best_s_eff_bits;
          ++accepted_3b;
        } else {
          for (size_t iy = 0; iy < bh; ++iy) {
            uint8_t* srow = rdo_sharpness.Row(by0 + iy) + bx0;
            for (size_t ix = 0; ix < bw; ++ix) {
              srow[ix] = saved_sharp[iy * bw + ix];
            }
          }
          epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale,
                                rdo_quant_field, rdo_sharpness, sigma_upd_rect);
        }
      }

      if (started_uniform && accepted_3b > 0 && cur_sh_bits > 0.0) {
        const double tot_drate =
            rate_term(cur_bits + cur_lf_bits + cur_sh_bits) -
            rate_term(cur_bits + cur_lf_bits);
        double tot_j = 0.0;
        double min_tot_rd = 1e30;
        for (size_t m = 0; m < 4; ++m) {
          if (weight_m[m] <= 0.0) continue;
          const double rd_m =
              (pre_3b_d[m] - cur_d[m]) / slope_m[m] - tot_drate;
          tot_j += weight_m[m] * rd_m;
          min_tot_rd = std::min(min_tot_rd, rd_m);
        }
        if (!(tot_j > 0.008 && min_tot_rd >= 0.0)) {
          FillPlane(u0_sharp, &rdo_sharpness, Rect(rdo_sharpness));
          cur_sh_bits = 0.0;
          epf_sigma->UpdateRect(lf, shared.ac_strategy, quant_scale,
                                rdo_quant_field, rdo_sharpness, full_blk_rect);
          JXL_RETURN_IF_ERROR(ReconstructLinearRgb(
              memory_manager, lf, shared, cur_pre_gab, feat_ptr, epf_sigma,
              epf_ws, full_px_rect, &cur_lin));
          JXL_RETURN_IF_ERROR(score_full_4m(cur_lin, &cur_d));
        } else {
          s3_changed = true;
        }
      } else if (accepted_3b > 0) {
        s3_changed = true;
      }
    }
  }

  // Stage 3C: Coupled post-LoopFilter transform-level RDOQ fidelity recovery.
  // Re-adapts per-transform RDOQ +1 levels on salient blocks where Stage 3
  // updated LoopFilter (epf_iters, gab_custom) and epf_sharpness.
  if (s3_changed && eob_scale_s2 > 0.0f && num_blocks >= 64) {
    struct CandTr3C {
      double score;
      uint32_t tr_idx;
      size_t group;
    };
    accumulate_4m_spatial_gain(blk_err_cur.data(), /*weight_by_step1=*/true);
    double sum_be_3c = 0.0;
    for (size_t b = 0; b < num_blocks; ++b) sum_be_3c += blk_err_cur[b];
    const double mean_be_3c =
        sum_be_3c / static_cast<double>(std::max<size_t>(1, num_blocks));
    if (mean_be_3c > 1e-12) {
      std::vector<CandTr3C> rp_cands;
      for (size_t g = 0; g < num_groups; ++g) {
        const Rect bgr = shared.frame_dim.BlockGroupRect(g);
        const auto& trs = group_caches[g].transforms;
        for (size_t i = 0; i < trs.size(); ++i) {
          const AcTransformCache& tr = trs[i];
          if (tr.cur_rdoq_level != 0) continue;
          const size_t cov = static_cast<size_t>(tr.cov_x) * tr.cov_y;
          if (cov > 4) continue;
          double tr_err = 0.0;
          for (size_t iy = 0; iy < tr.cov_y; ++iy) {
            for (size_t ix = 0; ix < tr.cov_x; ++ix) {
              tr_err += blk_err_cur[(bgr.y0() + tr.by + iy) * xsize_blocks +
                                    (bgr.x0() + tr.bx + ix)];
            }
          }
          if (tr_err <= 1.35 * mean_be_3c * static_cast<double>(cov)) continue;
          const float tr_bits = tr.slots[tr.cur_slot].bits;
          const double dbits_est =
              std::max(1.5, 0.14 * static_cast<double>(tr_bits));
          const double drate = rate_term(cur_bits + cur_sh_bits + dbits_est) -
                               rate_term(cur_bits + cur_sh_bits);
          const double proxy_j = 0.60 * tr_err - drate;
          if (proxy_j > 0.0003) {
            rp_cands.push_back({proxy_j, static_cast<uint32_t>(i), g});
          }
        }
      }
      if (!rp_cands.empty()) {
        std::sort(rp_cands.begin(), rp_cands.end(),
                  [](const CandTr3C& a, const CandTr3C& b) {
                    if (a.score != b.score) return a.score > b.score;
                    return (a.group != b.group) ? (a.group < b.group)
                                                : (a.tr_idx < b.tr_idx);
                  });
        std::vector<std::vector<Stage2QueueEntry>> queues(num_groups);
        for (const CandTr3C& c : rp_cands) {
          if (queues[c.group].size() < 12) {
            queues[c.group].push_back(
                {c.tr_idx, 0, 0, Stage2MoveKind::kTrRdoqPlus});
          }
        }
        size_t rp_moves_3c = 0;
        const size_t cap = std::min<size_t>(
            180, std::max<size_t>(4, num_tiles / 2));
        JXL_RETURN_IF_ERROR(run_stage2_queue(cap, queues, &rp_moves_3c));
      }
    }
  }


  if (frame_header.loop_filter.epf_iters == 0) {
    FillPlane(static_cast<uint8_t>(0), &shared.epf_sharpness,
              Rect(shared.epf_sharpness));
  } else {
    JXL_RETURN_IF_ERROR(CopyImageTo(rdo_sharpness, &shared.epf_sharpness));
  }
  enc_state->epf_optimized_in_rdo = true;

  if (total_moves > 0 || acs_moves > 0) {
    for (size_t g = 0; g < num_groups; ++g) {
      const Rect block_group_rect = shared.frame_dim.BlockGroupRect(g);
      for (const AcTransformCache& tr : group_caches[g].transforms) {
        if (tr.cur_quant_ac == tr.base_quant_ac && !tr.acs_changed) continue;
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

  if (rdoq_moves > 0 || acs_moves > 0) {
    JXL_ASSIGN_OR_RETURN(
        enc_state->rdoq_level_field,
        ImageSB::Create(memory_manager, xsize_blocks, ysize_blocks));
    ZeroFillImage(&enc_state->rdoq_level_field);
    for (size_t g = 0; g < num_groups; ++g) {
      const Rect block_group_rect = shared.frame_dim.BlockGroupRect(g);
      for (const AcTransformCache& tr : group_caches[g].transforms) {
        if (tr.cur_rdoq_level == 0) continue;
        block_group_rect.Row(&enc_state->rdoq_level_field, tr.by)[tr.bx] =
            tr.cur_rdoq_level;
      }
    }
  }

  if (cfl_moves > 0) {
    JXL_RETURN_IF_ERROR(CopyImageTo(st.rdo_ytox_map, &shared.cmap.ytox_map));
    JXL_RETURN_IF_ERROR(CopyImageTo(st.rdo_ytob_map, &shared.cmap.ytob_map));
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
