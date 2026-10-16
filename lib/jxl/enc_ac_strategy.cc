// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_ac_strategy.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/common.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/pack_signed.h"
#include "lib/jxl/quant_weights.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_ac_strategy.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/fast_math-inl.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_transforms-inl.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_debug_image.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/quantizer-inl.h"
#include "lib/jxl/simd_util.h"

// Some of the floating point constants in this file and in other
// files in the libjxl project have been obtained using the
// tools/optimizer/simplex_fork.py tool. It is a variation of
// Nelder-Mead optimization, and we generally try to minimize
// BPP * pnorm aggregate as reported by the benchmark_xl tool,
// but occasionally the values are optimized by using additional
// constraints such as maintaining a certain density, or ratio of
// popularity of integral transforms. Jyrki visually reviews all
// such changes and often makes manual changes to maintain good
// visual quality to changes where butteraugli was not sufficiently
// sensitive to some kind of degradation. Unfortunately image quality
// is still more of an art than science.

// Set JXL_DEBUG_AC_STRATEGY to 1 to enable debugging.
#ifndef JXL_DEBUG_AC_STRATEGY
#define JXL_DEBUG_AC_STRATEGY 0
#endif

// This must come before the begin/end_target, but HWY_ONCE is only true
// after that, so use an "include guard".
#ifndef LIB_JXL_ENC_AC_STRATEGY_
#define LIB_JXL_ENC_AC_STRATEGY_
// Parameters of the heuristic are marked with a OPTIMIZE comment.
namespace jxl {
namespace {

 // Debugging utilities.

// Returns a linear sRGB color (as bytes) for each AC strategy.
const uint8_t* TypeColor(uint8_t raw_strategy) {
  JXL_DASSERT(AcStrategy::IsRawStrategyValid(raw_strategy));
  static_assert(AcStrategy::kNumValidStrategies == 27, "Update colors");
  static constexpr uint8_t kColors[AcStrategy::kNumValidStrategies + 1][3] = {
      {0xFF, 0xFF, 0x00},  // DCT8       | yellow
      {0xFF, 0x80, 0x80},  // HORNUSS    | vivid tangerine
      {0xFF, 0x80, 0x80},  // DCT2x2     | vivid tangerine
      {0xFF, 0x80, 0x80},  // DCT4x4     | vivid tangerine
      {0x80, 0xFF, 0x00},  // DCT16x16   | chartreuse
      {0x00, 0xC0, 0x00},  // DCT32x32   | waystone green
      {0xC0, 0xFF, 0x00},  // DCT16x8    | lime
      {0xC0, 0xFF, 0x00},  // DCT8x16    | lime
      {0x00, 0xFF, 0x00},  // DCT32x8    | green
      {0x00, 0xFF, 0x00},  // DCT8x32    | green
      {0x00, 0xFF, 0x00},  // DCT32x16   | green
      {0x00, 0xFF, 0x00},  // DCT16x32   | green
      {0xFF, 0x80, 0x00},  // DCT4x8     | orange juice
      {0xFF, 0x80, 0x00},  // DCT8x4     | orange juice
      {0xFF, 0xFF, 0x80},  // AFV0       | butter
      {0xFF, 0xFF, 0x80},  // AFV1       | butter
      {0xFF, 0xFF, 0x80},  // AFV2       | butter
      {0xFF, 0xFF, 0x80},  // AFV3       | butter
      {0x00, 0xC0, 0xFF},  // DCT64x64   | capri
      {0x00, 0xFF, 0xFF},  // DCT64x32   | aqua
      {0x00, 0xFF, 0xFF},  // DCT32x64   | aqua
      {0x00, 0x40, 0xFF},  // DCT128x128 | rare blue
      {0x00, 0x80, 0xFF},  // DCT128x64  | magic ink
      {0x00, 0x80, 0xFF},  // DCT64x128  | magic ink
      {0x00, 0x00, 0xC0},  // DCT256x256 | keese blue
      {0x00, 0x00, 0xFF},  // DCT256x128 | blue
      {0x00, 0x00, 0xFF},  // DCT128x256 | blue
      {0x00, 0x00, 0x00}   // invalid    | black
  };
  raw_strategy =
      Clamp1<uint8_t>(raw_strategy, 0, AcStrategy::kNumValidStrategies);
  return kColors[raw_strategy];
}

const uint8_t* TypeMask(uint8_t raw_strategy) {
  JXL_DASSERT(AcStrategy::IsRawStrategyValid(raw_strategy));
  static_assert(AcStrategy::kNumValidStrategies == 27, "Update masks");
  // implicitly, first row and column is made dark
  static constexpr uint8_t kMask[AcStrategy::kNumValidStrategies + 1][64] = {
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT8
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 1, 1, 1, 1, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // HORNUSS
      {
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 0, 1, 0, 1, 0, 1, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 0, 1, 0, 1, 0, 1, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 0, 1, 0, 1, 0, 1, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 0, 1, 0, 1, 0, 1, 0,  //
      },                           // 2x2
      {
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
      },                           // 4x4
      {},                          // DCT16x16 (unused)
      {},                          // DCT32x32 (unused)
      {},                          // DCT16x8 (unused)
      {},                          // DCT8x16 (unused)
      {},                          // DCT32x8 (unused)
      {},                          // DCT8x32 (unused)
      {},                          // DCT32x16 (unused)
      {},                          // DCT16x32 (unused)
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT4x8
      {
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
      },                           // DCT8x4
      {
          1, 1, 1, 1, 1, 0, 0, 0,  //
          1, 1, 1, 1, 0, 0, 0, 0,  //
          1, 1, 1, 0, 0, 0, 0, 0,  //
          1, 1, 0, 0, 0, 0, 0, 0,  //
          1, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // AFV0
      {
          0, 0, 0, 0, 1, 1, 1, 1,  //
          0, 0, 0, 0, 0, 1, 1, 1,  //
          0, 0, 0, 0, 0, 0, 1, 1,  //
          0, 0, 0, 0, 0, 0, 0, 1,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // AFV1
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          1, 0, 0, 0, 0, 0, 0, 0,  //
          1, 1, 0, 0, 0, 0, 0, 0,  //
          1, 1, 1, 0, 0, 0, 0, 0,  //
          1, 1, 1, 1, 0, 0, 0, 0,  //
      },                           // AFV2
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 1,  //
          0, 0, 0, 0, 0, 0, 1, 1,  //
          0, 0, 0, 0, 0, 1, 1, 1,  //
      },                           // AFV3
      {}                           // invalid
  };
  raw_strategy =
      Clamp1<uint8_t>(raw_strategy, 0, AcStrategy::kNumValidStrategies);
  return kMask[raw_strategy];
}

Status DumpAcStrategy(const AcStrategyImage& ac_strategy, size_t xsize,
                      size_t ysize, const char* tag, AuxOut* aux_out,
                      const CompressParams& cparams) {
  JxlMemoryManager* memory_manager = ac_strategy.memory_manager();
  JXL_ASSIGN_OR_RETURN(Image3F color_acs,
                       Image3F::Create(memory_manager, xsize, ysize));
  for (size_t y = 0; y < ysize; y++) {
    float* JXL_RESTRICT rows[3] = {
        color_acs.PlaneRow(0, y),
        color_acs.PlaneRow(1, y),
        color_acs.PlaneRow(2, y),
    };
    const AcStrategyRow acs_row = ac_strategy.ConstRow(y / kBlockDim);
    for (size_t x = 0; x < xsize; x++) {
      AcStrategy acs = acs_row[x / kBlockDim];
      const uint8_t* JXL_RESTRICT color = TypeColor(acs.RawStrategy());
      for (size_t c = 0; c < 3; c++) {
        rows[c][x] = color[c] / 255.f;
      }
    }
  }
  size_t stride = color_acs.PixelsPerRow();
  for (size_t c = 0; c < 3; c++) {
    for (size_t by = 0; by < DivCeil(ysize, kBlockDim); by++) {
      float* JXL_RESTRICT row = color_acs.PlaneRow(c, by * kBlockDim);
      const AcStrategyRow acs_row = ac_strategy.ConstRow(by);
      for (size_t bx = 0; bx < DivCeil(xsize, kBlockDim); bx++) {
        AcStrategy acs = acs_row[bx];
        if (!acs.IsFirstBlock()) continue;
        const uint8_t* JXL_RESTRICT color = TypeColor(acs.RawStrategy());
        const uint8_t* JXL_RESTRICT mask = TypeMask(acs.RawStrategy());
        if (acs.covered_blocks_x() == 1 && acs.covered_blocks_y() == 1) {
          for (size_t iy = 0; iy < kBlockDim && by * kBlockDim + iy < ysize;
               iy++) {
            for (size_t ix = 0; ix < kBlockDim && bx * kBlockDim + ix < xsize;
                 ix++) {
              if (mask[iy * kBlockDim + ix]) {
                row[iy * stride + bx * kBlockDim + ix] = color[c] / 800.f;
              }
            }
          }
        }
        // draw block edges
        for (size_t ix = 0; ix < kBlockDim * acs.covered_blocks_x() &&
                            bx * kBlockDim + ix < xsize;
             ix++) {
          row[0 * stride + bx * kBlockDim + ix] = color[c] / 350.f;
        }
        for (size_t iy = 0; iy < kBlockDim * acs.covered_blocks_y() &&
                            by * kBlockDim + iy < ysize;
             iy++) {
          row[iy * stride + bx * kBlockDim + 0] = color[c] / 350.f;
        }
      }
    }
  }
  return DumpImage(cparams, tag, color_acs);
}

}  // namespace
}  // namespace jxl
#endif  // LIB_JXL_ENC_AC_STRATEGY_

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::AbsDiff;
using hwy::HWY_NAMESPACE::ConvertTo;
using hwy::HWY_NAMESPACE::CopySignToAbs;
using hwy::HWY_NAMESPACE::Eq;
using hwy::HWY_NAMESPACE::Floor;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::Gt;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::IfThenZeroElse;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::MaxOfLanes;
using hwy::HWY_NAMESPACE::Min;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Round;
using hwy::HWY_NAMESPACE::Sqrt;
#if HWY_TARGET != HWY_SCALAR
using hwy::HWY_NAMESPACE::Shuffle1032;
using hwy::HWY_NAMESPACE::Shuffle2301;
#endif

Status EstimateEntropy(const AcStrategy& acs, size_t x, size_t y,
                       const ACSConfig& config,
                       const float* JXL_RESTRICT cmap_factors, float* block,
                       float* full_scratch_space, uint32_t* quantized,
                       float best_cost_bound, float& entropy, float& q_out,
                       uint8_t& act_out) {
  entropy = 0.0f;
  float* mem = full_scratch_space;
  float* pixels_buf = full_scratch_space + AcStrategy::kMaxCoeffArea;
  float* scratch_space = full_scratch_space + 2 * AcStrategy::kMaxCoeffArea;
  float* row_buf = full_scratch_space + 3 * AcStrategy::kMaxCoeffArea;
  float* rec_y = full_scratch_space + 4 * AcStrategy::kMaxCoeffArea;
  const size_t size = (1 << acs.log2_covered_blocks()) * kDCTBlockSize;

  const size_t cov_x = acs.covered_blocks_x();
  const size_t cov_y = acs.covered_blocks_y();
  size_t cx = cov_x;
  size_t cy = cov_y;
  CoefficientLayout(&cy, &cx);
  const size_t row_stride = cx * kBlockDim;

  // Apply transform.
  for (size_t c = 0; c < 3; c++) {
    float* JXL_RESTRICT block_c = block + size * c;
    TransformFromPixels(acs.Strategy(), &config.Pixel(c, x, y),
                        config.src_stride, block_c, scratch_space);
  }
  HWY_CAPPED(float, kBlockDim) df;

  const size_t num_blocks = cov_x * cov_y;
  const size_t bx0 = x / 8;
  const size_t by0 = y / 8;
  float quant_norm16 = 0;
  float q_min = std::numeric_limits<float>::max();
  float q_max = 0.0f;
  float q_mean = 0.0f;
  if (num_blocks == 1) {
    quant_norm16 = config.Quant(bx0, by0);
    q_min = q_max = q_mean = quant_norm16;
  } else if (config.band_qf_stride > 0) {
    float q_norm2 = 0.0f;
    for (size_t iy = 0; iy < cov_y; iy++) {
      for (size_t ix = 0; ix < cov_x; ix++) {
        float qval = config.Quant(bx0 + ix, by0 + iy);
        q_min = std::min(q_min, qval);
        q_max = std::max(q_max, qval);
        qval *= qval;
        q_norm2 += qval;
        qval *= qval;
        quant_norm16 += qval * qval;
      }
    }
    quant_norm16 = FastPowf(quant_norm16 / num_blocks, 1.0f / 8.0f);
    q_norm2 = std::sqrt(q_norm2 / num_blocks);
    q_mean = q_norm2;
    quant_norm16 = 0.83f * quant_norm16 + 0.17f * q_norm2;
  } else if (num_blocks == 2) {
    for (size_t iy = 0; iy < cov_y; ++iy) {
      for (size_t ix = 0; ix < cov_x; ++ix) {
        const float qval = config.Quant(bx0 + ix, by0 + iy);
        q_min = std::min(q_min, qval);
        q_max = std::max(q_max, qval);
        q_mean += qval;
      }
    }
    q_mean /= num_blocks;
    quant_norm16 = q_max;
  } else {
    for (size_t iy = 0; iy < cov_y; iy++) {
      for (size_t ix = 0; ix < cov_x; ix++) {
        float qval = config.Quant(bx0 + ix, by0 + iy);
        q_min = std::min(q_min, qval);
        q_max = std::max(q_max, qval);
        q_mean += qval;
        qval *= qval;
        qval *= qval;
        qval *= qval;
        quant_norm16 += qval * qval;
      }
    }
    q_mean /= num_blocks;
    quant_norm16 /= num_blocks;
    quant_norm16 = FastPowf(quant_norm16, 1.0f / 16.0f);
  }
  float q_pool = q_max;
  if (num_blocks >= 4) {
    float mean_max_mixer = 1.0f;
    if (config.butteraugli_distance > 1.54138f) {
      mean_max_mixer = std::max(
          0.0f, 1.0f - (config.butteraugli_distance - 1.54138f) * 0.56391f);
    }
    q_pool = mean_max_mixer * q_max + (1.0f - mean_max_mixer) * q_mean;
  }
  q_out = q_pool;
  act_out = 0;

  const HWY_CAPPED(float, 8) df8;
  const HWY_CAPPED(int32_t, kBlockDim) di;
  const auto zero = Zero(df);
  const auto one = Set(df, 1.0f);

  float* w_row = row_buf;
  float* t0_row = row_buf + AcStrategy::kMaxBlockDim;
  float* cw_row = row_buf + 2 * AcStrategy::kMaxBlockDim;
  float* cs_row = row_buf + 3 * AcStrategy::kMaxBlockDim;

  const bool is_partial_block =
      (kPartialBlockKinds &
       (1u << static_cast<uint32_t>(acs.Strategy()))) != 0;
  const float inv_nblocks = 1.0f / static_cast<float>(num_blocks);
  const bool use_csf_w = (config.band_qf_stride > 0) && !is_partial_block;
  const bool use_q_rdo = (config.quant_scale > 0.0f) && !is_partial_block;
  const AcTokenCostModel* JXL_RESTRICT tcm = config.token_cost_model;
  const bool use_tcm = (tcm != nullptr);

  const float d = config.butteraugli_distance;
  float rate_scale = 1.0f;
  float dist_scale = 1.0f;
  if (!is_partial_block) {
    const float b = 0.5f * static_cast<float>(acs.log2_covered_blocks());
    const float a = 0.5f * static_cast<float>(FloorLog2Nonzero(cx / cy));
    if (b > 0.0f) {
      const float k_d =
          use_csf_w
              ? (-0.372f + 0.042f * Clamp1((d - 0.5f) / 0.35f, 0.0f, 1.0f))
              : -0.40f;
      dist_scale = 1.0f / (1.0f + k_d / (d + 1.4f));
      const float b_hi = std::max(0.0f, b - 2.0f);
      if (use_tcm) {
        const float d_ramp = Clamp1((d - 0.50f) / 0.35f, 0.0f, 1.0f);
        const float hq_mb = 1.0f + 0.18f * (1.0f - d_ramp);
        rate_scale =
            hq_mb * (1.0f + 0.05f * a + 0.25f * b_hi / (0.06f + b_hi));
      } else {
        const float mb_d =
            use_csf_w
                ? (0.985f + 0.015f * Clamp1((d - 0.5f) / 1.0f, 0.0f, 1.0f))
                : 1.0f;
        rate_scale =
            dist_scale * mb_d *
            (1.0f + 0.540f * (b * b) / (0.65f + b * b) + 0.110f * a +
             0.74f * b_hi / (0.06f + b_hi));
      }
    }
  } else {
    const AcStrategyType st = acs.Strategy();
    const float s = (st == AcStrategyType::IDENTITY) ? 3.0f
                    : (st == AcStrategyType::DCT2X2) ? 2.0f
                    : (st == AcStrategyType::DCT4X4) ? 1.0f
                                                     : 0.5f;
    const float a =
        (st == AcStrategyType::DCT4X8 || st == AcStrategyType::DCT8X4) ? 0.5f
                                                                       : 0.0f;
    if (use_tcm) {
      rate_scale = 1.0f + 0.04f * a;
      if (s > 1.0f && d < 5.0f) {
        const float d_ramp = Clamp1((d - 0.50f) / 0.35f, 0.0f, 1.0f);
        const float loc_favor = 0.08f + 0.10f * (1.0f - d_ramp);
        const float w_d = (5.0f - d) / 5.0f;
        rate_scale -= loc_favor * w_d * w_d;
      }
    } else if (s <= 1.0f) {
      const float s2 = s * s;
      rate_scale = 1.0f + (0.002f + 0.348f * s2) * s2 + 0.104f * a;
      if (d > 4.0f) {
        rate_scale += 0.5f * std::min(1.0f, 8.0f / (d - 4.0f));
      }
    } else {
      rate_scale = 0.9556f + 0.11595f * s;
      if (d < 5.0f) {
        const float d_ramp =
            (config.band_qf_stride > 0)
                ? Clamp1((d - 0.5f) / 0.35f, 0.0f, 1.0f)
                : 0.0f;
        const float loc_favor =
            (config.band_qf_stride > 0)
                ? ((0.400f - 0.015f * (3.0f - s)) -
                   (0.050f + 0.075f * (3.0f - s)) * d_ramp)
                : 0.400f;
        const float w_d = (5.0f - d) / 5.0f;
        rate_scale -= loc_favor * w_d * w_d;
      }
    }
  }

  float bq_c[3][4] = {};
  float q_all_c[3] = {quant_norm16, quant_norm16, quant_norm16};
  float band_w_c[3][4];
  float zero_thres_c[3][8];
  for (size_t c = 0; c < 3; ++c) {
    for (size_t s = 0; s < 4; ++s) {
      band_w_c[c][s] = 1.0f;
      zero_thres_c[c][s] = 0.50f;
      zero_thres_c[c][4 + s] = 0.0f;
    }
    if (use_csf_w) {
      for (size_t s = 0; s < 4; ++s) {
        float sq_sum = 0.0f;
        for (size_t iy = 0; iy < cov_y; ++iy) {
          for (size_t ix = 0; ix < cov_x; ++ix) {
            const float v = config.BandQuant(c, s, bx0 + ix, by0 + iy);
            sq_sum += v * v;
          }
        }
        bq_c[c][s] = std::sqrt(sq_sum * inv_nblocks);
      }
      q_all_c[c] = 0.30f * bq_c[c][0] + 0.30f * bq_c[c][1] +
                   0.25f * bq_c[c][2] + 0.15f * bq_c[c][3];
      ComputeBandThresholds(c, acs.Strategy(), cx, cy, q_all_c[c], bq_c[c],
                            zero_thres_c[c]);
      for (size_t s = 0; s < 4; ++s) {
        band_w_c[c][s] =
            Clamp1(1.0f - 2.2f * zero_thres_c[c][4 + s], 0.88f, 1.12f);
      }
    }
  }

  const float inv_npixels =
      1.0f / static_cast<float>(num_blocks * kDCTBlockSize);
  const float q_base = std::max(config.base_quant, 1e-4f);
  const float q_llf_fixed =
      (num_blocks == 1) ? (use_csf_w ? std::max(q_all_c[1], 1e-4f) : q_base)
                        : quant_norm16;

  auto eval_q = [&](float q_cand, bool rdo_mode, float& out_rate,
                    float& out_loss_scalar) {
    out_rate = 0.0f;
    float loss = 0.0f;
    float loss2 = 0.0f;
    const auto quant_ac_v = Set(df, q_cand);
    const auto inv_quant_ac_v = Set(df, 1.0f / std::max(q_cand, 1e-4f));
    const auto q_llf_v = Set(df, q_llf_fixed);
    const uint32_t qf_cand =
        use_tcm ? Clamp1<uint32_t>(
                      static_cast<uint32_t>(
                          std::lround(q_cand * config.inv_global_scale)),
                      1u, 255u)
                : 1u;
    const int ord = kStrategyOrder[acs.RawStrategy()];
    int32_t* JXL_RESTRICT q_coeffs = reinterpret_cast<int32_t*>(pixels_buf);

    for (size_t c : {1, 0, 2}) {
      const float* inv_matrix = config.dequant->InvMatrix(acs.Strategy(), c);
      const float* matrix = config.dequant->Matrix(acs.Strategy(), c);
      const float base_corr = (c == 2) ? 1.0f : 0.0f;
      const auto cmap_res_factor = Set(df, cmap_factors[c] - base_corr);
      auto entropy_v = Zero(df);
      auto nzeros_v = Zero(df);
      const size_t block_ctx =
          use_tcm ? tcm->block_ctx_map.Context(/*dc_idx=*/0, qf_cand, ord, c)
                  : 0;
      const uint32_t nzero_ctx =
          use_tcm ? tcm->block_ctx_map.NonZeroContext(
                        tcm->PredictNzeros(c, bx0, by0), block_ctx)
                  : 0;

      for (size_t v8 = 0; v8 < kBlockDim; ++v8) {
        for (size_t u8 = 0; u8 < kBlockDim; ++u8) {
          const bool is_llf = (v8 == 0 && u8 == 0);
          const uint8_t s = kCoeffBand[v8][u8];
          const float csf =
              use_csf_w ? config.csf_w[c][v8 * kBlockDim + u8] : 1.0f;
          const float ws = band_w_c[c][s] * csf;
          const float dz = is_llf ? 0.0f : zero_thres_c[c][4 + s];
          const float t0 = 0.50f + ((dz >= 0.0f) ? dz : (0.5f * dz));
          float cw =
              (!use_csf_w || is_llf)
                  ? 1.0f
                  : (0.94f + 0.0085f * static_cast<float>(v8 + u8));
          float cs = 0.0f;
          if (rdo_mode && use_tcm) {
            const size_t k_approx = v8 * kBlockDim + u8;
            const size_t k_freq = is_llf ? 0 : kCoeffFreqContext[k_approx];
            cw = is_llf ? 0.0f : tcm->rdo_c_one[block_ctx * 32 + k_freq];
            cs = is_llf ? 0.0f : tcm->rdo_c_slope[block_ctx * 32 + k_freq];
          } else if (!rdo_mode && c == 1) {
            const float quad_y = (v8 < 4 && u8 < 4) ? 0.58f : 0.64f;
            cs = is_llf ? 1e30f : (quad_y + dz);
          }
          for (size_t ix = 0; ix < cx; ++ix) {
            w_row[u8 * cx + ix] = ws;
            t0_row[u8 * cx + ix] = t0;
            cw_row[u8 * cx + ix] = cw;
            cs_row[u8 * cx + ix] = cs;
          }
        }
        for (size_t iy = 0; iy < cy; ++iy) {
          const size_t off = (v8 * cy + iy) * row_stride;
          for (size_t rx = 0; rx < row_stride; rx += Lanes(df)) {
            const auto in = Load(df, block + c * size + off + rx);
            const auto in_y =
                (c == 1) ? zero
                : (c == 0)
                    ? Mul(Load(df, rec_y + off + rx), cmap_res_factor)
                    : MulAdd(Load(df, rec_y + off + rx), cmap_res_factor,
                             Load(df, block + size + off + rx));
            const auto im = Load(df, inv_matrix + off + rx);
            const auto t0 = Load(df, t0_row + rx);
            const auto ws = Load(df, w_row + rx);
            const auto m = Load(df, matrix + off + rx);
            const auto cw = Load(df, cw_row + rx);

            const auto u_coeff = Mul(Sub(in, in_y), im);
            const auto val = Mul(u_coeff, quant_ac_v);
            if (rdo_mode) {
              const auto val0 = Mul(u_coeff, q_llf_v);
              const auto z = Div(Abs(val), t0);
              const auto z2 = Mul(z, z);
              const auto z4 = Mul(z2, z2);
              const auto w_act = Div(z4, Add(one, z4));
              if (c == 1) {
                Store(use_csf_w ? Mul(w_act, in) : in, df, rec_y + off + rx);
              }
              const auto err_denom =
                  Sqrt(Add(one, Mul(Set(df, 1.8f), z2)));
              const auto err = Div(val0, err_denom);
              Store(Mul(m, Mul(ws, err)), df, &mem[off + rx]);

              const auto r_vec =
                  use_tcm ? Mul(w_act, MulAdd(Load(df, cs_row + rx),
                                              Sub(Sqrt(Max(one, Abs(val))),
                                                  one),
                                              cw))
                          : Mul(cw, Mul(w_act, Sqrt(Abs(val))));
              if (v8 == 0 && rx < cx) {
                HWY_ALIGN float r_arr[kBlockDim];
                HWY_ALIGN float w_arr[kBlockDim];
                Store(r_vec, df, r_arr);
                Store(w_act, df, w_arr);
                for (size_t lx = 0; lx < std::min(cx, Lanes(df)); ++lx) {
                  mem[off + rx + lx] = 0.0f;
                  r_arr[lx] = 0.0f;
                  w_arr[lx] = 0.0f;
                }
                entropy_v = Add(entropy_v, Load(df, r_arr));
                nzeros_v = Add(nzeros_v, Load(df, w_arr));
              } else {
                entropy_v = Add(entropy_v, r_vec);
                nzeros_v = Add(nzeros_v, w_act);
              }
            } else {
              const auto rval = IfThenElseZero(Ge(Abs(val), t0), Round(val));
              const auto q_mag = Abs(rval);
              const auto q_is_zero = Eq(q_mag, zero);
              if (c == 1) {
                if (use_csf_w) {
                  const auto q_y_nz = Ge(Abs(val), Load(df, cs_row + rx));
                  const auto adj_rval = IfThenElseZero(
                      q_y_nz,
                      IfThenElse(
                          Eq(q_mag, one),
                          CopySignToAbs(Set(df, kDefaultQuantBias[1]), rval),
                          Sub(rval, Div(Set(df, kDefaultQuantBias[3]), rval))));
                  const auto deq_y = Mul(adj_rval, Mul(m, inv_quant_ac_v));
                  Store(MulAdd(Set(df, 0.65f), deq_y, Mul(Set(df, 0.35f), in)),
                        df, rec_y + off + rx);
                } else {
                  Store(in, df, rec_y + off + rx);
                }
              }
              const auto diff = Mul(ws, Sub(val, rval));
              Store(Mul(m, diff), df, &mem[off + rx]);
              const auto nz_one = IfThenZeroElse(q_is_zero, one);
              if (use_tcm) {
                Store(ConvertTo(di, rval), di, &q_coeffs[off + rx]);
              } else {
                entropy_v = MulAdd(cw, Sqrt(q_mag), entropy_v);
              }
              nzeros_v = Add(nzeros_v, nz_one);
            }
          }
        }
      }

      const float f_nzeros = GetLane(SumOfLanes(df, nzeros_v));
      float entropy_c = 0.0f;
      if (!rdo_mode && use_tcm) {
        const int32_t nzeros = static_cast<int32_t>(f_nzeros);
        float bits_c = tcm->ValBits(tcm->cluster_of_ctx[nzero_ctx],
                                    static_cast<uint32_t>(nzeros));
        if (nzeros > 0) {
          const size_t log2_blocks = acs.log2_covered_blocks();
          const size_t histo_offset =
              tcm->block_ctx_map.ZeroDensityContextsOffset(block_ctx);
          const coeff_order_t* JXL_RESTRICT order =
              &tcm->coeff_orders[CoeffOrderOffset(ord, c)];
          size_t prev =
              (nzeros > static_cast<int32_t>(size / 16)) ? 0 : 1;
          int32_t nzeros_left = nzeros;
          for (size_t k = num_blocks; k < size && nzeros_left != 0; ++k) {
            const int32_t coeff = q_coeffs[order[k]];
            const size_t ctx =
                histo_offset + ZeroDensityContext(nzeros_left, k, num_blocks,
                                                  log2_blocks, prev);
            const uint32_t u_coeff = PackSigned(coeff);
            bits_c += tcm->ValBits(tcm->cluster_of_ctx[ctx], u_coeff);
            const int32_t is_nz = (coeff != 0) ? 1 : 0;
            prev = static_cast<size_t>(is_nz);
            nzeros_left -= is_nz;
          }
        }
        entropy_c = tcm->bits_to_cost_mul * bits_c;
      }

      float masku_lut[3] = {12.0f, rdo_mode ? 16.0f : 0.0f, 4.0f};
      auto masku_off = Set(df8, masku_lut[c]);
      TransformToPixels(acs.Strategy(), &mem[0], pixels_buf, cov_x * 8,
                        scratch_space);

      static const float kChannelMul[3] = {
          static_cast<float>(pow(8.2, 8.0)),
          static_cast<float>(pow(1.0, 8.0)),
          static_cast<float>(pow(1.03, 8.0)),
      };
      static const float kChannelMul2[3] = {
          2.2f * 2.2f,
          1.0f,
          0.90f * 0.90f,
      };

      auto lossc = Zero(df8);
      auto lossc2 = Zero(df8);
      for (size_t iy = 0; iy < cov_y; iy++) {
        for (size_t ix = 0; ix < cov_x; ix++) {
          for (size_t dy = 0; dy < kBlockDim; dy += 2) {
            for (size_t dx = 0; dx < kBlockDim; dx += Lanes(df8)) {
              if (x + ix * 8 + dx + Lanes(df8) <= config.mask1x1_xsize) {
                const size_t p_off0 =
                    (iy * kBlockDim + dy) * (cov_x * kBlockDim) +
                    ix * kBlockDim + dx;
                const size_t p_off1 = p_off0 + cov_x * kBlockDim;
                auto masku0 =
                    Add(Load(df8, config.MaskingPtr1x1(x + ix * 8 + dx,
                                                       y + iy * 8 + dy)),
                        masku_off);
                auto masku1 =
                    Add(Load(df8, config.MaskingPtr1x1(x + ix * 8 + dx,
                                                       y + iy * 8 + dy + 1)),
                        masku_off);
                auto in0 = Mul(masku0, Load(df8, pixels_buf + p_off0));
                auto in1 = Mul(masku1, Load(df8, pixels_buf + p_off1));
#if HWY_TARGET != HWY_SCALAR
                if (use_csf_w) {
                  const auto v_sum = Add(in0, in1);
                  const auto h2_sum = Add(v_sum, Shuffle2301(v_sum));
                  const auto h4_sum = Add(h2_sum, Shuffle1032(h2_sum));
                  const auto s_ms =
                      MulAdd(Set(df8, 0.010f), h2_sum,
                             Mul(Set(df8, 0.0015f), h4_sum));
                  in0 = MulAdd(Set(df8, 0.948f), in0, s_ms);
                  in1 = MulAdd(Set(df8, 0.948f), in1, s_ms);
                  lossc2 = MulAdd(Set(df8, 10.0f), Mul(s_ms, s_ms), lossc2);
                }
#endif
                in0 = Mul(in0, in0);
                in1 = Mul(in1, in1);
                lossc2 = Add(lossc2, Add(in0, in1));
                in0 = Mul(in0, in0);
                in1 = Mul(in1, in1);
                in0 = Mul(in0, in0);
                in1 = Mul(in1, in1);
                lossc = Add(lossc, Add(in0, in1));
              }
            }
          }
        }
      }
      float w = 1.0f;
      if (c == 0 && num_blocks >= 2) {
        const float x_slope =
            use_csf_w
                ? (0.088f +
                   0.014f * Clamp1((config.butteraugli_distance - 0.5f) / 1.5f,
                                   0.0f, 1.0f))
                : 0.125f;
        w += std::min(2.6f, static_cast<float>(num_blocks) * x_slope);
      }
      loss += GetLane(SumOfLanes(df8, lossc)) * (kChannelMul[c] * w);
      loss2 += GetLane(SumOfLanes(df8, lossc2)) *
               (kChannelMul2[c] * ((w > 1.0f) ? std::pow(w, 0.25f) : 1.0f));

      if (rdo_mode && use_tcm) {
        const float dof_norm =
            1.0f / std::sqrt(1.0f + 0.35f * f_nzeros * inv_nblocks);
        const float nz_per_block = Clamp1(f_nzeros * inv_nblocks, 0.0f, 64.0f);
        const size_t nz_lo = static_cast<size_t>(nz_per_block);
        const size_t nz_hi = std::min<size_t>(nz_lo + 1, 64);
        const float nz_frac = nz_per_block - static_cast<float>(nz_lo);
        const float nz_bits =
            (1.0f - nz_frac) * tcm->rdo_nzero_bits[nzero_ctx * 65 + nz_lo] +
            nz_frac * tcm->rdo_nzero_bits[nzero_ctx * 65 + nz_hi];
        const float coeff_bits =
            dof_norm * GetLane(SumOfLanes(df, entropy_v));
        entropy_c = tcm->bits_to_cost_mul * (coeff_bits + nz_bits);
      } else if (!use_tcm) {
        entropy_c = config.cost_delta * GetLane(SumOfLanes(df, entropy_v));
        if (rdo_mode) {
          const float dof_norm =
              1.0f / std::sqrt(1.0f + 0.35f * f_nzeros * inv_nblocks);
          entropy_c *= dof_norm;
          const float f_nbits = std::log2(f_nzeros + 2.0f) - 1.0f;
          entropy_c += 0.45f * config.zeros_mul * f_nbits;
        } else {
          size_t num_nzeros = static_cast<size_t>(f_nzeros);
          size_t nbits = CeilLog2Nonzero(num_nzeros + 1) + 1;
          entropy_c += config.zeros_mul * (CeilLog2Nonzero(nbits + 17) + nbits);
        }
        if (c == 0 && num_blocks >= 2) {
          const float x_slope =
              use_csf_w
                  ? (0.088f +
                     0.014f *
                         Clamp1((config.butteraugli_distance - 0.5f) / 1.5f,
                                0.0f, 1.0f))
                  : 0.125f;
          entropy_c *=
              1.0f + std::min(2.6f, static_cast<float>(num_blocks) * x_slope);
        }
      }
      out_rate += entropy_c;
    }

    if (!rdo_mode && use_tcm) {
      out_rate += tcm->bits_to_cost_mul *
                  (tcm->acs_strategy_bits[acs.RawStrategy()] +
                   tcm->qf_block_bits);
    }

    const float d8 = std::pow(loss * inv_npixels, 1.0f / 8.0f);
    const float d2 = std::sqrt(loss2 * inv_npixels);
    const float d_blend =
        (config.band_qf_stride > 0) ? (0.91f * d8 + 0.09f * 2.18f * d2) : d8;
    const float q_norm =
        rdo_mode ? q_llf_fixed : std::max(q_cand, 1e-4f);
    out_loss_scalar = d_blend * (num_blocks * kDCTBlockSize) / q_norm;
  };

  const float d_rdo_norm = std::pow(std::max(d, 0.35f), -0.38f);
  const float q_min_floor =
      std::max(3.5f * config.quant_scale, 0.34f * q_base);
  const float q_max_ceil = std::min(
      static_cast<float>(Quantizer::kQuantMax - 1) * config.quant_scale,
      1.65f * q_base);

  if (use_q_rdo && num_blocks == 1) {
    const float q_0 = Clamp1(use_csf_w ? q_all_c[1] : q_base, q_min_floor,
                             q_max_ceil);
    const float step = std::max(0.5f * config.quant_scale, 0.14f * q_0);
    const float q_minus = std::max(q_min_floor, q_0 - step);
    const float q_plus = std::min(q_max_ceil, q_0 + step);
    const float lambda_q =
        3.2f * config.info_loss_multiplier * d_rdo_norm;
    const float lambda_reg = 4.2f;

    float r_0 = 0.0f;
    float l_0 = 0.0f;
    eval_q(q_0, /*rdo_mode=*/true, r_0, l_0);
    const float j0 = r_0 + lambda_q * l_0;

    auto score_q1x1 = [&](float qc) -> float {
      float r_cand = 0.0f;
      float l_cand = 0.0f;
      eval_q(qc, /*rdo_mode=*/true, r_cand, l_cand);
      const float dq = (qc - q_0) / std::max(step, 1e-4f);
      return r_cand + lambda_q * l_cand + lambda_reg * dq * dq;
    };

    const float jm =
        (q_minus < q_0 - 1e-5f) ? score_q1x1(q_minus) : (j0 + lambda_reg);
    const float jp =
        (q_plus > q_0 + 1e-5f) ? score_q1x1(q_plus) : (j0 + lambda_reg);
    const float d2_j = jm - 2.0f * j0 + jp;
    float delta = 0.0f;
    if (d2_j > 1e-4f) {
      delta = Clamp1((jm - jp) / (2.0f * d2_j), -1.0f, 1.0f);
    } else if (jm < jp && jm < j0) {
      delta = -0.8f;
    } else if (jp < jm && jp < j0) {
      delta = 0.8f;
    }
    const float q_rdo = Clamp1(q_0 + 0.50f * delta * step, q_min_floor,
                               q_max_ceil);
    quant_norm16 = q_rdo;
    config.MutableQuant(bx0, by0) = q_rdo;
    if (config.band_qf_stride > 0) {
      const float ratio = q_rdo / std::max(q_all_c[1], 1e-4f);
      for (size_t c = 0; c < 3; ++c) {
        for (size_t s = 0; s < 4; ++s) {
          if (config.band_qf_rows[c][s] != nullptr) {
            config.band_qf_rows[c][s][by0 * config.band_qf_stride + bx0] *=
                ratio;
          }
        }
      }
    }
    const float nonflat_w =
        Clamp1(r_0 / (12.0f * config.cost_delta * 0.1f), 0.0f, 1.0f);
    const float dct8_d_gate =
        Clamp1(1.02f - 0.48f * (d - 0.50f), 0.12f, 1.02f);
    const float dct8_step =
        nonflat_w * dct8_d_gate * Clamp1(1.0f + 0.45f * delta, 0.25f, 1.45f);
    q_out = Clamp1(q_rdo + dct8_step * config.quant_scale, q_min_floor,
                   q_max_ceil);
  }

  float base_rate = 0.0f;
  float base_loss_scalar = 0.0f;
  eval_q(quant_norm16, /*rdo_mode=*/false, base_rate, base_loss_scalar);
  entropy = rate_scale * base_rate +
            dist_scale * config.info_loss_multiplier * base_loss_scalar;
  if (config.band_qf_stride > 0 && num_blocks >= 2 && q_min > 1e-4f) {
    const float q_ratio = q_max / q_min;
    if (num_blocks <= 16 && q_ratio < 1.12f) {
      entropy *= 1.0f - 0.011f * ((1.12f - q_ratio) / 0.12f);
    } else if (q_ratio > 1.22f) {
      entropy *= 1.0f + 0.014f * Clamp1((q_ratio - 1.22f) / 0.35f, 0.0f, 1.0f);
    }
  }

  if (use_q_rdo && num_blocks >= 2 && entropy <= best_cost_bound) {
    const float q_0 = Clamp1(q_pool, q_min_floor, q_max_ceil);
    const float step = std::max(0.5f * config.quant_scale, 0.14f * q_0);
    const float q_minus = std::max(q_min_floor, q_0 - step);
    const float q_plus = std::min(q_max_ceil, q_0 + step);
    const float lambda_q_mb =
        3.5f * dist_scale * config.info_loss_multiplier * d_rdo_norm;
    const float lambda_reg_mb = 4.2f * static_cast<float>(num_blocks);

    float r_0_mb = 0.0f;
    float l_0_mb = 0.0f;
    eval_q(q_0, /*rdo_mode=*/true, r_0_mb, l_0_mb);
    const float j0 = r_0_mb + lambda_q_mb * l_0_mb;

    auto score_q_mb = [&](float qc) -> float {
      float r_cand = 0.0f;
      float l_cand = 0.0f;
      eval_q(qc, /*rdo_mode=*/true, r_cand, l_cand);
      const float dq = (qc - q_0) / std::max(step, 1e-4f);
      return r_cand + lambda_q_mb * l_cand + lambda_reg_mb * dq * dq;
    };

    const float jm =
        (q_minus < q_0 - 1e-5f) ? score_q_mb(q_minus) : (j0 + lambda_reg_mb);
    const float jp =
        (q_plus > q_0 + 1e-5f) ? score_q_mb(q_plus) : (j0 + lambda_reg_mb);
    const float d2_j = jm - 2.0f * j0 + jp;
    float delta_mb = 0.0f;
    if (d2_j > 1e-4f) {
      delta_mb = Clamp1((jm - jp) / (2.0f * d2_j), -1.0f, 1.0f);
    } else if (jm < jp && jm < j0) {
      delta_mb = -0.8f;
    } else if (jp < jm && jp < j0) {
      delta_mb = 0.8f;
    }
    const float nonflat_mb = Clamp1(
        (r_0_mb * inv_nblocks) / (8.0f * config.cost_delta * 0.1f), 0.0f, 1.0f);
    const float mb_d_gate =
        Clamp1(1.12f - 0.30f * (d - 0.50f), 0.48f, 1.12f);
    const float mb_base_steps = (num_blocks >= 16)  ? 1.35f
                                : (num_blocks >= 8) ? 1.25f
                                : (num_blocks >= 4) ? 1.15f
                                                    : 0.85f;
    const float mb_steps = nonflat_mb * mb_d_gate * mb_base_steps *
                           Clamp1(1.0f + 0.50f * delta_mb, 0.25f, 1.65f);
    q_out = Clamp1(q_0 + 0.40f * delta_mb * step + mb_steps * config.quant_scale,
                   q_min_floor, q_max_ceil);
  }

  if (use_q_rdo && entropy <= best_cost_bound) {
    // Compute 4-quadrant Luma (c = 1) non-zero activity at q_out for
    // ComputeBlockThresholds's high-frequency deadzone adaptation.
    const float* inv_matrix_y = config.dequant->InvMatrix(acs.Strategy(), 1);
    const auto q_y_v = Set(df, q_out * config.qm_multiplier[1]);
    auto quad_v0 = Zero(df);
    auto quad_v1 = Zero(df);
    auto quad_v2 = Zero(df);
    auto quad_v3 = Zero(df);
    HWY_ALIGN float quad_row8[2][8] = {};
    for (size_t v8 = 0; v8 < kBlockDim; ++v8) {
      const bool v8_hi = (v8 >= 4);
      for (size_t u8 = 0; u8 < kBlockDim; ++u8) {
        const bool is_llf = (v8 == 0 && u8 == 0);
        const uint8_t s = kCoeffBand[v8][u8];
        const size_t hfix =
            (static_cast<size_t>(v8_hi) * 2 + static_cast<size_t>(u8 >= 4));
        const float t0_q =
            is_llf ? 1e30f : (zero_thres_c[1][hfix] + zero_thres_c[1][4 + s]);
        for (size_t ix = 0; ix < cx; ++ix) {
          t0_row[u8 * cx + ix] = t0_q;
        }
      }
      for (size_t iy = 0; iy < cy; ++iy) {
        const size_t off = (v8 * cy + iy) * row_stride;
        for (size_t rx = 0; rx < row_stride; rx += Lanes(df)) {
          const auto in = Load(df, block + size + off + rx);
          const auto im = Load(df, inv_matrix_y + off + rx);
          const auto v0 = Abs(Mul(in, Mul(im, q_y_v)));
          const auto t0_q = Load(df, t0_row + rx);
          const auto q_nz = IfThenElseZero(Ge(v0, t0_q), Round(v0));
          if (cx == 1) {
            const auto prev_q = Load(df, quad_row8[v8_hi] + rx);
            Store(Add(prev_q, q_nz), df, quad_row8[v8_hi] + rx);
          } else if (rx < cx * 4) {
            if (!v8_hi) {
              quad_v0 = Add(quad_v0, q_nz);
            } else {
              quad_v2 = Add(quad_v2, q_nz);
            }
          } else {
            if (!v8_hi) {
              quad_v1 = Add(quad_v1, q_nz);
            } else {
              quad_v3 = Add(quad_v3, q_nz);
            }
          }
        }
      }
    }
    float hfNonZeros[4] = {};
    if (cx == 1) {
      hfNonZeros[0] =
          quad_row8[0][0] + quad_row8[0][1] + quad_row8[0][2] + quad_row8[0][3];
      hfNonZeros[1] =
          quad_row8[0][4] + quad_row8[0][5] + quad_row8[0][6] + quad_row8[0][7];
      hfNonZeros[2] =
          quad_row8[1][0] + quad_row8[1][1] + quad_row8[1][2] + quad_row8[1][3];
      hfNonZeros[3] =
          quad_row8[1][4] + quad_row8[1][5] + quad_row8[1][6] + quad_row8[1][7];
    } else {
      hfNonZeros[0] = GetLane(SumOfLanes(df, quad_v0));
      hfNonZeros[1] = GetLane(SumOfLanes(df, quad_v1));
      hfNonZeros[2] = GetLane(SumOfLanes(df, quad_v2));
      hfNonZeros[3] = GetLane(SumOfLanes(df, quad_v3));
    }
    const int32_t div = static_cast<int32_t>(num_blocks);
    const float min_hf_non_zeros =
        std::min({hfNonZeros[0], hfNonZeros[1], hfNonZeros[2], hfNonZeros[3]});
    int32_t activity = 15;
    if (min_hf_non_zeros < 15.0f * div) {
      activity = (static_cast<int32_t>(min_hf_non_zeros) + div / 2) / div;
    }
    const uint8_t q0_sub =
        (activity <= 2 && hfNonZeros[0] >= 2.0f * div) ? 1u : 0u;
    act_out = static_cast<uint8_t>((activity << 1) | q0_sub);
  }
  return true;
}

Status ProcessRectACS(const CompressParams& cparams, const ACSConfig& config,
                      const Rect& rect, const ColorCorrelationMap& cmap,
                      float* JXL_RESTRICT block,
                      uint32_t* JXL_RESTRICT quantized,
                      AcStrategyImage* ac_strategy) {
  float* JXL_RESTRICT scratch_space = block + 3 * AcStrategy::kMaxCoeffArea;
  const size_t bx = rect.x0();
  const size_t by = rect.y0();
  const size_t rx = rect.xsize();
  const size_t ry = rect.ysize();
  JXL_ENSURE(rx <= 32);
  JXL_ENSURE(ry <= 32);
  if (cparams.speed_tier > SpeedTier::kHare) return true;

  struct CandBlock {
    uint8_t x0, y0, w, h;
    AcStrategyType type;
    uint8_t act;
    float gain;
    float rate;
    float q;
  };
  struct LocalCand {
    uint64_t mask;
    float gain;
    float rate;
    float q;
    uint8_t x0, y0;
    AcStrategyType type;
    uint8_t act;
  };
  struct DPTable {
    float cost[528][528];
    uint16_t split[528][528];
    AcStrategyType leaf[32][32][9][9];
    uint8_t leaf_act[32][32][9][9];
    float leaf_q[32][32][9][9];
    AcStrategyType best_8x8_type[32][32];
    uint8_t best_8x8_act[32][32];
    float best_8x8_q[32][32];
    float entropy_estimate[32][32];
    float prefix_sum[33][33];
    CandBlock cands[6144];
    LocalCand local_cands[352];
  };
  static constexpr size_t kQMemWords = 655360;
  static_assert(sizeof(DPTable) <= kQMemWords * sizeof(uint32_t),
                "DPTable fits in per-thread quantized buffer");
  DPTable* JXL_RESTRICT dp = reinterpret_cast<DPTable*>(quantized);

  // Precompute CfL factors for each 8x8 color tile overlapping this AC group.
  float cmap_factors_tile[4][4][3];
  for (size_t ty_off = 0; ty_off < DivCeil(ry, kColorTileDimInBlocks);
       ++ty_off) {
    const size_t ty = by / kColorTileDimInBlocks + ty_off;
    for (size_t tx_off = 0; tx_off < DivCeil(rx, kColorTileDimInBlocks);
         ++tx_off) {
      const size_t tx = bx / kColorTileDimInBlocks + tx_off;
      cmap_factors_tile[ty_off][tx_off][0] =
          cmap.base().YtoXRatio(cmap.ytox_map.ConstRow(ty)[tx]);
      cmap_factors_tile[ty_off][tx_off][1] = 0.0f;
      cmap_factors_tile[ty_off][tx_off][2] =
          cmap.base().YtoBRatio(cmap.ytob_map.ConstRow(ty)[tx]);
    }
  }

  const bool enable_rdo = (config.band_qf_stride > 0);
  struct Transform1x1 {
    AcStrategyType type;
    int max_speed_tier;
  };
  static constexpr Transform1x1 kTransforms1x1[] = {
      {AcStrategyType::DCT, 9},      {AcStrategyType::DCT4X4, 5},
      {AcStrategyType::DCT2X2, 5},   {AcStrategyType::DCT4X8, 4},
      {AcStrategyType::DCT8X4, 4},   {AcStrategyType::IDENTITY, 5},
      {AcStrategyType::AFV0, 4},     {AcStrategyType::AFV1, 4},
      {AcStrategyType::AFV2, 4},     {AcStrategyType::AFV3, 4},
  };
  const int speed_tier = static_cast<int>(cparams.speed_tier);
  for (size_t iy = 0; iy <= ry; ++iy) {
    dp->prefix_sum[iy][0] = 0.0f;
  }
  for (size_t ix = 0; ix <= rx; ++ix) {
    dp->prefix_sum[0][ix] = 0.0f;
  }
  for (size_t iy = 0; iy < ry; iy++) {
    float row_sum = 0.0f;
    for (size_t ix = 0; ix < rx; ix++) {
      float best_entropy = 1e30f;
      AcStrategyType best_of_8x8s = AcStrategyType::DCT;
      float best_q = config.Quant(bx + ix, by + iy);
      uint8_t best_act = 0;
      const float* JXL_RESTRICT cmap_factors =
          cmap_factors_tile[iy / kColorTileDimInBlocks]
                           [ix / kColorTileDimInBlocks];
      for (const auto& tx : kTransforms1x1) {
        if (tx.max_speed_tier < speed_tier) continue;
        float cand_entropy = 1e30f;
        float cand_q = 0.0f;
        uint8_t cand_act = 0;
        JXL_RETURN_IF_ERROR(EstimateEntropy(
            AcStrategy::FromRawStrategy(tx.type), 8 * (bx + ix), 8 * (by + iy),
            config, cmap_factors, block, scratch_space, quantized,
            best_entropy, cand_entropy, cand_q, cand_act));
        if (cand_entropy < best_entropy) {
          best_entropy = cand_entropy;
          best_of_8x8s = tx.type;
          best_q = cand_q;
          best_act = cand_act;
        }
      }
      dp->entropy_estimate[iy][ix] = best_entropy;
      dp->best_8x8_type[iy][ix] = best_of_8x8s;
      dp->best_8x8_q[iy][ix] = best_q;
      dp->best_8x8_act[iy][ix] = best_act;
      row_sum += best_entropy;
      dp->prefix_sum[iy + 1][ix + 1] = dp->prefix_sum[iy][ix + 1] + row_sum;
    }
  }

  // Map 1D intervals [pos, pos + len) in [0, 32) to [0, 528).
  uint16_t iv_idx[32][33] = {};
  uint16_t next_iv = 0;
  for (size_t pos = 0; pos < 32; ++pos) {
    for (size_t len = 1; pos + len <= 32; ++len) {
      iv_idx[pos][len] = next_iv++;
    }
  }

  auto rect_base_sum = [&](size_t x0, size_t y0, size_t w, size_t h) -> float {
    return dp->prefix_sum[y0 + h][x0 + w] - dp->prefix_sum[y0][x0 + w] -
           dp->prefix_sum[y0 + h][x0] + dp->prefix_sum[y0][x0];
  };

  constexpr float kInf = 1e30f;
  const bool allow_16 = (cparams.decoding_speed_tier <= 4);
  const bool allow_16x32 = allow_16;
  const bool allow_32x32 = (cparams.decoding_speed_tier < 4);
  const bool allow_64x32 = (cparams.decoding_speed_tier <= 1);
  const bool allow_64x64 = (cparams.decoding_speed_tier < 4);

  size_t num_cands = 0;

  // Helper to run 64-bit bitmask non-guillotine set-packing on any sub-rect
  // [rx0, rx0 + rw) x [ry0, ry0 + rh) with rw <= 8, rh <= 8.
  auto solve_non_guillotine_subrect = [&](size_t rx0, size_t ry0, size_t rw,
                                          size_t rh, float dp_sub_cost,
                                          uint16_t* out_best_chosen,
                                          size_t* out_best_num,
                                          float* out_best_gain) -> void {
    *out_best_num = 0;
    const float sub_base = rect_base_sum(rx0, ry0, rw, rh);
    const float dp_sub_gain = std::max(0.0f, sub_base - dp_sub_cost);
    *out_best_gain = dp_sub_gain;

    size_t n_loc = 0;
    for (size_t i = 0; i < num_cands; ++i) {
      const CandBlock& cb = dp->cands[i];
      if (cb.x0 >= rx0 && cb.y0 >= ry0 && cb.x0 + cb.w <= rx0 + rw &&
          cb.y0 + cb.h <= ry0 + rh &&
          !(cb.w == rw && cb.h == rh) && n_loc < 352) {
        uint64_t mask = 0;
        const uint64_t row_bits =
            ((uint64_t{1} << cb.w) - 1) << (cb.x0 - rx0);
        for (size_t dy = 0; dy < cb.h; ++dy) {
          mask |= (row_bits << ((cb.y0 - ry0 + dy) * 8));
        }
        dp->local_cands[n_loc++] = {
            mask, cb.gain, cb.rate, cb.q, cb.x0, cb.y0, cb.type, cb.act,
        };
      }
    }
    if (n_loc <= 1) return;

    std::sort(dp->local_cands, dp->local_cands + n_loc,
              [](const LocalCand& a, const LocalCand& b) {
                return a.rate > b.rate;
              });
    float best_ng_gain = dp_sub_gain + 1e-3f;
    uint16_t cur_chosen[64];
    size_t cur_num = 0;
    size_t nodes = 0;
    constexpr size_t kMaxNodes = 2048;

    auto dfs = [&](auto& self, size_t idx, uint64_t used,
                   float cur_gain) -> void {
      if (cur_gain > best_ng_gain) {
        best_ng_gain = cur_gain;
        *out_best_num = cur_num;
        *out_best_gain = cur_gain;
        for (size_t i = 0; i < cur_num; ++i) {
          out_best_chosen[i] = cur_chosen[i];
        }
      }
      while (idx < n_loc && (dp->local_cands[idx].mask & used) != 0) {
        ++idx;
      }
      if (idx >= n_loc || ++nodes > kMaxNodes) return;
      uint64_t avail_mask = 0;
      float avail_gain = 0.0f;
      for (size_t k = idx; k < n_loc; ++k) {
        if ((dp->local_cands[k].mask & used) == 0) {
          avail_mask |= dp->local_cands[k].mask;
          avail_gain += dp->local_cands[k].gain;
        }
      }
      const float max_possible =
          cur_gain +
          std::min(avail_gain,
                   dp->local_cands[idx].rate *
                       static_cast<float>(hwy::PopCount(avail_mask)));
      if (max_possible <= best_ng_gain) return;

      const LocalCand& lc = dp->local_cands[idx];
      cur_chosen[cur_num++] = static_cast<uint16_t>(idx);
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

  for (size_t h = 1; h <= ry; ++h) {
    PinSeg y_segs[9];
    size_t num_y_segs = 0;
    if (enable_rdo && h >= 3 && h <= 12) {
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
          const uint16_t iy = iv_idx[y0][1];
          for (size_t x0 = 0; x0 < rx; ++x0) {
            const uint16_t ix = iv_idx[x0][1];
            dp->cost[iy][ix] = dp->entropy_estimate[y0][x0];
            dp->split[iy][ix] = 0;
            dp->leaf[y0][x0][1][1] = dp->best_8x8_type[y0][x0];
            dp->leaf_q[y0][x0][1][1] = dp->best_8x8_q[y0][x0];
            dp->leaf_act[y0][x0][1][1] = dp->best_8x8_act[y0][x0];
          }
        }
        continue;
      }

      PinSeg x_segs[9];
      size_t num_x_segs = 0;
      if (enable_rdo && w >= 3 && w <= 12 && num_y_segs > 0) {
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

      // Check if (w, h) corresponds to a valid single multi-block transform.
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

      const AcStrategy acs_direct = AcStrategy::FromRawStrategy(direct_type);

      const size_t k_dyadic_w =
          (w > 1) ? (size_t{1} << FloorLog2Nonzero(w - 1)) : 0;
      const size_t k_dyadic_h =
          (h > 1) ? (size_t{1} << FloorLog2Nonzero(h - 1)) : 0;

      for (size_t y0 = 0; y0 + h <= ry; ++y0) {
        const uint16_t iy = iv_idx[y0][h];
        for (size_t x0 = 0; x0 + w <= rx; ++x0) {
          const uint16_t ix = iv_idx[x0][w];
          float best_c = kInf;
          uint16_t best_s = 1;

          // Vertical splits: [x0, x0 + k) and [x0 + k, x0 + w)
          for (size_t k = 1; k < w; ++k) {
            if (!enable_rdo && k != k_dyadic_w) continue;
            const float c = dp->cost[iy][iv_idx[x0][k]] +
                            dp->cost[iy][iv_idx[x0 + k][w - k]];
            if (c < best_c) {
              best_c = c;
              best_s = static_cast<uint16_t>(k);
            }
          }
          // Horizontal splits: [y0, y0 + k) and [y0 + k, y0 + h)
          for (size_t k = 1; k < h; ++k) {
            if (!enable_rdo && k != k_dyadic_h) continue;
            const float c = dp->cost[iv_idx[y0][k]][ix] +
                            dp->cost[iv_idx[y0 + k][h - k]][ix];
            if (c < best_c) {
              best_c = c;
              best_s = static_cast<uint16_t>(32 + k);
            }
          }
          // Non-guillotine 5-rectangle pinwheel splits (CW and CCW):
          for (size_t yi = 0; yi < num_y_segs; ++yi) {
            const uint8_t ya = y_segs[yi].a;
            const uint8_t yb = y_segs[yi].b;
            const uint16_t iy_top_a = iv_idx[y0][ya];
            const uint16_t iy_top_b = iv_idx[y0][yb];
            const uint16_t iy_bot_a = iv_idx[y0 + ya][h - ya];
            const uint16_t iy_bot_b = iv_idx[y0 + yb][h - yb];
            const uint16_t iy_mid = iv_idx[y0 + ya][yb - ya];
            for (size_t xi = 0; xi < num_x_segs; ++xi) {
              const uint8_t xa = x_segs[xi].a;
              const uint8_t xb = x_segs[xi].b;
              const float c_mid = dp->cost[iy_mid][iv_idx[x0 + xa][xb - xa]];
              // CW: TL (xb x ya), TR ((w-xb) x yb), BR ((w-xa) x (h-yb)),
              //     BL (xa x (h-ya)), center ((xb-xa) x (yb-ya))
              const float c_cw =
                  dp->cost[iy_top_a][iv_idx[x0][xb]] +
                  dp->cost[iy_top_b][iv_idx[x0 + xb][w - xb]] +
                  dp->cost[iy_bot_b][iv_idx[x0 + xa][w - xa]] +
                  dp->cost[iy_bot_a][iv_idx[x0][xa]] + c_mid;
              if (c_cw < best_c) {
                best_c = c_cw;
                best_s = static_cast<uint16_t>(
                    0x4000u | (static_cast<uint16_t>(xa) << 10) |
                    (static_cast<uint16_t>(xb - xa) << 7) |
                    (static_cast<uint16_t>(ya) << 4) |
                    static_cast<uint16_t>(yb - ya));
              }
              // CCW: TL (xa x yb), TR ((w-xa) x ya), BR ((w-xb) x (h-ya)),
              //      BL (xb x (h-yb)), center ((xb-xa) x (yb-ya))
              const float c_ccw =
                  dp->cost[iy_top_b][iv_idx[x0][xa]] +
                  dp->cost[iy_top_a][iv_idx[x0 + xa][w - xa]] +
                  dp->cost[iy_bot_a][iv_idx[x0 + xb][w - xb]] +
                  dp->cost[iy_bot_b][iv_idx[x0][xb]] + c_mid;
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

          if (has_direct) {
            const float base_sum = rect_base_sum(x0, y0, w, h);
            const bool is_8x32 = (w == 4 && h == 1) || (w == 1 && h == 4);
            const bool eval_direct =
                enable_rdo
                    ? (is_8x32 ? (best_c < base_sum - 1e-4f)
                               : ((w * h <= 16) ||
                                  ((x0 & 1) == 0 && (y0 & 1) == 0) ||
                                  (best_c < base_sum - 1e-4f)))
                    : (x0 % w == 0 && y0 % h == 0);
            if (eval_direct) {
              float cand_entropy = kInf;
              float cand_q = 0.0f;
              uint8_t cand_act = 0;
              const float* JXL_RESTRICT cmap_factors =
                  cmap_factors_tile[y0 / kColorTileDimInBlocks]
                                   [x0 / kColorTileDimInBlocks];
              JXL_RETURN_IF_ERROR(EstimateEntropy(
                  acs_direct, (bx + x0) * 8, (by + y0) * 8, config,
                  cmap_factors, block, scratch_space, quantized, best_c,
                  cand_entropy, cand_q, cand_act));
              if (cand_entropy < best_c) {
                best_c = cand_entropy;
                best_s = 0;
                dp->leaf[y0][x0][h][w] = direct_type;
                dp->leaf_q[y0][x0][h][w] = cand_q;
                dp->leaf_act[y0][x0][h][w] = cand_act;
                const float gain = base_sum - cand_entropy;
                if (enable_rdo && gain > 0.0f && num_cands < 6144) {
                  dp->cands[num_cands++] = {
                      static_cast<uint8_t>(x0),
                      static_cast<uint8_t>(y0),
                      static_cast<uint8_t>(w),
                      static_cast<uint8_t>(h),
                      direct_type,
                      cand_act,
                      gain,
                      gain / static_cast<float>(w * h),
                      cand_q,
                  };
                }
              }
            }
          }

          if (enable_rdo &&
              ((w == 8 && h == 8) || (w == rx && h == ry && rx <= 8 && ry <= 8)) &&
              best_s != 0 && ((x0 % 4 == 0 && y0 % 4 == 0) || (w == rx && h == ry))) {
            uint16_t ng_chosen[64];
            size_t ng_num = 0;
            float ng_gain = 0.0f;
            solve_non_guillotine_subrect(x0, y0, w, h, best_c, ng_chosen,
                                         &ng_num, &ng_gain);
            if (ng_num > 0) {
              best_c = rect_base_sum(x0, y0, w, h) - ng_gain;
              best_s = 0x3FFFu;
            }
          }

          dp->cost[iy][ix] = best_c;
          dp->split[iy][ix] = best_s;
        }
      }
    }
  }

  auto commit_block_quant = [&](size_t x0, size_t y0, size_t w, size_t h,
                                AcStrategyType type, float q_val,
                                uint8_t act_val) {
    if (config.band_qf_stride > 0) {
      static constexpr float kBandRescaleMix[4] = {0.72f, 0.84f, 0.96f, 1.00f};
      float q_max = 0.0f;
      float q_rms = 0.0f;
      for (size_t iy = 0; iy < h; ++iy) {
        for (size_t ix = 0; ix < w; ++ix) {
          const float qv = config.Quant(bx + x0 + ix, by + y0 + iy);
          q_max = std::max(q_max, qv);
          q_rms += qv * qv;
        }
      }
      const float inv_nb = 1.0f / static_cast<float>(w * h);
      q_rms = std::sqrt(q_rms * inv_nb);
      float q_pool = q_max;
      if (w * h >= 4) {
        float mean_max_mixer = 1.0f;
        if (config.butteraugli_distance > 1.54138f) {
          mean_max_mixer = std::max(
              0.0f,
              1.0f - (config.butteraugli_distance - 1.54138f) * 0.56391f);
        }
        q_pool = mean_max_mixer * q_max + (1.0f - mean_max_mixer) * q_rms;
      }
      const float q_step_ratio = q_val / std::max(q_pool, 1e-4f);
      for (size_t c = 0; c < 3; ++c) {
        for (size_t s = 0; s < 4; ++s) {
          if (config.band_qf_rows[c][s] == nullptr) continue;
          float sq_sum = 0.0f;
          for (size_t iy = 0; iy < h; ++iy) {
            for (size_t ix = 0; ix < w; ++ix) {
              const size_t sbx = bx + x0 + ix;
              const size_t sby = by + y0 + iy;
              const float old_q = std::max(config.Quant(sbx, sby), 1e-4f);
              const float rescale = q_pool / old_q;
              const float band_rescale =
                  (rescale > 1.0f)
                      ? (1.0f + kBandRescaleMix[s] * (rescale - 1.0f))
                      : rescale;
              const float v = config.BandQuant(c, s, sbx, sby) * band_rescale;
              sq_sum += v * v;
            }
          }
          float bq = std::sqrt(sq_sum * inv_nb) * q_step_ratio;
          if (c == 1 && s == 0) {
            uint32_t bits = 0;
            memcpy(&bits, &bq, sizeof(bits));
            bits = (bits & ~31u) | (static_cast<uint32_t>(act_val) & 31u);
            memcpy(&bq, &bits, sizeof(bq));
          }
          config.band_qf_rows[c][s][(by + y0) * config.band_qf_stride +
                                    (bx + x0)] = bq;
        }
      }
      for (size_t iy = 0; iy < h; ++iy) {
        for (size_t ix = 0; ix < w; ++ix) {
          config.quant_field_row[(by + y0 + iy) * config.quant_field_stride +
                                 (bx + x0 + ix)] = q_val;
        }
      }
    } else if (config.band_qf_rows[1][0] != nullptr) {
      float bq = 1.0f;
      uint32_t bits = 0;
      memcpy(&bits, &bq, sizeof(bits));
      bits = (bits & ~31u) | (static_cast<uint32_t>(act_val) & 31u);
      memcpy(&bq, &bits, sizeof(bq));
      config.band_qf_rows[1][0][(by + y0) * config.quant_field_stride +
                                (bx + x0)] = bq;
    }
    config.quant_field_row[(by + y0) * config.quant_field_stride + (bx + x0)] =
        q_val;
  };

  // Reconstruct the globally optimal partition of [0, rx) x [0, ry).
  struct SubRect {
    uint8_t x0, y0, w, h;
  };
  SubRect stack[512];
  size_t sp = 0;
  stack[sp++] = {0, 0, static_cast<uint8_t>(rx), static_cast<uint8_t>(ry)};
  while (sp > 0) {
    const SubRect cur = stack[--sp];
    const uint16_t iy = iv_idx[cur.y0][cur.h];
    const uint16_t ix = iv_idx[cur.x0][cur.w];
    const uint16_t s = dp->split[iy][ix];
    if (s == 0) {
      const AcStrategyType ltype = dp->leaf[cur.y0][cur.x0][cur.h][cur.w];
      JXL_RETURN_IF_ERROR(ac_strategy->Set(bx + cur.x0, by + cur.y0, ltype));
      commit_block_quant(cur.x0, cur.y0, cur.w, cur.h, ltype,
                         dp->leaf_q[cur.y0][cur.x0][cur.h][cur.w],
                         dp->leaf_act[cur.y0][cur.x0][cur.h][cur.w]);
      continue;
    }
    if (enable_rdo && cur.w <= 8 && cur.h <= 8 &&
        (s == 0x3FFFu || (cur.w >= 4 && cur.h >= 4))) {
      uint16_t ng_chosen[64];
      size_t ng_num = 0;
      float ng_gain = 0.0f;
      const float compare_cost =
          (s == 0x3FFFu) ? kInf : dp->cost[iy][ix];
      solve_non_guillotine_subrect(cur.x0, cur.y0, cur.w, cur.h, compare_cost,
                                   ng_chosen, &ng_num, &ng_gain);
      if (ng_num > 0) {
        uint64_t covered = 0;
        for (size_t i = 0; i < ng_num; ++i) {
          const LocalCand& lc = dp->local_cands[ng_chosen[i]];
          covered |= lc.mask;
          JXL_RETURN_IF_ERROR(
              ac_strategy->Set(bx + lc.x0, by + lc.y0, lc.type));
          const AcStrategy acs_lc = AcStrategy::FromRawStrategy(lc.type);
          commit_block_quant(lc.x0, lc.y0, acs_lc.covered_blocks_x(),
                             acs_lc.covered_blocks_y(), lc.type, lc.q, lc.act);
        }
        for (size_t dy = 0; dy < cur.h; ++dy) {
          for (size_t dx = 0; dx < cur.w; ++dx) {
            if ((covered & (uint64_t{1} << (dy * 8 + dx))) == 0) {
              const size_t y8 = cur.y0 + dy;
              const size_t x8 = cur.x0 + dx;
              JXL_RETURN_IF_ERROR(ac_strategy->Set(
                  bx + x8, by + y8, dp->best_8x8_type[y8][x8]));
              commit_block_quant(x8, y8, 1, 1, dp->best_8x8_type[y8][x8],
                                 dp->best_8x8_q[y8][x8],
                                 dp->best_8x8_act[y8][x8]);
            }
          }
        }
        continue;
      }
    }
    if (s < 32) {
      const uint8_t k = static_cast<uint8_t>(s);
      stack[sp++] = {static_cast<uint8_t>(cur.x0 + k), cur.y0,
                     static_cast<uint8_t>(cur.w - k), cur.h};
      stack[sp++] = {cur.x0, cur.y0, k, cur.h};
    } else if (s < 64) {
      const uint8_t k = static_cast<uint8_t>(s - 32);
      stack[sp++] = {cur.x0, static_cast<uint8_t>(cur.y0 + k), cur.w,
                     static_cast<uint8_t>(cur.h - k)};
      stack[sp++] = {cur.x0, cur.y0, cur.w, k};
    } else if ((s & 0x4000u) != 0) {
      const bool ccw = (s & 0x2000u) != 0;
      const uint8_t xa = static_cast<uint8_t>((s >> 10) & 7u);
      const uint8_t xb = static_cast<uint8_t>(xa + ((s >> 7) & 7u));
      const uint8_t ya = static_cast<uint8_t>((s >> 4) & 7u);
      const uint8_t yb = static_cast<uint8_t>(ya + (s & 7u));
      // Center:
      stack[sp++] = {static_cast<uint8_t>(cur.x0 + xa),
                     static_cast<uint8_t>(cur.y0 + ya),
                     static_cast<uint8_t>(xb - xa),
                     static_cast<uint8_t>(yb - ya)};
      if (!ccw) {
        stack[sp++] = {cur.x0, static_cast<uint8_t>(cur.y0 + ya), xa,
                       static_cast<uint8_t>(cur.h - ya)};
        stack[sp++] = {static_cast<uint8_t>(cur.x0 + xa),
                       static_cast<uint8_t>(cur.y0 + yb),
                       static_cast<uint8_t>(cur.w - xa),
                       static_cast<uint8_t>(cur.h - yb)};
        stack[sp++] = {static_cast<uint8_t>(cur.x0 + xb), cur.y0,
                       static_cast<uint8_t>(cur.w - xb), yb};
        stack[sp++] = {cur.x0, cur.y0, xb, ya};
      } else {
        stack[sp++] = {cur.x0, static_cast<uint8_t>(cur.y0 + yb), xb,
                       static_cast<uint8_t>(cur.h - yb)};
        stack[sp++] = {static_cast<uint8_t>(cur.x0 + xb),
                       static_cast<uint8_t>(cur.y0 + ya),
                       static_cast<uint8_t>(cur.w - xb),
                       static_cast<uint8_t>(cur.h - ya)};
        stack[sp++] = {static_cast<uint8_t>(cur.x0 + xa), cur.y0,
                       static_cast<uint8_t>(cur.w - xa), ya};
        stack[sp++] = {cur.x0, cur.y0, xa, yb};
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
HWY_EXPORT(ProcessRectACS);

Status AcStrategyHeuristics::Init(const Image3F& src, const Rect& rect_in,
                                  ImageF* quant_field, const ImageF& mask1x1,
                                  DequantMatrices* matrices,
                                  ImageF (*band_qf)[4], float x_qm_multiplier,
                                  float b_qm_multiplier, float quant_scale,
                                  float inv_global_scale) {
  config.dequant = matrices;
  config.token_cost_model = nullptr;
  config.quant_scale = quant_scale;
  config.inv_global_scale = inv_global_scale;

  if (cparams.speed_tier >= SpeedTier::kCheetah) {
    JXL_RETURN_IF_ERROR(
        matrices->EnsureComputed(memory_manager, 1));  // DCT8 only
  } else {
    uint32_t acs_mask = 0;
    // All transforms up to 64x64.
    for (size_t i = 0; i < static_cast<size_t>(AcStrategyType::DCT128X128);
         i++) {
      acs_mask |= (1 << i);
    }
    JXL_RETURN_IF_ERROR(matrices->EnsureComputed(memory_manager, acs_mask));
  }

  // Image row pointers and strides.
  config.quant_field_row = quant_field->Row(0);
  config.quant_field_stride = quant_field->PixelsPerRow();
  config.base_quant =
      (quant_field->xsize() > 0 && quant_field->ysize() > 0)
          ? quant_field->ConstRow(0)[0]
          : 1.0f;
  config.mask1x1_xsize = mask1x1.xsize();
  if (mask1x1.xsize() > 0 && mask1x1.ysize() > 0) {
    config.masking1x1_field_row = mask1x1.Row(0);
    config.masking1x1_field_stride = mask1x1.PixelsPerRow();
  }

  for (size_t c = 0; c < 3; ++c) {
    for (size_t s = 0; s < 4; ++s) {
      config.band_qf_rows[c][s] = nullptr;
    }
  }
  config.band_qf_stride = 0;
  if (cparams.butteraugli_distance < 3.5f && cparams.resampling <= 1 &&
      cparams.gaborish != Override::kOff &&
      quant_field->xsize() * quant_field->ysize() > 64 && band_qf != nullptr &&
      band_qf[0][0].xsize() > 0 && band_qf[0][0].ysize() > 0) {
    config.band_qf_stride = band_qf[0][0].PixelsPerRow();
    for (size_t c = 0; c < 3; ++c) {
      for (size_t s = 0; s < 4; ++s) {
        if (band_qf[c][s].xsize() > 0 && band_qf[c][s].ysize() > 0) {
          config.band_qf_rows[c][s] = band_qf[c][s].Row(0);
        }
      }
    }
  } else if (band_qf != nullptr && band_qf[1][0].xsize() > 0 &&
             band_qf[1][0].ysize() > 0) {
    config.band_qf_rows[1][0] = band_qf[1][0].Row(0);
  }
  config.qm_multiplier[0] = x_qm_multiplier;
  config.qm_multiplier[1] = 1.0f;
  config.qm_multiplier[2] = b_qm_multiplier;
  for (size_t c = 0; c < 3; ++c) {
    const float* dct8_matrix = matrices->Matrix(AcStrategyType::DCT, c);
    const float m_ref = dct8_matrix[1];
    constexpr float kBetaCsf = 0.08f;
    float csf_sum = 0.0f;
    for (size_t i = 0; i < 64; ++i) {
      const float m_uv =
          (i == 0) ? m_ref : std::max(dct8_matrix[i], m_ref * 0.5f);
      const float w_uv = std::pow(m_ref / m_uv, kBetaCsf);
      config.csf_w[c][i] = w_uv;
      csf_sum += w_uv;
    }
    const float csf_norm = 61.8f / csf_sum;
    for (size_t i = 0; i < 64; ++i) {
      config.csf_w[c][i] *= csf_norm;
    }
  }

  config.src_rows[0] = rect_in.ConstPlaneRow(src, 0, 0);
  config.src_rows[1] = rect_in.ConstPlaneRow(src, 1, 0);
  config.src_rows[2] = rect_in.ConstPlaneRow(src, 2, 0);
  config.src_stride = src.PixelsPerRow();

  // Entropy estimate is composed of two factors:
  //  - estimate of the number of bits that will be used by the block
  //  - information loss due to quantization
  // The following constant controls the relative weights of these components.
  config.info_loss_multiplier = 1.2;
  config.zeros_mul = 9.3089059022677905;
  config.cost_delta = 10.833273317067883;
  config.butteraugli_distance = cparams.butteraugli_distance;

  static const float kBias = 0.13731742964354549;
  const float ratio = (cparams.butteraugli_distance + kBias) / (1.0f + kBias);

  static const float kPow1 = 0.33677806662454718;
  static const float kPow2 = 0.50990926717963703;
  static const float kPow3 = 0.36702940662370243;
  config.info_loss_multiplier *= std::pow(ratio, kPow1);
  config.zeros_mul *= std::pow(ratio, kPow2);
  config.cost_delta *= std::pow(ratio, kPow3);
  return true;
}

Status AcStrategyHeuristics::PrepareForThreads(std::size_t num_threads) {
  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;
  mem_per_thread = 8 * AcStrategy::kMaxCoeffArea + dct_scratch_size;
  size_t mem_bytes = num_threads * mem_per_thread * sizeof(float);
  JXL_ASSIGN_OR_RETURN(mem, AlignedMemory::Create(memory_manager, mem_bytes));
  qmem_per_thread = 655360;
  size_t qmem_bytes = num_threads * qmem_per_thread * sizeof(uint32_t);
  JXL_ASSIGN_OR_RETURN(qmem, AlignedMemory::Create(memory_manager, qmem_bytes));
  return true;
}

Status AcStrategyHeuristics::ProcessRect(const Rect& rect,
                                         const ColorCorrelationMap& cmap,
                                         AcStrategyImage* ac_strategy,
                                         size_t thread) {
  // In Cheetah mode, use DCT8 everywhere and uniform quantization.
  if (cparams.speed_tier >= SpeedTier::kCheetah) {
    ac_strategy->FillDCT8(rect);
    return true;
  }
  return HWY_DYNAMIC_DISPATCH(ProcessRectACS)(
      cparams, config, rect, cmap,
      mem.address<float>() + thread * mem_per_thread,
      qmem.address<uint32_t>() + thread * qmem_per_thread, ac_strategy);
}

Status AcStrategyHeuristics::Finalize(const FrameDimensions& frame_dim,
                                      const AcStrategyImage& ac_strategy,
                                      AuxOut* aux_out) {
  // Accounting and debug output.
  if (aux_out != nullptr) {
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

  if (JXL_DEBUG_AC_STRATEGY && WantDebugOutput(cparams)) {
    JXL_RETURN_IF_ERROR(DumpAcStrategy(ac_strategy, frame_dim.xsize,
                                       frame_dim.ysize, "ac_strategy", aux_out,
                                       cparams));
  }
  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
