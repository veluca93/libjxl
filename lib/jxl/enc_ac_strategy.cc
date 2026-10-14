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
#include "lib/jxl/common.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"
#include "lib/jxl/memory_manager_internal.h"
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
using hwy::HWY_NAMESPACE::Min;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Round;
using hwy::HWY_NAMESPACE::Sqrt;
#if HWY_TARGET != HWY_SCALAR
using hwy::HWY_NAMESPACE::Shuffle1032;
using hwy::HWY_NAMESPACE::Shuffle2301;
#endif

Status EstimateEntropy(const AcStrategy& acs, float entropy_mul, size_t x,
                       size_t y, const ACSConfig& config,
                       const float* JXL_RESTRICT cmap_factors, float* block,
                       float* full_scratch_space, uint32_t* quantized,
                       float& entropy, float& q_out, uint8_t& act_out) {
  entropy = 0.0f;
  float* mem = full_scratch_space;
  float* pixels_buf = full_scratch_space + AcStrategy::kMaxCoeffArea;
  float* scratch_space = full_scratch_space + 2 * AcStrategy::kMaxCoeffArea;
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
  if (config.band_qf_stride == 0) {
    if (acs.Strategy() == AcStrategyType::DCT16X8 ||
        acs.Strategy() == AcStrategyType::DCT8X16) {
      q_pool *= 0.90390420971782894f;
    } else if (acs.Strategy() == AcStrategyType::DCT16X16) {
      q_pool *= 0.89232976007290965f;
    }
  }
  q_out = q_pool;
  act_out = 0;

  const HWY_CAPPED(float, 8) df8;
  const auto zero = Zero(df);
  const auto one = Set(df, 1.0f);
  const auto quant = Set(df, quant_norm16);

  float* w_row = pixels_buf;
  float* t0_row = pixels_buf + AcStrategy::kMaxBlockDim;
  float* cw_row = pixels_buf + 2 * AcStrategy::kMaxBlockDim;
  float* w_q_row = pixels_buf + 3 * AcStrategy::kMaxBlockDim;
  float* t_tr_row = pixels_buf + 4 * AcStrategy::kMaxBlockDim;
  float* t0_q_row = pixels_buf + 5 * AcStrategy::kMaxBlockDim;
  auto loss = Zero(df8);
  auto loss2 = Zero(df8);

  const bool is_partial_block =
      ((1 << static_cast<size_t>(acs.Strategy())) & kPartialBlockKinds) != 0;
  const float inv_nblocks = 1.0f / static_cast<float>(num_blocks);
  const bool use_csf_w = (!is_partial_block && config.band_qf_stride > 0);
  const bool use_q_rdo = (!is_partial_block && config.quant_scale > 0.0f);

  static constexpr float kBandRescaleMix[4] = {0.72f, 0.84f, 0.96f, 1.00f};
  static constexpr float kBandErrWeight[4] = {0.81f, 0.96f, 1.18f, 1.40f};
  static constexpr float kAlphaBlk[4][3] = {
      {0.09624f, 0.19962f, 0.13015f},
      {0.30558f, 0.07055f, 0.06271f},
      {0.05012f, 0.19465f, 0.04615f},
      {0.20420f, 0.17975f, 0.03865f},
  };
  static constexpr float kAlphaVal[4][3] = {
      {0.27450f, 1.12558f, 0.98950f},
      {0.46522f, 0.40946f, 0.36582f},
      {0.28035f, 0.91827f, 1.55815f},
      {0.26873f, 0.68864f, 1.20822f},
  };
  static constexpr float kAlphaValNoRdo[4][3] = {
      {0.11965f, 0.49060f, 0.43129f},
      {0.20277f, 0.17847f, 0.15945f},
      {0.12220f, 0.40024f, 0.67915f},
      {0.11713f, 0.30015f, 0.52662f},
  };

  int32_t quant_orig = 1;
  float q_block_orig = quant_norm16;
  float dz_w = 0.0f;
  float dz_w_rt = 0.0f;
  float hr_w = 1.0f;
  float dct8_err_min = 0.0f;
  float zero_thres_q[3][8] = {};
  float trial_thres[4] = {0.58f, 0.64f, 0.64f, 0.64f};
  if (use_q_rdo) {
    quant_orig = Quantizer::ClampVal(q_pool * config.inv_global_scale + 0.5f);
    q_block_orig = config.quant_scale * quant_orig;
    const float d = config.butteraugli_distance;
    dz_w = use_csf_w ? Clamp1((d - 0.35f) / (d + 0.55f) * 1.55f, 0.0f, 1.0f)
                     : 0.0f;
    dz_w_rt = Clamp1((d - 0.50f) / (d + 0.55f) * 1.65f, 0.0f, 1.0f);
    hr_w = 1.0f - dz_w_rt;
    dct8_err_min =
        use_csf_w ? (0.42f + 1.85f * Clamp1(d - 0.95f, 0.0f, 1.20f)) : 0.0f;
    if (cx > 1 || cy > 1) {
      const float sub = Clamp1(0.003f * cx * cy, 0.0f, 0.08f);
      for (int i = 0; i < 4; ++i) {
        trial_thres[i] = std::max(0.54f, trial_thres[i] - sub);
      }
    }
    for (size_t c = 0; c < 3; ++c) {
      float bq_p[4] = {q_block_orig, q_block_orig, q_block_orig, q_block_orig};
      if (use_csf_w) {
        for (size_t s = 0; s < 4; ++s) {
          float sq_sum = 0.0f;
          for (size_t iy = 0; iy < cov_y; ++iy) {
            for (size_t ix = 0; ix < cov_x; ++ix) {
              const float old_q =
                  std::max(config.Quant(bx0 + ix, by0 + iy), 1e-4f);
              const float rescale = q_pool / old_q;
              const float band_rescale =
                  (rescale > 1.0f)
                      ? (1.0f + kBandRescaleMix[s] * (rescale - 1.0f))
                      : rescale;
              const float v =
                  config.BandQuant(c, s, bx0 + ix, by0 + iy) * band_rescale;
              sq_sum += v * v;
            }
          }
          bq_p[s] = std::sqrt(sq_sum * inv_nblocks);
        }
      }
      ComputeBandThresholds(c, acs.Strategy(), cx, cy, q_block_orig, bq_p,
                            zero_thres_q[c], dz_w);
    }
  }

  int32_t max_quant = 0;
  for (size_t c : {1, 0, 2}) {
    const float* inv_matrix = config.dequant->InvMatrix(acs.Strategy(), c);
    const float* matrix = config.dequant->Matrix(acs.Strategy(), c);
    const auto cmap_factor = Set(df, cmap_factors[c]);

    float band_w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float zero_thres[8] = {0.50f, 0.50f, 0.50f, 0.50f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (use_csf_w) {
      float bq[4];
      for (size_t s = 0; s < 4; ++s) {
        float sq_sum = 0.0f;
        for (size_t iy = 0; iy < cov_y; ++iy) {
          for (size_t ix = 0; ix < cov_x; ++ix) {
            const float v = config.BandQuant(c, s, bx0 + ix, by0 + iy);
            sq_sum += v * v;
          }
        }
        bq[s] = std::sqrt(sq_sum * inv_nblocks);
      }
      ComputeBandThresholds(c, acs.Strategy(), cx, cy, quant_norm16, bq,
                            zero_thres);
      for (size_t s = 0; s < 4; ++s) {
        band_w[s] = Clamp1(1.0f - 2.2f * zero_thres[4 + s], 0.88f, 1.12f);
      }
    }

    float band_w_q[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (use_q_rdo) {
      for (size_t s = 0; s < 4; ++s) {
        band_w_q[s] =
            (use_csf_w ? kBandErrWeight[s] : 1.0f) *
            Clamp1(1.0f - 3.5f * zero_thres_q[c][4 + s], 0.88f, 1.12f);
      }
    }
    const auto q_block_c_v = Set(df, q_block_orig * config.qm_multiplier[c]);
    const auto bias_c_v = Set(df, kDefaultQuantBias[c]);
    const auto bias3_v = Set(df, kDefaultQuantBias[3]);

    auto entropy_v = Zero(df);
    auto nzeros_v = Zero(df);
    auto r0_v = Zero(df);
    auto e0_v = Zero(df);
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
        const float csf =
            use_csf_w ? config.csf_w[c][v8 * kBlockDim + u8] : 1.0f;
        const float ws = band_w[s] * csf;
        const float dz = is_llf ? 0.0f : zero_thres[4 + s];
        const float t0 = 0.50f + ((dz >= 0.0f) ? dz : (0.5f * dz));
        const float cw =
            (!use_csf_w || is_llf)
                ? 1.0f
                : (0.94f + 0.0085f * static_cast<float>(v8 + u8));
        const float w_q = is_llf ? 0.0f : band_w_q[s];
        const float t_tr =
            is_llf ? 1e30f : (trial_thres[hfix] + zero_thres_q[c][4 + s]);
        const float t0_q =
            is_llf ? 1e30f
                   : (use_csf_w
                          ? (zero_thres_q[c][hfix] + zero_thres_q[c][4 + s])
                          : t_tr);
        for (size_t ix = 0; ix < cx; ++ix) {
          w_row[u8 * cx + ix] = ws;
          t0_row[u8 * cx + ix] = t0;
          cw_row[u8 * cx + ix] = cw;
          w_q_row[u8 * cx + ix] = w_q;
          t_tr_row[u8 * cx + ix] = t_tr;
          t0_q_row[u8 * cx + ix] = t0_q;
        }
      }
      for (size_t iy = 0; iy < cy; ++iy) {
        const size_t off = (v8 * cy + iy) * row_stride;
        for (size_t rx = 0; rx < row_stride; rx += Lanes(df)) {
          const auto in = Load(df, block + c * size + off + rx);
          const auto in_y =
              (c == 1) ? zero
                       : Mul(Load(df, block + size + off + rx), cmap_factor);
          const auto im = Load(df, inv_matrix + off + rx);
          const auto val = Mul(Sub(in, in_y), Mul(im, quant));
          const auto t0 = Load(df, t0_row + rx);
          const auto rval = IfThenElseZero(Ge(Abs(val), t0), Round(val));
          const auto ws = Load(df, w_row + rx);
          const auto diff = Mul(ws, Sub(val, rval));
          const auto m = Load(df, matrix + off + rx);
          Store(Mul(m, diff), df, &mem[off + rx]);
          const auto q_mag = Abs(rval);
          const auto q_is_zero = Eq(q_mag, zero);
          const auto cw = Load(df, cw_row + rx);
          entropy_v = MulAdd(cw, Sqrt(q_mag), entropy_v);
          nzeros_v = Add(nzeros_v, IfThenZeroElse(q_is_zero, one));

          if (use_q_rdo) {
            const auto v0 = Abs(Mul(in, Mul(im, q_block_c_v)));
            const auto t0_q = Load(df, t0_q_row + rx);
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

            const auto w_q = Load(df, w_q_row + rx);
            const auto t_tr = Load(df, t_tr_row + rx);
            auto eff_deq = [&](const auto& k) {
              const auto is_pos = Ge(k, one);
              const auto is_one = Eq(k, one);
              const auto adj = IfThenElse(
                  is_one, bias_c_v,
                  IfThenElseZero(is_pos, Sub(k, Div(bias3_v, Max(k, one)))));
              return MulAdd(Set(df, 0.65f), k, Mul(Set(df, 0.35f), adj));
            };

            const auto is_nz0 = Ge(v0, t_tr);
            const auto k0 = IfThenElseZero(is_nz0, Round(v0));
            const auto eff0 = eff_deq(k0);
            const auto err0 = Mul(w_q, Abs(Sub(v0, eff0)));
            r0_v = Add(r0_v, eff0);
            e0_v = Add(e0_v, err0);
          }
        }
      }
    }

    if (use_q_rdo) {
      float hfNonZeros[4] = {};
      if (cx == 1) {
        hfNonZeros[0] = quad_row8[0][0] + quad_row8[0][1] + quad_row8[0][2] +
                        quad_row8[0][3];
        hfNonZeros[1] = quad_row8[0][4] + quad_row8[0][5] + quad_row8[0][6] +
                        quad_row8[0][7];
        hfNonZeros[2] = quad_row8[1][0] + quad_row8[1][1] + quad_row8[1][2] +
                        quad_row8[1][3];
        hfNonZeros[3] = quad_row8[1][4] + quad_row8[1][5] + quad_row8[1][6] +
                        quad_row8[1][7];
      } else {
        hfNonZeros[0] = GetLane(SumOfLanes(df, quad_v0));
        hfNonZeros[1] = GetLane(SumOfLanes(df, quad_v1));
        hfNonZeros[2] = GetLane(SumOfLanes(df, quad_v2));
        hfNonZeros[3] = GetLane(SumOfLanes(df, quad_v3));
      }

      const int32_t div = static_cast<int32_t>(num_blocks);
      const float min_hf_non_zeros = std::min(
          {hfNonZeros[0], hfNonZeros[1], hfNonZeros[2], hfNonZeros[3]});
      int32_t activity = 15;
      if (min_hf_non_zeros < 15.0f * div) {
        activity = (static_cast<int32_t>(min_hf_non_zeros) + div / 2) / div;
      }
      if (c == 1) {
        const uint8_t q0_sub =
            (activity <= 2 && hfNonZeros[0] >= 2.0f * div) ? 1u : 0u;
        act_out = static_cast<uint8_t>((activity << 1) | q0_sub);
        const float luma_hf_dz =
            (activity > 0)
                ? std::max(0.0f,
                           -0.002f * hr_w + dz_w_rt * (0.0085f * activity))
                : 0.0f;
        for (int cc : {0, 2}) {
          zero_thres_q[cc][1] += 0.18f * luma_hf_dz;
          zero_thres_q[cc][2] += 0.18f * luma_hf_dz;
          zero_thres_q[cc][3] += 0.005f * dz_w_rt + 0.40f * luma_hf_dz;
        }
      }

      const float R0 = GetLane(SumOfLanes(df, r0_v));
      const float E0 = GetLane(SumOfLanes(df, e0_v));
      const float G1 = E0;
      const float G2 = E0;

      const float total_nz =
          hfNonZeros[0] + hfNonZeros[1] + hfNonZeros[2] + hfNonZeros[3];
      int32_t max_step = 0;
      float theta1 = 1e30f;
      float theta2 = 1e30f;
      if (acs.Strategy() == AcStrategyType::DCT) {
        if (use_csf_w ? (total_nz < 11.0f) : (R0 < 8.0f)) {
          max_step = 1;
          theta1 = use_csf_w ? dct8_err_min : 2.0f;
        }
      } else if (acs.Strategy() >= AcStrategyType::DCT16X16) {
        max_step = 2;
        const size_t scale_ix =
            (num_blocks == 4)
                ? 0
                : ((num_blocks == 8) ? 1 : ((num_blocks == 16) ? 2 : 3));
        const float val_mul =
            use_csf_w ? kAlphaVal[scale_ix][c] : kAlphaValNoRdo[scale_ix][c];
        const float base_cost =
            kAlphaBlk[scale_ix][c] * num_blocks * kDCTBlockSize +
            val_mul * R0;
        theta1 = base_cost;
        theta2 = 2.0f * base_cost;
      }

      const float j1 = theta1 - G1;
      const float j2 = (theta1 < G1)
                           ? (j1 + (theta2 - G2))
                           : (j1 + std::max(1.0f, theta2 - G2));
      const float j_max = (max_step >= 2) ? j2 : ((max_step == 1) ? j1 : 0.0f);
      auto eval_j_try = [&](int32_t q_try) -> float {
        const int32_t s = q_try - quant_orig;
        if (s < 0) {
          const float dq = static_cast<float>(-s);
          return 64.0f * num_blocks * dq * dq;
        }
        if (s > max_step) {
          const float ex = static_cast<float>(s - max_step);
          return j_max + 64.0f * num_blocks * ex * ex;
        }
        if (s == 0) return 0.0f;
        if (s == 1) return j1;
        return j2;
      };

      int32_t lo = std::max(1, quant_orig - 2);
      int32_t hi =
          std::min<int32_t>(Quantizer::kQuantMax - 1, quant_orig + max_step + 2);
      while (hi - lo > 2) {
        const int32_t m1 = lo + (hi - lo) / 3;
        const int32_t m2 = hi - (hi - lo) / 3;
        if (eval_j_try(m1) <= eval_j_try(m2)) {
          hi = m2;
        } else {
          lo = m1;
        }
      }
      int32_t q_try_best = quant_orig;
      float best_j_c = eval_j_try(quant_orig);
      for (int32_t qt = lo; qt <= hi; ++qt) {
        const float jc = eval_j_try(qt);
        if (jc < best_j_c) {
          best_j_c = jc;
          q_try_best = qt;
        }
      }
      const int32_t q_c =
          std::max(std::max(4, q_try_best / 2), q_try_best - activity);
      max_quant = std::max(q_c, max_quant);
    }

    auto lossc = Zero(df8);
    auto lossc2 = Zero(df8);
    {
      float masku_lut[3] = {
          12.0,
          0.0,
          4.0,
      };
      auto masku_off = Set(df8, masku_lut[c]);
      TransformToPixels(acs.Strategy(), &mem[0], pixels_buf, cov_x * 8,
                        scratch_space);

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
      static const double kChannelMul[3] = {
          pow(8.2, 8.0),
          pow(1.0, 8.0),
          pow(1.03, 8.0),
      };
      static const float kChannelMul2[3] = {
          2.2f * 2.2f,
          1.0f,
          0.90f * 0.90f,
      };
      lossc = Mul(Set(df8, kChannelMul[c]), lossc);
      lossc2 = Mul(Set(df8, kChannelMul2[c]), lossc2);
    }
    float entropy_c = config.cost_delta * GetLane(SumOfLanes(df, entropy_v));
    size_t num_nzeros = GetLane(SumOfLanes(df, nzeros_v));
    // Add #bit of num_nonzeros, as an estimate of the cost for encoding the
    // number of non-zeros of the block.
    size_t nbits = CeilLog2Nonzero(num_nzeros + 1) + 1;
    // Also add #bit of #bit of num_nonzeros, to estimate the ANS cost, with a
    // bias.
    entropy_c += config.zeros_mul * (CeilLog2Nonzero(nbits + 17) + nbits);
    if (c == 0 && num_blocks >= 2) {
      // It is X channel (red-green) and we often see ringing
      // in the large blocks. Let's punish that more here.
      const float x_slope =
          0.088f +
          0.014f * Clamp1((config.butteraugli_distance - 0.5f) / 1.5f, 0.0f,
                          1.0f);
      float w = (config.band_qf_stride > 0)
                    ? (1.0f + std::min(2.6f, num_blocks * x_slope))
                    : static_cast<float>(1.0 + std::min(3.0, num_blocks / 8.0));
      entropy_c *= w;
      lossc = Mul(lossc, Set(df8, w));
      lossc2 = Mul(lossc2, Set(df8, std::pow(w, 0.25f)));
    }
    entropy += entropy_c;
    loss = Add(loss, lossc);
    loss2 = Add(loss2, lossc2);
  }
  if (use_q_rdo) {
    q_out =
        q_pool + static_cast<float>(max_quant - quant_orig) * config.quant_scale;
  }
  const float inv_npixels = 1.0f / (num_blocks * kDCTBlockSize);
  const float d8 =
      std::pow(GetLane(SumOfLanes(df8, loss)) * inv_npixels, 1.0f / 8.0f);
  const float d2 = std::sqrt(GetLane(SumOfLanes(df8, loss2)) * inv_npixels);
  const float d_blend =
      (config.band_qf_stride > 0) ? (0.91f * d8 + 0.09f * 2.18f * d2) : d8;
  float loss_scalar = d_blend * (num_blocks * kDCTBlockSize) / quant_norm16;
  entropy *= entropy_mul;
  entropy += config.info_loss_multiplier * loss_scalar;
  if (config.band_qf_stride > 0 && num_blocks >= 2 && q_min > 1e-4f) {
    const float q_ratio = q_max / q_min;
    if (num_blocks <= 16 && q_ratio < 1.12f) {
      entropy *= 1.0f - 0.011f * ((1.12f - q_ratio) / 0.12f);
    } else if (q_ratio > 1.22f) {
      entropy *=
          1.0f + 0.014f * Clamp1((q_ratio - 1.22f) / 0.35f, 0.0f, 1.0f);
    }
  }
  return true;
}

Status FindBest8x8Transform(size_t x, size_t y, int encoding_speed_tier,
                            float butteraugli_target, const ACSConfig& config,
                            const float* JXL_RESTRICT cmap_factors,
                            AcStrategyImage* JXL_RESTRICT ac_strategy,
                            float* block, float* scratch_space,
                            uint32_t* quantized, float* entropy_out,
                            AcStrategyType& best_tx, float* q_out,
                            uint8_t* act_out) {
  struct TransformTry8x8 {
    AcStrategyType type;
    int encoding_speed_tier_max_limit;
    double entropy_mul;
  };
  static const TransformTry8x8 kTransforms8x8[] = {
      {
          AcStrategyType::DCT,
          9,
          0.8,
      },
      {
          AcStrategyType::DCT4X4,
          5,
          1.08,
      },
      {
          AcStrategyType::DCT2X2,
          5,
          0.95,
      },
      {
          AcStrategyType::DCT4X8,
          4,
          0.85931637428340035,
      },
      {
          AcStrategyType::DCT8X4,
          4,
          0.85931637428340035,
      },
      {
          AcStrategyType::IDENTITY,
          5,
          1.0427542510634957,
      },
      {
          AcStrategyType::AFV0,
          4,
          0.81779489591359944,
      },
      {
          AcStrategyType::AFV1,
          4,
          0.81779489591359944,
      },
      {
          AcStrategyType::AFV2,
          4,
          0.81779489591359944,
      },
      {
          AcStrategyType::AFV3,
          4,
          0.81779489591359944,
      },
  };
  double best = 1e30;
  best_tx = kTransforms8x8[0].type;
  *q_out = config.Quant(x / 8, y / 8);
  *act_out = 0;
  for (auto tx : kTransforms8x8) {
    if (tx.encoding_speed_tier_max_limit < encoding_speed_tier) {
      continue;
    }
    AcStrategy acs = AcStrategy::FromRawStrategy(tx.type);
    float entropy_mul = tx.entropy_mul / kTransforms8x8[0].entropy_mul;
    if ((tx.type == AcStrategyType::DCT2X2 ||
         tx.type == AcStrategyType::IDENTITY) &&
        butteraugli_target < 5.0) {
      float favor_2x2 = 0.40f;
      if (config.band_qf_stride > 0) {
        const float d_ramp =
            Clamp1((butteraugli_target - 0.5f) / 0.35f, 0.0f, 1.0f);
        favor_2x2 = (tx.type == AcStrategyType::DCT2X2)
                        ? (0.385f - 0.125f * d_ramp)
                        : (0.400f - 0.050f * d_ramp);
      }
      float weight = pow((5.0f - butteraugli_target) / 5.0f, 2.0f);
      entropy_mul -= favor_2x2 * weight;
    }
    if ((tx.type != AcStrategyType::DCT && tx.type != AcStrategyType::DCT2X2 &&
         tx.type != AcStrategyType::IDENTITY) &&
        butteraugli_target > 4.0) {
      static const float kAvoidEntropyOfTransforms = 0.5;
      float mul = 1.0;
      if (butteraugli_target < 12.0) {
        mul *= (12.0 - 4.0) / (butteraugli_target - 4.0);
      }
      entropy_mul += kAvoidEntropyOfTransforms * mul;
    }
    float entropy;
    float cand_q = 0.0f;
    uint8_t cand_act = 0;
    JXL_RETURN_IF_ERROR(EstimateEntropy(acs, entropy_mul, x, y, config,
                                        cmap_factors, block, scratch_space,
                                        quantized, entropy, cand_q, cand_act));
    if (entropy < best) {
      best_tx = tx.type;
      best = entropy;
      *q_out = cand_q;
      *act_out = cand_act;
    }
  }
  *entropy_out = best;
  return true;
}

Status ProcessRectACS(const CompressParams& cparams, const ACSConfig& config,
                      const Rect& rect, const ColorCorrelationMap& cmap,
                      float* JXL_RESTRICT block,
                      uint32_t* JXL_RESTRICT quantized,
                      AcStrategyImage* ac_strategy) {
  const float butteraugli_target = cparams.butteraugli_distance;
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
    CandBlock cands[4096];
    LocalCand local_cands[272];
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
  // Favor all 8x8 transforms (against 16x8 and larger transforms) at
  // low butteraugli_target distances.
  const float k8x8mul1 =
      enable_rdo
          ? (-0.375f +
             0.035f * Clamp1((butteraugli_target - 0.5f) / 0.4f, 0.0f, 1.0f))
          : -0.4f;
  static const float k8x8mul2 = 1.0f;
  static const float k8x8base = 1.4f;
  const float mul8x8 = k8x8mul2 + k8x8mul1 / (butteraugli_target + k8x8base);
  for (size_t iy = 0; iy <= ry; ++iy) {
    dp->prefix_sum[iy][0] = 0.0f;
  }
  for (size_t ix = 0; ix <= rx; ++ix) {
    dp->prefix_sum[0][ix] = 0.0f;
  }
  for (size_t iy = 0; iy < ry; iy++) {
    float row_sum = 0.0f;
    for (size_t ix = 0; ix < rx; ix++) {
      float entropy = 0.0f;
      AcStrategyType best_of_8x8s = AcStrategyType::DCT;
      float best_q = 0.0f;
      uint8_t best_act = 0;
      const float* JXL_RESTRICT cmap_factors =
          cmap_factors_tile[iy / kColorTileDimInBlocks]
                           [ix / kColorTileDimInBlocks];
      JXL_RETURN_IF_ERROR(FindBest8x8Transform(
          8 * (bx + ix), 8 * (by + iy), static_cast<int>(cparams.speed_tier),
          butteraugli_target, config, cmap_factors, ac_strategy, block,
          scratch_space, quantized, &entropy, best_of_8x8s, &best_q,
          &best_act));
      const float scaled_e = entropy * mul8x8;
      dp->entropy_estimate[iy][ix] = scaled_e;
      dp->best_8x8_type[iy][ix] = best_of_8x8s;
      dp->best_8x8_q[iy][ix] = best_q;
      dp->best_8x8_act[iy][ix] = best_act;
      row_sum += scaled_e;
      dp->prefix_sum[iy + 1][ix + 1] = dp->prefix_sum[iy][ix + 1] + row_sum;
    }
  }

  const float mb_scale =
      enable_rdo
          ? (0.985f +
             0.015f * Clamp1((butteraugli_target - 0.5f) / 1.0f, 0.0f, 1.0f))
          : 1.0f;
  const float entropy_mul16X8 = 1.21f * mb_scale;
  const float entropy_mul16X16 = 1.34f * mb_scale;
  const float entropy_mul16X32 = 1.49f * mb_scale;
  const float entropy_mul32X32 = 1.48f * mb_scale;
  const float entropy_mul64X32 = 2.25f * mb_scale;
  const float entropy_mul64X64 = 2.25f * mb_scale;

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
                                          uint8_t* out_best_chosen,
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
          !(cb.w == rw && cb.h == rh) && n_loc < 272) {
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
    uint8_t cur_chosen[64];
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
      cur_chosen[cur_num++] = static_cast<uint8_t>(idx);
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
      float direct_mul = 1.0f;
      if (w == 1 && h == 2 && allow_16) {
        has_direct = true;
        direct_type = AcStrategyType::DCT16X8;
        direct_mul = entropy_mul16X8;
      } else if (w == 2 && h == 1 && allow_16) {
        has_direct = true;
        direct_type = AcStrategyType::DCT8X16;
        direct_mul = entropy_mul16X8;
      } else if (w == 2 && h == 2 && allow_16) {
        has_direct = true;
        direct_type = AcStrategyType::DCT16X16;
        direct_mul = entropy_mul16X16;
      } else if (w == 4 && h == 2 && allow_16x32) {
        has_direct = true;
        direct_type = AcStrategyType::DCT16X32;
        direct_mul = entropy_mul16X32;
      } else if (w == 2 && h == 4 && allow_16x32) {
        has_direct = true;
        direct_type = AcStrategyType::DCT32X16;
        direct_mul = entropy_mul16X32;
      } else if (w == 4 && h == 4 && allow_32x32) {
        has_direct = true;
        direct_type = AcStrategyType::DCT32X32;
        direct_mul = entropy_mul32X32;
      } else if (w == 4 && h == 8 && allow_64x32) {
        has_direct = true;
        direct_type = AcStrategyType::DCT64X32;
        direct_mul = entropy_mul64X32;
      } else if (w == 8 && h == 4 && allow_64x32) {
        has_direct = true;
        direct_type = AcStrategyType::DCT32X64;
        direct_mul = entropy_mul64X32;
      } else if (w == 8 && h == 8 && allow_64x64) {
        has_direct = true;
        direct_type = AcStrategyType::DCT64X64;
        direct_mul = entropy_mul64X64;
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
            if (!enable_rdo && (w < h || k != k_dyadic_w)) continue;
            const float c = dp->cost[iy][iv_idx[x0][k]] +
                            dp->cost[iy][iv_idx[x0 + k][w - k]];
            if (c < best_c) {
              best_c = c;
              best_s = static_cast<uint16_t>(k);
            }
          }
          // Horizontal splits: [y0, y0 + k) and [y0 + k, y0 + h)
          for (size_t k = 1; k < h; ++k) {
            if (!enable_rdo && (h < w || k != k_dyadic_h)) continue;
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
            const bool eval_direct =
                enable_rdo
                    ? ((w * h <= 16) || ((x0 & 1) == 0 && (y0 & 1) == 0) ||
                       (best_c < base_sum - 1e-4f))
                    : (x0 % w == 0 && y0 % h == 0);
            if (eval_direct) {
              float cand_entropy = kInf;
              float cand_q = 0.0f;
              uint8_t cand_act = 0;
              const float* JXL_RESTRICT cmap_factors =
                  cmap_factors_tile[y0 / kColorTileDimInBlocks]
                                   [x0 / kColorTileDimInBlocks];
              JXL_RETURN_IF_ERROR(EstimateEntropy(
                  acs_direct, direct_mul, (bx + x0) * 8, (by + y0) * 8, config,
                  cmap_factors, block, scratch_space, quantized, cand_entropy,
                  cand_q, cand_act));
              if (cand_entropy < best_c) {
                best_c = cand_entropy;
                best_s = 0;
                dp->leaf[y0][x0][h][w] = direct_type;
                dp->leaf_q[y0][x0][h][w] = cand_q;
                dp->leaf_act[y0][x0][h][w] = cand_act;
                const float gain = base_sum - cand_entropy;
                if (enable_rdo && gain > 0.0f && num_cands < 4096) {
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
            uint8_t ng_chosen[64];
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
      const int32_t quant_orig =
          Quantizer::ClampVal(q_pool * config.inv_global_scale + 0.5f);
      const int32_t quant_final =
          Quantizer::ClampVal(q_val * config.inv_global_scale + 0.5f);
      const float q_step_ratio =
          static_cast<float>(quant_final) / static_cast<float>(quant_orig);
      for (size_t c = 0; c < 3; ++c) {
        for (size_t s = 0; s < 4; ++s) {
          if (config.band_qf_rows[c][s] == nullptr) continue;
          float sq_sum = 0.0f;
          for (size_t iy = 0; iy < h; ++iy) {
            for (size_t ix = 0; ix < w; ++ix) {
              const float old_q =
                  std::max(config.Quant(bx + x0 + ix, by + y0 + iy), 1e-4f);
              const float rescale = q_pool / old_q;
              const float band_rescale =
                  (rescale > 1.0f)
                      ? (1.0f + kBandRescaleMix[s] * (rescale - 1.0f))
                      : rescale;
              const float v =
                  config.BandQuant(c, s, bx + x0 + ix, by + y0 + iy) *
                  band_rescale;
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
                                 (bx + x0 + ix)] = q_pool;
        }
      }
    } else if (config.band_qf_rows[1][0] != nullptr) {
      float bq = 0.0f;
      uint32_t bits = static_cast<uint32_t>(act_val) & 31u;
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
      uint8_t ng_chosen[64];
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
                                  ImageF* quant_field, const ImageF& mask,
                                  const ImageF& mask1x1,
                                  DequantMatrices* matrices,
                                  ImageF (*band_qf)[4], float x_qm_multiplier,
                                  float b_qm_multiplier, float quant_scale,
                                  float inv_global_scale) {
  config.dequant = matrices;
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
  if (mask.xsize() > 0 && mask.ysize() > 0) {
    config.masking_field_row = mask.Row(0);
    config.masking_field_stride = mask.PixelsPerRow();
  }
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
  mem_per_thread = 7 * AcStrategy::kMaxCoeffArea + dct_scratch_size;
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
