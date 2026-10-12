// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_ADAPTIVE_QUANTIZATION_H_
#define LIB_JXL_ENC_ADAPTIVE_QUANTIZATION_H_

#include <jxl/cms_interface.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"

// Heuristics to find a good quantizer for a given image. InitialQuantField
// produces a quantization field (i.e. relative quantization amounts for each
// block) out of an opsin-space image. `InitialQuantField` uses heuristics,
// `FindBestQuantizer` (in non-fast mode) will run multiple encoding-decoding
// steps and try to improve the given quant field.

namespace jxl {

struct AuxOut;
class AcStrategyImage;

// Maps normalized 8x8 frequency coordinates (v8 = y / yblocks, u8 = x / xblocks)
// to one of 4 dyadic spatial-frequency bands:
//   s = 0: 1:1 finest features (max(u8, v8) in 6..7, wavelength ~2 px)
//   s = 1: 1:2 fine features   (max(u8, v8) in 4..5, wavelength ~4 px)
//   s = 2: 1:4 medium features (max(u8, v8) in 2..3, wavelength ~8 px)
//   s = 3: 1:8 coarse features (max(u8, v8) <= 1,    wavelength ~16 px)
static constexpr uint8_t kCoeffBand[8][8] = {
    {3, 3, 2, 2, 1, 1, 0, 0},
    {3, 3, 2, 2, 1, 1, 0, 0},
    {2, 2, 2, 2, 1, 1, 0, 0},
    {2, 2, 2, 2, 1, 1, 0, 0},
    {1, 1, 1, 1, 1, 1, 0, 0},
    {1, 1, 1, 1, 1, 1, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0},
};

static constexpr size_t kPartialBlockKinds =
    (1 << static_cast<size_t>(AcStrategyType::IDENTITY)) |
    (1 << static_cast<size_t>(AcStrategyType::DCT2X2)) |
    (1 << static_cast<size_t>(AcStrategyType::DCT4X4)) |
    (1 << static_cast<size_t>(AcStrategyType::DCT4X8)) |
    (1 << static_cast<size_t>(AcStrategyType::DCT8X4)) |
    (1 << static_cast<size_t>(AcStrategyType::AFV0)) |
    (1 << static_cast<size_t>(AcStrategyType::AFV1)) |
    (1 << static_cast<size_t>(AcStrategyType::AFV2)) |
    (1 << static_cast<size_t>(AcStrategyType::AFV3));

// Computes per-quadrant base thresholds zero_thres[0..3] (indexed by
// directional quadrant hfix = (v8 >= 4) * 2 + (u8 >= 4)) and jpegli-style
// per-band deadzone offsets zero_thres[4..7] (indexed by radial scale band
// s = kCoeffBand[v8][u8]) from the ratio between each group's reference
// quantizer and each band's perceptual target quantizer band_q[s].
static JXL_INLINE void ComputeBandThresholds(size_t c,
                                             AcStrategyType quant_kind,
                                             size_t xblocks, size_t yblocks,
                                             float q_block,
                                             const float band_q[4],
                                             float zero_thres[8],
                                             float pos_dz_scale = 1.0f) {
  static constexpr float kQuadY[4] = {0.58f, 0.64f, 0.64f, 0.64f};
  static constexpr float kQuadChroma[4] = {0.58f, 0.62f, 0.62f, 0.62f};
  const bool is_partial_block =
      ((1 << static_cast<size_t>(quant_kind)) & kPartialBlockKinds) != 0;
  const float num_blocks = static_cast<float>(xblocks * yblocks);
  const float multi_block_sub =
      (!is_partial_block && c == 1 && num_blocks > 1.0f)
          ? Clamp1(0.003f * num_blocks, 0.0f, 0.08f)
          : 0.0f;
  for (size_t hfix = 0; hfix < 4; ++hfix) {
    float q_t0 = (c == 1) ? kQuadY[hfix] : kQuadChroma[hfix];
    if (multi_block_sub > 0.0f) {
      q_t0 = std::max(0.54f, q_t0 - multi_block_sub);
    }
    zero_thres[hfix] = q_t0;
  }
  if (is_partial_block) {
    for (size_t s = 0; s < 4; ++s) {
      zero_thres[4 + s] = 0.0f;
    }
    return;
  }
  const float q_all = 0.30f * band_q[0] + 0.30f * band_q[1] +
                      0.25f * band_q[2] + 0.15f * band_q[3];
  const float q_ref = 0.50f * q_block + 0.50f * q_all;
  for (size_t s = 0; s < 4; ++s) {
    const float bq = std::max(band_q[s], 1e-4f);
    const float rho =
        (q_ref > 1e-4f) ? Clamp1(q_ref / bq, 0.80f, 1.25f) : 1.0f;
    const float alpha = rho * rho;
    const float slope = (alpha < 1.0f) ? 0.14f : (0.12f * pos_dz_scale);
    zero_thres[4 + s] =
        Clamp1(slope * (alpha - 1.0f), -0.022f, 0.024f * pos_dz_scale);
  }
}

// Returns an image subsampled by kBlockDim in each direction. If the value
// at pixel (x,y) in the returned image is greater than 1.0, it means that
// more fine-grained quantization should be used in the corresponding block
// of the input image, while a value less than 1.0 indicates that less
// fine-grained quantization should be enough. Returns a mask, too, which
// can later be used to make better decisions about ac strategy.
StatusOr<ImageF> InitialQuantField(float butteraugli_target,
                                   const Image3F& opsin, const Rect& rect,
                                   ThreadPool* pool, float rescale,
                                   ImageF* initial_quant_mask,
                                   ImageF* initial_quant_mask1x1,
                                   ImageF (*band_qf)[4] = nullptr);

float InitialQuantDC(float butteraugli_target);

Status AdjustQuantField(const AcStrategyImage& ac_strategy, const Rect& rect,
                        float butteraugli_target, ImageF* quant_field,
                        ImageF (*band_qf)[4] = nullptr);

// Returns a quantizer that uses an adjusted version of the provided
// quant_field. Also computes the dequant_map corresponding to the given
// dequant_float_map and chosen quantization levels.
// `linear` is only used in Kitten mode or slower.
Status FindBestQuantizer(const FrameHeader& frame_header, const Image3F* linear,
                         const Image3F& opsin, ImageF& quant_field,
                         PassesEncoderState* enc_state,
                         const JxlCmsInterface& cms, ThreadPool* pool,
                         AuxOut* aux_out, double rescale = 1.0);

}  // namespace jxl

#endif  // LIB_JXL_ENC_ADAPTIVE_QUANTIZATION_H_
