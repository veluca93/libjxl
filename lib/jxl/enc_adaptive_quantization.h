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

static JXL_INLINE float ComputeSharpSigmaLo(float butteraugli_distance) {
  const float d_c = Clamp1(butteraugli_distance, 0.35f, 2.5f);
  const float t =
      std::log(2.5f / d_c) * (1.0f / 1.9661128563728327f);  // ln(2.5 / 0.35)
  return t * t * (3.0f - 2.0f * t);
}

static JXL_INLINE float SnapToExactJxlF16(float value) {
  if (!(value > -65504.0f && value < 65504.0f)) {
    return (value > 0.0f) ? 65504.0f : ((value < 0.0f) ? -65504.0f : 0.0f);
  }
  uint32_t bits32 = 0;
  memcpy(&bits32, &value, sizeof(bits32));
  const uint32_t sign = bits32 >> 31;
  int32_t exp = static_cast<int32_t>((bits32 >> 23) & 0xFF) - 127;
  uint32_t mantissa32 = bits32 & 0x7FFFFFu;
  if (exp < -24) return 0.0f;
  if (exp >= -14) {
    const uint32_t rounded = bits32 + 0x1000u;
    if ((static_cast<int32_t>((rounded >> 23) & 0xFF) - 127) <= 15) {
      bits32 = rounded;
      exp = static_cast<int32_t>((bits32 >> 23) & 0xFF) - 127;
      mantissa32 = bits32 & 0x7FFFFFu;
    }
  }
  uint32_t biased_exp16 = 0;
  uint32_t mantissa16 = 0;
  if (exp < -14) {
    const uint32_t sub_exp = static_cast<uint32_t>(-14 - exp);
    mantissa16 = (1u << (10 - sub_exp)) + (mantissa32 >> (13 + sub_exp));
  } else {
    biased_exp16 = static_cast<uint32_t>(exp + 15);
    mantissa16 = mantissa32 >> 13;
  }
  if (biased_exp16 == 0) {
    const float out =
        (1.0f / 16384.0f) * (static_cast<float>(mantissa16) * (1.0f / 1024.0f));
    return sign ? -out : out;
  }
  const uint32_t out_bits32 =
      (sign << 31) | ((biased_exp16 + (127 - 15)) << 23) | (mantissa16 << 13);
  float out = 0.0f;
  memcpy(&out, &out_bits32, sizeof(out));
  return out;
}

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
                                             float pos_dz_scale = 1.0f,
                                             float sigma_lo = 0.0f) {
  static constexpr float kQuadY[4] = {0.58f, 0.64f, 0.64f, 0.64f};
  static constexpr float kQuadChroma[4] = {0.58f, 0.62f, 0.62f, 0.62f};
  const bool is_partial_block =
      ((1 << static_cast<size_t>(quant_kind)) & kPartialBlockKinds) != 0;
  const float num_blocks = static_cast<float>(xblocks * yblocks);
  const float multi_block_sub =
      (!is_partial_block && c == 1 && num_blocks > 1.0f)
          ? Clamp1(0.003f * num_blocks, 0.0f, 0.08f)
          : 0.0f;
  const float mb_sharp_sub =
      (!is_partial_block && c == 1 && num_blocks > 1.0f)
          ? (0.014f * sigma_lo * Clamp1((num_blocks - 1.0f) * 0.25f, 0.0f, 1.0f))
          : 0.0f;
  for (size_t hfix = 0; hfix < 4; ++hfix) {
    float q_t0 = (c == 1) ? kQuadY[hfix] : kQuadChroma[hfix];
    if (hfix >= 1 && mb_sharp_sub > 0.0f) {
      q_t0 -= mb_sharp_sub;
    }
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
        (q_ref > 1e-4f) ? Clamp1(q_ref / bq, 0.78f, 1.32f) : 1.0f;
    const float alpha = rho * rho;
    const float slope = (alpha < 1.0f) ? 0.14f : (0.15f * pos_dz_scale);
    zero_thres[4 + s] =
        Clamp1(slope * (alpha - 1.0f), -0.023f, 0.032f * pos_dz_scale);
  }
}

static JXL_INLINE void ComputeBlockThresholds(
    AcStrategyType quant_kind, size_t xblocks, size_t yblocks, float q_block,
    const float band_q[3][4], float butteraugli_distance, float dz_w,
    float eob_scale, float zero_thres[3][8]) {
  const float sigma_lo = ComputeSharpSigmaLo(butteraugli_distance);
  for (size_t c = 0; c < 3; ++c) {
    ComputeBandThresholds(c, quant_kind, xblocks, yblocks, q_block, band_q[c],
                          zero_thres[c], dz_w, sigma_lo);
  }
  const bool is_partial_block =
      ((1 << static_cast<size_t>(quant_kind)) & kPartialBlockKinds) != 0;
  uint32_t act_bits = 0;
  memcpy(&act_bits, &band_q[1][0], sizeof(act_bits));
  const int32_t activity = static_cast<int32_t>((act_bits >> 1) & 15u);
  const bool q0_sub = (act_bits & 1u) != 0;
  if (eob_scale <= 0.0f) {
    if (!is_partial_block) {
      const float low_d_keep = 1.0f - 0.70f * sigma_lo;
      for (int i = 1; i < 4; ++i) {
        zero_thres[1][i] += 0.01f * low_d_keep * activity;
      }
    }
    return;
  }

  const float dz_w_rt =
      Clamp1((butteraugli_distance - 0.38f) / (butteraugli_distance + 0.55f) *
                 1.65f,
             0.0f, 1.0f);
  const float hr_w = 1.0f - dz_w_rt;
  const float act_dz_keep = 1.0f - 0.72f * sigma_lo;
  const float struct_hf_sub =
      (activity >= 3)
          ? (0.010f * sigma_lo *
             Clamp1(static_cast<float>(activity - 2) * 0.20f, 0.0f, 1.0f))
          : 0.0f;
  if (!is_partial_block) {
    if (q0_sub) {
      zero_thres[1][0] -= 0.003f;
    }
    float luma_hf_dz = 0.0f;
    const float raw_act = static_cast<float>(activity);
    const float eff_act = raw_act / (1.0f + 0.032f * dz_w_rt * raw_act);
    if (activity > 0) {
      const float delta_hf =
          -0.002f * hr_w - struct_hf_sub +
          dz_w_rt * act_dz_keep * (0.0100f * eff_act);
      zero_thres[1][1] += delta_hf;
      zero_thres[1][2] += delta_hf;
      luma_hf_dz = std::max(0.0f, delta_hf);
    }
    zero_thres[1][3] +=
        -0.002f * hr_w - struct_hf_sub +
        dz_w_rt * act_dz_keep * (0.0025f + 0.0115f * eff_act);
    for (int cc : {0, 2}) {
      const float c_mf_ratio = band_q[cc][2] / std::max(q_block, 1e-4f);
      const float c_luma_mask_w =
          Clamp1(1.0f - (c_mf_ratio - 1.04f) * 4.0f, 0.0f, 1.0f);
      zero_thres[cc][1] += 0.18f * luma_hf_dz * c_luma_mask_w;
      zero_thres[cc][2] += 0.18f * luma_hf_dz * c_luma_mask_w;
      zero_thres[cc][3] += 0.005f * dz_w_rt + 0.40f * luma_hf_dz;
    }
  }
  for (int cc : {0, 2}) {
    const float c_scale = (cc == 0) ? 0.85f : 1.15f;
    zero_thres[cc][1] += 0.005f * c_scale * dz_w_rt;
    zero_thres[cc][2] += 0.005f * c_scale * dz_w_rt;
    zero_thres[cc][3] += 0.003f * c_scale * dz_w_rt;
    zero_thres[cc][4 + 0] += 0.004f * c_scale * dz_w_rt;
    zero_thres[cc][4 + 1] += 0.002f * c_scale * dz_w_rt;
  }
}

// Returns an image subsampled by kBlockDim in each direction. If the value
// at pixel (x,y) in the returned image is greater than 1.0, it means that
// more fine-grained quantization should be used in the corresponding block
// of the input image, while a value less than 1.0 indicates that less
// fine-grained quantization should be enough.
StatusOr<ImageF> InitialQuantField(float butteraugli_target,
                                   const Image3F& opsin, const Rect& rect,
                                   ThreadPool* pool, float rescale,
                                   ImageF* initial_quant_mask1x1,
                                   ImageF (*band_qf)[4] = nullptr,
                                   bool has_noise = false);

float InitialQuantDC(float butteraugli_target);

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
