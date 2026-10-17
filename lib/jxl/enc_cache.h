// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_CACHE_H_
#define LIB_JXL_ENC_CACHE_H_

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_progressive_split.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quant_weights.h"

namespace jxl {

struct AuxOut;

struct AcTokenCostModel {
  static constexpr size_t kMaxTokenAlphabet = 256;
  static constexpr size_t kSmallValLimit = 64;

  BlockCtxMap block_ctx_map;
  std::vector<coeff_order_t> coeff_orders;
  Image3I pass1_nzeros;
  std::vector<uint8_t> cluster_of_ctx;
  std::vector<HybridUintConfig> uint_config;
  std::vector<float> tok_bits;
  std::vector<float> small_val_bits;
  std::vector<float> rdo_c_one;
  std::vector<float> rdo_c_slope;
  std::vector<float> rdo_nzero_bits;
  float acs_strategy_bits[27] = {};
  float qf_block_bits = 2.2f;
  float bits_to_cost_mul = 1.0f;

  JXL_INLINE float LargeValBits(size_t cluster, uint32_t value) const {
    uint32_t tok = 0;
    uint32_t nbits = 0;
    uint32_t bits = 0;
    uint_config[cluster].Encode(value, &tok, &nbits, &bits);
    const size_t t_idx = cluster * kMaxTokenAlphabet +
                         std::min<size_t>(tok, kMaxTokenAlphabet - 1);
    return tok_bits[t_idx] + static_cast<float>(nbits);
  }

  JXL_INLINE float ValBits(size_t cluster, uint32_t value) const {
    if (JXL_LIKELY(value < kSmallValLimit)) {
      return small_val_bits[cluster * kSmallValLimit + value];
    }
    return LargeValBits(cluster, value);
  }

  JXL_INLINE int32_t PredictNzeros(size_t c, size_t bx0, size_t by0) const {
    const size_t lx = bx0 & 31;
    const size_t ly = by0 & 31;
    if (lx == 0 && ly == 0) return 32;
    const int32_t* JXL_RESTRICT row = pass1_nzeros.ConstPlaneRow(c, by0);
    if (ly == 0) return row[bx0 - 1];
    const int32_t* JXL_RESTRICT row_top =
        pass1_nzeros.ConstPlaneRow(c, by0 - 1);
    if (lx == 0) return row_top[bx0];
    return (row[bx0 - 1] + row_top[bx0] + 1) >> 1;
  }
};

// Contains encoder state.
struct PassesEncoderState {
  explicit PassesEncoderState(JxlMemoryManager* memory_manager)
      : shared(memory_manager) {}

  PassesSharedState shared;

  bool streaming_mode = false;
  bool initialize_global_state = true;
  size_t dc_group_index = 0;

  // Per-pass DCT coefficients for the image. One row per group.
  std::vector<std::unique_ptr<ACImage>> coeffs;

  // Raw data for special (reference+DC) frames.
  std::vector<std::unique_ptr<BitWriter>> special_frames;

  // For splitting into passes.
  ProgressiveSplitter progressive_splitter;

  CompressParams cparams;

  struct PassData {
    std::vector<std::vector<Token>> ac_tokens;
    std::vector<uint8_t> context_map;
    EntropyEncodingData codes;
  };

  std::vector<PassData> passes;
  std::vector<size_t> histogram_idx;

  // Block sizes seen so far.
  uint32_t used_acs = 0;
  // Coefficient orders that are non-default.
  std::vector<uint32_t> used_orders;

  // Multiplier to be applied to the quant matrices of the x channel.
  float x_qm_multiplier = 1.0f;
  float b_qm_multiplier = 1.0f;

  ImageF initial_quant_masking1x1;

  // Per-channel (X, Y, B), per-scale band (s = 0: 1:1, 1: 1:2, 2: 1:4, 3: 1:8)
  // target quantizer fields at 8x8 block resolution for intra-block AQ and RDO.
  ImageF band_qf[3][4];

  std::unique_ptr<AcTokenCostModel> token_cost_model;

  JxlMemoryManager* memory_manager() const { return shared.memory_manager; }
};

// Initialize per-frame information.
class ModularFrameEncoder;
Status InitializePassesEncoder(FrameHeader& frame_header,
                               const Image3F& opsin, const Rect& rect,
                               const JxlCmsInterface& cms, ThreadPool* pool,
                               PassesEncoderState* passes_enc_state,
                               ModularFrameEncoder* modular_frame_encoder,
                               AuxOut* aux_out);

Status ComputeACMetadata(ThreadPool* pool, PassesEncoderState* enc_state,
                         ModularFrameEncoder* modular_frame_encoder);

}  // namespace jxl

#endif  // LIB_JXL_ENC_CACHE_H_
