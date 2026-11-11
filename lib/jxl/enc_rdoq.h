// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_RDOQ_H_
#define LIB_JXL_ENC_RDOQ_H_

// Coupled Lagrangian Rate-Distortion Optimization (RDO) and Backward-DP
// Trellis Rate-Distortion Optimized Quantization (RDOQ) for VarDCT e7/e8/e9.

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"

namespace jxl {

// Calibrated Lagrangian multiplier lambda(d) mapping rate in bits to normalized
// distortion units.
float ComputeRDOLambda(float butteraugli_distance);

// Immutable per-pass rate model providing fast O(1) small-symbol bit estimates
// for AC tokens (NonZeroContext & ZeroDensityContext) and kACMeta control
// fields (ACS, raw_quant_field, CfL, EPF).
struct RDORateModel {
  static constexpr size_t kFastTableSize = 64;

  void InitPreset(const BlockCtxMap& bcm, float butteraugli_distance,
                  uint32_t decoding_speed_tier);

  void InitFromEmpiricalPass(const BlockCtxMap& bcm,
                             const EntropyEncodingData& codes,
                             const AcStrategyImage& ac_strategy,
                             const ImageI& raw_quant_field,
                             const ColorCorrelationMap& cmap,
                             const ImageB& epf_sharpness,
                             const FrameDimensions& frame_dim);

  // Bit cost of coding unsigned value `u_val` in AC context `ctx`.
  JXL_INLINE float TokenCost(size_t ctx, uint32_t u_val) const {
    if (ctx >= num_ac_contexts) ctx = 0;
    if (u_val < kFastTableSize) {
      return fast_cost[ctx * kFastTableSize + u_val];
    }
    return SlowTokenCost(ctx, u_val);
  }

  float AcsCost(uint8_t raw_acs, uint8_t prev_raw_acs) const;
  float QfDeltaCost(int32_t qf, int32_t prev_qf) const;
  float CflCost(int32_t m, int32_t pred_m) const;
  float EpfCost(uint8_t epf, uint8_t top_epf, uint8_t left_epf) const;

  size_t num_ac_contexts = 0;
  std::vector<float> fast_cost;
  bool has_empirical_ac = false;
  std::vector<std::vector<float>> sym_bits;
  std::vector<HybridUintConfig> uint_configs;
  std::vector<uint8_t> context_map;
  float unseen_base = 12.0f;

  // Empirically updated / preset kACMeta cost tables.
  float acs_cost_table[4][AcStrategy::kNumValidStrategies] = {};
  float qf_zero_cost[4] = {};
  float qf_nonzero_base[4] = {};
  float cfl_zero_cost = 1.0f;
  float cfl_nonzero_base = 2.8f;
  float epf_cost_table[4][8] = {};

 private:
  float SlowTokenCost(size_t ctx, uint32_t u_val) const;
};

// Runs coupled non-greedy Guillotine Partition DP + quantized-Yhat CfL RDO
// across all 8x8-block (64x64 px) color tiles. Populates `shared.ac_strategy`,
// initial `shared.raw_quant_field`, and `shared.cmap`.
Status ProcessAllTilesRDO(const CompressParams& cparams,
                          const FrameHeader& frame_header, const Image3F& opsin,
                          const Rect& rect, const ImageF& initial_quant_field,
                          const ImageF& initial_quant_masking1x1,
                          const RDORateModel& rate_model, ThreadPool* pool,
                          PassesEncoderState* enc_state, AuxOut* aux_out);

// Group-parallel causal Trellis RDOQ with 1D Viterbi / deadzone-widening AQ
// smoothing after DC + FindBestBlockEntropyModel. Writes refined
// `shared.raw_quant_field` across all covered sub-blocks and populates
// `enc_state->coeffs`.
Status ComputeCoefficientsRDOQ(const FrameHeader& frame_header,
                               const Image3F& opsin, const Rect& rect,
                               const ImageF& initial_quant_field,
                               const ImageF& initial_quant_masking1x1,
                               const RDORateModel& rate_model, ThreadPool* pool,
                               PassesEncoderState* enc_state);

// Transforms `opsin` blocks according to `ac_strategy` and extracts lowest
// frequencies into `dc` across all 3 channels.
Status ComputeDCFromOpsin(const Image3F& opsin, const Rect& rect,
                          const AcStrategyImage& ac_strategy, ThreadPool* pool,
                          Image3F* dc);

// Additive Rate-Distortion Optimization of EPF sharpness field using
// ReconstructImage + 2-pass kACMeta rate costs (with row-wise Viterbi at
// e8/e9).
Status ComputeEPFSharpnessRDO(const FrameHeader& frame_header,
                              const Image3F& orig_opsin, const Rect& rect,
                              const RDORateModel& rate_model, ThreadPool* pool,
                              PassesEncoderState* enc_state);

// Standalone helper for unit testing Backward-DP Trellis RDOQ on a single
// block.
struct RDOQTrellisResult {
  float rd_cost = 0.0f;
  float dist = 0.0f;
  float rate_bits = 0.0f;
  int32_t nzeros = 0;
};

RDOQTrellisResult RunTrellisRDOQSingleBlockForTest(
    size_t channel, AcStrategyType acs_type, const float* coeffs_in,
    const Quantizer& quantizer, float qm_multiplier, int32_t qf, float q_target,
    int32_t predicted_nzeros, size_t block_ctx, const BlockCtxMap& bcm,
    const coeff_order_t* order, const RDORateModel& rate_model, float lambda,
    int band_width, int multi_level, int32_t* quantized_out);

}  // namespace jxl

#endif  // LIB_JXL_ENC_RDOQ_H_
